/*********************                                                        */
/*! \file CROWNAnalysis.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "CROWNAnalysis.h"
#include "nodes/BoundedConstantNode.h"
#include "nodes/BoundedBatchNormNode.h"
#include "nodes/BoundedConvNode.h"
#include "conv/Patches.h"

#include "Debug.h"
#include "MStringf.h"
#include "LunaError.h"
#include "TimeUtils.h"

#include <limits>
#include <vector>
#include <sstream>

namespace NLR
{

torch::Tensor CROWNAnalysis::buildCenterInputForForward() const
{
    if ( !_torchModel || !_torchModel->hasInputBounds() )
        return torch::Tensor();

    torch::Tensor lb = _torchModel->getInputLowerBounds();
    torch::Tensor ub = _torchModel->getInputUpperBounds();
    if ( !lb.defined() || !ub.defined() )
        return torch::Tensor();

    lb = lb.to( torch::kFloat32 );
    ub = ub.to( torch::kFloat32 );
    torch::Tensor center = ( lb + ub ) / 2.0;

    // VNN-LIB bounds are flat, the model knows its own sample shape
    return _torchModel->toBatchedInput( center );
}

void CROWNAnalysis::ensureCenterActivations()
{
    if ( _hasCenterActivations )
        return;
    if ( !_torchModel )
        return;

    torch::NoGradGuard no_grad;
    try
    {
        torch::Tensor center = buildCenterInputForForward();
        if ( !center.defined() )
            return;
        _centerActivations = _torchModel->forwardAndStoreActivations( center );
        _hasCenterActivations = true;
    }
    catch ( const std::exception &e )
    {
        (void)e;
        _hasCenterActivations = false;
        _centerActivations.clear();
    }
    catch ( ... )
    {
        _hasCenterActivations = false;
        _centerActivations.clear();
    }
}

CROWNAnalysis::CROWNAnalysis( TorchModel *torchModel )
    : _torchModel( torchModel )
{
    const Vector<std::shared_ptr<BoundedTorchNode>> &nodes = _torchModel->getNodes();

    for ( unsigned i = 0; i < nodes.size(); ++i )
    {
        _nodes[i] = nodes[i];
    }
}

CROWNAnalysis::~CROWNAnalysis() {}

void CROWNAnalysis::run( bool enableGradients )
{
    log( "run() - Starting" );
    std::string stage = "start";
    try
    {
        _hasCenterActivations = false;
        _centerActivations.clear();
        _foundFirstUnsound = false;
        _firstUnsoundNode = 0;

        // Gradient-enabled mode (alpha-CROWN) uses single backward pass to avoid
        // in-place modification conflicts with autograd on shared alpha tensors
        bool useStandardCrown = LunaConfiguration::USE_STANDARD_CROWN;

        if ( useStandardCrown )
        {
            stage = "computeIBPBounds(standard)";
            computeIBPBounds();

            // Only set concrete bounds for INPUT initially (auto_LiRPA style)
            for ( const auto &p : _ibpBounds )
            {
                unsigned nodeIdx = p.first;
                if ( _nodes.exists( nodeIdx ) && _nodes[nodeIdx]->getNodeType() == NodeType::INPUT )
                {
                    _concreteBounds[nodeIdx] = p.second;
                    _torchModel->setConcreteBounds( nodeIdx, p.second );
                    _nodes[nodeIdx]->setBounds( p.second.lower(), p.second.upper() );
                }
            }

            unsigned outputIndex = getOutputIndex();
            auto &outputNode = _nodes[outputIndex];
            unsigned outputSize = outputNode->getOutputSize();
            std::string startKey = "/" + std::to_string( outputIndex );
            _setCurrentStart( startKey, outputSize );
            _currentStartSpecIndices.clear();

            log( Stringf(
                "run() - Single backward pass from output (lazy intermediate computation)" ) );

            stage = "checkPriorBounds(outputIndex)";
            checkPriorBounds( outputIndex );

            // Restore start context after checkPriorBounds() overwrites it with intermediate keys
            _setCurrentStart( startKey, outputSize );
            _currentStartSpecIndices.clear();

            stage =
                std::string( "backwardFrom(outputIndex=" ) + std::to_string( outputIndex ) + ")";
            backwardFrom( outputIndex );

            stage =
                std::string( "concretizeNode(outputIndex=" ) + std::to_string( outputIndex ) + ")";
            concretizeNode( outputIndex );
        }
        else
        {
            // CROWN-IBP: IBP for intermediates, CROWN for final
            stage = "computeIBPBounds(crown_ibp)";
            computeIBPBounds();

            unsigned outputIndex = getOutputIndex();
            auto &outputNode = _nodes[outputIndex];
            unsigned outputSize = outputNode->getOutputSize();
            std::string startKey = "/" + std::to_string( outputIndex );
            _setCurrentStart( startKey, outputSize );
            _currentStartSpecIndices.clear();

            stage =
                std::string( "backwardFrom(outputIndex=" ) + std::to_string( outputIndex ) + ")";
            backwardFrom( outputIndex );
            stage =
                std::string( "concretizeNode(outputIndex=" ) + std::to_string( outputIndex ) + ")";
            concretizeNode( outputIndex );
        }
    }
    catch ( const CommonError &e )
    {
        std::ostringstream oss;
        oss << "CROWNAnalysis::run CommonError code=" << e.getCode() << " stage=" << stage;
        const char *msg = e.getUserMessage();
        if ( msg && msg[0] != '\0' )
            oss << " message=" << msg;
        throw std::runtime_error( oss.str() );
    }
    catch ( const Error &e )
    {
        std::ostringstream oss;
        oss << "CROWNAnalysis::run Error class=" << e.getErrorClass() << " code=" << e.getCode()
            << " stage=" << stage;
        const char *msg = e.getUserMessage();
        if ( msg && msg[0] != '\0' )
            oss << " message=" << msg;
        throw std::runtime_error( oss.str() );
    }
    catch ( const std::exception &e )
    {
        log( Stringf( "run() - Exception caught: %s", e.what() ) );
        throw std::runtime_error( std::string( "CROWNAnalysis::run exception stage=" ) + stage +
                                  ": " + e.what() );
    }
    log( "run() - Completed" );
}

void CROWNAnalysis::computeIBPBounds()
{
    resetProcessingState();
    Vector<unsigned> forwardOrder = _torchModel->topologicalSort();
    std::size_t recomputedNodes = 0;
    std::size_t reusedNodes = 0;

    log( Stringf( "computeIBPBounds() - Processing %u nodes", forwardOrder.size() ) );

    for ( unsigned nodeIndex : forwardOrder )
    {

        if ( isProcessed( nodeIndex ) )
            continue;
        markProcessed( nodeIndex );

        auto &node = _nodes[nodeIndex];

        // Fixed intermediates never read recomputed IBP, so seeded bounds stand even when dirty
        const bool reuseSeededBounds = !LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS ||
                                       !_torchModel->isIncrementalNodeDirty( nodeIndex );
        if ( node->hasBounds() && reuseSeededBounds )
        {
            BoundedTensor<torch::Tensor> cachedBounds( node->getLower().detach(),
                                                       node->getUpper().detach() );
            _ibpBounds[nodeIndex] = cachedBounds;
            _concreteBounds[nodeIndex] = cachedBounds;
            _torchModel->setConcreteBounds( nodeIndex, cachedBounds );
            ++reusedNodes;
            continue;
        }
        ++recomputedNodes;

        Vector<BoundedTensor<torch::Tensor>> inputBounds = getInputBoundsForNode( nodeIndex );

        if ( node->getNodeType() == NodeType::INPUT )
        {
            if ( _torchModel->hasInputBounds() )
            {
                torch::Tensor inputLower = _torchModel->getInputLowerBounds();
                torch::Tensor inputUpper = _torchModel->getInputUpperBounds();
                _ibpBounds[nodeIndex] = BoundedTensor<torch::Tensor>( inputLower, inputUpper );
                continue;
            }
        }

        BoundedTensor<torch::Tensor> ibpBounds =
            node->computeIntervalBoundPropagation( inputBounds );

        if ( node->hasBounds() )
        {
            // Specification bounds at the output can have one row per property
            // clause rather than one entry per raw network output. They are not
            // neuron intervals, so keep the forward IBP result in that case.
            if ( nodeIndex == getOutputIndex() &&
                 ( ibpBounds.lower().sizes() != node->getLower().sizes() ||
                   ibpBounds.upper().sizes() != node->getUpper().sizes() ) )
            {
                node->clearBounds();
            }
        }

        if ( node->hasBounds() )
        {
            ibpBounds = BoundedTensor<torch::Tensor>(
                torch::maximum( ibpBounds.lower(), node->getLower() ),
                torch::minimum( ibpBounds.upper(), node->getUpper() ) );
        }

        // Detach to prevent gradient history leaking across iterations
        _ibpBounds[nodeIndex] =
            BoundedTensor<torch::Tensor>( ibpBounds.lower().detach(), ibpBounds.upper().detach() );
    }

    log( Stringf( "computeIBPBounds() - Completed, stored bounds for %u nodes",
                  _ibpBounds.size() ) );
    _torchModel->setForwardBoundWorkStats( recomputedNodes, reusedNodes );
}

bool CROWNAnalysis::getAlphaStartCacheInfo( const std::string &key,
                                            Vector<unsigned> &unstableIndices, bool &sparseMode,
                                            unsigned &nodeSize ) const
{
    auto it = _alphaStartCache.find( key );
    if ( it == _alphaStartCache.end() || !it->second.initialized )
    {
        return false;
    }
    unstableIndices = it->second.unstableIndices;
    sparseMode = it->second.sparseMode;
    nodeSize = it->second.nodeSize;
    return true;
}

void CROWNAnalysis::backwardFrom( unsigned startIndex, const Vector<unsigned> &unstableIndices,
                                  const torch::Tensor *C )
{
    unsigned current_dbg = startIndex;
    std::string stage = "start";
    try
    {
        stage = "checkStartIndexExists";
        if ( !_nodes.exists( startIndex ) )
        {
            log( Stringf( "backwardFrom() - Warning: start index %u not found in nodes.",
                          startIndex ) );
            return;
        }

        stage = "clearCrownState";
        clearCrownState();
        _nodesNeedingBounds.clear();

        auto &startNode = _nodes[startIndex];
        unsigned startSize = startNode->getOutputSize();
        unsigned outputIndex = getOutputIndex();

        torch::Tensor initMatrix;
        long numSpecs;

        bool isOutputNode = ( startIndex == outputIndex );

        // Patches mode avoids materializing full dense A for conv layers (auto_LiRPA approach).
        // GPU: always beneficial. CPU: only when dense identity exceeds threshold.
        static constexpr int64_t PATCHES_SIZE_THRESHOLD = 4096;

        bool usePatchesInit = false;
        std::vector<int64_t> patchOutputShape;

        bool cudaAvailable = torch::cuda::is_available();
        if ( LunaConfiguration::USE_PATCHES_MODE && !isOutputNode )
        {
            auto *convNode = dynamic_cast<BoundedConvNode *>( startNode.get() );
            if ( convNode )
            {
                const auto &shape = convNode->getOutputShape();
                if ( shape.size() == 4 )
                {
                    int64_t outFeatures = (int64_t)shape[1] * (int64_t)shape[2] * (int64_t)shape[3];
                    if ( cudaAvailable || outFeatures > PATCHES_SIZE_THRESHOLD )
                    {
                        patchOutputShape = { (int64_t)shape[0], (int64_t)shape[1],
                                             (int64_t)shape[2], (int64_t)shape[3] };
                        usePatchesInit = true;
                    }
                }
            }

            // BN preserves spatial shape; check Conv predecessor
            if ( !usePatchesInit )
            {
                auto *bnNode = dynamic_cast<BoundedBatchNormNode *>( startNode.get() );
                if ( bnNode && _torchModel->getDependenciesMap().exists( startIndex ) )
                {
                    const auto &deps = _torchModel->getDependencies( startIndex );
                    for ( unsigned i = 0; i < deps.size(); ++i )
                    {
                        if ( !_nodes.exists( deps[i] ) )
                            continue;
                        auto *predConv = dynamic_cast<BoundedConvNode *>( _nodes[deps[i]].get() );
                        if ( predConv )
                        {
                            const auto &shape = predConv->getOutputShape();
                            if ( shape.size() == 4 )
                            {
                                int64_t outFeatures =
                                    (int64_t)shape[1] * (int64_t)shape[2] * (int64_t)shape[3];
                                if ( cudaAvailable || outFeatures > PATCHES_SIZE_THRESHOLD )
                                {
                                    patchOutputShape = { (int64_t)shape[0], (int64_t)shape[1],
                                                         (int64_t)shape[2], (int64_t)shape[3] };
                                    usePatchesInit = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Skip dense identity when patches mode is active
        if ( isOutputNode && C != nullptr )
        {
            initMatrix = preprocessC( *C, outputIndex );
            numSpecs = initMatrix.size( 0 );
            log( Stringf( "backwardFrom() - Using explicitly provided C matrix (preprocessed), "
                          "shape [%ld, %ld, %ld]",
                          initMatrix.size( 0 ), initMatrix.size( 1 ), initMatrix.size( 2 ) ) );
        }
        else if ( isOutputNode && _torchModel->hasSpecificationMatrix() )
        {
            torch::Tensor specMatrix = _torchModel->getSpecificationMatrix();

            initMatrix = preprocessC( specMatrix, outputIndex );
            numSpecs = initMatrix.size( 0 );

            log( Stringf( "backwardFrom() - Using specification matrix from TorchModel "
                          "(preprocessed), shape [%ld, %ld, %ld]",
                          initMatrix.size( 0 ), initMatrix.size( 1 ), initMatrix.size( 2 ) ) );
        }
        else if ( !usePatchesInit )
        {
            initMatrix = preprocessC( torch::Tensor(), startIndex );
            numSpecs = initMatrix.size( 0 );
            if ( isOutputNode )
            {
                log( Stringf( "backwardFrom() - Using identity matrix (no specification matrix), "
                              "shape [%ld, %ld, %ld]",
                              initMatrix.size( 0 ), initMatrix.size( 1 ), initMatrix.size( 2 ) ) );
            }
            else
            {
                log( Stringf( "backwardFrom() - Using identity matrix for intermediate node %u "
                              "(output node is %u), shape [%ld, %ld, %ld]",
                              startIndex, outputIndex, initMatrix.size( 0 ), initMatrix.size( 1 ),
                              initMatrix.size( 2 ) ) );
            }
        }

        if ( usePatchesInit )
        {
            int64_t batch = patchOutputShape[0];
            int64_t outC = patchOutputShape[1];
            int64_t outH = patchOutputShape[2];
            int64_t outW = patchOutputShape[3];

            auto biasOpts =
                torch::TensorOptions().dtype( torch::kFloat32 ).device( _torchModel->getDevice() );

            if ( unstableIndices.empty() )
            {
                auto identityPatches = std::make_shared<Patches>(
                    torch::Tensor(), std::vector<int64_t>{ 1, 1 },
                    std::vector<int64_t>{ 0, 0, 0, 0 }, std::vector<int64_t>{ 0, 0, 0, 0 }, 0,
                    std::nullopt, patchOutputShape, std::vector<int64_t>{}, 1 );

                _lA[startIndex] = BoundA( identityPatches );
                _uA[startIndex] = BoundA( identityPatches );

                numSpecs = outC * outH * outW;
                _lowerBias[startIndex] = torch::zeros( { numSpecs, 1 }, biasOpts );
                _upperBias[startIndex] = torch::zeros( { numSpecs, 1 }, biasOpts );

                log( Stringf( "backwardFrom() - PATCHES identity (dense) for node %u [batch=%lld, "
                              "C=%lld, H=%lld, W=%lld], specs=%ld",
                              startIndex, (long long)batch, (long long)outC, (long long)outH,
                              (long long)outW, numSpecs ) );
            }
            else
            {
                unsigned numUnstable = unstableIndices.size();

                torch::Tensor c_idx = torch::zeros( { (long)numUnstable }, torch::kLong );
                torch::Tensor h_idx = torch::zeros( { (long)numUnstable }, torch::kLong );
                torch::Tensor w_idx = torch::zeros( { (long)numUnstable }, torch::kLong );

                auto c_acc = c_idx.accessor<int64_t, 1>();
                auto h_acc = h_idx.accessor<int64_t, 1>();
                auto w_acc = w_idx.accessor<int64_t, 1>();

                int64_t hw = outH * outW;
                for ( unsigned i = 0; i < numUnstable; ++i )
                {
                    unsigned flatIdx = unstableIndices[i];
                    c_acc[i] = flatIdx / hw;
                    int64_t rem = flatIdx % hw;
                    h_acc[i] = rem / outW;
                    w_acc[i] = rem % outW;
                }

                auto device = _torchModel->getDevice();
                std::vector<torch::Tensor> unstable_idx_vec = {
                    c_idx.to( device ), h_idx.to( device ), w_idx.to( device ) };

                auto identityPatches = std::make_shared<Patches>(
                    torch::Tensor(), std::vector<int64_t>{ 1, 1 },
                    std::vector<int64_t>{ 0, 0, 0, 0 }, std::vector<int64_t>{ 0, 0, 0, 0 }, 0,
                    unstable_idx_vec, patchOutputShape, std::vector<int64_t>{}, 1 );

                _lA[startIndex] = BoundA( identityPatches );
                _uA[startIndex] = BoundA( identityPatches );

                numSpecs = numUnstable;
                _lowerBias[startIndex] = torch::zeros( { (long)numUnstable, 1 }, biasOpts );
                _upperBias[startIndex] = torch::zeros( { (long)numUnstable, 1 }, biasOpts );

                log( Stringf( "backwardFrom() - PATCHES identity (sparse, %u unstable) for node %u",
                              numUnstable, startIndex ) );
            }
        }
        else if ( unstableIndices.empty() )
        {
            _lA[startIndex] = BoundA( initMatrix );
            _uA[startIndex] = BoundA( initMatrix );

            auto options =
                torch::TensorOptions().dtype( torch::kFloat32 ).device( _torchModel->getDevice() );
            _lowerBias[startIndex] = torch::zeros( { numSpecs, 1 }, options );
            _upperBias[startIndex] = torch::zeros( { numSpecs, 1 }, options );
        }
        else
        {
            unsigned numUnstable = unstableIndices.size();
            // Build sparse identity on CPU (accessor requires CPU), then move to device
            torch::Tensor identityMatrix =
                torch::zeros( { (long)numUnstable, 1, (long)startSize },
                              torch::TensorOptions().dtype( torch::kFloat32 ) );

            auto accessor = identityMatrix.accessor<float, 3>();
            for ( unsigned i = 0; i < numUnstable; ++i )
            {
                unsigned idx = unstableIndices[i];
                if ( idx < startSize )
                {
                    accessor[i][0][idx] = 1.0f;
                }
            }
            identityMatrix = identityMatrix.to( _torchModel->getDevice() );

            _lA[startIndex] = BoundA( identityMatrix );
            _uA[startIndex] = BoundA( identityMatrix );

            auto biasOpts =
                torch::TensorOptions().dtype( torch::kFloat32 ).device( _torchModel->getDevice() );
            _lowerBias[startIndex] = torch::zeros( { (long)numUnstable, 1 }, biasOpts );
            _upperBias[startIndex] = torch::zeros( { (long)numUnstable, 1 }, biasOpts );

            log( Stringf(
                "backwardFrom() - Initialized sparse C for node %u with %u unstable neurons",
                startIndex, numUnstable ) );
        }

        stage = "buildReachableSet";
        Set<unsigned> reachable;
        Queue<unsigned> work;
        work.push( startIndex );
        reachable.insert( startIndex );
        while ( !work.empty() )
        {
            unsigned n = work.peak();
            work.pop();
            if ( _torchModel->getDependenciesMap().exists( n ) )
            {
                const auto &deps = _torchModel->getDependencies( n );
                for ( unsigned d : deps )
                {
                    if ( !reachable.exists( d ) )
                    {
                        reachable.insert( d );
                        work.push( d );
                    }
                }
            }
        }

        stage = "computePendingCounts";
        Map<unsigned, unsigned> pending;
        for ( const auto &p : reachable )
        {
            unsigned v = p;
            unsigned cnt = 0;
            auto deps_of_v = _torchModel->getDependents( v );
            for ( unsigned dep : deps_of_v )
            {
                if ( reachable.exists( dep ) )
                    cnt++;
            }
            pending[v] = cnt;
        }

        Queue<unsigned> queue;
        queue.push( startIndex );

        Set<unsigned> processed;

        log( Stringf( "backwardFrom() - Starting scheduled processing from node %u (reachable=%u)",
                      startIndex, reachable.size() ) );

        while ( !queue.empty() )
        {
            unsigned current = queue.peak();
            current_dbg = current;
            stage = "loop";
            queue.pop();

            if ( processed.exists( current ) )
                continue;
            processed.insert( current );

            stage = "nodeLookup";
            if ( !_nodes.exists( current ) )
            {
                throw std::runtime_error(
                    "CROWNAnalysis::backwardFrom - missing node in _nodes map for index " +
                    std::to_string( current ) );
            }
            auto &node = _nodes[current];
            NodeType nodetype = node->getNodeType();

            if ( !_lA.exists( current ) && !_uA.exists( current ) )
            {
                log( Stringf( "No A matrices for node %u, skipping", current ) );
                // Still unblock dependencies to avoid deadlocking join points
                if ( _torchModel->getDependenciesMap().exists( current ) )
                {
                    const Vector<unsigned> &deps = _torchModel->getDependencies( current );
                    for ( unsigned inputIndex : deps )
                    {
                        if ( !reachable.exists( inputIndex ) )
                            continue;
                        if ( pending.exists( inputIndex ) && pending[inputIndex] > 0 )
                        {
                            pending[inputIndex] = pending[inputIndex] - 1;
                            if ( pending[inputIndex] == 0 )
                                queue.push( inputIndex );
                        }
                    }
                }
                continue;
            }

            stage = "getInputBoundsForNode";
            Vector<BoundedTensor<torch::Tensor>> inputBounds = getInputBoundsForNode( current );

            BoundA currentLowerAlpha = _lA.exists( current ) ? _lA[current] : BoundA();
            BoundA currentUpperAlpha = _uA.exists( current ) ? _uA[current] : BoundA();

            Vector<Pair<BoundA, BoundA>> A_matrices;
            torch::Tensor lbias, ubias;

            if ( checkIBPFirstLinear( current ) && !LunaConfiguration::USE_STANDARD_CROWN )
            {
                log( Stringf( "backwardFrom() -  Skipping CROWN backward for first linear layer %u "
                              "(using IBP)",
                              current ) );
                continue;
            }

            if ( !node->isPerturbed() )
            {
                throw std::runtime_error(
                    "CROWNAnalysis::backwardFrom - non-perturbed node carries A coefficients: "
                    "node=" +
                    std::to_string( current ) + " name=" + node->getNodeName().ascii() );
            }

            stage = "node.boundBackward";
            try
            {
                node->boundBackward( currentLowerAlpha, currentUpperAlpha, inputBounds, A_matrices,
                                     lbias, ubias );
            }
            catch ( const std::exception &e )
            {
                auto tensorShape = []( const torch::Tensor &tensor )
                {
                    std::ostringstream shape;
                    shape << "[";
                    for ( int64_t i = 0; i < tensor.dim(); ++i )
                    {
                        if ( i > 0 )
                            shape << ",";
                        shape << tensor.size( i );
                    }
                    shape << "]";
                    return shape.str();
                };
                auto boundShape = [&tensorShape]( const BoundA &A )
                {
                    if ( !A.defined() )
                        return std::string( "undefined" );
                    if ( A.isTensor() )
                        return std::string( "Tensor" ) + tensorShape( A.asTensor() );
                    const auto patches = A.asPatches();
                    return std::string( "Patches" ) + tensorShape( patches->patches );
                };

                std::ostringstream context;
                context << "CROWNAnalysis::backwardFrom node.boundBackward failed node=" << current
                        << " name=" << node->getNodeName().ascii()
                        << " type=" << nodeTypeToString( nodetype )
                        << " lA=" << boundShape( currentLowerAlpha )
                        << " uA=" << boundShape( currentUpperAlpha ) << " inputBounds=";
                for ( unsigned i = 0; i < inputBounds.size(); ++i )
                {
                    if ( i > 0 )
                        context << ";";
                    context << i << ":lower" << tensorShape( inputBounds[i].lower() ) << ",upper"
                            << tensorShape( inputBounds[i].upper() );
                }
                context << ": " << e.what();
                throw std::runtime_error( context.str() );
            }

            stage = "propagateToDependencies";
            if ( _torchModel->getDependenciesMap().exists( current ) )
            {
                const auto &dependencies = _torchModel->getDependencies( current );
                if ( dependencies.size() > 0 && A_matrices.size() != dependencies.size() )
                {
                    throw std::runtime_error(
                        "CROWNAnalysis::backwardFrom - dependency/A count mismatch: node=" +
                        std::to_string( current ) +
                        " dependencies=" + std::to_string( dependencies.size() ) +
                        " A_matrices=" + std::to_string( A_matrices.size() ) );
                }
                if ( dependencies.size() > 0 && ( lbias.defined() || ubias.defined() ) &&
                     !A_matrices[0].first().defined() && !A_matrices[0].second().defined() )
                {
                    throw std::runtime_error(
                        "CROWNAnalysis::backwardFrom - bias has no A carrier on dependency 0: "
                        "node=" +
                        std::to_string( current ) );
                }

                for ( unsigned i = 0; i < dependencies.size(); ++i )
                {
                    unsigned inputIndex = dependencies[i];

                    BoundA new_lA = A_matrices[i].first();
                    BoundA new_uA = A_matrices[i].second();

                    addBound( inputIndex, new_lA, new_uA );

                    // Attach bias to first dependency only to avoid double-counting in multi-input
                    // nodes
                    torch::Tensor propagated_lbias;
                    torch::Tensor propagated_ubias;
                    if ( i == 0 )
                    {
                        propagated_lbias = lbias.defined() ? lbias.clone() : torch::Tensor();
                        propagated_ubias = ubias.defined() ? ubias.clone() : torch::Tensor();

                        auto normalize_bias = [&]( const torch::Tensor &bias,
                                                   const BoundA &A_for_context ) -> torch::Tensor
                        {
                            if ( !bias.defined() || bias.numel() == 0 )
                                return bias;

                            int64_t A_spec = -1, A_batch = -1;
                            if ( A_for_context.defined() && A_for_context.isTensor() )
                            {
                                torch::Tensor A_tensor = A_for_context.asTensor();
                                if ( A_tensor.dim() >= 3 )
                                {
                                    A_spec = A_tensor.size( 0 );
                                    A_batch = A_tensor.size( 1 );
                                }
                                else if ( A_tensor.dim() == 2 )
                                {
                                    A_spec = A_tensor.size( 0 );
                                    A_batch = 1;
                                }
                            }

                            if ( bias.dim() == 0 )
                            {
                                if ( A_spec > 0 && A_batch > 0 )
                                {
                                    return bias.unsqueeze( 0 ).unsqueeze( 0 ).expand(
                                        { A_spec, A_batch } );
                                }
                                else
                                {
                                    return bias.unsqueeze( 0 ).unsqueeze( 0 );
                                }
                            }

                            if ( bias.dim() == 1 )
                            {
                                if ( A_spec > 0 && A_batch > 0 )
                                {
                                    if ( bias.size( 0 ) == A_spec )
                                    {
                                        return bias.unsqueeze( 1 ).expand( { A_spec, A_batch } );
                                    }
                                    else if ( bias.size( 0 ) == A_batch )
                                    {
                                        torch::Tensor result = bias.unsqueeze( 0 );
                                        if ( A_spec > 1 )
                                        {
                                            result = result.expand( { A_spec, A_batch } );
                                        }
                                        return result;
                                    }
                                    else
                                    {
                                        return bias.unsqueeze( 1 );
                                    }
                                }
                                else
                                {
                                    return bias.unsqueeze( 1 );
                                }
                            }

                            if ( bias.dim() == 2 )
                            {
                                if ( A_spec > 0 && A_batch > 0 )
                                {
                                    if ( bias.size( 0 ) == A_spec && bias.size( 1 ) == 1 &&
                                         A_batch > 1 )
                                    {
                                        return bias.expand( { A_spec, A_batch } );
                                    }
                                    if ( bias.size( 1 ) == A_batch && bias.size( 0 ) == 1 &&
                                         A_spec > 1 )
                                    {
                                        return bias.expand( { A_spec, A_batch } );
                                    }
                                    if ( bias.size( 0 ) == A_spec && bias.size( 1 ) == A_batch )
                                    {
                                        return bias;
                                    }
                                    else if ( bias.size( 0 ) == A_batch &&
                                              bias.size( 1 ) == A_spec )
                                    {
                                        return bias.transpose( 0, 1 );
                                    }
                                    else
                                    {
                                        if ( bias.numel() == A_spec * A_batch )
                                        {
                                            return bias.reshape( { A_spec, A_batch } );
                                        }
                                        return bias;
                                    }
                                }
                                else
                                {
                                    return bias;
                                }
                            }

                            return bias;
                        };

                        BoundA A_for_context =
                            currentLowerAlpha.defined() ? currentLowerAlpha : currentUpperAlpha;

                        if ( _lowerBias.exists( current ) )
                        {
                            auto cur = _lowerBias[current];
                            if ( propagated_lbias.defined() && cur.defined() )
                            {
                                torch::Tensor norm_propagated =
                                    normalize_bias( propagated_lbias, A_for_context );
                                torch::Tensor norm_cur = normalize_bias( cur, A_for_context );

                                propagated_lbias = norm_propagated + norm_cur;
                            }
                            else
                            {
                                propagated_lbias = cur.defined()
                                                       ? normalize_bias( cur, A_for_context )
                                                       : propagated_lbias;
                            }
                        }
                        else
                        {
                            propagated_lbias = normalize_bias( propagated_lbias, A_for_context );
                        }

                        if ( _upperBias.exists( current ) )
                        {
                            auto cur = _upperBias[current];
                            if ( propagated_ubias.defined() && cur.defined() )
                            {
                                torch::Tensor norm_propagated =
                                    normalize_bias( propagated_ubias, A_for_context );
                                torch::Tensor norm_cur = normalize_bias( cur, A_for_context );

                                propagated_ubias = norm_propagated + norm_cur;
                            }
                            else
                            {
                                propagated_ubias = cur.defined()
                                                       ? normalize_bias( cur, A_for_context )
                                                       : propagated_ubias;
                            }
                        }
                        else
                        {
                            propagated_ubias = normalize_bias( propagated_ubias, A_for_context );
                        }
                    }

                    addBias( inputIndex, propagated_lbias, propagated_ubias );
                }

                const Vector<unsigned> &deps = _torchModel->getDependencies( current );
                for ( unsigned inputIndex : deps )
                {
                    if ( !reachable.exists( inputIndex ) )
                        continue;
                    if ( pending.exists( inputIndex ) && pending[inputIndex] > 0 )
                    {
                        pending[inputIndex] = pending[inputIndex] - 1;
                        if ( pending[inputIndex] == 0 )
                        {
                            queue.push( inputIndex );
                        }
                    }
                }
            }

            if ( nodetype != NodeType::INPUT )
            {
                if ( _lA.exists( current ) )
                    _lA.erase( current );
                if ( _uA.exists( current ) )
                    _uA.erase( current );
                if ( _lowerBias.exists( current ) )
                    _lowerBias.erase( current );
                if ( _upperBias.exists( current ) )
                    _upperBias.erase( current );
            }
        }

        log( "backwardFrom() - Completed." );
    }
    catch ( const CommonError &e )
    {
        std::ostringstream oss;
        oss << "CROWNAnalysis::backwardFrom CommonError code=" << e.getCode() << " stage=" << stage
            << " startIndex=" << startIndex << " current=" << current_dbg;
        const char *msg = e.getUserMessage();
        if ( msg && msg[0] != '\0' )
            oss << " message=" << msg;
        throw std::runtime_error( oss.str() );
    }
    catch ( const Error &e )
    {
        std::ostringstream oss;
        oss << "CROWNAnalysis::backwardFrom Error class=" << e.getErrorClass()
            << " code=" << e.getCode() << " stage=" << stage << " startIndex=" << startIndex
            << " current=" << current_dbg;
        const char *msg = e.getUserMessage();
        if ( msg && msg[0] != '\0' )
            oss << " message=" << msg;
        throw std::runtime_error( oss.str() );
    }
    catch ( const std::exception &e )
    {
        throw std::runtime_error( std::string( "CROWNAnalysis::backwardFrom exception stage=" ) +
                                  stage + ": " + e.what() );
    }
}

void CROWNAnalysis::clearCrownState()
{
    _lA.clear();
    _uA.clear();
    _lowerBias.clear();
    _upperBias.clear();
}

void CROWNAnalysis::concretizeNode( unsigned startIndex, const Vector<unsigned> &unstableIndices )
{
    log( Stringf( "concretizeNode() - Starting concretization for node %u", startIndex ) );

    int inputIndex = -1;
    for ( const auto &p : _nodes )
    {
        if ( p.second->getNodeType() == NodeType::INPUT )
        {
            inputIndex = static_cast<int>( p.first );
            break;
        }
    }

    log( Stringf( "concretizeNode() - Input node index: %d", inputIndex ) );

    if ( inputIndex < 0 )
    {
        log( Stringf( "concretizeNode() - No input node found, using IBP bounds for node %u",
                      startIndex ) );
        if ( _ibpBounds.exists( startIndex ) )
        {
            _concreteBounds[startIndex] = _ibpBounds[startIndex];
            _torchModel->setConcreteBounds( startIndex, _ibpBounds[startIndex] );
            if ( _nodes.exists( startIndex ) )
            {
                _nodes[startIndex]->setBounds( _ibpBounds[startIndex].lower(),
                                               _ibpBounds[startIndex].upper() );
            }
            log( Stringf( "concretizeNode() - Stored IBP bounds as concrete bounds for node %u",
                          startIndex ) );
        }
        else
        {
            log( Stringf( "concretizeNode() - WARNING: No IBP bounds available for node %u",
                          startIndex ) );
        }
        return;
    }

    torch::Tensor inputLower, inputUpper;
    if ( _torchModel->hasInputBounds() )
    {
        inputLower = _torchModel->getInputLowerBounds();
        inputUpper = _torchModel->getInputUpperBounds();
    }
    else
    {
        unsigned inputSize = _torchModel->getInputSize();
        auto options =
            torch::TensorOptions().dtype( torch::kFloat32 ).device( _torchModel->getDevice() );
        inputLower = torch::zeros( { (long)inputSize }, options );
        inputUpper = torch::ones( { (long)inputSize }, options );
    }
    // Keep the lane dimension, the contraction reads one box row per lane
    inputLower = _torchModel->toBatchedInput( inputLower.to( torch::kFloat32 ) ).flatten( 1 );
    inputUpper = _torchModel->toBatchedInput( inputUpper.to( torch::kFloat32 ) ).flatten( 1 );

    log( Stringf( "concretizeNode() - Checking for A matrices at input node %d", inputIndex ) );
    log( Stringf( "concretizeNode() - _lA.exists(%d)=%d, _uA.exists(%d)=%d", inputIndex,
                  _lA.exists( inputIndex ), inputIndex, _uA.exists( inputIndex ) ) );

    if ( !_lA.exists( inputIndex ) && !_uA.exists( inputIndex ) )
    {
        log(
            Stringf( "concretizeNode() - No A matrices at input node, using IBP bounds for node %u",
                     startIndex ) );
        if ( _ibpBounds.exists( startIndex ) )
        {
            _concreteBounds[startIndex] = _ibpBounds[startIndex];
            _torchModel->setConcreteBounds( startIndex, _ibpBounds[startIndex] );
            if ( _nodes.exists( startIndex ) )
            {
                _nodes[startIndex]->setBounds( _ibpBounds[startIndex].lower(),
                                               _ibpBounds[startIndex].upper() );
            }
            log( Stringf( "concretizeNode() - Stored IBP bounds as concrete bounds for node %u",
                          startIndex ) );
        }
        else
        {
            log( Stringf( "concretizeNode() - WARNING: No IBP bounds available for node %u",
                          startIndex ) );
        }
        return;
    }

    BoundA lA_bound = _lA.exists( inputIndex ) ? _lA[inputIndex] : BoundA();
    BoundA uA_bound = _uA.exists( inputIndex ) ? _uA[inputIndex] : BoundA();

    torch::Tensor lBias =
        _lowerBias.exists( inputIndex ) ? _lowerBias[inputIndex] : torch::Tensor();
    torch::Tensor uBias =
        _upperBias.exists( inputIndex ) ? _upperBias[inputIndex] : torch::Tensor();

    bool lPatches = lA_bound.isPatches();
    bool uPatches = uA_bound.isPatches();

    torch::Tensor concreteLower, concreteUpper;

    if ( lPatches || uPatches )
    {
        // Patches-mode concretization via inplace_unfold + einsum (auto_LiRPA style)
        auto concretize_patches_side = [&]( const BoundA &A_bound, const torch::Tensor &bias,
                                            int sign ) -> torch::Tensor
        {
            if ( !A_bound.defined() )
                return torch::Tensor();

            auto patches_ptr = A_bound.asPatches();
            if ( !patches_ptr || patches_ptr->input_shape.empty() )
            {
                throw std::runtime_error(
                    "CROWNAnalysis::concretizeNode - Patches without input_shape" );
            }

            // input_shape is one sample behind a leading batch of 1, the box has a row per lane
            std::vector<int64_t> in_shape = patches_ptr->input_shape;
            in_shape[0] = inputLower.size( 0 );
            torch::Tensor xL_img = inputLower.reshape( in_shape ).to( torch::kFloat32 );
            torch::Tensor xU_img = inputUpper.reshape( in_shape ).to( torch::kFloat32 );

            torch::Tensor center = ( xU_img + xL_img ) / 2.0f;
            torch::Tensor diff = torch::max( xU_img - center, center - xL_img );
            // Enclose both FP32 endpoints despite midpoint rounding
            diff = torch::nextafter(
                diff, torch::full_like( diff, std::numeric_limits<float>::infinity() ) );

            torch::Tensor bound = patches_ptr->matmul( center );
            torch::Tensor bound_diff = patches_ptr->matmul( diff, /*patch_abs=*/true );

            if ( sign == 1 )
            {
                bound = bound + bound_diff;
            }
            else
            {
                bound = bound - bound_diff;
            }

            if ( bound.dim() > 2 )
            {
                bound = bound.reshape( { bound.size( 0 ), -1 } );
            }

            if ( bias.defined() )
            {
                torch::Tensor b = bias.to( torch::kFloat32 );
                if ( b.dim() == 2 )
                {
                    b = b.transpose( 0, 1 );
                }
                else if ( b.dim() == 1 )
                {
                    b = b.unsqueeze( 0 );
                }
                bound = bound + b;
            }

            return bound;
        };

        if ( lPatches )
        {
            concreteLower = concretize_patches_side( lA_bound, lBias, -1 );
        }
        if ( uPatches )
        {
            concreteUpper = concretize_patches_side( uA_bound, uBias, +1 );
        }

        if ( !lPatches && lA_bound.defined() )
        {
            torch::Tensor lA = lA_bound.asTensor();
            concreteLower = computeConcreteLowerBound( lA, lBias, inputLower, inputUpper );
        }
        if ( !uPatches && uA_bound.defined() )
        {
            torch::Tensor uA = uA_bound.asTensor();
            concreteUpper = computeConcreteUpperBound( uA, uBias, inputLower, inputUpper );
        }
    }
    else
    {
        torch::Tensor lA = lA_bound.asTensor();
        torch::Tensor uA = uA_bound.asTensor();

        computeConcreteBounds( lA, uA, lBias, uBias, inputLower, inputUpper, concreteLower,
                               concreteUpper );
    }

    log( Stringf(
        "concretizeNode() - Computed concrete bounds: lower.defined()=%d, upper.defined()=%d",
        concreteLower.defined(), concreteUpper.defined() ) );

    if ( concreteLower.defined() && concreteUpper.defined() )
    {
        if ( !unstableIndices.empty() )
        {
            if ( _ibpBounds.exists( startIndex ) )
            {
                auto ibp = _ibpBounds[startIndex];
                if ( ibp.lower().defined() && ibp.upper().defined() )
                {
                    torch::Tensor fullLower = ibp.lower().detach().clone();
                    torch::Tensor fullUpper = ibp.upper().detach().clone();

                    auto indexOptions =
                        torch::TensorOptions().dtype( torch::kLong ).device( fullLower.device() );
                    torch::Tensor indices = torch::tensor(
                        std::vector<int64_t>( unstableIndices.begin(), unstableIndices.end() ),
                        indexOptions );

                    const auto originalShape = fullLower.sizes().vec();
                    const int64_t lanes = inputLower.size( 0 );
                    torch::Tensor fullLowerRows = fullLower.reshape( { lanes, -1 } );
                    torch::Tensor fullUpperRows = fullUpper.reshape( { lanes, -1 } );

                    // Each lane writes its own row at the shared unstable columns
                    torch::Tensor concreteLowerRows = concreteLower.reshape( { lanes, -1 } );
                    torch::Tensor concreteUpperRows = concreteUpper.reshape( { lanes, -1 } );

                    // Preserve gradients for alpha-CROWN; detach otherwise
                    bool retainSparseGrad = ( LunaConfiguration::ANALYSIS_METHOD ==
                                              LunaConfiguration::AnalysisMethod::AlphaCROWN );
                    if ( !retainSparseGrad )
                    {
                        concreteLowerRows = concreteLowerRows.detach();
                        concreteUpperRows = concreteUpperRows.detach();
                    }
                    // A neuron stable in this lane keeps its IBP bound, as in a single call, the
                    // shared set only holds it because another lane has it unstable
                    const auto ibpLowerAtSet = fullLowerRows.index_select( 1, indices );
                    const auto ibpUpperAtSet = fullUpperRows.index_select( 1, indices );
                    const auto unstableInLane = ( ibpLowerAtSet < 0 ) & ( ibpUpperAtSet > 0 );
                    concreteLowerRows =
                        torch::where( unstableInLane, concreteLowerRows, ibpLowerAtSet );
                    concreteUpperRows =
                        torch::where( unstableInLane, concreteUpperRows, ibpUpperAtSet );

                    fullLowerRows = fullLowerRows.index_copy( 1, indices, concreteLowerRows );
                    fullUpperRows = fullUpperRows.index_copy( 1, indices, concreteUpperRows );

                    concreteLower = fullLowerRows.reshape( originalShape );
                    concreteUpper = fullUpperRows.reshape( originalShape );
                }
                else
                {
                    log( Stringf( "concretizeNode() - WARNING: Sparse CROWN computed but no IBP "
                                  "base for node %u",
                                  startIndex ) );
                    return;
                }
            }
            else
            {
                log( Stringf(
                    "concretizeNode() - WARNING: Sparse CROWN computed but no IBP base for node %u",
                    startIndex ) );
                return;
            }
        }

        auto refIt = _referenceBounds.find( startIndex );
        if ( refIt != _referenceBounds.end() )
        {
            const auto &refLower = refIt->second.first;
            const auto &refUpper = refIt->second.second;
            const bool retainGradients =
                LunaConfiguration::ANALYSIS_METHOD == LunaConfiguration::AnalysisMethod::AlphaCROWN;

            if ( refLower.defined() && refLower.sizes() == concreteLower.sizes() )
            {
                torch::Tensor clamped = torch::max( refLower, concreteLower );
                concreteLower = retainGradients
                                    ? clamped.detach() - concreteLower.detach() + concreteLower
                                    : clamped;
            }
            if ( refUpper.defined() && refUpper.sizes() == concreteUpper.sizes() )
            {
                torch::Tensor clamped = torch::min( refUpper, concreteUpper );
                concreteUpper = retainGradients
                                    ? concreteUpper - ( concreteUpper.detach() - clamped.detach() )
                                    : clamped;
            }
        }

        // IBP intersection disabled — can be unsound if IBP bounds have shape/layout mismatches
        const bool kIntersectWithIBP = false;
        if ( kIntersectWithIBP && _ibpBounds.exists( startIndex ) )
        {
            auto ibp = _ibpBounds[startIndex];
            if ( ibp.lower().defined() && ibp.upper().defined() )
            {
                concreteLower = torch::max( concreteLower, ibp.lower() );
                concreteUpper = torch::min( concreteUpper, ibp.upper() );
                log( Stringf( "concretizeNode() - Intersected with IBP bounds for node %u",
                              startIndex ) );
            }
        }

        BoundedTensor<torch::Tensor> concreteBounds( concreteLower, concreteUpper );
        _concreteBounds[startIndex] = concreteBounds;
        _torchModel->setConcreteBounds( startIndex, concreteBounds );
        if ( _nodes.exists( startIndex ) )
        {
            _nodes[startIndex]->setBounds( concreteLower, concreteUpper );
        }
        log( Stringf( "concretizeNode() - Stored CROWN concrete bounds for node %u", startIndex ) );
    }
    else
    {
        log( Stringf( "concretizeNode() - CROWN bounds undefined, falling back to IBP for node %u",
                      startIndex ) );
        if ( _ibpBounds.exists( startIndex ) )
        {
            _concreteBounds[startIndex] = _ibpBounds[startIndex];
            _torchModel->setConcreteBounds( startIndex, _ibpBounds[startIndex] );
            if ( _nodes.exists( startIndex ) )
            {
                _nodes[startIndex]->setBounds( _ibpBounds[startIndex].lower(),
                                               _ibpBounds[startIndex].upper() );
            }
        }
    }

    log( Stringf( "concretizeNode() - Finished, _concreteBounds.exists(%u)=%d", startIndex,
                  _concreteBounds.exists( startIndex ) ) );
}

torch::Tensor CROWNAnalysis::preprocessC( const torch::Tensor &C, unsigned startIndex )
{
    if ( !_nodes.exists( startIndex ) )
    {
        throw std::runtime_error( "CROWNAnalysis::preprocessC: startIndex not found in nodes" );
    }

    auto &startNode = _nodes[startIndex];
    unsigned outputSize = startNode->getOutputSize();

    if ( outputSize == 0 )
    {
        outputSize = 1;
    }

    if ( C.numel() == 0 )
    {
        auto opts =
            torch::TensorOptions().dtype( torch::kFloat32 ).device( _torchModel->getDevice() );
        torch::Tensor identity = torch::eye( outputSize, opts );
        torch::Tensor result = identity.unsqueeze( 1 );
        return result;
    }

    long batch_size, output_dim;

    if ( C.dim() == 3 )
    {

        bool isBatchSpecFormat = ( C.size( 0 ) == 1 && C.size( 1 ) > 1 );

        torch::Tensor C_processed;
        if ( isBatchSpecFormat )
        {
            batch_size = C.size( 0 );
            output_dim = C.size( 1 );
            C_processed = C.transpose( 0, 1 );
        }
        else
        {
            batch_size = C.size( 1 );
            output_dim = C.size( 0 );
            C_processed = C;
        }

        if ( C_processed.size( 2 ) != (long)outputSize )
        {
            throw std::runtime_error(
                Stringf( "CROWNAnalysis::preprocessC: C matrix output dimension (%ld) does not "
                         "match node output size (%u)",
                         C_processed.size( 2 ), outputSize )
                    .ascii() );
        }

        return C_processed;
    }
    else if ( C.dim() == 2 )
    {
        batch_size = C.size( 0 );
        output_dim = C.size( 1 );

        torch::Tensor C_transformed = C.transpose( 0, 1 );
        std::vector<int64_t> new_shape = { output_dim, batch_size, (long)outputSize };
        return C_transformed.reshape( new_shape );
    }
    else if ( C.dim() == 1 )
    {
        batch_size = 1;
        output_dim = C.size( 0 );

        torch::Tensor C_transformed = C.unsqueeze( 1 );
        std::vector<int64_t> new_shape = { output_dim, batch_size, (long)outputSize };
        return C_transformed.reshape( new_shape );
    }
    else
    {
        throw std::runtime_error( "CROWNAnalysis::preprocessC: unsupported C matrix dimension" );
    }
}

void CROWNAnalysis::concretizeBounds()
{
    concretizeNode( getOutputIndex() );
}

Vector<BoundedTensor<torch::Tensor>> CROWNAnalysis::getInputBoundsForNode( unsigned nodeIndex )
{
    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    const torch::Device device = _torchModel->getDevice();

    if ( _torchModel->getDependenciesMap().exists( nodeIndex ) &&
         !_torchModel->getDependencies( nodeIndex ).empty() )
    {

        for ( unsigned i = 0; i < _torchModel->getDependencies( nodeIndex ).size(); ++i )
        {
            unsigned inputIndex = _torchModel->getDependencies( nodeIndex )[i];

            auto node = _nodes[inputIndex];
            unsigned outputSize = node->getOutputSize();
            torch::Tensor lower, upper;

            std::string source = "unknown";
            if ( node->getNodeType() == NodeType::INPUT )
            {
                if ( _torchModel->hasInputBounds() )
                {
                    lower = _torchModel->getInputLowerBounds();
                    upper = _torchModel->getInputUpperBounds();
                }
                else
                {
                    auto options = torch::TensorOptions()
                                       .dtype( torch::kFloat32 )
                                       .device( _torchModel->getDevice() );
                    lower = torch::zeros( { (long)outputSize }, options );
                    upper = torch::ones( { (long)outputSize }, options );
                }
                source = "input";
            }
            else
            {
                if ( node->hasBounds() )
                {
                    lower = node->getLower();
                    upper = node->getUpper();
                    if ( _ibpBounds.exists( inputIndex ) )
                    {
                        lower = torch::maximum( lower, _ibpBounds[inputIndex].lower() );
                        upper = torch::minimum( upper, _ibpBounds[inputIndex].upper() );
                    }
                    source = "node";
                }
                else if ( LunaConfiguration::USE_STANDARD_CROWN &&
                          _concreteBounds.exists( inputIndex ) )
                {
                    lower = _concreteBounds[inputIndex].lower();
                    upper = _concreteBounds[inputIndex].upper();
                    source = "crown";
                }
                else if ( _ibpBounds.exists( inputIndex ) )
                {
                    lower = _ibpBounds[inputIndex].lower();
                    upper = _ibpBounds[inputIndex].upper();
                    source = "ibp";
                }
                else
                {
                    auto options = torch::TensorOptions().dtype( torch::kFloat32 ).device( device );
                    lower = torch::zeros( { (long)outputSize }, options );
                    upper = torch::ones( { (long)outputSize }, options );
                    source = "default";
                }
            }
            if ( lower.defined() && lower.device() != device )
            {
                lower = lower.to( device );
            }
            if ( upper.defined() && upper.device() != device )
            {
                upper = upper.to( device );
            }
            inputBounds.append( BoundedTensor<torch::Tensor>( lower, upper ) );
        }
    }
    else
    {
        log( Stringf( "getInputBoundsForNode() - Node %u has no dependencies", nodeIndex ) );
    }
    log( Stringf( "getInputBoundsForNode() - Completed for node %u with %u input bounds", nodeIndex,
                  inputBounds.size() ) );
    return inputBounds;
}

namespace
{

/// A is (spec, batch, *features) and a 2-D A is one lane (spec, features).
torch::Tensor concretizationA( const torch::Tensor &A )
{
    if ( A.dim() > 3 )
        return A.flatten( 2 );
    if ( A.dim() == 3 )
        return A;
    if ( A.dim() == 2 )
        return A.unsqueeze( 1 );
    throw std::runtime_error( "CROWNAnalysis concretization needs A of rank 2 or more" );
}

/// The box is one row of features per lane, or one row shared by every lane.
torch::Tensor concretizationX( const torch::Tensor &x, int64_t features )
{
    if ( features <= 0 || x.numel() % features != 0 )
    {
        throw std::runtime_error( "CROWNAnalysis concretization: input has " +
                                  std::to_string( x.numel() ) + " elements for " +
                                  std::to_string( features ) + " features" );
    }
    return x.reshape( { -1, features } );
}

/// Bias is (spec, batch), or a single column shared by every lane.
torch::Tensor concretizationBias( const torch::Tensor &bias, int64_t spec )
{
    if ( !bias.defined() )
        return torch::zeros( { spec, 1 }, torch::kFloat32 );
    torch::Tensor shaped = bias.numel() == spec ? bias.reshape( { spec, 1 } ) : bias;
    if ( shaped.dim() != 2 || shaped.size( 0 ) != spec )
    {
        throw std::runtime_error( "CROWNAnalysis concretization: bias does not match A" );
    }
    return shaped;
}

/// Every operand carries the lane count or 1, and the result has the larger of the two.
int64_t commonLaneCount( int64_t aLanes, int64_t boxLanes, int64_t biasLanes )
{
    int64_t lanes = 1;
    for ( const int64_t count : { aLanes, boxLanes, biasLanes } )
    {
        if ( count != 1 && lanes != 1 && count != lanes )
        {
            throw std::runtime_error( "CROWNAnalysis concretization: lane counts " +
                                      std::to_string( aLanes ) + ", " + std::to_string( boxLanes ) +
                                      " and " + std::to_string( biasLanes ) + " do not agree" );
        }
        lanes = std::max( lanes, count );
    }
    return lanes;
}

/// Sum over features of Apos * xPos + Aneg * xNeg, plus bias, as (batch, spec).
torch::Tensor concretize( const torch::Tensor &lA, const torch::Tensor &bias,
                          const torch::Tensor &xForPositive, const torch::Tensor &xForNegative )
{
    torch::Tensor A = concretizationA( lA.to( torch::kFloat32 ) );
    const int64_t spec = A.size( 0 );
    const int64_t features = A.size( 2 );
    torch::Tensor xPos = concretizationX( xForPositive.to( torch::kFloat32 ), features );
    torch::Tensor xNeg = concretizationX( xForNegative.to( torch::kFloat32 ), features );
    torch::Tensor b = concretizationBias( bias.to( torch::kFloat32 ), spec );
    (void)commonLaneCount( A.size( 1 ), xPos.size( 0 ), b.size( 1 ) );

    // Lane b contracts its own A slice with its own box row, so lanes never mix
    torch::Tensor term = torch::einsum( "sbf,bf->bs", { torch::clamp_min( A, 0 ), xPos } ) +
                         torch::einsum( "sbf,bf->bs", { torch::clamp_max( A, 0 ), xNeg } );
    return term + b.transpose( 0, 1 );
}

} // namespace

torch::Tensor CROWNAnalysis::computeConcreteLowerBound( const torch::Tensor &lA,
                                                        const torch::Tensor &lBias,
                                                        const torch::Tensor &xLower,
                                                        const torch::Tensor &xUpper )
{
    if ( !lA.defined() )
        return torch::Tensor();

    // LB = bias + Apos * xL + Aneg * xU
    return concretize( lA, lBias, xLower, xUpper );
}

torch::Tensor CROWNAnalysis::computeConcreteUpperBound( const torch::Tensor &uA,
                                                        const torch::Tensor &uBias,
                                                        const torch::Tensor &xLower,
                                                        const torch::Tensor &xUpper )
{
    if ( !uA.defined() )
        return torch::Tensor();

    // UB = bias + Apos * xU + Aneg * xL
    return concretize( uA, uBias, xUpper, xLower );
}

void CROWNAnalysis::computeConcreteBounds( const torch::Tensor &lA, const torch::Tensor &uA,
                                           const torch::Tensor &lBias, const torch::Tensor &uBias,
                                           const torch::Tensor &nodeLower,
                                           const torch::Tensor &nodeUpper,
                                           torch::Tensor &concreteLower,
                                           torch::Tensor &concreteUpper )
{
    concreteLower = computeConcreteLowerBound( lA, lBias, nodeLower, nodeUpper );
    concreteUpper = computeConcreteUpperBound( uA, uBias, nodeLower, nodeUpper );
}

unsigned CROWNAnalysis::getOutputIndex() const
{
    unsigned outputIndex = 0;
    for ( const auto &pair : _nodes )
    {
        if ( pair.first > outputIndex )
        {
            outputIndex = pair.first;
        }
    }
    return outputIndex;
}

BoundA CROWNAnalysis::addA( const BoundA &A1, const BoundA &A2 )
{
    if ( !A1.defined() )
        return A2;
    if ( !A2.defined() )
        return A1;

    if ( A1.isTensor() && A2.isTensor() )
    {
        torch::Tensor t1 = A1.asTensor();
        torch::Tensor t2 = A2.asTensor();

        if ( t1.dim() > 3 )
            t1 = t1.flatten( 2 );
        if ( t2.dim() > 3 )
            t2 = t2.flatten( 2 );

        if ( t1.sizes() != t2.sizes() )
        {
            torch::Tensor t1_norm = t1;
            torch::Tensor t2_norm = t2;

            if ( t1.dim() != t2.dim() )
            {
                if ( t1.dim() > t2.dim() )
                {
                    auto t1_sizes = t1.sizes().vec();
                    auto t2_sizes = t2.sizes().vec();
                    while ( t2_sizes.size() < t1_sizes.size() )
                    {
                        t2_sizes.insert( t2_sizes.begin(), 1 );
                    }
                    t2_norm = t2.view( t2_sizes );
                }
                else
                {
                    auto t1_sizes = t1.sizes().vec();
                    auto t2_sizes = t2.sizes().vec();
                    while ( t1_sizes.size() < t2_sizes.size() )
                    {
                        t1_sizes.insert( t1_sizes.begin(), 1 );
                    }
                    t1_norm = t1.view( t1_sizes );
                }
            }

            if ( t1_norm.sizes() != t2_norm.sizes() )
            {
                try
                {
                    return BoundA( t1_norm + t2_norm );
                }
                catch ( const std::exception &e )
                {
                    std::string shape1_str = "[";
                    for ( int i = 0; i < t1.dim(); ++i )
                    {
                        shape1_str += std::to_string( t1.size( i ) );
                        if ( i < t1.dim() - 1 )
                            shape1_str += ", ";
                    }
                    shape1_str += "]";

                    std::string shape2_str = "[";
                    for ( int i = 0; i < t2.dim(); ++i )
                    {
                        shape2_str += std::to_string( t2.size( i ) );
                        if ( i < t2.dim() - 1 )
                            shape2_str += ", ";
                    }
                    shape2_str += "]";

                    throw std::runtime_error(
                        "CROWNAnalysis::addA - A shape mismatch: A1.shape=" + shape1_str +
                        " vs A2.shape=" + shape2_str +
                        " (after normalization, broadcasting failed: " + std::string( e.what() ) +
                        ")" );
                }
            }

            return BoundA( t1_norm + t2_norm );
        }

        return BoundA( t1 + t2 );
    }
    else if ( A1.isPatches() && A2.isPatches() )
    {
        return BoundA( A1.asPatches()->add( A2.asPatches() ) );
    }
    else
    {
        const torch::Tensor denseSibling = A1.isTensor() ? A1.asTensor() : A2.asTensor();
        auto toTensor = [&denseSibling]( const BoundA &A ) -> torch::Tensor
        {
            if ( A.isTensor() )
            {
                torch::Tensor tensor = A.asTensor();
                return tensor.dim() > 3 ? tensor.flatten( 2 ) : tensor;
            }
            auto p = A.asPatches();
            if ( p->identity == 1 && !p->output_shape.empty() )
            {
                int64_t C = p->output_shape[1];
                int64_t H = p->output_shape[2];
                int64_t W = p->output_shape[3];
                int64_t n = C * H * W;
                auto opts = denseSibling.options();
                if ( p->unstable_idx.has_value() )
                {
                    auto &idx = p->unstable_idx.value();
                    int64_t batch = p->output_shape[0];
                    torch::Tensor eye = torch::eye( n, opts );
                    torch::Tensor flat_idx = idx[0] * ( H * W ) + idx[1] * W + idx[2];
                    torch::Tensor selected = eye.index_select( 0, flat_idx );
                    return selected.unsqueeze( 1 ).expand( { -1, batch, -1 } );
                }
                else
                {
                    int64_t batch = p->output_shape[0];
                    torch::Tensor eye = torch::eye( n, opts );
                    return eye.unsqueeze( 1 ).expand( { -1, batch, -1 } );
                }
            }
            if ( !p->input_shape.empty() )
            {
                torch::Tensor mat = p->to_matrix( p->input_shape );
                if ( mat.dim() == 5 )
                {
                    mat = mat.permute( { 1, 0, 2, 3, 4 } ).flatten( 2 );
                }
                else if ( mat.dim() > 3 )
                {
                    mat = mat.flatten( 2 );
                }
                return mat.to( denseSibling.options() );
            }
            throw std::runtime_error( "CROWNAnalysis::addA - Cannot convert Patches to matrix: "
                                      "input_shape empty and not identity" );
        };
        torch::Tensor t1 = toTensor( A1 );
        torch::Tensor t2 = toTensor( A2 );
        if ( t1.sizes() != t2.sizes() )
        {
            auto shapeString = []( const torch::Tensor &tensor )
            {
                std::string result = "[";
                for ( int64_t i = 0; i < tensor.dim(); ++i )
                {
                    if ( i > 0 )
                        result += ", ";
                    result += std::to_string( tensor.size( i ) );
                }
                return result + "]";
            };
            throw std::runtime_error(
                "CROWNAnalysis::addA - mixed Patches/Tensor shape mismatch after "
                "canonicalization: A1.shape=" +
                shapeString( t1 ) + " vs A2.shape=" + shapeString( t2 ) );
        }
        return BoundA( t1 + t2 );
    }
}

void CROWNAnalysis::addBound( unsigned nodeIndex, const BoundA &lA, const BoundA &uA )
{

    if ( _lA.exists( nodeIndex ) )
    {
        _lA[nodeIndex] = addA( _lA[nodeIndex], lA );
    }
    else
    {
        _lA[nodeIndex] = lA;
    }

    if ( _uA.exists( nodeIndex ) )
    {
        _uA[nodeIndex] = addA( _uA[nodeIndex], uA );
    }
    else
    {
        _uA[nodeIndex] = uA;
    }
}

void CROWNAnalysis::addBias( unsigned nodeIndex, const torch::Tensor &lBias,
                             const torch::Tensor &uBias )
{
    auto normalize_bias_shape = []( const torch::Tensor &bias ) -> torch::Tensor
    {
        if ( !bias.defined() || bias.numel() == 0 )
            return bias;

        if ( bias.dim() == 2 )
        {
            return bias;
        }

        if ( bias.dim() == 1 )
        {
            return bias.unsqueeze( 1 );
        }

        if ( bias.dim() == 0 )
        {
            return bias.unsqueeze( 0 ).unsqueeze( 0 );
        }

        if ( bias.numel() > 0 )
        {
            if ( bias.size( -1 ) == 1 && bias.dim() > 1 )
            {
                int64_t spec_size = bias.numel() / bias.size( -1 );
                return bias.reshape( { spec_size, 1 } );
            }
            else
            {
                return bias.flatten().unsqueeze( 1 );
            }
        }

        return bias;
    };

    if ( lBias.defined() && lBias.numel() > 0 )
    {
        torch::Tensor normalized_lBias = normalize_bias_shape( lBias );

        if ( _lowerBias.exists( nodeIndex ) )
        {
            torch::Tensor existing = _lowerBias[nodeIndex];
            torch::Tensor normalized_existing = normalize_bias_shape( existing );

            _lowerBias[nodeIndex] = normalized_existing + normalized_lBias;
        }
        else
        {
            _lowerBias[nodeIndex] = normalized_lBias;
        }
    }
    if ( uBias.defined() && uBias.numel() > 0 )
    {
        torch::Tensor normalized_uBias = normalize_bias_shape( uBias );

        if ( _upperBias.exists( nodeIndex ) )
        {
            torch::Tensor existing = _upperBias[nodeIndex];
            torch::Tensor normalized_existing = normalize_bias_shape( existing );

            _upperBias[nodeIndex] = normalized_existing + normalized_uBias;
        }
        else
        {
            _upperBias[nodeIndex] = normalized_uBias;
        }
    }
}

bool CROWNAnalysis::isProcessed( unsigned nodeIndex ) const
{
    return _torchModel->isProcessed( nodeIndex );
}

void CROWNAnalysis::resetProcessingState()
{
    _torchModel->resetProcessingState();
}

void CROWNAnalysis::clearConcreteBounds()
{
    _concreteBounds.clear();
    _torchModel->clearConcreteBounds();
}

void CROWNAnalysis::clearAllNodeBounds()
{
    // Prevents gradient history from previous runs leaking into new computation graph
    for ( auto &[nodeIdx, node] : _nodes )
    {
        if ( node )
        {
            node->clearBounds();
        }
    }
}

void CROWNAnalysis::setReferenceBounds(
    const std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> &ref )
{
    _referenceBounds = ref;
}

void CROWNAnalysis::clearReferenceBounds()
{
    _referenceBounds.clear();
}

void CROWNAnalysis::markProcessed( unsigned nodeIndex )
{
    _torchModel->markProcessed( nodeIndex );
}

void CROWNAnalysis::log( const String &message )
{
    (void)message;
}

torch::Tensor CROWNAnalysis::getIBPLowerBound( unsigned nodeIndex )
{
    if ( _ibpBounds.exists( nodeIndex ) )
    {
        return _ibpBounds[nodeIndex].lower();
    }
    return torch::Tensor();
}

torch::Tensor CROWNAnalysis::getIBPUpperBound( unsigned nodeIndex )
{
    if ( _ibpBounds.exists( nodeIndex ) )
    {
        return _ibpBounds[nodeIndex].upper();
    }
    return torch::Tensor();
}

torch::Tensor CROWNAnalysis::getCrownLowerBound( unsigned nodeIndex ) const
{
    if ( _lA.exists( nodeIndex ) )
    {
        BoundA b = _lA[nodeIndex];
        if ( b.isTensor() )
            return b.asTensor();
        return torch::Tensor();
    }
    return torch::Tensor();
}

torch::Tensor CROWNAnalysis::getCrownUpperBound( unsigned nodeIndex ) const
{
    if ( _uA.exists( nodeIndex ) )
    {
        BoundA b = _uA[nodeIndex];
        if ( b.isTensor() )
            return b.asTensor();
        return torch::Tensor();
    }
    return torch::Tensor();
}

bool CROWNAnalysis::hasIBPBounds( unsigned nodeIndex )
{
    return _ibpBounds.exists( nodeIndex );
}

bool CROWNAnalysis::hasCrownBounds( unsigned nodeIndex )
{
    return _lA.exists( nodeIndex ) || _uA.exists( nodeIndex );
}

unsigned CROWNAnalysis::getNumNodes() const
{
    return _nodes.size();
}

std::shared_ptr<BoundedTorchNode> CROWNAnalysis::getNode( unsigned index ) const
{
    if ( _nodes.exists( index ) )
    {
        return _nodes[index];
    }
    return nullptr;
}

unsigned CROWNAnalysis::getInputSize() const
{
    return _torchModel->getInputSize();
}

unsigned CROWNAnalysis::getOutputSize() const
{
    return _torchModel->getOutputSize();
}

torch::Tensor CROWNAnalysis::getConcreteLowerBound( unsigned nodeIndex )
{
    if ( _nodes.exists( nodeIndex ) && _nodes[nodeIndex]->hasBounds() )
    {
        return _nodes[nodeIndex]->getLower();
    }
    if ( _concreteBounds.exists( nodeIndex ) )
    {
        return _concreteBounds[nodeIndex].lower();
    }
    return torch::Tensor();
}

torch::Tensor CROWNAnalysis::getConcreteUpperBound( unsigned nodeIndex )
{
    if ( _nodes.exists( nodeIndex ) && _nodes[nodeIndex]->hasBounds() )
    {
        return _nodes[nodeIndex]->getUpper();
    }
    if ( _concreteBounds.exists( nodeIndex ) )
    {
        return _concreteBounds[nodeIndex].upper();
    }
    return torch::Tensor();
}

bool CROWNAnalysis::hasConcreteBounds( unsigned nodeIndex )
{
    if ( _nodes.exists( nodeIndex ) && _nodes[nodeIndex]->hasBounds() )
    {
        return true;
    }
    return _concreteBounds.exists( nodeIndex );
}

BoundedTensor<torch::Tensor> CROWNAnalysis::getOutputBounds() const
{
    unsigned outputIndex = getOutputIndex();
    if ( _concreteBounds.exists( outputIndex ) )
    {
        return _concreteBounds[outputIndex];
    }
    return BoundedTensor<torch::Tensor>( torch::Tensor(), torch::Tensor() );
}

BoundedTensor<torch::Tensor> CROWNAnalysis::getOutputIBPBounds() const
{
    unsigned outputIndex = getOutputIndex();
    if ( _ibpBounds.exists( outputIndex ) )
    {
        return _ibpBounds[outputIndex];
    }
    return BoundedTensor<torch::Tensor>( torch::Tensor(), torch::Tensor() );
}

BoundedTensor<torch::Tensor> CROWNAnalysis::getNodeIBPBounds( unsigned nodeIndex ) const
{
    if ( _ibpBounds.exists( nodeIndex ) )
    {
        return _ibpBounds[nodeIndex];
    }
    return BoundedTensor<torch::Tensor>();
}

BoundedTensor<torch::Tensor> CROWNAnalysis::getNodeCrownBounds( unsigned nodeIndex ) const
{
    if ( _lA.exists( nodeIndex ) || _uA.exists( nodeIndex ) )
    {
        torch::Tensor lA = getCrownLowerBound( nodeIndex );
        torch::Tensor uA = getCrownUpperBound( nodeIndex );
        return BoundedTensor<torch::Tensor>( lA, uA );
    }
    return BoundedTensor<torch::Tensor>();
}

BoundedTensor<torch::Tensor> CROWNAnalysis::getNodeConcreteBounds( unsigned nodeIndex ) const
{
    if ( _concreteBounds.exists( nodeIndex ) )
    {
        return _concreteBounds[nodeIndex];
    }
    return BoundedTensor<torch::Tensor>();
}

bool CROWNAnalysis::needsCROWNBounds( unsigned nodeIndex )
{
    if ( !_nodes.exists( nodeIndex ) )
    {
        return false;
    }

    auto &node = _nodes[nodeIndex];

    if ( _concreteBounds.exists( nodeIndex ) )
    {
        return false;
    }

    if ( !node->isPerturbed() )
    {
        return false;
    }

    // IBP is exact for first linear layers
    if ( checkIBPFirstLinear( nodeIndex ) )
    {
        if ( _ibpBounds.exists( nodeIndex ) )
        {
            _concreteBounds[nodeIndex] = _ibpBounds[nodeIndex];
            _torchModel->setConcreteBounds( nodeIndex, _ibpBounds[nodeIndex] );
        }
        return false;
    }

    // >80% stable neurons means IBP is sufficient
    if ( node->getNodeType() == NodeType::RELU )
    {
        if ( _ibpBounds.exists( nodeIndex ) )
        {
            auto bounds = _ibpBounds[nodeIndex];
            torch::Tensor lower = bounds.lower();
            torch::Tensor upper = bounds.upper();

            if ( lower.defined() && upper.defined() )
            {
                float stable_count = ( ( lower > 0 ) | ( upper < 0 ) ).sum().item<float>();
                float total = lower.numel();
                float stable_ratio = stable_count / total;

                if ( stable_ratio > 0.8 )
                {
                    _concreteBounds[nodeIndex] = _ibpBounds[nodeIndex];
                    _torchModel->setConcreteBounds( nodeIndex, _ibpBounds[nodeIndex] );
                    return false;
                }
            }
        }
    }

    return true;
}

bool CROWNAnalysis::checkIBPFirstLinear( unsigned nodeIndex )
{
    if ( !LunaConfiguration::ENABLE_FIRST_LINEAR_IBP )
    {
        return false;
    }

    return isFirstLinearLayer( nodeIndex );
}

bool CROWNAnalysis::isFirstLinearLayer( unsigned nodeIndex )
{
    if ( !_nodes.exists( nodeIndex ) )
    {
        return false;
    }

    auto &node = _nodes[nodeIndex];

    if ( node->getNodeType() != NodeType::LINEAR )
    {
        return false;
    }

    if ( !_torchModel->getDependenciesMap().exists( nodeIndex ) )
    {
        return false;
    }

    const Vector<unsigned> &dependencies = _torchModel->getDependencies( nodeIndex );

    // Single input dependency must be the INPUT node
    if ( dependencies.size() != 1 )
    {
        return false;
    }

    unsigned inputNodeIndex = dependencies[0];
    if ( !_nodes.exists( inputNodeIndex ) )
    {
        return false;
    }

    return _nodes[inputNodeIndex]->getNodeType() == NodeType::INPUT;
}

bool CROWNAnalysis::checkIBPIntermediate( unsigned nodeIndex )
{
    unsigned current = nodeIndex;
    Vector<unsigned> nodesToProcess;

    while ( !hasConcreteBounds( current ) )
    {
        if ( !_nodes.exists( current ) )
            break;
        if ( !_nodes[current]->isIBPIntermediate() )
            return false;

        nodesToProcess.append( current );

        if ( !_torchModel->getDependenciesMap().exists( current ) )
            break;
        const Vector<unsigned> &deps = _torchModel->getDependencies( current );
        if ( deps.empty() )
            break;
        current = deps[0];
    }

    for ( int i = (int)nodesToProcess.size() - 1; i >= 0; i-- )
    {
        computeIBPForNode( nodesToProcess[i] );
    }
    return true;
}

void CROWNAnalysis::computeIBPForNode( unsigned nodeIndex )
{
    if ( !_nodes.exists( nodeIndex ) )
        return;
    if ( hasConcreteBounds( nodeIndex ) )
        return;

    auto &node = _nodes[nodeIndex];

    Vector<BoundedTensor<torch::Tensor>> inputBounds = getInputBoundsForNode( nodeIndex );

    BoundedTensor<torch::Tensor> ibp = node->computeIntervalBoundPropagation( inputBounds );

    node->setBounds( ibp.lower(), ibp.upper() );

    _ibpBounds[nodeIndex] = ibp;
    _concreteBounds[nodeIndex] = ibp;
    _torchModel->setConcreteBounds( nodeIndex, ibp );
}

void CROWNAnalysis::computeIntermediateBoundsLazy( unsigned nodeIndex )
{
    if ( hasConcreteBounds( nodeIndex ) )
        return;

    if ( checkIBPIntermediate( nodeIndex ) )
        return;
    if ( checkIBPFirstLinear( nodeIndex ) )
    {
        computeIBPForNode( nodeIndex );
        return;
    }

    Vector<unsigned> unstableIndices;

    if ( _ibpBounds.exists( nodeIndex ) )
    {
        const int64_t lanes =
            _torchModel->hasInputBounds()
                ? _torchModel->toBatchedInput( _torchModel->getInputLowerBounds() ).size( 0 )
                : 1;
        torch::Tensor lb = _ibpBounds[nodeIndex].lower().reshape( { lanes, -1 } );
        torch::Tensor ub = _ibpBounds[nodeIndex].upper().reshape( { lanes, -1 } );

        // One shared index set per layer, the union of every lane's unstable neurons
        torch::Tensor unstableMask = ( ( lb < 0 ) & ( ub > 0 ) ).any( 0 );

        torch::Tensor indices = torch::nonzero( unstableMask ).flatten();

        if ( indices.numel() > 0 )
        {
            auto indices_cpu = indices.cpu();
            auto acc = indices_cpu.accessor<int64_t, 1>();
            for ( int64_t i = 0; i < acc.size( 0 ); ++i )
            {
                unstableIndices.append( static_cast<unsigned>( acc[i] ) );
            }
        }

        log( Stringf(
            "computeIntermediateBoundsLazy() - Node %u has %u unstable neurons out of %lld total",
            nodeIndex, unstableIndices.size(), (long long)lb.size( 1 ) ) );
    }

    // No unstable neurons — skip backward CROWN to avoid memory blowup on conv layers
    if ( unstableIndices.empty() && _ibpBounds.exists( nodeIndex ) )
    {
        auto ibp = _ibpBounds[nodeIndex];
        _concreteBounds[nodeIndex] = ibp;
        _torchModel->setConcreteBounds( nodeIndex, ibp );
        if ( _nodes.exists( nodeIndex ) )
        {
            _nodes[nodeIndex]->setBounds( ibp.lower(), ibp.upper() );
        }
        return;
    }

    auto &node = _nodes[nodeIndex];
    unsigned nodeSize = node->getOutputSize();
    std::string startKey = "/" + std::to_string( nodeIndex );
    _setCurrentStart(
        startKey, static_cast<int>( unstableIndices.empty() ? nodeSize : unstableIndices.size() ) );
    _currentStartSpecIndices = unstableIndices;

    backwardFrom( nodeIndex, unstableIndices );
    concretizeNode( nodeIndex, unstableIndices );
}

void CROWNAnalysis::checkPriorBounds( unsigned nodeIndex )
{
    if ( !_nodes.exists( nodeIndex ) )
        return;
    auto &node = _nodes[nodeIndex];

    if ( !_torchModel->getDependenciesMap().exists( nodeIndex ) )
        return;
    const Vector<unsigned> &deps = _torchModel->getDependencies( nodeIndex );

    for ( unsigned i = 0; i < deps.size(); ++i )
    {
        unsigned inputNode = deps[i];
        checkPriorBounds( inputNode );
    }

    const Vector<unsigned> &requiredInputs = node->getRequiresInputBounds();
    for ( unsigned i = 0; i < requiredInputs.size(); ++i )
    {
        unsigned inputIdx = requiredInputs[i];
        if ( inputIdx < deps.size() )
        {
            unsigned inputNode = deps[inputIdx];
            computeIntermediateBoundsLazy( inputNode );
        }
    }
}

void CROWNAnalysis::setInputBounds( const BoundedTensor<torch::Tensor> &inputBounds )
{
    log( "[CROWNAnalysis] Setting input bounds" );

    _torchModel->setInputBounds( inputBounds );

    log( "[CROWNAnalysis] Input bounds set successfully" );
}

} // namespace NLR
