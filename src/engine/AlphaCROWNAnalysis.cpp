/*********************                                                        */
/*! \file AlphaCROWNAnalysis.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "AlphaCROWNAnalysis.h"
#include "nodes/BoundedAlphaOptimizedNode.h"
#include "nodes/BoundedSigmoidNode.h"

#include "Debug.h"
#include "MStringf.h"
#include "LunaError.h"
#include "TimeUtils.h"

#include <sstream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_map>

namespace NLR
{

AlphaCROWNAnalysis::AlphaCROWNAnalysis( TorchModel *torchModel )
    : _torchModel( torchModel )
    , _alphaEnabled( true )
    , _initialized( false )
    , _iteration( LunaConfiguration::ALPHA_ITERATIONS )
    , _learningRate( LunaConfiguration::ALPHA_LR )
    , _optimizationStage( "init" )
{
    if ( !_torchModel )
    {
        throw LunaError( LunaError::UNINITIALIZED_NODE,
                         "AlphaCROWNAnalysis requires a valid TorchModel instance" );
    }

    _crownAnalysis = std::make_unique<CROWNAnalysis>( _torchModel );
    updateFromConfig();
}

AlphaCROWNAnalysis::~AlphaCROWNAnalysis() {}

void AlphaCROWNAnalysis::initializeAlphaParameters()
{
    log( "initializeAlphaParameters() - Starting alpha parameter initialization" );

    if ( _initialized )
    {
        log( "initializeAlphaParameters() - Alpha parameters already initialized" );
        return;
    }

    try
    {
        log( "initializeAlphaParameters() - Forward pass" );
        performForwardPass();

        log( "initializeAlphaParameters() - Prepare optimizable activations" );
        prepareOptimizableActivations();

        log( "initializeAlphaParameters() - Standard CROWN initialization pass" );
        performCROWNInitializationPass();

        log( "initializeAlphaParameters() - Alpha parameters will be created lazily per start" );

        _initialized = true;
        log(
            "initializeAlphaParameters() - Alpha parameter initialization completed successfully" );
    }
    catch ( const std::exception &e )
    {
        log( Stringf( "initializeAlphaParameters() - Exception: %s", e.what() ) );
        throw;
    }
}

void AlphaCROWNAnalysis::performForwardPass()
{
    log( "performForwardPass() - forward pass through model" );

    // autograd required for gradient flow through alpha parameters during optimization
    torch::GradMode::set_enabled( true );

    log( "performForwardPass() - Forward pass structure established with autograd enabled" );
}

void AlphaCROWNAnalysis::prepareOptimizableActivations()
{
    log( "prepareOptimizableActivations() - Finding alpha-optimizable activation nodes" );

    _optimizableNodes.clear();

    unsigned numNodes = _crownAnalysis->getNumNodes();
    for ( unsigned i = 0; i < numNodes; ++i )
    {
        auto node = _crownAnalysis->getNode( i );
        if ( !node )
        {
            continue;
        }

        auto alphaNode = std::dynamic_pointer_cast<BoundedAlphaOptimizeNode>( node );
        if ( alphaNode )
        {
            _optimizableNodes.push_back( std::make_pair( i, alphaNode ) );

            alphaNode->setOptimizationStage( "init" );
            alphaNode->setAlphaCrownAnalysis( this );

            log( Stringf( "prepareOptimizableActivations() - Added optimizable node %u (type=%d)",
                          i, (int)node->getNodeType() ) );
        }
    }

    log( Stringf( "prepareOptimizableActivations() - Found %u optimizable nodes",
                  (unsigned)_optimizableNodes.size() ) );
}

void AlphaCROWNAnalysis::performCROWNInitializationPass()
{
    log( "performCROWNInitializationPass() - Running standard CROWN to capture relaxation slopes" );

    LunaConfiguration::ENABLE_FIRST_LINEAR_IBP = false;

    _crownAnalysis->resetProcessingState();

    _crownAnalysis->run( false );

    log( "performCROWNInitializationPass() - CROWN initialization pass completed" );
}

AlphaParameters &AlphaCROWNAnalysis::ensureAlphaFor( unsigned nodeIndex,
                                                     const std::string &startKey, int specDim,
                                                     int outDim, const torch::Tensor &input_lb,
                                                     const torch::Tensor &input_ub )
{
    auto &perStart = _alphaByNodeStart[nodeIndex];
    auto it = perStart.find( startKey );

    // Bounds are one row per lane and every lane shares one unstable set, the union of theirs
    const int64_t lanes = input_lb.dim() >= 2 ? input_lb.size( 0 ) : 1;
    auto input_lb_flat = input_lb.detach().flatten();
    auto input_ub_flat = input_ub.detach().flatten();
    auto input_lb_rows = input_lb.detach().reshape( { lanes, -1 } );
    auto input_ub_rows = input_ub.detach().reshape( { lanes, -1 } );
    torch::Tensor unstableMask;
    auto nodePtr = _crownAnalysis->getNode( nodeIndex );
    if ( nodePtr && nodePtr->getNodeType() == NodeType::SIGMOID )
    {
        if ( lanes > 1 )
        {
            throw std::runtime_error(
                "AlphaCROWNAnalysis: sigmoid alpha does not support batch > 1 "
                "yet" );
        }
        unstableMask = ( input_ub_flat - input_lb_flat ) > 1e-6;
    }
    else
    {
        unstableMask = ( ( input_lb_rows < 0 ) & ( input_ub_rows > 0 ) ).any( 0 );
    }
    int numUnstable = unstableMask.sum().item<int>();
    bool need_new = ( it == perStart.end() );
    if ( !need_new )
    {
        const auto &ap = it->second;
        bool spec_mismatch = false;
        if ( ap.alpha.defined() )
        {
            const auto storedSpecDim = (int)ap.alpha.size( 1 );

            spec_mismatch = ( storedSpecDim != specDim ) &&
                            !( ap.hasSpecDefaultSlot && storedSpecDim == specDim + 1 );

            // Alpha holds one slope per lane, so a different lane count needs a new tensor
            spec_mismatch = spec_mismatch || ap.alpha.size( 2 ) != lanes;
        }

        // As in auto_LiRPA, an inherited entry stays as it is while every unstable neuron still has
        // a column. A neuron that became stable is masked in the relaxation, so a search node
        // keeps its parent's alphas and the answer does not depend on what the other lanes need
        bool unstable_mismatch = true;
        if ( ap.unstableMask.defined() && ap.unstableMask.sizes() == unstableMask.sizes() )
        {
            unstable_mismatch = ( unstableMask & ap.unstableMask.logical_not() ).any().item<bool>();
        }

        // A frozen entry keeps the unstable set it was built for, since the optimizer holds it
        if ( spec_mismatch || ( unstable_mismatch && !_freezeAlphaSets ) )
        {
            perStart.erase( it );
            need_new = true;
        }
        else
        {
            return it->second;
        }
    }

    if ( need_new )
    {
        auto options = input_lb.options().dtype( torch::kFloat32 ).requires_grad( false );

        AlphaParameters params;
        params.specDim = specDim;
        params.outDim = outDim;
        params.numUnstable = numUnstable;
        params.unstableMask = unstableMask.detach().clone();

        if ( numUnstable == 0 )
        {
            if ( nodePtr && nodePtr->getNodeType() == NodeType::SIGMOID )
            {
                params.alpha = torch::empty( { 2, specDim, lanes, 0, 8 }, options );
            }
            else
            {
                params.alpha = torch::empty( { 2, specDim, lanes, 0 }, options );
            }
            params.unstableIndices = torch::empty( { 0 }, torch::kLong );
            perStart[startKey] = std::move( params );
            return perStart[startKey];
        }

        params.unstableIndices =
            torch::nonzero( unstableMask ).flatten().to( torch::kLong ).detach();

        torch::Tensor slope_init;
        torch::Tensor full_slope_debug;
        {
            std::shared_ptr<BoundedAlphaOptimizeNode> nodePtr;
            for ( auto &p : _optimizableNodes )
            {
                if ( p.first == nodeIndex )
                {
                    nodePtr = p.second;
                    break;
                }
            }
            if ( nodePtr )
            {
                torch::Tensor full_slope = nodePtr->getCROWNSlope( true );
                full_slope_debug = full_slope;
                if ( full_slope.defined() && full_slope.numel() > 0 )
                {
                    // The CROWN slopes are per lane, a single row is shared by every lane
                    auto slope_rows =
                        full_slope.numel() == outDim
                            ? full_slope.reshape( { 1, outDim } ).expand( { lanes, outDim } )
                            : full_slope.reshape( { lanes, -1 } );
                    slope_init = slope_rows.index_select( 1, params.unstableIndices );
                }
                else
                {
                    slope_init = torch::full( { lanes, numUnstable }, 0.5f, options );
                }
            }
            else
            {
                slope_init = torch::full( { lanes, numUnstable }, 0.5f, options );
            }
        }

        bool useSparseSpec = false;
        Vector<unsigned> cachedUnstableSpecs;
        bool cachedSparseMode = false;
        unsigned cachedNodeSize = 0;
        if ( _crownAnalysis->getAlphaStartCacheInfo( startKey, cachedUnstableSpecs,
                                                     cachedSparseMode, cachedNodeSize ) )
        {
            useSparseSpec = cachedSparseMode && (int)cachedUnstableSpecs.size() == specDim;
        }

        // sigmoid: 8 slots per neuron (auto_LiRPA s-shaped tangent-point style)
        int alphaSlots = 1;
        if ( nodePtr && nodePtr->getNodeType() == NodeType::SIGMOID )
        {
            alphaSlots = 8;
        }

        int specDimAlpha = useSparseSpec ? ( specDim + 1 ) : specDim;
        torch::Tensor alpha;
        torch::Tensor expanded;
        if ( alphaSlots == 1 )
        {
            alpha = torch::zeros( { 2, specDimAlpha, lanes, numUnstable }, options );
            expanded = slope_init.view( { 1, 1, lanes, numUnstable } )
                           .expand( { 2, specDim, lanes, numUnstable } )
                           .contiguous()
                           .clone()
                           .to( options.dtype() );
        }
        else
        {
            auto mid_unstable = ( ( input_lb_flat + input_ub_flat ) * 0.5f )
                                    .index_select( 0, params.unstableIndices )
                                    .to( options.dtype() );
            torch::Tensor d_lower_unstable = mid_unstable.clone();
            torch::Tensor d_upper_unstable = mid_unstable.clone();
            if ( nodePtr )
            {
                auto sigmoidNode = std::dynamic_pointer_cast<BoundedSigmoidNode>( nodePtr );
                if ( sigmoidNode )
                {
                    auto d_pair = sigmoidNode->getDefaultTangentPoints( input_lb, input_ub );
                    d_lower_unstable = d_pair.first.flatten()
                                           .index_select( 0, params.unstableIndices )
                                           .to( options.dtype() );
                    d_upper_unstable = d_pair.second.flatten()
                                           .index_select( 0, params.unstableIndices )
                                           .to( options.dtype() );
                }
            }

            alpha = torch::zeros( { 2, specDimAlpha, 1, numUnstable, alphaSlots }, options );
            auto expanded_mid = mid_unstable.view( { 1, 1, 1, numUnstable } )
                                    .expand( { 2, specDim, 1, numUnstable } )
                                    .contiguous()
                                    .clone()
                                    .to( options.dtype() );
            auto expanded_d_lower = d_lower_unstable.view( { 1, 1, 1, numUnstable } )
                                        .expand( { 2, specDim, 1, numUnstable } )
                                        .contiguous()
                                        .clone()
                                        .to( options.dtype() );
            auto expanded_d_upper = d_upper_unstable.view( { 1, 1, 1, numUnstable } )
                                        .expand( { 2, specDim, 1, numUnstable } )
                                        .contiguous()
                                        .clone()
                                        .to( options.dtype() );

            alpha.index_put_(
                { torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0 },
                expanded_mid );
            alpha.index_put_(
                { torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 1 },
                expanded_mid );
            alpha.index_put_(
                { torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 2 },
                expanded_mid );
            alpha.index_put_(
                { torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 3 },
                expanded_mid );
            // auto_LiRPA style: tp_both_lower/tp_both_upper from precomputed defaults
            alpha.index_put_(
                { torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 4 },
                expanded_d_lower );
            alpha.index_put_(
                { torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 5 },
                expanded_d_lower );
            alpha.index_put_(
                { torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 6 },
                expanded_d_upper );
            alpha.index_put_(
                { torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 7 },
                expanded_d_upper );
            expanded = alpha.narrow( 1, 0, specDim );
        }
        if ( useSparseSpec )
        {
            alpha.narrow( 1, 1, specDim ).copy_( expanded );
            params.hasSpecDefaultSlot = true;
        }
        else
        {
            alpha.copy_( expanded );
        }

        // detach() makes alpha a proper leaf tensor; without it, clamp creates a
        // non-leaf that triggers "modified by inplace operation" errors from optimizer
        if ( alphaSlots == 1 )
        {
            alpha = torch::clamp( alpha, 0.0f, 1.0f ).detach();
        }
        else
        {
            alpha = alpha.detach();
        }
        alpha.set_requires_grad( true );

        params.alpha = alpha;
        perStart[startKey] = std::move( params );
    }
    return perStart[startKey];
}

torch::Tensor AlphaCROWNAnalysis::computeOptimizedBounds( LunaConfiguration::BoundSide side )
{
    log( Stringf( "computeOptimizedBounds() - Starting optimization for %s bounds",
                  side == LunaConfiguration::BoundSide::Lower ? "LOWER" : "UPPER" ) );
    if ( !_initialized )
    {
        log( "computeOptimizedBounds() - Initializing alpha parameters" );
        initializeAlphaParameters();
    }

    if ( !shouldPerformOptimization() )
    {
        log( "computeOptimizedBounds() - Alpha optimization not applicable, falling back to "
             "standard CROWN" );
        _crownAnalysis->run( false );
        return side == LunaConfiguration::BoundSide::Lower ? extractLowerBoundsFromCROWN()
                                                           : extractUpperBoundsFromCROWN();
    }

    LunaConfiguration::BOUND_SIDE = side;
    bool isLower = ( side == LunaConfiguration::BoundSide::Lower );

    auto seededBounds =
        _torchModel->hasPersistedIntermediateBounds()
            ? _torchModel->getPersistedIntermediateBounds()
            : std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>>{};

    _bestIntermediateBounds.clear();
    _crownAnalysis->clearAllNodeBounds();
    setOptimizationStage( "opt" );
    LunaConfiguration::ENABLE_FIRST_LINEAR_IBP = false;

    _crownAnalysis->setAlphaStartCacheEnabled( true );
    _crownAnalysis->clearConcreteBounds();
    _crownAnalysis->clearCrownState();
    _crownAnalysis->resetProcessingState();

    if ( !seededBounds.empty() && !LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS )
    {
        for ( auto &[nodeIdx, pair] : seededBounds )
        {
            _bestIntermediateBounds[nodeIdx] = { pair.first.detach().clone(),
                                                 pair.second.detach().clone() };
        }
        restoreBestIntermediateBounds();
    }
    else if ( !seededBounds.empty() )
    {
        // Recompute intermediates but hand the persisted bounds to concretizeNode's
        // reference clamp, so every recomputed node is intersected with the caller's
        // known-valid bounds and downstream layers see the tightened result
        _crownAnalysis->setReferenceBounds( seededBounds );
    }

    _crownAnalysis->run( true );

    snapshotBestIntermediateBounds();

    // STE stabilization: capture initial bounds, tighten monotonically
    if ( LunaConfiguration::STABILIZE_INTERMEDIATE_BOUNDS )
    {
        _referenceBounds.clear();
        for ( const auto &[nodeIdx, bounds] : _bestIntermediateBounds )
        {
            _referenceBounds[nodeIdx] = { bounds.first.detach().clone(),
                                          bounds.second.detach().clone() };
        }
        _crownAnalysis->setReferenceBounds( _referenceBounds );
    }

    torch::Tensor initialBound =
        isLower ? extractLowerBoundsFromCROWN() : extractUpperBoundsFromCROWN();

    auto alphaParams = collectAlphaParameters();
    if ( alphaParams.empty() )
    {
        _crownAnalysis->run( false );
        return isLower ? extractLowerBoundsFromCROWN() : extractUpperBoundsFromCROWN();
    }

    auto optimizer = std::make_shared<torch::optim::Adam>(
        alphaParams, torch::optim::AdamOptions( _learningRate )
                         .betas( std::make_tuple( 0.9, 0.999 ) )
                         .eps( 1e-8 ) );

    // From here the optimizer owns these tensors, so they are not rebuilt until the run ends
    struct FreezeAlphaSets
    {
        bool &flag;
        explicit FreezeAlphaSets( bool &f )
            : flag( f )
        {
            flag = true;
        }
        ~FreezeAlphaSets()
        {
            flag = false;
        }
    } freezeAlphaSets( _freezeAlphaSets );

    // Best bookkeeping follows auto_LiRPA: a lane improves when its bound summed over specs does,
    // and only improved lanes save their alphas and intermediate bounds
    torch::Tensor bestBounds;
    torch::Tensor savedReference;
    float currentLR = _learningRate;
    const unsigned early_stop_patience = LunaConfiguration::EARLY_STOP_PATIENCE;
    unsigned patience = 0;
    unsigned iterationsExecuted = 0;

    for ( unsigned iter = 0; iter < _iteration; ++iter )
    {
        iterationsExecuted = iter + 1;
        std::unique_ptr<torch::NoGradGuard> no_grad;
        if ( iter == _iteration - 1 )
        {
            no_grad = std::make_unique<torch::NoGradGuard>();
        }

        _crownAnalysis->clearConcreteBounds();
        _crownAnalysis->clearAllNodeBounds();
        _crownAnalysis->clearCrownState();
        _crownAnalysis->resetProcessingState();

        if ( !LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS && !_bestIntermediateBounds.empty() )
            restoreBestIntermediateBounds();

        _crownAnalysis->run( true );

        torch::Tensor currentBound =
            isLower ? extractLowerBoundsFromCROWN() : extractUpperBoundsFromCROWN();
        torch::Tensor currentBoundDetached = currentBound.detach();
        const int64_t lanes = currentBoundDetached.dim() >= 2 ? currentBoundDetached.size( 0 ) : 1;
        const torch::Tensor currentRows = currentBoundDetached.reshape( { lanes, -1 } );

        // A lane whose properties are all proven stops contributing loss, the others go on
        torch::Tensor stopLanes = torch::zeros( { lanes }, torch::kBool );
        if ( LunaConfiguration::STOP_ALPHA_ON_VERIFIED )
        {
            const torch::Tensor lowerRows =
                extractLowerBoundsFromCROWN().detach().reshape( { lanes, -1 } );
            const torch::Tensor upperRows =
                extractUpperBoundsFromCROWN().detach().reshape( { lanes, -1 } );
            for ( int64_t lane = 0; lane < lanes; ++lane )
            {
                stopLanes[lane] = _torchModel->isSpecVerified( lowerRows[lane], upperRows[lane] );
            }
        }
        const bool stopEveryLane = stopLanes.all().item<bool>();

        // Lower side maximizes the summed bound and the upper side minimizes it
        torch::Tensor loss;
        if ( iter < _iteration - 1 )
        {
            const torch::Tensor perLane = currentBound.reshape( { lanes, -1 } ).sum( 1 );
            const torch::Tensor active = stopLanes.logical_not().to( perLane.device() );
            loss = ( ( isLower ? -perLane : perLane ) * active.to( perLane.dtype() ) ).sum();
        }

        // Every iteration keeps the best bound of each lane that improved
        bool needUpdate = false;
        if ( !bestBounds.defined() )
        {
            bestBounds = currentRows.clone();
            savedReference = currentRows.clone();
        }
        else
        {
            const torch::Tensor reducedNow = currentRows.sum( 1 );
            const torch::Tensor reducedBest = bestBounds.sum( 1 );
            const torch::Tensor improved =
                isLower ? ( reducedNow > reducedBest ) : ( reducedNow < reducedBest );
            needUpdate = improved.any().item<bool>();
            if ( needUpdate )
            {
                const torch::Tensor merged = isLower ? torch::max( bestBounds, currentRows )
                                                     : torch::min( bestBounds, currentRows );
                bestBounds = torch::where( improved.unsqueeze( 1 ), merged, bestBounds );
            }
        }
        patience = needUpdate ? 0 : patience + 1;

        // Alphas and intermediate bounds are saved on the first iteration, in the second half, and
        // when patience runs out, and only for the lanes that beat their last saved result
        const bool shouldSaveBest =
            ( iter < 1 ) ||
            ( iter > static_cast<unsigned>( _iteration * LunaConfiguration::START_SAVE_BEST ) ) ||
            stopEveryLane || ( patience == early_stop_patience );
        if ( shouldSaveBest )
        {
            torch::Tensor savedLanes;
            if ( iter < 1 )
            {
                // On the bounds' device, since it is combined with them in torch::where below
                savedLanes = torch::ones(
                    { lanes },
                    torch::TensorOptions().dtype( torch::kBool ).device( currentRows.device() ) );
            }
            else
            {
                const torch::Tensor reducedNow = currentRows.sum( 1 );
                const torch::Tensor reducedSaved = savedReference.sum( 1 );
                savedLanes =
                    isLower ? ( reducedNow > reducedSaved ) : ( reducedNow < reducedSaved );
            }
            if ( savedLanes.any().item<bool>() )
            {
                savedReference =
                    torch::where( savedLanes.unsqueeze( 1 ), currentRows, savedReference );
                updateBestAlphas( savedLanes );
                snapshotBestIntermediateBounds( savedLanes );

                if ( LunaConfiguration::STABILIZE_INTERMEDIATE_BOUNDS )
                {
                    for ( const auto &[nodeIdx, bounds] : _bestIntermediateBounds )
                    {
                        _referenceBounds[nodeIdx] = { bounds.first.detach().clone(),
                                                      bounds.second.detach().clone() };
                    }
                    _crownAnalysis->setReferenceBounds( _referenceBounds );
                }
            }
        }

        if ( stopEveryLane )
        {
            log( Stringf( "All specs verified at iteration %u", iter ) );
            break;
        }

        if ( patience > early_stop_patience )
        {
            break;
        }

        if ( iter < _iteration - 1 && loss.defined() )
        {
            if ( !loss.requires_grad() )
            {
                break;
            }
            optimizer->zero_grad();
            loss.backward();

            _crownAnalysis->clearConcreteBounds();
            _crownAnalysis->resetProcessingState();
            optimizer->step();
            clipAlphaParameters();

            currentLR *= LunaConfiguration::ALPHA_LR_DECAY;
            for ( auto &group : optimizer->param_groups() )
            {
                if ( auto adamGroup =
                         dynamic_cast<torch::optim::AdamOptions *>( &group.options() ) )
                {
                    adamGroup->lr( currentLR );
                }
            }
        }
    }

    ( isLower ? LunaConfiguration::LAST_LOWER_ALPHA_ITERATIONS
              : LunaConfiguration::LAST_UPPER_ALPHA_ITERATIONS ) = iterationsExecuted;

    restoreBestAlphas();
    _crownAnalysis->clearReferenceBounds();
    _crownAnalysis->clearConcreteBounds();
    restoreBestIntermediateBounds();
    _crownAnalysis->resetProcessingState();
    _crownAnalysis->run( false );

    torch::Tensor finalBound =
        isLower ? extractLowerBoundsFromCROWN() : extractUpperBoundsFromCROWN();
    if ( bestBounds.defined() )
    {
        finalBound =
            isLower ? torch::max( finalBound, bestBounds ) : torch::min( finalBound, bestBounds );
    }

    log( Stringf( "computeOptimizedBounds() - %s bound optimization completed",
                  isLower ? "Lower" : "Upper" ) );

    return finalBound;
}

torch::Tensor AlphaCROWNAnalysis::extractLowerBoundsFromCROWN()
{
    unsigned outputIndex = _crownAnalysis->getOutputIndex();

    if ( _crownAnalysis->hasConcreteBounds( outputIndex ) )
    {
        torch::Tensor lowerBounds = _crownAnalysis->getConcreteLowerBound( outputIndex );

        return lowerBounds;
    }

    if ( _crownAnalysis->hasIBPBounds( outputIndex ) )
    {
        torch::Tensor lowerBounds = _crownAnalysis->getIBPLowerBound( outputIndex );
        return lowerBounds;
    }

    auto options =
        torch::TensorOptions().dtype( torch::kFloat32 ).device( _torchModel->getDevice() );
    return torch::zeros( { 1 }, options );
}

torch::Tensor AlphaCROWNAnalysis::extractUpperBoundsFromCROWN()
{
    unsigned outputIndex = _crownAnalysis->getOutputIndex();

    if ( _crownAnalysis->hasConcreteBounds( outputIndex ) )
    {
        torch::Tensor upperBounds = _crownAnalysis->getConcreteUpperBound( outputIndex );
        return upperBounds;
    }

    if ( _crownAnalysis->hasIBPBounds( outputIndex ) )
    {
        torch::Tensor upperBounds = _crownAnalysis->getIBPUpperBound( outputIndex );
        return upperBounds;
    }

    auto options =
        torch::TensorOptions().dtype( torch::kFloat32 ).device( _torchModel->getDevice() );
    return torch::zeros( { 1 }, options );
}

std::vector<torch::Tensor> AlphaCROWNAnalysis::collectAlphaParameters()
{
    std::vector<torch::Tensor> params;

    for ( auto &[nodeIdx, perStart] : _alphaByNodeStart )
    {
        for ( auto &[startKey, ap] : perStart )
        {
            if ( ap.alpha.requires_grad() )
            {
                params.push_back( ap.alpha );
            }
        }
    }

    log( Stringf( "collectAlphaParameters() - Collected %u alpha parameter tensors",
                  (unsigned)params.size() ) );
    return params;
}

void AlphaCROWNAnalysis::updateBestAlphas( const torch::Tensor &improvedLanes )
{
    if ( !improvedLanes.defined() || !improvedLanes.any().item<bool>() )
    {
        return;
    }

    log( Stringf( "updateBestAlphas() - Updating best alphas for %lld improved lanes",
                  (long long)improvedLanes.sum().item<int64_t>() ) );

    auto &currentStorage = _alphaByNodeStart;
    auto &bestStorage = _bestAlphaByNodeStart;

    torch::NoGradGuard no_grad;

    for ( auto &[nodeIdx, perStart] : currentStorage )
    {
        for ( auto &[startKey, ap] : perStart )
        {
            if ( !ap.alpha.defined() || ap.alpha.numel() == 0 )
            {
                continue;
            }

            auto &bestPerStart = bestStorage[nodeIdx];
            auto itBest = bestPerStart.find( startKey );
            if ( itBest == bestPerStart.end() )
            {
                AlphaParameters copy = ap;
                copy.alpha = ap.alpha.detach().clone();
                bestPerStart[startKey] = std::move( copy );
                continue;
            }

            auto &bestAp = itBest->second;
            auto currentAlpha = ap.alpha.detach();

            if ( !bestAp.alpha.defined() || bestAp.alpha.sizes() != currentAlpha.sizes() )
            {
                bestAp.alpha = currentAlpha.clone();
                continue;
            }

            // A lane that improved takes its whole alpha slice, every start node alike
            std::vector<int64_t> laneShape( currentAlpha.dim(), 1 );
            laneShape[2] = improvedLanes.size( 0 );
            bestAp.alpha =
                torch::where( improvedLanes.to( currentAlpha.device() ).reshape( laneShape ),
                              currentAlpha, bestAp.alpha );
        }
    }
}

void AlphaCROWNAnalysis::restoreBestAlphas()
{
    log( "restoreBestAlphas() - Restoring all alpha parameters to best found values" );

    auto &currentStorage = _alphaByNodeStart;
    auto &bestStorage = _bestAlphaByNodeStart;

    for ( auto &[nodeIdx, perStart] : currentStorage )
    {
        for ( auto &[startKey, ap] : perStart )
        {
            auto &bestAp = bestStorage[nodeIdx][startKey];
            if ( ap.alpha.defined() && bestAp.alpha.defined() &&
                 ap.alpha.sizes() == bestAp.alpha.sizes() )
            {
                ap.alpha.data().copy_( bestAp.alpha.data() );
            }
            else
            {
                if ( ap.alpha.defined() )
                {
                    bestAp.alpha = ap.alpha.detach().clone();
                }
            }
        }
    }
}

void AlphaCROWNAnalysis::snapshotBestIntermediateBounds( const torch::Tensor &improvedLanes )
{
    unsigned numNodesUpdated = 0;
    unsigned numNodesNew = 0;

    unsigned numNodes = _crownAnalysis->getNumNodes();
    for ( unsigned nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx )
    {
        auto node = _crownAnalysis->getNode( nodeIdx );
        if ( !node )
            continue;

        bool isPreActivation = false;

        for ( const auto &[actIdx, actNode] : _optimizableNodes )
        {
            (void)actNode;
            auto deps = _torchModel->getDependencies( actIdx );
            for ( unsigned depIdx : deps )
            {
                if ( depIdx == nodeIdx )
                {
                    isPreActivation = true;
                    break;
                }
            }
            if ( isPreActivation )
                break;
        }

        if ( !isPreActivation )
            continue;

        if ( _crownAnalysis->hasConcreteBounds( nodeIdx ) )
        {
            torch::Tensor currentLower = _crownAnalysis->getConcreteLowerBound( nodeIdx ).detach();
            torch::Tensor currentUpper = _crownAnalysis->getConcreteUpperBound( nodeIdx ).detach();

            auto it = _bestIntermediateBounds.find( nodeIdx );
            if ( it != _bestIntermediateBounds.end() )
            {
                // element-wise merge: tighter lower = max, tighter upper = min
                torch::Tensor &storedLower = it->second.first;
                torch::Tensor &storedUpper = it->second.second;

                torch::Tensor mergedLower = torch::max( storedLower, currentLower );
                torch::Tensor mergedUpper = torch::min( storedUpper, currentUpper );

                // Only a lane that improved merges its new bounds, the others keep what they had
                if ( improvedLanes.defined() && mergedLower.dim() >= 1 &&
                     mergedLower.size( 0 ) == improvedLanes.size( 0 ) )
                {
                    std::vector<int64_t> laneShape( mergedLower.dim(), 1 );
                    laneShape[0] = improvedLanes.size( 0 );
                    const auto rows = improvedLanes.to( mergedLower.device() ).reshape( laneShape );
                    mergedLower = torch::where( rows, mergedLower, storedLower );
                    mergedUpper = torch::where( rows, mergedUpper, storedUpper );
                }
                storedLower = mergedLower.detach().clone();
                storedUpper = mergedUpper.detach().clone();
                numNodesUpdated++;
            }
            else
            {
                _bestIntermediateBounds[nodeIdx] =
                    std::make_pair( currentLower.detach().clone(), currentUpper.detach().clone() );
                numNodesNew++;
            }
        }
    }

    if ( numNodesNew > 0 || numNodesUpdated > 0 )
    {
        log( Stringf(
            "snapshotBestIntermediateBounds() - Element-wise best: %u nodes updated, %u nodes new",
            numNodesUpdated, numNodesNew ) );
    }
}

void AlphaCROWNAnalysis::restoreBestIntermediateBounds()
{
    if ( _bestIntermediateBounds.empty() )
    {
        return;
    }

    log( "restoreBestIntermediateBounds() - Seeding best bounds into nodes" );

    unsigned numRestored = 0;
    for ( const auto &[nodeIdx, bounds] : _bestIntermediateBounds )
    {
        auto node = _crownAnalysis->getNode( nodeIdx );
        if ( node )
        {
            auto lower = bounds.first.detach();
            auto upper = bounds.second.detach();
            lower.requires_grad_( false );
            upper.requires_grad_( false );
            node->setBounds( lower, upper );
            numRestored++;
        }
    }

    log( Stringf( "restoreBestIntermediateBounds() - Restored bounds for %u nodes", numRestored ) );
}

torch::Tensor
AlphaCROWNAnalysis::computeLoss( const torch::Tensor &lowerBounds, const torch::Tensor &upperBounds,
                                 bool optimizeLower,
                                 c10::optional<torch::Tensor> stop_mask_opt /* (batch,1) bool */ )
{

    (void)stop_mask_opt;
    auto options =
        lowerBounds.defined()
            ? lowerBounds.options()
            : ( upperBounds.defined() ? upperBounds.options()
                                      : torch::TensorOptions().dtype( torch::kFloat32 ) );
    torch::Tensor loss_per_elem;

    if ( optimizeLower )
    {
        if ( !lowerBounds.defined() || lowerBounds.numel() == 0 )
        {
            return torch::zeros( { 1 }, options );
        }
        if ( lowerBounds.dim() == 1 )
        {
            loss_per_elem = -lowerBounds.sum().unsqueeze( 0 );
        }
        else
        {
            auto l = lowerBounds.sum( /*dim=*/1, /*keepdim=*/true );
            loss_per_elem = -l;
        }
    }
    else
    {
        if ( !upperBounds.defined() || upperBounds.numel() == 0 )
        {
            return torch::zeros( { 1 }, options );
        }
        if ( upperBounds.dim() == 1 )
        {
            loss_per_elem = upperBounds.sum().unsqueeze( 0 );
        }
        else
        {
            auto u = upperBounds.sum( /*dim=*/1, /*keepdim=*/true );
            loss_per_elem = u;
        }
    }

    auto loss = loss_per_elem.sum();

    return loss;
}

bool AlphaCROWNAnalysis::shouldPerformOptimization() const
{
    if ( !_alphaEnabled )
    {
        return false;
    }

    if ( !_initialized && _optimizableNodes.empty() )
    {
        unsigned numNodes = _crownAnalysis->getNumNodes();
        for ( unsigned i = 0; i < numNodes; ++i )
        {
            auto node = _crownAnalysis->getNode( i );
            if ( node && std::dynamic_pointer_cast<BoundedAlphaOptimizeNode>( node ) )
            {
                return true;
            }
        }
        return false;
    }

    return !_optimizableNodes.empty();
}

AlphaCROWNAnalysis::AlphaResult
AlphaCROWNAnalysis::getAlphaForNodeAllSpecs( unsigned nodeIndex, const std::string &startKey,
                                             int specDim, int outDim, const torch::Tensor &input_lb,
                                             const torch::Tensor &input_ub )
{

    auto &ap = ensureAlphaFor( nodeIndex, startKey, specDim, outDim, input_lb, input_ub );

    AlphaResult result;
    result.numUnstable = ap.numUnstable;
    result.outDim = ap.outDim;
    result.unstableMask = ap.unstableMask;
    result.unstableIndices = ap.unstableIndices;
    result.hasSpecDefaultSlot = ap.hasSpecDefaultSlot;

    // ReLU alpha keeps its lane axis, the sigmoid alpha is single lane and stays squeezed
    result.alpha = ap.alpha.dim() == 5 ? ap.alpha.squeeze( 2 ) : ap.alpha;

    return result;
}

void AlphaCROWNAnalysis::setOptimizationStage( const std::string &stage )
{
    _optimizationStage = stage;

    for ( auto &nodePair : _optimizableNodes )
    {
        auto node = nodePair.second;
        node->setOptimizationStage( stage );
    }

    log( Stringf( "setOptimizationStage() - Set optimization stage to '%s'", stage.c_str() ) );
}

unsigned AlphaCROWNAnalysis::getNumOptimizableNodes() const
{
    return static_cast<unsigned>( _optimizableNodes.size() );
}

std::vector<unsigned> AlphaCROWNAnalysis::getOptimizableNodeIndices() const
{
    std::vector<unsigned> indices;
    indices.reserve( _optimizableNodes.size() );

    for ( const auto &nodePair : _optimizableNodes )
    {
        indices.push_back( nodePair.first );
    }

    return indices;
}

void AlphaCROWNAnalysis::clipAlphaParameters()
{
    log( "clipAlphaParameters() - Clipping alpha parameters to valid ranges [0, 1]" );

    // .data().clamp_() modifies in-place without breaking autograd;
    // set_data() or new tensors cause version mismatch with squeeze() views
    torch::NoGradGuard no_grad;

    auto shouldClip01 = [&]( unsigned nodeIdx ) -> bool
    {
        auto node = _crownAnalysis->getNode( nodeIdx );
        // sigmoid uses tangent-point alpha, not [0,1] slopes
        return !( node && node->getNodeType() == NodeType::SIGMOID );
    };

    for ( auto &[nodeIdx, perStart] : _alphaByNodeStart )
    {
        for ( auto &[startKey, ap] : perStart )
        {
            (void)startKey;
            if ( !shouldClip01( nodeIdx ) )
                continue;
            ap.alpha.data().clamp_( 0.0f, 1.0f );
        }
    }

    log( "clipAlphaParameters() - Alpha parameter clipping completed" );
}

void AlphaCROWNAnalysis::resetForOptimization()
{
    log( "resetForOptimization() - Resetting analysis state for optimization" );

    setOptimizationStage( "opt" );

    for ( auto &[nodeIdx, perStart] : _alphaByNodeStart )
    {
        for ( auto &[startKey, ap] : perStart )
        {
            ap.alpha.requires_grad_( true );
        }
    }

    log( "resetForOptimization() - Analysis state reset for optimization" );
}

void AlphaCROWNAnalysis::resetAlphasFromCROWNSlopes( bool isLower )
{
    (void)isLower;

    log( "resetAlphasFromCROWNSlopes() - Re-initializing alpha parameters from CROWN slopes" );

    _alphaByNodeStart.clear();
    _bestAlphaByNodeStart.clear();
    _crownAnalysis->clearAlphaStartCache();

    log( "resetAlphasFromCROWNSlopes() - Cleared alpha storage, alphas will be re-created lazily" );
}

CROWNAnalysis *AlphaCROWNAnalysis::getCROWNAnalysis() const
{
    return _crownAnalysis.get();
}

TorchModel *AlphaCROWNAnalysis::getTorchModel() const
{
    return _torchModel;
}

void AlphaCROWNAnalysis::updateFromConfig()
{
    _alphaEnabled =
        ( LunaConfiguration::ANALYSIS_METHOD == LunaConfiguration::AnalysisMethod::AlphaCROWN );
    _iteration = LunaConfiguration::ALPHA_ITERATIONS;
    _learningRate = LunaConfiguration::ALPHA_LR;

    log( Stringf(
        "updateFromConfig() - Updated from LunaConfiguration: enable=%s, iterations=%u, lr=%.3f",
        _alphaEnabled ? "true" : "false", _iteration, _learningRate ) );
}

void AlphaCROWNAnalysis::log( const String &message )
{
    if ( LunaConfiguration::NETWORK_LEVEL_REASONER_LOGGING || LunaConfiguration::VERBOSE )
    {
        printf( "AlphaCROWNAnalysis: %s\n", message.ascii() );
    }
}

void AlphaCROWNAnalysis::preserveIntermediateBoundsForNextOptimization()
{
    _reuseIntermediateBounds = !_bestIntermediateBounds.empty();
    if ( _reuseIntermediateBounds )
    {
        log( "preserveIntermediateBoundsForNextOptimization() - Intermediate bounds preserved for "
             "reuse" );
    }
}

void AlphaCROWNAnalysis::seedAlphas(
    const std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>> &alphas,
    int64_t lanes )
{
    for ( auto &[nodeIdx, perStart] : alphas )
    {
        for ( auto &[startKey, ap] : perStart )
        {
            torch::Tensor seededAlpha = ap.alpha.defined() ? ap.alpha.detach() : torch::Tensor();
            if ( seededAlpha.defined() && seededAlpha.dim() >= 3 && seededAlpha.size( 2 ) != lanes )
            {
                // A one-lane parent warm starts every lane, any other lane count starts cold
                if ( seededAlpha.size( 2 ) != 1 )
                    continue;
                std::vector<int64_t> sizes = seededAlpha.sizes().vec();
                sizes[2] = lanes;
                seededAlpha = seededAlpha.expand( sizes );
            }

            AlphaParameters copy;
            copy.alpha = seededAlpha.defined() ? seededAlpha.clone() : torch::Tensor();
            if ( copy.alpha.defined() )
                copy.alpha.set_requires_grad( true );
            copy.unstableMask =
                ap.unstableMask.defined() ? ap.unstableMask.detach().clone() : torch::Tensor();
            copy.unstableIndices = ap.unstableIndices.defined()
                                       ? ap.unstableIndices.detach().clone()
                                       : torch::Tensor();
            copy.specDim = ap.specDim;
            copy.batchDim = ap.batchDim;
            copy.outDim = ap.outDim;
            copy.numUnstable = ap.numUnstable;
            copy.requiresGrad = true;
            copy.hasSpecDefaultSlot = ap.hasSpecDefaultSlot;
            _alphaByNodeStart[nodeIdx][startKey] = std::move( copy );
        }
    }
    _bestAlphaByNodeStart = _alphaByNodeStart;
    if ( _optimizableNodes.empty() )
        prepareOptimizableActivations();
    _initialized = true;
    log( "seedAlphas() - Seeded alpha parameters from BaB state" );
}

void AlphaCROWNAnalysis::seedIntermediateBounds(
    const std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> &bounds )
{
    for ( auto &[nodeIdx, pair] : bounds )
    {
        _bestIntermediateBounds[nodeIdx] = { pair.first.detach().clone(),
                                             pair.second.detach().clone() };
    }
    if ( !LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS )
    {
        restoreBestIntermediateBounds();
    }
    else if ( LunaConfiguration::STABILIZE_INTERMEDIATE_BOUNDS )
    {
        for ( auto &[nodeIdx, bp] : _bestIntermediateBounds )
        {
            _referenceBounds[nodeIdx] = { bp.first.detach().clone(), bp.second.detach().clone() };
        }
        _crownAnalysis->setReferenceBounds( _referenceBounds );
    }
    log( "seedIntermediateBounds() - Seeded intermediate bounds from BaB state" );
}

std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>>
AlphaCROWNAnalysis::extractAlphas() const
{
    const auto &source = _bestAlphaByNodeStart.empty() ? _alphaByNodeStart : _bestAlphaByNodeStart;
    std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>> result;
    for ( auto &[nodeIdx, perStart] : source )
    {
        for ( auto &[startKey, ap] : perStart )
        {
            AlphaParameters copy;
            copy.alpha = ap.alpha.defined() ? ap.alpha.detach().clone() : torch::Tensor();
            copy.unstableMask =
                ap.unstableMask.defined() ? ap.unstableMask.detach().clone() : torch::Tensor();
            copy.unstableIndices = ap.unstableIndices.defined()
                                       ? ap.unstableIndices.detach().clone()
                                       : torch::Tensor();
            copy.specDim = ap.specDim;
            copy.batchDim = ap.batchDim;
            copy.outDim = ap.outDim;
            copy.numUnstable = ap.numUnstable;
            copy.requiresGrad = ap.requiresGrad;
            copy.hasSpecDefaultSlot = ap.hasSpecDefaultSlot;
            result[nodeIdx][startKey] = std::move( copy );
        }
    }
    return result;
}

} // namespace NLR
