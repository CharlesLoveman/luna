/*********************                                                        */
/*! \file BoundedReLUNode.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "BoundedReLUNode.h"
#include "AlphaCROWNAnalysis.h"
#include "LunaConfiguration.h"
#include "LunaError.h"
#include "conv/Patches.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include "CROWNUtils.h"

namespace NLR
{

BoundedReLUNode::BoundedReLUNode( const torch::nn::ReLU &reluModule, const String &name )
    : BoundedAlphaOptimizeNode()
    , _reluModule( std::make_shared<torch::nn::ReLU>( reluModule ) )
{
    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;

    _requiresInputBounds.append( 0 );
    _ibpIntermediate = true;
}

torch::Tensor BoundedReLUNode::forward( const torch::Tensor &input )
{
    if ( input.dim() > 0 )
    {
        _input_size = input.numel();
        _output_size = input.numel();
    }

    return ( *_reluModule )( input );
}

void BoundedReLUNode::moveToDevice( const torch::Device &device )
{
    BoundedAlphaOptimizeNode::moveToDevice( device );
    if ( _reluModule )
    {
        _reluModule->ptr()->to( device );
    }
    if ( init_upper_d.defined() )
    {
        init_upper_d = init_upper_d.to( device );
    }
}

void BoundedReLUNode::boundBackward( const BoundA &last_lA, const BoundA &last_uA,
                                     const Vector<BoundedTensor<torch::Tensor>> &inputBounds,
                                     Vector<Pair<BoundA, BoundA>> &outputA_matrices,
                                     torch::Tensor &lbias, torch::Tensor &ubias )
{

    if ( inputBounds.size() < 1 )
    {
        throw std::runtime_error( "BoundedReLUNode expects at least one input" );
    }

    const auto &inputBound = inputBounds[0];
    torch::Tensor input_lower = inputBound.lower();
    torch::Tensor input_upper = inputBound.upper();

    // Bounds are one row per lane, a 1-D bound is a single lane
    const int64_t lanes = input_lower.dim() >= 2 ? input_lower.size( 0 ) : 1;
    input_lower = input_lower.reshape( { lanes, -1 } );
    input_upper = input_upper.reshape( { lanes, -1 } );

    const BoundA &effective_lA = last_lA;
    const BoundA &effective_uA = last_uA;
    const bool patchesPath = ( effective_lA.defined() && effective_lA.isPatches() ) ||
                             ( effective_uA.defined() && effective_uA.isPatches() );

    int specDim = 1;
    if ( effective_lA.defined() )
    {
        specDim = effective_lA.inferSpecDim();
    }
    else if ( effective_uA.defined() )
    {

        specDim = effective_uA.inferSpecDim();
    }
    _currentSpecDim = specDim;

    auto relaxation_result =
        _backwardRelaxation( effective_lA, effective_uA, input_lower, input_upper );

    // Patches unfold one slope image per row, so alpha's (spec, lanes, F) becomes (spec*lanes, F)
    // and the unfold selects the image of each spec and lane. CROWN slopes stay (lanes, F)
    bool perSpecSlopes = false;
    if ( patchesPath )
    {
        for ( torch::Tensor *slope :
              { &relaxation_result.lb_lower_d, &relaxation_result.lb_upper_d,
                &relaxation_result.ub_upper_d, &relaxation_result.ub_lower_d } )
        {
            if ( slope->defined() && slope->dim() == 3 )
            {
                perSpecSlopes = true;
                *slope =
                    slope->reshape( { slope->size( 0 ) * slope->size( 1 ), slope->size( 2 ) } );
            }
        }
    }

    BoundA new_lA, new_uA;

    if ( effective_lA.defined() )
    {
        auto aL_l = relaxation_result.lb_lower_d.defined() ? relaxation_result.lb_lower_d
                                                           : relaxation_result.d_lower;
        auto aU_l = relaxation_result.lb_upper_d.defined() ? relaxation_result.lb_upper_d
                                                           : relaxation_result.d_upper;
        auto bL_l = relaxation_result.bias_lower.defined() ? relaxation_result.bias_lower
                                                           : torch::zeros_like( input_lower );
        auto bU_l = relaxation_result.bias_upper.defined() ? relaxation_result.bias_upper
                                                           : torch::zeros_like( input_lower );

        new_lA = propagate( effective_lA, lanes, perSpecSlopes, aL_l, aU_l, bL_l, bU_l, lbias );
    }

    if ( effective_uA.defined() )
    {
        auto aU_u = relaxation_result.ub_upper_d.defined() ? relaxation_result.ub_upper_d
                                                           : relaxation_result.d_upper;
        auto aL_u = relaxation_result.ub_lower_d.defined() ? relaxation_result.ub_lower_d
                                                           : relaxation_result.d_lower;
        auto bU_u = relaxation_result.bias_upper.defined() ? relaxation_result.bias_upper
                                                           : torch::zeros_like( input_lower );
        auto bL_u = relaxation_result.bias_lower.defined() ? relaxation_result.bias_lower
                                                           : torch::zeros_like( input_lower );

        new_uA = propagate( effective_uA, lanes, perSpecSlopes, aU_u, aL_u, bU_u, bL_u, ubias );
    }

    if ( outputA_matrices.size() == 0 )
    {
        outputA_matrices.append( Pair<BoundA, BoundA>( new_lA, new_uA ) );
    }
}

BoundedTensor<torch::Tensor> BoundedReLUNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>> &inputBounds )
{

    if ( inputBounds.size() < 1 )
    {
        throw std::runtime_error( "ReLU module requires at least one input" );
    }

    const auto &inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower();
    torch::Tensor inputUpperBound = inputBoundsPair.upper();

    if ( _input_size == 0 && inputLowerBound.defined() )
    {
        _input_size = inputLowerBound.numel();
    }

    torch::Tensor lowerBound = torch::clamp_min( inputLowerBound, 0 );
    torch::Tensor upperBound = torch::clamp_min( inputUpperBound, 0 );

    if ( _output_size == 0 && lowerBound.defined() )
    {
        _output_size = lowerBound.numel();
    }

    return BoundedTensor<torch::Tensor>( lowerBound, upperBound );
}

unsigned BoundedReLUNode::getInputSize() const
{
    return _input_size;
}

unsigned BoundedReLUNode::getOutputSize() const
{
    if ( _output_size == 0 && _input_size > 0 )
    {
        return _input_size;
    }
    if ( _output_size == 0 )
    {
        return 2;
    }
    return _output_size;
}

void BoundedReLUNode::setInputSize( unsigned size )
{
    _input_size = size;
}

void BoundedReLUNode::setOutputSize( unsigned size )
{
    _output_size = size;
}

BoundedReLUNode::RelaxationResult
BoundedReLUNode::_backwardRelaxation( const BoundA &last_lA, const BoundA &last_uA,
                                      const torch::Tensor &input_lower,
                                      const torch::Tensor &input_upper )
{
    RelaxationResult result;

    auto [upper_d, upper_b] = _reluUpperBound( input_lower, input_upper );
    torch::Tensor lower_d = _computeStandardCROWNLowerBound( input_lower, input_upper );

    init_d = lower_d.detach().clone();
    init_upper_d = upper_d.detach().clone();

    result.d_lower = lower_d;
    result.d_upper = upper_d;
    // Lower relaxation passes through origin: y >= alpha*x + 0
    result.bias_lower = torch::zeros_like( input_lower );
    result.bias_upper = upper_b;

    _maskAlpha( input_lower, input_upper, upper_d, result );

    return result;
}

std::pair<torch::Tensor, torch::Tensor> BoundedReLUNode::_reluUpperBound( const torch::Tensor &lb,
                                                                          const torch::Tensor &ub )
{
    torch::Tensor lb_r = torch::clamp_max( lb, 0 );
    torch::Tensor ub_r = torch::clamp_min( ub, 0 );
    ub_r = torch::max( ub_r, lb_r + 1e-8 ); // Avoid division by zero

    torch::Tensor upper_d = ub_r / ( ub_r - lb_r );
    torch::Tensor upper_b = -lb_r * upper_d;

    return std::make_pair( upper_d, upper_b );
}

torch::Tensor BoundedReLUNode::_computeStandardCROWNLowerBound( const torch::Tensor &input_lower,
                                                                const torch::Tensor &input_upper )
{
    torch::Tensor slopes_lower = torch::zeros_like( input_lower );

    auto always_active_mask = input_lower >= 0;
    slopes_lower =
        torch::where( always_active_mask, torch::ones_like( slopes_lower ), slopes_lower );

    // Always compute to avoid GPU->CPU sync from .any().item<bool>()
    auto uncertain_mask = ( input_lower < 0 ) & ( input_upper > 0 );
    {
        torch::Tensor ub_r = torch::clamp_min( input_upper, 0 );
        torch::Tensor lb_r = torch::clamp_max( input_lower, 0 );
        ub_r = torch::max( ub_r, lb_r + 1e-8 );
        torch::Tensor upper_slope = ub_r / ( ub_r - lb_r );

        auto adaptive_mask = upper_slope > 0.5;
        torch::Tensor lower_slope = torch::where( adaptive_mask, torch::ones_like( input_lower ),
                                                  torch::zeros_like( input_lower ) );
        slopes_lower = torch::where( uncertain_mask, lower_slope, slopes_lower );
    }

    return slopes_lower;
}

// Deprecated - alpha is now fetched directly in boundBackward
void BoundedReLUNode::_maskAlpha( const torch::Tensor &input_lower,
                                  const torch::Tensor &input_upper, const torch::Tensor &upper_d,
                                  RelaxationResult &result )
{
    const int64_t lanes = input_lower.dim() >= 2 ? input_lower.size( 0 ) : 1;
    auto input_lb_rows = input_lower.reshape( { lanes, -1 } );
    auto input_ub_rows = input_upper.reshape( { lanes, -1 } );
    auto always_active_mask = input_lb_rows >= 0;
    auto always_inactive_mask = input_ub_rows <= 0;
    int outDim = (int)input_lb_rows.size( 1 );

    if ( isAlphaOptimizationEnabled() && _alphaCrownAnalysis &&
         ( _optimizationStage == "opt" || _optimizationStage == "reuse" ) && _currentSpecDim > 0 )
    {
        auto *crown = _alphaCrownAnalysis->getCROWNAnalysis();
        std::string startKey = crown->currentStartKey();
        if ( startKey.empty() )
            startKey = "default";

        int specDim = _currentSpecDim;
        Vector<unsigned> currentSpecIndices;
        bool hasSpecLookup = false;
        Vector<unsigned> cachedSpecIndices;
        bool cachedSparseMode = false;
        unsigned cachedNodeSize = 0;
        if ( crown )
        {
            currentSpecIndices = crown->currentStartSpecIndices();
            hasSpecLookup = crown->getAlphaStartCacheInfo( startKey, cachedSpecIndices,
                                                           cachedSparseMode, cachedNodeSize );
        }

        auto alphaResult = _alphaCrownAnalysis->getAlphaForNodeAllSpecs(
            getNodeIndex(), startKey, specDim, outDim, input_lower, input_upper );

        // Alpha columns are positional per unstable neuron, so a slot-count
        // mismatch means the cache is stale - fall back to CROWN slopes
        bool alpha_consistent =
            alphaResult.alpha.defined() && alphaResult.alpha.size( 2 ) == lanes &&
            alphaResult.alpha.size( -1 ) == (long)alphaResult.numUnstable &&
            alphaResult.unstableIndices.defined() &&
            alphaResult.unstableIndices.numel() == (long)alphaResult.numUnstable;

        if ( alpha_consistent && alphaResult.numUnstable > 0 && alphaResult.alpha.numel() > 0 )
        {
            // Clone alpha for fresh computation graph per iteration
            auto alpha_unstable = alphaResult.alpha.clone();

            if ( alphaResult.hasSpecDefaultSlot && hasSpecLookup && cachedSparseMode &&
                 cachedNodeSize > 0 )
            {
                auto lookup = torch::zeros( { (long long)cachedNodeSize },
                                            torch::TensorOptions()
                                                .dtype( torch::kLong )
                                                .device( alpha_unstable.device() ) );
                for ( int i = 0; i < (int)cachedSpecIndices.size(); ++i )
                {
                    unsigned idx = cachedSpecIndices[i];
                    if ( idx < cachedNodeSize )
                    {
                        lookup[idx] = i + 1;
                    }
                }

                if ( currentSpecIndices.size() > 0 )
                {
                    auto idxTensor = torch::empty( { (long long)currentSpecIndices.size() },
                                                   torch::TensorOptions()
                                                       .dtype( torch::kLong )
                                                       .device( alpha_unstable.device() ) );
                    for ( int i = 0; i < (int)currentSpecIndices.size(); ++i )
                    {
                        unsigned idx = currentSpecIndices[i];
                        idxTensor[i] = ( idx < cachedNodeSize )
                                           ? static_cast<int64_t>( lookup[idx].item<int64_t>() )
                                           : static_cast<int64_t>( 0 );
                    }
                    alpha_unstable = alpha_unstable.index_select( 1, idxTensor );
                    specDim = (int)alpha_unstable.size( 1 );
                }
            }

            // Avoid in-place ops (index_put_, scatter_) to prevent
            // "backward through graph a second time" errors
            auto options = alpha_unstable.options();

            int alphaSpecDim =
                (int)( alpha_unstable.defined() ? alpha_unstable.size( 1 ) : specDim );
            specDim = alphaSpecDim;

            const std::vector<int64_t> fullShape = { specDim, lanes, outDim };
            auto always_active_expanded = always_active_mask.unsqueeze( 0 ).expand( fullShape );

            // Alpha is (spec, lanes, unstable), scattered at the union set shared by every lane
            auto alpha_lower_unstable = alpha_unstable[0];
            auto alpha_upper_unstable = alpha_unstable[1];

            auto indices = alphaResult.unstableIndices.to( options.device() )
                               .reshape( { 1, 1, alphaResult.numUnstable } )
                               .expand( { specDim, lanes, alphaResult.numUnstable } );

            torch::Tensor alpha_lower_full =
                torch::zeros( fullShape, options ).scatter( 2, indices, alpha_lower_unstable );
            torch::Tensor alpha_upper_full =
                torch::zeros( fullShape, options ).scatter( 2, indices, alpha_upper_unstable );

            alpha_lower_full = torch::where( always_active_expanded,
                                             torch::ones( fullShape, options ), alpha_lower_full );
            alpha_upper_full = torch::where( always_active_expanded,
                                             torch::ones( fullShape, options ), alpha_upper_full );

            // A neuron that is stable in one lane takes that lane's fixed slope whatever its alpha
            // says, and one that became inactive after its alpha was built outputs zero
            auto always_inactive_expanded = always_inactive_mask.unsqueeze( 0 ).expand( fullShape );
            alpha_lower_full = torch::where( always_inactive_expanded,
                                             torch::zeros( fullShape, options ), alpha_lower_full );
            alpha_upper_full = torch::where( always_inactive_expanded,
                                             torch::zeros( fullShape, options ), alpha_upper_full );

            auto upper_d_rows = upper_d.reshape( { lanes, -1 } );
            auto upper_d_masked =
                torch::where( always_active_mask, torch::ones_like( upper_d_rows ), upper_d_rows );
            upper_d_masked = torch::where( always_inactive_mask,
                                           torch::zeros_like( upper_d_masked ), upper_d_masked );
            auto k_upper_spec = upper_d_masked.unsqueeze( 0 ).expand( fullShape );

            result.lb_lower_d = alpha_lower_full;
            result.lb_upper_d = k_upper_spec;

            result.ub_upper_d = k_upper_spec;
            result.ub_lower_d = alpha_upper_full;

            auto b_upper = -input_lb_rows * upper_d_rows;
            auto b_upper_masked = torch::where( always_active_mask | always_inactive_mask,
                                                torch::zeros_like( b_upper ), b_upper );
            result.bias_lower = torch::zeros_like( input_lb_rows );
            result.bias_upper = b_upper_masked;

            return;
        }
    }
}

void BoundedReLUNode::computeAlphaRelaxation( const torch::Tensor &last_lA,
                                              const torch::Tensor &last_uA,
                                              const torch::Tensor &input_lower,
                                              const torch::Tensor &input_upper,
                                              torch::Tensor &d_lower, torch::Tensor &d_upper,
                                              torch::Tensor &bias_lower, torch::Tensor &bias_upper )
{

    auto result =
        _backwardRelaxation( BoundA( last_lA ), BoundA( last_uA ), input_lower, input_upper );

    d_lower = result.d_lower;
    d_upper = result.d_upper;
    bias_lower = result.bias_lower;
    bias_upper = result.bias_upper;
}

torch::Tensor BoundedReLUNode::getCROWNSlope( bool isLowerBound ) const
{
    if ( isLowerBound )
    {
        if ( !hasInitD() )
        {
            return torch::full( { getOutputSize() }, 0.5f, torch::kFloat32 );
        }
        return init_d;
    }
    else
    {
        if ( !init_upper_d.defined() || init_upper_d.numel() == 0 )
        {
            return torch::full( { getOutputSize() }, 1.0f, torch::kFloat32 );
        }
        return init_upper_d;
    }
}

} // namespace NLR
