/*********************                                                        */
/*! \file BoundedSigmoidNode.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "BoundedSigmoidNode.h"
#include "AlphaCROWNAnalysis.h"
#include "configuration/LunaConfiguration.h"
#include "conv/Patches.h"
#include "Debug.h"
#include <algorithm>
#include <cmath>

namespace NLR
{

BoundedSigmoidNode::BoundedSigmoidNode( const torch::nn::Sigmoid &sigmoidModule,
                                        const String &name )
    : BoundedAlphaOptimizeNode()
    , _sigmoidModule( std::make_shared<torch::nn::Sigmoid>( sigmoidModule ) )
    , x_limit( LunaConfiguration::SIGMOID_CUTOFF_CONSTANT )
{
    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;
    num_points_pre = static_cast<int>( x_limit / step_pre );
    _lookupTablesInitialized = false;

    _requiresInputBounds.append( 0 );
    _ibpIntermediate = true;
}

torch::Tensor BoundedSigmoidNode::forward( const torch::Tensor &input )
{
    if ( input.dim() > 0 )
    {
        _input_size = input.numel();
        _output_size = input.numel();
    }

    return ( *_sigmoidModule )( input );
}

void BoundedSigmoidNode::moveToDevice( const torch::Device &device )
{
    BoundedAlphaOptimizeNode::moveToDevice( device );
    if ( _sigmoidModule )
    {
        _sigmoidModule->ptr()->to( device );
    }
    if ( d_lower.defined() )
        d_lower = d_lower.to( device );
    if ( d_upper.defined() )
        d_upper = d_upper.to( device );
    if ( dfunc_values.defined() )
        dfunc_values = dfunc_values.to( device );
    if ( init_lower_d.defined() )
        init_lower_d = init_lower_d.to( device );
    if ( init_upper_d.defined() )
        init_upper_d = init_upper_d.to( device );
    if ( mask_pos.defined() )
        mask_pos = mask_pos.to( device );
    if ( mask_neg.defined() )
        mask_neg = mask_neg.to( device );
    if ( mask_both.defined() )
        mask_both = mask_both.to( device );
    if ( lw.defined() )
        lw = lw.to( device );
    if ( lb.defined() )
        lb = lb.to( device );
    if ( uw.defined() )
        uw = uw.to( device );
    if ( ub.defined() )
        ub = ub.to( device );
}

torch::Tensor BoundedSigmoidNode::sigmoidFunc( const torch::Tensor &x )
{
    return torch::sigmoid( x );
}

torch::Tensor BoundedSigmoidNode::dsigmoidFunc( const torch::Tensor &x )
{
    auto s = torch::sigmoid( x );
    return s * ( 1 - s );
}

void BoundedSigmoidNode::precomputeRelaxation( double required_limit )
{
    double configured_limit = LunaConfiguration::SIGMOID_CUTOFF_CONSTANT;
    double target_limit = configured_limit;
    if ( required_limit > target_limit )
    {
        // Expand precompute range when observed bounds exceed config
        target_limit = required_limit;
    }

    if ( _lookupTablesInitialized && d_lower.defined() && target_limit <= x_limit )
    {
        return;
    }

    x_limit = target_limit;
    step_pre = 0.01;
    num_points_pre = static_cast<int>( x_limit / step_pre );
    int max_iter = 100;

    torch::Device device = _device;

    auto check_lower = [this]( const torch::Tensor &upper, const torch::Tensor &d ) -> torch::Tensor
    {
        torch::Tensor k = dsigmoidFunc( d );
        torch::Tensor y_d = sigmoidFunc( d );
        torch::Tensor y_upper = sigmoidFunc( upper );
        return ( k * ( upper - d ) + y_d ) <= y_upper;
    };

    auto check_upper = [this]( const torch::Tensor &lower, const torch::Tensor &d ) -> torch::Tensor
    {
        torch::Tensor k = dsigmoidFunc( d );
        torch::Tensor y_d = sigmoidFunc( d );
        torch::Tensor y_lower = sigmoidFunc( lower );
        return ( k * ( lower - d ) + y_d ) >= y_lower;
    };

    auto options = torch::TensorOptions().dtype( torch::kFloat32 ).device( device );
    torch::Tensor upper = step_pre * torch::arange( 0, num_points_pre + 5, options );
    torch::Tensor r = torch::zeros_like( upper );
    torch::Tensor l = -torch::ones_like( upper );

    while ( true )
    {
        torch::Tensor checked = check_lower( upper, l ).to( torch::kInt64 );
        l = checked * l + ( 1 - checked ) * ( l * 2 );
        if ( checked.sum().item<int64_t>() == l.numel() )
        {
            break;
        }
    }

    for ( int i = 0; i < max_iter; ++i )
    {
        torch::Tensor m = ( l + r ) / 2;
        torch::Tensor checked = check_lower( upper, m ).to( torch::kInt64 );
        l = checked * m + ( 1 - checked ) * l;
        r = checked * r + ( 1 - checked ) * m;
    }

    d_lower = l.clone();

    torch::Tensor lower = -step_pre * torch::arange( 0, num_points_pre + 5, options );
    l = torch::zeros_like( upper );
    r = torch::ones_like( upper );

    while ( true )
    {
        torch::Tensor checked = check_upper( lower, r ).to( torch::kInt64 );
        r = checked * r + ( 1 - checked ) * ( r * 2 );
        if ( checked.sum().item<int64_t>() == l.numel() )
        {
            break;
        }
    }

    for ( int i = 0; i < max_iter; ++i )
    {
        torch::Tensor m = ( l + r ) / 2;
        torch::Tensor checked = check_upper( lower, m ).to( torch::kInt64 );
        l = ( 1 - checked ) * m + checked * l;
        r = ( 1 - checked ) * r + checked * m;
    }

    d_upper = r.clone();
    _lookupTablesInitialized = true;
}

void BoundedSigmoidNode::precomputeDfuncValues()
{
    if ( !_lookupTablesInitialized )
    {
        precomputeRelaxation();
    }

    torch::Tensor upper =
        step_pre *
        torch::arange( 0, num_points_pre + 5, torch::TensorOptions().dtype( torch::kFloat32 ) );
    dfunc_values = dsigmoidFunc( upper );
}

torch::Tensor BoundedSigmoidNode::retrieveFromPrecompute( const torch::Tensor &precomputed_d,
                                                          const torch::Tensor &input_bound,
                                                          const torch::Tensor &default_d )
{
    if ( !_lookupTablesInitialized || !precomputed_d.defined() )
    {
        precomputeRelaxation();
    }

    torch::Tensor precomputed_d_aligned =
        precomputed_d.to( input_bound.device() ).to( input_bound.dtype() );
    torch::Tensor default_d_aligned =
        default_d.to( input_bound.device() ).to( input_bound.dtype() );

    torch::Tensor index =
        torch::max(
            torch::zeros(
                { input_bound.numel() },
                torch::TensorOptions().dtype( torch::kInt64 ).device( input_bound.device() ) ),
            ( input_bound / step_pre ).to( torch::kInt64 ).reshape( -1 ) ) +
        1;

    // Out-of-range indices fall back to default tangent points
    if ( index.max().item<int64_t>() >= precomputed_d_aligned.numel() )
    {
        torch::Tensor mask = ( index < precomputed_d_aligned.numel() ).view( input_bound.sizes() );
        torch::Tensor clamped_index =
            torch::clamp( index, 0, static_cast<int64_t>( precomputed_d_aligned.numel() - 1 ) );
        torch::Tensor selected = torch::index_select( precomputed_d_aligned, 0, clamped_index )
                                     .view( input_bound.sizes() );
        return torch::where( mask, selected, default_d_aligned ).view( input_bound.sizes() );
    }
    else
    {
        return torch::index_select( precomputed_d_aligned, 0, index ).view( input_bound.sizes() );
    }
}

std::pair<torch::Tensor, torch::Tensor>
BoundedSigmoidNode::generateDLowerUpper( const torch::Tensor &lower, const torch::Tensor &upper )
{
    double needed_limit = std::max( std::fabs( static_cast<double>( lower.min().item<float>() ) ),
                                    std::fabs( static_cast<double>( upper.max().item<float>() ) ) );
    needed_limit += 1.0;
    if ( !_lookupTablesInitialized || !d_lower.defined() || needed_limit > x_limit )
    {
        precomputeRelaxation( needed_limit );
    }

    if ( !d_lower.defined() || d_lower.device() != lower.device() ||
         d_lower.dtype() != lower.dtype() )
    {
        if ( !d_lower.defined() )
        {
            precomputeRelaxation();
        }
        d_lower = d_lower.to( lower.device() ).to( lower.dtype() );
        d_upper = d_upper.to( lower.device() ).to( lower.dtype() );
    }

    torch::Tensor d_lower_result = retrieveFromPrecompute( d_lower, upper, lower );
    torch::Tensor d_upper_result = retrieveFromPrecompute( d_upper, -lower, upper );

    return std::make_pair( d_lower_result, d_upper_result );
}

std::pair<torch::Tensor, torch::Tensor> BoundedSigmoidNode::retrieveDFromK( const torch::Tensor &k )
{
    if ( !_lookupTablesInitialized || !dfunc_values.defined() )
    {
        precomputeDfuncValues();
    }

    if ( !dfunc_values.defined() || dfunc_values.device() != k.device() ||
         dfunc_values.dtype() != k.dtype() )
    {
        if ( !dfunc_values.defined() )
        {
            precomputeDfuncValues();
        }
        dfunc_values = dfunc_values.to( k.device() ).to( k.dtype() );
    }

    torch::Tensor dfunc_flipped = torch::flip( dfunc_values, { 0 } );
    torch::Tensor d_indices = torch::searchsorted( dfunc_flipped, k, /*right=*/false );
    d_indices = num_points_pre - d_indices + 4;

    torch::Tensor d_left = d_indices * step_pre;
    torch::Tensor d_right = d_left + step_pre;
    torch::Tensor y_left = sigmoidFunc( d_left );
    torch::Tensor y_right = sigmoidFunc( d_right );

    torch::Tensor clamped_indices =
        torch::clamp( d_indices, 0, static_cast<int64_t>( dfunc_values.size( 0 ) - 1 ) );
    torch::Tensor k_left = dfunc_values.index_select( 0, clamped_indices );
    torch::Tensor k_right_indices =
        torch::clamp( d_indices + 1, 0, static_cast<int64_t>( dfunc_values.size( 0 ) - 1 ) );
    torch::Tensor k_right = dfunc_values.index_select( 0, k_right_indices );

    torch::Tensor denominator = ( k_left - k_right ).clamp_min( 1e-8 );
    torch::Tensor d_return =
        ( k_left * d_left - k_right * d_right - y_left + y_right ) / denominator;

    torch::Tensor mask_almost_the_same = ( k_left - k_right ).abs() < 1e-5;
    d_return = torch::where( mask_almost_the_same, d_left, d_return );
    torch::Tensor y_d = k_left * ( d_return - d_left ) + y_left;

    return std::make_pair( d_return, y_d );
}

BoundedTensor<torch::Tensor> BoundedSigmoidNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>> &inputBounds )
{

    if ( inputBounds.size() < 1 )
    {
        throw std::runtime_error( "Sigmoid module requires at least one input" );
    }

    const auto &inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower();
    torch::Tensor inputUpperBound = inputBoundsPair.upper();

    if ( _input_size == 0 && inputLowerBound.defined() )
    {
        _input_size = inputLowerBound.numel();
    }

    // Sigmoid is monotonic
    torch::Tensor lowerBound = sigmoidFunc( inputLowerBound );
    torch::Tensor upperBound = sigmoidFunc( inputUpperBound );

    if ( _output_size == 0 && lowerBound.defined() )
    {
        _output_size = lowerBound.numel();
    }

    return BoundedTensor<torch::Tensor>( lowerBound, upperBound );
}

unsigned BoundedSigmoidNode::getInputSize() const
{
    return _input_size;
}

unsigned BoundedSigmoidNode::getOutputSize() const
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

void BoundedSigmoidNode::setInputSize( unsigned size )
{
    _input_size = size;
}

void BoundedSigmoidNode::setOutputSize( unsigned size )
{
    _output_size = size;
}

void BoundedSigmoidNode::boundRelaxImpl( const torch::Tensor &input_lower,
                                         const torch::Tensor &input_upper )
{
    mask_pos = input_lower >= 0;
    mask_neg = input_upper <= 0;
    mask_both = torch::logical_not( torch::logical_or( mask_pos, mask_neg ) );

    lw = torch::zeros_like( input_lower );
    lb = torch::zeros_like( input_lower );
    uw = torch::zeros_like( input_lower );
    ub = torch::zeros_like( input_lower );

    torch::Tensor y_l = sigmoidFunc( input_lower );
    torch::Tensor y_u = sigmoidFunc( input_upper );

    torch::Tensor k_direct = ( y_u - y_l ) / ( input_upper - input_lower ).clamp_min( 1e-8 );
    torch::Tensor mask_almost_the_same = ( input_upper - input_lower ).abs() < 1e-4;
    k_direct = torch::where( mask_almost_the_same, dsigmoidFunc( input_lower ), k_direct );

    // Upper bound when input <= 0: direct secant line
    uw = torch::where( mask_neg, k_direct, uw );
    ub = torch::where( mask_neg, y_l - k_direct * input_lower, ub );

    // Lower bound when input >= 0: direct secant line
    lw = torch::where( mask_pos, k_direct, lw );
    lb = torch::where( mask_pos, y_l - k_direct * input_lower, lb );

    auto [d_lower_precomputed, d_upper_precomputed] =
        generateDLowerUpper( input_lower, input_upper );

    torch::Tensor k_lower = dsigmoidFunc( input_lower );
    torch::Tensor k_upper = dsigmoidFunc( input_upper );
    torch::Tensor mask_direct_lower = torch::logical_and( mask_both, k_direct < k_lower );
    torch::Tensor mask_direct_upper = torch::logical_and( mask_both, k_direct < k_upper );

    lw = torch::where( mask_direct_lower, k_direct, lw );
    lb = torch::where( mask_direct_lower, y_l - k_direct * input_lower, lb );

    uw = torch::where( mask_direct_upper, k_direct, uw );
    ub = torch::where( mask_direct_upper, y_l - k_direct * input_lower, ub );

    torch::Tensor mask_both_lower =
        torch::logical_and( mask_both, torch::logical_not( mask_direct_lower ) );
    torch::Tensor mask_both_upper =
        torch::logical_and( mask_both, torch::logical_not( mask_direct_upper ) );

    torch::Tensor k_lower_tangent = dsigmoidFunc( d_lower_precomputed );
    torch::Tensor y_lower_tangent = sigmoidFunc( d_lower_precomputed );
    lw = torch::where( mask_both_lower, k_lower_tangent, lw );
    lb = torch::where( mask_both_lower, y_lower_tangent - k_lower_tangent * d_lower_precomputed,
                       lb );

    torch::Tensor k_upper_tangent = dsigmoidFunc( d_upper_precomputed );
    torch::Tensor y_upper_tangent = sigmoidFunc( d_upper_precomputed );
    uw = torch::where( mask_both_upper, k_upper_tangent, uw );
    ub = torch::where( mask_both_upper, y_upper_tangent - k_upper_tangent * d_upper_precomputed,
                       ub );

    // mask_neg lower bound: midpoint tangent
    torch::Tensor m = ( input_lower + input_upper ) / 2;
    torch::Tensor y_m = sigmoidFunc( m );
    torch::Tensor k_m = dsigmoidFunc( m );
    lw = torch::where( mask_neg, k_m, lw );
    lb = torch::where( mask_neg, y_m - k_m * m, lb );

    // mask_pos upper bound: midpoint tangent
    uw = torch::where( mask_pos, k_m, uw );
    ub = torch::where( mask_pos, y_m - k_m * m, ub );
}

BoundedSigmoidNode::RelaxationResult
BoundedSigmoidNode::_backwardRelaxation( const BoundA & /* last_lA */, const BoundA & /* last_uA */,
                                         const torch::Tensor &input_lower,
                                         const torch::Tensor &input_upper )
{
    RelaxationResult result;

    boundRelaxImpl( input_lower, input_upper );

    result.d_lower = lw;
    result.d_upper = uw;
    result.bias_lower = lb;
    result.bias_upper = ub;

    // CROWN slopes become alpha initialization values
    init_lower_d = lw.detach().clone();
    init_upper_d = uw.detach().clone();

    if ( isAlphaOptimizationEnabled() && _alphaCrownAnalysis &&
         ( _optimizationStage == "opt" || _optimizationStage == "reuse" ) && _currentSpecDim > 0 )
    {
        auto *crown = _alphaCrownAnalysis->getCROWNAnalysis();
        std::string startKey = crown ? crown->currentStartKey() : std::string();
        if ( startKey.empty() )
            startKey = "default";

        auto lb_flat = input_lower.flatten();
        auto ub_flat = input_upper.flatten();
        int outDim = (int)lb_flat.numel();
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

        if ( alphaResult.numUnstable > 0 && alphaResult.alpha.defined() &&
             alphaResult.alpha.numel() > 0 )
        {
            // View into stored alpha so in-place projection updates optimizer parameters
            auto alpha_unstable = alphaResult.alpha;

            auto d_pair = generateDLowerUpper( input_lower, input_upper );
            auto d_lower_pre_flat = d_pair.first.flatten();
            auto d_upper_pre_flat = d_pair.second.flatten();

            // In-place projection: keep tangent-point channels inside valid interval-dependent
            // ranges
            if ( alpha_unstable.dim() == 3 && alpha_unstable.size( 2 ) >= 8 )
            {
                torch::NoGradGuard no_grad;
                int64_t specAlpha = alpha_unstable.size( 0 );
                auto idx = alphaResult.unstableIndices.to( lb_flat.device() );
                auto lb_unstable = lb_flat.index_select( 0, idx ).unsqueeze( 0 ).expand(
                    { specAlpha, alphaResult.numUnstable } );
                auto ub_unstable = ub_flat.index_select( 0, idx ).unsqueeze( 0 ).expand(
                    { specAlpha, alphaResult.numUnstable } );
                auto d_lower_unstable =
                    d_lower_pre_flat.index_select( 0, idx ).unsqueeze( 0 ).expand(
                        { specAlpha, alphaResult.numUnstable } );
                auto d_upper_unstable =
                    d_upper_pre_flat.index_select( 0, idx ).unsqueeze( 0 ).expand(
                        { specAlpha, alphaResult.numUnstable } );

                auto project_slot =
                    [&]( int slot, const torch::Tensor &lo, const torch::Tensor &hi )
                {
                    auto s = alpha_unstable.select( 2, slot );
                    auto p = torch::max( torch::min( s, hi ), lo );
                    alpha_unstable.index_put_(
                        { torch::indexing::Slice(), torch::indexing::Slice(), slot }, p );
                };

                // tp_pos / tp_neg in [lb, ub]
                project_slot( 0, lb_unstable, ub_unstable );
                project_slot( 1, lb_unstable, ub_unstable );
                project_slot( 2, lb_unstable, ub_unstable );
                project_slot( 3, lb_unstable, ub_unstable );
                // cross-zero lower tangents in [lb, d_lower_pre]
                project_slot( 4, lb_unstable, d_lower_unstable );
                project_slot( 5, lb_unstable, d_lower_unstable );
                // cross-zero upper tangents in [d_upper_pre, ub]
                project_slot( 6, d_upper_unstable, ub_unstable );
                project_slot( 7, d_upper_unstable, ub_unstable );
            }

            // Map sparse-spec alpha into current spec ordering
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
                    alpha_unstable = alpha_unstable.index_select( 0, idxTensor );
                    specDim = (int)alpha_unstable.size( 0 );
                }
            }

            auto options = alpha_unstable.options();
            specDim = (int)alpha_unstable.size( 0 );
            auto indices = alphaResult.unstableIndices.to( options.device() )
                               .unsqueeze( 0 )
                               .expand( { specDim, alphaResult.numUnstable } );
            auto make_alpha_full = [&]( const torch::Tensor &alpha_slice_2d ) {
                return torch::zeros( { specDim, outDim }, options )
                    .scatter( 1, indices, alpha_slice_2d );
            };

            torch::Tensor tp_pos_l, tp_pos_u;
            torch::Tensor tp_neg_l, tp_neg_u;
            torch::Tensor tp_both_lower_l, tp_both_lower_u;
            torch::Tensor tp_both_upper_l, tp_both_upper_u;

            if ( alpha_unstable.dim() == 3 && alpha_unstable.size( 2 ) >= 8 )
            {
                tp_pos_l = make_alpha_full( alpha_unstable.select( 2, 0 ) );
                tp_pos_u = make_alpha_full( alpha_unstable.select( 2, 1 ) );
                tp_neg_l = make_alpha_full( alpha_unstable.select( 2, 2 ) );
                tp_neg_u = make_alpha_full( alpha_unstable.select( 2, 3 ) );
                tp_both_lower_l = make_alpha_full( alpha_unstable.select( 2, 4 ) );
                tp_both_lower_u = make_alpha_full( alpha_unstable.select( 2, 5 ) );
                tp_both_upper_l = make_alpha_full( alpha_unstable.select( 2, 6 ) );
                tp_both_upper_u = make_alpha_full( alpha_unstable.select( 2, 7 ) );
            }
            else
            {
                // Backward-compatible fallback: share one alpha across all branches
                torch::Tensor alpha_base =
                    ( alpha_unstable.dim() == 3 ) ? alpha_unstable.select( 2, 0 ) : alpha_unstable;
                torch::Tensor alpha_full = make_alpha_full( alpha_base );
                tp_pos_l = alpha_full;
                tp_pos_u = alpha_full;
                tp_neg_l = alpha_full;
                tp_neg_u = alpha_full;
                tp_both_lower_l = alpha_full;
                tp_both_lower_u = alpha_full;
                tp_both_upper_l = alpha_full;
                tp_both_upper_u = alpha_full;
            }

            torch::Tensor y_l = sigmoidFunc( input_lower );
            torch::Tensor y_u = sigmoidFunc( input_upper );
            torch::Tensor k_direct =
                ( y_u - y_l ) / ( input_upper - input_lower ).clamp_min( 1e-8 );
            torch::Tensor mask_almost_the_same = ( input_upper - input_lower ).abs() < 1e-4;
            k_direct = torch::where( mask_almost_the_same, dsigmoidFunc( input_lower ), k_direct );

            auto mask_pos_flat = ( input_lower >= 0 ).flatten();
            auto mask_neg_flat = ( input_upper <= 0 ).flatten();
            auto mask_both_flat =
                torch::logical_not( torch::logical_or( mask_pos_flat, mask_neg_flat ) );
            auto k_lower_flat = dsigmoidFunc( input_lower ).flatten();
            auto k_upper_flat = dsigmoidFunc( input_upper ).flatten();
            auto k_direct_flat = k_direct.flatten();
            auto mask_direct_lower_flat =
                torch::logical_and( mask_both_flat, k_direct_flat < k_lower_flat );
            auto mask_direct_upper_flat =
                torch::logical_and( mask_both_flat, k_direct_flat < k_upper_flat );

            // Optimize tangent-point selection for cross-zero neurons
            auto opt_mask_lower_flat =
                torch::logical_and( mask_both_flat, torch::logical_not( mask_direct_lower_flat ) );
            auto opt_mask_upper_flat =
                torch::logical_and( mask_both_flat, torch::logical_not( mask_direct_upper_flat ) );

            auto lb_expanded = lb_flat.unsqueeze( 0 ).expand( { specDim, outDim } );
            auto ub_expanded = ub_flat.unsqueeze( 0 ).expand( { specDim, outDim } );
            auto d_lower_expanded = d_lower_pre_flat.unsqueeze( 0 ).expand( { specDim, outDim } );
            auto d_upper_expanded = d_upper_pre_flat.unsqueeze( 0 ).expand( { specDim, outDim } );

            auto clamp_interval = [&]( const torch::Tensor &t )
            { return torch::max( torch::min( t, ub_expanded ), lb_expanded ); };
            tp_pos_l = clamp_interval( tp_pos_l );
            tp_pos_u = clamp_interval( tp_pos_u );
            tp_neg_l = clamp_interval( tp_neg_l );
            tp_neg_u = clamp_interval( tp_neg_u );
            // Lower branch tangent in [lb, d_lower_pre]; upper branch in [d_upper_pre, ub]
            tp_both_lower_l =
                torch::max( torch::min( tp_both_lower_l, d_lower_expanded ), lb_expanded );
            tp_both_lower_u =
                torch::max( torch::min( tp_both_lower_u, d_lower_expanded ), lb_expanded );
            tp_both_upper_l =
                torch::min( torch::max( tp_both_upper_l, d_upper_expanded ), ub_expanded );
            tp_both_upper_u =
                torch::min( torch::max( tp_both_upper_u, d_upper_expanded ), ub_expanded );

            auto k_tp = [&]( const torch::Tensor &tp ) { return dsigmoidFunc( tp ); };
            auto b_tp = [&]( const torch::Tensor &tp )
            {
                auto k = dsigmoidFunc( tp );
                auto y = sigmoidFunc( tp );
                return y - k * tp;
            };

            auto k_both_lower_l = k_tp( tp_both_lower_l );
            auto k_both_lower_u = k_tp( tp_both_lower_u );
            auto k_both_upper_l = k_tp( tp_both_upper_l );
            auto k_both_upper_u = k_tp( tp_both_upper_u );
            auto b_both_lower_l = b_tp( tp_both_lower_l );
            auto b_both_lower_u = b_tp( tp_both_lower_u );
            auto b_both_upper_l = b_tp( tp_both_upper_l );
            auto b_both_upper_u = b_tp( tp_both_upper_u );

            auto k_neg_l = k_tp( tp_neg_l );
            auto k_neg_u = k_tp( tp_neg_u );
            auto k_pos_l = k_tp( tp_pos_l );
            auto k_pos_u = k_tp( tp_pos_u );
            auto b_neg_l = b_tp( tp_neg_l );
            auto b_neg_u = b_tp( tp_neg_u );
            auto b_pos_l = b_tp( tp_pos_l );
            auto b_pos_u = b_tp( tp_pos_u );

            torch::Tensor d_lower_spec =
                result.d_lower.flatten().unsqueeze( 0 ).expand( { specDim, outDim } ).clone();
            torch::Tensor d_upper_spec =
                result.d_upper.flatten().unsqueeze( 0 ).expand( { specDim, outDim } ).clone();
            torch::Tensor b_lower_spec =
                result.bias_lower.flatten().unsqueeze( 0 ).expand( { specDim, outDim } ).clone();
            torch::Tensor b_upper_spec =
                result.bias_upper.flatten().unsqueeze( 0 ).expand( { specDim, outDim } ).clone();

            auto opt_mask_lower = opt_mask_lower_flat.unsqueeze( 0 ).expand( { specDim, outDim } );
            auto opt_mask_upper = opt_mask_upper_flat.unsqueeze( 0 ).expand( { specDim, outDim } );
            auto mask_neg = mask_neg_flat.unsqueeze( 0 ).expand( { specDim, outDim } );
            auto mask_pos = mask_pos_flat.unsqueeze( 0 ).expand( { specDim, outDim } );

            // Lower-path coefficients (used by lA sign-split)
            auto d_lower_l_path = d_lower_spec.clone();
            auto b_lower_l_path = b_lower_spec.clone();
            auto d_upper_l_path = d_upper_spec.clone();
            auto b_upper_l_path = b_upper_spec.clone();

            d_lower_l_path = torch::where( mask_neg, k_neg_l, d_lower_l_path );
            b_lower_l_path = torch::where( mask_neg, b_neg_l, b_lower_l_path );
            d_lower_l_path = torch::where( opt_mask_lower, k_both_lower_l, d_lower_l_path );
            b_lower_l_path = torch::where( opt_mask_lower, b_both_lower_l, b_lower_l_path );

            d_upper_l_path = torch::where( mask_pos, k_pos_l, d_upper_l_path );
            b_upper_l_path = torch::where( mask_pos, b_pos_l, b_upper_l_path );
            d_upper_l_path = torch::where( opt_mask_upper, k_both_upper_l, d_upper_l_path );
            b_upper_l_path = torch::where( opt_mask_upper, b_both_upper_l, b_upper_l_path );

            // Upper-path coefficients (used by uA sign-split)
            auto d_lower_u_path = d_lower_spec.clone();
            auto b_lower_u_path = b_lower_spec.clone();
            auto d_upper_u_path = d_upper_spec.clone();
            auto b_upper_u_path = b_upper_spec.clone();

            d_lower_u_path = torch::where( mask_neg, k_neg_u, d_lower_u_path );
            b_lower_u_path = torch::where( mask_neg, b_neg_u, b_lower_u_path );
            d_lower_u_path = torch::where( opt_mask_lower, k_both_lower_u, d_lower_u_path );
            b_lower_u_path = torch::where( opt_mask_lower, b_both_lower_u, b_lower_u_path );

            d_upper_u_path = torch::where( mask_pos, k_pos_u, d_upper_u_path );
            b_upper_u_path = torch::where( mask_pos, b_pos_u, b_upper_u_path );
            d_upper_u_path = torch::where( opt_mask_upper, k_both_upper_u, d_upper_u_path );
            b_upper_u_path = torch::where( opt_mask_upper, b_both_upper_u, b_upper_u_path );

            result.lb_lower_d = d_lower_l_path;
            result.lb_upper_d = d_upper_l_path;
            result.ub_upper_d = d_upper_u_path;
            result.ub_lower_d = d_lower_u_path;

            result.lb_lower_b = b_lower_l_path;
            result.lb_upper_b = b_upper_l_path;
            result.ub_lower_b = b_lower_u_path;
            result.ub_upper_b = b_upper_u_path;

            result.bias_lower = b_lower_l_path;
            result.bias_upper = b_upper_u_path;
        }
    }

    return result;
}

torch::Tensor BoundedSigmoidNode::maybe_unfold_patches( const torch::Tensor &d_tensor,
                                                        const BoundA &last_A )
{
    if ( !d_tensor.defined() || !last_A.isPatches() )
    {
        return d_tensor;
    }
    auto patches = last_A.asPatches();

    if ( d_tensor.dim() == 3 )
    {
        return maybe_unfold_patches( d_tensor.unsqueeze( 0 ), last_A );
    }

    torch::Tensor d_unfolded = inplace_unfold(
        d_tensor, { patches->patches.size( -2 ), patches->patches.size( -1 ) }, patches->stride,
        patches->padding, patches->inserted_zeros, patches->output_padding );

    return d_unfolded.permute( { 0, 1, 2, 3, 4, 5 } ).unsqueeze( 0 );
}

void BoundedSigmoidNode::boundBackward( const BoundA &last_lA, const BoundA &last_uA,
                                        const Vector<BoundedTensor<torch::Tensor>> &inputBounds,
                                        Vector<Pair<BoundA, BoundA>> &outputA_matrices,
                                        torch::Tensor &lbias, torch::Tensor &ubias )
{

    if ( inputBounds.size() < 1 )
    {
        throw std::runtime_error( "BoundedSigmoidNode expects at least one input" );
    }

    const auto &inputBound = inputBounds[0];
    torch::Tensor input_lower = inputBound.lower();
    torch::Tensor input_upper = inputBound.upper();

    // Spec dim from A matrix for per-spec alpha
    int specDim = 1;
    if ( last_lA.defined() && last_lA.isTensor() )
    {
        specDim = last_lA.inferSpecDim();
    }
    else if ( last_uA.defined() && last_uA.isTensor() )
    {
        specDim = last_uA.inferSpecDim();
    }
    _currentSpecDim = specDim;

    auto relaxation_result = _backwardRelaxation( last_lA, last_uA, input_lower, input_upper );

    // Per-spec slopes: v [spec, out], A [spec, batch, out] -> [spec, 1, out]
    auto expand_like = []( torch::Tensor v, const torch::Tensor &A )
    {
        if ( !v.defined() || !A.defined() )
            return v;

        if ( v.dim() == 2 && A.dim() == 3 && v.size( 0 ) == A.size( 0 ) )
        {
            v = v.unsqueeze( 1 );
            try
            {
                return v.expand_as( A );
            }
            catch ( ... )
            {
                return v;
            }
        }

        // Flat-to-conv mismatch: reshape v when numel matches A's payload dims
        if ( v.numel() > 0 && A.dim() >= 3 )
        {
            int64_t payload_numel = 1;
            for ( int d = 2; d < A.dim(); ++d )
                payload_numel *= A.size( d );
            if ( v.numel() == payload_numel && v.dim() == 1 )
            {
                std::vector<int64_t> payload_shape;
                for ( int d = 2; d < A.dim(); ++d )
                    payload_shape.push_back( A.size( d ) );
                v = v.view( payload_shape ).unsqueeze( 0 ).unsqueeze( 0 );
            }
            else if ( v.numel() == payload_numel && v.dim() == 2 && v.size( 0 ) == 1 )
            {
                std::vector<int64_t> payload_shape;
                for ( int d = 2; d < A.dim(); ++d )
                    payload_shape.push_back( A.size( d ) );
                v = v.view( payload_shape ).unsqueeze( 0 ).unsqueeze( 0 );
            }
        }

        if ( A.dim() >= 2 && v.dim() == A.dim() - 2 )
        {
            v = v.unsqueeze( 0 ).unsqueeze( 0 );
        }
        else if ( A.dim() >= 1 && v.dim() == A.dim() - 1 )
        {
            v = v.unsqueeze( 0 );
        }

        try
        {
            return v.expand_as( A );
        }
        catch ( ... )
        {
            return v;
        }
    };

    auto reduce_bias_like_A = [&]( const torch::Tensor &term, const torch::Tensor &A )
    {
        if ( !term.defined() || !A.defined() )
            return term;
        if ( A.dim() >= 3 )
        {
            std::vector<int64_t> dims;
            for ( int64_t d = 2; d < term.dim(); ++d )
                dims.push_back( d );
            return dims.empty() ? term : term.sum( dims );
        }
        if ( A.dim() == 2 )
        {
            auto reduced = ( term.dim() >= 2 ) ? term.sum( { 1 } ) : term;
            return reduced.dim() == 1 ? reduced.unsqueeze( 1 ) : reduced;
        }
        return term;
    };

    BoundA new_lA, new_uA;

    if ( last_lA.defined() )
    {
        auto aL_l = relaxation_result.lb_lower_d.defined() ? relaxation_result.lb_lower_d
                                                           : relaxation_result.d_lower;
        auto aU_l = relaxation_result.lb_upper_d.defined() ? relaxation_result.lb_upper_d
                                                           : relaxation_result.d_upper;
        auto bL_l =
            relaxation_result.lb_lower_b.defined()
                ? relaxation_result.lb_lower_b
                : ( relaxation_result.bias_lower.defined() ? relaxation_result.bias_lower
                                                           : torch::zeros_like( input_lower ) );
        auto bU_l =
            relaxation_result.lb_upper_b.defined()
                ? relaxation_result.lb_upper_b
                : ( relaxation_result.bias_upper.defined() ? relaxation_result.bias_upper
                                                           : torch::zeros_like( input_lower ) );

        if ( last_lA.isTensor() )
        {
            torch::Tensor lA = last_lA.asTensor();
            auto Apos = torch::clamp_min( lA, 0 );
            auto Aneg = torch::clamp_max( lA, 0 );
            auto aL_l_exp = expand_like( aL_l, lA );
            auto aU_l_exp = expand_like( aU_l, lA );
            auto bL_l_exp = expand_like( bL_l, lA );
            auto bU_l_exp = expand_like( bU_l, lA );
            auto A_l = Apos * aL_l_exp + Aneg * aU_l_exp;
            new_lA = BoundA( A_l );
            auto b_l = Apos * bL_l_exp + Aneg * bU_l_exp;
            auto add_lbias = reduce_bias_like_A( b_l, lA );
            lbias = lbias.defined() ? ( lbias + add_lbias ) : add_lbias;
        }
        else
        {
            auto patches = last_lA.asPatches();
            torch::Tensor aL_l_unfolded = maybe_unfold_patches( aL_l, last_lA );
            torch::Tensor aU_l_unfolded = maybe_unfold_patches( aU_l, last_lA );
            torch::Tensor P = patches->patches;
            torch::Tensor Ppos = torch::clamp_min( P, 0 );
            torch::Tensor Pneg = torch::clamp_max( P, 0 );
            torch::Tensor P_new = Ppos * aL_l_unfolded + Pneg * aU_l_unfolded;
            new_lA = BoundA( patches->create_similar( P_new ) );

            if ( bL_l.dim() == 3 )
                bL_l = bL_l.unsqueeze( 0 );
            if ( bU_l.dim() == 3 )
                bU_l = bU_l.unsqueeze( 0 );
            torch::Tensor bL_l_unfolded = inplace_unfold(
                bL_l, { patches->patches.size( -2 ), patches->patches.size( -1 ) }, patches->stride,
                patches->padding, patches->inserted_zeros, patches->output_padding );
            torch::Tensor bL_l_ready = bL_l_unfolded.permute( { 0, 1, 2, 3, 4, 5 } ).unsqueeze( 0 );
            torch::Tensor bU_l_unfolded = inplace_unfold(
                bU_l, { patches->patches.size( -2 ), patches->patches.size( -1 ) }, patches->stride,
                patches->padding, patches->inserted_zeros, patches->output_padding );
            torch::Tensor bU_l_ready = bU_l_unfolded.permute( { 0, 1, 2, 3, 4, 5 } ).unsqueeze( 0 );
            torch::Tensor total_bias =
                ( Ppos * bL_l_ready + Pneg * bU_l_ready ).sum( { -3, -2, -1 } );
            total_bias = total_bias.permute( { 1, 0, 2, 3 } );
            lbias = lbias.defined() ? ( lbias + total_bias ) : total_bias;
        }
    }

    if ( last_uA.defined() )
    {
        auto aU_u = relaxation_result.ub_upper_d.defined() ? relaxation_result.ub_upper_d
                                                           : relaxation_result.d_upper;
        auto aL_u = relaxation_result.ub_lower_d.defined() ? relaxation_result.ub_lower_d
                                                           : relaxation_result.d_lower;
        auto bU_u =
            relaxation_result.ub_upper_b.defined()
                ? relaxation_result.ub_upper_b
                : ( relaxation_result.bias_upper.defined() ? relaxation_result.bias_upper
                                                           : torch::zeros_like( input_lower ) );
        auto bL_u =
            relaxation_result.ub_lower_b.defined()
                ? relaxation_result.ub_lower_b
                : ( relaxation_result.bias_lower.defined() ? relaxation_result.bias_lower
                                                           : torch::zeros_like( input_lower ) );

        if ( last_uA.isTensor() )
        {
            torch::Tensor uA = last_uA.asTensor();
            auto Apos = torch::clamp_min( uA, 0 );
            auto Aneg = torch::clamp_max( uA, 0 );
            auto aU_u_exp = expand_like( aU_u, uA );
            auto aL_u_exp = expand_like( aL_u, uA );
            auto bU_u_exp = expand_like( bU_u, uA );
            auto bL_u_exp = expand_like( bL_u, uA );
            auto A_u = Apos * aU_u_exp + Aneg * aL_u_exp;
            new_uA = BoundA( A_u );
            auto b_u = Apos * bU_u_exp + Aneg * bL_u_exp;
            auto add_ubias = reduce_bias_like_A( b_u, uA );
            ubias = ubias.defined() ? ( ubias + add_ubias ) : add_ubias;
        }
        else
        {
            auto patches = last_uA.asPatches();
            torch::Tensor aU_u_unfolded = maybe_unfold_patches( aU_u, last_uA );
            torch::Tensor aL_u_unfolded = maybe_unfold_patches( aL_u, last_uA );
            torch::Tensor P = patches->patches;
            torch::Tensor Ppos = torch::clamp_min( P, 0 );
            torch::Tensor Pneg = torch::clamp_max( P, 0 );
            torch::Tensor P_new = Ppos * aU_u_unfolded + Pneg * aL_u_unfolded;
            new_uA = BoundA( patches->create_similar( P_new ) );

            if ( bU_u.dim() == 3 )
                bU_u = bU_u.unsqueeze( 0 );
            if ( bL_u.dim() == 3 )
                bL_u = bL_u.unsqueeze( 0 );
            torch::Tensor bU_u_unfolded = inplace_unfold(
                bU_u, { patches->patches.size( -2 ), patches->patches.size( -1 ) }, patches->stride,
                patches->padding, patches->inserted_zeros, patches->output_padding );
            torch::Tensor bU_u_ready = bU_u_unfolded.permute( { 0, 1, 2, 3, 4, 5 } ).unsqueeze( 0 );
            torch::Tensor bL_u_unfolded = inplace_unfold(
                bL_u, { patches->patches.size( -2 ), patches->patches.size( -1 ) }, patches->stride,
                patches->padding, patches->inserted_zeros, patches->output_padding );
            torch::Tensor bL_u_ready = bL_u_unfolded.permute( { 0, 1, 2, 3, 4, 5 } ).unsqueeze( 0 );
            torch::Tensor total_bias =
                ( Ppos * bU_u_ready + Pneg * bL_u_ready ).sum( { -3, -2, -1 } );
            total_bias = total_bias.permute( { 1, 0, 2, 3 } );
            ubias = ubias.defined() ? ( ubias + total_bias ) : total_bias;
        }
    }

    if ( outputA_matrices.size() == 0 )
    {
        outputA_matrices.append( Pair<BoundA, BoundA>( new_lA, new_uA ) );
    }
}

void BoundedSigmoidNode::computeAlphaRelaxation(
    const torch::Tensor &last_lA, const torch::Tensor &last_uA, const torch::Tensor &input_lower,
    const torch::Tensor &input_upper, torch::Tensor &d_lower, torch::Tensor &d_upper,
    torch::Tensor &bias_lower, torch::Tensor &bias_upper )
{

    auto result =
        _backwardRelaxation( BoundA( last_lA ), BoundA( last_uA ), input_lower, input_upper );

    d_lower = result.d_lower;
    d_upper = result.d_upper;
    bias_lower = result.bias_lower;
    bias_upper = result.bias_upper;
}

torch::Tensor BoundedSigmoidNode::getCROWNSlope( bool isLowerBound ) const
{
    if ( isLowerBound )
    {
        if ( !init_lower_d.defined() || init_lower_d.numel() == 0 )
        {
            return torch::full( { getOutputSize() }, 0.25f, torch::kFloat32 );
        }
        return init_lower_d;
    }
    else
    {
        if ( !init_upper_d.defined() || init_upper_d.numel() == 0 )
        {
            return torch::full( { getOutputSize() }, 0.25f, torch::kFloat32 );
        }
        return init_upper_d;
    }
}

std::pair<torch::Tensor, torch::Tensor>
BoundedSigmoidNode::getDefaultTangentPoints( const torch::Tensor &lower,
                                             const torch::Tensor &upper )
{
    return generateDLowerUpper( lower, upper );
}

} // namespace NLR
