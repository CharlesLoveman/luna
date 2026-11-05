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
#include "conv/Patches.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

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
    auto inferSpecDim = []( const BoundA &A ) -> int
    {
        if ( !A.defined() )
            return 1;
        if ( A.isTensor() )
        {
            torch::Tensor t = A.asTensor();
            if ( !t.defined() )
                return 1;
            if ( t.dim() >= 2 )
                return (int)t.size( 0 );
            return 1;
        }
        if ( A.isPatches() )
        {
            auto p = A.asPatches();
            if ( !p )
                return 1;
            if ( p->unstable_idx.has_value() )
            {
                return (int)p->patches.size( 0 );
            }
            if ( p->patches.dim() == 7 )
            {
                return (int)( p->patches.size( 0 ) * p->patches.size( 2 ) * p->patches.size( 3 ) );
            }
            return (int)p->patches.size( 0 );
        }
        return 1;
    };
    if ( effective_lA.defined() )
    {
        specDim = inferSpecDim( effective_lA );
    }
    else if ( effective_uA.defined() )
    {
        specDim = inferSpecDim( effective_uA );
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

    // v is (lanes, F) for CROWN or (spec, lanes, F) for alpha, A is (spec, lanes or 1, *payload)
    auto broadcast_like = []( const torch::Tensor &v, const torch::Tensor &A )
    {
        if ( !v.defined() || !A.defined() )
            return v;
        if ( A.dim() < 3 || v.dim() < 2 || v.dim() > 3 )
        {
            throw std::runtime_error( "BoundedReLUNode: A must be (spec, batch, ...) and the "
                                      "slopes (batch, F) or (spec, batch, F)" );
        }
        std::vector<int64_t> shape = { v.dim() == 3 ? v.size( 0 ) : 1, v.size( -2 ) };
        int64_t payload_numel = 1;
        for ( int64_t d = 2; d < A.dim(); ++d )
        {
            shape.push_back( A.size( d ) );
            payload_numel *= A.size( d );
        }
        if ( v.size( -1 ) != payload_numel )
        {
            throw std::runtime_error( "BoundedReLUNode: slopes have " +
                                      std::to_string( v.size( -1 ) ) + " features for A with " +
                                      std::to_string( payload_numel ) );
        }
        return v.reshape( shape );
    };

    auto reduce_bias_like_A = [&]( const torch::Tensor &term, const torch::Tensor &A )
    {
        if ( !term.defined() || !A.defined() )
            return term;

        torch::Tensor result;
        if ( A.dim() >= 3 )
        {
            std::vector<int64_t> dims;
            for ( int64_t d = 2; d < term.dim(); ++d )
                dims.push_back( d );
            result = dims.empty() ? term : term.sum( dims );
            if ( result.dim() == 2 && result.size( 0 ) == A.size( 0 ) &&
                 result.size( 1 ) == A.size( 1 ) )
            {
                return result;
            }
        }
        else if ( A.dim() == 2 )
        {
            if ( term.dim() >= 2 )
            {
                result = term.sum( { 1 } );
            }
            else
            {
                result = term;
            }
            if ( result.dim() == 1 )
            {
                result = result.unsqueeze( 1 );
            }
            return result;
        }
        else
        {
            result = term;
        }

        if ( result.dim() == 1 )
        {
            result = result.unsqueeze( 1 );
        }

        return result;
    };

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

        if ( effective_lA.isTensor() )
        {
            torch::Tensor lA = effective_lA.asTensor();
            if ( lA.dim() == 2 )
                lA = lA.unsqueeze( 1 );
            auto Apos = torch::clamp_min( lA, 0 );
            auto Aneg = torch::clamp_max( lA, 0 );

            auto A_l = Apos * broadcast_like( aL_l, lA ) + Aneg * broadcast_like( aU_l, lA );
            auto b_l = Apos * broadcast_like( bL_l, lA ) + Aneg * broadcast_like( bU_l, lA );

            new_lA = BoundA( A_l );
            auto add_lbias = reduce_bias_like_A( b_l, lA );
            lbias = lbias.defined() ? ( lbias + add_lbias ) : add_lbias;
        }
        else
        {
            auto patches = effective_lA.asPatches();
            bool is_sparse = patches->unstable_idx.has_value();

            auto ensure_4d = [&patches]( torch::Tensor &t )
            {
                if ( !t.defined() )
                    return;
                auto &shape = patches->input_shape;
                if ( shape.size() < 4 )
                    return;
                int64_t C = shape[1], H = shape[2], W = shape[3];
                if ( t.dim() == 1 && t.size( 0 ) == C * H * W )
                    t = t.view( { 1, C, H, W } );
                else if ( t.dim() == 2 && t.size( 1 ) == C * H * W )
                    t = t.view( { t.size( 0 ), C, H, W } );
                else if ( t.dim() == 3 )
                    t = t.unsqueeze( 0 );
            };

            auto unfold_and_select_sparse = [&patches, lanes]( const torch::Tensor &t,
                                                               const torch::Tensor &P,
                                                               bool perSpec = false )
            {
                torch::Tensor unfolded = inplace_unfold(
                    t, { P.size( -2 ), P.size( -1 ) }, patches->stride, patches->padding,
                    patches->inserted_zeros, patches->output_padding );
                auto &idx = patches->unstable_idx.value();
                if ( !perSpec )
                {
                    // One image per lane shared by every spec, selected at each spec's location
                    return unfolded.permute( { 1, 2, 0, 3, 4, 5 } ).index( { idx[1], idx[2] } );
                }
                // One image per spec and lane, each spec reads its own image at its own location
                const int64_t specs = unfolded.size( 0 ) / lanes;
                auto rows = unfolded.reshape( { specs, lanes, unfolded.size( 1 ),
                                                unfolded.size( 2 ), unfolded.size( 3 ),
                                                unfolded.size( 4 ), unfolded.size( 5 ) } );
                auto arange_idx = torch::arange( specs, idx[1].options() );
                return rows.index( { arange_idx, torch::indexing::Slice(), idx[1], idx[2] } );
            };

            torch::Tensor P = patches->patches;
            torch::Tensor Ppos = torch::clamp_min( P, 0 );
            torch::Tensor Pneg = torch::clamp_max( P, 0 );

            if ( is_sparse )
            {
                ensure_4d( aL_l );
                ensure_4d( aU_l );
                torch::Tensor aL_sel = unfold_and_select_sparse( aL_l, P, perSpecSlopes );
                torch::Tensor aU_sel = unfold_and_select_sparse( aU_l, P, perSpecSlopes );

                torch::Tensor P_new = Ppos * aL_sel + Pneg * aU_sel;
                new_lA = BoundA( patches->create_similar( P_new ) );

                ensure_4d( bL_l );
                ensure_4d( bU_l );
                torch::Tensor bL_sel = unfold_and_select_sparse( bL_l, P );
                torch::Tensor bU_sel = unfold_and_select_sparse( bU_l, P );
                torch::Tensor term1 = ( Ppos * bL_sel ).sum( { -3, -2, -1 } );
                torch::Tensor term2 = ( Pneg * bU_sel ).sum( { -3, -2, -1 } );
                torch::Tensor total_bias = term1 + term2;
                lbias = lbias.defined() ? ( lbias + total_bias ) : total_bias;
            }
            else
            {
                torch::Tensor aL_l_unfolded = maybe_unfold_patches( aL_l, effective_lA );
                torch::Tensor aU_l_unfolded = maybe_unfold_patches( aU_l, effective_lA );

                torch::Tensor P_new = Ppos * aL_l_unfolded + Pneg * aU_l_unfolded;
                new_lA = BoundA( patches->create_similar( P_new ) );

                ensure_4d( bL_l );
                ensure_4d( bU_l );
                torch::Tensor bL_unfolded = inplace_unfold(
                    bL_l, { P.size( -2 ), P.size( -1 ) }, patches->stride, patches->padding,
                    patches->inserted_zeros, patches->output_padding );
                torch::Tensor bL_ready = bL_unfolded.permute( { 0, 1, 2, 3, 4, 5 } ).unsqueeze( 0 );
                torch::Tensor bU_unfolded = inplace_unfold(
                    bU_l, { P.size( -2 ), P.size( -1 ) }, patches->stride, patches->padding,
                    patches->inserted_zeros, patches->output_padding );
                torch::Tensor bU_ready = bU_unfolded.permute( { 0, 1, 2, 3, 4, 5 } ).unsqueeze( 0 );

                torch::Tensor term1 = ( Ppos * bL_ready ).sum( { -3, -2, -1 } );
                torch::Tensor term2 = ( Pneg * bU_ready ).sum( { -3, -2, -1 } );
                torch::Tensor total_bias = term1 + term2;
                total_bias = total_bias.permute( { 1, 0, 2, 3 } );
                lbias = lbias.defined() ? ( lbias + total_bias ) : total_bias;
            }
        }
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

        if ( effective_uA.isTensor() )
        {
            torch::Tensor uA = effective_uA.asTensor();
            if ( uA.dim() == 2 )
                uA = uA.unsqueeze( 1 );
            auto Apos = torch::clamp_min( uA, 0 );
            auto Aneg = torch::clamp_max( uA, 0 );

            auto A_u = Apos * broadcast_like( aU_u, uA ) + Aneg * broadcast_like( aL_u, uA );
            auto b_u = Apos * broadcast_like( bU_u, uA ) + Aneg * broadcast_like( bL_u, uA );

            new_uA = BoundA( A_u );
            auto add_ubias = reduce_bias_like_A( b_u, uA );
            ubias = ubias.defined() ? ( ubias + add_ubias ) : add_ubias;
        }
        else
        {
            auto patches = effective_uA.asPatches();
            bool is_sparse = patches->unstable_idx.has_value();

            auto ensure_4d = [&patches]( torch::Tensor &t )
            {
                if ( !t.defined() )
                    return;
                auto &shape = patches->input_shape;
                if ( shape.size() < 4 )
                    return;
                int64_t C = shape[1], H = shape[2], W = shape[3];
                if ( t.dim() == 1 && t.size( 0 ) == C * H * W )
                    t = t.view( { 1, C, H, W } );
                else if ( t.dim() == 2 && t.size( 1 ) == C * H * W )
                    t = t.view( { t.size( 0 ), C, H, W } );
                else if ( t.dim() == 3 )
                    t = t.unsqueeze( 0 );
            };

            auto unfold_and_select_sparse = [&patches, lanes]( const torch::Tensor &t,
                                                               const torch::Tensor &P,
                                                               bool perSpec = false )
            {
                torch::Tensor unfolded = inplace_unfold(
                    t, { P.size( -2 ), P.size( -1 ) }, patches->stride, patches->padding,
                    patches->inserted_zeros, patches->output_padding );
                auto &idx = patches->unstable_idx.value();
                if ( !perSpec )
                {
                    // One image per lane shared by every spec, selected at each spec's location
                    return unfolded.permute( { 1, 2, 0, 3, 4, 5 } ).index( { idx[1], idx[2] } );
                }
                // One image per spec and lane, each spec reads its own image at its own location
                const int64_t specs = unfolded.size( 0 ) / lanes;
                auto rows = unfolded.reshape( { specs, lanes, unfolded.size( 1 ),
                                                unfolded.size( 2 ), unfolded.size( 3 ),
                                                unfolded.size( 4 ), unfolded.size( 5 ) } );
                auto arange_idx = torch::arange( specs, idx[1].options() );
                return rows.index( { arange_idx, torch::indexing::Slice(), idx[1], idx[2] } );
            };

            torch::Tensor P = patches->patches;
            torch::Tensor Ppos = torch::clamp_min( P, 0 );
            torch::Tensor Pneg = torch::clamp_max( P, 0 );

            if ( is_sparse )
            {
                ensure_4d( aU_u );
                ensure_4d( aL_u );
                torch::Tensor aU_sel = unfold_and_select_sparse( aU_u, P, perSpecSlopes );
                torch::Tensor aL_sel = unfold_and_select_sparse( aL_u, P, perSpecSlopes );

                torch::Tensor P_new = Ppos * aU_sel + Pneg * aL_sel;
                new_uA = BoundA( patches->create_similar( P_new ) );

                ensure_4d( bU_u );
                ensure_4d( bL_u );
                torch::Tensor bU_sel = unfold_and_select_sparse( bU_u, P );
                torch::Tensor bL_sel = unfold_and_select_sparse( bL_u, P );
                torch::Tensor term1 = ( Ppos * bU_sel ).sum( { -3, -2, -1 } );
                torch::Tensor term2 = ( Pneg * bL_sel ).sum( { -3, -2, -1 } );
                torch::Tensor total_bias = term1 + term2;
                ubias = ubias.defined() ? ( ubias + total_bias ) : total_bias;
            }
            else
            {
                torch::Tensor aU_u_unfolded = maybe_unfold_patches( aU_u, effective_uA );
                torch::Tensor aL_u_unfolded = maybe_unfold_patches( aL_u, effective_uA );

                torch::Tensor P_new = Ppos * aU_u_unfolded + Pneg * aL_u_unfolded;
                new_uA = BoundA( patches->create_similar( P_new ) );

                ensure_4d( bU_u );
                ensure_4d( bL_u );
                torch::Tensor bU_unfolded = inplace_unfold(
                    bU_u, { P.size( -2 ), P.size( -1 ) }, patches->stride, patches->padding,
                    patches->inserted_zeros, patches->output_padding );
                torch::Tensor bU_ready = bU_unfolded.permute( { 0, 1, 2, 3, 4, 5 } ).unsqueeze( 0 );
                torch::Tensor bL_unfolded = inplace_unfold(
                    bL_u, { P.size( -2 ), P.size( -1 ) }, patches->stride, patches->padding,
                    patches->inserted_zeros, patches->output_padding );
                torch::Tensor bL_ready = bL_unfolded.permute( { 0, 1, 2, 3, 4, 5 } ).unsqueeze( 0 );

                torch::Tensor term1 = ( Ppos * bU_ready ).sum( { -3, -2, -1 } );
                torch::Tensor term2 = ( Pneg * bL_ready ).sum( { -3, -2, -1 } );
                torch::Tensor total_bias = term1 + term2;
                total_bias = total_bias.permute( { 1, 0, 2, 3 } );
                ubias = ubias.defined() ? ( ubias + total_bias ) : total_bias;
            }
        }
    }

    if ( outputA_matrices.size() == 0 )
    {
        outputA_matrices.append( Pair<BoundA, BoundA>( new_lA, new_uA ) );
    }
}

torch::Tensor BoundedReLUNode::maybe_unfold_patches( const torch::Tensor &d_tensor,
                                                     const BoundA &last_A )
{
    if ( !d_tensor.defined() || !last_A.isPatches() )
    {
        return d_tensor;
    }
    auto patches = last_A.asPatches();

    if ( d_tensor.dim() <= 2 && patches->input_shape.size() >= 4 )
    {
        int64_t C = patches->input_shape[1];
        int64_t H = patches->input_shape[2];
        int64_t W = patches->input_shape[3];
        int64_t batch = patches->input_shape[0];
        torch::Tensor d_4d;
        if ( d_tensor.dim() == 1 && d_tensor.size( 0 ) == C * H * W )
        {
            d_4d = d_tensor.view( { 1, C, H, W } );
        }
        else if ( d_tensor.dim() == 1 && d_tensor.size( 0 ) == batch * C * H * W )
        {
            d_4d = d_tensor.view( { batch, C, H, W } );
        }
        else if ( d_tensor.dim() == 2 && d_tensor.size( 1 ) == C * H * W )
        {
            d_4d = d_tensor.view( { d_tensor.size( 0 ), C, H, W } );
        }
        else
        {
            d_4d = d_tensor.view( { 1, C, H, W } );
        }
        return maybe_unfold_patches( d_4d, last_A );
    }

    if ( d_tensor.dim() == 3 )
    {
        return maybe_unfold_patches( d_tensor.unsqueeze( 0 ), last_A );
    }

    torch::Tensor d_unfolded = inplace_unfold(
        d_tensor, { patches->patches.size( -2 ), patches->patches.size( -1 ) }, patches->stride,
        patches->padding, patches->inserted_zeros, patches->output_padding );

    return d_unfolded.permute( { 0, 1, 2, 3, 4, 5 } ).unsqueeze( 0 );
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
