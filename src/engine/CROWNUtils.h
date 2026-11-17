/*********************                                                        */
/*! \file CROWNUtils.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __CROWNUtils_h__
#define __CROWNUtils_h__

#include "BoundResult.h"
#include "LunaError.h"
#include "Patches.h"
#include <torch/torch.h>

namespace NLR
{

inline torch::Tensor broadcastLike( const torch::Tensor &v, const torch::Tensor &A )
/*
 * v: (lanes, F) for CROWN or (spec, lanes, F) for alpha
 * A: (spec, lanes or 1, *payload)
 *
 * returns v reshaped to (spec or 1, lanes or 1, *payload)
 */
{
    if ( !v.defined() || !A.defined() )
    {
        throw LunaError( LunaError::INTERNAL_ERROR, "broadcastLike: tensors must be defined" );
    }
    if ( A.dim() < 3 || v.dim() < 2 || v.dim() > 3 )
    {
        throw LunaError( LunaError::UNEXPECTED_TENSOR_SHAPE,
                         "broadcastLike: A must be (spec, batch, ...) and the "
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
        throw LunaError( LunaError::UNEXPECTED_TENSOR_SHAPE,
                         ( "broadcastLike: slopes have " + std::to_string( v.size( -1 ) ) +
                           " features for A with " + std::to_string( payload_numel ) )
                             .c_str() );
    }
    return v.reshape( shape );
}

inline torch::Tensor reduceBiasLike( const torch::Tensor &term, const torch::Tensor &A )
{
    if ( !term.defined() || !A.defined() )
    {
        throw LunaError( LunaError::INTERNAL_ERROR, "reduceBiasLike: tensors must be defined" );
    }

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
}

inline torch::Tensor unfoldPatches( const torch::Tensor &d_tensor,
                                    const std::shared_ptr<Patches> patches )
{
    if ( !d_tensor.defined() )
    {
        throw LunaError( LunaError::INTERNAL_ERROR,
                         "Tried to unfold patches with undefined tensor" );
    }
    if ( patches == nullptr )
    {
        throw LunaError( LunaError::INTERNAL_ERROR, "Tried to unfold patches with null patches" );
    }

    torch::Tensor d_4d = patches->ensure_4d( d_tensor );

    torch::Tensor d_unfolded = inplace_unfold(
        d_4d, { patches->patches.size( -2 ), patches->patches.size( -1 ) }, patches->stride,
        patches->padding, patches->inserted_zeros, patches->output_padding );

    return d_unfolded.unsqueeze( 0 );
}

inline torch::Tensor unfoldAndSelectSparse( const std::shared_ptr<Patches> patches,
                                            const int64_t lanes, const torch::Tensor &t,
                                            const torch::Tensor &P, bool perSpec = false )
{
    torch::Tensor t_4d = patches->ensure_4d( t );
    torch::Tensor unfolded =
        inplace_unfold( t_4d, { P.size( -2 ), P.size( -1 ) }, patches->stride, patches->padding,
                        patches->inserted_zeros, patches->output_padding );
    auto &idx = patches->unstable_idx.value();
    if ( !perSpec )
    {
        // One image per lane shared by every spec, selected at each spec's location
        return unfolded.permute( { 1, 2, 0, 3, 4, 5 } ).index( { idx[1], idx[2] } );
    }
    // One image per spec and lane, each spec reads its own image at its own location
    const int64_t specs = unfolded.size( 0 ) / lanes;
    auto rows = unfolded.reshape( { specs, lanes, unfolded.size( 1 ), unfolded.size( 2 ),
                                    unfolded.size( 3 ), unfolded.size( 4 ), unfolded.size( 5 ) } );
    auto arange_idx = torch::arange( specs, idx[1].options() );
    return rows.index( { arange_idx, torch::indexing::Slice(), idx[1], idx[2] } );
};

inline torch::Tensor propagateTensor( const torch::Tensor &A, const torch::Tensor &aL,
                                      const torch::Tensor &aU, const torch::Tensor &bL,
                                      const torch::Tensor &bU, torch::Tensor &bias )
/*
 * Helper function to compute tensor case for propagate
 *
 * Returns new_A = A+ * aL + A- * aU
 * assigns bias  = reduce(A+ * bL + A- * bU)
 */
{
    torch::Tensor A_tensor = A;
    if ( A_tensor.dim() == 2 )
        A_tensor = A_tensor.unsqueeze( 1 );
    auto Apos = torch::clamp_min( A_tensor, 0 );
    auto Aneg = torch::clamp_max( A_tensor, 0 );

    auto newA = Apos * broadcastLike( aL, A_tensor ) + Aneg * broadcastLike( aU, A_tensor );
    auto new_bias = Apos * broadcastLike( bL, A_tensor ) + Aneg * broadcastLike( bU, A_tensor );

    auto add_bias = reduceBiasLike( new_bias, A_tensor );
    bias = bias.defined() ? ( bias + add_bias ) : add_bias;
    return newA;
}

inline std::shared_ptr<Patches>
propagateSparsePatches( const std::shared_ptr<Patches> patches, const int64_t lanes,
                        const bool perSpecSlopes, const torch::Tensor &aL, const torch::Tensor &aU,
                        const torch::Tensor &bL, const torch::Tensor &bU, torch::Tensor &bias )
/*
 * Helper function to compute sparse patches case for propagate
 *
 * Returns new_P = P+ * unfolded(aL) + P- * unfolded(aU)
 * assigns bias  = P+ * unfolded(bL) + P- * unfolded(bU)
 */
{
    torch::Tensor P = patches->patches;
    torch::Tensor Ppos = torch::clamp_min( P, 0 );
    torch::Tensor Pneg = torch::clamp_max( P, 0 );

    torch::Tensor aL_sel = unfoldAndSelectSparse( patches, lanes, aL, P, perSpecSlopes );
    torch::Tensor aU_sel = unfoldAndSelectSparse( patches, lanes, aU, P, perSpecSlopes );

    torch::Tensor P_new = Ppos * aL_sel + Pneg * aU_sel;

    torch::Tensor bL_sel = unfoldAndSelectSparse( patches, lanes, bL, P );
    torch::Tensor bU_sel = unfoldAndSelectSparse( patches, lanes, bU, P );
    torch::Tensor term1 = ( Ppos * bL_sel ).sum( { -3, -2, -1 } );
    torch::Tensor term2 = ( Pneg * bU_sel ).sum( { -3, -2, -1 } );
    torch::Tensor total_bias = term1 + term2;

    bias = bias.defined() ? ( bias + total_bias ) : total_bias;

    return patches->create_similar( P_new );
}

inline std::shared_ptr<Patches>
propagateDensePatches( const std::shared_ptr<Patches> patches, const torch::Tensor &aL,
                       const torch::Tensor &aU, const torch::Tensor &bL, const torch::Tensor &bU,
                       torch::Tensor &bias )
/*
 * Helper function to compute non-sparse patches case for propagate
 *
 * Returns new_P = P+ * unfolded(aL) + P- * unfolded(aU)
 * assigns bias  = P+ * unfolded(bL) + P- * unfolded(bU)
 */
{
    torch::Tensor P = patches->patches;
    torch::Tensor Ppos = torch::clamp_min( P, 0 );
    torch::Tensor Pneg = torch::clamp_max( P, 0 );
    torch::Tensor aL_unfolded = unfoldPatches( aL, patches );
    torch::Tensor aU_unfolded = unfoldPatches( aU, patches );

    torch::Tensor P_new = Ppos * aL_unfolded + Pneg * aU_unfolded;

    torch::Tensor bL_4d = patches->ensure_4d( bL );
    torch::Tensor bU_4d = patches->ensure_4d( bU );
    torch::Tensor bL_unfolded =
        inplace_unfold( bL_4d, { P.size( -2 ), P.size( -1 ) }, patches->stride, patches->padding,
                        patches->inserted_zeros, patches->output_padding );
    torch::Tensor bL_ready = bL_unfolded.permute( { 0, 1, 2, 3, 4, 5 } ).unsqueeze( 0 );
    torch::Tensor bU_unfolded =
        inplace_unfold( bU_4d, { P.size( -2 ), P.size( -1 ) }, patches->stride, patches->padding,
                        patches->inserted_zeros, patches->output_padding );
    torch::Tensor bU_ready = bU_unfolded.permute( { 0, 1, 2, 3, 4, 5 } ).unsqueeze( 0 );

    torch::Tensor term1 = ( Ppos * bL_ready ).sum( { -3, -2, -1 } );
    torch::Tensor term2 = ( Pneg * bU_ready ).sum( { -3, -2, -1 } );
    torch::Tensor total_bias = term1 + term2;
    total_bias = total_bias.permute( { 1, 0, 2, 3 } );

    bias = bias.defined() ? ( bias + total_bias ) : total_bias;

    return patches->create_similar( P_new );
}

inline BoundA propagate( const BoundA &A, const int64_t lanes, const bool perSpecSlopes,
                         const torch::Tensor &aL, const torch::Tensor &aU, const torch::Tensor &bL,
                         const torch::Tensor &bU, torch::Tensor &bias )
/*
 * A: last A bound
 * lanes: number of lanes, only used for sparse patches
 * perSpecSlopes: controls how sparse patches are unfolded for aL and aU
 * aL, aU: a terms
 * bL, bU: bias terms
 * bias: reference to output bias term
 *
 * For lower bound pass [aL, aU] = [a lower bound, a upper bound]
 * For upper bound pass [aL, aU] = [a upper bound, a lower bound]
 * Similar for bias terms.
 *
 * Returns new_A = A+ * aL + A- * aU
 * assigns bias  = reduce(A+ * bL + A- * bU)
 */
{
    if ( A.isTensor() )
    {
        return BoundA( propagateTensor( A.asTensor(), aL, aU, bL, bU, bias ) );
    }

    auto patches = A.asPatches();

    if ( patches->is_sparse() )
    {
        return BoundA(
            propagateSparsePatches( patches, lanes, perSpecSlopes, aL, aU, bL, bU, bias ) );
    }

    return BoundA( propagateDensePatches( patches, aL, aU, bL, bU, bias ) );
}

} // namespace NLR

#endif // __CROWNUtils_h__
