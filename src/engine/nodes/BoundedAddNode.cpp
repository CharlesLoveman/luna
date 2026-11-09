/*********************                                                        */
/*! \file BoundedAddNode.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "BoundedAddNode.h"

namespace NLR
{

static inline torch::Tensor firstDefinedBoundTensor( const BoundedTensor<torch::Tensor> &b )
{
    if ( b.lower().defined() )
        return b.lower();
    if ( b.upper().defined() )
        return b.upper();
    return torch::Tensor();
}

// The constant is the same in every lane, so one lane of it is what a spec row multiplies
static inline torch::Tensor oneLaneOf( const torch::Tensor &constant, const torch::Tensor &x )
{
    return x.defined() ? constant.narrow( 0, 0, 1 ).flatten() : constant.flatten();
}

static inline std::vector<int64_t> boundShapeVec( const BoundedTensor<torch::Tensor> &b )
{
    torch::Tensor t = firstDefinedBoundTensor( b );
    if ( !t.defined() )
        return {};
    return t.sizes().vec();
}

static inline bool shapesEqual( const std::vector<int64_t> &a, const std::vector<int64_t> &b )
{
    if ( a.size() != b.size() )
        return false;
    for ( size_t i = 0; i < a.size(); ++i )
    {
        if ( a[i] != b[i] )
            return false;
    }
    return true;
}

BoundedAddNode::BoundedAddNode()
{
    _input_size = 0;
    _output_size = 0;
    _nodeIndex = 0;
    _nodeName = "add";
}

torch::Tensor BoundedAddNode::forward( const torch::Tensor &input )
{
    if ( _constantValue.defined() )
    {
        return input + _constantValue;
    }
    return input;
}

void BoundedAddNode::moveToDevice( const torch::Device &device )
{
    BoundedTorchNode::moveToDevice( device );
    if ( _constantValue.defined() )
    {
        _constantValue = _constantValue.to( device );
    }
}

torch::Tensor BoundedAddNode::forward( const std::vector<torch::Tensor> &inputs )
{
    if ( inputs.size() == 1 )
    {
        return forward( inputs[0] );
    }
    else if ( inputs.size() == 2 )
    {
        return inputs[0] + inputs[1];
    }
    else
    {
        throw std::runtime_error( "BoundedAddNode::forward expects 1 or 2 inputs, got " +
                                  std::to_string( inputs.size() ) );
    }
}

torch::Tensor BoundedAddNode::broadcast_backward( const torch::Tensor &last_A,
                                                  const BoundedTensor<torch::Tensor> &input ) const
{
    if ( !last_A.defined() )
        return last_A;

    torch::Tensor x = firstDefinedBoundTensor( input );
    if ( !x.defined() )
    {
        return last_A;
    }

    std::vector<int64_t> target_shape = x.sizes().vec();

    // Bounds stored WITHOUT batch dim; A may have [spec, batch, features]
    if ( last_A.dim() >= 3 )
    {
        int64_t A_batch = last_A.size( 1 );
        if ( (int64_t)target_shape.size() == 0 )
        {
            target_shape = { A_batch };
        }
        if ( (int64_t)target_shape.size() < 2 )
        {
            target_shape.insert( target_shape.begin(), A_batch );
        }
        else if ( target_shape.size() >= 1 && target_shape[0] != A_batch )
        {
            target_shape.insert( target_shape.begin(), A_batch );
        }
    }
    else if ( last_A.dim() == 2 )
    {
        int64_t A_batch = 1;
        if ( (int64_t)target_shape.size() == 0 )
        {
            target_shape = { A_batch };
        }
        else if ( target_shape.size() >= 1 && target_shape[0] != A_batch )
        {
            target_shape.insert( target_shape.begin(), A_batch );
        }
    }

    torch::Tensor A = last_A;
    if ( A.dim() == 1 )
    {
        return A;
    }

    std::vector<int64_t> operand_shape = target_shape;
    const int64_t op_rank = (int64_t)operand_shape.size();

    int64_t spec_offset = -1;
    if ( A.dim() == op_rank + 1 && A.dim() >= 3 )
    {
        spec_offset = 1;
    }
    else if ( A.dim() == op_rank && A.dim() >= 2 )
    {
        spec_offset = 0;
    }
    else if ( A.dim() == 2 )
    {
        A = A.unsqueeze( 1 );
        spec_offset = 1;
    }
    else if ( A.dim() >= 3 )
    {
        spec_offset = 1;
    }
    else
    {
        return A;
    }

    if ( spec_offset == 1 && A.dim() < 3 )
        return A;

    const int64_t payload_start = ( spec_offset == 1 ) ? 2 : 1;

    // Broadcasting only expands an operand, so equal element counts mean A already matches it
    int64_t op_payload_numel = 1;
    for ( int64_t i = 1; i < op_rank; ++i )
        op_payload_numel *= operand_shape[(size_t)i];
    int64_t A_payload_numel = 1;
    for ( int64_t i = payload_start; i < A.dim(); ++i )
        A_payload_numel *= A.size( i );
    if ( op_payload_numel == A_payload_numel )
    {
        return A;
    }

    const int64_t A_payload_dims = A.dim() - payload_start;
    const int64_t op_payload_dims = op_rank - 1;
    if ( A_payload_dims > op_payload_dims )
    {
        const int64_t num_extra = A_payload_dims - op_payload_dims;
        std::vector<int64_t> sum_dims;
        sum_dims.reserve( (size_t)num_extra );
        for ( int64_t i = 0; i < num_extra; ++i )
        {
            sum_dims.push_back( payload_start + i );
        }
        if ( !sum_dims.empty() )
        {
            A = A.sum( sum_dims );
        }
    }

    // Reduce broadcasted dims where operand dim == 1 but A dim != 1
    std::vector<int64_t> keep_dims;
    if ( op_rank >= 2 )
    {
        for ( int64_t i = 1; i < op_rank; ++i )
        {
            int64_t op_dim = operand_shape[(size_t)i];
            int64_t a_dim_index = ( spec_offset == 1 ) ? ( i + 1 ) : i;
            if ( a_dim_index < 0 || a_dim_index >= A.dim() )
                continue;
            int64_t A_dim = A.size( a_dim_index );
            if ( op_dim == 1 && A_dim != 1 )
            {
                keep_dims.push_back( a_dim_index );
            }
        }
    }
    if ( !keep_dims.empty() )
    {
        A = A.sum( keep_dims, /*keepdim=*/true );
    }

    return A;
}

void BoundedAddNode::boundBackward( const BoundA &last_lA, const BoundA &last_uA,
                                    const Vector<BoundedTensor<torch::Tensor>> &inputBounds,
                                    Vector<Pair<BoundA, BoundA>> &outputA_matrices,
                                    torch::Tensor &lbias, torch::Tensor &ubias )
{

    outputA_matrices.clear();

    auto bound_one_side = [&]( const BoundA &last_A,
                               const BoundedTensor<torch::Tensor> &in ) -> BoundA
    {
        if ( !last_A.defined() )
            return BoundA();
        if ( last_A.isTensor() )
        {
            torch::Tensor A = broadcast_backward( last_A.asTensor(), in );
            return BoundA( A );
        }
        else
        {
            return last_A;
        }
    };

    if ( inputBounds.size() == 1 )
    {
        BoundA lA_x = bound_one_side( last_lA, inputBounds[0] );
        BoundA uA_x = bound_one_side( last_uA, inputBounds[0] );
        outputA_matrices.append( Pair<BoundA, BoundA>( lA_x, uA_x ) );

        if ( _constantValue.defined() )
        {
            if ( last_lA.isPatches() || last_uA.isPatches() )
            {
                throw std::runtime_error( "BoundedAddNode: Patches mode with constant bias not "
                                          "implemented (requires conversion)" );
            }

            torch::Tensor lA_tensor = last_lA.asTensor();
            torch::Tensor uA_tensor = last_uA.asTensor();

            if ( lA_tensor.defined() )
            {
                torch::Tensor x = firstDefinedBoundTensor( inputBounds[0] );
                torch::Tensor c_full =
                    x.defined() ? _constantValue.to( x.device() ).to( x.dtype() ) : _constantValue;
                if ( x.defined() )
                    c_full = c_full.expand_as( x );
                torch::Tensor constant_contrib;
                if ( lA_tensor.dim() == c_full.dim() )
                {
                    std::vector<int64_t> sum_dims;
                    for ( int64_t d = 1; d < lA_tensor.dim(); ++d )
                        sum_dims.push_back( d );
                    constant_contrib = ( lA_tensor * c_full ).sum( sum_dims );
                }
                else if ( lA_tensor.dim() == c_full.dim() + 1 )
                {
                    std::vector<int64_t> sum_dims;
                    for ( int64_t d = 2; d < lA_tensor.dim(); ++d )
                        sum_dims.push_back( d );
                    constant_contrib = ( lA_tensor * c_full.unsqueeze( 1 ) ).sum( sum_dims );
                }
                else if ( lA_tensor.dim() == 2 )
                {
                    torch::Tensor constant = oneLaneOf( c_full, x );
                    constant_contrib = torch::matmul( lA_tensor, constant );
                }
                else if ( lA_tensor.dim() == 3 )
                {
                    torch::Tensor constant = oneLaneOf( c_full, x );
                    constant_contrib =
                        torch::matmul( lA_tensor, constant.unsqueeze( -1 ) ).squeeze( -1 );
                }
                else
                {
                    throw std::runtime_error( "BoundedAddNode::boundBackward: unsupported last_lA "
                                              "shape for constant bias" );
                }

                if ( lbias.defined() )
                {
                    lbias = lbias + constant_contrib;
                }
                else
                {
                    lbias = constant_contrib;
                }
            }

            if ( uA_tensor.defined() )
            {
                torch::Tensor x = firstDefinedBoundTensor( inputBounds[0] );
                torch::Tensor c_full =
                    x.defined() ? _constantValue.to( x.device() ).to( x.dtype() ) : _constantValue;
                if ( x.defined() )
                    c_full = c_full.expand_as( x );
                torch::Tensor constant_contrib;
                if ( uA_tensor.dim() == c_full.dim() )
                {
                    std::vector<int64_t> sum_dims;
                    for ( int64_t d = 1; d < uA_tensor.dim(); ++d )
                        sum_dims.push_back( d );
                    constant_contrib = ( uA_tensor * c_full ).sum( sum_dims );
                }
                else if ( uA_tensor.dim() == c_full.dim() + 1 )
                {
                    std::vector<int64_t> sum_dims;
                    for ( int64_t d = 2; d < uA_tensor.dim(); ++d )
                        sum_dims.push_back( d );
                    constant_contrib = ( uA_tensor * c_full.unsqueeze( 1 ) ).sum( sum_dims );
                }
                else if ( uA_tensor.dim() == 2 )
                {
                    torch::Tensor constant = oneLaneOf( c_full, x );
                    constant_contrib = torch::matmul( uA_tensor, constant );
                }
                else if ( uA_tensor.dim() == 3 )
                {
                    torch::Tensor constant = oneLaneOf( c_full, x );
                    constant_contrib =
                        torch::matmul( uA_tensor, constant.unsqueeze( -1 ) ).squeeze( -1 );
                }
                else
                {
                    throw std::runtime_error( "BoundedAddNode::boundBackward: unsupported last_uA "
                                              "shape for constant bias" );
                }

                if ( ubias.defined() )
                {
                    ubias = ubias + constant_contrib;
                }
                else
                {
                    ubias = constant_contrib;
                }
            }
        }
    }
    else if ( inputBounds.size() == 2 )
    {
        const bool anyPatches = ( last_lA.defined() && last_lA.isPatches() ) ||
                                ( last_uA.defined() && last_uA.isPatches() );
        if ( anyPatches )
        {
            auto sx = boundShapeVec( inputBounds[0] );
            auto sy = boundShapeVec( inputBounds[1] );
            if ( !sx.empty() && !sy.empty() && !shapesEqual( sx, sy ) )
            {
                throw std::runtime_error( "BoundedAddNode: Patches mode Add requires identical "
                                          "input shapes (broadcasting not supported)" );
            }
        }

        BoundA lA_x = bound_one_side( last_lA, inputBounds[0] );
        BoundA uA_x = bound_one_side( last_uA, inputBounds[0] );
        BoundA lA_y = bound_one_side( last_lA, inputBounds[1] );
        BoundA uA_y = bound_one_side( last_uA, inputBounds[1] );
        outputA_matrices.append( Pair<BoundA, BoundA>( lA_x, uA_x ) );
        outputA_matrices.append( Pair<BoundA, BoundA>( lA_y, uA_y ) );
    }
    else
    {
        throw std::runtime_error( "BoundedAddNode::boundBackward expects 1 or 2 input bounds" );
    }
}

BoundedTensor<torch::Tensor> BoundedAddNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>> &inputBounds )
{

    if ( inputBounds.size() == 1 )
    {
        const auto &x = inputBounds[0];
        if ( _constantValue.defined() )
        {
            torch::Tensor lower = x.lower() + _constantValue;
            torch::Tensor upper = x.upper() + _constantValue;
            return BoundedTensor<torch::Tensor>( lower, upper );
        }
        else
        {
            return x;
        }
    }
    else if ( inputBounds.size() == 2 )
    {
        const auto &x = inputBounds[0];
        const auto &y = inputBounds[1];

        torch::Tensor lower = x.lower() + y.lower();
        torch::Tensor upper = x.upper() + y.upper();

        return BoundedTensor<torch::Tensor>( lower, upper );
    }
    else
    {
        throw std::runtime_error(
            "BoundedAddNode::computeIntervalBoundPropagation expects 1 or 2 input bounds" );
    }
}

void BoundedAddNode::setInputSize( unsigned size )
{
    _input_size = size;
}

void BoundedAddNode::setOutputSize( unsigned size )
{
    _output_size = size;
}

} // namespace NLR
