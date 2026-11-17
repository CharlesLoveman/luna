/*********************                                                        */
/*! \file Patches.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "Patches.h"
#include "LunaError.h"
#include <torch/nn/functional.h>
#include <stdexcept>
#include <algorithm>

namespace NLR
{

void Patches::simplify()
{
    std::vector<int64_t> full_stride = stride;
    if ( full_stride.size() == 1 )
    {
        full_stride = { full_stride[0], full_stride[0] };
    }

    if ( inserted_zeros > 0 && ( inserted_zeros + 1 ) == full_stride[0] &&
         full_stride[0] == full_stride[1] && ( patches.size( -1 ) % full_stride[1] == 0 ) &&
         ( patches.size( -2 ) % full_stride[0] == 0 ) )
    {

        std::vector<int64_t> full_stride_4 = { full_stride[1], full_stride[1], full_stride[0],
                                               full_stride[0] };

        std::vector<int64_t> padding_vec = unify_shape( padding );

        std::vector<int64_t> consumed_padding = {
            padding_vec[0], padding_vec[1] - ( full_stride[1] - 1 ), padding_vec[2],
            padding_vec[3] - ( full_stride[0] - 1 ) };

        std::vector<int64_t> output_padding_vec = unify_shape( output_padding );
        std::vector<int64_t> tentative_padding;
        bool all_non_negative = true;

        for ( size_t i = 0; i < 4; ++i )
        {
            int64_t val = consumed_padding[i] / full_stride_4[i] - output_padding_vec[i];
            if ( val < 0 )
                all_non_negative = false;
            tentative_padding.push_back( val );
        }

        if ( all_non_negative )
        {
            std::pair<int64_t, int64_t> remove_zero_start_idx = { padding_vec[2] % full_stride[0],
                                                                  padding_vec[0] % full_stride[1] };

            padding = tentative_padding;
            patches = remove_zeros( patches, inserted_zeros, remove_zero_start_idx );
            stride = { 1, 1 };
            inserted_zeros = 0;
            output_padding = { 0, 0, 0, 0 };
        }
    }
}

std::shared_ptr<Patches>
Patches::create_similar( std::optional<torch::Tensor> new_patches,
                         std::optional<std::vector<int64_t>> new_stride,
                         std::optional<std::vector<int64_t>> new_padding,
                         std::optional<std::vector<int64_t>> new_output_padding,
                         std::optional<int64_t> new_inserted_zeros, std::optional<int> new_identity,
                         std::optional<std::vector<int64_t>> new_input_shape ) const
{
    auto p = new_patches.has_value() ? new_patches.value() : patches;
    auto s = new_stride.has_value() ? new_stride.value() : stride;
    auto pad = new_padding.has_value() ? new_padding.value() : padding;
    auto out_pad = new_output_padding.has_value() ? new_output_padding.value() : output_padding;
    auto zeros = new_inserted_zeros.has_value() ? new_inserted_zeros.value() : inserted_zeros;
    auto id = new_identity.has_value() ? new_identity.value() : identity;
    auto in_shape = new_input_shape.has_value() ? new_input_shape.value() : input_shape;

    return std::make_shared<Patches>( p, s, pad, out_pad, zeros, unstable_idx, output_shape,
                                      in_shape, id );
}

std::shared_ptr<Patches> Patches::add( const std::shared_ptr<Patches> &other ) const
{
    if ( stride != other->stride )
    {
        throw std::runtime_error( "Patches addition requires same stride" );
    }

    torch::Tensor A1 = patches;
    torch::Tensor A2 = other->patches;

    std::vector<int64_t> sp = unify_shape( padding );
    std::vector<int64_t> op = unify_shape( other->padding );

    std::vector<int64_t> diff( 4 );
    int64_t diff_sum = 0;
    bool sp_ge_op = true;
    bool sp_le_op = true;

    std::vector<int64_t> new_pad( 4 );

    for ( int i = 0; i < 4; ++i )
    {
        diff[i] = sp[i] - op[i];
        diff_sum += std::abs( diff[i] );
        if ( diff[i] < 0 )
            sp_ge_op = false;
        if ( diff[i] > 0 )
            sp_le_op = false;
        new_pad[i] = std::max( sp[i], op[i] );
    }

    if ( diff_sum > 0 )
    {
        if ( sp_ge_op )
        {
            A2 = torch::nn::functional::pad( A2, torch::nn::functional::PadFuncOptions( diff ) );
        }
        else if ( sp_le_op )
        {
            std::vector<int64_t> neg_diff( 4 );
            for ( int i = 0; i < 4; ++i )
                neg_diff[i] = -diff[i];
            A1 =
                torch::nn::functional::pad( A1, torch::nn::functional::PadFuncOptions( neg_diff ) );
        }
        else
        {
            throw std::runtime_error( "Unsupported padding size difference in Patches::add" );
        }
    }

    torch::Tensor ret = A1 + A2;

    return std::make_shared<Patches>( ret, stride, new_pad, output_padding, inserted_zeros,
                                      unstable_idx, output_shape, input_shape, 0 );
}

torch::Tensor Patches::to_matrix( const std::vector<int64_t> &in_shape ) const
{
    if ( is_shape_used( output_padding ) )
    {
        throw std::runtime_error( "to_matrix with output_padding not supported" );
    }
    return patches_to_matrix( patches, in_shape, stride, padding, output_shape, unstable_idx,
                              inserted_zeros );
}

torch::Tensor Patches::matmul( const torch::Tensor &input, bool patch_abs,
                               const std::vector<int64_t> &expand_shape ) const
{
    // Mirrors auto_LiRPA Patches.matmul -- uses inplace_unfold + einsum

    torch::Tensor p = patches;
    if ( patch_abs )
    {
        p = p.abs();
    }

    torch::Tensor inp = input;
    if ( !expand_shape.empty() )
    {
        inp = inp.expand( torch::IntArrayRef( expand_shape ) );
    }

    std::vector<int64_t> kernel_size = { p.size( -2 ), p.size( -1 ) };

    torch::Tensor unfold_input =
        inplace_unfold( inp, kernel_size, stride, padding, inserted_zeros, output_padding );

    if ( unstable_idx.has_value() )
    {
        int64_t out_c = output_shape[1];
        torch::Tensor uf = unfold_input.unsqueeze( 0 ).expand( { out_c, -1, -1, -1, -1, -1, -1 } );

        const auto &idx = unstable_idx.value();
        using namespace torch::indexing;
        uf = uf.index( { idx[0], Slice(), idx[1], idx[2] } );

        return torch::einsum( "sbchw,sbchw->bs", { uf, p } );
    }
    else
    {
        return torch::einsum( "bijchw,sbijchw->bsij", { unfold_input, p } );
    }
}

torch::Tensor insert_zeros( const torch::Tensor &image, int64_t s )
{
    if ( s <= 0 )
        return image;

    int64_t N = image.size( 0 );
    int64_t C = image.size( 1 );
    int64_t H = image.size( 2 );
    int64_t W = image.size( 3 );

    int64_t new_H = H * ( s + 1 ) - s;
    int64_t new_W = W * ( s + 1 ) - s;

    torch::Tensor matrix = torch::zeros( { N, C, new_H, new_W }, image.options() );

    using namespace torch::indexing;
    matrix.index_put_( { Slice(), Slice(), Slice( None, None, s + 1 ), Slice( None, None, s + 1 ) },
                       image );

    return matrix;
}

torch::Tensor remove_zeros( const torch::Tensor &image, int64_t s,
                            const std::pair<int64_t, int64_t> &start_idx )
{
    if ( s <= 0 )
        return image;

    int64_t start_h = start_idx.first;
    int64_t start_w = start_idx.second;

    using namespace torch::indexing;
    return image.index(
        { Slice(), Slice(), Slice( start_h, None, s + 1 ), Slice( start_w, None, s + 1 ) } );
}

bool is_shape_used( const std::vector<int64_t> &shape, int expected )
{
    int64_t sum = 0;
    for ( auto s : shape )
        sum += s;
    return sum != expected;
}

torch::Tensor inplace_unfold( const torch::Tensor &image, const std::vector<int64_t> &kernel_size,
                              const std::vector<int64_t> &stride,
                              const std::vector<int64_t> &padding, int64_t inserted_zeros,
                              const std::vector<int64_t> &output_padding )
{

    torch::Tensor img = image;
    std::vector<int64_t> k_size = kernel_size;
    if ( k_size.size() == 1 )
        k_size = { k_size[0], k_size[0] };

    std::vector<int64_t> pad = unify_shape( padding );

    std::vector<int64_t> out_pad = unify_shape( output_padding );
    std::vector<int64_t> str = stride;
    if ( str.size() == 1 )
        str = { str[0], str[0] };

    if ( inserted_zeros > 0 )
    {
        img = insert_zeros( img, inserted_zeros );
    }

    if ( pad[0] != 0 || pad[1] != 0 || pad[2] != 0 || pad[3] != 0 )
    {
        img = torch::nn::functional::pad( img, torch::nn::functional::PadFuncOptions( pad ) );
    }

    torch::Tensor unfolded = img.unfold( 2, k_size[0], str[0] );
    unfolded = unfolded.unfold( 3, k_size[1], str[1] );

    unfolded = unfolded.permute( { 0, 2, 3, 1, 4, 5 } );

    if ( is_shape_used( output_padding ) )
    {
        using namespace torch::indexing;

        int64_t dim_h_start = out_pad[2] > 0 ? out_pad[2] : 0;
        int64_t dim_h_end = out_pad[3] > 0 ? -out_pad[3] : std::numeric_limits<int64_t>::max();

        int64_t dim_w_start = out_pad[0] > 0 ? out_pad[0] : 0;
        int64_t dim_w_end = out_pad[1] > 0 ? -out_pad[1] : std::numeric_limits<int64_t>::max();

        auto get_slice = []( int64_t start, int64_t end )
        {
            if ( end == std::numeric_limits<int64_t>::max() )
                return Slice( start, None );
            return Slice( start, end );
        };

        unfolded =
            unfolded.index( { Slice(), get_slice( dim_h_start, dim_h_end ),
                              get_slice( dim_w_start, dim_w_end ), Slice(), Slice(), Slice() } );
    }

    return unfolded;
}

void compute_patches_stride_padding(
    const std::vector<int64_t> &input_shape, const std::vector<int64_t> &patches_padding,
    const std::vector<int64_t> &patches_stride, const std::vector<int64_t> &op_padding,
    const std::vector<int64_t> &op_stride, int64_t inserted_zeros,
    const std::vector<int64_t> &output_padding, std::vector<int64_t> &new_padding,
    std::vector<int64_t> &new_stride, std::vector<int64_t> &new_output_padding )
{
    std::vector<int64_t> p_pad = unify_shape( patches_padding );
    std::vector<int64_t> p_str = ( patches_stride.size() == 1 )
                                     ? std::vector<int64_t>{ patches_stride[0], patches_stride[0] }
                                     : patches_stride;

    auto expand_stride = []( const std::vector<int64_t> &s )
    {
        if ( s.size() == 2 )
            return std::vector<int64_t>{ s[1], s[1], s[0], s[0] };
        if ( s.size() == 1 )
            return std::vector<int64_t>{ s[0], s[0], s[0], s[0] };
        return s;
    };

    std::vector<int64_t> full_p_stride = expand_stride( p_str );

    std::vector<int64_t> o_pad = unify_shape( op_padding );
    std::vector<int64_t> o_str_in =
        ( op_stride.size() == 1 ) ? std::vector<int64_t>{ op_stride[0], op_stride[0] } : op_stride;
    std::vector<int64_t> full_o_stride = expand_stride( o_str_in );

    new_padding.clear();
    new_stride.clear();

    for ( size_t i = 0; i < 4; ++i )
    {
        new_padding.push_back( p_pad[i] * full_o_stride[i] + o_pad[i] * ( inserted_zeros + 1 ) );
        new_stride.push_back( full_p_stride[i] * full_o_stride[i] );
    }

    std::vector<int64_t> out_pad = unify_shape( output_padding );
    new_output_padding.resize( 4 );

    int64_t H = input_shape[2];
    int64_t W = input_shape[3];

    new_output_padding[0] = out_pad[0];
    new_output_padding[1] = out_pad[1] + inserted_zeros * W % full_o_stride[1];
    new_output_padding[2] = out_pad[2];
    new_output_padding[3] = out_pad[3] + inserted_zeros * H % full_o_stride[2];

    if ( new_stride[0] == new_stride[1] && new_stride[2] == new_stride[3] &&
         new_stride[0] == new_stride[2] )
    {
        new_stride = { new_stride[2], new_stride[0] };
    }
    else
    {
        if ( new_stride[0] == new_stride[1] && new_stride[2] == new_stride[3] )
        {
            new_stride = { new_stride[2], new_stride[0] };
        }
    }
}

torch::Tensor create_valid_mask( const std::vector<int64_t> &output_shape,
                                 const torch::Device &device, torch::Dtype dtype,
                                 const std::vector<int64_t> &kernel_size,
                                 const std::vector<int64_t> &stride, int64_t inserted_zeros,
                                 const std::vector<int64_t> &padding,
                                 const std::vector<int64_t> &output_padding,
                                 const std::optional<std::vector<torch::Tensor>> &unstable_idx )
{
    std::vector<int64_t> shape_ones;
    for ( size_t i = 1; i < output_shape.size(); ++i )
        shape_ones.push_back( 1 );

    torch::Tensor one_d =
        torch::ones( shape_ones, torch::TensorOptions().device( device ).dtype( dtype ) );

    std::vector<int64_t> expand_shape;
    for ( size_t i = 1; i < output_shape.size(); ++i )
        expand_shape.push_back( output_shape[i] );
    one_d = one_d.expand( expand_shape );

    one_d = one_d.unsqueeze( 0 );

    torch::Tensor one_d_unfolded =
        inplace_unfold( one_d, kernel_size, stride, padding, inserted_zeros, output_padding );

    if ( unstable_idx.has_value() )
    {
        torch::Tensor ans = one_d_unfolded.permute( { 1, 2, 0, 3, 4, 5 } );
        const auto &indices = unstable_idx.value();
        if ( indices.size() >= 3 )
        {
            using namespace torch::indexing;
            ans = ans.index( { indices[1], indices[2] } );
        }
        return ans;
    }
    else
    {
        return one_d_unfolded.unsqueeze( 0 );
    }
}

torch::Tensor Patches::patches_to_matrix(
    const torch::Tensor &pieces, const std::vector<int64_t> &input_shape,
    const std::vector<int64_t> &stride_in, const std::vector<int64_t> &padding_in,
    const std::vector<int64_t> &output_shape,
    const std::optional<std::vector<torch::Tensor>> &unstable_idx, int64_t inserted_zeros )
{
    // as_strided for zero-copy view creation into sliding windows

    std::vector<int64_t> padding = unify_shape( padding_in );
    std::vector<int64_t> str = stride_in;
    if ( str.size() == 1 )
        str = { str[0], str[0] };

    torch::Tensor p = pieces;
    if ( p.dim() == 9 )
    {
        if ( p.size( 1 ) == 1 && p.size( 5 ) == 1 )
        {
            p = p.reshape( { p.size( 0 ), p.size( 2 ), p.size( 3 ), p.size( 4 ), p.size( 6 ),
                             p.size( 7 ), p.size( 8 ) } );
        }
    }

    int64_t output_channel, batch_size, output_x, output_y;
    int64_t input_channel = p.size( -3 );
    int64_t kernel_x = p.size( -2 );
    int64_t kernel_y = p.size( -1 );
    int64_t input_x = input_shape[input_shape.size() - 2];
    int64_t input_y = input_shape[input_shape.size() - 1];

    if ( inserted_zeros > 0 )
    {
        input_x = ( input_x - 1 ) * ( inserted_zeros + 1 ) + 1;
        input_y = ( input_y - 1 ) * ( inserted_zeros + 1 ) + 1;
    }

    torch::Tensor A_matrix;

    if ( !unstable_idx.has_value() )
    {
        if ( p.dim() != 7 )
        {
            throw std::runtime_error(
                "patches_to_matrix: expected 7D tensor for non-sparse case, got " +
                std::to_string( p.dim() ) + "D" );
        }

        output_channel = p.size( 0 );
        batch_size = p.size( 1 );
        output_x = p.size( 2 );
        output_y = p.size( 3 );

        int64_t padded_h = input_x + padding[2] + padding[3];
        int64_t padded_w = input_y + padding[0] + padding[1];

        A_matrix = torch::zeros(
            { batch_size, output_channel, output_x, output_y, input_channel, padded_h * padded_w },
            p.options() );

        auto orig_stride = A_matrix.strides();

        torch::Tensor matrix_strided =
            torch::as_strided( A_matrix,
                               { batch_size, output_channel, output_x, output_y, output_x, output_y,
                                 input_channel, kernel_x, kernel_y },
                               { orig_stride[0], orig_stride[1], orig_stride[2], orig_stride[3],
                                 padded_w * str[0], str[1], orig_stride[4], padded_w, 1 } );

        // Vectorized diagonal indexing -- single GPU kernel instead of O(output_x*output_y)
        torch::Tensor pieces_transposed = p.permute( { 1, 0, 2, 3, 4, 5, 6 } );

        {
            using namespace torch::indexing;
            auto dev_opts = torch::TensorOptions().dtype( torch::kLong ).device( p.device() );
            torch::Tensor first_indices = torch::arange( output_x * output_y, dev_opts );
            torch::Tensor second_indices = torch::div( first_indices, output_y, "trunc" );
            torch::Tensor third_indices = torch::fmod( first_indices, output_y );

            torch::Tensor pieces_reshaped =
                pieces_transposed.reshape( { batch_size, output_channel, output_x * output_y,
                                             input_channel, kernel_x, kernel_y } );

            matrix_strided.index_put_( { Slice(), Slice(), second_indices, third_indices,
                                         second_indices, third_indices, Slice(), Slice(), Slice() },
                                       pieces_reshaped );
        }

        A_matrix = A_matrix.view( { batch_size, output_channel * output_x * output_y, input_channel,
                                    padded_h, padded_w } );
    }
    else
    {
        const auto &idx = unstable_idx.value();
        int64_t unstable_size = idx[0].numel();
        batch_size = p.size( 1 );
        output_channel = output_shape[1];
        output_x = output_shape[2];
        output_y = output_shape[3];

        int64_t padded_h = input_x + padding[2] + padding[3];
        int64_t padded_w = input_y + padding[0] + padding[1];

        A_matrix = torch::zeros( { batch_size, unstable_size, input_channel, padded_h * padded_w },
                                 p.options() );

        auto orig_stride = A_matrix.strides();

        // Last dim is flattened (padded_h * padded_w), so row stride = padded_w
        torch::Tensor matrix_strided = torch::as_strided(
            A_matrix,
            { batch_size, unstable_size, output_x, output_y, input_channel, kernel_x, kernel_y },
            { orig_stride[0], orig_stride[1], padded_w * str[0], str[1], orig_stride[2], padded_w,
              1 } );

        torch::Tensor pieces_transposed = p.permute( { 1, 0, 2, 3, 4 } );

        // Vectorized fill at unstable positions -- single GPU kernel
        {
            using namespace torch::indexing;
            auto dev_opts = torch::TensorOptions().dtype( torch::kLong ).device( p.device() );
            torch::Tensor first_indices = torch::arange( unstable_size, dev_opts );

            matrix_strided.index_put_(
                { Slice(), first_indices, idx[1], idx[2], Slice(), Slice(), Slice() },
                pieces_transposed );
        }

        A_matrix =
            A_matrix.view( { batch_size, unstable_size, input_channel, padded_h, padded_w } );
    }

    {
        using namespace torch::indexing;
        A_matrix =
            A_matrix.index( { Slice(), Slice(), Slice(), Slice( padding[2], input_x + padding[2] ),
                              Slice( padding[0], input_y + padding[0] ) } );
    }

    if ( inserted_zeros > 0 )
    {
        using namespace torch::indexing;
        A_matrix =
            A_matrix.index( { Slice(), Slice(), Slice(), Slice( None, None, inserted_zeros + 1 ),
                              Slice( None, None, inserted_zeros + 1 ) } );
    }

    return A_matrix;
}

void Patches::ensure_4d( torch::Tensor &t )
{
    if ( !t.defined() )
        throw LunaError( LunaError::UNEXPECTED_TENSOR_SHAPE, "Tensor undefined in ensure_4d" );
    auto &shape = input_shape;
    if ( shape.size() < 4 )
        throw LunaError( LunaError::UNEXPECTED_TENSOR_SHAPE, "Expected patches with dimension 4" );
    int64_t C = shape[1], H = shape[2], W = shape[3];
    if ( t.dim() == 1 && t.size( 0 ) == C * H * W )
        t = t.view( { 1, C, H, W } );
    else if ( t.dim() == 2 && t.size( 1 ) == C * H * W )
        t = t.view( { t.size( 0 ), C, H, W } );
    else if ( t.dim() == 3 )
        t = t.unsqueeze( 0 );
}

} // namespace NLR
