/*********************                                                        */
/*! \file Patches.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __PATCHES_H__
#define __PATCHES_H__

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <optional>
#include "Debug.h"

namespace NLR
{

inline std::vector<int64_t> unify_shape( const std::vector<int64_t> &shape )
{
    if ( shape.size() == 4 )
        return shape;
    if ( shape.size() == 2 )
        return { shape[1], shape[1], shape[0], shape[0] }; // h,w -> left,right,top,bottom
    if ( shape.empty() )
        return { 0, 0, 0, 0 };
    if ( shape.size() == 1 )
        return { shape[0], shape[0], shape[0], shape[0] };
    return { 0, 0, 0, 0 };
}

class Patches
{
public:
    torch::Tensor patches;
    std::vector<int64_t> stride;
    std::vector<int64_t> padding;        // (left, right, top, bottom)
    std::vector<int64_t> output_padding; // (left, right, top, bottom)
    int64_t inserted_zeros;
    std::optional<std::vector<torch::Tensor>> unstable_idx;
    std::vector<int64_t> output_shape;
    std::vector<int64_t> input_shape;
    int identity;

    Patches()
        : inserted_zeros( 0 )
        , identity( 0 )
    {
        stride = { 1, 1 };
        padding = { 0, 0, 0, 0 };
        output_padding = { 0, 0, 0, 0 };
    }

    Patches( torch::Tensor p, std::vector<int64_t> s, std::vector<int64_t> pad,
             std::vector<int64_t> out_pad, int64_t zeros,
             std::optional<std::vector<torch::Tensor>> unstable, std::vector<int64_t> out_shape,
             std::vector<int64_t> in_shape, int id = 0 )
        : patches( p )
        , stride( s )
        , padding( pad )
        , output_padding( out_pad )
        , inserted_zeros( zeros )
        , unstable_idx( unstable )
        , output_shape( out_shape )
        , input_shape( in_shape )
        , identity( id )
    {
        simplify();
    }

    void simplify();

    std::shared_ptr<Patches>
    create_similar( std::optional<torch::Tensor> new_patches = std::nullopt,
                    std::optional<std::vector<int64_t>> new_stride = std::nullopt,
                    std::optional<std::vector<int64_t>> new_padding = std::nullopt,
                    std::optional<std::vector<int64_t>> new_output_padding = std::nullopt,
                    std::optional<int64_t> new_inserted_zeros = std::nullopt,
                    std::optional<int> new_identity = std::nullopt,
                    std::optional<std::vector<int64_t>> new_input_shape = std::nullopt ) const;

    std::shared_ptr<Patches> add( const std::shared_ptr<Patches> &other ) const;

    static torch::Tensor
    patches_to_matrix( const torch::Tensor &pieces, const std::vector<int64_t> &input_shape,
                       const std::vector<int64_t> &stride, const std::vector<int64_t> &padding,
                       const std::vector<int64_t> &output_shape,
                       const std::optional<std::vector<torch::Tensor>> &unstable_idx,
                       int64_t inserted_zeros );

    torch::Tensor to_matrix( const std::vector<int64_t> &in_shape ) const;

    // Bound computation via inplace_unfold + einsum (mirrors auto_LiRPA)
    torch::Tensor matmul( const torch::Tensor &input, bool patch_abs = false,
                          const std::vector<int64_t> &expand_shape = {} ) const;

    torch::Tensor ensure_4d( const torch::Tensor &t );

    inline bool is_sparse()
    {
        return unstable_idx.has_value();
    }
};

torch::Tensor insert_zeros( const torch::Tensor &image, int64_t s );
torch::Tensor remove_zeros( const torch::Tensor &image, int64_t s,
                            const std::pair<int64_t, int64_t> &start_idx = { 0, 0 } );
bool is_shape_used( const std::vector<int64_t> &shape, int expected = 0 );
torch::Tensor inplace_unfold( const torch::Tensor &image, const std::vector<int64_t> &kernel_size,
                              const std::vector<int64_t> &stride,
                              const std::vector<int64_t> &padding, int64_t inserted_zeros,
                              const std::vector<int64_t> &output_padding );

void compute_patches_stride_padding(
    const std::vector<int64_t> &input_shape, const std::vector<int64_t> &patches_padding,
    const std::vector<int64_t> &patches_stride, const std::vector<int64_t> &op_padding,
    const std::vector<int64_t> &op_stride, int64_t inserted_zeros,
    const std::vector<int64_t> &output_padding, std::vector<int64_t> &new_padding,
    std::vector<int64_t> &new_stride, std::vector<int64_t> &new_output_padding );

torch::Tensor create_valid_mask( const std::vector<int64_t> &output_shape,
                                 const torch::Device &device, torch::Dtype dtype,
                                 const std::vector<int64_t> &kernel_size,
                                 const std::vector<int64_t> &stride, int64_t inserted_zeros,
                                 const std::vector<int64_t> &padding,
                                 const std::vector<int64_t> &output_padding,
                                 const std::optional<std::vector<torch::Tensor>> &unstable_idx );

} // namespace NLR

#endif // __PATCHES_H__
