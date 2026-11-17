/*********************                                                        */
/*! \file BoundedBatchNormNode.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "BoundedBatchNormNode.h"
#include <stdexcept>
#include <sstream>

// Undefine Warning macro to avoid conflict with PyTorch
#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>

// Redefine Warning macro for CVC4 compatibility
#ifndef Warning
#define Warning ( !::CVC4::WarningChannel.isOn() ) ? ::CVC4::nullCvc4Stream : ::CVC4::WarningChannel
#endif

namespace NLR
{

using torch::indexing::Slice;

BoundedBatchNormNode::BoundedBatchNormNode( const torch::Tensor &scale, const torch::Tensor &B,
                                            const torch::Tensor &mean, const torch::Tensor &var,
                                            float eps, const String &name )
    : _scale( scale.detach().to( torch::kFloat32 ).contiguous() )
    , _bias( B.detach().to( torch::kFloat32 ).contiguous() )
    , _mean( mean.detach().to( torch::kFloat32 ).contiguous() )
    , _var( var.detach().to( torch::kFloat32 ).contiguous() )
    , _eps( eps )
{
    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;

    if ( _scale.defined() && _scale.dim() != 1 )
    {
        throw std::runtime_error( "BoundedBatchNormNode: scale must be 1D" );
    }
    if ( _bias.defined() && _bias.dim() != 1 )
    {
        throw std::runtime_error( "BoundedBatchNormNode: bias must be 1D" );
    }
    if ( _mean.defined() && _mean.dim() != 1 )
    {
        throw std::runtime_error( "BoundedBatchNormNode: mean must be 1D" );
    }
    if ( _var.defined() && _var.dim() != 1 )
    {
        throw std::runtime_error( "BoundedBatchNormNode: var must be 1D" );
    }
    auto c = _scale.numel();
    if ( _bias.numel() != c || _mean.numel() != c || _var.numel() != c )
    {
        throw std::runtime_error( "BoundedBatchNormNode: parameter channel sizes mismatch" );
    }
}

void BoundedBatchNormNode::ensureCachedParams( const torch::Device &device ) const
{
    if ( _cached_tmp_weight.defined() && _cached_device == device )
    {
        return;
    }
    auto options = torch::TensorOptions().dtype( torch::kFloat32 ).device( device );
    auto scale = _scale.to( options );
    auto var = _var.to( options );
    auto B = _bias.to( options );
    auto mean = _mean.to( options );
    _cached_tmp_weight = scale / torch::sqrt( var + _eps );
    _cached_tmp_bias = B - mean * _cached_tmp_weight;
    _cached_device = device;
}

torch::Tensor BoundedBatchNormNode::tmp_weight( const torch::Tensor &like ) const
{
    auto device = like.defined() ? like.device() : _scale.device();
    ensureCachedParams( device );
    return _cached_tmp_weight;
}

torch::Tensor BoundedBatchNormNode::tmp_bias( const torch::Tensor &like ) const
{
    auto device = like.defined() ? like.device() : _scale.device();
    ensureCachedParams( device );
    return _cached_tmp_bias;
}

torch::Tensor BoundedBatchNormNode::broadcast_channel_param( const torch::Tensor &param,
                                                             const torch::Tensor &x ) const
{
    if ( !param.defined() )
        return param;
    if ( !x.defined() )
        return param;
    // Channel dim = 1 for ONNX BatchNormalization; flattened activations need expansion
    int64_t C = param.numel();

    if ( x.dim() == 0 )
    {
        return param;
    }

    if ( x.dim() == 1 )
    {
        int64_t flat = x.numel();
        if ( flat == C )
            return param;
        if ( C > 0 && flat % C == 0 )
        {
            int64_t spatial = flat / C;
            return param.repeat_interleave( spatial );
        }
        std::ostringstream oss;
        oss << "BoundedBatchNormNode: cannot broadcast channel param: flat=" << flat
            << " is not divisible by C=" << C << " (nodeIndex=" << _nodeIndex
            << ", name=" << _nodeName.ascii() << ")";
        throw std::runtime_error( oss.str() );
    }

    if ( x.dim() == 2 )
    {
        int64_t flat = x.size( 1 );
        if ( flat == C )
            return param.view( { 1, C } );
        if ( C > 0 && flat % C == 0 )
        {
            int64_t spatial = flat / C;
            return param.repeat_interleave( spatial ).view( { 1, flat } );
        }
        std::ostringstream oss;
        oss << "BoundedBatchNormNode: cannot broadcast channel param for 2D input: flat=" << flat
            << " is not divisible by C=" << C << " (nodeIndex=" << _nodeIndex
            << ", name=" << _nodeName.ascii() << ")";
        throw std::runtime_error( oss.str() );
    }

    // Standard N-D conv-style layout: [N, C, ...]
    std::vector<int64_t> view_shape( x.dim(), 1 );
    view_shape[1] = C;
    return param.view( view_shape );
}

torch::Tensor BoundedBatchNormNode::forward( const torch::Tensor &input )
{
    torch::Tensor x = input.to( torch::kFloat32 ).contiguous();
    _last_input_shape.clear();
    for ( int i = 0; i < x.dim(); ++i )
        _last_input_shape.push_back( x.size( i ) );

    _input_size = x.numel() / ( x.size( 0 ) > 0 ? x.size( 0 ) : 1 );
    _output_size = _input_size;

    auto w = broadcast_channel_param( tmp_weight( x ), x );
    auto b = broadcast_channel_param( tmp_bias( x ), x );
    return w * x + b;
}

BoundedTensor<torch::Tensor> BoundedBatchNormNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>> &inputBounds )
{

    if ( inputBounds.size() < 1 )
    {
        throw std::runtime_error( "BoundedBatchNormNode: expects one input bound" );
    }
    torch::Tensor l = inputBounds[0].lower().to( torch::kFloat32 );
    torch::Tensor u = inputBounds[0].upper().to( torch::kFloat32 );

    _last_input_shape.clear();
    for ( int i = 0; i < l.dim(); ++i )
        _last_input_shape.push_back( l.size( i ) );

    if ( _input_size == 0 && l.defined() )
    {
        _input_size = l.numel() / ( l.size( 0 ) > 0 ? l.size( 0 ) : 1 );
    }
    if ( _output_size == 0 )
        _output_size = _input_size;

    auto w = broadcast_channel_param( tmp_weight( l ), l );
    auto b = broadcast_channel_param( tmp_bias( l ), l );

    auto w_pos = torch::clamp_min( w, 0 );
    auto w_neg = torch::clamp_max( w, 0 );

    torch::Tensor out_l = w_pos * l + w_neg * u + b;
    torch::Tensor out_u = w_pos * u + w_neg * l + b;

    return BoundedTensor<torch::Tensor>( out_l, out_u );
}

BoundA
BoundedBatchNormNode::boundOneSideTensor( const BoundA &last_A,
                                          const Vector<BoundedTensor<torch::Tensor>> &inputBounds,
                                          torch::Tensor &sum_bias )
{

    if ( !last_A.defined() )
    {
        auto options = torch::TensorOptions().dtype( torch::kFloat32 ).device( _device );
        sum_bias = torch::zeros( { 1 }, options );
        return BoundA();
    }
    if ( !last_A.isTensor() )
    {
        throw std::runtime_error(
            "BoundedBatchNormNode::boundOneSideTensor called with non-tensor last_A" );
    }
    if ( inputBounds.size() < 1 )
    {
        throw std::runtime_error( "BoundedBatchNormNode::boundOneSideTensor expects input bounds" );
    }

    auto A = last_A.asTensor().to( torch::kFloat32 );
    auto w_ch = tmp_weight( A );
    auto b_ch = tmp_bias( A );

    auto in_l = inputBounds[0].lower();
    std::vector<int64_t> xshape = in_l.defined() ? in_l.sizes().vec() : _last_input_shape;

    auto apply_affine = [&]( torch::Tensor A_reshaped, bool has_spec_dim, bool spec_first )
    {
        torch::Tensor w_view;
        torch::Tensor b_view;
        if ( A_reshaped.dim() == 5 )
        {
            w_view = w_ch.view( { 1, 1, -1, 1, 1 } );
            b_view = b_ch.view( { 1, 1, -1, 1, 1 } );
            torch::Tensor next_A = A_reshaped * w_view;
            torch::Tensor sum_spatial = A_reshaped.sum( { 3, 4 } );
            torch::Tensor sb = ( sum_spatial * b_ch.view( { 1, 1, -1 } ) ).sum( 2 );
            return std::make_pair( next_A, sb );
        }
        else if ( A_reshaped.dim() == 4 )
        {
            w_view = w_ch.view( { 1, -1, 1, 1 } );
            b_view = b_ch.view( { 1, -1, 1, 1 } );
            torch::Tensor next_A = A_reshaped * w_view;
            torch::Tensor sum_spatial = A_reshaped.sum( { 2, 3 } );
            torch::Tensor sb = ( sum_spatial * b_ch.view( { 1, -1 } ) ).sum( 1 );
            return std::make_pair( next_A, sb );
        }
        else if ( A_reshaped.dim() == 3 )
        {
            w_view = w_ch.view( { 1, 1, -1 } );
            torch::Tensor next_A = A_reshaped * w_view;
            torch::Tensor sb = ( A_reshaped * b_ch.view( { 1, 1, -1 } ) ).sum( 2 );
            return std::make_pair( next_A, sb );
        }
        else if ( A_reshaped.dim() == 2 )
        {
            w_view = w_ch.view( { 1, -1 } );
            torch::Tensor next_A = A_reshaped * w_view;
            torch::Tensor sb = ( A_reshaped * b_ch.view( { 1, -1 } ) ).sum( 1 );
            return std::make_pair( next_A, sb );
        }
        else
        {
            throw std::runtime_error( "BoundedBatchNormNode: unsupported tensor last_A dims" );
        }
        (void)has_spec_dim;
        (void)spec_first;
    };

    if ( A.dim() == 5 || A.dim() == 4 )
    {
        auto [next_A, sb] =
            apply_affine( A, /*has_spec_dim=*/( A.dim() == 5 ), /*spec_first=*/true );
        sum_bias = sb;
        return BoundA( next_A );
    }

    if ( A.dim() == 3 )
    {
        int64_t d0 = A.size( 0 ), d1 = A.size( 1 ), flat = A.size( 2 );

        int64_t C = w_ch.numel();

        if ( xshape.size() >= 2 )
        {
            int64_t H = 1, W = 1;
            if ( xshape.size() >= 4 )
            {
                H = xshape[2];
                W = xshape[3];
            }
            if ( C * H * W != flat && xshape.size() >= 4 )
            {
                int64_t spatial = flat / C;
                H = spatial;
                W = 1;
            }

            int64_t batch_from_bounds = ( xshape.size() > 0 ) ? xshape[0] : 1;
            bool is_BS = ( d0 == batch_from_bounds );

            torch::Tensor A5;
            if ( is_BS && xshape.size() >= 4 )
            {
                A5 = A.reshape( { d0, d1, C, H, W } );
                auto [next_A5, sb] = apply_affine( A5, true, false );
                sum_bias = sb;
                return BoundA( next_A5.reshape( { d0, d1, flat } ) );
            }
            else if ( !is_BS && xshape.size() >= 4 )
            {
                A5 = A.reshape( { d0, d1, C, H, W } );
                auto [next_A5, sb] = apply_affine( A5, true, true );
                sum_bias = sb;
                return BoundA( next_A5.reshape( { d0, d1, flat } ) );
            }
            else if ( xshape.size() == 2 )
            {
                torch::Tensor A3 = A.reshape( { d0, d1, C } );
                auto [next_A3, sb] = apply_affine( A3, true, false );
                sum_bias = sb;
                return BoundA( next_A3.reshape( { d0, d1, flat } ) );
            }
        }
        else
        {
            if ( C > 0 && flat % C == 0 )
            {
                int64_t spatial = flat / C;
                torch::Tensor A5 = A.reshape( { d0, d1, C, spatial, 1 } );
                auto [next_A5, sb] = apply_affine( A5, true, false );
                sum_bias = sb;
                return BoundA( next_A5.reshape( { d0, d1, flat } ) );
            }
        }
    }

    if ( A.dim() == 3 && xshape.size() == 1 )
    {
        int64_t d0 = A.size( 0 );
        int64_t d1 = A.size( 1 );
        int64_t A_flat = A.size( 2 );
        int64_t C = w_ch.numel();
        int64_t input_flat = xshape[0];

        if ( A_flat != input_flat && C > 0 && input_flat % C == 0 )
        {
            int64_t spatial = input_flat / C;
            int64_t H = static_cast<int64_t>( std::sqrt( spatial ) );
            int64_t W = spatial / H;

            if ( H * W == spatial )
            {
                torch::Tensor A_expanded;
                if ( A_flat == 1 )
                {
                    A_expanded = A.expand( { d0, d1, input_flat } );
                }
                else if ( A_flat == C )
                {
                    A_expanded = A.view( { d0, d1, C, 1, 1 } ).expand( { d0, d1, C, H, W } );
                    A_expanded = A_expanded.reshape( { d0, d1, input_flat } );
                }
                else
                {
                    A_expanded = A.expand( { d0, d1, input_flat } );
                }

                A_expanded = A_expanded.reshape( { d0, d1, C, H, W } );

                auto [next_A5, sb] = apply_affine( A_expanded, true, true );
                sum_bias = sb;
                return BoundA( next_A5.reshape( { d0, d1, input_flat } ) );
            }
        }
    }

    if ( A.dim() == 2 )
    {
        int64_t S = A.size( 0 );
        int64_t flat = A.size( 1 );
        int64_t C = w_ch.numel();
        if ( _last_input_shape.size() >= 4 )
        {
            int64_t H = _last_input_shape[2];
            int64_t W = _last_input_shape[3];
            if ( C * H * W == flat )
            {
                auto A5 = A.reshape( { S, 1, C, H, W } );
                auto [next_A5, sb] = apply_affine( A5, true, true );
                sum_bias = sb.squeeze( 1 );
                return BoundA( next_A5.reshape( { S, flat } ) );
            }
        }
        if ( C > 0 && flat % C == 0 )
        {
            int64_t spatial = flat / C;
            auto A5 = A.reshape( { S, 1, C, spatial, 1 } );
            auto [next_A5, sb] = apply_affine( A5, true, true );
            sum_bias = sb.squeeze( 1 );
            return BoundA( next_A5.reshape( { S, flat } ) );
        }
        if ( flat == C )
        {
            auto [next_A2, sb] = apply_affine( A, false, true );
            sum_bias = sb;
            return BoundA( next_A2 );
        }
    }

    std::ostringstream oss;
    oss << "BoundedBatchNormNode: unsupported tensor last_A shape for BN backward"
        << " (nodeIndex=" << _nodeIndex << ", name=" << _nodeName.ascii() << ")"
        << " A.dim=" << A.dim() << " A.sizes=" << A.sizes() << " C=" << w_ch.numel();
    if ( inputBounds.size() >= 1 && inputBounds[0].lower().defined() )
    {
        oss << " input.lower.sizes=" << inputBounds[0].lower().sizes();
    }
    throw std::runtime_error( oss.str() );
}

BoundA
BoundedBatchNormNode::boundOneSidePatches( const BoundA &last_A,
                                           const Vector<BoundedTensor<torch::Tensor>> &inputBounds,
                                           torch::Tensor &sum_bias )
{

    if ( !last_A.defined() )
    {
        auto options = torch::TensorOptions().dtype( torch::kFloat32 ).device( _device );
        sum_bias = torch::zeros( { 1 }, options );
        return BoundA();
    }
    if ( !last_A.isPatches() )
    {
        throw std::runtime_error(
            "BoundedBatchNormNode::boundOneSidePatches called with non-patches last_A" );
    }
    if ( inputBounds.size() < 1 )
    {
        throw std::runtime_error(
            "BoundedBatchNormNode::boundOneSidePatches expects input bounds" );
    }

    auto patches = last_A.asPatches();

    torch::Device device = _device;
    if ( inputBounds[0].lower().defined() )
        device = inputBounds[0].lower().device();
    ensureCachedParams( device );
    auto w = _cached_tmp_weight;
    auto b = _cached_tmp_bias;

    // Identity patches: materialize into diagonal channel-weight patches
    if ( patches->identity == 1 )
    {
        int64_t C = w.numel();
        int64_t batch = patches->output_shape[0];
        int64_t out_h = patches->output_shape[2];
        int64_t out_w = patches->output_shape[3];

        torch::Tensor eye_w =
            torch::eye( C, torch::TensorOptions().dtype( torch::kFloat32 ).device( device ) ) *
            w.view( { -1 } );

        if ( !patches->is_sparse() )
        {
            torch::Tensor P_new = eye_w.view( { C, 1, 1, 1, C, 1, 1 } )
                                      .expand( { C, batch, out_h, out_w, C, 1, 1 } )
                                      .contiguous();

            sum_bias = b.view( { C, 1, 1, 1 } ).expand( { C, batch, out_h, out_w } ).contiguous();

            return BoundA( patches->create_similar( P_new, std::vector<int64_t>{ 1, 1 },
                                                    std::vector<int64_t>{ 0, 0, 0, 0 },
                                                    std::vector<int64_t>{ 0, 0, 0, 0 }, 0, 0 ) );
        }
        else
        {
            auto &idx = patches->unstable_idx.value();
            torch::Tensor P_new = eye_w.index_select( 0, idx[0] );
            P_new = P_new.view( { (int64_t)idx[0].size( 0 ), 1, C, 1, 1 } )
                        .expand( { (int64_t)idx[0].size( 0 ), batch, C, 1, 1 } )
                        .contiguous();

            sum_bias = b.index_select( 0, idx[0] )
                           .unsqueeze( -1 )
                           .expand( { (int64_t)idx[0].size( 0 ), batch } )
                           .contiguous();

            return BoundA( patches->create_similar( P_new, std::vector<int64_t>{ 1, 1 },
                                                    std::vector<int64_t>{ 0, 0, 0, 0 },
                                                    std::vector<int64_t>{ 0, 0, 0, 0 }, 0, 0 ) );
        }
    }

    torch::Tensor P = ( patches->patches.scalar_type() == torch::kFloat32 )
                          ? patches->patches
                          : patches->patches.to( torch::kFloat32 );

    std::vector<int64_t> view( P.dim(), 1 );
    view[P.dim() - 3] = w.numel();
    torch::Tensor P_scaled = P * w.view( view );

    torch::Tensor in_l = inputBounds[0].lower();
    std::vector<int64_t> xshape = in_l.defined() ? in_l.sizes().vec() : _last_input_shape;
    if ( xshape.size() < 4 && _last_input_shape.size() >= 4 )
    {
        xshape = _last_input_shape;
    }
    if ( xshape.size() < 4 && xshape.size() >= 1 )
    {
        int64_t C = w.numel();
        int64_t flat = xshape.back();
        if ( C > 0 && flat % C == 0 )
        {
            int64_t spatial = flat / C;
            int64_t H = static_cast<int64_t>( std::sqrt( static_cast<double>( spatial ) ) );
            int64_t W = spatial / H;
            if ( H * W == spatial )
            {
                xshape = { 1, C, H, W };
            }
        }
    }
    if ( xshape.size() < 4 )
    {
        sum_bias = torch::zeros( { 1 }, P.options() );
        return BoundA( patches->create_similar( P_scaled ) );
    }

    int64_t C = w.numel();
    int64_t H = xshape[2];
    int64_t W = xshape[3];

    torch::Tensor bias_map = b.view( { -1, 1, 1 } ).expand( { C, H, W } ).unsqueeze( 0 );

    std::vector<int64_t> ksize = { P.size( -2 ), P.size( -1 ) };
    torch::Tensor bias_unfolded =
        inplace_unfold( bias_map, ksize, patches->stride, patches->padding, patches->inserted_zeros,
                        patches->output_padding );

    if ( patches->is_sparse() )
    {
        auto idx = patches->unstable_idx.value();
        if ( idx.size() < 3 )
        {
            sum_bias = torch::zeros( { 1 }, P.options() );
            return BoundA( patches->create_similar( P_scaled ) );
        }
        torch::Tensor bias_sel = bias_unfolded.index( { 0, idx[1], idx[2] } );
        if ( bias_sel.dim() == 4 )
        {
            bias_sel = bias_sel.unsqueeze( 0 );
        }
        bias_sel = bias_sel.permute( { 1, 0, 2, 3, 4 } );
        bias_sel =
            bias_sel.expand( { P.size( 0 ), P.size( 1 ), P.size( 2 ), P.size( 3 ), P.size( 4 ) } );

        // Use ORIGINAL P (before w scaling) for bias: sb = sum(A * b), not sum(A*w*b)
        torch::Tensor sb = ( P * bias_sel ).sum( { 2, 3, 4 } );
        sum_bias = sb;
    }
    else
    {
        torch::Tensor bias_ready = bias_unfolded.unsqueeze( 0 );
        if ( bias_ready.dim() == 7 )
        {
        }
        else if ( bias_ready.dim() == 6 )
        {
            bias_ready = bias_unfolded.unsqueeze( 0 ).unsqueeze( 0 );
        }
        // Use ORIGINAL P (before w scaling) for bias: sb = sum(A * b), not sum(A*w*b)
        torch::Tensor sb = ( P * bias_ready ).sum( { -3, -2, -1 } );
        sum_bias = sb;
    }

    return BoundA( patches->create_similar( P_scaled ) );
}

void BoundedBatchNormNode::boundBackward( const BoundA &last_lA, const BoundA &last_uA,
                                          const Vector<BoundedTensor<torch::Tensor>> &inputBounds,
                                          Vector<Pair<BoundA, BoundA>> &outputA_matrices,
                                          torch::Tensor &lbias, torch::Tensor &ubias )
{

    outputA_matrices.clear();

    BoundA lA_x, uA_x;
    torch::Tensor lbias_add, ubias_add;

    if ( last_lA.defined() )
    {
        if ( last_lA.isTensor() )
            lA_x = boundOneSideTensor( last_lA, inputBounds, lbias_add );
        else
            lA_x = boundOneSidePatches( last_lA, inputBounds, lbias_add );
    }
    if ( last_uA.defined() )
    {
        if ( last_uA.isTensor() )
            uA_x = boundOneSideTensor( last_uA, inputBounds, ubias_add );
        else
            uA_x = boundOneSidePatches( last_uA, inputBounds, ubias_add );
    }

    outputA_matrices.append( Pair<BoundA, BoundA>( lA_x, uA_x ) );

    if ( lbias_add.defined() && lbias_add.numel() > 0 )
    {
        lbias = lbias.defined() ? ( lbias + lbias_add ) : lbias_add;
    }
    if ( ubias_add.defined() && ubias_add.numel() > 0 )
    {
        ubias = ubias.defined() ? ( ubias + ubias_add ) : ubias_add;
    }
}

void BoundedBatchNormNode::moveToDevice( const torch::Device &device )
{
    BoundedTorchNode::moveToDevice( device );
    _scale = _scale.to( device );
    _bias = _bias.to( device );
    _mean = _mean.to( device );
    _var = _var.to( device );
    // Invalidate derived-param cache; rebuilt lazily on next use
    _cached_tmp_weight = torch::Tensor();
    _cached_tmp_bias = torch::Tensor();
}

} // namespace NLR
