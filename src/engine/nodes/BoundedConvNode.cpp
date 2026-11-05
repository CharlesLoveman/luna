/*********************                                                        */
/*! \file BoundedConvNode.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "BoundedConvNode.h"
#include "conv/MatrixConvolution.h"
#include "conv/Patches.h"
#include <torch/nn/functional.h>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <sstream>

namespace NLR
{

BoundedConvNode::BoundedConvNode( const torch::nn::Conv1d &convModule, ConvMode mode,
                                  const String &name )
    : conv1d( convModule )
    , conv_dim( 1 )
    , mode( mode )
{

    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;
    relu_followed = false;
    patches_start = true;

    initializeFromConv1d( convModule );
}

BoundedConvNode::BoundedConvNode( const torch::nn::Conv2d &convModule, ConvMode mode,
                                  const String &name )
    : conv2d( convModule )
    , conv_dim( 2 )
    , mode( mode )
{

    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;
    relu_followed = false;
    patches_start = true;

    initializeFromConv2d( convModule );
}

void BoundedConvNode::initializeFromConv1d( const torch::nn::Conv1d &convModule )
{
    if ( !convModule )
    {
        throw std::runtime_error( "Conv1d module is null" );
    }

    auto options = convModule->options;

    if ( std::holds_alternative<torch::ExpandingArray<1>>( options.padding() ) )
    {
        auto pad_array = std::get<torch::ExpandingArray<1>>( options.padding() );
        padding = { static_cast<int>( ( *pad_array )[0] ) };
    }
    else
    {
        padding = { 0 };
    }

    stride = { static_cast<int>( ( *options.stride() )[0] ) };
    dilation = { static_cast<int>( ( *options.dilation() )[0] ) };
    groups = options.groups();
    has_bias = options.bias();

    if ( conv1d && conv1d->weight.defined() )
    {
        auto weight = conv1d->weight;

        bool weightNeedsFix = false;
        if ( !weight.requires_grad() || !weight.is_contiguous() ||
             weight.dtype() != torch::kFloat32 )
        {
            weightNeedsFix = true;
        }

        if ( weightNeedsFix )
        {
            conv1d->weight = weight.contiguous().to( torch::kFloat32 ).requires_grad_( false );
        }

        if ( has_bias && conv1d->bias.defined() )
        {
            auto bias = conv1d->bias;
            bool biasNeedsFix = false;
            if ( !bias.requires_grad() || !bias.is_contiguous() || bias.dtype() != torch::kFloat32 )
            {
                biasNeedsFix = true;
            }

            if ( biasNeedsFix )
            {
                conv1d->bias = bias.contiguous().to( torch::kFloat32 ).requires_grad_( false );
            }
        }
    }

    stride_64 = std::vector<int64_t>( stride.begin(), stride.end() );
    padding_64 = std::vector<int64_t>( padding.begin(), padding.end() );
    dilation_64 = std::vector<int64_t>( dilation.begin(), dilation.end() );
}

void BoundedConvNode::initializeFromConv2d( const torch::nn::Conv2d &convModule )
{
    if ( !convModule )
    {
        throw std::runtime_error( "Conv2d module is null" );
    }

    auto options = convModule->options;

    if ( std::holds_alternative<torch::ExpandingArray<2>>( options.padding() ) )
    {
        auto pad_array = std::get<torch::ExpandingArray<2>>( options.padding() );
        padding = { static_cast<int>( ( *pad_array )[0] ), static_cast<int>( ( *pad_array )[1] ) };
    }
    else
    {
        padding = { 0, 0 };
    }

    stride = { static_cast<int>( ( *options.stride() )[0] ),
               static_cast<int>( ( *options.stride() )[1] ) };
    dilation = { static_cast<int>( ( *options.dilation() )[0] ),
                 static_cast<int>( ( *options.dilation() )[1] ) };
    groups = options.groups();
    has_bias = options.bias();

    if ( conv2d && conv2d->weight.defined() )
    {
        auto weight = conv2d->weight;

        // Do NOT use detach() - breaks computation graph needed for Alpha-CROWN
        bool weightNeedsFix = false;
        if ( !weight.requires_grad() || !weight.is_contiguous() ||
             weight.dtype() != torch::kFloat32 )
        {
            weightNeedsFix = true;
        }

        if ( weightNeedsFix )
        {
            conv2d->weight = weight.contiguous().to( torch::kFloat32 ).requires_grad_( false );
        }

        if ( has_bias && conv2d->bias.defined() )
        {
            auto bias = conv2d->bias;
            // Do NOT use detach() - breaks computation graph needed for Alpha-CROWN
            bool biasNeedsFix = false;
            if ( !bias.requires_grad() || !bias.is_contiguous() || bias.dtype() != torch::kFloat32 )
            {
                biasNeedsFix = true;
            }

            if ( biasNeedsFix )
            {
                conv2d->bias = bias.contiguous().to( torch::kFloat32 ).requires_grad_( false );
            }
        }
    }

    stride_64 = std::vector<int64_t>( stride.begin(), stride.end() );
    padding_64 = std::vector<int64_t>( padding.begin(), padding.end() );
    dilation_64 = std::vector<int64_t>( dilation.begin(), dilation.end() );
}

torch::Tensor BoundedConvNode::forward( const torch::Tensor &input )
{
    const auto device = input.device();
    torch::Tensor inputFloat =
        input.to( torch::TensorOptions().dtype( torch::kFloat32 ).device( device ) ).contiguous();

    // ONNX parsing sets proper 4D shape; forward() might receive flattened input
    if ( input_shape.empty() )
    {
        for ( int i = 0; i < input.dim(); ++i )
        {
            input_shape.push_back( input.size( i ) );
        }
    }

    torch::Tensor output;

    ensureWeightsOnDevice( device );
    const torch::Tensor &weight = _cached_weight;
    const torch::Tensor &bias = _cached_bias;

    if ( conv_dim == 1 )
    {
        if ( !conv1d )
        {
            throw std::runtime_error( "Conv1d module not initialized" );
        }

        output = torch::nn::functional::conv1d( inputFloat, weight,
                                                torch::nn::functional::Conv1dFuncOptions()
                                                    .bias( bias )
                                                    .stride( stride_64 )
                                                    .padding( padding_64 )
                                                    .dilation( dilation_64 )
                                                    .groups( groups ) );
    }
    else
    {
        if ( !conv2d )
        {
            throw std::runtime_error( "Conv2d module not initialized" );
        }

        if ( mode == ConvMode::MATRIX )
        {
            std::vector<int> kernel_size = { static_cast<int>( weight.size( 2 ) ),
                                             static_cast<int>( weight.size( 3 ) ) };

            std::vector<int> spatial_output = MatrixConvolution::computeConvOutputShape(
                { static_cast<int>( input.size( 2 ) ), static_cast<int>( input.size( 3 ) ) },
                kernel_size, stride, padding, dilation );

            torch::Tensor input_matrix = MatrixConvolution::im2col( inputFloat, kernel_size, stride,
                                                                    padding, dilation, groups );

            output =
                MatrixConvolution::matrixConvForward( input_matrix, weight, bias, spatial_output );
        }
        else
        {
            output = torch::nn::functional::conv2d( inputFloat, weight,
                                                    torch::nn::functional::Conv2dFuncOptions()
                                                        .bias( bias )
                                                        .stride( stride_64 )
                                                        .padding( padding_64 )
                                                        .dilation( dilation_64 )
                                                        .groups( groups ) );
        }
    }

    output_shape.clear();
    for ( int i = 0; i < output.dim(); ++i )
    {
        output_shape.push_back( output.size( i ) );
    }

    _input_size = input.numel() / input.size( 0 );
    _output_size = output.numel() / output.size( 0 );

    return output;
}

void BoundedConvNode::moveToDevice( const torch::Device &device )
{
    BoundedTorchNode::moveToDevice( device );
    if ( conv1d )
    {
        conv1d->to( device );
    }
    if ( conv2d )
    {
        conv2d->to( device );
    }
    _cached_weight = torch::Tensor();
    _cached_bias = torch::Tensor();
}

void BoundedConvNode::ensureWeightsOnDevice( const torch::Device &device ) const
{
    if ( _cached_weight.defined() && _cached_device == device )
    {
        return;
    }

    auto options = torch::TensorOptions().dtype( torch::kFloat32 ).device( device );
    if ( conv_dim == 1 && conv1d )
    {
        _cached_weight = conv1d->weight.to( options ).contiguous();
        if ( has_bias && conv1d->bias.defined() )
        {
            _cached_bias = conv1d->bias.to( options );
        }
        else
        {
            _cached_bias = torch::Tensor();
        }
    }
    else if ( conv_dim == 2 && conv2d )
    {
        _cached_weight = conv2d->weight.to( options ).contiguous();
        if ( has_bias && conv2d->bias.defined() )
        {
            _cached_bias = conv2d->bias.to( options );
        }
        else
        {
            _cached_bias = torch::Tensor();
        }
    }
    _cached_device = device;
}

void BoundedConvNode::boundBackward( const BoundA &last_lA, const BoundA &last_uA,
                                     const Vector<BoundedTensor<torch::Tensor>> &inputBounds,
                                     Vector<Pair<BoundA, BoundA>> &outputA_matrices,
                                     torch::Tensor &lbias, torch::Tensor &ubias )
{

    // Forward pass not always run before CROWN
    if ( input_shape.empty() )
    {
        if ( inputBounds.size() < 1 )
        {
            throw std::runtime_error(
                "BoundedConvNode: input_shape empty and no input bounds provided" );
        }

        auto &lb = inputBounds[0].lower();
        int64_t total_input_size = lb.numel();

        if ( conv_dim == 1 )
        {
            torch::Tensor weight = conv1d->weight;
            int64_t in_channels_per_group = weight.size( 1 );
            int64_t in_channels = in_channels_per_group * groups;

            int64_t L = total_input_size / in_channels;
            input_shape = { 1, static_cast<int>( in_channels ), static_cast<int>( L ) };

            int64_t kernel_length = weight.size( 2 );
            int64_t out_l =
                ( L + 2 * padding[0] - dilation[0] * ( kernel_length - 1 ) - 1 ) / stride[0] + 1;
            output_shape = { 1, static_cast<int>( weight.size( 0 ) ), static_cast<int>( out_l ) };

            _input_size = total_input_size;
            _output_size = weight.size( 0 ) * out_l;
        }
        else
        {
            torch::Tensor weight = conv2d->weight;
            int64_t in_channels_per_group = weight.size( 1 );
            int64_t in_channels = in_channels_per_group * groups;
            int64_t kernel_h = weight.size( 2 );
            int64_t kernel_w = weight.size( 3 );

            int64_t spatial_size = total_input_size / in_channels;

            int64_t H = 0, W = 0;

            int64_t sqrt_spatial = static_cast<int64_t>( std::sqrt( spatial_size ) );
            if ( sqrt_spatial * sqrt_spatial == spatial_size && sqrt_spatial >= kernel_h &&
                 sqrt_spatial >= kernel_w )
            {
                H = sqrt_spatial;
                W = sqrt_spatial;
            }
            else
            {
                throw std::runtime_error(
                    "BoundedConvNode: ambiguous flat 2D input shape; explicit shape required" );
            }

            input_shape = { 1, static_cast<int>( in_channels ), static_cast<int>( H ),
                            static_cast<int>( W ) };

            std::vector<int> kernel_size_vec = { static_cast<int>( kernel_h ),
                                                 static_cast<int>( kernel_w ) };
            std::vector<int> spatial_output = MatrixConvolution::computeConvOutputShape(
                { static_cast<int>( H ), static_cast<int>( W ) }, kernel_size_vec, stride, padding,
                dilation );

            output_shape = { 1, static_cast<int>( weight.size( 0 ) ), spatial_output[0],
                             spatial_output[1] };

            _input_size = total_input_size;
            _output_size = weight.size( 0 ) * spatial_output[0] * spatial_output[1];
        }
    }

    if ( inputBounds.size() < 1 )
    {
        throw std::runtime_error( "BoundedConvNode expects at least one input" );
    }

    torch::Device device = _device;
    if ( last_lA.defined() && last_lA.isTensor() )
    {
        device = last_lA.asTensor().device();
    }
    else if ( last_uA.defined() && last_uA.isTensor() )
    {
        device = last_uA.asTensor().device();
    }
    else if ( !inputBounds.empty() && inputBounds[0].lower().defined() )
    {
        device = inputBounds[0].lower().device();
    }

    ensureWeightsOnDevice( device );
    const torch::Tensor &weight = _cached_weight;
    const torch::Tensor &bias = _cached_bias;

    torch::Tensor lA_bias_contrib, uA_bias_contrib;
    BoundA lA_x = boundOneSide( last_lA, weight, bias, lA_bias_contrib );
    BoundA uA_x = boundOneSide( last_uA, weight, bias, uA_bias_contrib );

    if ( inputBounds.size() > 0 && inputBounds[0].lower().dim() == 1 )
    {
        if ( lA_x.isTensor() )
        {
            torch::Tensor t = lA_x.asTensor();
            if ( t.dim() == 4 )
            {
                lA_x = BoundA( t.reshape( { t.size( 0 ), -1 } ) );
            }
            else if ( t.dim() == 5 )
            {
                lA_x = BoundA( t.reshape( { t.size( 0 ), t.size( 1 ), -1 } ) );
            }
        }
        if ( uA_x.isTensor() )
        {
            torch::Tensor t = uA_x.asTensor();
            if ( t.dim() == 4 )
            {
                uA_x = BoundA( t.reshape( { t.size( 0 ), -1 } ) );
            }
            else if ( t.dim() == 5 )
            {
                uA_x = BoundA( t.reshape( { t.size( 0 ), t.size( 1 ), -1 } ) );
            }
        }
    }

    outputA_matrices.clear();
    outputA_matrices.append( Pair<BoundA, BoundA>( lA_x, uA_x ) );

    lbias = lA_bias_contrib;
    ubias = uA_bias_contrib;
}

BoundA BoundedConvNode::boundOneSide( const BoundA &last_A, const torch::Tensor &weight,
                                      const torch::Tensor &bias, torch::Tensor &sum_bias )
{
    if ( !last_A.defined() )
    {
        auto options = torch::TensorOptions().dtype( torch::kFloat32 ).device( _device );
        sum_bias = torch::zeros( { 1 }, options );
        return BoundA();
    }

    if ( last_A.isTensor() )
    {
        torch::Tensor last_A_tensor = last_A.asTensor();

        std::vector<int> output_padding = computeOutputPadding( input_shape, output_shape, weight );

        // Catch invalid inferred padding before LibTorch reports a shape error
        if ( conv_dim == 2 && output_padding.size() == 2 &&
             ( output_padding[0] >= std::max( stride[0], dilation[0] ) ||
               output_padding[1] >= std::max( stride[1], dilation[1] ) ) )
        {
            std::ostringstream message;
            message << "BoundedConvNode " << _nodeName.ascii()
                    << " computed invalid output_padding=[" << output_padding[0] << ","
                    << output_padding[1] << "] input_shape=[";
            for ( std::size_t i = 0; i < input_shape.size(); ++i )
            {
                message << ( i ? "," : "" ) << input_shape[i];
            }
            message << "] output_shape=[";
            for ( std::size_t i = 0; i < output_shape.size(); ++i )
            {
                message << ( i ? "," : "" ) << output_shape[i];
            }
            message << "] kernel=[" << weight.size( 2 ) << "," << weight.size( 3 ) << "] stride=["
                    << stride[0] << "," << stride[1] << "] padding=[" << padding[0] << ","
                    << padding[1] << "] dilation=[" << dilation[0] << "," << dilation[1] << "]";
            throw std::runtime_error( message.str() );
        }

        auto shape = last_A_tensor.sizes().vec();

        torch::Tensor reshaped_last_A;
        bool was_flat = false;

        if ( conv_dim == 1 )
        {
            if ( shape.size() == 4 )
            {
                reshaped_last_A =
                    last_A_tensor.reshape( { shape[0] * shape[1], shape[2], shape[3] } );
            }
            else if ( shape.size() == 3 && output_shape.size() >= 3 )
            {
                reshaped_last_A = last_A_tensor.reshape(
                    { shape[0] * shape[1], output_shape[1], output_shape[2] } );
                was_flat = true;
            }
            else if ( shape.size() == 2 && output_shape.size() >= 3 )
            {
                reshaped_last_A =
                    last_A_tensor.reshape( { shape[0], output_shape[1], output_shape[2] } );
                was_flat = true;
            }
            else
            {
                reshaped_last_A = last_A_tensor;
            }
        }
        else
        {
            if ( shape.size() == 5 )
            {
                reshaped_last_A =
                    last_A_tensor.reshape( { shape[0] * shape[1], shape[2], shape[3], shape[4] } );
            }
            else if ( shape.size() == 3 && output_shape.size() >= 4 )
            {
                reshaped_last_A = last_A_tensor.reshape(
                    { shape[0] * shape[1], output_shape[1], output_shape[2], output_shape[3] } );
                was_flat = true;
            }
            else if ( shape.size() == 2 && output_shape.size() >= 4 )
            {
                reshaped_last_A = last_A_tensor.reshape(
                    { shape[0], output_shape[1], output_shape[2], output_shape[3] } );
                was_flat = true;
            }
            else
            {
                reshaped_last_A = last_A_tensor;
            }
        }

        std::vector<int64_t> output_padding_64( output_padding.begin(), output_padding.end() );

        torch::Tensor next_A;
        if ( conv_dim == 1 )
        {
            next_A = torch::nn::functional::conv_transpose1d(
                reshaped_last_A, weight,
                torch::nn::functional::ConvTranspose1dFuncOptions()
                    .stride( stride_64 )
                    .padding( padding_64 )
                    .dilation( dilation_64 )
                    .groups( groups )
                    .output_padding( output_padding_64 ) );
        }
        else
        {
            next_A = torch::nn::functional::conv_transpose2d(
                reshaped_last_A, weight,
                torch::nn::functional::ConvTranspose2dFuncOptions()
                    .stride( stride_64 )
                    .padding( padding_64 )
                    .dilation( dilation_64 )
                    .groups( groups )
                    .output_padding( output_padding_64 ) );
        }

        if ( conv_dim == 1 )
        {
            if ( shape.size() == 4 )
            {
                next_A = next_A.view( { shape[0], shape[1], next_A.size( 1 ), next_A.size( 2 ) } );
            }
            else if ( shape.size() == 3 && output_shape.size() >= 3 )
            {
                next_A = next_A.view( { shape[0], shape[1], next_A.size( 1 ), next_A.size( 2 ) } );
            }
            else if ( shape.size() == 2 && output_shape.size() >= 3 )
            {
                next_A = next_A.view( { shape[0], next_A.size( 1 ), next_A.size( 2 ) } );
            }
        }
        else
        {
            if ( shape.size() == 5 )
            {
                next_A = next_A.view(
                    { shape[0], shape[1], next_A.size( 1 ), next_A.size( 2 ), next_A.size( 3 ) } );
            }
            else if ( shape.size() == 3 && output_shape.size() >= 4 )
            {
                next_A = next_A.view(
                    { shape[0], shape[1], next_A.size( 1 ), next_A.size( 2 ), next_A.size( 3 ) } );
            }
            else if ( shape.size() == 2 && output_shape.size() >= 4 )
            {
                next_A = next_A.view(
                    { shape[0], next_A.size( 1 ), next_A.size( 2 ), next_A.size( 3 ) } );
            }
        }

        if ( has_bias && bias.defined() )
        {
            if ( conv_dim == 1 )
            {
                if ( shape.size() == 4 )
                {
                    sum_bias = torch::einsum( "sbcl,c->sb", { last_A_tensor, bias } );
                }
                else if ( shape.size() == 3 )
                {
                    if ( was_flat && output_shape.size() >= 3 )
                    {
                        auto reshaped = last_A_tensor.reshape(
                            { shape[0], shape[1], output_shape[1], output_shape[2] } );
                        sum_bias = torch::einsum( "sbcl,c->sb", { reshaped, bias } );
                    }
                    else
                    {
                        sum_bias = torch::einsum( "bcl,c->b", { last_A_tensor, bias } );
                    }
                }
                else if ( shape.size() == 2 && output_shape.size() >= 3 )
                {
                    auto reshaped =
                        last_A_tensor.reshape( { shape[0], output_shape[1], output_shape[2] } );
                    sum_bias = torch::einsum( "bcl,c->b", { reshaped, bias } );
                }
                else
                {
                    sum_bias = torch::zeros( { 1 }, last_A_tensor.options() );
                }
            }
            else
            {
                if ( shape.size() == 5 )
                {
                    sum_bias = torch::einsum( "sbchw,c->sb", { last_A_tensor, bias } );
                }
                else if ( shape.size() == 4 )
                {
                    sum_bias = torch::einsum( "bchw,c->b", { last_A_tensor, bias } );
                }
                else if ( shape.size() == 3 && output_shape.size() >= 4 )
                {
                    auto reshaped = last_A_tensor.reshape(
                        { shape[0], shape[1], output_shape[1], output_shape[2], output_shape[3] } );
                    sum_bias = torch::einsum( "sbchw,c->sb", { reshaped, bias } );
                }
                else if ( shape.size() == 2 && output_shape.size() >= 4 )
                {
                    auto reshaped = last_A_tensor.reshape(
                        { shape[0], output_shape[1], output_shape[2], output_shape[3] } );
                    sum_bias = torch::einsum( "bchw,c->b", { reshaped, bias } );
                }
                else
                {
                    sum_bias = torch::zeros( { 1 }, last_A_tensor.options() );
                }
            }
        }
        else
        {
            if ( shape.size() >= 3 )
            {
                sum_bias = torch::zeros( { shape[0], shape[1] }, last_A_tensor.options() );
            }
            else if ( shape.size() == 2 )
            {
                sum_bias = torch::zeros( { 1, shape[0] }, last_A_tensor.options() );
            }
            else
            {
                sum_bias = torch::zeros( { 1 }, last_A_tensor.options() );
            }
        }

        return BoundA( next_A );
    }
    else
    {
        auto last_patches = last_A.asPatches();

        torch::Tensor pieces;

        if ( last_patches->identity == 0 )
        {
            torch::Tensor patches_tensor;
            bool is_sparse = last_patches->unstable_idx.has_value();

            if ( !relu_followed )
            {
                std::vector<int64_t> output_shape_vec;
                for ( int s : output_shape )
                    output_shape_vec.push_back( s );

                torch::Tensor mask = create_valid_mask(
                    output_shape_vec, last_patches->patches.device(), weight.scalar_type(),
                    { last_patches->patches.size( -2 ), last_patches->patches.size( -1 ) },
                    last_patches->stride, last_patches->inserted_zeros, last_patches->padding,
                    last_patches->output_padding, last_patches->unstable_idx );
                patches_tensor = last_patches->patches * mask;
            }
            else
            {
                patches_tensor = last_patches->patches;
            }

            if ( has_bias && bias.defined() )
            {
                if ( is_sparse )
                {
                    torch::Tensor bias_view = bias.view( { 1, 1, -1, 1, 1 } );
                    sum_bias = ( patches_tensor * bias_view ).sum( { -3, -2, -1 } );
                }
                else
                {
                    torch::Tensor bias_view = bias.view( { 1, 1, 1, 1, -1, 1, 1 } );
                    sum_bias = ( patches_tensor * bias_view ).sum( { -3, -2, -1 } );
                }
            }
            else
            {
                sum_bias = torch::zeros( { 1 }, patches_tensor.options() );
            }

            int64_t C = patches_tensor.size( -3 );
            int64_t H = patches_tensor.size( -2 );
            int64_t W = patches_tensor.size( -1 );

            torch::Tensor flattened = patches_tensor.reshape( { -1, C, H, W } );

            torch::Tensor weight_processed = insert_zeros( weight, last_patches->inserted_zeros );

            std::vector<int64_t> stride_64( stride.begin(), stride.end() );
            pieces = torch::nn::functional::conv_transpose2d(
                flattened, weight_processed,
                torch::nn::functional::ConvTranspose2dFuncOptions().stride( stride_64 ) );

            std::vector<int64_t> new_shape;
            for ( int i = 0; i < patches_tensor.dim() - 3; ++i )
            {
                new_shape.push_back( patches_tensor.size( i ) );
            }
            new_shape.push_back( pieces.size( -3 ) );
            new_shape.push_back( pieces.size( -2 ) );
            new_shape.push_back( pieces.size( -1 ) );

            pieces = pieces.view( new_shape );
        }
        else if ( last_patches->identity == 1 )
        {
            // Identity patches: directly use weight as patches (key efficiency win)
            int64_t batch_size = last_patches->output_shape[0];
            int64_t out_h = last_patches->output_shape[2];
            int64_t out_w = last_patches->output_shape[3];

            if ( last_patches->unstable_idx.has_value() )
            {
                const auto &unstable_idx = last_patches->unstable_idx.value();

                pieces = weight.view(
                    { weight.size( 0 ), 1, weight.size( 1 ), weight.size( 2 ), weight.size( 3 ) } );
                pieces = pieces.index_select( 0, unstable_idx[0] );
                pieces = pieces.expand( { -1, batch_size, -1, -1, -1 } );

                if ( has_bias && bias.defined() )
                {
                    sum_bias = bias.index_select( 0, unstable_idx[0] ).unsqueeze( -1 );
                    sum_bias = sum_bias.expand( { -1, batch_size } );
                }
                else
                {
                    sum_bias = torch::zeros( { 1 }, weight.options() );
                }
            }
            else
            {
                pieces = weight.view( { weight.size( 0 ), 1, 1, 1, weight.size( 1 ),
                                        weight.size( 2 ), weight.size( 3 ) } );
                std::vector<int64_t> expand_dims = {
                    weight.size( 0 ), batch_size,       out_h,           out_w,
                    weight.size( 1 ), weight.size( 2 ), weight.size( 3 ) };
                pieces = pieces.expand( expand_dims );

                if ( has_bias && bias.defined() )
                {
                    sum_bias = bias.view( { -1, 1, 1, 1 } )
                                   .expand( { weight.size( 0 ), batch_size, out_h, out_w } );
                }
                else
                {
                    sum_bias = torch::zeros( { 1 }, weight.options() );
                }
            }
        }

        std::vector<int64_t> new_padding_vec, new_stride_vec, new_output_padding_vec;

        std::vector<int64_t> p_pad = last_patches->padding;
        std::vector<int64_t> p_str = last_patches->stride;
        std::vector<int64_t> o_pad = {
            static_cast<int64_t>( padding[0] ), static_cast<int64_t>( padding[1] ),
            static_cast<int64_t>( padding[0] ), static_cast<int64_t>( padding[1] ) };
        std::vector<int64_t> o_str = { static_cast<int64_t>( stride[0] ),
                                       static_cast<int64_t>( stride[1] ) };
        std::vector<int64_t> out_pad_prev = last_patches->output_padding;
        std::vector<int64_t> in_shape_vec;
        for ( int s : input_shape )
            in_shape_vec.push_back( s );

        compute_patches_stride_padding( in_shape_vec, p_pad, p_str, o_pad, o_str,
                                        last_patches->inserted_zeros, out_pad_prev, new_padding_vec,
                                        new_stride_vec, new_output_padding_vec );

        // Patches too large - convert to dense matrix for efficiency
        if ( last_patches->inserted_zeros == 0 && !is_shape_used( new_output_padding_vec ) &&
             pieces.size( -1 ) > static_cast<int64_t>( input_shape[3] ) )
        {

            std::vector<int64_t> in_shape_64;
            for ( int s : input_shape )
                in_shape_64.push_back( static_cast<int64_t>( s ) );
            std::vector<int64_t> out_shape_64;
            for ( int64_t s : last_patches->output_shape )
                out_shape_64.push_back( s );

            torch::Tensor A_matrix =
                Patches::patches_to_matrix( pieces, in_shape_64, new_stride_vec, new_padding_vec,
                                            out_shape_64, last_patches->unstable_idx, 0 );

            if ( sum_bias.defined() && !last_patches->unstable_idx.has_value() )
            {
                sum_bias = sum_bias.permute( { 1, 0, 2, 3 } );
                sum_bias = sum_bias.reshape( { sum_bias.size( 0 ), -1 } );
                sum_bias = sum_bias.transpose( 0, 1 );
            }

            A_matrix = A_matrix.transpose( 0, 1 );

            return BoundA( A_matrix );
        }

        std::vector<int64_t> in_shape_64;
        for ( int s : input_shape )
            in_shape_64.push_back( static_cast<int64_t>( s ) );

        return BoundA( last_patches->create_similar( pieces, new_stride_vec, new_padding_vec,
                                                     new_output_padding_vec, 0, 0, in_shape_64 ) );
    }
}

std::vector<int> BoundedConvNode::computeOutputPadding( const std::vector<int> &input_shape,
                                                        const std::vector<int> &output_shape,
                                                        const torch::Tensor &weight ) const
{

    if ( conv_dim == 1 )
    {
        int kernel_l = weight.size( 2 );
        int needed_l = input_shape[2];
        int current_l = ( output_shape[2] - 1 ) * stride[0] - 2 * padding[0] +
                        dilation[0] * ( kernel_l - 1 ) + 1;
        int output_padding0 = needed_l - current_l;

        if ( output_padding0 < 0 )
        {
            output_padding0 = 0;
        }

        return { output_padding0 };
    }
    else
    {
        int kernel_h = weight.size( 2 );
        int kernel_w = weight.size( 3 );

        int needed_h = input_shape[2];
        int current_h = ( output_shape[2] - 1 ) * stride[0] - 2 * padding[0] +
                        dilation[0] * ( kernel_h - 1 ) + 1;
        int output_padding0 = needed_h - current_h;

        int needed_w = input_shape[3];
        int current_w = ( output_shape[3] - 1 ) * stride[1] - 2 * padding[1] +
                        dilation[1] * ( kernel_w - 1 ) + 1;
        int output_padding1 = needed_w - current_w;

        if ( output_padding0 < 0 )
        {
            output_padding0 = 0;
        }

        if ( output_padding1 < 0 )
        {
            output_padding1 = 0;
        }

        return { output_padding0, output_padding1 };
    }
}

BoundedTensor<torch::Tensor> BoundedConvNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>> &inputBounds )
{

    if ( inputBounds.empty() )
    {
        throw std::runtime_error( "No input bounds provided for IBP" );
    }

    const BoundedTensor<torch::Tensor> &input = inputBounds[0];

    const auto device = input.lower().defined() ? input.lower().device() : _device;
    ensureWeightsOnDevice( device );
    const torch::Tensor &weight = _cached_weight;
    const torch::Tensor &bias = _cached_bias;

    torch::Tensor input_lower = input.lower();
    torch::Tensor input_upper = input.upper();

    if ( input_lower.dim() == 1 || input_lower.dim() == 2 )
    {
        // Flat bounds are one row per lane, a 1-D bound is a single lane
        const int64_t lanes = input_lower.dim() == 2 ? input_lower.size( 0 ) : 1;
        if ( !input_shape.empty() && input_shape.size() >= 3 )
        {
            // input_shape is one sample behind a leading batch of 1
            std::vector<int64_t> laneShape = { lanes };
            laneShape.insert( laneShape.end(), input_shape.begin() + 1, input_shape.end() );
            input_lower = input_lower.reshape( laneShape );
            input_upper = input_upper.reshape( laneShape );
        }
        else if ( conv_dim == 1 )
        {
            int in_channels = weight.size( 1 ) * groups;
            int total_elements = input_lower.numel();

            if ( total_elements % in_channels == 0 )
            {
                int L = total_elements / in_channels;
                input_lower = input_lower.reshape( { in_channels, L } );
                input_upper = input_upper.reshape( { in_channels, L } );
            }
            else
            {
                throw std::runtime_error( "Input size incompatible with Conv1d weight dimensions" );
            }
        }
        else
        {
            int in_channels = weight.size( 1 ) * groups;
            int total_elements = input_lower.numel();

            if ( total_elements == 3072 && in_channels == 3 )
            {
                input_lower = input_lower.reshape( { 3, 32, 32 } );
                input_upper = input_upper.reshape( { 3, 32, 32 } );
            }
            else if ( total_elements % in_channels == 0 )
            {
                int spatial_size = total_elements / in_channels;
                int kernel_h = weight.size( 2 );
                int kernel_w = weight.size( 3 );
                int sqrt_spatial = static_cast<int>( std::sqrt( spatial_size ) );

                if ( sqrt_spatial * sqrt_spatial == spatial_size && sqrt_spatial >= kernel_h &&
                     sqrt_spatial >= kernel_w )
                {
                    input_lower =
                        input_lower.reshape( { in_channels, sqrt_spatial, sqrt_spatial } );
                    input_upper =
                        input_upper.reshape( { in_channels, sqrt_spatial, sqrt_spatial } );
                }
                else
                {
                    throw std::runtime_error(
                        "BoundedConvNode: ambiguous flat 2D input shape; explicit shape required" );
                }
            }
            else
            {
                throw std::runtime_error(
                    "Input size incompatible with convolution weight dimensions" );
            }
        }
    }

    if ( conv_dim == 1 && input_lower.dim() == 2 )
    {
        input_lower = input_lower.unsqueeze( 0 );
        input_upper = input_upper.unsqueeze( 0 );
    }
    else if ( conv_dim == 2 && input_lower.dim() == 3 )
    {
        input_lower = input_lower.unsqueeze( 0 );
        input_upper = input_upper.unsqueeze( 0 );
    }

    // Splitting a mixed-sign convolution into W+ and W- performs two large FP32
    // reductions and then subtracts them.  Their independent rounding can make
    // even a point interval miss the real-valued convolution.  Evaluate the
    // equivalent center-radius form in FP64, then round the result outward to
    // Luna's FP32 bound representation.
    torch::Tensor input_lower_64 = input_lower.to( torch::kFloat64 );
    torch::Tensor input_upper_64 = input_upper.to( torch::kFloat64 );
    torch::Tensor weight_64 = weight.to( torch::kFloat64 );
    torch::Tensor center = ( input_lower_64 + input_upper_64 ) / 2.0;
    torch::Tensor radius = torch::max( input_upper_64 - center, center - input_lower_64 );
    radius = torch::nextafter(
        radius, torch::full_like( radius, std::numeric_limits<double>::infinity() ) );

    torch::Tensor center_bound, radius_bound;

    if ( conv_dim == 1 )
    {
        auto options = torch::nn::functional::Conv1dFuncOptions()
                           .stride( stride_64 )
                           .padding( padding_64 )
                           .dilation( dilation_64 )
                           .groups( groups );
        center_bound = torch::nn::functional::conv1d( center, weight_64, options );
        radius_bound = torch::nn::functional::conv1d( radius, weight_64.abs(), options );
    }
    else
    {
        auto options = torch::nn::functional::Conv2dFuncOptions()
                           .stride( stride_64 )
                           .padding( padding_64 )
                           .dilation( dilation_64 )
                           .groups( groups );
        center_bound = torch::nn::functional::conv2d( center, weight_64, options );
        radius_bound = torch::nn::functional::conv2d( radius, weight_64.abs(), options );
    }

    if ( has_bias && bias.defined() )
    {
        std::vector<int64_t> bias_shape( center_bound.dim(), 1 );
        bias_shape[1] = bias.numel();
        center_bound = center_bound + bias.to( torch::kFloat64 ).view( bias_shape );
    }

    torch::Tensor lower_bound_64 = center_bound - radius_bound;
    torch::Tensor upper_bound_64 = center_bound + radius_bound;
    torch::Tensor lower_bound = lower_bound_64.to( torch::kFloat32 );
    torch::Tensor upper_bound = upper_bound_64.to( torch::kFloat32 );
    lower_bound = torch::nextafter(
        lower_bound, torch::full_like( lower_bound, -std::numeric_limits<float>::infinity() ) );
    upper_bound = torch::nextafter(
        upper_bound, torch::full_like( upper_bound, std::numeric_limits<float>::infinity() ) );

    if ( input.lower().dim() == 1 || ( input.lower().dim() == 2 && input.lower().size( 0 ) == 1 ) )
    {
        lower_bound = lower_bound.flatten();
        upper_bound = upper_bound.flatten();
    }

    return BoundedTensor<torch::Tensor>( lower_bound, upper_bound );
}

unsigned BoundedConvNode::getInputSize() const
{
    return _input_size;
}

unsigned BoundedConvNode::getOutputSize() const
{
    return _output_size;
}

void BoundedConvNode::setInputSize( unsigned size )
{
    _input_size = size;
}

void BoundedConvNode::setOutputSize( unsigned size )
{
    _output_size = size;
}

unsigned BoundedConvNode::inferOutputSize( unsigned inputSize ) const
{
    if ( conv_dim == 1 )
    {
        if ( !conv1d )
            return 0;
        torch::Tensor weight = conv1d->weight;

        int64_t out_channels = weight.size( 0 );
        int64_t in_channels_per_group = weight.size( 1 );
        int64_t in_channels = in_channels_per_group * groups;

        if ( in_channels == 0 )
            return 0;

        int64_t L = inputSize / in_channels;
        if ( L <= 0 )
            return 0;

        int64_t kernel_length = weight.size( 2 );
        int64_t out_l =
            ( L + 2 * padding[0] - dilation[0] * ( kernel_length - 1 ) - 1 ) / stride[0] + 1;

        return static_cast<unsigned>( out_channels * out_l );
    }
    else
    {
        if ( !conv2d )
            return 0;
        torch::Tensor weight = conv2d->weight;

        int64_t out_channels = weight.size( 0 );
        int64_t in_channels_per_group = weight.size( 1 );
        int64_t in_channels = in_channels_per_group * groups;

        if ( in_channels == 0 )
            return 0;

        int64_t spatial_dim_sq = inputSize / in_channels;
        if ( spatial_dim_sq <= 0 )
            return 0;

        int64_t H = static_cast<int64_t>( std::sqrt( spatial_dim_sq ) );
        int64_t W = H;

        std::vector<int> kernel_size = { static_cast<int>( weight.size( 2 ) ),
                                         static_cast<int>( weight.size( 3 ) ) };

        std::vector<int> spatial_output = MatrixConvolution::computeConvOutputShape(
            { static_cast<int>( H ), static_cast<int>( W ) }, kernel_size, stride, padding,
            dilation );

        if ( spatial_output.size() < 2 )
            return 0;

        return static_cast<unsigned>( out_channels * spatial_output[0] * spatial_output[1] );
    }
}

} // namespace NLR
