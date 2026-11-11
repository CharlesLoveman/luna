/*********************                                                        */
/*! \file OnnxToTorch.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "OnnxToTorch.h"
#include "../engine/TorchModel.h"
#include "../engine/nodes/BoundedTorchNode.h"
#include "../engine/nodes/BoundedConstantNode.h"
#include "../engine/nodes/BoundedInputNode.h"
#include "../engine/nodes/BoundedLinearNode.h"
#include "../engine/nodes/BoundedReLUNode.h"
#include "../engine/nodes/BoundedSigmoidNode.h"
#include "../engine/nodes/BoundedIdentityNode.h"
#include "../engine/nodes/BoundedReshapeNode.h"
#include "../engine/nodes/BoundedSubNode.h"
#include "../engine/nodes/BoundedAddNode.h"
#include "../engine/nodes/BoundedConvNode.h"
#include "../engine/nodes/BoundedConvTransposeNode.h"
#include "../engine/nodes/BoundedConcatNode.h"
#include "../engine/nodes/BoundedSliceNode.h"
#include "../engine/LunaError.h"
#include "BoundedSoftmaxNode.h"
#include "File.h"
#include "MString.h"
#include "Vector.h"
#include "Map.h"
#include "Set.h"
#include <fstream>
#include <memory>
#include <cstdlib>
#include <cstdio>
#include <torch/nn/options/activation.h>
#include <torch/torch.h>
#include <onnx.proto3.pb.h>

#include "TensorUtils.h"

#include <math.h>

#include "Debug.h"

void onnxToTorchMissingAttributeError( const luna_onnx::NodeProto &node,
                                       const String &attributeName )
{
    String errorMessage =
        Stringf( "OnnxToTorch: Onnx node of type %s is missing the expected attribute %s",
                 node.op_type().c_str(), attributeName.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchUnimplementedOperationError( const luna_onnx::NodeProto &node )
{
    String errorMessage = Stringf( "OnnxToTorch: Onnx '%s' operation not yet implemented for "
                                   "TorchModel conversion. Should be relatively easy to add.",
                                   node.op_type().c_str() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchUnimplementedAttributeError( const luna_onnx::NodeProto &node,
                                             const String &attributeName )
{
    String errorMessage = Stringf( "OnnxToTorch: Onnx '%s' operation with non-default value for "
                                   "attribute '%s' not yet supported.",
                                   node.op_type().c_str(), attributeName.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchUnsupportedOperationError( const luna_onnx::NodeProto &node )
{
    String errorMessage =
        Stringf( "OnnxToTorch: Onnx operation %s not currently supported by TorchModel conversion",
                 node.op_type().c_str() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchMissingNodeError( const String &missingNodeName )
{
    String errorMessage =
        Stringf( "OnnxToTorch: Internal invariant violated: missing node '%s' not found",
                 missingNodeName.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchUnexpectedNumberOfInputs( const luna_onnx::NodeProto &node,
                                          unsigned int actualNumberOfInputs,
                                          unsigned int lowerBound, unsigned int upperBound )
{
    String errorMessage;
    if ( lowerBound == upperBound )
    {
        errorMessage =
            Stringf( "OnnxToTorch: %s node expected to have exactly %d inputs, but found %d",
                     node.op_type().c_str(), lowerBound, actualNumberOfInputs );
    }
    else
    {
        errorMessage =
            Stringf( "OnnxToTorch: %s node expected to have between %d and %d inputs, but found %d",
                     node.op_type().c_str(), lowerBound, upperBound, actualNumberOfInputs );
    }
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchInvalidTensorShapeError( const String &nodeName, const String &reason )
{
    String errorMessage = Stringf( "OnnxToTorch: Invalid tensor shape for node '%s': %s",
                                   nodeName.ascii(), reason.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchUnsupportedDataTypeError( const luna_onnx::TensorProto_DataType &dataType )
{
    String errorMessage =
        Stringf( "OnnxToTorch: Support for Onnx constants of type '%s' not yet implemented.",
                 TensorProto_DataType_Name( dataType ).c_str() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchInvalidConstantNodeError( const luna_onnx::NodeProto &node, const String &reason )
{
    String errorMessage = Stringf( "OnnxToTorch: Invalid Constant node '%s': %s",
                                   node.name().c_str(), reason.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchTopologicalSortError( const String &reason )
{
    String errorMessage = Stringf( "OnnxToTorch: Topological sort failed: %s", reason.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchBoundedModuleCreationError( const String &operationType, const String &reason )
{
    String errorMessage =
        Stringf( "OnnxToTorch: Failed to create bounded module for operation '%s': %s",
                 operationType.ascii(), reason.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchFileReadError( const String &filename, const String &reason )
{
    String errorMessage =
        Stringf( "OnnxToTorch: Failed to read file '%s': %s", filename.ascii(), reason.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchModelParseError( const String &filename, const String &reason )
{
    String errorMessage = Stringf( "OnnxToTorch: Failed to parse ONNX model from file '%s': %s",
                                   filename.ascii(), reason.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchGraphProcessingError( const String &reason )
{
    String errorMessage = Stringf( "OnnxToTorch: Graph processing failed: %s", reason.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchTensorConversionError( const String &tensorName, const String &reason )
{
    String errorMessage = Stringf( "OnnxToTorch: Failed to convert tensor '%s': %s",
                                   tensorName.ascii(), reason.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchAttributeProcessingError( const luna_onnx::NodeProto &node,
                                          const String &attributeName, const String &reason )
{
    String errorMessage =
        Stringf( "OnnxToTorch: Failed to process attribute '%s' for node '%s': %s",
                 attributeName.ascii(), node.op_type().c_str(), reason.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchShapeMismatchError( const String &operation, const TensorShape &expectedShape,
                                    const TensorShape &actualShape )
{
    String expectedStr = "[";
    for ( unsigned int i = 0; i < expectedShape.size(); ++i )
    {
        if ( i > 0 )
            expectedStr += ", ";
        expectedStr += Stringf( "%u", expectedShape[i] );
    }
    expectedStr += "]";

    String actualStr = "[";
    for ( unsigned int i = 0; i < actualShape.size(); ++i )
    {
        if ( i > 0 )
            actualStr += ", ";
        actualStr += Stringf( "%u", actualShape[i] );
    }
    actualStr += "]";

    String errorMessage =
        Stringf( "OnnxToTorch: Shape mismatch in operation '%s': expected %s, got %s",
                 operation.ascii(), expectedStr.ascii(), actualStr.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchDimensionMismatchError( const String &operation, unsigned int expectedDim,
                                        unsigned int actualDim )
{
    String errorMessage = Stringf(
        "OnnxToTorch: Dimension mismatch in operation '%s': expected %u dimensions, got %u",
        operation.ascii(), expectedDim, actualDim );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchInvalidBroadcastError( const String &operation, const TensorShape &shape1,
                                       const TensorShape &shape2 )
{
    String shape1Str = "[";
    for ( unsigned int i = 0; i < shape1.size(); ++i )
    {
        if ( i > 0 )
            shape1Str += ", ";
        shape1Str += Stringf( "%u", shape1[i] );
    }
    shape1Str += "]";

    String shape2Str = "[";
    for ( unsigned int i = 0; i < shape2.size(); ++i )
    {
        if ( i > 0 )
            shape2Str += ", ";
        shape2Str += Stringf( "%u", shape2[i] );
    }
    shape2Str += "]";

    String errorMessage = Stringf(
        "OnnxToTorch: Invalid broadcast in operation '%s': cannot broadcast shapes %s and %s",
        operation.ascii(), shape1Str.ascii(), shape2Str.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchUnsupportedActivationError( const String &activationType )
{
    String errorMessage =
        Stringf( "OnnxToTorch: Unsupported activation function '%s'", activationType.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchInvalidWeightBiasError( const String &operation, const String &reason )
{
    String errorMessage =
        Stringf( "OnnxToTorch: Invalid weight/bias configuration in operation '%s': %s",
                 operation.ascii(), reason.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchMemoryAllocationError( const String &operation, const String &reason )
{
    String errorMessage = Stringf( "OnnxToTorch: Memory allocation failed in operation '%s': %s",
                                   operation.ascii(), reason.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

void onnxToTorchPyTorchError( const String &operation, const String &pytorchError )
{
    String errorMessage = Stringf( "OnnxToTorch: PyTorch error in operation '%s': %s",
                                   operation.ascii(), pytorchError.ascii() );
    throw LunaError( LunaError::ONNX_PARSING_ERROR, errorMessage.ascii() );
}

using namespace Operations;
using namespace AttributeUtils;

using TensorShape = Vector<unsigned int>;

OnnxToTorchParser::OnnxToTorchParser( const String &path )
{
    std::ifstream input( path.ascii(), std::ios::ate | std::ios::binary );
    if ( !input.is_open() )
    {
        onnxToTorchFileReadError( path, "Could not open file" );
    }
    std::streamsize size = input.tellg();

    input.seekg( 0, std::ios::beg );

    std::vector<char> buffer( size );
    input.read( buffer.data(), size );
    if ( input.gcount() != size )
    {
        onnxToTorchFileReadError(
            path, Stringf( "Failed to read entire file: expected %ld bytes, got %ld", (long)size,
                           (long)input.gcount() ) );
    }

    std::string model_string( buffer.data(), size );
    if ( !_onnx_model.ParseFromString( model_string ) )
    {
        onnxToTorchModelParseError( path, "Failed to parse ONNX protobuf" );
    }
    // run ONNX shape inference when intermediate shapes are missing
    if ( _onnx_model.graph().value_info_size() == 0 )
    {
        std::string temp_path = std::string( path.ascii() ) + ".inferred.onnx";

        std::string python_cmd = "python3 -c \"import onnx; from onnx import shape_inference; "
                                 "model = onnx.load('" +
                                 std::string( path.ascii() ) +
                                 "'); "
                                 "inferred = shape_inference.infer_shapes(model); "
                                 "onnx.save(inferred, '" +
                                 temp_path + "')\" 2>/dev/null";

        int result = system( python_cmd.c_str() );
        if ( result == 0 )
        {
            std::ifstream inferred_input( temp_path, std::ios::ate | std::ios::binary );
            if ( inferred_input.is_open() )
            {
                std::streamsize inferred_size = inferred_input.tellg();
                inferred_input.seekg( 0, std::ios::beg );
                std::vector<char> inferred_buffer( inferred_size );
                inferred_input.read( inferred_buffer.data(), inferred_size );

                std::string inferred_string( inferred_buffer.data(), inferred_size );
                _onnx_model.ParseFromString( inferred_string );
                inferred_input.close();

                std::remove( temp_path.c_str() );
            }
        }
    }
}

std::shared_ptr<NLR::TorchModel> OnnxToTorchParser::parse( const String &path )
{
    OnnxToTorchParser parser( path );
    return parser.processGraph();
}

std::shared_ptr<NLR::TorchModel> OnnxToTorchParser::processGraph()
{
    Map<String, luna_onnx::TensorProto> name_to_initializer;
    for ( const auto &initializer : _onnx_model.graph().initializer() )
    {
        name_to_initializer[initializer.name()] = initializer;
    }

    Map<String, luna_onnx::ValueInfoProto> name_to_input;
    for ( const auto &input : _onnx_model.graph().input() )
    {
        name_to_input[input.name()] = input;
    }

    Map<String, luna_onnx::NodeProto> name_to_node;
    for ( const auto &node : _onnx_model.graph().node() )
    {
        for ( int i = 0; i < node.output_size(); ++i )
        {
            name_to_node[node.output( i )] = node;
        }
    }

    Map<String, Vector<int>> shape_metadata;
    for ( const auto &input : _onnx_model.graph().input() )
    {
        if ( input.type().tensor_type().has_shape() )
        {
            const auto &shape = input.type().tensor_type().shape();
            Vector<int> dims;
            for ( int i = 0; i < shape.dim_size(); ++i )
            {
                if ( shape.dim( i ).has_dim_value() )
                {
                    dims.append( static_cast<int>( shape.dim( i ).dim_value() ) );
                }
                else
                {
                    dims.append( 1 );
                }
            }
            if ( !dims.empty() )
            {
                shape_metadata[input.name()] = dims;
            }
        }
    }

    for ( const auto &output : _onnx_model.graph().output() )
    {
        if ( output.type().tensor_type().has_shape() )
        {
            const auto &shape = output.type().tensor_type().shape();
            Vector<int> dims;
            for ( int i = 0; i < shape.dim_size(); ++i )
            {
                if ( shape.dim( i ).has_dim_value() )
                {
                    dims.append( static_cast<int>( shape.dim( i ).dim_value() ) );
                }
                else
                {
                    dims.append( -1 );
                }
            }
            if ( !dims.empty() )
            {
                shape_metadata[output.name()] = dims;
            }
        }
    }

    // multi-pass shape inference for cross-op dependencies
    for ( int pass = 0; pass < 3; ++pass )
    {
        int shapes_added = 0;

        for ( const auto &node : _onnx_model.graph().node() )
        {
            if ( node.output_size() > 0 )
            {
                String outputName = node.output( 0 );

                if ( shape_metadata.exists( outputName ) )
                    continue;

                Vector<int> output_shape;

                if ( node.op_type() == "MatMul" || node.op_type() == "Gemm" )
                {
                    if ( node.input_size() >= 2 )
                    {
                        String input0Name = node.input( 0 );
                        String input1Name = node.input( 1 );

                        if ( name_to_initializer.exists( input1Name ) )
                        {
                            Vector<int> input0_shape = shape_metadata.exists( input0Name )
                                                           ? shape_metadata[input0Name]
                                                           : Vector<int>();
                            const auto &weight_tensor = name_to_initializer[input1Name];
                            if ( weight_tensor.dims_size() == 2 && !input0_shape.empty() )
                            {
                                int batch = input0_shape[0];
                                int N = weight_tensor.dims( 1 );
                                output_shape.append( batch );
                                output_shape.append( N );
                            }
                        }
                        else if ( name_to_initializer.exists( input0Name ) )
                        {
                            Vector<int> input1_shape = shape_metadata.exists( input1Name )
                                                           ? shape_metadata[input1Name]
                                                           : Vector<int>();
                            const auto &weight_tensor = name_to_initializer[input0Name];
                            if ( weight_tensor.dims_size() == 2 )
                            {
                                int M = weight_tensor.dims( 0 );
                                if ( !input1_shape.empty() )
                                {
                                    int batch = input1_shape[0];
                                    output_shape.append( batch );
                                }
                                output_shape.append( M );
                            }
                        }
                    }
                }
                else if ( node.op_type() == "Relu" || node.op_type() == "Sigmoid" ||
                          node.op_type() == "Softmax" || node.op_type() == "Dropout" ||
                          node.op_type() == "BatchNormalization" )
                {
                    if ( node.input_size() > 0 && shape_metadata.exists( node.input( 0 ) ) )
                    {
                        output_shape = shape_metadata[node.input( 0 )];
                    }
                }
                else if ( node.op_type() == "Conv" )
                {
                    if ( node.input_size() >= 2 && shape_metadata.exists( node.input( 0 ) ) )
                    {
                        Vector<int> input_shape = shape_metadata[node.input( 0 )];
                        String weight_name = node.input( 1 );

                        if ( name_to_initializer.exists( weight_name ) && input_shape.size() >= 3 )
                        {
                            const auto &weight_tensor = name_to_initializer[weight_name];
                            int weight_dims = weight_tensor.dims_size();

                            std::vector<int64_t> strides = { 1, 1 };
                            std::vector<int64_t> pads = { 0, 0, 0, 0 };
                            std::vector<int64_t> dilations = { 1, 1 };

                            for ( const auto &attr : node.attribute() )
                            {
                                if ( attr.name() == "strides" )
                                {
                                    strides.clear();
                                    for ( int k = 0; k < attr.ints_size(); ++k )
                                        strides.push_back( attr.ints( k ) );
                                }
                                else if ( attr.name() == "pads" )
                                {
                                    pads.clear();
                                    for ( int k = 0; k < attr.ints_size(); ++k )
                                        pads.push_back( attr.ints( k ) );
                                }
                                else if ( attr.name() == "dilations" )
                                {
                                    dilations.clear();
                                    for ( int k = 0; k < attr.ints_size(); ++k )
                                        dilations.push_back( attr.ints( k ) );
                                }
                            }

                            int out_channels = weight_tensor.dims( 0 );

                            if ( weight_dims == 3 && input_shape.size() >= 3 )
                            {
                                int N = input_shape[0] > 0 ? input_shape[0] : 1;
                                int L = input_shape[2];
                                int k_l = weight_tensor.dims( 2 );
                                int s = strides.size() > 0 ? strides[0] : 1;
                                int p = pads.size() > 0 ? pads[0] : 0;
                                int d = dilations.size() > 0 ? dilations[0] : 1;
                                int out_l = ( L + 2 * p - d * ( k_l - 1 ) - 1 ) / s + 1;
                                output_shape.append( N );
                                output_shape.append( out_channels );
                                output_shape.append( out_l );
                            }
                            else if ( weight_dims == 4 && input_shape.size() >= 4 )
                            {
                                int N = input_shape[0] > 0 ? input_shape[0] : 1;
                                int H = input_shape[2];
                                int W = input_shape[3];
                                int k_h = weight_tensor.dims( 2 );
                                int k_w = weight_tensor.dims( 3 );
                                int s_h = strides.size() > 0 ? strides[0] : 1;
                                int s_w = strides.size() > 1 ? strides[1] : s_h;
                                int p_h = pads.size() > 0 ? pads[0] : 0;
                                int p_w = pads.size() > 1 ? pads[1] : p_h;
                                int d_h = dilations.size() > 0 ? dilations[0] : 1;
                                int d_w = dilations.size() > 1 ? dilations[1] : d_h;
                                int out_h = ( H + 2 * p_h - d_h * ( k_h - 1 ) - 1 ) / s_h + 1;
                                int out_w = ( W + 2 * p_w - d_w * ( k_w - 1 ) - 1 ) / s_w + 1;
                                output_shape.append( N );
                                output_shape.append( out_channels );
                                output_shape.append( out_h );
                                output_shape.append( out_w );
                            }
                        }
                    }
                }
                else if ( node.op_type() == "Flatten" )
                {
                    if ( node.input_size() > 0 && shape_metadata.exists( node.input( 0 ) ) )
                    {
                        Vector<int> input_shape = shape_metadata[node.input( 0 )];
                        if ( !input_shape.empty() )
                        {
                            int N = input_shape[0] > 0 ? input_shape[0] : 1;
                            int flat_size = 1;
                            for ( unsigned k = 1; k < input_shape.size(); ++k )
                            {
                                if ( input_shape[k] > 0 )
                                    flat_size *= input_shape[k];
                            }
                            output_shape.append( N );
                            output_shape.append( flat_size );
                        }
                    }
                }
                if ( !output_shape.empty() )
                {
                    shape_metadata[outputName] = output_shape;
                    shapes_added++;
                }
            }
        }

        if ( shapes_added == 0 )
            break;
    }

    for ( const auto &value_info : _onnx_model.graph().value_info() )
    {
        if ( value_info.type().tensor_type().has_shape() )
        {
            if ( shape_metadata.exists( value_info.name() ) )
                continue;

            const auto &shape = value_info.type().tensor_type().shape();
            Vector<int> dims;
            for ( int i = 0; i < shape.dim_size(); ++i )
            {
                if ( shape.dim( i ).has_dim_value() )
                {
                    dims.append( static_cast<int>( shape.dim( i ).dim_value() ) );
                }
                else
                {
                    dims.append( 1 );
                }
            }
            if ( !dims.empty() )
            {
                shape_metadata[value_info.name()] = dims;
            }
        }
    }

    Vector<String> processingOrder;
    Set<String> seenNames;

    // skip inputs that are also initializers (weights listed as inputs)
    for ( const auto &input : _onnx_model.graph().input() )
    {
        String inputName = input.name();
        if ( !name_to_initializer.exists( inputName ) )
        {
            processingOrder.append( inputName );
            seenNames.insert( inputName );
        }
    }

    for ( const auto &initializer : _onnx_model.graph().initializer() )
    {
        String initName = initializer.name();
        if ( !seenNames.exists( initName ) )
        {
            processingOrder.append( initName );
            seenNames.insert( initName );
        }
    }

    for ( const auto &node : _onnx_model.graph().node() )
    {
        for ( int i = 0; i < node.output_size(); ++i )
        {
            String outputName = node.output( i );
            if ( !seenNames.exists( outputName ) )
            {
                processingOrder.append( outputName );
                seenNames.insert( outputName );
            }
        }
    }

    Vector<std::shared_ptr<NLR::BoundedTorchNode>> nodes;
    Vector<unsigned> inputIndices;
    unsigned outputIndex = 0;

    String expectedOutputName;
    if ( _onnx_model.graph().output_size() > 0 )
    {
        expectedOutputName = _onnx_model.graph().output( 0 ).name();
    }

    Map<String, torch::Tensor> constantsMap;
    for ( const auto &node : _onnx_model.graph().node() )
    {
        for ( int i = 0; i < node.input_size(); ++i )
        {
            String inputName = node.input( i );
            if ( name_to_initializer.exists( inputName ) && !constantsMap.exists( inputName ) )
            {
                torch::Tensor constant =
                    ConstantProcessor::processInitializer( name_to_initializer[inputName] );
                constantsMap[inputName] = constant;
            }
            if ( name_to_node.exists( inputName ) && !constantsMap.exists( inputName ) )
            {
                const auto &producer = name_to_node[inputName];
                if ( producer.op_type() == "Constant" )
                {
                    torch::Tensor constant = ConstantProcessor::processConstantNode( producer );
                    constantsMap[inputName] = constant;
                }
            }
        }
    }

    Map<String, unsigned> nameToIndex;
    for ( unsigned i = 0; i < processingOrder.size(); ++i )
    {
        nameToIndex[processingOrder[i]] = i;
    }

    Map<unsigned, Vector<unsigned>> dependencies;

    for ( unsigned i = 0; i < processingOrder.size(); ++i )
    {
        const String &tensorName = processingOrder[i];
        if ( name_to_initializer.exists( tensorName ) )
        {
            torch::Tensor constant =
                ConstantProcessor::processInitializer( name_to_initializer[tensorName] );
            auto constantNode = std::make_shared<NLR::BoundedConstantNode>( constant, tensorName );
            constantNode->setNodeIndex( i );
            nodes.append( constantNode );
            continue;
        }

        if ( name_to_node.exists( tensorName ) )
        {
            const auto &node = name_to_node[tensorName];
            if ( node.op_type() == "Constant" )
            {
                try
                {
                    torch::Tensor constant = ConstantProcessor::processConstantNode( node );
                    auto constantNode =
                        std::make_shared<NLR::BoundedConstantNode>( constant, tensorName );
                    constantNode->setNodeIndex( i );
                    nodes.append( constantNode );
                }
                catch ( const LunaError &e )
                {
                    throw;
                }
                catch ( const std::exception &e )
                {
                    onnxToTorchInvalidConstantNodeError( node, e.what() );
                }
                continue;
            }
        }

        if ( name_to_input.exists( tensorName ) && !name_to_initializer.exists( tensorName ) )
        {
            inputIndices.append( i );

            unsigned inputSize = 1;
            TensorShape inputShape = BoundedOperationConverter::extractShapeFromNode(
                luna_onnx::NodeProto(), name_to_input, name_to_initializer, tensorName );
            if ( !inputShape.empty() )
            {
                inputSize = BoundedOperationConverter::computeTensorSize( inputShape );
            }
            auto inputNode = std::make_shared<NLR::BoundedInputNode>( i, inputSize, tensorName );
            std::vector<int64_t> sampleShape;
            for ( unsigned dim : inputShape )
            {
                sampleShape.push_back( static_cast<int64_t>( dim ) );
            }
            // Symbolic dims are already dropped, so a leading 1 on a rank 2+ input is the batch
            if ( sampleShape.size() >= 2 && sampleShape.front() == 1 )
            {
                sampleShape.erase( sampleShape.begin() );
            }
            if ( !sampleShape.empty() )
            {
                inputNode->setSampleShape( sampleShape );
            }
            inputNode->setNodeIndex( i );
            nodes.append( inputNode );
            continue;
        }

        if ( name_to_node.exists( tensorName ) )
        {
            const auto &node = name_to_node[tensorName];

            Vector<unsigned> deps;
            if ( node.op_type() == "Sub" && node.input_size() == 2 )
            {
                String input2Name = node.input( 1 );
                bool isSecondInputConstant = ( constantsMap.exists( input2Name ) ||
                                               name_to_initializer.exists( input2Name ) );

                if ( isSecondInputConstant )
                {
                    String input1Name = node.input( 0 );
                    if ( nameToIndex.exists( input1Name ) )
                    {
                        unsigned inputIndex = nameToIndex[input1Name];
                        deps.append( inputIndex );
                    }
                }
                else
                {
                    for ( int j = 0; j < node.input_size(); ++j )
                    {
                        String inputName = node.input( j );
                        if ( nameToIndex.exists( inputName ) )
                        {
                            unsigned inputIndex = nameToIndex[inputName];
                            deps.append( inputIndex );
                        }
                    }
                }
            }
            else if ( node.op_type() == "Add" && node.input_size() == 2 )
            {
                String input1Name = node.input( 0 );
                String input2Name = node.input( 1 );
                bool isFirstInputConstant = ( constantsMap.exists( input1Name ) ||
                                              name_to_initializer.exists( input1Name ) );
                bool isSecondInputConstant = ( constantsMap.exists( input2Name ) ||
                                               name_to_initializer.exists( input2Name ) );

                if ( isFirstInputConstant && !isSecondInputConstant )
                {
                    if ( nameToIndex.exists( input2Name ) )
                    {
                        unsigned inputIndex = nameToIndex[input2Name];
                        deps.append( inputIndex );
                    }
                }
                else if ( isSecondInputConstant && !isFirstInputConstant )
                {
                    if ( nameToIndex.exists( input1Name ) )
                    {
                        unsigned inputIndex = nameToIndex[input1Name];
                        deps.append( inputIndex );
                    }
                }
                else if ( !isFirstInputConstant && !isSecondInputConstant )
                {
                    for ( int j = 0; j < node.input_size(); ++j )
                    {
                        String inputName = node.input( j );
                        if ( nameToIndex.exists( inputName ) )
                        {
                            unsigned inputIndex = nameToIndex[inputName];
                            deps.append( inputIndex );
                        }
                    }
                }
            }
            else if ( node.op_type() == "BatchNormalization" )
            {
                // only X is a dependency; params are embedded
                if ( node.input_size() >= 1 )
                {
                    String xName = node.input( 0 );
                    if ( nameToIndex.exists( xName ) )
                    {
                        deps.append( nameToIndex[xName] );
                    }
                }
            }
            else
            {
                for ( int j = 0; j < node.input_size(); ++j )
                {
                    String inputName = node.input( j );
                    if ( nameToIndex.exists( inputName ) )
                    {
                        unsigned inputIndex = nameToIndex[inputName];
                        if ( !name_to_initializer.exists( inputName ) )
                        {
                            deps.append( inputIndex );
                        }
                    }
                }
            }
            if ( !deps.empty() )
            {
                dependencies[i] = deps;
            }
            std::shared_ptr<NLR::BoundedTorchNode> boundedNode;

            try
            {
                if ( node.op_type() == "Identity" )
                {
                    boundedNode = BoundedOperationConverter::convertIdentity( node, name_to_input,
                                                                              name_to_initializer );
                }
                else if ( node.op_type() == "Gemm" )
                {
                    boundedNode = BoundedOperationConverter::convertGemm(
                        node, constantsMap, name_to_input, name_to_initializer );
                }
                else if ( node.op_type() == "MatMul" )
                {
                    boundedNode = BoundedOperationConverter::convertMatMul(
                        node, constantsMap, name_to_input, name_to_initializer );
                }
                else if ( node.op_type() == "Add" )
                {
                    boundedNode = BoundedOperationConverter::convertAdd(
                        node, constantsMap, name_to_input, name_to_initializer, nodes,
                        nameToIndex );
                }
                else if ( node.op_type() == "Relu" || node.op_type() == "relu" )
                {
                    boundedNode = BoundedOperationConverter::convertRelu( node, name_to_input,
                                                                          name_to_initializer );
                }
                else if ( node.op_type() == "Sigmoid" || node.op_type() == "sigmoid" )
                {
                    boundedNode = BoundedOperationConverter::convertSigmoid( node, name_to_input,
                                                                             name_to_initializer );
                }
                else if ( node.op_type() == "Softmax" || node.op_type() == "softmax" )
                {
                    boundedNode = BoundedOperationConverter::convertSoftmax( node, name_to_input,
                                                                             name_to_initializer );
                }
                else if ( node.op_type() == "Reshape" )
                {
                    boundedNode = BoundedOperationConverter::convertReshape( node, name_to_input,
                                                                             name_to_initializer );
                }
                else if ( node.op_type() == "Flatten" )
                {
                    boundedNode = BoundedOperationConverter::convertFlatten( node, name_to_input,
                                                                             name_to_initializer );
                }
                else if ( node.op_type() == "Sub" )
                {
                    boundedNode = BoundedOperationConverter::convertSub(
                        node, constantsMap, name_to_input, name_to_initializer );
                }
                else if ( node.op_type() == "Conv" )
                {
                    boundedNode = BoundedOperationConverter::convertConv(
                        node, constantsMap, name_to_input, name_to_initializer, shape_metadata );
                    if ( !boundedNode )
                    {
                        onnxToTorchBoundedModuleCreationError( "Conv",
                                                               "Conversion returned nullptr" );
                    }
                }
                else if ( node.op_type() == "ConvTranspose" )
                {
                    boundedNode = BoundedOperationConverter::convertConvTranspose(
                        node, constantsMap, name_to_input, name_to_initializer );
                    if ( !boundedNode )
                    {
                        onnxToTorchBoundedModuleCreationError( "ConvTranspose",
                                                               "Conversion returned nullptr" );
                    }
                }
                else if ( node.op_type() == "BatchNormalization" )
                {
                    boundedNode = BoundedOperationConverter::convertBatchNormalization(
                        node, constantsMap, name_to_input, name_to_initializer );
                }
                else if ( node.op_type() == "Gather" )
                {
                    // conservative: treat as identity until proper bound propagation is added
                    boundedNode = BoundedOperationConverter::convertIdentity( node, name_to_input,
                                                                              name_to_initializer );
                }
                else if ( node.op_type() == "Cast" )
                {
                    // type conversion does not affect bounds
                    boundedNode = BoundedOperationConverter::convertIdentity( node, name_to_input,
                                                                              name_to_initializer );
                }
                else if ( node.op_type() == "Dropout" )
                {
                    // disabled during inference
                    boundedNode = BoundedOperationConverter::convertIdentity( node, name_to_input,
                                                                              name_to_initializer );
                }
                else if ( node.op_type() == "Concat" )
                {
                    boundedNode = BoundedOperationConverter::convertConcat(
                        node, name_to_input, name_to_initializer, shape_metadata );
                    if ( !boundedNode )
                    {
                        onnxToTorchBoundedModuleCreationError( "Concat",
                                                               "Conversion returned nullptr" );
                    }
                }
                else if ( node.op_type() == "Slice" )
                {
                    boundedNode = BoundedOperationConverter::convertSlice(
                        node, constantsMap, name_to_input, name_to_initializer, shape_metadata );
                    if ( !boundedNode )
                    {
                        onnxToTorchBoundedModuleCreationError( "Slice",
                                                               "Conversion returned nullptr" );
                    }
                }
                else
                {
                    onnxToTorchUnsupportedOperationError( node );
                }

                // must precede metadata assignment
                if ( node.output_size() > 0 )
                {
                    String outputName = node.output( 0 );
                    Vector<int> output_shape;

                    if ( node.op_type() == "MatMul" || node.op_type() == "Gemm" )
                    {
                        if ( node.input_size() >= 2 )
                        {
                            String input0Name = node.input( 0 );
                            String input1Name = node.input( 1 );

                            if ( name_to_initializer.exists( input1Name ) )
                            {
                                Vector<int> input0_shape = shape_metadata.exists( input0Name )
                                                               ? shape_metadata[input0Name]
                                                               : Vector<int>();
                                const auto &weight_tensor = name_to_initializer[input1Name];
                                if ( weight_tensor.dims_size() == 2 && !input0_shape.empty() )
                                {
                                    int batch = input0_shape[0];
                                    int N = weight_tensor.dims( 1 );
                                    output_shape.append( batch );
                                    output_shape.append( N );
                                }
                            }
                            else if ( name_to_initializer.exists( input0Name ) )
                            {
                                // Transposed: W @ X
                                Vector<int> input1_shape = shape_metadata.exists( input1Name )
                                                               ? shape_metadata[input1Name]
                                                               : Vector<int>();
                                const auto &weight_tensor = name_to_initializer[input0Name];
                                if ( weight_tensor.dims_size() == 2 )
                                {
                                    int M = weight_tensor.dims( 0 );
                                    if ( !input1_shape.empty() )
                                    {
                                        int batch = input1_shape[0];
                                        output_shape.append( batch );
                                    }
                                    output_shape.append( M );
                                }
                            }
                        }
                    }
                    else if ( node.op_type() == "Relu" || node.op_type() == "Sigmoid" ||
                              node.op_type() == "Softmax" )
                    {
                        if ( node.input_size() > 0 && shape_metadata.exists( node.input( 0 ) ) )
                        {
                            output_shape = shape_metadata[node.input( 0 )];
                        }
                    }

                    if ( !output_shape.empty() )
                    {
                        shape_metadata[outputName] = output_shape;
                    }
                }

                boundedNode->setNodeIndex( i );
                boundedNode->setNodeName( tensorName );
            }
            catch ( const LunaError &e )
            {
                throw;
            }
            catch ( const std::exception &e )
            {
                onnxToTorchBoundedModuleCreationError( node.op_type(), e.what() );
            }

            nodes.append( boundedNode );
            continue;
        }

        if ( _onnx_model.graph().output_size() > 0 &&
             tensorName == _onnx_model.graph().output( 0 ).name() )
        {
            outputIndex = i;
        }
    }

    for ( unsigned i = 0; i < nodes.size(); ++i )
    {
        auto node = nodes[i];
        if ( !node )
            continue;

        if ( ( node->getNodeType() == NLR::NodeType::RELU ||
               node->getNodeType() == NLR::NodeType::SOFTMAX ||
               node->getNodeType() == NLR::NodeType::IDENTITY ||
               node->getNodeType() == NLR::NodeType::RESHAPE ||
               node->getNodeType() == NLR::NodeType::FLATTEN ||
               node->getNodeType() == NLR::NodeType::ADD ||
               node->getNodeType() == NLR::NodeType::SUB ||
               node->getNodeType() == NLR::NodeType::CONV ||
               node->getNodeType() == NLR::NodeType::CONVTRANSPOSE ||
               node->getNodeType() == NLR::NodeType::BATCHNORM ||
               node->getNodeType() == NLR::NodeType::LINEAR ||
               node->getNodeType() == NLR::NodeType::CONCAT ||
               node->getNodeType() == NLR::NodeType::SLICE ) &&
             node->getInputSize() == 0 )
        {

            if ( dependencies.exists( i ) && !dependencies[i].empty() )
            {
                unsigned inputIdx = dependencies[i][0];
                if ( inputIdx < nodes.size() && nodes[inputIdx] )
                {
                    unsigned inferredSize = nodes[inputIdx]->getOutputSize();
                    if ( inferredSize > 0 )
                    {
                        if ( node->getNodeType() == NLR::NodeType::CONV )
                        {
                            auto convNode = std::dynamic_pointer_cast<NLR::BoundedConvNode>( node );
                            if ( convNode )
                            {
                                unsigned outputSize = convNode->inferOutputSize( inferredSize );
                                node->setInputSize( inferredSize );
                                node->setOutputSize( outputSize );
                            }
                            else
                            {
                                node->setInputSize( inferredSize );
                                node->setOutputSize( 2 );
                            }
                        }
                        else if ( node->getNodeType() == NLR::NodeType::CONVTRANSPOSE )
                        {
                            auto convtNode =
                                std::dynamic_pointer_cast<NLR::BoundedConvTransposeNode>( node );
                            if ( convtNode )
                            {
                                unsigned outputSize = convtNode->inferOutputSize( inferredSize );
                                node->setInputSize( inferredSize );
                                node->setOutputSize( outputSize );
                            }
                            else
                            {
                                node->setInputSize( inferredSize );
                                node->setOutputSize( 2 );
                            }
                        }
                        else if ( node->getNodeType() == NLR::NodeType::LINEAR )
                        {
                            node->setInputSize( inferredSize );
                        }
                        else if ( node->getNodeType() == NLR::NodeType::CONCAT )
                        {
                            node->setInputSize( inferredSize );
                        }
                        else if ( node->getNodeType() == NLR::NodeType::SLICE )
                        {
                            node->setInputSize( inferredSize );
                        }
                        else
                        {
                            node->setInputSize( inferredSize );
                            if ( node->getOutputSize() == 0 )
                            {
                                node->setOutputSize( inferredSize );
                            }
                        }
                    }
                }
            }
        }
    }

    if ( outputIndex == 0 && nodes.size() > 1 )
    {
        for ( int i = nodes.size() - 1; i >= 0; --i )
        {
            if ( nodes[i] && nodes[i]->getNodeType() != NLR::NodeType::CONSTANT &&
                 nodes[i]->getNodeType() != NLR::NodeType::INPUT )
            {
                outputIndex = i;
                break;
            }
        }
    }

    return std::make_shared<NLR::TorchModel>( nodes, inputIndices, outputIndex, dependencies );
}

namespace AttributeUtils
{

float getFloatAttribute( const luna_onnx::NodeProto &node, const String &name, float defaultValue )
{
    for ( const auto &attr : node.attribute() )
    {
        if ( attr.name() == name.ascii() )
        {
            return attr.f();
        }
    }
    return defaultValue;
}

int getIntAttribute( const luna_onnx::NodeProto &node, const String &name, int defaultValue )
{
    for ( const auto &attr : node.attribute() )
    {
        if ( attr.name() == name.ascii() )
        {
            return attr.i();
        }
    }
    return defaultValue;
}

Vector<int> getIntsAttribute( luna_onnx::NodeProto &node, const String &name,
                              const Vector<int> &defaultValue )
{
    for ( const auto &attr : node.attribute() )
    {
        if ( attr.name() == name.ascii() )
        {
            Vector<int> result;
            for ( int i = 0; i < attr.ints_size(); i++ )
            {
                result.append( attr.ints( i ) );
            }
            return result;
        }
    }
    return defaultValue;
}

String getStringAttribute( luna_onnx::NodeProto &node, const String &name,
                           const String &defaultValue )
{
    for ( const auto &attr : node.attribute() )
    {
        if ( attr.name() == name.ascii() )
        {
            return String( attr.s() );
        }
    }
    return defaultValue;
}

Map<String, torch::IValue> extractAttributes( luna_onnx::NodeProto &node )
{
    Map<String, torch::IValue> kwargs;

    for ( const auto &attr : node.attribute() )
    {
        String attr_name = attr.name();

        if ( attr.type() == luna_onnx::AttributeProto::INT )
        {
            kwargs[attr_name] = torch::IValue( static_cast<int64_t>( attr.i() ) );
        }
        else if ( attr.type() == luna_onnx::AttributeProto::FLOAT )
        {
            kwargs[attr_name] = torch::IValue( static_cast<double>( attr.f() ) );
        }
        else if ( attr.type() == luna_onnx::AttributeProto::STRING )
        {
            kwargs[attr_name] = torch::IValue( attr.s() );
        }
        else if ( attr.type() == luna_onnx::AttributeProto::INTS )
        {
            std::vector<int64_t> ints;
            for ( int i = 0; i < attr.ints_size(); i++ )
            {
                ints.push_back( attr.ints( i ) );
            }
            kwargs[attr_name] = torch::IValue( ints );
        }
        else if ( attr.type() == luna_onnx::AttributeProto::FLOATS )
        {
            std::vector<double> floats;
            for ( int i = 0; i < attr.floats_size(); i++ )
            {
                floats.push_back( attr.floats( i ) );
            }
            kwargs[attr_name] = torch::IValue( floats );
        }
        else if ( attr.type() == luna_onnx::AttributeProto::TENSOR )
        {
            kwargs[attr_name] = torch::IValue( torch::zeros( { 1 } ) );
        }
    }

    return kwargs;
}
} // namespace AttributeUtils

namespace Operations
{

torch::Tensor Constant::forward()
{
    return value;
}

torch::Tensor ReshapeImpl::forward( const torch::Tensor &input, const torch::Tensor &shape_tensor )
{
    std::vector<int64_t> shape = GraphUtils::instantiateReshapeTemplate( input, shape_tensor );
    return input.reshape( shape );
}

} // namespace Operations

namespace GraphUtils
{

Vector<String>
computeTopologicalOrder( const Map<String, luna_onnx::NodeProto> &name_to_node,
                         const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                         const Map<String, luna_onnx::TensorProto> &name_to_initializer )
{
    Vector<String> order;

    for ( auto it = name_to_input.begin(); it != name_to_input.end(); ++it )
    {
        order.append( it->first );
    }

    for ( auto it = name_to_initializer.begin(); it != name_to_initializer.end(); ++it )
    {
        order.append( it->first );
    }

    for ( auto it = name_to_node.begin(); it != name_to_node.end(); ++it )
    {
        const luna_onnx::NodeProto &node = it->second;
        for ( int i = 0; i < node.output_size(); ++i )
        {
            String output = node.output( i );
            bool already_exists = false;
            for ( unsigned j = 0; j < order.size(); ++j )
            {
                if ( order[j] == output )
                {
                    already_exists = true;
                    break;
                }
            }
            if ( !already_exists )
            {
                order.append( output );
            }
        }
    }

    return order;
}

Map<String, Set<String>> computeActivationDependencies( const luna_onnx::GraphProto &graph )
{
    Map<String, Set<String>> needed_by;

    for ( const auto &node : graph.node() )
    {
        String out_op_id = node.output( 0 );
        for ( int i = 0; i < node.input_size(); ++i )
        {
            String in_op_id = node.input( i );
            needed_by[in_op_id].insert( out_op_id );
        }
    }

    return needed_by;
}

std::vector<int64_t> instantiateReshapeTemplate( const torch::Tensor &input,
                                                 const torch::Tensor &shape_tensor )
{
    std::vector<int64_t> oldShape( input.sizes().begin(), input.sizes().end() );
    std::vector<int64_t> newShapeTemplate;

    torch::Tensor flattened_shape = shape_tensor.flatten();
    for ( int64_t i = 0; i < flattened_shape.numel(); ++i )
    {
        newShapeTemplate.push_back( flattened_shape[i].item<int64_t>() );
    }

    std::vector<int64_t> newShape;
    int inferredIndex = -1;
    int64_t knownProduct = 1;

    for ( size_t i = 0; i < newShapeTemplate.size(); ++i )
    {
        int64_t dim = newShapeTemplate[i];
        if ( dim == 0 )
        {
            dim = ( i < oldShape.size() ) ? oldShape[i] : 1;
        }
        if ( dim == -1 )
        {
            inferredIndex = i;
            newShape.push_back( 1 ); // Placeholder
        }
        else
        {
            newShape.push_back( dim );
            knownProduct *= dim;
        }
    }

    if ( inferredIndex != -1 )
    {
        int64_t total = input.numel();
        int64_t inferred = total / knownProduct;
        newShape[inferredIndex] = inferred;
    }

    return newShape;
}

} // namespace GraphUtils

namespace ConstantProcessor
{

torch::Tensor processInitializer( const luna_onnx::TensorProto &tensor )
{
    std::vector<int64_t> shape;
    for ( int i = 0; i < tensor.dims_size(); ++i )
    {
        shape.push_back( tensor.dims( i ) );
    }
    switch ( tensor.data_type() )
    {
    case luna_onnx::TensorProto_DataType_FLOAT:
    {
        if ( !tensor.raw_data().empty() )
        {
            const std::string &raw_data = tensor.raw_data();
            size_t num_elements = raw_data.size() / sizeof( float );
            std::vector<float> data( num_elements );
            std::memcpy( data.data(), raw_data.data(), raw_data.size() );
            torch::Tensor result = torch::tensor( data, torch::kFloat32 ).reshape( shape );
            return result;
        }

        if ( tensor.float_data_size() == 0 )
        {
            int64_t total_elements = 1;
            for ( int i = 0; i < tensor.dims_size(); ++i )
            {
                total_elements *= tensor.dims( i );
            }

            if ( total_elements == 0 )
            {
                torch::Tensor result = torch::empty( shape, torch::kFloat32 );
                return result;
            }

            std::string error_msg = "No data found in tensor (neither raw_data nor float_data). ";
            error_msg += "Shape: [";
            for ( int i = 0; i < tensor.dims_size(); ++i )
            {
                error_msg += std::to_string( tensor.dims( i ) );
                if ( i < tensor.dims_size() - 1 )
                    error_msg += ", ";
            }
            error_msg += "], Total elements: " + std::to_string( total_elements );
            onnxToTorchTensorConversionError( tensor.name(), error_msg );
        }
        std::vector<float> data;
        for ( int i = 0; i < tensor.float_data_size(); ++i )
        {
            data.push_back( tensor.float_data( i ) );
        }
        torch::Tensor result = torch::tensor( data, torch::kFloat32 ).reshape( shape );
        return result;
    }
    case luna_onnx::TensorProto_DataType_INT64:
    {
        if ( !tensor.raw_data().empty() )
        {
            const std::string &raw_data = tensor.raw_data();
            size_t num_elements = raw_data.size() / sizeof( int64_t );
            std::vector<int64_t> data( num_elements );
            std::memcpy( data.data(), raw_data.data(), raw_data.size() );
            return torch::tensor( data, torch::kInt64 ).reshape( shape );
        }

        std::vector<int64_t> data;
        for ( int i = 0; i < tensor.int64_data_size(); ++i )
        {
            data.push_back( tensor.int64_data( i ) );
        }
        return torch::tensor( data, torch::kInt64 ).reshape( shape );
    }
    case luna_onnx::TensorProto_DataType_INT32:
    {
        if ( !tensor.raw_data().empty() )
        {
            const std::string &raw_data = tensor.raw_data();
            size_t num_elements = raw_data.size() / sizeof( int32_t );
            std::vector<int32_t> data( num_elements );
            std::memcpy( data.data(), raw_data.data(), raw_data.size() );
            return torch::tensor( data, torch::kInt32 ).reshape( shape );
        }

        std::vector<int32_t> data;
        for ( int i = 0; i < tensor.int32_data_size(); ++i )
        {
            data.push_back( tensor.int32_data( i ) );
        }
        return torch::tensor( data, torch::kInt32 ).reshape( shape );
    }
    case luna_onnx::TensorProto_DataType_DOUBLE:
    {
        if ( !tensor.raw_data().empty() )
        {
            const std::string &raw_data = tensor.raw_data();
            size_t num_elements = raw_data.size() / sizeof( double );
            std::vector<double> data( num_elements );
            std::memcpy( data.data(), raw_data.data(), raw_data.size() );
            return torch::tensor( data, torch::kFloat64 ).reshape( shape );
        }

        std::vector<double> data;
        for ( int i = 0; i < tensor.double_data_size(); ++i )
        {
            data.push_back( tensor.double_data( i ) );
        }
        return torch::tensor( data, torch::kFloat64 ).reshape( shape );
    }
    default:
        onnxToTorchUnsupportedDataTypeError(
            static_cast<luna_onnx::TensorProto_DataType>( tensor.data_type() ) );
        return torch::Tensor(); // This line will never be reached, but satisfies compiler
    }
}

torch::Tensor processConstantNode( const luna_onnx::NodeProto &node )
{
    for ( const auto &attr : node.attribute() )
    {
        if ( attr.name() == "value" )
        {
            if ( attr.has_t() )
            {
                try
                {
                    torch::Tensor result = processInitializer( attr.t() );
                    return result;
                }
                catch ( const std::exception &e )
                {
                    throw;
                }
            }
        }
    }
    onnxToTorchInvalidConstantNodeError( node, "No valid value attribute found" );
    return torch::Tensor();
}

} // namespace ConstantProcessor

namespace BoundedOperationConverter
{

TensorShape extractShapeFromNode( const luna_onnx::NodeProto &node,
                                  const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                                  const Map<String, luna_onnx::TensorProto> &name_to_initializer,
                                  const String &tensorName )
{
    (void)node;
    if ( name_to_input.exists( tensorName ) )
    {
        const auto &inputInfo = name_to_input[tensorName];
        if ( inputInfo.type().tensor_type().has_shape() )
        {
            const auto &shape = inputInfo.type().tensor_type().shape();
            TensorShape result;
            for ( int i = 0; i < shape.dim_size(); ++i )
            {
                if ( shape.dim( i ).has_dim_value() )
                {
                    result.append( shape.dim( i ).dim_value() );
                }
            }
            return result;
        }
    }

    if ( name_to_initializer.exists( tensorName ) )
    {
        const auto &initializer = name_to_initializer[tensorName];
        TensorShape result;
        for ( int i = 0; i < initializer.dims_size(); ++i )
        {
            result.append( initializer.dims( i ) );
        }
        return result;
    }

    return TensorShape();
}

unsigned computeTensorSize( const TensorShape &shape )
{
    if ( shape.empty() )
        return 0;

    unsigned size = 1;
    for ( unsigned dim : shape )
    {
        size *= dim;
    }
    return size;
}

static TensorShape computeBroadcastShape( const String &operation, const TensorShape &a,
                                          const TensorShape &b )
{
    if ( a.empty() )
        return b;
    if ( b.empty() )
        return a;

    const unsigned ra = a.size();
    const unsigned rb = b.size();
    const unsigned r = ( ra >= rb ) ? ra : rb;

    std::vector<unsigned> out_rev;
    out_rev.reserve( r );

    for ( unsigned i = 0; i < r; ++i )
    {
        unsigned da = ( i < ra ) ? a[ra - 1 - i] : 1;
        unsigned db = ( i < rb ) ? b[rb - 1 - i] : 1;

        if ( da != db && da != 1 && db != 1 )
        {
            onnxToTorchInvalidBroadcastError( operation, a, b );
            return TensorShape();
        }
        out_rev.push_back( ( da >= db ) ? da : db );
    }

    TensorShape out;
    for ( auto it = out_rev.rbegin(); it != out_rev.rend(); ++it )
    {
        out.append( *it );
    }
    return out;
}

std::shared_ptr<NLR::BoundedTorchNode>
convertGemm( const luna_onnx::NodeProto &node, const Map<String, torch::Tensor> &constants,
             const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
             const Map<String, luna_onnx::TensorProto> &name_to_initializer )
{
    (void)name_to_input;
    (void)name_to_initializer;
    if ( node.input_size() < 2 )
    {
        onnxToTorchUnexpectedNumberOfInputs( node, node.input_size(), 2, 3 );
        return nullptr;
    }

    String weightName = node.input( 1 );
    String biasName = ( node.input_size() > 2 ) ? node.input( 2 ) : "";

    torch::Tensor weights;
    bool foundWeights = false;

    if ( constants.exists( weightName ) )
    {
        weights = constants[weightName];
        foundWeights = true;
    }
    else
    {
        Vector<String> possibleNames = { weightName, "weight", "W", "weights" };
        for ( const auto &name : possibleNames )
        {
            if ( constants.exists( name ) )
            {
                weights = constants[name];
                foundWeights = true;
                break;
            }
        }
    }

    if ( !foundWeights )
    {
        onnxToTorchInvalidWeightBiasError( "Gemm", "Weight tensor not found in constants" );
        return nullptr;
    }

    torch::Tensor bias;
    if ( biasName.length() > 0 && constants.exists( biasName ) )
    {
        bias = constants[biasName];
    }
    else
    {
        bias = torch::zeros( { weights.size( 0 ) } );
    }

    float alpha = AttributeUtils::getFloatAttribute( node, "alpha", 1.0f );
    float beta = AttributeUtils::getFloatAttribute( node, "beta", 1.0f );
    int transB = AttributeUtils::getIntAttribute( node, "transB", 0 );

    if ( transB == 0 )
    {
        weights = weights.transpose( -2, -1 );
    }

    if ( beta != 1.0f )
    {
        bias = beta * bias;
    }

    // network weights must not require grad; only alpha params are optimized
    weights = weights.contiguous().to( torch::kFloat32 ).detach().requires_grad_( false );
    bias = bias.contiguous().to( torch::kFloat32 ).detach().requires_grad_( false );

    auto linear_module = torch::nn::Linear( weights.size( 1 ), weights.size( 0 ) );
    linear_module->weight = weights;
    linear_module->bias = bias;

    auto boundedNode = std::make_shared<NLR::BoundedLinearNode>( linear_module, alpha );

    return boundedNode;
}

std::shared_ptr<NLR::BoundedTorchNode>
convertMatMul( const luna_onnx::NodeProto &node, const Map<String, torch::Tensor> &constants,
               const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
               const Map<String, luna_onnx::TensorProto> &name_to_initializer )
{
    (void)name_to_input;
    (void)name_to_initializer;

    if ( node.input_size() != 2 )
    {
        onnxToTorchUnexpectedNumberOfInputs( node, node.input_size(), 2, 2 );
        return nullptr;
    }

    String input1Name = node.input( 0 );
    String input2Name = node.input( 1 );

    torch::Tensor weights;
    bool foundWeights = false;
    bool weightsAreFirstInput = false;

    if ( constants.exists( input2Name ) )
    {
        weights = constants[input2Name];
        foundWeights = true;
    }
    else if ( constants.exists( input1Name ) )
    {
        weights = constants[input1Name];
        foundWeights = true;
        weightsAreFirstInput = true;
    }

    if ( !foundWeights )
    {
        onnxToTorchInvalidWeightBiasError(
            "MatMul", "Weight tensor not found in constants (checked both inputs)" );
        return nullptr;
    }

    torch::Tensor transposed_weights;
    int64_t in_features;
    int64_t out_features;

    if ( weightsAreFirstInput )
    {
        out_features = weights.size( -2 );
        in_features = weights.size( -1 );
        transposed_weights = weights;
    }
    else
    {
        in_features = weights.size( -2 );
        out_features = weights.size( -1 );
        transposed_weights = weights.transpose( -2, -1 );
    }

    transposed_weights =
        transposed_weights.contiguous().to( torch::kFloat32 ).detach().requires_grad_( false );

    auto linear_module =
        torch::nn::Linear( torch::nn::LinearOptions( in_features, out_features ).bias( false ) );
    linear_module->weight = transposed_weights;

    auto boundedNode = std::make_shared<NLR::BoundedLinearNode>( linear_module, 1.0f );

    return boundedNode;
}

std::shared_ptr<NLR::BoundedTorchNode>
convertAdd( const luna_onnx::NodeProto &node, const Map<String, torch::Tensor> &constants,
            const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
            const Map<String, luna_onnx::TensorProto> &name_to_initializer,
            const Vector<std::shared_ptr<NLR::BoundedTorchNode>> &existingNodes,
            const Map<String, unsigned> &nameToIndex )
{
    (void)existingNodes;
    (void)nameToIndex;

    if ( node.input_size() != 2 )
    {
        onnxToTorchUnexpectedNumberOfInputs( node, node.input_size(), 2, 2 );
        return nullptr;
    }

    String input1Name = node.input( 0 );
    String input2Name = node.input( 1 );

    TensorShape shape1 =
        extractShapeFromNode( node, name_to_input, name_to_initializer, input1Name );
    TensorShape shape2 =
        extractShapeFromNode( node, name_to_input, name_to_initializer, input2Name );

    unsigned size1 = computeTensorSize( shape1 );
    unsigned size2 = computeTensorSize( shape2 );

    // best-effort broadcast size
    unsigned broadcastOutputSize = 0;
    if ( !shape1.empty() || !shape2.empty() )
    {
        TensorShape outShape = computeBroadcastShape( "Add", shape1, shape2 );
        broadcastOutputSize = computeTensorSize( outShape );
    }
    if ( broadcastOutputSize == 0 )
    {
        broadcastOutputSize = ( size1 >= size2 ) ? size1 : size2;
    }

    torch::Tensor constantValue;
    bool isFirstInputConstant =
        ( constants.exists( input1Name ) || name_to_initializer.exists( input1Name ) );
    bool isSecondInputConstant =
        ( constants.exists( input2Name ) || name_to_initializer.exists( input2Name ) );

    auto boundedNode = std::make_shared<NLR::BoundedAddNode>();

    if ( isFirstInputConstant && !isSecondInputConstant )
    {
        if ( constants.exists( input1Name ) )
        {
            constantValue = constants[input1Name];
        }
        else if ( name_to_initializer.exists( input1Name ) )
        {
            constantValue =
                ConstantProcessor::processInitializer( name_to_initializer[input1Name] );
        }
        boundedNode->setConstantValue( constantValue );

        unsigned outputSize = broadcastOutputSize;
        boundedNode->setInputSize( size2 );
        boundedNode->setOutputSize( outputSize );
    }
    else if ( isSecondInputConstant && !isFirstInputConstant )
    {
        if ( constants.exists( input2Name ) )
        {
            constantValue = constants[input2Name];
        }
        else if ( name_to_initializer.exists( input2Name ) )
        {
            constantValue =
                ConstantProcessor::processInitializer( name_to_initializer[input2Name] );
        }
        boundedNode->setConstantValue( constantValue );

        unsigned outputSize = broadcastOutputSize;
        boundedNode->setInputSize( size1 );
        boundedNode->setOutputSize( outputSize );
    }
    else if ( !isFirstInputConstant && !isSecondInputConstant )
    {
        unsigned outputSize = broadcastOutputSize;
        boundedNode->setInputSize( size1 );
        boundedNode->setOutputSize( outputSize );
    }
    else
    {
        torch::Tensor const1, const2;
        if ( constants.exists( input1Name ) )
        {
            const1 = constants[input1Name];
        }
        else if ( name_to_initializer.exists( input1Name ) )
        {
            const1 = ConstantProcessor::processInitializer( name_to_initializer[input1Name] );
        }
        if ( constants.exists( input2Name ) )
        {
            const2 = constants[input2Name];
        }
        else if ( name_to_initializer.exists( input2Name ) )
        {
            const2 = ConstantProcessor::processInitializer( name_to_initializer[input2Name] );
        }

        torch::Tensor result = const1 + const2;
        return std::make_shared<NLR::BoundedConstantNode>( result, "add_folded" );
    }

    return boundedNode;
}

std::shared_ptr<NLR::BoundedTorchNode>
convertRelu( const luna_onnx::NodeProto &node,
             const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
             const Map<String, luna_onnx::TensorProto> &name_to_initializer )
{
    unsigned inputSize = 0;
    if ( node.input_size() > 0 )
    {
        String inputName = node.input( 0 );
        TensorShape inputShape =
            extractShapeFromNode( node, name_to_input, name_to_initializer, inputName );
        inputSize = computeTensorSize( inputShape );
    }

    auto relu_module = torch::nn::ReLU( torch::nn::ReLUOptions().inplace( false ) );
    auto boundedNode = std::make_shared<NLR::BoundedReLUNode>( relu_module );

    if ( inputSize > 0 )
    {
        boundedNode->setInputSize( inputSize );
        boundedNode->setOutputSize( inputSize );
    }

    return boundedNode;
}

std::shared_ptr<NLR::BoundedTorchNode>
convertIdentity( const luna_onnx::NodeProto &node,
                 const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                 const Map<String, luna_onnx::TensorProto> &name_to_initializer )
{
    unsigned inputSize = 0;
    if ( node.input_size() > 0 )
    {
        String inputName = node.input( 0 );
        TensorShape inputShape =
            extractShapeFromNode( node, name_to_input, name_to_initializer, inputName );
        inputSize = computeTensorSize( inputShape );
    }

    auto identity_module = torch::nn::Identity();
    auto boundedNode = std::make_shared<NLR::BoundedIdentityNode>( identity_module );

    if ( inputSize > 0 )
    {
        boundedNode->setInputSize( inputSize );
        boundedNode->setOutputSize( inputSize );
    }

    return boundedNode;
}

std::shared_ptr<NLR::BoundedTorchNode>
convertReshape( const luna_onnx::NodeProto &node,
                const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                const Map<String, luna_onnx::TensorProto> &name_to_initializer )
{
    unsigned inputSize = 0;
    unsigned outputSize = 0;

    if ( node.input_size() > 0 )
    {
        String inputName = node.input( 0 );
        TensorShape inputShape =
            extractShapeFromNode( node, name_to_input, name_to_initializer, inputName );
        inputSize = computeTensorSize( inputShape );
    }

    if ( node.output_size() > 0 )
    {
        String outputName = node.output( 0 );
        TensorShape outputShape =
            extractShapeFromNode( node, name_to_input, name_to_initializer, outputName );
        outputSize = computeTensorSize( outputShape );
    }

    torch::Tensor default_shape = torch::tensor( { -1 } );
    auto reshape_module = Operations::ReshapeWrapper( default_shape );
    auto boundedNode = std::make_shared<NLR::BoundedReshapeNode>( reshape_module );

    if ( inputSize > 0 )
    {
        boundedNode->setInputSize( inputSize );
    }
    if ( outputSize > 0 )
    {
        boundedNode->setOutputSize( outputSize );
    }

    return boundedNode;
}

std::shared_ptr<NLR::BoundedTorchNode>
convertFlatten( const luna_onnx::NodeProto &node,
                const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                const Map<String, luna_onnx::TensorProto> &name_to_initializer )
{
    int axis = AttributeUtils::getIntAttribute( node, "axis", 1 );

    unsigned inputSize = 0;
    unsigned outputSize = 0;
    TensorShape inputShape;

    if ( node.input_size() > 0 )
    {
        String inputName = node.input( 0 );
        inputShape = extractShapeFromNode( node, name_to_input, name_to_initializer, inputName );
        inputSize = computeTensorSize( inputShape );

        outputSize = inputSize;
    }

    auto flatten_module = Operations::FlattenWrapper( axis );
    auto boundedNode = std::make_shared<NLR::BoundedFlattenNode>( flatten_module );

    if ( inputSize > 0 )
    {
        boundedNode->setInputSize( inputSize );
        boundedNode->setOutputSize( outputSize );
        std::vector<int64_t> shapeVec;
        for ( unsigned dim : inputShape )
        {
            shapeVec.push_back( static_cast<int64_t>( dim ) );
        }
        boundedNode->setInputShape( shapeVec );
    }

    return boundedNode;
}

std::shared_ptr<NLR::BoundedTorchNode>
convertSigmoid( const luna_onnx::NodeProto &node,
                const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                const Map<String, luna_onnx::TensorProto> &name_to_initializer )
{
    unsigned inputSize = 0;
    if ( node.input_size() > 0 )
    {
        String inputName = node.input( 0 );
        TensorShape inputShape =
            extractShapeFromNode( node, name_to_input, name_to_initializer, inputName );
        inputSize = computeTensorSize( inputShape );
    }

    auto sigmoid_module = torch::nn::Sigmoid();
    auto boundedNode = std::make_shared<NLR::BoundedSigmoidNode>( sigmoid_module );

    if ( inputSize > 0 )
    {
        boundedNode->setInputSize( inputSize );
        boundedNode->setOutputSize( inputSize );
    }

    return boundedNode;
}

std::shared_ptr<NLR::BoundedTorchNode>
convertSoftmax( const luna_onnx::NodeProto &node,
                const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                const Map<String, luna_onnx::TensorProto> &name_to_initializer )
{
    int axis = AttributeUtils::getIntAttribute( node, "axis", -1 );

    unsigned inputSize = 0;
    if ( node.input_size() > 0 )
    {
        String inputName = node.input( 0 );
        TensorShape inputShape =
            extractShapeFromNode( node, name_to_input, name_to_initializer, inputName );
        inputSize = computeTensorSize( inputShape );
    }

    auto softmax_module = torch::nn::Softmax( torch::nn::SoftmaxOptions( axis ) );
    auto boundedNode = std::make_shared<NLR::BoundedSoftmaxNode>( softmax_module );

    if ( inputSize > 0 )
    {
        boundedNode->setInputSize( inputSize );
        boundedNode->setOutputSize( inputSize );
    }

    return boundedNode;
}

std::shared_ptr<NLR::BoundedTorchNode>
convertSub( const luna_onnx::NodeProto &node, const Map<String, torch::Tensor> &constants,
            const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
            const Map<String, luna_onnx::TensorProto> &name_to_initializer )
{
    if ( node.input_size() != 2 )
    {
        onnxToTorchUnexpectedNumberOfInputs( node, node.input_size(), 2, 2 );
        return nullptr;
    }

    String input1Name = node.input( 0 );
    String input2Name = node.input( 1 );

    TensorShape shape1 =
        extractShapeFromNode( node, name_to_input, name_to_initializer, input1Name );
    TensorShape shape2 =
        extractShapeFromNode( node, name_to_input, name_to_initializer, input2Name );

    unsigned size1 = computeTensorSize( shape1 );
    unsigned size2 = computeTensorSize( shape2 );

    bool isFirstInputConstant =
        ( constants.exists( input1Name ) || name_to_initializer.exists( input1Name ) );
    bool isSecondInputConstant =
        ( constants.exists( input2Name ) || name_to_initializer.exists( input2Name ) );
    auto boundedNode = std::make_shared<NLR::BoundedSubNode>();

    if ( isFirstInputConstant && !isSecondInputConstant )
    {
        torch::Tensor constantValue;
        if ( constants.exists( input1Name ) )
        {
            constantValue = constants[input1Name];
        }
        else if ( name_to_initializer.exists( input1Name ) )
        {
            constantValue =
                ConstantProcessor::processInitializer( name_to_initializer[input1Name] );
        }
        boundedNode->setConstantValue( constantValue, false );

        unsigned outputSize = size2;
        boundedNode->setInputSize( size2 );
        boundedNode->setOutputSize( outputSize );
    }
    else if ( isSecondInputConstant && !isFirstInputConstant )
    {
        torch::Tensor constantValue;
        if ( constants.exists( input2Name ) )
        {
            constantValue = constants[input2Name];
        }
        else if ( name_to_initializer.exists( input2Name ) )
        {
            constantValue =
                ConstantProcessor::processInitializer( name_to_initializer[input2Name] );
        }
        boundedNode->setConstantValue( constantValue, true );

        unsigned outputSize = size1;
        boundedNode->setInputSize( size1 );
        boundedNode->setOutputSize( outputSize );
    }
    else if ( !isFirstInputConstant && !isSecondInputConstant )
    {
        unsigned outputSize = ( size1 >= size2 ) ? size1 : size2;
        boundedNode->setInputSize( size1 );
        boundedNode->setOutputSize( outputSize );
    }
    else
    {
        torch::Tensor const1, const2;
        if ( constants.exists( input1Name ) )
        {
            const1 = constants[input1Name];
        }
        else if ( name_to_initializer.exists( input1Name ) )
        {
            const1 = ConstantProcessor::processInitializer( name_to_initializer[input1Name] );
        }
        if ( constants.exists( input2Name ) )
        {
            const2 = constants[input2Name];
        }
        else if ( name_to_initializer.exists( input2Name ) )
        {
            const2 = ConstantProcessor::processInitializer( name_to_initializer[input2Name] );
        }

        torch::Tensor result = const1 - const2;
        return std::make_shared<NLR::BoundedConstantNode>( result, "sub_folded" );
    }

    return boundedNode;
}

std::shared_ptr<NLR::BoundedTorchNode>
convertBatchNormalization( const luna_onnx::NodeProto &node,
                           const Map<String, torch::Tensor> &constants,
                           const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                           const Map<String, luna_onnx::TensorProto> &name_to_initializer )
{
    (void)name_to_input;
    (void)name_to_initializer;

    if ( node.input_size() != 5 )
    {
        onnxToTorchUnexpectedNumberOfInputs( node, node.input_size(), 5, 5 );
        return nullptr;
    }

    float eps = AttributeUtils::getFloatAttribute( node, "epsilon", 1e-5f );

    String scaleName = node.input( 1 );
    String BName = node.input( 2 );
    String meanName = node.input( 3 );
    String varName = node.input( 4 );

    auto getTensorConst = [&]( const String &name, const char *what ) -> torch::Tensor
    {
        if ( constants.exists( name ) )
        {
            return constants[name];
        }
        if ( name_to_initializer.exists( name ) )
        {
            return ConstantProcessor::processInitializer( name_to_initializer[name] );
        }
        onnxToTorchInvalidWeightBiasError(
            "BatchNormalization",
            Stringf( "%s tensor not found in constants/initializers", what ).ascii() );
        return torch::Tensor();
    };

    torch::Tensor scale = getTensorConst( scaleName, "scale" );
    torch::Tensor B = getTensorConst( BName, "B" );
    torch::Tensor mean = getTensorConst( meanName, "mean" );
    torch::Tensor var = getTensorConst( varName, "var" );

    if ( !scale.defined() || !B.defined() || !mean.defined() || !var.defined() )
    {
        return nullptr;
    }

    scale = scale.contiguous().to( torch::kFloat32 );
    B = B.contiguous().to( torch::kFloat32 );
    mean = mean.contiguous().to( torch::kFloat32 );
    var = var.contiguous().to( torch::kFloat32 );

    auto boundedNode = std::make_shared<NLR::BoundedBatchNormNode>( scale, B, mean, var, eps );

    if ( node.input_size() > 0 )
    {
        String inputName = node.input( 0 );
        TensorShape inputShape =
            extractShapeFromNode( node, name_to_input, name_to_initializer, inputName );
        unsigned inputSize = computeTensorSize( inputShape );
        if ( inputSize > 0 )
        {
            boundedNode->setInputSize( inputSize );
            boundedNode->setOutputSize( inputSize );
        }
    }

    return boundedNode;
}

std::shared_ptr<NLR::BoundedTorchNode> convertConstant( const torch::Tensor &value )
{
    auto boundedNode = std::make_shared<NLR::BoundedConstantNode>( value, "" );

    if ( value.defined() )
    {
        unsigned size = value.numel();
        boundedNode->setInputSize( 0 );
        boundedNode->setOutputSize( size );
    }

    return boundedNode;
}

std::shared_ptr<NLR::BoundedTorchNode>
convertConv( const luna_onnx::NodeProto &node, const Map<String, torch::Tensor> &constants,
             const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
             const Map<String, luna_onnx::TensorProto> &name_to_initializer,
             const Map<String, Vector<int>> &shape_metadata )
{
    (void)name_to_input;
    (void)name_to_initializer;

    Vector<int> input_shape;
    if ( node.input_size() > 0 )
    {
        String inputName = node.input( 0 );
        if ( shape_metadata.exists( inputName ) )
        {
            input_shape = shape_metadata[inputName];
        }
    }

    if ( node.input_size() < 2 )
    {
        onnxToTorchUnexpectedNumberOfInputs( node, node.input_size(), 2, 3 );
        return nullptr;
    }

    String weightName = node.input( 1 );
    String biasName = ( node.input_size() > 2 ) ? node.input( 2 ) : "";

    torch::Tensor weights;
    bool foundWeights = false;

    if ( constants.exists( weightName ) )
    {
        weights = constants[weightName];
        foundWeights = true;
    }
    else
    {
        Vector<String> possibleNames = { weightName, "weight", "W", "weights" };
        for ( const auto &name : possibleNames )
        {
            if ( constants.exists( name ) )
            {
                weights = constants[name];
                foundWeights = true;
                break;
            }
        }
    }

    if ( !foundWeights )
    {
        onnxToTorchInvalidWeightBiasError( "Conv", "Weight tensor not found in constants" );
        return nullptr;
    }

    torch::Tensor bias;
    bool has_bias = false;
    if ( biasName.length() > 0 && constants.exists( biasName ) )
    {
        bias = constants[biasName];
        has_bias = true;
    }

    auto kernel_shape = AttributeUtils::getIntsAttribute(
        const_cast<luna_onnx::NodeProto &>( node ), "kernel_shape", {} );
    int group = AttributeUtils::getIntAttribute( node, "group", 1 );

    bool is_conv1d = ( weights.dim() == 3 );
    bool is_conv2d = ( weights.dim() == 4 );

    if ( !is_conv1d && !is_conv2d )
    {
        onnxToTorchInvalidWeightBiasError(
            "Conv",
            Stringf( "Expected 3D (Conv1d) or 4D (Conv2d) weight tensor, got %ldD", weights.dim() )
                .ascii() );
        return nullptr;
    }

    auto strides =
        is_conv1d ? AttributeUtils::getIntsAttribute( const_cast<luna_onnx::NodeProto &>( node ),
                                                      "strides", { 1 } )
                  : AttributeUtils::getIntsAttribute( const_cast<luna_onnx::NodeProto &>( node ),
                                                      "strides", { 1, 1 } );
    auto pads = is_conv1d
                    ? AttributeUtils::getIntsAttribute( const_cast<luna_onnx::NodeProto &>( node ),
                                                        "pads", { 0, 0 } )
                    : AttributeUtils::getIntsAttribute( const_cast<luna_onnx::NodeProto &>( node ),
                                                        "pads", { 0, 0, 0, 0 } );
    auto dilations =
        is_conv1d ? AttributeUtils::getIntsAttribute( const_cast<luna_onnx::NodeProto &>( node ),
                                                      "dilations", { 1 } )
                  : AttributeUtils::getIntsAttribute( const_cast<luna_onnx::NodeProto &>( node ),
                                                      "dilations", { 1, 1 } );

    int out_channels = weights.size( 0 );
    int in_channels_per_group = weights.size( 1 );
    int in_channels = in_channels_per_group * group;

    if ( is_conv1d )
    {
        int kernel_length = weights.size( 2 );
        if ( !kernel_shape.empty() )
        {
            if ( kernel_shape.size() != 1 || kernel_shape[0] != kernel_length )
            {
                onnxToTorchAttributeProcessingError(
                    node, "kernel_shape", "Mismatch with weight tensor dimensions for Conv1d" );
            }
        }
    }
    else
    {
        int kernel_height = weights.size( 2 );
        int kernel_width = weights.size( 3 );
        if ( !kernel_shape.empty() )
        {
            if ( kernel_shape.size() != 2 || kernel_shape[0] != kernel_height ||
                 kernel_shape[1] != kernel_width )
            {
                onnxToTorchAttributeProcessingError(
                    node, "kernel_shape", "Mismatch with weight tensor dimensions for Conv2d" );
            }
        }
    }

    std::vector<int64_t> padding;
    if ( is_conv1d )
    {
        if ( pads.size() == 2 )
        {
            padding = { pads[0] };
        }
        else if ( pads.empty() )
        {
            padding = { 0 };
        }
        else
        {
            onnxToTorchAttributeProcessingError(
                node, "pads",
                Stringf( "Invalid padding size for Conv1d: %lu", pads.size() ).ascii() );
        }
    }
    else
    {
        if ( pads.size() == 4 )
        {
            if ( pads[0] == pads[2] && pads[1] == pads[3] )
            {
                padding = { pads[0], pads[1] };
            }
            else
            {
                padding = { pads[0], pads[1] };
            }
        }
        else if ( pads.size() == 2 )
        {
            padding = { pads[0], pads[1] };
        }
        else if ( pads.empty() )
        {
            padding = { 0, 0 };
        }
        else
        {
            onnxToTorchAttributeProcessingError(
                node, "pads",
                Stringf( "Invalid padding size for Conv2d: %lu", pads.size() ).ascii() );
        }
    }

    std::vector<int64_t> stride_vec( strides.begin(), strides.end() );
    std::vector<int64_t> dilation_vec( dilations.begin(), dilations.end() );

    weights = weights.contiguous().to( torch::kFloat32 ).detach().requires_grad_( false );
    if ( has_bias )
    {
        bias = bias.contiguous().to( torch::kFloat32 ).detach().requires_grad_( false );
    }

    std::shared_ptr<NLR::BoundedConvNode> boundedNode;

    if ( is_conv1d )
    {
        int kernel_length = weights.size( 2 );
        torch::nn::Conv1dOptions conv_options( in_channels, out_channels, kernel_length );
        conv_options.stride( stride_vec );
        conv_options.padding( padding );
        conv_options.dilation( dilation_vec );
        conv_options.groups( group );
        conv_options.bias( has_bias );

        auto conv_module = torch::nn::Conv1d( conv_options );

        conv_module->weight = weights;
        if ( has_bias )
        {
            conv_module->bias = bias;
        }

        boundedNode = std::make_shared<NLR::BoundedConvNode>( conv_module, NLR::ConvMode::PATCHES );
    }
    else
    {
        int kernel_height = weights.size( 2 );
        int kernel_width = weights.size( 3 );
        torch::nn::Conv2dOptions conv_options( in_channels, out_channels,
                                               { kernel_height, kernel_width } );
        conv_options.stride( stride_vec );
        conv_options.padding( padding );
        conv_options.dilation( dilation_vec );
        conv_options.groups( group );
        conv_options.bias( has_bias );

        auto conv_module = torch::nn::Conv2d( conv_options );

        conv_module->weight = weights;
        if ( has_bias )
        {
            conv_module->bias = bias;
        }
        boundedNode = std::make_shared<NLR::BoundedConvNode>( conv_module, NLR::ConvMode::MATRIX );
    }

    if ( !input_shape.empty() )
    {
        unsigned input_size = 1;
        for ( unsigned i = 1; i < input_shape.size(); ++i )
        {
            if ( input_shape[i] > 0 )
            {
                input_size *= input_shape[i];
            }
        }

        unsigned output_size = 0;
        if ( is_conv1d && input_shape.size() >= 3 )
        {
            // Conv1D: [N, C, L]
            int L = input_shape[2];
            if ( L > 0 )
            {
                int kernel_length = weights.size( 2 );
                int out_l = ( L + 2 * padding[0] - dilation_vec[0] * ( kernel_length - 1 ) - 1 ) /
                                stride_vec[0] +
                            1;
                output_size = out_channels * out_l;
            }
        }
        else if ( !is_conv1d && input_shape.size() >= 4 )
        {
            // Conv2D: [N, C, H, W]
            int H = input_shape[2];
            int W = input_shape[3];
            if ( H > 0 && W > 0 )
            {
                int kernel_height = weights.size( 2 );
                int kernel_width = weights.size( 3 );
                int out_h = ( H + 2 * padding[0] - dilation_vec[0] * ( kernel_height - 1 ) - 1 ) /
                                stride_vec[0] +
                            1;
                int out_w = ( W + 2 * padding[1] - dilation_vec[1] * ( kernel_width - 1 ) - 1 ) /
                                stride_vec[1] +
                            1;
                output_size = out_channels * out_h * out_w;
            }
        }

        if ( input_size > 0 && output_size > 0 )
        {
            boundedNode->setInputSize( input_size );
            boundedNode->setOutputSize( output_size );

            std::vector<int> input_shape_vec;
            std::vector<int> output_shape_vec;

            for ( int dim : input_shape )
            {
                if ( dim > 0 )
                {
                    input_shape_vec.push_back( dim );
                }
                else
                {
                    input_shape_vec.push_back( 1 );
                }
            }

            if ( is_conv1d && input_shape.size() >= 3 )
            {
                int N = ( input_shape[0] > 0 ) ? input_shape[0] : 1;
                int L = input_shape[2];
                int kernel_length = weights.size( 2 );
                int out_l = ( L + 2 * padding[0] - dilation_vec[0] * ( kernel_length - 1 ) - 1 ) /
                                stride_vec[0] +
                            1;
                output_shape_vec = { N, out_channels, out_l };
            }
            else if ( !is_conv1d && input_shape.size() >= 4 )
            {
                int N = ( input_shape[0] > 0 ) ? input_shape[0] : 1;
                int H = input_shape[2];
                int W = input_shape[3];
                int kernel_height = weights.size( 2 );
                int kernel_width = weights.size( 3 );
                int out_h = ( H + 2 * padding[0] - dilation_vec[0] * ( kernel_height - 1 ) - 1 ) /
                                stride_vec[0] +
                            1;
                int out_w = ( W + 2 * padding[1] - dilation_vec[1] * ( kernel_width - 1 ) - 1 ) /
                                stride_vec[1] +
                            1;
                output_shape_vec = { N, out_channels, out_h, out_w };
            }

            if ( !input_shape_vec.empty() && !output_shape_vec.empty() )
            {
                boundedNode->setInputShape( input_shape_vec );
                boundedNode->setOutputShape( output_shape_vec );
            }
        }
    }

    return boundedNode;
}

std::shared_ptr<NLR::BoundedTorchNode>
convertConvTranspose( const luna_onnx::NodeProto &node, const Map<String, torch::Tensor> &constants,
                      const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                      const Map<String, luna_onnx::TensorProto> &name_to_initializer )
{
    (void)name_to_input;
    (void)name_to_initializer;

    if ( node.input_size() < 2 )
    {
        onnxToTorchUnexpectedNumberOfInputs( node, node.input_size(), 2, 3 );
        return nullptr;
    }

    String weightName = node.input( 1 );
    String biasName = ( node.input_size() > 2 ) ? node.input( 2 ) : "";

    torch::Tensor weights;
    bool foundWeights = false;

    if ( constants.exists( weightName ) )
    {
        weights = constants[weightName];
        foundWeights = true;
    }
    else
    {
        Vector<String> possibleNames = { weightName, "weight", "W", "weights" };
        for ( const auto &name : possibleNames )
        {
            if ( constants.exists( name ) )
            {
                weights = constants[name];
                foundWeights = true;
                break;
            }
        }
    }

    if ( !foundWeights )
    {
        onnxToTorchInvalidWeightBiasError( "ConvTranspose",
                                           "Weight tensor not found in constants" );
        return nullptr;
    }

    torch::Tensor bias;
    bool has_bias = false;
    if ( biasName.length() > 0 && constants.exists( biasName ) )
    {
        bias = constants[biasName];
        has_bias = true;
    }

    auto kernel_shape = AttributeUtils::getIntsAttribute(
        const_cast<luna_onnx::NodeProto &>( node ), "kernel_shape", {} );
    auto strides = AttributeUtils::getIntsAttribute( const_cast<luna_onnx::NodeProto &>( node ),
                                                     "strides", { 1, 1 } );
    auto pads = AttributeUtils::getIntsAttribute( const_cast<luna_onnx::NodeProto &>( node ),
                                                  "pads", { 0, 0, 0, 0 } );
    auto dilations = AttributeUtils::getIntsAttribute( const_cast<luna_onnx::NodeProto &>( node ),
                                                       "dilations", { 1, 1 } );
    auto output_padding = AttributeUtils::getIntsAttribute(
        const_cast<luna_onnx::NodeProto &>( node ), "output_padding", { 0, 0 } );
    int group = AttributeUtils::getIntAttribute( node, "group", 1 );

    if ( weights.dim() != 4 )
    {
        onnxToTorchInvalidWeightBiasError(
            "ConvTranspose",
            Stringf( "Expected 4D weight tensor, got %ldD", weights.dim() ).ascii() );
        return nullptr;
    }

    int in_channels = weights.size( 0 );
    int out_channels_per_group = weights.size( 1 );
    int out_channels = out_channels_per_group * group;
    int kernel_height = weights.size( 2 );
    int kernel_width = weights.size( 3 );

    if ( !kernel_shape.empty() )
    {
        if ( kernel_shape.size() != 2 || kernel_shape[0] != kernel_height ||
             kernel_shape[1] != kernel_width )
        {
            onnxToTorchAttributeProcessingError( node, "kernel_shape",
                                                 "Mismatch with weight tensor dimensions" );
        }
    }

    std::vector<int64_t> padding;
    if ( pads.size() == 4 )
    {
        if ( pads[0] == pads[2] && pads[1] == pads[3] )
        {
            padding = { pads[0], pads[1] };
        }
        else
        {
            padding = { pads[0], pads[1] };
        }
    }
    else if ( pads.size() == 2 )
    {
        padding = { pads[0], pads[1] };
    }
    else if ( pads.empty() )
    {
        padding = { 0, 0 };
    }
    else
    {
        onnxToTorchAttributeProcessingError(
            node, "pads", Stringf( "Invalid padding size: %lu", pads.size() ).ascii() );
    }

    std::vector<int64_t> stride_vec( strides.begin(), strides.end() );
    std::vector<int64_t> dilation_vec( dilations.begin(), dilations.end() );
    std::vector<int64_t> output_padding_vec( output_padding.begin(), output_padding.end() );
    if ( output_padding_vec.size() != 2 || output_padding_vec[0] != 0 ||
         output_padding_vec[1] != 0 )
    {
        output_padding_vec = { 0, 0 };
    }
    if ( dilation_vec[0] != 1 || dilation_vec[1] != 1 )
    {
        onnxToTorchAttributeProcessingError( node, "dilations",
                                             "ConvTranspose only supports dilation [1, 1]" );
        return nullptr;
    }
    if ( stride_vec[0] != stride_vec[1] )
    {
        onnxToTorchAttributeProcessingError(
            node, "strides", "ConvTranspose requires symmetric stride (stride[0] == stride[1])" );
        return nullptr;
    }
    if ( group != 1 )
    {
        onnxToTorchAttributeProcessingError( node, "group",
                                             "ConvTranspose only supports group = 1" );
        return nullptr;
    }

    weights = weights.contiguous().to( torch::kFloat32 ).detach().requires_grad_( false );
    if ( has_bias )
    {
        bias = bias.contiguous().to( torch::kFloat32 ).detach().requires_grad_( false );
    }
    torch::nn::ConvTranspose2dOptions convt_options( in_channels, out_channels,
                                                     { kernel_height, kernel_width } );
    convt_options.stride( stride_vec );
    convt_options.padding( padding );
    convt_options.dilation( dilation_vec );
    convt_options.output_padding( output_padding_vec );
    convt_options.groups( group );
    convt_options.bias( has_bias );

    auto convtranspose_module = torch::nn::ConvTranspose2d( convt_options );

    convtranspose_module->weight = weights;
    if ( has_bias )
    {
        convtranspose_module->bias = bias;
    }

    auto boundedNode = std::make_shared<NLR::BoundedConvTransposeNode>( convtranspose_module,
                                                                        NLR::ConvMode::MATRIX );

    return boundedNode;
}

std::shared_ptr<NLR::BoundedTorchNode>
convertConcat( const luna_onnx::NodeProto &node,
               const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
               const Map<String, luna_onnx::TensorProto> &name_to_initializer,
               const Map<String, Vector<int>> &shape_metadata )
{
    int axis = AttributeUtils::getIntAttribute( node, "axis", 1 );

    unsigned numInputs = node.input_size();
    if ( numInputs < 2 )
    {
        onnxToTorchUnexpectedNumberOfInputs( node, numInputs, 2, 100 );
        return nullptr;
    }

    auto boundedNode = std::make_shared<NLR::BoundedConcatNode>( axis, numInputs );
    std::vector<unsigned> input_sizes_along_axis;
    unsigned output_size_total = 0;

    for ( unsigned i = 0; i < numInputs; ++i )
    {
        String inputName = node.input( i );
        TensorShape inputShape;
        if ( shape_metadata.exists( inputName ) )
        {
            const Vector<int> &shape_vec = shape_metadata[inputName];
            for ( unsigned j = 0; j < shape_vec.size(); ++j )
            {
                inputShape.append( static_cast<unsigned>( shape_vec[j] ) );
            }
        }
        else
        {
            inputShape =
                extractShapeFromNode( node, name_to_input, name_to_initializer, inputName );
        }

        if ( !inputShape.empty() && axis >= 0 && axis < (int)inputShape.size() )
        {
            unsigned size_along_axis = inputShape[axis];
            input_sizes_along_axis.push_back( size_along_axis );
            output_size_total += size_along_axis;
        }
        else
        {
        }
    }

    if ( !input_sizes_along_axis.empty() )
    {
        boundedNode->setInputSizes( input_sizes_along_axis );

        String firstInputName = node.input( 0 );

        TensorShape firstShape;
        if ( shape_metadata.exists( firstInputName ) )
        {
            const Vector<int> &shape_vec = shape_metadata[firstInputName];
            for ( unsigned j = 0; j < shape_vec.size(); ++j )
            {
                firstShape.append( static_cast<unsigned>( shape_vec[j] ) );
            }
        }
        else
        {
            firstShape =
                extractShapeFromNode( node, name_to_input, name_to_initializer, firstInputName );
        }

        if ( !firstShape.empty() )
        {
            unsigned firstTensorSize = computeTensorSize( firstShape );
            unsigned concatAxisSize = firstShape[axis];
            unsigned computedOutputSize = firstTensorSize / concatAxisSize * output_size_total;
            boundedNode->setInputSize( firstTensorSize );
            boundedNode->setOutputSize( computedOutputSize );
        }
    }
    return boundedNode;
}

std::shared_ptr<NLR::BoundedTorchNode>
convertSlice( const luna_onnx::NodeProto &node, const Map<String, torch::Tensor> &constantsMap,
              const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
              const Map<String, luna_onnx::TensorProto> &name_to_initializer,
              const Map<String, Vector<int>> &shape_metadata )
{
    int64_t start = 0;
    int64_t end = INT64_MAX;
    int64_t axes = 0;
    int64_t steps = 1;

    auto extractScalarFromTensor = [&]( const torch::Tensor &tensor ) -> int64_t
    {
        if ( tensor.numel() == 0 )
        {
            return 0;
        }
        torch::Tensor flat = tensor.flatten();
        return flat[0].item<int64_t>();
    };

    auto extractScalarFromName = [&]( const String &name ) -> std::pair<bool, int64_t>
    {
        if ( constantsMap.exists( name ) )
        {
            return { true, extractScalarFromTensor( constantsMap[name] ) };
        }
        if ( name_to_initializer.exists( name ) )
        {
            const auto &tensor_proto = name_to_initializer[name];
            torch::Tensor tensor = ConstantProcessor::processInitializer( tensor_proto );
            return { true, extractScalarFromTensor( tensor ) };
        }
        return { false, 0 };
    };

    bool has_attributes = false;
    for ( const auto &attr : node.attribute() )
    {
        if ( attr.name() == "starts" )
        {
            has_attributes = true;
            if ( attr.ints_size() > 0 )
            {
                start = attr.ints( 0 );
            }
        }
        else if ( attr.name() == "ends" )
        {
            if ( attr.ints_size() > 0 )
            {
                end = attr.ints( 0 );
            }
        }
        else if ( attr.name() == "axes" )
        {
            if ( attr.ints_size() > 0 )
            {
                axes = attr.ints( 0 );
            }
        }
        else if ( attr.name() == "steps" )
        {
            if ( attr.ints_size() > 0 )
            {
                steps = attr.ints( 0 );
            }
        }
    }

    if ( !has_attributes && node.input_size() >= 3 )
    {
        if ( node.input_size() > 1 )
        {
            String startsName = node.input( 1 );
            auto [found, val] = extractScalarFromName( startsName );
            if ( found )
            {
                start = val;
            }
        }

        if ( node.input_size() > 2 )
        {
            String endsName = node.input( 2 );
            auto [found, val] = extractScalarFromName( endsName );
            if ( found )
            {
                end = val;
            }
        }

        if ( node.input_size() > 3 && !node.input( 3 ).empty() )
        {
            String axesName = node.input( 3 );
            auto [found, val] = extractScalarFromName( axesName );
            if ( found )
            {
                axes = val;
            }
        }

        if ( node.input_size() > 4 && !node.input( 4 ).empty() )
        {
            String stepsName = node.input( 4 );
            auto [found, val] = extractScalarFromName( stepsName );
            if ( found )
            {
                steps = val;
            }
        }
    }

    if ( steps != 1 && steps != -1 )
    {
        onnxToTorchAttributeProcessingError(
            node, "steps",
            Stringf( "Slice only supports steps of 1 or -1, got %lld", (long long)steps ) );
        return nullptr;
    }

    if ( axes == 0 )
    {
    }
    auto boundedNode = std::make_shared<NLR::BoundedSliceNode>( start, end, axes, steps );

    String dataInputName = node.input( 0 );
    TensorShape inputShape;

    if ( shape_metadata.exists( dataInputName ) )
    {
        const Vector<int> &shape_vec = shape_metadata[dataInputName];
        for ( unsigned j = 0; j < shape_vec.size(); ++j )
        {
            inputShape.append( static_cast<unsigned>( shape_vec[j] ) );
        }
    }
    else
    {
        inputShape =
            extractShapeFromNode( node, name_to_input, name_to_initializer, dataInputName );
    }

    if ( !inputShape.empty() )
    {
        unsigned inputSize = computeTensorSize( inputShape );
        boundedNode->setInputSize( inputSize );

        std::vector<int64_t> input_shape_vec;
        for ( unsigned j = 0; j < inputShape.size(); ++j )
        {
            input_shape_vec.push_back( static_cast<int64_t>( inputShape[j] ) );
        }
        boundedNode->setInputShape( input_shape_vec );

        int64_t axis_idx = axes;
        if ( axis_idx < 0 )
        {
            axis_idx += static_cast<int64_t>( inputShape.size() );
        }

        if ( axis_idx >= 0 && axis_idx < static_cast<int64_t>( inputShape.size() ) )
        {
            int64_t axis_size = static_cast<int64_t>( inputShape[axis_idx] );

            int64_t fixed_start = start;
            int64_t fixed_end = end;

            if ( fixed_start < 0 )
            {
                fixed_start += axis_size;
            }
            if ( fixed_end < 0 )
            {
                if ( fixed_end == -9223372036854775807LL )
                {
                    fixed_end = 0;
                }
                else
                {
                    fixed_end += axis_size;
                }
            }
            if ( steps == -1 )
            {
                std::swap( fixed_start, fixed_end );
                fixed_end = fixed_end + 1;
            }
            fixed_end = std::min( fixed_end, axis_size );
            fixed_start = std::max( fixed_start, static_cast<int64_t>( 0 ) );

            int64_t slice_length = fixed_end - fixed_start;
            if ( slice_length < 0 )
                slice_length = 0;

            unsigned outputSize = inputSize / inputShape[axis_idx] * slice_length;
            boundedNode->setOutputSize( outputSize );
        }
        else
        {
            boundedNode->setOutputSize( inputSize );
        }
    }

    return boundedNode;
}

} // namespace BoundedOperationConverter
