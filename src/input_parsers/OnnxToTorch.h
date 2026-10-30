/*********************                                                        */
/*! \file OnnxToTorch.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __OnnxToTorchParser_h__
#define __OnnxToTorchParser_h__

#include "Map.h"
#include "Set.h"
#include "MString.h"
#include "Vector.h"
#include "onnx.proto3.pb.h"
#include "../engine/nodes/BoundedTorchNode.h"
#include "../engine/nodes/BoundedConstantNode.h"
#include "../engine/nodes/BoundedInputNode.h"
#include "../engine/nodes/BoundedLinearNode.h"
#include "../engine/nodes/BoundedReLUNode.h"
#include "../engine/nodes/BoundedIdentityNode.h"
#include "../engine/nodes/BoundedReshapeNode.h"
#include "../engine/nodes/BoundedFlattenNode.h"
#include "../engine/nodes/BoundedSubNode.h"
#include "../engine/nodes/BoundedBatchNormNode.h"

// avoid macro conflict with PyTorch
#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>
#include <memory>

namespace NLR
{
class TorchModel;
}

using TensorShape = Vector<unsigned int>;

void onnxToTorchMissingAttributeError( const luna_onnx::NodeProto &node,
                                       const String &attributeName );
void onnxToTorchUnimplementedOperationError( const luna_onnx::NodeProto &node );
void onnxToTorchUnimplementedAttributeError( const luna_onnx::NodeProto &node,
                                             const String &attributeName );
void onnxToTorchUnsupportedOperationError( const luna_onnx::NodeProto &node );
void onnxToTorchMissingNodeError( const String &missingNodeName );
void onnxToTorchUnexpectedNumberOfInputs( const luna_onnx::NodeProto &node,
                                          unsigned int actualNumberOfInputs,
                                          unsigned int lowerBound, unsigned int upperBound );
void onnxToTorchInvalidTensorShapeError( const String &nodeName, const String &reason );
void onnxToTorchUnsupportedDataTypeError( const luna_onnx::TensorProto_DataType &dataType );
void onnxToTorchInvalidConstantNodeError( const luna_onnx::NodeProto &node, const String &reason );
void onnxToTorchTopologicalSortError( const String &reason );
void onnxToTorchBoundedModuleCreationError( const String &operationType, const String &reason );
void onnxToTorchFileReadError( const String &filename, const String &reason );
void onnxToTorchModelParseError( const String &filename, const String &reason );
void onnxToTorchGraphProcessingError( const String &reason );
void onnxToTorchTensorConversionError( const String &tensorName, const String &reason );
void onnxToTorchAttributeProcessingError( const luna_onnx::NodeProto &node,
                                          const String &attributeName, const String &reason );
void onnxToTorchShapeMismatchError( const String &operation, const TensorShape &expectedShape,
                                    const TensorShape &actualShape );
void onnxToTorchDimensionMismatchError( const String &operation, unsigned int expectedDim,
                                        unsigned int actualDim );
void onnxToTorchInvalidBroadcastError( const String &operation, const TensorShape &shape1,
                                       const TensorShape &shape2 );
void onnxToTorchUnsupportedActivationError( const String &activationType );
void onnxToTorchInvalidWeightBiasError( const String &operation, const String &reason );
void onnxToTorchMemoryAllocationError( const String &operation, const String &reason );
void onnxToTorchPyTorchError( const String &operation, const String &pytorchError );

class OnnxToTorchParser
{
public:
    static std::shared_ptr<NLR::TorchModel> parse( const String &path );

private:
    OnnxToTorchParser( const String &path );
    std::shared_ptr<NLR::TorchModel> processGraph();
    luna_onnx::ModelProto _onnx_model;
};

namespace AttributeUtils
{
Map<String, torch::IValue> extractAttributes( luna_onnx::NodeProto &node );
float getFloatAttribute( const luna_onnx::NodeProto &node, const String &name,
                         float defaultValue = 0.0f );
int getIntAttribute( const luna_onnx::NodeProto &node, const String &name, int defaultValue = 0 );
Vector<int> getIntsAttribute( luna_onnx::NodeProto &node, const String &name,
                              const Vector<int> &defaultValue = {} );
String getStringAttribute( luna_onnx::NodeProto &node, const String &name,
                           const String &defaultValue = "" );
} // namespace AttributeUtils

namespace GraphUtils
{
Vector<String>
computeTopologicalOrder( const Map<String, luna_onnx::NodeProto> &name_to_node,
                         const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                         const Map<String, luna_onnx::TensorProto> &name_to_initializer );

Map<String, Set<String>> computeActivationDependencies( const luna_onnx::GraphProto &graph );

std::vector<int64_t> instantiateReshapeTemplate( const torch::Tensor &input,
                                                 const torch::Tensor &shape_tensor );
} // namespace GraphUtils

namespace ConstantProcessor
{
torch::Tensor processInitializer( const luna_onnx::TensorProto &tensor );
torch::Tensor processConstantNode( const luna_onnx::NodeProto &node );
} // namespace ConstantProcessor

namespace BoundedOperationConverter
{
TensorShape extractShapeFromNode( const luna_onnx::NodeProto &node,
                                  const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                                  const Map<String, luna_onnx::TensorProto> &name_to_initializer,
                                  const String &tensorName );
unsigned computeTensorSize( const TensorShape &shape );

std::shared_ptr<NLR::BoundedTorchNode>
convertGemm( const luna_onnx::NodeProto &node, const Map<String, torch::Tensor> &constants,
             const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
             const Map<String, luna_onnx::TensorProto> &name_to_initializer );
std::shared_ptr<NLR::BoundedTorchNode>
convertMatMul( const luna_onnx::NodeProto &node, const Map<String, torch::Tensor> &constants,
               const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
               const Map<String, luna_onnx::TensorProto> &name_to_initializer );
std::shared_ptr<NLR::BoundedTorchNode>
convertAdd( const luna_onnx::NodeProto &node, const Map<String, torch::Tensor> &constants,
            const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
            const Map<String, luna_onnx::TensorProto> &name_to_initializer,
            const Vector<std::shared_ptr<NLR::BoundedTorchNode>> &existingNodes,
            const Map<String, unsigned> &nameToIndex );
std::shared_ptr<NLR::BoundedTorchNode>
convertRelu( const luna_onnx::NodeProto &node,
             const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
             const Map<String, luna_onnx::TensorProto> &name_to_initializer );
std::shared_ptr<NLR::BoundedTorchNode>
convertSigmoid( const luna_onnx::NodeProto &node,
                const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                const Map<String, luna_onnx::TensorProto> &name_to_initializer );
std::shared_ptr<NLR::BoundedTorchNode>
convertSoftmax( const luna_onnx::NodeProto &node,
                const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                const Map<String, luna_onnx::TensorProto> &name_to_initializer );
std::shared_ptr<NLR::BoundedTorchNode>
convertIdentity( const luna_onnx::NodeProto &node,
                 const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                 const Map<String, luna_onnx::TensorProto> &name_to_initializer );
std::shared_ptr<NLR::BoundedTorchNode>
convertReshape( const luna_onnx::NodeProto &node,
                const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                const Map<String, luna_onnx::TensorProto> &name_to_initializer );
std::shared_ptr<NLR::BoundedTorchNode>
convertFlatten( const luna_onnx::NodeProto &node,
                const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                const Map<String, luna_onnx::TensorProto> &name_to_initializer );
std::shared_ptr<NLR::BoundedTorchNode>
convertSub( const luna_onnx::NodeProto &node, const Map<String, torch::Tensor> &constants,
            const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
            const Map<String, luna_onnx::TensorProto> &name_to_initializer );
std::shared_ptr<NLR::BoundedTorchNode>
convertConv( const luna_onnx::NodeProto &node, const Map<String, torch::Tensor> &constants,
             const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
             const Map<String, luna_onnx::TensorProto> &name_to_initializer,
             const Map<String, Vector<int>> &shape_metadata = Map<String, Vector<int>>() );
std::shared_ptr<NLR::BoundedTorchNode>
convertConvTranspose( const luna_onnx::NodeProto &node, const Map<String, torch::Tensor> &constants,
                      const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                      const Map<String, luna_onnx::TensorProto> &name_to_initializer );
std::shared_ptr<NLR::BoundedTorchNode>
convertBatchNormalization( const luna_onnx::NodeProto &node,
                           const Map<String, torch::Tensor> &constants,
                           const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
                           const Map<String, luna_onnx::TensorProto> &name_to_initializer );
std::shared_ptr<NLR::BoundedTorchNode>
convertConcat( const luna_onnx::NodeProto &node,
               const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
               const Map<String, luna_onnx::TensorProto> &name_to_initializer,
               const Map<String, Vector<int>> &shape_metadata );
std::shared_ptr<NLR::BoundedTorchNode>
convertSlice( const luna_onnx::NodeProto &node, const Map<String, torch::Tensor> &constants,
              const Map<String, luna_onnx::ValueInfoProto> &name_to_input,
              const Map<String, luna_onnx::TensorProto> &name_to_initializer,
              const Map<String, Vector<int>> &shape_metadata );
std::shared_ptr<NLR::BoundedTorchNode> convertConstant( const torch::Tensor &value );
} // namespace BoundedOperationConverter

#endif // __OnnxToTorchParser_h__
