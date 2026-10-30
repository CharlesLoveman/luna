target_sources(${LUNA_LIB} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/AlphaCROWNAnalysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CROWNAnalysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TorchModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/conv/MatrixConvolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/conv/Patches.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedAddNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedAlphaOptimizedNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedBatchNormNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedConcatNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedConstantNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedConvNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedConvTransposeNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedFlattenNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedIdentityNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedInputNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedLinearNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedReLUNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedReshapeNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedSigmoidNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedSoftmaxNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedSliceNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/BoundedSubNode.cpp
)

target_include_directories(${LUNA_LIB} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/conv
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes
)

# The path-based TorchModel constructors live in LUNAOnnxParser since
# they delegate to OnnxToTorchParser::parse and would otherwise force
# the onnx protobuf symbols into LUNACore. The class declaration stays
# in TorchModel.h (shipped with LUNACore); only the ctor bodies move.
if(LUNA_BUILD_ONNX_PARSER)
    target_sources(${LUNA_ONNX_PARSER_LIB} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/TorchModelOnnxFactory.cpp
    )
endif()
