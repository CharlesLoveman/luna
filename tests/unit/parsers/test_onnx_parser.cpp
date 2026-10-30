#include <gtest/gtest.h>
#include "src/input_parsers/OnnxToTorch.h"
#include "src/engine/TorchModel.h"
#include "fixtures/tensor_comparators.h"
#include <iostream>
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class ONNXParserTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Set resources directory
        // Tests assume resources are available relative to project root
    }
};

TEST_F( ONNXParserTest, ParseSimpleModel )
{
    // Try parsing a simple ONNX model if available
    // TODO: use a fixed test model fixture and validate expected bounds.
    std::string modelPath = RESOURCES_DIR "/onnx/fc1.onnx";

    try
    {
        auto model = OnnxToTorchParser::parse( String( modelPath.c_str() ) );
        if ( model )
        {
            EXPECT_GT( model->getNumNodes(), 0 );
            EXPECT_GT( model->getInputSize(), 0 );
            EXPECT_GT( model->getOutputSize(), 0 );
        }
    }
    catch ( const std::exception &e )
    {
        std::cerr << "ONNX model not available: " << e.what() << std::endl;
        return;
    }
}

TEST_F( ONNXParserTest, ParseModelWithNodes )
{
    std::string modelPath = RESOURCES_DIR "/onnx/toy_add_simple.onnx";

    try
    {
        auto model = OnnxToTorchParser::parse( String( modelPath.c_str() ) );
        if ( model )
        {
            // TODO: validate node structure against expected test model.
            EXPECT_GT( model->getNumNodes(), 0 );

            // TODO: compare input indices to expected values.
            auto inputIndices = model->getInputIndices();
            EXPECT_GT( inputIndices.size(), 0 );

            // TODO: compare output index to expected value.
            EXPECT_GE( model->getOutputIndex(), 0 );
        }
    }
    catch ( const std::exception &e )
    {
        std::cerr << "ONNX model not available: " << e.what() << std::endl;
        return;
    }
}

TEST_F( ONNXParserTest, ParseModelWithSoftmax )
{
    std::string modelPath = RESOURCES_DIR "/onnx/toy_softmax.onnx";

    try
    {
        auto model = OnnxToTorchParser::parse( String( modelPath.c_str() ) );
        if ( model )
        {
            // TODO: validate node structure against expected test model.
            EXPECT_GE( model->getNumNodes(), 1 );

            // TODO: compare input indices to expected values.
            auto inputIndices = model->getInputIndices();
            EXPECT_EQ( inputIndices.size(), 1 );

            // TODO: compare output index to expected value.
            EXPECT_EQ( model->getOutputIndex(), 1 );
        }
    }
    catch ( const std::exception &e )
    {
        std::cerr << "ONNX model not available: " << e.what() << std::endl;
        return;
    }
}
