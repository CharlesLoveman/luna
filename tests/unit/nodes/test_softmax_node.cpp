#include "gtest/gtest.h"
#include <gtest/gtest.h>
#include "TorchModel.h"
#include "model_builders.h"
#include "src/engine/nodes/BoundedSoftmaxNode.h"
#include "src/common/BoundedTensor.h"
#include "src/common/Vector.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>
#include "fixtures/test_utils.h"

using namespace NLR;
using namespace test;

std::shared_ptr<NLR::TorchModel> createMLPSoftmax( unsigned inputSize,
                                                   const std::vector<unsigned> &hiddenSizes,
                                                   unsigned outputSize )
{

    ModelBuilder builder;
    builder.addInput( inputSize, "input" );

    unsigned currentSize = inputSize;
    for ( size_t i = 0; i < hiddenSizes.size(); ++i )
    {
        builder.addLinear( currentSize, hiddenSizes[i], "fc" + std::to_string( i + 1 ) );
        builder.addSoftmax( "softmax" + std::to_string( i + 1 ) );
        currentSize = hiddenSizes[i];
    }

    builder.addLinear( currentSize, outputSize, "output" );
    return builder.build();
}

class SoftmaxNodeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        static bool threadsConfigured = false;
        if ( !threadsConfigured )
        {
            at::set_num_threads( 1 );
            at::set_num_interop_threads( 1 );
            threadsConfigured = true;
        }
        LunaConfiguration::ALPHA_ITERATIONS = 5;

        torch::nn::Softmax softmax( torch::nn::SoftmaxOptions( 1 ) );

        _softmaxNode = std::make_shared<BoundedSoftmaxNode>( softmax, "test_softmax" );
        _softmaxNode->setNodeIndex( 0 );
        _softmaxNode->setInputSize( 5 );
        _softmaxNode->setOutputSize( 5 );

        _model = createMLPSoftmax( 5, { 5 }, 5 );
    }

    std::shared_ptr<BoundedSoftmaxNode> _softmaxNode;
    std::shared_ptr<TorchModel> _model;
};

TEST_F( SoftmaxNodeTest, ForwardPassBasic )
{
    torch::Tensor input = torch::zeros( { 1, 5 } );
    torch::Tensor output = _softmaxNode->forward( input );

    // softmax(0) = 1/n
    torch::Tensor expected = torch::ones( { 1, 5 } ) * 0.2;
    EXPECT_TENSORS_CLOSE( output, expected, 1e-5, 1e-4 );
}

TEST_F( SoftmaxNodeTest, IBPRange )
{
    torch::Tensor lb = torch::zeros( { 1, 5 } ) - 5.0;
    torch::Tensor ub = torch::zeros( { 1, 5 } ) + 5.0;

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append( BoundedTensor<torch::Tensor>( lb, ub ) );

    auto result = _softmaxNode->computeIntervalBoundPropagation( inputBounds );

    // Softmax output should be in [0, 1]
    EXPECT_TRUE( torch::all( result.lower() >= 0 ).item<bool>() );
    EXPECT_TRUE( torch::all( result.upper() <= 1.0 ).item<bool>() );
}

TEST_F( SoftmaxNodeTest, SoundBounds )
{
    auto model = createMLPSoftmax( 5, { 10 }, 3 );
    auto inputBounds = BoundGenerator::randomBounds( { 1, 5 }, 0.0, 1.0, 0.2 );

    model->setInputBounds( inputBounds );
    auto ibpBounds = model->runIBP( inputBounds );
    auto crownBounds = model->runCROWN();
    auto alphaBounds = model->runAlphaCROWN();
    EXPECT_TRUE(
        SoundnessChecker::verifySoundnessBySampling( *model, inputBounds, ibpBounds, 500 ) );
    EXPECT_TRUE(
        SoundnessChecker::verifySoundnessBySampling( *model, inputBounds, crownBounds, 500 ) );
    EXPECT_TRUE(
        SoundnessChecker::verifySoundnessBySampling( *model, inputBounds, alphaBounds, 500 ) );
}

constexpr double BOX_RADIUS = 0.0039;

/// A box of radius BOX_RADIUS around a random center, one row per lane.
BoundedTensor<torch::Tensor> makeBoxes( const TorchModel &model, int64_t lanes )
{
    torch::manual_seed( 7 );
    std::vector<int64_t> shape = { lanes };
    for ( int64_t dimension : model.getInputSampleShape() )
        shape.push_back( dimension );
    torch::Tensor center = torch::rand( shape, torch::kFloat32 );
    return BoundedTensor<torch::Tensor>( center - BOX_RADIUS, center + BOX_RADIUS );
}

void waitForDevice()
{
    if ( torch::cuda::is_available() )
        torch::cuda::synchronize();
}

class SoftmaxNodeTestWithParam : public ::testing::TestWithParam<std::string>
{
protected:
    void SetUp() override
    {
        static bool threadsConfigured = false;
        if ( !threadsConfigured )
        {
            at::set_num_threads( 1 );
            at::set_num_interop_threads( 1 );
            threadsConfigured = true;
        }
        LunaConfiguration::ALPHA_ITERATIONS = 5;
    }
};

TEST_P( SoftmaxNodeTestWithParam, OnnxModels )
{
    std::shared_ptr<TorchModel> model;

    model = ModelBuilder::loadFromONNX( GetParam() );

    const auto boxes = makeBoxes( *model, 4 );
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = true;
    model->persistIntermediateBounds( {} );
    (void)model->runIBP( boxes );
    waitForDevice();
    auto ibpbounds = model->runIBP( boxes );
    auto crownBounds = model->runCROWN();
    auto aCrownBounds = model->runAlphaCROWN();
}

INSTANTIATE_TEST_SUITE_P( SoftmaxOnnxModels, SoftmaxNodeTestWithParam,
                          testing::Values( RESOURCES_DIR "/onnx/softmax/softmax_batch.onnx",
                                           RESOURCES_DIR
                                           "/onnx/softmax/relu_softmax_relu_batch.onnx",
                                           RESOURCES_DIR "/onnx/softmax/add_softmax_add_batch.onnx",
                                           RESOURCES_DIR "/onnx/softmax/toy_softmax.onnx" ) );
