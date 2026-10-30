#include <gtest/gtest.h>
#include "src/engine/nodes/BoundedSoftmaxNode.h"
#include "src/common/BoundedTensor.h"
#include "src/common/Vector.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class SoftmaxNodeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        torch::nn::Softmax softmax( torch::nn::SoftmaxOptions( 1 ) );
        _softmaxNode = std::make_shared<BoundedSoftmaxNode>( softmax, "test_softmax" );
        _softmaxNode->setNodeIndex( 0 );
        _softmaxNode->setInputSize( 5 );
        _softmaxNode->setOutputSize( 5 );
    }

    std::shared_ptr<BoundedSoftmaxNode> _softmaxNode;
};

TEST_F( SoftmaxNodeTest, ForwardPassBasic )
{
    torch::Tensor input = torch::zeros( { 1, 5 } );
    torch::Tensor output = _softmaxNode->forward( input );

    // softmax(0) = 1/n
    std::cout << output;
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
