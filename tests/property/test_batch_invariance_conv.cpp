#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "src/configuration/LunaConfiguration.h"
#include <torch/torch.h>

#include <string>
#include <vector>

using namespace NLR;

namespace
{

enum class Zoo
{
    ConvRelu,
    ConvBatchNorm,
    Residual,
    // Large enough that a backward pass from conv2 uses conv patches
    ConvLarge
};

std::string zooName( Zoo zoo )
{
    switch ( zoo )
    {
    case Zoo::ConvRelu:
        return "ConvRelu";
    case Zoo::ConvBatchNorm:
        return "ConvBatchNorm";
    case Zoo::Residual:
        return "Residual";
    case Zoo::ConvLarge:
        return "ConvLarge";
    }
    return "";
}

/// Per-sample input shape, 2 channels of a square image.
std::vector<int64_t> sampleShape( Zoo zoo )
{
    return zoo == Zoo::ConvLarge ? std::vector<int64_t>{ 2, 24, 24 }
                                 : std::vector<int64_t>{ 2, 6, 6 };
}

TorchGraphNode makeNode( unsigned index, const std::string &name, TorchGraphOperation operation,
                         std::vector<unsigned> inputs, std::vector<int64_t> shape )
{
    TorchGraphNode node;
    node.index = index;
    node.name = name;
    node.operation = operation;
    node.inputs = std::move( inputs );
    node.shape = std::move( shape );
    return node;
}

TorchGraphNode makeConv( unsigned index, const std::string &name, unsigned input,
                         int64_t inChannels, int64_t outChannels, int64_t stride,
                         std::vector<int64_t> outShape )
{
    auto node = makeNode( index, name, TorchGraphOperation::Conv2d, { input }, outShape );
    node.weight = torch::randn( { outChannels, inChannels, 3, 3 } ) * 0.3;
    node.bias = torch::randn( { outChannels } ) * 0.1;
    node.strides = { stride, stride };
    node.padding = { 1, 1 };
    node.dilations = { 1, 1 };
    return node;
}

TorchGraphNode makeLinear( unsigned index, unsigned input, int64_t inSize, int64_t outSize )
{
    auto node = makeNode( index, "linear" + std::to_string( index ), TorchGraphOperation::Linear,
                          { input }, { outSize } );
    node.weight = torch::randn( { outSize, inSize } ) * 0.3;
    node.bias = torch::randn( { outSize } ) * 0.1;
    return node;
}

std::shared_ptr<TorchModel> makeZooModel( Zoo zoo )
{
    torch::manual_seed( 2024 );
    TorchComputationGraph graph;
    auto &nodes = graph.nodes;
    nodes.push_back( makeNode( 0, "x", TorchGraphOperation::Input, {}, sampleShape( zoo ) ) );

    if ( zoo == Zoo::ConvRelu )
    {
        nodes.push_back( makeConv( 1, "conv1", 0, 2, 4, 1, { 4, 6, 6 } ) );
        nodes.push_back( makeNode( 2, "relu1", TorchGraphOperation::Relu, { 1 }, { 4, 6, 6 } ) );
        nodes.push_back( makeConv( 3, "conv2", 2, 4, 3, 2, { 3, 3, 3 } ) );
        nodes.push_back( makeNode( 4, "relu2", TorchGraphOperation::Relu, { 3 }, { 3, 3, 3 } ) );
        nodes.push_back( makeNode( 5, "flat", TorchGraphOperation::Flatten, { 4 }, { 27 } ) );
        nodes.push_back( makeLinear( 6, 5, 27, 5 ) );
    }
    else if ( zoo == Zoo::ConvBatchNorm )
    {
        nodes.push_back( makeConv( 1, "conv1", 0, 2, 4, 1, { 4, 6, 6 } ) );
        auto batchNorm = makeNode( 2, "bn", TorchGraphOperation::BatchNorm, { 1 }, { 4, 6, 6 } );
        batchNorm.scale = torch::rand( { 4 } ) + 0.5;
        batchNorm.offset = torch::randn( { 4 } ) * 0.2;
        batchNorm.mean = torch::randn( { 4 } ) * 0.1;
        batchNorm.variance = torch::rand( { 4 } ) + 0.5;
        nodes.push_back( batchNorm );
        nodes.push_back( makeNode( 3, "relu1", TorchGraphOperation::Relu, { 2 }, { 4, 6, 6 } ) );
        nodes.push_back( makeNode( 4, "flat", TorchGraphOperation::Flatten, { 3 }, { 144 } ) );
        nodes.push_back( makeLinear( 5, 4, 144, 5 ) );
    }
    else if ( zoo == Zoo::ConvLarge )
    {
        nodes.push_back( makeConv( 1, "conv1", 0, 2, 8, 1, { 8, 24, 24 } ) );
        nodes.push_back( makeNode( 2, "relu1", TorchGraphOperation::Relu, { 1 }, { 8, 24, 24 } ) );
        nodes.push_back( makeConv( 3, "conv2", 2, 8, 8, 1, { 8, 24, 24 } ) );
        nodes.push_back( makeNode( 4, "relu2", TorchGraphOperation::Relu, { 3 }, { 8, 24, 24 } ) );
        nodes.push_back( makeNode( 5, "flat", TorchGraphOperation::Flatten, { 4 }, { 4608 } ) );
        nodes.push_back( makeLinear( 6, 5, 4608, 5 ) );
    }
    else
    {
        nodes.push_back( makeConv( 1, "conv1", 0, 2, 4, 1, { 4, 6, 6 } ) );
        nodes.push_back( makeNode( 2, "relu1", TorchGraphOperation::Relu, { 1 }, { 4, 6, 6 } ) );
        nodes.push_back( makeConv( 3, "conv2", 2, 4, 4, 1, { 4, 6, 6 } ) );
        nodes.push_back( makeNode( 4, "add", TorchGraphOperation::Add, { 2, 3 }, { 4, 6, 6 } ) );
        nodes.push_back( makeNode( 5, "relu2", TorchGraphOperation::Relu, { 4 }, { 4, 6, 6 } ) );
        nodes.push_back( makeNode( 6, "flat", TorchGraphOperation::Flatten, { 5 }, { 144 } ) );
        nodes.push_back( makeLinear( 7, 6, 144, 5 ) );
    }
    graph.input_index = 0;
    graph.output_index = static_cast<unsigned>( nodes.size() - 1 );
    return TorchModel::fromGraph( graph );
}

BoundedTensor<torch::Tensor> makeBoxes( int64_t batch, Zoo zoo = Zoo::ConvRelu )
{
    torch::manual_seed( 31 );
    std::vector<int64_t> shape = { batch };
    for ( const int64_t dimension : sampleShape( zoo ) )
        shape.push_back( dimension );
    torch::Tensor center = torch::rand( shape ) * 2.0 - 1.0;
    torch::Tensor width = 0.02 + 0.3 * torch::rand( { batch, 1, 1, 1 } );
    return BoundedTensor<torch::Tensor>( center - width, center + width );
}

BoundedTensor<torch::Tensor> laneOf( const BoundedTensor<torch::Tensor> &boxes, int64_t lane )
{
    return BoundedTensor<torch::Tensor>( boxes.lower().narrow( 0, lane, 1 ).contiguous(),
                                         boxes.upper().narrow( 0, lane, 1 ).contiguous() );
}

void expectLanesMatchSingles( Zoo zoo, int64_t batch, LunaConfiguration::AnalysisMethod method,
                              double tolerance )
{
    const auto boxes = makeBoxes( batch, zoo );
    const auto batched = makeZooModel( zoo )->compute_bounds( boxes, nullptr, method, true, true );
    ASSERT_EQ( batched.lower().sizes(), ( std::vector<int64_t>{ batch, 5 } ) );
    for ( int64_t lane = 0; lane < batch; ++lane )
    {
        const auto single = makeZooModel( zoo )->compute_bounds( laneOf( boxes, lane ), nullptr,
                                                                 method, true, true );
        // Bounds in the thousands only resolve to a float32 ulp, so allow a few ulps of the scale
        const double scale = std::max( 1.0, std::max( single.lower().abs().max().item<double>(),
                                                      single.upper().abs().max().item<double>() ) );
        const double allowed = tolerance + 1e-6 * scale;
        // Positive means the batched lower bound is tighter, negative means looser
        const auto lowerGain = batched.lower().select( 0, lane ) - single.lower().flatten();
        const auto upperGain = single.upper().flatten() - batched.upper().select( 0, lane );
        EXPECT_LT( lowerGain.abs().max().item<double>(), allowed )
            << zooName( zoo ) << " lower bound differs in lane " << lane << ", tighter by at most "
            << lowerGain.max().item<double>() << " and looser by at most "
            << -lowerGain.min().item<double>();
        EXPECT_LT( upperGain.abs().max().item<double>(), allowed )
            << zooName( zoo ) << " upper bound differs in lane " << lane << ", tighter by at most "
            << upperGain.max().item<double>() << " and looser by at most "
            << -upperGain.min().item<double>();
    }
}

} // namespace

class ConvBatchInvarianceTest : public ::testing::TestWithParam<std::tuple<Zoo, int64_t>>
{
protected:
    void SetUp() override
    {
        at::set_num_threads( 1 );
        LunaConfiguration::ALPHA_ITERATIONS = 5;
        LunaConfiguration::EARLY_STOP_PATIENCE = 10;
        LunaConfiguration::OPTIMIZE_LOWER = true;
        LunaConfiguration::OPTIMIZE_UPPER = true;
    }

    // Alpha on the large net takes tens of seconds per lane on CPU, CROWN covers every batch size
    void skipSlowAlphaCases() const
    {
        if ( std::get<0>( GetParam() ) == Zoo::ConvLarge && std::get<1>( GetParam() ) > 3 )
            GTEST_SKIP() << "alpha-CROWN on the large net is too slow on CPU beyond batch 3";
    }
};

TEST_P( ConvBatchInvarianceTest, CrownMatchesSingleCalls )
{
    expectLanesMatchSingles( std::get<0>( GetParam() ), std::get<1>( GetParam() ),
                             LunaConfiguration::AnalysisMethod::CROWN, 1e-4 );
}

TEST_P( ConvBatchInvarianceTest, AlphaCrownMatchesSingleCalls )
{
    skipSlowAlphaCases();
    expectLanesMatchSingles( std::get<0>( GetParam() ), std::get<1>( GetParam() ),
                             LunaConfiguration::AnalysisMethod::AlphaCROWN, 1e-3 );
}

TEST_P( ConvBatchInvarianceTest, AlphaCrownOnIdenticalLanesMatchesTheSingleCall )
{
    skipSlowAlphaCases();
    // With the same box in every lane the shared unstable set is each lane's own set
    const Zoo zoo = std::get<0>( GetParam() );
    const int64_t batch = std::get<1>( GetParam() );
    const auto one = makeBoxes( 1, zoo );
    const BoundedTensor<torch::Tensor> boxes( one.lower().repeat( { batch, 1, 1, 1 } ),
                                              one.upper().repeat( { batch, 1, 1, 1 } ) );
    const auto batched = makeZooModel( zoo )->compute_bounds(
        boxes, nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN, true, true );
    const auto single = makeZooModel( zoo )->compute_bounds(
        one, nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN, true, true );
    const double scale = std::max( 1.0, single.lower().abs().max().item<double>() );
    for ( int64_t lane = 0; lane < batch; ++lane )
    {
        EXPECT_LT( ( batched.lower().select( 0, lane ) - single.lower().flatten() )
                       .abs()
                       .max()
                       .item<double>(),
                   1e-5 + 1e-6 * scale )
            << zooName( zoo ) << " lane " << lane;
    }
}

TEST_P( ConvBatchInvarianceTest, SeededIntermediatesOfConvolutionsKeepTheirShape )
{
    // A search node seeds every node with the bounds of an earlier pass over the same network
    const Zoo zoo = std::get<0>( GetParam() );
    const int64_t batch = std::get<1>( GetParam() );
    const auto boxes = makeBoxes( batch, zoo );
    auto model = makeZooModel( zoo );
    const auto first = model->compute_bounds(
        boxes, nullptr, LunaConfiguration::AnalysisMethod::CROWN, true, true );

    // A root solve recomputes every node and clips the result to the seeds
    const bool previous = LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS;
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = true;
    model->persistIntermediateBounds( model->getState().allBounds );
    BoundedTensor<torch::Tensor> seeded;
    EXPECT_NO_THROW( seeded = model->compute_bounds(
                         boxes, nullptr, LunaConfiguration::AnalysisMethod::CROWN, true, true ) );
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = previous;

    // Seeds that are the pass's own bounds change nothing
    ASSERT_EQ( seeded.lower().sizes(), first.lower().sizes() );
    EXPECT_LT( ( seeded.lower() - first.lower() ).abs().max().item<double>(), 1e-4 )
        << zooName( zoo );
}

TEST_P( ConvBatchInvarianceTest, FlatSeedsAreClippedToTheShapeOfAConvolution )
{
    // A caller that keeps bounds as plain vectors seeds a convolution with one flat row per lane
    const Zoo zoo = std::get<0>( GetParam() );
    const int64_t batch = std::get<1>( GetParam() );
    const auto boxes = makeBoxes( batch, zoo );
    auto model = makeZooModel( zoo );
    const auto first = model->compute_bounds(
        boxes, nullptr, LunaConfiguration::AnalysisMethod::CROWN, true, true );

    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> flat;
    for ( const auto &[node, bounds] : model->getState().allBounds )
        flat[node] = { bounds.first.reshape( { batch, -1 } ),
                       bounds.second.reshape( { batch, -1 } ) };

    const bool previous = LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS;
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = true;
    model->persistIntermediateBounds( flat );
    BoundedTensor<torch::Tensor> seeded;
    EXPECT_NO_THROW( seeded = model->compute_bounds(
                         boxes, nullptr, LunaConfiguration::AnalysisMethod::CROWN, true, true ) );
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = previous;

    ASSERT_EQ( seeded.lower().sizes(), first.lower().sizes() );
    EXPECT_LT( ( seeded.lower() - first.lower() ).abs().max().item<double>(), 1e-3 )
        << zooName( zoo );
}

TEST_P( ConvBatchInvarianceTest, AlphaCrownTightensCrownSoTheComparisonIsNotVacuous )
{
    skipSlowAlphaCases();
    const Zoo zoo = std::get<0>( GetParam() );
    const int64_t batch = std::get<1>( GetParam() );
    torch::manual_seed( 5 );
    std::vector<int64_t> shape = { batch };
    for ( const int64_t dimension : sampleShape( zoo ) )
        shape.push_back( dimension );
    torch::Tensor center = torch::rand( shape ) * 2.0 - 1.0;
    const BoundedTensor<torch::Tensor> boxes( center - 0.4, center + 0.4 );
    const auto crown = makeZooModel( zoo )->compute_bounds(
        boxes, nullptr, LunaConfiguration::AnalysisMethod::CROWN, true, true );
    const auto alpha = makeZooModel( zoo )->compute_bounds(
        boxes, nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN, true, true );
    EXPECT_GE( ( alpha.lower() - crown.lower() ).min().item<double>(), -1e-5 );
    EXPECT_GE( ( crown.upper() - alpha.upper() ).min().item<double>(), -1e-5 );
    EXPECT_GT( ( alpha.lower() - crown.lower() ).max().item<double>() +
                   ( crown.upper() - alpha.upper() ).max().item<double>(),
               1e-5 )
        << zooName( zoo ) << " alpha-CROWN did not move any bound";
}

INSTANTIATE_TEST_SUITE_P( Zoo, ConvBatchInvarianceTest,
                          ::testing::Combine( ::testing::Values( Zoo::ConvRelu, Zoo::ConvBatchNorm,
                                                                 Zoo::Residual, Zoo::ConvLarge ),
                                              ::testing::Values( 1, 2, 3, 4, 8 ) ),
                          []( const ::testing::TestParamInfo<std::tuple<Zoo, int64_t>> &info )
                          {
                              return zooName( std::get<0>( info.param ) ) + "Batch" +
                                     std::to_string( std::get<1>( info.param ) );
                          } );
