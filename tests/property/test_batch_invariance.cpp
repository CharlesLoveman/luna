#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "src/configuration/LunaConfiguration.h"
#include "fixtures/model_builders.h"
#include <torch/torch.h>

#include <functional>
#include <string>
#include <vector>

using namespace NLR;
using namespace test;

namespace
{

/// Lane i of a batched call must equal the single call on box i.
struct Lanes
{
    BoundedTensor<torch::Tensor> batched;
    std::vector<BoundedTensor<torch::Tensor>> singles;
};

/// Boxes with a different half-width per lane, so lanes have different unstable sets.
BoundedTensor<torch::Tensor> makeBatchBoxes( int64_t batch, const std::vector<int64_t> &sampleShape,
                                             uint64_t seed, double minWidth = 0.02,
                                             double widthSpread = 0.4 )
{
    torch::manual_seed( seed );
    std::vector<int64_t> shape = { batch };
    shape.insert( shape.end(), sampleShape.begin(), sampleShape.end() );
    torch::Tensor center = torch::rand( shape, torch::kFloat32 ) * 2.0 - 1.0;
    std::vector<int64_t> widthShape( shape.size(), 1 );
    widthShape[0] = batch;
    torch::Tensor width = minWidth + widthSpread * torch::rand( widthShape, torch::kFloat32 );
    return BoundedTensor<torch::Tensor>( center - width, center + width );
}

BoundedTensor<torch::Tensor> laneOf( const BoundedTensor<torch::Tensor> &boxes, int64_t lane )
{
    return BoundedTensor<torch::Tensor>( boxes.lower().narrow( 0, lane, 1 ).contiguous(),
                                         boxes.upper().narrow( 0, lane, 1 ).contiguous() );
}

Lanes runLanes( const std::function<std::shared_ptr<TorchModel>()> &makeModel,
                const BoundedTensor<torch::Tensor> &boxes,
                LunaConfiguration::AnalysisMethod method )
{
    Lanes result;
    const int64_t batch = boxes.lower().size( 0 );
    auto batchedModel = makeModel();
    result.batched = batchedModel->compute_bounds( boxes, nullptr, method, true, true );
    for ( int64_t lane = 0; lane < batch; ++lane )
    {
        auto singleModel = makeModel();
        result.singles.push_back(
            singleModel->compute_bounds( laneOf( boxes, lane ), nullptr, method, true, true ) );
    }
    return result;
}

void expectLanesMatch( const Lanes &lanes, double tolerance )
{
    const int64_t batch = static_cast<int64_t>( lanes.singles.size() );
    ASSERT_EQ( lanes.batched.lower().numel() % batch, 0 );
    const auto lowerByLane = lanes.batched.lower().reshape( { batch, -1 } );
    const auto upperByLane = lanes.batched.upper().reshape( { batch, -1 } );
    for ( int64_t lane = 0; lane < batch; ++lane )
    {
        const auto batchedLower = lowerByLane.narrow( 0, lane, 1 ).flatten();
        const auto batchedUpper = upperByLane.narrow( 0, lane, 1 ).flatten();
        const auto singleLower = lanes.singles[lane].lower().flatten();
        const auto singleUpper = lanes.singles[lane].upper().flatten();
        EXPECT_LT( ( batchedLower - singleLower ).abs().max().item<double>(), tolerance )
            << "lower bound differs in lane " << lane;
        EXPECT_LT( ( batchedUpper - singleUpper ).abs().max().item<double>(), tolerance )
            << "upper bound differs in lane " << lane;
    }
}

} // namespace

class BatchInvarianceTest : public ::testing::TestWithParam<int64_t>
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

    static std::shared_ptr<TorchModel> makeMlp()
    {
        torch::manual_seed( 1234 );
        return ModelBuilder::createMLP( 6, { 12, 10 }, 4, true, true );
    }
};

TEST_P( BatchInvarianceTest, MlpCrownMatchesSingleCalls )
{
    const auto boxes = makeBatchBoxes( GetParam(), { 6 }, 99 );
    expectLanesMatch( runLanes( makeMlp, boxes, LunaConfiguration::AnalysisMethod::CROWN ), 1e-4 );
}

TEST_P( BatchInvarianceTest, MlpAlphaCrownMatchesSingleCalls )
{
    const auto boxes = makeBatchBoxes( GetParam(), { 6 }, 99 );
    expectLanesMatch( runLanes( makeMlp, boxes, LunaConfiguration::AnalysisMethod::AlphaCROWN ),
                      1e-3 );
}

TEST_P( BatchInvarianceTest, MlpAlphaCrownMatchesSingleCallsWhenLanesImproveAtDifferentTimes )
{
    LunaConfiguration::ALPHA_ITERATIONS = 30;
    // A large patience keeps every run to the same iteration count, so lanes are comparable
    LunaConfiguration::EARLY_STOP_PATIENCE = 1000;
    const auto boxes = makeBatchBoxes( GetParam(), { 6 }, 7, 0.1, 0.9 );
    expectLanesMatch( runLanes( makeMlp, boxes, LunaConfiguration::AnalysisMethod::AlphaCROWN ),
                      1e-3 );
}

TEST_P( BatchInvarianceTest, EveryNodeBoundMatchesSingleCalls )
{
    // Lanes of very different widths, so a neuron is often stable in one lane and not in another
    const auto boxes = makeBatchBoxes( GetParam(), { 6 }, 21, 0.01, 1.2 );
    auto makeDeepMlp = []()
    {
        torch::manual_seed( 4321 );
        return ModelBuilder::createMLP( 6, { 14, 12, 10 }, 4, true, true );
    };
    auto batchedModel = makeDeepMlp();
    (void)batchedModel->compute_bounds( boxes, nullptr, LunaConfiguration::AnalysisMethod::CROWN,
                                        true, true );
    for ( int64_t lane = 0; lane < GetParam(); ++lane )
    {
        auto singleModel = makeDeepMlp();
        (void)singleModel->compute_bounds( laneOf( boxes, lane ), nullptr,
                                           LunaConfiguration::AnalysisMethod::CROWN, true, true );
        for ( unsigned node = 0; node < singleModel->getNumNodes(); ++node )
        {
            if ( !singleModel->hasConcreteBounds( node ) ||
                 !batchedModel->hasConcreteBounds( node ) )
                continue;
            const auto batched = batchedModel->getConcreteBounds( node );
            const auto single = singleModel->getConcreteBounds( node );
            const int64_t width = single.lower().numel();
            EXPECT_LT( ( batched.lower().reshape( { GetParam(), width } ).select( 0, lane ) -
                         single.lower().flatten() )
                           .abs()
                           .max()
                           .item<double>(),
                       1e-4 )
                << "lower bound of node " << node << " differs in lane " << lane;
            EXPECT_LT( ( batched.upper().reshape( { GetParam(), width } ).select( 0, lane ) -
                         single.upper().flatten() )
                           .abs()
                           .max()
                           .item<double>(),
                       1e-4 )
                << "upper bound of node " << node << " differs in lane " << lane;
        }
    }
}

TEST_P( BatchInvarianceTest, AlphaCrownActuallyTightensCrown )
{
    // Wide boxes so every lane has unstable neurons for alpha to work on
    const auto boxes = makeBatchBoxes( GetParam(), { 6 }, 99, 0.5, 0.5 );
    auto crown = makeMlp()->compute_bounds( boxes, nullptr,
                                            LunaConfiguration::AnalysisMethod::CROWN, true, true );
    auto alpha = makeMlp()->compute_bounds(
        boxes, nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN, true, true );

    // Never looser, and optimization must move at least one bound or the comparison is vacuous
    EXPECT_GE( ( alpha.lower() - crown.lower() ).min().item<double>(), -1e-5 );
    EXPECT_GE( ( crown.upper() - alpha.upper() ).min().item<double>(), -1e-5 );
    const double moved = ( alpha.lower() - crown.lower() ).max().item<double>() +
                         ( crown.upper() - alpha.upper() ).max().item<double>();
    EXPECT_GT( moved, 1e-5 );
}

TEST_P( BatchInvarianceTest, OutputBoundsAreBatchFirst )
{
    const int64_t batch = GetParam();
    const auto boxes = makeBatchBoxes( batch, { 6 }, 99 );
    auto model = makeMlp();
    const auto bounds = model->compute_bounds(
        boxes, nullptr, LunaConfiguration::AnalysisMethod::CROWN, true, true );
    EXPECT_EQ( bounds.lower().sizes(), ( std::vector<int64_t>{ batch, 4 } ) );
    EXPECT_EQ( bounds.upper().sizes(), ( std::vector<int64_t>{ batch, 4 } ) );
}

INSTANTIATE_TEST_SUITE_P( Batches, BatchInvarianceTest, ::testing::Values( 1, 2, 3, 4, 7, 8 ) );
