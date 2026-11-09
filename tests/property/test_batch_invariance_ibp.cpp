#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "src/configuration/LunaConfiguration.h"
#include "fixtures/model_builders.h"
#include <torch/torch.h>

#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace NLR;
using namespace test;

namespace
{

using Seeds = std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>>;

/// Boxes with a different half-width per lane, so lanes have different unstable sets.
BoundedTensor<torch::Tensor> makeBoxes( int64_t batch, const std::vector<int64_t> &sampleShape,
                                        uint64_t seed )
{
    torch::manual_seed( seed );
    std::vector<int64_t> shape = { batch };
    shape.insert( shape.end(), sampleShape.begin(), sampleShape.end() );
    torch::Tensor center = torch::rand( shape, torch::kFloat32 ) * 2.0 - 1.0;
    std::vector<int64_t> widthShape( shape.size(), 1 );
    widthShape[0] = batch;
    torch::Tensor width = 0.02 + 0.4 * torch::rand( widthShape, torch::kFloat32 );
    return BoundedTensor<torch::Tensor>( center - width, center + width );
}

torch::Tensor laneOf( const torch::Tensor &values, int64_t lane )
{
    return values.narrow( 0, lane, 1 ).contiguous();
}

BoundedTensor<torch::Tensor> laneOf( const BoundedTensor<torch::Tensor> &boxes, int64_t lane )
{
    return BoundedTensor<torch::Tensor>( laneOf( boxes.lower(), lane ),
                                         laneOf( boxes.upper(), lane ) );
}

/// Seeds that pull every non-input node toward its center, by a different fraction in each lane.
Seeds shrinkSeeds( const TorchModel &model, int64_t batch )
{
    Seeds seeds;
    for ( unsigned node : model.getAllNodeIndices() )
    {
        if ( !model.hasConcreteBounds( node ) || node == model.getOutputIndex() ||
             node == model.getInputIndices()[0] )
            continue;
        const auto bounds = model.getConcreteBounds( node );
        std::vector<int64_t> laneShape( bounds.lower().dim(), 1 );
        laneShape[0] = batch;
        torch::Tensor fraction = ( 0.05 + 0.4 * torch::arange( batch, bounds.lower().options() ) /
                                              static_cast<double>( batch ) )
                                     .reshape( laneShape );
        torch::Tensor span = bounds.upper() - bounds.lower();
        seeds[node] = { bounds.lower() + fraction * span, bounds.upper() - fraction * span };
    }
    return seeds;
}

Seeds laneOfSeeds( const Seeds &seeds, int64_t lane )
{
    Seeds result;
    for ( const auto &[node, bounds] : seeds )
        result[node] = { laneOf( bounds.first, lane ), laneOf( bounds.second, lane ) };
    return result;
}

/// Every node's bounds in lane i of the batched run must equal the single run on box i.
void expectEveryNodeMatches( const TorchModel &batched, const TorchModel &single, int64_t lane,
                             int64_t batch, double tolerance )
{
    for ( unsigned node : batched.getAllNodeIndices() )
    {
        ASSERT_EQ( batched.hasConcreteBounds( node ), single.hasConcreteBounds( node ) )
            << "node " << node;
        if ( !batched.hasConcreteBounds( node ) )
            continue;
        const auto many = batched.getConcreteBounds( node );
        const auto one = single.getConcreteBounds( node );
        ASSERT_EQ( many.lower().numel(), batch * one.lower().numel() ) << "node " << node;
        const auto manyLower = many.lower().reshape( { batch, -1 } ).narrow( 0, lane, 1 );
        const auto manyUpper = many.upper().reshape( { batch, -1 } ).narrow( 0, lane, 1 );
        EXPECT_LT( ( manyLower.flatten() - one.lower().flatten() ).abs().max().item<double>(),
                   tolerance )
            << "lower bound of node " << node << " differs in lane " << lane;
        EXPECT_LT( ( manyUpper.flatten() - one.upper().flatten() ).abs().max().item<double>(),
                   tolerance )
            << "upper bound of node " << node << " differs in lane " << lane;
    }
}

} // namespace

class BatchedIbpTest : public ::testing::TestWithParam<int64_t>
{
protected:
    void SetUp() override
    {
        at::set_num_threads( 1 );
        LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = true;
    }

    static std::shared_ptr<TorchModel> makeMlp()
    {
        torch::manual_seed( 1234 );
        return ModelBuilder::createMLP( 6, { 12, 10 }, 4, true, true );
    }

    /// The builder sizes its first dense layer from the channel count alone, so the convolutions
    /// must leave a 1x1 map: 5x5 to 3x3 to 1x1.
    static std::shared_ptr<TorchModel> makeCnn()
    {
        torch::manual_seed( 4321 );
        return ModelBuilder::createCNN( 1, 5, 5, { { 2, 3, 1, 0 }, { 3, 3, 1, 0 } }, { 8 }, 4 );
    }

    /// IBP on stacked boxes against IBP on each box alone, every node and the output.
    void expectMatchesSingleRuns( const std::function<std::shared_ptr<TorchModel>()> &makeModel,
                                  const std::vector<int64_t> &sampleShape, uint64_t seed,
                                  double tolerance = 1e-5 )
    {
        const int64_t batch = GetParam();
        const auto boxes = makeBoxes( batch, sampleShape, seed );
        auto batched = makeModel();
        const auto output = batched->runIBP( boxes );
        for ( int64_t lane = 0; lane < batch; ++lane )
        {
            auto single = makeModel();
            const auto expected = single->runIBP( laneOf( boxes, lane ) );
            EXPECT_LT( ( output.lower().reshape( { batch, -1 } ).narrow( 0, lane, 1 ).flatten() -
                         expected.lower().flatten() )
                           .abs()
                           .max()
                           .item<double>(),
                       tolerance )
                << "output lower bound differs in lane " << lane;
            expectEveryNodeMatches( *batched, *single, lane, batch, tolerance );
        }
    }
};

TEST_P( BatchedIbpTest, MlpMatchesSingleRuns )
{
    expectMatchesSingleRuns( makeMlp, { 6 }, 99 );
}

TEST_P( BatchedIbpTest, ConvMatchesSingleRuns )
{
    expectMatchesSingleRuns( makeCnn, { 1, 5, 5 }, 17 );
}

TEST_P( BatchedIbpTest, SeededLanesKeepTheirOwnSeedsWhenSeedsAreFinal )
{
    // Seeded nodes stand as given, so each lane must keep the seed row it was handed
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = false;
    const int64_t batch = GetParam();
    const auto boxes = makeBoxes( batch, { 6 }, 5 );
    auto first = makeMlp();
    (void)first->runIBP( boxes );
    const Seeds seeds = shrinkSeeds( *first, batch );

    auto batched = makeMlp();
    batched->persistIntermediateBounds( seeds );
    (void)batched->runIBP( boxes );
    for ( int64_t lane = 0; lane < batch; ++lane )
    {
        auto single = makeMlp();
        single->persistIntermediateBounds( laneOfSeeds( seeds, lane ) );
        (void)single->runIBP( laneOf( boxes, lane ) );
        expectEveryNodeMatches( *batched, *single, lane, batch, 1e-5 );
    }
}

TEST_P( BatchedIbpTest, SeededLanesMatchSingleRunsWhenOnlyDirtyNodesAreRecomputed )
{
    // Dirty nodes are recomputed from their seeded inputs while the rest are reused as seeded
    const int64_t batch = GetParam();
    const auto boxes = makeBoxes( batch, { 6 }, 8 );
    auto first = makeMlp();
    (void)first->runIBP( boxes );
    const Seeds seeds = shrinkSeeds( *first, batch );
    const std::vector<unsigned> dirty = { first->getOutputIndex() - 1 };

    auto batched = makeMlp();
    batched->persistIntermediateBounds( seeds );
    batched->setDirtyNodeIndices( dirty );
    (void)batched->runIBP( boxes );
    for ( int64_t lane = 0; lane < batch; ++lane )
    {
        auto single = makeMlp();
        single->persistIntermediateBounds( laneOfSeeds( seeds, lane ) );
        single->setDirtyNodeIndices( dirty );
        (void)single->runIBP( laneOf( boxes, lane ) );
        expectEveryNodeMatches( *batched, *single, lane, batch, 1e-5 );
    }
}

INSTANTIATE_TEST_SUITE_P( Batches, BatchedIbpTest, ::testing::Values( 1, 2, 3, 7 ) );
