#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "src/configuration/LunaConfiguration.h"
#include "fixtures/model_builders.h"
#include <torch/torch.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

using namespace NLR;
using test::ModelBuilder;

namespace
{

using Seeds = std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>>;
using Alphas = std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>>;

std::shared_ptr<TorchModel> makeModel()
{
    torch::manual_seed( 1234 );
    return ModelBuilder::createMLP( 6, { 12, 10 }, 4, true, true );
}

/// Flat bounds of every node except the input, which is what a search node seeds from its parent.
Seeds rootSeeds( const TorchModel &model )
{
    Seeds seeds;
    for ( unsigned node = 0; node < model.getNumNodes(); ++node )
    {
        const auto layer = model.getNode( node );
        if ( layer->getNodeType() == NodeType::INPUT )
            continue;
        if ( !layer->hasBounds() )
            continue;
        seeds[node] = { layer->getLower().detach().flatten().clone(),
                        layer->getUpper().detach().flatten().clone() };
    }
    return seeds;
}

/// An unstable neuron of the first ReLU's input is split, as a search node would do: even lanes
/// force it active and odd lanes force it inactive.
Seeds splitSeeds( const TorchModel &model, const Seeds &root, int64_t lane )
{
    // Tensors are handles, so the copy has to be deep or a split would edit the parent in place
    Seeds seeds;
    for ( const auto &[node, bounds] : root )
        seeds[node] = { bounds.first.clone(), bounds.second.clone() };
    const unsigned relu = model.getNodesByType( NodeType::RELU )[0];
    const unsigned preActivation = model.getDependencies( relu )[0];
    auto &lower = seeds.at( preActivation ).first;
    auto &upper = seeds.at( preActivation ).second;
    const torch::Tensor unstable = torch::nonzero( ( lower < 0 ) & ( upper > 0 ) ).flatten();
    EXPECT_GT( unstable.numel(), 0 ) << "no unstable neuron to split";
    if ( unstable.numel() == 0 )
        return seeds;
    const int64_t neuron = unstable[lane % unstable.numel()].item<int64_t>();
    if ( lane % 2 == 0 )
        lower[neuron] = 0.0;
    else
        upper[neuron] = 0.0;
    return seeds;
}

/// Stack flat per-lane seeds into one row per lane.
Seeds stackSeeds( const std::vector<Seeds> &perLane )
{
    Seeds stacked;
    for ( const auto &[node, bounds] : perLane[0] )
    {
        std::vector<torch::Tensor> lowers, uppers;
        for ( const auto &lane : perLane )
        {
            lowers.push_back( lane.at( node ).first.unsqueeze( 0 ) );
            uppers.push_back( lane.at( node ).second.unsqueeze( 0 ) );
        }
        stacked[node] = { torch::cat( lowers, 0 ), torch::cat( uppers, 0 ) };
    }
    return stacked;
}

BoundedTensor<torch::Tensor> box( int64_t lanes )
{
    torch::manual_seed( 77 );
    const torch::Tensor center = ( torch::rand( { 1, 6 } ) * 2.0 - 1.0 ).repeat( { lanes, 1 } );
    return BoundedTensor<torch::Tensor>( center - 0.4, center + 0.4 );
}

} // namespace

class SearchStateTest : public ::testing::TestWithParam<int64_t>
{
protected:
    void SetUp() override
    {
        at::set_num_threads( 1 );
        LunaConfiguration::ALPHA_ITERATIONS = 8;
        LunaConfiguration::EARLY_STOP_PATIENCE = 10;
        LunaConfiguration::OPTIMIZE_LOWER = true;
        LunaConfiguration::OPTIMIZE_UPPER = true;
        LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = true;
    }

    void TearDown() override
    {
        LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = true;
    }
};

// A search node seeds its parent's intermediate bounds and alphas and does not recompute them.
// Each lane carries its own seeds, and must come out as if it had been run alone.
void expectSeededLanesMatchSingles( int64_t batch, LunaConfiguration::AnalysisMethod method,
                                    bool warmStart = true )
{
    auto root = makeModel();
    (void)root->compute_bounds( box( 1 ), nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN,
                                true, true );
    const Seeds parentSeeds = rootSeeds( *root );
    const Alphas parentAlphas = root->getPersistedAlphas();
    ASSERT_FALSE( parentAlphas.empty() );

    std::vector<Seeds> laneSeeds;
    for ( int64_t lane = 0; lane < batch; ++lane )
        laneSeeds.push_back( splitSeeds( *root, parentSeeds, lane ) );

    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = false;

    auto batched = makeModel();
    batched->persistIntermediateBounds( stackSeeds( laneSeeds ) );
    if ( warmStart )
        batched->persistAlphas( parentAlphas );
    const auto together = batched->compute_bounds( box( batch ), nullptr, method, true, true );
    ASSERT_EQ( together.lower().sizes(), ( std::vector<int64_t>{ batch, 4 } ) );

    // Different seeds must give different answers, or the comparison below proves nothing
    if ( batch > 1 )
    {
        double spread = 0.0;
        for ( int64_t lane = 1; lane < batch; ++lane )
        {
            spread = std::max(
                spread, ( ( together.lower().select( 0, 0 ) - together.lower().select( 0, lane ) )
                              .abs()
                              .max() +
                          ( together.upper().select( 0, 0 ) - together.upper().select( 0, lane ) )
                              .abs()
                              .max() )
                            .item<double>() );
        }
        EXPECT_GT( spread, 1e-6 ) << "the lanes' seeded splits did not change the bounds";
    }

    for ( int64_t lane = 0; lane < batch; ++lane )
    {
        auto single = makeModel();
        single->persistIntermediateBounds( laneSeeds[lane] );
        if ( warmStart )
            single->persistAlphas( parentAlphas );
        const auto alone = single->compute_bounds( box( 1 ), nullptr, method, true, true );
        const double lowerGap = ( together.lower().select( 0, lane ) - alone.lower().flatten() )
                                    .abs()
                                    .max()
                                    .item<double>();
        const double upperGap = ( together.upper().select( 0, lane ) - alone.upper().flatten() )
                                    .abs()
                                    .max()
                                    .item<double>();
        EXPECT_LT( lowerGap, 1e-4 ) << "lower bound differs in lane " << lane << " by " << lowerGap;
        EXPECT_LT( upperGap, 1e-4 ) << "upper bound differs in lane " << lane << " by " << upperGap;
    }
}

TEST_P( SearchStateTest, SeededChildrenMatchSingleCallsAlpha )
{
    expectSeededLanesMatchSingles( GetParam(), LunaConfiguration::AnalysisMethod::AlphaCROWN );
}

TEST_P( SearchStateTest, SeededChildrenMatchSingleCallsAlphaColdStart )
{
    expectSeededLanesMatchSingles( GetParam(), LunaConfiguration::AnalysisMethod::AlphaCROWN,
                                   /*warmStart=*/false );
}

TEST_P( SearchStateTest, SeededChildrenMatchSingleCallsCrown )
{
    expectSeededLanesMatchSingles( GetParam(), LunaConfiguration::AnalysisMethod::CROWN );
}

// If seeds never reached the bounds, every comparison in this file would be vacuous.
TEST( SearchStateSeeds, ASplitSeedChangesTheBoundsOfASingleCall )
{
    at::set_num_threads( 1 );
    LunaConfiguration::ALPHA_ITERATIONS = 8;
    LunaConfiguration::OPTIMIZE_LOWER = true;
    LunaConfiguration::OPTIMIZE_UPPER = true;
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = true;

    auto root = makeModel();
    (void)root->compute_bounds( box( 1 ), nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN,
                                true, true );
    const Seeds parentSeeds = rootSeeds( *root );
    const Alphas parentAlphas = root->getPersistedAlphas();

    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = false;
    const auto run = [&]( const Seeds &seeds )
    {
        auto model = makeModel();
        model->persistIntermediateBounds( seeds );
        model->persistAlphas( parentAlphas );
        return model->compute_bounds( box( 1 ), nullptr,
                                      LunaConfiguration::AnalysisMethod::AlphaCROWN, true, true );
    };
    const auto parent = run( parentSeeds );
    const auto active = run( splitSeeds( *root, parentSeeds, 0 ) );
    const auto inactive = run( splitSeeds( *root, parentSeeds, 1 ) );
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = true;

    const auto spread =
        [&]( const BoundedTensor<torch::Tensor> &a, const BoundedTensor<torch::Tensor> &b )
    {
        return ( ( a.lower() - b.lower() ).abs().max() + ( a.upper() - b.upper() ).abs().max() )
            .item<double>();
    };
    EXPECT_GT( spread( parent, active ), 1e-6 ) << "forcing a neuron active changed nothing";
    EXPECT_GT( spread( parent, inactive ), 1e-6 ) << "forcing a neuron inactive changed nothing";
}

// Warm-started alphas must survive being seeded into a batch instead of being rebuilt cold.
TEST_P( SearchStateTest, WarmStartedAlphasAreKeptWhenSeedingABatch )
{
    const int64_t batch = GetParam();

    auto root = makeModel();
    (void)root->compute_bounds( box( 1 ), nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN,
                                true, true );
    const Seeds parentSeeds = rootSeeds( *root );
    const Alphas parentAlphas = root->getPersistedAlphas();

    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = false;
    LunaConfiguration::ALPHA_ITERATIONS = 1;

    // One iteration changes nothing, so the result is the parent's alphas applied to every lane
    auto warm = makeModel();
    warm->persistIntermediateBounds( stackSeeds( std::vector<Seeds>( batch, parentSeeds ) ) );
    warm->persistAlphas( parentAlphas );
    const auto warmBounds = warm->compute_bounds(
        box( batch ), nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN, true, true );

    auto alone = makeModel();
    alone->persistIntermediateBounds( parentSeeds );
    alone->persistAlphas( parentAlphas );
    const auto aloneBounds = alone->compute_bounds(
        box( 1 ), nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN, true, true );

    for ( int64_t lane = 0; lane < batch; ++lane )
    {
        EXPECT_LT( ( warmBounds.lower().select( 0, lane ) - aloneBounds.lower().flatten() )
                       .abs()
                       .max()
                       .item<double>(),
                   1e-4 )
            << "warm-started lane " << lane << " lost its alphas";
    }
}

INSTANTIATE_TEST_SUITE_P( Batches, SearchStateTest, ::testing::Values( 1, 2, 3, 4 ) );
