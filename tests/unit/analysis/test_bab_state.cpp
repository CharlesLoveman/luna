#include <gtest/gtest.h>
#include "src/engine/LunaError.h"
#include "src/engine/TorchModel.h"
#include "src/engine/CROWNAnalysis.h"
#include "src/engine/AlphaCROWNAnalysis.h"
#include "src/configuration/LunaConfiguration.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class BaBStateTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        torch::manual_seed( 42 );
        static bool threadsConfigured = false;
        if ( !threadsConfigured )
        {
            at::set_num_threads( 1 );
            at::set_num_interop_threads( 1 );
            threadsConfigured = true;
        }
        LunaConfiguration::resetToDefaults();
        LunaConfiguration::ANALYSIS_METHOD = LunaConfiguration::AnalysisMethod::AlphaCROWN;
        LunaConfiguration::ALPHA_ITERATIONS = 5;
        LunaConfiguration::OPTIMIZE_LOWER = true;
        LunaConfiguration::OPTIMIZE_UPPER = false;
        LunaConfiguration::STOP_CROWN_ON_VERIFIED = false;
        LunaConfiguration::STOP_ALPHA_ON_VERIFIED = false;
    }
};

TEST_F( BaBStateTest, loadState_setsInputBounds )
{
    auto model = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );
    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 3 } ), 0.1 );

    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> emptyBounds;
    std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>> emptyAlphas;

    model->loadState( inputBounds, emptyBounds, emptyAlphas );

    EXPECT_TRUE( model->hasInputBounds() );
    auto retrieved = model->getInputBounds();
    EXPECT_TRUE( torch::allclose( retrieved.lower(), inputBounds.lower() ) );
    EXPECT_TRUE( torch::allclose( retrieved.upper(), inputBounds.upper() ) );
}

TEST_F( BaBStateTest, computeBoundsAcceptsMultiBatchAtPublicBoundary )
{
    auto model = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );
    BoundedTensor<torch::Tensor> inputBounds( torch::zeros( { 2, 3 } ), torch::ones( { 2, 3 } ) );
    auto bounds =
        model->compute_bounds( inputBounds, nullptr, LunaConfiguration::AnalysisMethod::CROWN );
    EXPECT_EQ( bounds.lower().sizes(), ( std::vector<int64_t>{ 2, 2 } ) );
    EXPECT_EQ( bounds.upper().sizes(), ( std::vector<int64_t>{ 2, 2 } ) );
}

TEST_F( BaBStateTest, computeBoundsClearsNodeBoundsFromPriorInvocation )
{
    torch::manual_seed( 123 );
    auto reused = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );
    torch::manual_seed( 123 );
    auto fresh = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );
    auto narrow = BoundGenerator::epsilonBall( torch::zeros( { 1, 3 } ), 0.01 );
    auto wide = BoundGenerator::epsilonBall( torch::zeros( { 1, 3 } ), 1.0 );
    (void)reused->compute_bounds( narrow, nullptr, LunaConfiguration::AnalysisMethod::CROWN );
    auto reusedResult =
        reused->compute_bounds( wide, nullptr, LunaConfiguration::AnalysisMethod::CROWN );
    auto freshResult =
        fresh->compute_bounds( wide, nullptr, LunaConfiguration::AnalysisMethod::CROWN );
    EXPECT_TRUE(
        TensorComparator::allClose( reusedResult.lower(), freshResult.lower(), 1e-5, 1e-5 ) );
    EXPECT_TRUE(
        TensorComparator::allClose( reusedResult.upper(), freshResult.upper(), 1e-5, 1e-5 ) );
}

TEST_F( BaBStateTest, computeBoundsStartsCleanAfterFailedInvocation )
{
    auto reused = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );
    auto narrow = BoundGenerator::epsilonBall( torch::zeros( { 1, 3 } ), 0.01 );
    // Four elements are not a whole number of three-element samples
    BoundedTensor<torch::Tensor> wrongSampleSize( torch::zeros( { 1, 4 } ),
                                                  torch::ones( { 1, 4 } ) );
    (void)reused->compute_bounds( narrow, nullptr, LunaConfiguration::AnalysisMethod::CROWN );
    ASSERT_TRUE( reused->hasFinalAnalysisBounds() );
    EXPECT_THROW( reused->compute_bounds( wrongSampleSize, nullptr,
                                          LunaConfiguration::AnalysisMethod::CROWN ),
                  LunaError );
    EXPECT_FALSE( reused->hasFinalAnalysisBounds() );
    for ( unsigned i = 0; i < reused->getNumNodes(); ++i )
    {
        EXPECT_FALSE( reused->hasConcreteBounds( i ) );
    }
}

TEST_F( BaBStateTest, loadState_persistsIntermediateBounds )
{
    auto model = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );
    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 3 } ), 0.1 );

    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> intermBounds;
    intermBounds[2] = { torch::zeros( { 1, 5 } ) - 0.5, torch::zeros( { 1, 5 } ) + 0.5 };

    std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>> emptyAlphas;

    model->loadState( inputBounds, intermBounds, emptyAlphas );

    EXPECT_TRUE( model->hasPersistedIntermediateBounds() );
    auto &persisted = model->getPersistedIntermediateBounds();
    EXPECT_TRUE( persisted.count( 2 ) > 0 );
    EXPECT_TRUE( torch::allclose( persisted.at( 2 ).first, intermBounds[2].first ) );
}

TEST_F( BaBStateTest, loadState_persistsAlphas )
{
    auto model = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );
    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 3 } ), 0.1 );

    AlphaParameters ap;
    ap.alpha = torch::rand( { 2, 1, 1, 3 } );
    ap.unstableMask = torch::ones( { 5 }, torch::kBool );
    ap.unstableIndices = torch::arange( 3, torch::kLong );
    ap.specDim = 1;
    ap.batchDim = 1;
    ap.outDim = 5;
    ap.numUnstable = 3;

    std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>> alphaMap;
    alphaMap[2]["/output"] = ap;

    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> emptyBounds;
    model->loadState( inputBounds, emptyBounds, alphaMap );

    EXPECT_TRUE( model->hasPersistedAlphas() );
    auto &persisted = model->getPersistedAlphas();
    EXPECT_TRUE( persisted.count( 2 ) > 0 );
    EXPECT_TRUE( persisted.at( 2 ).count( "/output" ) > 0 );
    EXPECT_EQ( persisted.at( 2 ).at( "/output" ).alpha.sizes(), ap.alpha.sizes() );
}

TEST_F( BaBStateTest, loadProp_setsSpecificationMatrix )
{
    auto model = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );

    // 2D C matrix: y[0] - y[1] >= 0
    torch::Tensor C = torch::zeros( { 1, 2 } );
    C[0][0] = 1.0;
    C[0][1] = -1.0;

    model->loadProp( C );
    EXPECT_TRUE( model->hasSpecificationMatrix() );
    auto spec = model->getSpecificationMatrix();
    EXPECT_EQ( spec.dim(), 3 );
    EXPECT_EQ( spec.size( 0 ), 1 );
    EXPECT_EQ( spec.size( 2 ), 2 );
}

TEST_F( BaBStateTest, getState_afterCROWN_returnsAllBounds )
{
    auto model = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );
    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 3 } ), 0.1 );
    model->setInputBounds( inputBounds );

    model->runCROWN( inputBounds );

    auto state = model->getState();
    EXPECT_GT( state.allBounds.size(), 0u );

    // Should have bounds for at least input and output nodes
    bool hasOutputBounds = false;
    for ( auto &[idx, pair] : state.allBounds )
    {
        EXPECT_GT( pair.first.numel(), 0 );
        EXPECT_GT( pair.second.numel(), 0 );
        // upper >= lower
        EXPECT_TRUE( torch::all( pair.second >= pair.first - 1e-6 ).item<bool>() );
        if ( idx == model->getOutputIndex() )
            hasOutputBounds = true;
    }
    EXPECT_TRUE( hasOutputBounds );
}

TEST_F( BaBStateTest, incrementalBoundsReuseCleanNodesAndMatchFullRecomputation )
{
    auto model = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );
    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 3 } ), 0.1 );
    auto fullResult =
        model->compute_bounds( inputBounds, nullptr, LunaConfiguration::AnalysisMethod::CROWN );
    const auto parentBounds = model->getState().allBounds;

    model->persistIntermediateBounds( parentBounds );
    model->setDirtyNodeIndices( { model->getOutputIndex() } );
    auto incrementalResult =
        model->compute_bounds( inputBounds, nullptr, LunaConfiguration::AnalysisMethod::CROWN );
    const auto work = model->getForwardBoundWorkStats();
    EXPECT_GT( work.reusedNodes, 0u );
    EXPECT_LT( work.recomputedNodes, model->getNumNodes() );
    EXPECT_TRUE(
        TensorComparator::allClose( incrementalResult.lower(), fullResult.lower(), 1e-5, 1e-5 ) );
    EXPECT_TRUE(
        TensorComparator::allClose( incrementalResult.upper(), fullResult.upper(), 1e-5, 1e-5 ) );

    model->persistIntermediateBounds( parentBounds );
    std::vector<unsigned> allDirtyNodes;
    for ( unsigned node = 0; node < model->getNumNodes(); ++node )
    {
        allDirtyNodes.push_back( node );
    }
    model->setDirtyNodeIndices( allDirtyNodes );
    auto referenceResult =
        model->compute_bounds( inputBounds, nullptr, LunaConfiguration::AnalysisMethod::CROWN );
    EXPECT_EQ( model->getForwardBoundWorkStats().reusedNodes, 0u );
    EXPECT_TRUE( TensorComparator::allClose( incrementalResult.lower(), referenceResult.lower(),
                                             1e-5, 1e-5 ) );
    EXPECT_TRUE( TensorComparator::allClose( incrementalResult.upper(), referenceResult.upper(),
                                             1e-5, 1e-5 ) );
}

TEST_F( BaBStateTest, getState_afterAlphaCROWN_returnsAlphas )
{
    auto model = ModelBuilder::createMLP( 3, { 5, 5 }, 2, true, false );
    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 3 } ), 0.1 );
    model->setInputBounds( inputBounds );

    model->runAlphaCROWN( inputBounds, true, false );

    auto state = model->getState();

    // Should have bounds
    EXPECT_GT( state.allBounds.size(), 0u );

    // Should have alphas (model has ReLU layers)
    EXPECT_GT( state.alphas.size(), 0u );
    for ( auto &[nodeIdx, perStart] : state.alphas )
    {
        for ( auto &[startKey, ap] : perStart )
        {
            EXPECT_TRUE( ap.alpha.defined() );
            EXPECT_GT( ap.alpha.numel(), 0 );
        }
    }
}

TEST_F( BaBStateTest, clearPersistedState_removesAll )
{
    auto model = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );
    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 3 } ), 0.1 );

    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> intermBounds;
    intermBounds[2] = { torch::zeros( { 1, 5 } ) - 0.5, torch::zeros( { 1, 5 } ) + 0.5 };

    AlphaParameters ap;
    ap.alpha = torch::rand( { 2, 1, 1, 3 } );
    std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>> alphaMap;
    alphaMap[2]["/out"] = ap;

    model->loadState( inputBounds, intermBounds, alphaMap );
    EXPECT_TRUE( model->hasPersistedAlphas() );
    EXPECT_TRUE( model->hasPersistedIntermediateBounds() );

    model->clearPersistedState();
    EXPECT_FALSE( model->hasPersistedAlphas() );
    EXPECT_FALSE( model->hasPersistedIntermediateBounds() );
}

TEST_F( BaBStateTest, baselineRegression_CROWNMatchesWithLoadProp )
{
    auto model = ModelBuilder::createMLP( 5, { 10 }, 3, true, false );
    auto inputBounds = BoundGenerator::epsilonBall( torch::randn( { 1, 5 } ), 0.1 );

    // Old path: set spec + run
    torch::Tensor C = torch::eye( 3 );
    model->setInputBounds( inputBounds );
    model->setSpecificationMatrix( C.unsqueeze( 1 ) );
    auto baselineBounds = model->runCROWN( inputBounds );

    // New path: use loadProp + run on same model
    model->clearConcreteBounds();
    model->loadProp( C );
    auto newBounds = model->runCROWN( inputBounds );

    EXPECT_TRUE(
        TensorComparator::allClose( baselineBounds.lower(), newBounds.lower(), 1e-4, 1e-4 ) );
    EXPECT_TRUE(
        TensorComparator::allClose( baselineBounds.upper(), newBounds.upper(), 1e-4, 1e-4 ) );
}

TEST_F( BaBStateTest, alphaCROWN_warmStartAlphas_reproducesSoundBounds )
{
    auto model = ModelBuilder::createMLP( 3, { 5, 5 }, 2, true, false );
    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 3 } ), 0.1 );
    model->setInputBounds( inputBounds );

    // First run: compute bounds and extract state
    auto result1 = model->runAlphaCROWN( inputBounds, true, false );
    auto state = model->getState();

    // Second run: seed with extracted state and recompute
    model->loadState( inputBounds, state.allBounds, state.alphas );
    auto result2 = model->runAlphaCROWN( inputBounds, true, false );

    // Warm-started result should be sound
    EXPECT_TRUE( torch::all( result2.upper() >= result2.lower() - 1e-6 ).item<bool>() );

    // Verify soundness by checking center point
    EXPECT_TRUE( SoundnessChecker::centerContainedInBounds( inputBounds, result2, *model ) );
}

TEST_F( BaBStateTest, repeatedLoadStateAndRun )
{
    auto model = ModelBuilder::createMLP( 3, { 5, 5 }, 2, true, false );

    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> intermBounds;
    std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>> alphas;

    // Simulate 3 BaB iterations with narrowing bounds
    for ( int iter = 0; iter < 3; ++iter )
    {
        float eps = 0.3f - iter * 0.08f;
        auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 3 } ), eps );

        model->loadState( inputBounds, intermBounds, alphas );
        auto result = model->runAlphaCROWN( inputBounds, true, false );

        // Extract state for next iteration
        auto state = model->getState();
        intermBounds = state.allBounds;
        alphas = state.alphas;

        // Verify soundness each iteration
        EXPECT_TRUE( torch::all( result.upper() >= result.lower() - 1e-6 ).item<bool>() );
        EXPECT_TRUE( SoundnessChecker::centerContainedInBounds( inputBounds, result, *model ) );
    }
}

// BaB reuses one input box, so the center forward pass that caches node shapes runs once per box
TEST_F( BaBStateTest, centerForwardPassRunsOncePerInputBox )
{
    auto model = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );
    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 3 } ), 0.1 );

    model->compute_bounds( inputBounds, nullptr, LunaConfiguration::AnalysisMethod::CROWN );
    model->compute_bounds( inputBounds, nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN );
    model->compute_bounds( inputBounds, nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN );
    EXPECT_EQ( model->getShapeCachePassCount(), 1u );

    auto shiftedBounds = BoundGenerator::epsilonBall( torch::ones( { 1, 3 } ), 0.1 );
    model->compute_bounds( shiftedBounds, nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN );
    EXPECT_EQ( model->getShapeCachePassCount(), 2u );
}

// A search child starts from its parent's alphas, so it runs on its own step count and rate, as
// alpha-beta-CROWN does with a smaller learning rate than its cold root
TEST_F( BaBStateTest, warmStartRunsOnTheWarmIterationCountOnceAlphasArePersisted )
{
    auto model = ModelBuilder::createMLP( 3, { 5, 5 }, 2, true, true );
    auto inputBounds = BoundGenerator::randomBounds( { 1, 3 }, -1.0, 1.0, 1.0 );
    LunaConfiguration::ALPHA_ITERATIONS = 6;
    LunaConfiguration::WARM_ALPHA_ITERATIONS = 2;
    // Patience that never runs out, so the limit alone decides how many iterations run
    LunaConfiguration::EARLY_STOP_PATIENCE = 1000;

    model->compute_bounds( inputBounds, nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN );
    EXPECT_EQ( LunaConfiguration::LAST_LOWER_ALPHA_ITERATIONS, 6u ) << "a cold call is not warm";
    ASSERT_TRUE( model->hasPersistedAlphas() );

    model->compute_bounds( inputBounds, nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN );
    EXPECT_EQ( LunaConfiguration::LAST_LOWER_ALPHA_ITERATIONS, 2u ) << "a warm call is";
}

TEST_F( BaBStateTest, unsetWarmSettingsLeaveAWarmStartOnTheColdOnes )
{
    auto model = ModelBuilder::createMLP( 3, { 5, 5 }, 2, true, true );
    auto inputBounds = BoundGenerator::randomBounds( { 1, 3 }, -1.0, 1.0, 1.0 );
    LunaConfiguration::ALPHA_ITERATIONS = 6;
    LunaConfiguration::EARLY_STOP_PATIENCE = 1000;

    model->compute_bounds( inputBounds, nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN );
    model->compute_bounds( inputBounds, nullptr, LunaConfiguration::AnalysisMethod::AlphaCROWN );
    EXPECT_EQ( LunaConfiguration::LAST_LOWER_ALPHA_ITERATIONS, 6u );
}
