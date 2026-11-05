#include <gtest/gtest.h>
#include "src/configuration/LunaConfiguration.h"
#include "src/engine/CROWNAnalysis.h"
#include "src/engine/conv/Patches.h"
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

namespace
{

template <class Function>
void expectRuntimeErrorContaining( Function &&function, const std::string &expected )
{
    try
    {
        function();
        FAIL() << "Expected std::runtime_error containing: " << expected;
    }
    catch ( const std::runtime_error &error )
    {
        EXPECT_NE( std::string( error.what() ).find( expected ), std::string::npos )
            << "Unexpected diagnostic: " << error.what();
    }
}

class BackwardContractNode : public BoundedTorchNode
{
public:
    BackwardContractNode( bool perturbed, unsigned outputCount, bool firstOutputDefined,
                          bool produceBias )
        : _perturbed( perturbed )
        , _outputCount( outputCount )
        , _firstOutputDefined( firstOutputDefined )
        , _produceBias( produceBias )
    {
        _nodeName = "contract";
        _input_size = 2;
        _output_size = 2;
    }

    NodeType getNodeType() const override
    {
        return NodeType::IDENTITY;
    }
    String getNodeName() const override
    {
        return _nodeName;
    }
    unsigned getNodeIndex() const override
    {
        return _nodeIndex;
    }
    torch::Tensor forward( const torch::Tensor &input ) override
    {
        return input;
    }
    BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>> &inputBounds ) override
    {
        return inputBounds[0];
    }
    void boundBackward( const BoundA &last_lA, const BoundA &last_uA,
                        const Vector<BoundedTensor<torch::Tensor>> &inputBounds,
                        Vector<Pair<BoundA, BoundA>> &outputA, torch::Tensor &lbias,
                        torch::Tensor &ubias ) override
    {
        (void)inputBounds;
        outputA.clear();
        for ( unsigned i = 0; i < _outputCount; ++i )
        {
            bool defined = i > 0 || _firstOutputDefined;
            outputA.append( Pair<BoundA, BoundA>( defined ? last_lA : BoundA(),
                                                  defined ? last_uA : BoundA() ) );
        }
        if ( _produceBias )
        {
            lbias = torch::ones( { 2, 1 } );
            ubias = torch::ones( { 2, 1 } );
        }
    }
    unsigned getInputSize() const override
    {
        return _input_size;
    }
    unsigned getOutputSize() const override
    {
        return _output_size;
    }
    bool isPerturbed() const override
    {
        return _perturbed;
    }
    void setInputSize( unsigned size ) override
    {
        _input_size = size;
    }
    void setOutputSize( unsigned size ) override
    {
        _output_size = size;
    }
    void setNodeIndex( unsigned index ) override
    {
        _nodeIndex = index;
    }
    void setNodeName( const String &name ) override
    {
        _nodeName = name;
    }

private:
    bool _perturbed;
    unsigned _outputCount;
    bool _firstOutputDefined;
    bool _produceBias;
};

std::unique_ptr<TorchModel> makeContractModel( const std::shared_ptr<BackwardContractNode> &node,
                                               unsigned dependencyCount )
{
    Vector<std::shared_ptr<BoundedTorchNode>> nodes;
    Vector<unsigned> inputs;
    Map<unsigned, Vector<unsigned>> dependencies;
    Vector<unsigned> nodeDependencies;
    for ( unsigned i = 0; i < dependencyCount; ++i )
    {
        auto input = std::make_shared<BoundedInputNode>( i, 2, Stringf( "input%u", i ) );
        input->setNodeIndex( i );
        input->setOutputSize( 2 );
        nodes.append( input );
        inputs.append( i );
        dependencies[i] = Vector<unsigned>();
        nodeDependencies.append( i );
    }
    node->setNodeIndex( dependencyCount );
    nodes.append( node );
    dependencies[dependencyCount] = nodeDependencies;
    return std::make_unique<TorchModel>( nodes, inputs, dependencyCount, dependencies );
}

} // namespace

class CROWNAnalysisTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Use deterministic seed for reproducibility
        torch::manual_seed( 42 );
        static bool threadsConfigured = false;
        if ( !threadsConfigured )
        {
            at::set_num_threads( 1 );
            at::set_num_interop_threads( 1 );
            threadsConfigured = true;
        }
    }
};

TEST_F( CROWNAnalysisTest, IBPForwardBasic )
{
    // Create a simple MLP: Input(2) -> Linear(2,3) -> ReLU -> Linear(3,2) -> Output
    auto model = ModelBuilder::createMLP( 2, { 3 }, 2, true, false );

    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 2 } ), 0.1 );

    model->setInputBounds( inputBounds );

    NLR::CROWNAnalysis analysis( model.get() );
    analysis.setInputBounds( inputBounds );
    analysis.computeIBPBounds();

    auto ibpBounds = analysis.getOutputIBPBounds();

    // Verify bounds are computed
    EXPECT_GT( ibpBounds.lower().numel(), 0 );
    EXPECT_GT( ibpBounds.upper().numel(), 0 );

    // Verify upper >= lower
    torch::Tensor width = ibpBounds.upper() - ibpBounds.lower();
    EXPECT_TRUE( torch::all( width >= 0 ).item<bool>() );
}

TEST_F( CROWNAnalysisTest, IBPIgnoresPropertySizedCachedOutputBounds )
{
    auto model = ModelBuilder::createMLP( 2, { 3 }, 2, true, false );
    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 2 } ), 0.1 );
    model->setInputBounds( inputBounds );

    auto output = model->getNode( model->getOutputIndex() );
    ASSERT_NE( output, nullptr );
    output->setBounds( torch::zeros( { 99 } ), torch::ones( { 99 } ) );

    NLR::CROWNAnalysis analysis( model.get() );
    analysis.setInputBounds( inputBounds );
    analysis.computeIBPBounds();

    const auto bounds = analysis.getOutputIBPBounds();
    EXPECT_EQ( bounds.lower().numel(), 2 );
    EXPECT_EQ( bounds.upper().numel(), 2 );
}

// Fixed intermediates keep seeded node bounds even when a branch split marks every node dirty
TEST_F( CROWNAnalysisTest, FixedIntermediatesReuseSeededNodesEvenWhenDirty )
{
    auto model = ModelBuilder::createMLP( 2, { 3 }, 2, true, false );
    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 2 } ), 0.1 );
    model->setInputBounds( inputBounds );

    std::vector<unsigned> allNodes;
    std::size_t seededNodes = 0;
    for ( unsigned index = 0; index < model->getNumNodes(); ++index )
    {
        allNodes.push_back( index );
        auto node = model->getNode( index );
        if ( !node || node->getNodeType() == NodeType::INPUT )
            continue;
        const auto size = static_cast<int64_t>( node->getOutputSize() );
        node->setBounds( torch::full( { size }, -7.0 ), torch::full( { size }, 7.0 ) );
        ++seededNodes;
    }
    ASSERT_GT( seededNodes, 0u );
    model->setDirtyNodeIndices( allNodes );

    const bool previousRecompute = LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS;
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = false;
    NLR::CROWNAnalysis fixedAnalysis( model.get() );
    fixedAnalysis.setInputBounds( inputBounds );
    fixedAnalysis.computeIBPBounds();
    const auto fixedStats = model->getForwardBoundWorkStats();

    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = true;
    NLR::CROWNAnalysis recomputeAnalysis( model.get() );
    recomputeAnalysis.setInputBounds( inputBounds );
    recomputeAnalysis.computeIBPBounds();
    const auto recomputeStats = model->getForwardBoundWorkStats();
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = previousRecompute;

    EXPECT_EQ( fixedStats.reusedNodes, seededNodes );
    EXPECT_EQ( recomputeStats.reusedNodes, 0u );
}

TEST_F( CROWNAnalysisTest, CROWNTighterThanIBP )
{
    auto model = ModelBuilder::createMLP( 5, { 10, 10 }, 3, true, false );

    auto inputBounds = BoundGenerator::randomBounds( { 1, 5 }, 0.0, 1.0, 0.2 );
    model->setInputBounds( inputBounds );

    NLR::CROWNAnalysis analysis( model.get() );
    analysis.setInputBounds( inputBounds );
    analysis.computeIBPBounds();
    auto ibpBounds = analysis.getOutputIBPBounds();

    analysis.run();
    auto crownBounds = analysis.getOutputBounds();

    // CROWN should be at least as tight as IBP
    EXPECT_TRUE( SoundnessChecker::crownTighterThanIBP( ibpBounds, crownBounds ) );
}

TEST_F( CROWNAnalysisTest, CROWNSoundnessSimple )
{
    auto model = ModelBuilder::createMLP( 5, { 10 }, 3, true, false );

    auto inputBounds = BoundGenerator::epsilonBall( torch::randn( { 1, 5 } ), 0.1 );

    model->setInputBounds( inputBounds );

    NLR::CROWNAnalysis analysis( model.get() );
    analysis.setInputBounds( inputBounds );
    analysis.run();

    auto outputBounds = analysis.getOutputBounds();

    // Verify soundness by checking center point
    EXPECT_TRUE( SoundnessChecker::centerContainedInBounds( inputBounds, outputBounds, *model ) );
}

TEST_F( CROWNAnalysisTest, BackwardPropagation )
{
    auto model = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );

    auto inputBounds = BoundGenerator::wideBounds( { 1, 3 }, 0.0, 1.0 );
    model->setInputBounds( inputBounds );

    NLR::CROWNAnalysis analysis( model.get() );
    analysis.setInputBounds( inputBounds );
    analysis.run();

    // Verify bounds computed successfully
    auto bounds = analysis.getOutputBounds();
    EXPECT_GT( bounds.lower().numel(), 0 );
    EXPECT_GT( bounds.upper().numel(), 0 );
}

TEST_F( CROWNAnalysisTest, AddAMixedPatchesAndSpatialDenseUsesCanonicalLayout )
{
    auto model = ModelBuilder::createMLP( 4, {}, 4, false, false );
    NLR::CROWNAnalysis analysis( model.get() );

    auto patch_values = torch::arange( 1, 5, torch::kFloat32 ).reshape( { 1, 1, 2, 2, 1, 1, 1 } );
    auto patches = std::make_shared<Patches>(
        patch_values, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 0, 0, 0, 0 },
        std::vector<int64_t>{ 0, 0, 0, 0 }, 0, std::nullopt, std::vector<int64_t>{ 1, 1, 2, 2 },
        std::vector<int64_t>{ 1, 1, 2, 2 } );

    auto patch_matrix =
        patches->to_matrix( patches->input_shape ).permute( { 1, 0, 2, 3, 4 } ).flatten( 2 );
    auto spatial_dense = patch_matrix.reshape( { 4, 1, 1, 2, 2 } );
    auto expected = patch_matrix * 2;

    auto patches_first = analysis.addA( BoundA( patches ), BoundA( spatial_dense ) );
    auto dense_first = analysis.addA( BoundA( spatial_dense ), BoundA( patches ) );

    ASSERT_TRUE( patches_first.isTensor() );
    ASSERT_TRUE( dense_first.isTensor() );
    EXPECT_EQ( patches_first.asTensor().sizes(), torch::IntArrayRef( { 4, 1, 4 } ) );
    EXPECT_EQ( dense_first.asTensor().sizes(), torch::IntArrayRef( { 4, 1, 4 } ) );
    EXPECT_TRUE( torch::allclose( patches_first.asTensor(), expected ) );
    EXPECT_TRUE( torch::allclose( dense_first.asTensor(), expected ) );
}

TEST_F( CROWNAnalysisTest, AddAMixedPatchesAndDenseRejectsIncompatibleFeaturesClearly )
{
    auto model = ModelBuilder::createMLP( 4, {}, 4, false, false );
    NLR::CROWNAnalysis analysis( model.get() );

    auto patches = std::make_shared<Patches>(
        torch::ones( { 1, 1, 2, 2, 1, 1, 1 } ), std::vector<int64_t>{ 1, 1 },
        std::vector<int64_t>{ 0, 0, 0, 0 }, std::vector<int64_t>{ 0, 0, 0, 0 }, 0, std::nullopt,
        std::vector<int64_t>{ 1, 1, 2, 2 }, std::vector<int64_t>{ 1, 1, 2, 2 } );
    auto incompatible_dense = torch::ones( { 4, 1, 5 } );

    try
    {
        (void)analysis.addA( BoundA( patches ), BoundA( incompatible_dense ) );
        FAIL() << "Expected incompatible mixed A matrices to throw";
    }
    catch ( const std::runtime_error &error )
    {
        const std::string message = error.what();
        EXPECT_NE( message.find( "CROWNAnalysis::addA" ), std::string::npos );
        EXPECT_NE( message.find( "Patches" ), std::string::npos );
        EXPECT_NE( message.find( "[4, 1, 4]" ), std::string::npos );
        EXPECT_NE( message.find( "[4, 1, 5]" ), std::string::npos );
    }
}

TEST_F( CROWNAnalysisTest, ConcreteBoundsContractEachLaneWithItsOwnBox )
{
    auto model = ModelBuilder::createMLP( 3, {}, 2, false, false );
    NLR::CROWNAnalysis analysis( model.get() );
    // A is (spec=2, batch=2, features=3), spec 0 sums the inputs and spec 1 is -x0
    auto A = torch::zeros( { 2, 2, 3 } );
    A.select( 0, 0 ).fill_( 1.0 );
    A.index_put_( { 1, torch::indexing::Slice(), 0 }, torch::tensor( -1.0 ) );
    auto bias = torch::tensor( { { 10.0, 20.0 }, { 0.0, 0.0 } } );
    // Lane 0 box is [0, 1], lane 1 box is [2, 4]
    auto lower = torch::tensor( { { 0.0, 0.0, 0.0 }, { 2.0, 2.0, 2.0 } } );
    auto upper = torch::tensor( { { 1.0, 1.0, 1.0 }, { 4.0, 4.0, 4.0 } } );

    auto concrete = analysis.computeConcreteLowerBound( A, bias, lower, upper );

    // Result is (batch, spec): lane 0 gives 10 + 0 and -1, lane 1 gives 20 + 6 and -4
    auto expected = torch::tensor( { { 10.0, -1.0 }, { 26.0, -4.0 } } );
    EXPECT_EQ( concrete.sizes(), expected.sizes() );
    EXPECT_TRUE( torch::allclose( concrete, expected ) );
}

TEST_F( CROWNAnalysisTest, ConcreteUpperBoundsContractEachLaneWithItsOwnBox )
{
    auto model = ModelBuilder::createMLP( 3, {}, 2, false, false );
    NLR::CROWNAnalysis analysis( model.get() );
    auto A = torch::zeros( { 2, 2, 3 } );
    A.select( 0, 0 ).fill_( 1.0 );
    A.index_put_( { 1, torch::indexing::Slice(), 0 }, torch::tensor( -1.0 ) );
    auto bias = torch::tensor( { { 10.0, 20.0 }, { 0.0, 0.0 } } );
    auto lower = torch::tensor( { { 0.0, 0.0, 0.0 }, { 2.0, 2.0, 2.0 } } );
    auto upper = torch::tensor( { { 1.0, 1.0, 1.0 }, { 4.0, 4.0, 4.0 } } );

    auto concrete = analysis.computeConcreteUpperBound( A, bias, lower, upper );

    // Lane 0 gives 10 + 3 and 0, lane 1 gives 20 + 12 and -2
    auto expected = torch::tensor( { { 13.0, 0.0 }, { 32.0, -2.0 } } );
    EXPECT_TRUE( torch::allclose( concrete, expected ) );
}

TEST_F( CROWNAnalysisTest, ConcreteBoundsShareOneBoxAcrossLanes )
{
    auto model = ModelBuilder::createMLP( 3, {}, 2, false, false );
    NLR::CROWNAnalysis analysis( model.get() );
    auto A = torch::ones( { 1, 3, 3 } );
    auto bias = torch::zeros( { 1, 3 } );
    auto lower = torch::zeros( { 1, 3 } );
    auto upper = torch::ones( { 1, 3 } );

    auto concrete = analysis.computeConcreteUpperBound( A, bias, lower, upper );

    EXPECT_TRUE( torch::allclose( concrete, torch::full( { 3, 1 }, 3.0 ) ) );
}

TEST_F( CROWNAnalysisTest, ConcreteBoundsFlattenSpatialDenseFeatures )
{
    auto model = ModelBuilder::createMLP( 3, {}, 2, false, false );
    NLR::CROWNAnalysis analysis( model.get() );
    auto A = torch::ones( { 2, 1, 3, 2, 2 } );
    auto bias = torch::zeros( { 2, 1 } );
    auto lower = torch::ones( { 1, 3, 2, 2 } );
    auto upper = torch::ones( { 1, 3, 2, 2 } );

    auto concrete = analysis.computeConcreteLowerBound( A, bias, lower, upper );

    EXPECT_TRUE( torch::allclose( concrete, torch::full( { 1, 2 }, 12.0 ) ) );
}

TEST_F( CROWNAnalysisTest, PatchesConcretizationContainsOriginalFloatEndpoints )
{
    auto model = ModelBuilder::createMLP( 1, {}, 1, false, false );
    auto lower = torch::tensor( { -1.0077401399612427f }, torch::kFloat32 );
    auto upper = torch::tensor( { -0.9999401569366455f }, torch::kFloat32 );
    model->setInputBounds( BoundedTensor<torch::Tensor>( lower, upper ) );

    NLR::CROWNAnalysis analysis( model.get() );
    auto patches = std::make_shared<Patches>(
        torch::ones( { 1, 1, 1, 1, 1, 1, 1 }, torch::kFloat32 ), std::vector<int64_t>{ 1, 1 },
        std::vector<int64_t>{ 0, 0, 0, 0 }, std::vector<int64_t>{ 0, 0, 0, 0 }, 0, std::nullopt,
        std::vector<int64_t>{ 1, 1, 1, 1 }, std::vector<int64_t>{ 1, 1, 1, 1 } );
    analysis.addBound( 0, BoundA( patches ), BoundA( patches ) );
    analysis.addBias( 0, torch::zeros( { 1, 1 } ), torch::zeros( { 1, 1 } ) );

    analysis.concretizeNode( 1 );

    EXPECT_LE( analysis.getConcreteLowerBound( 1 ).item<float>(), lower.item<float>() );
    EXPECT_GE( analysis.getConcreteUpperBound( 1 ).item<float>(), upper.item<float>() );
}

TEST_F( CROWNAnalysisTest, BackwardRejectsDependencyAndCoefficientCountMismatch )
{
    auto node = std::make_shared<BackwardContractNode>( true, 1, true, false );
    auto model = makeContractModel( node, 2 );
    NLR::CROWNAnalysis analysis( model.get() );

    expectRuntimeErrorContaining( [&]() { analysis.backwardFrom( 2 ); },
                                  "dependency/A count mismatch" );
}

TEST_F( CROWNAnalysisTest, BackwardRejectsNonPerturbedNodeCarryingCoefficients )
{
    auto node = std::make_shared<BackwardContractNode>( false, 1, true, false );
    auto model = makeContractModel( node, 1 );
    NLR::CROWNAnalysis analysis( model.get() );

    expectRuntimeErrorContaining( [&]() { analysis.backwardFrom( 1 ); },
                                  "non-perturbed node carries A coefficients" );
}

TEST_F( CROWNAnalysisTest, BackwardRejectsBiasWithoutCoefficientCarrier )
{
    auto node = std::make_shared<BackwardContractNode>( true, 2, false, true );
    auto model = makeContractModel( node, 2 );
    NLR::CROWNAnalysis analysis( model.get() );

    expectRuntimeErrorContaining( [&]() { analysis.backwardFrom( 2 ); },
                                  "bias has no A carrier on dependency 0" );
}

TEST_F( CROWNAnalysisTest, ConcretizeRejectsFeatureMismatchInsteadOfFallingBackToIBP )
{
    auto node = std::make_shared<BackwardContractNode>( true, 1, true, false );
    auto model = makeContractModel( node, 1 );
    auto lower = torch::zeros( { 2 } );
    auto upper = torch::ones( { 2 } );
    model->setInputBounds( BoundedTensor<torch::Tensor>( lower, upper ) );
    NLR::CROWNAnalysis analysis( model.get() );
    analysis.addBound( 0, BoundA( torch::ones( { 2, 1, 3 } ) ),
                       BoundA( torch::ones( { 2, 1, 3 } ) ) );
    analysis.addBias( 0, torch::zeros( { 2, 1 } ), torch::zeros( { 2, 1 } ) );

    expectRuntimeErrorContaining( [&]() { analysis.concretizeNode( 1, {} ); },
                                  "concretization: input has 2 elements for 3 features" );
}
