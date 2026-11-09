#include <benchmark/benchmark.h>
#include "src/engine/TorchModel.h"
#include "src/configuration/LunaConfiguration.h"
#include "fixtures/model_builders.h"
#include <torch/cuda.h>
#include <torch/torch.h>

#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace NLR;
using namespace test;

namespace
{

constexpr double BOX_RADIUS = 0.0039;

/// The network under test, named by LUNA_BENCH_ONNX and loaded once for every benchmark.
std::shared_ptr<TorchModel> loadModel()
{
    const char *path = std::getenv( "LUNA_BENCH_ONNX" );
    if ( path == nullptr )
        return nullptr;
    return ModelBuilder::loadFromONNX( path );
}

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

} // namespace

/// Nothing is seeded, so every node is recomputed: the worst case for one call.
static void BM_FullIbp( benchmark::State &state )
{
    static const auto model = loadModel();
    if ( !model )
    {
        state.SkipWithError( "LUNA_BENCH_ONNX is not set" );
        return;
    }
    const int64_t lanes = state.range( 0 );
    const auto boxes = makeBoxes( *model, lanes );
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = true;
    model->persistIntermediateBounds( {} );
    (void)model->runIBP( boxes );
    waitForDevice();

    for ( auto _ : state )
    {
        auto bounds = model->runIBP( boxes );
        waitForDevice();
        benchmark::DoNotOptimize( bounds );
    }
    state.SetItemsProcessed( state.iterations() * lanes );
}

/// Every node is seeded and final, so the call only reuses them: the bookkeeping floor.
static void BM_SeededReuseIbp( benchmark::State &state )
{
    static const auto model = loadModel();
    if ( !model )
    {
        state.SkipWithError( "LUNA_BENCH_ONNX is not set" );
        return;
    }
    const int64_t lanes = state.range( 0 );
    const auto boxes = makeBoxes( *model, lanes );
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = true;
    model->persistIntermediateBounds( {} );
    (void)model->runIBP( boxes );

    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> seeds;
    for ( unsigned node : model->getAllNodeIndices() )
    {
        if ( !model->hasConcreteBounds( node ) || node == model->getOutputIndex() ||
             node == model->getInputIndices()[0] )
            continue;
        const auto bounds = model->getConcreteBounds( node );
        seeds[node] = { bounds.lower().clone(), bounds.upper().clone() };
    }
    model->persistIntermediateBounds( seeds );
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = false;
    (void)model->runIBP( boxes );
    waitForDevice();

    for ( auto _ : state )
    {
        auto bounds = model->runIBP( boxes );
        waitForDevice();
        benchmark::DoNotOptimize( bounds );
    }
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = true;
    model->persistIntermediateBounds( {} );
    state.SetItemsProcessed( state.iterations() * lanes );
}

BENCHMARK( BM_FullIbp )->Arg( 1 )->Arg( 8 )->Arg( 32 )->Arg( 128 )->Unit( benchmark::kMillisecond );
BENCHMARK( BM_SeededReuseIbp )
    ->Arg( 1 )
    ->Arg( 8 )
    ->Arg( 32 )
    ->Arg( 128 )
    ->Unit( benchmark::kMillisecond );

BENCHMARK_MAIN();
