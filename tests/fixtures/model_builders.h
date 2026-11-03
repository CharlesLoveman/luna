#ifndef __MODEL_BUILDERS_H__
#define __MODEL_BUILDERS_H__

#include "src/engine/TorchModel.h"
#include "src/common/BoundedTensor.h"
#include "src/engine/nodes/BoundedInputNode.h"
#include "src/engine/nodes/BoundedLinearNode.h"
#include "src/engine/nodes/BoundedReLUNode.h"
#include "src/engine/nodes/BoundedConvNode.h"
#include "src/engine/nodes/BoundedBatchNormNode.h"
#include "src/engine/nodes/BoundedAddNode.h"
#include "src/engine/nodes/BoundedSigmoidNode.h"
#include "src/engine/nodes/BoundedFlattenNode.h"
#include "src/engine/nodes/BoundedReshapeNode.h"
#include "src/input_parsers/Operations.h"
#include "src/engine/conv/ConvolutionMode.h"
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <string>

namespace test
{

/**
 * Fluent API for building test neural network models programmatically.
 */
class ModelBuilder
{
public:
    ModelBuilder();
    ~ModelBuilder() = default;

    // Build methods - create complete models
    static std::shared_ptr<NLR::TorchModel> createMLP( unsigned inputSize,
                                                       const std::vector<unsigned> &hiddenSizes,
                                                       unsigned outputSize, bool useRelu = true,
                                                       bool randomWeights = false );

    static std::shared_ptr<NLR::TorchModel>
    createCNN( unsigned inputChannels, unsigned inputHeight, unsigned inputWidth,
               const std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned>>
                   &convLayers, // (out_channels, kernel, stride, padding)
               const std::vector<unsigned> &fcSizes, unsigned outputSize );

    static std::shared_ptr<NLR::TorchModel> loadFromONNX( const std::string &onnxPath );
    static std::shared_ptr<NLR::TorchModel> loadFromONNXWithVNNLib( const std::string &onnxPath,
                                                                    const std::string &vnnlibPath );

    // Fluent API methods for building custom models
    ModelBuilder &addInput( unsigned size, const std::string &name = "input" );
    ModelBuilder &addLinear( unsigned inSize, unsigned outSize, const std::string &name = "" );
    ModelBuilder &addLinear( const torch::Tensor &weight, const torch::Tensor &bias,
                             const std::string &name = "" );
    ModelBuilder &addReLU( const std::string &name = "" );
    ModelBuilder &addConv2d( unsigned inChannels, unsigned outChannels, unsigned kernelSize,
                             unsigned stride = 1, unsigned padding = 0,
                             const std::string &name = "" );
    ModelBuilder &addBatchNorm( unsigned numFeatures, const std::string &name = "" );
    ModelBuilder &addAdd( const std::string &name = "" );
    ModelBuilder &addSigmoid( const std::string &name = "" );
    ModelBuilder &addFlatten( const std::string &name = "" );
    ModelBuilder &addReshape( const std::vector<int64_t> &shape, const std::string &name = "" );
    ModelBuilder &addSoftmax( const std::string &name = "", const int64_t dim = -1 );

    std::shared_ptr<NLR::TorchModel> build();

private:
    std::vector<std::shared_ptr<NLR::BoundedTorchNode>> _nodes;
    std::vector<unsigned> _inputIndices;
    unsigned _nextNodeIndex;
    Map<unsigned, Vector<unsigned>> _dependencies; // Map and Vector are in global namespace

    unsigned getNextNodeIndex()
    {
        return _nextNodeIndex++;
    }
    void addDependency( unsigned nodeIndex, unsigned dependencyIndex );
};

} // namespace test

#endif // __MODEL_BUILDERS_H__
