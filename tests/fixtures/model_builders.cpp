#include "model_builders.h"
#include "src/input_parsers/OnnxToTorch.h"
#include <torch/torch.h>
#include <random>

namespace test
{

ModelBuilder::ModelBuilder()
    : _nextNodeIndex( 0 )
{
}

std::shared_ptr<NLR::TorchModel> ModelBuilder::createMLP( unsigned inputSize,
                                                          const std::vector<unsigned> &hiddenSizes,
                                                          unsigned outputSize, bool useRelu,
                                                          bool randomWeights )
{

    ModelBuilder builder;
    builder.addInput( inputSize, "input" );

    unsigned currentSize = inputSize;
    for ( size_t i = 0; i < hiddenSizes.size(); ++i )
    {
        builder.addLinear( currentSize, hiddenSizes[i], "fc" + std::to_string( i + 1 ) );
        if ( useRelu )
        {
            builder.addReLU( "relu" + std::to_string( i + 1 ) );
        }
        currentSize = hiddenSizes[i];
    }

    builder.addLinear( currentSize, outputSize, "output" );
    return builder.build();
}

std::shared_ptr<NLR::TorchModel> ModelBuilder::createCNN(
    unsigned inputChannels, unsigned inputHeight, unsigned inputWidth,
    const std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned>> &convLayers,
    const std::vector<unsigned> &fcSizes, unsigned outputSize )
{

    ModelBuilder builder;
    // For CNN, we'd need to track spatial dimensions, which is complex
    // This is a simplified version - full implementation would track shape through layers
    builder.addInput( inputChannels * inputHeight * inputWidth, "input" );

    // For now, create a simple model - full CNN builder would require shape tracking
    // This is a placeholder - full implementation would be more complex
    unsigned currentChannels = inputChannels;
    for ( const auto &layer : convLayers )
    {
        unsigned outChannels = std::get<0>( layer );
        unsigned kernel = std::get<1>( layer );
        builder.addConv2d( currentChannels, outChannels, kernel, std::get<2>( layer ),
                           std::get<3>( layer ) );
        builder.addReLU();
        currentChannels = outChannels;
    }

    builder.addFlatten();
    unsigned currentSize = currentChannels; // Simplified - should compute from spatial dims
    for ( size_t i = 0; i < fcSizes.size(); ++i )
    {
        builder.addLinear( currentSize, fcSizes[i], "fc" + std::to_string( i + 1 ) );
        builder.addReLU( "relu" + std::to_string( i + 1 ) );
        currentSize = fcSizes[i];
    }

    builder.addLinear( currentSize, outputSize, "output" );
    return builder.build();
}

std::shared_ptr<NLR::TorchModel> ModelBuilder::loadFromONNX( const std::string &onnxPath )
{
    return OnnxToTorchParser::parse( String( onnxPath.c_str() ) );
}

std::shared_ptr<NLR::TorchModel>
ModelBuilder::loadFromONNXWithVNNLib( const std::string &onnxPath, const std::string &vnnlibPath )
{
    return std::make_shared<NLR::TorchModel>( String( onnxPath.c_str() ),
                                              String( vnnlibPath.c_str() ) );
}

ModelBuilder &ModelBuilder::addInput( unsigned size, const std::string &name )
{
    auto inputNode = std::make_shared<NLR::BoundedInputNode>( 0, size, String( name.c_str() ) );
    unsigned nodeIndex = getNextNodeIndex();
    inputNode->setNodeIndex( nodeIndex );
    inputNode->setNodeName( String( name.c_str() ) );
    _nodes.push_back( inputNode );
    _inputIndices.push_back( nodeIndex );
    _dependencies[nodeIndex] = Vector<unsigned>(); // No dependencies
    return *this;
}

ModelBuilder &ModelBuilder::addLinear( unsigned inSize, unsigned outSize, const std::string &name )
{
    torch::nn::Linear linear( torch::nn::LinearOptions( inSize, outSize ) );
    // Initialize with small random weights if desired
    linear->weight = torch::randn( { outSize, inSize } ) * 0.1;
    linear->bias = torch::zeros( { outSize } );
    return addLinear( linear->weight, linear->bias, name );
}

ModelBuilder &ModelBuilder::addLinear( const torch::Tensor &weight, const torch::Tensor &bias,
                                       const std::string &name )
{
    torch::nn::Linear linear( torch::nn::LinearOptions( weight.size( 0 ), weight.size( 1 ) ) );
    linear->weight = weight;
    linear->bias = bias;
    auto linearNode =
        std::make_shared<NLR::BoundedLinearNode>( linear, 1.0f, String( name.c_str() ) );
    unsigned nodeIndex = getNextNodeIndex();
    linearNode->setNodeIndex( nodeIndex );
    linearNode->setInputSize( weight.size( 1 ) );
    linearNode->setOutputSize( weight.size( 0 ) );
    if ( !name.empty() )
    {
        linearNode->setNodeName( String( name.c_str() ) );
    }
    _nodes.push_back( linearNode );

    // Depends on last node
    if ( !_nodes.empty() && _nodes.size() > 1 )
    {
        addDependency( nodeIndex, nodeIndex - 1 );
    }
    return *this;
}

ModelBuilder &ModelBuilder::addReLU( const std::string &name )
{
    torch::nn::ReLU relu( torch::nn::ReLUOptions{} );
    auto reluNode = std::make_shared<NLR::BoundedReLUNode>( relu, String( name.c_str() ) );
    unsigned nodeIndex = getNextNodeIndex();
    reluNode->setNodeIndex( nodeIndex );
    if ( !name.empty() )
    {
        reluNode->setNodeName( String( name.c_str() ) );
    }

    // Set sizes from previous node
    if ( !_nodes.empty() )
    {
        auto prevNode = _nodes.back();
        unsigned prevOutSize = prevNode->getOutputSize();
        reluNode->setInputSize( prevOutSize );
        reluNode->setOutputSize( prevOutSize );
    }

    _nodes.push_back( reluNode );

    // Depends on last node
    if ( !_nodes.empty() && _nodes.size() > 1 )
    {
        addDependency( nodeIndex, nodeIndex - 1 );
    }
    return *this;
}

ModelBuilder &ModelBuilder::addAddConstant( const torch::Tensor &constant, const std::string &name )
{
    auto addNode = std::make_shared<NLR::BoundedAddNode>();
    addNode->setConstantValue( constant );
    unsigned nodeIndex = getNextNodeIndex();
    addNode->setNodeIndex( nodeIndex );
    if ( !name.empty() )
    {
        addNode->setNodeName( String( name.c_str() ) );
    }

    // Keeps the size of the node before it
    if ( !_nodes.empty() )
    {
        unsigned prevOutSize = _nodes.back()->getOutputSize();
        addNode->setInputSize( prevOutSize );
        addNode->setOutputSize( prevOutSize );
    }

    _nodes.push_back( addNode );
    if ( _nodes.size() > 1 )
    {
        addDependency( nodeIndex, nodeIndex - 1 );
    }
    return *this;
}

ModelBuilder &ModelBuilder::addConv2d( unsigned inChannels, unsigned outChannels,
                                       unsigned kernelSize, unsigned stride, unsigned padding,
                                       const std::string &name )
{
    torch::nn::Conv2d conv( torch::nn::Conv2dOptions( inChannels, outChannels, kernelSize )
                                .stride( stride )
                                .padding( padding ) );
    // Initialize with small random weights
    conv->weight = torch::randn( { outChannels, inChannels, kernelSize, kernelSize } ) * 0.1;
    conv->bias = torch::zeros( { outChannels } );

    auto convNode = std::make_shared<NLR::BoundedConvNode>( conv, NLR::ConvMode::MATRIX,
                                                            String( name.c_str() ) );
    unsigned nodeIndex = getNextNodeIndex();
    convNode->setNodeIndex( nodeIndex );
    if ( !name.empty() )
    {
        convNode->setNodeName( String( name.c_str() ) );
    }
    _nodes.push_back( convNode );

    // Depends on last node
    if ( !_nodes.empty() && _nodes.size() > 1 )
    {
        addDependency( nodeIndex, nodeIndex - 1 );
    }
    return *this;
}

ModelBuilder &ModelBuilder::addBatchNorm( unsigned numFeatures, const std::string &name )
{
    // Create default batch norm parameters
    auto scale = torch::ones( { numFeatures } );
    auto bias = torch::zeros( { numFeatures } );
    auto mean = torch::zeros( { numFeatures } );
    auto var = torch::ones( { numFeatures } );
    float eps = 1e-5f;
    auto bnNode = std::make_shared<NLR::BoundedBatchNormNode>( scale, bias, mean, var, eps,
                                                               String( name.c_str() ) );
    unsigned nodeIndex = getNextNodeIndex();
    bnNode->setNodeIndex( nodeIndex );
    if ( !name.empty() )
    {
        bnNode->setNodeName( String( name.c_str() ) );
    }
    _nodes.push_back( bnNode );

    if ( !_nodes.empty() && _nodes.size() > 1 )
    {
        addDependency( nodeIndex, nodeIndex - 1 );
    }
    return *this;
}

ModelBuilder &ModelBuilder::addAdd( const std::string &name )
{
    auto addNode = std::make_shared<NLR::BoundedAddNode>();
    unsigned nodeIndex = getNextNodeIndex();
    addNode->setNodeIndex( nodeIndex );
    if ( !name.empty() )
    {
        addNode->setNodeName( String( name.c_str() ) );
    }
    _nodes.push_back( addNode );

    // Add nodes typically depend on two previous nodes - simplified for now
    if ( !_nodes.empty() && _nodes.size() > 1 )
    {
        addDependency( nodeIndex, nodeIndex - 1 );
    }
    return *this;
}

ModelBuilder &ModelBuilder::addSigmoid( const std::string &name )
{
    torch::nn::Sigmoid sigmoid;
    auto sigmoidNode = std::make_shared<NLR::BoundedSigmoidNode>( sigmoid, String( name.c_str() ) );
    unsigned nodeIndex = getNextNodeIndex();
    sigmoidNode->setNodeIndex( nodeIndex );
    if ( !name.empty() )
    {
        sigmoidNode->setNodeName( String( name.c_str() ) );
    }

    if ( !_nodes.empty() )
    {
        auto prevNode = _nodes.back();
        unsigned prevOutSize = prevNode->getOutputSize();
        sigmoidNode->setInputSize( prevOutSize );
        sigmoidNode->setOutputSize( prevOutSize );
    }

    _nodes.push_back( sigmoidNode );

    if ( !_nodes.empty() && _nodes.size() > 1 )
    {
        addDependency( nodeIndex, nodeIndex - 1 );
    }
    return *this;
}

ModelBuilder &ModelBuilder::addFlatten( const std::string &name )
{
    Operations::FlattenWrapper flatten_module( 1 ); // axis=1 (default)
    auto flattenNode = std::make_shared<NLR::BoundedFlattenNode>( flatten_module );
    unsigned nodeIndex = getNextNodeIndex();
    flattenNode->setNodeIndex( nodeIndex );
    if ( !name.empty() )
    {
        flattenNode->setNodeName( String( name.c_str() ) );
    }
    _nodes.push_back( flattenNode );

    if ( !_nodes.empty() && _nodes.size() > 1 )
    {
        addDependency( nodeIndex, nodeIndex - 1 );
    }
    return *this;
}

ModelBuilder &ModelBuilder::addReshape( const std::vector<int64_t> &shape, const std::string &name )
{
    torch::Tensor shape_tensor = torch::tensor( shape, torch::kInt64 );
    Operations::ReshapeWrapper reshape_module( shape_tensor );
    auto reshapeNode = std::make_shared<NLR::BoundedReshapeNode>( reshape_module );
    unsigned nodeIndex = getNextNodeIndex();
    reshapeNode->setNodeIndex( nodeIndex );
    if ( !name.empty() )
    {
        reshapeNode->setNodeName( String( name.c_str() ) );
    }
    _nodes.push_back( reshapeNode );

    if ( !_nodes.empty() && _nodes.size() > 1 )
    {
        addDependency( nodeIndex, nodeIndex - 1 );
    }
    return *this;
}

void ModelBuilder::addDependency( unsigned nodeIndex, unsigned dependencyIndex )
{
    if ( !_dependencies.exists( nodeIndex ) )
    {
        _dependencies[nodeIndex] = Vector<unsigned>();
    }
    _dependencies[nodeIndex].append( dependencyIndex );
}

std::shared_ptr<NLR::TorchModel> ModelBuilder::build()
{
    if ( _nodes.empty() )
    {
        throw std::runtime_error( "Cannot build model with no nodes" );
    }

    unsigned outputIndex = _nodes.size() - 1;

    Vector<std::shared_ptr<NLR::BoundedTorchNode>> nodes;
    for ( auto &node : _nodes )
    {
        nodes.append( node );
    }

    Vector<unsigned> inputIndices;
    for ( unsigned idx : _inputIndices )
    {
        inputIndices.append( idx );
    }

    return std::make_shared<NLR::TorchModel>( nodes, inputIndices, outputIndex, _dependencies );
}

} // namespace test
