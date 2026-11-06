/*********************                                                        */
/*! \file TorchModel.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __TorchModel_h__
#define __TorchModel_h__

#include "Map.h"
#include "Set.h"
#include "MString.h"
#include "Vector.h"
#include "nodes/BoundedTorchNode.h"
#include "nodes/BoundedInputNode.h"
#include "nodes/BoundedLinearNode.h"
#include "nodes/BoundedReLUNode.h"
#include "nodes/BoundedIdentityNode.h"
#include "nodes/BoundedConstantNode.h"
#include "nodes/BoundedReshapeNode.h"
#include "input_parsers/OutputConstraint.h"
#include "configuration/LunaConfiguration.h"

class CROWNAnalysis;
class AlphaCROWNAnalysis;

// avoid conflict with PyTorch
#ifdef Warning
#undef Warning
#endif

#include "AlphaParameters.h"

#include <torch/torch.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace NLR
{

/// \brief Mathematical operation kinds accepted by the tensor graph interface.
enum class TorchGraphOperation
{
    Input,
    Relu,
    Add,
    Flatten,
    Linear,
    Conv2d,
    BatchNorm,
    Unsupported
};

/// \brief One operation in a topologically ordered tensor computation graph.
struct TorchGraphNode
{
    /// \brief Stable graph-local node index.
    unsigned index = 0;
    /// \brief Stable source tensor name used to map bounds back to the caller.
    std::string name;
    /// \brief Operation described without ONNX or Luna node types.
    TorchGraphOperation operation = TorchGraphOperation::Unsupported;
    /// \brief Indices of dynamic input nodes, in operation input order.
    std::vector<unsigned> inputs;
    /// \brief Per-sample output shape, excluding the batch dimension.
    std::vector<int64_t> shape;
    /// \brief Optional constant operand used by operations such as Add.
    torch::Tensor constant;
    /// \brief Weights for Linear and Conv2d operations.
    torch::Tensor weight;
    /// \brief Optional bias for Linear and Conv2d operations.
    torch::Tensor bias;
    /// \brief Batch-normalization scale, offset, mean, and variance.
    torch::Tensor scale;
    torch::Tensor offset;
    torch::Tensor mean;
    torch::Tensor variance;
    /// \brief Convolution stride, symmetric padding, dilation, and groups.
    std::vector<int64_t> strides;
    std::vector<int64_t> padding;
    std::vector<int64_t> dilations;
    int64_t groups = 1;
    /// \brief Flatten axis or batch-normalization epsilon.
    int64_t axis = 1;
    double epsilon = 1.0e-5;
    /// \brief ONNX operation name for an explicit unsupported-operation error.
    std::string unsupported_operation;
};

/// \brief Operation-neutral graph passed from an embedding verifier to Luna.
struct TorchComputationGraph
{
    std::vector<TorchGraphNode> nodes;
    unsigned input_index = 0;
    unsigned output_index = 0;
};

class TorchModel
{
public:
    TorchModel( const Vector<std::shared_ptr<BoundedTorchNode>> &nodes,
                const Vector<unsigned> &inputIndices, unsigned outputIndex,
                const Map<unsigned, Vector<unsigned>> &dependencies );

    /// \brief Create a Luna model from an operation-neutral tensor graph.
    static std::shared_ptr<TorchModel> fromGraph( const TorchComputationGraph &graph );

    TorchModel( const String &onnxPath );

    TorchModel( const String &onnxPath, const String &vnnlibPath );

    torch::Tensor forward( const torch::Tensor &input );
    torch::Tensor forward( unsigned nodeIndex, Map<unsigned, torch::Tensor> &activations,
                           const Map<unsigned, torch::Tensor> &inputs );

    Map<unsigned, torch::Tensor> forwardAndStoreActivations( const torch::Tensor &input );
    Map<unsigned, torch::Tensor>
    forwardAndStoreActivations( const Map<unsigned, torch::Tensor> &inputs );

    unsigned getInputSize() const
    {
        return _input_size;
    }
    unsigned getOutputSize() const
    {
        return _output_size;
    }
    unsigned getNumNodes() const
    {
        return _nodes.size();
    }
    torch::Device getDevice() const
    {
        return _device;
    }
    void moveToDevice( const torch::Device &device );

    const Vector<std::shared_ptr<BoundedTorchNode>> &getNodes() const
    {
        return _nodes;
    }
    std::shared_ptr<BoundedTorchNode> getNode( unsigned index ) const;
    Vector<unsigned> getAllNodeIndices() const;
    Vector<unsigned> getNodesByType( NodeType type ) const;
    const Vector<unsigned> &getInputIndices() const
    {
        return _inputIndices;
    }
    unsigned getOutputIndex() const
    {
        return _outputIndex;
    }

    /// \brief Per-sample input shape without the batch dimension.
    std::vector<int64_t> getInputSampleShape() const;

    /// \brief Reshape flat or batched input values to `(B, *sampleShape)`.
    ///
    /// B is derived from the element count, so a flat `(n)` tensor is one sample and `(B, n)`
    /// or `(B, *sampleShape)` is B samples. Throws when the count is not a multiple.
    torch::Tensor toBatchedInput( const torch::Tensor &values ) const;

    /// \brief Shape every persisted bound to one row per lane.
    ///
    /// A seed with the node's feature count is shared by all lanes, one with lanes times that count
    /// is a row per lane, and anything else is an error.
    void normalizePersistedIntermediateBounds( int64_t lanes );

    // Remember the shape of one row of a node's bounds, so a flat seed can be given it back
    void rememberBoundRowShape( unsigned nodeIndex, const torch::Tensor &bounds );

    void setInputBounds( const BoundedTensor<torch::Tensor> &inputBounds );

    void setConcreteBounds( unsigned nodeIndex,
                            const BoundedTensor<torch::Tensor> &concreteBounds );
    void clearConcreteBounds();
    BoundedTensor<torch::Tensor> getConcreteBounds( unsigned nodeIndex ) const;
    bool hasConcreteBounds( unsigned nodeIndex ) const;

    BoundedTensor<torch::Tensor> getInputBounds() const;
    bool hasInputBounds() const;
    torch::Tensor getInputLowerBounds() const;
    torch::Tensor getInputUpperBounds() const;

    void setSpecificationMatrix( const torch::Tensor &specMatrix );
    void setSpecificationFromConstraints( const OutputConstraintSet &constraints );
    torch::Tensor getSpecificationMatrix() const;
    torch::Tensor getSpecificationThresholds() const;
    CMatrixResult getSpecificationMatrixResult() const;
    bool hasSpecificationMatrix() const;
    bool isSpecVerified( const torch::Tensor &lb, const torch::Tensor &ub ) const;

    BoundedTensor<torch::Tensor> compute_bounds(
        const BoundedTensor<torch::Tensor> &input_bounds,
        const torch::Tensor *specification_matrix = nullptr,
        LunaConfiguration::AnalysisMethod method = LunaConfiguration::AnalysisMethod::CROWN,
        bool bound_lower = true, bool bound_upper = true );

    BoundedTensor<torch::Tensor> runCROWN();
    BoundedTensor<torch::Tensor> runCROWN( const BoundedTensor<torch::Tensor> &inputBounds );
    BoundedTensor<torch::Tensor> runIBP( const BoundedTensor<torch::Tensor> &inputBounds );

    BoundedTensor<torch::Tensor> runAlphaCROWN( bool optimizeLower = true,
                                                bool optimizeUpper = false );
    BoundedTensor<torch::Tensor> runAlphaCROWN( const BoundedTensor<torch::Tensor> &inputBounds,
                                                bool optimizeLower = true,
                                                bool optimizeUpper = false );

    void setCROWNBounds( unsigned nodeIndex, const BoundedTensor<torch::Tensor> &bounds );
    void setAlphaCROWNBounds( unsigned nodeIndex, const BoundedTensor<torch::Tensor> &bounds );
    BoundedTensor<torch::Tensor> getCROWNBounds( unsigned nodeIndex ) const;
    BoundedTensor<torch::Tensor> getAlphaCROWNBounds( unsigned nodeIndex ) const;
    bool hasCROWNBounds( unsigned nodeIndex ) const;
    bool hasAlphaCROWNBounds( unsigned nodeIndex ) const;

    void setFinalAnalysisBounds( const BoundedTensor<torch::Tensor> &bounds );
    BoundedTensor<torch::Tensor> getFinalAnalysisBounds() const;
    bool hasFinalAnalysisBounds() const;

    struct BaBState
    {
        std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> allBounds;
        std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>> alphas;
    };

    void
    loadState( const BoundedTensor<torch::Tensor> &inputBounds,
               const std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>>
                   &intermediateBounds,
               const std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>>
                   &alphas );
    void loadProp( const torch::Tensor &C );
    BaBState getState() const;

    void persistAlphas(
        const std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>>
            &alphas );
    void persistIntermediateBounds(
        const std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> &bounds );
    bool hasPersistedAlphas() const;
    bool hasPersistedIntermediateBounds() const;
    const std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>> &
    getPersistedAlphas() const;
    const std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> &
    getPersistedIntermediateBounds() const;
    struct ForwardBoundWorkStats
    {
        std::size_t recomputedNodes{ 0 };
        std::size_t reusedNodes{ 0 };
    };
    void setDirtyNodeIndices( const std::vector<unsigned> &nodeIndices );
    bool isIncrementalNodeDirty( unsigned nodeIndex ) const;
    void setForwardBoundWorkStats( std::size_t recomputed, std::size_t reused );
    const ForwardBoundWorkStats &getForwardBoundWorkStats() const;
    // Center forward passes actually run, since repeated input boxes reuse the cached shapes
    std::size_t getShapeCachePassCount() const;
    void clearPersistedState();

    const Map<unsigned, Vector<unsigned>> &getDependenciesMap() const
    {
        return _dependencies;
    }

    void buildDependencyGraph();
    void buildDependents();
    void computeDegrees();

    Vector<unsigned> topologicalSort() const;
    Vector<unsigned> getRoots() const;
    Vector<unsigned> getLeaves() const;
    Vector<unsigned> getDependents( unsigned nodeIndex ) const;
    Vector<unsigned> getDependencies( unsigned nodeIndex ) const;

    unsigned getDegreeOut( unsigned nodeIndex ) const;
    unsigned getDegreeIn( unsigned nodeIndex ) const;
    void resetProcessingState();
    bool isProcessed( unsigned nodeIndex ) const;
    void markProcessed( unsigned nodeIndex );

    void log( const String &message ) const;

private:
    Vector<std::shared_ptr<BoundedTorchNode>> _nodes;
    Vector<unsigned> _inputIndices;
    unsigned _outputIndex;
    Map<unsigned, Vector<unsigned>> _dependencies;

    Map<unsigned, Vector<unsigned>> _dependents;
    Map<unsigned, unsigned> _degreeOut;
    Map<unsigned, unsigned> _degreeIn;
    Map<unsigned, bool> _processed;

    unsigned _input_size;
    unsigned _output_size;

    BoundedTensor<torch::Tensor> _inputBounds;
    Map<unsigned, BoundedTensor<torch::Tensor>> _concreteBounds;

    torch::Tensor _specificationMatrix;
    torch::Tensor _specificationThresholds;
    Vector<unsigned> _specificationBranchMapping;
    Vector<unsigned> _specificationBranchSizes;
    bool _hasSpecificationMatrix;
    bool _hasORBranches;
    Map<unsigned, BoundedTensor<torch::Tensor>> _crownBounds;
    Map<unsigned, BoundedTensor<torch::Tensor>> _alphaCrownBounds;
    BoundedTensor<torch::Tensor> _finalAnalysisBounds;
    bool _hasFinalAnalysisBounds;

    std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>> _persistedAlphas;
    // Shape of one row of each node's bounds as an earlier pass produced them, for flat seeds
    std::unordered_map<unsigned, std::vector<int64_t>> _boundRowShapes;
    bool _hasPersistedAlphas{ false };
    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>>
        _persistedIntermediateBounds;
    bool _hasPersistedIntermediateBounds{ false };
    std::unordered_set<unsigned> _dirtyNodeIndices;
    ForwardBoundWorkStats _forwardBoundWorkStats;
    // Input-box center whose forward pass last populated the node shape caches
    torch::Tensor _shapeCacheCenter;
    std::size_t _shapeCachePasses = 0;

    void validateNodeIndex( unsigned nodeIndex ) const;
    void cacheForwardShapesFromCenter();
    void resetAnalysisState();

    torch::Device _device;
};

} // namespace NLR

#endif // __TorchModel_h__
