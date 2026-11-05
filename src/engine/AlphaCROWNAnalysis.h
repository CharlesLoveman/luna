/*********************                                                        */
/*! \file AlphaCROWNAnalysis.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __AlphaCROWNAnalysis_h__
#define __AlphaCROWNAnalysis_h__

#include "CROWNAnalysis.h"
#include "AlphaParameters.h"
#include "BoundedTensor.h"
#include "Map.h"
#include "Vector.h"
#include "MString.h"
#include "configuration/LunaConfiguration.h"

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <unordered_map>

namespace NLR
{

class BoundedAlphaOptimizeNode;
class TorchModel;

class AlphaCROWNAnalysis
{
public:
    AlphaCROWNAnalysis( TorchModel *torchModel );

    ~AlphaCROWNAnalysis();

    void initializeAlphaParameters();

    struct AlphaResult
    {
        torch::Tensor alpha;
        torch::Tensor unstableMask;
        torch::Tensor unstableIndices;
        int numUnstable{ 0 };
        int outDim{ 0 };
        bool hasSpecDefaultSlot{ false };
    };

    AlphaResult getAlphaForNodeAllSpecs( unsigned nodeIndex, const std::string &startKey,
                                         int specDim, int outDim, const torch::Tensor &input_lb,
                                         const torch::Tensor &input_ub );

    void setOptimizationStage( const std::string &stage );

    std::string getOptimizationStage() const
    {
        return _optimizationStage;
    }

    unsigned getNumOptimizableNodes() const;

    std::vector<unsigned> getOptimizableNodeIndices() const;

    void resetForOptimization();

    void resetAlphasFromCROWNSlopes( bool isLower );

    CROWNAnalysis *getCROWNAnalysis() const;
    TorchModel *getTorchModel() const;

    torch::Tensor computeOptimizedBounds( LunaConfiguration::BoundSide side );

    void preserveIntermediateBoundsForNextOptimization();

    bool hasIntermediateBoundsForReuse() const
    {
        return _reuseIntermediateBounds;
    }

    /// \brief Warm start from persisted alphas for a batch of `lanes` search nodes.
    ///
    /// An alpha from a one-lane parent is copied to every lane, one with another lane count is
    /// left out so that entry starts cold.
    void
    seedAlphas( const std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>>
                    &alphas,
                int64_t lanes = 1 );
    void seedIntermediateBounds(
        const std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> &bounds );
    std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>>
    extractAlphas() const;

    void clipAlphaParameters();

    std::vector<torch::Tensor> collectAlphaParameters();

    bool isAlphaEnabled() const
    {
        return _alphaEnabled;
    }
    void setAlphaEnabled( bool enabled )
    {
        _alphaEnabled = enabled;
    }

    bool isInitialized() const
    {
        return _initialized;
    }

    unsigned getIterations() const
    {
        return _iteration;
    }
    void setIterations( unsigned iterations )
    {
        _iteration = iterations;
    }

    float getLearningRate() const
    {
        return _learningRate;
    }
    void setLearningRate( float lr )
    {
        _learningRate = lr;
    }
    void decayLearningRate()
    {
        _learningRate *= LunaConfiguration::ALPHA_LR_DECAY;
    }

    void setIteration( unsigned iteration )
    {
        LunaConfiguration::ALPHA_ITERATIONS = iteration;
        _iteration = iteration;
    }
    void setLrAlpha( float lr )
    {
        LunaConfiguration::ALPHA_LR = lr;
        _learningRate = lr;
    }
    void setKeepBest( bool keep )
    {
        LunaConfiguration::KEEP_BEST = keep;
    }
    void setOptimizer( const std::string &opt )
    {
        LunaConfiguration::OPTIMIZER = String( opt.c_str() );
    }
    void setBoundSide( LunaConfiguration::BoundSide side )
    {
        LunaConfiguration::BOUND_SIDE = side;
    }

    LunaConfiguration::BoundSide getBoundSide() const
    {
        return LunaConfiguration::BOUND_SIDE;
    }
    bool isOptimizingLower() const
    {
        return LunaConfiguration::BOUND_SIDE == LunaConfiguration::BoundSide::Lower;
    }
    bool isOptimizingUpper() const
    {
        return LunaConfiguration::BOUND_SIDE == LunaConfiguration::BoundSide::Upper;
    }

private:
    TorchModel *_torchModel;
    std::unique_ptr<CROWNAnalysis> _crownAnalysis;

    // nodeIndex -> startKey -> AlphaParameters
    // alpha shape: [2, spec, 1, numUnstable] where dim 0 separates lA/uA paths
    std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>>
        _alphaByNodeStart;

    std::vector<std::pair<unsigned, std::shared_ptr<BoundedAlphaOptimizeNode>>> _optimizableNodes;

    bool _alphaEnabled;
    bool _initialized;
    bool _reuseIntermediateBounds{ false };

    // While the optimizer runs its alpha tensors must stay the ones it holds, so their unstable
    // sets are not rebuilt when recomputed intermediate bounds move a neuron across zero
    bool _freezeAlphaSets{ false };
    unsigned _iteration;
    float _learningRate;
    std::string _optimizationStage;

    std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>>
        _bestAlphaByNodeStart;

    torch::Tensor _bestLowerBounds;
    torch::Tensor _bestUpperBounds;

    // per-node best intermediate bounds: nodeIndex -> (best_lower, best_upper)
    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> _bestIntermediateBounds;

    // STE stabilization: intermediate bounds are clamped to be at least as tight
    // as these references (value), while gradients flow unclamped (straight-through)
    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> _referenceBounds;

    // auto_LiRPA fix_interm_bounds: computed once, injected every iteration
    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> _initIntermediateBounds;

    void performForwardPass();
    void prepareOptimizableActivations();
    void performCROWNInitializationPass();
    AlphaParameters &ensureAlphaFor( unsigned nodeIndex, const std::string &startKey, int specDim,
                                     int outDim, const torch::Tensor &input_lb,
                                     const torch::Tensor &input_ub );

    /// \brief Copy the current alphas into the best ones for each lane that improved.
    ///
    /// improvedLanes is one bool per lane. The whole alpha slice of a lane moves, for every start
    /// node, as in auto_LiRPA.
    void updateBestAlphas( const torch::Tensor &improvedLanes );

    void restoreBestAlphas();

    /// \brief Merge the current intermediate bounds into the best ones, element-wise.
    ///
    /// improvedLanes is one bool per lane, an undefined tensor merges every lane.
    void snapshotBestIntermediateBounds( const torch::Tensor &improvedLanes = torch::Tensor() );
    void restoreBestIntermediateBounds();

    void _captureInitIntermediateBounds();
    void _injectInitIntermediateBounds();

    torch::Tensor computeLoss( const torch::Tensor &lowerBounds, const torch::Tensor &upperBounds,
                               bool optimizeLower,
                               c10::optional<torch::Tensor> stop_mask_opt = c10::nullopt );

    bool shouldPerformOptimization() const;

    torch::Tensor extractLowerBoundsFromCROWN();
    torch::Tensor extractUpperBoundsFromCROWN();

    void updateFromConfig();

    void log( const String &message );
};

} // namespace NLR

#endif // __AlphaCROWNAnalysis_h__
