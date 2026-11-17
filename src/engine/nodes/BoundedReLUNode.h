/*********************                                                        */
/*! \file BoundedReLUNode.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __BOUNDED_RELU_NODE_H__
#define __BOUNDED_RELU_NODE_H__

#include "BoundedAlphaOptimizedNode.h"

namespace NLR
{

class BoundedReLUNode : public NLR::BoundedAlphaOptimizeNode
{
public:
    BoundedReLUNode( const torch::nn::ReLU &reluModule, const String &name = "" );

    NLR::NodeType getNodeType() const override
    {
        return NLR::NodeType::RELU;
    }
    String getNodeName() const override
    {
        return _nodeName;
    }
    unsigned getNodeIndex() const override
    {
        return _nodeIndex;
    }

    torch::Tensor forward( const torch::Tensor &input ) override;

    void boundBackward( const BoundA &last_lA, const BoundA &last_uA,
                        const Vector<BoundedTensor<torch::Tensor>> &inputBounds,
                        Vector<Pair<BoundA, BoundA>> &outputA_matrices, torch::Tensor &lbias,
                        torch::Tensor &ubias ) override;

    BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>> &inputBounds ) override;

    unsigned getInputSize() const override;
    unsigned getOutputSize() const override;
    bool isPerturbed() const override
    {
        return true;
    }

    void setInputSize( unsigned size ) override;
    void setOutputSize( unsigned size ) override;

    void setNodeIndex( unsigned index ) override
    {
        _nodeIndex = index;
    }
    void setNodeName( const String &name ) override
    {
        _nodeName = name;
    }
    void moveToDevice( const torch::Device &device ) override;

    struct RelaxationResult
    {
        torch::Tensor d_lower;
        torch::Tensor d_upper;
        torch::Tensor bias_lower;
        torch::Tensor bias_upper;

        torch::Tensor lb_lower_d, ub_lower_d;
        torch::Tensor lb_upper_d, ub_upper_d;
    };

    RelaxationResult _backwardRelaxation( const BoundA &last_lA, const BoundA &last_uA,
                                          const torch::Tensor &input_lower,
                                          const torch::Tensor &input_upper );

    void computeAlphaRelaxation( const torch::Tensor &last_lA, const torch::Tensor &last_uA,
                                 const torch::Tensor &input_lower, const torch::Tensor &input_upper,
                                 torch::Tensor &d_lower, torch::Tensor &d_upper,
                                 torch::Tensor &bias_lower, torch::Tensor &bias_upper ) override;

private:
    std::shared_ptr<torch::nn::ReLU> _reluModule;

    std::pair<torch::Tensor, torch::Tensor> _reluUpperBound( const torch::Tensor &lb,
                                                             const torch::Tensor &ub );
    torch::Tensor _computeStandardCROWNLowerBound( const torch::Tensor &input_lower,
                                                   const torch::Tensor &input_upper );
    void _maskAlpha( const torch::Tensor &input_lower, const torch::Tensor &input_upper,
                     const torch::Tensor &upper_d, RelaxationResult &result );

    torch::Tensor init_upper_d;
    int _currentSpecDim{ 1 };

public:
    torch::Tensor getCROWNSlope( bool isLowerBound ) const override;
};

} // namespace NLR

#endif // __BOUNDED_RELU_NODE_H__
