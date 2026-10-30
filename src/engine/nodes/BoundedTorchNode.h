/*********************                                                        */
/*! \file BoundedTorchNode.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __BOUNDED_TORCH_NODE_H__
#define __BOUNDED_TORCH_NODE_H__

#include "Map.h"
#include "Vector.h"
#include "Pair.h"
#include "MString.h"
#include "BoundedTensor.h"
#include "BoundResult.h"

// Avoid conflict with PyTorch's Warning symbol
#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>
#include <memory>

// CVC4 compatibility
#ifndef Warning
#define Warning ( !::CVC4::WarningChannel.isOn() ) ? ::CVC4::nullCvc4Stream : ::CVC4::WarningChannel
#endif

namespace NLR
{

enum class NodeType
{
    INPUT,
    CONSTANT,
    LINEAR,
    RELU,
    RESHAPE,
    IDENTITY,
    SUB,
    FLATTEN,
    ADD,
    CONV,
    BATCHNORM,
    SIGMOID,
    CONVTRANSPOSE,
    CONCAT,
    SLICE,
    SOFTMAX
};

class BoundedTorchNode : public torch::nn::Module
{
public:
    virtual ~BoundedTorchNode() = default;

    virtual NodeType getNodeType() const = 0;
    virtual String getNodeName() const = 0;
    virtual unsigned getNodeIndex() const = 0;

    virtual torch::Tensor forward( const torch::Tensor &input ) = 0;
    virtual torch::Tensor forward( const std::vector<torch::Tensor> &inputs )
    {
        if ( inputs.size() == 1 )
        {
            return forward( inputs[0] );
        }
        else
        {
            throw std::runtime_error( "Multi-input forward not implemented for this module" );
        }
    }

    virtual BoundedTensor<torch::Tensor>
    computeIntervalBoundPropagation( const Vector<BoundedTensor<torch::Tensor>> &inputBounds ) = 0;

    virtual void boundBackward( const BoundA &last_lA, const BoundA &last_uA,
                                const Vector<BoundedTensor<torch::Tensor>> &inputBounds,
                                Vector<Pair<BoundA, BoundA>> &outputA_matrices,
                                torch::Tensor &lbias, torch::Tensor &ubias ) = 0;

    virtual unsigned getInputSize() const = 0;
    virtual unsigned getOutputSize() const = 0;
    virtual bool isPerturbed() const = 0;

    virtual void setInputSize( unsigned size ) = 0;
    virtual void setOutputSize( unsigned size ) = 0;

    virtual void setNodeIndex( unsigned index ) = 0;
    virtual void setNodeName( const String &name ) = 0;

    virtual void moveToDevice( const torch::Device &device )
    {
        _device = device;
        this->to( device );
    }

    const torch::Tensor &getLower() const
    {
        return _lower;
    }
    const torch::Tensor &getUpper() const
    {
        return _upper;
    }

    void setLower( const torch::Tensor &value )
    {
        _lower = value;
        _isLowerBoundCurrent = true;
    }

    void setUpper( const torch::Tensor &value )
    {
        _upper = value;
        _isUpperBoundCurrent = true;
    }

    void setBounds( const torch::Tensor &lower, const torch::Tensor &upper )
    {
        _lower = lower;
        _upper = upper;
        _isLowerBoundCurrent = true;
        _isUpperBoundCurrent = true;
    }

    void clearBounds()
    {
        _lower = torch::Tensor();
        _upper = torch::Tensor();
        _isLowerBoundCurrent = false;
        _isUpperBoundCurrent = false;
    }

    bool isLowerBoundCurrent() const
    {
        return _isLowerBoundCurrent;
    }
    bool isUpperBoundCurrent() const
    {
        return _isUpperBoundCurrent;
    }
    bool hasBounds() const
    {
        return _isLowerBoundCurrent && _isUpperBoundCurrent;
    }

    const Vector<unsigned> &getRequiresInputBounds() const
    {
        return _requiresInputBounds;
    }
    bool isIBPIntermediate() const
    {
        return _ibpIntermediate;
    }

protected:
    unsigned _nodeIndex;
    String _nodeName;
    unsigned _input_size;
    unsigned _output_size;
    torch::Device _device{ torch::kCPU };

    torch::Tensor _lower;
    torch::Tensor _upper;
    bool _isLowerBoundCurrent{ false };
    bool _isUpperBoundCurrent{ false };

    Vector<unsigned> _requiresInputBounds;
    bool _ibpIntermediate{ false };
};

} // namespace NLR

#endif // __TORCH_MODULE_BOUNDED_H__
