/*********************                                                        */
/*! \file BoundResult.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __BOUND_RESULT_H__
#define __BOUND_RESULT_H__

#include <torch/torch.h>
#include <memory>
#include <variant>
#include "conv/Patches.h"

namespace NLR
{

class BoundA
{
public:
    std::variant<torch::Tensor, std::shared_ptr<Patches>> data;

    BoundA()
        : data( torch::Tensor() )
    {
    }
    BoundA( torch::Tensor t )
        : data( t )
    {
    }
    BoundA( std::shared_ptr<Patches> p )
        : data( p )
    {
    }

    bool isTensor() const
    {
        return std::holds_alternative<torch::Tensor>( data );
    }

    bool isPatches() const
    {
        return std::holds_alternative<std::shared_ptr<Patches>>( data );
    }

    torch::Tensor asTensor() const
    {
        if ( isTensor() )
            return std::get<torch::Tensor>( data );
        return torch::Tensor();
    }

    std::shared_ptr<Patches> asPatches() const
    {
        if ( isPatches() )
            return std::get<std::shared_ptr<Patches>>( data );
        return nullptr;
    }

    bool defined() const
    {
        if ( isTensor() )
            return std::get<torch::Tensor>( data ).defined();
        if ( isPatches() )
            return std::get<std::shared_ptr<Patches>>( data ) != nullptr;
        return false;
    }

    int inferSpecDim() const
    /*
     * Returns the size of the spec dim for this BoundA
     */
    {
        if ( !defined() )
        {
            // Default value 1, could throw an error instead?
            return 1;
        }
        if ( isTensor() )
        {
            torch::Tensor t = asTensor();
            if ( !t.defined() )
                return 1;
            if ( t.dim() >= 2 )
                return (int)t.size( 0 );
            return 1;
        }
        if ( isPatches() )
        {
            auto p = asPatches();
            if ( !p )
                return 1;
            if ( p->is_sparse() )
            {
                return (int)p->patches.size( 0 );
            }
            if ( p->patches.dim() == 7 )
            {
                return (int)( p->patches.size( 0 ) * p->patches.size( 2 ) * p->patches.size( 3 ) );
            }
            return (int)p->patches.size( 0 );
        }
        return 1;
    }
};

} // namespace NLR

#endif // __BOUND_RESULT_H__
