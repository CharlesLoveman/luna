/*********************                                                        */
/*! \file BoundedSoftmaxNode.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "BoundedSoftmaxNode.h"
#include <ATen/core/TensorBody.h>
#include <ATen/ops/nan_to_num.h>
#include <ATen/ops/zeros_like.h>

namespace NLR
{

BoundedSoftmaxNode::BoundedSoftmaxNode( const torch::nn::Softmax &softmaxModule,
                                        const String &name )
    : BoundedTorchNode()
    , _softmaxModule( std::make_shared<torch::nn::Softmax>( softmaxModule ) )
{
    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;

    _requiresInputBounds.append( 0 );
    _ibpIntermediate = true;
}

torch::Tensor BoundedSoftmaxNode::forward( const torch::Tensor &input )
{
    if ( input.dim() > 0 )
    {
        _input_size = input.numel();
        _output_size = input.numel();
    }

    return ( *_softmaxModule )( input );
}

void BoundedSoftmaxNode::moveToDevice( const torch::Device &device )
{
    BoundedTorchNode::moveToDevice( device );
    if ( _softmaxModule )
    {
        _softmaxModule->ptr()->to( device );
    }
}

void BoundedSoftmaxNode::setInputSize( unsigned size )
{
    _input_size = size;
}

void BoundedSoftmaxNode::setOutputSize( unsigned size )
{
    _output_size = size;
}

const torch::Tensor move_dim_to_last( const torch::Tensor &t, int64_t dim, int64_t ndim )
{
    if ( dim != ndim - 1 )
    {
        return t.transpose( dim, -1 );
    }
    return t;
}

const torch::Tensor restore_dim( const torch::Tensor &t, const int64_t A_dim, const int64_t dim,
                                 const int64_t ndim )
{
    if ( dim != ndim - 1 )
    {

        return t.transpose( A_dim, -1 );
    }
    return t;
}

torch::Tensor expm1_div_x( const torch::Tensor &x )
{
    const torch::Tensor nonzero_x = torch::where( x == 0, torch::ones_like( x ), x );
    const torch::Tensor ratio = torch::expm1( x ) / nonzero_x;
    return torch::where( x == 0, torch::ones_like( x ), ratio );
}

typedef std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> JacobianComp;

std::tuple<JacobianComp, torch::Tensor>
softmax_lse_lower( const torch::Tensor &lb_p, const torch::Tensor &ub_p, const torch::Tensor &x0_p )
{
    const torch::Tensor max_u = std::get<0>( ub_p.max( -1, true ) );
    const torch::Tensor u_s = ub_p - max_u;
    const torch::Tensor l_s = lb_p - max_u;
    const torch::Tensor x0_s = x0_p - max_u;

    const torch::Tensor diff = u_s - l_s;

    const torch::Tensor exp_l = torch::exp( l_s );
    const torch::Tensor kappa = exp_l * expm1_div_x( diff );
    const torch::Tensor se_x0 = ( exp_l + kappa * ( x0_s - l_s ) ).sum( -1, true );
    const torch::Tensor L_x0 = torch::exp( x0_s ) / se_x0;
    const torch::Tensor V_k = L_x0 / se_x0;

    const JacobianComp J_L_comp( L_x0, V_k, kappa );
    const torch::Tensor dot_kappa_x0 = ( kappa * x0_p ).sum( -1, true );
    const torch::Tensor b_L = L_x0 - ( L_x0 * x0_p ) + ( V_k * dot_kappa_x0 );

    return std::tuple<JacobianComp, torch::Tensor>( J_L_comp, b_L );
}

std::tuple<JacobianComp, torch::Tensor>
softmax_lse_upper( const torch::Tensor &lb_p, const torch::Tensor &ub_p, const torch::Tensor &x0_p,
                   const torch::Tensor &sm_x0, const torch::Tensor &max_x0,
                   const torch::Tensor &sum_exp_x0 )
{
    const torch::Tensor max_u = std::get<0>( ub_p.max( -1, true ) );
    const torch::Tensor u_s = ub_p - max_u;
    const torch::Tensor l_s = lb_p - max_u;
    const torch::Tensor exp_u = torch::exp( u_s );
    const torch::Tensor exp_l = torch::exp( l_s );
    const torch::Tensor sum_exp_u = exp_u.sum( -1, true );
    const torch::Tensor sum_exp_l = exp_l.sum( -1, true );
    const torch::Tensor lower_denominator = exp_l + ( sum_exp_u - exp_u );
    const torch::Tensor upper_denominator = exp_u + ( sum_exp_l - exp_l );

    const torch::Tensor log_p_under = l_s - torch::log( lower_denominator );
    const torch::Tensor log_p_bar = u_s - torch::log( upper_denominator );
    const torch::Tensor p_under = torch::exp( log_p_under );
    const torch::Tensor p_bar = torch::exp( log_p_bar );
    const torch::Tensor log_p_diff = log_p_bar - log_p_under;

    const torch::Tensor use_expm1 = log_p_diff < 0.001;
    const torch::Tensor small_log_diff =
        torch::where( use_expm1, log_p_diff, torch::zeros_like( log_p_diff ) );
    const torch::Tensor c_k_expm1 = p_under * expm1_div_x( small_log_diff );
    const torch::Tensor safe_log_diff =
        torch::where( use_expm1, torch::ones_like( log_p_diff ), log_p_diff );
    const torch::Tensor c_k_secant = ( p_bar - p_under ) / safe_log_diff;
    const torch::Tensor c_k = torch::where( use_expm1, c_k_expm1, c_k_secant );
    const torch::Tensor a_k = p_under - c_k * log_p_under;

    const JacobianComp J_U_comp( c_k, c_k, sm_x0 );

    const torch::Tensor lse_x0 = max_x0 + torch::log( sum_exp_x0 );
    const torch::Tensor U_x0 = a_k - c_k * ( lse_x0 - x0_p );

    const torch::Tensor dot_sm_x0 = ( sm_x0 * x0_p ).sum( -1, true );
    const torch::Tensor b_U = U_x0 - ( c_k * x0_p ) + ( c_k * dot_sm_x0 );

    return std::tuple<JacobianComp, torch::Tensor>( J_U_comp, b_U );
}

torch::Tensor apply_jacobian_components( const torch::Tensor &A, const JacobianComp &J )
{
    const torch::Tensor &D = std::get<0>( J );
    const torch::Tensor &u = std::get<1>( J );
    const torch::Tensor &v = std::get<2>( J );
    const torch::Tensor term1 = A * D.unsqueeze( 0 );
    const torch::Tensor Au = ( A * u.unsqueeze( 0 ) ).sum( -1, true );
    const torch::Tensor term2 = Au * v.unsqueeze( 0 );
    return term1 - term2;
}

BoundA propagate( const BoundA &last_A,
                  const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> &J_pos_comp,
                  const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> &J_neg_comp,
                  const torch::Tensor &b_pos, const torch::Tensor &b_neg, const int64_t dim,
                  const int64_t A_dim, const int64_t ndim, torch::Tensor &bias )
{
    const torch::Tensor A_p =
        move_dim_to_last( last_A.asTensor(), A_dim, last_A.asTensor().ndimension() );
    const torch::Tensor A_pos = A_p.clamp( 0.0 );
    const torch::Tensor A_neg = A_p.clamp( {}, 0.0 );

    const torch::Tensor next_A_pos = apply_jacobian_components( A_pos, J_pos_comp );
    const torch::Tensor next_A_neg = apply_jacobian_components( A_pos, J_neg_comp );

    const BoundA next_A = BoundA( restore_dim( next_A_pos + next_A_neg, A_dim, dim, ndim ) );

    bias =
        ( ( A_pos * b_pos.unsqueeze( 0 ) ).sum( -1 ) + ( A_neg * b_neg.unsqueeze( 0 ) ).sum( -1 ) );
    if ( bias.ndimension() > 2 )
    {
        bias = bias.flatten( 2 ).sum( -1 );
    }
    return next_A;
}

void BoundedSoftmaxNode::boundBackward( const BoundA &last_lA, const BoundA &last_uA,
                                        const Vector<BoundedTensor<torch::Tensor>> &inputBounds,
                                        Vector<Pair<BoundA, BoundA>> &outputA_matrices,
                                        torch::Tensor &lbias, torch::Tensor &ubias )
{
    const torch::Tensor &lb = inputBounds[0].lower();
    const torch::Tensor &ub = inputBounds[0].upper();
    // require_finite_relaxation_bounds(lb, ub, 'softmax')

    const int64_t ndim = lb.ndimension();
    const int64_t axis = _softmaxModule->ptr()->options.dim();
    const int64_t dim = axis >= 0 ? axis : ndim + axis;
    const int64_t A_dim = dim < 0 ? dim : dim + 1;

    const torch::Tensor lb_p = move_dim_to_last( lb, dim, ndim );
    const torch::Tensor ub_p = move_dim_to_last( ub, dim, ndim );

    const torch::Tensor x0_p = ( lb_p + ub_p ) / 2.0;

    const torch::Tensor max_x0 = std::get<0>( x0_p.max( -1, true ) );
    const torch::Tensor x0_shifted = x0_p - max_x0;
    const torch::Tensor exp_x0 = torch::exp( x0_shifted );
    const torch::Tensor sum_exp_x0 = exp_x0.sum( -1, true );
    const torch::Tensor sm_x0 = exp_x0 / sum_exp_x0;

    const auto lse_lower = softmax_lse_lower( lb_p, ub_p, x0_p );
    const auto J_L_comp = std::get<0>( lse_lower );
    const torch::Tensor b_L = std::get<1>( lse_lower );

    const auto lse_upper = softmax_lse_upper( lb_p, ub_p, x0_p, sm_x0, max_x0, sum_exp_x0 );
    const auto J_U_comp = std::get<0>( lse_upper );
    const torch::Tensor b_U = std::get<1>( lse_upper );

    const BoundA lA = propagate( last_lA, J_L_comp, J_U_comp, b_L, b_U, dim, A_dim, ndim, lbias );
    const BoundA uA = propagate( last_uA, J_U_comp, J_L_comp, b_U, b_L, dim, A_dim, ndim, ubias );

    outputA_matrices.append( Pair<BoundA, BoundA>( lA, uA ) );
}

BoundedTensor<torch::Tensor> BoundedSoftmaxNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>> &inputBounds )
{
    const int64_t axis = _softmaxModule->ptr()->options.dim();
    const torch::Tensor &h_L = inputBounds[0].lower();
    const torch::Tensor &h_U = inputBounds[0].upper();
    const torch::Tensor shift = std::get<0>( h_U.max( axis, true ) );
    const torch::Tensor exp_L = torch::exp( h_L - shift );
    const torch::Tensor exp_U = torch::exp( h_U - shift );
    const torch::Tensor lower = exp_L / ( torch::sum( exp_U, axis, true ) - exp_U + exp_L );
    const torch::Tensor upper = exp_U / ( torch::sum( exp_L, axis, true ) - exp_L + exp_U );
    const torch::Tensor finite_inputs =
        ( torch::isfinite( h_L ).all( axis, true ) & torch::isfinite( h_U ).all( axis, true ) );
    const torch::Tensor lower_finite = torch::where(
        finite_inputs, torch::nan_to_num( lower, 0.0, 0.0, 0.0 ), torch::zeros_like( lower ) );
    const torch::Tensor upper_finite = torch::where(
        finite_inputs, torch::nan_to_num( upper, 1.0, 1.0, 1.0 ), torch::ones_like( upper ) );
    return BoundedTensor<torch::Tensor>( lower_finite, upper_finite );
}
} // namespace NLR
