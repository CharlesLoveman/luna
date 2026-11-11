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

const torch::Tensor moveDimToLast( const torch::Tensor &t, int64_t dim, int64_t ndim )
{
    if ( dim != ndim - 1 )
    {
        return t.transpose( dim, -1 );
    }
    return t;
}

const torch::Tensor restoreDim( const torch::Tensor &t, const int64_t A_dim, const int64_t dim,
                                const int64_t ndim )
{
    if ( dim != ndim - 1 )
    {

        return t.transpose( A_dim, -1 );
    }
    return t;
}

torch::Tensor expm1DivX( const torch::Tensor &x )
{
    const torch::Tensor nonzero_x = torch::where( x == 0, torch::ones_like( x ), x );
    const torch::Tensor ratio = torch::expm1( x ) / nonzero_x;
    return torch::where( x == 0, torch::ones_like( x ), ratio );
}

typedef std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> JacobianComp;

std::tuple<JacobianComp, torch::Tensor>
softmaxLseLower( const torch::Tensor &lb_p, const torch::Tensor &ub_p, const torch::Tensor &x0_p )
/*
 * lb_p: lower bound of the input to softmax
 * ub_p: upper bound of the input to softmax
 * x0_p: center of [lb_p, ub_p]
 *
 * Computes lower bound (39-40) in Wei et al 2023
 *   J_L = diag(L(x)) - exp(x0) * (1/sum_exp(x0))^2 * (exp(u) - exp(l)) / (u - l)
 * with
 *   L(x) = exp(x) / SE(x; l, u)
 *   Eq 9. SE(x; l, u) = sum((u - x)/(u - l) * exp(l) + (x - l)(u - l) * exp(u))
 */
{
    const torch::Tensor max_u = std::get<0>( ub_p.max( -1, true ) );
    const torch::Tensor u_s = ub_p - max_u;
    const torch::Tensor l_s = lb_p - max_u;
    const torch::Tensor x0_s = x0_p - max_u;

    const torch::Tensor diff = u_s - l_s;

    const torch::Tensor exp_l = torch::exp( l_s );

    // kappa = exp(l) * (exp(u - l) - 1) / (u - l)
    //       = (exp(u) - exp(l)) / (u - l)
    // avoids cancellation when interval is small
    const torch::Tensor kappa = exp_l * expm1DivX( diff );

    // se is the chordal upper bound on sum_exp(x0)
    // se_x0 = sum(exp(l) + (exp(u) - exp(l))/(u - l) * (x - l))
    //       = sum((u - l - x + l)/(u - l) * exp(l) + (x - l) * exp(u))
    //       = sum((u - x)/(u - l) * exp(l) + (x - l)/(u - l) * exp(u))
    //       = SE(x; l, u)
    const torch::Tensor se_x0 = ( exp_l + kappa * ( x0_s - l_s ) ).sum( -1, true );

    // L(x) = exp(x) / SE(x; l, u)
    const torch::Tensor L_x0 = torch::exp( x0_s ) / se_x0;
    // V_k = exp(x) / SE(x; l, u)^2
    const torch::Tensor V_k = L_x0 / se_x0;

    // J_L = diag(L(x)) - exp(x) * (1/SE(x; l, u))^2 * (exp(u) - exp(l)) / (u - l)
    //     = diag(L_x0) - outer(V_k, kappa)
    const JacobianComp J_L_comp( L_x0, V_k, kappa );

    // b_L = L_x0 - J_L @ x0_p
    const torch::Tensor dot_kappa_x0 = ( kappa * x0_p ).sum( -1, true );
    const torch::Tensor b_L = L_x0 - ( L_x0 * x0_p ) + ( V_k * dot_kappa_x0 );

    return std::tuple<JacobianComp, torch::Tensor>( J_L_comp, b_L );
}

std::tuple<JacobianComp, torch::Tensor>
softmaxLseUpper( const torch::Tensor &lb_p, const torch::Tensor &ub_p, const torch::Tensor &x0_p )
/* lb_p: lower bound of the input to softmax
 * ub_p: upper bound of the input to softmax
 * x0_p: center of [lb_p, ub_p]
 *
 * Computes upper bound (46-47) in Wei et al 2023.
 *   U_1 = -(p_bar - p_under)/(log(p_bar) - log(p_under)) * (exp(x) / SE(x) - 1)
 *   U_i = -(p_bar - p_under)/(log(p_bar) - log(p_under)) * (exp(x) / SE(x))
 *   J_U = diag(U_1) + U_i
 * with
 *   5a. p_under = 1 / sum(exp(u~))
 *   5b. p_bar = 1 / sum(exp(l~))
 *     u~_0 = 1
 *     u~_i = exp(u_i - l_1)
 *     l~_0 = 1
 *     l~_i = exp(l_i - u_1)
 */
{
    // exp(center), shifted to avoid overflow
    const torch::Tensor max_x0 = std::get<0>( x0_p.max( -1, true ) );
    const torch::Tensor x0_shifted = x0_p - max_x0;
    const torch::Tensor exp_x0 = torch::exp( x0_shifted );
    const torch::Tensor sum_exp_x0 = exp_x0.sum( -1, true );

    // softmax of center (shifted)
    const torch::Tensor sm_x0 = exp_x0 / sum_exp_x0;

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
    const torch::Tensor log_p_diff = log_p_bar - log_p_under;
    // lower bound on p
    const torch::Tensor p_under = torch::exp( log_p_under );
    // upper bound on p
    const torch::Tensor p_bar = torch::exp( log_p_bar );

    // Compute secant of exp
    // c_k_secant = (p_bar - p_under) / (log(p_bar) - log(p_under))
    // c_k_expm1 is equivalent to c_k_secant
    const torch::Tensor use_expm1 = log_p_diff < 0.001;
    const torch::Tensor small_log_diff =
        torch::where( use_expm1, log_p_diff, torch::zeros_like( log_p_diff ) );
    // safe when log_diff small
    const torch::Tensor c_k_expm1 = p_under * expm1DivX( small_log_diff );
    const torch::Tensor safe_log_diff =
        torch::where( use_expm1, torch::ones_like( log_p_diff ), log_p_diff );
    // used otherwise
    const torch::Tensor c_k_secant = ( p_bar - p_under ) / safe_log_diff;
    const torch::Tensor c_k = torch::where( use_expm1, c_k_expm1, c_k_secant );
    const torch::Tensor a_k = p_under - c_k * log_p_under;

    // J_U = diag(c_k) - outer(c_k, sm_x0)
    const JacobianComp J_U_comp( c_k, c_k, sm_x0 );

    const torch::Tensor lse_x0 = max_x0 + torch::log( sum_exp_x0 );
    const torch::Tensor U_x0 = a_k - c_k * ( lse_x0 - x0_p );

    // b_U = U_x0 - J_U @ x0_p
    const torch::Tensor dot_sm_x0 = ( sm_x0 * x0_p ).sum( -1, true );
    const torch::Tensor b_U = U_x0 - ( c_k * x0_p ) + ( c_k * dot_sm_x0 );

    return std::tuple<JacobianComp, torch::Tensor>( J_U_comp, b_U );
}

torch::Tensor applyJacobianComponents( const torch::Tensor &A, const JacobianComp &J )
/*
 * A: input tensor
 * J: Jacobian components (D, u, v), J = diag(D) * outer(u, v)
 * Returns A @ J
 */
{
    const torch::Tensor &D = std::get<0>( J );
    const torch::Tensor &u = std::get<1>( J );
    const torch::Tensor &v = std::get<2>( J );
    const torch::Tensor term1 = A * D.unsqueeze( 0 );
    const torch::Tensor Au = ( A * u.unsqueeze( 0 ) ).sum( -1, true );
    const torch::Tensor term2 = Au * v.unsqueeze( 0 );
    return term1 - term2;
}

torch::Tensor
propagateTensor( const torch::Tensor &last_A,
                 const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> &J_pos_comp,
                 const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> &J_neg_comp,
                 const torch::Tensor &b_pos, const torch::Tensor &b_neg, const int64_t dim,
                 const int64_t A_dim, const int64_t ndim, torch::Tensor &bias )
/*
 * Split A into A+ and A-, then back propagate A_next = (A+ * J+) + (A- * J-)
 */
{
    // Move softmax axis to the last dimension for A to agree with Jacobian components
    const torch::Tensor A_p = moveDimToLast( last_A, A_dim, last_A.ndimension() );
    const torch::Tensor A_pos = A_p.clamp_min( 0.0 );
    const torch::Tensor A_neg = A_p.clamp_max( 0.0 );

    const torch::Tensor next_A_pos = applyJacobianComponents( A_pos, J_pos_comp );
    const torch::Tensor next_A_neg = applyJacobianComponents( A_neg, J_neg_comp );

    // Restore softmax axis to its original position in the next_A matrix
    const torch::Tensor next_A = restoreDim( next_A_pos + next_A_neg, A_dim, dim, ndim );

    bias =
        ( ( A_pos * b_pos.unsqueeze( 0 ) ).sum( -1 ) + ( A_neg * b_neg.unsqueeze( 0 ) ).sum( -1 ) );
    if ( bias.ndimension() > 2 )
    {
        bias = bias.flatten( 2 ).sum( -1 );
    }
    return next_A;
}

torch::Tensor reshapePayload( const torch::Tensor &A, const torch::Tensor &input )
/*
 * A: a tensor of shape (spec, lanes or 1, payload_numel)
 * input: a tensor of shape (...input shape)
 *
 * Requires product(input shape) = payload_numel
 *
 * Returns A reshaped to (spec, lanes or 1, ...input shape)
 */
{
    std::vector<int64_t> shape = {};
    shape.push_back( A.size( 0 ) );
    shape.push_back( A.size( 1 ) );
    int64_t payload_numel = 1;
    for ( int64_t d = 1; d < input.dim(); ++d )
    {
        shape.push_back( input.size( d ) );
        payload_numel *= input.size( d );
    }
    if ( A.size( -1 ) != payload_numel )
    {
        throw std::runtime_error(
            "BoundedSoftmaxNode: expected A dim -1 == product(input sizes), got A dim -1: " +
            std::to_string( A.size( -1 ) ) +
            ", product(input sizes): " + std::to_string( payload_numel ) );
    }
    return A.reshape( shape );
}

torch::Tensor restoreShape( const torch::Tensor &A )
/*
 * Inverse of reshapePayload
 *
 * A: a tensor of shape (spec, lanes or 1, ...input shape)
 *
 * Returns A reshaped to (spec, lanes or 1, product(input shape))
 */
{
    std::vector<int64_t> shape = {};
    shape.push_back( A.size( 0 ) );
    shape.push_back( A.size( 1 ) );
    int64_t payload_numel = 1;
    for ( int64_t d = 2; d < A.dim(); ++d )
    {
        payload_numel *= A.size( d );
    }
    shape.push_back( payload_numel );
    return A.reshape( shape );
}

void BoundedSoftmaxNode::boundBackwardTensor(
    const torch::Tensor &last_lA, const torch::Tensor &last_uA,
    const Vector<BoundedTensor<torch::Tensor>> &inputBounds,
    Vector<Pair<BoundA, BoundA>> &outputA_matrices, torch::Tensor &lbias, torch::Tensor &ubias )
{
    const torch::Tensor &lb = inputBounds[0].lower();
    const torch::Tensor &ub = inputBounds[0].upper();

    const int64_t ndim = lb.ndimension();
    const int64_t axis = _softmaxModule->ptr()->options.dim();
    const int64_t dim = axis >= 0 ? axis : ndim + axis;
    const int64_t A_dim = dim < 0 ? dim : dim + 1;

    // Move softmax axis to the last dimension
    const torch::Tensor lb_p = moveDimToLast( lb, dim, ndim );
    const torch::Tensor ub_p = moveDimToLast( ub, dim, ndim );

    // Reshape to (spec, lanes or 1, input shape)
    const torch::Tensor lA = reshapePayload( last_lA, lb );
    const torch::Tensor uA = reshapePayload( last_uA, lb );

    // Center of interval [lb_p, ub_p]
    const torch::Tensor x0_p = ( lb_p + ub_p ) / 2.0;

    // log-sum-exp lower bound jacobian and bias
    const auto [J_L_comp, b_L] = softmaxLseLower( lb_p, ub_p, x0_p );
    // log-sum-exp upper bound jacobian and bias
    const auto [J_U_comp, b_U] = softmaxLseUpper( lb_p, ub_p, x0_p );

    // Compute new A from last A, jacobian, and bias terms
    const torch::Tensor new_lA =
        propagateTensor( lA, J_L_comp, J_U_comp, b_L, b_U, dim, A_dim, ndim, lbias );
    const torch::Tensor new_uA =
        propagateTensor( uA, J_U_comp, J_L_comp, b_U, b_L, dim, A_dim, ndim, ubias );

    // Reshape back to (spec, lanes or 1, product(input shape))
    outputA_matrices.append(
        Pair<BoundA, BoundA>( restoreShape( new_lA ), restoreShape( new_uA ) ) );
}

void BoundedSoftmaxNode::boundBackwardPatches(
    std::shared_ptr<Patches> last_lA, std::shared_ptr<Patches> last_uA,
    const Vector<BoundedTensor<torch::Tensor>> &inputBounds,
    Vector<Pair<BoundA, BoundA>> &outputA_matrices, torch::Tensor &lbias, torch::Tensor &ubias )
{
    throw std::runtime_error( "BoundedSoftmaxNode: Patches mode not implemented" );
}

void BoundedSoftmaxNode::boundBackward( const BoundA &last_lA, const BoundA &last_uA,
                                        const Vector<BoundedTensor<torch::Tensor>> &inputBounds,
                                        Vector<Pair<BoundA, BoundA>> &outputA_matrices,
                                        torch::Tensor &lbias, torch::Tensor &ubias )
/*
 * Backward bound propagation for softmax
 * Uses the log-sum-exp bounds in Wei et al 2023
 */
{
    if ( last_lA.isTensor() && last_uA.isTensor() )
    {
        boundBackwardTensor( last_lA.asTensor(), last_uA.asTensor(), inputBounds, outputA_matrices,
                             lbias, ubias );
    }
    else if ( last_lA.isPatches() && last_uA.isPatches() )
    {
        boundBackwardPatches( last_lA.asPatches(), last_uA.asPatches(), inputBounds,
                              outputA_matrices, lbias, ubias );
    }
    else
    {
        throw std::runtime_error( "BoundedSoftmaxNode: expected last_lA and last_uA to share the "
                                  "same type, got last_lA isTensor: " +
                                  std::to_string( last_lA.isTensor() ) +
                                  ", last_uA isTensor: " + std::to_string( last_uA.isTensor() ) );
    }
}

BoundedTensor<torch::Tensor> BoundedSoftmaxNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>> &inputBounds )
/*
 * Softmax is monotonic so Softmax([l, u]) = [Softmax(l), Softmax(u)]
 */
{
    const int64_t axis = _softmaxModule->ptr()->options.dim();
    const torch::Tensor &h_L = inputBounds[0].lower();
    if ( _output_size == 0 )
    {
        setOutputSize( h_L.numel() );
    }
    const torch::Tensor &h_U = inputBounds[0].upper();

    // Shift to avoid overflow in exp
    const torch::Tensor shift = std::get<0>( h_U.max( axis, true ) );
    const torch::Tensor exp_L = torch::exp( h_L - shift );
    const torch::Tensor exp_U = torch::exp( h_U - shift );
    const torch::Tensor lower = exp_L / ( torch::sum( exp_U, axis, true ) - exp_U + exp_L );
    const torch::Tensor upper = exp_U / ( torch::sum( exp_L, axis, true ) - exp_L + exp_U );

    // Clamp all values (including nan) to the range [0, 1]
    const torch::Tensor finite_inputs =
        ( torch::isfinite( h_L ).all( axis, true ) & torch::isfinite( h_U ).all( axis, true ) );
    const torch::Tensor lower_finite = torch::where(
        finite_inputs, torch::nan_to_num( lower, 0.0, 0.0, 0.0 ), torch::zeros_like( lower ) );
    const torch::Tensor upper_finite = torch::where(
        finite_inputs, torch::nan_to_num( upper, 1.0, 1.0, 1.0 ), torch::ones_like( upper ) );
    return BoundedTensor<torch::Tensor>( lower_finite.clamp( 0.0, 1.0 ),
                                         upper_finite.clamp( 0.0, 1.0 ) );
}
} // namespace NLR
