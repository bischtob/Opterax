"""Approximate gradients through Ensemble Kalman Inversion."""

import chex
import jax
import jax.numpy as jnp

from typing import Callable

def grad_eki(loss_fcn: Callable, obs_err_cov: chex.Array) -> Callable:
  """Produces an approximate gradient function.
  Args:
    loss_fcn: a symmetric loss function that evaluates the distance
    between truth and predictions.
    obs_err_cov: the covariance matrix of observational noise.
  Returns:
    A function that approximately evaluates the gradient of a forward model,
    given model parameters, forward model predictions, and truth to evaluate
    against.
  """

  # just like jax.grad, we use a closure here
  def _grad(params:chex.Array, y_pred:chex.Array, ys:chex.Array) -> chex.Array:
    n_par = params.shape[0]
    n_obs = y_pred.shape[0]

    # equation (13) in stuart and kovachki
    grad_y = jax.vmap(jax.grad(loss_fcn), in_axes=(1, None))(y_pred, ys).T

    # cross-covariances
    # TODO!: need to be able to work with JAX PyTrees
    cov_pe = jnp.atleast_2d(jnp.cov(params, y_pred, bias=True)[
                              :n_par, -n_obs:])  # n_par * n_obs
    cov_ee = jnp.atleast_2d(jnp.cov(y_pred, bias=True))  # n_obs * n_obs

    # ensemble gradient computation step
    # n_obs * n_obs \ [n_obs * n_ens] --> tmp is [n_obs * n_ens]
    tmp = jnp.matmul(jnp.linalg.inv(cov_ee + obs_err_cov), grad_y)
    grad = jnp.matmul(cov_pe, tmp)  # [n_par * n_ens]
    return grad

  return _grad
