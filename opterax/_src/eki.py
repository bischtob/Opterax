import jax
import jax.numpy as jnp

def grad_eki(loss_fcn, obs_noise_cov):
  # just like jax.grad, we use a closure here
  def _grad_eki(params, y_pred, ys):
    n_par = params.shape[0]
    n_obs = y_pred.shape[0]
  
    # equation (13) in stuart and kovachki
    grad_y = jax.vmap(jax.grad(loss_fcn), in_axes=(1, None))(y_pred, ys).T
  
    # cross-covariances
    cov_pe = jnp.atleast_2d(jnp.cov(params, y_pred, bias=True)[
                              :n_par, -n_obs:])  # n_par * n_obs
    cov_ee = jnp.atleast_2d(jnp.cov(y_pred, bias=True))  # n_obs * n_obs
  
    # ensemble gradient computation step
    # n_obs * n_obs \ [n_obs * n_ens] --> tmp is [n_obs * n_ens]
    tmp = jnp.matmul(jnp.linalg.inv(cov_ee + obs_noise_cov), grad_y)
    grad = jnp.matmul(cov_pe, tmp)  # [n_par * n_ens]
  
    return grad

  return _grad_eki
