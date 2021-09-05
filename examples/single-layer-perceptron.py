#!\bin\python

import functools
import optax
import opterax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# single-layer linear perceptron loss
@functools.partial(jax.vmap, in_axes=(None, 0))
def model(params, x):
  return jnp.tanh(jnp.dot(params, x))


# problem-dependent loss
def my_loss(y_pred, y):
  return jnp.mean(optax.l2_loss(y_pred, y))


# loss for gradient-based optimization
def model_loss(params, x, y): 
  y_pred = model(params, x)
  return my_loss(y_pred, y)


if __name__ == "__main__":
  key = jax.random.PRNGKey(42)

  # set optimizer to gradient-free or gradient-based optimization
  gradient_free = True
  plotting = False

  # Generate a batch of some data.
  target_params = 0.5 # nn node weights
  n_data = 16
  n_par = 10
  xs = jax.random.normal(key, (n_data, n_par))
  ys = jnp.tanh(jnp.sum(xs * target_params, axis=-1))

  # Combining gradient transforms using `optax.chain`.
  if gradient_free:
      # all we do is go down-gradient to achieve good results
      optimizer = optax.chain(
          optax.scale(-1.0) # since we are going downhill
      )
  else:
      # Exponential decay of the learning rate (only for gradient-based)
      start_learning_rate = 1e-1
      scheduler = optax.exponential_decay(
          init_value=start_learning_rate,
          transition_steps=1000,
          decay_rate=0.99)

      # conventionally, we add fancy things
      optimizer = optax.chain(
          optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
          optax.scale_by_adam(),  # Use the updates from adam.
          optax.scale_by_schedule(scheduler), 
          optax.scale(-1.0) # since we are going downhill
      )

  # Initialize parameters of the model + optimizer.
  if gradient_free:
      # EKI \w normal prior parameters
      n_data, n_par = xs.shape  # Number of synthetic observations from G(u)
      obs_err_cov = 1e-8 * jnp.eye(n_data)
      prior_mu = jnp.zeros(n_par) # prior mean
      prior_sig = jnp.eye(n_par) # prior covariance
      prior = jax.random.multivariate_normal # prior distrbution
      n_ens = 25 # number of ensemble members

      # sample a starting ensemble
      params = prior(key, prior_mu, prior_sig, (n_ens,)).T  # sample parameters from prior
  else:
      # conventionally, we guess and initial parameter vector
      params = jnp.array([0.0]*xs.shape[1])

  # Initialize optimizer
  n_iter = 100
  opt_state = optimizer.init(params)

  # A simple update loop.
  params_storage = [params.copy()]
  for _ in range(n_iter):
      if gradient_free:
          # we use EKI to approximate our gradients via ensembles
          # we get one prediction per ensemble member
          y_pred = jax.vmap(model, in_axes=(1, None))(params, xs).T
          # taking the approximate gradient of the loss has a slightly
          # different formulation using ensembles
          grads = opterax.grad_eki(my_loss, obs_err_cov)(params, y_pred, ys)
      else:
          # conventional gradient-based updates
          grads = jax.grad(model_loss)(params, xs, ys)
      updates, opt_state = optimizer.update(grads, opt_state)
      params = optax.apply_updates(params, updates)
      params_storage.append(params.copy())

  print("target params: ", target_params) 
  print("estimated params:")
  if gradient_free:
      print(jnp.mean(params, axis=1))
  else:
      print(params)

  if gradient_free and plotting:
    u_init = params_storage[1]
    u1_min = min(min(u[0, :]) for u in params_storage)
    u1_max = max(max(u[0, :]) for u in params_storage)
    u2_min = min(min(u[1, :]) for u in params_storage)
    u2_max = max(max(u[1, :]) for u in params_storage)
    xlim = (u1_min, u1_max)
    ylim = (u2_min, u2_max)
    for i, u in enumerate(params_storage):
        plt.cla()
        plt.title(str(i))
        plt.plot(u[0, :], u[1, :], 'kx')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()
        plt.pause(0.25)
