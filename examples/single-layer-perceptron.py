
import functools
import optax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# single-layer linear perceptron loss
@functools.partial(jax.vmap, in_axes=(None, 0))
def network(params, x):
    return jnp.tanh(jnp.dot(params, x)) # harder


# l2-norm loss
def compute_loss(params, x, y):
    y_pred = network(params, x)
    loss = jnp.mean(optax.l2_loss(y_pred, y))
    return loss


v_compute_loss = jax.vmap(compute_loss, in_axes=(1, None, None))


# update rule
def ensemble_grad(u, g, obs_mean, obs_noise_cov, dt=1.0, deterministic=True, key=jax.random.PRNGKey(0)):
    # u: n_par × n_ens, g: n_obs × n_ens
    n_par, n_ens = u.shape
    n_obs = g.shape[0]

    # cross-covariances
    cov_ug = jnp.atleast_2d(jnp.cov(u, g, bias=True)[
                            :n_par, -n_obs:])  # n_par × n_obs
    cov_gg = jnp.atleast_2d(jnp.cov(g, bias=True))  # n_obs × n_obs

    # scale noise using Δt
    scaled_obs_noise_cov = obs_noise_cov / dt  # n_obs × n_obs
    noise = jax.random.multivariate_normal(key, jnp.zeros(
        (n_obs, 1)), obs_noise_cov, (n_ens,)).T

    # add obs_mean (n_obs) to each column of noise (n_obs × n_ens) if
    # G is deterministic
    y = obs_mean + \
        noise if deterministic else jnp.repeat(obs_mean, n_ens, axis=1)

    # update step
    # n_obs × n_obs \ [n_obs × n_ens] --> tmp is [n_obs × n_ens]
    tmp = jnp.matmul(jnp.linalg.inv(cov_gg + scaled_obs_noise_cov), y - g)
    u_update = -jnp.matmul(cov_ug, tmp)  # [n_par × n_ens]

    return u_update


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    # set optimizer to gradient-free or gradient-based optimization
    gradient_free = True

    # Generate some data.
    # 16 batches, 2 nodes in network
    target_params = 0.5 # nn node weights
    xs = jax.random.normal(key, (16, 10))
    ys = jnp.tanh(jnp.sum(xs * target_params, axis=-1)) # harder

    # Exponential decay of the learning rate.
    start_learning_rate = 1e-1
    scheduler = optax.exponential_decay(
        init_value=start_learning_rate,
        transition_steps=1000,
        decay_rate=0.99)

    # Combining gradient transforms using `optax.chain`.
    if gradient_free:
        optimizer = optax.chain(
            # Clip by the gradient by the global norm.
            optax.trace(decay=0.25, nesterov=True, accumulator_dtype=None), # momentum
            optax.scale(-1.0)
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
            optax.scale_by_adam(),  # Use the updates from adam.
            # Use the learning rate from the scheduler.
            optax.scale_by_schedule(scheduler), 
            # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
            optax.scale(-1.0)
        )

    # EKI \w normal prior
    n_obs = 1  # Number of synthetic observations from G(u)
    noise_level = 1e-8  # Defining the observation noise level
    obs_mean = jnp.atleast_2d([0])
    obs_noise_cov = noise_level * jnp.eye(n_obs)
    noise = jax.random.multivariate_normal(key, jnp.zeros((n_obs, 1)), obs_noise_cov)
    n_par = xs.shape[1] # number of model parameters
    prior_mean = jnp.zeros(n_par) # prior mean
    prior_cov = jnp.eye(n_par) # prior covariance
    prior = jax.random.multivariate_normal # prior distrbution
    n_ens = 50  # number of ensemble members

    # Initialize parameters of the model + optimizer.
    if gradient_free:
        params = prior(key, prior_mean, prior_cov, (n_ens,)
                    ).T  # sample parameters from prior
    else:
        params = jnp.array([0.0]*xs.shape[1])  # Recall target_params=0.5.

    # Initialize optimizer
    n_iter = 250
    opt_state = optimizer.init(params)

    # A simple update loop.
    params_storage = [params.copy()]
    for _ in range(n_iter):
        if gradient_free:
            evals = jnp.atleast_2d(v_compute_loss(params, xs, ys))
            grads = ensemble_grad(params, evals, obs_mean, obs_noise_cov)
        else:
            grads = jax.grad(compute_loss)(params, xs, ys)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params_storage.append(params.copy())

    print("target params:")
    print(target_params)
    print("estimated params:")
    if gradient_free:
        print(jnp.mean(params, axis=1))
    else:
        print(params)

    # do plotting
    # u_init = params_storage[1]
    # u1_min = min(min(u[0, :]) for u in params_storage)
    # u1_max = max(max(u[0, :]) for u in params_storage)
    # u2_min = min(min(u[1, :]) for u in params_storage)
    # u2_max = max(max(u[1, :]) for u in params_storage)
    # xlim = (u1_min, u1_max)
    # ylim = (u2_min, u2_max)
    # for i, u in enumerate(params_storage):
    #     plt.cla()
    #     plt.title(str(i))
    #     plt.plot(u[0, :], u[1, :], 'kx')
    #     plt.xlim(xlim)
    #     plt.ylim(ylim)
    #     plt.show()
    #     plt.pause(0.25)
