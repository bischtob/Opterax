import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import random as rnd

# Update follows eqns. (4) and (5) of Schillings and Stuart (2017)

def update_ensemble(u, g, obs_mean, obs_noise_cov, dt=1.0, deterministic=True, key=rnd.PRNGKey(0)):
    # u: n_par × n_ens, g: n_obs × n_ens
    n_par, n_ens = u.shape
    n_obs = g.shape[0]

    # cross-covariances
    cov_ug = jnp.atleast_2d(jnp.cov(u, g, bias=True)[:n_par,-n_obs:]) # n_par × n_obs
    cov_gg = jnp.atleast_2d(jnp.cov(g, bias=True)) # n_obs × n_obs

    # scale noise using Δt
    scaled_obs_noise_cov = obs_noise_cov / dt # n_obs × n_obs
    noise = rnd.multivariate_normal(key, jnp.zeros((n_obs, 1)), obs_noise_cov, (n_ens,)).T

    # add obs_mean (n_obs) to each column of noise (n_obs × n_ens) if
    # G is deterministic
    y = obs_mean + noise if deterministic else jnp.repeat(obs_mean, n_ens, axis=1)

    # update step
    # n_obs × n_obs \ [n_obs × n_ens] --> tmp is [n_obs × n_ens]
    tmp = jnp.matmul(jnp.linalg.inv(cov_gg + scaled_obs_noise_cov), y - g)
    u_updated = u + jnp.matmul(cov_ug, tmp) # [n_par × n_ens]

    return u_updated


if __name__ == "__main__":
    # rng for reproducibility
    key = rnd.PRNGKey(0)

    # set up observational noise
    n_obs = 1  # Number of synthetic observations from G(u)
    noise_level = 1e-8  # Defining the observation noise level
    obs_noise_cov = noise_level * jnp.eye(n_obs)
    noise = rnd.multivariate_normal(key, jnp.zeros((n_obs, 1)), obs_noise_cov)

    # set up the loss function (unique minimum)
    @jax.jit
    def objective(u):
        du = u - jnp.array([1.0, -1.0])
        return jnp.linalg.norm(du)
    u_star = jnp.array([1.0, -1.0])  # Loss Function Minimum
    obs_mean = objective(u_star) + noise
    vobjective = jax.vmap(objective, in_axes=1)

    # set up prior
    n_par = len(u_star)
    prior_mean = jnp.zeros(n_par)
    prior_cov = jnp.eye(n_par)
    prior = rnd.multivariate_normal

    # TODO!: Here we would set up an instance of an optimizer object

    # do optimization loop
    n_ens = 50 # number of ensemble members
    n_iter = 1 # number of EKI iterations
    ensemble = prior(key, prior_mean, prior_cov, (n_ens,)).T
    storage_g = []
    storage_u = [ensemble.copy()]
    for i in range(n_iter):
        evaluations = jnp.expand_dims(vobjective(ensemble).T, axis=0)
        ensemble = update_ensemble(ensemble, evaluations, obs_mean, obs_noise_cov)
        storage_u.append(ensemble.copy())
        storage_g.append(evaluations.copy())

    print(obs_mean.shape)

    # # do plotting
    # u_init = storage_u[1]
    # u1_min = min(min(u[0, :]) for u in storage_u)
    # u1_max = max(max(u[0, :]) for u in storage_u)
    # u2_min = min(min(u[1, :]) for u in storage_u)
    # u2_max = max(max(u[1, :]) for u in storage_u)
    # xlim = (u1_min, u1_max)
    # ylim = (u2_min, u2_max)
    # for i, u in enumerate(storage_u):
    #     plt.cla()
    #     plt.title(str(i))
    #     plt.plot(u[0, :], u[1, :], 'kx')
    #     plt.xlim(xlim)
    #     plt.ylim(ylim)
    #     plt.show()
    #     plt.pause(0.25)
