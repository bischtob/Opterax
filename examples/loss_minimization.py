import jax.numpy as jnp

from jax import random as rnd

# Update follows eqns. (4) and (5) of Schillings and Stuart (2017)
def update_ensemble(u, g, obs_mean, obs_noise_cov, dt = 1.0, deterministic = True):
    # u: N_par × N_ens, g: N_obs × N_ens
    N_ens = u.shape[1]
    N_obs = g.shape[0]

    # means and covariances
    u_mean = jnp.mean(u, axis=1)
    g_mean = jnp.mean(g, axis=1)
    du = (u.transpose() - u_mean).transpose()
    dg = (g.transpose() - g_mean).transpose()
    cov_ug = jnp.matmul(du, dg.transpose()) # N_par × N_obs
    cov_gg = jnp.matmul(dg, dg.transpose()) # N_obs × N_obs

    # scale noise using Δt
    scaled_obs_noise_cov = obs_noise_cov / dt # N_obs × N_obs
    noise = rnd.normal(key, (N_obs, N_ens))

    # add obs_mean (N_obs) to each column of noise (N_obs × N_ens) if
    # G is deterministic
    y = obs_mean + noise if deterministic else jnp.repeat(obs_mean, N_ens, axis=1)

    # update step
    # N_obs × N_obs \ [N_obs × N_ens] --> tmp is [N_obs × N_ens]
    tmp = jnp.matmul(jnp.linalg.inv(cov_gg + scaled_obs_noise_cov), y - g)
    u_updated = u + jnp.matmul(cov_ug, tmp) # [N_par × N_ens]

    return u_updated

if __name__ == "__main__":
    # rng
    key = rnd.PRNGKey(0)

    N_par, N_obs, N_ens = 3, 1, 10
    u = jnp.ones((N_par, N_ens))
    g = jnp.ones((N_obs, N_ens))
    obs_mean = jnp.ones((N_obs, 1))
    obs_noise_cov = 1e-6 * jnp.eye(N_obs)

    update_ensemble(u, g, obs_mean, obs_noise_cov)