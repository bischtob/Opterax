import jax.numpy as jnp

from jax import random as rnd

# TODO!: Add reference for equations
def update_ensemble(u, g, obs_mean, obs_noise_cov, prior_mean, prior_cov, key=rnd.PRNGKey(0)):
    # u: N_par × N_ens
    N_par, N_ens = u.shape

    # means and covariances
    u_mean = jnp.mean(u, axis=1)
    g_mean = jnp.mean(g, axis=1)
    du = (u.T - u_mean).T
    cov_uu = jnp.matmul(du, du.T) # N_par × N_obs

    # update step
    E = (g.T - g_mean).T # N_obs × N_ens
    R = (g.T - obs_mean).T # N_obs × N_ens
    D = 1 / N_ens * (E.T * jnp.linalg.inv(obs_noise_cov) * R)  # N_ens × N_ens
    dt = 1 / (jnp.linalg.norm(D) + 1e-8)
    noise = rnd.multivariate_normal(key, jnp.zeros(N_par), cov_uu, (N_ens,)).T

    implicit1 = jnp.eye(N_par) + dt * (jnp.linalg.inv(prior_cov) * cov_uu).T
    implicit2 = u - dt * jnp.matmul(du, D) + dt * jnp.matmul(cov_uu, jnp.matmul(jnp.linalg.inv(prior_cov), prior_mean))
    implicit = jnp.matmul(jnp.linalg.inv(implicit1), implicit2)
    u_updated = implicit + jnp.sqrt(2 * dt) * noise

    return u_updated

if __name__ == "__main__":
    N_par, N_obs, N_ens = 3, 1, 10
    u = jnp.ones((N_par, N_ens))
    g = jnp.ones((N_obs, N_ens))
    obs_mean = jnp.ones((N_obs, 1))
    obs_noise_cov = 1e-6 * jnp.eye(N_obs)
    prior_mean = 10.0 * jnp.ones((N_par, 1))
    prior_cov = 1.0 * jnp.eye(N_par)

    print(update_ensemble(u, g, obs_mean, obs_noise_cov, prior_mean, prior_cov))