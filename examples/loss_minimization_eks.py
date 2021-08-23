import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import random as rnd

# TODO!: Add reference for equations
def update_ensemble(u, g, obs_mean, obs_noise_cov, prior_mean, prior_cov, key=rnd.PRNGKey(0)):
    # u: N_par × N_ens
    N_par, N_ens = u.shape

    # means and covariances
    u_mean = jnp.mean(u, axis=1)
    g_mean = jnp.mean(g, axis=1)
    du = (u.T - u_mean).T
    cov_uu = cov(u, u, corrected=False) # N_par × N_obs

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

def cov(A, B, corrected=False):
    A_mean = jnp.mean(A, axis=1)
    B_mean = jnp.mean(B, axis=1)
    dA = (A.T - A_mean).T
    dB = (B.T - B_mean).T
    n = A.shape[1]
    n = n-1 if corrected else n

    return jnp.matmul(dA, dB.T) / n

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
    prior_mean = 3.0 * jnp.ones(n_par)
    prior_cov = jnp.eye(n_par)
    prior = rnd.multivariate_normal

    # TODO!: Here we would set up an instance of an optimizer object

    # do optimization loop
    n_ens = 10 # number of ensemble members
    n_iter = 20 # number of EKI iterations
    ensemble = prior(key, prior_mean, prior_cov, (n_ens,)).T
    
    storage_g = []
    storage_u = [ensemble.copy()]
    for i in range(n_iter):
         evaluations = jnp.expand_dims(vobjective(ensemble).T, axis=0)
         ensemble = update_ensemble(ensemble, evaluations, obs_mean, obs_noise_cov, jnp.expand_dims(prior_mean, axis=1), prior_cov)
         storage_u.append(ensemble.copy())
         storage_g.append(evaluations.copy())

    # do plotting
    u_init = storage_u[1]
    u1_min = min(min(u[0, :]) for u in storage_u)
    u1_max = max(max(u[0, :]) for u in storage_u)
    u2_min = min(min(u[1, :]) for u in storage_u)
    u2_max = max(max(u[1, :]) for u in storage_u)
    xlim = (u1_min, u1_max)
    ylim = (u2_min, u2_max)
    for i, u in enumerate(storage_u):
        plt.cla()
        plt.title(str(i))
        plt.plot(u[0, :], u[1, :], 'kx')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()
        plt.pause(0.25)