#!/usr/bin/env python3

import jax
from jax import numpy as jnp

@jax.partial(jnp.vectorize, signature="(a,b),(c,b)->(s,a,c)", excluded=(2,))
def cov(A: jnp.ndarray, B: jnp.ndarray, corrected: bool = True) -> jnp.ndarray:
    """
        Computes the covariance matrix of two matrices assuming
        the 1st dimension is the sample dimension. If `corrected` is true, the
        corrected sample size is used.
        Example:
        :param A: Jax array.
        :param B: Jax array.
        :param corrected: boolean
        :returns: Jax array.
    """
    n = A.shape[1]
    n = n-1 if corrected else n
    A_mean = jnp.mean(A, axis=1)
    B_mean = jnp.mean(B, axis=1)

    # 1 / n * \sum ( (A - \mean(A)) \times (B - \mean(B)) )
    return jnp.matmul((A.T - A_mean).T, (B.T - B_mean).T) / n
