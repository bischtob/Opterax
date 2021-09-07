"""Tests for opterax._src.eki."""

from absl.testing import parameterized

import chex
import optax
import opterax
import jax.numpy as jnp
import numpy as np


class EkiTest(parameterized.TestCase):

  def setUp(self):
    """prepare test components."""
    super().setUp()
    def loss(y_pred, y):
      """standard l2-loss."""
      return jnp.mean(optax.l2_loss(y_pred, y))

    self.obs_err_cov = 1e-8 * jnp.eye(2)
    self.loss = loss
    self.params = jnp.array([[1.0, 1.5], [1.0, 1.5]])
    self.y_pred = jnp.array([[1.0, 0.5], [1.0, 0.5]])
    self.ys = jnp.array([[0.0, 0.0], [0.0, 0.0]])

  @chex.all_variants
  def test_eki(self):
    """test eki-based gradient approximation."""
    grad_fcn = opterax.grad_eki(self.loss, self.obs_err_cov)
    tmp = self.variant(grad_fcn)(self.params, self.y_pred, self.ys)

    np.testing.assert_allclose(tmp, [[-0.5, -0.25], [-0.5, -0.25]])
