"""Tests for opterax._src.loss."""

from absl.testing import parameterized

import chex
import jax.numpy as jnp
import numpy as np

from opterax._src import loss


class L2LossTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = jnp.array([-2., -1., 0.5, 1.])
    self.ts = jnp.array([-1.5, 0., -1, 1.])
    # compute expected outputs in numpy.
    self.exp = 0.5 * (self.ts - self.ys) ** 2

  @chex.all_variants
  def test_scalar(self):
    np.testing.assert_allclose(
        self.variant(loss.l2_loss)(self.ys[0], self.ts[0]), self.exp[0])

  @chex.all_variants
  def test_batched(self):
    np.testing.assert_allclose(
        self.variant(loss.l2_loss)(self.ys, self.ts), self.exp)
