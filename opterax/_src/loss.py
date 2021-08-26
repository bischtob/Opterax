"""Standard losses used in optimization.
"""

from typing import Optional

import chex

def l2_loss(
    predictions: chex.Array,
    targets: Optional[chex.Array] = None,
) -> chex.Array:
  """Calculates the L2 loss for a set of predictions.

  Note: the 0.5 term is standard in "Pattern Recognition and Machine Learning"
  by Bishop, but not "The Elements of Statistical Learning" by Tibshirani.

  References:
    [Chris Bishop, 2006](https://bit.ly/3eeP0ga)

  Args:
    predictions: a vector of arbitrary shape.
    targets: a vector of shape compatible with predictions; if not provides
      then it is assumed to be zero.

  Returns:
    the squared error loss.
  """
  chex.assert_type([predictions], float)
  errors = (predictions - targets) if (targets is not None) else predictions

  return 0.5 * (errors)**2
