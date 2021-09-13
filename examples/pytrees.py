import jax
from jax import numpy as jnp, random as jr
from flax import linen as nn, optim
import numpy as np

from flax import linen as nn
from jax.tree_util import tree_flatten, tree_unflatten


def make_spirals(n_samples, noise_std=0., rotations=1.):
    ts = jnp.linspace(0, 1, n_samples)
    rs = ts ** 0.5
    thetas = rs * rotations * 2 * np.pi
    signs = np.random.randint(0, 2, (n_samples,)) * 2 - 1
    labels = (signs > 0).astype(int)

    xs = rs * signs * jnp.cos(thetas) + np.random.randn(n_samples) * noise_std
    ys = rs * signs * jnp.sin(thetas) + np.random.randn(n_samples) * noise_std
    points = jnp.stack([xs, ys], axis=1)
    return points, labels


points, labels = make_spirals(88, noise_std=0.05)


class MLPClassifier(nn.Module):
    hidden_layers: int = 1
    hidden_dim: int = 2
    n_classes: int = 2

    @nn.compact
    def __call__(self, x):
        for layer in range(self.hidden_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.n_classes)(x)
        x = nn.log_softmax(x)
        return x


classifier_fns = MLPClassifier()


def cross_entropy(logprobs, labels):
    one_hot_labels = jax.nn.one_hot(labels, logprobs.shape[1])
    return -jnp.mean(jnp.sum(one_hot_labels * logprobs, axis=-1))


def loss_fn(params, batch):
    logits = classifier_fns.apply({'params': params}, batch[0])
    loss = jnp.mean(cross_entropy(logits, batch[1]))
    return loss


loss_and_grad_fn = jax.value_and_grad(loss_fn)


def init_fn(input_shape, seed):
    rng = jr.PRNGKey(jnp.array(seed, int))
    dummy_input = jnp.ones((1, *input_shape))
    params = classifier_fns.init(rng, dummy_input)['params']
    optimizer_def = optim.Adam(learning_rate=1e-3)
    optimizer = optimizer_def.create(params)
    return optimizer


@jax.jit  # jit makes it go brrr
def train_step_fn(optimizer, batch):
    loss = loss_fn(optimizer.target, batch)
    loss, grad = loss_and_grad_fn(optimizer.target, batch)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, loss


@jax.jit  # jit makes it go brrr
def predict_fn(optimizer, x):
    x = jnp.array(x)
    return classifier_fns.apply({'params': optimizer.target}, x)


if __name__ == "__main__":
    parallel_init_fn = jax.vmap(init_fn, in_axes=(None, 0))
    parallel_train_step_fn = jax.vmap(train_step_fn, in_axes=(0, None))

    N = 3
    seeds = jnp.linspace(0, N - 1, N)

    model_states = parallel_init_fn((2,), seeds)
    # for i in range(93):
    #     model_states, losses = parallel_train_step_fn(
    #         model_states, (points, labels))
        
    value_structured = model_states
    print("untransformed_structured={}".format(value_structured))

    # The leaves in value_flat correspond to the `*` markers in value_tree
    value_flat, value_tree = tree_flatten(value_structured)
    print("value_flat={}\nvalue_tree={}".format(value_flat, value_tree))

    # Transform the flat value list using an element-wise numeric transformer
    transformed_flat = list(map(lambda v: jnp.mean(v, axis=0), value_flat))
    print("transformed_flat={}".format(transformed_flat))

    # # Reconstruct the structured output, using the original
    # transformed_structured = tree_unflatten(value_tree, transformed_flat)
    # print("transformed_structured={}".format(transformed_structured))
