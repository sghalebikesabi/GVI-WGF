import jax
import jax.numpy as jnp


def rbf_kernel(x, y, axis=None):
    return jnp.exp(-jnp.mean((x - y) ** 2, axis=axis))


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_unflatten(treedef, keys)
