import jax


def vmap_lift(f, n, in_axes, out_axes):
    for _ in range(n):
        f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)
    return f
