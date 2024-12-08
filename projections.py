import jax
import jax.numpy as jnp
from jax import custom_jvp


def vector_to_batch_version(f):
    def f_batched(z, axis, *args, **kwargs):
        assert z.ndim >= 1, "input must be at least 1D"
        if z.ndim == 1:
            return f(z, *args, **kwargs)
        else:
            z_transposed = jnp.swapaxes(z, axis, -1)
            z_transposed_batched = jnp.reshape(
                z_transposed, (-1, z_transposed.shape[-1])
            )
            p_tranposed_batched = jax.vmap(f, in_axes=(0,) + len(args) * (None,))(
                z_transposed_batched, *args, **kwargs
            )
            p_transposed = jnp.reshape(p_tranposed_batched, z_transposed.shape)
            p = jnp.swapaxes(p_transposed, axis, -1)
            return p

    return f_batched


def vector_hardmax(z):
    assert z.ndim == 1, "input must be 1D"
    p = jnp.zeros_like(z).at[jnp.argmax(z)].set(1)
    return p


@custom_jvp
def vector_softmax(z):
    assert z.ndim == 1, "input must be 1D"
    m = jnp.max(z)
    z = z - m
    p = jnp.exp(z)
    return p / jnp.sum(p)


@vector_softmax.defjvp
def vector_softmax_jvp(primals, tangents):
    (z,) = primals
    (dz,) = tangents
    p = vector_softmax(z)
    dp = p * dz - (p * dz).sum() * p
    return p, dp


@custom_jvp
def vector_sparsemax(z):
    assert z.ndim == 1, "input must be 1D"
    z_sorted = jnp.sort(z, descending=True)
    z_cumsum = jnp.cumsum(z_sorted)
    arr = jnp.concatenate(
        [1 + z_sorted * jnp.arange(1, len(z) + 1) > z_cumsum, jnp.full((1,), False)]
    )
    k = jnp.argmin(arr) - 1
    tau = (z_cumsum[k] - 1) / (k + 1)
    return jnp.maximum(z - tau, 0)


@vector_sparsemax.defjvp
def vector_sparsemax_jvp(primals, tangents):
    (z,) = primals
    (dz,) = tangents
    p = vector_sparsemax(z)
    s = p > 0
    s_sum = s.sum()
    dp = s * dz - (s * dz).sum() / s_sum * s
    return p, dp


def vector_simplex_projection(z, a):
    K = len(z)
    z_sorted = jnp.sort(z, descending=True)
    z_cumsum = jnp.cumsum(z_sorted)
    arr = jnp.concatenate(
        [
            1 + z_sorted * jnp.arange(1, len(z) + 1) > z_cumsum - K * a,
            jnp.full((1,), False),
        ]
    )
    k = jnp.argmin(arr) - 1
    tau = (z_cumsum[k] - (K - k - 1) * a - 1) / (k + 1)
    return jnp.maximum(z - tau, -a)


def vector_entmax15(z):

    assert z.ndim == 1, "input must be 1D"
    z /= 2
    rho = jnp.arange(1, len(z) + 1)
    z_sorted = jnp.sort(z, descending=True)
    z_cumsum = jnp.cumsum(z_sorted)
    z2_cumsum = jnp.cumsum(z_sorted**2)
    mean = z_cumsum / rho
    var = z2_cumsum - mean**2 * rho
    delta = (1 - var) / rho
    taus = mean - jnp.sqrt(delta)
    arr = taus <= z_sorted
    k = jnp.argmin(arr) - 1
    tau = taus[k]
    return jnp.maximum(z - tau, 0) ** 2


hardmax = vector_to_batch_version(vector_hardmax)
softmax = vector_to_batch_version(vector_softmax)
sparsemax = vector_to_batch_version(vector_sparsemax)
simplex_projection = vector_to_batch_version(vector_simplex_projection)
entmax15 = vector_to_batch_version(vector_entmax15)


if __name__ == "__main__":
    import cvxpy as cp

    def cvx_entmax15(z):
        K = len(z)
        p = cp.Variable(K)
        objective = cp.Maximize(cp.sum(cp.multiply(p, z)) + 4 / 3 * cp.sum(p - p**1.5))
        constraints = [p >= 0, cp.sum(p) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return p.value

    import jax.random as jr

    z = jr.normal(jr.key(0), (10,))
    print(vector_entmax15(z))
    print(cvx_entmax15(z))
