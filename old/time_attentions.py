from math import sqrt

import jax

Array = jax.Array
from tqdm.auto import tqdm

from efficient_attention import monarch_attention


def standard_attention(query: Array, key: Array, value: Array):
    attn_matrix = jax.nn.softmax(query @ key.T, axis=-1)
    return attn_matrix @ value


def time_attentions(query, n_iters):
    from functools import partial
    from math import floor
    from time import time

    f = jax.jit(standard_attention)
    f(query, query, query).block_until_ready()
    start = time()
    for _ in tqdm(range(n_iters)):
        f(query, query, query).block_until_ready()
    end = time()
    standard_time = (end - start) / n_iters

    block_size = floor(sqrt(query.shape[0]))
    f = jax.jit(partial(monarch_attention, block_size=block_size))

    f(query, query, query).block_until_ready()
    start = time()
    for _ in tqdm(range(n_iters)):
        f(query, query, query).block_until_ready()
    end = time()

    efficient_time = (end - start) / n_iters

    return standard_time, efficient_time
