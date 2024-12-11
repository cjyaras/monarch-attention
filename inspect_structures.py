import pickle as pl

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax.traverse_util import flatten_dict

from projections import sparsemax
from solvers import low_rank_solve, monarch_solve


def main():
    with open("intermediates.pkl", "rb") as f:
        intermediates = flatten_dict(pl.load(f))

    all_queries = {int(k[4]): v[0] for k, v in intermediates.items() if k[-1] == "q"}  # type: ignore
    all_keys = {int(k[4]): v[0] for k, v in intermediates.items() if k[-1] == "k"}  # type: ignore

    layer = 11
    head = 5

    query = all_queries[layer][0, head]
    key = all_keys[layer][0, head]

    assert isinstance(query, jax.Array)
    assert isinstance(key, jax.Array)

    low_rank_history = low_rank_solve(
        query=query,
        key=key,
        num_steps=100,
        tx=optax.adam(0.1),
        rank=14,
        return_history=True,
    )
    monarch_history = monarch_solve(
        query=query,
        key=key,
        num_steps=100,
        tx=optax.adam(0.1),
        block_size=14,
        padding_type="pre",
        return_history=True,
    )
    # monarch_history_v2 = monarch_solve(
    #     query=Q,
    #     key=K,
    #     num_steps=100,
    #     tx=optax.sgd(1e5),
    #     block_size=14,
    #     padding_type="pre",
    #     return_history=True,
    # )

    plt.plot(
        jax.vmap(low_rank_history.loss_fn.__func__, in_axes=(0, None, None))(
            low_rank_history, query, key
        ),
        label=f"Low Rank",
    )

    plt.plot(
        jax.vmap(monarch_history.loss_fn.__func__, in_axes=(0, None, None))(
            monarch_history, query, key
        ),
        label=f"Monarch",
    )

    # Plot optimal value
    optimal = (
        1
        / 2
        * float(jnp.mean(jnp.square(sparsemax(query @ key.T, axis=-1) - query @ key.T)))
    )
    plt.axhline(optimal, color="black", linestyle="--", label="Optimal")

    plt.legend()
    plt.show()

    monarch_history = jax.vmap(lambda x: x.get_e2e())(monarch_history)
    low_rank_history = jax.vmap(lambda x: x.get_e2e())(low_rank_history)

    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(sparsemax(query @ key.T, axis=-1))
    ax[0].axis("off")
    ax[0].set_title("Original Sparsemax")

    ax[1].imshow(monarch_history[-1])
    ax[1].axis("off")
    ax[1].set_title("Monarch")

    ax[2].imshow(low_rank_history[-1])
    ax[2].axis("off")
    ax[2].set_title("LowRank")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
