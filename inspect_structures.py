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

    layer = 5
    head = 3

    Q = all_queries[layer][0, head]
    K = all_keys[layer][0, head]

    assert isinstance(Q, jax.Array)
    assert isinstance(K, jax.Array)

    # low_rank_history = low_rank_solve(
    #     Q=Q,
    #     K=K,
    #     num_steps=50,
    #     tx=optax.adam(0.3),
    #     rank=14,
    #     return_history=True,
    # )
    monarch_history = monarch_solve(
        Q=Q,
        K=K,
        num_steps=100,
        tx=optax.adam(0.5),
        block_size=14,
        padding_type="pre",
        return_history=True,
    )
    monarch_history_v2 = monarch_solve(
        Q=Q,
        K=K,
        num_steps=100,
        tx=optax.sgd(1e5),
        block_size=14,
        padding_type="pre",
        return_history=True,
    )

    plt.plot(
        jax.vmap(monarch_history.loss_fn.__func__, in_axes=(0, None, None))(
            monarch_history, Q, K
        ),
        label=f"Monarch",
    )

    plt.plot(
        jax.vmap(monarch_history_v2.loss_fn.__func__, in_axes=(0, None, None))(
            monarch_history_v2, Q, K
        ),
        label=f"Monarch V2",
    )
    # Plot optimal value
    optimal = 1 / 2 * float(jnp.mean(jnp.square(sparsemax(Q @ K.T, axis=-1) - Q @ K.T)))
    plt.axhline(optimal, color="black", linestyle="--", label="Optimal")

    plt.legend()
    plt.show()

    # exit()

    # monarch_history = jax.vmap(lambda x: x.get_e2e())(monarch_history)
    # low_rank_history = jax.vmap(lambda x: x.get_e2e())(low_rank_history)

    # fig, ax = plt.subplots(1, 3)

    # ax[0].imshow(sparsemax(Q @ K.T, axis=-1))
    # ax[0].axis("off")
    # ax[0].set_title("Original Sparsemax")

    # ax[1].imshow(monarch_history[-1])
    # ax[1].axis("off")
    # ax[1].set_title("Monarch")

    # ax[2].imshow(low_rank_history[-1])
    # ax[2].axis("off")
    # ax[2].set_title("LowRank")

    # fig.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
