#!/usr/bin/env python3
"""
Toy RL: concavity vs violations, with CLI flags.

This script exposes all key knobs from the earlier toy:
- concave bandit episodes
- sparse-path episodes
- horizon
- shaping
- learning rates
- number of seeds
- output directory
- optional fixed random seed offset

Example:
    python toy_rl_concavity_vs_violations_cli.py \
        --bandit-episodes 3000 \
        --sparse-episodes 20000 \
        --horizon 14 \
        --shaping 0.0 \
        --sparse-lr 0.08 \
        --n-seeds 1 \
        --outdir results

If you want a much sharper "phase transition"-like curve, try:
    python toy_rl_concavity_vs_violations_cli.py \
        --sparse-episodes 25000 \
        --horizon 16 \
        --shaping 0.0 \
        --n-seeds 1 \
        --sparse-lr 0.08
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def run_concave_bandit(
    episodes: int = 3000,
    lr: float = 0.035,
    baseline_decay: float = 0.99,
    reward_good: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """
    1-step Bernoulli bandit trained by REINFORCE.

    Action 1 -> reward_good
    Action 0 -> 0
    """
    rng = np.random.default_rng(seed)
    theta = 0.0
    baseline = 0.0
    perf = np.zeros(episodes, dtype=float)

    for t in range(episodes):
        p = float(sigmoid(theta))
        a = 1 if rng.random() < p else 0
        r = reward_good if a == 1 else 0.0
        grad = (1.0 - p) if a == 1 else -p

        theta += lr * (r - baseline) * grad
        baseline = baseline_decay * baseline + (1.0 - baseline_decay) * r

        perf[t] = reward_good * float(sigmoid(theta))

    return perf


def run_sparse_path(
    episodes: int = 6000,
    horizon: int = 8,
    lr: float = 0.06,
    shaping: float = 0.08,
    baseline_decay: float = 0.995,
    final_reward: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sparse deep-path problem.

    Correct action at every depth is 1.
    Big reward only if the full all-ones path is hit.
    Optional shaping rewards correct prefix length.
    """
    rng = np.random.default_rng(seed)
    theta = np.zeros(horizon, dtype=float)
    baseline = 0.0

    perf = np.zeros(episodes, dtype=float)
    probs = np.zeros((episodes, horizon), dtype=float)

    for t in range(episodes):
        p = sigmoid(theta)
        actions = (rng.random(horizon) < p).astype(np.int64)

        prefix_len = 0
        for a in actions:
            if a == 1:
                prefix_len += 1
            else:
                break

        reward = shaping * prefix_len + final_reward * float(prefix_len == horizon)
        grads = actions - p

        theta += lr * (reward - baseline) * grads
        baseline = baseline_decay * baseline + (1.0 - baseline_decay) * reward

        p_eval = sigmoid(theta)
        probs[t] = p_eval

        prefix_prod = 1.0
        expected_prefix = 0.0
        for i in range(horizon):
            prefix_prod *= p_eval[i]
            expected_prefix += prefix_prod

        perf[t] = shaping * expected_prefix + final_reward * float(np.prod(p_eval))

    return perf, probs


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, kernel, mode="same")


def aggregate_runs(
    n_seeds: int,
    bandit_episodes: int,
    bandit_lr: float,
    bandit_baseline_decay: float,
    reward_good: float,
    sparse_episodes: int,
    horizon: int,
    sparse_lr: float,
    shaping: float,
    sparse_baseline_decay: float,
    final_reward: float,
    seed_offset: int,
) -> dict:
    concave = np.stack(
        [
            run_concave_bandit(
                episodes=bandit_episodes,
                lr=bandit_lr,
                baseline_decay=bandit_baseline_decay,
                reward_good=reward_good,
                seed=seed_offset + s,
            )
            for s in range(n_seeds)
        ],
        axis=0,
    )

    sparse_runs = [
        run_sparse_path(
            episodes=sparse_episodes,
            horizon=horizon,
            lr=sparse_lr,
            shaping=shaping,
            baseline_decay=sparse_baseline_decay,
            final_reward=final_reward,
            seed=seed_offset + 10_000 + s,
        )
        for s in range(n_seeds)
    ]
    sparse = np.stack([r[0] for r in sparse_runs], axis=0)
    probs = np.stack([r[1] for r in sparse_runs], axis=0)

    return {
        "concave_mean": concave.mean(axis=0),
        "concave_std": concave.std(axis=0),
        "sparse_mean": sparse.mean(axis=0),
        "sparse_std": sparse.std(axis=0),
        "probs_mean": probs.mean(axis=0),
    }


def save_plots(results: dict, outdir: Path, mg_window: int, sd_window: int, legend_ncol: int) -> None:
    c_mean = results["concave_mean"]
    c_std = results["concave_std"]
    s_mean = results["sparse_mean"]
    s_std = results["sparse_std"]
    p_mean = results["probs_mean"]

    x1 = np.arange(1, len(c_mean) + 1)
    x2 = np.arange(1, len(s_mean) + 1)

    mg_c = np.diff(c_mean, prepend=c_mean[0])
    mg_s = np.diff(s_mean, prepend=s_mean[0])

    sd_c = np.diff(c_mean, n=2, prepend=[c_mean[0], c_mean[0]])
    sd_s = np.diff(s_mean, n=2, prepend=[s_mean[0], s_mean[0]])

    plt.figure(figsize=(9, 5.5))
    plt.plot(x1, c_mean, label="Concave regime")
    plt.fill_between(x1, c_mean - c_std, c_mean + c_std, alpha=0.2)
    plt.plot(x2, s_mean, label="Sparse deep-path regime")
    plt.fill_between(x2, s_mean - s_std, s_mean + s_std, alpha=0.2)
    plt.xlabel("Compute budget B (episodes)")
    plt.ylabel("Expected return")
    plt.title("Toy RL learning curves: concavity vs violations")
    plt.legend()
    plt.savefig(outdir / "toy_rl_learning_curves.png", dpi=130, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5.5))
    plt.plot(x1, moving_average(mg_c, mg_window), label="Concave: marginal gain")
    plt.plot(x2, moving_average(mg_s, mg_window), label="Sparse path: marginal gain")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Compute budget B (episodes)")
    plt.ylabel("Smoothed discrete marginal gain")
    plt.title("Marginal gains")
    plt.legend()
    plt.savefig(outdir / "toy_rl_marginal_gains.png", dpi=130, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5.5))
    plt.plot(x1, moving_average(sd_c, sd_window), label="Concave: second difference")
    plt.plot(x2, moving_average(sd_s, sd_window), label="Sparse path: second difference")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Compute budget B (episodes)")
    plt.ylabel("Smoothed second difference")
    plt.title("Concavity vs local convexity")
    plt.legend()
    plt.savefig(outdir / "toy_rl_second_differences.png", dpi=130, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for i in range(p_mean.shape[1]):
        plt.plot(x2, p_mean[:, i], label=f"Depth {i+1}")
    plt.xlabel("Compute budget B (episodes)")
    plt.ylabel("P(correct action)")
    plt.title("Sparse deep-path regime: depth-wise policy improvement")
    plt.legend(ncol=legend_ncol, fontsize=8)
    plt.savefig(outdir / "toy_rl_sparse_depth_probs.png", dpi=120, bbox_inches="tight")
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Toy RL simulator showing concavity vs sparse-discovery violations."
    )

    # Compute / aggregation knobs
    parser.add_argument("--n-seeds", type=int, default=40, help="Number of random seeds to average over.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset added to all seeds.")
    parser.add_argument("--outdir", type=str, default=".", help="Directory where plots are written.")

    # Bandit knobs
    parser.add_argument("--bandit-episodes", type=int, default=3000, help="Episodes for the concave bandit.")
    parser.add_argument("--bandit-lr", type=float, default=0.035, help="Learning rate for the bandit.")
    parser.add_argument(
        "--bandit-baseline-decay",
        type=float,
        default=0.99,
        help="Moving-baseline decay for the bandit.",
    )
    parser.add_argument(
        "--reward-good",
        type=float,
        default=1.0,
        help="Reward for the good action in the bandit.",
    )

    # Sparse-path knobs
    parser.add_argument("--sparse-episodes", type=int, default=6000, help="Episodes for the sparse path model.")
    parser.add_argument("--horizon", type=int, default=8, help="Depth of the sparse path.")
    parser.add_argument("--sparse-lr", type=float, default=0.06, help="Learning rate for the sparse path model.")
    parser.add_argument(
        "--shaping",
        type=float,
        default=0.08,
        help="Prefix shaping reward. Set to 0.0 for truly sparse rewards.",
    )
    parser.add_argument(
        "--sparse-baseline-decay",
        type=float,
        default=0.995,
        help="Moving-baseline decay for the sparse path model.",
    )
    parser.add_argument(
        "--final-reward",
        type=float,
        default=1.0,
        help="Reward for reaching the full correct path.",
    )

    # Plot smoothing knobs
    parser.add_argument("--mg-window", type=int, default=80, help="Moving-average window for marginal gains.")
    parser.add_argument("--sd-window", type=int, default=120, help="Moving-average window for second differences.")
    parser.add_argument(
        "--legend-ncol",
        type=int,
        default=2,
        help="Number of legend columns for the depth-wise probability plot.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n_seeds < 1:
        raise ValueError("--n-seeds must be >= 1")
    if args.bandit_episodes < 2 or args.sparse_episodes < 2:
        raise ValueError("Episode counts must be >= 2")
    if args.horizon < 1:
        raise ValueError("--horizon must be >= 1")
    if not (0.0 <= args.bandit_baseline_decay < 1.0):
        raise ValueError("--bandit-baseline-decay must be in [0, 1)")
    if not (0.0 <= args.sparse_baseline_decay < 1.0):
        raise ValueError("--sparse-baseline-decay must be in [0, 1)")
    if args.mg_window < 1 or args.sd_window < 1:
        raise ValueError("Smoothing windows must be >= 1")
    if args.legend_ncol < 1:
        raise ValueError("--legend-ncol must be >= 1")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    results = aggregate_runs(
        n_seeds=args.n_seeds,
        bandit_episodes=args.bandit_episodes,
        bandit_lr=args.bandit_lr,
        bandit_baseline_decay=args.bandit_baseline_decay,
        reward_good=args.reward_good,
        sparse_episodes=args.sparse_episodes,
        horizon=args.horizon,
        sparse_lr=args.sparse_lr,
        shaping=args.shaping,
        sparse_baseline_decay=args.sparse_baseline_decay,
        final_reward=args.final_reward,
        seed_offset=args.seed_offset,
    )

    save_plots(
        results=results,
        outdir=outdir,
        mg_window=args.mg_window,
        sd_window=args.sd_window,
        legend_ncol=args.legend_ncol,
    )

    print("Created files:")
    for name in [
        "toy_rl_learning_curves.png",
        "toy_rl_marginal_gains.png",
        "toy_rl_second_differences.png",
        "toy_rl_sparse_depth_probs.png",
    ]:
        print(f"- {outdir / name}")


if __name__ == "__main__":
    main()
