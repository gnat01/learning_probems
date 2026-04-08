#!/usr/bin/env python3
"""
Invisible Palette Toolkit

Three Bayesian estimators for the hidden number of occupied colours C in an urn,
using with-replacement samples from an empirical dataset of repeated labels 0..m-1.

Modes:
1) distinct_only          : uses only K_t = number of distinct colours observed
2) full_uniform          : uses full occupancy counts under a uniform multinomial over C colours
3) full_dirichlet        : uses full occupancy counts under a symmetric Dirichlet-multinomial prior

The script runs all three in one go, tracks posterior sharpening across rounds,
and writes plots + a CSV summary.

Example:
    python src/invisible_palette_toolkit.py \
      --counts 4,3,7 \
      --batch-size 8 \
      --rounds 20 \
      --c-max 15 \
      --alpha 0.3 \
      --outdir results/example_run

Alternative dataset generation:
    python src/invisible_palette_toolkit.py \
      --m 8 \
      --count-mode skew \
      --min-count 1 \
      --max-count 15 \
      --batch-size 10 \
      --rounds 30 \
      --c-max 25 \
      --alpha 0.5 \
      --outdir results/generated_skew_run
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

plt = None


def ensure_matplotlib() -> None:
    global plt
    if plt is not None:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as imported_plt
    plt = imported_plt


# -----------------------------
# Utilities
# -----------------------------
def logsumexp(logv: np.ndarray) -> float:
    m = np.max(logv)
    if np.isneginf(m):
        return float("-inf")
    return float(m + np.log(np.sum(np.exp(logv - m))))


def normalize_log_probs(logp: np.ndarray) -> np.ndarray:
    z = logsumexp(logp)
    return np.exp(logp - z)


def gammaln(x: float) -> float:
    return math.lgamma(x)


def parse_counts(counts_str: str) -> List[int]:
    vals = [int(x.strip()) for x in counts_str.split(",") if x.strip()]
    if any(v < 0 for v in vals):
        raise ValueError("All counts must be >= 0")
    return vals


def build_dataset_from_counts(counts: List[int]) -> List[int]:
    data = []
    for color, cnt in enumerate(counts):
        data.extend([color] * cnt)
    return data


def generate_counts(m: int, mode: str, min_count: int, max_count: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    if mode == "uniform":
        counts = [int(rng.integers(min_count, max_count + 1)) for _ in range(m)]
    elif mode == "skew":
        raw = rng.zipf(a=2.0, size=m)
        raw = np.clip(raw, 1, None).astype(float)
        scaled = min_count + (raw - raw.min()) * (max_count - min_count) / max(1.0, raw.max() - raw.min())
        counts = [int(round(x)) for x in scaled]
    elif mode == "one_heavy":
        counts = [min_count] * m
        counts[0] = max_count
    else:
        raise ValueError(f"Unknown count mode: {mode}")
    return counts


def posterior_mean(c_vals: np.ndarray, post: np.ndarray) -> float:
    return float(np.sum(c_vals * post))


# -----------------------------
# Stirling numbers of second kind, log-space
# S(n,k) = k*S(n-1,k) + S(n-1,k-1)
# -----------------------------
def log_stirling2_table(n_max: int, k_max: int) -> np.ndarray:
    table = np.full((n_max + 1, k_max + 1), -np.inf, dtype=float)
    table[0, 0] = 0.0
    log_k = np.zeros(k_max + 1, dtype=float)
    if k_max >= 1:
        log_k[1:] = np.log(np.arange(1, k_max + 1))
    for n in range(1, n_max + 1):
        max_k = min(n, k_max)
        a = table[n - 1, 1:max_k + 1] + log_k[1:max_k + 1]
        b = table[n - 1, :max_k]
        table[n, 1:max_k + 1] = np.logaddexp(a, b)
    return table


def log_falling_factorial(c: int, k: int) -> float:
    if k > c:
        return float("-inf")
    return gammaln(c + 1) - gammaln(c - k + 1)


def build_log_factorials(c_max: int) -> np.ndarray:
    vals = np.zeros(c_max + 1, dtype=float)
    for i in range(1, c_max + 1):
        vals[i] = vals[i - 1] + math.log(i)
    return vals


def build_log_falling_factorial_table(c_candidates: np.ndarray) -> np.ndarray:
    c_max = int(np.max(c_candidates))
    log_factorials = build_log_factorials(c_max)
    table = np.full((c_max + 1, len(c_candidates)), -np.inf, dtype=float)
    for k in range(c_max + 1):
        valid = c_candidates >= k
        table[k, valid] = log_factorials[c_candidates[valid]] - log_factorials[c_candidates[valid] - k]
    return table


def vectorized_lgamma(values: np.ndarray) -> np.ndarray:
    return np.fromiter((math.lgamma(float(v)) for v in values), dtype=float, count=len(values))


# -----------------------------
# Likelihoods
# -----------------------------
def loglik_distinct_only(c: int, t: int, k: int, logS2: np.ndarray) -> float:
    # P(K_t = k | C=c, uniform over c colours) = (c)_k * S(t,k) / c^t
    if k > c or k > t or k < 0:
        return float("-inf")
    return log_falling_factorial(c, k) + logS2[t, k] - t * math.log(c)


def loglik_full_uniform(c: int, counts_seen: List[int]) -> float:
    # Observed labeled occupancy under uniform multinomial, up to multinomial coefficient constant in c:
    # (c)_k / c^t
    t = sum(counts_seen)
    k = len(counts_seen)
    if k > c:
        return float("-inf")
    return log_falling_factorial(c, k) - t * math.log(c)


def loglik_full_dirichlet(c: int, counts_seen: List[int], alpha: float) -> float:
    # Labels are initially unidentified species, so include (c)_k for choosing which c colours were seen.
    t = sum(counts_seen)
    k = len(counts_seen)
    if k > c:
        return float("-inf")
    ll = log_falling_factorial(c, k)
    ll += gammaln(c * alpha) - gammaln(t + c * alpha)
    for x in counts_seen:
        ll += gammaln(x + alpha) - gammaln(alpha)
    return ll


# -----------------------------
# Priors on C
# -----------------------------
def build_log_prior(c_candidates: np.ndarray, prior_type: str, lam: float) -> np.ndarray:
    if prior_type == "uniform":
        return np.full_like(c_candidates, -math.log(len(c_candidates)), dtype=float)
    if prior_type == "geometric":
        # proportional to (1-lam)^(c-1) lam ; here lam is success probability in (0,1]
        if not (0 < lam <= 1):
            raise ValueError("--prior-lam must be in (0,1] for geometric prior")
        logp = np.array([math.log(lam) + (c - 1) * math.log(1 - lam) if lam < 1 else 0.0 for c in c_candidates], dtype=float)
        logp -= logsumexp(logp)
        return logp
    raise ValueError(f"Unknown prior_type: {prior_type}")


# -----------------------------
# Sequential experiment
# -----------------------------
@dataclass
class SequentialResult:
    rounds: List[int]
    cumulative_samples: List[int]
    distinct_seen: List[int]
    posteriors: Dict[str, np.ndarray]   # mode -> array shape (rounds, len(c_candidates))
    posterior_means: Dict[str, List[float]]
    observed_counts_per_round: List[List[int]]
    sampled_labels: List[int]


def run_experiment(
    data: List[int],
    c_candidates: np.ndarray,
    batch_size: int,
    rounds: int,
    alpha: float,
    log_prior: np.ndarray,
    seed: int,
) -> SequentialResult:
    rng = np.random.default_rng(seed)
    data_arr = np.asarray(data)

    sampled = []
    counts = Counter()

    max_t = batch_size * rounds
    c_max = int(np.max(c_candidates))
    logS2 = log_stirling2_table(max_t, min(max_t, c_max))
    log_c = np.log(c_candidates)
    log_falling = build_log_falling_factorial_table(c_candidates)
    round_ts = np.arange(batch_size, max_t + 1, batch_size, dtype=float)
    neg_t_log_c = -np.outer(round_ts, log_c)
    c_alpha = c_candidates.astype(float) * alpha
    lgamma_c_alpha = vectorized_lgamma(c_alpha)
    lgamma_t_plus_c_alpha = np.vstack([
        vectorized_lgamma(c_alpha + t) for t in round_ts
    ])
    log_gamma_alpha = gammaln(alpha)

    posteriors = {
        "distinct_only": [],
        "full_uniform": [],
        "full_dirichlet": [],
    }
    means = {k: [] for k in posteriors}
    round_ids = []
    cumulative_samples = []
    distinct_seen = []
    observed_counts_per_round = []

    for r in range(1, rounds + 1):
        draws = rng.choice(data_arr, size=batch_size, replace=True)
        sampled.extend(draws.tolist())
        counts.update(draws.tolist())

        t = len(sampled)
        occ = sorted(counts.values(), reverse=True)
        k = len(occ)
        round_idx = r - 1
        log_falling_k = log_falling[k]

        # distinct_only
        ll_distinct = log_falling_k + logS2[t, k] + neg_t_log_c[round_idx]
        post_distinct = normalize_log_probs(log_prior + ll_distinct)

        # full_uniform
        ll_fu = log_falling_k + neg_t_log_c[round_idx]
        post_fu = normalize_log_probs(log_prior + ll_fu)

        # full_dirichlet
        occ_term = sum(gammaln(x + alpha) - log_gamma_alpha for x in occ)
        ll_fd = log_falling_k + lgamma_c_alpha - lgamma_t_plus_c_alpha[round_idx] + occ_term
        post_fd = normalize_log_probs(log_prior + ll_fd)

        posteriors["distinct_only"].append(post_distinct)
        posteriors["full_uniform"].append(post_fu)
        posteriors["full_dirichlet"].append(post_fd)

        means["distinct_only"].append(posterior_mean(c_candidates, post_distinct))
        means["full_uniform"].append(posterior_mean(c_candidates, post_fu))
        means["full_dirichlet"].append(posterior_mean(c_candidates, post_fd))

        round_ids.append(r)
        cumulative_samples.append(t)
        distinct_seen.append(k)
        observed_counts_per_round.append(occ.copy())

    posteriors = {k: np.vstack(v) for k, v in posteriors.items()}

    return SequentialResult(
        rounds=round_ids,
        cumulative_samples=cumulative_samples,
        distinct_seen=distinct_seen,
        posteriors=posteriors,
        posterior_means=means,
        observed_counts_per_round=observed_counts_per_round,
        sampled_labels=sampled,
    )


# -----------------------------
# Output
# -----------------------------
def save_summary_csv(
    outdir: Path,
    result: SequentialResult,
    true_m: int,
) -> None:
    csv_path = outdir / "posterior_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "cumulative_samples",
            "distinct_seen",
            "posterior_mean_distinct_only",
            "posterior_mean_full_uniform",
            "posterior_mean_full_dirichlet",
            "true_m",
        ])
        for i, r in enumerate(result.rounds):
            writer.writerow([
                r,
                result.cumulative_samples[i],
                result.distinct_seen[i],
                result.posterior_means["distinct_only"][i],
                result.posterior_means["full_uniform"][i],
                result.posterior_means["full_dirichlet"][i],
                true_m,
            ])


def _save_heatmap(
    matrix: np.ndarray,
    c_candidates: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    ensure_matplotlib()
    plt.figure(figsize=(10, 5.5))
    plt.imshow(
        matrix.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[1, matrix.shape[0], c_candidates[0] - 0.5, c_candidates[-1] + 0.5],
    )
    plt.colorbar(label="Posterior probability")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_plots(
    outdir: Path,
    result: SequentialResult,
    c_candidates: np.ndarray,
    true_m: int,
) -> None:
    ensure_matplotlib()
    for mode, matrix in result.posteriors.items():
        _save_heatmap(
            matrix=matrix,
            c_candidates=c_candidates,
            xlabel="Round",
            ylabel="Candidate C",
            title=f"Posterior sharpening by round: {mode}",
            path=outdir / f"posterior_heatmap_{mode}.png",
        )

    # Posterior means across rounds
    plt.figure(figsize=(10, 5.5))
    for mode, vals in result.posterior_means.items():
        plt.plot(result.rounds, vals, label=mode)
    plt.axhline(true_m, linestyle="--", label="true m")
    plt.xlabel("Round")
    plt.ylabel("Posterior mean of C")
    plt.title("Posterior mean across rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "posterior_means_across_rounds.png", dpi=140)
    plt.close()

    # Distinct colours seen over rounds
    plt.figure(figsize=(10, 5.5))
    plt.plot(result.rounds, result.distinct_seen, label="Observed distinct colours")
    plt.axhline(true_m, linestyle="--", label="true m")
    plt.xlabel("Round")
    plt.ylabel("Distinct colours observed so far")
    plt.title("Observed distinct colours over rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "distinct_seen_across_rounds.png", dpi=140)
    plt.close()

    # Final posterior comparison
    plt.figure(figsize=(10, 5.5))
    width = 0.26
    x = np.arange(len(c_candidates))
    plt.bar(x - width, result.posteriors["distinct_only"][-1], width=width, label="distinct_only")
    plt.bar(x, result.posteriors["full_uniform"][-1], width=width, label="full_uniform")
    plt.bar(x + width, result.posteriors["full_dirichlet"][-1], width=width, label="full_dirichlet")
    plt.xticks(x, c_candidates)
    plt.xlabel("Candidate C")
    plt.ylabel("Posterior probability")
    plt.title("Final posterior comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "final_posterior_comparison.png", dpi=140)
    plt.close()


def save_run_info(
    outdir: Path,
    counts: List[int],
    args: argparse.Namespace,
    true_m: int,
    total_n: int,
) -> None:
    p = outdir / "run_info.txt"
    with p.open("w") as f:
        f.write(f"true_m={true_m}\n")
        f.write(f"counts={counts}\n")
        f.write(f"total_N={total_n}\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}={v}\n")


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bayesian toolkit for the Invisible Palette Problem."
    )

    dataset = parser.add_argument_group("dataset construction")
    dataset.add_argument(
        "--counts",
        type=str,
        default="",
        help="Comma-separated counts for colours 0..m-1, e.g. 4,3,7 . Overrides --m and count generation.",
    )
    dataset.add_argument("--m", type=int, default=5, help="Number of occupied colours when generating counts.")
    dataset.add_argument(
        "--count-mode",
        type=str,
        choices=["uniform", "skew", "one_heavy"],
        default="uniform",
        help="How to generate counts when --counts is not supplied.",
    )
    dataset.add_argument("--min-count", type=int, default=1, help="Minimum count per colour in generated data.")
    dataset.add_argument("--max-count", type=int, default=10, help="Maximum count per colour in generated data.")

    sampling = parser.add_argument_group("sampling process")
    sampling.add_argument("--batch-size", type=int, default=8, help="Samples drawn with replacement per round.")
    sampling.add_argument("--rounds", type=int, default=20, help="Number of sequential posterior updates.")
    sampling.add_argument("--seed", type=int, default=0, help="Random seed for count generation and sampling.")

    bayes = parser.add_argument_group("Bayesian model")
    bayes.add_argument("--c-max", type=int, default=20, help="Maximum candidate value of C to consider.")
    bayes.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Symmetric Dirichlet concentration parameter for the full_dirichlet model.",
    )
    bayes.add_argument(
        "--prior-type",
        type=str,
        choices=["uniform", "geometric"],
        default="uniform",
        help="Prior over candidate C.",
    )
    bayes.add_argument(
        "--prior-lam",
        type=float,
        default=0.2,
        help="Geometric prior parameter when --prior-type geometric.",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--outdir", type=str, default=".", help="Directory for plots and summary files.")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.min_count < 1 or args.max_count < args.min_count:
        raise ValueError("Need 1 <= min-count <= max-count")
    if args.c_max < 1:
        raise ValueError("--c-max must be >= 1")
    if args.alpha <= 0:
        raise ValueError("--alpha must be > 0")
    if args.counts:
        counts = parse_counts(args.counts)
        if len(counts) == 0:
            raise ValueError("--counts must not be empty if supplied")
        if args.c_max < len(counts):
            raise ValueError("--c-max must be at least the number of occupied colours in --counts")
    else:
        if args.m < 1:
            raise ValueError("--m must be >= 1")
        if args.c_max < args.m:
            raise ValueError("--c-max must be >= --m")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.counts:
        counts = parse_counts(args.counts)
    else:
        counts = generate_counts(
            m=args.m,
            mode=args.count_mode,
            min_count=args.min_count,
            max_count=args.max_count,
            seed=args.seed,
        )

    data = build_dataset_from_counts(counts)
    true_m = sum(1 for x in counts if x > 0)
    total_n = len(data)

    c_candidates = np.arange(1, args.c_max + 1)
    log_prior = build_log_prior(c_candidates, args.prior_type, args.prior_lam)

    result = run_experiment(
        data=data,
        c_candidates=c_candidates,
        batch_size=args.batch_size,
        rounds=args.rounds,
        alpha=args.alpha,
        log_prior=log_prior,
        seed=args.seed + 12345,
    )

    save_summary_csv(outdir, result, true_m)
    save_plots(outdir, result, c_candidates, true_m)
    save_run_info(outdir, counts, args, true_m, total_n)

    print("Created files:")
    for name in [
        "posterior_summary.csv",
        "posterior_heatmap_distinct_only.png",
        "posterior_heatmap_full_uniform.png",
        "posterior_heatmap_full_dirichlet.png",
        "posterior_means_across_rounds.png",
        "distinct_seen_across_rounds.png",
        "final_posterior_comparison.png",
        "run_info.txt",
    ]:
        print(f"- {outdir / name}")


if __name__ == "__main__":
    main()
