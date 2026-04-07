#!/usr/bin/env python3
"""
Invisible Palette Toolkit with posterior evolution plots and GIF animation.

Fixes in this version:
- GIFs are only reported if actually created
- robust GIF backend selection: imageio first, Pillow fallback
- explicit diagnostics if GIF creation is skipped or fails
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GIF_BACKEND = None
try:
    import imageio.v2 as imageio
    GIF_BACKEND = "imageio"
except Exception:
    imageio = None

if GIF_BACKEND is None:
    try:
        from PIL import Image
        GIF_BACKEND = "pillow"
    except Exception:
        Image = None


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


def log_stirling2_table(n_max: int, k_max: int) -> np.ndarray:
    table = np.full((n_max + 1, k_max + 1), -np.inf, dtype=float)
    table[0, 0] = 0.0
    for n in range(1, n_max + 1):
        for k in range(1, min(n, k_max) + 1):
            a = table[n - 1, k] + math.log(k) if np.isfinite(table[n - 1, k]) else -np.inf
            b = table[n - 1, k - 1]
            if np.isneginf(a):
                table[n, k] = b
            elif np.isneginf(b):
                table[n, k] = a
            else:
                m = max(a, b)
                table[n, k] = m + math.log(math.exp(a - m) + math.exp(b - m))
    return table


def log_falling_factorial(c: int, k: int) -> float:
    if k > c:
        return float("-inf")
    return gammaln(c + 1) - gammaln(c - k + 1)


def loglik_distinct_only(c: int, t: int, k: int, logS2: np.ndarray) -> float:
    if k > c or k > t or k < 0:
        return float("-inf")
    return log_falling_factorial(c, k) + logS2[t, k] - t * math.log(c)


def loglik_full_uniform(c: int, counts_seen: List[int]) -> float:
    t = sum(counts_seen)
    k = len(counts_seen)
    if k > c:
        return float("-inf")
    return log_falling_factorial(c, k) - t * math.log(c)


def loglik_full_dirichlet(c: int, counts_seen: List[int], alpha: float) -> float:
    t = sum(counts_seen)
    k = len(counts_seen)
    if k > c:
        return float("-inf")
    ll = log_falling_factorial(c, k)
    ll += gammaln(c * alpha) - gammaln(t + c * alpha)
    for x in counts_seen:
        ll += gammaln(x + alpha) - gammaln(alpha)
    return ll


def build_log_prior(c_candidates: np.ndarray, prior_type: str, lam: float) -> np.ndarray:
    if prior_type == "uniform":
        return np.full_like(c_candidates, -math.log(len(c_candidates)), dtype=float)
    if prior_type == "geometric":
        if not (0 < lam <= 1):
            raise ValueError("--prior-lam must be in (0,1] for geometric prior")
        logp = np.array([math.log(lam) + (c - 1) * math.log(1 - lam) if lam < 1 else 0.0 for c in c_candidates], dtype=float)
        logp -= logsumexp(logp)
        return logp
    raise ValueError(f"Unknown prior_type: {prior_type}")


@dataclass
class SequentialResult:
    rounds: List[int]
    cumulative_samples: List[int]
    distinct_seen: List[int]
    posteriors: Dict[str, np.ndarray]
    posterior_means: Dict[str, List[float]]
    observed_counts_per_round: List[List[int]]


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
    probs = np.ones(len(data), dtype=float) / len(data)
    counts = Counter()

    max_t = batch_size * rounds
    logS2 = log_stirling2_table(max_t, min(max_t, int(np.max(c_candidates))))

    posteriors = {"distinct_only": [], "full_uniform": [], "full_dirichlet": []}
    means = {k: [] for k in posteriors}
    round_ids, cumulative_samples, distinct_seen, observed_counts_per_round = [], [], [], []
    sampled = []

    for r in range(1, rounds + 1):
        draws = rng.choice(np.array(data), size=batch_size, replace=True, p=probs)
        sampled.extend(draws.tolist())
        counts.update(draws.tolist())

        t = len(sampled)
        occ = sorted(counts.values(), reverse=True)
        k = len(occ)

        ll_distinct = np.array([loglik_distinct_only(int(c), t, k, logS2) for c in c_candidates], dtype=float)
        post_distinct = normalize_log_probs(log_prior + ll_distinct)

        ll_fu = np.array([loglik_full_uniform(int(c), occ) for c in c_candidates], dtype=float)
        post_fu = normalize_log_probs(log_prior + ll_fu)

        ll_fd = np.array([loglik_full_dirichlet(int(c), occ, alpha) for c in c_candidates], dtype=float)
        post_fd = normalize_log_probs(log_prior + ll_fd)

        posteriors["distinct_only"].append(post_distinct)
        posteriors["full_uniform"].append(post_fu)
        posteriors["full_dirichlet"].append(post_fd)

        means["distinct_only"].append(posterior_mean(c_candidates, post_distinct))
        means["full_uniform"].append(posterior_mean(c_candidates, post_fu))
        means["full_dirichlet"].append(posterior_mean(candidates := c_candidates, post_fd))

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
    )


def save_summary_csv(outdir: Path, result: SequentialResult, true_m: int) -> None:
    csv_path = outdir / "posterior_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round", "cumulative_samples", "distinct_seen",
            "posterior_mean_distinct_only", "posterior_mean_full_uniform",
            "posterior_mean_full_dirichlet", "true_m", "occupancy_counts_desc"
        ])
        for i, r in enumerate(result.rounds):
            writer.writerow([
                r, result.cumulative_samples[i], result.distinct_seen[i],
                result.posterior_means["distinct_only"][i],
                result.posterior_means["full_uniform"][i],
                result.posterior_means["full_dirichlet"][i],
                true_m, " ".join(map(str, result.observed_counts_per_round[i]))
            ])


def _save_heatmap(matrix: np.ndarray, c_candidates: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(10, 5.5))
    plt.imshow(
        matrix.T, aspect="auto", origin="lower", interpolation="nearest",
        extent=[1, matrix.shape[0], c_candidates[0] - 0.5, c_candidates[-1] + 0.5]
    )
    plt.colorbar(label="Posterior probability")
    plt.xlabel("Round")
    plt.ylabel("Candidate C")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_static_plots(outdir: Path, result: SequentialResult, c_candidates: np.ndarray, true_m: int) -> None:
    for mode, matrix in result.posteriors.items():
        _save_heatmap(matrix, c_candidates, f"Posterior sharpening by round: {mode}", outdir / f"posterior_heatmap_{mode}.png")

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


def _plot_round_combined(outpath: Path, c_candidates: np.ndarray, result: SequentialResult, idx: int, true_m: int) -> None:
    r = result.rounds[idx]
    t = result.cumulative_samples[idx]
    k = result.distinct_seen[idx]
    occ = result.observed_counts_per_round[idx]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    modes = ["distinct_only", "full_uniform", "full_dirichlet"]
    titles = {
        "distinct_only": "Distinct only",
        "full_uniform": "Full counts + uniform multinomial",
        "full_dirichlet": "Full counts + Dirichlet-multinomial",
    }

    for ax, mode in zip(axes.flat[:3], modes):
        post = result.posteriors[mode][idx]
        ax.bar(c_candidates, post, width=0.8)
        ax.axvline(true_m, linestyle="--")
        ax.set_title(titles[mode])
        ax.set_xlabel("Candidate C")
        ax.set_ylabel("Posterior prob.")
        ax.set_ylim(0, max(0.25, float(post.max()) * 1.15))

    axes.flat[3].axis("off")
    info = (
        f"Round: {r}\n"
        f"Cumulative samples: {t}\n"
        f"Distinct seen: {k}\n"
        f"Observed occupancy (desc): {occ}\n\n"
        f"Posterior means:\n"
        f"  distinct_only   = {result.posterior_means['distinct_only'][idx]:.3f}\n"
        f"  full_uniform    = {result.posterior_means['full_uniform'][idx]:.3f}\n"
        f"  full_dirichlet  = {result.posterior_means['full_dirichlet'][idx]:.3f}\n"
        f"  true m          = {true_m}"
    )
    axes.flat[3].text(0.02, 0.98, info, va="top", ha="left", family="monospace")
    fig.suptitle("Posterior evolution by round", fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()


def _plot_round_mode(outpath: Path, c_candidates: np.ndarray, result: SequentialResult, idx: int, true_m: int, mode: str) -> None:
    titles = {
        "distinct_only": "Distinct only",
        "full_uniform": "Full counts + uniform multinomial",
        "full_dirichlet": "Full counts + Dirichlet-multinomial",
    }
    r = result.rounds[idx]
    t = result.cumulative_samples[idx]
    k = result.distinct_seen[idx]
    occ = result.observed_counts_per_round[idx]
    post = result.posteriors[mode][idx]

    plt.figure(figsize=(10, 6))
    plt.bar(c_candidates, post, width=0.8)
    plt.axvline(true_m, linestyle="--", label="true m")
    plt.xlabel("Candidate C")
    plt.ylabel("Posterior probability")
    plt.title(f"{titles[mode]} | round={r} | t={t} | distinct seen={k} | occ={occ}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()


def _build_gif(paths: List[Path], outpath: Path, gif_fps: float) -> bool:
    if len(paths) == 0:
        return False
    try:
        if GIF_BACKEND == "imageio":
            frames = [imageio.imread(p) for p in paths]
            imageio.mimsave(outpath, frames, duration=max(0.04, 1.0 / gif_fps))
            return outpath.exists() and outpath.stat().st_size > 0
        if GIF_BACKEND == "pillow":
            frames = [Image.open(p).convert("P") for p in paths]
            first, rest = frames[0], frames[1:]
            first.save(
                outpath,
                save_all=True,
                append_images=rest,
                duration=max(40, int(1000 / gif_fps)),
                loop=0,
            )
            return outpath.exists() and outpath.stat().st_size > 0
    except Exception as e:
        print(f"GIF creation failed for {outpath.name}: {e}")
        return False
    return False


def make_round_plots_and_gifs(outdir: Path, result: SequentialResult, c_candidates: np.ndarray, true_m: int, gif_fps: float, make_gif: bool) -> List[str]:
    combined_dir = outdir / "posterior_frames_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    mode_dirs = {}
    for mode in result.posteriors:
        d = outdir / f"posterior_frames_{mode}"
        d.mkdir(parents=True, exist_ok=True)
        mode_dirs[mode] = d

    combined_paths = []
    mode_paths = {mode: [] for mode in result.posteriors}

    for i in range(len(result.rounds)):
        cp = combined_dir / f"round_{i+1:03d}.png"
        _plot_round_combined(cp, c_candidates, result, i, true_m)
        combined_paths.append(cp)

        for mode, d in mode_dirs.items():
            mp = d / f"round_{i+1:03d}.png"
            _plot_round_mode(mp, c_candidates, result, i, true_m, mode)
            mode_paths[mode].append(mp)

    created = [
        "posterior_frames_combined/",
        "posterior_frames_distinct_only/",
        "posterior_frames_full_uniform/",
        "posterior_frames_full_dirichlet/",
    ]

    if make_gif:
        if GIF_BACKEND is None:
            print("GIF generation skipped: neither imageio nor Pillow is available.")
            return created

        if _build_gif(combined_paths, outdir / "posterior_evolution_combined.gif", gif_fps):
            created.append("posterior_evolution_combined.gif")

        for mode, paths in mode_paths.items():
            gif_name = f"posterior_evolution_{mode}.gif"
            if _build_gif(paths, outdir / gif_name, gif_fps):
                created.append(gif_name)

    return created


def save_run_info(outdir: Path, counts: List[int], args: argparse.Namespace, true_m: int, total_n: int) -> None:
    p = outdir / "run_info.txt"
    with p.open("w") as f:
        f.write(f"true_m={true_m}\n")
        f.write(f"counts={counts}\n")
        f.write(f"total_N={total_n}\n")
        f.write(f"gif_backend={GIF_BACKEND}\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}={v}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bayesian toolkit for the Invisible Palette Problem, with posterior evolution GIFs.")
    dataset = parser.add_argument_group("dataset construction")
    dataset.add_argument("--counts", type=str, default="", help="Comma-separated counts for colours 0..m-1, e.g. 4,3,7 . Overrides --m and count generation.")
    dataset.add_argument("--m", type=int, default=5, help="Number of occupied colours when generating counts.")
    dataset.add_argument("--count-mode", type=str, choices=["uniform", "skew", "one_heavy"], default="uniform", help="How to generate counts when --counts is not supplied.")
    dataset.add_argument("--min-count", type=int, default=1, help="Minimum count per colour in generated data.")
    dataset.add_argument("--max-count", type=int, default=10, help="Maximum count per colour in generated data.")

    sampling = parser.add_argument_group("sampling process")
    sampling.add_argument("--batch-size", type=int, default=8, help="Samples drawn with replacement per round.")
    sampling.add_argument("--rounds", type=int, default=20, help="Number of sequential posterior updates.")
    sampling.add_argument("--seed", type=int, default=0, help="Random seed for count generation and sampling.")

    bayes = parser.add_argument_group("Bayesian model")
    bayes.add_argument("--c-max", type=int, default=20, help="Maximum candidate value of C to consider.")
    bayes.add_argument("--alpha", type=float, default=0.5, help="Symmetric Dirichlet concentration parameter for the full_dirichlet model.")
    bayes.add_argument("--prior-type", type=str, choices=["uniform", "geometric"], default="uniform", help="Prior over candidate C.")
    bayes.add_argument("--prior-lam", type=float, default=0.2, help="Geometric prior parameter when --prior-type geometric.")

    viz = parser.add_argument_group("visualization")
    viz.add_argument("--make-gif", action="store_true", help="If supplied, build animated GIFs from the per-round posterior plots.")
    viz.add_argument("--gif-fps", type=float, default=2.0, help="Frames per second for posterior evolution GIFs.")

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
    if args.gif_fps <= 0:
        raise ValueError("--gif-fps must be > 0")
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
        counts = generate_counts(args.m, args.count_mode, args.min_count, args.max_count, args.seed)

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
    save_static_plots(outdir, result, c_candidates, true_m)
    created = make_round_plots_and_gifs(outdir, result, c_candidates, true_m, args.gif_fps, args.make_gif)
    save_run_info(outdir, counts, args, true_m, total_n)

    base_created = [
        "posterior_summary.csv",
        "posterior_heatmap_distinct_only.png",
        "posterior_heatmap_full_uniform.png",
        "posterior_heatmap_full_dirichlet.png",
        "posterior_means_across_rounds.png",
        "distinct_seen_across_rounds.png",
        "final_posterior_comparison.png",
        "run_info.txt",
    ] + created

    print("Created files:")
    for name in base_created:
        print(f"- {outdir / name}")


if __name__ == "__main__":
    main()
