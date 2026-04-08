#!/usr/bin/env python3
"""
Invisible Palette Joint Toolkit with posterior evolution plots and GIF animation.

This variant compares three inference views on the same sample stream:
- full_uniform
- fixed_alpha_dirichlet
- joint_C_alpha
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
from typing import List

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
    z = logsumexp(logp.ravel())
    return np.exp(logp - z)


def gammaln(x: float) -> float:
    return math.lgamma(x)


def parse_counts(counts_str: str) -> List[int]:
    vals = [int(x.strip()) for x in counts_str.split(",") if x.strip()]
    if any(v < 0 for v in vals):
        raise ValueError("All counts must be >= 0")
    return vals


def parse_alpha_candidates(alpha_str: str) -> np.ndarray:
    vals = [float(x.strip()) for x in alpha_str.split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError("--alpha-candidates must not be empty if supplied")
    if any(v <= 0 for v in vals):
        raise ValueError("All alpha candidates must be > 0")
    return np.array(sorted(set(vals)), dtype=float)


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


def vectorized_lgamma(values: np.ndarray) -> np.ndarray:
    flat = values.ravel()
    out = np.fromiter((math.lgamma(float(v)) for v in flat), dtype=float, count=len(flat))
    return out.reshape(values.shape)


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


def build_log_prior_c(c_candidates: np.ndarray, prior_type: str, lam: float) -> np.ndarray:
    if prior_type == "uniform":
        return np.full(len(c_candidates), -math.log(len(c_candidates)), dtype=float)
    if prior_type == "geometric":
        if not (0 < lam <= 1):
            raise ValueError("--prior-lam must be in (0,1] for geometric prior")
        logp = np.array([
            math.log(lam) + (c - 1) * math.log(1 - lam) if lam < 1 else 0.0
            for c in c_candidates
        ], dtype=float)
        logp -= logsumexp(logp)
        return logp
    raise ValueError(f"Unknown prior_type: {prior_type}")


def build_log_prior_alpha(alpha_candidates: np.ndarray, prior_type: str) -> np.ndarray:
    if prior_type == "uniform":
        return np.full(len(alpha_candidates), -math.log(len(alpha_candidates)), dtype=float)
    if prior_type == "log_uniform":
        logp = -np.log(alpha_candidates)
        logp -= logsumexp(logp)
        return logp
    raise ValueError(f"Unknown alpha_prior_type: {prior_type}")


def build_alpha_candidates(args: argparse.Namespace) -> np.ndarray:
    if args.alpha_candidates:
        return parse_alpha_candidates(args.alpha_candidates)
    if args.alpha_grid_scale == "linear":
        vals = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    else:
        vals = np.geomspace(args.alpha_min, args.alpha_max, args.alpha_points)
    return np.array(vals, dtype=float)


def posterior_mean(values: np.ndarray, posterior: np.ndarray) -> float:
    return float(np.sum(values * posterior))


def posterior_mode(values: np.ndarray, posterior: np.ndarray) -> float:
    return float(values[int(np.argmax(posterior))])


@dataclass
class JointSequentialResult:
    rounds: List[int]
    cumulative_samples: List[int]
    distinct_seen: List[int]
    observed_counts_per_round: List[List[int]]
    posterior_full_uniform_c: np.ndarray
    posterior_fixed_dirichlet_c: np.ndarray
    posterior_joint_c: np.ndarray
    posterior_joint_alpha: np.ndarray
    posterior_joint: np.ndarray
    posterior_mean_full_uniform_c: List[float]
    posterior_mean_fixed_dirichlet_c: List[float]
    posterior_mean_joint_c: List[float]
    posterior_mean_joint_alpha: List[float]


def run_experiment(
    data: List[int],
    c_candidates: np.ndarray,
    alpha_candidates: np.ndarray,
    batch_size: int,
    rounds: int,
    fixed_alpha: float,
    log_prior_c: np.ndarray,
    log_prior_joint: np.ndarray,
    seed: int,
) -> JointSequentialResult:
    rng = np.random.default_rng(seed)
    data_arr = np.asarray(data)
    counts = Counter()

    max_t = batch_size * rounds
    log_falling = build_log_falling_factorial_table(c_candidates)
    log_c = np.log(c_candidates.astype(float))

    c_alpha_grid = np.outer(alpha_candidates, c_candidates.astype(float))
    lgamma_c_alpha_grid = vectorized_lgamma(c_alpha_grid)
    lgamma_t_plus_c_alpha_grid = np.stack([
        vectorized_lgamma(c_alpha_grid + t) for t in range(max_t + 1)
    ], axis=0)
    log_gamma_alpha = vectorized_lgamma(alpha_candidates)
    lgamma_x_plus_alpha = np.stack([
        vectorized_lgamma(alpha_candidates + x) for x in range(max_t + 1)
    ], axis=0)

    fixed_c_alpha = c_candidates.astype(float) * fixed_alpha
    lgamma_fixed_c_alpha = vectorized_lgamma(fixed_c_alpha)
    lgamma_t_plus_fixed_c_alpha = np.stack([
        vectorized_lgamma(fixed_c_alpha + t) for t in range(max_t + 1)
    ], axis=0)
    log_gamma_fixed_alpha = gammaln(fixed_alpha)

    round_ids = []
    cumulative_samples = []
    distinct_seen = []
    observed_counts_per_round = []

    posterior_full_uniform_c = []
    posterior_fixed_dirichlet_c = []
    posterior_joint_c = []
    posterior_joint_alpha = []
    posterior_joint = []
    posterior_mean_full_uniform_c = []
    posterior_mean_fixed_dirichlet_c = []
    posterior_mean_joint_c = []
    posterior_mean_joint_alpha = []

    sampled = []
    for r in range(1, rounds + 1):
        draws = rng.choice(data_arr, size=batch_size, replace=True)
        sampled.extend(draws.tolist())
        counts.update(draws.tolist())

        t = len(sampled)
        occ = sorted(counts.values(), reverse=True)
        k = len(occ)

        ll_fu = log_falling[k] - t * log_c
        ll_fu[c_candidates < k] = -np.inf
        post_fu = normalize_log_probs(log_prior_c + ll_fu)

        fixed_occ_term = sum(gammaln(x + fixed_alpha) - log_gamma_fixed_alpha for x in occ)
        ll_fixed = log_falling[k] + lgamma_fixed_c_alpha - lgamma_t_plus_fixed_c_alpha[t] + fixed_occ_term
        ll_fixed[c_candidates < k] = -np.inf
        post_fixed = normalize_log_probs(log_prior_c + ll_fixed)

        occ_term_joint = np.zeros(len(alpha_candidates), dtype=float)
        for x in occ:
            occ_term_joint += lgamma_x_plus_alpha[x]
        occ_term_joint -= k * log_gamma_alpha
        ll_joint = (
            log_falling[k][np.newaxis, :]
            + lgamma_c_alpha_grid
            - lgamma_t_plus_c_alpha_grid[t]
            + occ_term_joint[:, np.newaxis]
        )
        ll_joint[:, c_candidates < k] = -np.inf
        post_joint = normalize_log_probs(log_prior_joint + ll_joint)
        post_joint_alpha = post_joint.sum(axis=1)
        post_joint_c_r = post_joint.sum(axis=0)

        posterior_full_uniform_c.append(post_fu)
        posterior_fixed_dirichlet_c.append(post_fixed)
        posterior_joint_c.append(post_joint_c_r)
        posterior_joint_alpha.append(post_joint_alpha)
        posterior_joint.append(post_joint)
        posterior_mean_full_uniform_c.append(posterior_mean(c_candidates.astype(float), post_fu))
        posterior_mean_fixed_dirichlet_c.append(posterior_mean(c_candidates.astype(float), post_fixed))
        posterior_mean_joint_c.append(posterior_mean(c_candidates.astype(float), post_joint_c_r))
        posterior_mean_joint_alpha.append(posterior_mean(alpha_candidates, post_joint_alpha))

        round_ids.append(r)
        cumulative_samples.append(t)
        distinct_seen.append(k)
        observed_counts_per_round.append(occ.copy())

    return JointSequentialResult(
        rounds=round_ids,
        cumulative_samples=cumulative_samples,
        distinct_seen=distinct_seen,
        observed_counts_per_round=observed_counts_per_round,
        posterior_full_uniform_c=np.stack(posterior_full_uniform_c, axis=0),
        posterior_fixed_dirichlet_c=np.stack(posterior_fixed_dirichlet_c, axis=0),
        posterior_joint_c=np.stack(posterior_joint_c, axis=0),
        posterior_joint_alpha=np.stack(posterior_joint_alpha, axis=0),
        posterior_joint=np.stack(posterior_joint, axis=0),
        posterior_mean_full_uniform_c=posterior_mean_full_uniform_c,
        posterior_mean_fixed_dirichlet_c=posterior_mean_fixed_dirichlet_c,
        posterior_mean_joint_c=posterior_mean_joint_c,
        posterior_mean_joint_alpha=posterior_mean_joint_alpha,
    )


def save_summary_csv(outdir: Path, result: JointSequentialResult, true_m: int) -> None:
    csv_path = outdir / "posterior_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "cumulative_samples",
            "distinct_seen",
            "posterior_mean_full_uniform_c",
            "posterior_mean_fixed_dirichlet_c",
            "posterior_mean_joint_c",
            "posterior_mean_joint_alpha",
            "true_m",
            "occupancy_counts_desc",
        ])
        for i, r in enumerate(result.rounds):
            writer.writerow([
                r,
                result.cumulative_samples[i],
                result.distinct_seen[i],
                result.posterior_mean_full_uniform_c[i],
                result.posterior_mean_fixed_dirichlet_c[i],
                result.posterior_mean_joint_c[i],
                result.posterior_mean_joint_alpha[i],
                true_m,
                " ".join(map(str, result.observed_counts_per_round[i])),
            ])


def _save_heatmap(
    matrix: np.ndarray,
    x_extent: tuple[float, float],
    y_extent: tuple[float, float],
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
        extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
    )
    plt.colorbar(label="Posterior probability")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_static_plots(
    outdir: Path,
    result: JointSequentialResult,
    c_candidates: np.ndarray,
    alpha_candidates: np.ndarray,
    true_m: int,
) -> None:
    ensure_matplotlib()
    _save_heatmap(
        result.posterior_full_uniform_c,
        (1, result.posterior_full_uniform_c.shape[0]),
        (c_candidates[0] - 0.5, c_candidates[-1] + 0.5),
        "Round",
        "Candidate C",
        "Posterior sharpening by round: full_uniform",
        outdir / "posterior_heatmap_full_uniform.png",
    )
    _save_heatmap(
        result.posterior_fixed_dirichlet_c,
        (1, result.posterior_fixed_dirichlet_c.shape[0]),
        (c_candidates[0] - 0.5, c_candidates[-1] + 0.5),
        "Round",
        "Candidate C",
        "Posterior sharpening by round: fixed_alpha_dirichlet",
        outdir / "posterior_heatmap_fixed_dirichlet.png",
    )
    _save_heatmap(
        result.posterior_joint_c,
        (1, result.posterior_joint_c.shape[0]),
        (c_candidates[0] - 0.5, c_candidates[-1] + 0.5),
        "Round",
        "Candidate C",
        "Posterior sharpening by round: joint marginal C",
        outdir / "posterior_heatmap_c.png",
    )
    _save_heatmap(
        result.posterior_joint_alpha,
        (1, result.posterior_joint_alpha.shape[0]),
        (alpha_candidates[0], alpha_candidates[-1]),
        "Round",
        "Candidate alpha",
        "Posterior sharpening by round: joint marginal alpha",
        outdir / "posterior_heatmap_alpha.png",
    )

    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(result.rounds, result.posterior_mean_full_uniform_c, label="full_uniform")
    ax1.plot(result.rounds, result.posterior_mean_fixed_dirichlet_c, label="fixed_alpha_dirichlet")
    ax1.plot(result.rounds, result.posterior_mean_joint_c, label="joint_C_alpha")
    ax1.axhline(true_m, linestyle=":", label="true m")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Posterior mean of C")
    ax1.set_title("Posterior means across rounds")
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(result.rounds, result.posterior_mean_joint_alpha, label="joint posterior mean alpha")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Posterior mean of alpha")
    ax2.legend()
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
    plt.bar(x - width, result.posterior_full_uniform_c[-1], width=width, label="full_uniform")
    plt.bar(x, result.posterior_fixed_dirichlet_c[-1], width=width, label="fixed_alpha_dirichlet")
    plt.bar(x + width, result.posterior_joint_c[-1], width=width, label="joint_C_alpha")
    plt.xticks(x, c_candidates)
    plt.xlabel("Candidate C")
    plt.ylabel("Posterior probability")
    plt.title("Final posterior comparison across model assumptions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "final_posterior_comparison.png", dpi=140)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    widths = np.diff(alpha_candidates).mean() if len(alpha_candidates) > 1 else max(0.1, alpha_candidates[0] * 0.25)
    plt.bar(alpha_candidates, result.posterior_joint_alpha[-1], width=widths * 0.9)
    plt.xlabel("Candidate alpha")
    plt.ylabel("Posterior probability")
    plt.title("Final marginal posterior for alpha")
    plt.tight_layout()
    plt.savefig(outdir / "final_posterior_alpha.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(
        result.posterior_joint[-1],
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[c_candidates[0] - 0.5, c_candidates[-1] + 0.5, alpha_candidates[0], alpha_candidates[-1]],
    )
    plt.colorbar(label="Posterior probability")
    plt.axvline(true_m, linestyle="--", color="white", linewidth=1)
    plt.xlabel("Candidate C")
    plt.ylabel("Candidate alpha")
    plt.title("Final joint posterior over (C, alpha)")
    plt.tight_layout()
    plt.savefig(outdir / "final_joint_posterior.png", dpi=140)
    plt.close()


def _plot_round_combined(
    outpath: Path,
    c_candidates: np.ndarray,
    alpha_candidates: np.ndarray,
    result: JointSequentialResult,
    idx: int,
    true_m: int,
) -> None:
    ensure_matplotlib()
    r = result.rounds[idx]
    t = result.cumulative_samples[idx]
    k = result.distinct_seen[idx]
    occ = result.observed_counts_per_round[idx]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(c_candidates, result.posterior_full_uniform_c[idx], width=0.8)
    axes[0, 0].axvline(true_m, linestyle="--")
    axes[0, 0].set_title("Full counts + uniform multinomial")
    axes[0, 0].set_xlabel("Candidate C")
    axes[0, 0].set_ylabel("Posterior prob.")

    axes[0, 1].bar(c_candidates, result.posterior_fixed_dirichlet_c[idx], width=0.8)
    axes[0, 1].axvline(true_m, linestyle="--")
    axes[0, 1].set_title("Fixed-alpha Dirichlet-multinomial")
    axes[0, 1].set_xlabel("Candidate C")
    axes[0, 1].set_ylabel("Posterior prob.")

    axes[1, 0].bar(c_candidates, result.posterior_joint_c[idx], width=0.8)
    axes[1, 0].axvline(true_m, linestyle="--")
    axes[1, 0].set_title("Joint model: marginal posterior for C")
    axes[1, 0].set_xlabel("Candidate C")
    axes[1, 0].set_ylabel("Posterior prob.")

    im = axes[1, 1].imshow(
        result.posterior_joint[idx],
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[c_candidates[0] - 0.5, c_candidates[-1] + 0.5, alpha_candidates[0], alpha_candidates[-1]],
    )
    axes[1, 1].axvline(true_m, linestyle="--", color="white", linewidth=1)
    axes[1, 1].set_title("Joint posterior over (C, alpha)")
    axes[1, 1].set_xlabel("Candidate C")
    axes[1, 1].set_ylabel("Candidate alpha")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    info = (
        f"Round: {r}\n"
        f"Cumulative samples: {t}\n"
        f"Distinct seen: {k}\n"
        f"Observed occupancy (desc): {occ}\n\n"
        f"Posterior mean C:\n"
        f"  full_uniform       = {result.posterior_mean_full_uniform_c[idx]:.3f}\n"
        f"  fixed_dirichlet    = {result.posterior_mean_fixed_dirichlet_c[idx]:.3f}\n"
        f"  joint_C_alpha      = {result.posterior_mean_joint_c[idx]:.3f}\n"
        f"  joint alpha mean   = {result.posterior_mean_joint_alpha[idx]:.4f}\n"
        f"  true m             = {true_m}"
    )
    fig.text(0.69, 0.48, info, va="top", ha="left", family="monospace", fontsize=9)

    fig.suptitle("Posterior evolution by round", fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()


def _plot_round_joint(
    outpath: Path,
    c_candidates: np.ndarray,
    alpha_candidates: np.ndarray,
    result: JointSequentialResult,
    idx: int,
    true_m: int,
) -> None:
    ensure_matplotlib()
    plt.figure(figsize=(8, 6))
    plt.imshow(
        result.posterior_joint[idx],
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[c_candidates[0] - 0.5, c_candidates[-1] + 0.5, alpha_candidates[0], alpha_candidates[-1]],
    )
    plt.colorbar(label="Posterior probability")
    plt.axvline(true_m, linestyle="--", color="white", linewidth=1)
    plt.xlabel("Candidate C")
    plt.ylabel("Candidate alpha")
    plt.title(f"Joint posterior | round={result.rounds[idx]} | t={result.cumulative_samples[idx]}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()


def _plot_round_c_marginal(
    outpath: Path,
    c_candidates: np.ndarray,
    posterior: np.ndarray,
    title: str,
    occ: List[int],
    round_id: int,
    true_m: int,
) -> None:
    ensure_matplotlib()
    plt.figure(figsize=(10, 6))
    plt.bar(c_candidates, posterior, width=0.8)
    plt.axvline(true_m, linestyle="--", label="true m")
    plt.xlabel("Candidate C")
    plt.ylabel("Posterior probability")
    plt.title(f"{title} | round={round_id} | occ={occ}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()


def _plot_round_alpha_marginal(outpath: Path, alpha_candidates: np.ndarray, result: JointSequentialResult, idx: int) -> None:
    ensure_matplotlib()
    widths = np.diff(alpha_candidates).mean() if len(alpha_candidates) > 1 else max(0.1, alpha_candidates[0] * 0.25)
    plt.figure(figsize=(10, 6))
    plt.bar(alpha_candidates, result.posterior_joint_alpha[idx], width=widths * 0.9)
    plt.xlabel("Candidate alpha")
    plt.ylabel("Posterior probability")
    plt.title(f"Joint alpha posterior | round={result.rounds[idx]} | occ={result.observed_counts_per_round[idx]}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()


def _build_gif(paths: List[Path], outpath: Path, gif_fps: float) -> bool:
    if len(paths) == 0:
        return False
    try:
        if GIF_BACKEND == "imageio":
            frames = [imageio.imread(p) for p in paths]
            imageio.mimsave(outpath, frames, format="GIF", duration=max(0.04, 1.0 / gif_fps), loop=0)
        elif GIF_BACKEND == "pillow":
            frames = [Image.open(p).convert("RGBA") for p in paths]
            first, rest = frames[0], frames[1:]
            first.save(
                outpath,
                save_all=True,
                append_images=rest,
                duration=max(40, int(1000 / gif_fps)),
                loop=0,
                optimize=False,
                disposal=2,
            )
        else:
            return False
        if not (outpath.exists() and outpath.stat().st_size > 0):
            return False
        if GIF_BACKEND == "pillow":
            with Image.open(outpath) as img:
                return getattr(img, "n_frames", 1) > 1
        return True
    except Exception as e:
        print(f"GIF creation failed for {outpath.name}: {e}")
        return False


def make_round_plots_and_gifs(
    outdir: Path,
    result: JointSequentialResult,
    c_candidates: np.ndarray,
    alpha_candidates: np.ndarray,
    true_m: int,
    gif_fps: float,
    make_gif: bool,
) -> List[str]:
    combined_dir = outdir / "posterior_frames_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    joint_dir = outdir / "posterior_frames_joint"
    joint_dir.mkdir(parents=True, exist_ok=True)
    full_uniform_dir = outdir / "posterior_frames_full_uniform"
    full_uniform_dir.mkdir(parents=True, exist_ok=True)
    fixed_dirichlet_dir = outdir / "posterior_frames_fixed_dirichlet"
    fixed_dirichlet_dir.mkdir(parents=True, exist_ok=True)
    joint_c_dir = outdir / "posterior_frames_joint_c"
    joint_c_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir = outdir / "posterior_frames_alpha"
    alpha_dir.mkdir(parents=True, exist_ok=True)

    combined_paths = []
    joint_paths = []
    full_uniform_paths = []
    fixed_dirichlet_paths = []
    joint_c_paths = []
    alpha_paths = []

    for i in range(len(result.rounds)):
        combined_path = combined_dir / f"round_{i+1:03d}.png"
        joint_path = joint_dir / f"round_{i+1:03d}.png"
        full_uniform_path = full_uniform_dir / f"round_{i+1:03d}.png"
        fixed_dirichlet_path = fixed_dirichlet_dir / f"round_{i+1:03d}.png"
        joint_c_path = joint_c_dir / f"round_{i+1:03d}.png"
        alpha_path = alpha_dir / f"round_{i+1:03d}.png"

        _plot_round_combined(combined_path, c_candidates, alpha_candidates, result, i, true_m)
        _plot_round_joint(joint_path, c_candidates, alpha_candidates, result, i, true_m)
        _plot_round_c_marginal(
            full_uniform_path,
            c_candidates,
            result.posterior_full_uniform_c[i],
            "Full counts + uniform multinomial",
            result.observed_counts_per_round[i],
            result.rounds[i],
            true_m,
        )
        _plot_round_c_marginal(
            fixed_dirichlet_path,
            c_candidates,
            result.posterior_fixed_dirichlet_c[i],
            "Fixed-alpha Dirichlet-multinomial",
            result.observed_counts_per_round[i],
            result.rounds[i],
            true_m,
        )
        _plot_round_c_marginal(
            joint_c_path,
            c_candidates,
            result.posterior_joint_c[i],
            "Joint model: marginal C posterior",
            result.observed_counts_per_round[i],
            result.rounds[i],
            true_m,
        )
        _plot_round_alpha_marginal(alpha_path, alpha_candidates, result, i)

        combined_paths.append(combined_path)
        joint_paths.append(joint_path)
        full_uniform_paths.append(full_uniform_path)
        fixed_dirichlet_paths.append(fixed_dirichlet_path)
        joint_c_paths.append(joint_c_path)
        alpha_paths.append(alpha_path)

    created = [
        "posterior_frames_combined/",
        "posterior_frames_joint/",
        "posterior_frames_full_uniform/",
        "posterior_frames_fixed_dirichlet/",
        "posterior_frames_joint_c/",
        "posterior_frames_alpha/",
    ]

    if make_gif:
        if GIF_BACKEND is None:
            print("GIF generation skipped: neither imageio nor Pillow is available.")
            return created
        gif_specs = [
            ("posterior_evolution_combined.gif", combined_paths),
            ("posterior_evolution_joint.gif", joint_paths),
            ("posterior_evolution_full_uniform.gif", full_uniform_paths),
            ("posterior_evolution_fixed_dirichlet.gif", fixed_dirichlet_paths),
            ("posterior_evolution_joint_c.gif", joint_c_paths),
            ("posterior_evolution_alpha.gif", alpha_paths),
        ]
        for gif_name, paths in gif_specs:
            if _build_gif(paths, outdir / gif_name, gif_fps):
                created.append(gif_name)
    return created


def save_run_info(
    outdir: Path,
    counts: List[int],
    alpha_candidates: np.ndarray,
    args: argparse.Namespace,
    true_m: int,
    total_n: int,
) -> None:
    p = outdir / "run_info.txt"
    with p.open("w") as f:
        f.write(f"true_m={true_m}\n")
        f.write(f"counts={counts}\n")
        f.write(f"total_N={total_n}\n")
        f.write(f"fixed_alpha={args.fixed_alpha}\n")
        f.write(f"alpha_candidates={alpha_candidates.tolist()}\n")
        f.write(f"gif_backend={GIF_BACKEND}\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}={v}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Joint Bayesian toolkit for the Invisible Palette Problem, comparing full_uniform, fixed-alpha Dirichlet, and joint (C, alpha)."
    )

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
    bayes.add_argument("--prior-type", type=str, choices=["uniform", "geometric"], default="uniform", help="Prior over candidate C.")
    bayes.add_argument("--prior-lam", type=float, default=0.2, help="Geometric prior parameter when --prior-type geometric.")
    bayes.add_argument("--fixed-alpha", type=float, default=0.5, help="Fixed alpha used by the fixed-alpha Dirichlet comparison model.")
    bayes.add_argument("--alpha-candidates", type=str, default="", help="Explicit comma-separated alpha grid, e.g. 0.1,0.2,0.5,1.0 . Overrides alpha grid construction flags.")
    bayes.add_argument("--alpha-min", type=float, default=0.1, help="Minimum alpha candidate when generating an alpha grid.")
    bayes.add_argument("--alpha-max", type=float, default=3.0, help="Maximum alpha candidate when generating an alpha grid.")
    bayes.add_argument("--alpha-points", type=int, default=25, help="Number of alpha candidates when generating an alpha grid.")
    bayes.add_argument("--alpha-grid-scale", type=str, choices=["linear", "log"], default="log", help="Spacing used to generate alpha candidates.")
    bayes.add_argument("--alpha-prior-type", type=str, choices=["uniform", "log_uniform"], default="log_uniform", help="Prior over candidate alpha values.")

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
    if args.fixed_alpha <= 0:
        raise ValueError("--fixed-alpha must be > 0")
    if args.gif_fps <= 0:
        raise ValueError("--gif-fps must be > 0")
    if args.alpha_min <= 0 or args.alpha_max <= 0:
        raise ValueError("--alpha-min and --alpha-max must be > 0")
    if args.alpha_max < args.alpha_min:
        raise ValueError("--alpha-max must be >= --alpha-min")
    if args.alpha_points < 1:
        raise ValueError("--alpha-points must be >= 1")
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
    alpha_candidates = build_alpha_candidates(args)
    log_prior_c = build_log_prior_c(c_candidates, args.prior_type, args.prior_lam)
    log_prior_alpha = build_log_prior_alpha(alpha_candidates, args.alpha_prior_type)
    log_prior_joint = log_prior_alpha[:, np.newaxis] + log_prior_c[np.newaxis, :]

    result = run_experiment(
        data=data,
        c_candidates=c_candidates,
        alpha_candidates=alpha_candidates,
        batch_size=args.batch_size,
        rounds=args.rounds,
        fixed_alpha=args.fixed_alpha,
        log_prior_c=log_prior_c,
        log_prior_joint=log_prior_joint,
        seed=args.seed + 12345,
    )

    save_summary_csv(outdir, result, true_m)
    save_static_plots(outdir, result, c_candidates, alpha_candidates, true_m)
    created = make_round_plots_and_gifs(
        outdir=outdir,
        result=result,
        c_candidates=c_candidates,
        alpha_candidates=alpha_candidates,
        true_m=true_m,
        gif_fps=args.gif_fps,
        make_gif=args.make_gif,
    )
    save_run_info(outdir, counts, alpha_candidates, args, true_m, total_n)

    base_created = [
        "posterior_summary.csv",
        "posterior_heatmap_full_uniform.png",
        "posterior_heatmap_fixed_dirichlet.png",
        "posterior_heatmap_c.png",
        "posterior_heatmap_alpha.png",
        "posterior_means_across_rounds.png",
        "distinct_seen_across_rounds.png",
        "final_posterior_comparison.png",
        "final_posterior_alpha.png",
        "final_joint_posterior.png",
        "run_info.txt",
    ] + created

    print("Created files:")
    for name in base_created:
        print(f"- {outdir / name}")


if __name__ == "__main__":
    main()
