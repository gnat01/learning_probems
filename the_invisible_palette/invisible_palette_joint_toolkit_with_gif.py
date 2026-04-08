#!/usr/bin/env python3
"""
Invisible Palette Joint Toolkit with posterior evolution plots and GIF animation.

This script compares three inference views on the same sample stream:
1) full_uniform
2) full_dirichlet
3) joint_c_alpha

The CLI is intentionally kept close to the original toolkit:
- the same urn construction flags,
- the same prior over C,
- one fixed --alpha for the fixed Dirichlet comparison,
- a built-in alpha grid for the joint model.
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


DEFAULT_ALPHA_GRID = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0], dtype=float)


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


def build_log_prior_alpha(alpha_candidates: np.ndarray) -> np.ndarray:
    logp = -np.log(alpha_candidates)
    logp -= logsumexp(logp)
    return logp


def posterior_mean(values: np.ndarray, posterior: np.ndarray) -> float:
    return float(np.sum(values * posterior))


@dataclass
class JointSequentialResult:
    rounds: List[int]
    cumulative_samples: List[int]
    distinct_seen: List[int]
    observed_counts_per_round: List[List[int]]
    posteriors_c: Dict[str, np.ndarray]
    posterior_alpha_joint: np.ndarray
    posterior_joint: np.ndarray
    posterior_means_c: Dict[str, List[float]]
    posterior_mean_alpha_joint: List[float]


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
    lgamma_t_plus_c_alpha_grid = np.stack([vectorized_lgamma(c_alpha_grid + t) for t in range(max_t + 1)], axis=0)
    log_gamma_alpha = vectorized_lgamma(alpha_candidates)
    lgamma_x_plus_alpha = np.stack([vectorized_lgamma(alpha_candidates + x) for x in range(max_t + 1)], axis=0)

    fixed_c_alpha = c_candidates.astype(float) * fixed_alpha
    lgamma_fixed_c_alpha = vectorized_lgamma(fixed_c_alpha)
    lgamma_t_plus_fixed_c_alpha = np.stack([vectorized_lgamma(fixed_c_alpha + t) for t in range(max_t + 1)], axis=0)
    log_gamma_fixed_alpha = gammaln(fixed_alpha)

    round_ids = []
    cumulative_samples = []
    distinct_seen = []
    observed_counts_per_round = []

    posteriors_c = {
        "full_uniform": [],
        "full_dirichlet": [],
        "joint_c_alpha": [],
    }
    posterior_alpha_joint = []
    posterior_joint = []
    posterior_means_c = {k: [] for k in posteriors_c}
    posterior_mean_alpha_joint = []

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

        occ_term_fixed = sum(gammaln(x + fixed_alpha) - log_gamma_fixed_alpha for x in occ)
        ll_fd = log_falling[k] + lgamma_fixed_c_alpha - lgamma_t_plus_fixed_c_alpha[t] + occ_term_fixed
        ll_fd[c_candidates < k] = -np.inf
        post_fd = normalize_log_probs(log_prior_c + ll_fd)

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
        post_joint_c = post_joint.sum(axis=0)
        post_joint_alpha = post_joint.sum(axis=1)

        posteriors_c["full_uniform"].append(post_fu)
        posteriors_c["full_dirichlet"].append(post_fd)
        posteriors_c["joint_c_alpha"].append(post_joint_c)
        posterior_alpha_joint.append(post_joint_alpha)
        posterior_joint.append(post_joint)

        posterior_means_c["full_uniform"].append(posterior_mean(c_candidates.astype(float), post_fu))
        posterior_means_c["full_dirichlet"].append(posterior_mean(c_candidates.astype(float), post_fd))
        posterior_means_c["joint_c_alpha"].append(posterior_mean(c_candidates.astype(float), post_joint_c))
        posterior_mean_alpha_joint.append(posterior_mean(alpha_candidates, post_joint_alpha))

        round_ids.append(r)
        cumulative_samples.append(t)
        distinct_seen.append(k)
        observed_counts_per_round.append(occ.copy())

    return JointSequentialResult(
        rounds=round_ids,
        cumulative_samples=cumulative_samples,
        distinct_seen=distinct_seen,
        observed_counts_per_round=observed_counts_per_round,
        posteriors_c={k: np.stack(v, axis=0) for k, v in posteriors_c.items()},
        posterior_alpha_joint=np.stack(posterior_alpha_joint, axis=0),
        posterior_joint=np.stack(posterior_joint, axis=0),
        posterior_means_c=posterior_means_c,
        posterior_mean_alpha_joint=posterior_mean_alpha_joint,
    )


def save_summary_csv(outdir: Path, result: JointSequentialResult, true_m: int) -> None:
    csv_path = outdir / "posterior_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "cumulative_samples",
            "distinct_seen",
            "posterior_mean_full_uniform",
            "posterior_mean_full_dirichlet",
            "posterior_mean_joint_c_alpha_c",
            "posterior_mean_joint_c_alpha_alpha",
            "true_m",
            "occupancy_counts_desc",
        ])
        for i, r in enumerate(result.rounds):
            writer.writerow([
                r,
                result.cumulative_samples[i],
                result.distinct_seen[i],
                result.posterior_means_c["full_uniform"][i],
                result.posterior_means_c["full_dirichlet"][i],
                result.posterior_means_c["joint_c_alpha"][i],
                result.posterior_mean_alpha_joint[i],
                true_m,
                " ".join(map(str, result.observed_counts_per_round[i])),
            ])


def _save_heatmap(matrix: np.ndarray, y_values: np.ndarray, title: str, ylabel: str, path: Path) -> None:
    ensure_matplotlib()
    plt.figure(figsize=(10, 5.5))
    plt.imshow(
        matrix.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[1, matrix.shape[0], y_values[0] - 0.5, y_values[-1] + 0.5] if ylabel == "Candidate C" else [1, matrix.shape[0], y_values[0], y_values[-1]],
    )
    plt.colorbar(label="Posterior probability")
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_static_plots(outdir: Path, result: JointSequentialResult, c_candidates: np.ndarray, alpha_candidates: np.ndarray, true_m: int) -> None:
    ensure_matplotlib()
    _save_heatmap(result.posteriors_c["full_uniform"], c_candidates, "Posterior sharpening by round: full_uniform", "Candidate C", outdir / "posterior_heatmap_full_uniform.png")
    _save_heatmap(result.posteriors_c["full_dirichlet"], c_candidates, "Posterior sharpening by round: full_dirichlet", "Candidate C", outdir / "posterior_heatmap_full_dirichlet.png")
    _save_heatmap(result.posteriors_c["joint_c_alpha"], c_candidates, "Posterior sharpening by round: joint marginal C", "Candidate C", outdir / "posterior_heatmap_joint_c.png")
    _save_heatmap(result.posterior_alpha_joint, alpha_candidates, "Posterior sharpening by round: joint marginal alpha", "Candidate alpha", outdir / "posterior_heatmap_joint_alpha.png")

    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(2, 1, 1)
    for mode, vals in result.posterior_means_c.items():
        ax1.plot(result.rounds, vals, label=mode)
    ax1.axhline(true_m, linestyle="--", label="true m")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Posterior mean of C")
    ax1.set_title("Posterior means across rounds")
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(result.rounds, result.posterior_mean_alpha_joint, label="joint_c_alpha")
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
    plt.bar(x - width, result.posteriors_c["full_uniform"][-1], width=width, label="full_uniform")
    plt.bar(x, result.posteriors_c["full_dirichlet"][-1], width=width, label="full_dirichlet")
    plt.bar(x + width, result.posteriors_c["joint_c_alpha"][-1], width=width, label="joint_c_alpha")
    plt.xticks(x, c_candidates)
    plt.xlabel("Candidate C")
    plt.ylabel("Posterior probability")
    plt.title("Final posterior comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "final_posterior_comparison.png", dpi=140)
    plt.close()

    bar_width = np.diff(alpha_candidates).mean() if len(alpha_candidates) > 1 else max(0.1, alpha_candidates[0] * 0.2)
    plt.figure(figsize=(10, 5.5))
    plt.bar(alpha_candidates, result.posterior_alpha_joint[-1], width=bar_width * 0.9)
    plt.xlabel("Candidate alpha")
    plt.ylabel("Posterior probability")
    plt.title("Final marginal posterior for alpha")
    plt.xscale("log")
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
    plt.yscale("log")
    plt.title("Final joint posterior over (C, alpha)")
    plt.tight_layout()
    plt.savefig(outdir / "final_joint_posterior.png", dpi=140)
    plt.close()


def _plot_round_combined(outpath: Path, c_candidates: np.ndarray, alpha_candidates: np.ndarray, result: JointSequentialResult, idx: int, true_m: int) -> None:
    ensure_matplotlib()
    r = result.rounds[idx]
    t = result.cumulative_samples[idx]
    occ = result.observed_counts_per_round[idx]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(c_candidates, result.posteriors_c["full_uniform"][idx], width=0.8)
    axes[0, 0].axvline(true_m, linestyle="--")
    axes[0, 0].set_title("Full counts + uniform multinomial")
    axes[0, 0].set_xlabel("Candidate C")
    axes[0, 0].set_ylabel("Posterior prob.")

    axes[0, 1].bar(c_candidates, result.posteriors_c["full_dirichlet"][idx], width=0.8)
    axes[0, 1].axvline(true_m, linestyle="--")
    axes[0, 1].set_title("Full counts + fixed-alpha Dirichlet")
    axes[0, 1].set_xlabel("Candidate C")
    axes[0, 1].set_ylabel("Posterior prob.")

    axes[1, 0].bar(c_candidates, result.posteriors_c["joint_c_alpha"][idx], width=0.8)
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
    axes[1, 1].set_yscale("log")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    info = (
        f"Round: {r}\n"
        f"Cumulative samples: {t}\n"
        f"Observed occupancy: {occ}\n\n"
        f"Posterior mean C:\n"
        f"  full_uniform    = {result.posterior_means_c['full_uniform'][idx]:.3f}\n"
        f"  full_dirichlet  = {result.posterior_means_c['full_dirichlet'][idx]:.3f}\n"
        f"  joint_c_alpha   = {result.posterior_means_c['joint_c_alpha'][idx]:.3f}\n"
        f"Joint mean alpha:\n"
        f"  {result.posterior_mean_alpha_joint[idx]:.4f}\n"
        f"true m: {true_m}"
    )
    fig.text(0.69, 0.48, info, va="top", ha="left", family="monospace", fontsize=9)

    fig.suptitle("Posterior evolution by round", fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()


def _plot_round_joint(outpath: Path, c_candidates: np.ndarray, alpha_candidates: np.ndarray, result: JointSequentialResult, idx: int, true_m: int) -> None:
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
    plt.yscale("log")
    plt.title(f"Joint posterior | round={result.rounds[idx]} | t={result.cumulative_samples[idx]}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()


def _plot_round_c_marginal(outpath: Path, c_candidates: np.ndarray, posterior: np.ndarray, title: str, occ: List[int], round_id: int, true_m: int) -> None:
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


def _plot_round_alpha_marginal(outpath: Path, alpha_candidates: np.ndarray, posterior: np.ndarray, occ: List[int], round_id: int) -> None:
    ensure_matplotlib()
    width = np.diff(alpha_candidates).mean() if len(alpha_candidates) > 1 else max(0.1, alpha_candidates[0] * 0.2)
    plt.figure(figsize=(10, 6))
    plt.bar(alpha_candidates, posterior, width=width * 0.9)
    plt.xlabel("Candidate alpha")
    plt.ylabel("Posterior probability")
    plt.xscale("log")
    plt.title(f"Joint alpha posterior | round={round_id} | occ={occ}")
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
            first.save(outpath, save_all=True, append_images=rest, duration=max(40, int(1000 / gif_fps)), loop=0, optimize=False, disposal=2)
        else:
            return False
        return outpath.exists() and outpath.stat().st_size > 0
    except Exception as e:
        print(f"GIF creation failed for {outpath.name}: {e}")
        return False


def make_round_plots_and_gifs(outdir: Path, result: JointSequentialResult, c_candidates: np.ndarray, alpha_candidates: np.ndarray, true_m: int, gif_fps: float, make_gif: bool) -> List[str]:
    frame_specs = {
        "combined": [],
        "joint": [],
        "full_uniform": [],
        "full_dirichlet": [],
        "joint_c": [],
        "joint_alpha": [],
    }
    for name in frame_specs:
        (outdir / f"posterior_frames_{name}").mkdir(parents=True, exist_ok=True)

    for i in range(len(result.rounds)):
        occ = result.observed_counts_per_round[i]
        round_id = result.rounds[i]
        combined_path = outdir / "posterior_frames_combined" / f"round_{i+1:03d}.png"
        joint_path = outdir / "posterior_frames_joint" / f"round_{i+1:03d}.png"
        fu_path = outdir / "posterior_frames_full_uniform" / f"round_{i+1:03d}.png"
        fd_path = outdir / "posterior_frames_full_dirichlet" / f"round_{i+1:03d}.png"
        jc_path = outdir / "posterior_frames_joint_c" / f"round_{i+1:03d}.png"
        ja_path = outdir / "posterior_frames_joint_alpha" / f"round_{i+1:03d}.png"

        _plot_round_combined(combined_path, c_candidates, alpha_candidates, result, i, true_m)
        _plot_round_joint(joint_path, c_candidates, alpha_candidates, result, i, true_m)
        _plot_round_c_marginal(fu_path, c_candidates, result.posteriors_c["full_uniform"][i], "Full counts + uniform multinomial", occ, round_id, true_m)
        _plot_round_c_marginal(fd_path, c_candidates, result.posteriors_c["full_dirichlet"][i], "Full counts + fixed-alpha Dirichlet", occ, round_id, true_m)
        _plot_round_c_marginal(jc_path, c_candidates, result.posteriors_c["joint_c_alpha"][i], "Joint model: marginal C", occ, round_id, true_m)
        _plot_round_alpha_marginal(ja_path, alpha_candidates, result.posterior_alpha_joint[i], occ, round_id)

        frame_specs["combined"].append(combined_path)
        frame_specs["joint"].append(joint_path)
        frame_specs["full_uniform"].append(fu_path)
        frame_specs["full_dirichlet"].append(fd_path)
        frame_specs["joint_c"].append(jc_path)
        frame_specs["joint_alpha"].append(ja_path)

    created = [f"posterior_frames_{name}/" for name in frame_specs]
    if make_gif:
        gif_targets = {
            "combined": "posterior_evolution_combined.gif",
            "joint": "posterior_evolution_joint.gif",
            "full_uniform": "posterior_evolution_full_uniform.gif",
            "full_dirichlet": "posterior_evolution_full_dirichlet.gif",
            "joint_c": "posterior_evolution_joint_c.gif",
            "joint_alpha": "posterior_evolution_joint_alpha.gif",
        }
        if GIF_BACKEND is None:
            print("GIF generation skipped: neither imageio nor Pillow is available.")
            return created
        for key, gif_name in gif_targets.items():
            if _build_gif(frame_specs[key], outdir / gif_name, gif_fps):
                created.append(gif_name)
    return created


def save_run_info(outdir: Path, counts: List[int], args: argparse.Namespace, true_m: int, total_n: int, alpha_candidates: np.ndarray) -> None:
    p = outdir / "run_info.txt"
    with p.open("w") as f:
        f.write(f"true_m={true_m}\n")
        f.write(f"counts={counts}\n")
        f.write(f"total_N={total_n}\n")
        f.write(f"joint_alpha_grid={alpha_candidates.tolist()}\n")
        f.write(f"gif_backend={GIF_BACKEND}\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}={v}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Joint comparison toolkit for the Invisible Palette Problem.")
    dataset = parser.add_argument_group("dataset construction")
    dataset.add_argument("--counts", type=str, default="", help="Comma-separated counts for colours 0..m-1, e.g. 4,3,7 . Overrides generated counts.")
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
    bayes.add_argument("--alpha", type=float, default=0.5, help="Fixed alpha for the full_dirichlet comparison model.")
    bayes.add_argument("--prior-type", type=str, choices=["uniform", "geometric"], default="uniform", help="Prior over candidate C.")
    bayes.add_argument("--prior-lam", type=float, default=0.2, help="Geometric prior parameter when --prior-type geometric.")

    viz = parser.add_argument_group("visualization")
    viz.add_argument("--gif-fps", type=float, default=2.0, help="Frames per second for posterior evolution GIFs.")
    viz.add_argument("--no-gif", action="store_true", help="Disable GIF creation. By default, GIFs are generated.")

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
    alpha_candidates = DEFAULT_ALPHA_GRID.copy()
    log_prior_c = build_log_prior_c(c_candidates, args.prior_type, args.prior_lam)
    log_prior_alpha = build_log_prior_alpha(alpha_candidates)
    log_prior_joint = log_prior_alpha[:, np.newaxis] + log_prior_c[np.newaxis, :]

    result = run_experiment(
        data=data,
        c_candidates=c_candidates,
        alpha_candidates=alpha_candidates,
        batch_size=args.batch_size,
        rounds=args.rounds,
        fixed_alpha=args.alpha,
        log_prior_c=log_prior_c,
        log_prior_joint=log_prior_joint,
        seed=args.seed + 12345,
    )

    save_summary_csv(outdir, result, true_m)
    save_static_plots(outdir, result, c_candidates, alpha_candidates, true_m)
    created = make_round_plots_and_gifs(outdir, result, c_candidates, alpha_candidates, true_m, args.gif_fps, make_gif=not args.no_gif)
    save_run_info(outdir, counts, args, true_m, total_n, alpha_candidates)

    base_created = [
        "posterior_summary.csv",
        "posterior_heatmap_full_uniform.png",
        "posterior_heatmap_full_dirichlet.png",
        "posterior_heatmap_joint_c.png",
        "posterior_heatmap_joint_alpha.png",
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
