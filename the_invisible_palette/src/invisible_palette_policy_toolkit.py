#!/usr/bin/env python3
"""
Adaptive policy benchmark for the Invisible Palette problem.

Outputs:
- stopping time distributions over repeated runs
- posterior width versus total samples
- regret-like comparison between fixed-budget and adaptive-budget schemes
- skew impact on learning C and alpha under a fixed budget
- budget-sweep accuracy and estimate-distribution plots
- averaged final posterior marginals for C and alpha by policy
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math
import os
import tempfile
from collections import Counter, defaultdict
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


def logsumexp(logv: np.ndarray) -> float:
    m = np.max(logv)
    if np.isneginf(m):
        return float("-inf")
    return float(m + np.log(np.sum(np.exp(logv - m))))


def normalize_log_probs(logp: np.ndarray) -> np.ndarray:
    z = logsumexp(logp.ravel())
    return np.exp(logp - z)


def vectorized_lgamma(values: np.ndarray) -> np.ndarray:
    flat = values.ravel()
    out = np.fromiter((math.lgamma(float(v)) for v in flat), dtype=float, count=len(flat))
    return out.reshape(values.shape)


def parse_float_list(value: str) -> np.ndarray:
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected a non-empty comma-separated float list")
    return np.array(vals, dtype=float)


def parse_int_list(value: str) -> List[int]:
    vals = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected a non-empty comma-separated int list")
    return vals


def build_dataset_from_counts(counts: List[int]) -> List[int]:
    data = []
    for color, cnt in enumerate(counts):
        data.extend([color] * cnt)
    return data


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


def generate_dirichlet_counts(m: int, total_balls: int, true_alpha: float, seed: int) -> List[int]:
    if total_balls < m:
        raise ValueError("--total-balls must be at least --m")
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.full(m, true_alpha, dtype=float))
    if total_balls == m:
        return [1] * m
    extra = rng.multinomial(total_balls - m, probs)
    return (extra + 1).tolist()


def posterior_interval_width(values: np.ndarray, posterior: np.ndarray, mass: float) -> float:
    cdf = np.cumsum(posterior)
    lo_q = (1.0 - mass) / 2.0
    hi_q = 1.0 - lo_q
    lo_idx = int(np.searchsorted(cdf, lo_q, side="left"))
    hi_idx = int(np.searchsorted(cdf, hi_q, side="left"))
    lo_idx = min(max(lo_idx, 0), len(values) - 1)
    hi_idx = min(max(hi_idx, 0), len(values) - 1)
    return float(values[hi_idx] - values[lo_idx])


@dataclass
class PosteriorSnapshot:
    cumulative_samples: int
    round_id: int
    posterior_c: np.ndarray
    posterior_alpha: np.ndarray
    mean_c: float
    mean_alpha: float
    width_c: float
    width_alpha: float
    width_log_alpha: float


class JointInferenceEngine:
    def __init__(self, c_candidates: np.ndarray, alpha_candidates: np.ndarray, log_prior_joint: np.ndarray) -> None:
        self.c_candidates = c_candidates
        self.alpha_candidates = alpha_candidates
        self.log_prior_joint = log_prior_joint
        self.log_falling = build_log_falling_factorial_table(c_candidates)
        self.c_alpha_grid = np.outer(alpha_candidates, c_candidates.astype(float))
        self.lgamma_c_alpha_grid = vectorized_lgamma(self.c_alpha_grid)
        self.log_gamma_alpha = vectorized_lgamma(alpha_candidates)

    def posterior_from_occ(self, occ: List[int]) -> PosteriorSnapshot:
        t = sum(occ)
        k = len(occ)
        occ_term = np.zeros(len(self.alpha_candidates), dtype=float)
        for x in occ:
            occ_term += vectorized_lgamma(self.alpha_candidates + x)
        occ_term -= k * self.log_gamma_alpha
        ll_joint = (
            self.log_falling[k][np.newaxis, :]
            + self.lgamma_c_alpha_grid
            - vectorized_lgamma(self.c_alpha_grid + t)
            + occ_term[:, np.newaxis]
        )
        ll_joint[:, self.c_candidates < k] = -np.inf
        post_joint = normalize_log_probs(self.log_prior_joint + ll_joint)
        post_alpha = post_joint.sum(axis=1)
        post_c = post_joint.sum(axis=0)
        mean_c = float(np.sum(self.c_candidates * post_c))
        mean_alpha = float(np.sum(self.alpha_candidates * post_alpha))
        width_c = posterior_interval_width(self.c_candidates.astype(float), post_c, 0.9)
        width_alpha = posterior_interval_width(self.alpha_candidates, post_alpha, 0.9)
        width_log_alpha = posterior_interval_width(np.log(self.alpha_candidates), post_alpha, 0.9)
        return PosteriorSnapshot(
            cumulative_samples=t,
            round_id=0,
            posterior_c=post_c,
            posterior_alpha=post_alpha,
            mean_c=mean_c,
            mean_alpha=mean_alpha,
            width_c=width_c,
            width_alpha=width_alpha,
            width_log_alpha=width_log_alpha,
        )


@dataclass
class PolicyRunResult:
    policy_name: str
    budget_rounds: int
    budget_samples: int
    true_alpha: float
    true_m: int
    replicate: int
    stop_round: int
    stop_samples: int
    final_mean_c: float
    final_mean_alpha: float
    final_width_c: float
    final_width_alpha: float
    final_width_log_alpha: float
    c_abs_error: float
    alpha_log_error: float
    regret_like: float
    final_posterior_c: np.ndarray
    final_posterior_alpha: np.ndarray
    snapshots: List[PosteriorSnapshot]


def should_stop_width(snapshot: PosteriorSnapshot, args: argparse.Namespace) -> bool:
    return (
        snapshot.round_id >= args.policy_min_rounds
        and snapshot.width_c <= args.policy_c_width_threshold
        and snapshot.width_log_alpha <= args.policy_log_alpha_width_threshold
    )


def should_stop_stability(history: List[PosteriorSnapshot], args: argparse.Namespace) -> bool:
    if len(history) < args.policy_stability_patience + 1:
        return False
    if history[-1].round_id < args.policy_min_rounds:
        return False
    recent = history[-(args.policy_stability_patience + 1):]
    c_diffs = [abs(recent[i].mean_c - recent[i - 1].mean_c) for i in range(1, len(recent))]
    alpha_diffs = [abs(math.log(recent[i].mean_alpha) - math.log(recent[i - 1].mean_alpha)) for i in range(1, len(recent))]
    return max(c_diffs) <= args.policy_c_mean_tolerance and max(alpha_diffs) <= args.policy_log_alpha_mean_tolerance


def compute_regret_like(snapshot: PosteriorSnapshot, true_m: int, true_alpha: float, sample_budget: int, cost_weight: float, alpha_weight: float) -> float:
    c_error = abs(snapshot.mean_c - true_m)
    alpha_log_error = abs(math.log(snapshot.mean_alpha) - math.log(true_alpha))
    sample_fraction = snapshot.cumulative_samples / sample_budget
    return float(c_error + alpha_weight * alpha_log_error + cost_weight * sample_fraction)


def run_policy(
    policy_name: str,
    data: List[int],
    true_alpha: float,
    true_m: int,
    replicate: int,
    budget_rounds: int,
    engine: JointInferenceEngine,
    args: argparse.Namespace,
    seed: int,
) -> PolicyRunResult:
    rng = np.random.default_rng(seed)
    data_arr = np.asarray(data)
    counts = Counter()
    snapshots: List[PosteriorSnapshot] = []
    budget_samples = args.batch_size * budget_rounds

    for round_id in range(1, budget_rounds + 1):
        draws = rng.choice(data_arr, size=args.batch_size, replace=True)
        counts.update(draws.tolist())
        occ = sorted(counts.values(), reverse=True)
        snapshot = engine.posterior_from_occ(occ)
        snapshot.round_id = round_id
        snapshots.append(snapshot)

        if policy_name == "fixed_budget":
            continue
        if policy_name == "width_stop" and should_stop_width(snapshot, args):
            break
        if policy_name == "stability_stop" and should_stop_stability(snapshots, args):
            break

    final = snapshots[-1]
    c_abs_error = abs(final.mean_c - true_m)
    alpha_log_error = abs(math.log(final.mean_alpha) - math.log(true_alpha))
    regret_like = compute_regret_like(final, true_m, true_alpha, budget_samples, args.regret_cost_weight, args.regret_alpha_weight)
    return PolicyRunResult(
        policy_name=policy_name,
        budget_rounds=budget_rounds,
        budget_samples=budget_samples,
        true_alpha=true_alpha,
        true_m=true_m,
        replicate=replicate,
        stop_round=final.round_id,
        stop_samples=final.cumulative_samples,
        final_mean_c=final.mean_c,
        final_mean_alpha=final.mean_alpha,
        final_width_c=final.width_c,
        final_width_alpha=final.width_alpha,
        final_width_log_alpha=final.width_log_alpha,
        c_abs_error=c_abs_error,
        alpha_log_error=alpha_log_error,
        regret_like=regret_like,
        final_posterior_c=final.posterior_c,
        final_posterior_alpha=final.posterior_alpha,
        snapshots=snapshots,
    )


def save_benchmark_csv(outdir: Path, results: List[PolicyRunResult]) -> None:
    with (outdir / "policy_benchmark_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy_name",
            "budget_rounds",
            "budget_samples",
            "true_alpha",
            "true_m",
            "replicate",
            "stop_round",
            "stop_samples",
            "final_mean_c",
            "final_mean_alpha",
            "final_width_c",
            "final_width_alpha",
            "final_width_log_alpha",
            "c_abs_error",
            "alpha_log_error",
            "regret_like",
        ])
        for row in results:
            writer.writerow([
                row.policy_name,
                row.budget_rounds,
                row.budget_samples,
                row.true_alpha,
                row.true_m,
                row.replicate,
                row.stop_round,
                row.stop_samples,
                row.final_mean_c,
                row.final_mean_alpha,
                row.final_width_c,
                row.final_width_alpha,
                row.final_width_log_alpha,
                row.c_abs_error,
                row.alpha_log_error,
                row.regret_like,
            ])


def save_width_curves_csv(outdir: Path, rows: List[tuple[float, int, float, float]]) -> None:
    with (outdir / "posterior_width_curves.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_alpha", "cumulative_samples", "mean_width_c", "mean_width_log_alpha"])
        for row in rows:
            writer.writerow(row)


def save_skew_impact_csv(outdir: Path, rows: List[tuple[float, float, float, float, float]]) -> None:
    with (outdir / "skew_impact_fixed_budget.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_alpha", "mean_c_abs_error", "mean_alpha_log_error", "mean_width_c", "mean_width_log_alpha"])
        for row in rows:
            writer.writerow(row)


def unique_in_order(values: List) -> List:
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def grouped_boxplot(ax, groups: List[List[float]], group_labels: List[str], series_labels: List[str], title: str, ylabel: str) -> None:
    n_series = len(series_labels)
    positions = []
    series_data = []
    for g_idx, group in enumerate(groups):
        center = g_idx * (n_series + 1.5)
        for s_idx in range(n_series):
            positions.append(center + s_idx)
            series_data.append(group[s_idx] if s_idx < len(group) else [])
    bp = ax.boxplot(series_data, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
    colors = plt.cm.Set2(np.linspace(0, 1, n_series))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % n_series])
    ticks = [g_idx * (n_series + 1.5) + (n_series - 1) / 2 for g_idx in range(len(group_labels))]
    ax.set_xticks(ticks)
    ax.set_xticklabels(group_labels, rotation=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    handles = [plt.Line2D([0], [0], color=colors[i], lw=8) for i in range(n_series)]
    ax.legend(handles, series_labels, fontsize=8)


def plot_stopping_time_distribution(outdir: Path, results: List[PolicyRunResult]) -> None:
    ensure_matplotlib()
    budgets = sorted(unique_in_order([r.budget_samples for r in results]))
    policies = sorted(unique_in_order([r.policy_name for r in results]))
    groups = []
    labels = []
    for budget in budgets:
        labels.append(str(budget))
        group = []
        for policy in policies:
            vals = [r.stop_samples for r in results if r.budget_samples == budget and r.policy_name == policy]
            group.append(vals)
        groups.append(group)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    grouped_boxplot(ax, groups, labels, policies, "Stopping time distribution by budget", "Stopping samples")
    ax.set_xlabel("Budget samples")
    plt.tight_layout()
    plt.savefig(outdir / "stopping_time_distribution.png", dpi=140)
    plt.close()


def plot_width_curves(outdir: Path, width_curve_rows: List[tuple[float, int, float, float]]) -> None:
    ensure_matplotlib()
    by_alpha_c: Dict[float, List[tuple[int, float]]] = defaultdict(list)
    by_alpha_a: Dict[float, List[tuple[int, float]]] = defaultdict(list)
    for true_alpha, samples, width_c, width_log_alpha in width_curve_rows:
        by_alpha_c[true_alpha].append((samples, width_c))
        by_alpha_a[true_alpha].append((samples, width_log_alpha))

    plt.figure(figsize=(10, 5.5))
    for alpha, pairs in sorted(by_alpha_c.items()):
        pairs = sorted(pairs)
        plt.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=f"true alpha={alpha:g}")
    plt.xlabel("Total samples")
    plt.ylabel("90% posterior width for C")
    plt.title("Posterior width for C versus total samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "posterior_width_vs_samples_c.png", dpi=140)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    for alpha, pairs in sorted(by_alpha_a.items()):
        pairs = sorted(pairs)
        plt.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=f"true alpha={alpha:g}")
    plt.xlabel("Total samples")
    plt.ylabel("90% posterior width for log(alpha)")
    plt.title("Posterior width for alpha versus total samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "posterior_width_vs_samples_alpha.png", dpi=140)
    plt.close()


def plot_regret_like(outdir: Path, results: List[PolicyRunResult], max_budget_samples: int) -> None:
    ensure_matplotlib()
    filtered = [r for r in results if r.budget_samples == max_budget_samples]
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in filtered:
        grouped[row.policy_name].append(row.regret_like)
    policies = list(grouped.keys())
    means = [float(np.mean(grouped[p])) for p in policies]
    stds = [float(np.std(grouped[p])) for p in policies]
    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(policies))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, policies)
    plt.ylabel("Regret-like score")
    plt.title(f"Regret-like comparison at budget={max_budget_samples} samples")
    plt.tight_layout()
    plt.savefig(outdir / "regret_like_comparison.png", dpi=140)
    plt.close()


def plot_skew_impact(outdir: Path, skew_rows: List[tuple[float, float, float, float, float]]) -> None:
    ensure_matplotlib()
    alphas = [r[0] for r in skew_rows]
    c_err = [r[1] for r in skew_rows]
    a_err = [r[2] for r in skew_rows]
    c_width = [r[3] for r in skew_rows]
    a_width = [r[4] for r in skew_rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(alphas, c_err, marker="o")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("Fixed-budget C error vs true alpha")
    axes[0, 0].set_xlabel("True alpha")
    axes[0, 0].set_ylabel("Mean abs error in C")

    axes[0, 1].plot(alphas, a_err, marker="o")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Fixed-budget alpha error vs true alpha")
    axes[0, 1].set_xlabel("True alpha")
    axes[0, 1].set_ylabel("Mean abs log-error in alpha")

    axes[1, 0].plot(alphas, c_width, marker="o")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Fixed-budget posterior width for C")
    axes[1, 0].set_xlabel("True alpha")
    axes[1, 0].set_ylabel("Mean 90% width for C")

    axes[1, 1].plot(alphas, a_width, marker="o")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("Fixed-budget posterior width for log(alpha)")
    axes[1, 1].set_xlabel("True alpha")
    axes[1, 1].set_ylabel("Mean 90% width for log(alpha)")

    plt.tight_layout()
    plt.savefig(outdir / "skew_impact_fixed_budget.png", dpi=140)
    plt.close()


def plot_estimate_distributions_by_policy(outdir: Path, results: List[PolicyRunResult], true_alpha_values: np.ndarray, max_budget_samples: int) -> None:
    ensure_matplotlib()
    filtered = [r for r in results if r.budget_samples == max_budget_samples]
    policies = sorted(unique_in_order([r.policy_name for r in filtered]))

    fig_c, axes_c = plt.subplots(1, len(true_alpha_values), figsize=(5 * len(true_alpha_values), 4), squeeze=False)
    fig_a, axes_a = plt.subplots(1, len(true_alpha_values), figsize=(5 * len(true_alpha_values), 4), squeeze=False)
    for idx, true_alpha in enumerate(true_alpha_values):
        ax_c = axes_c[0, idx]
        ax_a = axes_a[0, idx]
        groups_c = [[[
            r.final_mean_c for r in filtered if r.true_alpha == float(true_alpha) and r.policy_name == policy
        ][0 if False else slice(None)] for policy in policies]]
        groups_a = [[[
            r.final_mean_alpha for r in filtered if r.true_alpha == float(true_alpha) and r.policy_name == policy
        ][0 if False else slice(None)] for policy in policies]]
        grouped_boxplot(ax_c, groups_c, [f"alpha={true_alpha:g}"], policies, f"Estimated C | true alpha={true_alpha:g}", "Posterior mean C")
        grouped_boxplot(ax_a, groups_a, [f"alpha={true_alpha:g}"], policies, f"Estimated alpha | true alpha={true_alpha:g}", "Posterior mean alpha")
        ax_c.axhline(filtered[0].true_m, linestyle="--", color="black", linewidth=1)
        ax_a.axhline(float(true_alpha), linestyle="--", color="black", linewidth=1)
    fig_c.tight_layout()
    fig_c.savefig(outdir / "estimated_c_distribution_by_policy.png", dpi=140)
    plt.close(fig_c)
    fig_a.tight_layout()
    fig_a.savefig(outdir / "estimated_alpha_distribution_by_policy.png", dpi=140)
    plt.close(fig_a)


def plot_average_final_posteriors(outdir: Path, results: List[PolicyRunResult], true_alpha_values: np.ndarray, c_candidates: np.ndarray, alpha_candidates: np.ndarray, max_budget_samples: int) -> None:
    ensure_matplotlib()
    filtered = [r for r in results if r.budget_samples == max_budget_samples]
    policies = sorted(unique_in_order([r.policy_name for r in filtered]))

    fig_c, axes_c = plt.subplots(1, len(true_alpha_values), figsize=(5 * len(true_alpha_values), 4), squeeze=False)
    fig_a, axes_a = plt.subplots(1, len(true_alpha_values), figsize=(5 * len(true_alpha_values), 4), squeeze=False)
    for idx, true_alpha in enumerate(true_alpha_values):
        ax_c = axes_c[0, idx]
        ax_a = axes_a[0, idx]
        for policy in policies:
            rows = [r for r in filtered if r.true_alpha == float(true_alpha) and r.policy_name == policy]
            mean_post_c = np.mean(np.stack([r.final_posterior_c for r in rows], axis=0), axis=0)
            mean_post_a = np.mean(np.stack([r.final_posterior_alpha for r in rows], axis=0), axis=0)
            ax_c.plot(c_candidates, mean_post_c, label=policy)
            ax_a.plot(alpha_candidates, mean_post_a, label=policy)
        ax_c.set_title(f"Average final posterior for C | true alpha={true_alpha:g}")
        ax_c.set_xlabel("Candidate C")
        ax_c.set_ylabel("Posterior probability")
        ax_a.set_title(f"Average final posterior for alpha | true alpha={true_alpha:g}")
        ax_a.set_xlabel("Candidate alpha")
        ax_a.set_ylabel("Posterior probability")
        ax_a.set_xscale("log")
    axes_c[0, 0].legend()
    axes_a[0, 0].legend()
    fig_c.tight_layout()
    fig_c.savefig(outdir / "average_final_posterior_c_by_policy.png", dpi=140)
    plt.close(fig_c)
    fig_a.tight_layout()
    fig_a.savefig(outdir / "average_final_posterior_alpha_by_policy.png", dpi=140)
    plt.close(fig_a)


def plot_budget_sweep_errors(outdir: Path, results: List[PolicyRunResult], true_alpha_values: np.ndarray) -> None:
    ensure_matplotlib()
    policies = sorted(unique_in_order([r.policy_name for r in results]))
    budgets = sorted(unique_in_order([r.budget_samples for r in results]))

    fig_c, axes_c = plt.subplots(1, len(true_alpha_values), figsize=(5 * len(true_alpha_values), 4), squeeze=False)
    fig_a, axes_a = plt.subplots(1, len(true_alpha_values), figsize=(5 * len(true_alpha_values), 4), squeeze=False)
    for idx, true_alpha in enumerate(true_alpha_values):
        ax_c = axes_c[0, idx]
        ax_a = axes_a[0, idx]
        for policy in policies:
            c_vals = []
            a_vals = []
            for budget in budgets:
                rows = [r for r in results if r.true_alpha == float(true_alpha) and r.policy_name == policy and r.budget_samples == budget]
                c_vals.append(float(np.mean([r.c_abs_error for r in rows])))
                a_vals.append(float(np.mean([r.alpha_log_error for r in rows])))
            ax_c.plot(budgets, c_vals, marker="o", label=policy)
            ax_a.plot(budgets, a_vals, marker="o", label=policy)
        ax_c.set_title(f"C error vs budget | true alpha={true_alpha:g}")
        ax_c.set_xlabel("Budget samples")
        ax_c.set_ylabel("Mean abs error in C")
        ax_a.set_title(f"Alpha error vs budget | true alpha={true_alpha:g}")
        ax_a.set_xlabel("Budget samples")
        ax_a.set_ylabel("Mean abs log-error in alpha")
    axes_c[0, 0].legend()
    axes_a[0, 0].legend()
    fig_c.tight_layout()
    fig_c.savefig(outdir / "budget_sweep_c_error.png", dpi=140)
    plt.close(fig_c)
    fig_a.tight_layout()
    fig_a.savefig(outdir / "budget_sweep_alpha_error.png", dpi=140)
    plt.close(fig_a)


def plot_budget_sweep_estimate_distributions(outdir: Path, results: List[PolicyRunResult], true_alpha_values: np.ndarray) -> None:
    ensure_matplotlib()
    policies = sorted(unique_in_order([r.policy_name for r in results]))
    budgets = sorted(unique_in_order([r.budget_samples for r in results]))

    fig_c, axes_c = plt.subplots(len(true_alpha_values), 1, figsize=(12, 4 * len(true_alpha_values)), squeeze=False)
    fig_a, axes_a = plt.subplots(len(true_alpha_values), 1, figsize=(12, 4 * len(true_alpha_values)), squeeze=False)
    for idx, true_alpha in enumerate(true_alpha_values):
        groups_c = []
        groups_a = []
        labels = []
        for budget in budgets:
            labels.append(str(budget))
            group_c = []
            group_a = []
            for policy in policies:
                rows = [r for r in results if r.true_alpha == float(true_alpha) and r.policy_name == policy and r.budget_samples == budget]
                group_c.append([r.final_mean_c for r in rows])
                group_a.append([r.final_mean_alpha for r in rows])
            groups_c.append(group_c)
            groups_a.append(group_a)
        grouped_boxplot(axes_c[idx, 0], groups_c, labels, policies, f"Estimated C distribution by budget | true alpha={true_alpha:g}", "Posterior mean C")
        grouped_boxplot(axes_a[idx, 0], groups_a, labels, policies, f"Estimated alpha distribution by budget | true alpha={true_alpha:g}", "Posterior mean alpha")
        axes_c[idx, 0].set_xlabel("Budget samples")
        axes_a[idx, 0].set_xlabel("Budget samples")
    fig_c.tight_layout()
    fig_c.savefig(outdir / "budget_sweep_estimated_c_distribution.png", dpi=140)
    plt.close(fig_c)
    fig_a.tight_layout()
    fig_a.savefig(outdir / "budget_sweep_estimated_alpha_distribution.png", dpi=140)
    plt.close(fig_a)


def save_run_info(outdir: Path, args: argparse.Namespace, alpha_grid: np.ndarray, budget_rounds_list: List[int]) -> None:
    with (outdir / "run_info.txt").open("w") as f:
        f.write(f"alpha_candidates={alpha_grid.tolist()}\n")
        f.write(f"budget_rounds_list={budget_rounds_list}\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}={v}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive policy benchmark for support-size and alpha inference in the Invisible Palette problem.")
    parser.add_argument("--m", type=int, default=8, help="True number of occupied colours in generated urns.")
    parser.add_argument("--total-balls", type=int, default=80, help="Total empirical balls in each generated urn.")
    parser.add_argument("--true-alpha-values", type=str, default="0.1,0.3,1.0,3.0", help="Comma-separated true alpha values used to generate urns.")
    parser.add_argument("--replicates", type=int, default=24, help="Number of repeated urn/sample experiments per true alpha and policy.")
    parser.add_argument("--batch-size", type=int, default=8, help="Samples drawn with replacement per round.")
    parser.add_argument("--max-rounds", type=int, default=20, help="Maximum rounds available to a policy.")
    parser.add_argument("--budget-rounds-list", type=str, default="", help="Comma-separated budget round values for the budget sweep. Defaults to a short sweep ending at --max-rounds.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--c-max", type=int, default=20, help="Maximum candidate C considered by the posterior.")
    parser.add_argument("--prior-type", type=str, choices=["uniform", "geometric"], default="uniform", help="Prior over candidate C.")
    parser.add_argument("--prior-lam", type=float, default=0.2, help="Geometric prior parameter when --prior-type geometric.")
    parser.add_argument("--alpha-candidates", type=str, default="", help="Explicit comma-separated alpha grid.")
    parser.add_argument("--alpha-min", type=float, default=0.05, help="Minimum alpha candidate when generating an alpha grid.")
    parser.add_argument("--alpha-max", type=float, default=5.0, help="Maximum alpha candidate when generating an alpha grid.")
    parser.add_argument("--alpha-points", type=int, default=31, help="Number of alpha candidates when generating an alpha grid.")
    parser.add_argument("--alpha-grid-scale", type=str, choices=["linear", "log"], default="log", help="Spacing used for generated alpha grids.")
    parser.add_argument("--alpha-prior-type", type=str, choices=["uniform", "log_uniform"], default="log_uniform", help="Prior over candidate alpha values.")
    parser.add_argument("--policy-min-rounds", type=int, default=4, help="Minimum rounds before an adaptive policy can stop.")
    parser.add_argument("--policy-c-width-threshold", type=float, default=2.0, help="Stopping threshold for 90%% posterior width of C.")
    parser.add_argument("--policy-log-alpha-width-threshold", type=float, default=0.9, help="Stopping threshold for 90%% posterior width of log(alpha).")
    parser.add_argument("--policy-stability-patience", type=int, default=3, help="Consecutive rounds required for stability stopping.")
    parser.add_argument("--policy-c-mean-tolerance", type=float, default=0.1, help="Tolerance for posterior mean C changes under stability stopping.")
    parser.add_argument("--policy-log-alpha-mean-tolerance", type=float, default=0.08, help="Tolerance for log posterior mean alpha changes under stability stopping.")
    parser.add_argument("--regret-cost-weight", type=float, default=0.5, help="Cost weight in the regret-like score.")
    parser.add_argument("--regret-alpha-weight", type=float, default=1.0, help="Weight on alpha log-error in the regret-like score.")
    parser.add_argument("--outdir", type=str, default=".", help="Directory for benchmark outputs.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.m < 1:
        raise ValueError("--m must be >= 1")
    if args.total_balls < args.m:
        raise ValueError("--total-balls must be at least --m")
    if args.replicates < 1:
        raise ValueError("--replicates must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be >= 1")
    if args.c_max < args.m:
        raise ValueError("--c-max must be >= --m")
    if args.alpha_min <= 0 or args.alpha_max <= 0:
        raise ValueError("--alpha-min and --alpha-max must be > 0")
    if args.alpha_max < args.alpha_min:
        raise ValueError("--alpha-max must be >= --alpha-min")
    if args.alpha_points < 1:
        raise ValueError("--alpha-points must be >= 1")


def build_alpha_grid(args: argparse.Namespace) -> np.ndarray:
    if args.alpha_candidates:
        return parse_float_list(args.alpha_candidates)
    if args.alpha_grid_scale == "linear":
        return np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    return np.geomspace(args.alpha_min, args.alpha_max, args.alpha_points)


def build_budget_rounds_list(args: argparse.Namespace) -> List[int]:
    if args.budget_rounds_list:
        budgets = parse_int_list(args.budget_rounds_list)
    else:
        candidates = sorted(set([max(2, args.max_rounds // 4), max(3, args.max_rounds // 2), max(4, (3 * args.max_rounds) // 4), args.max_rounds]))
        budgets = [b for b in candidates if b <= args.max_rounds]
    budgets = sorted(set(budgets))
    return budgets


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    true_alpha_values = parse_float_list(args.true_alpha_values)
    alpha_candidates = build_alpha_grid(args)
    budget_rounds_list = build_budget_rounds_list(args)
    c_candidates = np.arange(1, args.c_max + 1)
    log_prior_c = build_log_prior_c(c_candidates, args.prior_type, args.prior_lam)
    log_prior_alpha = build_log_prior_alpha(alpha_candidates, args.alpha_prior_type)
    log_prior_joint = log_prior_alpha[:, np.newaxis] + log_prior_c[np.newaxis, :]
    engine = JointInferenceEngine(c_candidates, alpha_candidates, log_prior_joint)

    policies = ["fixed_budget", "width_stop", "stability_stop"]
    all_results: List[PolicyRunResult] = []
    width_curve_accum: Dict[tuple[float, int], List[tuple[float, float]]] = defaultdict(list)
    skew_fixed_rows: Dict[float, List[PolicyRunResult]] = defaultdict(list)

    max_budget_rounds = max(budget_rounds_list)
    max_budget_samples = args.batch_size * max_budget_rounds

    for alpha_idx, true_alpha in enumerate(true_alpha_values):
        for replicate in range(args.replicates):
            urn_seed = args.seed + alpha_idx * 100000 + replicate
            counts = generate_dirichlet_counts(args.m, args.total_balls, float(true_alpha), urn_seed)
            data = build_dataset_from_counts(counts)

            for budget_rounds in budget_rounds_list:
                for policy_offset, policy_name in enumerate(policies):
                    policy_seed = urn_seed + 1000 * (budget_rounds + 1) + 1000000 * (policy_offset + 1)
                    result = run_policy(policy_name, data, float(true_alpha), args.m, replicate, budget_rounds, engine, args, policy_seed)
                    all_results.append(result)
                    if policy_name == "fixed_budget" and budget_rounds == max_budget_rounds:
                        skew_fixed_rows[float(true_alpha)].append(result)
                        for snap in result.snapshots:
                            width_curve_accum[(float(true_alpha), snap.cumulative_samples)].append((snap.width_c, snap.width_log_alpha))

    width_curve_rows = []
    for (alpha, samples), vals in sorted(width_curve_accum.items()):
        width_curve_rows.append((alpha, samples, float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals]))))

    skew_rows = []
    for alpha in sorted(skew_fixed_rows):
        rows = skew_fixed_rows[alpha]
        skew_rows.append((
            alpha,
            float(np.mean([r.c_abs_error for r in rows])),
            float(np.mean([r.alpha_log_error for r in rows])),
            float(np.mean([r.final_width_c for r in rows])),
            float(np.mean([r.final_width_log_alpha for r in rows])),
        ))

    save_benchmark_csv(outdir, all_results)
    save_width_curves_csv(outdir, width_curve_rows)
    save_skew_impact_csv(outdir, skew_rows)
    plot_stopping_time_distribution(outdir, all_results)
    plot_width_curves(outdir, width_curve_rows)
    plot_regret_like(outdir, all_results, max_budget_samples)
    plot_skew_impact(outdir, skew_rows)
    plot_estimate_distributions_by_policy(outdir, all_results, true_alpha_values, max_budget_samples)
    plot_average_final_posteriors(outdir, all_results, true_alpha_values, c_candidates, alpha_candidates, max_budget_samples)
    plot_budget_sweep_errors(outdir, all_results, true_alpha_values)
    plot_budget_sweep_estimate_distributions(outdir, all_results, true_alpha_values)
    save_run_info(outdir, args, alpha_candidates, budget_rounds_list)

    print("Created files:")
    for name in [
        "policy_benchmark_summary.csv",
        "posterior_width_curves.csv",
        "skew_impact_fixed_budget.csv",
        "stopping_time_distribution.png",
        "posterior_width_vs_samples_c.png",
        "posterior_width_vs_samples_alpha.png",
        "regret_like_comparison.png",
        "skew_impact_fixed_budget.png",
        "estimated_c_distribution_by_policy.png",
        "estimated_alpha_distribution_by_policy.png",
        "average_final_posterior_c_by_policy.png",
        "average_final_posterior_alpha_by_policy.png",
        "budget_sweep_c_error.png",
        "budget_sweep_alpha_error.png",
        "budget_sweep_estimated_c_distribution.png",
        "budget_sweep_estimated_alpha_distribution.png",
        "run_info.txt",
    ]:
        print(f"- {outdir / name}")


if __name__ == "__main__":
    main()
