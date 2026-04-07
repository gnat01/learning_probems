from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def run_concave_bandit(episodes=3000, lr=0.035, seed=0):
    rng = np.random.default_rng(seed)
    theta = 0.0
    baseline = 0.0
    perf = np.zeros(episodes)
    for t in range(episodes):
        p = sigmoid(theta)
        a = 1 if rng.random() < p else 0
        r = float(a == 1)
        grad = (1 - p) if a == 1 else -p
        theta += lr * (r - baseline) * grad
        baseline = 0.99 * baseline + 0.01 * r
        perf[t] = sigmoid(theta)
    return perf

def run_sparse_path(episodes=6000, horizon=8, lr=0.06, shaping=0.08, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.zeros(horizon)
    baseline = 0.0
    perf = np.zeros(episodes)
    probs = np.zeros((episodes, horizon))
    for t in range(episodes):
        p = sigmoid(theta)
        actions = (rng.random(horizon) < p).astype(np.int64)
        prefix_len = 0
        for a in actions:
            if a == 1:
                prefix_len += 1
            else:
                break
        reward = shaping * prefix_len + float(prefix_len == horizon)
        grads = actions - p
        theta += lr * (reward - baseline) * grads
        baseline = 0.995 * baseline + 0.005 * reward

        p_eval = sigmoid(theta)
        probs[t] = p_eval
        prefix_prod = 1.0
        expected_prefix = 0.0
        for i in range(horizon):
            prefix_prod *= p_eval[i]
            expected_prefix += prefix_prod
        perf[t] = shaping * expected_prefix + np.prod(p_eval)
    return perf, probs

def moving_average(x, w):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")

def main():
    outdir = Path(".")
    n_seeds = 40
    concave = np.stack([run_concave_bandit(seed=s) for s in range(n_seeds)])
    sparse_runs = [run_sparse_path(seed=s) for s in range(n_seeds)]
    sparse = np.stack([r[0] for r in sparse_runs])
    probs = np.stack([r[1] for r in sparse_runs])

    c_mean, c_std = concave.mean(0), concave.std(0)
    s_mean, s_std = sparse.mean(0), sparse.std(0)
    p_mean = probs.mean(0)

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
    plt.plot(x1, moving_average(mg_c, 80), label="Concave: marginal gain")
    plt.plot(x2, moving_average(mg_s, 80), label="Sparse path: marginal gain")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Compute budget B (episodes)")
    plt.ylabel("Smoothed discrete marginal gain")
    plt.title("Marginal gains")
    plt.legend()
    plt.savefig(outdir / "toy_rl_marginal_gains.png", dpi=130, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5.5))
    plt.plot(x1, moving_average(sd_c, 120), label="Concave: second difference")
    plt.plot(x2, moving_average(sd_s, 120), label="Sparse path: second difference")
    plt.axhline(0, linewidth=1)
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
    plt.legend(ncol=2, fontsize=8)
    plt.savefig(outdir / "toy_rl_sparse_depth_probs.png", dpi=120, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
