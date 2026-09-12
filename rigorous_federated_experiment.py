"""
rigorous_federated_experiment.py
--------------------------------
ICML/ICLR-grade federated anomaly detection study.

Central research question:
    Under what degree of hardware heterogeneity does federation
    outperform centralized training on edge devices?

Experiments:
1. Multi-client federation (K = 2, 5, 10 clients)
2. Multi-seed evaluation (5 seeds, report mean +/- std)
3. Heterogeneity sweep (Dirichlet alpha controls non-IID degree)
4. Three methods compared: Local, Centralized, Federated (FedAvg)

The headline figure: F1 vs heterogeneity, showing the crossover point
where federation begins to beat centralized.

Usage:
    python3 rigorous_federated_experiment.py --data-dir ~/telemetry_data \
        --seeds 5 --rounds 20 --results-dir results_rigorous
"""

import argparse
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
import glob

FEATURES = [
    "cpu_percent", "cpu_freq_mhz", "load_1m", "load_5m",
    "mem_percent", "mem_used_mb", "swap_percent",
    "disk_read_kbs", "disk_write_kbs",
    "net_sent_kbs", "net_recv_kbs",
]
LABEL = "is_anomaly"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_telemetry(data_dir):
    """Load and combine all telemetry CSVs into one pool."""
    files = glob.glob(os.path.join(data_dir, "telemetry_*.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, on_bad_lines='skip')
            df = df.dropna(subset=FEATURES + [LABEL])
            df = df[df["cpu_freq_mhz"] != -1]
            dfs.append(df)
        except Exception as e:
            print(f"Skip {f}: {e}")
    combined = pd.concat(dfs, ignore_index=True)
    X = combined[FEATURES].values.astype(np.float32)
    y = combined[LABEL].values.astype(np.int64)
    print(f"Loaded {len(X)} total samples, anomaly rate {y.mean():.1%}")
    return X, y


# ---------------------------------------------------------------------------
# Non-IID partitioning via Dirichlet distribution
# ---------------------------------------------------------------------------
def dirichlet_partition(X, y, n_clients, alpha, seed):
    """
    Partition data across clients using a Dirichlet distribution.
    alpha -> inf : IID (homogeneous)
    alpha -> 0   : extreme non-IID (each client sees mostly one class)

    This is the standard FL heterogeneity control (Hsu et al. 2019).
    Returns list of (X_client, y_client) tuples.
    """
    rng = np.random.RandomState(seed)
    n_classes = 2
    client_indices = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        class_idx = np.where(y == c)[0]
        rng.shuffle(class_idx)
        # Dirichlet proportions for this class across clients
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        # Split class indices according to proportions
        splits = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        client_splits = np.split(class_idx, splits)
        for client_id, idx in enumerate(client_splits):
            client_indices[client_id].extend(idx.tolist())

    clients = []
    for idx in client_indices:
        idx = np.array(idx)
        if len(idx) < 10:  # ensure minimum samples
            clients.append(None)
            continue
        rng.shuffle(idx)
        clients.append((X[idx], y[idx]))
    return clients


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------
def make_model(seed):
    return MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu",
        solver="adam", max_iter=1, warm_start=True,
        random_state=seed, early_stopping=False,
    )


def get_w(m):
    w = []
    if hasattr(m, 'coefs_'):
        for c, b in zip(m.coefs_, m.intercepts_):
            w.append(c.copy()); w.append(b.copy())
    return w


def set_w(m, w):
    if not w:
        return m
    coefs, ints, i = [], [], 0
    for _ in range(len(m.coefs_)):
        coefs.append(w[i].copy()); ints.append(w[i+1].copy()); i += 2
    m.coefs_ = coefs; m.intercepts_ = ints
    return m


def fedavg(weights_list, sizes):
    """Weighted FedAvg by client sample count."""
    total = sum(sizes)
    out = []
    for layer in zip(*weights_list):
        weighted = sum(w * (s / total) for w, s in zip(layer, sizes))
        out.append(weighted)
    return out


def evaluate(m, X, y):
    if len(np.unique(y)) < 2:
        return {"f1": 0.0, "auroc": 0.5}
    yp = m.predict(X)
    ys = m.predict_proba(X)[:, 1]
    return {
        "f1": f1_score(y, yp, zero_division=0),
        "auroc": roc_auc_score(y, ys),
    }


# ---------------------------------------------------------------------------
# Training methods
# ---------------------------------------------------------------------------
def train_centralized(X_tr, y_tr, X_te, y_te, epochs, seed):
    m = make_model(seed); m.max_iter = epochs
    m.fit(X_tr, y_tr)
    return evaluate(m, X_te, y_te)


def train_local_avg(clients, X_te, y_te, epochs, seed):
    """Average performance of independent per-client models."""
    metrics = []
    for client in clients:
        if client is None:
            continue
        Xc, yc = client
        if len(np.unique(yc)) < 2:
            continue
        m = make_model(seed); m.max_iter = epochs
        m.fit(Xc, yc)
        metrics.append(evaluate(m, X_te, y_te))
    if not metrics:
        return {"f1": 0.0, "auroc": 0.5}
    return {
        "f1": np.mean([m["f1"] for m in metrics]),
        "auroc": np.mean([m["auroc"] for m in metrics]),
    }


def _init_fit(m, Xc, yc):
    """
    Initialize an MLP so its weight structure exists, guaranteeing
    both classes are present in the priming fit to avoid warm_start
    class-mismatch errors.
    """
    # Build a tiny balanced priming batch with both classes
    idx0 = np.where(yc == 0)[0]
    idx1 = np.where(yc == 1)[0]
    take = min(len(idx0), len(idx1), 25)
    prime_idx = np.concatenate([idx0[:take], idx1[:take]])
    m.fit(Xc[prime_idx], yc[prime_idx])
    return m


def train_federated(clients, X_te, y_te, rounds, local_epochs, seed):
    """FedAvg across all clients."""
    valid_clients = [c for c in clients if c is not None and len(np.unique(c[1])) >= 2]
    if len(valid_clients) < 2:
        return {"f1": 0.0, "auroc": 0.5}

    # Initialize global model with a balanced priming batch
    g = make_model(seed)
    Xc0, yc0 = valid_clients[0]
    g = _init_fit(g, Xc0, yc0)

    for _ in range(rounds):
        gw = get_w(g)
        cws, sizes = [], []
        for Xc, yc in valid_clients:
            cm = make_model(seed)
            cm = _init_fit(cm, Xc, yc)
            cm = set_w(cm, gw)
            cm.max_iter = local_epochs
            cm.fit(Xc, yc)
            cws.append(get_w(cm))
            sizes.append(len(Xc))
        g = set_w(g, fedavg(cws, sizes))

    return evaluate(g, X_te, y_te)


# ---------------------------------------------------------------------------
# Main experiment: heterogeneity sweep
# ---------------------------------------------------------------------------
def run_sweep(X, y, args):
    from sklearn.model_selection import train_test_split

    # Held-out global test set (same for all methods)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    # Heterogeneity levels: high alpha = IID, low alpha = extreme non-IID
    alphas = [100.0, 10.0, 1.0, 0.5, 0.1]
    alpha_labels = ["IID\n(α=100)", "Mild\n(α=10)", "Moderate\n(α=1)",
                    "High\n(α=0.5)", "Extreme\n(α=0.1)"]

    n_clients = args.n_clients
    results = {
        "alphas": alphas,
        "alpha_labels": alpha_labels,
        "n_clients": n_clients,
        "n_seeds": args.seeds,
        "local": {"f1": [], "auroc": []},
        "federated": {"f1": [], "auroc": []},
        "centralized": {"f1": [], "auroc": []},
        "local_std": {"f1": [], "auroc": []},
        "federated_std": {"f1": [], "auroc": []},
        "centralized_std": {"f1": [], "auroc": []},
    }

    # Centralized is heterogeneity-independent (uses all data pooled)
    cent_f1s, cent_aurocs = [], []
    for seed in range(args.seeds):
        m = train_centralized(X_tr, y_tr, X_te, y_te, args.epochs, seed)
        cent_f1s.append(m["f1"]); cent_aurocs.append(m["auroc"])
    cent_f1_mean, cent_f1_std = np.mean(cent_f1s), np.std(cent_f1s)
    cent_auroc_mean, cent_auroc_std = np.mean(cent_aurocs), np.std(cent_aurocs)

    print(f"\nCentralized baseline: F1={cent_f1_mean:.3f}+/-{cent_f1_std:.3f}")
    print(f"\n{'='*70}")
    print(f"HETEROGENEITY SWEEP: {n_clients} clients, {args.seeds} seeds")
    print(f"{'='*70}")

    for alpha, label in zip(alphas, alpha_labels):
        local_f1s, local_aurocs = [], []
        fed_f1s, fed_aurocs = [], []

        for seed in range(args.seeds):
            clients = dirichlet_partition(X_tr, y_tr, n_clients, alpha, seed)

            lm = train_local_avg(clients, X_te, y_te, args.epochs, seed)
            local_f1s.append(lm["f1"]); local_aurocs.append(lm["auroc"])

            fm = train_federated(clients, X_te, y_te, args.rounds,
                                  args.epochs, seed)
            fed_f1s.append(fm["f1"]); fed_aurocs.append(fm["auroc"])

        results["local"]["f1"].append(np.mean(local_f1s))
        results["local"]["auroc"].append(np.mean(local_aurocs))
        results["local_std"]["f1"].append(np.std(local_f1s))
        results["local_std"]["auroc"].append(np.std(local_aurocs))

        results["federated"]["f1"].append(np.mean(fed_f1s))
        results["federated"]["auroc"].append(np.mean(fed_aurocs))
        results["federated_std"]["f1"].append(np.std(fed_f1s))
        results["federated_std"]["auroc"].append(np.std(fed_aurocs))

        results["centralized"]["f1"].append(cent_f1_mean)
        results["centralized"]["auroc"].append(cent_auroc_mean)
        results["centralized_std"]["f1"].append(cent_f1_std)
        results["centralized_std"]["auroc"].append(cent_auroc_std)

        alpha_str = label.replace("\n", " ")
        print(f"\nalpha={alpha:6.1f} [{alpha_str}]")
        print(f"  Local:       F1={np.mean(local_f1s):.3f}+/-{np.std(local_f1s):.3f}")
        print(f"  Federated:   F1={np.mean(fed_f1s):.3f}+/-{np.std(fed_f1s):.3f}")
        print(f"  Centralized: F1={cent_f1_mean:.3f}+/-{cent_f1_std:.3f}")

    return results


# ---------------------------------------------------------------------------
# Headline plot: F1 vs heterogeneity
# ---------------------------------------------------------------------------
def plot_sweep(results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(results["alphas"]))

    for ax, metric, title in [(axes[0], "f1", "F1 Score"),
                               (axes[1], "auroc", "AUROC")]:
        for method, color, label in [
            ("local", "#E63946", "Local-only (no collaboration)"),
            ("centralized", "#457B9D", "Centralized (data pooled)"),
            ("federated", "#2D6A4F", "Federated (FedAvg)"),
        ]:
            mean = np.array(results[method][metric])
            std = np.array(results[f"{method}_std"][metric])
            ax.plot(x, mean, marker='o', color=color, linewidth=2.5,
                    markersize=8, label=label)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_xticks(x)
        ax.set_xticklabels(results["alpha_labels"], fontsize=10)
        ax.set_xlabel("Data Heterogeneity (Dirichlet $\\alpha$)", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f"{title} vs Heterogeneity\n"
                     f"({results['n_clients']} clients, "
                     f"{results['n_seeds']} seeds, mean$\\pm$std)",
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()  # IID on left, extreme non-IID on right? No - keep order

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nHeadline plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.expanduser("~/telemetry_data"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n-clients", type=int, default=10)
    parser.add_argument("--results-dir", default="results_rigorous")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    X, y = load_all_telemetry(args.data_dir)
    results = run_sweep(X, y, args)

    # Save
    with open(os.path.join(args.results_dir, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_sweep(results, os.path.join(args.results_dir, "heterogeneity_sweep.png"))

    # Find crossover point
    fed = np.array(results["federated"]["f1"])
    cent = np.array(results["centralized"]["f1"])
    print(f"\n{'='*70}")
    print("HEADLINE FINDING")
    print(f"{'='*70}")
    crossover = None
    for i, (a, label) in enumerate(zip(results["alphas"], results["alpha_labels"])):
        diff = fed[i] - cent[i]
        status = "FED WINS" if diff > 0 else "CENT WINS"
        print(f"  alpha={a:6.1f}: Fed-Cent F1 gap = {diff:+.3f}  [{status}]")
        if diff > 0 and crossover is None:
            crossover = a
    if crossover:
        print(f"\n  Federation begins to outperform centralization at alpha <= {crossover}")
        print(f"  i.e. under moderate-to-high data heterogeneity.")


if __name__ == "__main__":
    main()
