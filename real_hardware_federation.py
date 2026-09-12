"""
real_hardware_federation.py
----------------------------
Runs the real 2-client federated anomaly detection experiment on the
Pi 4 + gateway telemetry collected by telemetry_logger.py.

Compares three settings on REAL physical hardware:
1. Local-only    -- each device trains on its own data only
2. Centralized   -- all data pooled, one model
3. Federated     -- FedAvg across the two real clients

This produces the numbers for the "Real-Hardware Validation" table
and the convergence plot in the poster.

Usage:
    python3 real_hardware_federation.py \
        --data-dir ~/telemetry_data \
        --pi-device-id pi4_edge \
        --gateway-device-id mac_gateway \
        --rounds 20 --epochs 5 \
        --results-dir results
"""

import argparse
import glob
import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

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
def load_device(data_dir, device_id):
    """Load the most recent telemetry CSV for a given device_id."""
    pattern = os.path.join(data_dir, f"telemetry_{device_id}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No telemetry files found for device_id="
                                f"'{device_id}' matching {pattern}")
    # Use the most recent file
    path = files[-1]
    df = pd.read_csv(path, on_bad_lines='skip')
    df = df.dropna(subset=FEATURES + [LABEL])
    print(f"  Loaded {len(df)} rows from {os.path.basename(path)} "
          f"(anomaly rate {df[LABEL].mean():.1%})")
    return df


# ---------------------------------------------------------------------------
# Model utilities (sklearn MLP with manual weight get/set for FedAvg)
# ---------------------------------------------------------------------------
def make_model(seed=42):
    return MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu",
        solver="adam", max_iter=1, warm_start=True,
        random_state=seed,
    )


def get_w(m):
    w = []
    for c, b in zip(m.coefs_, m.intercepts_):
        w.append(c.copy()); w.append(b.copy())
    return w


def set_w(m, w):
    coefs, ints, i = [], [], 0
    for _ in range(len(m.coefs_)):
        coefs.append(w[i].copy()); ints.append(w[i+1].copy()); i += 2
    m.coefs_ = coefs; m.intercepts_ = ints
    return m


def _balanced_init_fit(m, X, y):
    """Prime the model with a small balanced batch so warm_start works."""
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    take = min(len(idx0), len(idx1), 25)
    if take == 0:
        # Fallback: just use whatever is available
        m.fit(X[:20], y[:20])
        return m
    prime_idx = np.concatenate([idx0[:take], idx1[:take]])
    m.fit(X[prime_idx], y[prime_idx])
    return m


def evaluate(m, X, y):
    if len(np.unique(y)) < 2:
        return {"f1": 0.0, "auroc": 0.5}
    yp = m.predict(X)
    ys = m.predict_proba(X)[:, 1]
    return {
        "f1": f1_score(y, yp, zero_division=0),
        "auroc": roc_auc_score(y, ys),
    }


def model_size_bytes(m):
    """Approximate serialized size of model weights, for comm-cost estimate."""
    total = 0
    for c in m.coefs_:
        total += c.nbytes
    for b in m.intercepts_:
        total += b.nbytes
    return total


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def run_local(devices, rounds, epochs, seed):
    """
    Each device trains independently, evaluated on its own held-out split.

    IMPORTANT: total training budget is matched to federated training
    (rounds x epochs total iterations) so the comparison is fair -- local
    clients are NOT shortchanged on training time relative to federated
    clients, who receive `rounds` separate bursts of `epochs` iterations
    each (rounds * epochs effective passes over their local data).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Local-only training")
    print(f"(matched budget: {rounds} x {epochs} = {rounds*epochs} total iterations, "
          f"same as federated)")
    print("="*60)
    results = {}
    for name, (X, y) in devices.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y)
        m = make_model(seed)
        m = _balanced_init_fit(m, X_tr, y_tr)
        m.max_iter = epochs
        for _ in range(rounds):
            m.fit(X_tr, y_tr)  # repeated short fits, same total budget as federated
        metrics = evaluate(m, X_te, y_te)
        results[name] = metrics
        print(f"  {name}: F1={metrics['f1']:.3f}  AUROC={metrics['auroc']:.3f}")
    return results


def run_centralized(devices, rounds, epochs, seed):
    """
    Pool all devices' data into one model.

    Same matched-budget logic as run_local: trains for rounds*epochs total
    iterations so centralized isn't shortchanged relative to federated either.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Centralized training")
    print(f"(matched budget: {rounds} x {epochs} = {rounds*epochs} total iterations)")
    print("="*60)
    Xs = np.concatenate([X for X, y in devices.values()])
    ys = np.concatenate([y for X, y in devices.values()])
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, ys, test_size=0.25, random_state=seed, stratify=ys)
    m = make_model(seed)
    m = _balanced_init_fit(m, X_tr, y_tr)
    m.max_iter = epochs
    for _ in range(rounds):
        m.fit(X_tr, y_tr)
    metrics = evaluate(m, X_te, y_te)
    print(f"  Centralized: F1={metrics['f1']:.3f}  AUROC={metrics['auroc']:.3f}")
    return metrics


def run_federated(devices, rounds, epochs, seed):
    """FedAvg across the real physical clients, tracking per-round metrics."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Federated training (FedAvg)")
    print("="*60)

    # Per-client train/test split; keep a global test set for fair comparison
    client_splits = {}
    all_X_te, all_y_te = [], []
    for name, (X, y) in devices.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y)
        client_splits[name] = (X_tr, y_tr, X_te, y_te)
        all_X_te.append(X_te); all_y_te.append(y_te)
    X_te_global = np.concatenate(all_X_te)
    y_te_global = np.concatenate(all_y_te)

    # Initialize global model ONCE
    g = make_model(seed)
    first_name = list(client_splits.keys())[0]
    X0, y0, _, _ = client_splits[first_name]
    g = _balanced_init_fit(g, X0, y0)

    history = {name: {"f1": [], "auroc": []} for name in client_splits}
    history["global"] = {"f1": [], "auroc": []}

    total_comm_bytes = 0
    per_round_bytes = None

    # Persistent per-client models: each client keeps its OWN model object
    # across rounds so local optimizer state is not artificially reset by
    # re-instantiating MLPClassifier every round. Weights are overwritten
    # with the global model's weights each round (as FedAvg specifies),
    # but the same model object continues training round-over-round rather
    # than restarting from a freshly seeded random init each time.
    client_models = {}
    for name, (X_tr, y_tr, X_te, y_te) in client_splits.items():
        cm = make_model(seed)
        cm = _balanced_init_fit(cm, X_tr, y_tr)
        cm = set_w(cm, get_w(g))  # sync to global model's initial weights
        cm.max_iter = epochs
        client_models[name] = cm

    for rnd in range(1, rounds + 1):
        gw = get_w(g)
        cws, sizes = [], []
        for name, (X_tr, y_tr, X_te, y_te) in client_splits.items():
            cm = client_models[name]
            cm = set_w(cm, gw)  # sync to current global weights, keep same object
            cm.fit(X_tr, y_tr)  # continue training this round (warm_start preserves weights)
            cws.append(get_w(cm))
            sizes.append(len(X_tr))

            # Track this client's standalone metric trajectory too
            m_eval = evaluate(cm, X_te, y_te)
            history[name]["f1"].append(m_eval["f1"])
            history[name]["auroc"].append(m_eval["auroc"])

            if per_round_bytes is None:
                per_round_bytes = model_size_bytes(cm)

        # FedAvg: weighted average by client sample count
        total = sum(sizes)
        new_w = []
        for layer_tensors in zip(*cws):
            weighted = sum(w * (s / total) for w, s in zip(layer_tensors, sizes))
            new_w.append(weighted)
        g = set_w(g, new_w)

        # Communication this round: each client sends weights up, server
        # sends aggregated weights back down (upload + download)
        total_comm_bytes += per_round_bytes * len(client_splits) * 2

        g_metrics = evaluate(g, X_te_global, y_te_global)
        history["global"]["f1"].append(g_metrics["f1"])
        history["global"]["auroc"].append(g_metrics["auroc"])

        if rnd % 5 == 0 or rnd == 1:
            print(f"  Round {rnd:2d}: Global F1={g_metrics['f1']:.3f}  "
                  f"AUROC={g_metrics['auroc']:.3f}")

    final = evaluate(g, X_te_global, y_te_global)
    print(f"\n  Final federated: F1={final['f1']:.3f}  AUROC={final['auroc']:.3f}")
    print(f"  Total communication: {total_comm_bytes/1024:.1f} KB "
          f"({total_comm_bytes/1024/rounds:.2f} KB/round)")

    return final, history, total_comm_bytes


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_convergence(history, rounds, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(1, rounds + 1)

    colors = {"global": "#2D6A4F"}
    client_names = [k for k in history if k != "global"]
    palette = ["#457B9D", "#E63946", "#F4A261", "#8E7CC3"]
    for i, name in enumerate(client_names):
        colors[name] = palette[i % len(palette)]

    for ax, metric, title in [(axes[0], "f1", "F1 Score"),
                               (axes[1], "auroc", "AUROC")]:
        for name in client_names:
            ax.plot(x, history[name][metric], label=f"{name} (local)",
                    color=colors[name], linewidth=1.8, alpha=0.8)
        ax.plot(x, history["global"][metric], label="Global (federated)",
                color=colors["global"], linewidth=2.5, marker='o', markersize=4)
        ax.set_xlabel("Federation round")
        ax.set_ylabel(title)
        ax.set_title(f"{title} per Federation Round")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nConvergence plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/telemetry_data"))
    ap.add_argument("--pi-device-id", default="pi4_edge")
    ap.add_argument("--gateway-device-id", default="mac_gateway")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading device telemetry...")
    df_pi = load_device(args.data_dir, args.pi_device_id)
    df_gw = load_device(args.data_dir, args.gateway_device_id)

    scaler = StandardScaler().fit(
        pd.concat([df_pi[FEATURES], df_gw[FEATURES]])
    )

    X_pi = scaler.transform(df_pi[FEATURES]).astype(np.float32)
    y_pi = df_pi[LABEL].values.astype(np.int64)
    X_gw = scaler.transform(df_gw[FEATURES]).astype(np.float32)
    y_gw = df_gw[LABEL].values.astype(np.int64)

    devices = {
        "pi4 (ARM64)": (X_pi, y_pi),
        "gateway (x86_64)": (X_gw, y_gw),
    }

    local_results = run_local(devices, args.rounds, args.epochs, args.seed)
    centralized_result = run_centralized(devices, args.rounds, args.epochs, args.seed)
    federated_result, history, comm_bytes = run_federated(
        devices, args.rounds, args.epochs, args.seed)

    plot_convergence(history, args.rounds,
                     os.path.join(args.results_dir, "convergence_plot.png"))

    summary = {
        "local": local_results,
        "centralized": centralized_result,
        "federated": federated_result,
        "communication_bytes_total": comm_bytes,
        "communication_kb_total": round(comm_bytes / 1024, 1),
        "communication_kb_per_round": round(comm_bytes / 1024 / args.rounds, 2),
        "rounds": args.rounds,
        "epochs_per_round": args.epochs,
        "device_rows": {
            args.pi_device_id: len(df_pi),
            args.gateway_device_id: len(df_gw),
        },
        "device_anomaly_rate": {
            args.pi_device_id: round(float(df_pi[LABEL].mean()), 4),
            args.gateway_device_id: round(float(df_gw[LABEL].mean()), 4),
        },
    }

    with open(os.path.join(args.results_dir, "real_hardware_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("SUMMARY TABLE (for poster)")
    print("="*60)
    print(f"{'Setting':<22} {'F1':>8} {'AUROC':>8} {'Comm.':>10}")
    for name, m in local_results.items():
        print(f"{name + ' local':<22} {m['f1']:>8.3f} {m['auroc']:>8.3f} {'0 KB':>10}")
    print(f"{'Centralized':<22} {centralized_result['f1']:>8.3f} "
          f"{centralized_result['auroc']:>8.3f} {'N/A':>10}")
    print(f"{'Federated':<22} {federated_result['f1']:>8.3f} "
          f"{federated_result['auroc']:>8.3f} "
          f"{summary['communication_kb_total']:>7.1f} KB")
    print(f"\nDataset rows: {summary['device_rows']}")
    print(f"Anomaly rates: {summary['device_anomaly_rate']}")


if __name__ == "__main__":
    main()
