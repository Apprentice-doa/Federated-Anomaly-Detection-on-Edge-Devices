"""
client_count_isolation_experiment.py
-------------------------------------
Tests whether FedAvg's failure on 2 real clients is caused by CLIENT COUNT
specifically, isolated from heterogeneity (which is held constant).

Design: split the SAME real Pi and gateway data into K sub-clients each
(K=1,2,3,4), so we go from 2 total clients up to 8 total clients, while
each sub-client still only contains data from ONE real device (so the
genuine ARM64-vs-x86_64 heterogeneity between Pi-derived and
gateway-derived sub-clients is fully preserved, not diluted).

If FedAvg's performance improves as client count rises (holding real
heterogeneity fixed), this supports the "few clients -> averaging
misalignment dominates" hypothesis. If it doesn't improve, the client-count
hypothesis is wrong and something else explains the 2-client failure.

Usage:
    python3 client_count_isolation_experiment.py \
        --data-dir ~/telemetry_data \
        --pi-device-id pi4_edge --gateway-device-id mac_gateway \
        --rounds 20 --epochs 5 --seeds 3 \
        --results-dir results_client_count
"""

import argparse
import glob
import json
import os
import warnings
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
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

def load_device(data_dir, device_id):
    pattern = os.path.join(data_dir, f"telemetry_{device_id}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files for device_id='{device_id}'")
    df = pd.read_csv(files[-1], on_bad_lines='skip')
    df = df.dropna(subset=FEATURES + [LABEL])
    return df


def make_model(seed):
    return MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu",
        solver="adam", max_iter=1, warm_start=True, random_state=seed,
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
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    take = min(len(idx0), len(idx1), 25)
    if take == 0:
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
    return {"f1": f1_score(y, yp, zero_division=0),
            "auroc": roc_auc_score(y, ys)}


def split_into_subclients(X, y, k, seed):
    """Split one device's data into k roughly-equal sub-clients, preserving
    class balance in each via stratified splitting."""
    if k == 1:
        return [(X, y)]
    rng = np.random.RandomState(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)
    splits0 = np.array_split(idx0, k)
    splits1 = np.array_split(idx1, k)
    subclients = []
    for s0, s1 in zip(splits0, splits1):
        idx = np.concatenate([s0, s1])
        rng.shuffle(idx)
        subclients.append((X[idx], y[idx]))
    return subclients


def run_federated(all_clients, X_te_global, y_te_global, rounds, epochs, seed):
    """FedAvg across an arbitrary number of clients, each with its own
    persistent model object (fixes the optimizer-reset bug)."""
    g = make_model(seed)
    X0, y0 = all_clients[0]
    g = _balanced_init_fit(g, X0, y0)

    client_models = []
    client_data = []
    for Xc, yc in all_clients:
        if len(np.unique(yc)) < 2:
            continue  # skip degenerate sub-clients
        cm = make_model(seed)
        cm = _balanced_init_fit(cm, Xc, yc)
        cm = set_w(cm, get_w(g))
        cm.max_iter = epochs
        client_models.append(cm)
        client_data.append((Xc, yc))

    if len(client_models) < 2:
        return {"f1": 0.0, "auroc": 0.5}, []

    f1_trace = []
    for rnd in range(1, rounds + 1):
        gw = get_w(g)
        cws, sizes = [], []
        for i, (Xc, yc) in enumerate(client_data):
            cm = client_models[i]
            cm = set_w(cm, gw)
            cm.fit(Xc, yc)
            cws.append(get_w(cm))
            sizes.append(len(Xc))

        total = sum(sizes)
        new_w = []
        for layer_tensors in zip(*cws):
            weighted = sum(w * (s / total) for w, s in zip(layer_tensors, sizes))
            new_w.append(weighted)
        g = set_w(g, new_w)

        m_eval = evaluate(g, X_te_global, y_te_global)
        f1_trace.append(m_eval["f1"])

    final = evaluate(g, X_te_global, y_te_global)
    return final, f1_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/telemetry_data"))
    ap.add_argument("--pi-device-id", default="pi4_edge")
    ap.add_argument("--gateway-device-id", default="mac_gateway")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--results-dir", default="results_client_count")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading real device data...")
    df_pi = load_device(args.data_dir, args.pi_device_id)
    df_gw = load_device(args.data_dir, args.gateway_device_id)

    scaler = StandardScaler().fit(pd.concat([df_pi[FEATURES], df_gw[FEATURES]]))

    X_pi = scaler.transform(df_pi[FEATURES]).astype(np.float32)
    y_pi = df_pi[LABEL].values.astype(np.int64)
    X_gw = scaler.transform(df_gw[FEATURES]).astype(np.float32)
    y_gw = df_gw[LABEL].values.astype(np.int64)

    # Held-out global test set: stratified slice from BOTH devices, fixed
    # across all conditions for fair comparison
    X_pi_tr, X_pi_te, y_pi_tr, y_pi_te = train_test_split(
        X_pi, y_pi, test_size=0.2, random_state=0, stratify=y_pi)
    X_gw_tr, X_gw_te, y_gw_tr, y_gw_te = train_test_split(
        X_gw, y_gw, test_size=0.2, random_state=0, stratify=y_gw)
    X_te_global = np.concatenate([X_pi_te, X_gw_te])
    y_te_global = np.concatenate([y_pi_te, y_gw_te])

    # Sweep: K sub-clients PER DEVICE (so total clients = 2*K)
    k_values = [1, 2, 3, 4]
    results = {"k_per_device": [], "total_clients": [], "f1_mean": [],
               "f1_std": [], "f1_all_seeds": []}

    print(f"\n{'='*70}")
    print("CLIENT COUNT ISOLATION SWEEP (real Pi/gateway heterogeneity held fixed)")
    print(f"{'='*70}")

    for k in k_values:
        seed_f1s = []
        for seed in range(args.seeds):
            pi_subclients = split_into_subclients(X_pi_tr, y_pi_tr, k, seed)
            gw_subclients = split_into_subclients(X_gw_tr, y_gw_tr, k, seed)
            all_clients = pi_subclients + gw_subclients

            final, trace = run_federated(
                all_clients, X_te_global, y_te_global,
                args.rounds, args.epochs, seed)
            seed_f1s.append(final["f1"])

        f1_mean = np.mean(seed_f1s)
        f1_std = np.std(seed_f1s)
        total_clients = 2 * k

        results["k_per_device"].append(k)
        results["total_clients"].append(total_clients)
        results["f1_mean"].append(f1_mean)
        results["f1_std"].append(f1_std)
        results["f1_all_seeds"].append(seed_f1s)

        print(f"\nK={k} sub-clients/device -> {total_clients} total clients:")
        print(f"  Federated F1 = {f1_mean:.3f} +/- {f1_std:.3f}  "
              f"(seeds: {[round(s,3) for s in seed_f1s]})")

    with open(os.path.join(args.results_dir, "client_count_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    x = results["total_clients"]
    y = results["f1_mean"]
    yerr = results["f1_std"]
    ax.errorbar(x, y, yerr=yerr, marker='o', markersize=8, linewidth=2.5,
               color="#2D6A4F", capsize=5)
    ax.set_xlabel("Total number of federated clients", fontsize=12)
    ax.set_ylabel("Federated F1 score", fontsize=12)
    ax.set_title("Does Federation Improve with More Clients?\n"
                 "(real Pi/gateway heterogeneity held constant, "
                 f"{args.seeds} seeds, mean\u00b1std)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "client_count_isolation.png"),
               dpi=150, bbox_inches='tight')

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    if results["f1_mean"][-1] > results["f1_mean"][0]:
        print(f"  F1 IMPROVED from {results['f1_mean'][0]:.3f} ({results['total_clients'][0]} clients) "
              f"to {results['f1_mean'][-1]:.3f} ({results['total_clients'][-1]} clients)")
        print("  -> SUPPORTS the client-count hypothesis")
    else:
        print(f"  F1 did NOT improve from {results['f1_mean'][0]:.3f} ({results['total_clients'][0]} clients) "
              f"to {results['f1_mean'][-1]:.3f} ({results['total_clients'][-1]} clients)")
        print("  -> Client count alone does NOT explain the 2-client failure")


if __name__ == "__main__":
    main()
