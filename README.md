# Does Federated Learning Need a Crowd?

**Client Count, Not Heterogeneity, Explains FedAvg's Failure on Real Edge Hardware**

> *Deep Learning Indaba 2026 — RIAD Poster Session · Poster GP-80*
> Daniel Akhabue · David Akhabue
>
> 🏆 **Best Poster Award — Deep Learning Indaba 2026**

---

## The One-Sentence Summary

Federated learning is supposed to improve accuracy over training on a single device — we tested that assumption on real hardware and found it fails when the federation has only 2 devices, and we proved exactly why.

---

## Background

Edge devices (security cameras, drones, IoT sensors) face three binding constraints that make cloud-based training impractical:

- **Compute scarcity** — ARM-class CPUs, 4–8 GB RAM
- **Bandwidth scarcity** — 2G/EDGE or intermittent links
- **Data sovereignty** — raw sensor data must not leave the device

FedAvg satisfies all three by exchanging only model weight updates. Prior work shows this works at moderate-to-large fleet sizes. What is under-examined is whether the benefit holds in the **minimal-client regime** — two or three physical devices — the realistic starting point for most edge deployments.

---

## What We Did

### Hardware

| Device | Architecture | Role |
|---|---|---|
| Raspberry Pi 4 | ARM64 | Edge node |
| Laptop/Gateway | x86_64 | Gateway node |

Two physically different devices — different ISA, thermal behaviour, and frequency scaling — producing naturally non-IID telemetry without synthetic partitioning.

### Task

Anomaly detection on real system telemetry (1 Hz, ≈4 hours each device). Real anomalies injected via `stress-ng`: CPU spikes, memory pressure, I/O storms, disk thrashing, cache pressure.

**11 features:** `cpu_percent`, `cpu_freq_mhz`, `load_1m`, `load_5m`, `mem_percent`, `mem_used_mb`, `swap_percent`, `disk_read_kbs`, `disk_write_kbs`, `net_sent_kbs`, `net_recv_kbs`

| Device | Rows | Anomaly % |
|---|---|---|
| Pi 4 (ARM64) | 14,354 | 23.1% |
| Gateway (x86_64) | 14,313 | 23.2% |
| Combined | 28,667 | 23.2% |

### Model

A tiny MLP: `11 → 32 → 16 → 2` (ReLU), ~700 parameters, <10 KB, <1 ms/sample inference on the Pi. Deliberately minimal to fit constrained hardware.

### Three Methods — Identical 100-Iteration Budget

| Method | Description |
|---|---|
| **Local-only** | Each device trains on its own data only |
| **Centralized** | All data pooled, one model trained |
| **Federated (FedAvg)** | 20 rounds × 5 local epochs, weights averaged each round |

---

## Results

### FedAvg Fails on 2 Real Clients

| Setting | Seed 42 | Seed 1 | Seed 7 |
|---|---|---|---|
| Pi local-only | 0.952 | 0.945 | 0.952 |
| Centralized | 0.895 | 0.893 | 0.890 |
| Gateway local-only | 0.851 | 0.848 | 0.836 |
| **Federated** | **0.789** | **0.770** | **0.749** |

Federation is **worst in every seed** — 0.10–0.14 F1 points below centralized, despite an identical matched training budget. The federated model peaks near round 5 (~0.83) then **degrades** through round 20 (~0.75–0.79): more rounds actively hurt performance.

Communication cost: 290 KB total, 14.5 KB/round — viable on 2G, but bandwidth savings do not offset the accuracy cost.

### Client-Count Isolation Study

To separate heterogeneity from client count, we split each real device's data into `k` sub-clients (k = 1–4), keeping Pi-derived and gateway-derived data strictly separated. Real ARM64-vs-x86_64 heterogeneity is held **constant**; only total client count changes.

| k / device | Total clients | F1 mean | F1 std |
|---|---|---|---|
| 1 | 2 | 0.816 | 0.025 |
| 2 | 4 | 0.830 | 0.016 |
| 3 | 6 | 0.839 | 0.003 |
| 4 | 8 | 0.839 | 0.005 |

Federated accuracy improves monotonically and variance shrinks sharply as client count rises — **with heterogeneity never changing**. Client count, not heterogeneity, is the causal variable.

### Supporting Evidence: 10-Client Sweep

Pooled dataset partitioned across 10 simulated clients at 5 heterogeneity levels (α ∈ {100, 10, 1, 0.5, 0.1}, Dirichlet), 5 seeds each. At K=10, federation wins across IID to moderate heterogeneity (F1 0.84 vs. 0.82 centralized) — consistent with the isolation study: above the critical client-count threshold, FedAvg works as expected.

---

## Why This Happens

Two independently trained neural networks solving the same task may arrive at very different weight configurations. FedAvg averages these directly — frequently producing a result that resembles neither original solution. With more clients, misalignment statistically washes out. With only two, there is no cancellation.

**Analogy:** Two people solve a maze — Person A goes left-right-left, Person B goes right-left-right. Averaging their directions gives "go straight" — which hits a wall. The average of two valid paths is not necessarily a valid path.

---

## Practical Implication

For a small pilot fleet (2–5 devices): **test local-only and centralized pooling as baselines before committing to FedAvg.** If privacy prohibits pooling, split each device's data into virtual sub-clients within FedAvg — the isolation study shows this recovers much of the accuracy loss without adding physical hardware.

---

## Repository Structure

```
├── telemetry_logger.py                  # Logs system telemetry at 1 Hz
├── real_hardware_federation.py          # Exp 1: real 2-device FedAvg vs baselines
├── client_count_isolation_experiment.py # Exp 2: client-count isolation sweep
├── rigorous_federated_experiment.py     # Exp 3: 10-client heterogeneity sweep
├── results/                             # Seed 42 results & convergence plot
├── results_seed1/                       # Seed 1 results
├── results_seed2/                       # Seed 7 results
├── results_client_count/                # Client-count isolation results & plot
├── results_rigorous/                    # Heterogeneity sweep results & plot
└── poster_a/                            # LaTeX poster source
```

---

## Limitations

- Only 2 physical devices; sub-clients are data partitions, not genuinely independent hardware
- FedProx and SCAFFOLD not yet tested as remedies at the 2-client scale
- Single task domain; generalization to image/video tasks not yet evaluated
- Energy consumption and unreliable network conditions not yet measured

---

## References

1. McMahan et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*.
2. Hsu et al. (2019). Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification. *arXiv:1909.06335*.
3. Li et al. (2020). Federated Optimization in Heterogeneous Networks. *MLSys*.
4. Karimireddy et al. (2020). SCAFFOLD: Stochastic Controlled Averaging for FL. *ICML*.
5. He et al. (2021). Federated Learning for IoT: On-Device Anomaly Detection. *arXiv:2106.07976*.
6. Marfo et al. (2025). A Framework for Tiny FL in Resource-Constrained IIoT Environments. *IEEE*.
7. Farooq et al. (2024). Harnessing FL for Anomaly Detection in Supercomputer Nodes. *FGCS, 161*, 673–685.
8. Labate et al. (2026). Towards Secure and Scalable Energy Theft Detection. *arXiv:2602.16181*.

---

## Contact

Daniel Akhabue — danakhabue@gmail.com · Deep Learning Indaba 2026, RIAD Poster Session, GP-80
