"""
telemetry_logger.py  (unified, cross-platform: Linux/Pi + macOS)
----------------------------------------------------------------
Collects rich system telemetry once per second and injects REAL anomalies
via stress-ng on a fixed, labeled schedule. Runs identically on the
Raspberry Pi and the Mac so the two datasets are directly comparable.

Key improvements over the first collection:
  * Runs for hours (default 4h), not 40 min  -> ~14k rows/device
  * REAL stress-ng anomalies on BOTH devices (no synthetic data)
  * Categorical anomaly_type label, not just binary
  * Balanced, rotating anomaly schedule so each type is well-sampled
  * Mixed idle/light-load "normal" periods (realistic baseline)
  * Identical schema on both platforms

Usage (run inside tmux on each device):
  # Pi:
  python3 telemetry_logger.py --device-id pi4_edge --duration-hours 4
  # Mac:
  python3 telemetry_logger.py --device-id mac_gateway --duration-hours 4

Dependencies:
  pip install psutil
  stress-ng:
    Pi:  sudo apt install stress-ng
    Mac: brew install stress-ng
"""

import argparse
import csv
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone

import psutil

# -------------------------------------------------------------------------
# Anomaly schedule
# -------------------------------------------------------------------------
# Each entry: (anomaly_type, stress-ng args, duration_seconds)
# We rotate through these on a fixed cycle so every type is well-sampled.
ANOMALY_LIBRARY = [
    ("cpu_spike",     ["--cpu", "0", "--cpu-load", "90"],        45),
    ("mem_pressure",  ["--vm", "2", "--vm-bytes", "60%"],        45),
    ("io_storm",      ["--io", "4"],                              40),
    ("disk_thrash",   ["--hdd", "2", "--hdd-bytes", "256M"],     40),
    ("cpu_cache",     ["--cache", "2"],                           35),
]

# Normal period between anomalies (seconds). Randomized in a band so the
# data isn't perfectly periodic (which a model could trivially exploit).
NORMAL_MIN_S = 90
NORMAL_MAX_S = 180

# During "normal" periods, sometimes apply light background load so that
# "normal" is not artificially pristine. Probability per normal period.
LIGHT_LOAD_PROB = 0.4
LIGHT_LOAD_ARGS = ["--cpu", "1", "--cpu-load", "20"]


# -------------------------------------------------------------------------
# Platform helpers
# -------------------------------------------------------------------------
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


def get_cpu_freq_mhz():
    try:
        f = psutil.cpu_freq()
        return f.current if f else -1.0
    except Exception:
        return -1.0


def get_load_avg():
    try:
        return os.getloadavg()  # (1m, 5m, 15m)
    except (OSError, AttributeError):
        return (-1.0, -1.0, -1.0)


def get_temperature():
    """Best-effort CPU temperature. Pi exposes it; many Macs don't via psutil."""
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for _, entries in temps.items():
                if entries:
                    return entries[0].current
    except Exception:
        pass
    # Pi-specific fallback
    if IS_LINUX:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            pass
    return -1.0


# -------------------------------------------------------------------------
# stress-ng control
# -------------------------------------------------------------------------
def stress_available():
    try:
        subprocess.run(["stress-ng", "--version"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def launch_stress(args, duration_s):
    """Launch stress-ng in the background for duration_s seconds."""
    cmd = ["stress-ng"] + args + ["--timeout", f"{duration_s}s"]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


# -------------------------------------------------------------------------
# Telemetry sampling
# -------------------------------------------------------------------------
SCHEMA = [
    "timestamp", "device_id", "platform",
    "cpu_percent", "cpu_freq_mhz", "load_1m", "load_5m",
    "mem_percent", "mem_used_mb", "swap_percent",
    "disk_read_kbs", "disk_write_kbs",
    "net_sent_kbs", "net_recv_kbs",
    "temperature_c",
    "anomaly_type", "is_anomaly",
]


def sample(device_id, anomaly_type, prev_disk, prev_net, dt):
    cpu = psutil.cpu_percent(interval=None)
    freq = get_cpu_freq_mhz()
    l1, l5, _ = get_load_avg()
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()

    disk = psutil.disk_io_counters()
    net = psutil.net_io_counters()

    if prev_disk and dt > 0:
        disk_r = (disk.read_bytes - prev_disk.read_bytes) / 1024.0 / dt
        disk_w = (disk.write_bytes - prev_disk.write_bytes) / 1024.0 / dt
    else:
        disk_r = disk_w = 0.0
    if prev_net and dt > 0:
        net_s = (net.bytes_sent - prev_net.bytes_sent) / 1024.0 / dt
        net_r = (net.bytes_recv - prev_net.bytes_recv) / 1024.0 / dt
    else:
        net_s = net_r = 0.0

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_id": device_id,
        "platform": platform.system(),
        "cpu_percent": round(cpu, 2),
        "cpu_freq_mhz": round(freq, 1),
        "load_1m": round(l1, 3),
        "load_5m": round(l5, 3),
        "mem_percent": round(vm.percent, 2),
        "mem_used_mb": round(vm.used / 1024 / 1024, 1),
        "swap_percent": round(sw.percent, 2),
        "disk_read_kbs": round(disk_r, 2),
        "disk_write_kbs": round(disk_w, 2),
        "net_sent_kbs": round(net_s, 2),
        "net_recv_kbs": round(net_r, 2),
        "temperature_c": round(get_temperature(), 1),
        "anomaly_type": anomaly_type,
        "is_anomaly": 0 if anomaly_type == "normal" else 1,
    }
    return row, disk, net


# -------------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", required=True,
                    help="e.g. pi4_edge or mac_gateway")
    ap.add_argument("--duration-hours", type=float, default=4.0)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/telemetry_data"))
    ap.add_argument("--no-stress", action="store_true",
                    help="log telemetry only, no anomaly injection (debug)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir,
                            f"telemetry_{args.device_id}_{ts}.csv")

    have_stress = (not args.no_stress) and stress_available()
    if not args.no_stress and not have_stress:
        print("WARNING: stress-ng not found. Install it, or run with "
              "--no-stress. Logging telemetry WITHOUT anomalies.")

    print(f"[{args.device_id}] platform={platform.system()} "
          f"stress={'on' if have_stress else 'off'}")
    print(f"[{args.device_id}] writing to {out_path}")
    print(f"[{args.device_id}] duration={args.duration_hours}h")

    end_time = time.time() + args.duration_hours * 3600
    prev_disk = psutil.disk_io_counters()
    prev_net = psutil.net_io_counters()
    psutil.cpu_percent(interval=None)  # prime
    last_t = time.time()

    # State machine: alternate NORMAL period and ANOMALY period.
    anomaly_idx = 0
    state = "normal"
    state_until = time.time() + random.uniform(NORMAL_MIN_S, NORMAL_MAX_S)
    current_anomaly_type = "normal"
    active_proc = None

    # Optionally start light load during the first normal period
    if have_stress and random.random() < LIGHT_LOAD_PROB:
        active_proc = launch_stress(LIGHT_LOAD_ARGS,
                                    int(state_until - time.time()))

    rows_written = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA)
        writer.writeheader()

        while time.time() < end_time:
            now = time.time()

            # Transition state if the current period elapsed
            if now >= state_until:
                # Clean up any running stress process
                if active_proc and active_proc.poll() is None:
                    active_proc.terminate()
                    try:
                        active_proc.wait(timeout=3)
                    except Exception:
                        active_proc.kill()
                active_proc = None

                if state == "normal":
                    # Begin an anomaly period (rotate through library)
                    atype, sargs, dur = ANOMALY_LIBRARY[
                        anomaly_idx % len(ANOMALY_LIBRARY)]
                    anomaly_idx += 1
                    state = "anomaly"
                    current_anomaly_type = atype
                    state_until = now + dur
                    if have_stress:
                        active_proc = launch_stress(sargs, dur)
                else:
                    # Begin a normal period
                    state = "normal"
                    current_anomaly_type = "normal"
                    dur = random.uniform(NORMAL_MIN_S, NORMAL_MAX_S)
                    state_until = now + dur
                    if have_stress and random.random() < LIGHT_LOAD_PROB:
                        active_proc = launch_stress(LIGHT_LOAD_ARGS, int(dur))

            # Sample telemetry
            dt = now - last_t
            last_t = now
            row, prev_disk, prev_net = sample(
                args.device_id, current_anomaly_type,
                prev_disk, prev_net, dt if dt > 0 else 1.0)
            writer.writerow(row)
            rows_written += 1

            if rows_written % 60 == 0:
                f.flush()
                elapsed_min = (now - (end_time - args.duration_hours*3600)) / 60
                print(f"[{args.device_id}] {rows_written} rows | "
                      f"{elapsed_min:.0f} min | state={state} "
                      f"({current_anomaly_type})")

            # Sleep to maintain ~1 Hz
            sleep_for = 1.0 - (time.time() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)

    # Final cleanup
    if active_proc and active_proc.poll() is None:
        active_proc.terminate()

    print(f"[{args.device_id}] DONE. {rows_written} rows -> {out_path}")


if __name__ == "__main__":
    main()
