from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


G = 1.0
DT = 0.05
TRAIN_STEPS = 100
VAL_STEPS = 100
TEST_STEPS = 1000
TEST_LONG_STEPS = 2000
TEST_HALF_STEPS = TEST_STEPS * 2
THETA_MIN = 0.30
THETA_MAX = 1.30
PTHETA_MAX = 1.00
PPHI_MIN = 0.25
PPHI_MAX = 0.95
ENERGY_MIN = 0.30
ENERGY_MAX = 2.40
SEED = 20260424


def safe_sin(theta):
    return np.clip(np.sin(theta), 1e-8, None)


def wrap_phi(phi):
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def get_hamiltonian(state):
    arr = np.asarray(state, dtype=np.float64)
    if arr.ndim == 1:
        theta = arr[0]
        p_theta = arr[2]
        p_phi = arr[3]
        s = safe_sin(theta)
        return 0.5 * (p_theta ** 2 + (p_phi ** 2) / (s ** 2)) + G * (1.0 - np.cos(theta))
    theta = arr[..., 0]
    p_theta = arr[..., 2]
    p_phi = arr[..., 3]
    s = safe_sin(theta)
    return 0.5 * (p_theta ** 2 + (p_phi ** 2) / (s ** 2)) + G * (1.0 - np.cos(theta))


def vector_field(_t, y):
    theta, phi, p_theta, p_phi = y
    s = safe_sin(theta)
    c = np.cos(theta)
    dtheta = p_theta
    dphi = p_phi / (s ** 2)
    dp_theta = (p_phi ** 2) * c / (s ** 3) - G * s
    dp_phi = 0.0
    return np.array([dtheta, dphi, dp_theta, dp_phi], dtype=np.float64)


def sample_initial_condition(rng: np.random.Generator):
    while True:
        theta = rng.uniform(THETA_MIN, THETA_MAX)
        phi = rng.uniform(-np.pi, np.pi)
        p_theta = rng.uniform(-PTHETA_MAX, PTHETA_MAX)
        p_phi = rng.uniform(PPHI_MIN, PPHI_MAX)
        state = np.array([theta, phi, p_theta, p_phi], dtype=np.float64)
        energy = get_hamiltonian(state)
        if ENERGY_MIN <= energy <= ENERGY_MAX:
            return state


def integrate_trajectory(y0, steps, dt, *, atol=1e-12, rtol=1e-12):
    t_eval = np.linspace(0.0, steps * dt, steps + 1, dtype=np.float64)
    sol = solve_ivp(
        vector_field,
        (0.0, steps * dt),
        y0,
        t_eval=t_eval,
        method="DOP853",
        atol=atol,
        rtol=rtol,
    )
    if sol.status != 0 or not np.all(np.isfinite(sol.y)):
        raise RuntimeError(f"Reference integration failed for y0={y0.tolist()}")
    traj = sol.y.T
    if np.any((traj[:, 0] <= 1e-3) | (traj[:, 0] >= np.pi - 1e-3)):
        raise RuntimeError("Trajectory approached a polar singularity too closely.")
    return traj


def generate_split(num_trajs, steps, dt, rng):
    trajs = []
    energies = []
    pphi_drifts = []
    while len(trajs) < num_trajs:
        y0 = sample_initial_condition(rng)
        try:
            traj = integrate_trajectory(y0, steps, dt)
        except RuntimeError:
            continue
        trajs.append(traj)
        energies.append(float(np.max(np.abs((get_hamiltonian(traj) - get_hamiltonian(y0)) / (get_hamiltonian(y0) + 1e-12)))))
        pphi_series = traj[:, 3]
        pphi_drifts.append(float(np.max(np.abs((pphi_series - pphi_series[0]) / (abs(pphi_series[0]) + 1e-12)))))
        if len(trajs) % max(1, num_trajs // 8) == 0:
            print(
                f"  generated {len(trajs)}/{num_trajs} "
                f"(max sampled energy drift {max(energies):.2e}, max p_phi drift {max(pphi_drifts):.2e})"
            )
    array = np.stack(trajs, axis=0)
    meta = {
        "num_trajs": int(num_trajs),
        "steps": int(steps),
        "dt": float(dt),
        "reference_solver": "DOP853",
        "atol": 1e-12,
        "rtol": 1e-12,
        "theta_range": [THETA_MIN, THETA_MAX],
        "p_theta_range": [-PTHETA_MAX, PTHETA_MAX],
        "p_phi_range": [PPHI_MIN, PPHI_MAX],
        "energy_range": [ENERGY_MIN, ENERGY_MAX],
        "max_sampled_energy_drift": float(max(energies) if energies else 0.0),
        "max_sampled_p_phi_drift": float(max(pphi_drifts) if pphi_drifts else 0.0),
    }
    return torch.tensor(array, dtype=torch.float64), meta


def generate_half_step_test(save_dir: Path):
    test_path = save_dir / "test.pt"
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    test = torch.load(test_path, map_location="cpu", weights_only=False).numpy()
    trajs = []
    energies = []
    pphi_drifts = []
    for idx, traj in enumerate(test):
        refined = integrate_trajectory(traj[0], TEST_HALF_STEPS, DT / 2.0)
        trajs.append(refined)
        e0 = get_hamiltonian(refined[0])
        energies.append(float(np.max(np.abs((get_hamiltonian(refined) - e0) / (abs(e0) + 1e-12)))))
        pphi_series = refined[:, 3]
        pphi_drifts.append(float(np.max(np.abs((pphi_series - pphi_series[0]) / (abs(pphi_series[0]) + 1e-12)))))
        print(f"  half-step test {idx + 1}/{len(test)}")
    tensor = torch.tensor(np.stack(trajs, axis=0), dtype=torch.float64)
    torch.save(tensor, save_dir / "test_half.pt")
    half_meta = {
        "num_trajs": int(len(test)),
        "steps": int(TEST_HALF_STEPS),
        "dt": float(DT / 2.0),
        "reference_solver": "DOP853",
        "atol": 1e-12,
        "rtol": 1e-12,
        "max_sampled_energy_drift": float(max(energies) if energies else 0.0),
        "max_sampled_p_phi_drift": float(max(pphi_drifts) if pphi_drifts else 0.0),
        "source_split": "test.pt initial conditions",
    }
    meta_path = save_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"system": "spherical_pendulum", "seed": SEED, "dt": DT}
    meta["test_half"] = half_meta
    meta_path.write_text(json.dumps(meta, indent=2))
    return half_meta


def main():
    parser = argparse.ArgumentParser(description="Generate spherical pendulum benchmark trajectories.")
    parser.add_argument("--half-test-only", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    save_dir = Path("data") / "spherical_pendulum"
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.half_test_only:
        print("[Spherical pendulum] generating half-step multi-IC test split ...")
        half_meta = generate_half_step_test(save_dir)
        print(
            "[Spherical pendulum] wrote test_half.pt "
            f"(max energy drift {half_meta['max_sampled_energy_drift']:.2e})"
        )
        return

    print("[Spherical pendulum] generating training split ...")
    train, train_meta = generate_split(1024, TRAIN_STEPS, DT, rng)
    torch.save(train, save_dir / "train.pt")

    print("[Spherical pendulum] generating validation split ...")
    val, val_meta = generate_split(128, VAL_STEPS, DT, rng)
    torch.save(val, save_dir / "val.pt")

    print("[Spherical pendulum] generating multi-IC test split ...")
    test, test_meta = generate_split(20, TEST_STEPS, DT, rng)
    torch.save(test, save_dir / "test.pt")

    print("[Spherical pendulum] generating half-step multi-IC test split ...")
    half_meta = generate_half_step_test(save_dir)

    print("[Spherical pendulum] generating representative long-rollout split ...")
    test_long, test_long_meta = generate_split(10, TEST_LONG_STEPS, DT, rng)
    torch.save(test_long, save_dir / "test_long.pt")

    meta = {
        "system": "spherical_pendulum",
        "seed": SEED,
        "dt": DT,
        "train": train_meta,
        "val": val_meta,
        "test": test_meta,
        "test_half": half_meta,
        "test_long": test_long_meta,
    }
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print("[Spherical pendulum] wrote:")
    for name in ["train.pt", "val.pt", "test.pt", "test_long.pt", "meta.json"]:
        print(f"  - {save_dir / name}")


if __name__ == "__main__":
    main()
