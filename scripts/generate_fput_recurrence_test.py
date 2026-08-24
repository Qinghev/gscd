import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from generate_fput import DT, N_PARTICLES, SUBSTEPS, integrate


OUT_DIR = REPO / "data" / "fput"


def sine_mode(k: int) -> np.ndarray:
    idx = np.arange(1, N_PARTICLES + 1, dtype=np.float64)
    return np.sin(np.pi * k * idx / (N_PARTICLES + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260411)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    amplitudes = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    mode1 = sine_mode(1)
    mode2 = sine_mode(2)
    families = []
    trajectories = []

    for amp in amplitudes:
        q0 = amp * mode1
        p0 = np.zeros(N_PARTICLES, dtype=np.float64)
        trajectories.append(integrate(q0, p0, args.steps, SUBSTEPS))
        families.append({
            "family": "pure_mode1",
            "amplitude": amp,
            "momentum_std": 0.0,
        })
        print(f"generated pure mode-1 amplitude={amp:.2f}")

    for amp in amplitudes:
        q0 = amp * mode1 + 0.25 * amp * mode2
        p0 = rng.normal(0.0, 0.005, N_PARTICLES).astype(np.float64)
        trajectories.append(integrate(q0, p0, args.steps, SUBSTEPS))
        families.append({
            "family": "mode1_mode2_mixed",
            "amplitude": amp,
            "mode2_ratio": 0.25,
            "momentum_std": 0.005,
        })
        print(f"generated mixed mode-1/mode-2 amplitude={amp:.2f}")

    data = np.stack(trajectories, axis=0)
    out_path = OUT_DIR / "test_recurrence.pt"
    meta_path = OUT_DIR / "test_recurrence_meta.json"

    torch.save(torch.tensor(data, dtype=torch.float64), out_path)
    meta = {
        "system": "FPUT_beta",
        "protocol": "recurrence_aware_multi_ic",
        "num_ics": int(data.shape[0]),
        "steps": args.steps,
        "shape": list(data.shape),
        "dt": DT,
        "substeps": int(SUBSTEPS),
        "seed": args.seed,
        "dtype": "float64",
        "families": families,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
