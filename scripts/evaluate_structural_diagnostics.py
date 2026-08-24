from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import make_J  # noqa: E402
from train import DTYPE, SYSTEM_DIM, SYSTEM_DT, build_model  # noqa: E402


ACTIVE_SYSTEMS = (
    "fput",
    "toda",
    "phi4",
    "double_pendulum",
    "spherical_pendulum",
)


def load_tensor(system: str, split: str) -> torch.Tensor:
    path = REPO / "data" / system / f"{split}.pt"
    if not path.is_file() and split == "test":
        candidates = (
            REPO / "data" / system / "test_long.pt",
            REPO / "data" / system / "test_recurrence.pt",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.ndim == 2:
        data = data.unsqueeze(0)
    return data.to(dtype=DTYPE)


def finite_difference_pairs(data: torch.Tensor, dt: float, max_samples: int):
    states = data[:, :-1].reshape(-1, data.shape[-1])
    velocities = ((data[:, 1:] - data[:, :-1]) / dt).reshape(-1, data.shape[-1])
    if len(states) > max_samples:
        indices = torch.linspace(0, len(states) - 1, max_samples).round().long()
        states = states.index_select(0, indices)
        velocities = velocities.index_select(0, indices)
    return states.numpy(), velocities.numpy()


def projected_linear_fit_score(
    system: str, ridge: float, max_train_samples: int, max_test_samples: int
):
    dt = float(SYSTEM_DT[system])
    train_x, train_y = finite_difference_pairs(
        load_tensor(system, "train"), dt, max_train_samples
    )
    test_x, test_y = finite_difference_pairs(
        load_tensor(system, "test"), dt, max_test_samples
    )
    train_x_t = torch.from_numpy(train_x).to(dtype=DTYPE)
    train_y_t = torch.from_numpy(train_y).to(dtype=DTYPE)
    test_x_t = torch.from_numpy(test_x).to(dtype=DTYPE)
    test_y_t = torch.from_numpy(test_y).to(dtype=DTYPE)
    dim = train_x.shape[1]
    gram = train_x_t.T @ train_x_t + ridge * torch.eye(dim, dtype=DTYPE)
    fitted_a = torch.linalg.solve(gram, train_x_t.T @ train_y_t).T
    j = make_J(dim // 2, dtype=DTYPE)
    fitted_m = -j @ fitted_a
    symmetric_m = 0.5 * (fitted_m + fitted_m.T)
    projected_a = j @ symmetric_m
    prediction = test_x_t @ projected_a.T
    residual = torch.sum((test_y_t - prediction) ** 2)
    centered = torch.sum((test_y_t - test_y_t.mean(dim=0, keepdim=True)) ** 2)
    return {
        "r_linear_squared": float(1.0 - residual / centered),
        "ridge": float(ridge),
        "train_samples": int(len(train_x)),
        "test_samples": int(len(test_x)),
        "difference": "forward",
        "projection": "A -> J sym(-J A)",
    }


def checkpoint_path(system: str, seed: int) -> Path:
    return (
        REPO
        / "checkpoints"
        / system
        / "gscd"
        / "multiseed"
        / f"seed_{seed}"
        / "best.pt"
    )


def cayley_diagnostics(system: str, seeds):
    rows = []
    dim = SYSTEM_DIM[system]
    dt = float(SYSTEM_DT[system])
    for seed in seeds:
        path = checkpoint_path(system, seed)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        args = checkpoint.get("args", {})
        model = build_model(
            "gscd",
            dim,
            hidden=int(args.get("hidden", 128)),
            sparsity=str(args.get("sparsity", "dense")),
            separable=not bool(args.get("non_separable", False)),
        )
        model.load_state_dict(checkpoint["model_state"])
        with torch.no_grad():
            matrix = model.linear_part.M
            cayley = model.linear_part.cayley_map(dt)
            rows.append(
                {
                    "seed": int(seed),
                    "condition_M": float(torch.linalg.cond(matrix)),
                    "spectral_radius_C": float(
                        torch.max(torch.abs(torch.linalg.eigvals(cayley)))
                    ),
                }
            )
    condition = np.asarray([row["condition_M"] for row in rows])
    radius = np.asarray([row["spectral_radius_C"] for row in rows])
    return {
        "rows": rows,
        "condition_M_mean": float(condition.mean()),
        "condition_M_std": float(condition.std(ddof=1)),
        "spectral_radius_C_mean": float(radius.mean()),
        "spectral_radius_C_std": float(radius.std(ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=list(ACTIVE_SYSTEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--max-train-samples", type=int, default=100_000)
    parser.add_argument("--max-test-samples", type=int, default=100_000)
    parser.add_argument(
        "--output",
        default=str(REPO / "results" / "structural_diagnostics.json"),
    )
    args = parser.parse_args()

    payload = {
        "linear_fit": {
            system: projected_linear_fit_score(
                system,
                args.ridge,
                args.max_train_samples,
                args.max_test_samples,
            )
            for system in args.systems
        },
        "cayley": {
            system: cayley_diagnostics(system, args.seeds)
            for system in args.systems
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
