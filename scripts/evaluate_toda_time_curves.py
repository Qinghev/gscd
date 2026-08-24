from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_multi_ic import build_model_from_checkpoint, load_benchmark_module


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
PAPER_FIGURES = REPO / "paper" / "figures"
DTYPE = torch.float64

MODEL_LABELS = {
    "gscd": "GSC-HNN",
    "hnn": "HNN",
    "node": "NODE",
    "gsympnet": "G-SympNet",
}

MODEL_COLORS = {
    "gscd": "#0072B2",
    "hnn": "#228B22",
    "node": "#D55E00",
    "gsympnet": "#6A3D9A",
}


def checkpoint_path(system: str, model: str, seed: int) -> Path:
    return REPO / "checkpoints" / system / model / "multiseed" / f"seed_{seed}" / "best.pt"


def load_test_data(system: str, max_ics: int | None, max_steps: int | None) -> torch.Tensor:
    path = REPO / "data" / system / "test.pt"
    data = torch.load(path, map_location="cpu", weights_only=False).to(dtype=DTYPE)
    if data.ndim == 2:
        data = data.unsqueeze(0)
    if max_ics is not None:
        data = data[:max_ics]
    if max_steps is not None:
        data = data[:, : max_steps + 1]
    return data


def relative_energy_errors(bench, system: str, pred_np: np.ndarray, e0: np.ndarray) -> np.ndarray:
    values = np.asarray([bench.energy_fn(system, state) for state in pred_np], dtype=np.float64)
    return np.abs((values - e0) / (e0 + 1e-12))


def rollout_seed_curves(
    bench,
    system: str,
    model_name: str,
    seed: int,
    test_data: torch.Tensor,
    device: torch.device,
) -> dict[str, list[float]]:
    ckpt = checkpoint_path(system, model_name, seed)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    model, _payload, _loaded_name = build_model_from_checkpoint(bench, system, ckpt, device)
    dt = bench.SYSTEM_DT[system]
    gt_np = test_data.cpu().numpy()
    u = test_data[:, 0, :].to(device)
    if hasattr(model, "prepare_initial_state"):
        u = model.prepare_initial_state(u)

    e0 = np.asarray(
        [bench.energy_fn(system, state) for state in u.detach().cpu().numpy()],
        dtype=np.float64,
    )
    mse_curve: list[float] = []
    drift_curve: list[float] = []

    with torch.no_grad():
        for idx in range(test_data.shape[1]):
            pred_np = u.detach().cpu().numpy()
            diff = pred_np - gt_np[:, idx, :]
            mse_curve.append(float(np.mean(diff**2)))
            drift_curve.append(float(np.mean(relative_energy_errors(bench, system, pred_np, e0))))
            if idx < test_data.shape[1] - 1:
                u = model.step(u, dt)

    return {"mse": mse_curve, "relative_energy_error": drift_curve}


def evaluate(args: argparse.Namespace) -> dict:
    bench = load_benchmark_module()
    device = torch.device(args.device)
    test_data = load_test_data(args.system, args.max_ics, args.max_steps)
    time_grid = (np.arange(test_data.shape[1], dtype=np.float64) * bench.SYSTEM_DT[args.system]).tolist()

    output: dict = {
        "system": args.system,
        "models": args.models,
        "seeds": args.seeds,
        "num_ics": int(test_data.shape[0]),
        "steps": int(test_data.shape[1] - 1),
        "dt": float(bench.SYSTEM_DT[args.system]),
        "time": time_grid,
        "curves": {},
    }

    for model_name in args.models:
        seed_curves = []
        start = time.time()
        for seed in args.seeds:
            print(f"Evaluating {args.system}/{model_name}/seed_{seed}", flush=True)
            seed_curves.append(rollout_seed_curves(bench, args.system, model_name, seed, test_data, device))
        mse = np.asarray([row["mse"] for row in seed_curves], dtype=np.float64)
        drift = np.asarray([row["relative_energy_error"] for row in seed_curves], dtype=np.float64)
        output["curves"][model_name] = {
            "seed_count": int(mse.shape[0]),
            "mse_mean": mse.mean(axis=0).tolist(),
            "relative_energy_error_mean": drift.mean(axis=0).tolist(),
            "eval_seconds": float(time.time() - start),
        }
    return output


def plot(payload: dict, output_stem: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.5,
        }
    )
    time_grid = np.asarray(payload["time"], dtype=np.float64)
    plot_slice = slice(1, None)
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.35), constrained_layout=True)

    panels = [
        ("mse_mean", "Mean rollout MSE"),
        ("relative_energy_error_mean", "Mean relative energy error"),
    ]
    for ax, (key, ylabel) in zip(axes, panels):
        for model_name in payload["models"]:
            values = np.asarray(payload["curves"][model_name][key], dtype=np.float64)
            ax.plot(
                time_grid[plot_slice],
                np.maximum(values[plot_slice], 1e-16),
                label=MODEL_LABELS.get(model_name, model_name),
                color=MODEL_COLORS.get(model_name, "0.35"),
            )
        ax.set_yscale("log")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", linestyle=":", alpha=0.35)
    axes[0].legend(loc="lower right", frameon=True, edgecolor="black")

    for base in (PAPER_FIGURES / "toda", RESULTS / "figures" / "toda"):
        base.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "png"):
            target = base / f"{output_stem}.{suffix}"
            fig.savefig(target, dpi=300, bbox_inches="tight")
            print(f"Wrote {target}", flush=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default="toda")
    parser.add_argument("--models", nargs="+", default=["gscd", "hnn", "node", "gsympnet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_ics", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_stem", default="toda_time_curves_2panel_5seed")
    args = parser.parse_args()

    payload = evaluate(args)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / f"{args.output_stem}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}", flush=True)
    plot(payload, args.output_stem)


if __name__ == "__main__":
    main()
