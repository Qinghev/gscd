from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_double_pendulum_shadowing import (
    COLORS,
    MODEL_LABELS,
    build_model_from_checkpoint,
    checkpoint_path,
    load_benchmark_module,
    load_test_trajectories,
    rollout,
    wrapped_state_error,
)


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
PAPER_FIGURES = REPO / "paper" / "figures"

MODEL_ORDER = ["gscd", "hnn", "hnn_implicit", "node", "gsympnet"]


def trajectory_curves(gt: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    diff = wrapped_state_error(pred, gt)
    scale = np.sqrt(np.mean(np.sum(gt**2, axis=1))) + 1e-12
    rel_err = np.linalg.norm(diff, axis=1) / scale
    angle_sq = np.mean(diff[:, :2] ** 2, axis=1)
    return {
        "angle_sq": angle_sq,
        "relative_error": rel_err,
        "shadow_survival_tau050": (rel_err <= 0.50).astype(np.float64),
    }


def evaluate(args: argparse.Namespace) -> dict:
    bench = load_benchmark_module()
    device = torch.device(args.device)
    test_data = load_test_trajectories()
    if args.max_ics is not None:
        test_data = test_data[: args.max_ics]
    if args.max_steps is not None:
        test_data = test_data[:, : args.max_steps + 1]
    dt = bench.SYSTEM_DT["double_pendulum"]
    time_grid = (np.arange(test_data.shape[1], dtype=np.float64) * dt).tolist()

    raw_seed: list[dict] = []
    summary: dict[str, dict] = {}
    for model_name in args.models:
        seed_rows = []
        for seed in args.seeds:
            ckpt_path = checkpoint_path(model_name, seed)
            if not ckpt_path.exists():
                print(f"Skipping missing checkpoint {ckpt_path}", flush=True)
                continue
            print(f"Evaluating double_pendulum/{model_name}/seed_{seed}", flush=True)
            model = build_model_from_checkpoint(bench, ckpt_path, device, test_data.shape[-1])
            seed_angle_sq = []
            seed_rel = []
            seed_survival = []
            elapsed = 0.0
            for ic_idx in range(len(test_data)):
                pred, rollout_seconds = rollout(model, test_data[ic_idx], dt, device)
                elapsed += rollout_seconds
                curves = trajectory_curves(test_data[ic_idx].numpy(), pred)
                seed_angle_sq.append(curves["angle_sq"])
                seed_rel.append(curves["relative_error"])
                seed_survival.append(curves["shadow_survival_tau050"])
            angle_sq = np.asarray(seed_angle_sq, dtype=np.float64)
            rel_err = np.asarray(seed_rel, dtype=np.float64)
            survival = np.asarray(seed_survival, dtype=np.float64)
            row = {
                "model": model_name,
                "seed": int(seed),
                "ic_count": int(test_data.shape[0]),
                "steps": int(test_data.shape[1] - 1),
                "eval_seconds": float(elapsed),
                "curves": {
                    "angle_rmse": np.sqrt(angle_sq.mean(axis=0)).tolist(),
                    "median_relative_error": np.median(rel_err, axis=0).tolist(),
                    "shadow_survival_tau050": survival.mean(axis=0).tolist(),
                },
            }
            raw_seed.append(row)
            seed_rows.append(row)

        if seed_rows:
            curves = {}
            for key in seed_rows[0]["curves"]:
                values = np.asarray([row["curves"][key] for row in seed_rows], dtype=np.float64)
                curves[f"{key}_mean"] = values.mean(axis=0).tolist()
                curves[f"{key}_seed_std"] = values.std(axis=0, ddof=1).tolist()
            angle_values = np.asarray([row["curves"]["angle_rmse"] for row in seed_rows], dtype=np.float64)
            denom = np.arange(1, angle_values.shape[1] + 1, dtype=np.float64)
            cumulative_angle = np.sqrt(np.cumsum(angle_values**2, axis=1) / denom[None, :])
            curves["cumulative_angle_rmse_mean"] = cumulative_angle.mean(axis=0).tolist()
            curves["cumulative_angle_rmse_seed_std"] = cumulative_angle.std(axis=0, ddof=1).tolist()
            summary[model_name] = {
                "seed_count": len(seed_rows),
                "ic_per_seed": int(seed_rows[0]["ic_count"]),
                "steps": int(seed_rows[0]["steps"]),
                "curves": curves,
                "final_angle_rmse": float(curves["angle_rmse_mean"][-1]),
                "final_median_relative_error": float(curves["median_relative_error_mean"][-1]),
                "final_shadow_survival_tau050": float(curves["shadow_survival_tau050_mean"][-1]),
            }

    return {
        "command": sys.argv,
        "system": "double_pendulum",
        "state_error_definition": {
            "periodic_difference_interval": "[-pi, pi)",
            "periodic_indices": [0, 1],
        },
        "models": args.models,
        "seeds": args.seeds,
        "num_ics": int(test_data.shape[0]),
        "steps": int(test_data.shape[1] - 1),
        "dt": float(dt),
        "time": time_grid,
        "raw_seed": raw_seed,
        "summary": summary,
    }


def plot(payload: dict, output_stem: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.35,
        }
    )
    time_grid = np.asarray(payload["time"], dtype=np.float64)
    plot_slice = slice(1, None)
    fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.18), constrained_layout=True)
    panels = [
        ("cumulative_angle_rmse_mean", "Running angle RMSE", "linear", False),
        ("median_relative_error_mean", "Median relative error", "linear", False),
        ("shadow_survival_tau050_mean", r"Fraction below $\tau_{0.50}$ threshold", "linear", True),
    ]
    models = [model for model in MODEL_ORDER if model in payload["summary"]]
    for ax, (key, ylabel, scale, higher_better) in zip(axes, panels):
        for model_name in models:
            values = np.asarray(payload["summary"][model_name]["curves"][key], dtype=np.float64)
            y_values = values[plot_slice]
            if scale == "log":
                y_values = np.maximum(y_values, 1e-16)
            ax.plot(
                time_grid[plot_slice],
                y_values,
                label=MODEL_LABELS.get(model_name, model_name),
                color=COLORS.get(model_name, "0.35"),
            )
        if scale == "log":
            ax.set_yscale("log")
        if higher_better:
            ax.set_ylim(-0.04, 1.04)
        elif scale == "linear":
            _, ymax = ax.get_ylim()
            ax.set_ylim(0.0, ymax)
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", linestyle=":", alpha=0.35)
    axes[0].legend(loc="lower right", frameon=True, edgecolor="black")

    for base in (PAPER_FIGURES / "double_pendulum", RESULTS / "figures" / "double_pendulum"):
        base.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "png"):
            target = base / f"{output_stem}.{suffix}"
            fig.savefig(target, dpi=300, bbox_inches="tight")
            print(f"Wrote {target}", flush=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate double pendulum time resolved shadowing curves.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max_ics", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_stem", default="double_pendulum_time_curves_5seed")
    args = parser.parse_args()

    start = time.time()
    payload = evaluate(args)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / f"{args.output_stem}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}", flush=True)
    plot(payload, args.output_stem)
    print(f"Done in {time.time() - start:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
