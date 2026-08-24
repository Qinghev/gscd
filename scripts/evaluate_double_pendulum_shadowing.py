import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_final_benchmarks import state_difference


REPO = Path(__file__).resolve().parent.parent
BENCH_PATH = REPO / "scripts" / "run_final_benchmarks.py"
RESULTS = REPO / "results"
FIG_DIR = RESULTS / "figures" / "double_pendulum"
DTYPE = torch.float64
MODEL_LABELS = {
    "gscd": "GSC-HNN",
    "hnn": "HNN",
    "hnn_implicit": "HNN-Implicit",
    "scd": "SCD",
    "node": "NODE",
    "gsympnet": "G-SympNet",
}
COLORS = {
    "gscd": "#0072B2",
    "hnn": "#009E73",
    "hnn_implicit": "#CC79A7",
    "scd": "#A6761D",
    "node": "#D55E00",
    "gsympnet": "#6A3D9A",
}
MODEL_ORDER = ["gscd", "hnn", "hnn_implicit", "scd", "node", "gsympnet"]


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("run_final_benchmarks", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def checkpoint_path(model, seed):
    return REPO / "checkpoints" / "double_pendulum" / model / "multiseed" / f"seed_{seed}" / "best.pt"


def load_test_trajectories():
    data = torch.load(REPO / "data" / "double_pendulum" / "test.pt", map_location="cpu", weights_only=False)
    if data.ndim == 2:
        data = data.unsqueeze(0)
    return data.to(dtype=DTYPE)


def wrapped_state_error(pred: np.ndarray, gt: np.ndarray):
    return state_difference("double_pendulum", pred, gt)


def shadowing_time(rel_err: np.ndarray, threshold: float, dt: float):
    idx = np.flatnonzero(rel_err > threshold)
    if len(idx) == 0:
        return float((len(rel_err) - 1) * dt)
    return float(idx[0] * dt)


def build_model_from_checkpoint(bench, ckpt_path, device, dim):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model_name = args.get("model", ckpt_path.parent.parent.name)
    model = bench.build_model(
        model_name,
        dim,
        hidden=args.get("hidden", 128),
        separable=bench.infer_separable(ckpt, dim),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def rollout(model, traj, dt, device):
    u = traj[0].unsqueeze(0).to(device)
    if hasattr(model, "prepare_initial_state"):
        u = model.prepare_initial_state(u)
    pred = []
    start = time.time()
    with torch.no_grad():
        for idx in range(len(traj)):
            pred.append(u.squeeze(0).detach().cpu().numpy())
            if idx < len(traj) - 1:
                u = model.step(u, dt)
    return np.stack(pred, axis=0), time.time() - start


def evaluate_pair(gt: np.ndarray, pred: np.ndarray, dt: float):
    diff = wrapped_state_error(pred, gt)
    scale = np.sqrt(np.mean(np.sum(gt ** 2, axis=1))) + 1e-12
    rel_err = np.linalg.norm(diff, axis=1) / scale
    angle_rmse = float(np.sqrt(np.mean(diff[:, :2] ** 2)))
    return {
        "mse": float(np.mean(diff ** 2)),
        "angle_rmse": angle_rmse,
        "median_rel_error": float(np.median(rel_err)),
        "shadow_time_tau025": shadowing_time(rel_err, 0.25, dt),
        "shadow_time_tau050": shadowing_time(rel_err, 0.50, dt),
    }


def summarise(raw_rows):
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[row["model"]].append(row)

    fields = ["mse", "angle_rmse", "median_rel_error", "shadow_time_tau025", "shadow_time_tau050"]
    summary = []
    for model, rows in sorted(grouped.items(), key=lambda item: MODEL_ORDER.index(item[0]) if item[0] in MODEL_ORDER else len(MODEL_ORDER)):
        per_seed = defaultdict(list)
        seed_times = defaultdict(float)
        for row in rows:
            per_seed[row["seed"]].append(row)
            seed_times[row["seed"]] = max(seed_times[row["seed"]], row["eval_seconds"])
        item = {
            "system": "double_pendulum",
            "model": model,
            "num_seeds": len(per_seed),
            "ic_per_seed": len(rows) // max(len(per_seed), 1),
            "steps": rows[0]["steps"],
        }
        for field in fields:
            arr = np.asarray([np.mean([r[field] for r in seed_rows]) for seed_rows in per_seed.values()], dtype=float)
            item[f"{field}_seed_mean"] = float(arr.mean())
            item[f"{field}_seed_std"] = float(arr.std(ddof=1))
        eval_arr = np.asarray(list(seed_times.values()), dtype=float)
        item["eval_seconds_seed_mean"] = float(eval_arr.mean())
        item["eval_seconds_seed_std"] = float(eval_arr.std(ddof=1))
        summary.append(item)
    return summary


def write_outputs(raw_rows, summary_rows, argv, output_stem):
    out_json = RESULTS / f"{output_stem}.json"
    out_md = RESULTS / f"{output_stem}.md"
    payload = {
        "command": argv,
        "state_error_definition": {
            "periodic_difference_interval": "[-pi, pi)",
            "periodic_indices": [0, 1],
        },
        "raw": raw_rows,
        "summary": summary_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    lines = [
        "# Double Pendulum Shadowing Metrics",
        "",
        "| Model | Seeds | ICs/seed | Steps | Angle RMSE | Median RelErr | Shadow Time @0.25 | Shadow Time @0.50 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {MODEL_LABELS.get(row['model'], row['model'])} | {row['num_seeds']} | {row['ic_per_seed']} | {row['steps']} | "
            f"{row['angle_rmse_seed_mean']:.4e} +/- {row['angle_rmse_seed_std']:.2e} | "
            f"{row['median_rel_error_seed_mean']:.4e} +/- {row['median_rel_error_seed_std']:.2e} | "
            f"{row['shadow_time_tau025_seed_mean']:.4e} +/- {row['shadow_time_tau025_seed_std']:.2e} | "
            f"{row['shadow_time_tau050_seed_mean']:.4e} +/- {row['shadow_time_tau050_seed_std']:.2e} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    return out_json, out_md


def plot_summary(summary_rows, output_stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    row_map = {row["model"]: row for row in summary_rows}
    metrics = [
        ("angle_rmse_seed_mean", "angle_rmse_seed_std", "Angle RMSE"),
        ("median_rel_error_seed_mean", "median_rel_error_seed_std", "Median relative error"),
        ("shadow_time_tau025_seed_mean", "shadow_time_tau025_seed_std", "Shadow time @ 0.25"),
        ("shadow_time_tau050_seed_mean", "shadow_time_tau050_seed_std", "Shadow time @ 0.50"),
    ]
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    for ax, (mean_key, std_key, title) in zip(axes.ravel(), metrics):
        models = [m for m in MODEL_ORDER if m in row_map]
        xs = np.arange(len(models))
        ax.bar(
            xs,
            [row_map[m][mean_key] for m in models],
            yerr=[row_map[m][std_key] for m in models],
            capsize=3,
            color=[COLORS[m] for m in models],
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=20, ha="right")
        if "shadow" not in title.lower():
            ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    out_pdf = FIG_DIR / f"{output_stem}.pdf"
    out_png = FIG_DIR / f"{output_stem}.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_pdf, out_png


def main():
    parser = argparse.ArgumentParser(description="Evaluate double pendulum shadowing metrics on the multi-IC test set.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+", default=["gscd", "hnn", "hnn_implicit", "scd", "node"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output_stem", default="double_pendulum_shadowing_metrics")
    args = parser.parse_args()

    bench = load_benchmark_module()
    device = torch.device(args.device)
    test_data = load_test_trajectories()
    dt = bench.SYSTEM_DT["double_pendulum"]
    raw_rows = []
    for model_name in args.models:
        for seed in args.seeds:
            ckpt_path = checkpoint_path(model_name, seed)
            if not ckpt_path.exists():
                continue
            model = build_model_from_checkpoint(bench, ckpt_path, device, test_data.shape[-1])
            for ic_idx in range(len(test_data)):
                pred, elapsed = rollout(model, test_data[ic_idx], dt, device)
                metrics = evaluate_pair(test_data[ic_idx].numpy(), pred, dt)
                raw_rows.append(
                    {
                        "system": "double_pendulum",
                        "model": model_name,
                        "seed": seed,
                        "ic_index": ic_idx,
                        "steps": int(test_data.shape[1] - 1),
                        "checkpoint": str(ckpt_path),
                        "eval_seconds": elapsed,
                        **metrics,
                    }
                )

    summary_rows = summarise(raw_rows)
    out_json, out_md = write_outputs(raw_rows, summary_rows, sys.argv, args.output_stem)
    out_pdf, _ = plot_summary(summary_rows, args.output_stem)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
