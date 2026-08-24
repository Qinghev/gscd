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


REPO = Path(__file__).resolve().parent.parent
BENCH_PATH = REPO / "scripts" / "run_final_benchmarks.py"
RESULTS = REPO / "results"
FIG_DIR = RESULTS / "figures" / "fput"
DTYPE = torch.float64

MODEL_LABELS = {"gscd": "GSC-HNN", "hnn": "HNN", "node": "NODE", "gsympnet": "G-SympNet"}
COLORS = {"gscd": "#0072B2", "hnn": "#009E73", "node": "#D55E00", "gsympnet": "#6A3D9A"}


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("run_final_benchmarks", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def checkpoint_path(model, seed):
    return REPO / "checkpoints" / "fput" / model / "multiseed" / f"seed_{seed}" / "best.pt"


def load_test_trajectories(path_str=None):
    if path_str is None:
        path = REPO / "data" / "fput" / "test_recurrence.pt"
    else:
        path = Path(path_str)
        if not path.is_absolute():
            path = REPO / path
    data = torch.load(path, map_location="cpu", weights_only=False).to(dtype=DTYPE)
    if data.ndim == 2:
        data = data.unsqueeze(0)
    return path, data


def build_model(bench, ckpt_path, dim, device):
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
    return model, ckpt, model_name


def modal_basis(n_particles: int):
    j = np.arange(1, n_particles + 1, dtype=np.float64)[:, None]
    k = np.arange(1, n_particles + 1, dtype=np.float64)[None, :]
    basis = np.sqrt(2.0 / (n_particles + 1.0)) * np.sin(np.pi * j * k / (n_particles + 1.0))
    omega = 2.0 * np.sin(np.pi * np.arange(1, n_particles + 1, dtype=np.float64) / (2.0 * (n_particles + 1.0)))
    return basis, omega


def modal_energies(traj):
    n_particles = traj.shape[-1] // 2
    basis, omega = modal_basis(n_particles)
    q = traj[:, :n_particles]
    p = traj[:, n_particles:]
    q_modes = q @ basis
    p_modes = p @ basis
    return 0.5 * (p_modes ** 2 + (omega[None, :] ** 2) * (q_modes ** 2))


def first_recurrence_time(mode1_energy, dt):
    start = max(10, int(0.05 * len(mode1_energy)))
    target = 0.9 * float(mode1_energy[0])
    peaks = []
    for idx in range(start + 1, len(mode1_energy) - 1):
        if mode1_energy[idx] >= mode1_energy[idx - 1] and mode1_energy[idx] >= mode1_energy[idx + 1]:
            peaks.append(idx)
    for idx in peaks:
        if mode1_energy[idx] >= target:
            return idx * dt
    fallback = start + int(np.argmax(mode1_energy[start:]))
    return fallback * dt


def evaluate_trajectory(bench, model, traj, device):
    dt = bench.SYSTEM_DT["fput"]
    gt_np = traj.cpu().numpy()
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
    pred_np = np.stack(pred, axis=0)
    elapsed = time.time() - start
    pred_energy = np.array([bench.energy_fn("fput", u_state) for u_state in pred_np])
    e0 = pred_energy[0]

    gt_modes = modal_energies(gt_np)
    pred_modes = modal_energies(pred_np)
    gt_mode1 = gt_modes[:, 0]
    pred_mode1 = pred_modes[:, 0]
    gt_spec = gt_modes[:, :4].mean(axis=0)
    pred_spec = pred_modes[:, :4].mean(axis=0)
    gt_spec /= max(gt_spec.sum(), 1e-12)
    pred_spec /= max(pred_spec.sum(), 1e-12)
    gt_trec = first_recurrence_time(gt_mode1, dt)
    pred_trec = first_recurrence_time(pred_mode1, dt)

    return {
        "steps": int(len(traj) - 1),
        "mse": float(np.mean((pred_np - gt_np) ** 2)),
        "max_energy_drift": float(np.max(np.abs((pred_energy - e0) / (e0 + 1e-12)))),
        "mode1_energy_rmse": float(np.sqrt(np.mean((pred_mode1 - gt_mode1) ** 2))),
        "first_recurrence_rel_error": float(abs(pred_trec - gt_trec) / max(abs(gt_trec), dt)),
        "modal_spectrum_l1": float(np.sum(np.abs(pred_spec - gt_spec))),
        "eval_seconds": elapsed,
    }


def summarise(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["system"], row["model"])].append(row)

    fields = [
        "mse",
        "max_energy_drift",
        "mode1_energy_rmse",
        "first_recurrence_rel_error",
        "modal_spectrum_l1",
        "eval_seconds",
    ]
    summary = []
    for (system, model), items in sorted(grouped.items()):
        by_seed = defaultdict(list)
        for row in items:
            by_seed[row["seed"]].append(row)
        seed_stats = {field: [] for field in fields}
        for seed_rows in by_seed.values():
            for field in fields:
                values = [row[field] for row in seed_rows]
                seed_stats[field].append(np.mean(values) if field != "eval_seconds" else np.sum(values))
        row = {
            "system": system,
            "model": model,
            "num_seeds": len(by_seed),
            "ic_per_seed": int(len(items) / max(len(by_seed), 1)),
            "steps": int(items[0]["steps"]),
        }
        for field in fields:
            arr = np.asarray(seed_stats[field], dtype=float)
            row[f"{field}_seed_mean"] = float(arr.mean())
            row[f"{field}_seed_std"] = float(arr.std(ddof=1))
        summary.append(row)
    return summary


def write_outputs(raw_rows, summary_rows, output_stem, argv):
    out_json = RESULTS / f"{output_stem}.json"
    out_md = RESULTS / f"{output_stem}.md"
    payload = {"command": argv, "raw": raw_rows, "summary": summary_rows}
    out_json.write_text(json.dumps(payload, indent=2))
    lines = [
        "# FPUT Recurrence-Aware Evaluation",
        "",
        "| Model | Seeds | ICs/seed | Steps | Rollout MSE | Energy Drift | Mode-1 RMSE | First Recurrence RelErr | Modal Spectrum L1 | Eval time / seed (s) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['num_seeds']} | {row['ic_per_seed']} | {row['steps']} | "
            f"{row['mse_seed_mean']:.4e} +/- {row['mse_seed_std']:.2e} | "
            f"{row['max_energy_drift_seed_mean']:.4e} +/- {row['max_energy_drift_seed_std']:.2e} | "
            f"{row['mode1_energy_rmse_seed_mean']:.4e} +/- {row['mode1_energy_rmse_seed_std']:.2e} | "
            f"{row['first_recurrence_rel_error_seed_mean']:.4e} +/- {row['first_recurrence_rel_error_seed_std']:.2e} | "
            f"{row['modal_spectrum_l1_seed_mean']:.4e} +/- {row['modal_spectrum_l1_seed_std']:.2e} | "
            f"{row['eval_seconds_seed_mean']:.2f} +/- {row['eval_seconds_seed_std']:.2f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    return out_json, out_md


def plot_summary(summary_rows, output_stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("mse_seed_mean", "mse_seed_std", "Rollout MSE"),
        ("max_energy_drift_seed_mean", "max_energy_drift_seed_std", "Energy drift"),
        ("mode1_energy_rmse_seed_mean", "mode1_energy_rmse_seed_std", "Mode-1 energy RMSE"),
        ("first_recurrence_rel_error_seed_mean", "first_recurrence_rel_error_seed_std", "Recurrence-time rel. error"),
    ]
    order = ["gscd", "hnn", "node", "gsympnet"]
    row_map = {row["model"]: row for row in summary_rows}

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    for ax, (mean_key, std_key, title) in zip(axes.ravel(), metrics):
        labels = [MODEL_LABELS[m] for m in order if m in row_map]
        colors = [COLORS[m] for m in order if m in row_map]
        means = [row_map[m][mean_key] for m in order if m in row_map]
        errs = [row_map[m][std_key] for m in order if m in row_map]
        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=errs, capsize=3, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=10)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.35)
    for suffix in ["pdf", "png"]:
        out = FIG_DIR / f"{output_stem}.{suffix}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gscd", "node", "hnn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--test_path", default="data/fput/test_recurrence.pt")
    parser.add_argument("--output_stem", default="multi_ic_fput_recurrence")
    args = parser.parse_args()

    device = torch.device(args.device)
    bench = load_benchmark_module()
    _, trajectories = load_test_trajectories(args.test_path)

    raw_rows = []
    for model_name in args.models:
        for seed in args.seeds:
            ckpt = checkpoint_path(model_name, seed)
            if not ckpt.exists():
                print(f"Skipping missing checkpoint: {ckpt}")
                continue
            model, ckpt_obj, resolved_model = build_model(bench, ckpt, trajectories.shape[-1], device)
            for ic_idx, traj in enumerate(trajectories):
                metric = evaluate_trajectory(bench, model, traj, device)
                metric.update({
                    "system": "fput",
                    "model": resolved_model,
                    "seed": seed,
                    "ic_index": ic_idx,
                    "checkpoint": str(ckpt),
                    "best_val_loss": ckpt_obj.get("val_loss"),
                })
                raw_rows.append(metric)
                print(
                    f"fput/{resolved_model}/seed_{seed}/ic_{ic_idx}: "
                    f"mse={metric['mse']:.4e}, drift={metric['max_energy_drift']:.4e}, "
                    f"mode1={metric['mode1_energy_rmse']:.4e}, trec={metric['first_recurrence_rel_error']:.4e}"
                )

    summary = summarise(raw_rows)
    out_json, out_md = write_outputs(raw_rows, summary, args.output_stem, sys.argv)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    plot_summary(summary, args.output_stem)


if __name__ == "__main__":
    main()
