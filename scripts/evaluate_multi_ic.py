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


REPO = Path(__file__).parent.parent
BENCH_PATH = REPO / "scripts" / "run_final_benchmarks.py"
RESULTS = REPO / "results"
FIG_DIR = RESULTS / "figures" / "multi_ic"
DTYPE = torch.float64

MODEL_LABELS = {
    "gscd": "GSC-HNN",
    "hnn": "HNN",
    "hnn_implicit": "HNN-Implicit",
    "scd": "SCD",
    "node": "NODE",
    "srnn": "SRNN-lite",
    "gsympnet": "G-SympNet",
}
SYSTEM_LABELS = {
    "henon_heiles": "Henon-Heiles",
    "double_pendulum": "Double pendulum",
    "harmonic": "Harmonic",
    "fput": "FPUT",
    "phi4": "$\\phi^4$ lattice",
    "toda": "Toda lattice",
    "kepler": "Kepler orbit",
    "charged_particle": "Charged particle",
}
COLORS = {
    "gscd": "#0072B2",
    "hnn": "#009E73",
    "hnn_implicit": "#CC79A7",
    "scd": "#A6761D",
    "node": "#D55E00",
    "srnn": "#7F3C8D",
    "gsympnet": "#6A3D9A",
}


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("run_final_benchmarks", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def checkpoint_path(system, model, seed=None):
    if seed is None:
        return REPO / "checkpoints" / system / model / "best.pt"
    return REPO / "checkpoints" / system / model / "multiseed" / f"seed_{seed}" / "best.pt"


def load_test_trajectories(system, override_path=None):
    if override_path is not None:
        path = Path(override_path)
        if not path.is_absolute():
            path = REPO / path
        if not path.exists():
            raise FileNotFoundError(f"Explicit test path not found: {path}")
    else:
        data_dir = REPO / "data" / system
        test_path = data_dir / "test.pt"
        long_path = data_dir / "test_long.pt"
        if test_path.exists():
            path = test_path
        elif long_path.exists():
            path = long_path
        else:
            raise FileNotFoundError(f"No test.pt or test_long.pt found for {system}")
    data = torch.load(path, map_location="cpu", weights_only=False).to(dtype=DTYPE)
    if data.ndim == 2:
        data = data.unsqueeze(0)
    if data.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D trajectory tensor in {path}, got {tuple(data.shape)}")
    return path, data


def build_model_from_checkpoint(bench, system, ckpt_path, device, test_path=None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model_name = args.get(
        "model",
        ckpt_path.parent.parent.parent.name
        if ckpt_path.parent.name.startswith("seed_")
        else ckpt_path.parent.name,
    )
    hidden = args.get("hidden", 128)

    _, test_data = load_test_trajectories(system, override_path=test_path)
    dim = test_data.shape[-1]
    model = bench.build_model(
        model_name,
        dim,
        hidden=hidden,
        sparsity=args.get("sparsity", "dense"),
        separable=bench.infer_separable(ckpt, dim),
        implicit_max_iters=int(args.get("implicit_max_iters") or 5),
        implicit_tol=float(args.get("implicit_tol") or 1e-6),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt, model_name


def evaluate_one_trajectory(bench, system, model, traj, device, max_steps=None):
    if max_steps is not None:
        traj = traj[: max_steps + 1]
    dt = bench.SYSTEM_DT[system]
    gt_np = traj.cpu().numpy()
    u = traj[0].unsqueeze(0).to(device)
    if hasattr(model, "prepare_initial_state"):
        u = model.prepare_initial_state(u)
    e0 = bench.energy_fn(system, u.squeeze(0).detach().cpu().numpy())
    max_drift = 0.0
    se_sum = 0.0
    count = 0
    start = time.time()
    with torch.no_grad():
        for idx in range(len(traj)):
            pred_np = u.squeeze(0).detach().cpu().numpy()
            diff = bench.state_difference(system, pred_np, gt_np[idx])
            se_sum += float(np.sum(diff**2))
            count += diff.size
            drift = abs((bench.energy_fn(system, pred_np) - e0) / (e0 + 1e-12))
            max_drift = max(max_drift, drift)
            if idx < len(traj) - 1:
                u = model.step(u, dt)
    return {
        "steps": int(len(traj) - 1),
        "mse": se_sum / max(count, 1),
        "max_energy_drift": max_drift,
        "eval_seconds": time.time() - start,
    }


def summarise(raw_rows):
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[(row["system"], row["model"])].append(row)

    summary = []
    for (system, model), rows in sorted(grouped.items()):
        by_seed = defaultdict(list)
        for row in rows:
            by_seed[row["seed"]].append(row)

        seed_mse = []
        seed_drift = []
        seed_time = []
        for seed_rows in by_seed.values():
            seed_mse.append(np.mean([row["mse"] for row in seed_rows]))
            seed_drift.append(np.mean([row["max_energy_drift"] for row in seed_rows]))
            seed_time.append(np.sum([row["eval_seconds"] for row in seed_rows]))

        all_mse = np.array([row["mse"] for row in rows], dtype=float)
        all_drift = np.array([row["max_energy_drift"] for row in rows], dtype=float)
        seed_mse = np.array(seed_mse, dtype=float)
        seed_drift = np.array(seed_drift, dtype=float)
        seed_time = np.array(seed_time, dtype=float)
        summary.append({
            "system": system,
            "model": model,
            "num_seeds": len(by_seed),
            "seeds": sorted(int(seed) for seed in by_seed),
            "num_ic_total": len(rows),
            "ic_per_seed": int(len(rows) / max(len(by_seed), 1)),
            "steps": int(rows[0]["steps"]),
            "mse_seed_mean": float(seed_mse.mean()),
            "mse_seed_std": float(seed_mse.std(ddof=1)),
            "drift_seed_mean": float(seed_drift.mean()),
            "drift_seed_std": float(seed_drift.std(ddof=1)),
            "mse_all_ic_mean": float(all_mse.mean()),
            "mse_all_ic_std": float(all_mse.std(ddof=1)),
            "drift_all_ic_mean": float(all_drift.mean()),
            "drift_all_ic_std": float(all_drift.std(ddof=1)),
            "eval_seconds_seed_mean": float(seed_time.mean()),
            "eval_seconds_seed_std": float(seed_time.std(ddof=1)),
        })
    return summary


def format_pm(mean, std):
    return f"{mean:.4e} +/- {std:.2e}"


def write_outputs(raw_rows, summary_rows, output_stem, argv):
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / f"{output_stem}.json"
    out_md = RESULTS / f"{output_stem}.md"
    payload = {
        "command": argv,
        "state_error_definition": {
            "periodic_difference_interval": "[-pi, pi)",
            "periodic_indices": {
                "double_pendulum": [0, 1],
                "spherical_pendulum": [1],
            },
        },
        "raw": raw_rows,
        "summary": summary_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Multi-Initial-Condition Evaluation",
        "",
        "Each row aggregates all stored test initial conditions for each seed-isolated checkpoint.",
        "The seed-level columns average over initial conditions first, then report mean +/- std across training seeds.",
        "",
        "| System | Model | Seeds | ICs/seed | Steps | MSE seed mean +/- std | Drift seed mean +/- std | MSE all IC mean +/- std | Drift all IC mean +/- std | Eval time/seed (s) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['system']} | {row['model']} | {row['num_seeds']} | {row['ic_per_seed']} | {row['steps']} | "
            f"{format_pm(row['mse_seed_mean'], row['mse_seed_std'])} | "
            f"{format_pm(row['drift_seed_mean'], row['drift_seed_std'])} | "
            f"{format_pm(row['mse_all_ic_mean'], row['mse_all_ic_std'])} | "
            f"{format_pm(row['drift_all_ic_mean'], row['drift_all_ic_std'])} | "
            f"{format_pm(row['eval_seconds_seed_mean'], row['eval_seconds_seed_std'])} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    return out_json, out_md


def plot_summary(summary_rows, systems, models, output_stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    row_map = {(row["system"], row["model"]): row for row in summary_rows}
    x = np.arange(len(systems))
    width = 0.16
    offsets = {
        "gscd": -2.0 * width,
        "hnn": -1.0 * width,
        "hnn_implicit": 0.0,
        "scd": 1.0 * width,
        "node": 2.0 * width,
        "srnn": 3.0 * width,
        "gsympnet": 4.0 * width,
    }

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    for ax, mean_key, std_key, ylabel in [
        (axes[0], "mse_seed_mean", "mse_seed_std", "Rollout MSE"),
        (axes[1], "drift_seed_mean", "drift_seed_std", "Max relative energy drift"),
    ]:
        for model in models:
            xs, means, errors = [], [], []
            for i, system in enumerate(systems):
                row = row_map.get((system, model))
                if row is None:
                    continue
                xs.append(x[i] + offsets.get(model, 0.0))
                means.append(row[mean_key])
                errors.append(row[std_key])
            ax.bar(
                xs,
                means,
                width=width,
                yerr=errors,
                capsize=3,
                label=MODEL_LABELS.get(model, model.upper()),
                color=COLORS.get(model, "gray"),
                edgecolor="black",
                linewidth=0.4,
                alpha=0.92,
            )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([SYSTEM_LABELS.get(system, system) for system in systems])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.35)
    axes[0].legend(loc="upper left", frameon=True, edgecolor="black")
    fig.suptitle("Multi-initial-condition evaluation", y=1.03, fontsize=12)

    outputs = []
    for suffix in ["pdf", "png"]:
        out = FIG_DIR / f"{output_stem}.{suffix}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        outputs.append(out)
    plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=["double_pendulum", "spherical_pendulum"])
    parser.add_argument("--models", nargs="+", default=["gscd", "hnn", "hnn_implicit", "node", "gsympnet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_ics", type=int, default=None)
    parser.add_argument("--output_stem", default="multi_ic_summary")
    parser.add_argument("--test_path", default=None,
                        help="optional explicit test tensor path; overrides data/<system>/test.pt")
    args = parser.parse_args()

    device = torch.device(args.device)
    bench = load_benchmark_module()
    raw_rows = []
    for system in args.systems:
        test_path, test_data = load_test_trajectories(system, override_path=args.test_path)
        if args.max_ics is not None:
            test_data = test_data[: args.max_ics]
        print(f"{system}: evaluating {len(test_data)} initial conditions from {test_path}")
        for model_name in args.models:
            for seed in args.seeds:
                ckpt = checkpoint_path(system, model_name, seed)
                if not ckpt.exists():
                    print(f"Skipping missing checkpoint: {ckpt}")
                    continue
                model, ckpt_obj, resolved_model = build_model_from_checkpoint(
                    bench, system, ckpt, device, test_path=args.test_path
                )
                for ic_idx, traj in enumerate(test_data):
                    metric = evaluate_one_trajectory(
                        bench,
                        system,
                        model,
                        traj,
                        device,
                        max_steps=args.max_steps,
                    )
                    metric.update({
                        "system": system,
                        "model": resolved_model,
                        "seed": seed,
                        "ic_index": ic_idx,
                        "checkpoint": str(ckpt),
                        "best_val_loss": ckpt_obj.get("val_loss"),
                    })
                    raw_rows.append(metric)
                    print(
                        f"  {system}/{resolved_model}/seed_{seed}/ic_{ic_idx}: "
                        f"mse={metric['mse']:.4e}, "
                        f"drift={metric['max_energy_drift']:.4e}, "
                        f"time={metric['eval_seconds']:.2f}s"
                    )

    summary = summarise(raw_rows)
    out_json, out_md = write_outputs(raw_rows, summary, args.output_stem, sys.argv)
    fig_paths = plot_summary(summary, args.systems, args.models, args.output_stem)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    for fig_path in fig_paths:
        print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
