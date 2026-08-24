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
FIG_DIR = RESULTS / "figures" / "spherical_pendulum"
DTYPE = torch.float64

sys.path.insert(0, str(REPO / "src"))

from generate_spherical_pendulum import get_hamiltonian  # noqa: E402  # pyright: ignore[reportMissingImports]


MODEL_LABELS = {
    "gscd": "GSC-HNN",
    "gscd_sym": "GSC-sym",
    "gscd_exp": "GSC-exp",
    "gscd_nosplit": "GSC-no-split",
    "hnn": "HNN",
    "hnn_implicit": "HNN-Implicit",
    "hnn_symp": "HNN-Symp",
    "sympnet": "SympNet-like",
    "gsympnet": "G-SympNet",
    "node": "NODE",
}
COLORS = {
    "gscd": "#0072B2",
    "gscd_sym": "#56B4E9",
    "gscd_exp": "#009E73",
    "gscd_nosplit": "#CC79A7",
    "hnn": "#1B9E77",
    "hnn_implicit": "#7F3C8D",
    "hnn_symp": "#4C78A8",
    "sympnet": "#A6761D",
    "gsympnet": "#6A3D9A",
    "node": "#D55E00",
}
MODEL_ORDER = [
    "gscd",
    "gscd_sym",
    "gscd_exp",
    "gscd_nosplit",
    "hnn",
    "hnn_implicit",
    "hnn_symp",
    "sympnet",
    "gsympnet",
    "node",
]


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("run_final_benchmarks", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def checkpoint_path(model, seed):
    return REPO / "checkpoints" / "spherical_pendulum" / model / "multiseed" / f"seed_{seed}" / "best.pt"


def load_test_trajectories():
    data = torch.load(REPO / "data" / "spherical_pendulum" / "test.pt", map_location="cpu", weights_only=False)
    if data.ndim == 2:
        data = data.unsqueeze(0)
    return data.to(dtype=DTYPE)


def wrapped_error(pred: np.ndarray, gt: np.ndarray):
    return state_difference("spherical_pendulum", pred, gt)


def max_relative_drift(series):
    s0 = float(series[0])
    denom = max(abs(s0), 1e-12)
    return float(np.max(np.abs((series - s0) / denom)))


def symmetric_cloud_rmse(a: np.ndarray, b: np.ndarray):
    if len(a) == 0 or len(b) == 0:
        return np.nan
    d_ab = np.min(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1), axis=1)
    d_ba = np.min(np.sum((b[:, None, :] - a[None, :, :]) ** 2, axis=-1), axis=1)
    return float(np.sqrt(0.5 * (d_ab.mean() + d_ba.mean())))


def phi_velocity(state: np.ndarray):
    theta = float(state[0])
    s = max(np.sin(theta), 1e-8)
    return float(state[3] / (s ** 2))


def initial_meridian_section(traj: np.ndarray):
    """Return positive crossings of phi - phi(0) = 0 mod 2*pi.

    The benchmark compares returns to each trajectory's initial meridian.  It
    does not rotate states to, or claim to use, the global phi = 0 meridian.
    """
    phi = traj[:, 1]
    shifted = phi - phi[0]
    levels = shifted / (2.0 * np.pi)
    points = []
    for idx in range(len(traj) - 1):
        lo = levels[idx]
        hi = levels[idx + 1]
        if hi <= lo:
            continue
        start = int(np.floor(lo)) + 1
        end = int(np.floor(hi))
        if end < start:
            continue
        for target in range(start, end + 1):
            target_phi = target * 2.0 * np.pi
            alpha = (target_phi - shifted[idx]) / max(shifted[idx + 1] - shifted[idx], 1e-12)
            state = traj[idx] + alpha * (traj[idx + 1] - traj[idx])
            if phi_velocity(state) <= 0.0:
                continue
            points.append([state[0], state[2]])
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def return_map_cloud(section: np.ndarray):
    if len(section) < 2:
        return np.zeros((0, 4), dtype=np.float64)
    return np.concatenate([section[:-1], section[1:]], axis=1)


def build_model_from_checkpoint(bench, ckpt_path, device, dim):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model_name = args.get("model", ckpt_path.parent.parent.name)
    model = bench.build_model(
        model_name,
        dim,
        hidden=args.get("hidden", 128),
        sparsity=args.get("sparsity", "dense"),
        separable=bench.infer_separable(ckpt, dim),
        implicit_max_iters=args.get("implicit_max_iters", 5),
        implicit_tol=args.get("implicit_tol", 1e-6),
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
    diff = wrapped_error(pred, gt)
    energy = np.asarray(get_hamiltonian(pred), dtype=np.float64)
    pphi = pred[:, 3]
    gt_section = initial_meridian_section(gt)
    pred_section = initial_meridian_section(pred)
    gt_return = return_map_cloud(gt_section)
    pred_return = return_map_cloud(pred_section)
    return {
        "mse": float(np.mean(diff ** 2)),
        "max_energy_drift": max_relative_drift(energy),
        "max_p_phi_drift": max_relative_drift(pphi),
        "phase_rmse": float(np.sqrt(np.mean(diff[:, 1] ** 2))),
        "section_cloud_rmse": symmetric_cloud_rmse(pred_section, gt_section),
        "return_map_cloud_rmse": symmetric_cloud_rmse(pred_return, gt_return),
        "crossing_count_relerr": float(abs(len(pred_section) - len(gt_section)) / max(len(gt_section), 1)),
        "num_section_crossings": int(len(pred_section)),
        "eval_seconds": 0.0,
    }


def summarise(raw_rows):
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[row["model"]].append(row)

    fields = [
        "mse",
        "max_energy_drift",
        "max_p_phi_drift",
        "phase_rmse",
        "section_cloud_rmse",
        "return_map_cloud_rmse",
        "crossing_count_relerr",
    ]
    summary = []
    for model, rows in sorted(grouped.items(), key=lambda item: MODEL_ORDER.index(item[0]) if item[0] in MODEL_ORDER else len(MODEL_ORDER)):
        per_seed = defaultdict(list)
        seed_times = defaultdict(float)
        for row in rows:
            per_seed[row["seed"]].append(row)
            seed_times[row["seed"]] = max(seed_times[row["seed"]], row["eval_seconds"])
        item = {
            "system": "spherical_pendulum",
            "model": model,
            "num_seeds": len(per_seed),
            "ic_per_seed": len(rows) // max(len(per_seed), 1),
            "steps": rows[0]["steps"],
        }
        for field in fields:
            values = []
            for seed_rows in per_seed.values():
                vals = [r[field] for r in seed_rows if not np.isnan(r[field])]
                values.append(float(np.mean(vals)) if vals else np.nan)
            arr = np.asarray(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            item[f"{field}_seed_mean"] = float(arr.mean()) if len(arr) else np.nan
            item[f"{field}_seed_std"] = float(arr.std(ddof=1)) if len(arr) else np.nan
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
        "section_definition": {
            "equation": "phi - phi_initial = 0 mod 2*pi",
            "orientation": "positive",
            "initial_crossing_excluded": True,
        },
        "state_error_definition": {
            "periodic_difference_interval": "[-pi, pi)",
            "periodic_indices": [1],
        },
        "raw": raw_rows,
        "summary": summary_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    lines = [
        "# Spherical pendulum invariant and meridian return audit",
        "",
        "| Model | Seeds | ICs/seed | Steps | MSE | Energy drift | $p_\\phi$ drift | Section RMSE | Return-map RMSE | Crossing-count rel. err. |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {MODEL_LABELS.get(row['model'], row['model'])} | {row['num_seeds']} | {row['ic_per_seed']} | {row['steps']} | "
            f"{row['mse_seed_mean']:.4e} +/- {row['mse_seed_std']:.2e} | "
            f"{row['max_energy_drift_seed_mean']:.4e} +/- {row['max_energy_drift_seed_std']:.2e} | "
            f"{row['max_p_phi_drift_seed_mean']:.4e} +/- {row['max_p_phi_drift_seed_std']:.2e} | "
            f"{row['section_cloud_rmse_seed_mean']:.4e} +/- {row['section_cloud_rmse_seed_std']:.2e} | "
            f"{row['return_map_cloud_rmse_seed_mean']:.4e} +/- {row['return_map_cloud_rmse_seed_std']:.2e} | "
            f"{row['crossing_count_relerr_seed_mean']:.4e} +/- {row['crossing_count_relerr_seed_std']:.2e} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    return out_json, out_md


def plot_summary(summary_rows, output_stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    row_map = {row["model"]: row for row in summary_rows}
    metrics = [
        ("max_p_phi_drift_seed_mean", "max_p_phi_drift_seed_std", "$p_\\phi$ drift"),
        ("section_cloud_rmse_seed_mean", "section_cloud_rmse_seed_std", "Meridian section RMSE"),
        ("return_map_cloud_rmse_seed_mean", "return_map_cloud_rmse_seed_std", "Return-map RMSE"),
        ("crossing_count_relerr_seed_mean", "crossing_count_relerr_seed_std", "Crossing-count rel. err."),
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
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=18, ha="right")
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
    parser = argparse.ArgumentParser(description="Evaluate spherical pendulum invariant and meridian-return indicators.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gscd", "gscd_sym", "gscd_exp", "gscd_nosplit", "hnn", "hnn_implicit", "hnn_symp", "sympnet", "gsympnet", "node"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output_stem", default="spherical_pendulum_meridian_5seed")
    args = parser.parse_args()

    bench = load_benchmark_module()
    device = torch.device(args.device)
    test_data = load_test_trajectories()
    dt = bench.SYSTEM_DT["spherical_pendulum"]
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
                metrics["eval_seconds"] = elapsed
                raw_rows.append(
                    {
                        "system": "spherical_pendulum",
                        "model": model_name,
                        "seed": seed,
                        "ic_index": ic_idx,
                        "steps": int(test_data.shape[1] - 1),
                        "checkpoint": str(ckpt_path),
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
