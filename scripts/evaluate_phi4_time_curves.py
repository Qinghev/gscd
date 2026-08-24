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


def energy_errors(bench, system: str, states: np.ndarray, e0: np.ndarray) -> np.ndarray:
    values = np.asarray([bench.energy_fn(system, state) for state in states], dtype=np.float64)
    return np.abs((values - e0) / (e0 + 1e-12))


def modal_energy_spectrum(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states[None, :]
    n = states.shape[-1] // 2
    q = states[:, :n]
    p = states[:, n:]
    q_hat = np.fft.rfft(q, axis=1, norm="ortho")
    p_hat = np.fft.rfft(p, axis=1, norm="ortho")
    k = np.arange(q_hat.shape[1], dtype=np.float64)
    omega_sq = 1.0 + 4.0 * np.sin(np.pi * k / n) ** 2
    return 0.5 * (np.abs(p_hat) ** 2 + omega_sq[None, :] * np.abs(q_hat) ** 2)


def modal_spectrum_errors(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred_spec = modal_energy_spectrum(pred)
    truth_spec = modal_energy_spectrum(truth)
    return np.linalg.norm(pred_spec - truth_spec, axis=1) / (np.linalg.norm(truth_spec, axis=1) + 1e-12)


def rollout_seed_curves(
    bench,
    system: str,
    model_name: str,
    seed: int,
    test_data: torch.Tensor,
    device: torch.device,
) -> dict:
    ckpt = checkpoint_path(system, model_name, seed)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    model, payload, _loaded_name = build_model_from_checkpoint(bench, system, ckpt, device)
    dt = bench.SYSTEM_DT[system]
    gt_np = test_data.cpu().numpy()
    u = test_data[:, 0, :].to(device)
    if hasattr(model, "prepare_initial_state"):
        u = model.prepare_initial_state(u)

    initial = u.detach().cpu().numpy()
    e0 = np.asarray([bench.energy_fn(system, state) for state in initial], dtype=np.float64)
    curves = {"mse": [], "relative_energy_error": [], "modal_spectrum_error": []}
    se_sum = 0.0
    count = 0
    max_energy = np.zeros(test_data.shape[0], dtype=np.float64)
    start = time.time()

    with torch.no_grad():
        for idx in range(test_data.shape[1]):
            pred_np = u.detach().cpu().numpy()
            truth_np = gt_np[:, idx, :]
            diff = pred_np - truth_np
            mse_ic = np.mean(diff**2, axis=1)
            energy_err = energy_errors(bench, system, pred_np, e0)
            spectrum_err = modal_spectrum_errors(pred_np, truth_np)

            se_sum += float(np.sum(diff**2))
            count += diff.size
            max_energy = np.maximum(max_energy, energy_err)
            curves["mse"].append(float(mse_ic.mean()))
            curves["relative_energy_error"].append(float(energy_err.mean()))
            curves["modal_spectrum_error"].append(float(spectrum_err.mean()))
            if idx < test_data.shape[1] - 1:
                u = model.step(u, dt)

    return {
        "model": model_name,
        "seed": int(seed),
        "best_val_loss": float(payload.get("best_val_loss", np.nan)),
        "eval_seconds": float(time.time() - start),
        "mse": float(se_sum / count),
        "max_relative_energy_error_ic_mean": float(max_energy.mean()),
        "curves": curves,
    }


def summarise(seed_rows: list[dict]) -> dict:
    by_model: dict[str, list[dict]] = {}
    for row in seed_rows:
        by_model.setdefault(row["model"], []).append(row)

    summary = {}
    metrics = ["mse", "max_relative_energy_error_ic_mean"]
    for model_name, rows in by_model.items():
        entry = {"seed_count": len(rows)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            entry[f"{metric}_seed_mean"] = float(values.mean())
            entry[f"{metric}_seed_std"] = float(values.std(ddof=1))
        curves = {}
        for key in rows[0]["curves"]:
            values = np.asarray([row["curves"][key] for row in rows], dtype=np.float64)
            curves[f"{key}_mean"] = values.mean(axis=0).tolist()
        entry["curves"] = curves
        summary[model_name] = entry
    return summary


def evaluate(args: argparse.Namespace) -> dict:
    bench = load_benchmark_module()
    device = torch.device(args.device)
    test_data = load_test_data(args.system, args.max_ics, args.max_steps)
    time_grid = (np.arange(test_data.shape[1], dtype=np.float64) * bench.SYSTEM_DT[args.system]).tolist()

    seed_rows = []
    for model_name in args.models:
        for seed in args.seeds:
            print(f"Evaluating {args.system}/{model_name}/seed_{seed}", flush=True)
            seed_rows.append(rollout_seed_curves(bench, args.system, model_name, seed, test_data, device))

    raw_seed = [{key: value for key, value in row.items() if key != "curves"} for row in seed_rows]
    return {
        "system": args.system,
        "models": args.models,
        "seeds": args.seeds,
        "num_ics": int(test_data.shape[0]),
        "steps": int(test_data.shape[1] - 1),
        "dt": float(bench.SYSTEM_DT[args.system]),
        "time": time_grid,
        "raw_seed": raw_seed,
        "summary": summarise(seed_rows),
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
            "lines.linewidth": 1.4,
        }
    )
    time_grid = np.asarray(payload["time"], dtype=np.float64)
    plot_slice = slice(1, None)
    fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.15), constrained_layout=True)
    panels = [
        ("mse_mean", "Mean rollout MSE"),
        ("relative_energy_error_mean", "Mean relative energy error"),
        ("modal_spectrum_error_mean", "Mean modal-spectrum error"),
    ]
    for ax, (key, ylabel) in zip(axes, panels):
        for model_name in payload["models"]:
            values = np.asarray(payload["summary"][model_name]["curves"][key], dtype=np.float64)
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

    base = RESULTS / "figures" / "phi4"
    base.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        target = base / f"{output_stem}.{suffix}"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        print(f"Wrote {target}", flush=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default="phi4")
    parser.add_argument("--models", nargs="+", default=["gscd", "hnn", "node", "gsympnet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_ics", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_stem", default="phi4_time_curves_5seed")
    args = parser.parse_args()

    payload = evaluate(args)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / f"{args.output_stem}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}", flush=True)
    plot(payload, args.output_stem)


if __name__ == "__main__":
    main()
