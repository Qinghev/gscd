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


def toda_lax_eigenvalues(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states[None, :]
    n = states.shape[-1] // 2
    q = states[:, :n]
    p = states[:, n:]
    a = np.exp(0.5 * (q - np.roll(q, -1, axis=1)))

    mats = np.zeros((states.shape[0], n, n), dtype=np.float64)
    idx = np.arange(n)
    nxt = (idx + 1) % n
    mats[:, idx, idx] = p
    mats[:, idx, nxt] = a
    mats[:, nxt, idx] = a
    return np.linalg.eigvalsh(mats)


def invariant_errors(states: np.ndarray, lax0: np.ndarray, p_sum0: np.ndarray, p_norm0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    n = states.shape[-1] // 2
    lax = toda_lax_eigenvalues(states)
    lax_err = np.linalg.norm(lax - lax0, axis=1) / (np.linalg.norm(lax0, axis=1) + 1e-12)
    p_sum = states[:, n:].sum(axis=1)
    momentum_err = np.abs(p_sum - p_sum0) / (p_norm0 + 1e-12)
    return lax_err, momentum_err


def energy_errors(bench, system: str, states: np.ndarray, e0: np.ndarray) -> np.ndarray:
    values = np.asarray([bench.energy_fn(system, state) for state in states], dtype=np.float64)
    return np.abs((values - e0) / (e0 + 1e-12))


def reference_metrics(bench, system: str, test_data: torch.Tensor) -> dict:
    gt = test_data.cpu().numpy()
    initial = gt[:, 0, :]
    n = initial.shape[-1] // 2
    lax0 = toda_lax_eigenvalues(initial)
    p_sum0 = initial[:, n:].sum(axis=1)
    p_norm0 = np.linalg.norm(initial[:, n:], axis=1)
    e0 = np.asarray([bench.energy_fn(system, state) for state in initial], dtype=np.float64)

    curves = {"lax_spectrum_error": [], "momentum_error": [], "relative_energy_error": []}
    max_lax = np.zeros(gt.shape[0], dtype=np.float64)
    max_momentum = np.zeros(gt.shape[0], dtype=np.float64)
    max_energy = np.zeros(gt.shape[0], dtype=np.float64)
    for idx in range(gt.shape[1]):
        lax_err, mom_err = invariant_errors(gt[:, idx, :], lax0, p_sum0, p_norm0)
        en_err = energy_errors(bench, system, gt[:, idx, :], e0)
        max_lax = np.maximum(max_lax, lax_err)
        max_momentum = np.maximum(max_momentum, mom_err)
        max_energy = np.maximum(max_energy, en_err)
        curves["lax_spectrum_error"].append(float(lax_err.mean()))
        curves["momentum_error"].append(float(mom_err.mean()))
        curves["relative_energy_error"].append(float(en_err.mean()))

    return {
        "curves": curves,
        "max_lax_spectrum_error_ic_mean": float(max_lax.mean()),
        "max_lax_spectrum_error_ic_max": float(max_lax.max()),
        "max_momentum_error_ic_mean": float(max_momentum.mean()),
        "max_momentum_error_ic_max": float(max_momentum.max()),
        "max_relative_energy_error_ic_mean": float(max_energy.mean()),
        "max_relative_energy_error_ic_max": float(max_energy.max()),
    }


def rollout_seed_metrics(
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
    n = initial.shape[-1] // 2
    lax0 = toda_lax_eigenvalues(initial)
    p_sum0 = initial[:, n:].sum(axis=1)
    p_norm0 = np.linalg.norm(initial[:, n:], axis=1)
    e0 = np.asarray([bench.energy_fn(system, state) for state in initial], dtype=np.float64)

    curves = {"mse": [], "relative_energy_error": [], "lax_spectrum_error": [], "momentum_error": []}
    max_lax = np.zeros(test_data.shape[0], dtype=np.float64)
    max_momentum = np.zeros(test_data.shape[0], dtype=np.float64)
    max_energy = np.zeros(test_data.shape[0], dtype=np.float64)
    se_sum = 0.0
    count = 0
    start = time.time()

    with torch.no_grad():
        for idx in range(test_data.shape[1]):
            pred_np = u.detach().cpu().numpy()
            diff = pred_np - gt_np[:, idx, :]
            mse_ic = np.mean(diff**2, axis=1)
            lax_err, mom_err = invariant_errors(pred_np, lax0, p_sum0, p_norm0)
            en_err = energy_errors(bench, system, pred_np, e0)

            se_sum += float(np.sum(diff**2))
            count += diff.size
            max_lax = np.maximum(max_lax, lax_err)
            max_momentum = np.maximum(max_momentum, mom_err)
            max_energy = np.maximum(max_energy, en_err)
            curves["mse"].append(float(mse_ic.mean()))
            curves["relative_energy_error"].append(float(en_err.mean()))
            curves["lax_spectrum_error"].append(float(lax_err.mean()))
            curves["momentum_error"].append(float(mom_err.mean()))
            if idx < test_data.shape[1] - 1:
                u = model.step(u, dt)

    return {
        "model": model_name,
        "seed": int(seed),
        "best_val_loss": float(payload.get("best_val_loss", np.nan)),
        "eval_seconds": float(time.time() - start),
        "mse": float(se_sum / count),
        "max_relative_energy_error_ic_mean": float(max_energy.mean()),
        "max_lax_spectrum_error_ic_mean": float(max_lax.mean()),
        "max_momentum_error_ic_mean": float(max_momentum.mean()),
        "curves": curves,
    }


def summarise(seed_rows: list[dict]) -> dict:
    by_model: dict[str, list[dict]] = {}
    for row in seed_rows:
        by_model.setdefault(row["model"], []).append(row)

    summary = {}
    metrics = [
        "mse",
        "max_relative_energy_error_ic_mean",
        "max_lax_spectrum_error_ic_mean",
        "max_momentum_error_ic_mean",
    ]
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
            seed_rows.append(rollout_seed_metrics(bench, args.system, model_name, seed, test_data, device))

    raw_seed = [{key: value for key, value in row.items() if key != "curves"} for row in seed_rows]
    return {
        "system": args.system,
        "models": args.models,
        "seeds": args.seeds,
        "num_ics": int(test_data.shape[0]),
        "steps": int(test_data.shape[1] - 1),
        "dt": float(bench.SYSTEM_DT[args.system]),
        "time": time_grid,
        "reference": reference_metrics(bench, args.system, test_data),
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
        ("lax_spectrum_error_mean", "Mean Lax spectrum drift"),
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
    parser.add_argument("--output_stem", default="toda_invariants_5seed")
    parser.add_argument("--figure_stem", default="toda_time_curves_5seed")
    args = parser.parse_args()

    payload = evaluate(args)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / f"{args.output_stem}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}", flush=True)
    plot(payload, args.figure_stem)


if __name__ == "__main__":
    main()
