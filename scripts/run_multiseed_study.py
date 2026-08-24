import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).parent.parent
BENCH_PATH = REPO / "scripts" / "run_final_benchmarks.py"
def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("run_final_benchmarks", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_reference_args(system, model):
    candidates = (
        REPO / "checkpoints" / system / model / "multiseed" / "seed_0" / "best.pt",
        REPO / "checkpoints" / system / model / "best.pt",
    )
    ckpt_path = next((path for path in candidates if path.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            "Reference checkpoint not found; tried "
            + ", ".join(str(path) for path in candidates)
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("args", {})


def checkpoint_path(system, model, seed):
    return REPO / "checkpoints" / system / model / "multiseed" / f"seed_{seed}" / "best.pt"


def train_log_path(system, model, seed):
    return REPO / "checkpoints" / system / model / "multiseed" / f"seed_{seed}" / "train_log.json"


def build_train_command(system, model, seed):
    args = load_reference_args(system, model)
    cmd = [
        sys.executable,
        "src/train.py",
        "--system", system,
        "--model", model,
        "--epochs", str(args.get("epochs", 200)),
        "--stride", str(args.get("stride", 10)),
        "--patience", str(args.get("patience", 30)),
        "--hidden", str(args.get("hidden", 128)),
        "--batch_size", str(args.get("batch_size", 32)),
        "--window", str(args.get("window", 10)),
        "--lr", str(args.get("lr", 3e-4)),
        "--sparsity", str(args.get("sparsity", "dense")),
        "--implicit_max_iters", str(args.get("implicit_max_iters", 5)),
        "--implicit_tol", str(args.get("implicit_tol", 1e-6)),
        "--seed", str(seed),
        "--tag", f"multiseed/seed_{seed}",
    ]
    if args.get("data_ratio", 1.0) != 1.0:
        cmd.extend(["--data_ratio", str(args["data_ratio"])])
    if args.get("non_separable", False):
        cmd.append("--non_separable")
    return cmd


def run_training(system, model, seed):
    ckpt_path = checkpoint_path(system, model, seed)
    log_path = train_log_path(system, model, seed)
    if ckpt_path.exists() and log_path.exists():
        print(f"Skipping cached training for {system}/{model}/seed_{seed}")
        return ckpt_path
    cmd = build_train_command(system, model, seed)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/mplbench")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True, env=env)
    return ckpt_path


def summarise(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["system"], row["model"])].append(row)

    summary = []
    for (system, model), entries in sorted(grouped.items()):
        mse_values = np.array([entry["mse"] for entry in entries], dtype=float)
        drift_values = [entry["max_energy_drift"] for entry in entries if entry["max_energy_drift"] is not None]
        time_values = np.array([entry["eval_seconds"] for entry in entries], dtype=float)
        row = {
            "system": system,
            "model": model,
            "num_seeds": len(entries),
            "mse_mean": float(mse_values.mean()),
            "mse_std": float(mse_values.std(ddof=1)),
            "eval_seconds_mean": float(time_values.mean()),
            "eval_seconds_std": float(time_values.std(ddof=1)),
        }
        if drift_values:
            drift_values = np.array(drift_values, dtype=float)
            row["drift_mean"] = float(drift_values.mean())
            row["drift_std"] = float(drift_values.std(ddof=1))
        else:
            row["drift_mean"] = None
            row["drift_std"] = None
        summary.append(row)
    return summary


def format_pm(mean, std):
    if mean is None:
        return "--"
    return f"{mean:.4e} ± {std:.2e}"


def output_paths(output_stem: str):
    return (
        REPO / "results" / f"{output_stem}.json",
        REPO / "results" / f"{output_stem}.md",
    )


def write_outputs(raw_rows, summary_rows, systems, models, seeds, output_stem):
    out_json, out_md = output_paths(output_stem)
    payload = {
        "systems": systems,
        "models": models,
        "seeds": seeds,
        "raw": raw_rows,
        "summary": summary_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Multiseed Summary",
        "",
        f"Systems: {', '.join(systems)}",
        f"Models: {', '.join(models)}",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "| System | Model | Seeds | Rollout MSE (mean ± std) | Max Energy Drift (mean ± std) | Eval Time (s, mean ± std) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['system']} | {row['model']} | {row['num_seeds']} | "
            f"{format_pm(row['mse_mean'], row['mse_std'])} | "
            f"{format_pm(row['drift_mean'], row['drift_std'])} | "
            f"{format_pm(row['eval_seconds_mean'], row['eval_seconds_std'])} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    return out_json, out_md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=["toda"])
    parser.add_argument("--models", nargs="+", default=["gscd", "node", "hnn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_stem", default="multiseed_summary",
                        help="basename under results/ for the JSON and markdown summaries")
    parser.add_argument("--use_existing", action="store_true",
                        help="evaluate existing multiseed checkpoints even if train_log.json is absent")
    args = parser.parse_args()

    bench = load_benchmark_module()
    raw_rows = []
    for system in args.systems:
        for model in args.models:
            for seed in args.seeds:
                if args.use_existing:
                    ckpt_path = checkpoint_path(system, model, seed)
                    if not ckpt_path.exists():
                        raise FileNotFoundError(f"Missing checkpoint for {system}/{model}/seed_{seed}: {ckpt_path}")
                else:
                    ckpt_path = run_training(system, model, seed)
                metric = bench.evaluate_checkpoint(system, ckpt_path, device=args.device)
                metric["seed"] = seed
                raw_rows.append(metric)
                print(
                    f"Evaluated {system}/{model}/seed_{seed}: "
                    f"mse={metric['mse']:.4e}, "
                    f"drift={bench.format_value(metric['max_energy_drift'])}, "
                    f"time={metric['eval_seconds']:.2f}s"
                )

    summary_rows = summarise(raw_rows)
    out_json, out_md = write_outputs(raw_rows, summary_rows, args.systems, args.models, args.seeds, args.output_stem)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
