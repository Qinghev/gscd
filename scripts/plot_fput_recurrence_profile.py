"""Render the manuscript's seed-by-initial-condition FPUT recurrence profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

MODEL_ORDER = ("gscd", "hnn", "node", "gsympnet")
MODEL_LABELS = {
    "gscd": "GSC-HNN",
    "hnn": "HNN",
    "node": "NODE",
    "gsympnet": "G-SympNet",
}
MODEL_COLORS = {
    "gscd": "#0072B2",
    "hnn": "#009E73",
    "node": "#D55E00",
    "gsympnet": "#6A3D9A",
}
SEED_MARKERS = ("o", "s", "^", "D", "P")
METRICS = (
    ("mse", "Rollout\nMSE"),
    ("max_energy_drift", "Energy\ndrift"),
    ("mode1_energy_rmse", "Mode-1\nRMSE"),
    ("first_recurrence_rel_error", "Recurrence\nerror"),
    ("modal_spectrum_l1", "Modal\nspectrum L1"),
)
FAMILIES = (
    ("pure_mode1", "Pure mode-1"),
    ("mode1_mode2_mixed", "Mode-1/mode-2 mixed"),
)


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["raw"]
    ic_count = max(int(row["ic_index"]) for row in rows) + 1
    if ic_count % 2:
        raise ValueError(
            "The archived FPUT protocol requires equal pure and mixed families"
        )
    split = ic_count // 2
    for row in rows:
        row.setdefault(
            "family",
            "pure_mode1" if int(row["ic_index"]) < split else "mode1_mode2_mixed",
        )
    return rows


def render(rows: list[dict], output_dir: Path, stem: str) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    fig, axes = plt.subplots(
        2, 5, figsize=(7.05, 3.45), constrained_layout=True, squeeze=False
    )

    for row_index, (family, family_label) in enumerate(FAMILIES):
        family_ic_order = sorted(
            {int(row["ic_index"]) for row in rows if row["family"] == family}
        )
        family_ic_position = {
            ic_index: position
            for position, ic_index in enumerate(family_ic_order)
        }
        family_center = (len(family_ic_order) - 1) / 2.0
        for column_index, (metric, title) in enumerate(METRICS):
            ax = axes[row_index, column_index]
            for model_index, model in enumerate(MODEL_ORDER):
                model_rows = [
                    row
                    for row in rows
                    if row["model"] == model and row["family"] == family
                ]
                for item in model_rows:
                    seed = int(item["seed"])
                    ic_index = int(item["ic_index"])
                    jitter = (
                        0.025 * (family_ic_position[ic_index] - family_center)
                        + 0.006 * (seed - 2)
                    )
                    ax.scatter(
                        model_index + jitter,
                        float(item[metric]),
                        s=12,
                        marker=SEED_MARKERS[seed % len(SEED_MARKERS)],
                        facecolor=MODEL_COLORS[model],
                        edgecolor="black",
                        linewidth=0.25,
                        alpha=0.78,
                    )

            if metric == "first_recurrence_rel_error":
                ax.set_yscale("symlog", linthresh=1e-4, linscale=0.75)
            else:
                ax.set_yscale("log")
            ax.set_title(title, pad=3)
            ax.set_xlim(-0.45, len(MODEL_ORDER) - 0.55)
            ax.set_xticks([])
            ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.3)
            if column_index == 0:
                ax.set_ylabel(family_label)

    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=MODEL_LABELS[model],
        )
        for model in MODEL_ORDER
    ]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        ncol=len(model_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("pdf", "png"):
        target = output_dir / f"{stem}.{suffix}"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        outputs.append(target)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS / "gsympnet_fput_recurrence_5seed.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS / "figures" / "fput"
    )
    parser.add_argument("--stem", default="fput_recurrence_profile_5seed")
    args = parser.parse_args()

    for output in render(load_rows(args.input), args.output_dir, args.stem):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
