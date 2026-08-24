from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
MODEL_ORDER = (
    "gscd",
    "gscd_nosplit",
    "gsympnet",
    "hnn",
    "hnn_implicit",
    "hnn_symp",
    "node",
)
MODEL_LABELS = {
    "gscd": "GSC-HNN",
    "gscd_nosplit": "GSC-no-split",
    "gsympnet": "G-SympNet",
    "hnn": "HNN",
    "hnn_implicit": "HNN-Implicit",
    "hnn_symp": "HNN-sep-symp",
    "node": "NODE",
}
MODEL_COLORS = {
    "gscd": "#0072B2",
    "gscd_nosplit": "#CC79A7",
    "gsympnet": "#6A3D9A",
    "hnn": "#009E73",
    "hnn_implicit": "#7F3C8D",
    "hnn_symp": "#4C78A8",
    "node": "#D55E00",
}


def read_summary(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    if isinstance(summary, dict):
        return summary
    return {row["model"]: row for row in summary}


def merged_summary(paths: list[Path]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in paths:
        rows.update(read_summary(path))
    missing = [model for model in MODEL_ORDER if model not in rows]
    if missing:
        raise KeyError(f"Missing spherical pendulum summaries: {missing}")
    return rows


def normalized(
    rows: dict[str, dict], model: str, mean_key: str, std_key: str
) -> tuple[float, float]:
    denominator = float(rows["gscd"][mean_key])
    return (
        float(rows[model][mean_key]) / denominator,
        float(rows[model][std_key]) / denominator,
    )


def render(rows: dict[str, dict], output_dir: Path, stem: str) -> list[Path]:
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
        1, 3, figsize=(6.7, 2.45), constrained_layout=True, sharey=True
    )
    y = np.arange(len(MODEL_ORDER), dtype=float)

    panels = (
        (
            axes[0],
            (("mse_seed_mean", "mse_seed_std", "o", 0.0, None),),
            "Rollout MSE",
        ),
        (
            axes[1],
            (
                (
                    "section_cloud_rmse_seed_mean",
                    "section_cloud_rmse_seed_std",
                    "o",
                    -0.12,
                    "section",
                ),
                (
                    "return_map_cloud_rmse_seed_mean",
                    "return_map_cloud_rmse_seed_std",
                    "s",
                    0.12,
                    "return",
                ),
            ),
            "Initial-meridian geometry",
        ),
        (
            axes[2],
            (
                (
                    "max_p_phi_drift_seed_mean",
                    "max_p_phi_drift_seed_std",
                    "o",
                    0.0,
                    None,
                ),
            ),
            r"$p_\phi$ drift",
        ),
    )

    for ax, series, title in panels:
        for mean_key, std_key, marker, offset, label in series:
            for index, model in enumerate(MODEL_ORDER):
                ratio, error = normalized(rows, model, mean_key, std_key)
                ax.errorbar(
                    ratio,
                    y[index] + offset,
                    xerr=error,
                    fmt=marker,
                    markersize=4.0,
                    color=MODEL_COLORS[model],
                    markeredgecolor="black",
                    markeredgewidth=0.3,
                    elinewidth=0.8,
                    capsize=1.5,
                    label=label if index == 0 else None,
                )
        ax.axvline(1.0, color="0.35", linewidth=0.8, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.2, 5.5)
        ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
        ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
        ax.set_xlabel("ratio to GSC-HNN")
        ax.set_title(title, pad=4)
        ax.grid(True, axis="x", which="major", linestyle=":", alpha=0.3)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, loc="lower right", fontsize=7)

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
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            RESULTS / "spherical_pendulum_meridian_5seed.json",
            RESULTS / "gsympnet_spherical_pendulum_meridian_5seed.json",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS / "figures" / "spherical_pendulum",
    )
    parser.add_argument(
        "--stem",
        default="spherical_pendulum_metric_dotplot_with_gsympnet_5seed",
    )
    args = parser.parse_args()

    for output in render(merged_summary(args.inputs), args.output_dir, args.stem):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
