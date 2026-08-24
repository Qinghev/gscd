from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import GSCD_Integrator, make_J  # noqa: E402
from train import DTYPE, SYSTEM_DIM, SYSTEM_DT, build_model  # noqa: E402


SYSTEMS = ("double_pendulum", "spherical_pendulum")


def torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(system: str) -> Path:
    """Resolve the primary seed-0 checkpoint, with a legacy fallback."""
    candidates = [
        REPO / "checkpoints" / system / "gscd" / "multiseed" / "seed_0" / "best.pt",
        REPO / "checkpoints" / system / "gscd" / "best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No GSC-HNN checkpoint found for {system}; tried "
        + ", ".join(str(p) for p in candidates)
    )


def load_gscd(system: str, checkpoint: Path, device: torch.device, max_iters: int, tol: float):
    payload = torch_load(checkpoint, map_location=device)
    args = payload.get("args", {})
    dim = SYSTEM_DIM[system]
    model_name = args.get("model", "gscd")
    if model_name != "gscd":
        raise ValueError(f"Expected gscd checkpoint for {system}, got {model_name}")
    model = build_model(
        model_name,
        dim,
        hidden=int(args.get("hidden", 128)),
        sparsity=args.get("sparsity", "dense"),
        separable=not bool(args.get("non_separable", True)),
    ).to(device=device, dtype=DTYPE)
    if not isinstance(model, GSCD_Integrator):
        raise TypeError(f"Expected GSCD_Integrator for {system}, got {type(model).__name__}")
    model.load_state_dict(payload["model_state"])
    model.eval()
    model.nonlin_part.implicit_max_iters = int(max_iters)
    model.nonlin_part.implicit_tol = float(tol)
    return model, payload


def sample_states(system: str, samples: int, device: torch.device) -> torch.Tensor:
    path = REPO / "data" / system / "test.pt"
    data = torch_load(path, map_location="cpu")
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"Expected tensor test data at {path}")
    flat = data.reshape(-1, data.shape[-1]).to(device=device, dtype=DTYPE)
    if samples >= flat.shape[0]:
        return flat
    idx = torch.linspace(0, flat.shape[0] - 1, samples, device=device).round().long()
    return flat.index_select(0, idx)


def hamiltonian_field(nonlin, x: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        xt = x.detach().clone().requires_grad_(True)
        H = nonlin.hamiltonian(xt).sum()
        grad = torch.autograd.grad(H, xt, create_graph=False)[0]
    return grad @ J.T


def residual_stage_audit(
    model: GSCD_Integrator, states: torch.Tensor, dt: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rho_F, iteration counts, and stopping-test convergence flags."""
    J = make_J(model.n_dof, device=states.device, dtype=states.dtype)
    u = model.linear_part(states, dt / 2.0)
    v = u + dt * hamiltonian_field(model.nonlin_part, u, J)
    max_iters = int(model.nonlin_part.implicit_max_iters)
    tol = float(model.nonlin_part.implicit_tol)
    done = torch.zeros(u.shape[0], dtype=torch.bool, device=u.device)
    counts = torch.zeros(u.shape[0], dtype=torch.long, device=u.device)

    for _ in range(max_iters):
        midpoint = (u + v) / 2.0
        v_new = u + dt * hamiltonian_field(model.nonlin_part, midpoint, J)
        diff = torch.linalg.norm(v_new - v, dim=-1)
        active = ~done
        counts += active.long()
        v = torch.where(active.unsqueeze(-1), v_new, v)
        done |= active & (diff < tol)
        if bool(done.all()):
            break

    midpoint = (u + v) / 2.0
    F = v - u - dt * hamiltonian_field(model.nonlin_part, midpoint, J)
    rho = torch.linalg.norm(F, dim=-1) / (
        1.0 + torch.linalg.norm(u, dim=-1) + torch.linalg.norm(v, dim=-1)
    )
    return rho.detach(), counts.detach(), done.detach()


def one_step(model: GSCD_Integrator, u: torch.Tensor, dt: float) -> torch.Tensor:
    if u.ndim == 1:
        u_batch = u.unsqueeze(0)
        return model.step(u_batch, dt).squeeze(0)
    return model.step(u, dt)


def finite_difference_jvp(
    model: GSCD_Integrator, u: torch.Tensor, dt: float, direction: torch.Tensor, eps: float
) -> torch.Tensor:
    norm_d = torch.linalg.norm(direction).clamp_min(1e-30)
    step = eps * (1.0 + torch.linalg.norm(u)) / norm_d
    return (one_step(model, u + step * direction, dt) - one_step(model, u - step * direction, dt)) / (
        2.0 * step
    )


def symplectic_and_fb_audit(
    model: GSCD_Integrator,
    states: torch.Tensor,
    dt: float,
    pairs: int,
    fd_eps: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = states.shape[-1]
    J = make_J(dim // 2, device=states.device, dtype=states.dtype)
    gen = torch.Generator(device=states.device)
    gen.manual_seed(seed)
    eps_omega: List[torch.Tensor] = []
    eps_fb: List[torch.Tensor] = []

    for u in states:
        point_max = torch.tensor(0.0, device=states.device, dtype=states.dtype)
        for _ in range(pairs):
            a = torch.randn(dim, generator=gen, device=states.device, dtype=states.dtype)
            b = torch.randn(dim, generator=gen, device=states.device, dtype=states.dtype)
            a = a / torch.linalg.norm(a).clamp_min(1e-30)
            b = b / torch.linalg.norm(b).clamp_min(1e-30)
            da = finite_difference_jvp(model, u, dt, a, fd_eps)
            db = finite_difference_jvp(model, u, dt, b, fd_eps)
            defect = torch.abs(da @ (J @ db) - a @ (J @ b))
            point_max = torch.maximum(point_max, defect)
        eps_omega.append(point_max.detach())

        fwd = one_step(model, u, dt)
        back = one_step(model, fwd, -dt)
        eps_fb.append(torch.linalg.norm(back - u).detach())

    return torch.stack(eps_omega), torch.stack(eps_fb)


def fmt(x: float) -> str:
    if not math.isfinite(x):
        return "--"
    return f"{x:.3e}"


def fmt_tex(x: float) -> str:
    if not math.isfinite(x):
        return "--"
    if x == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(x))))
    mantissa = x / (10.0**exponent)
    return f"${mantissa:.3f}\\times 10^{{{exponent}}}$"


def summarize(values: torch.Tensor) -> Dict[str, float]:
    arr = values.detach().cpu().numpy().astype(float)
    return {
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
    }


def audit_system(
    system: str,
    device: torch.device,
    samples: int,
    pairs: int,
    fd_eps: float,
    seed: int,
    max_iters: int,
    tol: float,
) -> Dict[str, object]:
    checkpoint = resolve_checkpoint(system)
    model, payload = load_gscd(system, checkpoint, device, max_iters=max_iters, tol=tol)
    states = sample_states(system, samples, device)
    dt = float(SYSTEM_DT[system])
    start = time.time()
    rho, counts, converged = residual_stage_audit(model, states, dt)
    eps_omega, eps_fb = symplectic_and_fb_audit(
        model, states, dt, pairs=pairs, fd_eps=fd_eps, seed=seed
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - start
    count_arr = counts.detach().cpu().numpy().astype(float)
    return {
        "system": system,
        "checkpoint": str(checkpoint.relative_to(REPO)),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_val_loss": float(payload.get("val_loss", float("nan"))),
        "checkpoint_saved_implicit_max_iters": payload.get("args", {}).get("implicit_max_iters"),
        "checkpoint_saved_implicit_tol": payload.get("args", {}).get("implicit_tol"),
        "training_solver_default_if_unsaved": {"implicit_max_iters": 5, "implicit_tol": 1e-6},
        "evaluation_role": "fixed-checkpoint strict-control audit; no retraining",
        "dt": dt,
        "samples": int(states.shape[0]),
        "random_pairs_per_sample": int(pairs),
        "finite_difference_eps": float(fd_eps),
        "implicit_max_iters": int(max_iters),
        "implicit_tol": float(tol),
        "max_rho_F": summarize(rho)["max"],
        "p95_rho_F": summarize(rho)["p95"],
        "median_rho_F": summarize(rho)["median"],
        "cap_reached_count": int((counts >= max_iters).sum().item()),
        "converged_count": int(converged.sum().item()),
        "median_iterations": float(np.median(count_arr)),
        "max_iterations": int(np.max(count_arr)),
        "max_epsilon_omega": summarize(eps_omega)["max"],
        "median_epsilon_omega": summarize(eps_omega)["median"],
        "max_epsilon_fb": summarize(eps_fb)["max"],
        "median_epsilon_fb": summarize(eps_fb)["median"],
        "elapsed_seconds": float(elapsed),
    }


def write_markdown(rows: Iterable[Dict[str, object]], path: Path) -> None:
    labels = {
        "double_pendulum": "Double pendulum",
        "spherical_pendulum": "Spherical pendulum",
    }
    lines = [
        "# Finite Solve Symplecticity Audit",
        "",
        "The table audits the deployed finite fixed-point realization of the nonseparable GSC-HNN branch.",
        "",
        "| System | Checkpoint | Samples | max rho_F | median iters | max epsilon_omega | max epsilon_fb |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {system} | `{checkpoint}` | {samples} | {rho} | {iters:.1f} | {omega} | {fb} |".format(
                system=labels.get(str(row["system"]), str(row["system"])),
                checkpoint=row["checkpoint"],
                samples=int(row["samples"]),
                rho=fmt(float(row["max_rho_F"])),
                iters=float(row["median_iterations"]),
                omega=fmt(float(row["max_epsilon_omega"])),
                fb=fmt(float(row["max_epsilon_fb"])),
            )
        )
    lines.extend(
        [
            "",
            "Definitions: rho_F is the normalized implicit midpoint residual of the residual substep; "
            "epsilon_omega is the randomized symplectic form residual of the full terminated single step map; "
            "epsilon_fb is the forward backward residual of the full terminated single step map. "
            "Both map defects are unnormalized, matching the manuscript definitions.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(rows: Iterable[Dict[str, object]], path: Path) -> None:
    labels = {
        "double_pendulum": "Double pendulum",
        "spherical_pendulum": "Spherical pendulum",
    }
    lines = [
        "% Auto-generated by scripts/audit_finite_solve_symplecticity.py",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{\\textbf{Finite solve audit for the nonseparable branch.} "
        "The indicators are computed on the GSC-HNN checkpoints using the deployed fixed-point midpoint realization. "
        "Here $\\rho_F$ is the normalized nonlinear residual, $\\epsilon_\\omega$ is the randomized symplectic form residual, "
        "and $\\epsilon_{\\mathrm{fb}}$ is the forward backward residual.}",
        "\\label{tab:finite_solve_audit}",
        "\\footnotesize",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "System & max $\\rho_F$ & median iters & max $\\epsilon_\\omega$ & max $\\epsilon_{\\mathrm{fb}}$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            "{system} & {rho} & {iters:.1f} & {omega} & {fb} \\\\".format(
                system=labels.get(str(row["system"]), str(row["system"])),
                rho=fmt_tex(float(row["max_rho_F"])),
                iters=float(row["median_iterations"]),
                omega=fmt_tex(float(row["max_epsilon_omega"])),
                fb=fmt_tex(float(row["max_epsilon_fb"])),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--fd-eps", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", default=str(REPO / "results" / "finite_solve_symplecticity_audit.json"))
    parser.add_argument("--out-md", default=str(REPO / "results" / "finite_solve_symplecticity_audit.md"))
    parser.add_argument("--out-tex", default=str(REPO / "paper" / "finite_solve_audit_table.tex"))
    args = parser.parse_args()

    device = torch.device(args.device)
    rows = []
    for system in args.systems:
        print(f"[audit] {system} on {device}")
        rows.append(
            audit_system(
                system,
                device=device,
                samples=args.samples,
                pairs=args.pairs,
                fd_eps=args.fd_eps,
                seed=args.seed,
                max_iters=args.max_iters,
                tol=args.tol,
            )
        )

    payload = {
        "generated_by": "scripts/audit_finite_solve_symplecticity.py",
        "generated_at_unix": time.time(),
        "device": str(device),
        "map_defect_normalization": "none",
        "primary_training_solver_default": {"implicit_max_iters": 5, "implicit_tol": 1e-6},
        "audit_is_retraining": False,
        "rows": rows,
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_tex = Path(args.out_tex)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(rows, out_md)
    write_latex(rows, out_tex)
    print(f"[audit] wrote {out_json}")
    print(f"[audit] wrote {out_md}")
    print(f"[audit] wrote {out_tex}")


if __name__ == "__main__":
    main()
