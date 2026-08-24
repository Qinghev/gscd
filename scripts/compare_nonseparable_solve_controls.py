from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audit_finite_solve_symplecticity import (  # noqa: E402
    finite_difference_jvp,
    fmt,
    hamiltonian_field,
    load_gscd,
    resolve_checkpoint,
    sample_states,
    sha256,
    summarize,
    torch_load,
)
from model import GSCD_Integrator, make_J  # noqa: E402
from train import DTYPE, SYSTEM_DT  # noqa: E402


SYSTEMS = ("double_pendulum", "spherical_pendulum")


def field_with_graph(nonlin, x: torch.Tensor, J: torch.Tensor, create_graph: bool) -> torch.Tensor:
    xt = x.detach().clone().requires_grad_(True)
    H = nonlin.hamiltonian(xt.unsqueeze(0)).sum()
    grad = torch.autograd.grad(H, xt, create_graph=create_graph)[0]
    return J @ grad


def residual_vector(nonlin, u: torch.Tensor, v: torch.Tensor, dt: float, J: torch.Tensor, create_graph: bool) -> torch.Tensor:
    mid = (u + v) / 2.0
    return v - u - dt * field_with_graph(nonlin, mid, J, create_graph=create_graph)


def fixed_point_stage(
    model: GSCD_Integrator,
    u: torch.Tensor,
    dt: float,
    max_iters: int,
    tol: float,
) -> Tuple[torch.Tensor, int, float]:
    J = make_J(model.n_dof, device=u.device, dtype=u.dtype)
    v = u + dt * hamiltonian_field(
        model.nonlin_part, u.unsqueeze(0), J
    ).squeeze(0)
    iters = 0
    for _ in range(max_iters):
        mid = (u + v) / 2.0
        f_mid = hamiltonian_field(model.nonlin_part, mid.unsqueeze(0), J).squeeze(0)
        v_new = u + dt * f_mid
        diff = float(torch.linalg.norm(v_new - v).detach().cpu())
        v = v_new
        iters += 1
        if diff < tol:
            break
    F = residual_vector(model.nonlin_part, u, v, dt, J, create_graph=False)
    rho = float((torch.linalg.norm(F) / (1.0 + torch.linalg.norm(u) + torch.linalg.norm(v))).detach().cpu())
    return v.detach(), iters, rho


def newton_stage(
    model: GSCD_Integrator,
    u: torch.Tensor,
    dt: float,
    max_iters: int,
    tol: float,
    step_tol: float,
) -> Tuple[torch.Tensor, int, float, bool]:
    J = make_J(model.n_dof, device=u.device, dtype=u.dtype)
    v, _, _ = fixed_point_stage(model, u, dt, max_iters=2, tol=tol)
    dim = u.numel()
    eye = torch.eye(dim, device=u.device, dtype=u.dtype)
    converged = False
    iters = 0
    for _ in range(max_iters):
        v_req = v.detach().clone().requires_grad_(True)
        F = residual_vector(model.nonlin_part, u.detach(), v_req, dt, J, create_graph=True)
        rho = torch.linalg.norm(F) / (1.0 + torch.linalg.norm(u) + torch.linalg.norm(v_req))
        if float(rho.detach().cpu()) <= tol:
            converged = True
            v = v_req.detach()
            break
        rows = []
        for i in range(dim):
            grad_i = torch.autograd.grad(F[i], v_req, retain_graph=True)[0]
            rows.append(grad_i)
        A = torch.stack(rows, dim=0)
        try:
            delta = torch.linalg.solve(A, -F.detach())
        except RuntimeError:
            delta = torch.linalg.lstsq(A + 1e-12 * eye, -F.detach()).solution
        alpha = 1.0
        base_norm = torch.linalg.norm(F.detach())
        best_v = v_req.detach()
        best_norm = base_norm
        for _ls in range(8):
            candidate = v_req.detach() + alpha * delta
            cand_F = residual_vector(model.nonlin_part, u.detach(), candidate, dt, J, create_graph=False)
            cand_norm = torch.linalg.norm(cand_F)
            if cand_norm <= best_norm:
                best_v = candidate.detach()
                best_norm = cand_norm.detach()
                break
            alpha *= 0.5
        v = best_v
        iters += 1
        rel_step = torch.linalg.norm(alpha * delta) / (1.0 + torch.linalg.norm(v))
        if float(best_norm.detach().cpu()) / float((1.0 + torch.linalg.norm(u) + torch.linalg.norm(v)).detach().cpu()) <= tol and float(rel_step.detach().cpu()) <= step_tol:
            converged = True
            break
    F = residual_vector(model.nonlin_part, u, v, dt, J, create_graph=False)
    rho = float((torch.linalg.norm(F) / (1.0 + torch.linalg.norm(u) + torch.linalg.norm(v))).detach().cpu())
    return v.detach(), iters, rho, converged


def make_step_fn(model: GSCD_Integrator, solver: str, max_iters: int, tol: float) -> Callable[[torch.Tensor, float], torch.Tensor]:
    def step(u: torch.Tensor, dt: float) -> torch.Tensor:
        scalar = u.ndim == 1
        batch = u.unsqueeze(0) if scalar else u
        outs = []
        for row in batch:
            half = model.linear_part(row.unsqueeze(0), dt / 2.0).squeeze(0)
            if solver == "fixed":
                mid, _, _ = fixed_point_stage(model, half, dt, max_iters=max_iters, tol=tol)
            elif solver == "newton":
                mid, _, _, _ = newton_stage(model, half, dt, max_iters=max_iters, tol=tol, step_tol=tol)
            else:
                raise ValueError(solver)
            out = model.linear_part(mid.unsqueeze(0), dt / 2.0).squeeze(0)
            outs.append(out)
        stacked = torch.stack(outs, dim=0)
        return stacked.squeeze(0) if scalar else stacked

    return step


def stage_metrics(
    model: GSCD_Integrator,
    states: torch.Tensor,
    dt: float,
    solver: str,
    max_iters: int,
    tol: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    rhos = []
    counts = []
    converged_count = 0
    for row in states:
        half = model.linear_part(row.unsqueeze(0), dt / 2.0).squeeze(0)
        if solver == "fixed":
            _, iters, rho = fixed_point_stage(model, half, dt, max_iters=max_iters, tol=tol)
            converged_count += int(rho <= max(tol * 10.0, 1e-14))
        else:
            _, iters, rho, converged = newton_stage(model, half, dt, max_iters=max_iters, tol=tol, step_tol=tol)
            converged_count += int(converged)
        rhos.append(rho)
        counts.append(iters)
    return (
        torch.tensor(rhos, dtype=torch.float64),
        torch.tensor(counts, dtype=torch.float64),
        converged_count,
    )


def symp_fb_metrics(
    step_fn: Callable[[torch.Tensor, float], torch.Tensor],
    dim: int,
    states: torch.Tensor,
    dt: float,
    pairs: int,
    fd_eps: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    J = make_J(dim // 2, device=states.device, dtype=states.dtype)
    gen = torch.Generator(device=states.device)
    gen.manual_seed(seed)
    omegas = []
    fbs = []
    for u in states:
        point_max = torch.tensor(0.0, device=states.device, dtype=states.dtype)
        for _ in range(pairs):
            a = torch.randn(dim, generator=gen, device=states.device, dtype=states.dtype)
            b = torch.randn(dim, generator=gen, device=states.device, dtype=states.dtype)
            a = a / torch.linalg.norm(a).clamp_min(1e-30)
            b = b / torch.linalg.norm(b).clamp_min(1e-30)
            da = finite_difference_custom(step_fn, u, dt, a, fd_eps)
            db = finite_difference_custom(step_fn, u, dt, b, fd_eps)
            point_max = torch.maximum(
                point_max,
                torch.abs(da @ (J @ db) - a @ (J @ b)),
            )
        omegas.append(point_max.detach())
        fwd = step_fn(u, dt)
        back = step_fn(fwd, -dt)
        fbs.append(torch.linalg.norm(back - u).detach())
    return torch.stack(omegas), torch.stack(fbs)


def finite_difference_custom(
    step_fn: Callable[[torch.Tensor, float], torch.Tensor],
    u: torch.Tensor,
    dt: float,
    direction: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    norm_d = torch.linalg.norm(direction).clamp_min(1e-30)
    step = eps * (1.0 + torch.linalg.norm(u)) / norm_d
    return (step_fn(u + step * direction, dt) - step_fn(u - step * direction, dt)) / (2.0 * step)


def rollout_mse(step_fn: Callable[[torch.Tensor, float], torch.Tensor], system: str, device: torch.device, steps: int, max_ics: int) -> float:
    data = torch_load(REPO / "data" / system / "test.pt", map_location="cpu")
    data = data[:max_ics, : steps + 1].to(device=device, dtype=DTYPE)
    dt = float(SYSTEM_DT[system])
    u = data[:, 0, :]
    preds = []
    for _ in range(steps):
        u = step_fn(u, dt)
        preds.append(u)
    pred = torch.stack(preds, dim=1)
    return float(torch.mean((pred - data[:, 1:, :]) ** 2).detach().cpu())


def audit_config(
    system: str,
    model: GSCD_Integrator,
    states: torch.Tensor,
    device: torch.device,
    solver: str,
    max_iters: int,
    tol: float,
    pairs: int,
    fd_eps: float,
    seed: int,
    rollout_steps: int,
    rollout_ics: int,
) -> Dict[str, object]:
    dt = float(SYSTEM_DT[system])
    t0 = time.time()
    rho, counts, converged = stage_metrics(model, states, dt, solver, max_iters, tol)
    step_fn = make_step_fn(model, solver, max_iters, tol)
    omega, fb = symp_fb_metrics(step_fn, states.shape[-1], states, dt, pairs, fd_eps, seed)
    mse = rollout_mse(step_fn, system, device, rollout_steps, rollout_ics)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "system": system,
        "solver": solver,
        "max_iters": int(max_iters),
        "tol": float(tol),
        "samples": int(states.shape[0]),
        "converged_count": int(converged),
        "max_rho_F": summarize(rho)["max"],
        "median_rho_F": summarize(rho)["median"],
        "median_iterations": float(torch.median(counts).item()),
        "max_iterations": int(torch.max(counts).item()),
        "max_epsilon_omega": summarize(omega)["max"],
        "median_epsilon_omega": summarize(omega)["median"],
        "max_epsilon_fb": summarize(fb)["max"],
        "median_epsilon_fb": summarize(fb)["median"],
        "rollout_steps": int(rollout_steps),
        "rollout_ics": int(rollout_ics),
        "rollout_mse": float(mse),
        "elapsed_seconds": float(time.time() - t0),
    }


def write_md(rows: List[Dict[str, object]], path: Path) -> None:
    lines = [
        "# Nonseparable Solver-Control Comparison",
        "",
        "| System | Solver | max iters | tol | max rho_F | median iters | max epsilon_omega | max epsilon_fb | rollout MSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {system} | {solver} | {max_iters} | {tol} | {rho} | {iters:.1f} | {omega} | {fb} | {mse} |".format(
                system=row["system"],
                solver=row["solver"],
                max_iters=row["max_iters"],
                tol=fmt(float(row["tol"])),
                rho=fmt(float(row["max_rho_F"])),
                iters=float(row["median_iterations"]),
                omega=fmt(float(row["max_epsilon_omega"])),
                fb=fmt(float(row["max_epsilon_fb"])),
                mse=fmt(float(row["rollout_mse"])),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--fd-eps", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--rollout-ics", type=int, default=4)
    parser.add_argument("--out-json", default=str(REPO / "results" / "nonseparable_solver_control_comparison.json"))
    parser.add_argument("--out-md", default=str(REPO / "results" / "nonseparable_solver_control_comparison.md"))
    args = parser.parse_args()

    device = torch.device(args.device)
    configs = [
        ("fixed", 5, 1e-6),
        ("fixed", 8, 1e-8),
        ("fixed", 12, 1e-10),
        ("fixed", 20, 1e-12),
        ("newton", 8, 1e-10),
    ]
    rows: List[Dict[str, object]] = []
    for system in args.systems:
        checkpoint = resolve_checkpoint(system)
        model, payload = load_gscd(system, checkpoint, device, max_iters=5, tol=1e-6)
        states = sample_states(system, args.samples, device)
        for solver, max_iters, tol in configs:
            print(f"[compare] {system} {solver} max_iters={max_iters} tol={tol:g}")
            row = audit_config(
                system,
                model,
                states,
                device,
                solver,
                max_iters,
                tol,
                args.pairs,
                args.fd_eps,
                args.seed,
                args.rollout_steps,
                args.rollout_ics,
            )
            row["checkpoint"] = str(checkpoint.relative_to(REPO))
            row["checkpoint_sha256"] = sha256(checkpoint)
            row["checkpoint_epoch"] = int(payload.get("epoch", -1))
            row["evaluation_role"] = "fixed-checkpoint solver sensitivity; no retraining"
            rows.append(row)
    payload = {
        "generated_by": "scripts/compare_nonseparable_solve_controls.py",
        "generated_at_unix": time.time(),
        "device": str(device),
        "map_defect_normalization": "none",
        "primary_training_solver_default": {"implicit_max_iters": 5, "implicit_tol": 1e-6},
        "audit_is_retraining": False,
        "rows": rows,
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(rows, out_md)
    print(f"[compare] wrote {out_json}")
    print(f"[compare] wrote {out_md}")


if __name__ == "__main__":
    main()
