"""Shared benchmark utilities used by the multiseed evaluation scripts.

This module intentionally contains no benchmark driver side effects.  It
provides the model reconstruction, Hamiltonians, and endpoint evaluator that
the public evaluation scripts import.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train import DTYPE, SYSTEM_DIM, SYSTEM_DT, build_model as _build_model  # noqa: E402


def build_model(
    name: str,
    dim: int,
    hidden: int = 128,
    sparsity: str = "dense",
    separable: bool = True,
    implicit_max_iters: int = 5,
    implicit_tol: float = 1e-6,
):
    """Build a released model with all solver controls made explicit."""
    return _build_model(
        name,
        dim,
        hidden=hidden,
        sparsity=sparsity,
        separable=separable,
        implicit_max_iters=implicit_max_iters,
        implicit_tol=implicit_tol,
    )


def infer_separable(checkpoint: Dict[str, Any], dim: Optional[int] = None) -> bool:
    """Recover the residual branch from checkpoint arguments.

    ``dim`` is accepted for compatibility with the existing evaluation scripts.
    """
    del dim
    args = checkpoint.get("args", {})
    return not bool(args.get("non_separable", False))


def _split_state(state) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(state, dtype=np.float64)
    n = array.shape[-1] // 2
    return array[..., :n], array[..., n:]


def energy_fn(system: str, state) -> float:
    """Reference Hamiltonian for every active manuscript benchmark."""
    q, p = _split_state(state)

    if system == "fput":
        q_ext = np.concatenate(([0.0], q, [0.0]))
        delta = np.diff(q_ext)
        return float(
            0.5 * np.sum(p**2)
            + 0.5 * np.sum(delta**2)
            + 0.7 / 4.0 * np.sum(delta**4)
        )

    if system == "toda":
        return float(0.5 * np.sum(p**2) + np.sum(np.exp(q - np.roll(q, -1))))

    if system == "phi4":
        delta = np.roll(q, -1) - q
        return float(
            0.5 * np.sum(p**2)
            + 0.5 * np.sum(delta**2)
            + 0.5 * np.sum(q**2)
            + 0.25 * np.sum(q**4)
        )

    if system == "double_pendulum":
        theta1, theta2 = q
        p1, p2 = p
        cosine = np.cos(theta1 - theta2)
        determinant = 1.0 + np.sin(theta1 - theta2) ** 2
        kinetic = (p1**2 - 2.0 * cosine * p1 * p2 + 2.0 * p2**2) / (
            2.0 * determinant
        )
        potential = -9.81 * (2.0 * np.cos(theta1) + np.cos(theta2))
        return float(kinetic + potential)

    if system == "spherical_pendulum":
        theta = q[0]
        p_theta, p_phi = p
        sine = max(abs(float(np.sin(theta))), 1e-8)
        return float(
            0.5 * p_theta**2
            + 0.5 * p_phi**2 / sine**2
            + 1.0
            - np.cos(theta)
        )

    raise KeyError(f"No reference Hamiltonian is defined for system {system!r}")


def _test_path(system: str) -> Path:
    root = REPO / "data" / system
    for name in ("test.pt", "test_long.pt"):
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No test.pt or test_long.pt found under {root}")


def _checkpoint_model_name(path: Path, checkpoint: Dict[str, Any]) -> str:
    args = checkpoint.get("args", {})
    if args.get("model"):
        return str(args["model"])
    if path.parent.name.startswith("seed_"):
        return path.parent.parent.parent.name
    return path.parent.name


PERIODIC_STATE_INDICES = {
    "double_pendulum": (0, 1),
    "spherical_pendulum": (1,),
}


def state_difference(system: str, prediction, reference):
    """Return a state difference with periodic angles on their shortest arcs.

    The function accepts either NumPy arrays or PyTorch tensors.  Keeping this
    operation in the shared benchmark module prevents endpoint evaluators from
    silently using different definitions of pendulum rollout MSE.
    """
    indices = PERIODIC_STATE_INDICES.get(system, ())
    if torch.is_tensor(prediction) or torch.is_tensor(reference):
        difference = prediction - reference
        if indices:
            difference = difference.clone()
            index = torch.as_tensor(
                indices, device=difference.device, dtype=torch.long
            )
            wrapped = torch.remainder(
                difference.index_select(-1, index) + torch.pi, 2 * torch.pi
            ) - torch.pi
            difference.index_copy_(-1, index, wrapped)
        return difference

    difference = np.asarray(prediction) - np.asarray(reference)
    if indices:
        difference = np.array(difference, copy=True)
        difference[..., list(indices)] = (
            difference[..., list(indices)] + np.pi
        ) % (2.0 * np.pi) - np.pi
    return difference


def evaluate_checkpoint(
    system: str,
    checkpoint_path,
    device: str = "cpu",
    max_steps: Optional[int] = None,
    max_ics: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one checkpoint on the released held-out trajectories."""
    checkpoint_path = Path(checkpoint_path)
    torch_device = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=torch_device, weights_only=False
    )
    args = checkpoint.get("args", {})
    test_path = _test_path(system)
    trajectories = torch.load(
        test_path, map_location="cpu", weights_only=False
    ).to(dtype=DTYPE)
    if trajectories.ndim == 2:
        trajectories = trajectories.unsqueeze(0)
    if max_ics is not None:
        trajectories = trajectories[:max_ics]
    if max_steps is not None:
        trajectories = trajectories[:, : max_steps + 1]

    dim = int(trajectories.shape[-1])
    model_name = _checkpoint_model_name(checkpoint_path, checkpoint)
    model = build_model(
        model_name,
        dim,
        hidden=int(args.get("hidden", 128)),
        sparsity=str(args.get("sparsity", "dense")),
        separable=infer_separable(checkpoint, dim),
        implicit_max_iters=int(args.get("implicit_max_iters") or 5),
        implicit_tol=float(args.get("implicit_tol") or 1e-6),
    ).to(device=torch_device, dtype=DTYPE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dt = float(SYSTEM_DT[system])
    squared_error = 0.0
    value_count = 0
    trajectory_drifts = []
    started = time.time()
    for trajectory in trajectories:
        state = trajectory[0].unsqueeze(0).to(torch_device)
        if hasattr(model, "prepare_initial_state"):
            state = model.prepare_initial_state(state)
        initial_energy = energy_fn(system, state.squeeze(0).detach().cpu().numpy())
        max_drift = 0.0
        with torch.no_grad():
            for index, reference in enumerate(trajectory):
                prediction = state.squeeze(0).detach().cpu()
                difference = state_difference(system, prediction, reference)
                squared_error += float(torch.sum(difference**2))
                value_count += difference.numel()
                energy = energy_fn(system, prediction.numpy())
                max_drift = max(
                    max_drift,
                    abs((energy - initial_energy) / (initial_energy + 1e-12)),
                )
                if index + 1 < len(trajectory):
                    state = model.step(state, dt)
        trajectory_drifts.append(max_drift)

    return {
        "system": system,
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "test_path": str(test_path),
        "num_ics": int(len(trajectories)),
        "steps": int(trajectories.shape[1] - 1),
        "periodic_state_indices": list(PERIODIC_STATE_INDICES.get(system, ())),
        "periodic_difference_interval": "[-pi, pi)",
        "mse": squared_error / max(value_count, 1),
        "max_energy_drift": float(np.mean(trajectory_drifts)),
        "eval_seconds": float(time.time() - started),
    }


def format_value(value) -> str:
    return "--" if value is None else f"{float(value):.4e}"


__all__ = [
    "DTYPE",
    "SYSTEM_DIM",
    "SYSTEM_DT",
    "build_model",
    "energy_fn",
    "evaluate_checkpoint",
    "format_value",
    "infer_separable",
    "PERIODIC_STATE_INDICES",
    "state_difference",
]
