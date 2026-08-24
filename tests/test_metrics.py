from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_spherical_pendulum_meridian import initial_meridian_section
from run_final_benchmarks import state_difference
from model import HNN_Implicit


def test_double_pendulum_wraps_both_angles() -> None:
    prediction = np.array([np.pi - 0.01, -np.pi + 0.02, 3.0, 4.0])
    reference = np.array([-np.pi + 0.01, np.pi - 0.02, 1.0, 1.0])
    difference = state_difference("double_pendulum", prediction, reference)

    np.testing.assert_allclose(difference[:2], [-0.02, 0.04], atol=1e-14)
    np.testing.assert_allclose(difference[2:], [2.0, 3.0], atol=0.0)


def test_spherical_pendulum_wraps_phi_but_not_theta() -> None:
    prediction = torch.tensor(
        [np.pi - 0.01, np.pi - 0.01, 3.0, 4.0], dtype=torch.float64
    )
    reference = torch.tensor(
        [-np.pi + 0.01, -np.pi + 0.01, 1.0, 1.0], dtype=torch.float64
    )
    difference = state_difference("spherical_pendulum", prediction, reference)

    assert torch.allclose(
        difference,
        torch.tensor(
            [2.0 * np.pi - 0.02, -0.02, 2.0, 3.0], dtype=torch.float64
        ),
        atol=1e-14,
        rtol=0.0,
    )


def test_initial_meridian_is_not_the_global_zero_meridian() -> None:
    phi = np.array([0.7, 3.0, 2.0 * np.pi, 0.7 + 2.0 * np.pi, 8.0])
    trajectory = np.column_stack(
        [
            np.linspace(1.0, 1.4, len(phi)),
            phi,
            np.linspace(0.0, 0.4, len(phi)),
            np.ones(len(phi)),
        ]
    )

    section = initial_meridian_section(trajectory)
    assert section.shape == (1, 2)
    np.testing.assert_allclose(section[0], [1.3, 0.3], atol=1e-14)


def test_hnn_implicit_uses_archived_gelu_activation() -> None:
    model = HNN_Implicit(dim=4, hidden=8, layers=3)
    assert all(isinstance(model.net[index], nn.GELU) for index in (1, 3, 5))
