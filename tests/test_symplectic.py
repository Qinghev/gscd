import sys
sys.path.insert(0, "src")
import torch
import math
from model import (
    GradientSympNet,
    HNN_Implicit,
    HNN_Residual,
    SymplecticLinear,
    make_J,
)

DTYPE = torch.float64
torch.manual_seed(0)


def test_symplectic_linear():
    print("=" * 55)
    print("Test 1: SymplecticLinear — Cayley symplectic condition")
    print("=" * 55)
    for dim in [4, 8, 64]:
        sl = SymplecticLinear(dim, dtype=DTYPE)
        with torch.no_grad():
            sl.R_raw.data = torch.randn(dim, dim, dtype=DTYPE)
        for h in [0.01, 0.1, 0.5]:
            Phi = sl.cayley_map(h)
            J   = make_J(dim // 2, dtype=DTYPE)
            err = (Phi.T @ J @ Phi - J).norm().item()
            status = "✅" if err < 1e-10 else "❌ FAIL"
            print(f"  dim={dim:3d}, h={h:.2f}  ||Phi^T J Phi - J||_F = {err:.3e}  {status}")
            assert err < 1e-10
    print()


def test_M_positive_definite():
    print("=" * 55)
    print("Test 2: SymplecticLinear — M positive definite")
    print("=" * 55)
    for dim in [4, 64]:
        sl = SymplecticLinear(dim, eps=1e-6, dtype=DTYPE)
        with torch.no_grad():
            sl.R_raw.data = torch.randn(dim, dim, dtype=DTYPE)
        M      = sl.M
        eigvals = torch.linalg.eigvalsh(M)
        min_ev  = eigvals.min().item()
        status  = "✅" if min_ev > 0 else "❌ FAIL"
        print(f"  dim={dim:3d}  min eigenvalue of M = {min_ev:.4e}  {status}")
        assert min_ev > 0
    print()


def test_L_in_sp():
    print("=" * 55)
    print("Test 3: L = JM  ∈  sp(2n, R)  i.e. JL + L^T J = 0")
    print("=" * 55)
    for dim in [4, 64]:
        sl = SymplecticLinear(dim, dtype=DTYPE)
        with torch.no_grad():
            sl.R_raw.data = torch.randn(dim, dim, dtype=DTYPE)
        L   = sl.L
        J   = make_J(dim // 2, dtype=DTYPE)
        err = (J @ L + L.T @ J).norm().item()
        status = "✅" if err < 1e-10 else "❌ FAIL"
        print(f"  dim={dim:3d}  ||JL + L^T J||_F = {err:.3e}  {status}")
        assert err < 1e-10
    print()


def test_gradient_flow():
    print("=" * 55)
    print("Test 4: Gradient flows through Cayley step (autograd)")
    print("=" * 55)
    dim = 8
    sl  = SymplecticLinear(dim, dtype=DTYPE)
    u   = torch.randn(4, dim, dtype=DTYPE)
    u_out = sl(u, h=0.1)
    loss  = u_out.sum()
    try:
        loss.backward()
        grad_norm = sl.R_raw.grad.norm().item()
        status = "✅" if not math.isnan(grad_norm) and grad_norm > 0 else "❌ FAIL"
        print(f"  Gradient norm wrt R_raw = {grad_norm:.4e}  {status}")
        assert not math.isnan(grad_norm) and grad_norm > 0
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        raise
    print()


def test_energy_conservation_linear():
    print("=" * 55)
    print("Test 5: Cayley step preserves quadratic Hamiltonian")
    print("=" * 55)
    dim = 4
    n   = dim // 2
    sl  = SymplecticLinear(dim, dtype=DTYPE)
    with torch.no_grad():
        sl.R_raw.data = torch.eye(dim, dtype=DTYPE)

    u   = torch.randn(1, dim, dtype=DTYPE)
    with torch.no_grad():
        M   = sl.M
        H0  = (u @ M @ u.T).squeeze().item() / 2
        Phi = sl.cayley_map(0.1)
        for _ in range(1000):
            u = u @ Phi.T
        H1  = (u @ M @ u.T).squeeze().item() / 2
    err = abs(H1 - H0) / abs(H0)
    status = "✅" if err < 1e-10 else "❌ FAIL"
    print(f"  dH/H0 over 1000 Cayley steps = {err:.3e}  {status}")
    assert err < 1e-10
    print()


def test_gradient_sympnet_is_symplectic():
    print("=" * 55)
    print("Test 6: GradientSympNet shear composition is symplectic")
    print("=" * 55)
    dim = 4
    model = GradientSympNet(dim=dim, hidden=8, layers=1, blocks=2, dtype=DTYPE)
    model.train()
    u0 = torch.randn(dim, dtype=DTYPE, requires_grad=True)
    J = make_J(dim // 2, dtype=DTYPE)

    def one_step(x):
        return model.step(x.unsqueeze(0), 0.05).squeeze(0)

    Phi = torch.autograd.functional.jacobian(one_step, u0)
    err = (Phi.T @ J @ Phi - J).norm().item()
    status = "PASS" if err < 1e-9 else "FAIL"
    print(f"  ||Phi^T J Phi - J||_F = {err:.3e}  {status}")
    assert err < 1e-9
    print()


def test_tolerance_stopping_is_batch_independent():
    torch.manual_seed(7)
    target = torch.tensor([[20.0, -15.0, 10.0, -8.0]], dtype=DTYPE)
    peer = torch.tensor([[0.01, -0.02, 0.03, -0.04]], dtype=DTYPE)

    residual = HNN_Residual(
        n_dof=2,
        hidden=16,
        layers=2,
        separable=False,
        implicit_max_iters=8,
        implicit_tol=1e-5,
        dtype=DTYPE,
    ).eval()
    alone = residual(target.clone(), 2.0)
    batched = residual(torch.cat([target, peer]), 2.0)[:1]
    assert torch.allclose(alone, batched, atol=1e-13, rtol=1e-13)

    implicit = HNN_Implicit(
        dim=4,
        hidden=16,
        layers=2,
        implicit_max_iters=8,
        implicit_tol=1e-5,
        dtype=DTYPE,
    ).eval()
    alone = implicit.step(target.clone(), 2.0)
    batched = implicit.step(torch.cat([target, peer]), 2.0)[:1]
    assert torch.allclose(alone, batched, atol=1e-13, rtol=1e-13)


def test_full_state_branches_start_from_euler_predictor():
    torch.manual_seed(11)
    u = torch.randn(3, 4, dtype=DTYPE)
    dt = 0.07
    J = make_J(2, dtype=DTYPE)

    residual = HNN_Residual(
        n_dof=2,
        hidden=16,
        layers=2,
        separable=False,
        implicit_max_iters=0,
        dtype=DTYPE,
    ).eval()
    u_req = u.clone().requires_grad_(True)
    grad = torch.autograd.grad(residual.hamiltonian(u_req).sum(), u_req)[0]
    expected = u + dt * (grad @ J.T)
    assert torch.allclose(residual(u.clone(), dt), expected, atol=1e-13, rtol=1e-13)

    implicit = HNN_Implicit(
        dim=4,
        hidden=16,
        layers=2,
        implicit_max_iters=0,
        dtype=DTYPE,
    ).eval()
    u_req = u.clone().requires_grad_(True)
    grad = torch.autograd.grad(implicit.hamiltonian(u_req).sum(), u_req)[0]
    expected = u + dt * (grad @ J.T)
    assert torch.allclose(implicit.step(u.clone(), dt), expected, atol=1e-13, rtol=1e-13)

if __name__ == "__main__":
    test_symplectic_linear()
    test_M_positive_definite()
    test_L_in_sp()
    test_gradient_flow()
    test_energy_conservation_linear()
    test_gradient_sympnet_is_symplectic()
    test_tolerance_stopping_is_batch_independent()
    test_full_state_branches_start_from_euler_predictor()
    print("All tests completed.")
