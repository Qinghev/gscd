import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


def make_J(n: int, device=None, dtype=torch.float64) -> Tensor:
    J = torch.zeros(2*n, 2*n, device=device, dtype=dtype)
    J[:n, n:] =  torch.eye(n, device=device, dtype=dtype)
    J[n:, :n] = -torch.eye(n, device=device, dtype=dtype)
    return J


def rk4_step(f, u: Tensor, dt: float) -> Tensor:
    k1 = f(u)
    k2 = f(u + 0.5*dt*k1)
    k3 = f(u + 0.5*dt*k2)
    k4 = f(u + dt*k3)
    return u + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)


class SymplecticLinear(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6,
                 sparsity: str = 'dense',
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        assert dim % 2 == 0, "dim must be even (2n)"
        self.n   = dim // 2
        self.dim = dim
        self.eps = eps
        self.sparsity = sparsity
        self.dtype = dtype

        R_init = torch.eye(dim, dtype=dtype) * 0.1
        self.R_raw = nn.Parameter(R_init)

    @property
    def R(self) -> Tensor:
        R = torch.triu(self.R_raw)
        if self.sparsity == 'banded':
            mask = torch.eye(self.dim, device=R.device) + \
                   torch.diag(torch.ones(self.dim-1, device=R.device), diagonal=1)
            R = R * mask.to(self.dtype)
        return R

    @property
    def M(self) -> Tensor:
        R = self.R
        return R.T @ R + self.eps * torch.eye(self.dim,
                                               device=R.device, dtype=self.dtype)

    @property
    def L(self) -> Tensor:
        J = make_J(self.n, device=self.R.device, dtype=self.dtype)
        return J @ self.M

    def cayley_map(self, h: float) -> Tensor:
        L  = self.L
        I  = torch.eye(self.dim, device=L.device, dtype=self.dtype)
        A  = I - (h / 2.0) * L
        B  = I + (h / 2.0) * L
        Phi = torch.linalg.solve(A, B)
        return Phi

    def forward(self, u: Tensor, h: float) -> Tensor:
        Phi = self.cayley_map(h)
        return u @ Phi.T

    def symplectic_error(self) -> Tensor:
        Phi = self.cayley_map(1.0)
        J   = make_J(self.n, device=Phi.device, dtype=self.dtype)
        err = Phi.T @ J @ Phi - J
        return err.norm()


class SymmetricLinear(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        assert dim % 2 == 0, "dim must be even (2n)"
        self.n = dim // 2
        self.dim = dim
        self.eps = eps
        self.dtype = dtype
        self.M_raw = nn.Parameter(torch.eye(dim, dtype=dtype) * 0.1)

    @property
    def M(self) -> Tensor:
        sym = 0.5 * (self.M_raw + self.M_raw.T)
        return sym + self.eps * torch.eye(self.dim, device=sym.device, dtype=self.dtype)

    @property
    def L(self) -> Tensor:
        J = make_J(self.n, device=self.M_raw.device, dtype=self.dtype)
        return J @ self.M

    def cayley_map(self, h: float) -> Tensor:
        L = self.L
        I = torch.eye(self.dim, device=L.device, dtype=self.dtype)
        A = I - (h / 2.0) * L
        B = I + (h / 2.0) * L
        return torch.linalg.solve(A, B)

    def forward(self, u: Tensor, h: float) -> Tensor:
        Phi = self.cayley_map(h)
        return u @ Phi.T

    def symplectic_error(self) -> Tensor:
        Phi = self.cayley_map(1.0)
        J = make_J(self.n, device=Phi.device, dtype=self.dtype)
        err = Phi.T @ J @ Phi - J
        return err.norm()


class FreeLinearCayley(nn.Module):

    def __init__(self, dim: int, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.dim = dim
        self.n = dim // 2
        self.dtype = dtype
        self.A_raw = nn.Parameter(torch.eye(dim, dtype=dtype) * 0.1)

    @property
    def L(self) -> Tensor:
        return self.A_raw

    def cayley_map(self, h: float) -> Tensor:
        L = self.L
        I = torch.eye(self.dim, device=L.device, dtype=self.dtype)
        A = I - (h / 2.0) * L
        B = I + (h / 2.0) * L
        return torch.linalg.solve(A, B)

    def forward(self, u: Tensor, h: float) -> Tensor:
        Phi = self.cayley_map(h)
        return u @ Phi.T

    def symplectic_error(self) -> Tensor:
        Phi = self.cayley_map(1.0)
        J = make_J(self.n, device=Phi.device, dtype=self.dtype)
        err = Phi.T @ J @ Phi - J
        return err.norm()


class ExpSymplecticLinear(SymplecticLinear):

    def forward(self, u: Tensor, h: float) -> Tensor:
        Phi = torch.matrix_exp(h * self.L)
        return u @ Phi.T

    def symplectic_error(self) -> Tensor:
        Phi = torch.matrix_exp(self.L)
        J = make_J(self.n, device=Phi.device, dtype=self.dtype)
        err = Phi.T @ J @ Phi - J
        return err.norm()


class HNN_Residual(nn.Module):

    def __init__(self, n_dof: int, hidden: int = 128, layers: int = 3,
                 separable: bool = True, implicit_max_iters: int = 5,
                 implicit_tol: float = 1e-6,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        self.n_dof = n_dof
        self.separable = separable
        self.implicit_max_iters = implicit_max_iters
        self.implicit_tol = implicit_tol
        self.dtype = dtype

        in_dim = n_dof if separable else 2*n_dof
        net_layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net_layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        net_layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*net_layers).to(dtype=dtype)

    def hamiltonian(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)

    def vector_field(self, u: Tensor) -> Tensor:
        if self.separable:
            n = self.n_dof
            if self.training:
                q = u[:, :n]
                if not q.requires_grad:
                    q = q.requires_grad_(True)
                V_q = self.hamiltonian(q).sum()
                grad_q = torch.autograd.grad(V_q, q, create_graph=True)[0]
            else:
                with torch.enable_grad():
                    q = u[:, :n].detach().requires_grad_(True)
                    V_q = self.hamiltonian(q).sum()
                    grad_q = torch.autograd.grad(V_q, q, create_graph=False)[0]
            zeros = torch.zeros_like(grad_q)
            return torch.cat([zeros, -grad_q], dim=-1)

        if self.training:
            u_req = u if u.requires_grad else u.requires_grad_(True)
            H = self.hamiltonian(u_req).sum()
            grad = torch.autograd.grad(H, u_req, create_graph=True)[0]
        else:
            with torch.enable_grad():
                u_req = u.detach().requires_grad_(True)
                H = self.hamiltonian(u_req).sum()
                grad = torch.autograd.grad(H, u_req, create_graph=False)[0]
        J = make_J(self.n_dof, device=u.device, dtype=self.dtype)
        return grad @ J.T

    def forward(self, u: Tensor, dt: float) -> Tensor:
        if self.separable:
            n = self.n_dof
            if self.training:
                q = u[:, :n]
                if not q.requires_grad: q = q.requires_grad_(True)
                V_q = self.hamiltonian(q).sum()
                grad_q = torch.autograd.grad(V_q, q, create_graph=True)[0]
            else:
                with torch.enable_grad():
                    q = u[:, :n].detach().requires_grad_(True)
                    V_q = self.hamiltonian(q).sum()
                    grad_q = torch.autograd.grad(V_q, q, create_graph=False)[0]
            p_new = u[:, n:] - dt * grad_q
            return torch.cat([u[:, :n], p_new], dim=-1)
        else:
            def get_field(x):
                with torch.enable_grad():
                    xt = x.detach().requires_grad_(True) if not self.training else x
                    H = self.hamiltonian(xt).sum()
                    grad = torch.autograd.grad(H, xt, create_graph=self.training)[0]
                    J = make_J(self.n_dof, device=u.device, dtype=self.dtype)
                    return grad @ J.T

            f0 = get_field(u)
            u_next = u + dt * f0
            active = torch.ones(u.shape[0], dtype=torch.bool, device=u.device)
            
            max_iters = self.implicit_max_iters
            tol = self.implicit_tol
            for _ in range(max_iters):
                u_mid = (u + u_next) / 2.0
                f_mid = get_field(u_mid)
                u_next_new = u + dt * f_mid
                
                diff = torch.norm(u_next_new - u_next, dim=-1)
                u_next = torch.where(active.unsqueeze(-1), u_next_new, u_next)
                active = active & (diff >= tol)
                if not bool(active.any()):
                    break
            
            return u_next


class GSCD_NoSplit(nn.Module):

    def __init__(self, dim: int, hidden: int = 128, layers: int = 3,
                 sparsity: str = 'dense', separable: bool = True,
                 eps: float = 1e-6, dtype: torch.dtype = torch.float64,
                 linear_variant: str = "spd_cayley",
                 implicit_max_iters: int = 5,
                 implicit_tol: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.n_dof = dim // 2
        self.dtype = dtype
        self.linear_variant = linear_variant

        if linear_variant == "spd_cayley":
            self.linear_part = SymplecticLinear(dim, eps=eps, sparsity=sparsity, dtype=dtype)
        elif linear_variant == "sym_cayley":
            self.linear_part = SymmetricLinear(dim, eps=eps, dtype=dtype)
        elif linear_variant == "free_cayley":
            self.linear_part = FreeLinearCayley(dim, dtype=dtype)
        elif linear_variant == "spd_exp":
            self.linear_part = ExpSymplecticLinear(dim, eps=eps, sparsity=sparsity, dtype=dtype)
        else:
            raise ValueError(f"Unknown linear_variant: {linear_variant}")
        self.nonlin_part = HNN_Residual(
            self.n_dof,
            hidden=hidden,
            layers=layers,
            separable=separable,
            implicit_max_iters=implicit_max_iters,
            implicit_tol=implicit_tol,
            dtype=dtype,
        )

    def vector_field(self, u: Tensor) -> Tensor:
        linear = u @ self.linear_part.L.T
        nonlinear = self.nonlin_part.vector_field(u)
        return linear + nonlinear

    def step(self, u: Tensor, dt: float) -> Tensor:
        return rk4_step(self.vector_field, u, dt)

    def rollout(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        traj = [u0]
        u = u0
        for _ in range(steps):
            u = self.step(u, dt)
            traj.append(u)
        return torch.stack(traj, dim=0)

    def forward(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        return self.rollout(u0, dt, steps)


class GSCD_Integrator(nn.Module):

    def __init__(self, dim: int, hidden: int = 128, layers: int = 3,
                 sparsity: str = 'dense', separable: bool = True,
                 eps: float = 1e-6, dtype: torch.dtype = torch.float64,
                 linear_variant: str = "spd_cayley",
                 implicit_max_iters: int = 5,
                 implicit_tol: float = 1e-6):
        super().__init__()
        self.dim   = dim
        self.n_dof = dim // 2
        self.dtype = dtype
        self.linear_variant = linear_variant

        if linear_variant == "spd_cayley":
            self.linear_part = SymplecticLinear(dim, eps=eps, sparsity=sparsity, dtype=dtype)
        elif linear_variant == "sym_cayley":
            self.linear_part = SymmetricLinear(dim, eps=eps, dtype=dtype)
        elif linear_variant == "free_cayley":
            self.linear_part = FreeLinearCayley(dim, dtype=dtype)
        elif linear_variant == "spd_exp":
            self.linear_part = ExpSymplecticLinear(dim, eps=eps, sparsity=sparsity, dtype=dtype)
        else:
            raise ValueError(f"Unknown linear_variant: {linear_variant}")
        self.nonlin_part  = HNN_Residual(
            self.n_dof, hidden=hidden,
            layers=layers, separable=separable,
            implicit_max_iters=implicit_max_iters,
            implicit_tol=implicit_tol,
            dtype=dtype,
        )

    def step(self, u: Tensor, dt: float) -> Tensor:
        u = self.linear_part(u, dt / 2.0)
        u = self.nonlin_part(u, dt)
        u = self.linear_part(u, dt / 2.0)
        return u

    def rollout(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        traj = [u0]
        u = u0
        for _ in range(steps):
            u = self.step(u, dt)
            traj.append(u)
        return torch.stack(traj, dim=0)

    def forward(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        return self.rollout(u0, dt, steps)


class NeuralODE(nn.Module):

    def __init__(self, dim: int, hidden: int = 128, layers: int = 3,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        net = [nn.Linear(dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*net).to(dtype=dtype)
        self.dtype = dtype

    def vector_field(self, u: Tensor) -> Tensor:
        return self.net(u)

    def step(self, u: Tensor, dt: float) -> Tensor:
        return rk4_step(self.vector_field, u, dt)

    def rollout(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        traj = [u0]
        u = u0
        for _ in range(steps):
            u = self.step(u, dt)
            traj.append(u)
        return torch.stack(traj, dim=0)

    def forward(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        return self.rollout(u0, dt, steps)


class SCD(nn.Module):

    def __init__(self, dim: int, hidden: int = 128, layers: int = 3,
                 margin: float = 0.01, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.dim    = dim
        self.margin = margin
        self.dtype  = dtype
        self.A_raw = nn.Parameter(torch.randn(dim, dim, dtype=dtype) * 0.1)
        net = [nn.Linear(dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*net).to(dtype=dtype)

    @property
    def L_stable(self) -> Tensor:
        A = self.A_raw
        return A - A.T - self.margin * torch.eye(self.dim,
                                                  device=A.device, dtype=self.dtype)

    def vector_field(self, u: Tensor) -> Tensor:
        L = self.L_stable
        return u @ L.T + self.net(u)

    def step(self, u: Tensor, dt: float) -> Tensor:
        return rk4_step(self.vector_field, u, dt)

    def rollout(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        traj = [u0]
        u = u0
        for _ in range(steps):
            u = self.step(u, dt)
            traj.append(u)
        return torch.stack(traj, dim=0)

    def forward(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        return self.rollout(u0, dt, steps)


class HNN(nn.Module):

    def __init__(self, dim: int, hidden: int = 128, layers: int = 3,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        net = [nn.Linear(dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net += [nn.Linear(hidden, 1)]
        self.net   = nn.Sequential(*net).to(dtype=dtype)
        self.dtype = dtype
        self.dim   = dim
        self.n     = dim // 2

    def hamiltonian(self, u: Tensor) -> Tensor:
        return self.net(u).squeeze(-1)

    def vector_field(self, u: Tensor) -> Tensor:
        if self.training:
            u_req = u if u.requires_grad else u.requires_grad_(True)
            H     = self.hamiltonian(u_req).sum()
            grad  = torch.autograd.grad(H, u_req, create_graph=True)[0]
        else:
            with torch.enable_grad():
                u_req = u.detach().requires_grad_(True)
                H     = self.hamiltonian(u_req).sum()
                grad  = torch.autograd.grad(H, u_req, create_graph=False)[0]
        J     = make_J(self.n, device=u.device, dtype=self.dtype)
        return grad @ J.T

    def step(self, u: Tensor, dt: float) -> Tensor:
        return rk4_step(self.vector_field, u, dt)

    def rollout(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        traj = [u0]
        u = u0
        for _ in range(steps):
            u = self.step(u, dt)
            traj.append(u)
        return torch.stack(traj, dim=0)

    def forward(self, u0, dt, steps):
        return self.rollout(u0, dt, steps)


class HNN_Symp(nn.Module):

    def __init__(self, dim: int, hidden: int = 128, layers: int = 3,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        n = dim // 2
        self.n = n
        self.dtype = dtype
        def mlp(in_d):
            net = [nn.Linear(in_d, hidden), nn.Tanh()]
            for _ in range(layers - 1):
                net += [nn.Linear(hidden, hidden), nn.Tanh()]
            net += [nn.Linear(hidden, 1)]
            return nn.Sequential(*net).to(dtype=dtype)
        self.V_net = mlp(n)
        self.T_net = mlp(n)

    def V(self, q: Tensor) -> Tensor:
        return self.V_net(q).squeeze(-1)

    def T(self, p: Tensor) -> Tensor:
        return self.T_net(p).squeeze(-1)

    def step(self, u: Tensor, dt: float) -> Tensor:
        n   = self.n
        if self.training:
            q   = u[:, :n]; p = u[:, n:]
            if not q.requires_grad: q = q.requires_grad_(True)
            dV  = torch.autograd.grad(self.V(q).sum(), q, create_graph=True)[0]
            p_h = p - (dt/2) * dV
            
            if not p_h.requires_grad: p_h = p_h.requires_grad_(True)
            dT  = torch.autograd.grad(self.T(p_h).sum(), p_h, create_graph=True)[0]
            q_n = q + dt * dT
            
            if not q_n.requires_grad: q_n = q_n.requires_grad_(True)
            dV2  = torch.autograd.grad(self.V(q_n).sum(), q_n, create_graph=True)[0]
            p_n  = p_h - (dt/2) * dV2
        else:
            with torch.enable_grad():
                q   = u[:, :n].detach().requires_grad_(True)
                p   = u[:, n:].detach().requires_grad_(True)
                dV  = torch.autograd.grad(self.V(q).sum(), q, create_graph=False)[0]
                p_h = p - (dt/2) * dV
                
                p_r = p_h.detach().requires_grad_(True)
                dT  = torch.autograd.grad(self.T(p_r).sum(), p_r, create_graph=False)[0]
                q_n = u[:, :n] + dt * dT
                
                q_r2 = q_n.detach().requires_grad_(True)
                dV2  = torch.autograd.grad(self.V(q_r2).sum(), q_r2, create_graph=False)[0]
                p_n  = p_h - (dt/2) * dV2

        return torch.cat([q_n, p_n], dim=-1)

    def rollout(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        traj = [u0]
        u = u0
        for _ in range(steps):
            u = self.step(u, dt)
            traj.append(u)
        return torch.stack(traj, dim=0)

    def forward(self, u0, dt, steps):
        return self.rollout(u0, dt, steps)


class SympNetLite(nn.Module):

    def __init__(self, dim: int, hidden: int = 128, layers: int = 2,
                 blocks: int = 4, dtype: torch.dtype = torch.float64):
        super().__init__()
        assert dim % 2 == 0, "dim must be even (2n)"
        self.dim = dim
        self.n = dim // 2
        self.blocks = blocks
        self.dtype = dtype

        def mlp():
            net = [nn.Linear(self.n, hidden), nn.Tanh()]
            for _ in range(max(0, layers - 1)):
                net += [nn.Linear(hidden, hidden), nn.Tanh()]
            net += [nn.Linear(hidden, self.n)]
            return nn.Sequential(*net).to(dtype=dtype)

        self.f_blocks = nn.ModuleList([mlp() for _ in range(blocks)])
        self.g_blocks = nn.ModuleList([mlp() for _ in range(blocks)])

    def step(self, u: Tensor, dt: float) -> Tensor:
        q = u[:, :self.n]
        p = u[:, self.n:]
        h = dt / float(self.blocks)
        for f_net, g_net in zip(self.f_blocks, self.g_blocks):
            p = p + h * f_net(q)
            q = q + h * g_net(p)
        return torch.cat([q, p], dim=-1)

    def rollout(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        traj = [u0]
        u = u0
        for _ in range(steps):
            u = self.step(u, dt)
            traj.append(u)
        return torch.stack(traj, dim=0)

    def forward(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        return self.rollout(u0, dt, steps)


class _ScalarPotential(nn.Module):

    def __init__(self, dim: int, hidden: int = 128, layers: int = 2,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        net = [nn.Linear(dim, hidden), nn.Tanh()]
        for _ in range(max(0, layers - 1)):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*net).to(dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class GradientSympNet(nn.Module):

    def __init__(self, dim: int, hidden: int = 128, layers: int = 2,
                 blocks: int = 4, dtype: torch.dtype = torch.float64):
        super().__init__()
        assert dim % 2 == 0, "dim must be even (2n)"
        self.dim = dim
        self.n = dim // 2
        self.blocks = blocks
        self.dtype = dtype
        self.V_blocks = nn.ModuleList([
            _ScalarPotential(self.n, hidden=hidden, layers=layers, dtype=dtype)
            for _ in range(blocks)
        ])
        self.T_blocks = nn.ModuleList([
            _ScalarPotential(self.n, hidden=hidden, layers=layers, dtype=dtype)
            for _ in range(blocks)
        ])

    def _potential_grad(self, potential: nn.Module, x: Tensor) -> Tensor:
        if self.training:
            value = potential(x).sum()
            return torch.autograd.grad(value, x, create_graph=True, retain_graph=True)[0]
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            value = potential(x_req).sum()
            grad = torch.autograd.grad(value, x_req, create_graph=False)[0]
        return grad.detach()

    def step(self, u: Tensor, dt: float) -> Tensor:
        if self.training and not u.requires_grad:
            u = u.requires_grad_(True)
        q = u[:, :self.n]
        p = u[:, self.n:]
        h = dt / float(self.blocks)
        for V_net, T_net in zip(self.V_blocks, self.T_blocks):
            p = p - h * self._potential_grad(V_net, q)
            q = q + h * self._potential_grad(T_net, p)
        return torch.cat([q, p], dim=-1)

    def rollout(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        traj = [u0]
        u = u0
        for _ in range(steps):
            u = self.step(u, dt)
            traj.append(u)
        return torch.stack(traj, dim=0)

    def forward(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        return self.rollout(u0, dt, steps)


class SRNNLite(HNN_Symp):

    def __init__(self, dim: int, hidden: int = 128, layers: int = 3,
                 dtype: torch.dtype = torch.float64):
        super().__init__(dim=dim, hidden=hidden, layers=layers, dtype=dtype)
        self.dim = dim
        corr_layers = [nn.Linear(dim, hidden), nn.Tanh()]
        for _ in range(max(1, layers - 1)):
            corr_layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        corr_layers += [nn.Linear(hidden, dim)]
        self.correction_head = nn.Sequential(*corr_layers).to(dtype=dtype)
        final = self.correction_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def prepare_initial_state(self, u0: Tensor) -> Tensor:
        return u0 + self.correction_head(u0)

    def rollout(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        return super().rollout(u0, dt, steps)

    def forward(self, u0, dt, steps):
        return super().forward(u0, dt, steps)


class HNN_Implicit(nn.Module):

    def __init__(self, dim: int, hidden: int = 128, layers: int = 3,
                 implicit_max_iters: int = 5, implicit_tol: float = 1e-6,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        self.dim = dim
        self.n = dim // 2
        self.dtype = dtype
        self.implicit_max_iters = implicit_max_iters
        self.implicit_tol = implicit_tol
        
        net = [nn.Linear(dim, hidden), nn.GELU()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.GELU()]
        net += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*net).to(dtype=dtype)

    def hamiltonian(self, u: Tensor) -> Tensor:
        return self.net(u).squeeze(-1)

    def step(self, u: Tensor, dt: float) -> Tensor:
        def get_field(x):
            with torch.enable_grad():
                xt = x
                if not xt.requires_grad:
                    xt = xt.requires_grad_(True)
                H = self.hamiltonian(xt).sum()
                grad = torch.autograd.grad(H, xt, create_graph=self.training)[0]
                J = make_J(self.n, device=u.device, dtype=self.dtype)
                return grad @ J.T

        f0 = get_field(u)
        u_next = u + dt * f0
        active = torch.ones(u.shape[0], dtype=torch.bool, device=u.device)
        
        max_iters = self.implicit_max_iters
        tol = self.implicit_tol
        for _ in range(max_iters):
            u_mid = (u + u_next) / 2.0
            f_mid = get_field(u_mid)
            u_next_new = u + dt * f_mid
            
            diff = torch.norm(u_next_new - u_next, dim=-1)
            u_next = torch.where(active.unsqueeze(-1), u_next_new, u_next)
            active = active & (diff >= tol)
            if not bool(active.any()):
                break
        
        return u_next

    def rollout(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        traj = [u0]
        u = u0
        for _ in range(steps):
            u = self.step(u, dt)
            traj.append(u)
        return torch.stack(traj, dim=0)

    def forward(self, u0: Tensor, dt: float, steps: int) -> Tensor:
        return self.rollout(u0, dt, steps)
