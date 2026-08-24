import torch
import numpy as np
from scipy.integrate import solve_ivp
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

M1, M2, L1, L2, G = 1.0, 1.0, 1.0, 1.0, 9.81
SEED = 20260810

def get_hamiltonian(y):
    th1, th2, p1, p2 = y
    c, s = np.cos(th1 - th2), np.sin(th1 - th2)
    det = 1 + s**2
    m11 = 1 / det
    m12 = -c / det
    m22 = 2 / det
    T = 0.5 * (m11*p1**2 + 2*m12*p1*p2 + m22*p2**2)
    V = -( (M1+M2)*G*L1*np.cos(th1) + M2*G*L2*np.cos(th2) )
    return T + V

def vector_field(t, y):
    th1, th2, p1, p2 = y
    c, s = np.cos(th1 - th2), np.sin(th1 - th2)
    det = 1 + s**2
    
    m11 = 1 / det
    m12 = -c / det
    m22 = 2 / det
    dth1 = m11*p1 + m12*p2
    dth2 = m12*p1 + m22*p2
    
    dT_ddelta = ( p1*p2*s*det - (0.5*p1**2 - p1*p2*c + p2**2)*2*s*c ) / det**2
                
    dp1 = -( (M1+M2)*G*L1*np.sin(th1) + dT_ddelta )
    dp2 = -( M2*G*L2*np.sin(th2) - dT_ddelta )
    
    return [dth1, dth2, dp1, dp2]

def generate_trajectories(num=100, steps=1000, dt=0.05, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    print(f">>> Integrating {num} Double Pendulums for {steps} steps (dt={dt})...")
    data = []
    attempts = 0
    max_attempts = max(10 * num, num + 10)
    while len(data) < num and attempts < max_attempts:
        attempts += 1
        th1 = rng.uniform(-np.pi, np.pi)
        th2 = rng.uniform(-np.pi, np.pi)
        p1, p2 = 0.0, 0.0
        y0 = [th1, th2, p1, p2]
        
        t_span = (0, steps * dt)
        t_eval = np.linspace(0, steps * dt, steps + 1)
        
        sol = solve_ivp(vector_field, t_span, y0, t_eval=t_eval, method='LSODA', atol=1e-12, rtol=1e-12)
        
        if sol.status == 0:
            data.append(sol.y.T)
        if len(data) % 50 == 0 and sol.status == 0:
            E0 = get_hamiltonian(y0)
            E1 = get_hamiltonian(sol.y.T[-1])
            print(f"    Traj {len(data)}/{num}, E0: {E0:.4f}, dE: {np.abs(E1-E0):.2e}")

    if len(data) != num:
        raise RuntimeError(
            f"Generated only {len(data)}/{num} successful trajectories "
            f"after {attempts} attempts."
        )
    return torch.tensor(np.stack(data), dtype=torch.float64)

if __name__ == "__main__":
    os.makedirs("data/double_pendulum", exist_ok=True)
    rng = np.random.default_rng(SEED)
    train = generate_trajectories(num=1024, steps=100, dt=0.05, rng=rng)
    torch.save(train, "data/double_pendulum/train.pt")
    val = generate_trajectories(num=128, steps=100, dt=0.05, rng=rng)
    torch.save(val, "data/double_pendulum/val.pt")
    test = generate_trajectories(num=10, steps=1000, dt=0.05, rng=rng)
    torch.save(test, "data/double_pendulum/test.pt")
    print("Double Pendulum Data Ready.")
