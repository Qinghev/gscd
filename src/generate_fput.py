import numpy as np
import torch
import json
import os
import time

SAVE_DIR = "data/fput"
os.makedirs(SAVE_DIR, exist_ok=True)

N_PARTICLES = 32
BETA        = 0.7

DT       = 0.1          
H_GT     = 0.001        
SUBSTEPS = int(round(DT / H_GT))   
N_TRAIN_STEPS = 500
N_IC          = 128
N_TEST_STEPS  = 100_000
SEED = 20260810

def fput_force(q: np.ndarray) -> np.ndarray:
    N = len(q)
    q_ext = np.empty(N + 2, dtype=q.dtype)
    q_ext[0] = 0.0; q_ext[1:-1] = q; q_ext[-1] = 0.0
    delta = q_ext[1:] - q_ext[:-1]
    f_spring = delta + BETA * delta**3
    return f_spring[1:] - f_spring[:-1]

def fput_hamiltonian(q: np.ndarray, p: np.ndarray) -> float:
    N = len(q)
    q_ext = np.empty(N + 2, dtype=q.dtype)
    q_ext[0] = 0.0; q_ext[1:-1] = q; q_ext[-1] = 0.0
    delta = q_ext[1:] - q_ext[:-1]
    V = 0.5 * np.sum(delta**2) + (BETA / 4.0) * np.sum(delta**4)
    T = 0.5 * np.sum(p**2)
    return T + V

def störmer_verlet_step(q, p, h):
    p_half = p + (h / 2.0) * fput_force(q)
    q_next = q + h * p_half
    p_next = p_half + (h / 2.0) * fput_force(q_next)
    return q_next, p_next

def integrate(q0, p0, n_coarse, substeps):
    traj = np.empty((n_coarse + 1, 2 * N_PARTICLES), dtype=np.float64)
    traj[0] = np.concatenate([q0, p0])
    q, p = q0.copy(), p0.copy()
    h_fine = DT / substeps
    for i in range(n_coarse):
        for _ in range(substeps):
            q, p = störmer_verlet_step(q, p, h_fine)
        traj[i+1] = np.concatenate([q, p])
    return traj

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    print(f"[FPUT 3.0] Generating {N_IC} training trajectories...")
    train_list = []
    t0 = time.time()
    for i in range(N_IC):
        amp = rng.uniform(0.05, 0.2)
        q0 = amp * np.sin(np.pi * np.arange(1, N_PARTICLES + 1) / (N_PARTICLES + 1))
        p0 = rng.normal(0, 0.01, N_PARTICLES)
        train_list.append(integrate(q0, p0, N_TRAIN_STEPS, SUBSTEPS))
        if i % 32 == 0: print(f"  {i}/{N_IC}...")
    
    train_data = np.stack(train_list, axis=0)
    print(f"[FPUT] Training generation done in {time.time()-t0:.1f}s")
    
    val_list = []
    for _ in range(32):
        amp = rng.uniform(0.05, 0.2)
        q0 = amp * np.sin(
            np.pi * np.arange(1, N_PARTICLES + 1) / (N_PARTICLES + 1)
        )
        p0 = rng.normal(0, 0.01, N_PARTICLES)
        val_list.append(integrate(q0, p0, N_TRAIN_STEPS, SUBSTEPS))
    val_data = np.stack(val_list, axis=0)
    
    q0_te = 0.1 * np.sin(np.pi * np.arange(1, N_PARTICLES + 1) / (N_PARTICLES + 1))
    p0_te = np.zeros(N_PARTICLES)
    test_traj = integrate(q0_te, p0_te, N_TEST_STEPS, SUBSTEPS)
    
    torch.save(torch.tensor(train_data, dtype=torch.float64), f"{SAVE_DIR}/train.pt")
    torch.save(torch.tensor(val_data,   dtype=torch.float64), f"{SAVE_DIR}/val.pt")
    torch.save(torch.tensor(test_traj,  dtype=torch.float64), f"{SAVE_DIR}/test_long.pt")
    
    print(f"[FPUT] Saved to {SAVE_DIR}/")
    print(f"  train: {train_data.shape}, val: {val_data.shape}, test: {test_traj.shape}")
