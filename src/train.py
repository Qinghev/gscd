import argparse, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from model   import GSCD_Integrator, GSCD_NoSplit, NeuralODE, SCD, HNN, HNN_Symp, HNN_Implicit, SRNNLite, SympNetLite, GradientSympNet
from dataset import load_dataset


DTYPE = torch.float64
torch.set_default_dtype(DTYPE)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(gpu_id=None):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cuda")


SYSTEM_DT = {
    "harmonic":         0.1,
    "harmonic_noisy":   0.1,
    "fput":             0.1,
    "fput_noisy":       0.1,
    "phi4":             0.05,
    "toda":             0.05,
    "kepler":           0.02,
    "charged_particle": 0.05,
    "henon_heiles":     0.1,
    "double_pendulum":  0.05,
    "spherical_pendulum": 0.05,
    "damped_pendulum":  0.05,
}

SYSTEM_DIM = {
    "harmonic":         4,
    "harmonic_noisy":   4,
    "fput":             64,
    "fput_noisy":       64,
    "phi4":             64,
    "toda":             64,
    "kepler":           4,
    "charged_particle": 6,
    "henon_heiles":     4,
    "double_pendulum":  4,
    "spherical_pendulum": 4,
    "damped_pendulum":  2,
}


def build_model(name: str, dim: int, hidden: int = 128,
                sparsity: str = 'dense', separable: bool = True,
                implicit_max_iters: int = 5,
                implicit_tol: float = 1e-6) -> nn.Module:
    kw = dict(dim=dim, hidden=hidden, dtype=DTYPE)
    split_kw = dict(
        sparsity=sparsity,
        separable=separable,
        implicit_max_iters=implicit_max_iters,
        implicit_tol=implicit_tol,
    )
    if   name == "gscd":      return GSCD_Integrator(linear_variant="spd_cayley", **split_kw, **kw)
    elif name == "gscd_sym":  return GSCD_Integrator(linear_variant="sym_cayley", **split_kw, **kw)
    elif name == "gscd_free": return GSCD_Integrator(linear_variant="free_cayley", **split_kw, **kw)
    elif name == "gscd_exp":  return GSCD_Integrator(linear_variant="spd_exp", **split_kw, **kw)
    elif name == "gscd_nosplit": return GSCD_NoSplit(linear_variant="spd_cayley", **split_kw, **kw)
    elif name == "node":      return NeuralODE(**kw)
    elif name == "scd":       return SCD(**kw)
    elif name == "hnn":       return HNN(**kw)
    elif name == "hnn_symp":  return HNN_Symp(**kw)
    elif name == "srnn":      return SRNNLite(**kw)
    elif name == "sympnet":   return SympNetLite(**kw)
    elif name == "gsympnet":  return GradientSympNet(**kw)
    elif name == "hnn_implicit": return HNN_Implicit(
        implicit_max_iters=implicit_max_iters,
        implicit_tol=implicit_tol,
        **kw,
    )
    else: raise ValueError(f"Unknown model: {name}")


def trajectory_loss(model, batch: torch.Tensor, dt: float,
                    window: int) -> torch.Tensor:
    u0  = batch[:, 0, :]
    gt  = batch[:, 1:, :]
    if hasattr(model, "prepare_initial_state"):
        u0 = model.prepare_initial_state(u0)
    pred = model(u0, dt=dt, steps=window)
    pred = pred[1:].permute(1, 0, 2)
    return F.mse_loss(pred, gt)


import torch.nn.functional as F


def train(args):
    set_global_seed(args.seed)
    device = get_device(getattr(args, 'gpu', None))

    data_ratio = getattr(args, 'data_ratio', 1.0)
    system_dir = args.system if data_ratio == 1.0 else f"{args.system}_data_starvation"
    ckpt_dir = Path("checkpoints") / system_dir / args.model
    if args.tag:
        ckpt_dir = ckpt_dir / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.json"
    summary_path = ckpt_dir / "train_summary.json"

    data_dir = Path("data") / args.system
    dt       = SYSTEM_DT[args.system]
    dim      = SYSTEM_DIM[args.system]

    print(f"\n{'='*60}")
    print(f"  System : {args.system}  (dim={dim}, dt={dt})")
    print(f"  Model  : {args.model}")
    print(f"  Seed   : {args.seed}")
    print(f"  Device : {device}")
    print(f"  DType  : {DTYPE}")
    print(f"{'='*60}\n")

    train_loader, val_loader, test_traj = load_dataset(
        data_dir,
        window     = args.window,
        stride     = args.stride,
        batch_size = args.batch_size,
        normalize  = False,
        data_ratio = data_ratio,
        seed       = args.seed,
    )
    test_traj = test_traj.to(device, dtype=DTYPE)

    model = build_model(
        args.model,
        dim,
        hidden=args.hidden,
        sparsity=args.sparsity,
        separable=not args.non_separable,
        implicit_max_iters=args.implicit_max_iters,
        implicit_tol=args.implicit_tol,
    ).to(device)

    n_gpus = torch.cuda.device_count()
    use_dp = (n_gpus > 1) and (getattr(args, 'gpu', None) is None)
    if use_dp:
        print(f"  [DataParallel] Using {n_gpus} GPUs: {list(range(n_gpus))}")
        model = nn.DataParallel(model)

    opt   = Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    param_count = int(sum(p.numel() for p in model.parameters()))
    train_start = time.time()
    best_epoch = None
    peak_gpu_mem_mb = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val   = math.inf
    log        = []
    patience   = args.patience
    min_delta  = args.min_delta
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0     = time.time()
        tr_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device=device, dtype=DTYPE)
            opt.zero_grad()
            loss = trajectory_loss(model, batch, dt, args.window)
            if torch.isnan(loss):
                print(f"[Epoch {epoch}] NaN loss detected — skipping batch")
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= max(len(train_loader), 1)
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch   = batch.to(device=device, dtype=DTYPE)
                val_loss += trajectory_loss(model, batch, dt, args.window).item()
        val_loss /= max(len(val_loader), 1)

        symp_err_str = ""
        if args.model.startswith("gscd"):
            with torch.no_grad():
                core = model.module if isinstance(model, nn.DataParallel) else model
                se = core.linear_part.symplectic_error().item()
            symp_err_str = f"  symp_err={se:.2e}"

        elapsed = time.time() - t0
        print(f"[{epoch:4d}/{args.epochs}]  "
              f"train={tr_loss:.4e}  val={val_loss:.4e}"
              f"{symp_err_str}  ({elapsed:.1f}s)")

        entry = {"epoch": epoch, "train": tr_loss, "val": val_loss,
                 "lr": sched.get_last_lr()[0], "epoch_seconds": elapsed}
        if args.model.startswith("gscd"):
            entry["symp_err"] = se
        log.append(entry)

        relative_improvement = (best_val - val_loss) / (best_val + 1e-30)
        if val_loss < best_val and relative_improvement > min_delta:
            best_val   = val_loss
            best_epoch = epoch
            no_improve = 0
            m_state = (model.module if isinstance(model, nn.DataParallel)
                       else model).state_dict()
            torch.save({
                "epoch": epoch,
                "model_state": m_state,
                "opt_state":   opt.state_dict(),
                "val_loss":    val_loss,
                "args":        vars(args),
            }, ckpt_dir / "best.pt")
        elif val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            m_state = (model.module if isinstance(model, nn.DataParallel)
                       else model).state_dict()
            torch.save({
                "epoch": epoch,
                "model_state": m_state,
                "opt_state":   opt.state_dict(),
                "val_loss":    val_loss,
                "args":        vars(args),
            }, ckpt_dir / "best.pt")
            no_improve += 1
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    total_train_seconds = time.time() - train_start
    if device.type == "cuda":
        peak_gpu_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    summary = {
        "system": args.system,
        "model": args.model,
        "seed": args.seed,
        "tag": args.tag,
        "device": str(device),
        "param_count": param_count,
        "epochs_requested": args.epochs,
        "epochs_completed": len(log),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "total_train_seconds": total_train_seconds,
        "gpu_hours": total_train_seconds / 3600.0 if device.type == "cuda" else 0.0,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest val loss: {best_val:.4e}")
    print(f"Checkpoint saved to: {ckpt_dir / 'best.pt'}")
    print(f"Training summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSC-HNN Training Script")
    parser.add_argument("--model",      type=str,   default="gscd",
                        choices=["gscd","gscd_sym","gscd_free","gscd_exp","gscd_nosplit","node","scd","hnn","hnn_symp","srnn","sympnet","gsympnet","hnn_implicit"])
    parser.add_argument("--system",     type=str,   default="fput",
                        choices=["harmonic","fput","phi4","toda","kepler","charged_particle","henon_heiles","harmonic_noisy","fput_noisy","double_pendulum","spherical_pendulum","damped_pendulum"])
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--data_ratio", type=float, default=1.0, 
                        help="Ratio of training/val data to keep (e.g., 0.1 for 10%% data starvation)")
    parser.add_argument("--window",     type=int,   default=10,
                        help="rollout steps per training sample")
    parser.add_argument("--stride",     type=int,   default=10,
                        help="stride between window starts (larger = fewer samples, faster)")
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--hidden",     type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--sparsity",   type=str,   default="dense", choices=["dense", "banded"])
    parser.add_argument("--non_separable", action="store_true", help="use H(q,p) instead of V(q)")
    parser.add_argument("--implicit_max_iters", type=int, default=5,
                        help="maximum fixed-point iterations for finite solve implicit midpoint branches")
    parser.add_argument("--implicit_tol", type=float, default=1e-6,
                        help="stopping tolerance for finite solve implicit midpoint branches")
    parser.add_argument("--gpu",        type=int,   default=None,
                        help="Specific GPU id to lock this process to (for multi-GPU parallel runs). "
                             "If None, uses all visible GPUs via DataParallel.")
    parser.add_argument("--patience",   type=int,   default=30,
                        help="early stopping patience epochs")
    parser.add_argument("--min_delta",  type=float, default=1e-3,
                        help="minimum *relative* val improvement to reset patience (default 0.1%%)")
    parser.add_argument("--seed",       type=int,   default=0,
                        help="global random seed for Python, NumPy, PyTorch, and DataLoader shuffling")
    parser.add_argument("--tag",        type=str,   default="",
                        help="optional subdirectory under checkpoints/<system>/<model>/ for isolated runs")
    args = parser.parse_args()
    train(args)
