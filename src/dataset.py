import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional


class TrajectoryDataset(Dataset):
    def __init__(self,
                 traj:   torch.Tensor,
                 window: int = 10,
                 stride: int = 1,
                 mean:   Optional[torch.Tensor] = None,
                 std:    Optional[torch.Tensor] = None):
        self.window = window
        if traj.ndim == 3:
            N_ic, T_steps, dim = traj.shape
            self.dim = dim
            self.starts = []
            for i in range(N_ic):
                for s in range(0, T_steps - window, stride):
                    self.starts.append((i, s))
        else:
            T, dim = traj.shape
            self.dim = dim
            self.starts = [(None, s) for s in range(0, T - window, stride)]
            
        if mean is not None and std is not None:
            traj = (traj - mean) / std
        self.traj = traj

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        i, s = self.starts[idx]
        if i is None:
            return self.traj[s : s + self.window + 1]
        else:
            return self.traj[i, s : s + self.window + 1]


def load_dataset(data_dir: str,
                 window:   int = 10,
                 stride:   int = 1,
                 batch_size: int = 64,
                 normalize: bool = False,
                 data_ratio: float = 1.0,
                 seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """Load train / val splits from data_dir and return DataLoaders.

    Returns: (train_loader, val_loader, test_traj)
    """
    root = Path(data_dir)
    train = torch.load(root / "train.pt", weights_only=True)
    val_path = root / "val.pt"
    val = torch.load(val_path, weights_only=True) if val_path.exists() else None

    val_invalid = (
        val is None or
        val.ndim != train.ndim or
        val.shape[-1] != train.shape[-1]
    )
    if val_invalid:
        if train.ndim != 3 or len(train) < 2:
            raise ValueError(
                f"Validation split at {val_path} is missing or malformed, "
                f"and no fallback split can be created from train.pt."
            )
        n_val = max(1, min(128, len(train) // 8))
        print(
            f"⚠️ VALIDATION FALLBACK ⚠️: using the last {n_val} training trajectories "
            f"as validation because {val_path} is missing or malformed."
        )
        val = train[-n_val:].clone()
        train = train[:-n_val].clone()

    if data_ratio < 1.0:
        n_train = max(1, int(len(train) * data_ratio))
        n_val = max(1, int(len(val) * data_ratio))
        
        train = train[:n_train]
        val = val[:n_val]
        print(f"⚠️ DATA STARVATION ⚠️: Reduced to {n_train} train trajectories, {n_val} val trajectories (ratio: {data_ratio})")

    mean = std = None
    if normalize:
        mean = torch.load(root / "mean.pt", weights_only=True)
        std  = torch.load(root / "std.pt",  weights_only=True)

    train_ds = TrajectoryDataset(train, window=window, stride=stride,
                                  mean=mean, std=std)
    val_ds   = TrajectoryDataset(val,   window=window, stride=stride,
                                  mean=mean, std=std)

    drop_last_train = True if len(train_ds) > batch_size else False
    loader_generator = None
    if seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True,  drop_last=drop_last_train,  num_workers=0,
                               generator=loader_generator)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                               shuffle=False, drop_last=False, num_workers=0)

    test_path = root / "test_long.pt"
    if not test_path.exists():
        test_path = root / "test.pt"
    test_traj = torch.load(test_path, weights_only=True)
    return train_loader, val_loader, test_traj
