# Learned Split Hamiltonian Maps: Modified Dynamics and Finite-Iteration Effects

This repository contains the code and numerical data for the paper
"Learned Split Hamiltonian Maps: Modified Dynamics and Finite-Iteration
Effects." The method, referred to as GSC-HNN in the code, combines a Cayley
linear Hamiltonian step with a neural Hamiltonian residual in a symmetric
composition.

The numerical experiments cover the FPUT-beta lattice, Toda lattice,
phi4 lattice, double pendulum, and spherical pendulum. The repository includes
the datasets and trained models used to obtain the reported results.

## Installation

The tensor files are stored with Git LFS. Install Git LFS before cloning the
repository.

```bash
git lfs install
git clone https://github.com/Qinghev/gscd.git
cd gscd
conda env create -f environment.yml
conda activate gsc-hnn-siads
```

The Python dependencies can alternatively be installed with

```bash
pip install -r requirements.txt
```

The experiments use double-precision PyTorch. A CUDA device is recommended for
training, but the saved models can also be evaluated on CPU.

## Repository Structure

- `src/` contains the model definitions, training code, dataset utilities, and
  system generators.
- `scripts/` contains the multiseed training, evaluation, plotting, and
  finite-solve diagnostic scripts.
- `data/` contains the training, validation, and test trajectories.
- `checkpoints/` contains the trained models for five random seeds.
- `results/` contains the numerical results and figures used in the paper.
- `experiments/` gives the commands and settings for each benchmark.
- `REPRODUCIBILITY.md` records the common numerical protocol.

## Experiments

The settings shared by all experiments are described in
`REPRODUCIBILITY.md`. Commands specific to each system are given in
`experiments/<system>/README.md`.

For example, the FPUT recurrence experiment can be evaluated with

```bash
python scripts/evaluate_fput_recurrence.py \
  --models gscd hnn hnn_symp node gsympnet \
  --seeds 0 1 2 3 4 --device cuda:0 \
  --output_stem gsympnet_fput_recurrence_5seed
```

The common multiseed training driver is

```bash
python scripts/run_multiseed_study.py \
  --systems fput toda phi4 \
  --models gscd hnn node gsympnet \
  --seeds 0 1 2 3 4 --device cuda:0
```

The full-state residual experiments use an explicit-Euler predictor followed
by fixed-point iterations for the implicit midpoint equation. Training and the
main evaluations use at most five iterations with tolerance `1e-6`; the
fixed-checkpoint numerical audit uses at most 12 iterations with tolerance
`1e-10`.

## Data and Trained Models

The repository contains 22 data files and 125 trained models. Checkpoints are
organized as

```text
checkpoints/<system>/<model>/multiseed/seed_<n>/best.pt
```

Each checkpoint stores the model parameters together with the training
arguments. File sizes and SHA-256 hashes are listed in
`results/sisc_current/artifact_manifest.json`.

## Citation

If you use this code or the accompanying data, please cite

> Qinghe Wang, Xuan Wu, and Kailiang Wu. "Learned Split Hamiltonian Maps:
> Modified Dynamics and Finite-Iteration Effects."

Citation metadata are also provided in `CITATION.cff`.

## License

The code is available under the BSD 3-Clause License. See `LICENSE` for
details.
