# Reproducibility protocol

## Environment

Create the Python environment from `requirements.txt` or `environment.yml`.
The artifact host Python, package, CUDA, cuDNN, and GPU versions are recorded
in `results/sisc_current/artifact_manifest.json`. This legacy directory name is
retained for artifact identity; it does not identify the target journal.

All reported models use float64, hidden width 128, rollout windows of length
10, stride 10, batch size 32, Adam, and five seeds numbered 0 through 4.
Per-checkpoint saved arguments in the artifact manifest are authoritative when
a model-specific setting differs from these shared values. GSC-HNN, HNN, and
NODE use tanh activations. The archived HNN-Implicit checkpoints use GELU;
`src/model.py` retains GELU for that class so the released weights reconstruct
their training-time map.

## Active benchmarks

| System | Residual branch | Epoch cap | Test ICs | Rollout steps |
| --- | --- | ---: | ---: | ---: |
| FPUT-beta | position dependent | 200 | 12 | 20,000 |
| Toda lattice | position dependent | 200 | 16 | 2,000 |
| phi4 lattice | position dependent | 200 | 16 | 2,000 |
| Double pendulum | full state | 500 | 10 | 1,000 |
| Spherical pendulum | full state | 500 | 20 | 1,000 |

Benchmark-specific commands and result filenames are listed in
`experiments/<system>/README.md`.

## Solver controls

The full-state residual branch starts from an explicit-Euler predictor and
then applies a fixed-point realization of implicit midpoint. In batched
execution, convergence is tracked independently for each state, so a state's
returned iterate does not depend on the other states in its batch.

- Primary training and evaluation: five iterations, increment tolerance
  `1e-6`.
- Strict fixed-checkpoint audit: 12 iterations, increment tolerance `1e-10`.

The strict audit re-evaluates the primary seed-0 checkpoints and does not
retrain them. Historical `fp5`, `fp8`, and `fp12` retraining pilots are not
part of this release.

The audit reports the normalized midpoint equation residual `rho_F`.
Forward backward and randomized symplectic form defects are unnormalized,
matching the manuscript definitions. Tangent actions in the latter indicator
are estimated by centered finite differences with the step recorded in the
result JSON.

## Periodic coordinate error

All manuscript-facing endpoint evaluators use the same shortest-arc state
difference. Both double pendulum angles are wrapped to `[-pi, pi)`. For the
spherical pendulum, only the azimuth `phi` is periodic; the polar coordinate
`theta` is not wrapped. The authoritative common pendulum archive is
`results/multi_ic_pendula_wrapped_5seed.json`.

## Training

The multiseed driver reconstructs model-specific settings from saved checkpoint
arguments and writes isolated `seed_0` through `seed_4` directories:

```bash
python scripts/run_multiseed_study.py \
  --systems fput toda phi4 \
  --models gscd hnn node gsympnet \
  --seeds 0 1 2 3 4 --device cuda:0

python scripts/run_multiseed_study.py \
  --systems double_pendulum spherical_pendulum \
  --models gscd hnn hnn_implicit node gsympnet \
  --seeds 0 1 2 3 4 --device cuda:0
```

Retraining validates the statistical protocol but is not expected to recreate
bit-identical weights.

## Exact reproduction boundary

Exact numerical reproduction of the archived results requires the released
tensors and checkpoints. The legacy FPUT and double pendulum generators did not
record the random seed used for the primary tensors, so regenerating those
datasets is not bit-equivalent. The current generators use explicit seeds; the
FPUT generator also replaces the legacy duplicated validation trajectory with
independent validation initial conditions. Those corrections define a new
training protocol and do not retroactively reproduce the archived checkpoints.
All archived data files and 125 checkpoints have byte size and SHA-256 recorded
in the artifact manifest.

Run the package-integrity gate before evaluation:

```bash
python scripts/verify_release.py --require-artifacts
```

A `PASS` verifies the released code, data, checkpoints, solver controls,
experiment indexes, and absence of host specific absolute workspace paths.
