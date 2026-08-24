# Double pendulum

- Branch: residual depending on the full state with finite fixed-point midpoint solve.
- Training and primary endpoint control: 5 iterations, tolerance `1e-6`.
- Training budget: 500 epochs, seeds 0--4.
- Primary evaluation: 10 held-out initial conditions and 1,000 saved steps.
- Generator: `src/generate_double_pendulum.py`.
- Common endpoint evaluator: `scripts/evaluate_multi_ic.py`; both angular
  differences are wrapped to `[-pi, pi)`.
- Shadowing evaluator: `scripts/evaluate_double_pendulum_shadowing.py`.
- Time-curve evaluator: `scripts/evaluate_double_pendulum_time_curves.py`.
- Strict fixed-checkpoint audit: 12 iterations, tolerance `1e-10`; no retraining.
- Primary archives: `results/multi_ic_pendula_wrapped_5seed.json`,
  `results/gsympnet_double_pendulum_shadowing_5seed.json`, and
  `results/double_pendulum_time_curves_5seed.json`.

```bash
python scripts/evaluate_multi_ic.py \
  --systems double_pendulum spherical_pendulum \
  --models gscd hnn hnn_implicit node gsympnet \
  --seeds 0 1 2 3 4 --device cuda:0 \
  --output_stem multi_ic_pendula_wrapped_5seed
python scripts/evaluate_double_pendulum_shadowing.py \
  --models gscd hnn hnn_implicit node gsympnet \
  --seeds 0 1 2 3 4 --device cuda:0 \
  --output_stem double_pendulum_shadowing_5seed
python scripts/evaluate_double_pendulum_time_curves.py --device cuda:0 \
  --output_stem double_pendulum_time_curves_5seed
```

The released tensors are authoritative because the legacy generator did not
record the random seed used for the primary data.
