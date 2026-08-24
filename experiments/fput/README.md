
- Branch: position dependent residual.
- Training budget: 200 epochs, seeds 0--4.
- Primary evaluation: 12 recurrence-aware held-out initial conditions and
  20,000 saved steps.
- Generator: `src/generate_fput.py`.
- Recurrence test generator: `scripts/generate_fput_recurrence_test.py`.
- Evaluator: `scripts/evaluate_fput_recurrence.py`.
- Manuscript scatter-profile renderer:
  `scripts/plot_fput_recurrence_profile.py`.
- Primary archive: `results/gsympnet_fput_recurrence_5seed.json`.
- The five HNN-Symp checkpoints used in the reported long-horizon comparison
  are included in the release manifest.
- Pre-rendered manuscript figure:
  `results/figures/fput/fput_recurrence_profile_5seed.pdf`.

```bash
python scripts/evaluate_fput_recurrence.py --models gscd hnn hnn_symp node gsympnet \
  --seeds 0 1 2 3 4 --device cuda:0 \
  --output_stem gsympnet_fput_recurrence_5seed
python scripts/plot_fput_recurrence_profile.py
```

The released tensors are authoritative because the legacy training generator
did not record the random seed used for the primary data.
