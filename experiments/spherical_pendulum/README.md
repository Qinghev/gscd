- Branch: residual depending on the full state with finite fixed-point midpoint solve.
- Training and primary endpoint control: 5 iterations, tolerance `1e-6`.
- Training budget: 500 epochs, seeds 0--4.
- Primary evaluation: 20 held-out initial conditions and 1,000 saved steps.
- Generator: `src/generate_spherical_pendulum.py`.
- Evaluator: `scripts/evaluate_spherical_pendulum_meridian.py`.
- Evaluated quantities: rollout MSE, conserved momentum, meridian section,
  return map, and crossing count.
- Periodic state error wraps only the azimuth `phi`; the polar angle `theta`
  is not treated as a `2*pi`-periodic coordinate.
- Section crossings use each trajectory's initial meridian,
  `phi - phi_initial = 0 mod 2*pi`, with positive orientation and the initial
  crossing excluded.
- Manuscript dot-plot renderer:
  `scripts/plot_spherical_pendulum_metric_dotplot.py`.
- Strict fixed-checkpoint audit: 12 iterations, tolerance `1e-10`; no retraining.
- Primary archives: `results/spherical_pendulum_meridian_5seed.json` and
  `results/gsympnet_spherical_pendulum_meridian_5seed.json`.
- The five-seed GSC-no-split and HNN-Symp checkpoints used in the reported
  comparison are included in the release manifest.
- Pre-rendered manuscript figure:
  `results/figures/spherical_pendulum/spherical_pendulum_metric_dotplot_with_gsympnet_5seed.pdf`.

```bash
python scripts/evaluate_spherical_pendulum_meridian.py \
  --models gscd gscd_nosplit hnn hnn_implicit hnn_symp node gsympnet \
  --seeds 0 1 2 3 4 --device cuda:0 \
  --output_stem spherical_pendulum_meridian_5seed
python scripts/plot_spherical_pendulum_metric_dotplot.py
```
