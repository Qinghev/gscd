# Toda lattice

- Branch: position dependent residual.
- Training budget: 200 epochs, seeds 0--4.
- Primary evaluation: 16 held-out initial conditions and 2,000 saved steps.
- Generator: `src/generate_toda.py`.
- Endpoint evaluator: `scripts/evaluate_multi_ic.py`.
- Time-curve evaluator: `scripts/evaluate_toda_time_curves.py`.
- Lax-invariant evaluator: `scripts/evaluate_toda_invariants.py`.
- Primary archives: `results/multi_ic_toda_5seed.json`,
  `results/toda_time_curves_5seed.json`, and
  `results/toda_invariants_5seed.json`.

```bash
python scripts/evaluate_multi_ic.py --systems toda \
  --models gscd hnn node gsympnet --seeds 0 1 2 3 4 --device cuda:0 \
  --output_stem multi_ic_toda_5seed
python scripts/evaluate_toda_invariants.py --device cuda:0 \
  --output_stem toda_invariants_5seed \
  --figure_stem toda_time_curves_5seed
```
