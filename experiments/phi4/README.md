- Branch: position dependent residual.
- Training budget: 200 epochs, seeds 0--4.
- Primary evaluation: 16 held-out initial conditions and 2,000 saved steps.
- Generator: `src/generate_phi4.py`.
- Endpoint evaluator: `scripts/evaluate_multi_ic.py`.
- Time-curve evaluator: `scripts/evaluate_phi4_time_curves.py`.
- Primary archive: the `phi4` rows of
  `results/gsympnet_multi_ic_5seed.json`.

```bash
python scripts/evaluate_multi_ic.py --systems phi4 \
  --models gscd hnn node gsympnet --seeds 0 1 2 3 4 --device cuda:0 \
  --output_stem phi4_multi_ic_5seed
python scripts/evaluate_phi4_time_curves.py --device cuda:0 \
  --output_stem phi4_time_curves_5seed
```

The released `gsympnet_multi_ic_5seed.json` is a public subset of the original
multi-system batch archive. It retains the `phi4` rows used here and the
superseded double-pendulum rows for traceability; inactive H\'enon--Heiles rows
and their unreleased checkpoints are omitted. The command stored in the JSON
describes the released subset.
