# Public reproduction code

This is the single compact, data-free reference implementation of the espresso
reconstruction used for the public release. It contains no internal paths,
manuscript files, or results. Use a permitted shot CSV as follows:

```bash
pip install -r requirements.txt
python reconstruct.py shot.csv --out fit.json
python reconstruct.py shot.csv --tds 8.57 --out fit_tds.json
```

Required columns are `elapsed`, `pressure`, `flow_out`, and
`current_total_shot_weight` (or `weight`). The same code and fitting routine
are used for every shot. If `--tds` is provided, the endpoint-TDS residual is
included; otherwise it is omitted. No second model or second code path is
distributed. The fit uses differential evolution (seed 42, 60 iterations,
population 15) followed by L-BFGS-B. TDS and EY are reconstructed outputs; EY
is never a fitting target.
