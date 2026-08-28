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

Additional reproducibility steps:

```bash
python screen_dataset.py summary.csv screened.csv
python batch_reconstruct.py manifest.csv fits.csv
python neutralize.py fits.csv neutralization.csv
python trajectory_pca.py trajectories.csv descriptors.csv pca_scores.csv
python stability_analysis.py same_shot_metrics.csv between_program_metrics.csv stability_summary.csv
python endpoint_matching.py manifest.csv fits.csv telemetry_dir endpoint_output
```

The screening input is one row per shot and must contain the screening summary
columns (`n_samples`, `duration_s`, `final_weight_g`, `onset_s`,
`max_pressure_bar`, `max_flow_out_ml_s`, `monotonicity_pass`, and
`apparent_resistance_pass`). The batch manifest must contain `csv_path` and may
contain `tds_percent` and `dose_g`. Neutralization uses the fitted parameter
columns. PCA expects long-format normalized-progress trajectories with
`shot_id`, `progress`, and the five columns `wetting`, `retained_liquid`,
`swelling`, `resistance_relief`, and `viscosity_factor`.
The script derives ten descriptors: onset and AUC for wetting/connectivity and
resistance relief, peak and AUC for retained liquid, swelling, and viscosity
factor.

| Manuscript result | Public script |
|---|---|
| Dataset screening / population construction | `screen_dataset.py` |
| Batch reconstruction and R² summary | `batch_reconstruct.py`, `reconstruct.py` |
| Fig. 2e post-fit neutralization | `neutralize.py` |
| Fig. 4a,b trajectory descriptors and PCA | `trajectory_pca.py` |
| SI 24-shot × 16-start stability | `stability_analysis.py` |
| Fig. 5 July-2026 endpoint matching | `endpoint_matching.py` |

Raw Visualizer.coffee data are intentionally not included. Input data must be
supplied separately by the user under the applicable data terms.

The endpoint script reproduces the cohort counts (17 pairs, 19 unique shots,
7 strict pairs) directly from the July-2026 manifest. With the raw-telemetry
validity rule documented in `endpoint_matching.py`, the recomputed median RMS
distances are 3.606 bar (pressure), 1.104 g s^-1 (outlet flow), and 4.287
bar s g^-1 (apparent resistance); values in the manuscript are rounded or may
reflect the archived production distance export.
