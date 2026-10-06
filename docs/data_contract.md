# Dataset and experiment contract

These formats are planned interfaces. Writers and loaders are TODO.

## Raw run

Each unique `data/raw/<run_id>/` contains `cases.csv` and `manifest.json`.
Never overwrite or modify raw runs during preprocessing. Generated files are
ignored by Git, so retain durable copies outside Git when data generation begins.

Each attempted case has these CSV columns:

| Columns | Meaning |
| --- | --- |
| schema_version, run_id, case_id, geometry_id | Provenance and stable grouping |
| aspect_ratio, area_m2, taper_ratio, sweep_deg, airfoil | Geometry definition |
| alpha_deg, beta_deg, velocity_m_s | Operating state, degrees and m/s |
| density_kg_m3, dynamic_viscosity_pa_s, speed_of_sound_m_s | Atmospheric inputs |
| cl, cd | Whole-wing coefficients in freestream axes |
| status, converged | converged, unconverged, or failed; explicit Boolean |
| runtime_s, iterations, residual, error_message | Diagnostics for every attempt |

Null coefficients are empty CSV fields, not zeros. Unconverged finite iterates
may be retained for diagnostics but never used as valid training labels.
Every attempt must remain visible, including exceptions. A scaffold
`NotImplementedError` must abort the workflow, not create a pseudo solver dataset.

The manifest records resolved configuration, random seed, UTC timestamp, commit
if available, dependency versions, units/reference definitions, schema version,
sampling strategy, attempted/success/failed counts, and polar provenance. Define
case IDs from geometry and operating-state inputs, with a replicate index if
repeated measurements are requested. Verify an ID always identifies the same
geometry. Geometry identity must remain consistent across raw runs.

## Preparation and experiment artifacts

1. Read raw data without modification; write exclusion reasons/counts separately.
2. Split unique geometry IDs with the configured seed. A wing's operating states
   stay entirely in one split. Save exact split membership and raw input hashes.
3. Fit airfoil encoding and feature/target scalers only on training rows. Freeze
   them for validation, test and inference. Named-airfoil one-hot encoding does
   not generalize to unseen shapes; reject unknown airfoils in the baseline.
4. Save processed splits under `data/processed/<run_id>/`; put model weights and
   fitted preprocessing under `models/<run_id>/`. Save a schema with feature order,
   units, airfoil categories and target order `[cl, cd]` next to the artifacts.
5. Save resolved settings, seeds, split membership, artifact paths, versions,
   training history, metrics and plots under `results/<run_id>/`. Use relative
   artifact paths where possible. Do not overwrite an existing run.

Validation chooses hyperparameters and checkpoints. The test set is evaluated
after selection. Report CL and CD metrics separately in original physical units,
including MAE, RMSE, R2 and error distributions by geometry and operating state.
Avoid percentage errors near zero lift or drag.

Timing comparisons use identical cases and hardware, explicit batch size,
warmups, repeated measurements and consistent units. Separate model loading,
preprocessing and inference timing. Report solver cold/warm polar cache timing
and failed-case counts. No speedup claims exist until measurements are implemented.
