# Experiment map

Run commands from a workspace created by `reproduce.py prepare`. Recorded-result workspaces allow direct score verification; training workspaces start without previous experiment outputs. Use a separate workspace when repeating an output that already exists.

| Paper evidence | Code | Inputs / recorded result |
|---|---|---|
| Complete reference and adaptations, Section 3 | `src/models/msda_irstd.py`, `subpixel_logits.py`, `multi_phase_stem.py`, `detector_variants.py` | `configs/models/`, grouped protocol |
| Data preparation, Section 4.1 / Table S1 | `scripts/audit_data.py`, `audit_backgrounds.py`, `prepare_split.py` | Data archive: original fingerprints, similarity edges, background groups, frozen members |
| Six-detector comparison, Figure 3 | `benchmarks/detectors/{campaign,comparator,train_one,plot_comparison}.py` | Four YAML definitions; seed 17, 50 epochs; `results/summary.csv` |
| Factorial study, Tables 3–5 and S3 | `scripts/run_factorial.py`, `evaluate_study.py`, `analyze_factorial.py` | Twelve grouped test predictions; paired seeds 17/29/43 |
| IoU curves, Figures 4–5 | `scripts/analyze_saved_predictions.py`, `plot_figures.py` | Same twelve predictions; `analysis/threshold_analysis.json` |
| Matching and duplicates, Figure 6 / Table S4 | `scripts/analyze_predictions.py`, `plot_match_turnover.py` | Six A/H predictions; `analysis/prediction_diagnostics.json` |
| Real-image examples, Figure 7 | `scripts/select_examples.py`, `plot_figures.py` | Fixed first qualifying seed/image/annotation; confidence ≥0.25; actual source images |
| Candidate geometry / phase control, Table S5 | `scripts/analyze_geometry.py`, `check_geometry_boundaries.py`, `check_phase_control.py` | Training boxes; seed-17 development H and phase-collapsed results |
| Group sensitivity, Table S6 | `src/cluster_ap.py`, `scripts/validate_cluster_ap.py`, `analyze_factorial.py`, `analyze_groups.py` | Fixed group membership; paired bootstrap and leave-one-group-out results |
| Longer training, Table S6 | `scripts/train.py`, `evaluate_long.py`; `configs/long_schedule.json` | Independent 50/100-epoch schedules; terminal development A/HS scores at seed 17 |
| SIDD transfer, Tables 6 and S6 | `scripts/train_original_split.py`, `audit_sidd.py`, `prepare_sidd_sensitivity.py`, `evaluate_study.py` | Original-split A/HS weights; six complete prediction sets; screened and alternate annotations |
| Grouped A/H cost, Table 7 | `scripts/benchmark_grouped.py`; `configs/diagnostics.json` | Six grouped weights; `results/cost_grouped/` |
| Original-split A/HS cost, Table 8 | `scripts/benchmark_inference_warm.py`, `benchmark_memory.py` | Six original-split weights; warmed timing and separate single-input memory |
| Architecture / data diagrams, Figures 1–2 | `scripts/draw_overview_figures.py` | Actual model configuration and frozen population counts |

Data and measurements are JSON dictionaries compressed with gzip. Their keys are the relative files expanded by `prepare`; they are not executable archives. `results/index.json` associates each unchanged prediction file with its runtime destination and SHA-256. The CSV provides a directly browsable result index. AP values in CSV are fractions, not percentages.

To regenerate diagnostics in a recorded workspace, first move its existing diagnostic JSON to a backup filename, then run the corresponding script. For group intervals run `validate_cluster_ap.py` before `analyze_factorial.py`. The latter recomputes all registered resamples from retained predictions. The grouping sensitivity script also draws actual-image alignment panels and therefore needs SIRST images.

Cost commands need CUDA, the appropriate checkpoints and 32 specified training images. Run them separately on an idle GPU. The measurements use batch one, FP32, unfused models, synchronized warm-up and repeated blocks. They measure the local hardware configuration; regenerated timings are not expected to be bit-identical.

The original archive converter is `scripts/prepare_sirst_archive.py`: place the matching `SIRST-UAVB.zip` at the workspace root, run the converter, `reproduce.py register`, and `reproduce.py attach-data --sirst-root data/processed/sirst_uavb`. To rerun data-only grouping, archive the workspace's existing staged manifest before running the three preparation scripts in table order. Do not change the final evaluation population based on scores.

SIDD's screened primary population has 3,373 images; alternate first/last-source annotation populations have 4,017 unique images. `attach_sidd.py` extracts only fingerprint-verified image members needed for those fixed records. The original audit and sensitivity code is retained for reconstructing their definitions. The paper reports bounding-box evaluation, although SIDD originated as an instance-segmentation dataset.

Relocation changes path-dependent source/protocol hashes. `register` records the current execution identities before training; original checkpoint SHA-256 values and retained prediction bytes remain unchanged. No training was repeated or test score optimized while simplifying this package.
