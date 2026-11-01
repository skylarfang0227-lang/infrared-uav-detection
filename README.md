# Infrared UAV Detection

Reproducible experiments for **Infrared Tiny UAV Detection with a Sub-Pixel Prediction Head**.

The complete MSDA-IRSTD reconstruction is compared with a sub-pixel P2 prediction head, a four-padding input stem and their combination. Four additional lightweight detectors provide a matched single-seed comparison. The repository includes the experiments used in the final paper and appendix, without exploratory campaigns, editing records or packaging logs.

## Results

| Grouped configuration | UAV AP50:95 (%) | Scope |
|---|---:|---|
| A: complete reference | 27.99 | Three-seed mean |
| H: prediction head | 29.38 | Three-seed mean |
| S: input stem | 26.87 | Three-seed mean |
| HS: both | 29.25 | Three-seed mean |

[Per-run results](results/summary.csv) include all 28 retained assessments: 12 main component results, four additional detectors, four development controls, two terminal 100-epoch comparisons and six SIDD transfer results. SIDD scores use the separately trained original-split models. YOLO11n's zero-score run is retained. The fixed-budget comparisons do not establish an external state-of-the-art ranking.

## Recompute scores and figures

Python 3.12; recorded versions: PyTorch 2.7.1, CUDA 12.8 and Ultralytics 8.3.161. CPU is sufficient for the following checks. CUDA training uses the matching PyTorch/torchvision wheels; the original GPU was an RTX 4060 Ti 8 GB.

```console
python -m pip install -r requirements.txt
python scripts/download_results.py
python reproduce.py prepare --output ../uav-review --mode recorded
cd ../uav-review
python reproduce.py verify
python reproduce.py rescore --all
python reproduce.py figures
```

Predictions are distributed as a checksum-verified [release asset](https://github.com/skylarfang0227-lang/infrared-uav-detection/releases/tag/v1.0.0), keeping the Git repository small. The standalone submission ZIP already contains them, so the download command performs verification only. Data and result metadata are losslessly compressed; `prepare` expands them into a separate execution workspace. The original checkout is unchanged.

`figures` regenerates the six charts and diagrams that need no source images. Attach the matching SIRST snapshot and run `figures --with-examples` for all seven manuscript figures. All plots use Python and measured data; example panels use actual images and retained predictions.

## Train and evaluate

Provide the matching processed SIRST-UAVB directory containing `images/train`, `images/val`, `labels/train` and `labels/val`. Every image and label is fingerprint-checked. Labels are normalized YOLO boxes: 0 = UAV, 1 = Bird. Source: [SIRST-UAVB](https://github.com/JN-Yang/PConv-SDloss-Data).

```console
python reproduce.py prepare --output ../uav-training --mode training --sirst-root /path/to/sirst_uavb
cd ../uav-training
python reproduce.py train --experiment factorial
python scripts/evaluate_study.py --study dev
python scripts/evaluate_study.py --study test
python reproduce.py train --experiment detectors
```

The factorial controller runs matched preflight/resume checks, then A/H/S/HS at seeds 17/29/43 and the phase-collapsed control at seed 17. All use scratch initialization, 50 epochs and terminal EMA. The four additional detectors use seed 17 with the same recipe. Training never evaluates the final test population. An existing run is not silently overwritten.

For the independent longer-budget comparison:

```console
python reproduce.py train --experiment long
python scripts/evaluate_long.py
```

For SIDD, use `train --experiment original` or import the recorded original-split checkpoints. Place the fingerprint-matched `SIDD-main.zip` at the workspace root, then run `scripts/attach_sidd.py` followed by `scripts/evaluate_study.py --study sidd`. Source: [SIDD](https://github.com/Dang-zy/SIDD). No target fitting or calibration is performed.

Raw images and large model weights are not bundled. [Checkpoint identities](configs/checkpoints.json) support verified import with `reproduce.py import-weights --mapping weights-map.json`, where the JSON maps a listed destination to an available checkpoint file. New training can regenerate the models. Historical weights support inference; exact resume requires checkpoints from the current packaged implementation.

## Structure

| Directory | Purpose |
|---|---|
| `src/` | Reference, adaptations, training engine and grouped AP computation |
| `scripts/` | Training, data preparation, evaluation, diagnostics and plotting |
| `configs/` | Fixed scientific settings, model definitions and checkpoint identities |
| `benchmarks/detectors/` | Four native comparison detectors and their matched controller |
| `data/` | Compressed annotations, group definitions and fixed split members |
| `results/` | Per-run results, measured diagnostics and prediction download index |
| `tests/` | Model geometry, finite gradients and exact state reload checks |

See [Experiments and reproduction commands](docs/EXPERIMENTS.md) for every paper experiment, including cost and robustness analysis. Prepared workspaces contain generated `analysis/`, `experiments/`, `figures/` and `logs/` directories; these are not versioned process archives.

## License

AGPL-3.0. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md). Dataset rights remain with their providers. This is an independent MSDA-IRSTD reconstruction, with the documented local assumptions shared across comparisons.
