# Reproducibility notes

## Paper-to-code map

| Study component | Implementation |
| --- | --- |
| Slim 3D U-Net / U-Net++ | `py_code/unet_slim.py`, `py_code/unet_pp_slim.py` |
| Fully automatic baseline | `py_code/training_utils/baseline_training.py` |
| GTVt three-slice adaptation | `py_code/training_utils/idl_gtvt_training.py` |
| GTVt interaction dataset | `py_code/dataset_utils/idl_gtvt_dataset.py` |
| GTVn click-guided workflow | `py_code/training_utils/idl_gtvn_training.py` |
| GTVn attention-map dataset | `py_code/dataset_utils/idl_gtvn_dataset.py` |
| DSC, MSD and HD95 | `py_code/metric_utils/metric_func.py` |
| Cross-dataset analysis | `py_code/research_utils/cross_dataset.py` |
| Inter-observer variation | `py_code/research_utils/iov.py` |

`hyper/*.json` stores archived experiment definitions, `dataset_split/*.json` stores cohort partitions, and `settings/core.json` supplies local data roots and devices.

## Recommended sequence

1. Prepare and quality-check registered NIfTI data using [DATA.md](DATA.md).
2. Create `settings/core.json` and configure cohort roots and devices.
3. Review split JSON files and preserve patient grouping.
4. Train or provide a compatible baseline checkpoint.
5. Explicitly select the intended GTVt simulation or GTVn inference block in `main_training.py`.
6. Explicitly select the intended analysis block in `main_research.py`.
7. Archive configs, logs, package versions, seeds and Git commit with the results.

## Limitations

- Clinical datasets and trained weights are not included.
- The exact historical environment lock file was not retained.
- Entry points contain development-specific experiment IDs and manually selected blocks.
- `main_training.py` launches a particular GTVt simulation if run unchanged.
- Registration/resampling is assumed rather than implemented in the loader.
- External-cohort interactions are retrospective simulations derived from reference labels, not prospective clinician interactions.
- The clinical UI and observer-study interface are intentionally excluded.

Report the commit, Python/PyTorch/CUDA versions, GPU, cohort/split, modalities, preprocessing, checkpoint provenance, hyperparameters, interaction-generation method, sample count and patient-level aggregation strategy.
