# iDL-3D simulation code

Research code for simulating interactive deep learning (iDL) in three-dimensional head-and-neck gross tumour volume segmentation.

This public release contains the Python training, inference, retrospective interaction-simulation and analysis code. The clinical interaction UI, observer-study interface and related graphical assets are intentionally excluded.

The workflow has two target-specific interaction modes:

- **GTVt (primary tumour):** a slim 3D U-Net baseline is adapted for an individual case using reference-derived delineations on three orthogonal slices. The simulation can perturb the selected centre to study sensitivity to interaction placement.
- **GTVn (nodal disease):** one simulated click identifies each included lymph-node component. Clicks are converted to exponentially decaying distance maps and concatenated with the multimodal images during inference.

The associated cross-centre study evaluates direct application of AUH-trained models on NKI and MDA cohorts against training from scratch and transfer learning.

> [!IMPORTANT]
> This is research software, not a medical device. It is not validated for diagnosis, treatment planning or clinical decision-making.

## What is included

| Path | Purpose |
| --- | --- |
| `py_code/main_training.py` | Manually configured training and simulation driver |
| `py_code/main_research.py` | Analysis and figure-generation driver |
| `py_code/training_utils/` | Baseline, GTVt-iDL and GTVn-iDL workflows |
| `py_code/dataset_utils/` | NIfTI loading, preprocessing and augmentation |
| `py_code/loss_utils/` | Unified focal-loss variants |
| `py_code/metric_utils/` | DSC, surface-distance and added-path-length metrics |
| `py_code/research_utils/` | Cross-dataset, interaction and IOV analyses |
| `py_code/unet_slim.py` | Slim 3D U-Net architecture |
| `py_code/unet_pp_slim.py` | Slim 3D U-Net++ architecture |
| `hyper/` | Archived experiment hyperparameters |
| `dataset_split/` | Archived cohort split identifiers |
| `settings/core.example.json` | Public configuration template |

## Quick start

The code was developed with Python 3.8/3.9.

```bash
git clone https://github.com/630084142/iDL_3d.git
cd iDL_3d
python -m venv .venv
```

Activate the environment, install a CPU/CUDA-compatible PyTorch build using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy and edit the settings template:

```bash
cp settings/core.example.json settings/core.json
```

On Windows PowerShell, use `Copy-Item settings/core.example.json settings/core.json`.

Set the cohort paths and CUDA devices in `settings/core.json`. See [Installation](docs/INSTALLATION.md) and [Data configuration](docs/DATA.md).

## Running simulations

`main_training.py` is an archived experiment driver, not a command-line application. Its active block currently launches a particular GTVt simulation. Review and explicitly select the intended dataset, checkpoint, device and operation before running:

```bash
python py_code/main_training.py
```

Likewise, select the required analysis block before running:

```bash
python py_code/main_research.py
```

No images, clinical labels or model weights are distributed. A run requires locally prepared data plus compatible checkpoints under `train_results/`. See [Reproducibility notes](docs/REPRODUCIBILITY.md) for the paper-to-code map and known limitations.

## Model weights

The complete AUH simulation uses two pretrained model types:

1. an AUH baseline model for the initial GTVt prediction and subsequent three-slice case adaptation;
2. an AUH click-guided iDL model for GTVn inference.

Reserved locations are provided under `weights/auh/gtvt_baseline/` and `weights/auh/gtvn_idl/`. Actual checkpoint files are ignored by Git. See [weights/README.md](weights/README.md) for filenames, compatibility and integration notes.

The current archived code selects models from the historical `train_results/` experiment hierarchy. A direct loader will be added after the released checkpoints and their metadata are available.

## Data and privacy

Clinical datasets are not included. Access is governed by the originating institutions and study agreements. Never commit patient images, labels, identifiable metadata, credentials or local model outputs. NIfTI volumes and model checkpoints are ignored by default.

## Citation

Please use [`CITATION.cff`](CITATION.cff) and cite the original workflow:

> Z. Wei et al., ?An interactive deep-learning workflow for head and neck gross tumour volume segmentation,? *Physics and Imaging in Radiation Oncology*, vol. 35, 100820, 2025. https://doi.org/10.1016/j.phro.2025.100820

The cross-centre transferability manuscript associated with this snapshot is in preparation; add its final bibliographic record after publication.

## License

No open-source license has been selected yet. Until a `LICENSE` file is added, copyright is retained by the authors and reuse is not automatically granted. Select an appropriate license before announcing a public release.
