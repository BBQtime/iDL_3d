# Interactive Deep Learning for 3D Head-and-Neck GTV Segmentation

[![Code status: available](https://img.shields.io/badge/code-available-brightgreen)](https://github.com/BBQtime/iDL_3d) [![AUH weights status: planned](https://img.shields.io/badge/AUH_iDL_weights-planned-yellow)](#model-weights) [![MDA data status: TCIA submission planned](https://img.shields.io/badge/MDA_multi--annotator_data-TCIA_submission_planned-orange)](#planned-mda-multi-annotator-dataset-release)

Research code for simulating interactive deep learning (iDL) in three-dimensional head-and-neck gross tumour volume segmentation.

This public release contains the Python training, inference, retrospective interaction-simulation and analysis code. The clinical interaction UI, observer-study interface and related graphical assets are intentionally excluded.

The workflow has two target-specific interaction modes:

- **GTVt (primary tumour):** a slim 3D U-Net baseline is adapted for an individual case using reference-derived delineations on three orthogonal slices. The simulation can perturb the selected centre to study sensitivity to interaction placement.
- **GTVn (nodal disease):** one simulated click identifies each included lymph-node component. Clicks are converted to exponentially decaying distance maps and concatenated with the multimodal images during inference.

The associated cross-centre study evaluates direct application of AUH-trained models on NKI and MDA cohorts against training from scratch and transfer learning.

> [!IMPORTANT]
> This is research software, not a medical device. It is not validated for diagnosis, treatment planning or clinical decision-making.

## Release status

| Component | Status | Public scope |
| --- | --- | --- |
| Simulation code | **Available** | Python training, simulation and analysis code; clinical UI excluded |
| AUH iDL weights | **Planned** | GTVt-iDL and GTVn-iDL checkpoints only |
| MDA multi-annotator dataset | **TCIA submission planned** | De-identified CT/T1/T2 imaging and multi-observer GTVt/GTVn contours; accession pending |

Status badges and this table will be updated when each release becomes available.

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
| `docs/MDA_DATASET.md` | Planned MDA multi-annotator data release description and status |

## Quick start

The code was developed with Python 3.8/3.9.

```bash
git clone https://github.com/BBQtime/iDL_3d.git
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

The public weight release is limited to two AUH iDL checkpoints:

1. `weights/auh/gtvt_idl/idl_gtvt_auh.pt` for the three-orthogonal-slice GTVt iDL workflow;
2. `weights/auh/gtvn_idl/idl_gtvn_auh.pt` for click-guided GTVn iDL inference.

Reserved locations are provided under `weights/auh/gtvt_idl/` and `weights/auh/gtvn_idl/`. Actual checkpoint files are ignored by Git. See [weights/README.md](weights/README.md) for compatibility and integration notes.

The current archived code selects models from the historical `train_results/` experiment hierarchy. A direct loader will be added after these two checkpoints and their metadata are available.


## Planned MDA multi-annotator dataset release

The manuscript describes a planned public release of MDA-derived, de-identified DICOM imaging and contour data through [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/). The cohort contains 65 patients with CT, T1-weighted MRI and T2-weighted MRI. Three to four physicians delineated GTVt and GTVn for each patient, yielding 197 observer entries. PET is not included because PET and MR were acquired in different patient positions and could not be aligned reliably for this study.

| Item | Manuscript description |
| --- | --- |
| Patients | 65 |
| Imaging | CT, T1 and T2; 1 mm isotropic study preprocessing |
| Annotations | GTVt and GTVn from 3-4 physicians per patient |
| Observer entries | 197 |
| Study split | 45 patients / 136 observer entries for 3-fold cross-validation; 20 patients / 61 delineations held out for testing |
| Distribution | TCIA submission planned; accession link pending |

All observer entries from the same patient were kept in the same partition. The final TCIA package, citation, license and accession link may change during archive curation. This README will link to the collection when TCIA makes it available. See [the dataset release notes](docs/MDA_DATASET.md) for details.

## Data and privacy

No clinical dataset is currently included in this repository. The MDA multi-annotator release is planned through TCIA; AUH and NKI imaging and contours are not publicly available because of institutional data-use agreements and patient-privacy restrictions. Never commit patient images, labels, identifiable metadata, credentials or local model outputs directly to this Git repository.

## Citation

Please use [`CITATION.cff`](CITATION.cff) or copy the BibTeX entries below in this order.

1. **Current cross-centre manuscript (submitted)**

```bibtex
@unpublished{wei_crosscentre_idl_submitted,
  author = {Wei, Zixiang and Nijkamp, Jasper and Eriksen, Jesper Grau and
            Gouw, Zeno A. R. and Dede, Cem and Wahid, Kareem A. and
            Sonke, Jan-Jakob and Fuller, Clifton D. and
            Korreman, Stine Sofia and Ren, Jintao},
  title  = {Cross-centre transferability of interactive deep learning for
            head-and-neck gross tumour volume segmentation},
  note   = {Submitted}
}
```

2. **Original iDL workflow**

```bibtex
@article{wei_interactive_idl_2025,
  author  = {Wei, Zixiang and Ren, Jintao and Eriksen, Jesper Grau and
             Jensen, Kenneth and Mortensen, Hanna Rahbek and
             Korreman, Stine Sofia and Nijkamp, Jasper},
  title   = {An interactive deep-learning workflow for head and neck gross
             tumour volume segmentation},
  journal = {Physics and Imaging in Radiation Oncology},
  volume  = {35},
  pages   = {100820},
  year    = {2025},
  doi     = {10.1016/j.phro.2025.100820}
}
```

## License

No open-source license has been selected yet. Until a `LICENSE` file is added, copyright is retained by the authors and reuse is not automatically granted. Select an appropriate license before announcing a public release.
