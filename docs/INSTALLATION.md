# Installation

## Environment

Python 3.8 or 3.9 is recommended because those versions were used during development.

Windows PowerShell:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch build matching the machine's CUDA driver or CPU environment using the [official instructions](https://pytorch.org/get-started/locally/), then install the project dependencies:

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The code can fall back to CPU, but 3D training and case-specific GTVt adaptation are intended for a CUDA-capable GPU.

## Configuration

Create the local settings file from the public template:

```bash
cp settings/core.example.json settings/core.json
```

Edit `cuda.visible.devices`, every relevant path in `dataset.dir`, image geometry, worker counts and `train.results.dir`. The local `settings/core.json` is ignored because it can reveal infrastructure paths.

## Smoke checks

From the repository root:

```bash
python -m compileall -q py_code
python -c "import sys; sys.path.insert(0, 'py_code'); import global_utils.global_core as g; print(g.PROJ_DIR)"
```

The import check validates configuration loading, not datasets or weights.

## Common problems

- **Missing module:** activate the environment and run from the repository root.
- **CUDA error:** align `cuda.visible.devices`, the PyTorch build and available hardware.
- **Missing NIfTI file:** check the exact cohort convention in [DATA.md](DATA.md).
- **Missing checkpoint:** model weights are not bundled; provide compatible experiment directories under `train_results/`.
- **Out of memory:** the default aligned volume is 240? voxels; use suitable hardware or adjust the experiment.
