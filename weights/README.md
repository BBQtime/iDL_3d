# Model weights

Place the approved AUH pretrained models under this directory. Two model types are required for the complete simulation workflow:

```text
weights/
??? auh/
    ??? gtvt_baseline/
    ?   ??? baseline_auh.pt
    ??? gtvn_idl/
        ??? idl_gtvn_auh.pt
```

The filenames above are the recommended public-release names. The archived training code does not currently hard-code these names.

## Model roles

- `gtvt_baseline/baseline_auh.pt`: multimodal baseline model used to produce the initial GTVt segmentation. GTVt iDL resets to this model before each patient's three-slice case-specific adaptation.
- `gtvn_idl/idl_gtvn_auh.pt`: click-guided GTVn model. Its input includes the click-derived distance map and the configured image modalities.

## Important compatibility note

The archived code saves and loads complete PyTorch model objects with `torch.save(model, ...)` and `torch.load(...)`, not only `state_dict` dictionaries. Only load weights from a trusted source, and retain the Python, PyTorch and model-code versions used to create them.

The existing simulation discovers checkpoints through the historical `train_results/<experiment>/fold=.../epoch=.../` hierarchy and uses adjacent `hyper.json` and validation-metric JSON files to select the best fold. This `weights/` directory is a clean staging location for the public AUH models. After the actual files and their metadata are available, add a direct-path loader or reconstruct the legacy result hierarchy before running simulations.

All `*.pt`, `*.pth` and `*.ckpt` files are ignored by Git. Do not commit weights until their redistribution permission, license, provenance and patient-data risk have been reviewed. For public binary distribution, prefer a versioned GitHub Release or Git LFS rather than ordinary Git history.
