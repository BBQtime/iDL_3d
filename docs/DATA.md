# Data configuration

No clinical data are distributed with this repository. Obtain data through the relevant institutional or public-dataset process and comply with approvals, data-use agreements and de-identification requirements.

## Preprocessing assumptions

The loaders expect multimodal NIfTI volumes already registered into a common space. The archived settings use a centre-aligned `(depth, height, width)` shape of `(240, 240, 240)`, 1 mm isotropic spacing, CT windowing, per-volume normalization, CT/PET/T1/T2 channels when available, and binary GTVt/GTVn labels.

Registration and resampling are not performed by the dataset loader. Verify orientation, spacing, origin and dimensions before use.

## Cohort layouts

Set cohort roots in `settings/core.json`. `<id>` denotes a patient or observer-entry identifier.

AU and observer-study data are loaded directly below the cohort root:

```text
HNCDL_<id>_CT.nii
HNCDL_<id>_PT.nii
HNCDL_<id>_T1dr.nii
HNCDL_<id>_T2dr.nii
HNCDL_<id>_GTVt.nii
HNCDL_<id>_GTVn.nii
HNCDL_<id>_GTVn_clicks.nii.gz
```

AU extended data use the same names inside `<root>/HNCDL_<id>/`.

NKI data use `<root>/<id>/<id>_{CT,PTdr,T1dr,T2dr,GTVt,GTVn}.nii` plus `<id>_GTVn_clicks.nii.gz`.

MDA data use `<root>/<id>/CT`, `T1dr`, `T2dr`, `GTVt.nii`, `GTVn.nii` and `GTVn_clicks.nii.gz`. The archived loader appends no extension to the image-channel names. PET was excluded from the MDA experiments.

HECKTOR files are expected directly under the root with names such as `<id>_CT.nii.gz`. Review the nodal-click path before use: the archived adapter expects `<id>_GTVn_clicks.nii.gz.gz`, which may reflect a historical filename.

## Interaction simulation

- **GTVt:** extracts transverse, coronal and sagittal label slices through a selected centre. `gravity.center.bias.range` controls perturbation experiments.
- **GTVn:** a binary click volume identifies included connected components and is converted to `exp(-0.1 ? Euclidean distance)`.

Visually inspect aligned images, labels and clicks. Keep all observer entries from one patient in the same split to prevent leakage.
