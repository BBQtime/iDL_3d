# MDA multi-annotator dataset: planned public release

## Release status

**Status:** TCIA submission planned; collection accession pending.

The associated manuscript states that MDA-derived, de-identified DICOM imaging and contour data will be submitted for public release through [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/). No MDA clinical data are stored in this Git repository.

## Cohort described in the manuscript

| Item | Description |
| --- | --- |
| Institution | MD Anderson Cancer Center (MDA) |
| Patients | 65 |
| Imaging used | CT, T1-weighted MRI and T2-weighted MRI |
| PET | Not included; PET and MR acquisitions used different patient positions, preventing reliable alignment for this study |
| Targets | Primary-tumour GTV (GTVt) and nodal GTV (GTVn) |
| Annotators | 3-4 physicians per patient |
| Observer entries | 197 |
| Ethics approval reported in manuscript | MD Anderson IRB reference RCR03-0800 |

Images were resampled to 1 mm isotropic resolution and centrally cropped to 240 x 240 x 240 voxels for the reported experiments. These are study preprocessing details, not a promise that the TCIA archive will distribute preprocessed arrays.

## Study partition

The analysis used a patient-level split so that every observer entry for one patient remained in the same partition:

- 45 patients and 136 observer entries were used for grouped 3-fold cross-validation;
- 20 patients and 61 delineations were held out for testing;
- the test cohort produced 61 GTVt and 63 GTVn within-patient observer-pair comparisons for the inter-observer-variation analysis.

The repository's `dataset_split/mda.json` records the archived research split identifiers. The future TCIA collection may use different public identifiers after curation.

## Planned archive integration

When TCIA publishes the collection, this document and the root README should be updated with:

1. the TCIA collection name and accession URL;
2. the dataset citation and DOI, if assigned;
3. the archive license/data-usage terms;
4. a checksum or version for the downloaded package;
5. a de-identified conversion script mapping the TCIA layout to the loader format in `docs/DATA.md`;
6. any difference between the manuscript cohort and the final curated release.

Until an accession link appears here, the MDA dataset status remains **planned**, not publicly available.
