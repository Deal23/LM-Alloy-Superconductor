# Data Inventory

This document summarizes the data files included in `Data/` and the role of each file in the machine-learning workflow.

## Source Data

| File | Description |
| --- | --- |
| `20240322_MDR_OAndM.csv` | Source MDR dataset used as the starting point for this study. |
| `mdr.csv` | Filtered subset derived from `20240322_MDR_OAndM.csv`, keeping records marked as available. |

The source dataset is associated with:

1. https://doi.org/10.48505/nims.3739
2. *Science and Technology of Advanced Materials* **16**, 033503 (2015).
3. *Physical Review B* **103**, 014509 (2021).

## Cleaned and Augmented Data

| File | Description |
| --- | --- |
| `mdr_clean.csv` | Fully cleaned dataset used for baseline model training. Columns: `formula`, `Tc`. |
| `mdr_clean_wt.csv` | Cleaned dataset before full weight/atomic-ratio conversion. |
| `mdr_clean_os2.csv` | Dataset with 2x SMOTE oversampling for non-superconductor records. |
| `mdr_clean_os5.csv` | Dataset with 5x SMOTE oversampling for non-superconductor records. |
| `mdr_clean_od2.csv` | Dataset with 2x random duplication oversampling. |
| `mdr_clean_od5.csv` | Dataset with 5x random duplication oversampling. |
| `mdr_duplicated.csv` | Formulas in the cleaned dataset that have multiple reported `Tc` values. |

## Prediction Inputs

| File | Description |
| --- | --- |
| `alloy_element.csv` | Element table used to identify metallic elements for alloy generation. |
| `alloy4pre.csv` | Alloy compositions outside the training set for prediction. |
| `other4pre.csv` | Additional high-temperature superconductor and non-superconductor records for prediction. |
| `example4pre.csv` | Small example prediction dataset. |

## Processing Records

| File | Description |
| --- | --- |
| `Data_clean.ipynb` | Notebook used for data cleaning, feature construction, oversampling, and quality checks. |
| `数据清洗.txt` | Original data-cleaning record. |
| `数据说明.txt` | Original Chinese data-description note. |

## Notes

- Chemical compositions are represented as formula strings and parsed into element-ratio features for model training.
- `Tc` denotes superconducting critical temperature in kelvin.
- The repository preserves the original intermediate datasets to make the data-cleaning pathway inspectable.
