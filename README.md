# LM-Alloy-Superconductor

Machine-learning workflow and curated datasets for identifying liquid metal-based alloy superconductors with tree-based regression models.

If this repository is helpful for your research, please cite:

```bibtex
@article{Hua2025LiquidMetalAlloySuperconductor,
  title = {Tree model machine learning to identify liquid metal-based alloy superconductor},
  author = {Hua, Chen and Liu, Jing},
  journal = {Journal of Materials Science},
  volume = {60},
  number = {28},
  pages = {11857--11877},
  year = {2025},
  doi = {10.1007/s10853-025-11121-z},
  url = {https://doi.org/10.1007/s10853-025-11121-z}
}
```

## Overview

This repository contains the data-processing records, model-training notebooks, hyperparameter-search outputs, and alloy-screening results reported in the study. The workflow starts from a superconducting-materials dataset, cleans and featurizes chemical formulas, trains tree-based regression models for critical-temperature prediction, and screens binary/ternary liquid-metal alloy compositions.

## Repository Layout

```text
LM-Alloy-Superconductor/
|-- Data/                         # source, cleaned, augmented, and prediction datasets
|-- Code/
|   |-- 1_Tree_model_train/        # model training, comparison, and prediction notebooks
|   |-- 2_Hyperparameter_opt/      # hyperparameter-search script and outputs
|   |-- 3_Num_effect2prediction/   # sampling-size effect on alloy prediction
|   |-- 4_alloy_predict/           # binary and ternary alloy screening scripts/results
|   |-- 5_GaInSn_best_predict/     # Ga-In-Sn focused prediction and visualization
|   `-- server_script.slurm        # example HPC submission script
|-- Pic/                           # plotting notebook and figure files
|-- docs/                          # data inventory and reproducibility notes
|-- CITATION.cff                   # citation metadata for GitHub and Zenodo
|-- LICENSE                        # MIT License
|-- environment.yml                # conda environment
`-- requirements.txt               # pip dependencies
```

The original directory names are retained to preserve relative paths used by the notebooks and scripts.

## Data

Core datasets are stored in `Data/`.

- `20240322_MDR_OAndM.csv`: source MDR dataset used in the study.
- `mdr.csv`: filtered records whose source status is available.
- `mdr_clean.csv`: cleaned dataset used for baseline model training.
- `mdr_clean_wt.csv`: cleaned dataset before full atomic-ratio conversion.
- `mdr_clean_os2.csv`, `mdr_clean_os5.csv`: SMOTE-oversampled datasets.
- `mdr_clean_od2.csv`, `mdr_clean_od5.csv`: randomly duplicated oversampled datasets.
- `alloy4pre.csv`, `other4pre.csv`, `example4pre.csv`: prediction-target datasets.
- `alloy_element.csv`: element classification table used for alloy screening.

See [docs/DATA.md](docs/DATA.md) for a file-level inventory and source notes.

## Reproducibility

Create a Python environment with either conda:

```bash
conda env create -f environment.yml
conda activate lm-alloy-superconductor
```

or pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The main workflow is:

1. Clean and inspect the source data with `Data/Data_clean.ipynb`.
2. Train and compare tree models with `Code/1_Tree_model_train/LMSC-Regression.ipynb`.
3. Run hyperparameter screening with `Code/2_Hyperparameter_opt/Hype_opt.py`.
4. Screen candidate alloy systems with `Code/4_alloy_predict/alloy_predict.py`.
5. Analyze Ga-In-Sn candidates with `Code/5_GaInSn_best_predict/best-Alloy.ipynb`.

More detailed execution notes are provided in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Important Notes

- Some prediction scripts expect trained model files such as `ET.joblib`, `RF.joblib`, and their corresponding feature-name files under `Code/1_Tree_model_train/2-model/`. If these files are not present, regenerate them with `LMSC-Regression.ipynb` before running downstream prediction scripts.
- Several scripts use relative paths. Run each script from its own directory unless you update the paths explicitly.
- The repository includes result CSV files generated during the reported workflow so that key screening outputs can be inspected without rerunning every calculation.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Chen Hua

Email: h_uachen@163.com
