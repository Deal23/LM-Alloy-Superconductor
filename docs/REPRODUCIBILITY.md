# Reproducibility Notes

This repository preserves the analysis workflow used for tree-model prediction of liquid metal-based alloy superconductors. The files are organized by workflow stage.

## Environment

Use `environment.yml` with conda or `requirements.txt` with pip. The notebooks and scripts use:

- pandas and NumPy for data handling
- scikit-learn, imbalanced-learn, CatBoost, XGBoost, and LightGBM for model training/comparison
- joblib for trained model serialization
- matplotlib, seaborn, and mpltern for plotting
- periodictable and mendeleev for element metadata

## Recommended Execution Order

1. Data cleaning:

   ```bash
   jupyter notebook Data/Data_clean.ipynb
   ```

2. Model training and model comparison:

   ```bash
   cd Code/1_Tree_model_train
   jupyter notebook LMSC-Regression.ipynb
   ```

3. Hyperparameter screening:

   ```bash
   cd Code/2_Hyperparameter_opt
   python Hype_opt.py
   ```

4. Alloy-system screening:

   ```bash
   cd Code/4_alloy_predict
   python alloy_predict.py
   ```

5. Ga-In-Sn focused prediction:

   ```bash
   cd Code/5_GaInSn_best_predict
   jupyter notebook best-Alloy.ipynb
   ```

## Model Artifacts

Downstream prediction scripts expect trained model artifacts under `Code/1_Tree_model_train/2-model/`, including files such as:

- `ET.joblib`
- `ET_feature.joblib`
- `RF.joblib`
- `RF_feature.joblib`

If these files are not available after cloning, regenerate them from `Code/1_Tree_model_train/LMSC-Regression.ipynb`.

## Path Conventions

Many scripts use relative paths inherited from the original workflow. For reproducible execution, run each script from the directory where it is located unless the paths are updated.

## Computational Notes

- `Code/4_alloy_predict/alloy_predict.py` uses multiprocessing and defaults to 30 workers. Adjust `num_workers` for local hardware.
- `Code/server_script.slurm` is provided as an example HPC submission script and may require cluster-specific edits.
- Random alloy compositions are generated stochastically. For exact repeatability, set NumPy random seeds before generation.
