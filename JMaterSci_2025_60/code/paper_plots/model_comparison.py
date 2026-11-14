# Adapted from Code/1_Tree_model_train/LMSC-Regression.ipynb at commit 82898a6.

# %% Original notebook cell 5
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_OUTPUT = PROJECT_ROOT / 'paper/figures/model_comparison'
PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Read the CSV file
df = pd.read_csv(PROJECT_ROOT / 'data/processed/mdr_clean.csv')

# Parse chemical formulas into dictionaries
def parse_formula(formula):
    formula = str(formula)  # ensure that formula is a string
    elements_ratios = {}
    for element, ratio in re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', formula):
        if element in elements_ratios:
            elements_ratios[element] += float(ratio) if ratio else 1
        else:
            elements_ratios[element] = float(ratio) if ratio else 1
    return elements_ratios

# Build molecular data records
molecular_data = []
for index, formula in df.iloc[1:, 0].items():
    elements_ratios = parse_formula(formula)
    transition_temp = df.iloc[index, 1] if not pd.isna(df.iloc[index, 1]) else None
    molecular_data.append({
        'elements_ratios': elements_ratios,
        'transition_temp': transition_temp
    })

# Preprocess data
def preprocess_molecular_data(molecular_data):
    all_elements = set()
    for data in molecular_data:
        all_elements.update(data['elements_ratios'].keys())

    feature_data = []
    target_data = []

    for data in molecular_data:
        feature_row = {element: data['elements_ratios'].get(element, 0) for element in all_elements}
        feature_data.append(feature_row)
        target_data.append(data['transition_temp'])

    return pd.DataFrame(feature_data), pd.Series(target_data)

X, y = preprocess_molecular_data(molecular_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Evaluate prediction accuracy
def right_count(y_pred, y_true):
    relative_error = abs(y_pred - y_true)
    return (relative_error < 5).sum().item()

def mare_c(y_true, y_pred):
    # Use only samples with true values greater than 5
    mask = (y_true > 5)
    y_true_nonzero = y_true[mask]
    y_pred_nonzero = y_pred[mask]

    if len(y_true_nonzero) == 0:
        return float('nan')  # Return NaN if there are no nonzero samples

    return np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero))

def smape_c(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    return np.mean(diff / denominator) * 100  # return as a percentage

# Performance-evaluation function
def performancecalculator(trueVals, predVals, name):
    fig_height = 5  #cm
    fig_width = 8.5 #cm
    plt.rcParams['font.family'] = 'Arial'  # set the font to Arial
    plt.figure(figsize=(fig_width/2.54, fig_height/2.54))
    # mt = max(trueVals)
    # mp = max(predVals)
    mt = 150
    mp = 150
    plt.subplot(1, 2, 1)
    plt.scatter(trueVals, predVals, color='#1f77b4', label='Predictions',s=3,marker='o',edgecolor='none')
    plt.plot([0, mt], [0, mp], 'k--', lw=0.5, label='Ideal fit')
    accuracy = right_count(predVals, trueVals) / len(predVals) * 100
    plt.title(
        f'Model: {name}\n'
        f'Test Accuracy: {accuracy:.2f}%',
        fontsize=8)
    plt.legend(fontsize=5)
    plt.xlabel('Experiment Value (K)',fontsize=8)
    plt.ylabel('Predicted Value (K)',fontsize=8,labelpad=-0.5)
    plt.xticks(fontsize=6)  # ensure that all x-axis labels are visible
    plt.yticks(fontsize=6)  # set the y-axis tick-label font size
    plt.xticks(range(0, mt+1, 30))
    plt.yticks(range(0, mt+1, 30))


    plt.subplot(1, 2, 2)
    residuals = predVals - trueVals
    mare = mare_c(trueVals, predVals)
    smape = smape_c(trueVals, predVals)
    rmse = np.sqrt(mean_squared_error(trueVals, predVals))
    r2 = r2_score(trueVals, predVals)
    plt.title(
        f'RMSE: {rmse:.4f}  $R^2$: {r2:.4f}\n'
        f'MARE: {mare:.4f}  SMAPE: {smape:.2f}%',
        fontsize=8, loc='left', x = -3e-1)
    plt.scatter(trueVals, residuals, color='#ff7f0e', label='Residuals',s=3,marker='o',edgecolor='none')
    plt.hlines(y=0, xmin=0, xmax=mt, colors='black', linestyles='dashed',lw=0.5)
    plt.hlines(y=5, xmin=0, xmax=mt, colors='black', linestyles='solid',lw=0.5)
    plt.hlines(y=-5, xmin=0, xmax=mt, colors='black', linestyles='solid',lw=0.5)
    plt.xlabel('Experiment Value (K)',fontsize=8)
    plt.ylabel('Residual (K)',fontsize=8,labelpad=-0.5)
    plt.xticks(fontsize=6)  # ensure that all x-axis labels are visible
    plt.yticks(fontsize=6)  # set the y-axis tick-label font size
    plt.xticks(range(0, mt+1, 30))
    plt.yticks(range(-90, 91, 30))
    plt.legend(fontsize=5)
    # Adjust layout
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT / f'{name.lower().replace("-", "_").replace(" ", "_")}.png', dpi=300, bbox_inches='tight', pad_inches=0.05,transparent=True)
    plt.show()

# Train and evaluate models
def train_evaluate_model(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    performancecalculator(y_test, predictions, name)


# # Use extra-trees regression
# train_evaluate_model(ExtraTreesRegressor(), X_train, y_train, X_test, y_test, 'Extra-Trees')

# # Use decision-tree analysis
# train_evaluate_model(DecisionTreeRegressor(), X_train, y_train, X_test, y_test, 'Decision Tree')

# # Use gradient-boosting regression analysis
# train_evaluate_model(GradientBoostingRegressor(), X_train, y_train, X_test, y_test, 'Gradient Boosting')

# # Use random-forest analysis
# train_evaluate_model(RandomForestRegressor(), X_train, y_train, X_test, y_test, 'Random Forest')

# Use gradient-boosting regression analysis
train_evaluate_model(HistGradientBoostingRegressor(), X_train, y_train, X_test, y_test, 'HistGB')

# # Use CatBoost regression analysis
# train_evaluate_model(CatBoostRegressor(), X_train, y_train, X_test, y_test, 'CatBoost')

# # Use XGBoost regression analysis
# train_evaluate_model(XGBRegressor(), X_train, y_train, X_test, y_test, 'XGBoost')

# # Use LightGBM regression analysis
# train_evaluate_model(LGBMRegressor(), X_train, y_train, X_test, y_test, 'LightGBM')
