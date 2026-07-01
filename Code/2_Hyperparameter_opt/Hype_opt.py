import pandas as pd
import re
import numpy as np
import time
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

df = pd.read_csv('../../Data/mdr_clean.csv')

# Hyperparameter search space
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],  # number of trees
    'max_depth': [None, 10, 20],  # maximum depth
    'min_samples_split': [2, 5, 10],  # minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # minimum samples required at a leaf node
    'criterion': ['squared_error', 'friedman_mse'],  # split criterion
}
# Parse a chemical formula into an element-ratio dictionary
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

# Preprocess molecular data
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

# Count predictions with absolute error below 5 K
def right_count(y_pred, y_true):
    relative_error = np.abs(y_pred - y_true)
    return (relative_error < 5).sum()

# Hyperparameter optimization with cross-validation
def hyperparameter_optimization(X_train, y_train, param_grid):
    results_list = []

    for params in ParameterGrid(param_grid):
        start_time = time.time()
        
        et = ExtraTreesRegressor(random_state=42, **params)
        scores = cross_val_score(et, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        et.fit(X_train, y_train)
        y_pred = et.predict(X_test)
        # Switch to the alternative model block when needed.
        # rf = RandomForestRegressor(random_state=42, **params)
        # scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        # rf.fit(X_train, y_train)
        # y_pred = rf.predict(X_test)           
        
        mean_score = np.mean(scores)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        accuracy = right_count(y_pred, y_test) / len(y_test) * 100
        elapsed_time = time.time() - start_time
        print(f"fit time for params {params}: {elapsed_time:.2f} seconds")
        
        results_list.append({
            'params': params,
            'mean_mse': mean_score,
            'mse': mse,
            'r2': r2,
            'accuracy': accuracy,
            'fit_time': elapsed_time
        })

    # Save the result table
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('Hype_model.csv', index=False)
    return results_df

# Run hyperparameter optimization
grid_search_result = hyperparameter_optimization(X_train, y_train, param_grid)
