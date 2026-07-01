import os
import numpy as np
import pandas as pd
import re
import time
import itertools
from joblib import load
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import math

# Alloy-system settings.
num_elements = 2            # number of elements in each alloy
num_formulas = 1000         # number of random alloy formulas to generate
num_workers = 30            # number of worker processes

# Configuration for model files and logs.
# Modify `model_path` and `feature_names_path` to select the prediction model.
# Modify `al1oy_names_path` to select the prediction target list.
CONFIG = {
    'model_path': '../model/ET.joblib',
    'feature_names_path': '../model/ET_feature.joblib',
    'al1oy_names_path':'../model/alloy_element.csv',
    'log_file': f'alloy_log_{datetime.now().strftime("%m%d%H%M")}.txt',
    'epoch':math.ceil(math.comb(66, num_elements)*2/3600)
}


# Load model and feature names.
model = load(CONFIG['model_path'])
feature_names = load(CONFIG['feature_names_path'])

# Generate formulas.
def generate_formulas(elements, num_formulas):
    formulas = []
    while len(formulas) < num_formulas:
        coefficients = np.random.rand(len(elements))
        coefficients /= coefficients.sum()
        coefficients = np.round(coefficients, 3)
        if np.all(coefficients >= 0) and np.isclose(coefficients.sum(), 1):
            formula = ''.join(f"{elements[i]}{coefficients[i]:.3f}" for i in range(len(elements)) if coefficients[i] != 0)
            formulas.append(formula)
    return formulas


# Parse a chemical formula into an element-ratio dictionary.
def parse_formula(formula):
    formula = str(formula)  # ensure that formula is a string
    elements_ratios = {}
    for element, ratio in re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', formula):
        if element in elements_ratios:
            elements_ratios[element] += float(ratio) if ratio else 1
        else:
            elements_ratios[element] = float(ratio) if ratio else 1
    return elements_ratios

# Prediction function.
def predict_transition_temp(model, feature_names, elements_ratios):
    feature_row = {element: elements_ratios.get(element, 0) for element in feature_names}
    input_features = pd.DataFrame([feature_row], columns=feature_names)
    predicted_temp = model.predict(input_features)
    return predicted_temp[0]

def f_pre(formula):
    try:
        return predict_transition_temp(model, feature_names, parse_formula(formula))
    except Exception as e:
        # Write processing errors to the log file.
        log_message(f"Error processing formula {formula}: {e}")
        return None

# Write a message to the log file.
def log_message(message):
    with open(CONFIG['log_file'], "a") as log_file:
        log_file.write(message + "\n")
        
# Read the alloy-element table.
df = pd.read_csv(CONFIG['al1oy_names_path'])  
metals_df = df[df['type'] == 'Metal']
metal_elements = metals_df['Element'].tolist()

# Create the alloy-formula table.
elements_combination = []
alloy_formula_file_name = f"{num_elements}-Alloy.csv"       # output file for predictions
for ec in itertools.combinations(metal_elements, num_elements):
    elements_combination.append('-'.join(ec))
df = pd.DataFrame(elements_combination, columns=['elements'])
df['calculated'] = 'no'

# Iterate over each row in the DataFrame.
def update_cal(df,num_formulas,epoch=10):
    start_time = time.time()
    c_epoch = 0
    for index, row in df.iterrows(): 
        # Process rows that have not been calculated.
        if row['calculated'] == 'no':
            alloy_time = time.time()
            # Read the element list.
            elements = row['elements'].split('-')
            formulas = generate_formulas(elements, num_formulas)
            predict = pd.DataFrame(formulas, columns=['formula'])
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(f_pre, predict['formula']))
            predict['Predicted_Tc'] = results
            # Find the maximum predicted value and its index.
            max_index = predict['Predicted_Tc'].idxmax()
            max_predicted_tc = predict.at[max_index, 'Predicted_Tc']
            # Update the corresponding row in the original DataFrame.
            df.at[index, 'Predicted_Tc'] = max_predicted_tc
            # Store the formula associated with the maximum prediction.
            df.at[index, 'Elements'] = predict.at[max_index, 'formula']
            # Mark the row as calculated.
            df.at[index, 'calculated'] = 'yes'
            alloy_time = time.time() - alloy_time
            c_epoch +=1
            if c_epoch > int(len(df)/epoch):
                break
    elapsed_time = time.time() - start_time
    log_message(f"\n{row['elements']} finished in {alloy_time:.2f} seconds")
    log_message(f"\nElapsed time: {elapsed_time:.2f} seconds")
    
    return df

if os.path.exists(alloy_formula_file_name):
    epoch = CONFIG['epoch']
    log_message(f"File {alloy_formula_file_name} already exists; resuming the previous calculation.")
    log_message(f"Total calculation rounds: {epoch}")
    df = pd.read_csv(alloy_formula_file_name)
    for i in range(epoch):
        update_cal(df,num_formulas,epoch)
        df.to_csv(alloy_formula_file_name,index=False)
        log_message(f"\n-----------------------Epoch {i+1} finished-----------------------------")

else:
    log_message(f"File {alloy_formula_file_name} does not exist; creating it and starting a new calculation.")
    epoch = CONFIG['epoch']
    df.to_csv(alloy_formula_file_name, index=False)
    log_message(f"Total calculation rounds: {epoch}")
    for i in range(epoch):
        update_cal(df,num_formulas,epoch)
        df.to_csv(alloy_formula_file_name,index=False)
        log_message(f"\n-----------------------Epoch {i+1} finished-----------------------------")
