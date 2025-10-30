import os
import numpy as np
import pandas as pd
import re
import time
import itertools
from joblib import load
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

num_workers = 30
# 配置文件路径和日志文件名
# 修改'model_path'以及'feature_names_path'以选择合适的预测模型
CONFIG = {
    'model_path': '../1_Tree_model_train/2-model/ET.joblib',
    'feature_names_path': '../1_Tree_model_train/2-model/ET_feature.joblib',
    'output_dir': './Alloy/',
    'log_file': f'alloy_log_{datetime.now().strftime("%m%d%H%M")}.txt',
    'top_file': '1-3Alloy.csv'
}

# 确保输出目录存在
# os.makedirs(CONFIG['output_dir'], exist_ok=True)

# 加载模型和特征名称
model = load(CONFIG['model_path'])
feature_names = load(CONFIG['feature_names_path'])

# 定义预测函数
def predict_transition_temp(model, feature_names, elements_ratios):
    feature_row = {element: elements_ratios.get(element, 0) for element in feature_names}
    input_features = pd.DataFrame([feature_row], columns=feature_names)
    predicted_temp = model.predict(input_features)
    return predicted_temp[0]

def f_pre(formula):
    try:
        return predict_transition_temp(model, feature_names, parse_formula(formula))
    except Exception as e:
        # 将错误信息记录到日志文件中
        log_message(f"Error processing formula {formula}: {e}")
        return None  # 或者返回一个默认值，取决于您的业务逻辑

# 生成随机化学式
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

# 化学式拆分为字典
def parse_formula(formula):
    formula = str(formula)  # 确保formula是字符串
    elements_ratios = {}
    for element, ratio in re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', formula):
        if element in elements_ratios:
            elements_ratios[element] += float(ratio) if ratio else 1
        else:
            elements_ratios[element] = float(ratio) if ratio else 1
    return elements_ratios

# 将消息写入日志文件
def log_message(message):
    with open(CONFIG['log_file'], "a") as log_file:
        log_file.write(message + "\n")

# 主函数，执行预测
def main(elements, num_formulas):
    file_name = '-'.join(elements)
    file_path = os.path.join(CONFIG['output_dir'], f'{file_name}.csv')
    formulas = generate_formulas(elements, num_formulas)
    predict = pd.DataFrame(formulas, columns=['formula'])
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(f_pre, predict['formula']))
    predict['Predicted_Tc'] = results
    # 保存top 10结果到top.csv
    predict.insert(0, 'elements', file_name)
    top_10 = predict.nlargest(10, 'Predicted_Tc')
    if os.path.exists(CONFIG['top_file']):
        top_10.to_csv(CONFIG['top_file'], mode='a', header=False, index=False)
    else:
        top_10.to_csv(CONFIG['top_file'], index=False)

if __name__ == "__main__":
    start_time = time.time()
    elements = ['Ga', 'Bi', 'In', 'Sn', 'Zn', 'Ag', 'Sb', 'Cu']
    num_elements = 3

    for r in range(1, num_elements + 1):          #-------------------从一元单质到二元合金到三元合金
        for elements_combination in itertools.combinations(elements, r):
            alloy_time = time.time()
            num_formulas = 1 if r == 1 else 500   #-------------------修改生成样本数量
            main(elements_combination, num_formulas)
            cycle_time = time.time() - alloy_time
            log_message(f"Finished predicting for alloy: {elements_combination};{cycle_time:.2f} seconds")

    elapsed_time = time.time() - start_time
    log_message(f"\nElapsed time: {elapsed_time:.2f} seconds")