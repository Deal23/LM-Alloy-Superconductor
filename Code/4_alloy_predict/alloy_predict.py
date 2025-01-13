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

# 合金体系
num_elements = 2            # n元合金
num_formulas = 1000         # 生成随机合金数量
num_workers = 30            # 多进程处理调用核心数目

# 配置文件路径和日志文件名
# 修改'model_path'以及'feature_names_path'以选择合适的预测模型
# 修改'al1oy_names_path'以选择合适预测对象
CONFIG = {
    'model_path': '../model/ET.joblib',
    'feature_names_path': '../model/ET_feature.joblib',
    'al1oy_names_path':'../model/alloy_element.csv',
    'log_file': f'alloy_log_{datetime.now().strftime("%m%d%H%M")}.txt',
    'epoch':math.ceil(math.comb(66, num_elements)*2/3600)
}


# 加载模型和特征名称
model = load(CONFIG['model_path'])
feature_names = load(CONFIG['feature_names_path'])

# 生成分子式
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

# 将消息写入日志文件
def log_message(message):
    with open(CONFIG['log_file'], "a") as log_file:
        log_file.write(message + "\n")
        
# 读取合金元素CSV文件
df = pd.read_csv(CONFIG['al1oy_names_path'])  
metals_df = df[df['type'] == 'Metal']
metal_elements = metals_df['Element'].tolist()

# 创建合金分子式CSV文件
elements_combination = []
alloy_formula_file_name = f"{num_elements}-Alloy.csv"       #----------------保存预测数据到文件-------------
for ec in itertools.combinations(metal_elements, num_elements):
    elements_combination.append('-'.join(ec))
df = pd.DataFrame(elements_combination, columns=['elements'])
df['calculated'] = 'no'

# 遍历DataFrame中的每一行
def update_cal(df,num_formulas,epoch=10):
    start_time = time.time()
    c_epoch = 0
    for index, row in df.iterrows(): 
    # 检查'calculated'列的值是否为'no'
        if row['calculated'] == 'no':
            alloy_time = time.time()
        # 读取'formulas'列的值
            elements = row['elements'].split('-')
            formulas = generate_formulas(elements, num_formulas)
            predict = pd.DataFrame(formulas, columns=['formula'])
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(f_pre, predict['formula']))
            predict['Predicted_Tc'] = results
            # 找到预测结果的最大值及其对应的索引
            max_index = predict['Predicted_Tc'].idxmax()
            max_predicted_tc = predict.at[max_index, 'Predicted_Tc']
            # 将最大值更新到原始DataFrame的相应行
            df.at[index, 'Predicted_Tc'] = max_predicted_tc
            # 将最大值对应的elements也保存到df
            df.at[index, 'Elements'] = predict.at[max_index, 'formula']  # 假设我们只需要第一个元素
            # 将'calculated'列的值更新为'yes'
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
    log_message(f"文件 {alloy_formula_file_name} 已存在，执行上次计算。")
    log_message(f"共 {epoch} 轮计算")
    df = pd.read_csv(alloy_formula_file_name)
    for i in range(epoch):
        update_cal(df,num_formulas,epoch)
        df.to_csv(alloy_formula_file_name,index=False)
        log_message(f"\n-----------------------Epoch {i+1} finished-----------------------------")

else:
    log_message(f"文件 {alloy_formula_file_name} 不存在，将创建并开始新的计算。")
    epoch = CONFIG['epoch']
    df.to_csv(alloy_formula_file_name, index=False)
    log_message(f"共 {epoch} 轮计算")
    for i in range(epoch):
        update_cal(df,num_formulas,epoch)
        df.to_csv(alloy_formula_file_name,index=False)
        log_message(f"\n-----------------------Epoch {i+1} finished-----------------------------")