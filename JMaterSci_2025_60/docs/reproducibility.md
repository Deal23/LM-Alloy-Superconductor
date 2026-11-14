# 运行与复现

## 环境

使用 Python 3.11 或更高版本，通过 `docs/environment/environment.yml` 或 `docs/environment/requirements.txt` 安装依赖。模型比较还需要 LightGBM、XGBoost 和 CatBoost。

## 工作流

以下命令在项目根目录运行：

| 步骤 | 命令 | 输出 |
| --- | --- | --- |
| 数据清洗 | `python code/data/prepare_data.py` | `data/generated/` 中的清洗数据、重复记录和摘要 |
| 数据增强 | `python code/data/augment.py` | `data/generated/augmented.csv` |
| 模型训练 | `python code/modeling/train.py --models ET RF` | `data/models/` 中的模型及 `data/generated/training/` 中的评估结果 |
| 超参数搜索 | `python code/modeling/tune.py --model ET` | `data/generated/tuning/` 中的交叉验证结果 |
| 批量预测 | `python code/prediction/predict.py` | `data/generated/prediction.csv` |
| 合金筛选 | `python code/prediction/screen.py` | `data/generated/screening.csv` |
| 结果绘图 | `python code/visualization/plot_results.py` | `paper/figures/generated/` |
| 测试 | `python -m unittest discover -s tests -v` | 测试结果 |

训练默认使用 90/10 随机行划分，调参使用五折交叉验证。脚本参数可通过 `--help` 查看；图片与数据的具体对应关系见[工作流索引](workflow.md)。

论文原始 ET/RF 搜索各有 270 组，其中 `squared_error` 和 `friedman_mse` 各占一半。当前 scikit-learn 将两者用于这些回归模型时视为等价，因此脚本仅用 `squared_error` 计算 135 组；新结果保留这一参数标签，历史调参表保持原样。

## 历史参数

- `train.py --evaluation resubstitution`：在训练数据上评估。
- `train.py --compare --legacy-skip-first` 和 `tune.py --legacy-skip-first`：使用原比较或搜索流程的首行跳过方式。
- `augment.py --method legacy-smote`：使用原 Notebook 的标准化坐标重构方式；`--method smote` 则先进行逆标准化。
- 历史超参数表中的 `mean_mse` 为负交叉验证分数，`mse` 为留出集均方误差。
