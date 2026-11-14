# 工作流索引

本文列出代码、数据和图片的对应关系。数据来源及校验值见[数据说明](provenance/data.md)和 `provenance/artifact_manifest.csv`；路径均相对于项目根目录。

MDR 人工筛选、修正和原始文件说明见[数据整理记录](provenance/records/README.md)；环境和复现范围见[复现指南](reproducibility.md)。

## 整体关系

```text
data/source/mdr.csv ── prepare_data.py ──> data/generated/mdr_clean.csv
data/processed/mdr_clean.csv ──┬── augment.py ──> data/generated/augmented.csv
                              ├── train.py ────> data/models/ + data/generated/training/
                              └── tune.py ─────> data/generated/tuning/

data/models/ + data/inputs/ ─────── predict.py ─────> data/generated/prediction.csv
data/models/ + 元素及采样设置 ───── screen.py ──────> data/generated/screening.csv

历史 data/processed/ + data/research/ ── plot_results.py ──> paper/figures/generated/
历史 data/processed/ + data/research/ ── paper_plots/ ──────> paper/figures/<研究主题>/
```

箭头表示数据读写。训练数据可通过 `--data` 指定；预测和筛选读取模型及特征文件。

## 脚本输入和输出

| 脚本 | 默认输入或依赖 | 默认输出 | 说明 |
| --- | --- | --- | --- |
| `code/data/prepare_data.py` | `data/source/mdr.csv`；`code/data/atomic_masses.csv` | `data/generated/mdr_clean.csv` 及重复、拒绝记录和摘要 | 当前清洗流程；不会覆盖 `data/processed/` 快照 |
| `code/data/augment.py` | `data/processed/mdr_clean.csv` | `data/generated/augmented.csv` | 重复采样或 SMOTE；`data/augmented/` 是历史快照 |
| `code/modeling/train.py` | `data/processed/mdr_clean.csv` | `data/models/` 中模型、特征和指标；`data/generated/training/` 中评估结果 | 可选择模型和评估方式 |
| `code/modeling/tune.py` | `data/processed/mdr_clean.csv` | `data/generated/tuning/` | 交叉验证调参；不直接输出训练好的预测模型 |
| `code/prediction/predict.py` | `data/inputs/alloy_prediction.csv`；`data/models/` 中所选模型和特征 | `data/generated/prediction.csv` | 输入、模型和输出均可通过参数指定 |
| `code/prediction/screen.py` | `data/models/`；元素可由参数指定；`alloys` 模式默认读取 `data/inputs/alloy_element.csv` | `data/generated/screening.csv` | 默认 Ga-In-Sn 筛选 |
| `code/visualization/plot_results.py` | 已保存的 `data/processed/`、`data/research/` 中的历史数据 | `paper/figures/generated/` | 重绘辅助图，不读取 `data/generated/` 作为默认数据源 |
| `code/paper_plots/data_plot.py` | MDR 清洗快照、历史筛选/预测表，以及代码中保留的实验数据 | `paper/figures/<研究主题>/` | 由原 Notebook 绘图单元适配；详见下表 |
| `code/paper_plots/gain_sn_plot.py` | `data/research/ga_in_sn/ga_in_sn_rf.csv` | `paper/figures/ga_in_sn/fig4d_1.png` | Ga-In-Sn 三元图 |
| `code/paper_plots/model_comparison.py` | `data/processed/mdr_clean.csv` | `paper/figures/model_comparison/` 中模型比较图 | 会重新训练多个模型，耗时且需要额外依赖 |

`code/common.py` 被数据处理、训练、调参、预测、筛选和辅助绘图脚本**导入**，提供路径、化学式解析、特征与模型读写等公共功能；这些入口脚本之间没有直接的自动调用关系。`code/paper_plots/` 的绘图脚本是从历史 Notebook 提取并适配的独立脚本，不依赖 `common.py`。

## 论文绘图代码与图片

以下图片按主题分别位于 `paper/figures/data_analysis/`、`paper/figures/alloy_screening/`、`paper/figures/model_comparison/` 和 `paper/figures/ga_in_sn/`；`data_plot.py` 的单元号对应文件中的 `# %% Original notebook cell ...`。输出为 PNG，重复运行会覆盖同名图片。

| 图片 | 脚本及单元 | 主要输入 |
| --- | --- | --- |
| `fig2a.png` | `data_plot.py`，1 | `data/processed/mdr_clean.csv` 的 Tc 分布 |
| `fig2b.png` | `data_plot.py`，3 | 同上，化学式的元素数量 |
| `fig2c.png` | `data_plot.py`，5 | 同上，Tc > 5 K 的元素频率和内置周期表位置 |
| `fig4b.png` | `data_plot.py`，7 | 单元中保留的采样实验数据 |
| `fig4c_2.png` | `data_plot.py`，9 | `data/research/screening/alloy_2_et.csv` |
| `fig4d_3.png` | `data_plot.py`，11 | `data/research/screening/alloy_3_rf.csv` |
| `fig4d_1.png` | `gain_sn_plot.py`，3 | `data/research/ga_in_sn/ga_in_sn_rf.csv` |
| `fig5_extra_trees.png` | `data_plot.py`，13 | `data/research/prediction/updated_mdr_et.csv` |
| `fig5d.png` | `data_plot.py`，15 | 单元中保留的实验数据 |
| `fig7a.png`、`fig7b.png`、`fig7c.png`、`fig7d.png` | `data_plot.py`，17、19、21、23（依次） | 对应单元中保留的实验数据 |

`paper/published/figures/` 保存论文及补充材料图片，来源见 `docs/provenance/artifact_manifest.csv` 和[补充材料说明](provenance/supplementary.md)。

## 数据与历史结果目录

| 路径 | 用途及关系 |
| --- | --- |
| `data/source/` | 原始导出及来源数据；`mdr.csv` 是当前清洗入口的默认输入 |
| `data/processed/` | 已保存的清洗、去重和质量比数据；默认训练和部分绘图读取此处 |
| `data/augmented/` | 保存的历史增强数据，不随 `augment.py` 自动更新 |
| `data/inputs/` | 预测示例、元素列表等；预测和筛选脚本可读取 |
| `data/non-superconductors/` | 保留的非超导体来源数据；不是上述脚本的默认输入 |
| `data/research/` | 保存的历史表格，主要供辅助重绘或论文绘图使用；不等同于当前脚本的新输出 |
| `paper/published/tables/` | 从作者稿提取的表格，来源见[补充材料说明](provenance/supplementary.md) |
| `data/generated/`、`data/models/` | 当前运行产生的结果或本地模型；不替代历史快照 |
