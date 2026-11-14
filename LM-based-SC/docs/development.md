# 开发与命令

## 开发约定

数据与训练模块位于 `lmsc/`，预测和扫描位于 `lmsc/discovery/`，分析和绘图位于 `lmsc/analysis/`；共享路径由 `lmsc/paths.py` 定义。

```bash
pip install -r config/requirements.txt
python -m lmsc verify
python -m lmsc test -q
```

完整模型比较使用 `config/environment.yml`，环境名为 `lmsc`。加载 joblib 模型时使用可信、版本兼容的环境。

## 命令

| 命令（前缀 `python -m lmsc`） | 用途 | 主要产物 |
|---|---|---|
| `build` | 源表清洗、过滤、原子分数归一化 | `data/superconductors.csv` |
| `train` | 模型比较、Optuna、重复分组验证、holdout | `models/`、`results/training/` |
| `predict Nb3Sn MgB2` | 化学式预测 | `results/predictions/predict_log.csv` |
| `predict --search --elements In Sn Nb` | 整数配比搜索 | `composition_optimization.csv` |
| `predict --phase-diagram --elements Nb Ga` | 二元/三元 Tc–组成图 | `results/predictions/phase_diagrams/` |
| `predict --analyze` | 分析预测日志 | `results/predictions/` |
| `anomaly` | OOF 残差、重复组成和新颖性诊断 | `results/training/anomaly_*` |
| `scan --bases Ga Bi In Sn` | 锚定基元素的三元扫描 | `phase_diagrams/base_scan/` |
| `elements` | 周期表频率图与计数表 | `results/data/` |
| `space` | 元素相空间、弦图、小提琴、矩阵 | `results/data/` |
| `screening` | 自备合金筛选表的可视化 | `results/predictions/` |
| `plot-training` | 从已保存结果重绘训练图 | `results/training/` |
| `verify` | 核对 Git 中冻结资产 SHA256 | 只读 |
| `test -q` | 核心回归测试 | 不要求模型文件 |

数据、训练、预测、扫描及绘图分析子命令可加 `--help`；`plot-training` 直接重绘，不提供参数。

## 数据与实验管理

```bash
python -m lmsc build --output artifacts/superconductors.csv --audit artifacts/cleaning_audit.csv
python -m lmsc verify
```

审计的 `source_row` 是源 CSV 数据行的零基位置，不含表头。

训练和预测输出到 `results/`，结果保存在本地。修改数据过滤、单位策略、特征或划分后，可更新资产清单并重新训练、报告指标。

开发规范见仓库根目录 [CONTRIBUTING.md](../../CONTRIBUTING.md)。
