# LM-based SC

基于化学组成预测超导转变温度 Tc，涵盖合金与金属–非金属化合物。 [项目总览](../README.md)

项目从论文工作流发展而来，以加权 ExtraTrees 为基线，支持数据构建、评估、配方预测、组成搜索与可视化。

## 范围与数据

- 至少含一种金属；包含全金属合金、金属间化合物及硼化物、碳化物、氮化物、硫族化合物等金属–非金属体系。
- 按元素组成排除含 H 或 O 的记录，保留符合筛选规则的纯金属。
- 排除 Tc > 2 K 的纯 Ga 记录。
- 排除已核实的单层 FeSe 薄膜、高压及存疑记录，详见[数据说明](data/README.md)。
- 源数据来自 `JMaterSci_2025_60/data/processed/mdr_clean.csv`。最终数据 12,290 行、8,076 个唯一化学式、81 个元素原子分数特征，目标为 `Tc_K`。
- 使用当前数据训练模型后进行预测。

## 快速开始

在本项目目录运行。以下环境激活命令适用于 macOS/Linux；Windows 使用 `.venv\Scripts\activate`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r config/requirements.txt
python -m lmsc verify
python -m lmsc test -q

# 生成独立数据表和逐行筛选记录
python -m lmsc build --output artifacts/superconductors.csv --audit artifacts/cleaning_audit.csv
```

完整模型比较、调参及已有模型的兼容环境：

```bash
mamba env create -f config/environment.yml
mamba activate lmsc

# 训练后运行预测
python -m lmsc train
python -m lmsc predict Nb3Sn MgB2 FeSe
python -m lmsc predict --search --elements In Sn Nb --components 2,3 --denominator 12 --top-k 10
python -m lmsc predict --phase-diagram --elements Ga In Ag
```

命令说明：`python -m lmsc --help`、`python -m lmsc predict --help`。训练与预测输出写入 `models/` 和 `results/`。

## 结构

```text
LM-based-SC/
├── lmsc/             # 核心数据与训练模块；统一入口 python -m lmsc
│   ├── discovery/    # 预测、组成搜索与扫描
│   └── analysis/     # 数据、模型与结果分析
├── data/             # 数据集及资产校验清单
├── models/           # 特征顺序、模型说明、本地模型
├── results/          # 本地生成，不纳入 Git
│   ├── data/         # 数据分析图表
│   ├── training/     # 评估、调参、异常诊断图表
│   └── predictions/  # 预测、搜索及组成图
├── tests/
├── docs/             # 方法与开发说明
└── config/           # 完整研究环境与轻量测试依赖
```

共享路径集中在 `lmsc/paths.py`，结果按分析任务组织。

## 数据与预测

- 直接使用已清洗源表中的组成，不重复转换质量百分比；详见[数据说明](data/README.md)。
- 成分模型以化学组成为输入；压力、物态和制备条件可能影响 Tc。
- 树间标准差描述模型间离散程度；Tc–组成图展示预测值，区别于热力学相图。
- 模型文件保存在本地。

更多：[开发与命令](docs/development.md) · [模型](models/README.md) · [结果说明](results/README.md)

## 方法来源与许可

本项目基于 Chen Hua, Jing Liu. *Tree model machine learning to identify liquid metal-based alloy superconductor*. **J Mater Sci 60**, 11857–11877 (2025)。https://doi.org/10.1007/s10853-025-11121-z

引用元数据见 [CITATION.cff](CITATION.cff)，论文实现见 [JMaterSci_2025_60](../JMaterSci_2025_60/)。代码使用 [MIT License](LICENSE)，第三方材料遵循各自许可。
