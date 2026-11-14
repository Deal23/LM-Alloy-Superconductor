# 数据

`source.csv` 是 `JMaterSci_2025_60/data/processed/mdr_clean.csv` 的快照（20,995 行）。`superconductors.csv` 由 `python -m lmsc build` 生成，共 12,290 行、8,076 个唯一化学式。

构建规则：排除含 H 或 O 的组成，保留至少含一种金属元素的记录，排除 Tc > 2 K 的纯 Ga；保留原化学式和 Tc，将元素下标归一化为原子分数。其余纯金属及重复记录均保留。

另排除 6 条单层 FeSe 薄膜记录（80、65、60、55、55、40 K），以及已核实高压记录 Li 20 K、Sc 19.6 K 和 BaFe₂As₂ 35 K。TiS₂ 45 K 对应文献标题中的材料为 LiₓTi₁.₁S₂，组成和数值对应关系待确认，暂排除；YbFe₄Al₈ 50 K 的超导归属被[后续研究](https://arxiv.org/abs/2105.05064)质疑，也暂排除。

原始 MDR 表未逐条记录压力或薄膜状态，文献标题涉及高压或薄膜并不意味着所有记录均在该条件下测量；以上是逐项核查后的排除，不是完整的常压块体筛选。重复组成及其他待核查记录保留。

| 筛选步骤 | 行数 |
|---|---:|
| 源表 | 20,995 |
| 排除 H/O | −8,563 |
| 排除无金属 | −122 |
| 排除非晶 Ga | −9 |
| 排除 FeSe 薄膜 | −6 |
| 排除已核实高压、存疑组成及超导归属 | −5 |
| 最终 | 12,290 |

建模表包含 `formula`、81 个元素原子分数和目标 `Tc_K`；金属定义见 `lmsc/composition.py`。源表中的质量百分比转换及历史人工清洗说明见 `JMaterSci_2025_60/docs/provenance/data.md`。

`assets.json` 记录数据和特征列表的校验信息；运行 `python -m lmsc verify` 校验资产。可用 `python -m lmsc build --output artifacts/superconductors.csv --audit artifacts/cleaning_audit.csv` 生成独立结果和逐行筛选记录。
