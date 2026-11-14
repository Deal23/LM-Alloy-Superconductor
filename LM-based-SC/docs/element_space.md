# 元素相空间可视化

数据源：`data/superconductors.csv`，12,290 条最终建模记录（化学式 + Tc_K）。图表由以下命令生成。

复现：

```bash
python -m lmsc space
```

输出到 `results/data/`。

## 三类划分

最终数据不含 O/H，原六家族中的 Cuprate、Hydride、Oxide 不再出现。当前按元素集合分为三类：

| 类别 | 规则 | n | Tc 中位数 | Tc 最大 |
|---|---|---:|---:|---:|
| Alloy | 不含 Fe，且只含金属元素 | 4,579 | 3.47 K | 29 K |
| Fe-based | 含 Fe；优先判定，任意配比 | 1,636 | 14.50 K | 54 K |
| Metal-nonmetal | 不含 Fe，且含至少一种非金属/准金属 | 6,075 | 5.00 K | 41.4 K |

三类互斥，判定顺序固定：

1. 先挑出所有含 Fe 的记录，归为 `Fe-based`；
2. 在剩余记录中挑出只含金属的，归为 `Alloy`；
3. 剩下的归为 `Metal-nonmetal`。

非金属/准金属集合与 `lmsc/composition.py` 的 `NONMETALS` 一致：`B C N F Si P S Cl Ge As Se Br Sb Te I`。分类只看元素集合，不看配比；该分类用于可视化。

## 图 1 `data_element_chord.png` — 三类弦图

单行 3 面板：Alloy / Fe-based / Metal-nonmetal，共用 Tc 色标。

- 每面板取共现频率最高的 top-25 元素；
- 弦宽 ∝ 元素对共现次数；
- 弦色 = 该元素对的最大 Tc；
- 单质不构成元素对，已排除。

三类信号概览：

- **Alloy**：Nb–{Sn, Ga, Al, Ta}、V–Ga 等 A15 邻近组合最亮；
- **Fe-based**：所有含 Fe 体系，Fe–As、Fe–Se、Fe–Te、Fe–P 等较突出；
- **Metal-nonmetal**：不含 Fe 的金属-非金属体系，覆盖 B/C/Si/Ge/As/Se/Te 等，Nb–Ge、Mg–B 等分散出现。

## 图 2 `data_element_violin.png` — 元素数 × 三类 Tc 分布

- 横轴：元素种类数 2–8+；
- 每个元素数下并列 3 条小提琴；
- 黑线 = 中位数，同色折线连接各类中位数；
- 每格样本数 ≥3 才画小提琴；
- 只画元素数 ≥2 的记录，单质不计入小提琴图。

## 图 3 `data_element_anchor.png` — Ga / Bi / Sn / In 锚定矩阵

- 锚：Ga、Bi、Sn、In；排除含 O/H 的记录（当前最终表已无 O/H）；
- 每个锚取“共现次数 top-25 ∪ 与锚共现 Tc 最高 top-25”，再按最高 Tc 保留 25 个元素；
- 排序按二元 `anchor+A` 的最大 Tc 降序；
- 上三角 = `anchor+A+B` 的 max/median/min 三条色带；
- 对角线右上 = `anchor+A` 三条色带，左下 = `anchor+A` 样本数；
- 下三角 = 三元组合样本数（灰色，log 标度）；
- Tc 标尺动态，使用幂律拉伸低 Tc 区。

## 口径与限制

- 统计用“包含”语义：一个多元化合物会把它所有的元素对/三元组都计入。
- 图只依赖元素集合，不依赖配比，因此不受源数据原子百分比/质量百分比混用影响。
- 最终数据的单位限制仍见 `../data/README.md`；本图不做定量配比结论。
- 相图与元素相空间图不同：前者是模型预测的 Tc-成分图，不是热力学相边界。
