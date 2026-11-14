# 数据

| 目录 | 内容 |
| --- | --- |
| `source/` | MDR 原始导出及人工整理的 `mdr.csv` |
| `processed/` | 清洗数据、转换前数据及重复记录 |
| `augmented/` | 历史重复采样与 SMOTE 数据快照 |
| `inputs/` | 合金元素表及预测输入示例 |
| `non-superconductors/` | 文献衍生的候选非超导体及化学式列表 |
| `research/` | 调参、预测、合金筛选和实验数据 |
| `models/` | 本地训练模型与特征信息 |

数据来源和处理记录见[数据溯源](../docs/provenance/data.md)；CSV 的行数、字段、编码和校验和见[数据清单](../docs/provenance/data_inventory.csv)。
