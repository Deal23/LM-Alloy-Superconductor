# 数据文件原始说明

> 本表保留历史文件名及原记录的用途描述。当前路径和增强方式的注意事项见[记录索引](README.md)。

| 原始文件或目录 | 原始说明 |
| --- | --- |
| `20240322_MDR_OAndM.csv` | 官网原始数据 |
| `mdr.csv` | 数据清洗后数据（未进行 wt、at 转换） |
| `mdr_clean_wt.csv` | 数据清洗后数据（进行 wt 转换） |
| `mdr_clean.csv` | 数据清洗后数据（完全清洗） |
| `mdr_clean_os2.csv` | 数据清洗后数据（对非超导体数据进行 2 倍 SMOTE 过采样） |
| `mdr_clean_os5.csv` | 数据清洗后数据（对非超导体数据进行 5 倍 SMOTE 过采样） |
| `mdr_clean_od2.csv` | 数据清洗后数据（对非超导体数据进行 2 倍随机复制过采样） |
| `mdr_clean_od5.csv` | 数据清洗后数据（对非超导体数据进行 5 倍随机复制过采样） |
| `mdr_duplicated.csv` | 数据清洗后部分数据（同分子式多 Tc 值） |
| `非超导文件夹` | 非超导数据整理 |
| `alloy_element.csv` | 元素周期表金属元素识别 |
| `alloy4pre.csv` | 训练集以外的合金数据，用于预测 |
| `other4pre.csv` | 训练集以外的高温超导和非超导数据，用于预测 |
| `Data_clean.ipynb` | 数据处理 Python 程序 |
| `数据清洗.txt` | 数据处理结果记录 |
