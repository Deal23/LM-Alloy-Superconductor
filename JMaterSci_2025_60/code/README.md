# Python 脚本

在项目根目录运行 `python code/<分类>/<脚本名>.py --help` 查看参数。

| 脚本 | 用途 |
| --- | --- |
| `common.py` | 化学式解析、特征、指标及公共路径 |
| `data/prepare_data.py` | 数据清洗、重复与异常记录 |
| `data/atomic_masses.csv` | 数据清洗所用的固定元素原子量表 |
| `data/augment.py` | 重复采样和 SMOTE 数据增强 |
| `modeling/train.py` | 模型训练、比较及评估 |
| `modeling/tune.py` | ET/RF 超参数网格搜索 |
| `prediction/predict.py` | 批量预测 CSV 数据 |
| `prediction/screen.py` | 合金筛选、采样规模实验和 Ga-In-Sn 预测 |
| `visualization/plot_results.py` | 从已保存的表格重新绘图 |
| `paper_plots/` | 从原始论文 Notebook 提取并适配的绘图脚本 |

运行方法见[复现指南](../docs/reproducibility.md)；原始绘图 Notebook 与脚本的对应关系见[论文绘图说明](paper_plots/README.md)。

原子量表取自 `periodictable` 2.1.0，包含 118 个元素；没有标准原子量的放射性元素采用该版本给出的代表质量数。
