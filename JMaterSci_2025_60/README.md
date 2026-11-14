# 液态金属合金超导体：树模型研究

基于树模型预测合金超导临界温度的 Python 工作流，包含论文数据和历史结果。

Chen Hua, Jing Liu. *Journal of Materials Science* **60**, 11857–11877 (2025). [论文 DOI](https://doi.org/10.1007/s10853-025-11121-z) · [引用信息](CITATION.cff)

## 快速开始

```bash
conda env create -f docs/environment/environment.yml
conda activate lm-alloy-superconductor
python -m unittest discover -s tests -v
python code/visualization/plot_results.py
```

绘图使用已保存的结果。其他操作见[脚本说明](code/README.md)、[工作流索引](docs/workflow.md)和[复现指南](docs/reproducibility.md)。

## 项目结构

| 目录 | 内容 |
| --- | --- |
| [data/](data/README.md) | 原始数据、历史结果、预测输入与本地模型 |
| [paper/](paper/README.md) | 论文最终材料及代码生成的图片 |
| [code/](code/README.md) | 处理、训练、预测、筛选和绘图 |
| [docs/](docs/workflow.md) | 数据关系和复现说明 |
| [tests/](tests/README.md) | 数据与工作流测试 |

软件采用 [MIT 许可](LICENSE)；第三方数据和图片保留各自权利。
