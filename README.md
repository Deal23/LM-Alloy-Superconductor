# LM-Alloy-Superconductor

Chen Hua, Jing Liu. **Tree model machine learning to identify liquid metal-based alloy superconductor.** *Journal of Materials Science* **60**, 11857–11877 (2025).

[论文 DOI](https://doi.org/10.1007/s10853-025-11121-z) · [引用信息](CITATION.cff) · [MIT License](LICENSE)

面向超导材料的数据驱动研究：从论文复现与数据整理，到临界温度（Tc）预测、候选筛选和可视化。

## 项目

| 目录 | 简介 |
| --- | --- |
| [JMaterSci_2025_60](JMaterSci_2025_60/) | 上述 *Journal of Materials Science* 论文的研究代码、数据与结果复现 |
| [LM-based-SC](LM-based-SC/) | 金属基超导材料的 Tc 预测、配比搜索与可视化 |
| [WEB-APP](WEB-APP/) | 模型训练、预测与筛选的浏览器工作台 |

各项目可独立运行。`LM-based-SC` 基于论文项目的 `mdr_clean` 数据构建建模表；Web 工作台使用该表的本地副本。数据范围与清洗规则见[数据说明](LM-based-SC/data/README.md)。

开发与验证约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 快速开始

克隆仓库后，在仓库根目录选择需要的项目：

```bash
git clone https://github.com/Deal23/LM-Alloy-Superconductor.git
cd LM-Alloy-Superconductor
```

**论文研究**：配置环境并运行测试；复现流程和历史结果见[项目文档](JMaterSci_2025_60/README.md)。

```bash
cd JMaterSci_2025_60
conda env create -f docs/environment/environment.yml
conda activate lm-alloy-superconductor
python -m unittest discover -s tests -v
```

**体系预测**：安装轻量依赖并运行测试；训练、预测和筛选方法见[项目文档](LM-based-SC/README.md)。

```bash
cd LM-based-SC
python -m venv .venv
source .venv/bin/activate
pip install -r config/requirements.txt
python -m lmsc test -q
```

**Web 应用**：macOS/Linux 使用以下命令；Windows 安装与启动方式见 [Web 文档](WEB-APP/README.md)。启动后访问 <http://localhost:5050>。

```bash
cd WEB-APP
./scripts/install.sh
./scripts/start.sh
```

以上命令分别从仓库根目录进入项目；体系预测的虚拟环境激活命令适用于 macOS/Linux。

## 引用与许可

使用论文方法或数据时，请引用上述文献，并注明实际使用的项目及版本。代码采用 MIT 许可；第三方数据、论文和图像遵循各自的许可。
