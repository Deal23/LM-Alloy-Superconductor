# LMSC · 超导材料预测

基于树模型的超导临界温度（Tc）预测与高通量筛选工具。支持单点和批量预测、模型训练、数据集管理及结果可视化。

当前版本：**0.1.1**。应用可在左侧导航检查仓库默认分支中的新版本，并在确认后更新程序；数据、模型和结果保留在本地。发布新版本时更新 `app/__init__.py` 中的版本号并推送代码；依赖变化时重新运行安装脚本。更新后重启应用，远端已删除的旧程序文件不会自动移除。更新会覆盖程序文件，请先备份本地代码修改；无法连接 GitHub 时可下载新版 ZIP 解压到新目录，并按文件位置说明迁移数据和模型。

## 快速开始

使用 Python 3.10+（推荐 3.12）：

```bash
./scripts/install.sh
./scripts/start.sh
```

浏览器将打开 <http://127.0.0.1:5050>。可通过 `./scripts/start.sh --port 8080` 指定端口。Windows 用户依次运行 `scripts\install.bat` 和 `scripts\start.bat`。脚本在项目内创建 `.venv/`。Windows 安装脚本还会为当前用户注册 `lmsc` 命令：安装完成后打开新的终端，在任意目录运行 `lmsc` 即可启动；也可运行 `lmsc --port 8080`。如果已有同名命令，注册会跳过，仍可使用 `scripts\start.bat`。移动项目后需重新运行安装脚本以更新启动器。

手动安装（macOS/Linux）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B run.py
```

默认安装 Flask、NumPy、pandas、scikit-learn、joblib、Matplotlib 及其必要的传递依赖，支持预测、训练、CSV 导入及可视化。Excel 导入可按需安装：

```bash
# Excel .xlsx 导入（可选，CSV 不需要）
.venv/bin/python -m pip install 'openpyxl>=3.1.0'
```

Windows 将 `.venv/bin/python` 替换为 `.venv\Scripts\python.exe`，安装后重启服务。

首次使用时，可在「训练模型」中使用默认数据集训练，或将可信的 `*.joblib` 和对应的 `*_feature.joblib` 放入 `ml/models/`。加载已有模型时使用与训练时相同的 scikit-learn 版本。

默认训练数据为本项目的 `data/superconductors.csv`，来自 `LM-based-SC/data/superconductors.csv` 的冻结版本（12,290 条、81 个元素原子分数，SHA-256：`7ea2fede6f0b3e633f5d388ef35a4bf7ff326898ea670cf58e1271a3b7e076fb`）。目标列为 `Tc_K`，列 `Tc` 是元素锝。**默认数据和基于它训练的模型**排除含 O/H 的化学式，且至少包含一种金属。自定义 `formula,Tc` 数据不受此体系限制：含 O/H 或纯非金属均可，训练时按原子分数编码；对应模型只能预测训练特征覆盖的元素。旧模型保留原有输入编码和默认体系限制。训练/测试为随机行切分，指标可能受重复组成影响，不能代表严格的组成外推能力。

概览页可直接命名并训练默认 ET；训练页可自定义算法及参数。训练进度根据已完成的树数更新，模型可在训练前命名，或在「本地模型」页重命名。训练任务保存在服务进程内，切换页面不会中断；服务重启会中止任务，页面刷新后不会自动恢复进度显示。

## 界面与交互

- **概览**：快速预测、示例化学式、模型状态和评估指标。
- **单点预测**：Enter 提交、启发式可靠度评分和会话内最近 8 条记录（刷新即清除）。
- **批量预测**：支持换行、中英文逗号或分号；显示成功/失败统计、Top 10 Tc 对比，支持 UTF-8 CSV 导出；单次最多 1000 条。
- **数据集可视化**：从数据集页面进入，分析 Tc 分布、元素频次和组成元素数量并下载 SVG；当前数据集的高清图表可导出 PNG（默认 300 DPI，可设置 72–600 DPI）。

- **体验**：浅色/深色主题记忆、移动端导航遮罩、键盘操作、页面锚点链接（如 `/#batch`）。
- **成分相图**：二元 Tc 曲线、三元填色及多元固定组分切片（最多十种元素），支持 PNG 和完整预测 CSV。二元最多 1,000 等分，三元切片最多 10,000 等分；后台分块计算，图像最多显示 1,000 等分，完整结果保存于 `results/phase/`，可按需清理。高分辨率可能耗时很长并占用数 GB 磁盘。图为模型预测，不代表物相稳定性。
- **数据集**：基础数据与自定义数据集中展示，可进入详情查看说明、统计和分页预览；导入区提供 `formula,Tc` CSV 示例与模板。

所有计算在运行服务的机器完成。默认只监听 `127.0.0.1`；需要局域网访问时显式指定 `--host 0.0.0.0`。模型可靠度不是实验成功概率，预测结果需实验验证。

## 项目结构

```text
app/       Web 界面、API 与服务
docs/      文件位置与清理说明
ml/        模型训练、预测、筛选与可视化
data/      唯一内置基础数据及其说明（保留，不作缓存清理）
results/   运行产物：logs/、predictions/、phase/
utils/     数据处理与通用工具
scripts/   可选的安装和启动脚本
tests/     自动化测试
run.py     应用入口
```

环境、数据和模型的位置见 [文件位置说明](docs/FILE_LOCATIONS.md)；清理步骤见 [目录与清理说明](docs/CLEANUP.md)。

基础数据的来源和字段见 [`data/README.md`](data/README.md)。

## 哪些位置可以清理？

停止服务后，可按需管理以下文件：

| 位置（相对 `WEB-APP/`） | 用途 | 清理影响 |
| --- | --- | --- |
| `results/logs/` | 日志 | 可按需清理 |
| `results/predictions/` | 高通量候选 CSV、任务日志 | 先备份有用结果；清理后需重新计算 |
| `data/uploads/` | 上传数据与 `datasets.json` 索引 | 整体备份或通过页面删除 |
| `ml/models/` | 模型、特征、指标、训练信息 | 通过页面管理，配套文件一并处理 |
| `.venv/` | 项目 Python 环境 | 可重建；删除后需重新运行安装脚本 |
| `__pycache__/`、`.pytest_cache/` | Python/测试缓存 | 可清理；启动脚本已关闭新增字节码缓存 |
| `data/`、`app/`、`ml/`、`utils/` | 数据和源代码 | 项目运行所需 |

自定义数据导入后保存到 `data/uploads/`，可在数据集页面预览、删除或在训练页面选择。浏览器下载的 PNG/CSV 位于浏览器下载目录；预测记录保存在当前页面，主题偏好保存在浏览器中。

## 测试

安装 pytest 后，在项目目录运行：

```bash
python -B -m pytest -q -p no:cacheprovider
```

如已安装 Node.js，还可以运行零额外依赖的前端逻辑测试：

```bash
node --test tests/frontend.test.cjs
```

## 引用与许可

论文信息见 [CITATION.cff](CITATION.cff)，代码采用 [MIT 许可](LICENSE)。第三方数据和论文材料的使用权利需另行核对。
