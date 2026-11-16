# LMSC · 超导材料预测

基于树模型的超导临界温度（Tc）预测与高通量筛选工具。支持单点和批量预测、模型训练、数据集管理及结果可视化。

当前版本：**0.1.4** · Python 3.10+（推荐 3.12）

## 快速开始

下载并解压项目，在 `WEB-APP` 目录打开终端。首次安装需要联网。

**Windows**（安装 Python 时勾选 Add Python to PATH）：

```powershell
.\scripts\install.bat
.\scripts\start.bat
```

安装后，打开新终端也可在任意目录运行 `lmsc`。如已有同名命令，安装脚本不会覆盖它；移动项目后需重新运行安装脚本。

**macOS / Linux**：

```bash
./scripts/install.sh
./scripts/start.sh
```

浏览器打开 <http://127.0.0.1:5050>，按 `Ctrl+C` 停止服务。启动命令可加 `--port 8080` 指定端口。虚拟环境默认在用户本地目录，不在项目或同步文件夹内：macOS/Linux 为 `~/.local/share/LMSC/venvs/<项目路径哈希>/`，Windows 为 `%LOCALAPPDATA%\LMSC\venvs\<项目路径哈希>\`。每个项目路径使用独立环境；移动项目后需重新安装。可设置绝对路径环境变量 `LMSC_VENV_DIR` 自定义位置，安装、启动和注册时必须一致，建议选择同步目录以外的位置。

手动安装（macOS/Linux）：

```bash
VENV_DIR="$(python3 scripts/venv_path.py)"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install -r requirements.txt
"$VENV_DIR/bin/python" -B run.py
```

CSV 导入无需额外依赖。Excel 导入可选装 `openpyxl`：

```bash
# Excel .xlsx 导入（可选，CSV 不需要）
"$(python3 scripts/venv_path.py)/bin/python" -m pip install 'openpyxl>=3.1.0'
```

Windows 可运行 `python scripts\venv_path.py` 查看环境路径，然后使用该目录下的 `Scripts\python.exe` 安装可选依赖；安装后重启服务。旧版项目内 `.venv/` 和 `.venv-windows/` 不再使用：确认新环境可用后可手动删除。已有模型可能依赖旧版 scikit-learn，请在外置环境安装与模型训练时兼容的版本。

首次使用时，可在「训练模型」中使用默认数据集训练，或将可信的 `*.joblib` 和对应的 `*_feature.joblib` 放入 `ml/models/`。加载已有模型时使用与训练时相同的 scikit-learn 版本。

默认训练数据为本项目的 `data/superconductors.csv`，来自 `LM-based-SC/data/superconductors.csv` 的冻结版本（12,290 条、81 个元素原子分数，SHA-256：`7ea2fede6f0b3e633f5d388ef35a4bf7ff326898ea670cf58e1271a3b7e076fb`）。目标列为 `Tc_K`，列 `Tc` 是元素锝。**默认数据和基于它训练的模型**排除含 O/H 的化学式，且至少包含一种金属。自定义 `formula,Tc` 数据不受此体系限制：含 O/H 或纯非金属均可，训练时按原子分数编码；对应模型只能预测训练特征覆盖的元素。旧模型保留原有输入编码和默认体系限制。训练/测试为随机行切分，指标可能受重复组成影响，不能代表严格的组成外推能力。

概览页可直接命名并训练默认 ET；训练页可自定义算法及参数。训练进度根据已完成的树数更新，模型可在训练前命名，或在「本地模型」页重命名。训练任务保存在服务进程内，切换页面不会中断；服务重启会中止任务，页面刷新后不会自动恢复进度显示。

## 功能

- 单点和批量预测：批量最多 1,000 条，支持 CSV 导出。
- 模型训练、管理与数据集导入、预览和可视化。
- 成分相图：支持二元、三元及多元固定组分切片，并导出 PNG 和预测 CSV。高分辨率计算可能耗时很长并占用数 GB 磁盘；图像不代表物相稳定性。

所有计算在运行服务的机器完成。默认只监听 `127.0.0.1`；需要局域网访问时显式指定 `--host 0.0.0.0`。模型可靠度不是实验成功概率，预测结果需实验验证。

## 更新与数据

在左侧导航检查更新并确认安装。通过启动脚本或 `run.py` 运行时，更新完成后自动重启；其他运行方式需手动重启。更新会覆盖程序文件并删除远端不存在的程序文件，请先保存本地代码修改。同步期间使用临时备份，成功后自动清除，失败时尝试恢复。数据、模型、结果和运行环境保留，详见 [文件位置说明](docs/FILE_LOCATIONS.md)。依赖变化时需重新运行安装脚本。

无法连接 GitHub 时，可下载新版 ZIP 解压到新目录，再迁移数据和模型。文件位置见 [文件位置说明](docs/FILE_LOCATIONS.md)，清理方法见 [目录与清理说明](docs/CLEANUP.md)，基础数据来源见 [数据说明](data/README.md)。

发布新版本时，更新 `app/__init__.py` 的版本号及本文版本号，再推送代码。

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
