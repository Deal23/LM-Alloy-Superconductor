# 文件与存储位置

项目内路径相对于 `WEB-APP/`；运行时目录按需创建。

## 项目文件

| 路径 | 内容 |
|---|---|
| `run.py`、`app/` | Web 入口、API、服务与界面 |
| `ml/`、`utils/` | 模型、预测、筛选、可视化与数据处理代码 |
| `scripts/`、`tests/` | 安装启动脚本与测试 |
| `requirements.txt` | Python 依赖清单 |
| `data/superconductors.csv` | 默认建模数据，是 LM-based SC 建模表的副本 |
| `data/uploads/` | 用户上传数据及 `datasets.json` 索引 |
| `ml/models/` | 本地模型、特征、指标及训练元数据 |
| `results/predictions/`、`results/phase/` | 筛选结果及成分图网格数据 |
| `results/logs/` | 轮转应用日志 |
| `logs/`、`results/` | 本地日志及运行结果 |
| `.venv/`、`.venv-windows/` | 旧版项目内环境，不再使用；确认新环境正常后可删除 |

更新保留用户数据、模型、结果和环境。`app/`、`ml/`（除 `ml/models/`）、`utils/`、`scripts/` 中的程序文件与远端同步；本地修改会被覆盖，远端不存在的程序文件会被删除。根目录的入口、依赖清单和说明文件也会被覆盖。更新失败时会尝试恢复临时备份；清理方法见 [CLEANUP.md](CLEANUP.md)。

更新临时文件位于系统临时目录，名称以 `lmsc-update-` 或 `lmsc-recovery-` 开头。

## 配置

`app/config.py` 支持 `PROJECT_ROOT`、`MODEL_DIR`、`DATA_DIR`、`RESULTS_DIR`、`UPLOADS_DIR` 和 `SECRET_KEY`；`run.py` 支持 `PORT`。部分服务仍根据 `PROJECT_ROOT` 推导路径，单独修改目录变量不会迁移全部文件。部署时应配置自己的 `SECRET_KEY`。

基础依赖通过安装脚本安装到项目外：macOS/Linux 使用 `~/.local/share/LMSC/venvs/<项目路径哈希>/`，Windows 使用 `%LOCALAPPDATA%\LMSC\venvs\<项目路径哈希>\`。运行 `python3 scripts/venv_path.py`（Windows 使用 `python`）查看实际位置；可通过绝对路径 `LMSC_VENV_DIR` 覆盖，安装及启动时必须保持一致。项目路径变化后需重新安装；不要直接移动虚拟环境。Excel 导入可选安装 `openpyxl`，使用已有模型时应保持 scikit-learn 版本兼容。

## 浏览器与缓存

浏览器管理导出文件的下载位置，并在 localStorage 中保存主题。单点和批量预测保存在当前页面内存，任务状态保存在服务进程内存。日志、`__pycache__/` 和 `.pytest_cache/` 可在停止服务后清理；pip 等工具的用户级缓存由各工具管理。
