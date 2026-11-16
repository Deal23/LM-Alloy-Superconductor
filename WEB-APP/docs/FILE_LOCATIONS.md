# 文件与存储位置

以下路径相对于 `WEB-APP/`；运行时目录按需创建。

## 项目文件

| 路径 | 内容 |
|---|---|
| `run.py`、`app/` | Web 入口、API、服务与界面 |
| `ml/`、`utils/` | 模型、预测、筛选、可视化与数据处理代码 |
| `scripts/`、`tests/` | 安装启动脚本与测试 |
| `requirements.txt`、`.venv/` | Python 依赖清单与本地虚拟环境 |
| `data/superconductors.csv` | 默认建模数据，是 LM-based SC 建模表的副本 |
| `data/uploads/` | 用户上传数据及 `datasets.json` 索引 |
| `ml/models/` | 本地模型、特征、指标及训练元数据 |
| `results/predictions/`、`results/phase/` | 筛选结果及成分图网格数据 |
| `results/logs/` | 轮转应用日志 |
| `logs/`、`results/` | 本地日志及运行结果 |
| `.venv/`、`.venv-windows/` | 本地 Python 环境 |

更新保留上述用户数据、模型、结果和环境，也不清理项目其他目录。`app/`、`ml/`（除 `ml/models/`）、`utils/`、`scripts/` 中的程序文件会与远端同步：本地修改被覆盖，远端不存在的程序文件被删除；隐藏文件和缓存不在同步范围。根目录的入口、依赖清单及说明文件会被覆盖。更新期间的临时程序备份在同步成功后自动清除；同步失败时尝试恢复，如果恢复失败则保留临时恢复副本并提示路径。它不是用户数据备份，请自行备份重要数据及代码。旧版产生的 `.update-backups/` 不会自动删除。清理步骤见 [CLEANUP.md](CLEANUP.md)。

临时文件位于操作系统的临时目录；进程或系统意外中断时，可能需要手动清理以 `lmsc-update-` 开头的临时目录。恢复失败时的 `lmsc-recovery-` 目录请先检查并保存需要的文件，再手动清理。

## 配置

`app/config.py` 支持 `PROJECT_ROOT`、`MODEL_DIR`、`DATA_DIR`、`RESULTS_DIR`、`UPLOADS_DIR` 和 `SECRET_KEY`；`run.py` 支持 `PORT`。部分服务仍根据 `PROJECT_ROOT` 推导路径，单独修改目录变量不会迁移全部文件。部署时应配置自己的 `SECRET_KEY`。

基础依赖通过安装脚本安装在 `.venv/`；Excel 导入可选安装 `openpyxl`。迁移项目或使用已有模型时，建议重建虚拟环境并保持 scikit-learn 版本兼容。

## 浏览器与缓存

浏览器管理导出文件的下载位置，并在 localStorage 中保存主题。单点和批量预测保存在当前页面内存，任务状态保存在服务进程内存。日志、`__pycache__/` 和 `.pytest_cache/` 可在停止服务后清理；pip 等工具的用户级缓存由各工具管理。
