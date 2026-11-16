# 目录与清理说明

项目内路径相对 `WEB-APP/`。完整位置见 [文件位置说明](FILE_LOCATIONS.md)。

## 约定

```text
app/、ml/、utils/       源代码，不清理
data/                   内置输入数据，不清理
  uploads/              上传文件 + datasets.json 索引，需备份
ml/models/              训练模型及配套特征/指标/元数据，需备份
results/
  logs/                 服务日志，可清理
  predictions/          筛选结果 CSV 和任务日志，按需备份
  phase/                成分相图完整预测 CSV，可能很大，按需备份
```

Python 环境位于项目外，路径及配置方法见 [文件位置说明](FILE_LOCATIONS.md)。

服务日志自动轮转；筛选任务日志与结果保存在 `results/predictions/`。

## 清理规则

停止服务和后台任务后，按需清理：

- 日志、`__pycache__/` 和 `.pytest_cache/` 可直接删除。
- 虚拟环境可通过安装脚本重建；额外安装的依赖需重新安装，已有模型需使用兼容的 scikit-learn 版本。旧版项目内 `.venv/`、`.venv-windows/` 不再使用。
- 上传数据优先在「数据集」页面删除，以同步文件和索引。
- 模型优先在「本地模型」页面删除；手动管理时，将模型、特征、指标和元数据文件成套处理。
- `results/` 中的预测和相图结果按需备份或删除，删除后需重新计算。

## 不落盘的内容

- 单点和批量预测结果默认保存在浏览器页面内存，不自动存数据库。
- Web 图表预览通过内存 PNG 返回，下载位置由浏览器决定。
- 主题只在浏览器 localStorage 中保存。

Matplotlib、pip 等工具可能使用用户级缓存，可通过 `python -m pip cache info` 查看 pip 缓存。
