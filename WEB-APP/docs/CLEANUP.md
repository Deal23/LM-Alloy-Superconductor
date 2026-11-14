# 目录与清理说明

所有路径相对 `WEB-APP/`。环境、配置和数据位置见 [FILE_LOCATIONS.md](FILE_LOCATIONS.md)。

## 约定

```text
.venv/                  可重新安装的 Python 环境
app/、ml/、utils/       源代码，不清理
data/                   内置输入数据，不清理
  uploads/              上传文件 + datasets.json 索引，需备份
ml/models/              训练模型及配套特征/指标/元数据，需备份
results/
  logs/                 服务日志，可清理
  predictions/          筛选结果 CSV 和任务日志，按需备份
  phase/                成分相图完整预测 CSV，可能很大，按需备份
```

`results/` 保存日志和研究结果。服务停止后，可按需清理日志及不再需要的图表。

服务日志采用轮转：单文件最多约 1 MiB，保留 2 个备份，通常总量约 3 MiB；筛选任务自己的文本日志位于 `results/predictions/`，需按研究任务管理。

## 清理规则

1. **先停止服务与后台筛选**，避免仍有写入；不要在训练期间清理模型。
2. 日志、`__pycache__/`、`.pytest_cache/` 可删除，不影响研究数据。
3. `.venv/` 可删除重建，加载已有模型时使用兼容的 scikit-learn 版本。
4. 上传数据优先在「数据集」页面删除，让文件与索引同步；要批量清空新上传数据，应将整个 `data/uploads/` 备份或删除。不要只删 `datasets.json` 后留下无法从界面管理的文件。
5. 模型优先在「本地模型」页面删除。手动处理时，一个模型的 `key.joblib`、`key_feature.joblib`、`key_metrics.json`、`key_metadata.json` 应成套保留或移除。
6. `results/predictions/` 清理后，任务结果需重新计算。

## 不落盘的内容

- 单点和批量预测结果默认保存在浏览器页面内存，不自动存数据库。
- Web 图表预览通过内存 PNG 返回，下载位置由浏览器决定。
- 主题只在浏览器 localStorage 中保存。

Matplotlib、pip 等工具可能使用用户级缓存；可通过 `.venv/bin/python -m pip cache info` 查看 pip 缓存。
