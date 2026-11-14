# 模型

| 文件 | 说明 |
|---|---|
| `feature_columns.json` | 81 个元素特征的固定顺序 |
| `model.joblib` | 训练生成的本地模型，不进入 Git |

训练以当前建模表为输入，评估指标与模型参数保存在本地 `results/training/`。

```bash
python -m lmsc train  # 覆盖当前模型、训练图表和指标
python -m lmsc predict Nb3Sn MgB2 FeSe
```

Python API：

```python
from lmsc.discovery.predict import load_artifacts, predict_batch

model, columns = load_artifacts()
print(predict_batch(model, columns, ["Nb3Sn", "MgB2"]))
```

加载已有模型时，使用可信来源及与训练时兼容的环境。
