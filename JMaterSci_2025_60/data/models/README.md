# 本地模型

`code/modeling/train.py` 在此目录写入模型、特征顺序和指标文件。模型二进制及训练生成的元数据由 Git 忽略；使用训练得到的模型预测时，必须同时保留对应的特征顺序文件。

历史 ET/RF 模型超过 GitHub 的常规文件大小限制，不随仓库分发。可使用训练脚本重新生成，或在本地提供兼容的历史文件：

```bash
python code/prediction/predict.py --model ET --model-dir data/models/legacy
```

运行方法见[复现指南](../../docs/reproducibility.md)。请仅加载可信来源的 joblib/pickle 文件。
