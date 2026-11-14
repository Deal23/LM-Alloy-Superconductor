# 论文绘图代码

代码提取自 Git 提交 `82898a6302875ed6423fad9b2f9bd24f890ee6bd` 的 Notebook；`# %%` 标明原代码单元索引。输出改为 300 dpi PNG，周期表位置改用内置元素数据，不再依赖 `mendeleev`；适配了输入输出路径和 scikit-learn 的 RMSE 接口，启用了原先注释的图片保存语句。

| 脚本 | 原始 Notebook | 范围 |
| --- | --- | --- |
| `data_plot.py` | `Pic/data_plot.ipynb` | 全部代码单元 |
| `model_comparison.py` | `Code/1_Tree_model_train/LMSC-Regression.ipynb` | 模型比较单元（索引 5） |
| `gain_sn_plot.py` | `Code/5_GaInSn_best_predict/best-Alloy.ipynb` | 三元图单元（索引 3） |

可在支持 `# %%` 的编辑器中按需运行单元，或使用 Python 运行整个脚本；输出位于对应的 `paper/figures/<研究主题>/`，同名文件会被覆盖。部分单元需要 `seaborn`。模型比较单元会重新训练模型，并需要 CatBoost、XGBoost 和 LightGBM。
