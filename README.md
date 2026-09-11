# CUMCM 2026 A：统一建模代码

本仓库实现《药材的烘干问题》handoff 决策中的可复现代码：

- 一维径向圆柱热质传递主模型（问题1–4）；
- 问题4的移动材料坐标与已知半径收缩；
- 仅作用于外表面的潜热敏感性情景；
- 粗网格轴对称二维模型，用于一维降维检查；
- 官方 Excel 模板导出、事件记录、能量台账和验证报告。

## 运行

使用题目指定的 Python 环境执行：

```bash
/path/to/.venv-math-modeling/bin/python src/model.py
/path/to/.venv-math-modeling/bin/python src/postprocess.py
/path/to/.venv-math-modeling/bin/python validation.py
```

也可以从项目根目录使用 `python -m src.model`。输入文件位于 `A题/附件/`，原始输入只读。输出写入 `results/`，图形写入 `figures/`。

主模型不加入潜热；`results/energy_ledger.json` 是独立的表面潜热情景，不能解释为题面唯一正确的闭合。`results/shrinkage_scenarios.json` 记录轴向长度情景，主生产结果使用长度不变；`results/two_d_validation.json` 记录二维诊断。事件统一为全域最大含水率（包含中心对称重构和 Robin 表面重构）首次严格低于 0.15 kg/kg。

代码、结果和验证记录应以同一次运行生成的 `results/复现清单.json` 为准；未提供外部实验验证，因此输出是给定经验公式和假设下的模型预测。
