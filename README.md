# 中药材圆柱干燥模型最终基线

本仓库保存依据《final决策清单.md》和论文初稿整理的最终基线代码、完整结果、论文六个小表以及问题一二维轴对称验证结果。

## 目录

- `src/model.py`：一维径向主模型，覆盖问题一至问题四。
- `src/axisym_q1_72h.py`：问题一二维轴对称 (r)-(z) 验证，仿真至 72.0000 小时。
- `src/rebuild_chinese_tables.py`：生成与题目正文版式一致的六个中文小表。
- `src/decision_delivery.py`：基线诊断、敏感性和守恒检查。
- `data/附件/附件1.xlsx`、`data/附件/附件2.xlsx`：模型输入数据。
- `results/result1.xlsx` 至 `results/result4.xlsx`：按题目完整网格输出。
- `results/正文六个小表.xlsx`：论文正文六个小表，表头全中文，数值保留四位小数。
- `results/axisym_q1_72h/`：二维轴对称逐时对比、长时指标和期刊版图件。

## 运行

```bash
python -m pip install -r requirements.txt
python src/model.py
python src/rebuild_full_workbooks.py
python src/rebuild_chinese_tables.py
python src/axisym_q1_72h.py
python src/decision_delivery.py
python src/round_workbooks.py
python src/round_text_results.py
```

问题一、问题二采用固定时间步后向欧拉与节点中心有限体积法；问题二至问题四采用物性双向耦合和步内皮卡迭代；问题四使用材料坐标、实测半径的 PCHIP 插值，长度不变为基线情景；潜热仅作诊断，不进入生产主模型。问题三、问题四在 7200.0000 秒后采用 50.0000 摄氏度和 0.0500 的分段环境值。

六个正文小表的时间点严格采用题目要求：问题一为 100、300、600、900、1200、1500、1800 秒；问题二为 0.5000 至 3.0000 小时、间隔 0.5000 小时；问题三、问题四按 6.0000 小时递增至终止采样时刻。问题四增加“药材表面”列。
