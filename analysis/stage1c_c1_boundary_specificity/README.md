# Stage1C-C1 独立诊断

仅冻结有标签模型诊断，无训练、无新预测头、无通道组或交换。起点 C0 `ae073c5`；历史目录均只读。

## 阅读入口

- [C1_DESIGN.md](C1_DESIGN.md)：执行规则、主/敏感性参数、控制与统计单位。
- [SMOKE_REPORT.md](SMOKE_REPORT.md)：原定2L实施门槛；[EXECUTION_NOTES.md](EXECUTION_NOTES.md)：审计输出和量纲澄清的时间记录。
- [C1_REPORT.md](C1_REPORT.md)：8L执行和定量结果。
- [C1_FINDINGS.md](C1_FINDINGS.md)：Q1–Q4与下一步决策。
- [C1_VERIFICATION.md](C1_VERIFICATION.md)：测试、独立复算与历史SHA。

## 结果字段

`runs/{discovery,probe}` 各含4例×2原裁块。`losses.csv` 是各独立原始目标，无额外LossNorm。`contributions.csv` 分开保存梯度范数、`inner=sum(gFM)`、`abs_gxa=sum|gFM|`、`unit_response=-inner/||FM||`。它们是特征坐标；`parameter_norms.csv`、`raw/parameters/*.npy` 是另外的卷积核坐标，不能互换。

`local_attribution.csv` 中 `delta_hat` 和 `delta_actual` 是**源目标自身**的一阶预测和实际变化；`boundary_delta` 才是交叉终点的真实Boundary CE变化。`interventions.csv` 含最终CE、Dice、轮廓F1、预测改变率和实际L2。无效干预不填零。

`attribution_by_case.csv`：

- `local_GT_gradient` 用本例GT梯度，只是本地归因，不是迁移。
- `discovery_B_transfer` 用冻结discovery边界支持评分；B→B为主要终点，B→I是支持控制。
- `discovery_support_transfer` 分别使用冻结的B/I评分；I→I是补充控制。

各源目标损失单位不同，因此跨目标只比较排名、符号及影响强度相关性，不把Transition的Delta数值当作Boundary CE的定量预测。`absolute_spearman` 使用有符号净贡献的绝对值；`local_sum_absolute_gxa_rho` 是另一个幅值指标，不改变原有符号评分。方向准确率必须同时读取可判定率；真实变化与源预测分别采用各自单位的数值区间。另存剔除真实近零变化后的相关性，不把数值噪声排序当作有效信号。

`gradient_relations_by_case.csv` 只比较同裁块的目标梯度；`supported_feature_gradient_relations.csv` 额外限定相同M。`feature_contribution_stability.csv` 是空间归约后的通道响应模式；`parameter_direction_stability.csv` 才能在相同卷积核坐标上比较不同病例方向。不直接比较不同裁块空间坐标的特征梯度。

原始归档：每裁块五层 `raw/features/*.npz` 含激活、B/I掩码、五个主目标完整特征梯度；参数数组按实际输出通道轴展平；`raw/*_pairs.npz` 含各控制原生位置对；`raw/*_prediction.npz` 含最终logits、**有标签**GT和原生B/I支持。无无标签GT。margin0.5/2与额外置换的梯度统计/逐通道贡献/损失均保留；其完整图可由固定模型及代码重算。

`FROZEN_PLAN.json` 只使用discovery；`FROZEN_SENSITIVITY.json` 在probe访问前冻结margin0.5/2评分，候选不变。主评分对每病例先平均两裁块，再平均病例。`attribution_summary.json` 与 `paired_case_comparisons.json` 使用相同2000组病例重采样，单位为4个probe病例，不能把通道当样本量。

所有距离和边界带宽为原生体素单位。轮廓F1沿用B：6邻接前景表面，Chebyshev容差1体素，排除裁块边缘2体素。端点距预测轮廓的补充统计是裁块内距离，不能等同于整病例表面指标。

## 同节点映射

|阶段|实际特征位置/形状（含batch）|独立卷积核参考|
|---|---|---|
|E3|block_three输出，1×64×28×28×20|block_three.conv.6.weight，axis0|
|E4|block_four输出，1×128×14×14×10|block_four.conv.6.weight，axis0|
|D3|block_seven输入，skip融合后，1×64×28×28×20|block_six_up.conv.0.weight，axis1，仅上采样分支|
|D2|block_eight输入，skip融合后，1×32×56×56×40|block_seven_up.conv.0.weight，axis1，仅上采样分支|
|D1|out_conv输入，1×16×112×112×80|block_nine.conv.0.weight，axis0|

## 原始执行命令

从项目根目录，Python为 `/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python -B`。实际GPU0需要主机CUDA访问。已存在的结果目录不会被覆盖；以下是实际执行顺序，不能对原目录重复写入。

```bash
python -B analysis/stage1c_c1_boundary_specificity/test_all.py
python -B analysis/stage1c_c1_boundary_specificity/runner.py --phase smoke --output analysis/stage1c_c1_boundary_specificity/runs/smoke --device cuda:0
python -B analysis/stage1c_c1_boundary_specificity/runner.py --phase discovery --output analysis/stage1c_c1_boundary_specificity/runs/discovery --device cuda:0
python -B analysis/stage1c_c1_boundary_specificity/freeze_sensitivity.py
python -B analysis/stage1c_c1_boundary_specificity/runner.py --phase probe --output analysis/stage1c_c1_boundary_specificity/runs/probe --discovery analysis/stage1c_c1_boundary_specificity/runs/discovery --device cuda:0
python -B analysis/stage1c_c1_boundary_specificity/summarize_c1.py
python -B analysis/stage1c_c1_boundary_specificity/verify_c1.py
python -B analysis/stage1c_c1_boundary_specificity/control_summary_c1.py
python -B analysis/stage1c_c1_boundary_specificity/figures_c1.py
```

本次还执行了低梯度有效性汇总审计：保留 `TRANSFER_REFERENCE_BEFORE_VALIDITY_AUDIT.csv` 后重汇总，并运行 `post_audit.py` 确认3360条固定迁移结果不变，再用 `report_c1.py` 生成最终报告。复现主要实验无需引入未匹配梯度；`summary_validity.py` 已包含最终严格检查。审计脚本依赖本次保存的审计前参考表，不是新的实验或通道选择步骤。

所有L病例参与过Baseline训练，probe已有Stage1B探索历史。这里没有独立泛化验证，不解析test.list选通道或调参数。历史文件仅SHA保护；HDF5容器不做全字节哈希，以避免读取U标签字节。
