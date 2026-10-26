# C15 真实数据 smoke：通过

冻结LA10 iter15000 Student/Teacher，cuda:0 A100，eval，seed142，原112×112×80裁块。C0固定两个L病例 `06SR5RBREL16DQ6M8LWS`、`1D7CUD1955YZPGK8XHJX`，各两裁块。未打开U图像或标签；Teacher未前向。

- 4/4裁块完成，96次单通道干预，96次有效，E3/D3/D2/D1各3个冻结通道，剂量0.01和0.03。
- 五节点均为原特征接口同一tensor对象；D3/D2 skip融合差为0，捕获/干预各调用一次，操作进入最终D1路径。
- 实际L2与目标匹配通过，最大相对误差 1.74e-06；最大alpha 0.035059，没有剪裁或将失败填零。
- A/C0/C1历史兼容性 120 项，最大相对误差 4.99e-07；联合及原LossNorm直接autograd重构70项，最大相对误差 8.28e-07。
- 所有参数/BN buffers、四种RNG状态及.grad槽不变；Teacher冻结、无残留hook。`execution_results.json`中13项状态检查全部为true。
- 无optimizer.step、checkpoint写入、正式训练或C2/C3。
- 运行826.06秒，峰值GPU分配4152.35MiB。

90项CPU单元测试（20新、70历史）及原verify_core通过。首个smoke曾因新分析代码未正确处理A归档的不可达头NaN而失败，完整保留在 `runs/smoke_failed01/`；修正和新增测试见EXECUTION_NOTES.md。不可达路径保留状态，重构梯度和时不贡献，但四头分母保持4。

为减少大数组中间乘积，完整诊断使用等价FP64归约：对本smoke全部4裁块×5节点、300组统计逐项核对，最大相对误差4.44e-13；余弦及有效性标签也一致。旧核心源码留在 `runs/smoke/source_snapshot/`，前后SHA见REDUCTION_EQUIVALENCE.json。此替换不更改网络、目标、配对、节点、预算或评分。

smoke的lambda=1、beta=.01仅作工程联合重构，不是完整诊断候选。工程gate允许进入预先固定的discovery标定及一次8L诊断。它不证明发现可靠边界通道或局部配对创新。
