# C1_VERIFICATION

结论：PASS（实施、复算和保护），不表示Transition研究假设成立。

## 实际执行

最终70项单元测试通过：C1新增18、C0 21、Stage0 7、Stage1A 10、Stage1B 14；verify_core.py通过。测试通过导入旧测试类执行，不调用旧报告写入入口。初次smoke前68项通过，之后新增D1余弦结构与低梯度汇总两项，最终70项全回归通过。

真实原定2L smoke：4crop/360次操作；完整4 discovery+4 probe：16crop/7392次操作。五层对象身份、原始模块、D3/D2 skip求和、形状/梯度形状、同节点削弱、一次hook、最终预测路径通过。全部正式操作都改变最终logits；argmax可能不变。实际能量最大相对误差1.7037452e-05，未匹配0次，未裁剪alpha。

Student/Teacher strict checkpoint加载，iter15000与原SHA通过。参数、全部state_dict/BN buffers精确不变；两模型eval；Teacher requires_grad=False且无前向；所有.grad均None；CPU/CUDA/NumPy/Python RNG不变；无遗留hook。只打开8个L病例image/label，数据大小/mtime不变，不打开U容器。test.list仅历史SHA保护，不解析或用于选择。

## 独立与兼容性检查

- 四尺度A原CE+Dice及B原边界CE公式复算；正式160条最终头参数梯度比较，另20条C0特征梯度比较；最大相对差异0。smoke另有60条比较。C1 Region纯CE与原A定义区分，不偷换Baseline训练目标。
- NumPy从原始特征/梯度/M独立复算36960个主目标一阶预测。最大inner误差7.22e-16，最大Delta_hat误差1.54e-16。
- 从原始最终logits、GT与位置对复算Region/Boundary/各Transition损失，最大绝对误差7.86e-12（阈值1e−10）。原生类别方向、相邻距离、置换多重集合、B/I各类别计数与不重叠、各层平均权重映射通过。
- discovery全部冻结评分可由其原始贡献表重建；主计划及margin0.5/2补充计划在probe前冻结，SHA/时间顺序通过；核心运行源码在discovery/probe间SHA相同。
- 源预测阈值按源目标单位传播，真实Boundary变化用Boundary阈值；整体真实近零比例8.455%，B为1.272%，I为15.639%，不强制正负。相同目标Taylor仅辅助检查；local GT归因与固定迁移分表。
- 3个probe裁块的Far源梯度低于固定阈值；相似度无效，本地病例归因要求两crop都可靠，否则未定义，不用低梯度冒充可靠零。完整特征Far覆盖81.25%，其余四主目标100%；D1支持结构零单独解释。
- 汇总层有效性修订后3360条固定迁移记录与修订前一致；原始运行数据及主结论不变。配对重采样单位是4病例，不是通道/体素/位置对。参数梯度与特征梯度文件/坐标独立。

## 历史保护与限制

1198项历史SHA复核通过，改动0。保护包括Baseline六组Results、checkpoint、summary、配置/快照、原生产代码、数据划分及Stage0/1A/1B/C0（C0的486项完整清单预核验通过）。HDF5不做容器全字节哈希，避免读取U标签字节；L访问记录和只读文件状态另核验。

0 optimizer步骤，0 checkpoint写入；未修改train.py、objectives.py、VNet默认行为或LossNorm。未运行C2/C3、正式训练、新辅助头/SDF损失、多通道组或交换。没有独立泛化验证。详细证据：TEST_RESULTS.json、VERIFICATION_RESULTS.json、POST_AUDIT_VERIFICATION.json、PROTECTION_BEFORE.json及各run的execution_results.json/source_at_run.json。

smoke后的归档补充、阈值量纲澄清和低信号汇总修订见EXECUTION_NOTES.md，均保留实际执行来源和原始结果，不将未执行检查列为通过。
