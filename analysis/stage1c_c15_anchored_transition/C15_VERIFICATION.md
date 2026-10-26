# C15 验证与历史保护：PASS

验证依据为TEST_RESULTS.json、VERIFICATION_RESULTS.json和三阶段execution_results.json，均实际执行。

| 测试套件 | 项数 | 结果 |
| --- | --- | --- |
| test_stage1c_c15.py | 21 | PASS |
| test_stage1c_c1.py | 18 | PASS |
| test_stage1c_c0.py | 21 | PASS |
| test_stage_features.py | 7 | PASS |
| test_stage1a_diagnostics.py | 10 | PASS |
| test_stage1b_boundary.py | 14 | PASS |
| verify_core.py | 独立脚本 | PASS |

共91项单元测试，另原verify_core通过。新21项覆盖端点CE/方向、重复端点、两种平移、阈值与几何区别、空/非有限区域、零/低梯度、固定预算/通道/输出路径、skip节点、grad槽、Teacher冻结、U输入禁止、LossNorm重构、统计归约与softplus极值。旧70项覆盖Stage0/A/B/C0/C1全部原测试合同。

| 独立复核项目 | 结果 |
| --- | --- |
| test_suites | PASS |
| three_execution_gates | PASS |
| state_rng_BN_grad_slots_Lonly | PASS |
| discovery_only_plan | PASS |
| plan_hash_unchanged | PASS |
| core_sources_match_runs | PASS |
| smoke_reduction_equivalence | PASS |
| A_C0_C1_compatibility | PASS |
| direct_joint_LossNorm_autograd | PASS |
| true_pair_geometry_orientation | PASS |
| permuted_endpoint_marginals | PASS |
| independent_raw_losses | PASS |
| feature_mask_native_average | PASS |
| independent_signed_GxA | PASS |
| anchor_endpoint_CE_numeric_equivalence | PASS |
| all_parameter_gram_diagonal | PASS |
| fixed_intervention_count | PASS |
| energy_matching_valid | PASS |
| alpha_never_clipped | PASS |
| no_E4_interventions | PASS |
| fixed_channels_no_probe_reselection | PASS |
| discovery_scores_reconstruct | PASS |
| historical_sha256_1931 | PASS |

- A/C0/C1兼容720项，最大相对误差6.19e-07（预设1e−4或绝对1e−9）。
- 联合梯度/原LossNorm直接autograd重构140项，在smoke和main首裁块分别检查，保留feature/parameter/LossNorm_parameter坐标；其他裁块使用已核验的线性组合，并不声称逐项都另跑了联合backward。
- 全16crop×5节点原始数组独立FP64 G×A重算，最大相对误差2.94e-13；原生边界平均池化mask最大误差0。
- 全16crop独立Anchor/True/Perm算子重算；显式匹配PyTorch threshold20，不用放宽容差掩盖差异。Anchor与精确端点CE最大差5.55e-12，极值单测允许默认softplus线性近似的2.1e−9。
- 所有现有参数逐项范数平方与全θ Gram对角一致；不可达辅助头记unreachable，归档NaN仅在已验证的图路径求和中不贡献，四头分母保持4。零/不可达不是可靠零分。
- 384/384物理干预有效；384次最终logits发生非零变化，0次无效。alpha不剪裁，空间支持和实际L2匹配。
- Student/Teacher完整state_dict（含BN buffers）逐bit不变、Teacher无梯度、所有.grad=None、eval保持、CPU/CUDA0/NumPy/Python RNG不变、无残留hook；数据文件size/mtime不变。checkpoint strict加载且SHA不变。
- 历史SHA核对1931文件，变更0：原Baseline关键源文件、全部历史分析/配置/快照、六组Results、summary、checkpoint和划分列表保持。HDF5只读L image/label，不哈希U容器或读取U GT。test.list只作文件保护，不用于选择。
- 原Stage0九阶段接口、VNet默认forward、训练/评估、LossNorm/warm-up/EMA/Optimizer均未修改。Teacher生产训练语义不变，本轮只eval诊断。

首次smoke失败和第一次NumPy复核差异详见EXECUTION_NOTES.md，未抹去失败或重跑完整8L来挑结果。统计优化额外验证20节点/300组，最大相对差4.44e-13，前后核心SHA及旧源码快照可复核。

CPU命令：`python -B analysis/stage1c_c15_anchored_transition/test_all_c15.py`。GPU冻结三阶段使用`run_c15.py --phase smoke|calibration|main --device cuda:0 --output .../runs/<phase>`，依次执行，输出目录必须不存在。随后`check_reductions_c15.py`（smoke后）、`summarize_c15.py`、`verify_c15.py`、本报告生成器。真实复现需本地旧A/B/C0/C1原始数组、数据和checkpoint；大型NPZ按现有规则仅本地保存。

边界：未执行新U前向、真实train-mode mini-batch、独立有标签泛化验证或正式训练；MT为原2U四crop的只读梯度参考。不虚称这些测试已通过。无C2/C3、通道组/交换或新预测头。
