# C1 变更清单

仅新增 `analysis/stage1c_c1_boundary_specificity/` 与 `tests/test_stage1c_c1.py`。Baseline、Stage0/1A/1B/C0源码与结果未修改；1198项SHA两次结束核验通过。

- 方案/结论：C1_DESIGN.md、C1_REPORT.md、C1_FINDINGS.md、C1_VERIFICATION.md、SMOKE_REPORT.md、README.md、EXECUTION_NOTES.md。
- 运行：bridge.py只读复用原模块；control_pairs.py构造并审计对照；targets.py独立原始目标；runner.py执行冻结诊断；freeze_sensitivity.py在probe前冻结预定敏感性评分。
- 统计：measures.py、summary_validity.py、summarize_c1.py、control_summary_c1.py。空间梯度与参数坐标分开；病例是重采样单位；源信号无效不当可靠零。
- 验证：test_all.py、verify_c1.py、post_audit.py及18项C1新增测试；最终70项新旧单元测试与verify_core通过。
- 交付：report_c1.py、figures_c1.py生成报告与PNG/PDF。runs/保存原定smoke、discovery、probe，包含原始梯度数组和位置对、最终logits/有标签GT、损失/贡献/消融CSV、状态与来源SHA。
- 保护与复现：PREFLIGHT.json、PROTECTION_BEFORE.json、TEST_RESULTS.json、VERIFICATION_RESULTS.json、POST_AUDIT_VERIFICATION.json、FINAL_STATUS.json、ENVIRONMENT.json。所有新增文件与原始数组的大小/SHA见ARTIFACT_MANIFEST.json（清单自身除外）。

未新增训练损失/预测头/优化器或训练命令，未读取无标签GT，未进行多通道联合干预或C2/C3。
