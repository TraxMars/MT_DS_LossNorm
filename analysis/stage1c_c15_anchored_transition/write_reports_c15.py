"""Generate final human-readable reports after independent verification passes."""
import json
import numpy as np
import pandas as pd
from c15_common import ROOT,STAGES,INTERVENTION_STAGES,write_json

def read(p):return json.loads(p.read_text())
def md(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def fmt(v,d=3):return 'NA' if v is None or pd.isna(v) else f'{v:.{d}f}'

def run():
    v=read(ROOT/'VERIFICATION_RESULTS.json');assert v['status']=='PASS'
    plan=read(ROOT/'runs/calibration/FROZEN_PLAN.json');test=read(ROOT/'TEST_RESULTS.json');eq=read(ROOT/'REDUCTION_EQUIVALENCE.json')
    results={n:read(ROOT/'runs'/n/'execution_results.json') for n in ('smoke','calibration','main')}
    pairs=pd.read_csv(ROOT/'summary/pair_geometry_and_marginals.csv');grad=pd.read_csv(ROOT/'summary/gradient_global_pairs.csv')
    conflicts=pd.read_csv(ROOT/'runs/main/conflicts.csv');g=conflicts[(conflicts.budget=='r30')&(conflicts.increment=='transition')]
    fixed=pd.read_csv(ROOT/'summary/attribution_probe_case.csv');fixed=fixed[(fixed.method=='discovery_fixed')&(fixed.dose==.03)]
    con=pd.read_csv(ROOT/'runs/main/interventions.csv');actual=con[(con.dose==.03)&con.case.isin(['0RZDK210BSMWAA6467LU','1GU15S0GJ6PFNARO469W','23X6SY44VT9KFHR7S7OC','38CWS74285MFGZZXR09Z'])]
    classes=pd.read_csv(ROOT/'summary/endpoint_class_groups.csv');classes=classes[classes.objective=='anchor']
    gt=[]
    for s in STAGES:
        def value(kind,left,right):return grad[(grad.stage==s)&(grad.coordinate_kind==kind)&(grad.left==left)&(grad.right==right)].cosine.mean()
        gt.append([s,fmt(value('feature','anchor','transition')),fmt(value('parameter','anchor','transition')),fmt(value('feature','boundary','transition')),fmt(value('parameter','boundary','transition'))])
    attribution=[]
    for s in INTERVENTION_STAGES:
        r=[]
        for n in ('boundary','anchor','true_r30','perm_r30'):r.append(fmt(fixed[(fixed.stage==s)&(fixed.objective==n)].signed_rho.mean()))
        attribution.append([s,*r])
    class_table=[]
    for (fg,bg),r in classes.groupby(['fg_correct','bg_correct']):
        class_table.append(['正确' if fg else '错误','正确' if bg else '错误',f'{100*r.fraction.mean():.3f}%',fmt(r.anchor_mean.mean()),fmt(r.true_transition_mean.mean())])
    actual_table=[]
    for s,r in actual.groupby('stage'):
        actual_table.append([s,*[f'{r[c].mean():+.3g}' for c in ('boundary_delta','endpoint_error_delta','boundary_f1_delta','dice_delta','changed_prediction_fraction')]])
    lnorm=pd.read_csv(ROOT/'runs/main/lossnorm.csv');ln=lnorm[lnorm.coordinate_kind=='all_parameters']
    ln_table=[]
    for name in ('boundary','anchor','true_r10','true_r30','perm_r10','perm_r30'):
        r=ln[ln.objective==name];ln_table.append([name,fmt(100*r.aux_to_sup_raw_ratio.mean(),2)+'%',fmt(100*r.aux_to_sup_raw_ratio.max(),2)+'%',fmt(r.independent_aux_to_joint_aux_ratio.mean(),4)])
    report=f'''# C15 完整诊断报告

工程结论：通过。方法结论：Anchor补足了纯Transition缺失的绝对分类约束；当前没有证据支持真实局部Transition比同端点置换提供额外、可重复的边界质量价值。未训练新模型。

## 起点与执行

工作仓库MT_DS_LossNorm_Boundary，分支`experiment/stage1c-c15-anchored-transition`，起点`d7b50ccb5b458a9d5a72e65fa0b21c0cd6cbcf56`。GitHub main `493fa83eca062aa13eee9c4c2dafca56cd27fea8`，438个远程blob与本地相同，另41个本地图件/完整统计文件保留。执行中复核远程HEAD未变；未merge/reset/pull/push。

LA10 iteration15000配对checkpoint SHA256 `85489c52c0905aebde65eacf5cea215aac51ee825ed3e5a845da86fe7f23e168`，Student/Teacher strict加载。模型eval，Student允许autograd，Teacher冻结且不前向，FP32网络/FP64诊断、seed142、原112×112×80裁块、TF32关闭。C1原4 discovery/4 probe L病例，每例两crop。probe参加过训练及前期分析，不是独立盲验证。没有用test.list选方案，也没有打开U容器。

先2L×2crop smoke，后4 discovery L标定，再仅一次8L×2crop完整诊断。耗时分别{results['smoke']['elapsed_seconds']:.2f}s、{results['calibration']['elapsed_seconds']:.2f}s、{results['main']['elapsed_seconds']:.2f}s。完整运行384次物理干预、3456条分目标归因比较；全部384次能量有效，不能将目标/通道/体素/位置对数当病例N。完整峰值GPU分配{results['main']['peak_allocated_mib']:.2f}MiB。

完整冻结计划：λ10={plan['lambdas']['r10']:.9f}，主λ30={plan['lambdas']['r30']:.9f}；共同模拟β={plan['outer_beta']:.12f}。标定坐标为全部现有可训练参数θ，True增量预算0.10/0.30，β控制discovery六候选中最大的辅助/监督范数比为0.10。C/D用同λ，不为Perm单独归一化。固定计划SHA256 `27a395986bfc732a06ff9def1a3ae19edffc9efa7d2530f96f2726e998f3b5b3`。probe无重排、筛除或权重修改。

干预通道E3/D3=[0,32,63]，D2=[0,16,31]，D1=[0,8,15]；E4只作梯度统计。均来自旧C1 discovery冻结计划中的C0锚点。主剂量.03，敏感性.01；复用C1同GT边界掩码和q，α=目标L2/||FM||，α上限.5、不剪裁。最大实际α={con.alpha.max():.6f}，最大L2相对误差{abs(con.actual_l2/con.target_l2-1).max():.3g}。

## 目标与分类锚定

全部目标使用最终D1 logits。A为原宽2体素平衡Boundary CE；B为同原始pair多重集合的Anchor=.5mean[softplus(−z+)+softplus(z−)]；C=B+λTrue；D=B+λPerm。Region CE、纯Transition仅作解释。原始目标分别保存，C/D总值不能直接评优。

原始2K pair/crop中，两端状态如下。分类比例先crop、后病例等权；组内loss为非空crop的描述平均，稀少双错误组不作病例级推断，空组不填0：

{md(['FG端','BG端','比例','Anchor均值','True均值'],class_table)}

True margin≥1比例{100*pairs.margin_satisfied_fraction.mean():.2f}%；在已满足margin的pair内仍至少一端错误{100*pairs.wrong_given_satisfied.mean():.2f}%。Anchor对FG错误端的单pair导数≤−.25，对BG错误端≥.25；正λTrue也保持这些分类修正方向。所有真实端点的Anchor、True、Perm原始logit梯度方向均正确，但纯True仍允许整体z偏移，其相对gap不锚定分类阈值。

必须区分两种平移：同一点两个类别logit同时平移（softmax gauge）使Anchor/True都不变；两个端点的z同时加常量使True不变、Anchor通常改变。Anchor在数值上等价端点分类CE，默认PyTorch softplus的threshold=20线性近似使真实数据最大差{v['max_anchor_endpoint_CE_difference']:.3g}，不是所有极值下逐bit相等。

独立z空间max-abs .05示意步后pair错误率：原59.2651%，Anchor59.2133%，主C59.2255%，主D59.1980%。这只是冻结logit空间的小步示意，不是参数训练或模型准确率提升。仅双端相反符号时计算边上零交点；无交点记NA。既有共同交点的归一化中点误差约Anchor .233051→.232656、C .233035→.232785、D .233051→.232648，纳入的共同pair略不同，不能据此比较优劣。即使两端分类CE近零，z+=(1000)、z−=(−50)的线性交点仍为.95238而非.5。Anchor与联合目标均不是完整几何轮廓距离损失。

## 梯度互补与冲突

节点E3/E4为编码block输出，D3/D2分别为block_seven/block_eight输入（skip相加后），D1为out_conv输入。捕获、特征接口、真实削弱同tensor且调用一次。对应阶段卷积参数沿用原映射；D3/D2反卷积权重只描述上采样分支，不能冒充融合后的特征。详细参数/通道轴见runs/main/parameter_mapping.json。

下表为8病例两crop等权平均，有符号余弦：

{md(['阶段','特征 Anchor/True','参数 Anchor/True','特征 Boundary/True','参数 Boundary/True'],gt)}

全θ Anchor/True余弦均值−.476599，16/16 crop为负；主λ的True增量/Anchor范数范围.057565～.400531，均值.197149。Perm全θ余弦+.827538，有效增量比.786281～1.743652，均值1.251362。相同λ不意味着相同有效梯度强度，Perm更强是重要解释限制。

全θ主C抵消率均值.228503，主D .043956。增加True的额外Anchor一阶变化率−λ dot(gA,gT)为正，表示参数域增量冲突；本轮两个λ下所有阶段与全θ的总方向−(||gA||²+λdot)仍为下降，没有总方向升高Anchor的观测。discovery安全部分来自预设范数预算，本身不是互补性证据。

特征域多为正且非共线，参数域可反向，源于网络Jacobian投影。只能说存在不同的重加权与非共线分量，不能将其自动解释为有益边界互补，也不能把负余弦称为有害通道。逐通道正负dot、零/低阈值状态、范数、abs G×A见channel_relations/contributions及summary/channel_dot_signs.csv。

参数共享坐标可比较病例间方向；特征仅比较空间归约后的单位L2贡献，不跨病例比较体素坐标。D1参数True跨病例cos约.975、同病例双crop约.990，但高稳定不代表质量价值；E4 True跨病例约.106，E3约.298，D3约.252，位置/病例稳定性有限。详表保留全部方法及discovery/probe分别结果。

## 严格配对负对照

True/Perm均每crop2048 pair，完整FG/BG端点多重集合及次数相同，Anchor完全共用，端点logit/CE/熵Wasserstein均为0。True距离1 voxel；Perm平均47.990 voxel，相同pair仅约.052%。平均gap同为7.209，gap标准差True4.511、Perm9.990；True原始loss .125298、Perm1.655370。差异说明配对改变联合难度与梯度权重，不说明True更优。

True/Perm logit梯度cos均值.375095，相对梯度差2.326468。这个可计算差异并未转化为下述固定评分的额外边界解释能力。C1补充远距离匹配只复用原存档：保留率49.55%，True共同子集loss .0810089、Far匹配 .0809413；匹配不足且未完全相同，不能作为纯几何因果对照。未扩大负对照或重新挑配对。

## 有限干预与跨目标归因

所有目标使用同一个边界空间支持、节点、通道、剂量和实际L2。固定预测=discovery每病例双crop平均、再4病例平均的有符号单位L2响应×probe实际预算。probe GT只定义允许的支持与评价；没有使用probe梯度重排通道。local_GT_attribution另列，不能混入跨病例迁移结论。

主剂量.03，预测与Boundary实际变化均超各自ε，4个probe病例的Spearman均值：

{md(['阶段','Boundary归因','Anchor归因','Anchor+True','Anchor+Perm'],attribution)}

True相对Perm的signed相关性病例配对差四层均0；三候选通道的排序分辨率很低，区间[0,0]不是方法等价证明。True相对Anchor仅E3均值差+.375，探索区间[−.25,.875]，且两者平均rho仍为负。主C并未稳定超过Boundary参照。D2绝对强度rho由Anchor0变为C .75，但D也为.75；D3的abs rho为Anchor与C均0、D为−.375，未形成跨层/独立轮廓一致优势。λ10时E3仍为−.75，未事后改主权重。

主剂量方向准确率，各方法相同：E3 .333、D3 .750、D2 .583、D1 1.000。D1的rho=1是有限通道和局部结构的忠实性表现，不是可靠边界通道发现。主剂量病例均值方向覆盖100%；物理干预层面，.01下全8L 4/192（2.083%）Boundary变化落入不确定区间，probe 4/96（4.167%），.03为0。近零均标indeterminate，没有强制赋正负。完整ε源于零剂量同batch对照，数据逐项保存。

以下为probe .03共同物理干预池的实际最终指标均值，各评分方法共用这一池，不能把它当C/D模型性能差异：

{md(['阶段','ΔBoundaryCE','Δ端点错误','Δ轮廓F1','ΔDice','预测改变率'],actual_table)}

probe .03有58.33%的单次轮廓F1变化为0，.01为78.125%；E3/D3的F1相关性只有3/4病例可算，不给完整4例CI。D1四方法F1相关性均−.125、D2均−.375，没有独立轮廓优势。F1越高越好、loss越低越好，相关方向必须按原符号解释，不翻转评分制造优势。端点错误、CE、Dice、预测改变率全部保存，绝不以自目标Taylor代替质量终点。

## LossNorm与未来训练

原Lsup为四头mean[(CE+Dice)/2]，Lmt为四头softmax概率MSE mean；warm-up后N(Lsup)+.3N(Lmt)，N(L)=L/stopgrad(max(L,1e−8))。本轮梯度关系全部用原始目标，不独立归一化边界目标后冒充raw。

仅模拟N(Lsup+βLaux)及其参数贡献(gSup+βgAux)/den。共同β的全θ加权辅助/监督比，以及独立归一化风险参照：

{md(['辅助候选','平均β||gAux||/||gSup||','最大值','独立Norm辅助/联合Norm辅助均值'],ln_table)}

主C的全θ比平均3.276%，stage D1均值10.509%，说明全局预算不保证每阶段相同比例。Perm主档probe最大10.679%略超过discovery预算，未裁剪或重定标。独立Norm在当前候选及纯Transition参照中主要使辅助贡献缩小（非放大）：所有全θ记录比率约.00882～.57202；不能将理论“小loss可能放大”误报为本轮已经放大。极小loss放大由单元测试单独验证，未来训练损失动态变化仍需监测。

MT只读取Stage1A原2U×2crop梯度存档，按原可达头sum/4及.3/N分母重构；没有打开U图像或GT。summary/lossnorm_MT_stage_reference.csv与新L贡献同参数坐标列示，例如D1主C联合Norm均值.354819、辅助分量.031917，历史MT加权Norm .727821；E3分别1.024160/.023521/2.226389。病例和模式不同，这不是新混合mini-batch或train-mode检验。Baseline VolumeDataset会读取U label，但新路径完全不实例化它，不擅改旧训练语义。

若将来试训，先把D1辅助放入监督目标内再统一用原监督LossNorm；必须维持原MT、warm-up、EMA和Optimizer，并另记录每个分量。当前β/λ仅是冻结eval标定，不能视为训练全过程的安全保证。

## 验证、失败记录与交付

{test['unittest_total']}项单元测试及verify_core通过；{v['compatibility_checks']}项A/C0/C1兼容检查、{v['direct_reconstruction_checks']}项联合/归一化直接autograd重构通过；1931项历史SHA全部不变。详细最大误差及检查矩阵见C15_VERIFICATION.md/VERIFICATION_RESULTS.json。原始参数、BN、RNG、checkpoint和所有历史结果保护通过。

保留第一次smoke的不可达NaN重构失败，修复只在新代码。独立NumPy复核还修正了softplus阈值分支的算子匹配，不调宽原始loss容差、不重跑模型。全部20真实节点、300组优化统计等价性最大相对误差{eq['max_relative_error']:.3g}。详见EXECUTION_NOTES.md。

原始数据位于runs/main/raw（NPZ本地保留，按仓库规则不纳入Git），标量/通道表在runs/main，病例级汇总和PNG/PDF在summary。无正式训练、C2、C3、通道组、交换或九层新头。建议及五问结论见C15_FINDINGS.md。
'''
    (ROOT/'C15_REPORT.md').write_text(report)
    findings='''# C15 结论与下一步建议

**工程通过；真实局部Transition的额外边界质量价值未获支持。没有训练新模型。**

**Q1：纯Transition的主要缺陷是否得到分类锚定补充？**

是，在目标层面补足。原pair中94.17%满足margin，其中56.83%仍有错误端点。Anchor引入相对z=0的绝对分类项，错误端点有明确修正梯度，且与Transition使用完全相同的端点多重集合和重复权重。这不是已经训练后提升分类的证据；Anchor也不直接约束轮廓几何距离。

**Q2：Transition与Anchor是重复、冲突还是互补？**

它改变端点的联合难度重加权，存在非共线分量。特征域通常正相关；全参数域16/16裁块为负，平均cos=−.4766，D1代表卷积约−.8296，属于增量冲突。本轮预算下联合方向仍降低Anchor，但“受控”不等于“有益互补”。不将负余弦解释为有害通道，也不将高稳定或低余弦解释为边界可靠性。

**Q3：真实局部配对是否超越置换配对？**

没有得到额外边界质量证据。置换完整保留端点/CE/熵边际，确实改变gap与梯度，但主C/D对probe真实Boundary CE的signed相关性四层均相同。True相对Anchor仅E3有不确定的小样本差异，未超Boundary参照，独立轮廓F1没有一致优势。D2绝对强度改善也由Perm复现。不同有效梯度强度、少量候选和匹配不足进一步限制因果解释。不能据此宣称两方法等价，也不能声称局部配对创新成立。

**Q4：T-Pilot推荐？**

若预算有限，优先Baseline对照下的D1 Boundary CE小规模试训，不把Anchor+Transition作为默认主方案。Anchor补足目标缺陷且实现可控，允许作为后续次级假设，但若专门试Anchor+True，至少同时保留Anchor-only；若声称局部配对价值，还须Anchor+Perm对照。只比较Boundary与联合目标无法辨别收益来自Anchor还是配对。试训需另行授权与设计，本次不执行。当前没有依据继续大规模冻结归因、C2/C3或通道交互来寻找积极结果。

**Q5：有无阻止实际训练的实现或数值问题？**

本轮冻结诊断未发现未解决的实现/数值阻塞：91项单测与verify_core、真实smoke、唯一一次8L诊断、梯度/状态/历史完整性复核通过。仍需另行验证真实训练批次、Dropout/BN/Teacher行为和动态LossNorm贡献；冻结eval权重不是训练期保证。本轮仅新L前向，MT为历史eval存档参考，没有完成新U或train-mode实验。

证据限于4个probe训练病例、每例两固定crop、每层3个固定通道；均曾参与Baseline训练及前期分析。D1高相关性和[0,0]配对区间受候选数/秩分辨率限制，不是独立泛化或等价性证明。近零方向、硬指标常数及缺失病例均保留状态。

详见[C15_REPORT.md](C15_REPORT.md)、[C15_VERIFICATION.md](C15_VERIFICATION.md)、[C15_SMOKE_REPORT.md](C15_SMOKE_REPORT.md)。完成C1.5后停止。
'''
    (ROOT/'C15_FINDINGS.md').write_text(findings)
    verification=f'''# C15 验证与历史保护：PASS

验证依据为TEST_RESULTS.json、VERIFICATION_RESULTS.json和三阶段execution_results.json，均实际执行。

{md(['测试套件','项数','结果'],[[r['suite'],r.get('tests_run','独立脚本'),r['status']] for r in test['suites']])}

共{test['unittest_total']}项单元测试，另原verify_core通过。新21项覆盖端点CE/方向、重复端点、两种平移、阈值与几何区别、空/非有限区域、零/低梯度、固定预算/通道/输出路径、skip节点、grad槽、Teacher冻结、U输入禁止、LossNorm重构、统计归约与softplus极值。旧70项覆盖Stage0/A/B/C0/C1全部原测试合同。

{md(['独立复核项目','结果'],[[k,'PASS' if ok else 'FAIL'] for k,ok in v['checks'].items()])}

- A/C0/C1兼容{v['compatibility_checks']}项，最大相对误差{v['max_compatibility_relative_error']:.3g}（预设1e−4或绝对1e−9）。
- 联合梯度/原LossNorm直接autograd重构{v['direct_reconstruction_checks']}项，在smoke和main首裁块分别检查，保留feature/parameter/LossNorm_parameter坐标；其他裁块使用已核验的线性组合，并不声称逐项都另跑了联合backward。
- 全16crop×5节点原始数组独立FP64 G×A重算，最大相对误差{v['max_feature_signed_GxA_relative_error']:.3g}；原生边界平均池化mask最大误差{v['max_mask_error']:.3g}。
- 全16crop独立Anchor/True/Perm算子重算；显式匹配PyTorch threshold20，不用放宽容差掩盖差异。Anchor与精确端点CE最大差{v['max_anchor_endpoint_CE_difference']:.3g}，极值单测允许默认softplus线性近似的2.1e−9。
- 所有现有参数逐项范数平方与全θ Gram对角一致；不可达辅助头记unreachable，归档NaN仅在已验证的图路径求和中不贡献，四头分母保持4。零/不可达不是可靠零分。
- 384/384物理干预有效；{v['final_path_changed']}次最终logits发生非零变化，{v['invalid_interventions']}次无效。alpha不剪裁，空间支持和实际L2匹配。
- Student/Teacher完整state_dict（含BN buffers）逐bit不变、Teacher无梯度、所有.grad=None、eval保持、CPU/CUDA0/NumPy/Python RNG不变、无残留hook；数据文件size/mtime不变。checkpoint strict加载且SHA不变。
- 历史SHA核对{v['history']['checked_files']}文件，变更0：原Baseline关键源文件、全部历史分析/配置/快照、六组Results、summary、checkpoint和划分列表保持。HDF5只读L image/label，不哈希U容器或读取U GT。test.list只作文件保护，不用于选择。
- 原Stage0九阶段接口、VNet默认forward、训练/评估、LossNorm/warm-up/EMA/Optimizer均未修改。Teacher生产训练语义不变，本轮只eval诊断。

首次smoke失败和第一次NumPy复核差异详见EXECUTION_NOTES.md，未抹去失败或重跑完整8L来挑结果。统计优化额外验证20节点/300组，最大相对差{eq['max_relative_error']:.3g}，前后核心SHA及旧源码快照可复核。

CPU命令：`python -B analysis/stage1c_c15_anchored_transition/test_all_c15.py`。GPU冻结三阶段使用`run_c15.py --phase smoke|calibration|main --device cuda:0 --output .../runs/<phase>`，依次执行，输出目录必须不存在。随后`check_reductions_c15.py`（smoke后）、`summarize_c15.py`、`verify_c15.py`、本报告生成器。真实复现需本地旧A/B/C0/C1原始数组、数据和checkpoint；大型NPZ按现有规则仅本地保存。

边界：未执行新U前向、真实train-mode mini-batch、独立有标签泛化验证或正式训练；MT为原2U四crop的只读梯度参考。不虚称这些测试已通过。无C2/C3、通道组/交换或新预测头。
'''
    (ROOT/'C15_VERIFICATION.md').write_text(verification)
    print('C15 final reports written')

if __name__=='__main__':run()
