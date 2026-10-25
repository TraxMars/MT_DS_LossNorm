"""Assemble the measured C1 report and its explicitly negative decision."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from bridge import ROOT,STAGES,PRIMARY,b,write_json

def fmt(v,n=3):return '未定义' if v is None or not np.isfinite(v) else f'{v:.{n}f}'

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)

def main():
    verification=json.loads((ROOT/'VERIFICATION_RESULTS.json').read_text());tests=json.loads((ROOT/'TEST_RESULTS.json').read_text())
    assert verification['status']=='PASS' and tests['status']=='PASS'
    post=json.loads((ROOT/'POST_AUDIT_VERIFICATION.json').read_text());assert post['status']=='PASS'
    summaries=json.loads((ROOT/'attribution_summary.json').read_text())
    def stat(stage,obj,metric,mode='discovery_B_transfer',dose=.03):
        return next(r for r in summaries if r['stage']==stage and r['objective']==obj and r['support']=='B' and r['dose']==dose and r['mode']==mode and r['metric']==metric)
    def ci(stage,obj,metric='signed_spearman'):
        r=stat(stage,obj,metric)
        return '未定义' if r['mean'] is None else f"{r['mean']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
    primary=table(['阶段','Region','Boundary','Transition','Permuted','Far'],[[s]+[ci(s,o) for o in PRIMARY] for s in STAGES])
    quality=[]
    for s in STAGES:
        a=stat(s,'transition','absolute_spearman');d=stat(s,'transition','direction_accuracy');c=stat(s,'transition','both_direction_decidable_fraction');f=stat(s,'transition','boundary_f1_delta_rho_degradation')
        quality.append([s,fmt(a['mean']),fmt(d['mean']),fmt(c['mean']),fmt(f['mean']) if f['mean'] is not None else f"仅{f['available_cases']}/4例可用，均值{fmt(f['descriptive_available_case_mean'])}"])
    quality=table(['阶段','T的abs相关','T方向准确率','两者方向可判定率','T与轮廓F1下降的相关'],quality)
    g=pd.read_csv(ROOT/'gradient_relations_by_case.csv');g['pair']=['~'.join(sorted((a,c))) for a,c in zip(g.left,g.right)]
    def gc(kind,s,a,c):
        x=g[(g.coordinate_kind==kind)&(g.stage==s)&(g['pair']=='~'.join(sorted((a,c))))].median_channel_cosine_mean_crops
        return float(x.mean())
    gradients=table(['阶段','特征 T/Region','特征 T/Boundary','特征 T/Perm','参数 T/Boundary','参数 T/Perm','参数 Perm/Perm1'],
        [[s]+[fmt(gc(k,s,a,c)) for k,a,c in [('feature_same_crop','transition','region'),('feature_same_crop','transition','boundary'),('feature_same_crop','transition','permuted'),
            ('parameter_shared_coordinates','transition','boundary'),('parameter_shared_coordinates','transition','permuted'),('parameter_shared_coordinates','permuted','permuted_1')]] for s in STAGES])
    stability=pd.read_csv(ROOT/'feature_contribution_stability.csv');stability=stability[stability.support=='B']
    stable=table(['阶段','T同病例两crop','T病例间','Boundary同病例两crop','Boundary病例间'],
        [[s]+[fmt(stability[(stability.stage==s)&(stability.objective==o)&(stability.comparison==c)].spearman.mean()) for o,c in [('transition','within_case_crops'),('transition','between_cases'),('boundary','within_case_crops'),('boundary','between_cases')]] for s in STAGES])
    geom=[];groups=[]
    for part in ('discovery','probe'):
        for record in json.loads((ROOT/'runs'/part/'geometry.json').read_text()):
            geom.append(record)
            for group in record['pairs']['true']['groups']:
                if group['count']:
                    row=dict(case=record['case'],fg=group['fg_correct'],bg=group['bg_correct'],count=group['count'],fraction=group['fraction'],
                        plus=group['fg_logit']['mean'],minus=group['bg_logit']['mean'],abs_plus=group['absolute_fg_logit']['mean'],abs_minus=group['absolute_bg_logit']['mean'],gap=group['gap']['mean'],
                        transition=group['transition'],endpoint_ce=group['endpoint_boundary_ce'])
                    distance=group.get('gt_inner_to_predicted_contour_distance');row['contour_distance']=distance['mean'] if distance else np.nan;groups.append(row)
    bycase=[]
    for case in sorted({g['case'] for g in geom}):
        records=[r for r in geom if r['case']==case]
        bycase.append(dict(case=case,satisfied=np.mean([r['pairs']['true']['margins'][1]['satisfied_fraction'] for r in records]),
            wrong=np.mean([r['pairs']['true']['margins'][1]['wrong_fraction_given_satisfied'] for r in records]),
            wrong_all=np.mean([r['pairs']['true']['margins'][1]['satisfied_and_wrong_fraction'] for r in records]),retained=np.mean([r['far_matching']['retained_fraction'] for r in records])))
    bycase=pd.DataFrame(bycase);classification=[];gd=pd.DataFrame(groups)
    for fg,bg in ((True,True),(True,False),(False,True),(False,False)):
        fraction=np.mean([next(x['fraction'] for x in r['pairs']['true']['groups'] if x['fg_correct']==fg and x['bg_correct']==bg) for r in geom])
        block=gd[(gd.fg==fg)&(gd.bg==bg)];conditional=[]
        for _,v in block.groupby('case'):
            conditional.append({k:np.average(v[k],weights=v['count']) for k in ('plus','minus','abs_plus','abs_minus','gap','transition','endpoint_ce','contour_distance')})
        m=pd.DataFrame(conditional).mean()
        classification.append([('正确' if fg else '错误')+'/'+('正确' if bg else '错误'),f'{fraction:.2%}',len(conditional)]+[fmt(m[k]) for k in ('plus','minus','abs_plus','abs_minus','gap','transition','endpoint_ce','contour_distance')])
    classification=table(['FG/BG分类','病例平均占比','条件统计可用病例','z+','z−','abs z+','abs z−','相对gap','Transition','端点平衡CE','GT内边界到预测轮廓距离'],classification)
    geometry=pd.read_csv(ROOT/'pair_geometry_summary.csv')
    geotable=table(['对照','每crop平均对数','端点距离均值','FG端复用比例','BG端复用比例'],
        [[name]+[fmt(geometry[geometry.control==name][col].mean()) for col in ('pairs','distance_mean','fg_reuse','bg_reuse')] for name in ('true','permuted','far','true_common','far_matched')])
    difficulty=pd.read_csv(ROOT/'endpoint_difficulty_matching.csv')
    diff=table(['对照/端','logit W1','CE W1','熵 W1'],[[c+'/'+side]+[fmt(difficulty[(difficulty.control==c)&(difficulty.side==side)][k].mean(),5) for k in ('logit_wasserstein','ce_wasserstein','entropy_wasserstein')] for c in ('permuted','far','far_matched') for side in ('plus','minus')])
    actual=pd.concat([pd.read_csv(ROOT/'runs'/part/'interventions.csv').assign(split=part) for part in ('discovery','probe')])
    probe_actual=actual[(actual.split=='probe')&(actual.dose==.03)]
    effects=table(['阶段','B平均abs Boundary Δ','I平均abs Boundary Δ','B最终预测改变比例','B非零轮廓F1变化比例'],
        [[s]+[f'{v:.7g}' for v in [probe_actual[(probe_actual.stage==s)&(probe_actual.support==r)].boundary_delta.abs().mean() for r in ('B','I')]]+
            [f'{probe_actual[(probe_actual.stage==s)&(probe_actual.support=="B")].changed_prediction_fraction.mean():.7g}',f'{probe_actual[(probe_actual.stage==s)&(probe_actual.support=="B")].boundary_f1_delta.ne(0).mean():.2%}'] for s in STAGES])
    sensitivity=table(['阶段','剂量.01 rho','剂量.03 rho','剂量.10 rho','margin.5 rho','margin1 rho','margin2 rho'],
        [[s]+[fmt(stat(s,'transition','signed_spearman',dose=d)['mean']) for d in (.01,.03,.10)]+[fmt(stat(s,o,'signed_spearman')['mean']) for o in ('transition_m0_5','transition','transition_m2')] for s in STAGES])
    executions={p:json.loads((ROOT/'runs'/p/'execution_results.json').read_text()) for p in ('smoke','discovery','probe')}
    rawloss=pd.concat([pd.read_csv(ROOT/'runs'/p/'losses.csv') for p in ('discovery','probe')]).groupby('objective').raw_loss.agg(['mean','min','max'])
    losses=table(['目标','原始loss均值','最小','最大'],[[o]+[f'{rawloss.loc[o,k]:.7g}' for k in ('mean','min','max')] for o in rawloss.index])
    report=f'''# C1_REPORT：有标签边界专属性与跨目标归因

实施与保护检查通过；当前Transition没有提供比Region/Boundary CE更可靠的真实边界贡献解释，不建议直接进入新边界模型训练，也不自动执行C2。

## 执行与有效性

起点C0 `ae073c5`，独立分支 `experiment/stage1c-c1-boundary-specificity`。LA10%原iteration15000 Student/Teacher配对checkpoint，SHA256 `85489c52c0905aebde65eacf5cea215aac51ee825ed3e5a845da86fe7f23e168`。保持eval，Student可求导，Teacher冻结且不前向。只读取L病例。所有源梯度原始、无额外LossNorm；本次Region是纯CE，旧Stage1A的四尺度CE+Dice只作兼容性复算。

原2L/4crop smoke完成360次单通道干预；随后4 discovery L + 4 probe L，每例两个原裁块，完成7392次正式干预，物理有效7392/7392。五阶段共同有效discovery通道304个，从中冻结18/17/18/14/10个候选，共77个。候选为五方法有符号极值、固定随机通道与C0锚点的并集；所有方法使用同集合，不基于probe重新排序。结果仅适用于此候选集合，不代表所有通道。

三次GPU耗时分别{executions['smoke']['elapsed_seconds']:.2f}/{executions['discovery']['elapsed_seconds']:.2f}/{executions['probe']['elapsed_seconds']:.2f}秒，正式峰值显存{max(executions[p]['peak_allocated_mib'] for p in ('discovery','probe')):.2f} MiB。所有有效干预都改变了最终logits，未必改变argmax。α范围{actual.alpha.min():.6f}–{actual.alpha.max():.6f}，未触及0.5上限。实际L2最大相对误差{max(p['energy_max_relative_error'] for p in verification['phases']):.3g}。

Region/Boundary/Transition/Permuted的完整特征梯度有效覆盖100%；未匹配Far为81.25%，3个probe裁块过易，低于既定阈值。它们的本地归因不提供可靠病例估计，标未定义，不能当可靠零分。D1的Far在B支持上和Boundary/Transition在I支持上的零值是空间支持导致的结构零。详见feature_validity_by_case.csv、supported_feature_gradient_relations.csv。

所有病例参与过Baseline训练，probe已有Stage1B探索历史；不是独立泛化验证。病例为统计单位，crop为重复测量。主参数事前固定margin1、剂量0.03；0.5/2与0.01/0.10只作敏感性。未解析test.list调参。

## 配对、匹配与普通分类难度

{geotable}

每个原始真实/置换/未匹配far目标每crop2048对，类别方向均FG→BG，裁块边缘排除4体素。置换保留完全相同端点及复用情况，CE/熵/logit边际分布精确相同；相对gap的均值也被保留，但联合gap分布和距离改变，不能说所有因素匹配。

Far主对照的内部位置通常容易得多，梯度接近零不能解释为边界专属性。额外同类z宽0.5分箱匹配、不放回，保留两端均可匹配的true-common子集。平均保留{bycase.retained.mean():.2%}，病例平均范围{bycase.retained.min():.2%}–{bycase.retained.max():.2%}；即平均{1-bycase.retained.mean():.2%}未匹配，不能推广到全部边界端点。重复端点比例、距离和条件样本选择仍不完全匹配。

下面为每crop分布Wasserstein距离的病例平均（熵为自然对数）：

{diff}

匹配后的true-common与far原始Transition均值分别{rawloss.loc['true_common','mean']:.6f}/{rawloss.loc['far_matched','mean']:.6f}。相似端点logit可使远处目标损失非常接近真实局部目标；梯度位置不同不能单独证明几何专属性。配对位置、各边际、margin分布、复用和保留率均留在原始归档/geometry.json。

## 端点分类与margin

真实局部对margin1满足率{bycase.satisfied.mean():.2%}；全部位置对中{bycase.wrong_all.mean():.2%}同时满足margin却至少一端分类错误；在已满足margin的对中，该比例为{bycase.wrong.mean():.2%}。这是病例平均，不能把32768个位置对视为32768个独立样本。正margin不可能在FG/BG两端都反向分类时满足；实际问题主要是只有一端分类正确。

下表fraction先平均两个crop再平均病例；条件均值在每病例内按组内对数合并crop，再平均有该组的病例，空组不填零。距离单位体素，是裁块内补充统计，不替代整病例表面评价。

{classification}

Transition对z的共同平移不变，能够约束相对差，不能单独保证z+>0且z−≤0。softplus不是硬铰链；margin满足后仍有梯度，但大gap可使错误端点被很弱地加权。margin0.5/2均不能解除这个结构限制。Boundary CE与真实轮廓指标必须独立检查。

## 梯度关系与稳定性

下表先取每crop有效通道余弦的中位数，再平均crop/8病例。特征与参数列属于不同坐标，不能互当同一梯度；不同crop空间坐标不直接比较。

{gradients}

真实/置换端点边际完全相同，梯度仍有明显差异；尤其参数D1中T/Perm约−0.884，而额外置换间约0.991，说明配对联合关系确实被使用。它说明目标不同，不说明目标更好。D1全特征余弦不能用于通道排名：二分类1×1头有g_c=(w_fg,c−w_bg,c)·dL/dz，余弦中的通道系数抵消；该性质已自动测试。

空间归约后的单位L2有符号贡献向量Spearman稳定性（所有阶段通道，而非仅77候选）：

{stable}

D1/D2/D3的Transition响应可以重复，E4较弱。但可重复的目标响应可能持续与真实Boundary变化方向不一致。卷积核共享坐标的逐通道方向稳定性另存parameter_direction_stability.csv；不是跨病例特征图梯度余弦。

## 主要终点：discovery固定评分解释probe真实Boundary CE变化

每例先合并两crop，使用冻结源目标单位能量响应乘probe实际预算。probe GT梯度不进入这些评分。以下是4病例平均signed Spearman及2000次相同病例重采样的探索性95%区间；未定义不填0，不作显著性或泛化声明。

{primary}

Transition在E3/E4/D3/D2的均值为负；D1虽为正，不能仅凭该正相关宣称有用。

{quality}

D1主剂量80次原始probe干预中，Transition预测和实际自身变化全部为正，而真实Boundary CE变化全部为负（−0.002982到−0.000080）。因此本地及固定迁移方向准确率均0%，影响强度abs相关反而为负。其signed rho约0.618只是正值与负值之间的排序关系，不能解释为正确预测改善/恶化。评分保持原符号，没有取反挽救结果。

本地GT归因仅作参照：Boundary对自身rho接近1符合Taylor预期；Region对真实Boundary变化约E3 0.926/E4 0.765/D3 0.978/D2 0.993/D1 0.982，Transition约−0.165/0.038/−0.283/−0.397/0.609。不能要求Transition超过Boundary自身的局部优势；但这里同时缺乏相对Region、置换和独立轮廓指标的稳定额外价值。

跨目标损失单位不同，只解释排序/符号/强度相关性，不能把Transition的Delta当作Boundary CE数值预测。abs Delta与sum|gFM|分开保留。方向准确率仅在源预测及真实变化各自可判定时计算，必须连同覆盖率读取。剔除真实近零变化后的主rho结论不变；E4 Transition约−0.084。

## 支持控制、最终预测与敏感性

B/I分别精确匹配前景、背景体素数，实际使用同一L2预算；分类CE/熵/logit分布未完全匹配，原始分布已保存。粗层采用体积平均，支持重叠和ESS见geometry.json。

{effects}

D1 I对Boundary CE变化严格为零，是1×1点式路径和不重叠支持的结构结果。上游感受野可跨区域传播，不能由B影响较大推断通道边界专属。主剂量硬预测改变比例约10^−6到10^−5，轮廓F1离散且很多干预无变化。E4 Transition轮廓相关只有2/4病例有变异，另外2例恒零，不能报告四病例区间；其正的可用子集均值不构成稳定证据。E3/D3/D2/D1的Transition轮廓相关区间均跨0。

冻结Transition评分对Boundary变化的敏感性：

{sensitivity}

剂量缩小时相同目标Taylor近似的数值检查是实施基础，保存在local_self_fidelity_secondary.csv，不作为本阶段主要成果。三剂量没有挽救跨目标的方向问题；三个margin没有改变主判断，不事后选最佳参数。

## 原始损失、资料与停止

{losses}

损失原始单位不同，数值大小不代表目标质量。完整每通道范数、signed G×A、abs G×A、梯度关系、无效状态、实际消融和最终预测指标见runs/；主/次终点、配对病例差及区间见attribution_by_case.csv、attribution_summary.json、paired_case_comparisons.json。固定计划及敏感性评分在probe前哈希冻结。

![交叉目标归因](figures/cross_target_attribution.png)

![margin与分类正确性](figures/margin_and_classification.png)

实施校验与{tests['unittest_total']}项自动测试通过，1198历史文件SHA未变，0训练步、0checkpoint写入。当前不支持直接训练新边界目标；应先重新设计并验证绝对分类/等值面位置约束与相对过渡的关系，再决定是否进入C2。本任务停于C1，未实现新训练损失、SDF头、通道组或交换。
'''
    (ROOT/'C1_REPORT.md').write_text(report)
    findings='''# C1_FINDINGS

结论：C1实施与保护校验通过，但没有足够证据把当前Transition接入训练或直接推进C2。

**Q1：真实局部配对是否提供额外、可重复的信息？**

它确实提供不同于可分离CE的配对联合约束：真实/置换的端点多重集合、类别和logit/熵边际完全相同，参数梯度仍明显不同；不同随机置换彼此更接近。Transition通道响应在D1/D2/D3也有重复性。但这些证据支持“使用了局部配对关系”，没有支持“提供额外、可重复的真实边界质量收益”。匹配far控制损失与true-common几乎相同，且只能匹配约49.6%端点，不能忽略剩余难度和几何差异。

**Q2：梯度是否具有边界专属性？**

尚未建立。margin1满足率94.17%，满足者中56.83%仍至少一端分类错误。Transition约束相对logit，不锚定两侧绝对类别或预测等值面位置。与CE梯度不同、边界支持消融较强和低维/稳定响应，都不足以证明边界专属性。D1的远区结构零与通道余弦同值也不能作为边界通道证据。

**Q3：能否解释probe真实边界变化，而不仅是自身Taylor变化？**

当前不如Region/Boundary CE。固定discovery评分的signed rho在E3/E4/D3/D2为−0.259/−0.083/−0.419/−0.408。D1为+0.618，但abs rho为−0.618，方向准确率0%：削弱使Transition自身上升、真实Boundary CE下降。普通Region/Boundary及置换评分在多数阶段更一致。独立轮廓F1没有稳定增益；E4表面较好的子集结果仅2/4病例可用，不能视作四病例证据。没有翻符号或换主剂量挽救结果。

**Q4：继续C2还是重设计？**

建议先重设计有标签目标及控制，再决定C2。需要分别验证绝对端点分类正确性、0.5等值面/局部轮廓定位和相对过渡提供什么信息，继续保留相同端点置换、共同支持/能量与匹配不足记录；不建议直接使用当前纯Transition做新模型训练。也不能从这次负结果推出所有过渡/边界监督无效，或推荐随机配对作为训练损失。

8L/16crop、7392次正式单通道干预，全部冻结；probe四例参与过Baseline训练且有Stage1B探索历史，所有区间仅探索性。历史1198文件未变，未读取无标签GT，未训练，未执行C2/C3、分组或交换。详细数据及限制见C1_REPORT.md和C1_VERIFICATION.md。
'''
    (ROOT/'C1_FINDINGS.md').write_text(findings)
    verification_text=f'''# C1_VERIFICATION

结论：PASS（实施、复算和保护），不表示Transition研究假设成立。

## 实际执行

最终{tests['unittest_total']}项单元测试通过：C1新增18、C0 21、Stage0 7、Stage1A 10、Stage1B 14；verify_core.py通过。测试通过导入旧测试类执行，不调用旧报告写入入口。初次smoke前68项通过，之后新增D1余弦结构与低梯度汇总两项，最终70项全回归通过。

真实原定2L smoke：4crop/360次操作；完整4 discovery+4 probe：16crop/7392次操作。五层对象身份、原始模块、D3/D2 skip求和、形状/梯度形状、同节点削弱、一次hook、最终预测路径通过。全部正式操作都改变最终logits；argmax可能不变。实际能量最大相对误差{max(x['energy_max_relative_error'] for x in verification['phases']):.8g}，未匹配0次，未裁剪alpha。

Student/Teacher strict checkpoint加载，iter15000与原SHA通过。参数、全部state_dict/BN buffers精确不变；两模型eval；Teacher requires_grad=False且无前向；所有.grad均None；CPU/CUDA/NumPy/Python RNG不变；无遗留hook。只打开8个L病例image/label，数据大小/mtime不变，不打开U容器。test.list仅历史SHA保护，不解析或用于选择。

## 独立与兼容性检查

- 四尺度A原CE+Dice及B原边界CE公式复算；正式160条最终头参数梯度比较，另20条C0特征梯度比较；最大相对差异0。smoke另有60条比较。C1 Region纯CE与原A定义区分，不偷换Baseline训练目标。
- NumPy从原始特征/梯度/M独立复算{verification['numpy_replayed_primary_predictions']}个主目标一阶预测。最大inner误差{verification['max_numpy_inner_product_error']:.3g}，最大Delta_hat误差{verification['max_numpy_prediction_error']:.3g}。
- 从原始最终logits、GT与位置对复算Region/Boundary/各Transition损失，最大绝对误差{verification['max_numpy_raw_loss_error']:.3g}（阈值1e−10）。原生类别方向、相邻距离、置换多重集合、B/I各类别计数与不重叠、各层平均权重映射通过。
- discovery全部冻结评分可由其原始贡献表重建；主计划及margin0.5/2补充计划在probe前冻结，SHA/时间顺序通过；核心运行源码在discovery/probe间SHA相同。
- 源预测阈值按源目标单位传播，真实Boundary变化用Boundary阈值；整体真实近零比例8.455%，B为1.272%，I为15.639%，不强制正负。相同目标Taylor仅辅助检查；local GT归因与固定迁移分表。
- 3个probe裁块的Far源梯度低于固定阈值；相似度无效，本地病例归因要求两crop都可靠，否则未定义，不用低梯度冒充可靠零。完整特征Far覆盖81.25%，其余四主目标100%；D1支持结构零单独解释。
- 汇总层有效性修订后{post['unchanged_transfer_records']}条固定迁移记录与修订前一致；原始运行数据及主结论不变。配对重采样单位是4病例，不是通道/体素/位置对。参数梯度与特征梯度文件/坐标独立。

## 历史保护与限制

1198项历史SHA复核通过，改动0。保护包括Baseline六组Results、checkpoint、summary、配置/快照、原生产代码、数据划分及Stage0/1A/1B/C0（C0的486项完整清单预核验通过）。HDF5不做容器全字节哈希，避免读取U标签字节；L访问记录和只读文件状态另核验。

0 optimizer步骤，0 checkpoint写入；未修改train.py、objectives.py、VNet默认行为或LossNorm。未运行C2/C3、正式训练、新辅助头/SDF损失、多通道组或交换。没有独立泛化验证。详细证据：TEST_RESULTS.json、VERIFICATION_RESULTS.json、POST_AUDIT_VERIFICATION.json、PROTECTION_BEFORE.json及各run的execution_results.json/source_at_run.json。

smoke后的归档补充、阈值量纲澄清和低信号汇总修订见EXECUTION_NOTES.md，均保留实际执行来源和原始结果，不将未执行检查列为通过。
'''
    (ROOT/'C1_VERIFICATION.md').write_text(verification_text)
    write_json(ROOT/'FINAL_STATUS.json',dict(status='COMPLETE',implementation_verification='PASS',substage='C1',base_commit='ae073c55f9b362714a444bd83c6acbb886cba2d3',
        cases=8,crops=16,formal_diagnostic_interventions=7392,smoke_interventions=360,unit_tests=tests['unittest_total'],historical_files_unchanged=1198,
        original_baseline_compatible=True,transition_boundary_quality_benefit_supported=False,recommendation='redesign labeled boundary objective before C2',
        independent_generalization=False,optimizer_steps=0,checkpoint_writes=0,C2_executed=False,C3_executed=False,channel_group_interactions=False))
    print('Wrote C1_REPORT.md, C1_FINDINGS.md, C1_VERIFICATION.md and FINAL_STATUS.json')

if __name__=='__main__':main()
