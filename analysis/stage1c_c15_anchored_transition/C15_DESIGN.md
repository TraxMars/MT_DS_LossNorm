# C1.5：分类锚定与局部过渡互补性（执行前固定）

本地起点 `d7b50ccb5b458a9d5a72e65fa0b21c0cd6cbcf56`，新分支 `experiment/stage1c-c15-anchored-transition`。远程main锁定 `493fa83eca062aa13eee9c4c2dafca56cd27fea8`：438个远程文件与本地Git blob一致，本地另外41个图件/完整统计文件。历史1931项SHA预核验通过。仅新增本目录与C15测试，不更改Baseline、Stage0/A/B/C0/C1、checkpoint、Results或数据。

用户已澄清仅在Boundary仓库执行后文C1.5；不执行误附的D0/D1-A，不创建其目录，不进行train-mode或新的U前向。只进行冻结eval诊断，Student保留autograd，Teacher冻结且不前向；无优化器、训练、C2/C3、通道组或新预测头。

## 模型与数据

LA10% iteration15000 原配对checkpoint，SHA256 `85489c52c0905aebde65eacf5cea215aac51ee825ed3e5a845da86fe7f23e168`。原seed142、112×112×80、TF32关、CuDNN deterministic、FP32网络，诊断目标FP64。C1原4 discovery L/4 probe L、每例两固定crop；先C0两L四crop smoke。probe已参与Baseline训练及前期分析，不能称独立盲验证。test.list仅完整性哈希，不用于选择。

执行顺序：CPU测试 → 2L smoke → 4 discovery L梯度预算标定 → 冻结λ、模拟β与通道 → 单次完整8L诊断。smoke仅工程检查，λ=1、β=.01用于联合梯度恒等式检查，不是方法候选。完整诊断不依据probe结果再调整定义、权重或排名；允许正常修正程序错误，但必须记录失败与修正，不抹去原始产物。

## 目标和配对

原生6邻接FG正端/BG负端，复用C0相同种子/边缘4/最多2048、至少32对，并逐点核对C1归档。主margin1。Anchor=.5 mean[softplus(-z+) + softplus(z−)]；True=mean softplus(1-z++z−)。重复端点按原始pair出现次数计权，不去重后重加权。Perm复用C1主置换，端点多重集合完全相同，Anchor完全共用同一个标量。

候选：Boundary宽2，Anchor，Anchor+λTrue，Anchor+λPerm。Region纯CE和纯True只作参照，不作为训练候选。补充远距离匹配只复用C1既有pair/匹配覆盖和原损失，不新增对照或重新挑匹配。原始目标分别保存，不能用C/D总loss大小直接论优劣。

两种平移必须区分：同时对每点fg/bg两个类别logits加同值（softmax gauge）不改变z，Anchor/True均不变；对两端z加同值改变绝对p_fg，True不变而Anchor通常改变。Anchor锚定分类阈值，不直接度量轮廓距离。即使z+=(1000)、z−=(-50)使分类CE近零，线性插值零交点仍为1000/1050而非GT体素面中点0.5。

## 梯度预算，先于probe冻结

先计算Boundary/Anchor/True/Perm/Region及原四尺度监督的独立梯度。标定坐标θ为现有模型全部可训练参数；D1目标不可达的辅助头参数记None，范数中为0但保留不可达标签。使用同一θ坐标，不把不同层特征的空间大小混加为参数梯度。

预定增量预算ρ∈{0.10,0.30}，主ρ=.30。令R=max_discovery-crops(||G_true||/||G_anchor||)，固定λρ=ρ/R。这样discovery的True增量范数不超过预算；probe的实际比率另外记录，不剪裁。C/D用相同λ，不为Perm单独重定标，记录其实际增量比例。若参考范数无效则标定失败，不以probe补选或改预算。

模拟未来训练外权重β：在两个已冻结λ以及Boundary、Anchor共六个候选中，取discovery最大 ||G_aux||/||G_sup||；共同β=.10/该最大值。这里只控制辅助梯度预算，β不通过loss值或probe指标优化，所有候选同β。它仅用于LossNorm模拟，不将其接入训练。

## 梯度关系和优化冲突

在E3/E4/D3/D2/D1真实节点上记录特征梯度；分别记录原阶段卷积参数梯度以及全部θ的Gram/各参数范数。E3/E4输出，D3/D2为skip融合后block_seven/block_eight输入，D1为out_conv输入。仅最终D1目标用于特征归因；四头监督只作参数级训练模拟，避免D3/D2辅助头与下游干预不一致。

记录A/T、Boundary/T、A/Perm的有符号余弦、dot、范数比、逐通道正负/零内积、真实梯度线性组合、抵消率，以及去除Anchor方向后的增量范数比例。阈值沿用绝对1e-12、相对1e-6；零、不可达、低阈值、非有限分别标记。

对gJ=gA+λgT，Anchor沿未实际执行的梯度下降方向的一阶变化率为 −(||gA||²+λ dot(gA,gT))；额外项为−λ dot。负dot表示对Anchor的增量冲突，但只有总括号<0才表示此坐标下联合方向可能升高Anchor。Boundary同样检验。它是冻结局部推断，不代表真实SGD轨迹或有害通道。

跨crop/病例不直接比较空间坐标。参数共享坐标可比较；特征采用空间归约后的有符号贡献向量比较。病例为统计单位，crop为重复；4 probe病例用相同2000组配对重采样，区间仅探索性，不报告显著性或独立泛化。

## 端点与分界线

按基线FG/BG正确性的四组合分别记录Anchor、True、联合目标、两端logit导数、满足margin但错误比例。分别保留单pair导数和按重复端点汇总后的真实logit梯度。独立输出空间示意步长为0.05的max-abs归一化logit下降方向，只用于观察分类方向及线性插值零交点；不更新参数、不冒充网络训练。两端同号无边内零交点，必须标无交点，不能填0距离。

## 有限单通道干预

仅E3/D3/D2/D1；每层从C1冻结候选中取C0锚点0/C//2/C−1（若缺失则按通道号补齐），只取共同有效者，先于新probe固定。每层3个、共12通道。E4仅梯度统计。B支持与C1完全相同，平均池化至真实特征网格，预算基准q原样读取C1对应crop的固定q。剂量.03主、.01敏感性；同stage/crop/剂量全部目标和通道同目标L2，α=δ/||FM||≤.5，失败不剪裁或填零。

完整16crop共384次单通道操作（另smoke96次），不做数千次筛选或多通道组。Δhat=−αΣgFM保持原符号；两联合目标从相同基础梯度线性组合。记录实际Boundary CE、端点分类错误、轮廓F1、全CE、Dice、预测改变率及实际L2。Boundary既是候选又是参照端点，绝不宣称它是完全独立评价；独立硬轮廓、端点错误、Dice等必须一起解释。

每crop在非零干预前，以四干预stage×batch4零剂量标定C0 ε=max(1e−9,1e−6 max(|L0|,1e−3),5 max|null−base|)。真实近零方向不可判定。Local GT梯度仅作本地检查；主要跨病例评分是discovery每病例平均后的单位L2有符号响应，probe不能重排通道。相同目标Taylor只作数值兼容检查。

完整诊断前细化有效性统计：主Boundary CE signed/abs相关性仅在预测与实际变化均超过各自ε时计算；不足3通道标记不可计算。另列所有有效物理干预的未过滤raw相关性作为描述，不冒充可靠方向。不同目标单位不同：固定预测ε用discovery目标标量数值下界，实际Boundary变化用本crop零干预标定ε。病例CI必须四例均有效，不能删除缺失病例后宣称完整4例证据。轮廓/Dice等离散指标的常数响应单独标记。

## LossNorm兼容性与L-only限制

原Lsup为四尺度(CE+Dice)/2平均，原Lmt为四尺度概率MSE平均；normalize=L/stopgrad(max(L,1e−8))。先记录原始梯度。模拟N(Lsup+βLaux)的贡献 (gSup+βgAux)/max(Lsup+βLaux,eps)，与N(Lsup)+βN(Laux)分开，后者仅风险参照。模拟联合标量先转原FP32再执行原normalize_loss，归约差异和直接autograd重构分别记录。

本轮不打开任何U容器。MT梯度只能只读复用Stage1A中C0原2U×2crop存档，重构原mean四头与0.3权重LossNorm，比较五个共享参数坐标上的贡献；不是新2L+2U train-mode或真实未来混合批次验证。原VolumeDataset确实读取所有样本label，训练loss只截取L标签；新路径不实例化该Dataset。此问题只报告，不擅改Baseline语义。

## 工程门槛与决策

原checkpoint strict加载、五节点身份/skip/grad形状，重复权重和置换边际完全保留，联合/归一化梯度重构，70项旧回归加新测试，全部state/BN/.grad/RNG/历史1931项SHA保护。工程通过不等于边界创新成立。若Anchor补足绝对分类且尺度可控、无未解释不稳定，可讨论有限T-Pilot；真实/置换不可区分必须认定局部配对独特价值未获支持，不调整权重找积极结论。交付五份C15报告后停止。
