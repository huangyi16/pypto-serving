# DeepSeek V4.1 Flash 在 Ascend A5 上的 MoonEP Prefill 集成设计

**设计基线为 2026-11-20 的 `pypto-serving@ffab1ab` 和该版本锁定的 `pypto-lib@3dc359fe`。** 本文针对单个 TP1、CP1、DP=EP4/8 的 Serving worker。性能目标是降低长输入请求的首 token 延迟（TTFT），并提高给定 P95 TTFT 约束下的持续输入吞吐。文中的容量和流量来自代码推导，性能目标及调度策略等待 A5 实测。相邻的开发树 `pypto-lib@c87ee612` 已更新专家核，部署复现时应记录实际 `PYPTO_LIB_ROOT` 及其提交。[Serving 模型入口](../../pypto_serving/cli/main.py#L380) · [内核目录解析](../../pypto_serving/model/deepseek_v41/npu_runner.py#L3627)

## 1. 方案与成功条件

在 V4.1 完整 prefill 设备图中接入 MoonEP 的两项机制。第一项让激活直接写入目的专家的输入行，省去接收端对 hidden 的再次搬运。第二项让热点专家在空闲 rank 上拥有临时权重副本，缩短最慢 rank 的完成时间。两项可独立启用和计时。全组空轮裁剪作为共用的基线优化单独交付，以便算出 MoonEP 自身的增益。

生产数据面以 PyPTO 的设备窗口和现有 MX 专家核为基础。复用上游 MoonEP 的计划语义、参考实现和对照用例，把推理前向需要的能力移植进同一个 L3 程序。先在固定属主上兑现直写收益，再通过预置副本实验测出动态放置的性能上限。只有上限足够、在线代价可覆盖的层和负载，才启用在线复制。[MoonEP 实现](https://github.com/MoonshotAI/MoonEP) · [算法与均衡条件](https://arxiv.org/html/2607.24653v2#S5.SS2.SSS1)

两个业务目标可能选出不同参数。低并发长输入以同一请求到达后首个有效输出 token 的 P95 延迟最小为目标。高并发以固定流量组合、给定 P95 TTFT 上限时，成功完成请求的输入 tokens/s 最大为目标。同时报告 P50/P99 TTFT、TPOT、错误率和可用 KV 页。服务方尚未指定 SLO 时，用原版在指定参考负载的 P95 TTFT 作临时上限，并给出多个负载点的延迟与吞吐曲线。此处不预设某个加速比。

## 2. 基线调用链与成本量级

| 层次 | 当前实际实现 | 设计切入点 |
| --- | --- | --- |
| Serving 调度 | 默认每步全组最多 1024 个输入 token，V4.1 支持最高 4096 的步预算。请求保留所属 DP 缓存分区。prefill 与 decode 分步执行 | 复用已知的每分区有效行数，让各卡采用一致的有效轮数 |
| Worker 与编译 | `stage_full` 组织两阶段 counts、页表、行映射及 Engram 输入。`compile_prefill_fwd` 构建完整程序，`PrefillForwardSegment.run` 向 persistent worker 提交一次 | 在模型编译、绑定和设备图内部选择后端。每层 gate、plan、dispatch、combine 在设备侧衔接 |
| 模型 | 40 层，隐藏维 5120，专家中间维 2304，384 个 routed experts，每 token 选 6 个，另有 1 个 shared expert | 接在 attention 之后的 routed-MoE。shared expert 的结果、HC 状态、KV 所属分区和 DSpark target hidden 都沿原接口消费 |
| MoE 通信 | `dispatch` 在设备上按专家计数，发送到固定属主的来源分区。`dispatch_gather_pack` 复制有效 hidden、拼接元数据并打包 MX scale | 提前生成目的专家内的最终行偏移。FP8 hidden 可直写，scale 按 A5 核的读取格式组织 |
| 计算与返回 | W13、加权 SwiGLU 与量化、W2 之后，`ffn_scatter` 把 BF16 输出写回源 rank。`combine_scattered` 将 shared 输出与六条结果按既有次序以 FP32 累加 | 解耦专家权重地址、输入行地址和返回地址，复用 scatter 与源端归并 |

源码入口见 [分区与预算](../../pypto_serving/cli/main.py#L407)、[完整 prefill 编译](../../pypto_serving/model/deepseek_v41/npu_runner.py#L1574)、[worker 绑定与执行](../../pypto_serving/model/deepseek_v41/npu_runner.py#L2351)、[设备层循环](https://github.com/hw-native-sys/pypto-lib/blob/3dc359fe57419506b42a19989ff072a7898a50ed/models/deepseek_v4_1_flash/prefill_fwd.py#L652) 和 [MoE 接线](https://github.com/hw-native-sys/pypto-lib/blob/3dc359fe57419506b42a19989ff072a7898a50ed/models/deepseek_v4_1_flash/prefill_moe.py#L408)。

**区分请求 chunk、MoE 轮次和规划范围。** 用 `R` 表示 EP rank 数，`T` 表示每来源 rank 每轮容量，`s[r]` 表示该阶段该 rank 的有效行数。本版 Serving 给 prefill 传入共享 decode 核的容量 `T = round_up(12 × 6, 16) = 80`。编译容量 1024 时每层运行 13 轮。EP8 单个分区有 1024 个 token 时，前 12 轮各有 480 条实际路由，末轮 384 条。八个分区各有 128 个 token 时，只有头两轮有数据，分别为 3840、2304 条路由。单卡没有源 token 时仍可能承接其他卡的专家计算。应按全组有效范围安排 collective。[分轮和有效行计算](https://github.com/hw-native-sys/pypto-lib/blob/3dc359fe57419506b42a19989ff072a7898a50ed/models/deepseek_v4_1_flash/prefill_moe.py#L550)

| 对象 | 单位与推导 | 对性能的含义 |
| --- | --- | --- |
| 发送的一条路由 | FP8 hidden 为 5120 B，group-32 scale 为 160 B，另有 FP32 权重与返回信息。单条 BF16 专家输出为 10240 B | 消掉 hidden 的中间读写，理论上少 10240 B **本地显存流量**。同一 token 去同一远端的重复路径每去重一份，理论上少 5280 B **链路载荷** |
| 旧接收 hidden 窗口 | 每 rank 预留 `(384/R) × R × 80 × 5120 = 150 MiB` | 这是分配容量。EP8 一轮全组至多 3840 条有效路由，全部落到同一 rank 的紧凑 hidden 为 18.75 MiB，物理窗口另计分组对齐、元数据和 bank 数 |
| 一份专家副本 | W1/W3/W2 共 `3 × 5120 × 2304` 个 FP4 参数，每 32 个值一个 scale 字节，合计 18,800,640 B，即 17.93 MiB | EP8 若每卡预留 48 个副本槽，单组槽约 860.6 MiB。槽预留量、一次实际搬运量和关键路径等待应分别计量 |

## 3. 上游复用边界

Ascend 已有 MoonEP NPU 移植。A5 支持 CANN SHMEM、MTE 与 URMA，提供 planning、dispatch、combine 和权重预取。上游 PR #40 的 prepared bindings 可以在启动时准备通信缓冲与专家地址，BF16 PUSH 提供同步和异步两种提交。所查 NPU dispatch 要求 BF16 hidden。上游公开 A5 的一组算子实验为 EP8、S=8192、H=7168、K=8、E=48。此项目的核心形状是 S≤80、H=5120、K=6、E=384，且来源长度各异。[Ascend 移植](https://raw.githubusercontent.com/Ascend/MoonEP/dev_npu/npu_patch/README.md) · [dispatch 类型约束](https://raw.githubusercontent.com/Ascend/MoonEP/dev_npu/npu_patch/native/csrc/ops/moonep_dispatch.cpp) · [上游 PR 和实验条件](https://github.com/MoonshotAI/MoonEP/pull/40)

1. **参考实现**。固定 Ascend 分支的完整 SHA，在独立 A5 算子回放中使用它的计划、专家复制表、前缀和和逆映射作对照。先在等长输入验证上游可支持的 EP4/8、K6 形状，再分别验证本设计的变长输入。两个合法 planner 可能给出不同放置，用路由守恒、专家语义和最终结果判定等价。
2. **生产集成**。把前向 planner 的分配逻辑接成图内设备任务，沿用 PyPTO 的 rank、窗口和完成依赖。规划与传输可分别替换。优先复用当前可运行的 `pld` 搬运。对目标偏移写入、同目的去重或大块副本传输，按目标 A5 拓扑实测 MTE/URMA 的交叉点。Simpler 的 extern 在 persistent executor 内以任务执行，适配时需要符合其参数、core 分工和完成语义。[PyPTO extern 执行约定](https://github.com/hw-native-sys/pypto-lib/blob/3dc359fe57419506b42a19989ff072a7898a50ed/docs/pypto-coding/cce-extern-kernel.md#L20)
3. **验证可复用的传输能力**。开发树中的 A5 `indexed_all_to_all_v` 可作紧凑行发送的带宽对照。其 L2、PER_PEER 接口把结果放到按来源划分的区域，支持 INT8/FP32 载荷。按专家目标偏移写入、MX scale 重排、BF16 回传仍需在本路径实现。只有现有原语在实测中限制收益时，才向 PyPTO/Simpler 下沉所需的原语。[A5 索引传输接口](https://github.com/hw-native-sys/pypto/blob/f887db8e/docs/zh/user/distributed/20-all_to_all.md#L137)

具体移植以 Ascend 后端的 `PlanningOp` / `MoonEPCommPlan` 为起点，保留「每条路由的目的位置、需要复制的专家、各专家累计行数」三种输出语义。把上游分配与目的索引生成的设备逻辑接入 prefill 任务，替换固定 `S×K` 容量假设和副本选择代价。以相同路由分别驱动上游参考、PyPTO 紧凑直写和在线副本路径，核对这些输出所表达的逻辑路由。训练用的梯度归约属于上游训练流程，推理侧只分配前向张量。[Ascend Buffer 与设备计划接口](https://raw.githubusercontent.com/Ascend/MoonEP/dev_npu/npu_patch/api.py)

## 4. 设备图与路由协议

### 4.1 用真实行数驱动有效轮次

`PrefillMetadataResources.stage_full` 已掌握所有分区的 counts。给新 prefill 入口增加每个阶段全组一致的 I32 有效轮数 `ceil(max_r s[r] / T)`，在已有 host 元数据准备时计算并复制到各 rank。轮内仍取 `active[r,j] = clamp(s[r] − jT, 0, T)`，以 `6 × sum(active[:,j])` 为本轮有效路由总量。空来源参与计划、接收、计算、回传和完成通知。全组空尾轮可以跳过。把供后续层读取的 padding 写入契约与 MoE collective 分开，验证两阶段 replay、HC、末行 logits 和 DSpark taps。epoch 继续按编译上界为每层预留区间，所有 rank 对相同层和轮生成相同编号。[两阶段元数据](../../pypto_serving/model/deepseek_v41/npu_runner.py#L2172) · [原有 epoch 与窗口复用](https://github.com/hw-native-sys/pypto-lib/blob/3dc359fe57419506b42a19989ff072a7898a50ed/models/deepseek_v4_1_flash/prefill_moe.py#L456)

### 4.2 固定属主的紧凑直写

首版在每个来源 rank 用 gate 的 top-6 扫描有效行，沿用现有计数和路由位置。记 `h[s,e]` 为来源 s 发往专家 e 的条数。目的属主汇总本地专家计数，为每个专家分配行起点 `base[e]`，向每个来源返回其区间起点。来源按稳定的 `(token,k)` 顺序写入，每条路由的地址为

```text
row(s,e,i) = base[e] + sum(h[u,e] for u < s) + i
0 <= i < h[s,e]
```

先保留原有专家 stride，只验证提前确定偏移能否消除 hidden 中间写读。再切换成紧凑专家段，W13 根据描述符取 `row_base`。按 MX scale 的物理 tile 粒度对齐每个非空段，第一版可用 32 行与现有 gather/pack 对齐。逻辑行与物理 padding 分别维护，GEMM、归并计数和性能计数器均使用逻辑行。若每个 rank 最多有 `L+B` 个执行槽，`L=384/R`，物理行容量的安全上界可取 `R×T×6 + (L+B)×(a−1)`，其中 `a` 是段对齐粒度。这个上界适用于固定属主和小副本池遇到极端热点时的落点。

FP8 hidden 直接落最终行。分散来源同时填充同一 scale tile 时，先各写自己的 160 字节原始 scale 行，目的 AIV 在该 tile 完整后一次性打包成 W13 所读的 MX_A 布局。整 tile 只有一个写方且目标对齐时，再试由来源直接填最终 scale。路由权重保持 FP32。同一 token 发往同一远端的去重作为单独实验，比较省下的传输与接收端展开的读写时间。每条专家路由仍保留独立权重和返回位置。先比较增加的前缀交换延迟与省掉的 `dispatch_gather_pack`，再决定在哪些规模启用直写。[发送与接收布局](https://github.com/hw-native-sys/pypto-lib/blob/3dc359fe57419506b42a19989ff072a7898a50ed/models/deepseek_v4_1_flash/prefill_moe.py#L212)

### 4.3 有界的在线动态副本

规划分两层。**放置计划**决定哪几个专家能在何处计算，同一层连续有效轮可以复用。**行计划**决定本轮每条实际路由的去处和位置，每轮随 gate 更新。起始实现按一轮规划、一个 bank 执行，得到可核对的完整路径，再扩大同层放置复用范围。业务 SLO 约束下优先利用同一次 forward 内已经到达的 token。

1. 来源生成有效直方图及本地路由列表。动态模式第一版向各 rank 发布相同的 `h[R,384]` 视图，EP8 的完整 I32 表为 12 KiB/参与者，实际网络流量取决于交换方式。各 rank 根据同一计数、同一上轮放置和同一组预先标定的整数成本作确定性决策，专家 ID、rank 和槽号作平局顺序。这样可独立算出相同的行区间，省去逐 token 跨卡广播计划。对比集中计算后仅回传区间的实现，以 gate 到首批有效发送的实耗时取舍。
2. 以 home 分配为初始可行解。对预计最慢的 rank，从重专家中选候选，优先尝试已有该层副本且有余量的目的 rank，再考虑占用新槽。枚举少量能改变矩阵 tile 数的切分点及尾部，保留预计能降低**整组完成时间**的移动。使用实际源/目标链路、热槽命中、W13/W2 每专家行数、AIV 开销估算。截断候选数量和搜索轮数，均衡负载可提前退出。每个专家的剩余行始终可由 home 接收。
3. 对选定的专家 e，把各执行 rank 配额写成覆盖 `[0,sum_s h[s,e])` 的互斥连续区间。来源 s 同样拥有按来源前缀得到的区间。二者取交集，即可得到 `(src, expert, dst, slot, count)` 的非空段和目的偏移。每个目的 rank 按槽排列这些段，固定属主是每个专家只有一个区间的特例。行数无法被 R 整除时允许各 rank 行数不同。上游每 rank 接收 `S×K` 的保证适用于等长来源和充分副本容量，项目按有效路由全覆盖交付。
4. 放置范围可试同层 1、2、4 个有效轮。最低成本方案用已观测热点决定副本是否延续，本轮精确计数负责行分配。另一候选先计算一个小窗口内各轮 gate，再做一次综合放置。后者需在 `prefill_moe_slabs` 中把 HC/gate 前置，保存各轮量化输入、权重、返回行和 HC 结果，计入额外 HBM 及首轮等待。窗口末尾按剩余有效轮收缩。离线回放可以利用未来路由估算上限，生产选择器只使用决策时已产生的数据。

用设备实测构建代价表。固定版本的 W13 主 tile 是 64 行、N/K tile 为 256。一个 64 行专家拆成 32+32 后，两卡各自都需处理一个不满的 M tile。拟合时记录每卡活跃专家数、各专家行数、`sum(ceil(n_e/64))`、链路争用和可用核数，并用多专家混合回放校验单专家曲线。复制阈值由「未来可用轮内预计节省的最慢卡时间，大于规划增量、首次装载暴露时间和分段开销」确定。[W13 工作分配](https://github.com/hw-native-sys/pypto-lib/blob/3dc359fe57419506b42a19989ff072a7898a50ed/models/deepseek_v4_1_flash/kernels/expert_routed_w13.cpp#L82) · [矩阵 tile 与 scale 寻址](https://github.com/hw-native-sys/pypto-lib/blob/3dc359fe57419506b42a19989ff072a7898a50ed/models/deepseek_v4_1_flash/kernels/expert_routed_mxfp4.hpp#L24)

对短轮次，先根据部署时标定的层、全组有效行数和已有热槽，选择是否值得交换完整直方图。阈值只使用各 rank 已经共享的信息，冷启动默认 home。进入动态模式后，用下面的微秒成本比较候选放置。两端的完成时间都包含已有工作队列和 tile 尾部，预测表按 EP4/8 和实际链路分别标定。复制代价只对缺失的目的槽计一次，规划要用的交换与可提前进行的推送也只计其未被覆盖部分。

```text
gain(move) = predicted_group_finish(home or current placement)
           − predicted_group_finish(candidate placement, including exposed copy and dispatch)
accept when gain(move) > measured planning increment + prediction error margin
```

相同分数优先复用常驻权重。另加一组静态热门副本对照，在同样的 B、同样的工作区和同一个保留集上测量。若热点随请求快速变化，在线规划才有额外机会。若固定少数副本已覆盖大部分倾斜，可以在相应层直接采用该放置，省去重复搜索。

### 4.4 专家核、权重预取与逆映射

以下为拟新增的内部设备契约。静态容量在编译时确定，实际计数和偏移为设备上的整型值。表中的维度都表示每 rank、每个正在使用的 bank。

| 数据 | 拥有方与生命周期 | 消费约定 |
| --- | --- | --- |
| 输入 hidden、MX scale、路由权重 | 发送方写入目的 bank，W13 完成读取后可复用对应输入 | hidden 为 E4M3，scale 为 E8M0，group=32。每个有效行绑定原 FP32 路由权重 |
| 专家段表，最多 `L+B` 个段 | 行计划生成，随本轮更新 | 每段给出全局 expert、逻辑行数、输入物理起点、紧凑输出起点和权重位置。home 使用原权重库的索引及 scale stride。副本使用槽内独立索引及 scale stride |
| 回传信息 | 来源写入，目的 scatter 读取 | 最小语义为 `(src_rank, local_token×6+k)`。按来源有序的段可把 src_rank 放在段描述符中。多轮并发另由 bank 与轮号区分 |
| 副本槽，最多 `B` 个 | 设备放置计划分配，同一层跨轮复用 | 身份为 `(layer_id, global_expert_id)`。W1/W3/W2 和对应 scale 可消费后发布 ready。覆盖前等到该槽最后一次权重读取完成 |

当前 W13 用 `expert×stride+row` 读激活，并用同一 expert 编号索引权重。改为分别读取输入段、输出段和权重位置。W2 使用相同的权重映射与对应输出起点。SwiGLU 保持原有截断、FP32 路由加权、MXFP8 量化和舍入位置。W2 的 BF16 结果按返回信息写回源端 `[T×6,5120]`，源端按 k 顺序 FP32 加到 shared 输出上。将线性权重移到 combine 会改变中间量化结果，数值回放须覆盖这个边界。[专家计算与 scatter](https://github.com/hw-native-sys/pypto-lib/blob/3dc359fe57419506b42a19989ff072a7898a50ed/models/deepseek_v4_1_flash/expert_routed.py#L379)

home 继续使用与 decode 共用的常驻权重。三个投影的 FP4 payload 可从常驻库按专家取出。当前 MX_B_NN scale 的物理次序将输出块排在专家/K-group 之前。第一版按输出块从 home 收集目标专家的 scale 到单专家布局，并随 packed FP4 投影发到目的槽。目的专家核按单专家 stride 读取。回放确认 scale 字节和原库逻辑坐标逐项一致。若 scale 收集成为热点，再评估启动时建立小型 scale 镜像，按实际常驻字节计入预算。[权重加载与 scale 合并](../../pypto_serving/model/deepseek_v41/weight_loader.py#L507)

## 5. 通信计算重叠与资源预算

| 顺序 | 最早可启动的工作 | 必须满足的依赖 |
| --- | --- | --- |
| gate 结束 | 直方图、shared expert、输入 scale 准备 | 本轮有效 token 与 gate 输出已就绪 |
| 放置和行区间确定 | 发送激活，同时从 home PUSH 缺失权重。已有权重的段可优先排入计算 | 各参与者使用同一计划代次。先发布数据完成，再发布对应 ready |
| 一批专家段的输入就绪 | 对权重也已就绪的 home/热槽段执行 W13、SwiGLU、W2。远端槽随后加入 | 输入含该段所有来源和已打包的 scale。首版按整轮等待确保正确，优化版把现有整组专家调用拆为少量可独立提交的段组 |
| W2 输出就绪 | 对有效行 scatter。shared 与六条回传齐后源端归并、HC post | 返回位置唯一。设备窗口复用沿用 epoch 及 peer-reader 完成协议 |

优先试单 bank。若泳道显示相邻轮的发送与计算存在可用重叠，再试双 bank，显式分开输入、回传及完成信号，副本槽按最后消费者释放。已有的 `moe_round_release` 保护的是跨 rank 窗口复用，跳过空轮或提早复用时要验证同一 persistent worker 连跑多个 chunk 的结果。MTE/URMA、AIV 核数、通信分段长度和是否按目的去重均用**通信与真实 GEMM 并跑**的结果取值。单独测得的通信带宽作为诊断数据。[当前轮间依赖](https://github.com/hw-native-sys/pypto-lib/blob/3dc359fe57419506b42a19989ff072a7898a50ed/models/deepseek_v4_1_flash/prefill_moe.py#L456)

每卡增量预算按下式计算。输入窗口若已替换旧窗，用峰值生命周期核算可回收部分。workspace 的释放以 worker 实际报告的 HBM 为准。

```text
新增峰值 ≈ 同时存活的副本槽数 × 18,800,640 B
         + 各 bank 的 hidden、scale、返回与元数据窗口
         + 计划表、scale 打包与前视 gate 工作区
         + 专家核新增临时张量、对齐与运行时开销
         − 在同一峰值时刻已释放的旧工作区
```

Serving 锁定版本的生产 `allocate` 会根据 `request_slots` 调用 `plan_decode_cache_pools` 并传入具体 plan。资源类也支持一个按常驻后余量计算的 callback。实施时在 worker 启动、权重与各缓冲布局已知时选定全组共同可用的 B，显式保留配置要求的 KV 页和 DSpark 资源。向固定 cache plan 加入容量核算，或完整接通已有 callback 及其 host 元数据尺寸、最终上报页数。按最紧张的 rank 选可容纳的 B，从小槽池起步，B=0 始终能执行固定属主模式。副本槽可跨层复用内存，切换 layer 时更新权重身份。[生产资源规划](../../pypto_serving/model/deepseek_v41/npu_runner.py#L3828) · [常驻资源分配与 cache plan 分支](../../pypto_serving/model/deepseek_v41/npu_runner.py#L2790)

跨卡窗口由 L3 统一分配，worker 在启动时为其预留容量。同层跨轮命中的范围限于一次 forward 内，同一份物理空间可供不同层顺序使用。L2 获取输入、回传和副本窗口在本 rank 的视图。home 从原常驻权重向目的窗口 PUSH，每个目的槽约定唯一的 home 写方。通知在本次数据及 scale 写入可见之后发出，目标 AIV/AIC 以对应完成依赖启动。这样可以先用现有 PyPTO 通信域跑通全图。若独立实测表明 Ascend SHMEM 后端更快，进入生产前单独验证同一 worker 内 rank 映射、地址导出/注册、跨任务可见性与写完成事件。首次握手、窗口注册和槽表建立在启动路径完成。跨域拷贝如确有需要，把它纳入 D 的计时。

## 6. 改动位置与交付顺序

| 阶段 | 主要改动位置与交付物 | 进入下一阶段的依据 |
| --- | --- | --- |
| P0 基线与回放 | 在 `pypto-lib/models/deepseek_v4_1_flash/prefill_moe.py` 的现有 standalone/golden 入口接真实 gate 路由回放，增加按层、轮、rank 的设备诊断。Serving 以 `stage_full` 提供各 rank counts | 同一输入可重复得到每层有效行、负载直方图、关键路径与完整 TTFT，能区分全组空轮和空来源 |
| P1 有效轮和固定属主直写 | 修改 `prefill_fwd.py`、`prefill_moe.py` 及 `PrefillMetadataResources`。先实现有效轮对照，再实现固定 stride 直写、紧凑段和 scale 打包 | 路由守恒、真实 golden 通过。分别量出空轮收益、直写收益、额外计数交换及窗口峰值 |
| P2 预置副本的收益上限 | 修改 `expert_routed.py`、`kernels/expert_routed_w13.cpp`、`expert_routed_w2.cpp`、`expert_routed_mxfp4.hpp`，加入段表和 home/slot 两套读取位置。先把指定专家预置在槽中 | 在相同真实路由和实际 A5 GEMM 上，确认哪些层、负载和切分方式有可重复的动态放置收益 |
| P3 在线计划与真实复制 | 在同一模型目录加入 prefill 专用的 planner 与 packed 预取实现。`prefill_moe_slabs` 管理同层槽复用。`V41ResidentResources` / `PrefillForwardSegment` 负责预算、生命周期及 ABI 绑定 | 在线全链路优于固定属主直写，能够分别解释规划、缺失装载和资源争用的代价 |
| P4 调校与服务验证 | 依第 7 节比较单/双 bank、热槽优先、窗口和传输后端。由 `cli/main.py` 的 V4.1 配置入口，经 runtime/build options 传到 `compile_prefill_fwd`。升级 Serving 的 lib gitlink | 在固定部署、真实流量和独立保留集上取得端到端正收益，并具备按配置切回基线的能力 |

对外只建议增加一个 V4.1 prefill 后端选择，例如拟议的 `--v41-prefill-ep-backend native|moonep`。首批以显式启用验证，`moonep` 内部根据已经验证的形状选择固定属主直写或有副本的执行。基准工具另行选择实验臂。由入口一次校验目标平台、拓扑与静态参数。编译产物的身份必须覆盖后端、EP、物理 slab 容量、槽容量、bank 数、量化布局及内核源码。沿用现有 JIT 缓存，热切换和冷启动分别验证，特别是独立 prefill 变体与已导入的 decode 核共享配置的情况。[编译入口与容量检查](../../pypto_serving/model/deepseek_v41/npu_runner.py#L1574) · [Serving 编译器](../../pypto_serving/model/common/compiler/compiler.py#L71)

发布时先在 EP4、EP8 分别完成可回退的小流量部署。小预算或低收益形状在同一预分配容量内选择 home 映射，所有 rank 共用该决策。运行中的 collective 若失败，沿用现有 worker 失败与请求事务处理，在恢复的 worker 上选择已知良好的后端。设备数据已写入的半轮沿现有失败处理退出。[persistent worker 的错误处理](../../pypto_serving/model/deepseek_v41/npu_runner.py#L2391)

## 7. 证明收益来自哪里

### 7.1 同路由、同输入的对照

| 实验臂 | 执行方式 | 可归因的差值 |
| --- | --- | --- |
| N | 锁定版本的完整原生 Serving | 业务端总基线 |
| N1 | N 加全组空尾轮裁剪 | `T_N − T_N1` 为分轮优化收益 |
| A | N1 加固定属主紧凑直写 | `T_N1 − T_A` 为布局和通信的净收益 |
| B | A 的相同数据面，重放选定放置与行计划，副本物理预置 | `T_A − T_B` 为该动态放置可兑现的乐观上限，已包含专家拆分后的实际矩阵乘 |
| C | 执行在线规划，副本物理预置 | `T_C − T_B` 估计规划及依赖引入的暴露时间 |
| D | 在线规划，按实际槽命中/缺失装载和消费 | `T_D − T_C` 估计装载、等待和资源竞争。`T_A − T_D` 为动态放置净收益，`T_N − T_D` 为总收益 |

B/C/D 固定硬件、工作区上限、输入和目标布局。归因时先记录 D 的放置决定、逻辑槽状态和输入，B/C 使用相同逻辑命中轨迹，仅让数据物理预置。C 实际运行 planner，核对所选放置和行区间。用 profiler 单独说明预置占用。另测不约束决策的真实在线策略。D 与各级基线还应在整服务复测，允许策略在经过验证的形状选择 A。上限很小时集中优化直写。上限可观而 D 落后时，根据 C−B、D−C 选择下一项工作。

若同一小槽池会在一段长回放中多次覆盖，在 B/C 对选中的同层规划窗口逐次预填，填充在该窗口计时前完成。报告每个窗口的实测分布和覆盖率。N、N1、A、D 以完整 chunk 和真实服务作为主对比，这样能够保留窗口边界的等待和重叠。正式记录运行时保存 A5 型号和连线、卡号、checkpoint 版本、Serving/lib/PyPTO/Simpler/PTOAS/ISA/CANN 版本、编译形状、bank 和槽池大小。对照实验共用这些条件。

### 7.2 工作负载与测量口径

- **形状矩阵**。EP4、EP8，T 初始固定 80。覆盖 1、79、80、81 行边界、单长请求集中到一张卡、各卡均匀、多个空来源、持续到达请求中的尾 chunk。对照全局预算 80、1024、4096，prompt 采用短输入、1K、跨多个 chunk 的 4K/8K 和实际业务长度分位数。固定允许的模型上下文长度、采样、并发、KV 和 DSpark 设置。
- **路由矩阵**。回放真实 checkpoint 多层、多种 prompt 的 top-6、权重和激活。取均匀、单热点、多个热点同属一个 rank、热点换层/换轮、同 token 多路由去同一目的的诊断用例。按 prompt 划分调参与保留集。分别报告第一次装载、同层热槽和跨层覆盖代价。
- **参数搜索**。先固定 T=80，扫描同层放置复用 1/2/4 个有效轮、B=0 和小副本池（例如 1、2、4、8 个槽）、单/双 bank、分段就绪策略。以设备 trace 定位瓶颈后才试不同 AIV 核数、MTE/URMA 或更大物理 slab。更大 T 需要 prefill 专用编译入口与独立 scratch，现有 Serving 会校验共享 decode 容量。逐项测量 HBM、首轮发出时刻和整层时间。
- **设备指标**。记录每层每轮真实路由数、每卡有效专家段与 64 行 tile 数、最慢 rank、gate 完成到首批发送、最后一段输入/权重就绪、W13/W2 与 scatter 完成、bank 等待、预取字节/新增副本数/承接行数、峰值 HBM。跨 rank 时钟经验证能对齐时才合并设备时间轴，另取 worker 整次提交到完成的墙钟时间。多流有重叠，用时间依赖归因。
- **端到端指标**。warmup 后重放相同请求到达序列，用流式首个实际 token 计算 TTFT，另量固定短输出的 prefill 主导路径和正常输出长度下的 TPOT。固定相同到达率比延迟，再扫描到达率，在相同 P95 SLO、无持续队列增长时比较成功输入 tokens/s。保留 P99、错误率、各长度桶和 KV 容量。原版/候选交错重复，给出配对置信区间。正式测速关闭逐路由转储和详细泳道。[Serving profiling 工作流](profile.md)

### 7.3 正确性与放量判据

逐层核对每条有效 `(src,token,k)` 恰好执行一次、执行槽对应原 expert、目的行无覆盖、BF16 结果回到原位置。先逐字节核对 packed FP4 及 MX scale，随后比较 W13、加权和量化后的 SwiGLU、W2、shared 合并与 HC 输出。沿用现有 `golden_moe_slabs` 容差。单个新路由映射的正确性要求精确相等。重复运行覆盖非满尾段、零有效来源、余数分配、全热点、B=0、槽覆写、同 worker 连续 chunk 和重用窗口。最后用真实 checkpoint 核对逐请求 logits、greedy 首 token、跨 chunk 的后续生成和 DSpark 开关两条路径。[算子 golden 与阈值](https://github.com/hw-native-sys/pypto-lib/blob/3dc359fe57419506b42a19989ff072a7898a50ed/models/deepseek_v4_1_flash/prefill_moe.py#L1377) · [V4.1 HTTP 回归用例](../../tests/test_deepseek_v41_serving.py#L65)

以下命令是**当前基线**可用的回放入口。第一条在锁定的 pypto-lib 根目录运行，第二条在 pypto-serving 根目录运行。设备编号按实际分配替换，HTTP 用例需设置真实 checkpoint 的 `PYPTO_DSV41_MODEL_DIR`、锁定的 `PYPTO_LIB_ROOT` 和四卡 `TASK_DEVICE`。算子命令使用现有合成输入与 golden，真实 gate 轨迹重放在 P0 增加。

```bash
python models/deepseek_v4_1_flash/prefill_moe.py --platform a5 --ep 4 \
  --device 0,1,2,3 --experts-per-rank 96 --entry slabs --tokens 80 \
  --slab-capacity 1024 --num-tokens-per-rank 1024,0,0,0 --enable-chip-swimlane 1
```

```bash
python -m pytest -q tests/test_deepseek_v41_serving.py
```

**发布判据**。先通过数值与端到端回归。对要启用动态副本的目标负载，要求 D 相对 A 的优势在保留集上可重复。对完整产品模式，要求相对 N 的主指标改善，配对差值的 95% 置信区间排除零，并在实际 KV 页数与并发约束下复现。P99 TTFT、TPOT 和可接受退化范围随业务 SLO 一起锁定。若只有紧凑直写取得净收益，先发布它，对已证明有收益的层或流量桶再放开动态副本。这样可以分别交付布局收益、动态放置收益和最终用户看见的首 token 收益。
