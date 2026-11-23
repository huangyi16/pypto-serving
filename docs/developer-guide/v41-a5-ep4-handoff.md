# DeepSeek V4.1 Flash · Ascend A5 EP4 上机交接

本文记录已验证的代码版本、上机操作和实验结果。历史上机操作使用 UTC+8，最新实验时间以优化日志中的 UTC 为准。接手时以当前 Git、作业和设备状态为准。

## 暂停时的作业与实验包

截至 2026-11-23 00:56 UTC，第 79 轮候选已完成，第 62 轮同场对照正在运行，原生随后运行。第 84 轮 16K 扫描仍在等待同一四卡锁。目标已暂停，服务器上已经提交的作业继续按原队列运行。[第 79 轮记录](v41-a5-ep4-optimization-log.md#prefill-79--return-eight-compact-rows-per-store-with-32-scatter-blocks) [第 84 轮长度扫描](v41-a5-ep4-optimization-log.md#sweep-84--compare-direct-expert-operand-stores-through-16384-tokens)

第 79 轮和第 84 轮已结束。第 66 轮候选组通过，同场对照在预热时出现设备同步错误 507000 和清理失败，请求在 600 秒后超时。整轮于 02:15 UTC 自行退出，退出码为 1，原生组未启动。

迁移作业在第 66 轮退出后取得四卡锁，已于 02:59 UTC 完成。最初暂停的 16 组启动器中，第 87 轮启动器后来不在进程表中，且没有退出记录或用例输出。迁移完成后确认其余 15 组的 32 个进程仍在独立的 freezer cgroup 中，状态为 `FROZEN`。迁移状态及进程清单保存在服务器 `/tmp/ep4-weight-move.mReub9`，冻结组路径见其中的 `paused-cgroup.txt`。

四份本地实验包均已复核 SHA-256。各轮的清单和生成代码检查与对应实验包一起保存。

| 轮次 | 本地实验包 |
| --- | --- |
| 88 | `/tmp/v41-prefill-88.wo5OVW/p88.tar.gz` |
| 89 | `/tmp/v41-prefill-89.fgEXiV/prefill-89-local.tar.gz` |
| 90 | `/tmp/v41-prefill-90.W3uDbb/prefill-90-local.tar.gz` |
| 91 | `/tmp/v41-prefill-91-ordered.qRSuV9/prefill-91-local.tar.gz` |

## 后续实验

1. 收齐第 79 轮候选、第 62 轮和原生的同场样本。单独记录八行合并回传相对第 62 轮的增量。第 84 轮用原生先跑、第 62 轮后跑的顺序复核六档输入，从 1,024 到 16,384 token 各测四条，按长度报告 Prefill P50、P95、TTFT、配对结果和绝对节省。前一次 16K 扫描中，长档的降幅较稳定，短档受运行顺序和波动影响较大。[第 79 轮](v41-a5-ep4-optimization-log.md#prefill-79--return-eight-compact-rows-per-store-with-32-scatter-blocks) [第 84 轮](v41-a5-ep4-optimization-log.md#sweep-84--compare-direct-expert-operand-stores-through-16384-tokens)
2. 优先验证第 91 轮异步回传。用第 80 轮作 MoonEP 同场对照，对比逐组 BF16 写入与每远端卡一次 URMA 提交。先跑集中到单个远端卡的 474 行、单源、混合源、79 行尾轮四项回放。重点核对空 owner 的通知、URMA 完成后才发信号，以及多轮串行复用时的数据一致性。再测原生、第 80 轮和候选的稳态 P50、P95。第 91 轮只引入已有运行时的 URMA workspace，沿用原有回传窗口和完成顺序。上机所需实验包在本地，上传待授权。[第 91 轮原型](v41-a5-ep4-optimization-log.md#prefill-91-local-serving-prototype)
3. 分别验证前向搬运和回传发布的增量。前向依次量化第 83、85、86、87 轮的块间均衡、元数据合包、整组激活写入和索引读取。以第 86 轮作对照核验第 88 轮的路由权重复用，以第 87 轮作对照核验第 90 轮的量化 scale 直读。回传发布先测第 81、82 轮，再对第 82 轮验证第 89 轮三个归并计数器，比较减少等待条件与增加原子加、清零的净效果。[前向搬运](v41-a5-ep4-optimization-log.md#prefill-83--balance-direct-operand-pushes-by-16-row-work-group) [回传发布](v41-a5-ep4-optimization-log.md#prefill-81--publish-compact-return-blocks-directly-from-scatter)
4. 将实测有增量的前向与回传机制组合，重新跑同场原生和优化前的 MoonEP，复测标准负载和 16K。三组固定 80 行、同一空尾轮策略和单槽串行复用，沿用原来的专家核、attention、Decode、shared expert、请求调度和权重加载。按同场热态 Serving Prefill P50 评估 30% 目标，按实际输运、写入和发布机制归因。两组共有的 W2 完成登记留在对照中。[固定配置验收](v41-a5-ep4-optimization-log.md#fixed-configuration-moonep-acceptance) [暖态追踪归因](v41-a5-ep4-optimization-log.md#return-entry-delay-across-full-and-tail-rounds)

## 服务器存储整理

2026-11-23 UTC 盘点时，`/home/hy` 实占约 635 GiB，EP4 预打包权重占 296 GiB，旧 EP1 全量预打包文件占约 269 GiB。EP4 比测继续使用现有 `sidecars/DeepSeek-V4.1-Flash-1017-ep4.safetensors`。EP1 的原始 checkpoint、重建脚本、日志和小样本保留在原位置。

已归档第 22、50、59、67 组已结束诊断的 28 份 `merged_swimlane_*.json`。每份 `.json.gz` 解压后已和原文件逐字节比对，共释放 19.83 GiB。原始 `chip_swimlane_records.json`、依赖图和名称映射仍可直接分析。需要打开某份合并追踪时，可在原目录通过 `gzip -dk 文件名.json.gz` 恢复。

旧 EP1 全量预打包文件已迁到 `/mnt/weight/a5-ep1-validation.5tvlKN/flash-1017-routed-experts.pack`。迁移于 01:51 UTC 完成，副本通过全量逐字节比对，大小为 288,777,895,936 字节，继续使用原来的属主和 0600 权限。原路径 `/home/hy/a5-ep1-validation.5tvlKN/prepack-v41/flash-1017-routed-experts.pack` 已指向新位置。整理后 `/home/hy` 实占 345.81 GiB，根分区可用 496 GiB，使用率为 86%。

EP4 预打包权重已迁到 `/mnt/weight/v41-a5-moonep-prefill-ep4/DeepSeek-V4.1-Flash-1017-ep4.safetensors`。317,105,642,208 字节通过全量逐字节比对，属主为 `23178:23178`，权限为 `0600`。原 `sidecars/DeepSeek-V4.1-Flash-1017-ep4.safetensors` 路径已通过符号链接指向新位置，并以运行账号验证可读。本次迁移前后，`/home/hy` 实占从 346.79 GiB 降到 51.46 GiB，释放约 295.33 GiB。迁移后根分区可用 768 GiB，使用率为 78%。

## 当前验收结果

默认配置下，第 62 轮在两次同场测试中将热态 Prefill P50 降低 **2.22%** 和 **1.83%**，两次均改善全部 12 对请求。30% 目标尚待验证。对照均为每轮 80 行、单通信槽和串行复用。标准负载含 1,024、3,070、6,142 token 各四条，第 84 轮扩展到 16,384 token。[同场对照复核](v41-a5-ep4-optimization-log.md#matched-serving-control-audit) [六档长度扫描](v41-a5-ep4-optimization-log.md#sweep-84--compare-direct-expert-operand-stores-through-16384-tokens)

第 23 轮合并发送侧路由描述符中的权重字段，并跳过无数据来源卡的激活推送。与较早的第 16 轮原生对照相比，热态 Prefill P50 从 2.238 秒降至 2.168 秒，下降 3.14%。[固定配置验收记录](v41-a5-ep4-optimization-log.md#fixed-configuration-moonep-acceptance)

第 47 轮在固定 80 行、单槽串行的条件下，只合并了 MoonEP 推送核的连续写入。两组冻结回放和三组 Serving 正确性检查均通过。热态 Prefill P50 为原生 3.437 秒、第 23 轮控制 2.368 秒、第 47 轮候选 4.045 秒。第 47 轮较同场第 23 轮慢 70.84%。第 23 轮对本次原生的数值降幅为 31.12%，同时本次原生 P50 较历史高 53.60%，收益归因尚未证实。按停止条件结束本轮优化。[第 47 轮结果](v41-a5-ep4-optimization-log.md#matched-device-result)

新目标继续验证固定配置。第 48 轮反序以第 23 轮 MoonEP 先跑，P50 为 3.676 秒，后跑的原生为 2.966 秒。两种顺序各取同等样本，合并 P50 为原生 3.374 秒和 MoonEP 2.985 秒，降幅 11.51%。[反序复测结果](v41-a5-ep4-optimization-log.md#prefill-48--repeat-the-single-slot-controls-in-reverse-order)

第 49 轮只改紧凑窗口的行规划。两组冻结回放和三组 Serving 检查均通过。热态 Prefill P50 为原生 3.749 秒、第 23 轮控制 2.402 秒、第 49 轮候选 2.427 秒，改动没有测得额外收益。[第 49 轮结果](v41-a5-ep4-optimization-log.md#prefill-49-device-result)

第 50 轮两组单槽追踪均已完成。输入 1,024 token 时，原生最慢 rank 的设备跨度为 609.3 ms，MoonEP 为 620.2 ms。MoonEP 接收端的空直方图在各自重建路径上约占 5.4–5.6 ms。源 rank 的路由元数据耗时和回传前的间隙均比原生长。[第 50 轮追踪](v41-a5-ep4-optimization-log.md#diagnostic-50--capture-the-current-single-slot-device-paths)

第 51 轮省去单源 compact 接收端的空直方图，两种冻结回放和三组 Serving 检查全部通过。热态 Prefill P50 为原生 2.293 秒、第 23 轮控制 2.905 秒、第 51 轮候选 2.641 秒。候选比同场 compact 控制快 9.09%，比原生慢 15.20%。[第 51 轮结果](v41-a5-ep4-optimization-log.md#prefill-51-device-result)

第 52 轮完成 80 行单槽的六档长度扫描，使用第 23 轮和原生对照。16,384 token 的热态 Prefill P50 从 14.583 秒降至 12.671 秒，降幅 13.11%。24 对请求中有 21 对改善。[单槽长度扫描](v41-a5-ep4-optimization-log.md#sweep-52-device-result)

第 53 轮合并尺度和路由描述符时，设备运行时拒绝同一通信窗口中的重叠任务参数。第 54 轮只用一个可写参数传递合并载荷。两种冻结回放和三组 Serving 检查均通过，候选 P50 为 2.349 秒，较同场原生 2.506 秒下降 6.29%，12 对请求全部改善。同场第 23 轮控制首跑 P50 为 4.264 秒，仍有明显的复测波动。[第 54 轮结果](v41-a5-ep4-optimization-log.md#prefill-54-device-result)

第 54 轮反序复测中，原生和 MoonEP 的 P50 分别为 2.364 和 3.000 秒。两种顺序等量合并后，第 54 轮 P50 高于原生 1.59%。第 55 轮候选在最新一轮的 P50 为 2.323 秒，比同场原生低 1.76%，12 对请求中有 11 对改善。第 55 轮全部检查通过。[反序复测](v41-a5-ep4-optimization-log.md#prefill-54-opposing-order-comparison) [第 55 轮结果](v41-a5-ep4-optimization-log.md#prefill-55-device-result)

第 56 轮把单源紧凑布局的对齐行计划随激活就绪信号发布，省去独立的直方图等待和行规划。两种数值回放和三组 Serving 检查均通过。原生、第 55 轮对照和第 56 轮候选的热态 P50 分别为 2.309、2.305 和 2.327 秒，候选比原生慢 0.79%。第 55 轮等量合并两种顺序后比原生低 0.33%。[第 56 轮结果](v41-a5-ep4-optimization-log.md#prefill-56-device-result)

第 57 轮把合并包的尺度复制从逐路改到 16 个 token 一组的解包阶段。两种数值回放及三组 Serving 检查均通过。候选热态 P50 为 2.250 秒，同场 compact 对照为 2.238 秒，候选慢 0.53%。同场原生 P50 为 3.619 秒，较上一轮同一原生对照高 56.75%。候选对这个原生样本的降幅为 37.84%，第 60 轮将反序复测。[第 57 轮结果](v41-a5-ep4-optimization-log.md#prefill-57-device-result)

第 58 轮每笔最多回传四个连续 BF16 行。按 1,560 个真实层轮估算，跨卡回传写入次数减少 74.69%。两组数值回放和三组 Serving 检查均通过。原生、第 56 轮 MoonEP 对照和回传候选的热态 Prefill P50 分别为 2.229、2.203 和 2.204 秒。候选较同场原生低 1.13%，较 MoonEP 对照高 0.07%。第 59 轮先预热，再采集第 55 和 56 轮的调度追踪。第 60 轮复用紧凑包的 32 字节前缀暂存六个路由权重，已通过本机编译并排队真机验证。[第 58 轮结果](v41-a5-ep4-optimization-log.md#prefill-58-device-result) [第 59 轮](v41-a5-ep4-optimization-log.md#diagnostic-59--capture-scheduler-timing-around-compact-transport) [第 60 轮](v41-a5-ep4-optimization-log.md#prefill-60--read-route-weights-from-the-prepared-compact-packet)

第 61 轮利用第 58 轮的紧凑布局，每组最多 16 个专家行的尺度按 W13 所需格式发送，并附同组 FP32 权重。按真实路由估算，跨卡前向写入减少 43.27%，前向传输字节增加 3.19%。真机单源、混合源和 79 个活跃行的尾包回放均已通过。同场热态 Prefill P50 为原生 2.184 秒、第 58 轮 2.236 秒、候选 2.193 秒。候选比第 58 轮低 1.92%，比原生高 0.41%。MoE 每轮保持默认 80 行。[第 61 轮结果](v41-a5-ep4-optimization-log.md#prefill-61-device-result)

第 62 轮将打包尺度和 FP32 权重直接写入专家输入窗口，省掉接收端复制。默认 80 行、单槽串行以及原专家启动规格相同。三项真机数值回放和三组 Serving 均已通过。原生、第 61 轮和第 62 轮的热态 Prefill P50 分别为 **2.085、2.086 和 2.038 秒**。第 62 轮较原生下降 **2.22%**，较 MoonEP 对照下降 **2.30%**，两组对照的 12 对请求均改善。[第 62 轮真机结果](v41-a5-ep4-optimization-log.md#prefill-62-device-result)

第 59 轮首次因隔离目录权限在采集前退出，私有副本重试已取得原设备锁，开始第 55 轮暖态追踪。第 60 轮三组均通过验证，原生、第 57 轮 MoonEP 对照和预打包权重候选的热态 Prefill P50 分别为 2.127、2.186 和 2.218 秒。两种顺序等量合并后，第 57 轮比原生低 21.93%。首跑曾测得的 30% 以上降幅在反序中未复现。[诊断重试](v41-a5-ep4-optimization-log.md#diagnostic-59--capture-scheduler-timing-around-compact-transport) [第 60 轮结果](v41-a5-ep4-optimization-log.md#prefill-60-device-result)

第 64 轮把单源专家计数和偏移随行计划写进现有直方图窗口，省掉整个接收端安装任务。混合源保留接收打包。通信缓冲区、每轮 80 行、单槽复用及专家核启动规格相同。三项真机数值回放和三组 Serving 均通过。原生、第 62 轮和候选的热态 Prefill P50 分别为 2.129、2.090 和 2.094 秒。候选较原生低 1.64%，较第 62 轮高 0.19%。[第 64 轮真机结果](v41-a5-ep4-optimization-log.md#prefill-64-device-result)

第 65 轮复用紧凑回传行号，把单源路由位置合成每轮一次本地向量写，并按实际活跃行读取。三项真机回放和三组 Serving 检查均通过。候选热态 Prefill P50 为 2.104 秒，比同场 MoonEP 对照低 1.54%，比同场原生 2.057 秒高 2.29%。[第 65 轮真机结果](v41-a5-ep4-optimization-log.md#prefill-65-device-result)

沿用冷态四卡追踪的依赖和核顺序，模拟去掉 MoonEP 传输任务耗时可缩短约 72 毫秒，连同这些任务前的间隙一起去掉约缩短 135 毫秒。暖态追踪用于区分这些间隙中的实际就绪与调度开销。[四卡回放](v41-a5-ep4-optimization-log.md#observed-order-replay-of-the-original-compact-trace)

第 59 轮已完成第 55、56 轮的四卡暖态追踪，输入均为 1,024 token。两轮源端推送的最慢块中位耗时分别为 49.2 和 48.9 微秒，块平均耗时的中位数均约为 9 微秒。去掉独立行计划后，推送块耗时分布仍接近。第 56 轮模拟把传输任务体耗时置零可缩短 58.8 毫秒。同条件的第 67 轮原生追踪已完成。原生最慢 rank 为 612.30 毫秒，第 56 轮为 613.59 毫秒。第 62 轮已进入同宽 Serving 对照。[暖态归因](v41-a5-ep4-optimization-log.md#diagnostic-59-warm-prefill-56-candidate) [原生暖态结果](v41-a5-ep4-optimization-log.md#diagnostic-67-warm-native-result)

第 66 轮复用原窗口的尾部暂存，远端卡只接收一次相同 token 的激活，再在原打包任务中补齐专家行。按真实路由估算，跨卡激活字节减少约 45.5%，同时增加接收端本地复制。主机核验确认 80 行单槽、缓冲区规格和专家核提交一致，已排队和第 58 轮、原生对照。[第 66 轮](v41-a5-ep4-optimization-log.md#prefill-66--share-one-staged-activation-per-remote-token-owner)

第 68 轮让已有推送任务按最多 16 行的连续块发送激活，沿用第 66 轮的接收扩展。按相同路由估算，跨卡激活写入从 558,268 笔降到 23,040 笔，激活字节减少 34.01%。主机核验保持每轮 80 行、单槽串行及原专家提交，已排队真机验证。[第 68 轮](v41-a5-ep4-optimization-log.md#prefill-68--move-the-staged-activation-prefix-in-16-row-tiles)

第 69 轮在第 68 轮的分块激活传输上，按 token 批量发送尺度与 FP32 路由权重。独立路由表沿用紧凑回传行序。模型估算远端描述符写入从 558,268 笔降至 27,720 笔。主机核验及 79 行尾包编译已通过。服务器已校验并排队，待第 68 轮完成后验证真机精度和同宽 Serving 延迟。[第 69 轮](v41-a5-ep4-optimization-log.md#prefill-69--stage-token-metadata-with-the-compact-route-map)

第 70 轮在原激活窗口尾部合并传输头与激活，首个头部同时携带紧凑路由表。相对第 69 轮，模型估算远端前向写入减少 54.61%，载荷字节增加 1.95%，源端增加一次片上组装。主机核验通过，保持默认 80 行和单通信槽，真机对照在第 69 轮之后。[第 70 轮](v41-a5-ep4-optimization-log.md#prefill-70--fuse-remote-activation-tiles-and-metadata-into-one-payload-write)

第 71 轮复用紧凑路由表，在源卡现有打包核内直接读本地激活、尺度和权重。单源推送只剩三组远端块，启动块数由 56 降到 15。默认 80 行、单槽及专家核启动规格已通过生成代码核验。服务器已校验实验包并排队，待第 70 轮完成后运行真机回放和 Serving 对照。[第 71 轮](v41-a5-ep4-optimization-log.md#prefill-71--pack-source-owned-expert-rows-directly-from-the-compact-route-map)

第 72 轮在组装 MoonEP 单源传输帧时直接解包尺度，省去四张卡各自的独立解包任务。完整 Serving 主机编译和服务器归档校验已通过。暖态追踪中尺度解包在 520 次源端推送前均先于其他前驱完成，这个版本先保留作备选。[第 72 轮](v41-a5-ep4-optimization-log.md#prefill-72--unpack-the-moonep-frame-scales-in-the-source-push)

第 73 轮将同一 16-token 块的尺度和权重提前组装一次，供三个远端卡复用。按 1,560 个真实层轮估算，帧头组装从 23,040 次减到 7,680 次。默认 80 行、单槽及专家核启动规格已通过生成代码核验。服务器已校验归档并排队，使用第 71 轮与原生作对照。[第 73 轮](v41-a5-ep4-optimization-log.md#prefill-73--prepare-each-compact-frame-header-once-before-push)

第 74 轮由源端传输块直接发布就绪，接收卡等待本卡的五个标志。旧暖态记录中，推送结束到发布开始累计 4.10 ms，发布执行累计 2.32 ms。方案复用现有信号窗口和单通信槽，主机核验通过，服务器已校验归档并排队。真机对照将衡量额外标志的净延迟成本。[第 74 轮](v41-a5-ep4-optimization-log.md#prefill-74--signal-each-source-frame-directly-from-the-push-task)

第 75 轮在原接收打包核内合并最多四个连续激活行。按真实路由估算，远端接收卡的复制操作减少 23.89%，字节数相同。完整 Serving 主机编译和生成代码核验已通过。归档已校验并排队，真机以第 73 轮和原生作对照。[第 75 轮](v41-a5-ep4-optimization-log.md#prefill-75--copy-up-to-four-adjacent-staged-activation-rows-at-a-time)

第 76 轮移除紧凑接收打包到 W13 之间的冗余空任务。三项数值回放和三组 Serving 检查均通过。热态 Prefill P50 为原生 2.140 秒、第 58 轮对照 2.147 秒、候选 2.162 秒。候选比原生高 1.07%，比第 58 轮高 0.72%。后续使用原有第 58 轮交接方式。[第 76 轮真机结果](v41-a5-ep4-optimization-log.md#prefill-76-device-result)

第 77 轮复用行计划的空余位携带活跃专家列表，单源接收打包从 32 块降到 16 块。真实路由模型中空闲块中位数从 21 降到 3，最忙块路由数 P95 仍为 67。主机核验已通过，补充了同一远端卡命中 20 个专家的数值回放。真机回放与 Serving 对照已在原四卡锁排队。[第 77 轮](v41-a5-ep4-optimization-log.md#prefill-77--index-active-experts-in-the-compact-receive-pack)

第 78 轮用片上向量排序和一次扫描生成单源路由表，保留专家内原有行序和传输格式。切片及完整 Serving 主机编译通过，生成代码中只有 MoonEP 的路由规划核改变。服务器已校验归档并排队，将在四项数值回放后测原生、第 73 轮和候选。[第 78 轮](v41-a5-ep4-optimization-log.md#prefill-78--sort-the-compact-source-route-map-in-unified-buffer)

第 79 轮在第 62 轮的单源回传中把每笔连续结果从 4 行扩到 8 行，回传启动块从 64 减到 32。集中路由、单源、混合源和尾轮四项真机回放均通过。候选的 12 条热态请求 Prefill P50 为 2.109 秒，P95 为 4.302 秒，同场对照待齐。[第 79 轮](v41-a5-ep4-optimization-log.md#prefill-79--return-eight-compact-rows-per-store-with-32-scatter-blocks)

第 80 轮把单源回传再扩大到每笔 16 行，启动 16 块。真实路由模型估计远端写入从第 62 轮的 141,309 笔降到 37,095 笔，BF16 数据量相同。主机核验通过，真机已在第 79 轮后排队。[第 80 轮](v41-a5-ep4-optimization-log.md#prefill-80--return-sixteen-compact-rows-per-store-with-16-scatter-blocks)

第 81 轮利用 16 块连续回传，由各回传块写完后直接向源卡发布完成，源卡等待 48 个远端标志。生成代码已核验写入顺序、原轮间释放和专家核启动规格。服务器已校验并排队，使用第 80 轮和原生作对照。[第 81 轮](v41-a5-ep4-optimization-log.md#prefill-81--publish-compact-return-blocks-directly-from-scatter)

第 82 轮把每卡需要的标志数放进源卡现有路由表的空余位，接收方只发布有数据的块，空接收方由第一个块确认。真实路由估计标志数减少 50.48%。主机核验通过，服务器已校验并排队，将以第 81 轮和原生作对照。[第 82 轮](v41-a5-ep4-optimization-log.md#prefill-82--register-only-the-compact-return-flags-that-cover-each-owner)

13 组同场记录的原生热态 P50 在 2.085 至 3.749 秒之间。第 49 和 57 轮的候选对各自原生降幅一度超过 30%，两版也都慢于同场 MoonEP 对照。归因需结合反序复测和同场增量。[对照复核](v41-a5-ep4-optimization-log.md#matched-serving-control-audit)

第 83 轮在第 62 轮上把发送工作按 16 行传输组均衡分给原 56 块。真实路由估计最忙块的路由数 P95 从 101 降至 32，传输地址和专家核保持一致。主机核验通过，归档已校验并排队。[第 83 轮](v41-a5-ep4-optimization-log.md#prefill-83--balance-direct-operand-pushes-by-16-row-work-group)

第 85 轮把每组的地址、行数和路由索引合并进一个 64 字节本地包，发送块每组只读一次。真实路由估计相关本地元数据写入少 22.69%，额外占用约 28 KiB 本地暂存。完整主机核验通过，服务器已校验并排队，使用第 83 轮和原生作对照。[第 85 轮](v41-a5-ep4-optimization-log.md#prefill-85--build-one-local-metadata-packet-per-direct-push-group)

第 86 轮在原 16 行传输组内直接把激活加载到片上目标行，再整组发送。真实路由估算远端激活写入减少 86.55%，前向全部写入减少 68.20%。第 87 轮在此基础上用 A5 行索引 gather 把每组多次源读取合成一次。两版均通过生成代码核验，服务器已校验并排队。[第 86 轮](v41-a5-ep4-optimization-log.md#prefill-86--store-each-direct-push-activation-group-as-one-tile) [第 87 轮](v41-a5-ep4-optimization-log.md#prefill-87--gather-a-compact-activation-group-with-one-indexed-load)

第 88 轮独立基于第 86 轮，每轮读取一次 FP32 路由权重，并将每组所需的权重放进本地发送包。路由回放估计相关本地访问少约 168 MiB，每卡暂存增加约 30 KiB。主机编译和生成代码核验通过，实验包等待授权上传。[第 88 轮](v41-a5-ep4-optimization-log.md#prefill-88--pack-exact-routing-weights-once-per-source-round)

按原顺序重放 1,024 token 暖态设备追踪，把回传 scatter、通知及其入口等待置零可缩短约 43 毫秒，把源端推送及其入口等待置零可缩短约 32 毫秒。第 89 轮把远端回传的逐块完成条件合并为每卡一个计数，条件中位数从 24 降到 3。完整主机核验通过，实际收益取决于原子加和清零成本。[分阶段追踪回放](v41-a5-ep4-optimization-log.md#warm-transport-sensitivity-by-phase) [第 89 轮](v41-a5-ep4-optimization-log.md#prefill-89--wait-for-one-return-count-per-remote-owner)

原生和 MoonEP 的 W2 结束到 scatter 启动间隔中，约四分之三来自 W2 的完成登记。MoonEP 尾轮少 16 个活跃行，scatter 本体中位数减少约 9 微秒，入口等待仍约 42 微秒。合并回传写入主要压缩本体，发布机制变化的收益需单独测量。[整轮与尾轮的回传间隔](v41-a5-ep4-optimization-log.md#return-entry-delay-across-full-and-tail-rounds)

第 90 轮在第 87 轮上把量化 scale 直接读入发送组，省去每条路由一次片上搬运。真实路由回放估计少搬约 112 MiB，跨卡数据量相同。主机核验确认只有 MoonEP 推送核改变。[第 90 轮](v41-a5-ep4-optimization-log.md#prefill-90--load-scale-rows-directly-into-the-compact-send-tile)

第 91 轮本地原型让非源卡的 W2 写入已有回传窗口，再用一笔 URMA 提交发送该卡的连续 BF16 结果。按已有路由，从第 80 轮的 37,095 笔远端写入压到 4,680 笔提交。六个切片和两套完整 Serving 均通过主机编译。原有设备核、原生提交与分配顺序、专家启动规格均匹配，通知等待异步回传任务完成。真机正确性和延迟待测。[异步回传原型](v41-a5-ep4-optimization-log.md#prefill-91-local-serving-prototype)

下方保留此前探索的数据。第 34 轮的 21.17% 和第 20 轮的 16K 扫测都使用双槽与轮间重叠，属于历史配置。第 46 轮实验包保留供回顾。

## 历史实测与归因

MoonEP 独立收益以 Serving 默认宽度为准。TP1 下默认宽度为 80 行，由 `12 × 6 = 72` 向上对齐到 16 的倍数得到。原生和 compact 都沿用这个宽度，一个完整的 1,024 token 分块有 13 轮。两组使用相同的空尾轮裁剪、请求负载和运行参数。256 行结果作为扩宽实验保留。

原 8,320 token 上下文配置中，目前默认宽度 P50 最好的是 Prefill 34。沿用 Prefill 16 原生对照，热态 Prefill P50 从 2.238 s 降至 1.764 s，下降 21.17%。P95 为 3.841 s，TTFT P50/P95 为 1.850/3.977 s。单源和混合来源冻结回放、12 条计时请求、长问答和两轮并发均通过。按这组原生对照，30% 对应 P50 为 1.566 s，还需减少 198 ms。第 25 轮原生对照也启用了本地提前准备，回读后以该组 P50 的 70% 核定 MoonEP 独立收益的目标值。[Prefill 34 逐轮记录](v41-a5-ep4-optimization-log.md#prefill-34--reuse-the-active-expert-plan-in-scale-packing-and-scatter)

16,384 token 长度扫测使用第 16 轮原生路径和第 20 轮 MoonEP，两组均为默认 80 行，服务上下文上限为 16,640 token，按 1,024 token 分块。两组各跑六个输入长度，每种四条，24 对请求的 Prefill 和 TTFT 均有改善。16,384 token 的 Prefill P50 从 12.056 s 降到 10.286 s，改善 14.69%，TTFT P50 从 12.161 s 降到 10.390 s。8,192 token 组的 Prefill 相对降幅最大，为 16.90%。两组均通过 16,384 token 长问答和两轮并发验证。[16K 扫测完整结果](v41-a5-ep4-optimization-log.md#default-width-prompt-length-sweep-through-16384-tokens)

同宽 1,024 token 芯片追踪已确认双槽能让相邻轮的通信和专家计算并行。最慢 rank 的芯片执行跨度从 627.418 ms 降到 493.932 ms，下降 21.28%。源卡的激活推送与专家计算重叠 9.267 ms，仍有 36.087 ms 推送时间未与专家计算重叠。正式端到端收益使用上面的热态 Serving P50。[80 行配对追踪](v41-a5-ep4-optimization-log.md#diagnostic-22--compare-native-and-compact-chip-paths-at-80-rows)

第 30 轮按 token 推送的 80 行单源回放最慢 rank 中位耗时下降 11.72%。正式 Serving P50 为 1.924 s，比原双槽高 5.22%。正确性检查全部通过，后续将用同一真实模型追踪查看推送、专家计算和等待间隙。第 33–35 轮针对有效专家计划的改动均已完成编译并排队。[第 30 轮真机结果](v41-a5-ep4-optimization-log.md#prefill-30--reuse-source-activations-across-the-routes-of-each-token)

第 37 轮把单源有效专家计划随激活发布，复用载荷就绪通知，省去接收端单独的直方图等待和行规划。两种 slab 形状与完整 Serving 已通过本机编译，真机验证排在第 35 轮之后。真实单源路由显示，按 token 和目标卡去重可少发 45.50% 的前向激活字节，保留逐路 BF16 回传时往返逻辑载荷合计少 15.48%。[第 37 轮与 MoonEP 开销分析](v41-a5-ep4-optimization-log.md#prefill-37--publish-each-active-expert-plan-with-its-activation-payload)

第 33 轮稀疏专家计划通过两种冻结回放和 Serving 正确性验证。单源回放最慢 rank 中位耗时下降 21.33%，Serving Prefill P50 为 2.182 s，较第 20 轮高 19.33%。[第 33 轮结果](v41-a5-ep4-optimization-log.md#prefill-33--give-w13-and-w2-a-plan-containing-only-active-experts)

第 38 轮用下一次载荷通知完成单源双槽复用，完整分块中省掉前 11 轮的独立释放握手。两种冻结回放和 Serving 正确性均通过，热态 Prefill P50 为 1.953 s，较第 20 轮高 6.81%。第 39 轮继续做单独对照，把这 11 轮的回传通知进一步收敛到源卡。[第 38 轮同步优化](v41-a5-ep4-optimization-log.md#prefill-38--reuse-interior-single-source-slots-on-the-next-payload-epoch)

第 36 轮芯片追踪中，按 token 调度把源卡推送活动从 45.354 ms 降到 16.989 ms，四卡最慢跨度仅缩短 0.73%，最慢卡转移到了接收端。第 40 轮改按目标卡和连续专家行分段，第 41 轮直接发布接收计划并沿用第 20 轮的专家核。两轮已通过本机编译并排队真机对照。[第 36 轮芯片追踪](v41-a5-ep4-optimization-log.md#diagnostic-36--explain-the-token-push-regression-in-the-real-model)

第 42 轮诊断发现回传任务按整个窗口排序。第 44 轮基于第 34 轮，把实际回传与归并窗口都改成逐槽视图，保留运行时的张量生命周期跟踪。已通过本机编译并排队真机。[第 44 轮逐槽回传](v41-a5-ep4-optimization-log.md#prefill-44--track-only-the-return-rows-of-each-slot)

第 45 轮用归档的四卡任务图核实了同层相邻槽的等待。每卡发现 480 对相邻轮的直方图任务被整体窗口排序，源卡的激活推送和就绪通知也各有 480 对。第 25 轮原生对照最近一次检查已通过服务健康检查，计时结果待回读。[第 45 轮任务依赖](v41-a5-ep4-optimization-log.md#diagnostic-45--locate-tensor-map-waits-across-the-two-communication-slots)

第 46 轮基于第 34 轮，让 80 行单源激活推送按 3,456 行的槽视图登记写入。运行时重叠检查确认两个槽互不重叠，两种冻结形状与完整 Serving 通过本机编译。实验脚本校验各组启动参数，并用第 25 轮原生对照计算独立收益。第 25 轮结果暂不可用时，先汇报相对第 34 轮的增量。实验包已就绪，等待恢复 SSH 后上机。[第 46 轮逐槽激活推送](v41-a5-ep4-optimization-log.md#prefill-46--track-activation-writes-by-communication-slot)

| 项目 | 原生或优化前对照 | 已验证结果 |
| --- | ---: | ---: |
| Serving 从启动到 HTTP /health | 1,700.303 s | 权重预打包后首次 334.124 s，重复启动 308.063 s |
| 扩宽组合测试的热态 Prefill P50 | 80 行原生为 2.430 s | 256 行 MoonEP compact 为 1.665 s，降低 31.47% |
| 扩宽组合测试的热态 Prefill P95 | 80 行原生为 4.663 s | 256 行 MoonEP compact 为 3.252 s，降低 30.26% |
| 扩宽组合测试的首次 token 延迟 P50 / P95 | 2.509 / 4.750 s | 1.744 / 3.337 s |

31.47% 是组合配置的收益。每轮 MoE 从 80 行扩大到 256 行后，完整的 1,024 token 分块从 13 轮减到 4 轮，专家核也按新宽度特化。对照同时改变了空尾轮裁剪和通信后端。MoonEP 集成的独立热态 Prefill 延迟收益尚未达到 30% 目标。

早期普通配置下，80 行 compact 的 Prefill P50/P95 为 2.297/6.341 s，裁剪空尾轮的 80 行原生为 2.180/4.933 s。compact 的 P50 高 5.3%。同轮锁定原生的 P50 为 2.223 s。这组历史对照使用的是当时的实现，详见[逐轮优化记录的 Prefill 01](v41-a5-ep4-optimization-log.md#prefill-01--reverse-order-and-isolate-empty-round-trimming)。

80 行 Prefill 16 对照统一使用相同的空尾轮裁剪。原生 P50/P95 为 2.238/4.465 s，原 compact 为 2.302/4.597 s，移除单源空发送后为 2.237/4.539 s。该项相对原 compact 改善 P50 2.85%，相对原生改善 0.04%。单源和混合来源冻结校验、12 条计时请求、长问答和两轮并发均通过。

80 行 Prefill 18 的回传裁剪通过了相同的正确性检查，P50/P95 为 2.312/4.601 s，P50 比 Prefill 16 compact 高 3.36%。后续继续用 Prefill 16 作 80 行 compact 对照。

80 行 Prefill 19 把单源行偏移并入直方图任务，P50/P95 为 2.176/4.442 s。沿用 Prefill 16 同宽对照，P50 比原 compact 低 2.73%，比原生低 2.77%。两组冻结回放和完整 Serving 校验均通过。第 26 轮已排队，验证把这项改动叠加到双槽后的增量。[Prefill 19 逐轮记录](v41-a5-ep4-optimization-log.md#prefill-19--build-the-sole-sources-offsets-during-histogram-publication)

256 行 Prefill 15b 的三组 Serving 验证均通过。修复 scatter 分片数后，原生还通过了冻结校验，P50/P95 为 1.506/3.033 s。同宽 Prefill 14 compact 为 1.453/2.904 s，P50 改善 3.53%。激活去重候选的 P50 为 1.524 s，比原 compact 高 4.88%。同宽度 30% 目标对应 P50 约 1.054 s。详见[逐轮优化记录的 Prefill 15b 和 16](v41-a5-ep4-optimization-log.md#prefill-15b--correct-wide-native-scatter-before-comparison)。

Prefill 17 把路由权重合入已有回传描述字。候选通过两项冻结回放和完整 Serving 校验，256 行 P50/P95 为 1.360/2.872 s。沿用已校验的 256 行对照，P50 比原 compact 低 6.44%，比原生低 9.74%。详见[Prefill 17 逐轮记录](v41-a5-ep4-optimization-log.md#prefill-17--put-the-route-weight-in-the-existing-return-descriptor)。

组合配置的测试条件是固定 12 条请求、TP1/CP1/DP4/EP4、1,024 token 分块、关闭 DSpark 和 prefix cache。按实际输入 1,024、3,070、6,142 token 分组，Prefill P50 分别降低 31.08%、31.47%、31.42%。用较早一次原生 P50 2.223 s 比较，同一候选结果降低 25.10%。

复现扩宽组合时，使用 `PYPTO_V41_PREFILL_MOE_TOKENS=256` 和 `--v41-prefill-ep-backend moonep`。该候选通过 12 条计时请求，每条生成 13 token。另外通过 3,070 token 长问答（答案 `323`）和两轮各 8 条并发请求。主机上共通过 48 项定向测试，覆盖 CLI、Prefill 轮次元数据、sidecar 和路由回放。并发波次的结果用于正确性验证。

## 代码与数据位置

本地父目录是 `/home/huangyi/devspace/inference/serving/.worktrees/v41-a5-moonep-prefill`。服务器父目录是 `/home/hy/hw-native-sys/.worktrees/v41-a5-moonep-prefill`。两个目录下都有 `pypto-serving` 和 `pypto-lib`，配套分支均为 `feat/v41-a5-moonep-prefill`。

| 仓库 | 本地提交 | 锁定原生对照 |
| --- | --- | --- |
| Serving | `6f1ffca` MoonEP 集成、测试和记录，`c8ffcd6` 权重预打包 | `ffab1ab07e4f926c112ce416589892df9dca2556` |
| pypto-lib | `7785e97b59067fdc43ec871636a770c5c519cdce` MoonEP 设备实现 | `3dc359fe57419506b42a19989ff072a7898a50ed` |

Serving 的 `6f1ffca` 已记录 pypto-lib 的子模块提交。本地两个分支的提交已创建。最后一次检查服务器时，两个 `HEAD` 仍指向原生基准提交，验收源码以工作树修改的形式同步到了服务器。核对服务器版本时同时查看 `git status` 和实际文件。`verification/ep4/prefill-12-verified-source.tar.gz` 保存了验收后同步的 14 个文件，SHA-256 为 `da27300de6cae6ecbadfa2232625af2227f6b819b587c797fc5ba0ffb097a5a3`。代码文件在同步时已与正式真机实验快照逐字节比较。

服务器上的关键位置如下。

| 目录或文件 | 用途 |
| --- | --- |
| `/home/hy/hw-native-env/v41-a5` | 宿主机运行环境、虚拟环境和设备锁 |
| `/mnt/weight/DeepSeek-V4.1-Flash-1017` | 真机 checkpoint |
| `verification/ep4/env.sh` | 装载 CANN、PyPTO、Python 以及 HCCL 环境，设置模型、缓存和逻辑卡 |
| `verification/ep4/sidecars/DeepSeek-V4.1-Flash-1017-ep4.safetensors` | 295.33 GiB EP4 预打包权重，由现有隔离运行身份读取 |
| `verification/ep4/baseline/{pypto-serving,pypto-lib}` | 两个锁定原生提交的源码快照 |
| `verification/ep4/benchmark-00-workload.json` | 固定请求，SHA-256 为 `e538a50d91ef60a00178490e8615f4ed82a276ba06e304f817748086c9dbfb6f` |
| `verification/ep4/experiments/prefill-12` | 正式 CLI 验收时用的源码快照 |
| `verification/ep4/benchmark-12-comparison.json` | 原生、原型和正式 CLI 的逐请求对照 |

## 如何上机

### 登录、核对状态和分配设备

使用团队提供的 SSH 配置登录已分配的 A5 主机，在服务器上设置下面两个变量。已验证的环境为 Ascend 950DT、CANN 9.2.0、PTOAS 0.72 和 Python 3.12.16，软件路径由 `env.sh` 装载。

```bash
ep4_project=/home/hy/hw-native-sys/.worktrees/v41-a5-moonep-prefill
ep4_verify="$ep4_project/verification/ep4"

hostname
for repo in pypto-serving pypto-lib; do
    git -C "$ep4_project/$repo" status --short --branch
    git -C "$ep4_project/$repo" rev-parse HEAD
done
sha256sum "$ep4_verify/benchmark-00-workload.json" \
    "$ep4_verify/prefill-12-verified-source.tar.gz"
stat -c '%s bytes %n' \
    "$ep4_verify/sidecars/DeepSeek-V4.1-Flash-1017-ep4.safetensors"
ps -eo pid,etime,args | grep -E '[r]un-verified-cards.sh|[b]enchmark-12b.py'
npu-smi info
```

如需确认服务器仍是验收源码，可将同步包内的代码文件与当前工作树逐个比较。

```bash
(
    set -euo pipefail
    archive="$ep4_verify/prefill-12-verified-source.tar.gz"
    while IFS= read -r path; do
        case "$path" in
            */docs/*) continue ;;
        esac
        tar -xOzf "$archive" "$path" | cmp - "$ep4_project/$path"
    done < <(tar -tzf "$archive")
)
```

真机入口是 `run-verified-cards.sh <作业名> <启动脚本> [参数...]`。外层通过 `flock` 使用 `/home/hy/hw-native-env/v41-a5/cards-4-7.lock`，拿锁后再次检查 4–7 卡是否空闲。有占用时返回 `3`。内层通过 systemd scope 和 `setpriv` 以 UID/GID `23178` 运行，物理卡 **4–7** 映射为进程内的 **0–3**。设定 `TASK_DEVICE=0,1,2,3` 后，CLI 使用 `--devices "$TASK_DEVICE"`。作用域限额为 12 个逻辑 CPU 等效、384 GiB 内存、768 个任务和 9,000 秒运行时间。CPU、内存和磁盘仍由同机任务共享。HTTP 服务绑定 `127.0.0.1`，测试驱动为每次启动选择端口。

之前尝试过容器启动，普通 runc 路径在 ACL 初始化时报错。宿主机启动路径通过了实际设备校验和完整 HTTP 测试。`run-verified-cards.sh` 会清理继承环境，模型和调优环境变量应由传入的启动脚本在加载 `env.sh` 后设置。

### 在当前同步源码上复现 80 行原生与 256 行 compact 的组合对照

用同一个已有驱动 `benchmark-12b.py` 跑两组。下面的包装脚本选择源码并给每次运行独立的输出名。原生使用锁定快照，compact 使用服务器当前的两个配套工作树。若要按验收时的源码逐字复现，可先核对同步包，也可把包装脚本中的 compact 源码路径指到上述 `experiments/prefill-12` 快照。

```bash
ep4_run="ep4-$(date -u +%Y%m%dT%H%M%SZ)"
test -s "$ep4_verify/benchmark-00-workload.json"
test -s "$ep4_verify/sidecars/DeepSeek-V4.1-Flash-1017-ep4.safetensors"

cat > "$ep4_verify/$ep4_run.sh" <<'BASH'
#!/bin/bash
source "$(dirname "$0")/env.sh"
cd "$here"
arm=$1
export PYPTO_V41_TEST_ARM="$2-$arm"
export PYPTO_V41_PREPACKED_WEIGHTS="$here/sidecars/DeepSeek-V4.1-Flash-1017-ep4.safetensors"
export PYPTO_CACHE=1
unset PYPTO_RUNTIME_LOG
case "$arm" in
    native)
        serving="$here/baseline/pypto-serving"
        export PYPTO_LIB_ROOT="$here/baseline/pypto-lib"
        export PYPTO_LIB_REVISION=3dc359fe57419506b42a19989ff072a7898a50ed
        ;;
    compact)
        serving="$project/pypto-serving"
        export PYPTO_LIB_ROOT="$project/pypto-lib"
        export PYPTO_V41_PREFILL_MOONEP_LAYOUT=compact
        export PYPTO_V41_PREFILL_MOE_TOKENS=256
        ;;
    *) exit 2 ;;
esac
export PYTHONPATH="$serving:$PYPTO_LIB_ROOT:$previous/pypto/python:$previous/pypto/runtime/python:$previous/pypto/runtime"
exec python "$here/benchmark-12b.py" "$arm"
BASH
chmod 0644 "$ep4_verify/$ep4_run.sh"
bash -n "$ep4_verify/$ep4_run.sh"

nohup bash -c '
    set -euo pipefail
    verify=$1
    run=$2
    for arm in native compact; do
        label="$run-$arm"
        bash "$verify/run-verified-cards.sh" "$label" "$verify/$run.sh" "$arm" "$run" \
            > "$verify/$label.log" 2>&1
    done
' bash "$ep4_verify" "$ep4_run" > "$ep4_verify/$ep4_run-suite.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$ep4_verify/$ep4_run-suite.pid"
printf 'run=%s\n' "$ep4_run"
```

驱动会保存完整 CLI 到 `command.json`。此次对照的主要参数是 `--tp 1 --dp 4 --ep 4 --block-size 128 --max-model-len 8320 --max-num-seqs 8 --max-num-batched-tokens 1024 --enable-chunked-prefill --long-prefill-token-threshold 1024 --no-enable-prefix-caching`。候选还带有 `--v41-prefill-ep-backend moonep`。两组均使用相同 sidecar、JIT 缓存策略和固定输入文件，并将调试计时、设备追踪和路由录制留在独立的诊断运行里。

### 观察进度、确认退出和生成对照

SSH 重连后重新设置 `ep4_verify` 和启动时打印的 `ep4_run`。通过 PID 查进程，通过每组的 `.exit` 确认结果。区分等待设备锁、启动到健康、首次预热和正式计时这几段。

```bash
ps -p "$(cat "$ep4_verify/$ep4_run-suite.pid")" -o pid,etime,stat,args
for arm in native compact; do
    label="$ep4_run-$arm"
    result="$ep4_verify/benchmark-12b-$label"
    if test -f "$ep4_verify/$label.exit"; then
        printf '%s ' "$label"
        cat "$ep4_verify/$label.exit"
    fi
    if test -f "$ep4_verify/$label.log"; then
        tail -n 6 "$ep4_verify/$label.log"
    fi
    if test -f "$result/server.log"; then
        tail -n 8 "$result/server.log"
    fi
done
```

正常完成时两组 `.exit` 都是 `0`。用原有汇总器计算新的配对结果，引用本次的原生目录。

```bash
test "$(cat "$ep4_verify/$ep4_run-native.exit")" = 0 &&
test "$(cat "$ep4_verify/$ep4_run-compact.exit")" = 0 &&
sed "s/benchmark-11-native/benchmark-12b-$ep4_run-native/" \
    "$ep4_verify/summarize-11.py" > "$ep4_verify/$ep4_run-report.py" &&
python3 "$ep4_verify/$ep4_run-report.py" "$ep4_run-comparison.json" \
    "benchmark-12b-$ep4_run-compact"
```

每组目录包含 `warmup-result.json`、`http.json`、`answer.json`、`wave-0.json`、`wave-1.json`、`command.json` 和 `server.log`。测试驱动在退出时关闭自身启动的 HTTP 进程，随后可用 `npu-smi info` 核对设备释放状态。

驱动在每条串行请求前后读取 `/metrics/json`，把各 replica 的 `vllm_histograms.prefill` 的 `sum` 差作为该条服务端 Prefill 耗时，要求 `count` 恰好增加 1。单独采集客户端 TTFT。12 条请求按 `[1024, 3070, 6142]` 实际输入长度重复 4 次，最多生成 16 token，实测每条生成 13 token。对照报告对服务端 Prefill 计算中位数和 nearest-rank P95，收益公式为 `1 - candidate / native`。健康检查、首次预热、问答和并发验证各有记录，正式计时取热态的 12 个串行样本。

### 单独测量 MoonEP 集成收益

在同一对 Serving 和 pypto-lib 源码上，用 `benchmark-12b.py` 复跑固定的 12 条请求。两组都取消设置 `PYPTO_V41_PREFILL_MOE_TOKENS`，使用默认的 80 行 Prefill 宽度。原生设置 `PYPTO_V41_PREFILL_TRIM_EMPTY_ROUNDS=1`，MoonEP compact 默认裁剪空尾轮。两组沿用上面的 checkpoint、分块、缓存和预热条件，报告热态 Prefill P50/P95 和 TTFT。每组使用独立 Serving 进程，保留实际 CLI 和逐请求结果。

第 13 轮沿用这组 80 行条件，加入省去副本放置任务的 compact 候选。三组源码、启动包装脚本和主机编译结果见[逐轮优化记录的 Prefill 13](v41-a5-ep4-optimization-log.md#prefill-13--skip-replica-placement-in-fixed-owner-dispatch)。

第 14 轮已完成 80 行真机对照。原生、原 compact 和单来源优化的 compact 热态 Prefill P50 分别为 2.160、2.304 和 2.231 秒。三组均通过 12 条同序请求、长问答和两轮并发。单来源改动缩小了与原生的差距，仍未达到 30% 的同宽收益目标。[第 14 轮结果](v41-a5-ep4-optimization-log.md#prefill-14--compact-dispatch-from-one-active-source)

256 行对照使用隔离的 Serving 配置，让原生和 compact 都走 `moe_tokens=256` 的编译和调度路径，每个完整分块使用 4 轮。首个宽原生进程已完成设备分配和健康检查，但回传分区参数与启动的 worker 数不一致，长问答校验失败。第 15b 轮把同一处修正应用到所有对照臂，先验证宽原生 golden，再采集三组 Serving 指标。80 行和 256 行都以各自通过正确性检查的同宽原生为基线。[第 15b 轮记录](v41-a5-ep4-optimization-log.md#prefill-15b--correct-wide-native-scatter-before-comparison)

### 单独验证 MoE 设备路径和权重阶段

需要独立检查紧凑通信和专家计算时，可以在同一隔离方式下复用冻结的 1,024 token golden。`moe-04/golden-data` 保存输入和期望输出目录。这个用例使用四卡行数 `1024,0,0,0`、96 experts/rank，运行 5 轮预热和 20 轮计时，比较 `next_pre_mix`、`x_mixed` 和 `x_next`。

```bash
test -d "$(cat "$ep4_verify/moe-04/golden-data")" &&
bash "$ep4_verify/run-verified-cards.sh" "$ep4_run-golden-t256" \
    "$ep4_verify/moe-04.sh" 256 \
    > "$ep4_verify/$ep4_run-golden-t256.log" 2>&1
```

正常上机直接复用现有 sidecar。检查 `server.log` 是否出现 `Mapped prepacked V4.1 weight bank`。若需要重建，先检查本地盘空间和现有文件权限，使用同一宿主机环境和隔离运行身份，在 CPU 专用作用域内执行 `python -m pypto_serving.tools.prepack_deepseek_v41 "$PYPTO_DSV41_MODEL_DIR" --ranks 4 --output <新文件绝对路径>`。原始构建用时约 1,318 秒，产物为 317,105,642,208 字节。文件通过 checkpoint 指纹和 EP 布局校验，供 Serving mmap 使用。启动收益的原始对照使用当时的 JIT 策略，并单独记录页缓存驻留比例。

## 此前做了什么

| 阶段 | 已完成的改动和实验结论 |
| --- | --- |
| 启动优化 | 确认 checkpoint 权重整理是主要开销。增加 v2 sidecar 写入工具，流式保存现有权重布局，保留 float8 元数据。首次与重复启动均通过长问答及并发回归。 |
| MoonEP 集成 | 接入 `fixed`、`compact`、`online`，实现路由录制、回放和有界副本实验。EP4 初始 golden 覆盖 `native`、`fixed`、`compact` 和 `online`，四个 rank 的活跃行数为 `81,80,79,0`。 |
| 基线和筛选 | 一组带 profiler 的早期结果未能在普通配置复现。原生空尾轮裁剪在固定负载的中位数改善约 1.9%。逐专家批量写元数据通过 golden，但真实 Serving 中位数相对原 compact 恶化 24.4%，已回退。 |
| 宽度实验 | 使用同一 1,024 token golden 测试 80、128、160、192、256 行，并用真实路由回放和芯片追踪定位。256 行的独立用例通过 golden，逐轮四卡最大有效耗时的中位数从 80 行的 15.618 ms 降到 9.025 ms。256 行原型在同轮 Serving 对照中降低 Prefill P50 34.0%。 |
| 正式源码 | 使 Prefill 宽度成为独立编译参数，通过静态构造专门化 gate 和 shared expert。正式 CLI 在同一工作进程为 Decode 编译 80 行、为 Prefill 编译 256 行，完整分块使用四轮。修复了启动时配置探测未传递 MoonEP 选择的问题，正式用例测得组合 P50 降幅 31.47%。 |

代码地图如下。

- Serving 入口在 `pypto_serving/cli/main.py` 和 `pypto_serving/model/deepseek_v41/npu_executor.py`。`npu_runner.py` 的 `compile_prefill_fwd`、`make_production_bindings` 和 `PrefillMetadataResources` 依次关联编译选择和调度器的 `moe_rounds`。
- pypto-lib 的相关文件在 `models/deepseek_v4_1_flash/`。`prefill_fwd.py` 是完整入口，`prefill_moe.py` 管理分轮和合并，`prefill_moonep.py` 实现直接派发、紧凑布局和回传。`gate.py`、`expert_shared.py` 和 `expert_routed.py` 提供共享的计算部分。
- 权重预打包工具在 `pypto_serving/tools/prepack_deepseek_v41.py`。HTTP 请求记录使用 `scripts/v41_prefill_bench.py`，冒烟测试在 `tests/test_deepseek_v41_serving.py`。

## 已见症状与证据

| 症状 | 已确认的情况 |
| --- | --- |
| 已通过健康检查，首个请求仍在等待 | 两个通信域的对端状态等待累计约 185 秒，日志出现 `P2P status unconfirmed`。正式计时在预热完成后开始。 |
| DSpark 开启后首个 Decode 报 `MallocPhysical -> 207001` | 之前 EP4 k5 的额外 HCCL 通信窗口发生设备内存分配失败。日志在 `host-retry-compact-k5.log`，30% 结果覆盖 k0。 |
| 路由回放 fixture 报 `207001` | 旧独立 fixture 预留 16 GiB ring heap，改用 Serving 的 10 GiB ring heap 和对应依赖池后跑通。对照在优化记录的 Prefill 07。 |
| 宽度改动在 host 编译阶段失败 | PyPTO 对依赖调用的 `constexpr` 需要显式绑定，函数内计算的局部量无法作为依赖特化参数。W13 padded/direct 形状在构造阶段静态选择，详见 Prefill 12。 |
| 相同宽度在诊断和正式对照中收益不同 | 192 行的早期普通运行曾回退，启用 `PYPTO_RUNTIME_LOG=timing` 的结果明显不同。正式 256 行结果取普通配置和同轮原生基线，历史对照保留各自的日志配置。 |

原始证据都在服务器 `verification/ep4` 下。启动看 `host-retry-compact-k0.log`、`startup-01-pack.log`、`startup-01-first.log` 和 `startup-01-repeat.log`。最终测试看 `benchmark-11-native/`、`benchmark-12b-t256/` 与 `benchmark-12-comparison.json`。256 行 kernel golden 在 `moe-11-t256.log`。主机测试在 `unit-12.log` 与 `unit-12-cli.log`。

补充资料见 [原始集成设计](v41-a5-moonep-prefill-design.md)、[调优指南](v41-a5-moonep-tuning.md) 和 [逐轮优化记录](v41-a5-ep4-optimization-log.md)。记录一次新实验时保留改动与源码版本、A/B 配置、正确性、原始输出、量化结果和保留或回退决定。
