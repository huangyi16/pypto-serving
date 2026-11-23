# DeepSeek V4.1 A5 EP4 optimization log

## Fixed-configuration MoonEP acceptance

The current objective measures only changes inside the MoonEP implementation, with the original native reference held fixed. Both arms use 80 rows per round, the same empty-tail policy, one communication slot, and serial round reuse. Keep expert kernel scheduling, Decode, attention, shared expert, request scheduling, and weight loading fixed. Only warm Serving Prefill P50 counts toward the 30% target, using `1 - MoonEP P50 / native P50`.

Prefill 62 has two corroborating same-suite measurements. Its warm P50 falls 2.22% and 1.83% against the respective native arms, with improvements in all 12 matched requests in both suites. See the [matched Serving control audit](#matched-serving-control-audit) for the full attribution. The 80-row one-slot [Sweep 84](#sweep-84--compare-direct-expert-operand-stores-through-16384-tokens) uses Prefill 62 and covers six lengths through 16,384 tokens.

Prefill 23 established the earlier eligible reference. Its single-source dispatch suppresses empty activation pushes, and its fused route descriptor carries the exact FP32 weight in the existing return metadata. The scale-pack task materializes the weight for the unchanged expert input. Prefill 23 retains Prefill 16's one-slot round ordering and the same 80-row expert partition. The standard 12 matched requests include four requests each at 1,024, 3,070 and 6,142 input tokens.

Recomputed the following from the archived `http.json` files for `p16-20261121T045325Z` and `p23dr-20261121T074020Z`. The three arms exited 0. Their recorded commands match after normalizing port and backend selection. All arms generated 13 tokens and stopped at EOS on the same request indices and input lengths. The long answer and both eight-request waves passed in each arm. Prefill 23 uses the validated Prefill 16 native reference from the earlier suite.

| Arm | Warm Prefill P50 (s) | Warm Prefill P95 (s) | P50 reduction vs native |
|---|---:|---:|---:|
| Prefill 16 native | 2.237811 | 4.464890 | — |
| Prefill 16 compact, one slot | 2.236836 | 4.539110 | 0.04% |
| Prefill 23 compact, one slot | 2.167525 | 4.364125 | 3.14% |

The 30% requirement is 1.566468 s under this reference. Prefill 23 is 601.057 ms above that threshold. Its P50 fell 3.10% from Prefill 16 compact, and 11 of 12 matched requests improved over native. Prefill 12 combines MoonEP with a width increase. Prefill 20 introduces two communication slots and inter-round overlap. Prefill 34 inherits those changes and modifies routed-expert scheduling. Prefills 44 and 46 explicitly target inter-round overlap. These historical results are ineligible under the current configuration.

The audit stopped at the verified result below the threshold. After the goal was resumed, Prefill 47 started from the eligible Prefill 23 source and retained this acceptance configuration.

## Acceptance

- Reduce the repeatable process-start to HTTP `/health` time for the real checkpoint. Track checkpoint staging, kernel preparation, device initialization, and first-request communicator setup separately.
- Measure standalone MoonEP against native with the original Serving default of 80 MoE rows per round in both arms. The target is at least a 30% reduction in median warm Prefill time, calculated as `1 - candidate_seconds / native_seconds`. Hold empty-round trimming, model, prompts, topology, scheduler, warmup, and runtime settings fixed. Report tail latency and require the correctness checks to pass. The initial EP4 comparison and the later 256-row experiments record the wider configurations.
- Keep raw samples and logs under the isolated server worktree at `/home/hy/hw-native-sys/.worktrees/v41-a5-moonep-prefill/verification/ep4`. Record the changed code, expected mechanism, measured result and next decision in each iteration.

## Startup 00 · Existing EP4 checkpoint path

Server `rack06os7`, physical cards 4–7, Ascend 950DT, CANN 9.2.0, PTOAS 0.72, TP1/DP4/EP4. The host launch retains the 12-CPU-equivalent quota and 384 GiB memory limit. The checkpoint is `/mnt/weight/DeepSeek-V4.1-Flash-1017`. Serving base `ffab1ab07e4f926c112ce416589892df9dca2556` and lib base `3dc359fe57419506b42a19989ff072a7898a50ed` both carry the MoonEP changes in the isolated worktrees.

| Case | Health startup | Checkpoint staging | Resident upload and allocation | Result |
| --- | ---: | ---: | ---: | --- |
| MoonEP compact, DSpark off, 1,024-token prefill budget | 1,700.303 s | 1,386.30 s | 197.51 s | Passed |
| MoonEP compact, DSpark on, 1,024-token prefill budget | 2,294.475 s | 1,649.96 s | 443.05 s | First request failed during DSpark decode |

The weight bank was 295.3 GiB. Peak checkpoint-stage RSS was 319.1 GiB. Copies into the shared bank took 6.93 s within staging. The HTTP test returned the expected answer `323` for a 3,070-token prompt, then passed both waves of eight simultaneous requests. The first request includes lazy HCCL setup. Logs are `host-retry-compact-k0.log` and `host-retry-compact-k0-pytest/test_v41_dp4_http_answer_k0_0/server.log`.

The hardware MoE golden fixtures passed for `native`, `fixed`, `compact` and `online`, with 384 routed experts and per-rank counts `81,80,79,0`. Their elapsed times include initialization. The DSpark compatibility case reached health, then all four ranks returned `MallocPhysical -> 207001` for the additional decode communication window. CANN 9.2.0 defines this code as device memory allocation failure. The log is `host-retry-compact-k5.log`. The queued online HTTP case was deferred to prioritize startup optimization.

## Startup 01 · Reuse an EP4 prepacked weight bank

Checkpoint staging accounts for 81.5% of the measured health startup. Serving already maps a fingerprinted V4.1 sidecar and uploads from the mapping. Add a producer for the existing slot-major v2 format that uses the same checkpoint staging implementation, then streams the packed storage to a temporary file and atomically publishes it. The writer preserves float8 payloads as bytes with dtype metadata and avoids a second model-sized serialization buffer.

```bash
python -m pypto_serving.tools.prepack_deepseek_v41 "$PYPTO_DSV41_MODEL_DIR" \
  --ranks 4 --output "$PWD/sidecars/DeepSeek-V4.1-Flash-1017-ep4.safetensors"
export PYPTO_V41_PREPACKED_WEIGHTS="$PWD/sidecars/DeepSeek-V4.1-Flash-1017-ep4.safetensors"
```

Pack once on server-local storage. Run the same compact k0 HTTP test and verify that the reader maps the sidecar, the answers and concurrency checks still pass, and health startup decreases. Record file size, generation time, page-cache residency and both first-use and repeat-start times. Prepack is CPU-only and runs after the in-flight DSpark test. Logs are `startup-01-pack.log` and the associated startup cases under the same verification directory.

The sidecar round-trip and routed-weight budget tests passed on the server's existing Python environment. The round-trip compared bytes, shapes and dtypes through the production reader and confirmed that an EP4 sidecar is rejected for EP8. The hardware checkpoint pack completed in 1,318.4 s, publishing a 317,105,642,208-byte (295.33 GiB) file on the server's local ext4 volume. The first and repeat startup tests are `startup-01-first` and `startup-01-repeat`.

The startup comparison keeps the original JIT policy. The host sets `PYPTO_CACHE_DIR` but leaves `PYPTO_CACHE` unset, and this PyPTO build defaults to a disabled cache. Cache enablement can be measured independently after the sidecar comparison.

| Sidecar case | Health startup | Initial sidecar cache residency | Resident upload and allocation | HTTP regression |
| --- | ---: | ---: | ---: | --- |
| First use | 334.124 s | 30.1% | 216.73 s | Passed |
| Repeat start | 308.063 s | 95.2% | 185.77 s | Passed |

First use reduced the measured health startup by 80.35%, and the repeat start reduced it by 81.88%. The production loader mapped the 295.3 GiB bank both times. Each run returned `323` with 3,070 input tokens and 13 output tokens and passed both eight-request waves. The full pytest runs took 532.73 s and 509.77 s including HTTP regression and shutdown.

The first request has a separate cold cost. In the host's Simpler revision `355ce3015e3e344d1fc3a61c206ee01823ce292c`, A5 IPC domain initialization enables all peers, then polls their P2P status in sequence with up to 30 seconds per peer. The compact k0 log records two domains with all three peers unconfirmed under the current device mapping. Keep this time outside health startup and warm prefill measurements. Preserve the original runtime during the sidecar comparison.

## Prefill 00 · Native and MoonEP compact reference measurements

The remote `benchmark-00-suite.sh` started the native case at 01:02:38 on November 21, server time, after both sidecar HTTP regressions passed. The native arm uses pristine archives of the locked Serving and lib revisions. The compact arm uses the paired MoonEP worktrees. Both use the same EP4 sidecar, physical cards 4–7, a 1,024-token Prefill budget, DSpark off, prefix caching off, and enabled PyPTO JIT caching under the isolated verification directory.

The fixed synthetic workload has 12 distinct prompts using nominal 1,024, 3,072 and 6,144-token lengths, with four repetitions per length and a maximum of 16 new tokens per request. Each arm runs a warm-up request before measurements. The existing benchmark client saves unprofiled HTTP metrics and a separate e2e-profile run. Use the `WorkerProcess.batch_prefill` spans to sum chunk times per request and retain the unprofiled TTFT distribution as the external check. This first comparison uses single-request arrivals. Mixed-rank concurrent load and separate real-route replay are additional held-out checks.

The launcher, captured commands, shared workload JSON and raw metrics are in the server verification directory under `benchmark-00*`. Run `python benchmark-00-report.py benchmark-00-native benchmark-00-compact` there to join worker span durations with prompt token counts and verify matched chunk counts before computing the comparison.

The native run completed with zero errors in both sets of 12 requests. The profiled worker prefill P50/P95 were 3.016/5.685 s. Unprofiled TTFT P50/P95 were 2.538/4.777 s. The profiled TTFT P50 was 4.032 s, so keep these observations separate from the unprofiled request results. The stage trace confirmed one, three and six chunks for 1,024, 3,070 and 6,142 input tokens respectively. The compact case started at 01:12:43 server time.

The compact run also completed both sets of requests with zero errors. The matching prompt lengths and chunk counts were verified by `benchmark-00-report.py`. Raw paired output is `benchmark-00-comparison.json`.

| Actual prompt tokens | Native prefill P50 | Compact prefill P50 | Change |
| --- | ---: | ---: | ---: |
| 1,024 | 1.059 s | 0.789 s | 25.5% lower |
| 3,070 | 3.016 s | 2.269 s | 24.8% lower |
| 6,142 | 5.495 s | 4.730 s | 13.9% lower |

Across all 12 requests, profiled prefill P50 was 24.8% lower and P95 was 9.2% lower. Unprofiled TTFT P50 was 4.8% lower (2.538 to 2.416 s), while P95 increased 3.2% (4.777 to 4.929 s). An unprofiled matched repeat is needed to establish how much stage time is saved under the normal runtime. The 30% goal remains open.

## Prefill 01 · Reverse order and isolate empty-round trimming

Measure compact, locked native and native with empty-round trimming (N1). Use the same 12 requests in the same order. Query the existing `request_prefill_time_seconds` histogram around each completed serial request with profiling inactive. The change in its sum is the request's server-side duration from first scheduling to first token when the count increases by exactly one. Keep client TTFT and this scheduler-level measurement alongside Prefill 00's worker-stage trace. The three-arm comparison tests whether empty-round trimming explains any improvement before changing transport kernels.

The compact case started at 01:25:21 server time under the same card lock. The raw per-request results are in `benchmark-01-<arm>.log`. The report script `benchmark-01-report.py` regenerates paired statistics from those records and checks sample order and prompt lengths.

Compact completed all 12 requests with an unprofiled server Prefill P50/P95 of 2.297/6.341 s and TTFT P50/P95 of 2.379/6.443 s. The first 6,142-token sample took 6.341 s in Prefill. The remaining three took 4.548–4.574 s. Preserve the slow sample for tail comparisons. Native began at 01:34:35 server time.

The reversed-order native case completed all 12 requests. Native unprofiled Prefill P50/P95 were 2.223/5.406 s and TTFT P50/P95 were 2.302/5.491 s. Compact was 3.3% slower at Prefill P50, with increases of 4.6%, 3.3%, and 0.9% across the three lengths. This run did not reproduce the first profiled comparison's improvement. N1 started at 01:43:42 server time.

All three arms completed, with identical token counts and 13 generated tokens per request. `benchmark-01-comparison.json` preserves the 36 request results.

| Arm | Server Prefill P50 / P95 | Client TTFT P50 / P95 |
| --- | ---: | ---: |
| Locked native | 2.223 / 5.406 s | 2.302 / 5.491 s |
| Native with empty-round trimming | 2.180 / 4.933 s | 2.261 / 5.021 s |
| MoonEP compact | 2.297 / 6.341 s | 2.379 / 6.443 s |

Empty-round trimming improved the median by 1.9% overall. This workload largely fills its 1,024-token chunks, and the 1,024-token group itself did not improve. Compact was 5.3% slower than the trimmed native arm. Use the unprofiled native P50 as the reference for subsequent same-workload experiments.

## Prefill 02 · Batch route metadata stores

In the original compact path, each routed row issues separate 32-byte remote stores for weight and inverse-route metadata. A fixed owner assigns one contiguous row interval to each source and expert. Build the metadata for that interval in the existing push task and issue two stores for the interval. Bound each store by its actual valid row count to preserve the next source's rows. Activation and scale transfers, output ordering, and the online replica path retain their current logic.

The candidate `prefill_moonep.py` has SHA-256 `c00f164696d32f45c67f4f50e37e45c06ada65a8978483b3cb56c593721c20bc`. The previous compact file has SHA-256 `267475c52f6852269dd785e22aa61f17a390f8de132c425590a10c867eb0ee9d`. The candidate is staged in `experiments/prefill-02/pypto-lib` so it does not change the ongoing Prefill 01 jobs.

The `moe-02-suite.sh` job waited for the Prefill 01 group to release the cards and started the control at 01:52:59 server time. It replays the same hot-route golden fixture with ranks `81,80,79,0`, 20 timed rounds and five warmup rounds per arm. The compact control reuses its existing compiler output and lets the runtime rebuild missing binaries. The candidate uses a new compile. Regressions in golden outputs or device wall time reject the change. Use a full Serving repeat to confirm any kernel gain with real checkpoint routes.

The control passed all three output comparisons. Its fastest-rank device median was 2,112.4 μs, and the per-round max across ranks had a median of 4,000.6 μs. Keep both figures because this fixture intentionally skews routing. The candidate's first compile rejected a branch-assigned tile extent as a runtime variable before device execution. Logs are `moe-02-baseline.log` and `moe-02-candidate.log`.

The second candidate specializes the metadata tile row count as an explicit `pl.constexpr` argument. Its `prefill_moe.py` and `prefill_moonep.py` SHA-256 hashes are `e3ea395eb5f7941bcc751642ead6429c6aa60e040e6a70aa7a39dd959b28b012` and `8a54a232041518b3870f2e41a340c2e0449ee6cfef8c65525b8bebcdc2dd251c`. The isolated source is `experiments/prefill-02b/pypto-lib`. Its hardware run is `moe-02b-candidate.log`.

The revised candidate passed `next_pre_mix`, `x_mixed`, and `x_next` against the frozen EP4 golden. With 20 timed rounds and five warmup rounds, the fastest-rank median was 2,094.7 μs, 0.8% below the compact control. The per-round max across ranks had a median of 3,690.4 μs, 7.8% below the control. The latter includes rank start skew. Run the same unprofiled Serving workload with this source to check transfer of the improvement.

The full Serving repeat is `benchmark-02-compact`. It selects the isolated second candidate and collects the same server and client metrics as Prefill 01.

All 12 requests passed with matching input and output token counts. The full Serving Prefill P50/P95 were 2.858/7.349 s and TTFT P50/P95 were 2.937/7.434 s. Prefill P50 regressed 24.4% relative to the original compact path and 28.5% relative to locked native. The report is `benchmark-02-comparison.json`. Reverted the batched metadata change in the development source based on the real-checkpoint regression. The isolated candidate and logs remain available.

## Prefill 03 · Test a larger MoE slab in the existing fixture

The fixed 80-row MoE extent yields 13 rounds per full 1,024-token Serving chunk. Each round performs histogram publication, row planning, payload exchange and release. A larger extent can reduce that repeated work. Test 128 rows using the existing standalone `--tokens` option, the Prefill 02 candidate, and the same 128-row fixture and golden outputs. Rank counts `81,80,79,0` require two rounds at 80 rows and one at 128. Keep the test at 20 timed rounds and five warmup rounds. This tests a round-boundary case, so a hardware gain here requires confirmation at the Serving chunk shape. The job `moe-03-after-serving.sh` waits for the current Serving repeat and writes to `moe-03-t128.log`.

The 128-row candidate passed the same three golden comparisons. Its fastest-rank median fell from 2,094.7 to 1,219.9 μs and the per-round max across ranks fell from 3,690.4 to 2,886.7 μs, reductions of 41.8% and 21.8% respectively. The sweep used the isolated metadata-batching source for both widths.

## Prefill 04 · Match the Serving chunk capacity

Compare 80 and 128 MoE rows at a 1,024-token chunk with active rank counts `1024,0,0,0` and the existing spread-route generator. Use the original compact transport for both runs, retaining 20 timed rounds and five warmup rounds. The 80-row case saves the golden input and output in the isolated `moe-04` work directory. The 128-row case replays those files. This tests whether fewer rounds help under the long-request shape after reverting metadata batching. The server launcher is `moe-04-suite.sh`. If the device improvement holds, the next Serving experiment will specialize the Prefill width independently of the Decode extent.

Both widths passed all golden comparisons. From 80 to 128 rows, the fastest-rank median fell from 13.481 to 8.761 ms, a 35.0% reduction. The median of the per-round max across ranks fell from 15.618 to 12.053 ms, a 22.8% reduction. The 128-row run had a 134.683 ms maximum sample and a higher mean. Retain the outlier in `moe-04-t128.log`.

## Prefill 05 · Sweep 160 and 192 rows

Re-run the same 1,024-token golden and measurement with 160 and 192 MoE rows. These widths take seven and six rounds. The 160-row case also avoids padding in the shared expert's 80-row W13 tiles. The `moe-05-suite.sh` group began at 02:22:43 server time. Keep raw time and validation results in `moe-05-t160.log` and `moe-05-t192.log`.

The 160-row case passed all golden comparisons. Its fastest-rank median was 8.068 ms, 40.1% below the 80-row control. The median of the max across ranks was 11.088 ms, 29.0% below the control, with a 11.504 ms maximum sample. The 192-row case started at 02:25:22 server time.

The 192-row case also passed all golden comparisons. Its fastest-rank median was 7.306 ms, 45.8% below the 80-row control. The median of the max across ranks was 9.616 ms, 38.4% below the control, with a 10.417 ms maximum sample. Select 192 rows for the next matched Serving test.

## Prefill 06 · Replay a wider-width artifact through Serving

The current Serving compile imports shared MoE kernels for Decode before Prefill and they capture the 80-row Decode extent. To test a wider Prefill using the existing library, compile its exact Serving signature in a fresh process with the selected width, and load the artifact in an isolated Serving source snapshot. Decode continues to use the ordinary path. The experiment checks the layout, EP, capacity and argument names before binding the artifact and sets the scheduler's Prefill round count to the artifact's explicit width.

The experimental runner `experiments/prefill-width/pypto-serving/pypto_serving/model/deepseek_v41/npu_runner.py` has SHA-256 `ddc27917b21fa4314476146eb6ab3f8eb37bce5a34277821f6e7fa8bb21193db`. It is staged with `prefill-width-build.py` and `benchmark-06.sh`. The 80-row replay control started after the sweep. Its log is `benchmark-06-t80.log`. The 192-row case is queued behind the control and logs to `benchmark-06-t192.log`. A repeatable win in Serving is required before adapting the shared model source to support separate widths.

The 80-row replay control passed 12 requests with matching 1,024, 3,070 and 6,142-token inputs. Prefill P50/P95 were 2.334/4.686 s. TTFT P50/P95 were 2.414/4.771 s. The 192-row job started next and will check the 3,070-token answer after timing.

The 192-row case returned the expected `323` answer and passed all 12 measured requests. Prefill P50/P95 were 3.075/6.436 s and TTFT P50/P95 were 3.155/6.527 s. The Prefill median regressed 31.7% relative to the 80-row replay control and 38.3% relative to locked native. Each arm generated the same 13 output tokens per request. The report is `benchmark-06-comparison.json`. Retain 80 rows in the worktree. Wider Prefill remains an isolated experiment.

## Prefill 07 · Capture real model routing

The spread-route fixture improved with wider slabs while the real-checkpoint HTTP workload regressed. Capture original compact routing at 80 rows from the real checkpoint with the normal EP4 HTTP regression. The routing capture is diagnostic. Inspect route distribution, experts and active rows per MoE round. Then compare A5 critical paths on matched real routes to identify the limiting computation or transfer.

The server scope started at 02:48:14 server time. It runs `diag-07.sh`, with `diag-07-compact.log` and `diag-07-routes/` in the isolated verification directory. The replay queue waits for the HTTP test to pass, summarizes the first chunk at each tested MoE width, and replays the same captured gate decisions in the checkpoint-free forward at widths 80 and 192. It uses a level-1 chip swimlane for diagnosis. Per-layer round and owner counts are saved in `diag-07-route-pattern.json`.

The first captured chunk has counts `1024,0,0,0` in both stages. At 80 rows it contains 520 layer-rounds and a sum of 8,931 busiest-owner 64-row tiles. Repartitioning the same routes at 192 rows gives 240 layer-rounds and 5,332 busiest-owner tiles. The sum of busiest-owner routed rows is nearly flat (91,609 versus 91,285). These routing counts alone do not explain the wider Serving regression.

The capture passed the 3,070-token answer and both concurrent waves, saving seven route files. The first level-1 replay started at 02:56:40 server time.

The original 80-row replay compiled and validated the trace inputs, then failed while uploading a 1,323,827,200-byte device tensor with ACL error 207001. The standalone fixture reserved 16 GiB of ring heap, while the corresponding Serving case uses 10 GiB. The `diag-07b-replay-suite.sh` rerun uses an isolated lib copy with Serving's ring-heap and dependency-pool sizes, keeping the captured routes and both MoE widths.

Both matched replays passed the fixture's finite-output and zero-weight checks and emitted level-1 task traces for all four ranks. The longest per-rank AICore time span fell from 692.160 to 448.618 ms at 192 rows, a 35.2% reduction. Routed expert W13/W2 were the largest named groups on 80 rows. Rank 0's total routed-expert core busy time fell from 10,443 to 4,814 core-ms, while shared-expert core busy time grew from 318 to 440 core-ms. Busy times sum cores and can overlap. This replay validates the routed time benefit with matched routes, while the real-checkpoint Serving result still needs direct attribution. The paired report `diag-07b-width-comparison.json` uses merged timing events to recover callable names via dependency records.

## Prefill 08 · Capture the real Serving device path

Use the validated 80-row and 192-row Prefill artifacts and the same real 1,024-token request with one output token. Capture level-1 chip timing and worker profiling in an isolated copy of the replay adapter. Compare the per-rank device spans with the host Prefill window for the corresponding response. This establishes whether the regression is present at the device dispatch while loading the real weights with ordinary Serving residency.

The `diag-08-suite.sh` job started at 03:16:37 server time. The replay adapter and copied compiler artifacts are under `experiments/diag-08` and `diag-08-artifacts`. Both arms use the same sidecar and cap their response at one token so the trace targets Prefill. Artifact capture and conversion add host work, so use device spans and task durations for attribution and the unprofiled Prefill 06 numbers for service throughput.

Both real-weight arms returned one token from the 1,024-token request. The worker trace confirmed `1024,0,0,0` rows and 13 versus six MoE rounds. The longest per-rank AICore span fell from 653.387 to 408.151 ms, a 37.5% reduction. Rank 0 routed-expert busy time fell from 11,051 to 5,013 core-ms. The two-pass tracing and conversion dominated the host time, so this diagnostic does not overturn the unprofiled regression. Raw output is `diag-08-comparison.json`.

### 80-row transport path reconstruction

Reanalyze the real 1,024-token trace with `simpler_setup.tools.critical_path` and the original `chip_swimlane_records.json`, `deps.json`, and `name_map.json`. The trace contains 520 layer-rounds and identifies every timed task on the source rank and one receiving rank. The observed rank-local path uses dependencies and same-core predecessors. The transport group includes histogram, row planning, activation send and packing, return scatter and reduction, and synchronization. Its task time excludes overlaps on the observed path. Gaps are measured before grouped tasks and can include scheduling, peer work, or wait time.

| Rank | AICore span | Transport task time on path | Gaps before transport tasks | Routed expert task time on path |
| --- | ---: | ---: | ---: | ---: |
| 0, source | 648.985 ms | 106.328 ms | 110.331 ms | 136.405 ms |
| 1, receiver | 653.385 ms | 62.267 ms | 209.582 ms | 136.298 ms |

The trace labels both payload and return synchronization as `combine_notify` and `combine_wait`. Separate them by round order and check the local push-to-pack and scatter-to-reduce sequence for every round. For the rank pair, the reciprocal histogram, payload and return dependencies bound the rank 1 clock adjustment between 26.231 and 26.251 ms. On rank 1, histogram-wait start through row-plan completion spans 120.590 ms in aggregate, with a median round of 85.9 µs and P95 of 847.6 µs. With the midpoint clock adjustment, rank 1 scale-pack starts a median 12.0 µs after rank 0 payload notification starts. The clock bounds put that median between 2.4 and 21.7 µs. This supports testing synchronization and preparation on the dispatch and return paths. The rank-local gaps include work on peers and cannot be added across ranks or treated as removable. This is the historical protocol before Prefill 14–18. See `diag-08-path-summary.json`, `diag-08-epochs-summary.json`, and their analysis scripts in `verification/ep4`.

## Prefill 09 · Compare ordinary dispatch timing

Rerun the same real-checkpoint width comparison using runtime timing logs without chip swimlane or host profiling. Keep the fixed prompt set and cap diagnostic responses at one output token so each sample measures only the Prefill path and its first token. Join runtime `STRACE` timing with server Prefill metrics per request. This tests whether the wider-device gain persists under ordinary persistent execution.

The `diag-09-suite.sh` job runs the 80- and 192-row controls with `PYPTO_RUNTIME_LOG=timing` and the same stored compiler manifests as Prefill 06. Per-request metrics and worker logs are under `diag-09-t80` and `diag-09-t192`.

The 80-row arm passed all 12 requests. The parser checked exactly one, three, or six dispatches on each rank for each prompt length, and reports a 2.320 s median server Prefill, 1.821 s median device effective duration, and 1.857 s median chip-worker host duration. The 192-row arm started at 03:39:57 server time.

The 192-row arm passed all 12 requests and reports a 1.650 s median server Prefill, 1.170 s median device effective duration and 1.204 s median chip-worker host duration. Compared with 80 rows, server Prefill improved 28.9% and device effective duration improved 35.8%. Both arms limited output to one token. Report `diag-09-comparison.json` preserves the 24 joined HTTP and runtime samples.

## Prefill 10 · Restore ordinary Decode interleaving

Prefill 06 allowed up to 16 output tokens and measured a wider-width regression. Prefill 09 stopped after the first token and measured a gain. Repeat with the same timing logger and restore the 16-token cap. Join only the initial Prefill dispatches to each request's Prefill metric and record subsequent Decode dispatches separately. This isolates whether interleaving with Decode accounts for the difference.

The `diag-10-suite.sh` group started at 03:49:09 server time. Its two arms reuse the same compiled manifests and workload. The parser checks that Prefill and Decode have distinct callable identities on each rank and preserves the exact server metrics and raw runtime logs.

The 80-row arm passed all 12 requests. Its median Prefill was 2.303 s and its median device effective time for Prefill dispatches was 1.822 s, close to the one-output control. The 192-row arm started at 03:57:10 server time.

The 192-row arm passed all 12 requests. Server Prefill P50 was 1.571 s and TTFT P50 was 1.651 s. The Prefill device-effective P50 was 1.168 s. Relative to the 80-row arm, the measured Prefill reduction was 31.8% and the device-effective reduction was 35.9%. The original Prefill 06 regression did not reproduce when runtime timing was enabled, even with Decode interleaved. Relative to the previously measured 2.223 s locked-native median, 1.571 s is a 29.3% reduction, with a different logging configuration. The results and individual dispatch counts are in `diag-10-comparison.json`.

## Prefill 11 · Extend the width sweep and repeat the ordinary Serving comparison

Test a 256-row width on the same frozen 1,024-token EP4 fixture to see whether four rounds can improve on 192 rows. The job `moe-11-t256` keeps the original compact implementation and the same device timing setup as Prefill 04 and 05. Repeat 192-row and locked-native Serving with the original 16-token workload and ordinary logging to verify the unprofiled objective against a fresh native control.

The 256-row run completed in 142.69 s and passed all three frozen golden comparisons. Across 20 timed rounds, the median of the per-round max across four ranks was 9.025 ms, down 6.1% from 192 rows and 42.2% from 80 rows. The fastest-rank median was 5.851 ms. The maximum sample was 11.176 ms. Raw samples are in `moe-11-t256.log`.

The `benchmark-11-suite.sh` job started at 10:01:28 server time and runs 192 rows, locked native and then 256 rows on the existing isolated physical cards 4–7. It uses the same fixed prompts, token cap, sidecar and scheduler options from Prefill 01, unsets runtime timing, and checks the expected 3,070-token answer after each timed arm. The 256-row arm builds the complete Serving entry before launching the server.

The 192-row ordinary run passed all 12 requests with 13 output tokens per request and returned the expected answer. Prefill P50 was 1.676 s and TTFT P50 was 1.756 s. Prefill medians for 1,024, 3,070 and 6,142 tokens were 0.584, 1.676 and 3.352 s. Its first warmup took 184.95 s with 30-second HCCL peer status warnings during lazy initialization. This warmup is outside the reported latency samples. The native arm started at 10:09:27 server time.

The fresh native arm passed the same checks and reported Prefill P50/P95 of 2.430/4.663 s and TTFT P50/P95 of 2.509/4.750 s. At 192 rows the Prefill P50 reduction is 31.0% against this matched control. Prefill P95 for that arm is 4.998 s, including a slow sample in the 6,142-token group. Against the original 2.223 s native median the same 1.676 s measurement is a 24.6% reduction. Keep both comparisons. The verified per-request report is `benchmark-11-native-t192-comparison.json`. The 256-row arm started at 10:18:31 server time.

All 12 requests and the answer check passed at 256 rows. Prefill P50/P95 were 1.603/3.104 s and TTFT P50/P95 were 1.690/3.190 s. Compared with the same-suite native arm, Prefill P50 fell 34.0% and P95 fell 33.4%. Compared with the earlier 2.223 s native median, the reduction was 27.9%. Per-prompt medians were 0.591, 1.603 and 3.082 s against the same-suite native values of 0.920, 2.430 and 4.651 s. The complete 36-request report is `benchmark-11-comparison.json`.

## Prefill 12 · Specialize Prefill width in the ordinary compiler

Give Prefill its own MoE row extent and pass it to the shared gate, expert projection and routed compute as compile-time values. Preserve the existing Decode ABI and use the normal in-process compiler and JIT cache for both entries. Validate the new source path against the separately compiled artifact, including the host round count and exact compiled dimensions.

The first compile in `experiments/prefill-12` failed because a called JIT function's `constexpr` defaults are not used to bind a dependency. The second run used explicit values at every shared call site and reached the W13 padding function. It failed because the dependency resolver cannot bind locally computed constants as parameters. Inline the padding where the W13 shape is already available and keep the incoming width explicit. Logs are `compile-12-t256.log` and `compile-12b-t256.log`.

The inline W13 branch then lost the tensor metadata required by the external projection, whose direct and padded input shapes differ. Use the existing lib pattern of building a JIT function with statically captured dimensions, and select W13 padding at construction time. The next compile passed shared-expert inference and rejected a Python conditional expression in the gate DSL. Apply the same construction pattern to the gate. Logs are `compile-12c-t256.log` and `compile-12d-t256.log`. The static gate and expert source is staged for `compile-12e-t256`.

`compile-12e-t256` passed both standard compiler calls in one process. The assertions confirmed an 80-row Decode, a 256-row Prefill with four rounds per 1,024 tokens, and 6,144 maximum routed rows per round across EP4. The normal CLI check `benchmark-12-t256` is queued behind the replay-artifact test and uses the same ordinary workload and locked native reference. It sets the independent width with `PYPTO_V41_PREFILL_MOE_TOKENS=256`.

The integrated and replay artifacts have identical serialized distributed ABIs and the same 214 generated kernel source paths. The 80-row native Prefill and Decode compile also passed in one process, preserving 13 Prefill rounds at a 1,024-token capacity. Nine host tests passed in `unit-12.log`, covering Prefill round metadata, sidecar I/O and route replay.

The first CLI run failed before opening a device because the engine configuration probe still asked for native bindings, although the CLI requested MoonEP. Pass the same selected backend to the probe and the worker executor. The corrected normal CLI run `benchmark-12b-t256` started at 10:29:20 server time and reached worker initialization. It runs the 12 measured prompts, the long answer check, and both eight-request waves through the same server.

The corrected run exited successfully. Health startup was 290.100 s with the prepacked bank. The first warmup took 184.92 s, including host IPC setup. All 12 timed requests produced 13 output tokens, the 3,070-token answer was `323`, and both concurrent waves passed eight arithmetic checks. The CLI suite passed 39 additional unit tests.

| Arm | Server Prefill P50 / P95 | Client TTFT P50 / P95 |
| --- | ---: | ---: |
| Same-session locked native | 2.430 / 4.663 s | 2.509 / 4.750 s |
| Integrated MoonEP compact, 256 rows | 1.665 / 3.252 s | 1.744 / 3.337 s |

The combined 256-row compact configuration reduced median warm Prefill by 31.47% and P95 by 30.26% against the 80-row locked native control from Prefill 11. The median reductions by prompt length were 31.08%, 31.47% and 31.42% at 1,024, 3,070 and 6,142 tokens. Against the earlier native run's 2.223 s median, the reduction was 25.10%. Both arms used the same checkpoint, sidecar, scheduler and unprofiled runtime settings with DSpark off. The combined configuration meets the 30% latency threshold against the same-session baseline. The 256-row setting is opt-in for EP4 compact. The concurrent checks provide correctness evidence. See `benchmark-12-comparison.json` for the complete timed samples.

The measured difference combines the independent Prefill width specialization, its expert-kernel settings, round trimming, and the compact transport. The earlier 80-row comparison in Prefill 01 measured 2.223 s for locked native and 2.297 s for compact. A 256-row native Serving run at the same source revision and with matching round trimming is missing. The compact receive layout has a smaller row footprint by construction. The standalone latency contribution of that layout and its effect on the feasible native width remain unmeasured.

## Prefill 13 · Skip replica placement in fixed-owner dispatch

**Acceptance for MoonEP alone.** Reduce median warm Serving Prefill by at least 30% relative to native with the same Prefill width and empty-round trimming. Keep the checkpoint, sidecar, scheduling, EP4 device allocation, workload, warmup, and runtime logging settings identical. Report P95 and TTFT, and require the existing long-answer and concurrent correctness checks. The Prefill 12 comparison remains the result for the combined configuration.

**Candidate.** The default compact path and the fixed-stride diagnostic both use the original expert owners. They previously submitted `moonep_placement` after the histogram wait to write a no-move record, then read that record in row planning and payload push. The candidate compiles those layouts with `moonep_row_plan` depending directly on histogram completion. Replica-enabled layouts still take the placement path. Row offsets, payloads, and round release use the same protocol. The predicted mechanism is one fewer task on the path to the first payload push per layer and round. Check the generated task graph for that change before attributing any time difference to it.

| Arm | Serving | pypto-lib | Width and trimming |
| --- | --- | --- | --- |
| Native control | `6f1ffca` | `7785e97b`, `native` | 80 rows, empty tails trimmed |
| Compact control | `6f1ffca` | `7785e97b`, `compact` | 80 rows, empty tails trimmed |
| Compact candidate | `6f1ffca` | `7785e97b` with the `prefill_moonep.py` change | 80 rows, empty tails trimmed |

The control `prefill_moonep.py` SHA-256 is `c39ba3ccd8b5e0b39715e64b2a1e586d1cb927ca4d8f052c4b2974b3392a7de7`. The 12 source files in the historical verification archive match the two control commits byte for byte. The first candidate passed local syntax and lint checks. The default local Python lacked the PyPTO binding, and the session was denied an SSH socket. The approval policy rejected escalation.

**Host code generation.** A local PyPTO build at `3100121698416e3b2d583564dabc7c731f2de186` and PTOAS 0.72 were available in the existing A5 toolchain artifacts. The first candidate (`e89cdb946b49c1c09b56c5fbdb91de39a06164815c12a6b9fa940893143a2c60`) failed to infer the `move` tensor metadata at the `copy_replica` call. Moving the tensor declarations outside the layout branch resolved type inference, and assigning the task ID from a uniquely named `pl.at` binding fixed an SSA scope error in the intermediate candidate. The revised source SHA-256 is `8a04678e7daaa3a6183ab4fdcc8a1e4884a281495aa9fafadfee5efb51f6ffca`.

The EP4 slab entry generated A5 code with `--compile-only --enable-dep-gen` at 80 rows, slab capacity 128, and counts `81,80,79,0`. Control compact and revised `fixed`, `compact`, and `online` passed. The control compact graph registered 40 kernels and set `moonep_placement` as the row-plan dependency. The revised compact and fixed graphs registered 39 kernels and depend directly on `moonep_histogram_wait`. The revised online graph still schedules placement and uses its task ID as the row-plan dependency. Raw output is in `compile-control-compact.log` and `compile-candidate-c-{fixed,compact,online}.log`. Seven Serving metadata and sidecar tests passed, as did two lib route-trace tests. These results cover host code generation and contracts. EP4 device execution, golden outputs, and latency measurements are pending.

The full Serving `compile_decode_fwd` and `compile_prefill_fwd` calls also generated code in the same process with TP1/DP4/EP4 and a 1,024-token Prefill budget. Decode stayed at 80 rows. All three 80-row arms generated 13 rounds. The compact control registered 147 kernels and the candidate registered 146. Both request the same 207,758,592-byte shared EP window per rank. Native and candidate compact also generated 256-row graphs with four rounds. At 256 rows the generated shared EP window is 717,739,776 bytes per rank for native and 241,145,088 bytes for compact. The FP8 hidden receive window accounts for 480.000 and 44.531 MiB of those respective allocations. These figures describe requested communication windows. Device allocation under the checkpoint's memory pressure and the latency effect remain unmeasured. Host logs are in `serving-{control-native-80,control-compact-80,compact-80,native-256,compact-256}.log`.

The candidate also generated the 80-row `preplaced` and `planned` graphs through the Serving compiler with route replay enabled. The logs are `serving-candidate-{preplaced,planned}-80.log`. This covers compilation through the replay placement branch. Replay correctness still requires device execution.

### EP4 run configuration

Stage the two control commits under `verification/ep4/experiments/prefill-13/{pypto-serving,pypto-lib-control}` and a copy of the lib control with the single candidate file under `verification/ep4/experiments/prefill-13/pypto-lib-candidate`. Verify the two file hashes above before running. Use the existing EP4 isolation and frozen golden first. Check `fixed`, `compact`, and `online` on the `81,80,79,0` fixture, then check `compact` on `1024,0,0,0` at 80 rows. Keep the golden and trace runs outside formal Serving timing.

The revised staging archive `prefill-13-source.tar.gz` has SHA-256 `f9716c44b9c3ef39c7accacf8a2b62d597fc6481f8da4f1290180490d1c6dc78`. It contains the three source snapshots under `prefill-13/` and the wrapper below as `prefill-13.sh`. The two staged lib trees differ only in `prefill_moonep.py`, and the wrapper passed `bash -n`. The original archive with SHA-256 `259e43ff960814c5dc9dd4b2b430b1c7d311a77d20d5ebfd950db48e81f64408` contains the candidate that failed host code generation.

Save this wrapper in the EP4 verification directory and invoke it with a unique suite label through `run-verified-cards.sh`. Run `native`, `compact-control`, and `compact-candidate` serially under that label.

```bash
#!/bin/bash
source "$(dirname "$0")/env.sh"
set -euo pipefail
cd "$here"
arm=$1
experiment="$here/experiments/prefill-13"
export PYPTO_V41_TEST_ARM="$2-$arm"
export PYPTO_V41_PREPACKED_WEIGHTS="$here/sidecars/DeepSeek-V4.1-Flash-1017-ep4.safetensors"
export PYPTO_CACHE=1
export PYPTO_V41_PREFILL_TRIM_EMPTY_ROUNDS=1
export PYPTO_V41_PREFILL_MOONEP_LAYOUT=compact
unset PYPTO_RUNTIME_LOG PYPTO_V41_PREFILL_MOE_TOKENS
unset PYPTO_V41_PREFILL_ROUTE_TRACE_DIR PYPTO_V41_PREFILL_ROUTE_REPLAY_DIR
case "$arm" in
    native) backend=native; lib=pypto-lib-control ;;
    compact-control) backend=compact; lib=pypto-lib-control ;;
    compact-candidate) backend=compact; lib=pypto-lib-candidate ;;
    *) exit 2 ;;
esac
export PYPTO_LIB_ROOT="$experiment/$lib"
export PYTHONPATH="$experiment/pypto-serving:$PYPTO_LIB_ROOT:$previous/pypto/python:$previous/pypto/runtime/python:$previous/pypto/runtime"
exec python "$here/benchmark-12b.py" "$backend"
```

Use the same fixed 12 requests already stored in `benchmark-00-workload.json`. Preserve each arm's `.exit`, `command.json`, `http.json`, server log, warmup, and correctness outputs. Before comparing, confirm 80 rows and matched round trimming from the compiled configuration, along with identical request order and token counts. Compare the candidate first to compact control for the effect of the task removal, then to native for the 30% criterion. Use `1 - candidate_P50 / native_P50` for the acceptance calculation. Retain the candidate for performance only if the golden passes and ordinary Serving shows a repeatable gain. If the trace instead places the remaining histogram wait on the critical path, the next candidate is to plan and publish compact offsets from the sole active source on single-source rounds.

## Prefill 14 · Compact dispatch from one active source

The first captured real 1,024-token chunk in Prefill 07 had counts `1024,0,0,0` in both stages. Prefill 13 still exchanges a complete histogram before sending hidden rows. In a chunk with exactly one active source, its local route counts already determine every aligned expert segment and its outgoing offsets. The experiment builds on the compiled Prefill 13 candidate and stays in an isolated source snapshot while device evidence is pending.

Serving supplies each rank with the same `moe_sole_source` value per stage, computed once from the chunk's existing per-rank counts. A rank ID selects the single-source compact path. `-1` selects the existing general exchange for mixed or empty chunks. A receiver only reads its own slice of the source's counts, and the source plans all destination offsets from its local counts. The source skips `moonep_histogram_wait`; other ranks wait only for its publication. Payload and return-window synchronization follow the existing protocol. The slab fixture derives the rank from its existing count vector and uses the same dispatch implementation.

At EP4 with 96 experts per rank, the histogram loops issue 48 inter-rank writes per round in the general path and three in the single-source path. The logical histogram size per write is 384 bytes. That gives 18,432 versus 1,152 logical metadata bytes per round, excluding local writes, notifications, and transport overhead. This is a protocol count from the source. The trace needs to establish a shorter gate-to-first-payload interval and lower complete MoE time. Run the mixed-source control to check the cost of the selection branch.

**Host validation.** The initial slab compile rejected a scalar index/count mismatch in the fixture. After matching the DSL index type, `compile-slab-compact-80b.log` passed A5 host code generation for EP4, 80 rows, capacity 1,024, and counts `1024,0,0,0`. The emitted task graph submits histogram wait only when the local rank differs from the sole source. Eight Serving metadata and sidecar tests passed, including a shared-metadata test that exercises single-source rank changes and resets through mixed and empty chunks. SSH continued to fail before authentication with `socket: Operation not permitted`. No EP4 device outputs or latency samples have been produced for Prefill 14.

The normal Serving compiler also generated the compact graphs at 80 and 256 rows with the new metadata in the full Prefill signature and Decode at 80 rows in the same process. The 80-row graph reads `moe_sole_source` from the stage metadata and contains the same conditional histogram wait as the slab graph. Full compile logs are `serving-compact-{80,256}.log`.

**Device comparison.** Use the same 80-row EP4 setup and trimming as Prefill 13. Reuse the frozen single-source golden and the `81,80,79,0` mixed-source fixture. Verify `next_pre_mix`, `x_mixed`, and `x_next`, including return traffic and repeated window reuse. Compare the candidate to Prefill 13 compact for the incremental effect, and to same-width native for the 30% Serving target. Keep diagnostic traces separate from the 12 timed requests, long answer check, and two concurrent waves. Candidate retention remains pending those results.

The isolated `prefill-14-source.tar.gz` has SHA-256 `86aff287c8a66eef55111429d47cf12f33b3a35c57c3e95867d94f103b4150be`. Stage its `prefill-14/` directory under EP4 `experiments/` and `prefill-14.sh` in the verification directory. The three wrapper arms share the same Serving snapshot. They select the original native lib, Prefill 13 compact lib, or Prefill 14 compact lib. Serving stages the new input only when the compiled lib exposes it. The candidate `prefill_moonep.py` hash is `67ec8ef4a39b063471bbc20df2f27a52a4b9761b9826fff77a15717dde19ff34`. The launcher passed `bash -n`, and the package comparison showed changes confined to the four Prefill lib files, Serving metadata staging, and its host test.

**EP4 execution.** SSH access was restored and the uploaded package and workload hashes matched. Suite `p14-20261121T041100Z` started on `rack06os7` at 04:11 UTC using `run-verified-cards.sh` on physical cards 4–7. It runs Prefill 13 and Prefill 14 with the frozen single-source and mixed-source goldens, then runs same-width native, Prefill 13 compact, and Prefill 14 compact Serving. The suite PID is 2060988. The single-source control passed the three frozen output comparisons in 231.36 s and the single-source candidate started at 04:15:03 UTC. See the per-arm logs and exit files under `verification/ep4`.

Both 80-row single-source arms exited 0 and passed `next_pre_mix`, `x_mixed`, and `x_next` on the frozen `1024,0,0,0` replay. Across 20 timed iterations after five warmups, the fastest-rank median fell from 13.493 to 12.944 ms (4.1%). The median of the per-iteration maximum across ranks rose from 15.025 to 18.152 ms (20.8%). Mean host union time, which includes dispatch overhead and rank start skew, fell from 48.820 to 35.461 ms. The three conventions give different signals, so the selection decision awaits the matched Serving run. The mixed-source control started at 04:19:03 UTC.

Both mixed-source arms exited 0 and passed the same three output comparisons on `81,80,79,0`. Across the same warmup and timed counts, the fastest-rank median rose from 2.124 to 2.138 ms (0.7%). The per-iteration maximum median rose from 3.557 to 3.926 ms (10.4%), and mean host union time rose from 6.414 to 6.762 ms (5.4%). The native 80-row Serving control started at 04:24:11 UTC after all four golden runs passed.

The native 80-row Serving control reached health in 298.046 s and exited 0. Warm Prefill P50/P95 were 2.160/4.325 s. Client TTFT P50/P95 were 2.240/4.414 s. All 12 requests had the expected input lengths and 13 generated tokens, and the 3,070-token answer and both eight-request waves passed. A 30% reduction against this control requires P50 at or below 1.512 s. Prefill 13 compact started at 04:33:34 UTC, followed by Prefill 14 compact in the suite.

The 80-row Prefill 13 compact control reached health in 290.054 s, and the Prefill 14 candidate in 288.077 s. All three Serving arms exited 0, each passing the same 12 requests, long answer, and both eight-request waves. The saved CLI commands match after normalizing the port and backend flag. Prefill width and empty-round trimming were held fixed. Full samples are in `p14-20261121T041100Z-comparison.json`.

| 80-row arm | Prefill P50 / P95 | Client TTFT P50 / P95 | Prefill P50 reduction vs native |
| --- | ---: | ---: | ---: |
| Native | 2.160 / 4.325 s | 2.240 / 4.414 s | 0.00% |
| Prefill 13 compact | 2.304 / 5.838 s | 2.384 / 5.925 s | -6.65% |
| Prefill 14 compact | 2.231 / 4.467 s | 2.312 / 4.565 s | -3.29% |

The single-source histogram change reduced Prefill P50 by 3.15% against Prefill 13 compact. At 1,024, 3,070, and 6,142 input tokens, the medians moved from 0.816/2.304/4.666 s to 0.781/2.231/4.454 s. The native medians for those lengths were 0.747/2.160/4.310 s. The compact control's 5.838 s tail sample was a 6,142-token request. Retain Prefill 14 as an experimental control for evaluating further changes. It remains slower than native on this workload and does not meet the latency target.

## Prefill 15 · Send each token once to each remote owner

**Route evidence.** Read the seven real captures from Prefill 07. Three captures were successive single-source chunks comprising 3,070 tokens. Four captures were mixed-source chunks from the concurrent requests. Using the existing home-owner route replay at 80 rows, count every remote `(source, token, expert)` route and every unique `(source, token, destination)` per layer and round. Each omitted copy accounts for 5,280 bytes of FP8 hidden and raw scales. Return results still occupy one 10,240-byte BF16 vector per remote expert route. The 256-row pass over the same routes produced the same aggregate counts.

| Chunks | Remote expert routes | Unique remote token destinations | Input payload reducible | Input plus return payload reducible | Busiest-link input reducible at 80 rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single-source | 558,268 | 304,282 | 45.50% | 15.48% | 54.63% |
| Mixed-source | 57,607 | 31,670 | 45.02% | 15.32% | 56.86% |

The busiest-link metric sums the maximum directed-link route count in each layer-round before and after deduplication. The raw route summary is `route-summary.json`. These percentages count logical payload bytes. They exclude transport overhead, local copies, and compute time.

**Candidate.** Start from Prefill 14. For each remote destination, use the first top-k route for a token as the canonical row. The sender writes its FP8 hidden and raw scales straight into that expert's compact segment. Other routes to the same rank carry the canonical row through an unused word of the existing return descriptor. The receiver's scale-pack task fills only alias hidden rows and their scale data. Every expert still receives its own route weight and its output returns to its original top-k slot. The candidate changes only compact dispatch and leaves the width and collective topology fixed. The hypothesis is that the reduction in remote activation transfer outweighs canonical-row selection and local duplicate-row copies. Both operations need measurement on the same frozen input.

The 80-row EP4 slab generated host code with PTOAS 0.72 for the `1024,0,0,0` fixture. Full Serving compile also passed at compact widths 80 and 256 with Decode at 80. The device sources and route summary were staged under `verification/ep4/experiments/prefill-15`. The `prefill-15-source.tar.gz` SHA-256 is `06995794264a3fc65a236278d141109956b3bd3adbd527c388eba0a4149e1f5d`. The compact candidate source hash is `a90c732cde064537ca4cbbf7edc0a9b5bd94e702c45420756a01d443b32ba646`.

The `prefill-15.sh` wrapper accepts a width of 80 or 256 for all three arms. Its isolated Serving snapshot adds EP4 native to the existing width override, so the 256-row native and compact arms run the same CLI compiler path and both use four MoE rounds per full chunk. Keep the native memory budget identical. If native cannot allocate, retain the allocation evidence and use the 80-row comparison for latency attribution. Device golden checks and any Serving latency gain for Prefill 15 are pending.

The follow-on suite `p15-20261121T042026Z` was queued after Prefill 14, PID 2196296. It waits for the first suite to finish successfully and checks its same-width Serving report before requesting the card lock. If the first suite already meets 30%, it exits without launching Prefill 15. Otherwise it runs the two 80-row frozen goldens with dedup enabled, followed by native, Prefill 14 compact, and Prefill 15 compact at matched 256-row widths. The device and memory allocation policy is the same for every arm.

The Prefill 14 suite exited 0 without meeting the latency target. The Prefill 15 suite acquired the allocation and started its single-source golden at 04:52:11 UTC.

The gate passed for the frozen 80-row single-source `1024,0,0,0` replay on cards 4–7. All three output tensors passed and the arm exited 0. Across 20 timed samples after five warmups, the fastest-rank median was 14.024 ms, the per-iteration maximum median was 15.549 ms, and the mean host union was 41.558 ms. The respective Prefill 14 snapshot measurements on the same fixture were 12.944, 18.152, and 35.461 ms. These are separate device runs with the same frozen input and show different effects across timing conventions. The mixed-source golden started at 04:54:40 UTC.

The frozen 80-row mixed-source `81,80,79,0` replay also passed all three output comparisons and exited 0. Its fastest-rank median, per-iteration maximum median, and mean host union were 2.221, 3.809, and 7.314 ms, respectively, versus 2.138, 3.926, and 6.762 ms for Prefill 14 on the same fixture. The same-width 256-row Serving series started with native at 04:57:17 UTC.

The 256-row native arm reached health in 288.072 s and ran all 12 requests, but ten stopped at the 16-token cap and the other two generated 5 and 8 tokens. The 3,070-token check generated 128 tokens and failed to answer `323`. The native arm and suite exited 1, and neither compact Serving arm started. Discard these timing samples as a comparison baseline. The issue was correctness beyond the original native width. Device allocation succeeded. See `p15-20261121T042026Z-native.log` and its request directory.

### Prefill 15b · Correct wide native scatter before comparison

The generated native 256-row orchestration launches 32 AIV blocks for `ffn_scatter`. Its generated C++ divides and takes the remainder of the row count by 64, using `ROUTED_VECTOR_WORKER_TILE` from the module imported for 80-row Decode. Rows assigned to partitions 32–63 are skipped. Build scatter's partition with the `vector_workers` compile-time argument that also sets its launch. This is a baseline correctness fix. Apply the same lib file in the corrected native and both compact snapshots. All three keep 256-row Prefill, 80-row Decode, the same empty-round policy, and the original request validations. Treat the corrected 256-row native latency as eligible only after the frozen golden, long answer, and both concurrent waves pass.

Host compilation with PTOAS 0.72 confirmed launch and partition both use 32 blocks at 256 rows, and both still use 64 blocks at 80 rows. Decode compiled at 80 rows in both processes. The corrected `expert_routed.py` SHA-256 is `f7c3e242f9951bc047d34149e079841cf0eb42757870fb333fb8ec74740d4c71` in each remote `experiments/prefill-15b` lib snapshot. Suite `p15b-20261121T051014Z` (PID 2581115) waits for Prefill 16. If the 30% target remains open, it will run the native 256-row frozen golden and rerun all three 256-row Serving arms after that gate passes.

The corrected 256-row native arm passed all three frozen tensor comparisons on `1024,0,0,0` and exited 0 in 144.58 s. Across 20 timed iterations after five warmups, the fastest-rank median was 5.192 ms, the per-iteration maximum median was 9.066 ms, and mean host union was 32.861 ms. Native 256-row Serving started at 05:42:03 UTC. Qualification still requires the timed requests, long answer and both concurrent waves at that width.

The corrected 256-row native Serving arm reached health in 296.102 s. Prefill 14 compact started at 05:51:20 UTC and the Prefill 15 activation deduplication arm started at 06:00:33 UTC. The latter reached health in 290.111 s.

**Matched 256-row Serving result.** All three arms and the suite exited 0. Each arm passed the 12 fixed requests at 13 generated tokens, the 3,070-token answer `323`, and both eight-request waves. The saved commands match after normalizing port and backend selection. Complete per-request samples are in `p15b-20261121T051014Z-comparison.json`.

| 256-row arm | Prefill P50 / P95 | Client TTFT P50 / P95 | Prefill P50 reduction vs native |
| --- | ---: | ---: | ---: |
| Corrected native | 1.506 / 3.033 s | 1.587 / 3.119 s | 0.00% |
| Prefill 14 compact | 1.453 / 2.904 s | 1.532 / 2.990 s | 3.53% |
| Prefill 15 activation deduplication | 1.524 / 3.100 s | 1.604 / 3.185 s | -1.18% |

The 30% target at this width requires P50 of about 1.054 s. For 1,024, 3,070, and 6,142 input tokens, the medians were 0.521/1.506/3.014 s for native, 0.502/1.453/2.867 s for Prefill 14, and 0.506/1.524/3.025 s for Prefill 15. Activation deduplication raised P50 by 4.88% against Prefill 14 despite reducing logical remote payload bytes. Its current alias selection and local expansion are possible costs. The Serving test did not time those stages separately. Keep Prefill 14 as the measured wide compact control and discard this deduplication implementation as a latency optimization.

## Prefill 16 · Remove empty activation sends from single-source chunks

**Candidate.** Prefill 14 restricts histogram traffic when the scheduler identifies a sole source, but it still submits `moonep_push` on every rank. The other three ranks launch 56 AIV blocks each, scan their empty send counts, and publish an arrival that the active source and its peers all wait for. Isolate this remaining synchronization cost using the Prefill 14 data path. The sole source submits push and publish and can pack its local receive rows once those writes complete. Each other rank skips push and waits for the source's payload epoch before reading its receive rows. Mixed-source chunks still submit push on every rank and wait for all peers. Route counts, activation bytes, return and release protocols, window sizes, and width are the same as Prefill 14. The EP4 single-source graph submits one push grid and three peer payload waits per round, down from four grids and twelve peer waits.

**Host validation.** The 80-row EP4 slab generated A5 code with PTOAS 0.72 for `1024,0,0,0`. The generated orchestration places push and publish inside the sender branch and the payload wait inside the receiver branch. The scale-pack task still depends on receive completion and the row plan. Full Serving host compilation passed at 80 and 256 rows with 80-row Decode. Logs are `compile-slab-compact-80.log` and `serving-compact-{80,256}.log` in the local Prefill 16 experiment.

The source was staged under `verification/ep4/experiments/prefill-16/pypto-lib-candidate`. It differs from Prefill 14 only in `prefill_moonep.py`, whose verified SHA-256 is `9589ad2d2b4bed6ce631976981d3d997e1690de3eec155503e82fd641a939a95`. The suite `p16-20261121T045325Z` (PID 2483975) was queued after Prefill 15. It skips execution if the preceding matched report meets the target. Otherwise it runs the same frozen single-source and mixed-source goldens, followed by native, Prefill 14 compact, and Prefill 16 compact with all three arms at one width. It uses 256 rows if the preceding native 256-row arm completed successfully, otherwise 80.

After the native 256-row arm in Prefill 15 failed correctness, Prefill 16 selected the 80-row comparison. Its frozen single-source device golden passed all three tensor checks and exited 0. Across 20 timed samples after five warmups, the fastest-rank median was 12.958 ms, the per-iteration maximum median was 16.144 ms, and mean host union was 37.360 ms. On the same fixture Prefill 14 had recorded 12.944, 18.152, and 35.461 ms. The mixed-source fixture started at 05:09:16 UTC. Treat the timings as diagnostic until the matched Serving suite finishes.

The mixed-source golden also passed all three tensors and exited 0. Its fastest-rank median, per-iteration maximum median, and mean host union were 2.143, 3.755, and 6.625 ms, versus 2.138, 3.926, and 6.762 ms for Prefill 14 on the same input in the previous suite. The matched 80-row Serving series started with native at 05:11:48 UTC.

The 80-row native arm reached health in 290.041 s, passed the 12 fixed requests, long answer, and both eight-request waves, and exited 0. Warm Prefill P50/P95 were 2.238/4.465 s and TTFT P50/P95 were 2.318/4.560 s. The target against this run requires P50 at or below 1.566 s. Prefill 14 compact started at 05:21:02 UTC, followed by Prefill 16 compact.

The 80-row Prefill 14 compact control reached health in 290.071 s and exited 0 after the same request checks. Warm Prefill P50/P95 were 2.302/4.597 s and TTFT P50/P95 were 2.382/4.684 s. Its Prefill P50 was 2.89% higher than the native control. The saved commands match when port and backend selection are normalized. Prefill 16 compact started at 05:30:16 UTC.

**Matched 80-row Serving result.** Prefill 16 compact reached health in 286.085 s and exited 0. All three arms passed the fixed 12 requests at 13 generated tokens, the 3,070-token answer `323`, and both eight-request waves. The persisted commands match after normalizing the port and backend argument. The suite exited 0 and saved every timed sample in `p16-20261121T045325Z-comparison.json`.

| 80-row arm | Prefill P50 / P95 | Client TTFT P50 / P95 | Prefill P50 reduction vs native |
| --- | ---: | ---: | ---: |
| Native | 2.238 / 4.465 s | 2.318 / 4.560 s | 0.00% |
| Prefill 14 compact | 2.302 / 4.597 s | 2.382 / 4.684 s | -2.89% |
| Prefill 16 compact | 2.237 / 4.539 s | 2.316 / 4.623 s | 0.04% |

Removing the empty sends lowered Prefill P50 by 2.85% against Prefill 14 compact. At 1,024, 3,070 and 6,142 input tokens, candidate medians were 0.780/2.237/4.484 s versus 0.795/2.302/4.561 s for the compact control and 0.804/2.238/4.445 s for native. The candidate is effectively tied with native at P50 and is 1.66% slower at P95 on this sample. Retain it as the measured control for the next protocol changes. The 30% target remains open. Prefill 15b started the corrected 256-row native golden at 05:39:30 UTC.

## Prefill 17 · Put the route weight in the existing return descriptor

**Evidence and candidate.** The historical real-weight 80-row chip trace in `diag-08-t80-summary.json` reported 364.69 core-ms for `moonep_push_spmd` on rank 0 across 520 layer-rounds. This is summed core time and includes all work in the push task. Prefill 14 issues two separate 32-byte metadata stores per compact route, one for its FP32 weight and one for its return descriptor. The candidate puts the FP32 bits in word three of the existing descriptor and issues one metadata store. The scale-pack task loads a tile of descriptors and recovers the exact FP32 bits into the existing compute tensor. Scatter keeps the original source and top-k words. Width, expert placement, payloads, and collective ordering use the Prefill 14 configuration. The isolated hypothesis is that one fewer remote store per route outweighs the local tile extraction cost.

**Host validation.** The 80-row slab compiled with PTOAS 0.72 and the full Serving compile passed at 80 and 256 rows with Decode at 80. The generated compact orchestration has no `recv_weights` input for push or pack. The existing receive-window allocation stays in the benchmark ABI so this iteration measures the transport change with the same capacity. Logs are `compile-slab-compact-80.log` and `serving-compact-{80,256}.log` in the local Prefill 17 experiment.

The isolated candidate is staged under `verification/ep4/experiments/prefill-17/pypto-lib-candidate`. Its only change from Prefill 14 is `prefill_moonep.py` with SHA-256 `b88a8185a491a5f245d263239c218052008772d203296b4fabdb2a984f6e07e8`. Device validation remains pending. Use the two frozen goldens and measure against Prefill 14 first. Combine with deduplication or the one-sender payload protocol only after the individual changes have Serving evidence.

Suite `p17-20261121T051629Z` (PID 2622099) waits for Prefill 15b. It skips device work if Prefill 16 or 15b meets the target. Otherwise it runs both 80-row goldens and the metadata-fusion candidate against the preceding completed same-width native and Prefill 14 controls. It prefers the corrected 256-row report from Prefill 15b, falling back to the 80-row Prefill 16 controls if that report fails validation. The wide candidate contains the same corrected scatter file as Prefill 15b. Reusing the preceding controls saves two Serving startups. Repeat a matched pair before relying on a threshold result from controls reused across suites.

Prefill 15b completed successfully below the target. Prefill 17 selected its 256-row controls and started the first 80-row frozen golden at 06:10:00 UTC.

Both 80-row goldens exited 0 and passed `next_pre_mix`, `x_mixed`, and `x_next`. With five warmups and 20 measured iterations per replay, single-source fastest-rank median, per-iteration maximum median, and mean host union were 12.699, 16.777, and 37.132 ms. On the mixed-source replay they were 2.125, 3.804, and 6.665 ms. These are device diagnostics at 80 rows. The same-width 256-row Serving candidate started at 06:15:16 UTC.

The 256-row candidate reached health in 286.067 s. It passed the fixed 12 requests at 13 tokens, the 3,070-token answer, and both concurrent waves. The normalized command matches the two 256-row Prefill 15b controls. The arm and suite exited 0 and saved the comparison in `p17-20261121T051629Z-comparison.json`.

| 256-row arm | Prefill P50 / P95 | Client TTFT P50 / P95 | Prefill P50 reduction vs native |
| --- | ---: | ---: | ---: |
| Prefill 15b native control | 1.506 / 3.033 s | 1.587 / 3.119 s | 0.00% |
| Prefill 14 compact control | 1.453 / 2.904 s | 1.532 / 2.990 s | 3.53% |
| Prefill 17 compact | 1.360 / 2.872 s | 1.439 / 2.959 s | 9.74% |

The fused descriptor reduced P50 by 6.44% against Prefill 14 compact. Candidate medians for 1,024, 3,070, and 6,142 input tokens were 0.462/1.360/2.779 s. The candidate is the fastest observed matched-width compact arm. Its native control was reused from the previous suite. Carry the mechanism into a combined candidate and repeat controls if a candidate reaches the 30% target. Prefill 18 started its single-source golden at 06:24:25 UTC.

## Prefill 18 · Acknowledge returns and window reuse through the sole source

**Candidate.** Prefill 16 restricts the forward payload notifications, while every rank still exchanges return-completion and window-release notifications with all peers. For a single-source chunk, each expert rank can publish its scatter completion only to the source. The source waits for every peer scatter and its own scatter before reducing the complete set of return rows. It then publishes the reuse epoch to all peers. A receiver can start its next round after that epoch. At that point all previous scatter tasks have finished and the source has consumed the returned rows. Retain the existing combine computation, local zero initialization dependencies, and mixed-source protocol. The EP4 single-source case reduces each of the four peer notification or wait phases from 12 operations to three without changing the width or transmitted tensors.

**Host validation.** The 80-row slab generated code with PTOAS 0.72 for the frozen `1024,0,0,0` shape. The generated orchestration gates scatter publication, return wait, release publication, and reuse wait by rank role. Serving host compilation passed at 80 and 256 rows with Decode at 80. Logs are `compile-slab-compact-80.log` and `serving-compact-{80,256}.log` in the local Prefill 18 experiment. The remote snapshot is `verification/ep4/experiments/prefill-18/pypto-lib-candidate`. Its only change from Prefill 16 is `prefill_moe.py`, SHA-256 `a2d264c5df6a7992322f5addfb032cc547f100da0a1e52ae8bdf1f9fa40743cd`. Compare to Prefill 16 only after its Serving result. Device validation of Prefill 18 has not started.

Suite `p18-20261121T052209Z` (PID 2667722) is queued behind Prefill 17. It skips device work if a preceding comparison reaches the target. Otherwise it requires the validated Prefill 16 report, runs both 80-row goldens, then uses Prefill 16 native and compact as same-width controls for the 80-row candidate. This isolates the return protocol from the already measured activation protocol and avoids repeating the two control startups.

The 80-row frozen single-source arm exited 0 and passed all three tensor checks. Across 20 timed iterations after five warmups, the fastest-rank median was 12.763 ms, the per-iteration maximum median was 16.625 ms, and mean host union was 36.537 ms. Prefill 16 recorded 12.958, 16.144, and 37.360 ms respectively on the same replay. These diagnostic metrics give mixed signals. The mixed-source replay started at 06:27:03 UTC.

The 80-row mixed-source arm also passed the three frozen tensor checks and exited 0. Its fastest-rank median, per-iteration maximum median, and mean host union were 2.148, 3.797, and 6.658 ms, compared with 2.143, 3.755, and 6.625 ms for Prefill 16. The 80-row Serving candidate started at 06:29:40 UTC.

The candidate reached health in 290.043 s and passed the 12 fixed requests, long answer, and both eight-request waves. The arm and suite exited 0. The normalized commands match both Prefill 16 controls. Warm Prefill P50/P95 were 2.312/4.601 s and TTFT P50/P95 were 2.391/4.688 s. Prefill P50 was 3.36% higher than Prefill 16 compact and 3.31% higher than the 80-row native control. Candidate P50s for 1,024, 3,070, and 6,142 tokens were 0.788/2.312/4.542 s. Retain Prefill 16 as the 80-row compact control. The result is in `p18-20261121T052209Z-comparison.json`. Prefill 19 started at 06:39:04 UTC.

## Prefill 19 · Build the sole source's offsets during histogram publication

**Evidence and candidate.** On the historical 80-row Prefill 08 trace, source-rank row planning occupied 13.4 ms across 516 tasks on the observed path. Prefill 14 already reduced this task's reads by using only the sole source's counts. Prefill 19 starts from Prefill 16 and removes the separate source-side plan in the compact single-source path. The source notifies the peers with their counts, then walks its existing histogram cursor to assign packed offsets and write its own receive counts and starts. Peers still plan their local receives after the notification, while the source can push after its histogram task. The source omits the histogram store to itself. The mixed-source and replica-enabled paths keep their existing plan. This isolates the task boundary and repeated count reads with the same route order, alignment, payloads, width, and return protocol.

**Host validation.** The 80-row EP4 slab compiled for `1024,0,0,0` with PTOAS 0.72. The generated graph submits `moonep_row_plan` only when the local rank differs from `sole_source`, including every rank in the mixed-source case. Push retains explicit dependencies on scale readiness and the completed plan or histogram. Full Serving host compilation passed at 80 and 256 rows with 80-row Decode. The experiment changes `prefill_moonep.py` with SHA-256 `8c250872554861ee0d37bd4f8f078931a9e2316c80e7d429868ad285286c7b76`. Both frozen goldens and same-width Serving comparisons are pending.

The remote snapshot `verification/ep4/experiments/prefill-19/pypto-lib-candidate` copies Prefill 16 and replaces only that source file. The source package and reference hashes matched and all three wrapper scripts passed `bash -n`. Suite `p19-20261121T054019Z` (PID 2794479) is queued behind Prefill 18. It checks earlier qualified reports for the target and otherwise runs both 80-row goldens before evaluating the candidate against the validated Prefill 16 80-row native and compact arms. A result at the target with reused controls requires a fresh matched repeat.

Both 80-row frozen replays exited 0 with all three tensors passing. With 20 measured iterations after five warmups, the single-source fastest-rank median, per-iteration maximum median, and mean host union were 12.806, 13.598, and 40.739 ms. Prefill 16 measured 12.958, 16.144, and 37.360 ms on that replay. Mixed-source values were 2.126, 3.805, and 6.678 ms, compared with 2.143, 3.755, and 6.625 ms for Prefill 16. The timing conventions give different signals. The 80-row Serving arm started at 06:45:37 UTC.

The candidate reached health in 292.097 s and passed the fixed requests, long answer, and both concurrent waves. The arm and suite exited 0. The normalized commands match the Prefill 16 controls. Warm Prefill P50/P95 were 2.176/4.442 s and TTFT P50/P95 were 2.255/4.528 s. Prefill P50 improved by 2.73% against Prefill 16 compact and 2.77% against the 80-row native control. At 1,024, 3,070, and 6,142 input tokens, the candidate medians were 0.745/2.176/4.372 s. Comparing matching request IDs, 12 of 12 Prefill times were lower than native and eight were lower than Prefill 16 compact. This is the best observed default-width result with reused controls. Use the change as a candidate for combination, with a paired repeat needed before treating its small gain as stable. The report is `p19-20261121T054019Z-comparison.json`. Prefill 20 started its first 80-row golden at 06:55:07 UTC.

## Overlap opportunity within one layer

The current slab reuses its scratch after result copies complete. The 80-row Prefill 08 source-rank trace verifies this ordering for all 480 pairs of successive rounds within layers. Across those pairs, the next round's row preparation through histogram start took 102.500 ms. Through push start it took 155.455 ms. The preceding rounds occupied 185.204 ms between scale-pack end and scatter start. Summing the shorter interval in each pair gives 101.942 ms of ideal overlap for local preparation and 151.158 ms for preparation through push start. These totals use historical phase spans, assume perfect scheduling and independent storage, and ignore competition for AIC and AIV resources. They are bounds on overlap of the measured intervals, with no prediction for unprofiled Serving.

The same reconstruction at 192 rows covers 200 successive round pairs. It gives 50.886 ms of ideal overlap for local preparation and 72.086 ms for preparation through push start. At that width, the estimated overlap for the transport preparation adds 21.200 ms beyond local preparation in one historical 1,024-token trace. The estimates shrink as the number of adjacent rounds falls. Keep the 80-row and 192-row evidence with their respective traces. The 192-row report is `diag-08-overlap-192-summary.json`, generated with `analyze-diag08-round-overlap.py` and six rounds per layer.

Compact receive rows fit two activation and per-row metadata windows at 49.90 MiB per rank at width 80, or 92.96 MiB at width 256. A single native receive window uses 156.56 MiB and 501.00 MiB at those widths. This count includes FP8 hidden, scale bytes, weight and route words. It excludes return, sync, and local scratch storage. The smaller footprint makes it practical to examine a separate workspace for an adjacent round while holding width fixed. Measure isolated lookahead and the common local preparation gain before attributing a Serving difference to compact transport. See `analyze-diag08-overlap.py` and `diag-08-overlap-summary.json` in `verification/ep4`.

The single-source compiler path can use tighter activation slots. One source emits at most `width * 6` routes across all destinations. For any destination, up to 96 experts can each add 31 padding rows, and the resulting size is a multiple of 32. At width 256 the bound is 4,512 rows per slot. Two slots fit in 9,024 rows of the existing 9,120-row EP4 compact window. At width 80 two slots require up to 6,912 rows compared with the current 4,896. Replay of the three real single-source chunks observed at most 1,696 packed rows at 256 and 1,312 at 80. The larger bounds cover every legal route without relying on those observations. An overlapping prototype needs independent slot staging and arrival epochs. Additional return storage or an explicit handoff before reusing the return window is still required. The mixed-source layout can continue to use the full receive window. `analyze-single-source-windows.py` and `diag-07-single-source-window-summary.json` preserve the calculation and route check.

### Owner-specific activation readiness estimate

Replay the same three single-source route captures with the existing 56 push blocks and four owner-specific groups of 14 blocks. For each round, count the busiest block's routes and each owner's 64-row expert tiles. The common notification model waits for the busiest push block before any owner computes. The split model lets each owner compute when its own push group finishes. With the modeled aggregate push cost set to one quarter of the expert cost, early per-owner readiness improves the combined phase proxy by 3.70% at width 80 and 2.83% at width 256. The 256-row proxy improvement ranges from 1.76% to 4.23% when the relative cost varies from one eighth to one. This model assumes linear costs and concurrent scheduling of the four groups. It excludes the extra launch and synchronization costs. Defer owner-specific groups while measuring adjacent-round overlap. See `analyze-owner-dispatch.py` and `diag-07-owner-dispatch-summary.json` in the verification directory.

## Prefill 20 · Overlap neighboring rounds using two compact slots

**Candidate.** Start from Prefill 16 and alternate slots within each EP4 single-source compact layer. Give each slot its own activation segments, histogram row, gather and output scratch, result rows, and the existing arrival and release protocol with slot-specific offsets. A new round depends on the previous occupant of its own slot. The layer waits for both completions before advancing. Mixed-source rounds keep slot zero and the full activation window. Width, route order, expert tiles, transmitted payloads, and the release condition for a reused slot are held constant. The 256-row activation window remains 9,120 rows. The 80-row activation window grows from 4,896 to 6,912 rows, which adds 10.27 MiB per rank including per-row metadata. Both widths allocate one extra return slot and local scratch.

**Host validation.** The 80- and 256-row EP4 slab compiles passed with PTOAS 0.72 and the frozen single-source shape. Generated orchestration selects the slot from round parity, indexes the corresponding completion for row staging, and joins both slots at the layer boundary. It retains automatic dependencies for shared external tensor handles. A device trace is needed to establish the realized overlap. The first full Serving compile referenced the return-row constant before `TOPK` was defined. Moving the calculation after `TOPK` fixed the error. Full Serving host compilation then passed at Prefill widths 80 and 256 with Decode at 80.

The shared entry also passed a 256-row native Serving compile with 80-row Decode. The existing Prefill round-metadata and route replay suite passed all eight host tests.

The source snapshot is `verification/ep4/experiments/prefill-20/pypto-lib-candidate`. The four Prefill source hashes and common corrected `expert_routed.py` hash passed verification against `lib-source.sha256`. The latter compiles the 80-row scatter with the same 64 partitions as its control. Suite `p20-20261121T060336Z` (PID 2893552) is queued after Prefill 19 on cards 4–7. It runs 80-row single- and mixed-source goldens for the alternating and serial paths, then the 256-row single-source golden. The first Serving measurement uses 80 rows because the historical 80-row trace exposes more overlap and has a matched Prefill 16 native and compact control. A meaningful reduction will need a trace or a local-lookahead ablation to attribute the incremental gain to overlapping MoonEP transport. Device validation is pending.

The 80-row single-source fixture exited 0 with all three tensors passing across repeated slot reuse. With five warmups and 20 measurements, the fastest-rank median dropped from Prefill 16's 12.958 ms to 9.234 ms, a 28.74% reduction. The per-iteration maximum median dropped from 16.144 to 11.841 ms. Mean host union rose from 37.360 to 40.611 ms. These metrics describe the frozen device replay. The mixed-source fixture started at 06:57:49 UTC. The default-width Serving result is pending.

The 80-row mixed-source fixture also passed all three tensors and exited 0. Its fastest-rank median, per-iteration maximum median, and mean host union were 2.142, 3.835, and 7.246 ms, against Prefill 16's 2.143, 3.755, and 6.625 ms. The supplementary 256-row single-source replay passed. Default 80-row Serving started at 07:02:51 UTC.

The Serving arm reached health in 300.068 s. It passed the 12 fixed requests with 13 output tokens each, the long answer check, and both eight-request waves. The arm and suite exited 0. The normalized launch command matched the two Prefill 16 controls. The report `p20-20261121T060336Z-comparison.json` records the following warm timings in seconds.

| Default 80-row arm | Prefill P50 | Prefill P95 | TTFT P50 | TTFT P95 |
|---|---:|---:|---:|---:|
| Prefill 16 native | 2.237811 | 4.464890 | 2.317592 | 4.560076 |
| Prefill 16 compact | 2.236836 | 4.539110 | 2.316223 | 4.623488 |
| Prefill 20 two-slot compact | 1.828501 | 3.786430 | 1.909366 | 3.875626 |

Against the reused native control, warm Prefill fell 18.29% at P50 and 15.20% at P95. TTFT fell 17.61% at P50 and 15.01% at P95. Prefill P50 fell 18.26% from Prefill 16 compact and 15.96% from Prefill 19. All 12 matching requests had lower Prefill than each of those three arms. Candidate P50s at 1,024, 3,070, and 6,142 tokens were 0.633, 1.829, and 3.708 s. The current target against this native sample is 1.566468 s. It requires a further 14.33% decrease from this candidate. The observed benefit includes both local lookahead and the independent communication slots. Prefill 25 measures their contributions and Diagnostic 22 checks cross-round scheduling. A fresh native and compact pair remains necessary at the target. Prefill 21 started its first frozen replay at 07:12:16 UTC.

## Prefill 21 · Send the compact row plan with the activation

**Candidate.** Start from Prefill 16. For an EP4 compact chunk with one active source, the source can compute every expert's aligned start and count in the histogram task. Write both fields as one word per expert into the existing receive metadata window. The high 16 bits hold the start and the low 16 bits hold the count. The compile-time compact bound keeps the word nonnegative in `INT32`. Metadata and activation become readable at the same payload epoch. The passive ranks arm the payload wait after their current row staging task and submit no histogram, histogram wait, or separate row plan. The existing scale-pack grid reads the row descriptions and initializes the unchanged expert inputs. Its metadata writers each own a full 64-byte line. Activation, scale, route weights, scatter, and return ordering are unchanged. The mixed-source path keeps its histogram and planner.

Each single-source layer-round across EP4 goes from four histogram launches, three histogram waits, and four row plans to one source task for the same work. The packed descriptions retain the same logical histogram size. The modified scale-pack grid does extra local descriptor reads and the receivers no longer plan while the activation is in flight. Timed Serving will resolve the tradeoff.

**Host validation.** The first two 80-row slab compiles reported a reused scalar name with incompatible types and an index-to-INT32 write mismatch. The next build caught a scope conflict between the push and pack loops. Distinct temporaries and explicit INT32 casts resolved all three errors. The 80-row slab compiled at PTOAS 0.72. The generated graph gates histogram, histogram wait, and row plan by the packed-plan predicate and keeps scale pack after payload completion.

The 256-row slab and both full Serving Prefill widths compiled with the same toolchain. Serving retained Decode at 80 rows. The verified snapshot is `verification/ep4/experiments/prefill-21/pypto-lib-candidate`, with source hashes in `lib-source.sha256` and the common corrected scatter file. Suite `p21-20261121T061915Z` (PID 2960217) is queued after Prefill 20. It uses the two frozen 80-row shapes and the measured 80-row Prefill 16 native and compact controls. It skips device work if a previous qualified comparison has already met the target. Device correctness and latency are pending.

Both 80-row frozen replays exited 0 and passed all three tensor checks. With five warmups and 20 measurements, the single-source fastest-rank median, per-iteration maximum median, and mean host union were 12.507, 13.379, and 41.384 ms. Prefill 16 recorded 12.958, 16.144, and 37.360 ms. Mixed-source values were 2.157, 3.883, and 7.478 ms, compared with the control's 2.143, 3.755, and 6.625 ms. The 80-row Serving candidate started at 07:17:32 UTC.

The candidate reached health in 286.109 s. The arm and suite exited 0 after the fixed 12 requests, long answer, and both eight-request waves. The normalized commands match the Prefill 16 controls. Warm Prefill P50/P95 were 2.231/4.490 s and TTFT P50/P95 were 2.309/4.576 s. Prefill P50 fell 0.24% against Prefill 16 compact and 0.29% against native. Prefill P95 fell 1.07% against Prefill 16 compact and rose 0.57% against native. The 1,024-, 3,070-, and 6,142-token P50s were 0.773, 2.231, and 4.464 s. This sample gives insufficient evidence to carry the packed row plan into the two-slot path. Retain Prefill 20 as the measured leader at 80 rows. See `p21-20261121T061915Z-comparison.json`.

## Diagnostic 22 · Compare native and compact chip paths at 80 rows

The earlier real-checkpoint trace compared two compact widths. Capture a matched native and compact pair at 80 rows to quantify how much time remains in transport, expert compute, common preparation, and round transitions. Use Prefill 16 native and the lowest measured P50 among the qualified 80-row compact candidates from Prefill 16 and 18–21. Keep the same Serving snapshot, fixed 1,024-token prompt, sidecar, trimming, allocation, and NPU runtime. Both arms use one output token, diagnostic level-1 chip timing, and dependency capture. Save each compiled artifact and each rank's captures under that arm's output directory. Use the unprofiled reports for Serving latency.

The `diag22-20261121T062324Z` suite (PID 2974853) is queued after Prefill 21. It skips both captures if a previous qualified comparison reaches the 30% target. The selector passed against the completed reports on the server at staging time. The driver is adapted from the validated Prefill 08 one-request driver. The isolated Serving patch adds artifact persistence, chip timing, and dependency collection to the normal compiler path. Shell syntax and Python compilation passed. The device gate requires a successful diagnostic request and a raw chip capture for each of the four ranks in both arms. Trace-derived changes to phase spans remain pending.

`analyze-diag22-paths.py` reuses Simpler's trace graph loader and rank-local observed path. It also measures intersections between the task envelopes of activation push or scale pack and routed expert execution on the same rank. A pair of these phases in one round already has an explicit dependency, so intersection in the paired trace is evidence of inter-round scheduling. Task envelopes include gaps between blocks and do not measure simultaneous active cores. The historical serial 80-row capture check found 520 round-staging tasks, zero unresolved tasks, and zero measured overlap on each of the first two ranks. The script and check output `diag-22-analyzer-check-t80.json` were saved in the verification directory.

The native capture began at 07:26:54 UTC after the Prefill 21 Serving result. The selector chose Prefill 20 as the compact arm. Native Serving exited during startup at 07:31:11 UTC after `rtMalloc` returned 207001 while allocating 22,649,241,600 bytes for the worker. The host showed other active processes on all four selected cards when checked after the failure. No diagnostic request or trace was captured.

The retry `diag22r-20261121T074020Z-native` exited 0 with a successful diagnostic reply and four Prefill chip captures. Each graph resolved all named tasks and recorded 520 MoE rounds. Rank 0–3 device spans were 627.418, 607.724, 589.138, and 584.698 ms. Activation push and scale pack had zero measured task-envelope overlap with routed expert work on all four ranks. On rank 0, the reconstructed path attributed 133.39 ms to routed expert tasks, 38.10 ms to activation tasks, 24.25 ms to local preparation, 15.79 ms to route metadata, 14.87 ms to result transfer and reduction, and 10.52 ms to return and reuse signals. Preceding gaps for the last two groups were 35.84 and 29.22 ms. See `diag22r-20261121T074020Z-native-paths.json` for all path groups and rank results. The matched Prefill 20 compact capture started at 07:50:10 UTC.

The diagnostic retry finished successfully for both backends. Both used a 1,024-token prompt and generated one token. Their recorded CLI matches after allowing for backend, port, and profile output directory. Prefill 20 recorded 520 rounds and no unresolved task names on each of the four ranks. Its device spans were 493.932, 485.622, 477.446, and 479.609 ms in rank order. The longest observed rank span fell 21.28% from native in this diagnostic replay.

The archived `analyze-diag22-core-spans.py` measures intersections of active task blocks rather than their envelopes. On the source, 9.267 ms of activation push, 2.962 ms of scale packing, and 6.332 ms of scatter coincided with routed expert execution in Prefill 20. Native measured zero for these intersections. The three compact receiver ranks each recorded about 2.1 ms of scale packing and 6.9–7.5 ms of scatter coincident with experts. Per-phase intersections can overlap each other. Source push still occupied 45.354 ms, of which 36.087 ms did not coincide with routed experts. Compact source gaps before route metadata tasks and local preparation fell from 53.019 and 39.442 ms to 6.929 and 18.455 ms on the reconstructed path. Gaps before result transfer and reduction were 28.510 ms. The technical evidence is in the `diag22r-20261121T074020Z-{native,compact}-{paths,core}.json` files. Prefill 23 restarted its frozen single-source fixture at 07:58:50 UTC.

## Prefill 23 · Combine one-sender activation with the fused route descriptor

**Candidate.** The matched 80-row Prefill 16 candidate improved on its compact control by 2.85%, and the 256-row Prefill 17 candidate improved on its compact control by 6.44%. Start from the verified 256-row Prefill 17 snapshot. Apply Prefill 16's one-sender push, publication, and wait ordering around the existing fused-descriptor push loop. Mixed-source dispatch retains the original peer participation pattern. The combined candidate keeps width, trimming, row plan, weights, return traffic, and device allocation fixed. Test the additional effect of restricting empty activation sends at width 256 against Prefill 17 and the corrected same-width native baseline. The separate measured gains have different widths and can interact.

**Host validation.** The 80-row single-source and 256-row mixed-source slabs compiled at PTOAS 0.72. Full Serving compact compiled at widths 80 and 256 with Decode at 80. The C++ graph keeps the one-sender gate on push, and the fused compact push and pack use the route descriptor for the FP32 weight. Only `prefill_moonep.py` changed from the wide Prefill 17 snapshot. Its SHA-256 is `c9f0a7ffe96a0a6a47eb32229b54d3331bc3429f92f085cfc9065f881729a906`.

The original 256-row suite `p23-20261121T062850Z` was superseded before acquiring devices. Its waiter exited at 06:41:19 UTC and the benchmark produced no samples. Replacement suite `p23d-20261121T064119Z` (PID 3053257) waits behind Diagnostic 22 and runs the frozen single- and mixed-source inputs and Serving at the default 80 rows. It reuses the qualified 80-row Prefill 16 native and compact controls. This compares the fused descriptor against the already measured one-sender compact path at the default width. Only earlier completed 80-row Serving reports can satisfy the skip gate. Device correctness and latency remain pending.

The 80-row single-source retry exited 0 and passed all three tensor checks. Across 20 measurements after five warmups, the fastest-rank median was 12.592 ms, the per-iteration maximum median was 16.023 ms, and the mean host union was 34.522 ms. Prefill 16 measured 12.958, 16.144, and 37.360 ms on the same frozen input. The mixed-source replay started at 08:01:23 UTC.

The 80-row mixed-source retry also exited 0 with all three tensor checks passing. Its fastest-rank median, per-iteration maximum median, and mean host union were 2.135, 3.277, and 6.790 ms, compared with 2.143, 3.755, and 6.625 ms for Prefill 16. The 80-row fixed-workload Serving arm started at 08:04:00 UTC as `p23dr-20261121T074020Z-compact-candidate`.

The Serving arm and suite exited 0. The arm reached health in 310.118 s and passed the 12 fixed requests, long answer, and both eight-request waves. The normalized launch command and request IDs match Prefill 16 native, Prefill 16 compact, and Prefill 20. P50/P95 were 2.168/4.364 s for Prefill and 2.250/4.452 s for TTFT. Prefill P50 improved 3.10% over the one-slot Prefill 16 compact control and 3.14% over native. Eleven of 12 matching requests improved relative to each of those controls. The medians by prompt length were 0.718, 2.168, and 4.298 s. The one-slot candidate remains slower than the two-slot Prefill 20 arm at P50. Retain it as the control for Prefill 24 and test the transport change on the two-slot path independently. See `p23dr-20261121T074020Z-comparison.json`.

## Prefill 24 · Read route weights from the descriptor in W13

**Candidate.** Prefill 23 packs the FP32 route weight into the compact descriptor and the scale-pack grid copies it into an intermediate tensor. The W13 vector subcore then loads that tensor and writes the same weight into the output's active-row order. Let compact W13 load the valid descriptors for each tile and gather the FP32 word directly before its existing contiguous store. The gather reads the original bits and keeps the active-row order used by SwiGLU. The scale-pack grid handles scales only. Return descriptors, topology, activation ordering, and MoE width are unchanged from Prefill 23. The common W13 path for native and other layouts still reads contiguous weights. The experiment retains the existing scratch reservation for this comparison, with no compact producer or consumer for the intermediate weights. Measure whether removing the intermediate transfer outweighs the descriptor load and gather on W13's vector subcore.

**Build validation.** The slab initially rejected a locally reassigned constexpr and an aliased weight parameter with missing shape inference. Pass the two weight-layout scalars to the C++ kernel and select the input tensor in the layout-specific call. A5 80-row compact slab code generation then passed with PTOAS 0.72. The compiled scale pack has no weight input or store. The remote CANN toolchain compiled and linked the updated W13 for AIV and AIC through `pypto.runtime.kernel_compiler`, producing 241,248 and 240,528 bytes respectively. Logs and tested source are in `verification/ep4/experiments/.prefill-24-ccec.oPlxhI`.

Full Serving native and compact code generation passed at 80 rows with Decode at 80. The compact graph passes the descriptor window and scalars 8 and 3 to W13 after dispatch completion. The remote source and both compiler logs were staged under `verification/ep4/experiments/prefill-24`, with verified source hashes in `lib-source.sha256`. Suite `p24-20261121T065156Z` (PID 3104886) waits for the default-width Prefill 23 suite. If that suite passes and earlier 80-row results remain below the target, run both 80-row frozen goldens and fixed-workload Serving, using Prefill 23 compact and Prefill 16 native as the controls. Device correctness and latency are pending.

At 08:13:35 UTC the retry `p24r-20261121T074020Z` started its 80-row single-source golden. It exited 0 and passed all three tensors. The fastest-rank median, per-iteration maximum median, and mean host union were 12.552, 16.283, and 37.133 ms, compared with 12.592, 16.023, and 34.522 ms for Prefill 23. Its mixed-source golden requested the device lock at 08:16:08 UTC.

The 80-row mixed-source retry exited 1. `next_pre_mix` and `x_mixed` passed. `x_next` exceeded the 0.003 error threshold at 16.0884% of entries, against the allowed 2%. The suite stopped at the correctness gate before Serving. The difference is specific to this candidate so far, with a passing mixed-source Prefill 23 descriptor control. Prefill 32 uses that working descriptor unpack path. Prefill 26 continued independently at 08:21:20 UTC. See `p24r-20261121T074020Z-golden-mixed-80.log`.

## Prefill 25 · Separate local preparation from two-slot transport

**Control.** Start from Prefill 16 and adopt the two local slab scratch slots from Prefill 20 for default 80-row native and compact single-source chunks. Gather and gate preparation may enter the next slot after its last local occupant completes. Native's window-reuse task and compact's histogram still wait for the immediately preceding round before beginning transport. Keep the original receive and return window sizes, epochs, communication ordering, and mixed-source behavior. This control measures the opportunity available from local preparation in both backends. Compare the compact control to full two-slot compact to isolate the incremental effect of opening the next transport window.

Both 80-row slab builds and both full Serving builds passed on the host with PTOAS 0.72 and 80-row Decode. Generated C++ indexes the current slot's completion for row preparation and the previous slot's completion for transport. The source and logs are verified and staged under `verification/ep4/experiments/prefill-25`.

After Prefill 20 passed Serving, suite `p25-20261121T071321Z` (PID 3206364) was queued behind Prefill 26. It runs the 80-row single- and mixed-source goldens in both backends, then both fixed-workload Serving arms. Compare each lookahead arm to its Prefill 16 control and compare Prefill 20 to both lookahead arms. The compact lookahead comparison measures the added transport-slot effect. The native lookahead comparison holds the common preparation opportunity fixed. The suite writes `p25-20261121T071321Z-ablation.json`. Device work is pending.

Both retry native replays exited 0 and passed all three output tensors. After five warmups, the fastest-rank median, per-iteration maximum median, and mean host union across 20 measurements were 10.828, 14.273, and 34.839 ms on the 1,024-row single-source fixture. Values on the mixed-source fixture were 1.994, 3.664, and 6.508 ms. The compact single-source arm requested the card lock at 09:30:18 UTC. The end-to-end comparison requires the two ordinary Serving arms.

The compact single-source arm exited 0 and passed the same three tensors. The measured values were 11.086, 15.008, and 33.282 ms. The compact mixed-source arm requested the card lock at 09:46:41 UTC.

The compact mixed-source replay exited 0 and passed all three tensors. Fastest-rank median, per-iteration maximum median, and mean host union were 2.136, 3.810, and 6.647 ms. All four replay arms have passed. The native lookahead Serving arm requested the EP4 lock at 09:54:23 UTC.

At 10:19:03 UTC, the native lookahead Serving log reported a successful health check after 296.044 s. The timed requests had yet to report results. The existing suite continues to own both lookahead arms.

## Prefill 26 · Use the fused source row plan in the two-slot path

**Candidate.** Apply the measured Prefill 19 source-planning change on top of Prefill 20. Start the aligned row prefix at the current receive slot's base on the sole source and its receivers. The source writes its own counts and starts from its cursor and skips the separate planner and the self histogram store. Peer histogram rows, payload epoch, window release and layer completion retain their Prefill 20 semantics. This isolates the row-plan reduction when two communication slots are available. Separate measured effects may interact.

The 80-row slab and full Serving compact compiled on the host with PTOAS 0.72 and 80-row Decode. Only `prefill_moonep.py` differs from Prefill 20, with SHA-256 `5c0f33ed28c1c25dfb186fb57539b38f8b337451347d81b005577e6e46188ff3`. Source and compiler logs were verified and staged at `verification/ep4/experiments/prefill-26`. Suite `p26-20261121T071321Z` (PID 3206363) waits for Prefill 24 and checks completed 80-row reports for the target. It then runs both 80-row goldens and fixed-workload Serving against Prefill 20 compact and the same-width native control. Device work is pending.

The retry `p26r-20261121T074020Z` passed all three 80-row single-source tensors. The fastest-rank median, per-iteration maximum median, and mean host union were 8.731, 12.195, and 32.922 ms. Prefill 20 measured 9.234, 11.841, and 40.611 ms on the same fixture. The per-iteration maximum median rose 2.99% and the mean host union fell 18.93%. The mixed-source arm requested the same lock at 08:26:23 UTC.

## Prefill 27 · Give compact single-source layers a third communication slot

**Candidate.** Start from the measured two-slot Prefill 20 snapshot. Test whether allowing another 80-row round into the pipeline improves overlap between compact dispatch, routed experts, and result return. Activate the third slot only at the default EP4 width for single-source chunks. Give it a separate receive segment, return range, local scratch, and activation, scatter-completion, and release counter lanes. Its histogram counter uses the next padded peer row, which is free when only one source publishes. The existing two slots retain their original addresses. Each slot waits for its own prior occupant and the layer joins all three before proceeding. Mixed-source chunks still use slot zero. Width, route order, expert tiles, and bytes sent per token stay fixed.

The compact receive reservation grows from 6,912 to 10,368 rows. The third slot adds 17.61 MiB per rank to activation and its metadata, 4.69 MiB to the return window, and 13.28 MiB to the explicit slab tensors. These sizes describe the fixed tensors. Total device memory also depends on how many per-round temporary tensors are live. Compare directly with Prefill 20 to see whether more concurrency outweighs the increased working set.

**Host validation.** The DSL rejected a conditional expression for the array extent and lowered an alternate local conditional assignment to a scalar. Pass the existing static allocation count from each caller into the slab function. Both 80-row slab builds and the complete 80-row native and compact Serving builds then passed with PTOAS 0.72 and 80-row Decode. Generated code indexes the three completion IDs by round modulo three for the sole source path and joins those IDs before clearing the signals. The remote sources and logs are under `verification/ep4/experiments/prefill-27`, with `lib-source.sha256` verified before and after staging. Suite `p27-20261121T072328Z` (PID 3263172) waits for Prefill 25, checks earlier 80-row reports for the target, then runs the two frozen goldens and fixed Serving workload. Device latency is pending.

## Prefill 28 · Pull the compact expert outputs from their owners

**Candidate.** Start from the verified Prefill 20 two-slot source. For default-width EP4 compact rounds with one source, W2 writes BF16 results into the active slot of the local symmetric return window in expert and active-row order. The source's histogram builds one compact address per route from the unpadded expert prefix on its owner and the route's ordinal within that expert. The source's existing combine task waits for W2 completion on all owners and pulls each BF16 row in the original TopK order before FP32 accumulation. Keep the return and reuse epochs. The owner-side scatter task and the corresponding route descriptor stores are omitted in these rounds. Mixed-source rounds continue to use scatter to the destination layout.

Each 80-row slot reserves 1,920 return rows for the existing W2 output shape, compared with 480 in Prefill 20. Across two slots this adds 28.125 MiB per rank of fixed return window and replaces the W2 temporary output for the direct branch. The timed comparison measures whether removing a separate dispatch compensates for the added source-side descriptor work, concentration of reads on the source rank, and tracked writes to the return window. Tile sizes and the request workload stay the same.

**Host validation.** Explicit caller-owned output tensors and individually typed local and remote loads resolved the compiler's output inference and tile view errors. A manual dependency scope made the W2 completion ID inaccessible to its consumer and was removed. The compiled graph passes the selected symmetric view to W2, feeds its completion ID to the existing notification, and joins the notified data and local preparation before the remote reads. Both 80-row slab compiles and complete native and compact Serving compiles passed with PTOAS 0.72 and 80-row Decode. `host-codegen.txt`, the patch, and both source manifests are archived in `verification/ep4/experiments/prefill-28`. The source hash verification succeeded against Prefill 20 before applying the patch and against the compiled snapshot after applying it.

Suite `p28-20261121T075615Z` (PID 3423109) follows the Prefill 27 retry. It checks the existing same-width target reports, then runs both frozen goldens and the fixed Serving workload with Prefill 20 compact and Prefill 16 native as controls. Device correctness and latency are pending.

The archived two-slot suite exited 0 after both frozen fixtures and Serving. Pulling results at the source measured 1.927419 s warm Prefill P50, up 5.41% from the 1.828501 s Prefill 20 compact control. The comparison is `p28-20261121T075615Z-comparison.json`. This result informs the return operation in Prefill 58.

## Prefill 29 · Load source route weights once per push block

**Evidence and candidate.** Diagnostic 22 measured 45.354 ms of active source push in the 80-row two-slot path. In the generated push kernel, each nonempty expert assigned to a block reloads the complete 80-by-8 route-weight tile from local memory. Keep the tile live across the block's expert loop and read it once. A block with no nonempty assigned experts now also loads the tile. The new live range and the load on idle blocks need to be weighed against the removed reads. Keep the worker count, route traversal, payload and notifications from Prefill 20.

The frozen single-source slab and full compact Serving graph compiled at 80 rows with PTOAS 0.72 and 80-row Decode. Generated AIV code places the weight `TLOAD` just before the expert loop and keeps the position and payload loads in the loop. The verified candidate changes only `prefill_moonep.py`, SHA-256 `d812903bad3be943cc79cd86d0e7c8ffc097fd11e368cabc069c6e464b2de54e`. The exact patch, manifests, and compiler results are archived under `verification/ep4/experiments/prefill-29`. Suite `p29-20261121T080314Z` (PID 3456704) follows Prefill 28 and measures the default-width goldens and ordinary Serving against Prefill 20 compact and Prefill 16 native. Device correctness and latency are pending.

## Prefill 30 · Reuse source activations across the routes of each token

**Evidence.** On 1,560 recorded 80-row single-source layer-rounds, the 56-block expert partition assigned the busiest block a median 63 routes and a P95 of 101. With 40 token blocks the busiest block gets 12 at both percentiles. The sum of the busiest-block route counts falls by 81.46% across 736,800 route offers. `analyze-prefill-30-push.py` reuses the recorded layer histograms and writes `prefill-30-load-model.json`. These are work counts. The device compiler and run determine the cost of the less-contiguous destination addresses and the benefit of scheduling fewer blocks.

**Candidate.** Start from Prefill 20. On a single-source 80-row compact round, record each route's ordinal within its expert in the source histogram. The push task assigns at most two tokens to each of 40 blocks. Load the FP8 activation and unpacked scale once per token, then send to each of its six existing packed destinations. Keep the return descriptors, FP32 weight bits, expert offsets, traffic bytes, epochs, and output order. The mixed-source path retains its expert-based push.

An initial code-generation attempt reused the fallback task ID as a scope name. Binding the two task IDs separately before assigning the shared notification dependency resolved SSA verification. Both frozen slab shapes and full compact Serving compiled at 80 rows with PTOAS 0.72 and 80-row Decode. Generated AIV code loads activation and scale outside the six-route loop and the graph uses 40 push blocks for a sole source. Only `prefill_moonep.py` differs from Prefill 20, SHA-256 `cce641dd61e49b17aa9d10590b6212e7b3230f5c9cfe6f87f7243a998efbc1e7`. The source and compiler summaries were verified and staged at `verification/ep4/experiments/prefill-30`.

Suite `p30-20261121T080854Z` (PID 3501534) waits for the active Prefill 23 run. It uses the same device lock as Prefill 24 and the other queued suites. This prioritizes the observed source-push bottleneck. The suite runs both 80-row goldens and Serving against Prefill 20 compact and the same-width native control. The single-source arm took the device lock after Prefill 24's single-source replay and was compiling on the server at 08:16:21 UTC.

The 80-row single-source replay passed all three tensors and exited 0. After five warmups, 20 measured replays had a fastest-rank median of 8.395 ms, a per-iteration maximum median of 10.453 ms, and a mean host union of 36.165 ms. The same Prefill 20 fixture recorded 9.234, 11.841, and 40.611 ms. The new maximum-rank median fell 11.72%. The mixed-source replay requested the lock at 08:18:43 UTC.

The mixed-source replay also passed all three tensors and exited 0. Its fastest-rank median, per-iteration maximum median, and mean host union were 2.154, 3.455, and 6.260 ms versus 2.142, 3.835, and 7.246 ms for Prefill 20. The default-width fixed-workload Serving arm started at 08:23:45 UTC and the result is pending.

The Serving arm and suite exited 0, and all 12 pinned requests, the long answer, and both eight-request waves passed. Startup to health was 304.105 s. The normalized CLI and request IDs match Prefill 16 and Prefill 20. Prefill P50/P95 were 1.924/3.800 s and TTFT P50/P95 were 2.003/3.884 s. The candidate improved P50 by 14.02% versus native. Its P50 was 5.22% slower than Prefill 20 and 10 of 12 matching requests took longer. At 1,024, 3,070, and 6,142 tokens, the medians were 0.694/1.924/3.750 s versus 0.633/1.829/3.708 s for Prefill 20. The regression was most pronounced in the shortest prompt group. Retain Prefill 20 as the ordinary Serving leader. The formal result is in `p30-20261121T080854Z-comparison.json`.

## Prefill 31 · Overlap local post-mix with the return-window release

**Candidate.** In Prefill 20 the local HC post-mix waits for peers to acknowledge release of the distributed return window. It consumes the local result already produced by combine reduction and the local HC coefficients. At 80 rows in the two-slot, single-source path, let post-mix start from those two producers while the release handshake runs. Add the release task to the existing slot completion latch with the two local result copies, so the next occupant cannot use the window or scratch early. The layer still joins both slots before clearing the signals or advancing. Mixed-source and native post-mix retain their original dependency.

The compiler could not type the release ID as a returned dependency. Passing a one-element `TASK_ID` array from the slab caller preserved the existing tensor return type and made the release edge explicit. Both 80-row slab compiles and full compact Serving with 80-row Decode passed on PTOAS 0.72. Generated orchestration selects combine completion for post-mix on the single-source compact path and keeps peer release in the slot latch. Only `prefill_moe.py` changed from Prefill 20, SHA-256 `112c81abbbfa1b366651ddc3e6b25e7705ad617ccdc618ee5e393974f4c9bb04`. The checked source is staged under `verification/ep4/experiments/prefill-31`. Suite `p31-20261121T081259Z` (PID 3522935) follows Prefill 29 and checks the earlier default-width reports before both goldens and fixed-workload Serving. Device correctness and latency are pending.

## Prefill 32 · Fuse the route descriptor in both communication slots

**Candidate.** Prefill 23 reduced 80-row Serving Prefill P50 by 3.10% relative to the one-slot compact control. Apply only its descriptor send and receive changes to the faster two-slot Prefill 20 path. The existing route descriptor carries the FP32 weight bits in word three, and the compact scale-pack task extracts them in the same expert-row order. This removes one per-route remote store while preserving the aligned per-slot ranges and their readiness epochs. Test the incremental effect with Prefill 20 compact as the control.

The 80-row single- and mixed-source slabs and full compact Serving passed host compilation with PTOAS 0.72 and 80-row Decode. The verified patch changes just `prefill_moonep.py` from Prefill 20, SHA-256 `36e1f19fd3f3009aac2268123518864028fdb72924e04fc010423f2f66c6cc2e`. The source, patch, and compiler summaries are staged under `verification/ep4/experiments/prefill-32`. Suite `p32-20261121T081606Z` (PID 3539689) follows Prefill 31 and checks the preceding default-width results before taking the device lock. Device correctness and latency are pending.

## Prefill 33 · Give W13 and W2 a plan containing only active experts

**Evidence.** Across 1,560 single-source 80-row layer-rounds and all four ranks, a median 13 of 96 experts received rows. The mean was 13.014 and the P95 was 22. There were 81,210 active rank-expert entries out of 599,040 allocated entries. Each W13 block currently acquires both a count and a compact input start for every local expert. Each W2 block reads all local counts again. The model implies 14 packed plan reads per block on average, including the plan header, versus 192 current metadata reads in W13 and 96 in W2. The counts are in `prefill-33-expert-counts.json`.

**Candidate.** Start from two-slot Prefill 20. In the existing receiver row plan for a default-width EP4 single-source compact round, append an entry only for a nonempty expert. Each word packs count, local expert ID, and aligned activation start. The 7-bit count holds up to 80 and the 7-bit expert field covers 96 IDs. Keep the original expert-ID order and compact output accumulation order. W13 and W2 enumerate the packed plan and use its expert IDs to choose the same weight slices. The original count and start arrays serve scale pack, SwiGLU, and scatter. Native and mixed-source kernels use the dense metadata path.

Both slab shapes and full 80-row native and compact Serving generated A5 code with PTOAS 0.72 and 80-row Decode. An explicit INT32 cast kept the plan index stable in the DSL loop. The full source hashes were checked before and after applying the patch in `verification/ep4/experiments/prefill-33`. On the server the CANN toolchain built both AIV and AIC versions of W13 and W2. `ccec.log` records the four binaries. Suite `p33-20261121T082414Z` (PID 3607696) waits on Prefill 30 and uses the same card lock and frozen fixtures as earlier experiments. Measure its incremental effect against the verified two-slot Prefill 20 control. Device validation is pending.

Both 80-row frozen device replays exited 0 and passed `next_pre_mix`, `x_mixed`, and `x_next`. Across 20 measured iterations after five warmups, single-source fastest-rank median, per-iteration maximum median, and mean host union were 7.353, 9.316, and 34.499 ms. Prefill 20 recorded 9.234, 11.841, and 40.611 ms. The maximum median fell 21.33% and the host union fell 15.05%. The mixed-source maximum median was 3.808 ms against 3.835 ms. The ordinary Serving arm took the EP4 lock at about 09:18 UTC after the 16K sweep and the Prefill 25 native single-source replay.

Serving reached health in 306.063 s and the suite exited 0. All 12 timed requests produced 13 output tokens, the long answer was correct, and both eight-request waves passed. The launch command matches both controls after allowing for the backend selector and port. Prefill P50/P95 were 2.182/3.941 s and TTFT P50/P95 were 2.263/4.026 s. Prefill P50 increased 19.33% relative to Prefill 20 compact and fell 2.49% relative to Prefill 16 native. Only one of 12 paired requests improved against Prefill 20. The three length-group Prefill medians were 0.998, 2.182, and 3.932 s. Keep Prefill 20 as the default-width Serving leader. The device replay and Serving results support a separate analysis of the real-model task path before crediting further sparse-plan changes. See `p33-20261121T082414Z-comparison.json`.

## Prefill 34 · Reuse the active expert plan in scale packing and scatter

**Candidate.** Start from Prefill 33. Use the packed local plan to bound the scale-pack and scatter loops in each 80-row EP4 compact single-source round. Both read count and aligned start from the same entry used by W13 and W2. Scale pack assigns the active entries round-robin to its original 32 blocks. Scatter walks the sorted entries and produces the same cumulative active output offsets. The native and mixed paths preserve dense counts and starts.

Both slab shapes and full compact Serving generated A5 code with PTOAS 0.72 and 80-row Decode. Only `prefill_moonep.py` changed from Prefill 33, SHA-256 `92d7043101e5dcfc3c729d9fffd8865663698dc10c4149526f7276cdf10ddfca`. The server checked the original and candidate manifests and staged the patch at `verification/ep4/experiments/prefill-34`. Suite `p34-20261121T082704Z` (PID 3623891) waits for a completed and validated Prefill 33 comparison. It measures the two frozen cases and fixed 80-row Serving against Prefill 33 compact and Prefill 16 native.

The Prefill 34 single-source replay exited 0 and passed the three frozen tensors. Fastest-rank median, per-iteration maximum median, and mean host union were 7.289, 10.245, and 27.700 ms. The maximum median increased 9.98% and mean host union fell 19.71% relative to Prefill 33. The mixed-source arm requested the EP4 lock at 09:41:25 UTC.

The mixed-source replay also exited 0 and passed all three tensors. Its three timing statistics were 2.145, 3.770, and 6.608 ms. The Serving process took the EP4 lock near 09:54 UTC after the Prefill 25 compact mixed-source replay.

The Serving arm and suite exited 0. All 12 requests generated 13 tokens and stopped at EOS. The 3,070-token answer and both eight-request waves passed. The recorded CLI matches Prefill 16 native and Prefill 20 compact after normalizing the port and backend selector. Warm Prefill P50/P95 were 1.764/3.841 s and TTFT P50/P95 were 1.850/3.977 s. Prefill P50 fell 21.17% against Prefill 16 native, 3.52% against Prefill 20 compact, and 19.15% against Prefill 33. The 1,024-, 3,070-, and 6,142-token group medians were 0.602, 1.764, and 3.548 s. Ten of 12 requests improved against Prefill 20. Prefill P95 rose 1.44% against Prefill 20. This is the lowest measured P50 on the original default-width workload. See `p34-20261121T082704Z-comparison.json` and `p20-20261121T060336Z-comparison.json`.

## Prefill 35 · Carry the active row total in the same expert plan

**Candidate.** Start from Prefill 34. Add the total active rows in the high bits of the plan header. The low seven bits contain the active expert count used by W13, W2, scale packing, and scatter. SwiGLU uses the total directly in the compact 80-row single-source branch, removing a separate load and reduction of all counts on each vector block. The receiver planner writes the packed plan for these rounds and the dense count and start arrays for the other layouts. The expert order, operand layouts, and result order are the same as Prefill 34.

Both slab shapes and full native and compact Serving generated A5 code with PTOAS 0.72 and 80-row Decode. The server verified the patch against Prefill 34 and compiled W13 and W2 for AIV and AIC and SwiGLU for AIV with CANN. The SPMD dependency on plan completion is explicit through dispatch and routed W13. The source manifests and five CANN build results are in `verification/ep4/experiments/prefill-35`. Suite `p35-20261121T083031Z` (PID 3645407) waits for a completed and validated Prefill 34 comparison. Its primary incremental control is Prefill 34 compact. Device validation is pending.

## Diagnostic 36 · Explain the token push regression in the real model

The 80-row token push reduced the single-source frozen replay maximum-rank median by 11.72% and increased warm Serving Prefill P50 by 5.22% against Prefill 20. The 1,024-token prompt group regressed most. Run the real-model Diagnostic 22 workload on Prefill 30 with the same single 1,024-token prompt, one output token, and 80-row EP4 profile build. Compare with the existing Prefill 20 dual-slot capture. Use the saved rank paths and core records to inspect source route metadata, push span, expert overlap, and rank release gaps. A compile-only combination of token push and early post-mix from Prefill 31 passed both slab builds and full Serving. Its device trial was deferred after the standalone regression.

The device arm and suite exited 0 after writing all four complete traces. Both captures returned one output token from the 1,024-token prompt, with the same recorded command after normalizing the port and profile path. Both resolved every task name and recorded 520 MoE rounds on each rank. Prefill 30 source push activity fell from 45.354 to 16.989 ms. Its overlap with routed experts fell from 9.267 to 3.868 ms, leaving 13.121 ms of push outside that overlap compared with 36.087 ms in Prefill 20.

The source-rank span fell from 493.932 to 475.846 ms. Rank 1 became the slowest at 490.338 ms, up from its previous 485.622 ms. The maximum across ranks fell only 0.73%. The source path's gaps before result transfer and reduction grew from 28.510 to 33.821 ms. Rank 1 still had 57.446 ms of gaps before route metadata and 35.048 ms before result transfer and reduction. The task path suggests investigating receiver readiness and the result phase along with source push. The ordinary 12-request replay remains the end-to-end latency result. See `diag36-20261121T083703Z-compact-{paths,core-spans}.json`.

Suite `diag36-20261121T083703Z` (PID 3666735) waits for Prefill 33 and skips the trace when a prior 80-row report reaches the target. It uses the existing device lock, Diagnostic 22 driver, and both timing analyzers. The script is staged at `verification/ep4/experiments/diag-36`.

## Default-width queue recovery

After the first Diagnostic 22 native startup failed, the original Prefill 23, 26, 25, and 27 first golden invocations each exited 3 with `Cards 4-7 are occupied`. Prefill 24 stopped at the gate requiring a completed Prefill 23 comparison. None of these invocations produced a timing or correctness result. The original logs and exit files remain available.

At 07:40:20 UTC, restarted the same six suites with the suffix `20261121T074020Z` under `experiments/ep4-retry-20261121T074020Z`. `run-map.txt` pairs the original and retry labels. The queue runs `diag22r`, `p23dr`, `p24r`, `p26r`, `p25r`, and `p27r` in that order. The retry copies preserve the trained model, 80-row configuration, fixture inputs, card scope, and comparison gates. The existing device lock still covers each arm. When the selected cards are occupied, the retry runner releases the lock and checks again after 30 seconds. No other jobs were stopped. At 07:42 UTC the six new workers were live, and the native diagnostic had compiled the Prefill and mapped the weight sidecar.

## Replica placement estimate on captured routes

Use the seven Prefill 07 captures to count 64-row expert tiles. For each layer-round, evaluate moving tiles of one expert from its home rank to another rank and calculate the new maximum rank tile count. Count equal tiles with zero transfer or planner cost. The adaptive estimate picks the best expert and destination each round. The fixed estimate picks at most one expert and destination per layer using only the first 1,024-token chunk, then scores that assignment on the two subsequent chunks and the four mixed-source captures. The fixed estimate can vary how many tiles it moves in each round. It assumes the selected weights are resident.

| Width and evaluation set | Fixed replica tile reduction | One adaptive replica, no copy cost | Ideal balance of all current tiles |
| --- | ---: | ---: | ---: |
| 80, held-out chunks of the same long request | 3.46% | 5.72% | 18.34% |
| 80, mixed-source chunks | 1.98% | 5.99% | 18.89% |
| 256, held-out chunks of the same long request | 5.02% | 7.95% | 18.17% |
| 256, mixed-source chunks | 2.19% | 5.99% | 18.89% |

The percentages compare sums of maximum rank tile counts across layer-rounds. At 256 rows, keeping the selected fixed replicas would require 22, 11, 5, and 2 expert copies by rank, or up to 394.45 MiB on a single rank. The single-replica route estimate captures only a fraction of the available balance even before accounting for planning or copy costs. Defer an online single-replica tuning run while the direct transport candidates are being measured. `analyze-replicas.py` and `replica-tile-summary.json` in `verification/ep4` preserve the method, counts, and selected assignments. This is a tile-count estimate on the captured workload. End-to-end latency depends on the observed execution order and actual kernel costs.

## Default-width prompt length sweep through 16,384 tokens

Run Prefill 16 native and Prefill 20 MoonEP compact with the pinned Prefill 15 Serving snapshot. Both use 80 rows per MoE round, a 1,024-token Prefill chunk, a 16,640-token context limit, TP1/DP4/EP4, 128-token KV blocks, the same EP4 sidecar, and empty-tail trimming. Prefix caching and DSpark are disabled. A single card-lock scope runs the native and compact Serving processes consecutively.

The shared workload has four requests at each of 1,024, 3,070, 6,142, 8,192, 12,288, and 16,384 actual input tokens. It retains the original three short-length groups and extends the same long-context arithmetic prompt with three new length groups. The V4.1 chat tokenizer checked all input lengths before scheduling. Each arm warms up at 1,024 and 16,384 tokens, then sends the 24 measured requests serially with greedy decoding and a 16-token output cap. The driver requires exactly one server Prefill histogram observation per request and checks the length against returned usage. It records time to the first emitted text or reasoning token from the HTTP stream.

The run directory is `verification/ep4/b16k-20261121T085000Z`. Its `workload.json` has SHA-256 `1b57f626bdd0e59050a5d002865cac95a664412f091c85031fc4d32f1e6e8722`. The driver and wrapper are `benchmark-16k.py` and `prefill-16k.sh` in the verification directory. The latter delegates card acquisition to the existing retry runner.

The suite exited 0. Both arms generated exactly 13 tokens on all 24 timed requests and stopped at EOS. The 3,070- and 16,384-token answer checks returned `323`, and both eight-request concurrent waves passed on each backend. The launch commands match after normalizing the port and MoonEP selector. Prefill and TTFT both improved in all 24 matched requests.

| Input tokens | Native Prefill P50 (s) | MoonEP Prefill P50 (s) | Prefill reduction | Native TTFT P50 (s) | MoonEP TTFT P50 (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.798 | 0.741 | 7.17% | 0.873 | 0.815 |
| 3,070 | 2.194 | 1.993 | 9.17% | 2.272 | 2.071 |
| 6,142 | 4.323 | 3.813 | 11.81% | 4.408 | 3.896 |
| 8,192 | 6.047 | 5.025 | 16.90% | 6.135 | 5.113 |
| 12,288 | 9.006 | 7.631 | 15.27% | 9.104 | 7.727 |
| 16,384 | 12.056 | 10.286 | 14.69% | 12.161 | 10.390 |

The absolute Prefill P50 savings grow with input length, from 57 ms at 1,024 tokens to 1.771 s at 16,384 tokens. The largest median percentage reduction is 16.90% at 8,192 tokens. Between 8,192 and 16,384 tokens, each additional input token adds about 0.734 ms to native Prefill and 0.642 ms to MoonEP Prefill. At 16,384 tokens, the median rates are 1,359 and 1,593 input tokens/s. The corresponding Prefill P95 values are 12.270 and 10.341 s, and TTFT P95 values are 12.374 and 10.444 s. With four requests per length, P95 is the observed maximum. The original short-context run used an 8,320-token context limit, one warmup request, and a different request order. The sweep identifies performance by length under the new matched setting. Raw per-request times and the normalized comparison are archived in `{native,compact}/http.json` and `comparison.json`.

### MoonEP transport and synchronization opportunities at 80 rows

In the three existing single-source real-model captures, 1,560 layer-rounds produced 736,800 TopK routes. Of these, 558,268 went to another rank. There were 304,282 distinct pairs of a source token and a remote owner, counted separately per layer. Prefill 20 sends 5,280 logical activation bytes per route, including 160 scale bytes, and returns 10,240 BF16 bytes per route. Sending the activation once per remote owner would reduce the forward activation payload by 45.50% and the forward-plus-return payload by 15.48% with per-route results preserved. These are logical byte counts from the captured routes. They exclude padded descriptors, physical transfer overhead, copy costs, and time. The calculation is archived as `route-payload.py` and `route-payload.json` in the sweep run directory.

Prioritize removing the single-source histogram rendezvous and receiver row-plan task. The source already computes the route counts and can produce an ordered compact plan for each owner with the aligned row start and count for each active expert. The existing payload-ready epoch can publish that plan with the activation data. Keep window reuse dependent on the completion of its consumers and preserve the mixed-source path. Prefill 26 moved source offsets into histogram publication and retained the receiver rendezvous and plan. It passed both frozen replays and Serving. Its warm Prefill P50 was 1.879 s, up 2.74% from Prefill 20, and P95 was 3.740 s, down 1.23%, in the original 12-request workload. See `p26r-20261121T074020Z-comparison.json`. A new test needs to remove waiting from a measured path and account for the added source work.

The paired 1,024-token capture measured 36.087 ms of source push outside routed expert execution and a 28.510 ms reconstructed gap before result transfer and reduction. These describe different timing views and cannot be added as recoverable latency.

Keep the returned BF16 values and the source's FP32 TopK accumulation order. Prefill 28 tests dropping an intermediate result pass by writing W2 outputs directly to a symmetric window for the source to combine. The alternative should be evaluated against both link traffic and the new pull dependencies. Activation deduplication comes after the critical-path work. It needs an input layout that lets the expert consume a shared activation without re-expanding it through a separate receiver copy. The 256-row Prefill 15b deduplication candidate with receiver expansion regressed 4.88% against its same-width compact control. In the 80-row Prefill 30 experiment, reusing local loads and keeping the wire payload unchanged improved frozen replay timing but regressed Serving P50 by 5.22%.

## Prefill 37 · Publish each active expert plan with its activation payload

Start from Prefill 35 and specialize the 80-row EP4 single-source compact path. The source's histogram task produces its send offsets and each owner's ordered active expert plan from the same route counts. Each 104-word tile occupies that round's row of the existing symmetric metadata window. The payload-ready epoch covers both the plan and the activation. Receivers skip their local histogram submission, histogram wait, and row plan. The source skips the separate row plan. Scale packing and expert kernels read a view of their slot in the window. The order of experts, starts, weights, BF16 results, and window reuse epochs is preserved. The mixed-source path uses the existing dense histogram.

Both default-width slab shapes and full native and compact Serving generated A5 code with PTOAS 0.72 and 80-row Decode on the host. The emitted graph guards the histogram wait and row plan by the dense-path predicate and passes the slot-specific view to the consumers after the payload wait. The server verified the base and candidate source hashes before staging `verification/ep4/experiments/prefill-37`. Suite `p37-20261121T091611Z` is queued after Prefill 35. It compares the two default-width goldens and original fixed-workload Serving with Prefill 35 compact, Prefill 20 compact, and Prefill 16 native. Device correctness and performance remain to be measured.

## Prefill 38 · Reuse interior single-source slots on the next payload epoch

The paired 1,024-token trace places 3.626 ms of compact source tasks and 11.437 ms of preceding gaps in the combined return and reuse signal group. This experiment addresses the slot-release notification and wait within that group. Prefill 18 narrowed the notification fanout while preserving a separate release epoch. Prefill 31 overlaps the release with post-mix. Prefill 38 skips both release tasks on rounds with a known same-slot successor.

The compiler assigned the same callable IDs to return and reuse notifications and waits. The first extract grouped them by callable name in `experiments/prefill-38/diag22-source-path-detail.json`. To distinguish the stages, follow each notification's predecessor. Scatter completion identifies the result notification. Reduction completion identifies the reuse notification. Each wait inherits the phase of its notification dependency. All 520 tasks per phase and rank were resolved. On the reconstructed source path, reuse contributes 2.514 ms of task time and 7.549 ms of preceding gaps. Return contributes 1.112 ms of task time and 3.888 ms of preceding gaps. The dependency-classified report is `experiments/prefill-39/diag22-signal-deps.json`. Path duration and preceding gaps describe the observed schedule. The unprofiled comparison measures the change in request latency.

Start from verified two-slot Prefill 20. On default-width EP4 compact rounds with a sole source, all ranks finish reading the activation slot before their scatter-completion notification. The source waits for those notifications and accumulates all returns in the original TopK order. Its own completion still gates reuse of that slot by the next round. Inactive ranks finish their local scratch readers before reuse and wait for the next payload epoch before reading the receive slot. The source is the only rank receiving routed results in these rounds. A full 1,024-token chunk skips the extra release on 11 of 13 rounds. The last occupant of each slot runs the existing release protocol at the layer boundary. Mixed-source chunks, native execution, other widths, and scatter-completion signals use the existing graph.

The 1,024-row single-source slab, 128-row mixed-source slab, and full native and compact Serving builds passed host code generation with PTOAS 0.72 and 80-row Decode. Generated orchestration skips both release tasks for single-source loop indices 0–10 and connects post-mix to combine completion on those rounds. The server matched the Prefill 20 file before applying the patch and verified the candidate `prefill_moe.py` SHA-256 `61a87cc5925ebef9058729fe025b910e977c9d315228acf49dc7abedf3dd70c0`. The patch, source hashes, and compiler checks are under `verification/ep4/experiments/prefill-38`. Suite `p38-20261121T092210Z` (PID 3809101) is waiting for Prefill 33. It measures both 80-row frozen fixtures and the 12-request Serving workload against Prefill 20 compact and Prefill 16 native. Device correctness and latency remain pending.

After Prefill 33 exited 0, the Prefill 38 single-source replay requested the card lock at 09:27:56 UTC. The suite and its card runner remained live during the Prefill 25 replays.

The single-source replay then exited 0 and passed all three frozen tensors over 13 rounds. After five warmups and 20 measurements, fastest-rank median, per-iteration maximum median, and mean host union were 8.698, 12.483, and 32.747 ms. The maximum median increased 5.42% and mean host union fell 19.36% relative to Prefill 20. The mixed-source arm requested the lock at 09:43:58 UTC.

The mixed-source replay exited 0 and passed all three tensors. Fastest-rank median, per-iteration maximum median, and mean host union were 2.148, 3.688, and 6.528 ms. The ordinary Serving arm requested the card lock at 09:51:45 UTC.

Serving reached health in 302.115 s and the suite exited 0. All 12 measured requests generated 13 tokens and stopped at EOS. The long answer and both eight-request concurrent waves passed. Prefill P50/P95 were 1.953/3.801 s and TTFT P50/P95 were 2.032/3.898 s. The normalized command matches Prefill 20. Prefill P50 increased 6.81% and P95 increased 0.38% relative to Prefill 20. Only one paired request improved. The three length medians were 0.705, 1.953, and 3.789 s. Against native the P50 reduction was 12.72%. Keep the release change isolated. Prefill 39 tests the effect of also restricting the result notification and dropping inactive combine work, with Prefill 38 as its incremental comparison. See `p38-20261121T092210Z-comparison.json`.

## Prefill 39 · Signal only the single-source rank before it combines results

Start from Prefill 38 and use the same default-width single-source rounds with an interior successor. Each receiver publishes scatter completion only to the source. The source's wait depends directly on its local scatter and still waits for three remote scatters before exact TopK-order FP32 accumulation. Inactive ranks join scatter-notification completion with local BF16 zero initialization and skip the peer return wait and empty reduction grid. The next payload epoch still gates reads of the reused activation slot. The final occupant of each slot and the mixed-source path retain the original return and release protocol.

A qualifying EP4 round uses three remote result notifications, three remote wait conditions, and one combine grid. Prefill 38 uses twelve notifications, twelve wait conditions, and four combine grids for this round. This experiment holds the activation path and result values fixed while testing the remaining return cost. The dependency-classified Prefill 22 capture establishes where to look if skipping the extra work changes the observed schedule.

Both frozen slab shapes and both complete Serving graphs compiled with PTOAS 0.72 at default width. Generated code gates the source and receiver tasks by role and retains the existing protocol in the mixed-source slab. The server verified the Prefill 38 base and candidate `prefill_moe.py` SHA-256 `fccbb67ded0c29e5a79e195b5223162acd52e47ac648876acf7c43723fd7ac80` under `verification/ep4/experiments/prefill-39`. Suite `p39-20261121T093235Z` (PID 3849082) waits for validated Prefill 38 results before device replay and the fixed Serving comparison.

## Prefill 40 · Balance source push with contiguous expert-row ranges

The three real-model route captures contain 1,560 sole-source 80-row rounds and 736,800 route offers. In Prefill 20 each of 56 push blocks handles complete experts, with 63 routes at P50 and 101 at P95 on the busiest block. Splitting each expert into contiguous eight-row ranges and assigning 14 workers to each of the four destinations lowers those figures to 17 and 24 in the route model. The sum of maximum worker routes falls 70.44%, and the median work list grows from 49 experts to 88 ranges. Four-row ranges drop the busiest-block P95 to 20 and require 143 median entries. Eight rows balances the additional plan entries with worst-block work. The analysis and input selection are in `analyze-push-strips.py` and `push-strips-report.json` under the experiment directory.

Compare this contiguous schedule against Prefill 20. The default-width sole source builds each destination's compact list of expert IDs and starting row indices in the existing planning task. Fourteen push blocks read each list round-robin. Within a work item the block sends consecutive rows of the same destination expert using the existing payload and descriptor stores. Expert row alignment, route order, transport bytes, and release epochs are preserved. The unchanged tile loads can occur once per range, which can offset the work balance. Serving verifies whether this schedule improves model latency.

The first host build rejected an untyped expert index as an SSA alias across the striped and expert branches. Explicit INDEX casts and a push-specific name resolved it. Both 80-row slab shapes and complete compact and native Serving passed host code generation with PTOAS 0.72. The server verified the base and candidate `prefill_moonep.py` SHA-256 `2424e7237c5e41fa0816ea5a43f6c0fdabe44280ae73b91faf828ab597daec2f` under `verification/ep4/experiments/prefill-40`. Suite `p40-20261121T094233Z` (PID 3884182) waits for Prefill 38, then runs both frozen fixtures and the fixed-workload Serving comparison against Prefill 20 compact and Prefill 16 native.

## Prefill 41 · Publish the dense receive plan with activation readiness

Diagnostic 36 moved the longest chip span to a receiver. Rank 1 still had 57.446 ms of reconstructed gaps before route metadata. Prefill 37 publishes the sparse plan from the source and inherits the expert kernels introduced in Prefill 33. This experiment isolates removal of the histogram rendezvous and per-receiver row plan on the faster Prefill 20 Serving baseline.

For default-width single-source EP4 compact, the source's existing histogram task computes the aligned send offsets and packs each receiver's 96 counts and starts into one INT32 tile. Seven low bits hold the count, which is at most 80. The high bits hold the absolute receive-row start for the active slot. The source writes one row into each owner's existing symmetric metadata buffer and uses activation-ready publication for the plan and activation. The separate histogram notification, histogram wait, and row-plan task are skipped. Receivers still wait for activation readiness before reading either input.

The existing scale-pack grid reads the window's plan words. Its first worker also vector-unpacks the 96 counts and starts into the same local tensors consumed by the unchanged routed expert kernels and scatter. This preserves input order and output accumulation. The plan uses the same 384 logical metadata bytes per owner as the earlier histogram. The mixed-source graph keeps its original histogram protocol and row planner.

The first host build rejected uses of a conditionally defined scale-pack count. Initializing the two scalar temporaries explicitly and naming them separately from the push loop resolved SSA validation. Both frozen slab shapes and both full 80-row Serving graphs passed with PTOAS 0.72. The emitted graph gates the histogram and row-plan tasks and places both metadata tensor stores in the existing scale-pack task. The server verified the Prefill 20 base and candidate `prefill_moonep.py` SHA-256 `bdba910ef29096b56c1581923c358dfd848ec174cb2aeb5ff6d850cf117890cc` under `verification/ep4/experiments/prefill-41`. Suite `p41-20261121T094654Z` (PID 3898387) waits for Prefill 38 before both frozen device replays and fixed-workload Serving against Prefill 20 compact and Prefill 16 native.

## Diagnostic 42 · Separate receiver planner gaps and adjacent-slot task edges

Reanalyse rank 1 of Diagnostics 22 and 36 at default width. The `moonep_row_plan` task accounts for 51.849 and 53.332 ms of gaps immediately before it on the two observed paths. Its own task contributions are 1.214 and 1.072 ms. In Diagnostic 36, the selected predecessor is `moonep_histogram_wait` for 28.227 ms of these gaps, including 17.381 ms attributed to a recorded local data edge. The rest of the path also includes same-core scheduling and unrelated predecessors. Both captures are level 1 and record neither scheduler task times nor cross-rank clock alignment. These gaps localize the stalled work before the planner, and their precise split between peer signals and local scheduling requires further capture.

Both dependency files have 519 tensor-map edges between consecutive `moonep_scatter` tasks, including the 480 adjacent round pairs within the 40 layers. The generated Prefill 20 submission passes the full return window as an InOut tensor. Those tasks write the distinct return rows of alternating slots inside each layer. The histogram also has 1,038 tensor-map records between neighboring occurrences. Same-name row plan and scale-pack edges connect tasks 46 positions apart in submission order. The observed row-plan gaps therefore do not establish adjacent-slot serialization by the planner itself. The scripts and JSON reports are archived under `verification/ep4/experiments/diagnostic-42`.

## Prefill 43 · Test the scatter-only manual submission scope

Start from Prefill 20 and keep W2 completion, return notification, combination, and slot release edges explicit. Applying `pl.no_dep(returned)` at a JIT kernel call failed distributed-tensor inference. A manual submission region around scatter compiled after placing the scatter task ID in a caller-owned one-element array, which kept the following notification's dependency visible across the scope boundary. Default-width single- and mixed-source slab builds, a 256-row compatibility build, and both full default-width Serving graphs passed host code generation. The compiled graph and source hashes were archived at `verification/ep4/experiments/prefill-43`. Suite `p43-20261121T100357Z` initially waited for Prefill 38.

Audit the exact saved Prefill 20 task graphs at rank 0 and 1 before running this variant. Use the task table order for corresponding rounds and dependency topology for reachability. Task IDs encode three runtime rings, so numeric ID order alone cannot prune predecessor walks. Projecting the scatter-only manual scope preserves all 520 W2-to-scatter, plan-to-scatter, and scatter-to-combine paths on each rank. It also preserves all 440 same-slot successor paths and removes the forced ordering of all 480 adjacent opposite-slot scatter pairs. Rank-local duration-weighted paths change from 140.446 to 139.923 ms on rank 0 and from 74.818 to 64.408 ms on rank 1. These are graph projections that hold the captured task times fixed.

The same audit found 1,040 creator-retention records into scatter on each rank. The manual scope disables creator retention along with tensor-map tracking. At 10:08:25 UTC, withdrew only the waiting Prefill 43 suite before it acquired the card lock. The host build, source, initial graph projection, corrected topology audit, and withdrawal record remain archived. The replacement uses slot-shaped views so the runtime continues to track the input tensor lifetimes.

## Prefill 44 · Track only the return rows of each slot

Build on the verified Prefill 34 Serving leader. Give both compact scatter and combine reduction a view of the return rows in their current slot. At the default width each view has 480 rows, derived from 80 source rows and six routes per row. Its starting row is `window_slot * 480`. Both the remote write and local read use a row index relative to the same view. The existing tensor tracker retains all input creators and continues to track overlapping views. The original return-notification and slot-release edges remain in the graph. This tests whether making the two slot ranges explicit opens useful overlap in the return phase.

Both default-width frozen shapes and full native and compact Serving compiled with PTOAS 0.72. The generated compact graph registers the scatter as InOut and the reduction as Input on their slot views. The device kernel unpacks the view's start offset before translating the base to the peer rank, so slot one addresses the same rows as the original absolute offsets. The server verified the Prefill 34 sources and both candidate hashes under `verification/ep4/experiments/prefill-44`. Suite `p44-20261121T101108Z` (PID 3979488) waits for Prefill 38 before its two frozen fixtures and fixed-workload Serving comparison with Prefill 34 compact and Prefill 16 native.

After Prefill 38 exited 0, the Prefill 44 single-source arm requested the device lock at 10:13:38 UTC. The suite and its card runner were both live at 10:14:39 UTC.

## Diagnostic 45 · Locate tensor-map waits across the two communication slots

Inspect the four archived Prefill 20 compact graphs from Diagnostic 22. Align same-phase task occurrences by the dependency task table and classify tensor-map wait records by layer, round, consumer dtype, and view shape. The 1,024-token request contains 40 layers of 13 rounds at width 80. This gives 480 adjacent opposite-slot pairs within layers for each phase that runs every round. Exclude the layer boundaries because the slab completion join orders those tasks.

| Phase | Ranks | Tracked window | Opposite-slot wait records per affected rank |
|---|---|---|---:|
| Histogram | 0–3 | INT32 signal `[4, 128]` and INT32 metadata `[4, 384]` | 960 |
| Activation push | 0 | INT8 activations `[6912, 5120]`, UINT8 scales `[6912, 160]`, FP32 weights and INT32 routes `[6912, 8]` | 1,920 |
| Activation publication | 0 | INT32 signal `[4, 96, 128]` | 480 |
| Return scatter | 0–3 | BF16 return `[960, 5120]` | 480 |

Histogram submits on receiver ranks even in sole-source rounds, where only the source writes histogram rows and publishes its signal. The source push tasks hand the runtime full payload tensors for both 3,456-row slots. These full-window arguments explain the adjacent automatic waits. The four activation tensors produce four records per push pair, and the two histogram tensors produce two per histogram pair. This count measures dependencies. The latency test must also account for remaining dependency paths and scheduling.

Prefill 44 isolates the return-window change. Prefills 37 and 41 test replacing the histogram rendezvous by publishing receive metadata with activation readiness. To evaluate additional slot views, first project the remaining rank-local graph after removing only the opposite-slot tensor-map waits for each window class, keeping explicit and creator edges. The four-rank count above completed. The graph projection and server-side archival were pending when the SSH control connection closed. The last confirmed device state was native lookahead past health with Prefills 35, 39, 40, 41, and 44 queued.

## Prefill 46 · Track activation writes by communication slot

Diagnostic 45 found 1,920 automatic wait records across 480 neighboring push pairs on the source rank. Start from Prefill 34 and register the four activation destinations with per-slot views in the default 80-row single-source path. Each view starts at `window_slot * 3456` and spans 3,456 rows. The push kernel subtracts that start from its absolute row before using the view. The activation, scale, weight, and route writes use the same payload and physical addresses in both variants. Mixed-source and other layouts use the existing full-window arguments. This iteration targets source push ordering. The publication, receive packing, and expert work still use the original dependencies and addressing.

The DSL rejected reassigning distributed tensors to views with a different type, even when the assignments were in separate branches. The compiled implementation shares the push kernel source between the two call sites. The generated A5 orchestration registers four InOut slot views in the single-source branch and the original full windows in the mixed branch. It carries the original plan, scale, and histogram dependencies into the push task and its task ID into the publication task. The AIV kernel applies each view's `start_offset` to its local base before peer translation. The store and scale put then add the relative row. A host check using the runtime's actual `Tensor::view` and `ChipTensorMapEntry::check_overlap` passed for all four payload shapes. Opposite slots were disjoint in both directions, and same-slot and whole-window consumers still covered each producer view. A whole-window read can retain dependencies on both slot writers.

PTOAS 0.72 passed both frozen 80-row shapes, a 256-row compatibility shape, and full 80-row native and compact Serving host code generation. The 80-row fixtures were 1,024 single-source rows and the 128-row mixed fixture `81,80,79,0`. Only `prefill_moonep.py` changes from Prefill 34. The applied patch matched candidate SHA-256 `9c645e31e5cdb1b14c0df393c34ad53c449776e676cfec80b437416489fc3bdd`. The local archive `/tmp/v41-prefill-46.54JTIi/prefill-46.tar.gz` contains the patch, manifests, compile logs, overlap check, and existing-suite staging script. Server staging and device verification await an authenticated SSH session.

The device comparison uses the two frozen fixtures, then the same 12-request 1,024-, 3,070-, and 6,142-token workload as Prefill 34. Prefill 34 compact is the incremental control. Use the read-back Prefill 25 native lookahead result to confirm the fair native reference before attributing a MoonEP-only gain. The host geometry check verifies one source of ordering. Device traces and the paired Serving run measure whether useful overlap and latency improve.

### Same-width attribution and target check

Recomputed the 12-request summaries from the archived Prefill 16 native, Prefill 20 compact, and Prefill 34 compact samples with the existing summarizer. All 12 Prefill 34 requests improved over their Prefill 16 native pairs with the same prompt-token counts, completion-token counts, and stop reason. Replayed the 16K summarizer against the archived 24 request pairs and reproduced the original report exactly.

| Prompt tokens | Prefill 16 native P50 (s) | Prefill 34 P50 (s) | Reduction | P50 for 30% (s) | Further reduction needed (ms) |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.804246 | 0.602261 | 25.11% | 0.562972 | 39.289 |
| 3,070 | 2.237811 | 1.764092 | 21.17% | 1.566468 | 197.624 |
| 6,142 | 4.445488 | 3.548256 | 20.18% | 3.111841 | 436.415 |

These thresholds use the original native reference. At the mixed-workload median, improving Prefill 34 from 1.764092 s to 1.566468 s requires a further 11.20% drop. Once the Prefill 25 native lookahead result is available, use 70% of that arm's P50 for the strict target. Compare Prefill 25 native with Prefill 16 native to measure the shared local-preparation opportunity on native. Compare Prefill 20 compact with Prefill 25 compact to measure the added effect of the independent transport slots. The direct comparison of the best compact candidate with Prefill 25 native holds local lookahead available in both backends.

The Prefill 46 local archive now includes a suite that uses the existing device runner and 12-request summarizer. It checks that the completed arms share the same command after normalizing the port and MoonEP selector. When the Prefill 25 native arm has exited successfully it is the native reference in the comparison. Otherwise the suite emits only the incremental report with Prefill 34 as its reference. The inherited shortcut for a prior 30% result was removed because it used reports based on Prefill 16 native. Shell syntax and both report layouts passed offline with real archived 80-row inputs. The command check accepted the matching 8,320-context arms and rejected substituting the 16,640-context native sweep. The revised archive SHA-256 is `540533a7afbe2725d950f4e3b7b5c2e562911065cbce906a7b7ec9b9cef6ea46`. The Prefill 46 source manifest is unchanged from the verified host build.

At 10:29 UTC, the server responded to a read-only connection check. The prior control socket was absent and the configured public-key connection failed authentication. The approval review had rejected retrieving credentials from the goal record. The latest confirmed device evidence is the Prefill 25 health check above. After restoring authenticated access, read back the existing suites before scheduling another device run. The Prefill 46 archive is verified locally and ready for staging.

## Prefill 47 · Batch contiguous writes inside MoonEP push

Use the hash-verified Prefill 23 compact source at the original 80-row width and one communication slot. The existing push kernel sends activation, scale, and fused route descriptor separately for every routed row. Its generated code fences each of the three writes. Rows assigned to one source and destination expert are already contiguous in the receiver's compact windows. The candidate assembles up to four such rows in each temporary push tile and then stores each of the three tiles. The final tile writes its actual valid row count. A source with `c` rows for an expert executes `3 * ceil(c / 4)` writes instead of `3 * c`. That count isolates the reduction in write and fence operations. The per-row loads and tile moves still cost time.

The candidate diff touches only `prefill_moonep.py`. It uses the same receiver row offsets, payload bytes, expert ownership, push worker partition, publication epoch, and serial round reuse. The compact receive and expert input layouts are preserved. The push tile extent is local to the communication kernel. The compiled slab continues to execute 13 rounds for a full 1,024-token chunk, with 80 rows per round and one completion slot.

The first host build exposed a tile-name conflict between push and scale packing. A distinct push scratch name resolved it. Candidate and control passed A5 PTOAS 0.72 code generation for the 1,024-token single-source slab, the `81,80,79,0` mixed-source slab, and the 80-row native slab. After normalizing debug comments and generated variable IDs, all 26 native kernels matched. The only changed compact kernel was `moonep_push` in both slabs. Compact orchestration changed only the position of the push scale input argument. The three generated remote stores each use the tail's actual row extent and keep their release fence.

The verified base `prefill_moonep.py` SHA-256 is `c9f0a7ffe96a0a6a47eb32229b54d3331bc3429f92f085cfc9065f881729a906`. The candidate SHA-256 is `e598d931664950e82252e0a0f140758fa6bc5dc5a1926ecc051a81cd6f2b33a1`. The local archive is `/tmp/v41-prefill-47.apzLYW/prefill-47.tar.gz` with SHA-256 `667218b55852b95d0a4b31113dc4c0630f47db983afe10b1a448d03e62fc9f6a`. Local staging passed the source checksums, reproduced the exact candidate, and generated the expected native, compact control, and candidate launch selections.

The packaged device suite uses the existing card lock and two frozen golden fixtures, then runs fresh Prefill 16 native, Prefill 23 compact, and Prefill 47 compact with the fixed 12-request workload. It checks recorded launch parameters and uses the existing Prefill summarizer. The server responded to the SSH probe and rejected the configured public-key login. Deployment, device correctness, and latency for Prefill 47 remain pending authenticated access.

Reused the Prefill 40 route-count report to estimate operation counts for the current push. The captured 1,560 single-source layer-rounds contain 736,800 route records and 223,403 groups of up to four adjacent records for the same expert. The latter count is independent of worker assignment. It corresponds to 670,209 fenced writes with Prefill 47 versus 2,210,400 with Prefill 23, a 69.68% reduction in operation count on those routes. Payload byte counts are identical. The estimate excludes the cost of assembling each tile and is not a timing result.

Full 80-row Serving native and compact host code generation also passed for the control and candidate. This local integration check reconstructed a single-slot Prefill entry from the archived Prefill 20 entry by restoring the original scratch and window extents and signal-clear signature. Both arms use that same local entry, the same single-slot slab, and the verified Prefill 23 or Prefill 47 push file. The remote package stages its patch on the actual Prefill 23 tree. Of 109 generated Prefill kernels in each layout, only compact `moonep_push` differed after normalizing comments and generated identifiers. All 45 Decode kernels matched in both layouts. Native orchestration matched. Compact orchestration changed only the push scale input argument position. The four logs and the comparison are in the local Prefill 47 directory under `serving-{native,compact}.log`, `control/serving-{native,compact}.log`, and `serving-codegen.txt`. The follow-up SSH probe returned the same authentication failure.

Authenticated access was restored on November 22. The server's Prefill 23 `prefill_fwd.py` and `prefill_moe.py` matched the reconstructed single-slot local inputs byte for byte, with SHA-256 `43994320dba5c1f9ea38495de6634da0854d2de08dc643995b9fd4af5248fe8c` and `3f00bb655d6303aadd149348e66fcd148c0d63af0bdbdb451f2b35206150d6d3`. The archive and both source manifests passed on the server. Staged under `verification/ep4/experiments/prefill-47`.

Suite `p47-20261122T141415Z` (PID 2316551) acquired the existing cards 4–7 lock and started the single-source frozen replay at 14:15:53 UTC. It runs both frozen replays and all three fresh Serving arms under one lock. The fixed-workload reports and correctness artifacts will be stored under the suite label. The configuration remains the fixed 80-row, one-slot acceptance setup.

The single-source replay exited 0 after passing `next_pre_mix`, `x_mixed`, and `x_next`. Across 20 measured iterations after five warmups, the fastest-rank median, per-iteration maximum median, and mean host union were 12.187, 15.364, and 37.354 ms. Prefill 23 recorded 12.592, 16.023, and 34.522 ms on this fixture. Mixed-source replay started at 14:18:31 UTC.

The mixed-source replay exited 0 and passed the same three tensors. The three timing statistics were 2.105, 3.443, and 6.873 ms, compared with 2.135, 3.277, and 6.790 ms for Prefill 23. The native Serving arm started at 14:21:07 UTC under the same card lock. Frozen replay timings characterize the fixtures. The warm Serving result determines acceptance.

The fresh native arm reached health in 294.039 s and exited 0 after the fixed requests and correctness checks. Prefill P50/P95 were 3.437360/6.947958 s and TTFT P50/P95 were 3.515739/7.033068 s. Prefill medians at 1,024, 3,070, and 6,142 input tokens were 1.502029, 3.437360, and 6.148665 s. This arm is substantially slower than the archived 2.237811 s native median. Its numerical 30% threshold is 2.406152 s. Compare both fresh compact arms and examine the shift against the archived controls before attributing a threshold result to the transport change. The fresh Prefill 23 compact arm started at 14:30:40 UTC under the same lock.

The fresh Prefill 23 compact arm exited 0 after all requests and correctness checks. Its Prefill P50/P95 were 2.367815/4.648769 s and TTFT P50/P95 were 2.453014/4.733018 s. This unchanged MoonEP control has a numerical reduction of 31.12% against the fresh native arm. At 3,070 tokens fresh native samples ranged from 2.576 to 5.304 s, while the compact control ranged from 2.356 to 2.402 s. The fresh native command matches the archived native after port normalization, and the workload file matches its earlier checksum. Reusing the original native benchmark configuration does not by itself resolve the run-to-run shift. The Prefill 47 compact arm started at 14:40:19 UTC.

### Matched device result

Both golden arms, all three Serving arms, the locked runner, and the suite exited 0. The three Serving CLI commands match after normalizing the port and MoonEP selector. All 12 requests in each arm used the same order and token counts, generated 13 tokens, and stopped at EOS. The long-answer check and both eight-request waves passed. The reported medians and nearest-rank P95 values were independently recomputed from the saved rows.

| 80-row, one-slot arm | Prefill P50 (s) | Prefill P95 (s) | TTFT P50 (s) | TTFT P95 (s) |
|---|---:|---:|---:|---:|
| Prefill 16 native | 3.437360 | 6.947958 | 3.515739 | 7.033068 |
| Prefill 23 compact control | 2.367815 | 4.648769 | 2.453014 | 4.733018 |
| Prefill 47 compact candidate | 4.045291 | 7.833128 | 4.124265 | 7.919243 |

| Input tokens | Native Prefill P50 (s) | Prefill 23 P50 (s) | Prefill 47 P50 (s) |
|---:|---:|---:|---:|
| 1,024 | 1.502029 | 0.847511 | 1.499809 |
| 3,070 | 3.437360 | 2.367815 | 4.045291 |
| 6,142 | 6.148665 | 4.624645 | 7.282317 |

The candidate increased warm Prefill P50 by 17.69% against fresh native and 70.84% against its fresh compact control. None of its 12 matched requests improved against the compact control. The lower source replay median and lower count of fenced writes did not carry through to Serving. The result does not isolate the time spent on tile assembly from other effects.

The unchanged Prefill 23 control is 31.12% faster than this fresh native sample. Relative to their archived measurements, the fresh native P50 is 53.60% higher and the fresh Prefill 23 P50 is 9.24% higher. The native within-length spread also grew. This shift leaves 30% attributable to MoonEP unverified. Stop after this comparison under the fixed-configuration objective. Keep Prefill 47 as an isolated experiment. The raw report is `p47-20261122T141415Z-comparison.json` in the server verification directory, SHA-256 `b73c1e40a1b768a6a67cd3ec96ec3a8dcb97cd23674c703932515ec93da06ce2`.

## Prefill 48 · Repeat the single-slot controls in reverse order

The renewed objective retains the 30% threshold and the fixed configuration, and continues after an unsuccessful iteration. Repeat the two eligible Prefill 47 controls in reverse order to test whether the observed 31.12% separation is reproducible. Start a fresh Prefill 23 compact process, then a fresh Prefill 16 native process, under one unchanged cards 4–7 lock. Both use their existing launch wrappers, default 80 rows, one slot, serial round reuse, the original empty-tail policy and the same 12-request driver. The two workloads use the same prompt order and output limit. The final report checks normalized launch commands and retains both correctness suites and raw samples.

The scripts were archived under `verification/ep4/experiments/prefill-48` after server-side package verification. Suite `p48-20261122T145612Z` (PID 2444344) started compact control at 14:56:53 UTC. This iteration changes only measurement order.

Prefill 23 compact exited 0 after the fixed requests and correctness checks. Its Prefill P50/P95 were 3.675902/9.693941 s and TTFT P50/P95 were 3.757298/9.779923 s. Its length-group Prefill medians at 1,024, 3,070, and 6,142 tokens were 1.303775, 3.675902, and 7.056258 s. It was the first arm of this suite. The same implementation recorded 2.367815 s when it ran second in Prefill 47. Native started at 15:06:41 UTC.

Both Serving arms, the locked runner and the suite exited 0. The normalized launch commands matched. Native Prefill P50/P95 were 2.965771/7.035477 s and TTFT P50/P95 were 3.050206/7.123249 s. The raw report is `p48-20261122T145612Z-comparison.json` with SHA-256 `70cf4f3bf14d0cb2cf0721301c7fd9261d72442186f96f2464797ca17bdd2bff`.

| Sample selection | Native Prefill P50 (s) | Prefill 23 P50 (s) | Prefill reduction |
|---|---:|---:|---:|
| Prefill 47, native first | 3.437360 | 2.367815 | 31.12% |
| Prefill 48, compact first | 2.965771 | 3.675902 | -23.94% |
| First-position arms only | 3.437360 | 3.675902 | -6.94% |
| Second-position arms only | 2.965771 | 2.367815 | 20.16% |
| Both orders pooled, 24 requests per backend | 3.373556 | 2.985369 | 11.51% |

The pooled reductions by length were 16.79%, 11.51%, and -5.83% at 1,024, 3,070, and 6,142 tokens. This calculation equally represents both orientations and verifies identical request indices, actual input lengths, generated lengths and EOS in all four arms. One suite per order is enough to contradict the stability of a 30% result. The runs show that comparison order and run conditions need to be accounted for. They do not isolate a single cause of the swings. Continue with the transport-only candidate and verify any numerical threshold crossing against matched controls.

The pooled analysis and reproduction script are archived in `verification/ep4/experiments/prefill-48` as `order-comparison.json` and `compare-order.py`.

## Prefill 49 · Tile the compact row planner's metadata transfers

The native dispatcher also writes route positions one row at a time, so generic coalescing of those stores is outside the attribution scope. Prefill 49 instead targets the aligned offset scan that is specific to compact MoonEP. At default 80 rows, a sole source computes the 32-row-aligned starts for each destination's 96 experts. A receiving rank computes only its own starts from the source histogram. Read each group of 96 counts as one tile. Accumulate starts in their original expert order in local tile storage, and commit the resulting counts and offsets to views of the original shared tensors. For EP4 this replaces 1,824 scalar global-memory loads or stores per single-source planner round with 19 tile transfers across the four ranks. These counts describe generated operations, not their latency. The distribution, payloads, epochs, 80-row width and single-slot completion ordering are preserved.

The initial host build lowered tensor views created inside the planner to tile values. Move the views to the caller scope. A subsequent build rejected mixing MTE3 and scalar writes to the same result in one function even through exclusive branches. Use mutually exclusive planner submissions, with one task per round. The original scalar planner stays on the mixed-source and other layouts. Both branch submissions have the same single predecessor and five tensor arguments. The result views alias the original allocations.

The verified Prefill 23 baseline and final candidate passed 80-row single-source and mixed-source host code generation. The candidate also passed the native slab and full native and compact Serving compile with Decode at 80 rows. Only the compact row-plan variants changed in the generated C++ kernels after normalizing generated names and comments. All native and Decode kernels matched. The candidate changes only `prefill_moonep.py` with SHA-256 `54b1f3d4a5f04c77d43bb1d6f75beac05b09e7d7cea3330e843a2f6aaf6cbefc`. The archive `prefill-49.tar.gz` has SHA-256 `0e733d1d550c7a4563584c8bcf34f96ecf26bd5fc7642a7aa6f8df7b106015d8`. The server verified and staged the package under `verification/ep4/experiments/prefill-49`.

After the reverse-order suite exited 0, suite `p49-20261122T151707Z` (PID 2518012) acquired the card lock and started the single-source frozen replay at 15:17:42 UTC. It runs both frozen replays and the fixed 12-request Serving workload on fresh Prefill 16 native, Prefill 23 compact, and Prefill 49 compact within the same lock.

The single-source replay exited 0 and passed `next_pre_mix`, `x_mixed`, and `x_next`. Across 20 measured iterations after five warmups, the fastest-rank median, per-iteration maximum median, and mean host union were 12.557, 15.067, and 42.292 ms. Prefill 23 recorded 12.592, 16.023, and 34.522 ms on this fixture. Mixed-source replay started at 15:20:11 UTC.

The mixed-source replay exited 0 and passed the same three tensors. Its three timing statistics were 2.122, 3.699, and 7.106 ms, versus 2.135, 3.277, and 6.790 ms in Prefill 23. The mixed-source branch uses the original scalar planner. Native Serving started at 15:22:37 UTC.

The emitted scalar planner matched Prefill 23 after normalizing its generated name and comments in both the mixed slab and full Serving. As a check on the observed run-to-run fluctuation, a read-only monitor captures the existing test scope's CPU throttling and memory limit counters at the warmup and 12-request report boundaries. It uses the host's cgroup v1 counters. It leaves the device allocation, scope quota, inference process and benchmark driver unchanged.

Native reached health in 318.094 s and exited 0 after the workload and correctness checks. Its Prefill P50/P95 were 3.749463/6.264280 s and TTFT P50/P95 were 3.829403/6.349349 s. The cgroup had 558 throttled periods and zero memory limit events at both the warmup and 12-request report boundaries. Prefill 23 compact started at 15:32:24 UTC.

The fresh Prefill 23 control reached health in 320.102 s and exited 0 after the workload and correctness checks. Prefill P50/P95 were 2.401975/4.733849 s and TTFT P50/P95 were 2.481427/4.818353 s. The same control recorded 2.367815 s when it previously ran second in Prefill 47 and 3.675902 s when it ran first in Prefill 48. During its timed window both cgroup counters again stayed constant, at 916 throttled periods and zero memory limit events. Prefill 49 compact started at 15:42:08 UTC.

### Prefill 49 device result

Both frozen fixtures, all three Serving arms, the runner and the suite exited 0. The recorded commands matched after normalizing the port and MoonEP selector. Each arm passed the fixed 12 requests, the 3,070-token answer, and both eight-request waves. The ordered input lengths and generated 13-token EOS outputs match across the three arms. Recomputed P50 and P95 from the raw rows.

| 80-row, one-slot arm | Prefill P50 (s) | Prefill P95 (s) | TTFT P50 (s) | TTFT P95 (s) |
|---|---:|---:|---:|---:|
| Prefill 16 native | 3.749463 | 6.264280 | 3.829403 | 6.349349 |
| Prefill 23 compact | 2.401975 | 4.733849 | 2.481427 | 4.818353 |
| Prefill 49 compact | 2.426850 | 4.871123 | 2.505614 | 4.957444 |

| Input tokens | Native Prefill P50 (s) | Prefill 23 P50 (s) | Prefill 49 P50 (s) |
|---:|---:|---:|---:|
| 1,024 | 1.685201 | 0.862021 | 0.874368 |
| 3,070 | 3.749463 | 2.401975 | 2.426850 |
| 6,142 | 5.911990 | 4.686339 | 4.716997 |

Prefill 49 improved three of 12 matched requests against Prefill 23 and increased its overall P50 by 1.04%. Its numerical reduction against native was 35.28%, versus 35.94% for the existing compact control. The Prefill 48 order reversal still contradicts a reproducible 30% gain for the fixed-configuration control. The raw report is `p49-20261122T151707Z-comparison.json` with SHA-256 `85609e7cb6aa2c72f603c254992d3ff3d0c0afff85bf1b5fb31ff5da613a40a2`.

The candidate arm's timed window also recorded constant CPU quota throttling and memory limit counters. All six snapshots are in `experiments/prefill-49/scope-intervals.log` with SHA-256 `a61106ee43782241313f620497a0f80adc19add614a79dbfb80a9948dc336b0d`. These counters leave the timing variation unexplained.

## Diagnostic 50 · Capture the current single-slot device paths

Record Prefill 23 compact and Prefill 16 native at default 80 rows on the same 1,024-token prompt and one generated token, compact first. Reuse the Diagnostic 22 driver, tracing-only Serving snapshot, and one cards 4–7 lock. The source trees, chat workload, empty-tail policy and single-slot ordering are the ones used for the fixed-configuration Serving runs. The capture is for attribution and selection of the next transport change. The goal remains the ordinary unprofiled warm Serving result.

The checksum-verified suite is at `verification/ep4/experiments/diag-50`. Suite `diag50-20261122T155230Z` (PID 2590121) acquired the lock and started compact at 15:52:38 UTC. Serving was compiling at 15:52:53 UTC.

Compact exited 0 and native started at 16:01:05 UTC. The compact capture resolves all tasks and records 520 rounds per rank. The device spans are 620.197, 616.515, 598.815 and 603.766 ms in rank order. On the source, push accounts for 33.846 ms on the reconstructed local path. The three receivers run 520 local histograms apiece with no active routes. They take 7.29–7.32 ms of aggregate task time and 5.38–5.56 ms on their respective reconstructed paths.

Receiver row-plan tasks follow 66.6–67.5 ms of gaps per rank. The local path assigns 47.3–51.9 ms of those gaps to edges from `moonep_histogram_wait` and 14.6–19.2 ms to edges from `_shared_swiglu_vf` on the same core. Edges from `moonep_wait` to scale packing contribute 50.6–51.5 ms of gaps per receiver. This capture has local task times and dependencies. Quantifying how much of each gap is remote communication would require aligned rank clocks and scheduler readiness records. The extracts are archived beside the run as `-compact-paths.json`, `-compact-histogram-work.json`, and `-compact-readiness-gaps.json`.

Both arms, the card runner and the suite exited 0. The commands match after accounting for port, trace path and the compact backend selector. Both answer records show 1,024 input tokens and one generated token, and all eight rank traces resolve 520 rounds with no unnamed tasks.

| Rank | Native device span (ms) | MoonEP device span (ms) |
| ---: | ---: | ---: |
| 0, source | 592.028 | 620.197 |
| 1 | 609.328 | 616.515 |
| 2 | 587.314 | 598.815 |
| 3 | 586.472 | 603.766 |

The longest rank span increased by 10.869 ms (1.78%) for compact in this capture. Source route-metadata tasks on the reconstructed path take 25.302 ms for compact and 15.737 ms for native. Source gaps before result transfer and reduction total 51.765 and 36.403 ms, respectively. The three receivers execute less activation work in compact (5.9–6.2 ms compared with 15.9–16.4 ms), with longer gaps before that work (55.5–56.7 ms compared with 41.8–42.6 ms). These measurements motivate cuts in metadata, transfer submission and result readiness within the round. The profile contains one request per arm. The ordinary Serving suite measures the warm P50.

The source path assigns 25.645 ms of compact gaps before reduction to 512 local `combine_wait → combine_reduce` edges. Native assigns 17.724 ms to 511 such edges. Two compact reductions also follow same-core `engram_widen` tasks, which account for 7.281 ms. Gaps between W2 and scatter are almost equal at 18.551 and 18.544 ms. The two reports are archived as `-compact-return-gaps.json` and `-native-return-gaps.json`; the extraction script is under `experiments/diag-50/source-return-gaps.py`.

On the 620.197 ms reconstructed source path, route metadata, activation transport, result transfer and reduction, and completion and reuse tasks occupy 86.992 ms (14.03%). Their preceding gaps occupy another 102.961 ms. For scale, a 30% reduction from the 609.328 ms native device span would require removing 193.667 ms from this compact capture. Some categorized work is required, and the gaps can include peer execution or scheduler delay. These rank-local totals describe the current profiled path. The warm Serving measurements determine acceptance.

The compact source publishes every payload and each peer publishes return completion. For each of the 520 rounds, the consumer's start cannot precede the start of its notifier. Under a constant relative clock offset per rank, these reciprocal dependencies give intervals 17.9–20.0 µs wide. Across the three receivers, the bounds on median scale-pack start after source notification start span 2.0–22.5 µs. The bounds on median source reduction start after the last return notifier starts are 1.9–21.1 µs. The latter interval sums to 2.430–11.129 ms across all rounds. This includes time within the notifying task and any later scheduling delay. `diag50-20261122T155230Z-compact-clock-bounds.json` has SHA-256 `ed21f136e19f6585bfa91d72ec16d684d51dc17aa54209b03ac176b1877f3b38`.

A dependency-only replay adds 3,120 cross-rank forward and return edges and 11,520 within-layer serial reuse edges to the four recorded graphs. Using the recorded task spans as vertex durations, the longest replay path is 362.209 ms. Setting the route, payload, return and reuse tasks to zero gives 282.178 ms on the same graph. The 80.031 ms difference describes the modeled task work. The measured 620.197 ms span also contains scheduling, resource contention and dependencies the replay does not capture. The replay cannot predict Serving latency. `experiments/diag-59/transport-path-budget.py` generated `diag50-20261122T155230Z-compact-transport-budget.json`, SHA-256 `6bfa938533c06d0d9fca35cff8511917230c5a944947846cda104f955d81fdac`.

Using the midpoint of each clock interval, compare phases through the owner that publishes return readiness last in each round. Source metadata start to push start accumulates 23.369 ms for native and 35.967 ms for compact. The round medians are 42.9 and 68.0 µs. Payload notification to that owner's pack start has similar medians (12.46 and 12.05 µs) and different totals (14.273 and 31.627 ms). The first layer's first round accounts for 15.175 ms of the 17.354 ms difference (4.476 ms native, 19.652 ms compact). Diagnostic 50 traced the first request after health. The observed first-round delay motivates warming the diagnostic before recording the steady-state request. The two reports are `diag50-20261122T155230Z-{native,compact}-round-phase-spans.json`, SHA-256 `526ef0519774e1edcd25b8283b967e60597793f4a3f5f8336724d2b287527958` and `c237f3e5896c5e7a545b2f32875db004c8b11e8a33d38ef76a2a92e18bea0190`.

## Prefill 51 · Skip the empty local histogram on compact receivers

In a compact chunk with one active source, peer ranks receive the histogram and plan only their own experts. The Prefill 23 graph still calls `moonep_histogram` on every rank. The passive rank loads its local indices and zeros the send counts, although the row planner reads the source histogram and the rank never submits activation push. The native protocol still publishes a histogram per source to its fixed receive lanes.

Keep the Prefill 23 histogram code and submit it only on the source for compact sole-source chunks. On a passive rank, anchor the existing histogram wait to `scale_ready` in that same round. The source and mixed-source paths keep their histogram tasks. The wait still uses the existing source rank and epoch. Planning, payload, result and release notifications follow the original protocol. The local scale-preparation task consumes the quantized input from the same round, whose staging is ordered by the single completion slot. This removes a redundant transport task within a round with the original inter-round reuse and expert kernel schedule.

Both 80-row slabs, the native slab and full native and compact Serving passed PTOAS 0.72 host code generation. After normalizing generated names and comments, all 109 device kernels in each full Prefill build and all 45 Decode kernels matched Prefill 23. The compact orchestration gates histogram submission by source role and feeds the existing wait from the selected task ID. The candidate changes only `prefill_moonep.py`, SHA-256 `9dffbbbb8a7233f8f28eca8ba2c38e142ca19c1c66db7cba0c5cd53a947d6f26`. The source manifests and archive SHA-256 `6cf284bdb2fe4d8332b28300ab19ad63845535732a0d548017b6b12e38075547` passed on the server. Staged under `verification/ep4/experiments/prefill-51`.

The packaged suite retains the two frozen fixtures and measures the candidate, Prefill 23 compact, then the original native arm with one card lock. This reverses the Prefill 49 Serving order after the observed order sensitivity. Suite `p51-20261122T161125Z` started the single-source replay at 16:11:29 UTC and the mixed-source replay at 16:13:57 UTC. Both exited 0 and passed `next_pre_mix`, `x_mixed` and `x_next`. Candidate Serving started at 16:16:24 UTC.

After five warmups and 20 measurements per fixture, the per-iteration maximum rank median was 14.688 ms in the single-source replay and 3.761 ms in the mixed-source replay. The candidate Serving arm reached health in 324.107 s and exited 0. All 12 measured requests produced 13 tokens and stopped at EOS. The long answer and both eight-request waves passed. Prefill P50/P95 were 2.641295/4.992983 s and TTFT P50/P95 were 2.719054/5.078096 s. Prefill medians at 1,024, 3,070 and 6,142 tokens were 0.935439, 2.641295 and 4.836604 s. The fresh Prefill 23 control started at 16:26:10 UTC.

The Prefill 23 arm reached health in 302.049 s and exited 0 with matched token usage and all Serving checks passed. Its Prefill P50 was 2.905335 s, with 1.266318, 2.905335 and 5.446263 s at the three lengths. The candidate's median was 9.09% lower. Nine of 12 matched requests improved. The three per-length reductions were 26.13%, 9.09% and 11.19%. This is a single ordering and the control also has wider within-length spreads. The final native arm started at 16:35:35 UTC.

### Prefill 51 device result

The native arm reached health in 302.107 s. All three Serving arms and the suite exited 0 with normalized commands and matching requests. All 36 timed responses generated 13 tokens and finished at EOS. The long answer and both concurrent waves passed on all arms.

| Input tokens | Native Prefill P50 (s) | Prefill 23 P50 (s) | Prefill 51 P50 (s) |
| ---: | ---: | ---: | ---: |
| 1,024 | 0.802467 | 1.266318 | 0.935439 |
| 3,070 | 2.292743 | 2.905335 | 2.641295 |
| 6,142 | 4.495357 | 5.446263 | 4.836604 |
| All 12 | 2.292743 | 2.905335 | 2.641295 |

The candidate was 15.20% slower than native at P50 and slower on all 12 paired requests. Its 9.09% reduction from the compact control warrants an opposing-order repeat if considering adoption. The 30% requirement for this native sample is 1.604920 s. The suite report is `p51-20261122T161125Z-comparison.json`, SHA-256 `5093e5024681323bdf4190e1b9c2a17ef3bedf81767ecfcc0932a15122ff68ba`.

## Sweep 52 · Extend the single-slot comparison to 16,384 tokens

Run Prefill 23 compact first and Prefill 16 native second, at 80 rows and with one serial communication slot. Reuse `benchmark-16k.py`, the pinned Prefill 15 Serving snapshot and the checksum-verified workload. The two arms use a 16,640-token context limit, 1,024-token chunks, matched warmups, and four timed requests at each of 1,024, 3,070, 6,142, 8,192, 12,288 and 16,384 tokens. The report checks token usage and completion, normalized launch commands, both long answers and both eight-request waves. It records Prefill and TTFT P50/P95 and per-length paired improvements. The reporting script reproduced the archived two-slot sweep's aggregate Prefill statistics in a scratch run.

Suite `sweep52-20261122T161341Z` acquired cards 4–7 after Prefill 51 exited and started compact Serving at 16:45:03 UTC. Both runs use the existing card-lock runner. The measure by length is supplementary to the original 12-request acceptance workload.

### Sweep 52 device result

Compact finished at 16:57:02 UTC and native finished at 17:09:41 UTC. The runner exited 0. Both arms matched the launch configuration and token usage, generated 13 tokens at EOS on all 24 timed requests, and passed the two long answers and two concurrent waves.

| Input tokens | Native Prefill P50 (s) | Prefill 23 P50 (s) | Prefill reduction | Improved pairs | Native TTFT P50 (s) | Prefill 23 TTFT P50 (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.998767 | 0.856781 | 14.22% | 3/4 | 1.073664 | 0.932482 |
| 3,070 | 3.413619 | 2.382904 | 30.19% | 3/4 | 3.492467 | 2.461296 |
| 6,142 | 6.328070 | 4.655829 | 26.43% | 3/4 | 6.413280 | 4.741250 |
| 8,192 | 7.484211 | 6.179162 | 17.44% | 4/4 | 7.574307 | 6.268484 |
| 12,288 | 11.149115 | 9.344542 | 16.19% | 4/4 | 11.247371 | 9.442310 |
| 16,384 | 14.582650 | 12.671271 | 13.11% | 4/4 | 14.688397 | 12.776824 |

The absolute median saving at 16,384 tokens is 1.911 s. From 8,192 through 16,384 tokens, the relative reduction declines from 17.44% to 13.11% while all 12 pairs improve. The largest relative reduction occurs at 3,070 tokens, where one of four pairs slows down. Across all 24 requests, Prefill P50 decreases by 20.53% (6.805606 to 5.408259 s) and 21 pairs improve. This compact-first sweep still leaves the 30% warm acceptance criterion unverified. Prefill 51 and previous reverse-order controls show material run-to-run variation. Report `sweep52-20261122T161341Z/comparison.json` has SHA-256 `abd6bb4246a9ce765c5a0ca75a553aa7e22348aa476d4f2590c333143574fdf2`.

The three slower pairs are the first 1,024, 3,070 and 6,142 token requests in the paired order. Their changes range from 0.05% to 1.23% slower. Later pairs at those lengths vary widely. At 16,384 tokens, the four paired reductions are 13.88%, 15.89%, 9.72% and 13.01%. The median saving grows by 0.500 s from 8K to 12K and by 0.107 s from 12K to 16K. The first three pairs and broad short-prompt ranges emphasize the need for the native-first Sweep 84 confirmation before attributing the short-prompt gains.

## Prefill 53 · Transfer the compact route descriptor with its scales

Start from Prefill 23. Its compact push sends a separate 160-byte quantization scale row and 32-byte descriptor for each routed activation. Store both in one 192-byte row of the existing scale window. The descriptor occupies the first 32 bytes so it falls in the cache line invalidated by `remote_store`, which supplies the scalar route reads in scatter. The two typed views have the same backing and the same byte stride. Activation and result bytes, exact FP32 weights, expert indices and consumer order are preserved. This tests one fewer remote operation and release fence per route against the added 160-byte on-core move and strided receiver reads. Native publishes one route tile for an entire expert lane, so it has no corresponding per-route descriptor store to remove.

Both compact slab shapes, the matching native slab and both full 80-row Prefill and Decode builds passed PTOAS 0.72. Against the Prefill 23 graphs, the compact code differs only in push, scale packing and scatter addressing. The compact task submission sequence and block counts match. All 109 native Prefill kernels and 45 Decode kernels match. The doorbell and epoch edges retain the one-slot serial order. Generated push has two GM release fences per route, down from three. The host creates the two typed views from the same window and allocates the same total window bytes.

The source manifests and package SHA-256 `3a6cb5a3fabe935f8a3b962b7871899bb6559a3c8dcd1cc602e00c5261cb6f70` passed server verification under `experiments/prefill-53`. Suite `p53-20261122T161938Z` acquired the lock after Sweep 52 and started the single-source fixture at 17:09:51 UTC. After the two fixtures, it runs the original Serving workload in native, Prefill 23, candidate order.

The first frozen replay exited 1 during task submission. The chip entry declares the scale view as `OUTPUT_EXISTING` and the descriptor view as `INPUT`; both cover the same window bytes. The host task argument validator rejects an overlapping argument paired with a writable argument. No output tensors were validated and the suite stopped before mixed-source or Serving. The error is in `p53-20261122T161938Z-golden-single-80.log`.

## Prefill 54 · Pass the merged payload through one writable window view

Keep the 192-byte format and assign it to the existing INT32 route argument. The same argument carries both reads in the compact scale-pack task. It loads the 160 scale bytes as 40 words and reinterprets the tile as UINT8. The descriptor stays in the first cache line. The legacy scale argument in the common chip entry is unused in this graph. Bind it to a small read-only view of the existing weight window. The generated argument list passes both legacy views as inputs and only the packet window as a writable argument. This satisfies the host submit validation with the same total communication allocation.

Both compact slab shapes, the native slab and all four full Prefill and Decode builds passed PTOAS 0.72. The generated compact packet has one 48-word store and two GM fences per route including the activation store. The task submissions and block dimensions still match Prefill 23. The only changed compact device kernels are push, scale packing and scatter addressing. Native Prefill and both Decode kernel sets match.

The server verified the fixed snapshot and package SHA-256 `e447bafb4fc2a56b1026fb61885afab35fc3acf012fd7222b5f67706d51560ee` under `experiments/prefill-54`. Suite `p54-20261122T171835Z` took the same cards 4–7 lock and started the single-source replay at 17:18:36 UTC and the mixed-source replay at 17:21:04 UTC. Both passed `next_pre_mix`, `x_mixed` and `x_next` and exited 0.

The initial Serving script omitted the native launch. The compact control had already started at 17:23:31 UTC. Before the candidate arm began, its launcher was corrected to run the original native wrapper after the candidate within the held lock. The revised order is Prefill 23, Prefill 54, native. The active recovery launcher, the initial locked script and the corrected locked script are archived in `experiments/prefill-54`; subsequent runs call native directly from the locked script.

The Prefill 23 control reached health in 302.118 s and exited 0. Prefill P50 was 4.263717 s overall, with 2.054618, 4.263717 and 7.012892 s at 1,024, 3,070 and 6,142 tokens. These medians are higher than the earlier runs of the same control. The candidate started at 17:33:22 UTC.

The Prefill 54 candidate reached health in 336.066 s and finished its Serving checks. The 12-request Prefill P50 was 2.348771 s, with 0.838585, 2.348771 and 4.633020 s at the three lengths. The recovered launcher started native at 17:43:19 UTC within the same lock.

### Prefill 54 device result

The native arm reached health in 316.056 s. All three Serving arms, the card runner and the suite exited 0. The normalized commands, per-request input and output token counts, long answer and concurrent waves all passed.

| Input tokens | Native Prefill P50 (s) | Prefill 23 P50 (s) | Prefill 54 P50 (s) | Prefill 54 reduction vs native |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1.084715 | 2.054618 | 0.838585 | 22.69% |
| 3,070 | 2.506441 | 4.263717 | 2.348771 | 6.29% |
| 6,142 | 5.059117 | 7.012892 | 4.633020 | 8.42% |
| All 12 | 2.506441 | 4.263717 | 2.348771 | 6.29% |

All 12 candidate requests improved over native in this ordering. Prefill P95 fell from 5.259717 to 4.640934 s. The 30% line under the fresh native control is 1.754509 s. The compact control ran first and had a 46.75% higher median than the same backend in Prefill 51. The 44.91% candidate-to-compact reduction in this suite reflects a comparison with that slower sample. The opposite order in Prefill 55 will provide a fresh Prefill 54 control. The full report is `p54-20261122T171835Z-comparison.json`, SHA-256 `dc4970d05d205bae8108b3adf41332813e5207aa00ca98270d608324b4c124c9`.

## Prefill 55 · Read the scale and route descriptor in one load

Start from Prefill 54. In compact scale packing, load each valid 192-byte packet into one tile. The descriptor fields use the INT32 tile's FP32 view, and the scale bytes use its UINT8 subview with the same valid row count. The two strided GM reads become one contiguous load. The generated code retains the same scale padding, on-core moves and MX packing.

Both compact slabs, the native slab and full Serving compiled at 80 rows with PTOAS 0.72. Only `moonep_pack_scale` changed from Prefill 54. Native, Decode, task submission order and block dimensions match. The server verified the source against Prefill 54 and package SHA-256 `5e6542c2d74206702f4ef74da0022d75f6388872681d70e90b353f80b3518cb1`. Suite `p55-20261122T172927Z` acquired the lock after Prefill 54 exited 0 and started the first frozen replay at 17:53:01 UTC. After replay, it measures native first, then Prefill 54 compact and this variant under the same lock.

The single-source and mixed-source frozen replays exited 0 and passed `next_pre_mix`, `x_mixed` and `x_next`. Native Serving started at 17:58:08 UTC.

The native arm reached health in 298.043 s and exited 0 after its Serving checks. The 12-request Prefill P50 was 2.364402 s, with 0.846875, 2.364402 and 4.634655 s at 1,024, 3,070 and 6,142 tokens. The Prefill 54 compact control started at 18:07:27 UTC.

### Prefill 54 opposing-order comparison

The Prefill 54 control reached health in 360.091 s, passed Serving checks and exited 0. Compare its raw rows with both Prefill 54 and Prefill 55 native arms. The four normalized commands match, and all paired requests match input length, 13 output tokens and EOS.

| Order | Native P50 (s) | Prefill 54 P50 (s) | Prefill 54 reduction | Improved pairs |
| --- | ---: | ---: | ---: | ---: |
| Native ran last in Prefill 54 | 2.506441 | 2.348771 | 6.29% | 12/12 |
| Native ran first in Prefill 55 | 2.364402 | 3.000044 | -26.88% | 0/12 |
| Equal samples from both orders | 2.387923 | 2.425968 | -1.59% | 12/24 |

Pooled per-length medians increased by 15.26%, 1.59% and 1.22% at 1,024, 3,070 and 6,142 tokens. The results reverse across the two runs and do not establish a reproducible reduction from Prefill 54. The report `p55-20261122T172927Z-p54-orders.json` has SHA-256 `c4d62147a72fc302a8293974c9d7b66de71cedeb6a0dc2a5c6b13f690c7d1730`. The Prefill 55 candidate started at 18:18:00 UTC.

### Prefill 55 device result

The candidate reached health in 324.040 s. The three Serving arms, runner and suite exited 0. Normalized commands and all 36 measured responses match the specified prompt lengths and 13-token EOS. Both frozen fixtures, the long answer and the two concurrent waves passed.

| Input tokens | Native Prefill P50 (s) | Prefill 54 P50 (s) | Prefill 55 P50 (s) | Prefill 55 reduction vs native |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.846875 | 1.770246 | 0.825858 | 2.48% |
| 3,070 | 2.364402 | 3.000044 | 2.322864 | 1.76% |
| 6,142 | 4.634655 | 5.078218 | 4.559674 | 1.62% |
| All 12 | 2.364402 | 3.000044 | 2.322864 | 1.76% |

Eleven of 12 pairs improved over native. Prefill P95 fell from 4.649700 to 4.580222 s. The numerical reduction against the compact control is 22.57%, with the same control's opposing-order shift recorded above. The 30% threshold for this native sample is 1.655081 s. The comparison report `p55-20261122T172927Z-comparison.json` has SHA-256 `4c53cb55f0b3d9d6f7e0d2cf91cc0233d6dbc8e04729055552aaea46b98a801c`.

## Prefill 56 · Publish the aligned row plan with activation readiness

Start from Prefill 55. In EP4 at 80 rows with one source, only the source knows the complete route counts needed to assign the compact aligned rows. In its existing histogram task, it encodes each expert's start and count into a 32-bit word. It writes a tile of 96 words to the receiver and retains the same tile locally for push. Push reads one word per expert in place of separate count and start reads. Native forward lanes have fixed starts and still need publication from each source.

The three receivers skip their empty histograms and histogram waits. All four ranks skip the separate row-plan task. Ready notification for activations also publishes the plan. The scale-pack task reads the plan with one tile load per block, and its first block writes the count and start arrays for the original expert kernels with two tile stores. The mixed-source path keeps the separate histogram and plan. Packet bytes, ordered expert inputs, output reduction, round width, release epochs and the single-slot reuse graph retain their existing semantics. This revisits the Prefill 21 protocol with one packed source read per expert and vector plan materialization.

Both frozen slab shapes and the matching native slab compiled. Full native and compact Serving also compiled at 80 rows with PTOAS 0.72. Only the compact histogram, push and scale-pack kernels changed. All expert and native kernels and both Decode builds matched; the 134 block counts and 250 submit sites per Prefill arm match Prefill 23. The compact orchestration guards the two removed tasks and keeps the same 80-row staging and one-element completion array.

The server verified the base and candidate source manifests and archive SHA-256 `a6dfdcb998e3d42cfd5dfce9f4d6200d919719b58173a4c2f86950c4f182f63b` under `experiments/prefill-56`. Suite `p56-20261122T180311Z` acquired the card lock after Prefill 55 exited successfully. The single-source frozen replay started at 18:27:53 UTC. The original workload will run with the candidate first, then Prefill 55 and native under one lock.

Both frozen replays exited 0 and passed `next_pre_mix`, `x_mixed` and `x_next`. The per-iteration maximum rank median was 13.700 ms for single source and 3.780 ms for mixed source, compared with 16.081 and 3.767 ms in the Prefill 55 replays. The single-source host union mean increased from 37.217 to 41.786 ms. The candidate Serving arm started at 18:32:48 UTC.

The candidate reached health in 300.074 s and exited 0 after 12 measured requests and the Serving checks. Its warm Prefill P50 was 2.326871 s, with 0.828863, 2.326871 and 4.557209 s at 1,024, 3,070 and 6,142 tokens. The compact control started at 18:42:09 UTC.

The Prefill 55 control exited 0 after 12 measured requests. Its warm Prefill P50 was 2.304872 s, with 0.827667, 2.304872 and 4.544165 s at the three lengths. The native arm started at 18:51:31 UTC.

### Prefill 56 device result

The native arm reached health in 298.092 s. All three Serving arms, the card runner and the suite exited 0. The normalized launch commands and 36 measured responses matched the workload. Every timed response generated 13 tokens and finished at EOS. The two frozen replays, long answer and two concurrent waves passed.

| Input tokens | Native Prefill P50 (s) | Prefill 55 P50 (s) | Prefill 56 P50 (s) | Prefill 56 reduction vs native |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.826609 | 0.827667 | 0.828863 | -0.27% |
| 3,070 | 2.308697 | 2.304872 | 2.326871 | -0.79% |
| 6,142 | 4.516088 | 4.544165 | 4.557209 | -0.91% |
| All 12 | 2.308697 | 2.304872 | 2.326871 | -0.79% |

The candidate was 0.95% slower than the compact control at P50. It improved two of 12 pairs against native and four against the compact control. The respective Prefill P95 values for native, Prefill 55 and Prefill 56 were 4.519652, 4.559354 and 4.578599 s. The 30% line for this native sample is 1.616088 s. The comparison report `p56-20261122T180311Z-comparison.json` has SHA-256 `d0eda5e643d5d9a2598619b2fb8325ff3c07bb0fe7801271ed30d59930b8b0ef`.

Prefill 55 ran after native in its own suite and before native here. Pooling the two orders with equal sample counts gives native P50 of 2.322623 s and Prefill 55 P50 of 2.314961 s, a 0.33% reduction, with 14 of 24 pairs improving. All four normalized commands and request usage match. The report `p56-20261122T180311Z-p55-orders.json` has SHA-256 `12233df33ef3963c40a20d3c06fb6fa3b2c5e1a9a77862820840389e09c9a664`.

## Prefill 57 · Prepare the packed scale row once per source token

Start from Prefill 55. In the compact path, the existing scale-unpack task writes 192-byte rows with a zeroed 32-byte descriptor prefix. Its 16-token tile performs the scale-byte move once. Push loads the full row for each route and writes the route, source and exact FP32 weight into the prefix in its own tile. This removes the per-route scale move and keeps the 192-byte received format, cache-line placement, remote writes and release fences. Each source round adds 2,560 bytes of scale scratch. Native puts scales into separate per-source lanes and has no per-route compact descriptor prefix to construct.

Both frozen slab shapes, the native slab and full Serving passed host compilation at 80 rows with PTOAS 0.72. Only compact `dispatch_scale_unpack` and `moonep_push` changed. The 134 block counts and 250 submit sites per arm match. Native, Decode and the expert kernels match Prefill 55. Generated push retains both GM fences and uses a 192-byte load with no scale move. The unpack task stores prefixed rows in 16-token tiles.

The server verified the base, candidate and archive SHA-256 `f2e9bbc089ff8625ae071bfde599a2faa2a36a881387845fcb4ba2ee921165c2` under `experiments/prefill-57`. Suite `p57-20261122T183501Z` acquired the lock after Prefill 56 exited and started the single-source replay at 19:00:54 UTC. Its Serving order is native, Prefill 55, candidate.

The single-source and mixed-source frozen replays passed `next_pre_mix`, `x_mixed` and `x_next` and exited 0. Native Serving started at 19:05:47 UTC.

The native arm reached health in 298.094 s and finished its Serving checks with exit 0. Its 12-request Prefill P50 was 3.618883 s, with 1.460157, 3.618883 and 5.873150 s at 1,024, 3,070 and 6,142 tokens. This native P50 is 56.75% above the preceding suite's 2.308697 s. Both runs reported 100% sidecar page-cache residency. Prefill 55 compact control started at 19:15:19 UTC.

The source push still loads the 80-token weight tile once per active expert. The same three route captures used for Prefill 58 produce 81,210 such loads and 7,680 active 16-token tiles across 1,560 rounds. The existing packet prefix has room for six FP32 route weights. Packing them with the scales could let push pick its weight from the packet it already reads. This would read weights once per token tile, while making packet preparation depend on both quantization and routing. The estimated valid weight bytes read are 155.92 MB for the current load sites and 2.95 MB for token tiles. The count excludes cache effects and carries no latency prediction. See `experiments/prefill-57/route-weight-load-count.py` and `p57-route-weight-load-count.json`, SHA-256 `e26c3fe140ff8343223b7b9ad52d488b0eea92ea4fbc5fb199f591cc6e7a74b4`.

### Prefill 57 device result

The candidate reached health in 294.108 s. All three Serving arms, both frozen replays and the suite exited 0. Normalized commands matched, each measured response generated 13 tokens at EOS, and each arm passed its long answer and both concurrent waves.

| Input tokens | Native Prefill P50 (s) | Prefill 55 P50 (s) | Prefill 57 P50 (s) | Prefill 57 reduction vs native |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1.460157 | 0.793964 | 0.809781 | 44.54% |
| 3,070 | 3.618883 | 2.237714 | 2.249580 | 37.84% |
| 6,142 | 5.873150 | 4.360986 | 4.385654 | 25.33% |
| All 12 | 3.618883 | 2.237714 | 2.249580 | 37.84% |

Prefill 57 was 0.53% slower than the compact control at P50 and improved two of 12 pairs against it. Prefill P95 was 6.405376 s for native, 4.374995 s for Prefill 55, and 4.433518 s for Prefill 57. Both compact arms were faster than all 12 pairs from the native run. The same Prefill 55 control reduced the pooled P50 by 0.33% in the earlier two opposing orders. The unusually high native values in this run and the lack of an incremental gain leave the 30% result unverified. Prefill 60 repeats Prefill 57 and native in the opposite order. The report `p57-20261122T183501Z-comparison.json` has SHA-256 `c1e86bcfddadfcd69395470fb1dad4e9cbbefaa562a8466404dc2b949ad4e136`.

## Prefill 58 · Return contiguous expert rows in batches of four

Start from Prefill 56. The sole-source compact plan assigns each owner a contiguous range in the existing result window, sized by its actual number of active expert rows. The first two words of the existing plan also carry the owner's return start and count. The owners already produce contiguous BF16 rows in W2. The scatter task loads and transfers up to four adjacent rows at once, using the actual valid row count on the tail. The source records the new result row for each route in its histogram task and keeps the original TopK order for FP32 reduction. Mixed-source scatter and reduction retain their existing order.

This changes return addressing and the remote write size. The same 480 result rows are allocated in one slot. W2 output shape, expert submissions and block sizes, return readiness and round release retain their existing behavior. Native uses per-source lanes and has no sender-owned compact plan. The older Prefill 28 pull transferred results with one remote read per route and regressed against its own control.

Three archived real-model route captures contain 1,560 single-source layer-rounds. Using each owner's actual row count, the estimated total write count is 186,521 instead of 736,800, including local owner stores. Cross-rank writes are 141,309 instead of 558,268, a 74.69% reduction. BF16 byte counts are identical. The script and report are archived under `experiments/prefill-58`; `return-write-count-58.json` has SHA-256 `76f5b9e943b1f343d1298eb6293ccdb4b398419e38ce357a8269415aa5d42e9b`.

Both frozen slab shapes and full native and compact Serving compiled at 80 rows with PTOAS 0.72. The changed compact kernels are histogram, push, scale pack, scatter and combine. All routed expert, native and Decode kernels match Prefill 56. The 134 block counts and 250 submit sites match in the generated compact graph. A single-slot live test is required for precision and latency.

The server verified the source manifests and archive SHA-256 `2faa0713c6561b6c152f4576054334bed398256c1f1e7d114aaf1621f3e30aaa`. Suite `p58-20261122T184315Z` acquired the card lock after Prefill 57 exited and started the single-source replay at 19:33:59 UTC. Its Serving order is native, Prefill 56, candidate.

The single-source and mixed-source fixtures passed `next_pre_mix`, `x_mixed`, and `x_next` and exited 0. Native Serving started at 19:39:05 UTC.

Native reached health in 310.101 s and exited 0 after its Serving checks. Its 12-request Prefill P50 was 2.229302 s, with 0.768110, 2.229302 and 4.414317 s at 1,024, 3,070 and 6,142 input tokens. The normalized native launch command and the 12 prompt and output token counts match Prefill 57. Both runs had 100% sidecar page-cache residency. The native P50 fell by 38.40% from the previous run. Prefill 56 compact control started at 19:48:35 UTC.

Prefill 56 control exited 0 with Prefill P50 of 2.202713 s. Its per-length P50 was 0.757007, 2.202713 and 4.364431 s. Each of the 12 requests on both arms generated 13 output tokens and stopped at EOS. The Prefill 58 candidate started at 19:57:53 UTC.

### Prefill 58 device result

Both frozen replays, all three Serving arms, the matched card run and the suite exited 0. The normalized commands matched. All 36 timed requests used the fixed input order and generated 13 output tokens at EOS. Each arm also passed the long answer and both concurrent waves.

| Input tokens | Samples per arm | Native P50 (s) | Prefill 56 P50 (s) | Prefill 58 P50 (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 4 | 0.768110 | 0.757007 | 0.759380 |
| 3,070 | 4 | 2.229302 | 2.202713 | 2.204174 |
| 6,142 | 4 | 4.414317 | 4.364431 | 4.382660 |
| All | 12 | 2.229302 | 2.202713 | 2.204174 |

The candidate's Prefill P50 was 1.13% lower than native and 0.07% higher than Prefill 56. Five of twelve matched requests improved against Prefill 56. Candidate, Prefill 56 and native Prefill P95 were 4.402375, 4.471014 and 4.430205 s. Candidate and Prefill 56 TTFT P50 were 2.287893 and 2.281575 s. The four-row return combines many modeled writes without a measured median improvement. Retain Prefill 56 as the faster standalone compact control. Prefill 61 uses the validated Prefill 58 layout as its incremental control for the separate forward-packet change. The archived result is `p58-20261122T184315Z-comparison.json`.

## Diagnostic 59 · Capture scheduler timing around compact transport

The Diagnostic 50 capture located gaps before row planning, scale packing and combine, without the scheduler's dispatch and finish records. Reuse the matched 1,024-token diagnostic driver and raise the trace-only Serving snapshot to chip swimlane level 3. This enables scheduler task timing and scheduler phase records. Request host phase records where the runtime supports them. Verify the artifact coverage and the clock bounds before assigning gaps between ranks. The clock analysis script is `experiments/diag-59/clock-bounds.py`; it reproduces the bounds above on the archived Diagnostic 50 capture.

The staged runner is under `experiments/diag-59`. The trace-only change in `npu_runner.py` is the swimlane level. Its SHA-256 is `1a2c8ab202103c12a379e50d7d95b475e7051b7aaeac76b7972370c108766744`. After health, `diag-59.py` completes one request before starting host profiling and repeating the same 1,024-input, one-output-token request. It archives both answers and lists device captures produced by the measured request in `measured-captures.json`. Device diagnostic collection is configured at worker creation and may leave captures from both requests. The analysis can select a dispatch and infer the sole source from the compact notification trace, accounting for the scheduler's ordinary replica rotation. Driver SHA-256 is `e531362651d9ec4d7a32620b70d76933bc8a98d8c1e63d27c9b020cb48ff9d5c`. Shell and Python syntax checks passed. Diagnostic `diag59-20261122T185540Z` is queued after Prefill 58. It uses the same lock with Prefill 55 and Prefill 56, isolating the dispatch handshake and row plan changes.

Diagnostic 59 launched at 20:07:21 UTC after Prefill 58 exited 0. The matched wrapper exited 126 before starting Serving because the isolated account could not traverse the root-owned experiment directory. Preserve that directory's mode 0700. The retry copies the same driver and Serving package into an account-owned mode-0700 directory and regenerates wrappers with the adjusted paths. The original and copied driver and package match byte for byte. The isolated account passed read-access and shell syntax checks for the required inputs. Retry `diag59r-20261122T200956Z` is queued on the original cards 4–7 lock while Prefill 60 runs.

The archived `scheduler-spans.py` joins scheduler records to AICore activity by run epoch, core and register task ID. It reports per-kernel dispatch, execution and finish intervals, and the gap from the last timed local predecessor to dispatch. That last interval includes readiness effects from outside the local graph. The script identified level 1 and zero scheduler records on all four ranks of the archived Diagnostic 50 device run. The level 3 timing path awaits the Diagnostic 59 captures. The script SHA-256 is `917fb85f0a49899331be6afc43eef2e17887ef04bc3792fe583a6aae8704f82a`.

### Observed-order replay of the original compact trace

Replay all 71,923 tasks from the four-card Diagnostic 50 compact capture. Reuse the previous explicit forward, return and slot-reuse dependencies, add each task's observed first-core predecessor, and tighten the card clock bounds with those cross-rank dependencies. Relative to the resulting readiness time, keep each recorded task's residual delay. The unchanged replay recovers the observed 620.197 ms span. At low, midpoint and high feasible clock offsets, the estimated reductions under synthetic interventions are as follows.

| Work set to zero | Midpoint reduction (ms) | Range across the three clock alignments (ms) |
| --- | ---: | ---: |
| Source histogram, plan and push task bodies | 40.878 | 36.075–45.312 |
| Return and release task bodies | 23.597 | 22.637–24.600 |
| All measured MoonEP transport task bodies | 72.372 | 68.692–75.340 |
| All transport task bodies and their readiness-to-start residuals | 135.383 | 135.373–135.393 |

Zeroing the transport bodies and their residuals leaves a 484.814 ms span. A 30% reduction from the matched cold native device span of 609.328 ms would require 426.530 ms. The replay preserves the observed first-core order and all delays outside the selected tasks. It measures sensitivity to those assumptions on a single cold capture. Some residuals include unobserved readiness and scheduling, so their removal has no direct implementation equivalent. Use the warm scheduler trace to identify which delays follow MoonEP readiness publication and which can move when the transport path changes.

The analysis script is `experiments/diag-59/anchored-transport.py`, SHA-256 `bafddf31a4284c02630bb0726793e50c2306fcce45f0a6f273cd6f4b0eb293b2`. The report is `diag50-20261122T155230Z-compact-anchored-transport.json`, SHA-256 `36f41796f946858af09c3cf9417af65e21c44560b94c0f831a83c8724ab268cc`.

The private Diagnostic 59 retry took the original device lock at 20:40:40 UTC and started the Prefill 55 control. The driver and Serving process are both live under the locked scope.

The control reached health in 292.052 s. The archived `transport-readiness.py` separates the timestamp intervals around push, activation packing, combine and same-layer round reuse. It joins scheduler dispatch and subtask FIN records to device execution. The start of the latest peer notification, adjusted by the clock bounds, gives a lower bound for when a counter might become visible. The script also totals the portion of each resulting interval with all recorded AIVs executing. An idle AIV does not prove that a particular task was eligible for dispatch. A deferred-wait kernel's subtask FIN can precede counter completion, and the async-poll records describe scheduler batches. Keep the remaining delay unattributed. The script SHA-256 is `5518d9f40e86aa8ab36635ac455db64d54c0e284cd047332dd1b1a7acd66d309`. Running it on the four actual Diagnostic 50 traces showed level 1 on every rank. That capture has no scheduler records for this split.

The first Prefill 55 request emitted raw Level 3 timing on all four cards. The source rank's prewarm capture joined 18,361 logical tasks with scheduler records and reported no dropped scheduler phase records. Over 520 source pushes, the interval from the last recorded local subtask FIN to dispatch totaled 4.688 ms and had median 8.124 µs. The interval from last timed producer core-end to dispatch totaled 5.615 ms, with zero time in which all 64 recorded AIV cores were executing. This rules out full AIV saturation in those measured windows. Eligibility on a specific core and the origin of the residual delay remain unclassified. The scheduler recorded 51,894 async-poll intervals and 1,040 processed completions on that rank. These are cumulative batch events in the prewarm capture.

The trace exporter writes timing and dependency passes separately before combining them. Rank-local dispatch identifiers differed while the first request's conversions were underway. Use `measured-capture-view.py` to select the four prefills listed by the driver's measured-request manifest, then link the raw timing, graph and names into a common analysis view. The helper's SHA-256 is `5e137683de28bd50cd62f8ad3db56f3a4dc21f8fe82afb657b33f9dcbcc162f0`. Analysis job `diag59r-20261122T200956Z-analysis` waits for each arm to exit successfully, then computes the clock bounds, readiness decomposition and observed-order replay.

The full four-card Level 3 prewarm analysis verified the cross-rank phase calculation with no dropped scheduler phase records. From the start of the latest peer release to the next same-layer row staging dispatch, the 480 windows totaled 37.415–54.043 ms per rank across the conservative clock bounds. Within the corresponding readiness-lower-bound windows, all AIVs were simultaneously executing for at most 0.320 ms on any rank. The prewarm report SHA-256 is `7f1376c5dc4f85d6d310148b9645e06bedcede7ed143f473be79d3cbea1a9e9c`. The runtime reclaimed these raw prewarm shards when the next request collected its own trace. The prewarm script remains archived as `transport-readiness-v1.py`, and the reports were saved outside the reclaimed build subtree.

The control completed warmup and issued the measured request. Its rank 0 Level 3 `d2` raw record was written after the warmup answer. Follow actual dependency edges through untimed nodes until the nearest timed predecessors. This identifies `prefill_moe_results` as the latest timed predecessor for all 480 same-layer staging tasks. The median from latest upstream subtask FIN to staging dispatch is 5.955 µs. Looking only at direct timed predecessors gave 989.147 µs because their timing omitted intermediate dependencies. On this rank, local release start to post-mix start totals 15.077 ms, post-mix execution 7.598 ms, and post-mix end to staging dispatch 28.841 ms. The full release-to-staging interval includes the fixed post-mix and result-copy path.

The updated analyzer is `transport-readiness.py`, SHA-256 `dd0031a352039aa0f467ce7af6e1dda5955f169ca4dc715c0bd91eaae844e5fd`. The rank 0 warm probe is `diag59r-20261122T200956Z-control-rank0-warm-prereq-probe.json`, SHA-256 `2a19a31da2affddd8f23389b29dcf8343e55c21c511abe17ed61dbfab011a0fd`.

### Diagnostic 59 warm Prefill 55 control

The control and its analysis exited 0. The measured request used 1,024 input tokens and returned one token at the generation limit. Its manifest selected `d2` on four ranks. Rank 1 published payload readiness, and constant offset bounds fit. All four Level 3 scheduler streams report zero dropped phase records.

| Observed stage | Timing over the warm request |
| --- | ---: |
| Source push, latest traced local subtask FIN to dispatch | 4.701 ms on rank 1, median 8.305 µs |
| Receiver activation pack, latest traced local subtask FIN to dispatch | 46.174–47.043 ms per receiving rank, medians 85.930–87.764 µs |
| Receiver activation pack, source payload notify start to dispatch | 2.567–19.310 ms across receiving ranks and clock bounds |
| Source combine, latest traced local subtask FIN to dispatch | 28.975 ms, median 32.320 µs |
| Source combine, latest peer return notify start to dispatch | 3.072–13.431 ms across clock bounds |
| Same-layer staging, latest traced local subtask FIN to dispatch | 2.948–3.298 ms per rank, medians 5.901–6.010 µs |

The readiness-lower-bound windows for these stages contained zero time with all 64 recorded AIVs executing. The receiver activation windows begin well before source publication in aggregate. This favors reducing source preparation and push work. Source combine also remains gated by peer return completion. After including untimed dependencies, the next-round staging window mainly follows the fixed result copy. The async-poll record counts represent batch processing and do not identify an exact counter-visibility timestamp.

The warm source rank recorded 18.350 ms in histogram task spans, 7.549 ms in row planning and 29.586 ms in push task spans. The push medians were 53.946 µs end to end, 8.105 µs between the first and last subtask starts, and 49.155 µs for the longest individual subtask in each push. All 520 pushes had 56 timed subtasks. This measures substantial work inside the actual push blocks, the part addressed by the packed position and staged activation candidates. The source timing report is `diag59r-20261122T200956Z-control-measured-source-pack.json`, SHA-256 `5fd568edaaf20c9c44bf163de4782e1fe122f151a95d39aa08a8bf5b00656a30`.

The midpoint observed-order replay recovered 625.238 ms over 71,923 tasks. Zeroing source histogram, plan and push bodies shortens it by 36.882 ms. Zeroing return and release bodies shortens it by 23.748 ms. Zeroing all transport bodies shortens it by 67.644 ms. Zeroing those bodies and their preceding residuals leaves 489.168 ms. These are interventions on the measured warm trace with fixed core order and residuals outside the selected tasks. The Prefill 56 arm started at 20:58:20 UTC.

The warm artifacts share prefix `diag59r-20261122T200956Z-control-measured-`. SHA-256 for `readiness.json` is `c2eec12da0aa4cb6ac8c9c214f877744c78aecb491d9981ed95877823a9f1f46`. SHA-256 for `replay.json` is `d0840b17f5fd23510541f68db2fc26402fefc66c0c95a34a3629ad9bf4d2ea47`.

### Diagnostic 59 warm Prefill 56 candidate

The candidate, suite and four-rank analysis exited 0. After one 1,024-input, one-output-token warmup, the measured request used the same length and source rank as Prefill 55. All ranks reported zero dropped scheduler phase records. The observed-order replay spans 614.150 ms over 66,723 tasks, compared with 625.238 ms and 71,923 tasks for Prefill 55. Zeroing the Prefill 56 source preparation bodies removes 33.781 ms and zeroing return and release removes 23.743 ms. Across all compact transport, zeroing bodies removes 58.791 ms, and including their preceding residuals leaves 490.128 ms. These figures are for an instrumented request with fixed-order replays.

For the three receiving ranks, the latest traced local subtask FIN to pack dispatch rises to 115.631–117.002 ms after removing the independent row-plan dependency. The offset bounds place the source payload notification to pack dispatch at 2.740–16.922 ms in aggregate per receiver. All 64 AIVs were never simultaneously active in the corresponding readiness-lower-bound windows. The pack still follows the source's payload notification. Measuring from the last local predecessor now covers more of the source's work. The replay and readiness report SHA-256 hashes are `2793161f7283c924b623ac6bfc6e443fd6a663c37204fbafa678e4365a8037fb` and `caf15c5266ea7f855e7aa4847fe760cf1b8026da3b3c49e768d4cd31f8b54412`.

Both compact arms were also processed with the same per-round phase analyzer used for the cold native comparison. At midpoint clock offsets, the interval from source metadata start to push start decreases from 37.006 ms to 19.890 ms across 520 rounds, with medians of 69.279 and 36.688 µs. Push start to payload notify increases from 32.795 to 35.734 ms. Payload notify to the last-returning owner's pack start is similar at 15.132 and 14.930 ms. The last-returning owner is stable across feasible clock offsets for 274 Prefill 55 rounds and 292 Prefill 56 rounds. These interval sums identify where compact work moved when the separate row planner was removed. The per-arm reports end in `-measured-round-phase-spans.json` and hash to `9a40f60c0a2d6d0481593323869e8fdc642ad2487d2d78d4e1e013a2a42895fc` and `4b25506ab204096f87f1a9b98f9ca287e71da51ea6487730c112bba403f8df7e`.

For comparison with the native capture, the manifest-selected four ranks also give local first-task-start to last-task-end spans. Prefill 55 records 624.647, 624.016, 618.811 and 618.098 ms. Prefill 56 records 613.586, 612.966, 609.992 and 608.301 ms in rank order. The script `experiments/diag-59/measured-rank-spans.py` hashes to `9f98fc96bc31f69a2fa82f5d428d7b6adf7c57670558b052577c2b9360e68680`. The two `-measured-rank-spans.json` reports hash to `565ad70b1b9b0b049ce06cc7a3920838e6287c9f5e88daab67bdaf95adb2850e` and `f041a794798431f26a464017d5418dc36efbb6596c5cfb1a98df8c0e993dcf2f`.

#### Source push block imbalance

Each warm arm recorded 520 source pushes with 56 subtasks per push. Prefill 55 has a median per-round block mean of 8.929 µs and a median per-round longest block of 49.155 µs. Prefill 56 measures 8.843 and 48.928 µs respectively. Both arms have a median of 30 blocks under 5 µs per push. The imbalance persists after removing the separate row plan.

Independently, the 1,560 archived single-source layer-rounds contain 736,800 routes. Assignment by expert ID modulo 56 gives a median maximum of 63 routes on the busiest worker, with P95 of 101 and a median of 22 idle workers. Distributing the same route counts evenly over 56 workers would give a median maximum of nine. This count model has no timing or memory-transaction cost. It motivates moving remote work away from the busiest expert-assigned blocks.

The script `experiments/diag-59/push-imbalance.py` has SHA-256 `c56830a5bdd4b616ecfd9de8460e7c22ae7c423adee65c26ad51755c3835f576`. The two `measured-push-imbalance.json` reports hash to `b11731ba208bda9a6da641abc9029bf01d035964ba5da4823215660acb870ce8` and `7b1c8ac92a32112744b0b33c5b4b98385aa5a346ecd9fcd0d6f5b1e5fb406b4d`. The same-profile native arm is queued on the device lock. Prefill 61 acquired the lock next at 21:16 UTC.

## Prefill 60 · Read route weights from the prepared compact packet

Start from Prefill 57. Its compact scale-unpack task already writes a 32-byte prefix ahead of each token's scales. Load the six exact FP32 route weights with the same 16-token tile and write their bits into this prefix. Push chooses the weight from the packet already loaded for the route and reads it before writing route index, source rank and selected weight into the descriptor. Receivers consume those three descriptor fields and the same scale bytes. The extra input to preparation makes it depend on routing as well as quantization. The scratch size, sent bytes, communication epochs and slot are the same as Prefill 57. Native sends scales and weights in separate lane buffers.

Both compact slabs, the native slab, and complete native and compact Serving passed host compilation with PTOAS 0.72. The 109 native Prefill kernels and both sets of 45 Decode kernels match Prefill 57. Compact device changes are limited to `dispatch_scale_unpack` and `moonep_push`, with identical 250 submit sites and 134 block assignments. The generated push has no 80-row weight load and still has two GM release fences. Packet preparation has one valid 16-row weight load and one extra tile move.

The server verified the source and archive SHA-256 `20a4e7ee8517cdfa8c61a861764b0fedb7df031cbf2de2538423fd54aa20e846` under `experiments/prefill-60`. Suite `p60-20261122T191315Z` is queued after Diagnostic 59 and requires the Prefill 57 compact control to pass. It runs the two frozen fixtures, then Serving in candidate, Prefill 57, native order under the original lock.

Prefill 60 took the lock after the first Diagnostic 59 attempt exited. The single-source and mixed-source replays both exited 0. The candidate began Serving at 20:12:36 UTC and reached health in 294.093 s.

The candidate exited 0 with warm Prefill P50/P95 of 2.217976/4.413646 s and TTFT P50/P95 of 2.296757/4.497464 s. Its Prefill P50 values were 0.761317, 2.217976 and 4.406976 s at 1,024, 3,070 and 6,142 input tokens. All 12 measured requests returned 13 output tokens at EOS. The long answer and both concurrent waves passed.

The Prefill 57 control started at 20:21:50 UTC, reached health in 310.046 s and exited 0. It recorded warm Prefill P50/P95 2.186241/4.352117 s and length medians 0.762277, 2.186241 and 4.302580 s. The normalized compact commands, prompt lengths, request order and output lengths match. The Prefill 60 candidate P50 was 1.45% higher than its same-run compact control. Two of 12 paired requests improved. The matched native arm began at 20:31:19 UTC.

### Prefill 60 device result

Both frozen replays, all three Serving arms and the matched suite exited 0. The normalized launch commands match. All 36 timed requests returned 13 tokens at EOS in the fixed input order. The native arm's warm Prefill P50/P95 was 2.127020/4.280170 s, with P50 of 0.736461, 2.127020 and 4.260594 s at 1,024, 3,070 and 6,142 input tokens. Native TTFT P50/P95 was 2.204922/4.366759 s. Prefill 57 and Prefill 60 were 2.78% and 4.28% slower than same-run native by Prefill P50, respectively. Prefill 60 improved zero of 12 pairs against native. Retain the Prefill 57 control over the prepared-weight variant.

Compare Prefill 57 with native across the two opposite orders. The native P50 moved from 3.618883 s with native first to 2.127020 s with native last, a 41.22% drop under the same normalized command and input sequence. Equal sample pooling gives native 2.833536 s and Prefill 57 2.212174 s, a numerical reduction of 21.93% with 13 improved pairs out of 24. The pooled P50s at 1,024, 3,070 and 6,142 input tokens are 0.951957/0.780880, 2.712562/2.212174 and 5.000628/4.354494 s. The first run's apparent 30% result does not reproduce with the reversed order. Raw results are in `p60-20261122T191315Z-comparison.json` and `p57-20261122T183501Z-comparison.json`.

## Prefill 61 · Send expert-ready scales and weights in one aligned packet

Start from Prefill 58. In the sole-source compact path, the sender knows each expert's aligned receive rows, and the owner uses the return plan in place of per-route descriptors. In the existing push task, group an expert's scales and FP32 route weights into packets of up to 16 active rows. Transpose the scale pairs into the exact MX_A_ZZ byte order used by W13, with code 127 on the inactive rows. Group weights follow the padded scale bytes. Store each packet into 14 rows of the existing 48-word receive window, starting at its 16-aligned group offset. Activation stores, receiver plan, epoch notifications and four-row BF16 return follow Prefill 58. Mixed sources continue to send per-route packets with descriptors.

The receiving task consumes the packet with one load and copies its already-packed scales into the existing expert operand tensor. It reads weight bits from the same packet and copies only active weights. This removes receiver scale-transpose work at the cost of source packing and metadata padding. The routed expert kernels, MoE width of 80 rows and serial single-slot graph are unchanged. Native keeps its per-source expert lanes and individual return descriptors.

The three real-model captures include 1,560 layer-rounds. Compact cross-rank metadata writes fall from 558,268 to 75,097 in the route-count model. Including the unchanged activation writes, total cross-rank forward writes fall from 1,116,536 to 633,365 (43.27%). Transmitted forward bytes rise by 3.19%, from 2,965,519,616 to 3,060,192,896, because the scale packets carry padding. The modeled counts are in `experiments/prefill-61/forward-group-count.json`, SHA-256 `58a74fff4a0abf8a938fb26e63825650473fdf31155146ad3bd323ae4a4c5ae7`.

Both frozen slab shapes, the native slab and complete native and compact Serving passed host compilation with PTOAS 0.72. Only `moonep_push` and `moonep_pack_scale` changed in the compact graph. The generated packet transfer is 14 by 48 INT32 words. Native Prefill, both Decode builds, routed expert kernels, 250 submit sites and 134 block assignments all match Prefill 58. The server verified archive SHA-256 `9cd3cc3d19f9c224d34634a5fa1762130c37b9e1adbb8dea2e2b5226f788e4cd`. Suite `p61-20261122T193335Z` waits for Prefill 60 and a successful Prefill 58 control. Serving order is candidate, Prefill 58, native.

The frozen 1,024-token replay exercises up to five groups per expert, all with 16 valid rows. Add a one-round single-source replay with active counts `79,0,0,0` and capacity 80 to check the partial group. `make-tail-golden.py` reuses the frozen weights and first slab inputs and calls the existing `golden_moe_slabs` reference. Its three generated outputs match the original reference exactly on all 79 active rows. The reference gate selects six experts with 79 rows each, giving every expert a final group of 15. The queued suite runs single, mixed and tail replays before Serving. The fixture coverage report `experiments/prefill-61/golden-tail-coverage.json` has SHA-256 `1e9324e3669a7d09fbf0f17d4988e4d24cb941609728bcb771d89cf2b0c039e7`.

### Prefill 61 device result

The single-source, mixed-source and 79-row tail replays, all three Serving arms, matched-command validation and report generator exited 0. Twelve measured requests per arm contain four requests for each of the three lengths. All paired request IDs and token counts match.

| Input tokens | Native P50 (s) | Prefill 58 control P50 (s) | Prefill 61 candidate P50 (s) |
| ---: | ---: | ---: | ---: |
| 1,024 | 0.726869 | 0.768294 | 0.755344 |
| 3,070 | 2.183673 | 2.235504 | 2.192666 |
| 6,142 | 4.379483 | 4.449654 | 4.369360 |
| All twelve requests | 2.183673 | 2.235504 | 2.192666 |

The candidate lowers P50 by 1.92% against Prefill 58, with 11 of 12 paired requests improving. Against native, its pooled P50 rises 0.41%, with 5 of 12 pairs improving. The 30% target on this matched native run is 1.528571 s, 0.664096 s below the candidate. Reports are in `p61-20261122T193335Z-comparison.json`, SHA-256 `3a3e545e3a5d4eae92246fa255c5214bed20cd851fdefc39fc9644be020f5c11`. The same-profile Diagnostic 67 native capture started after this suite released the lock.

## Prefill 62 · Write compact metadata directly into the expert operand windows

Start from Prefill 61. Send each group's MX_A_ZZ scale bytes and exact FP32 weights with two stores into separate expert operand windows in the existing communication slot. The same windows receive locally packed operands for mixed sources, which retain their existing per-route wire format. The sole-source receiver uses one vector block to unpack the row plan. Activation placement, four-row return stores and the release protocol follow Prefill 61.

The route-count model over 1,560 archived layer-rounds estimates 708,462 cross-rank forward writes instead of 633,365 and 3,055,386,688 forward bytes instead of 3,060,192,896. Direct stores eliminate the receiver's packet reads and copies. The 32-to-1 change in pack blocks removes 193,440 receive-side blocks across four ranks in the captured workload. Shared operand storage rises by 646,272 bytes per rank, while the generated graph stops allocating 802,944 bytes of private operand scratch in each active layer-round. The modeled results are in `experiments/prefill-62/direct-operand-count.json`, SHA-256 `b0ebd47d7ae9991a015787937b0768fd0fb3b05e81099eb41e0fb44fa6d54dfc`.

Both compact slabs, the native slab and full native and compact Serving passed host compilation with PTOAS 0.72. The canonical kernel comparison changes only `moonep_push` and `moonep_pack_scale`. Native Prefill, both Decode builds and routed expert kernels match Prefill 61. Counting dynamic block sizes and mixed launches gives 230 block assignments and 289 submit sites in the control. The candidate changes only the compact pack block count and adds one dependency-only fence that joins MoonEP operand readiness before W13. The sequence of other submit sites, the expert launch sizes, 80-row width and serial single slot match. The audit is in `experiments/prefill-62/host-codegen.txt`.

The server verified both source manifests and archive SHA-256 `11dc1062210f1a99a509069ad9fc44d1ac33671d4748687db2f203a8639e6277`. Suite `p62-20261122T195718Z` waits for a successful Prefill 61. It runs the single, mixed and 79-row tail replays and then Serving in native, Prefill 61, candidate order. Device precision and latency are pending.

### Prefill 62 device result

The three frozen replays, three Serving arms and suite exited 0. All normalized launch commands matched. Each arm ran 12 warm requests with four requests at each input length. The long answer and concurrency waves passed.

| Input tokens | Native Prefill P50 (s) | Prefill 61 P50 (s) | Prefill 62 P50 (s) | Reduction vs native |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.689090 | 0.692044 | 0.676551 | 1.82% |
| 3,070 | 2.084674 | 2.086486 | 2.038489 | 2.22% |
| 6,142 | 4.172891 | 4.181944 | 4.093428 | 1.90% |
| All 12 | 2.084674 | 2.086486 | 2.038489 | 2.22% |

The candidate P50 is 2.30% lower than the compact control. All 12 request pairs improved against each control. Prefill P95 was 4.181267, 4.194752 and 4.112069 s in native, control and candidate order. The same-run comparison remains below the 30% requirement. Report `p62-20261122T195718Z-comparison.json` has SHA-256 `e775841857bdf683e4f79a64a843ba1d238399042a78b9a472aadc60620ccc19`.

## Prefill 63 · Restrict return notifications to the single result reader

Prototype the source-only completion protocol on Prefill 58's serial single-slot graph. In a single-source round, the other three ranks signal their return completion to the source. That rank waits for all three signals and still accumulates in exact TopK order. The other ranks join their own scatter notification and local output initialization before release. Every rank still participates in the separate round-release and window-reuse barrier. Mixed-source and native retain all-to-all return completion.

The single-source protocol uses three remote return signals, three wait conditions and one reduction grid, down from twelve, twelve and four. Native slabs, native Serving and both Decode builds emit the same kernels as Prefill 58. Compact slabs and Serving change only the return notification kernel and emit its original body separately for activation notification. All 230 generated block assignments match. The earlier two-slot Prefill 39 used the same notification idea on interior rounds and was 2.38% slower than its Prefill 38 control, with P50 of 1.999497 s vs 1.953066 s. Keep the single-slot prototype at host verification until the warm scheduler trace shows enough return-handshake cost to justify a device comparison.

## Prefill 64 · Consume the receive plan directly from the compact histogram window

Start from Prefill 62. Single-source ranks already receive expert-ready activations, scales and FP32 weights. Append the 96 counts and 96 aligned start offsets to each existing 96-word owner plan in the histogram store. W13, SwiGLU and W2 consume the same count and offset values through contiguous views. The payload-ready chain gates those reads. Single-source ranks skip the receive pack task. For mixed sources, finish all histogram reads before storing the locally computed counts and starts in two slices of the same window. The mixed-source pack task keeps its 32 blocks and its per-route wire format.

The plan write grows from 384 to 1,152 bytes per owner. The route-count model over 1,560 archived layer-rounds estimates 3,594,240 additional cross-rank plan bytes and 6,240 fewer receive pack tasks across the four ranks. The number of plan writes, sizes of shared buffers, 80-row width, serial slot reuse and expert kernel launches are unchanged. `plan-window-count.json` has SHA-256 `c5ae4aaaab6feba94061e6b1ea50c2aa7155d6b7b6c31d1196ac163e72a26991`.

Both compact slabs, the native slab and full native and compact Serving passed host compilation at 80 rows. Only the four MoonEP kernels `moonep_histogram`, `moonep_row_plan`, `moonep_pack_scale` and `moonep_scatter` differ from Prefill 62. Native Prefill, both Decode builds and the routed expert kernels match. The generated graph keeps 230 block assignments and drops one MoonEP readiness fence. The two plan views begin at columns 96 and 192 of the existing `[4,384]` histogram. The mixed-source planner passes the whole histogram as its single writable argument. Host compilation leaves the device correctness and latency effects to the frozen replays and Serving comparison.

The fixed and online standalone layouts also compiled with the shared count and start view orientation. The server verified source manifests and archive SHA-256 `4af26e744210d54b7039e74cc0c110fb015601af929d14cd92489b9765314896` under `experiments/prefill-64`. Suite `p64-20261122T202030Z` waits for a successful Prefill 62 run. It uses the same cards and records single-source, mixed-source and 79-row tail replays, then Serving in candidate, Prefill 62, native order.

After Prefill 62 exited 0, Prefill 64 acquired the device lock and started its single-source replay at 22:49:52 UTC. Prefill 76 is waiting for the same device lock.

All three frozen replays passed each of the three numerical comparisons. The candidate Serving arm exited 0 with Prefill P50 of 2.094150 s on 12 warm requests, each generating 13 output tokens and stopping at EOS. The Prefill 62 arm started at 23:06:58 UTC and uses the matching normalized launch command. The native arm and same-run speedup are pending. The candidate `http.json` hash is `30766ebae0e611e3b7f26c3bc095bac46e60a038e464f7d06300303362718bc0`.

The Prefill 62 control arm also exited 0, with the same tokens and completion behavior. Its Prefill P50 is 2.090198 s and its P95 is 4.197928 s. The candidate P50 is 0.19% higher and its P95 is 4.351425 s. The candidate is faster in three of 12 pairs. The native arm started at 23:16:17 UTC. Two-arm report `p64-20261122T202030Z-partial-comparison.json` has SHA-256 `58c7175973cc592e9ae18a6254dbf7e7af973d62bb8b891c0dd4d9d44f2c94ca`.

The separately collected frozen replays show lower effective device median at each shape. In the 1,024-token replay it moves from 14.930 to 14.116 ms, and the mean host union span grows from 36.407 to 41.606 ms. The frozen routing patterns and benchmark timing boundaries differ from the matched Serving requests. The same-run Serving comparison measures the change at whole-request scope.

### Prefill 64 device result

All three Serving arms and the suite exited 0. The normalized commands and token counts match. The matched check includes the long answer and both concurrency waves.

| Input tokens | Native Prefill P50 (s) | Prefill 62 P50 (s) | Prefill 64 P50 (s) |
| ---: | ---: | ---: | ---: |
| 1,024 | 0.707731 | 0.694471 | 0.695916 |
| 3,070 | 2.129084 | 2.090198 | 2.094150 |
| 6,142 | 4.269807 | 4.191062 | 4.195690 |
| All 12 | 2.129084 | 2.090198 | 2.094150 |

Prefill 64 P50 is 1.64% lower than native and 0.19% higher than Prefill 62. It improves 10 of 12 pairs against native and three against Prefill 62. Whole-workload P95 is 4.291894, 4.197928 and 4.351425 s respectively. The same-run 30% threshold is 1.490359 s. Report `p64-20261122T202030Z-comparison.json` hashes to `4aa0620dc7c83e9baa988aad3674533843139657c49a6e87a053fb0b9ab51e01`.

## Prefill 65 · Pack source route positions in the compact return-row order

Start from the device-validated Prefill 58 layout. In a single-source round, the histogram task already assigns each route a compact return row in owner and expert order. Save the route's ordinal within its expert in UB during the counting pass. When building the return lookup, write the route ID into a dense UB tile at that return row. Store the position tile once at the end of the task. The local source plan carries each expert's dense start in its spare high bits. Strip those bits from the transmitted plan and keep the existing owner return header. The push task starts its position load on a 32-byte boundary and skips the leading entries. It reads only the active positions and that prefix.

Mixed sources retain the per-expert sparse position layout. Their scalar writes and the new single-source vector store use separate local buffers, keeping both memory paths coherent. The extra packed buffer and load padding add 2,272 logical scratch bytes per source-dispatch instance. The communication buffers, activation and descriptor packets, notifications and round release keep their existing sizes and order.

The position model uses three archived real-model captures containing 1,560 single-source layer-rounds and 736,800 routes. It replaces 736,800 scalar position stores with 1,560 vector stores. The existing full-row position loads span 25,987,200 bytes. The candidate spans 4,063,508 valid bytes, or 5,184,000 bytes after 32-byte rounding, a modeled reduction of 80.05%. The new position stores write 2,995,200 bytes. These are local position counts. Cross-rank payload bytes follow Prefill 58. The model and report are `experiments/prefill-65/count-route-positions.py` and `route-position-count.json`, with report SHA-256 `c1c231a8cfb6ba36d2e441622464ac3592d1c2d33a540db3fec77f433a926c17`.

Host compilation passes for both compact slabs, the 79-row tail slab, the native slab, full Prefill and Decode. The canonical compact kernel comparison changes only `moonep_histogram` and `moonep_push`. All routed expert and native kernels, 230 block assignments, 289 submit sites and communication window specifications match Prefill 58. The generated push uses dynamically sized position loads. Width remains 80 rows with one serially reused slot.

The earlier 1,024-token Diagnostic 50 measured 18.225 ms in source histogram tasks and 34.528 ms in source push tasks. The active spans total 52.753 ms in a 620.197 ms rank span. The Prefill 58 matched native P50 puts the 30% target at 1.560511 s, 0.643663 s below the compact candidate. The source position change covers part of two tasks in the earlier device decomposition. Use the warm scheduler capture to assess additional savings in readiness and return on the critical path.

The server verified archive SHA-256 `1d115a7916d04e184601376e95bc4a4add0219c468b81e767c5eef5202f42463` and the source manifests. Suite `p65-20261122T202844Z` waits for Prefill 64 to finish and uses Prefill 58 as its compact control. It runs the single-source, mixed-source and 79-row tail replays, then Serving in candidate, Prefill 58, native order. The 79-row replay exercises nonzero alignment prefixes on the source position views. Device precision and latency are pending.

Prefill 65 acquired the four-card lock at 00:01:13 UTC on November 23. The single-source, mixed-source and 79-row tail replays exited 0. Candidate Serving started at 00:08:38 UTC and reached health in 298.120 s.

The candidate exited 0. Warm Prefill P50 and P95 are 2.103818 and 4.234568 s over 12 requests. Prefill P50 by input length is 0.697177 s at 1,024, 2.103818 s at 3,070 and 4.222569 s at 6,142 tokens. All 12 generated 13 tokens with the expected stop reason. The candidate `http.json` has SHA-256 `63223300f657cc2b79f6a46c8aeed1a098c7ac9b54ede67754146e98f8590e27`. The same-suite Prefill 58 control started at 00:17:55 UTC and its normalized launch command matches the candidate. The native arm follows it.

The Prefill 58 control exited 0. Its P50 and P95 are 2.136809 and 4.302171 s. The candidate improves all 12 paired requests, with overall P50 and P95 reductions of 1.54% and 1.57%. Control P50 by input length is 0.710592, 2.136809 and 4.288593 s. Both arms passed the long-answer and concurrency checks. The native arm started at 00:27:16 UTC. All three normalized launch commands match. The partial report `p65-20261122T202844Z-partial-comparison.json` hashes to `febd80ba26103fd1cd2f8b1752d2b2291cb1ff195fbb1b30394be41f714f128b`.

### Prefill 65 device result

The native arm and suite exited 0 at 00:36:21 UTC. The three arms match on all 12 ordered prompt lengths, output lengths and stop reasons. The long-answer and both concurrency checks passed.

| Arm | Warm Prefill P50 (s) | Warm Prefill P95 (s) | 1,024 P50 (s) | 3,070 P50 (s) | 6,142 P50 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 2.056734 | 4.126367 | 0.681600 | 2.056734 | 4.117178 |
| Prefill 58 control | 2.136809 | 4.302171 | 0.710592 | 2.136809 | 4.288593 |
| Prefill 65 candidate | 2.103818 | 4.234568 | 0.697177 | 2.103818 | 4.222569 |

Packing source positions improves Prefill 58 by 1.89%, 1.54% and 1.54% P50 across the three lengths. Against native, the candidate's overall P50 and P95 increase by 2.29% and 2.62%. All 12 native pairs favor native. The 30%-lower P50 is 1.439714 s in this suite. The full report `p65-20261122T202844Z-comparison.json` hashes to `216113a1a1d4d1f3be53d3168d40f05271230a554666e17a95f1bae46fb70424`. Prefill 79 acquired the original four-card lock next and started the spill fixture at 00:36:25 UTC.

## Prefill 66 · Share one staged activation per remote token owner

Start from Prefill 58. In a single-source round, the histogram marks the first route from each token to each owner. The source sends that activation once to an 80-row staging slice beginning at row 4,816 in the existing 4,896-row receive window. The remote owner uses the original route ID in the scale/route packet to expand the staged activation into the original expert input rows. This expansion runs inside the existing receive scale pack task. The source's local experts retain direct activation stores. Mixed-source dispatch, scale and route packets, exact FP32 weights and BF16 return data follow the control.

The aligned expert input for a single source uses at most `80 × 6 + 96 × 31 = 3,456` rows. That leaves the staging slice disjoint even when all active routes target one owner. All readers remain behind the existing payload notification and slot release. No communication window is allocated and the expert kernel submissions, worker counts, round width and serial slot reuse match Prefill 58.

On the 1,560 archived single-source layer-rounds, 558,268 cross-rank routes reference 304,282 unique token-owner pairs. Sending one 5,120-byte activation per pair saves 1,300,408,320 logical cross-rank bytes, or 45.50% of activation bytes, 43.85% of forward bytes including descriptors and 14.98% of forward-plus-return bytes. The receiver performs 558,268 local copies, equivalent to 5,716,664,320 bytes of combined reads and writes. The model is `activation-traffic.json`, SHA-256 `9bccf5c2424296680cfae93ceb833dfbd5426e77e54b05385cbc146b505f3767`. The measured trade-off is the shorter source push against the added receive work. The earlier Prefill 15b deduplication variant at 256 rows regressed 4.88% against its same-width compact control.

Host compilation passed for both compact slabs, the native slab, the 79-row tail and complete native and compact Serving. Only `moonep_histogram`, `moonep_push` and `moonep_pack_scale` changed. All routed expert, native Prefill and Decode kernels, 230 block assignments, 289 submit sites and communication buffer specifications match. Generated code uses the same staging offset for the sender and owner. Device precision and latency require the matched replays and Serving test.

The server verified archive SHA-256 `9012e60ecb8b27f8afecb20da6c1cfaba42143d1b24c3a0a4ea8ca9b4a9eefa2` and both source manifests. Suite `p66-20261122T203833Z` waits for Prefill 65 to finish. It runs the single, mixed and 79-row tail replays before candidate, Prefill 58 and native Serving on the original lock.

## Diagnostic 67 · Capture warm native on the same scheduler profile

The cold native trace and the warm compact scheduler trace cannot by themselves attribute steady-state latency differences. Queue one native tracing arm `diag67-20261122T204618Z` after Diagnostic 59. Use the exact native pypto-lib snapshot from Prefill 16 and the same 1,024-input, one-output-token warmup and measured requests as the compact tracing arms. The trace Serving package and ordinary benchmark Serving package differ only by the capture configuration in `npu_runner.py`. The run uses the existing device lock, prepacked sidecar and 80-row settings. It does not replace the uninstrumented acceptance benchmark.

The postprocessor `experiments/diag-59/warm-native-phases.py` waits for a successful capture through analysis job `diag67-20261122T204618Z-analysis`. It builds the view from the measured request manifest and runs the archived clock-bounds and per-round phase analyses. On the existing Diagnostic 50 native capture, the source detector recovers rank 0 with a median push span of 57.911 µs. The other three ranks have medians of 16.830–17.980 µs. It will infer the warm source independently from native push spans and record all rank durations. The script hash is `bc78aeb66866e1b197abbe8a0d768f11fcb5b074ef62f08ef6a2ae1d3c1102f7`.

### Diagnostic 67 warm native result

The native suite and analysis exited 0. Warmup and measurement each used 1,024 input tokens and one output token. The manifest selects the dedicated measured `d2` traces on all four ranks. Source identification from the native push durations gives rank 1, matching the measured Prefill 56 arm. Constant clock offsets are feasible.

| Rank | Native device span (ms) | Prefill 56 device span (ms) |
| ---: | ---: | ---: |
| 0 | 609.419 | 613.586 |
| 1, source | 612.304 | 612.966 |
| 2 | 607.576 | 609.992 |
| 3 | 608.142 | 608.301 |

The longest rank span is 1.282 ms higher for Prefill 56 (+0.21%). Per-round timing was measured over 520 rounds using the midpoint clock alignment. The owner below is the one publishing return readiness last in its round. That choice is stable across clock bounds in 295 native rounds and 292 Prefill 56 rounds.

| Phase | Native total (ms) | Prefill 56 total (ms) |
| --- | ---: | ---: |
| Row staging to source metadata | 191.629 | 188.791 |
| Source metadata start to push start | 23.307 | 19.890 |
| Push start to payload notify start | 35.650 | 35.734 |
| Payload notify to owner pack start | 13.020 | 14.930 |
| Owner pack start to scatter start | 227.128 | 231.716 |
| Scatter start to owner return notify | 32.067 | 31.981 |
| Last return notify to reduce start | 7.789 | 7.842 |
| Source reduction span | 7.665 | 7.684 |

Native source push records 31.936 ms of task spans and a 58.092 µs median. The other native ranks record 10.421–10.581 ms each in their own push spans. Prefill 56's source metadata interval saves 3.418 ms, while the receiver-side notify-to-pack and pack-to-scatter intervals add 1.911 and 4.589 ms. The latter interval includes expert execution and the gaps between kernels. Local and cross-rank intervals support separate attribution checks within the round. Ordinary uninstrumented Serving remains the acceptance measurement.

The native `-measured-source-push.json`, `-measured-clocks.json` and `-measured-round-phase-spans.json` hashes are `48eafd9c4e17fc9f9a2782ae2e8e2d3c6d45d68e53acd34044d3f83a27330641`, `fee99488f3f41ff8deeacde354c0e53102215542464728f906cf9a4250557595` and `066a13963bd172722749a2db8ea1f24ded363fd0b13bb0b6d4985310863e4210`. Prefill 62 acquired the device lock next. Its single-source replay exited 0 and the mixed-source replay started at 22:17:07 UTC.

#### Pack completion to routed expert start

Split the pack-to-scatter interval at the actual W13, SwiGLU and W2 task boundaries on each rank. Using each arm's last-returning owner, the segments sum to the previously reported 227.128 ms native and 231.716 ms compact over 520 rounds. The intervals have no overlapping stage bodies in these traces.

| Segment | Native (ms) | Prefill 56 (ms) |
| --- | ---: | ---: |
| Pack body | 8.060 | 3.801 |
| Pack end to W13 start | 9.998 | 14.647 |
| W13 body | 74.750 | 78.444 |
| W13 end to SwiGLU start | 21.929 | 23.730 |
| SwiGLU body | 8.587 | 6.828 |
| SwiGLU end to W2 start | 20.087 | 20.913 |
| W2 body | 59.835 | 59.926 |
| W2 end to scatter start | 23.881 | 23.426 |

Using the native last-returning rank for both arms in each of the 520 rounds, the total comparable interval changes from 227.128 to 225.848 ms. The pack saves 4.207 ms, while pack end to W13 start still grows by 3.859 ms. In the 134 rounds with the same stable last owner in both arms, the gap grows from 2.340 to 3.682 ms. The timings identify a compact handoff cost to investigate. Generated compact code submits a dummy task waiting for the pack and its plan before W13, even though the pack already depends on that plan. The native launch uses the gather pack task directly. The observed gap can include other scheduling delays.

The paired analysis is `experiments/diag-59/owner-pack-stages.py` with report `owner-pack-stages.json`. Their SHA-256 hashes are `e2e3e07b7678cbe9b005fe90c47b5c60b639eb40d5bb66862a0694e1f4744503` and `65f76b373f291589ba561139c8bde0272dc1f0ade9dfbac5cdb93fb058fa4874`.

The level-3 scheduler records further split the pack-to-W13 gap. For the full 520-round pairing using the native-selected rank in both arms, pack AICore end to recorded FIN totals 2.307 ms native and 4.878 ms compact. Recorded FIN to W13 dispatch totals 6.753 and 8.039 ms, and dispatch to first execution totals 0.938 and 0.941 ms. This puts about 2.57 ms of the 3.86 ms difference before FIN and 1.29 ms between FIN and W13 dispatch. The sum retains three negative end-to-FIN intervals in the native trace. The dependency-only task removed in Prefill 76 falls after FIN. Prefill 62 and Prefill 64 test less receiver pack work and fewer receiver pack blocks.

The additional postprocessor `experiments/diag-59/owner-pack-scheduler.py` and report `owner-pack-scheduler.json` hash to `5beb991784bd2ec5665f11c3214e3bee799f429e428bcd71e1ee378e917b82e7` and `61b8ca4c994918367b294c29566b69d31e06c8562294f651e8517d89d27d0e2c`.

For the same-native-owner pairing, pack end to recorded FIN is longer in 414 of 520 compact rounds. Its marginal median changes from 2.751 to 8.247 µs and its paired excess median is 4.440 µs. The largest 26 paired excesses account for 0.575 ms of the total 2.571 ms increase (22.36%). The same split after FIN shows 424 slower rounds, a 2.501 µs paired excess median and 0.310 of 1.286 ms excess in the largest 26 rounds. The overhead is distributed across the request, which motivates reducing the number of receive pack blocks. The script `experiments/diag-59/owner-pack-tails.py` and report `owner-pack-tails.json` hash to `0772056206da80a68e509b2102f65230bed665dd8744376224c1ab70175a2ba5` and `5e9de06d772c6764909f1c85b4cc9234f601803fde2d9c3b6c1eaa281dbdcab4`.

#### Transport budget against the aligned warm native trace

At the midpoint clock offsets, the native four-rank task envelope spans 612.304 ms and the observed Prefill 56 envelope spans 614.150 ms. Thirty percent below the native span is 428.613 ms. In the existing fixed-order Prefill 56 replay, zeroing all designated transport task bodies leaves 555.359 ms. Zeroing their entry gaps as well leaves 490.128 ms, 61.515 ms above the same-trace 30% threshold. The replay retains measured order, other residuals and task durations. Transport changes can also affect contention and those retained residuals on device. This quantifies the gap for the specific instrumented 1,024-token request and points to examining the timing effects at both ends of transport handoffs. The Serving criterion uses the uninstrumented matched measurements.

The script `experiments/diag-59/phase-budget-reference.py` reuses the trace reader and clock bounds and combines the existing Prefill 56 replay with the native envelope. Its SHA-256 is `e6b82da49a8186ac2b16a6812ebe97c2ecb9ba7f24570b786a6d69e0bd9b8ecb`. The report `phase-budget-reference.json` hashes to `259eefe0687307330155c6423b5ecda8e5c76930c46764e78d8c5d3d3fee3a40`.

The follow-up replay matches the 614.150 and 490.128 ms references at the midpoint clock offsets. For each immediate nontransport successor, it finds the last observed input among logical dependencies and recorded core-order edges. Removing the residual at successors with a last logical input from transport brings the span to 469.402 ms, 40.789 ms above the trace threshold. It selects 2,068 of 2,080 W13 tasks and 2,070 of 2,080 HC post tasks. All W13 direct-successor gaps on their own bring the 490.128 ms replay to 475.986 ms. The early and late alignments give spans within 0.002 ms of each result.

Removing every direct-successor residual yields 420.003 ms. This includes 2,080 SwiGLU and 2,080 W2 residuals whose recorded last inputs were outside transport. This broader result depends on eliminating wait outside the demonstrated MoonEP handoffs. The device comparison requires changing only MoonEP while retaining the same expert execution. The replay keeps the observed order and provides an attribution diagnostic for a single 1,024-token capture.

The archived script `experiments/diag-59/transport-following-gaps-v2.py` and report `transport-following-gaps-v2.json` have SHA-256 `b24dd559816d305b134c9007dcac4d8a30dc5bfa61d02bdb73db1d51d9344f8b` and `216d455723ce06ad43ab7a53bfc7ad22ce2a173288f507d70e72b5d94246b7e9`.

#### Return scatter and completion publication

Split the four-rank warm traces over 520 rounds, selecting the last-notifying native owner and using that same owner in both arms. The source is rank 1 in both. The comparison keeps the sign of scheduler-to-device time differences.

| Return phase | Native (ms) | Prefill 56 (ms) |
| --- | ---: | ---: |
| Scatter AICore execution span | 18.108 | 17.305 |
| Scatter AICore end to recorded FIN | 11.201 | 11.152 |
| Recorded FIN to notify dispatch | 1.826 | 2.218 |
| Notify dispatch to AICore start | 0.933 | 0.924 |
| Notify AICore execution span | 2.331 | 2.420 |

More than 11 ms in each arm lies between the selected scatter's AICore end and its registered FIN. On the source, the offset bounds put the signed interval from the latest notify start to reduce dispatch at 2.132–11.279 ms native and 2.102–11.412 ms compact. Reduce dispatch to first execution totals about 0.91 ms in both. The counter can become visible during the notify task. Prefill 56 uses the earlier per-route scatter, which limits inference about the sixteen-block candidate. The result motivates checking whether publishing each compact block's completed stores directly can remove the runtime FIN handoff before source reduction.

`experiments/diag-59/return-notify-stages.py` and `return-notify-stages.json` hash to `fe6c14255a50626440886eb1c5d85ef9b409a140ddc5886a415aafd8cac4b5bc` and `d43eb515b317dd146ec498f8fa3575032c97847e112a368a629b6b31a4c81e6c`.

#### Warm transport sensitivity by phase

Reuse the Prefill 56 measured views, clock bounds and archived observed-order replay. Reproducing the midpoint span yields 614.150 ms. Partition transport tasks into forward dispatch, receive pack, return scatter and publication, return wait and reduction, and round release. The source has two `combine_notify` tasks per round. Its first notification belongs to forward dispatch and the second to return publication. All other ranks' notifications belong to return publication. Retain all other task times, delays and first-core order for each intervention.

| Selected transport tasks | Tasks | Reduction without task bodies (ms) | Reduction without entry gaps (ms) | Reduction without either (ms) |
| --- | ---: | ---: | ---: | ---: |
| Forward dispatch | 3,120 | 34.127 | 16.647 | 38.875 |
| Source push alone | 520 | 27.761 | 6.721 | 31.957 |
| Receive pack | 2,080 | 3.514 | 7.024 | 10.396 |
| Return scatter and publication | 4,160 | 11.861 | 27.633 | 42.941 |
| Return wait and reduction | 4,160 | 8.006 | 16.658 | 24.484 |
| Round release | 4,160 | 3.620 | 9.277 | 12.875 |
| Source push plus return scatter and publication | 4,680 | 40.081 | 34.452 | 76.361 |

The combined push and return-publication row spans 75.442–77.036 ms across the three clock alignments. The largest return opportunity in this trace is before the selected tasks start. The largest source push opportunity is in the task body. This separates the mechanisms tested by the direct activation transfer candidates and the direct return publications. Entry gaps include recorded scheduling and readiness effects, and the results are device-span sensitivities for one 1,024-token capture.

The narrower Prefill 81 intervention covers 2,080 return notifications and the three peer ranks' 3,120 return wait and empty reduction tasks. At the midpoint, zeroing notification bodies alone saves 1.191 ms, and including their entry gaps saves 5.799 ms. Zeroing bodies and gaps for all 5,200 affected tasks saves 5.912 ms, with a 4.070–8.033 ms alignment range. Adding source push bodies and gaps to that intervention saves 42.185 ms. The fixed-order replay holds scatter itself intact in these cases and omits the new joins and per-block publications. The larger 42.941 ms return-publication row above also removes scatter and its entry gaps. The report `diag59r-20261122T200956Z-return-notify-sensitivity-001624.json` hashes to `1d5bbd6c74e076ba6f12f2142e1c61c6440bec322cfdec29f52d1fb76ce1ec03`.

#### Return entry delay across full and tail rounds

Follow the native last-notifying owner in both captures for each of 520 rounds. Compare scatter body and the preceding entry interval across 480 full 80-row rounds and 40 tail rounds with 64 active rows.

| Arm | Full W2 end to scatter start P50 (µs) | Tail W2 end to scatter start P50 (µs) | Full scatter body P50 (µs) | Tail scatter body P50 (µs) |
| --- | ---: | ---: | ---: | ---: |
| Native | 43.718 | 45.934 | 35.531 | 24.128 |
| Prefill 56 | 41.311 | 42.075 | 33.917 | 24.537 |

Prefill 56 accumulates 21.822 ms before scatter and 11.152 ms between scatter end and recorded FIN. The 26 longest intervals account for 7.13% and 7.39% of those totals. Pearson correlation between the W2-to-scatter interval and scatter body is -0.113, compared with -0.077 native. Shorter scatter work still incurs comparable entry delay in this capture. Narrowing work inside a launch and eliminating a MoonEP handoff therefore address different parts of the span. The report `diag59r-20261122T200956Z-return-entry-tail-002232.json` hashes to `ecc2ef4e7db7bdd3097b65ae53794d9e477b152ec5cca2bc5472ef5426611d5a`.

The same-owner scheduler split accounts for the full W2-to-scatter gap. Prefill 56 spends 16.577 ms from W2 AICore end to recorded W2 FIN, 4.209 ms from FIN to scatter dispatch, and 1.037 ms from dispatch to first execution. Native spends 17.951, 4.881 and 1.050 ms respectively. All intervals are nonnegative in both arms. The W2 end-to-FIN interval contributes 76% of the compact entry delay and 75% of native. This shared completion interval falls outside the measured cost of the MoonEP scatter itself. The report `diag59r-20261122T200956Z-w2-scatter-handoff-002550.json` hashes to `3dac10fa6f6e7e80205065bf93ba4ae4e3fd6ba3b2cedb451e0302de0716894a`.

A follow-up replay retains the measured W2 end-to-FIN interval on the existing W2-to-scatter dependency at all 2,080 sites. The original 614.150 ms span is reproduced before and after adding these constraints. With their delays retained, zeroing transport bodies and entry gaps leaves 506.653 ms, versus 490.128 ms in the earlier replay. Including the same directly attributed successor gaps leaves 485.937 ms, versus 469.402 ms. This corresponds to a 20.64% reduction from the 612.304 ms native device span and leaves 57.324 ms to the 30%-lower device threshold. The fixed-order counterfactual uses one 1,024-token trace. Its result depends on retaining the other measured durations and scheduling gaps. The report `diag59r-20261122T200956Z-transport-zero-retain-w2-fin-003009.json` hashes to `cbd68cb308bc05ef51ac4e665f88d87723f477b6f7897cec509ed8dd3507bfec`.

## Prefill 68 · Move the staged activation prefix in 16-row tiles

Start from Prefill 58 and use the same receiver expansion as Prefill 66. In each single-source round, the existing push task gives the remote staging prefix to the workers covering its 16-row tiles. Each of the three peers receives one contiguous copy of the valid activation prefix. The source's own expert activations still go directly to their expert rows. The per-route loop sends the existing scale and descriptor packets to every owner and retains the same mixed-source path. The receiver uses the existing pack task to materialize the unchanged expert input rows.

Across the 1,560 captured single-source layer-rounds, the model gives 23,040 issued remote activation tile stores, compared with 558,268 per-route stores in Prefill 58 and 304,282 per-token-per-owner stores in Prefill 66. The 16-row tiles reduce issued remote activation stores by 95.87% from Prefill 58. They carry 1,886,208,000 activation bytes, 34.01% below Prefill 58 and 21.07% above Prefill 66. The receiver copy count remains 558,268 rows. The model counts tile store instructions and logical bytes. The NPU can split the tile store into smaller memory transactions. Counts and layout are in `activation-tile-count.json`, SHA-256 `dd02702425ab5837388f0e7acd063c03fb0f903be58394cff298a6d428fa0bf6`.

Both compact slabs, the native slab, the 79-row tail and full native and compact Serving passed host compilation. Only `moonep_push` and `moonep_pack_scale` differ from Prefill 58. The receive pack kernel matches Prefill 66. The generated store supports a valid row count below 16 on the tail. Native and routed expert kernels, 230 block assignments, 289 submit sites and all communication buffer specifications match. MoE stays at 80 rows with one serially reused slot.

The server verified archive SHA-256 `c658a590b2c0c2b0a7a4a3052fd48df36dea6405afb191f7f35d6e69bac64de3` and the source manifests. Suite `p68-20261122T210006Z` waits for Prefill 66 to finish and uses Prefill 58 and native as its controls. Device precision and latency remain pending.

## Prefill 69 · Stage token metadata with the compact route map

Extend Prefill 68's single-source staging. Each remote owner receives the 16-row activation tiles plus token replicas of the exact FP32 routing weights and MX scales. The source histogram orders the route IDs by the already assigned compact return row, and each owner receives a 480-word route map. A second row-plan field gives each expert's starting route ID index. After the original payload notification, the receive pack expands token packets into the original aligned activation, MX scale and weight rows. The source's local experts and mixed-source rounds use the original per-route packets.

The route map occupies the first 10 rows of the existing remote descriptor window. Token packets occupy its existing tail at row 4,816. The owner plan uses 192 of the 384 existing histogram columns and the source map adds 1,920 bytes of local scratch. Expert work, 80-row rounds, communication windows, one serial slot, publish and wait tasks, and round release remain the same.

On the 1,560 archived layer-rounds, Prefill 68 issues 558,268 remote per-route descriptor tile stores. The candidate issues an estimated 23,040 token-packet tile stores and 4,680 route-map tile stores, a 95.03% reduction. Logical descriptor payload decreases from 107,187,456 to 79,718,400 bytes, or 25.63%, plus 1,797,120 bytes of additional owner-plan fields. Activation staging and return payloads follow Prefill 68. The candidate shifts some work to the receiver, which now reads staged scales and weights per valid expert row. `metadata-store-count.json` has SHA-256 `f54abec717edd7122ed703907466dcfce36a0b1874182a7f3c886edf5b5bd532`.

The single-source, mixed-source, 79-row tail and native slabs passed host compilation, as did full native and compact Prefill and Decode. The source comparison changes one MoonEP module. Only `moonep_histogram`, `moonep_push` and `moonep_pack_scale` change in the compact device kernels. The 109 native Prefill kernels, both 45-kernel Decode builds, routed expert kernels, 230 block assignments and 289 submit sites match Prefill 68. Generated code writes 16-token metadata tiles and a 10-row route map within the existing windows. Device precision and warm Serving latency are pending.

The server verified archive SHA-256 `84a3095b7007ecb91e3872c4dfc7a10c664a188cd961bf6c53900e27e4644f56` and both source manifests. Suite `p69-20261122T212137Z` waits for Prefill 68 to exit successfully. It runs the single-source, mixed-source and 79-row tail replays, then candidate, Prefill 68 and native Serving on the same device lock.

## Prefill 70 · Fuse remote activation tiles and metadata into one payload write

Start from Prefill 69's compact single-source plan. Each staging tile starts with a 5,120-byte header row and holds up to 16 following activation rows. The header contains the corresponding token scale and exact FP32 weight bytes. The first header also contains the 480-word route map in compact return order. The header and actual activation rows share one remote tile write. The receiver reads the same route IDs and operands from this layout and fills the original expert rows. The push and pack paths for local experts and mixed-source rounds are unchanged at runtime.

The five frames fit into rows 4,811–4,895 of the existing 4,896-row activation window. A tail frame has one header plus its actual activation rows. The activation window size, payload and return readiness dependencies, single communication slot, 80-row round width and expert launch shapes match Prefill 69.

The model for 1,560 archived layer-rounds counts 23,040 remote forward tile writes, down from 50,760 in Prefill 69 (54.61%). Forward payload rises from 1,965,926,400 to 2,004,172,800 bytes (1.95%) because every header occupies one full row. The sender also assembles the activation tile into its contiguous frame in UB. The NPU timing will determine how that copy and padding trade off against fewer remote writes. `frame-store-count.json` has SHA-256 `4a7472e6056bcfd7c3395b3e3b6818571559602e2d9fee09e4bb809eeccde53b`.

Host compilation passed the single-source, mixed-source, 79-row tail and native slabs and both full Serving variants. Only `moonep_push` and `moonep_pack_scale` differ in compact kernel bodies. Native Prefill and both Decode builds, routed expert kernels, 230 block assignments, 289 submit sites and all communication buffer specifications match Prefill 69. The compiled remote store has a 17-row physical tile and a dynamic valid row count. Device precision and warm Serving latency are pending.

The server verified archive SHA-256 `79aecb964593796a5b29d95f9ce97545e664ba676299234d9a41622421a3a6cb` and both source manifests. Suite `p70-20261122T213004Z` waits for a successful Prefill 69. It runs three frozen replays, then candidate, Prefill 69 and native Serving with the same device lock.

## Prefill 71 · Pack source-owned expert rows directly from the compact route map

Extend Prefill 70's dense route map to the source's own experts. The source histogram keeps the same per-owner counts, aligned starts and return lookup. It skips writing the single-source sparse route positions and keeping a second local plan for push. The three remote owners receive the same fused frames as Prefill 70. In the existing source pack task, the compact map indexes the original activation, MX scale and FP32 routing weight for each expert row. The source writes the expert input rows directly and skips its per-route descriptor transfers. The mixed-source payload path retains its original representation and 56 push blocks.

The sole-source push uses 15 blocks for three peers and up to five frames per peer. Across the 1,560 archived layer-rounds, this reduces source push blocks from 87,360 to 23,400 and removes 178,532 scalar position writes and the same number of local descriptor stores, equivalent to 34,278,144 logical descriptor bytes. The 178,532 local activation stores move into the existing pack task. Cross-rank payload and return traffic follow Prefill 70. `source-local-count.json` hashes to `0fc95cbe8f226ebe720d7dca1b6bdc18d2c56f5ea0e4ce2e9712baad20bfba84`.

All four host slabs and full native and compact Serving compile. Compact kernel differences are confined to `moonep_histogram`, `moonep_push` and `moonep_pack_scale`. Native Prefill, both Decode builds, expert bodies and launch shapes, communication buffer specifications, the 80-row width and serial slot reuse match Prefill 70. The compiler emits one additional dependency-only submit site in the zero-block guard, which is unreachable for both generated push counts, 15 and 56. The 289 reachable submit sites and the other 229 block assignments match the control. The changes in work placement and input reads require the matched device replays and Serving comparison.

The server verified archive SHA-256 `216143c6c985c169c06ad82fd3f78c8552a95d2ddd1db6f78b0904cda682409c` and both source manifests. Suite `p71-20261122T214039Z` waits for Prefill 70 to succeed, then runs the three frozen replays and the candidate, Prefill 70 and native Serving arms under the original device lock.

## Prefill 72 · Unpack the MoonEP frame scales in the source push

Extend Prefill 71. Each single-source push block already owns one 16-token frame. It loads the corresponding packed MX scale tile from the gate output and transposes its pairs while assembling the existing packet. The first peer's workers also store the unpacked local scale rows consumed by the source pack after payload completion. The separate `dispatch_scale_unpack` stage is skipped on all ranks in a single-source MoonEP round. Mixed-source and native still use the original stage.

On 1,560 archived layer-rounds, the model removes 6,240 standalone scale-unpack tasks and 199,680 scheduled blocks across the four ranks. The source transposes the 7,680 active scale tiles once for each of the three receiving peers, for 23,040 transposes. The packets, their valid tail rows, the order of experts and the local scale and weight values follow Prefill 71. `scale-stage-count.json` has SHA-256 `68645aea8e6ab22b490e63056ed259059a79e027c5c87d8d390746b54cfacc64`.

The four slab builds and both full Serving builds compiled. After normalizing generated operand names, only `moonep_push` changes in compact kernel code. The audit confirms the same native kernels and launch sites, both Decode builds, experts, collective buffers, 80-row width and serial single slot. The compact graph has the same 230 block assignments and 290 static submit sites. Only the standalone scale-unpack site becomes conditional on mixed input sources. Hardware correctness and warm latency are pending.

The Prefill 56 warm trace provides a check on the timing assumption behind fusion. Each of the four ranks records 520 standalone unpack tasks and 16,640 block instances. Unpack spans total 5.149 ms on the source and 3.262–3.322 ms per receiver. In all 520 source pushes, the latest FIN among the other traced local prerequisites is later than the scale-unpack FIN. The script `experiments/diag-59/scale-unpack-cost.py` has SHA-256 `621f08facf9b99b661e7b58af803841acaa2dc12c1e501595e693bee0930fbce`. Its report `diag59r-20261122T200956Z-candidate-measured-scale-unpack.json` hashes to `1f3ad4bc0f4d980f04ce3a84da7ee021bef70342ad45a7fd21f6225a7ff591d5`.

The server verified archive SHA-256 `b35d899595b2bbbc5f882c17a76be63ff923422b66956d839122e10632cc6999` and both source manifests. Keep Prefill 72 staged while testing whether the earlier source preparation task can build each header once for all three peers. Integration into push would move three copies of the transpose onto the currently longer branch.

## Prefill 73 · Prepare each compact frame header once before push

Use Prefill 71 as the matched control. The warm source prerequisites favor preparing token metadata before the push. Five source workers transpose the gate's packed scales and assemble the exact FP32 weights into a 3,072-byte header prefix per 16-token tile. The same workers write the unpacked scales for local expert input packing. The existing push reads each prepared prefix for all three peers, adds the route map to the first frame and sends the original valid activation rows. Single-source receivers skip the standalone scale task. Mixed-source and native keep the existing scale stage.

For the 1,560 captured layer-rounds, metadata assembly falls from 23,040 to 7,680 instances. The separate preparation work drops from four 32-block scale tasks per layer-round to one five-block MoonEP task. This removes 4,680 tasks and 191,880 scheduled block instances across four ranks in the count model. The header prefix scratch is 15,360 bytes per compact dispatch and the model adds 23,592,960 bytes of source-local header stores across the captures. The remote frame and return formats are identical to Prefill 71. `frame-prep-count.json` has SHA-256 `26bd799d2307811791e014a0ec71afee876c7f7601500441516a482a57555969`.

The four host slabs and both full Serving builds compile. The compact variant adds `moonep_prepare_frame` and changes `moonep_push`. The audit matches every other kernel, the native launch sites, both Decode builds and the original expert launch shapes. Removing the one new frame-prep site leaves the original 290 static submit sites and 230 block assignments, after accounting for generated IDs. Communication buffer specifications, 80-row width and single-slot serial reuse match. The source push waits for the existing compact plan and the prepared metadata. Hardware correctness and latency are pending.

The fixed and online host slabs also compile. The server verified archive SHA-256 `56fe12de194d92a973fbb03b378b171b147726ec53af840eff9a753b7250f745` and both source manifests. Suite `p73-20261122T215201Z` waits for successful Prefill 71, then uses the existing card lock for three frozen replays and candidate, Prefill 71 and native Serving arms.

## Prefill 74 · Signal each source frame directly from the push task

The measured Prefill 56 source records 520 payload publications. Push end to publication start totals 4.100 ms, with median 5.883 µs. Within this interval, recorded scheduler FIN to publication dispatch totals 1.895 ms, with median 3.274 µs. The publication bodies add 2.320 ms over the request. These spans describe a local 1,024-token trace. The script `experiments/diag-59/payload-notify-gap.py` has SHA-256 `802a720329de026b001c6e112adbff71c842e23c1d7862fca083781bb1e7cae4`. The report `diag59r-20261122T200956Z-candidate-measured-notify-gap.json` hashes to `1050ffeeb14128fc1184d06414b56ae0eaaad1127fcbf9b7d3482a686f4df913`.

Use Prefill 73 as the control. In a compact single-source round, each push block writes its valid frame, orders the write with the generated GM release and sets the current epoch in that peer's frame flag. Empty frame blocks still publish their epoch. The receiver registers the five conditions for its own frames in the existing wait task. The source's local pack depends directly on push completion. Mixed rounds retain the original separate publisher and payload predicate.

The frame flags occupy five elements of the existing `data_arrived` window. The existing end-of-entry clear task clears the additional flags for compact builds. One serial communication slot, the per-round release protocol, tail policy and packet format remain the same. The installed A5 host-build-graph runtime supports up to 64 completions per task. The generated code shows the frame TSTORE, cache invalidate and GM release fence before TNOTIFY.

Across 1,560 archived layer-rounds, the model removes 1,560 standalone publish tasks and raises the total payload signals from 4,680 to 23,400. The number of receiver wait tasks stays at 4,680. There are 12 additional signal clears per rank per entry exit. `frame-notify-count.json` hashes to `5539b53e14384380f5f0b9a340f7dee4f85d9376d372004e62c86d55c5029133`. The device comparison will measure the trade-off between the shorter dependency chain and the additional signals.

All four slabs and both full Serving variants compile. Only `moonep_push`, `moonep_wait` and the compact `moe_signal_clear` change in the compact device bodies. All other kernel bodies, native launch sites, both Decode builds, the 80-row width, communication-buffer specifications and all 231 block assignments match. The 291 static submit sites likewise match, with the separate payload publisher now conditional on the original mixed-source path. Hardware precision, repeated-entry behavior and warm latency require device validation.

Fixed and online host slabs also compile. The server verified archive SHA-256 `f420ea2004d778c091268e0ecd0c45bd1d6671f22c36a753578cd92ad1bd9068` and both source manifests. Suite `p74-20261122T220157Z` waits for successful Prefill 73, then uses the existing card lock for all three frozen replays and the candidate, Prefill 73 and native Serving comparisons.

### Contiguous receives in the staged activation path

Archive the count of consecutive source tokens targeting each expert in the same three routing captures. Runs stop at each 16-token source frame and each 32-row expert pack group. This preserves the current expert row order, staging windows and scales. The scan covers 1,560 layer-rounds. On remote owners, 558,268 per-row activation copies can be expressed as 417,990 contiguous runs, a 25.13% reduction in copy operations with the same byte count. Of those runs, 334,043 (79.92%) are single rows. The source's 178,532 own rows form 135,200 runs. These are route counts. Tile setup and run detection take device time. The report `experiments/diag-59/activation-copy-runs.json` has SHA-256 `deef109ea1ccda7cce92d911f964f145c0713d96608f67fa3bb0a6478414f86b`. The script hash is `b4a2942aff97dc03510b91602e956c0a7c17329b916d31bfe17ab84d09e40ac8`.

## Prefill 75 · Copy up to four adjacent staged activation rows at a time

Use Prefill 73 as the control so this experiment isolates receive-side copying. The existing scale pack tracks a run of consecutive physical input rows within each expert pack group. It writes up to four rows with one dynamically sized tile load and store. A frame header creates a gap in remote staging addresses and ends a run automatically. Source-owned experts use the same run logic on their local inputs. Expert row order, per-route scales, FP32 weights and remote frame payloads follow Prefill 73.

The four-row cap keeps the activation tile at 20 KiB per copy. The 1,560 archived layer-rounds give 424,918 remote-owner copies instead of 558,268, a reduction of 23.89% at the same byte count. The source has at most 137,151 own copies under the capped model, compared with 178,532 per-row copies. The local count is an upper bound because local activations can also cross a token frame boundary. The shorter tile retains most of the savings of a 16-row copy. Run tracking adds scalar work. `four-row-count.json` has SHA-256 `8526174a34538abb74569398e28851fc419288bba414ac82b3d12e1ee30c8fc3`.

The compact single-source, mixed-source and 79-row tail slabs, native slab and both full Serving builds compile. Compact generated code changes only `moonep_pack_scale` and reads and writes the exact valid run length. All expert kernels, native generated code, both Decode builds, 231 block assignments, 291 submit sites, buffer specifications, 80-row width and one serial communication slot match the control. Correctness and latency require the matched device tests.

The server verified archive SHA-256 `d445a919aa2eb3ac8cd1e01cae1efc4f1fba736e5d0efbc44345df1d0456215b` and both source manifests. Suite `p75-20261122T220751Z` waits for Prefill 74 to finish and requires the Prefill 73 control to pass. It uses the existing device lock for the three frozen replays and candidate, Prefill 73 and native Serving comparisons.

### Peer coverage within staged frames

The same 1,560 layer-rounds contain 23,040 remote frames. Each receiving owner needs at least one row in all but four frames. The existing full-prefix transfer moves 368,400 activation rows. Sending the interval from the first needed token to the last needed token in each frame still moves 355,475 rows, a 3.51% activation-byte reduction. A fully sparse encoding could reach 304,282 rows, a 17.40% activation-byte reduction, and would require packing the holes and reindexing them on the owner. Keep the current source frame ranges in the next device comparison. The simple prefix and suffix trim has a smaller measured opportunity than the contiguous receive copy model. The `activation-frame-cover.py` script and `activation-frame-cover.json` report are in `experiments/diag-59`, with SHA-256 `20b4f612317f4455a2574073cdee847157d736e5e7236344203576623a6b9288` and `fffb9fe5d95f809ce87b0b5e5bb355376240798e3040f621dfa4c3b8c9c30501`.

## Prefill 76 · Use the compact pack completion directly for W13

Use the device-validated Prefill 58 to isolate the handoff change on a verified wire format. In EP4 compact 80-row rounds, the pack already depends on payload readiness and the row plan. This layout has no replica copy. Pass the original pack completion to W13, removing the dependency-only task that joined pack completion with the same plan. The change applies to the EP4 compact sender-plan layout.

Source and receiver payload work, expert input buffers and the later completion and release tasks follow Prefill 58. The compact code removes one runtime submission per rank and round, or 2,080 submissions in a four-rank capture with 520 rounds. The per-phase warm trace identified the pack-to-W13 gap. Its device duration includes all scheduler work and contention in the interval.

All four host slabs and both Serving builds compile. The 109 compact kernels, 109 native Prefill kernels and both sets of 45 Decode kernels match Prefill 58 canonically. The only compact submit-site change removes the dummy task between packing and W13. W13's arguments, launch dimensions, the pack dependencies, all 230 block assignments, the 80-row width and the single-slot communication buffers match. Hardware precision and latency are pending.

The fixed and online host slabs also compile. The server verified archive SHA-256 `1e1635108e1e9c39beb76d4bc7457d6b462e816f69388c0578b85acdcd52c077` and both source manifests. Suite `p76-20261122T222532Z` waits for Prefill 62 to release the original device lock. Its three frozen replays and candidate, Prefill 58 and native Serving arms use the same fixed configuration.

Prefill 76 acquired the lock after Prefill 64 and started the single-source replay at 23:25:32 UTC. The replay exited 0 and the mixed-source replay started at 23:28:09 UTC.

The mixed-source and 79-row tail replays also exited 0. Candidate Serving started at 23:33:10 UTC and reached health in 306.094 s.

The candidate Serving arm exited 0. It produced 13 output tokens at EOS on all 12 warm requests and achieved Prefill P50 of 2.162488 s. The medians for the three prompt lengths are 0.718694, 2.162488 and 4.341085 s. The Prefill 58 control started at 23:42:35 UTC. Its normalized command matches the candidate. The candidate `http.json` hashes to `7a5db232b3ef4206016c00b72a7cdce8b23d57c71399a05631ea6fffe2ecf16e`.

The Prefill 58 control exited 0 with Prefill P50 of 2.146937 s and P95 of 4.458181 s. The candidate P50 is 0.72% higher and its P95 of 4.354323 s is 2.33% lower. Seven of the 12 candidate requests are faster than their same-order compact counterpart. The native arm started at 23:51:47 UTC. Two-arm report `p76-20261122T222532Z-partial-comparison.json` hashes to `45e206de4045d115d30fa80febbe1adf8ee78b363f3141f858e7a6d047dccb7d`.

### Prefill 76 device result

The native arm and suite exited 0 at 00:01:10 UTC on November 23. All three normalized commands match. Each arm produced 13 output tokens with the same stop reason on each of the 12 ordered requests. The long-answer and both concurrency checks passed in all arms.

| Arm | Warm Prefill P50 (s) | Warm Prefill P95 (s) | 1,024 P50 (s) | 3,070 P50 (s) | 6,142 P50 (s) |
|---|---:|---:|---:|---:|---:|
| Native | 2.139514 | 4.310530 | 0.711072 | 2.139514 | 4.291082 |
| Prefill 58 control | 2.146937 | 4.458181 | 0.720016 | 2.146937 | 4.327734 |
| Prefill 76 candidate | 2.162488 | 4.354323 | 0.718694 | 2.162488 | 4.341085 |

The candidate P50 and P95 increase by 1.07% and 1.02% versus native. No candidate request improves over native, and two Prefill 58 requests improve. Keep the Prefill 58 handoff for the device comparison. The full `p76-20261122T222532Z-comparison.json` hashes to `62a4d218f30a0eafd96c057538445e211b0a0b51caef8eeb5dfd63d8efd4144e`. Prefill 65 acquired the four-card lock and started its first frozen replay at 00:01:13 UTC.

## Prefill 77 · Index active experts in the compact receive pack

Starting from Prefill 73, encode the receiver's active expert list in the spare bits of the existing per-owner route offsets. The low nine bits retain the absolute route start. The next seven bits hold the corresponding entry of the active list, with the length recorded in the high bits of word zero. The owner still installs the full expert count and start vectors, then uses 16 receive pack blocks to process the active list. Mixed-source packing uses the original 32-block stride. The route map, expert row order, frame transfers and return lanes follow Prefill 73.

On the 6,240 owner-rounds from the three real-model routing captures, the idle block median drops from 21 to 3. The median route count of the busiest block drops from 37 to 35, the P95 remains at 67, and the maximum drops from 134 to 93. The model uses one 32-row group per pack tile. The worst groups per block fall from six to four. This halves the scheduled pack blocks in these single-source rounds and specifically removes 33,280 block instances in a four-rank 520-round request. Source list construction adds work. The owner-plan write stays 192 words. The `experiments/diag-59/owner-pack-blocks.py` script and `owner-pack-blocks.json` report have SHA-256 `658bdb31951495d99c88022e16873ce035f09ad1a9247108e8381d33e7cadb45` and `fcce10beb805f9d18970e14b0c362de665f4b5ec094240ed65d551ab98deae77`.

The four slabs and both full Serving variants compile. The only changed compact kernel bodies are `moonep_histogram` and `moonep_pack_scale`. The generated 16- and 32-block cases preserve the 291 reachable submit sites, with a dependency-only fallback guarded by the unreachable zero-block branch. Native and Decode kernels, expert launch sites and block assignments, the 80-row width, communication buffer specifications and serial single-slot reuse match Prefill 73.

The frozen routing biases exercise at most six active experts per owner. The added 79-token fixture uses the frozen 79-row inputs and weights and sets the source gate bias to select 20 nonconsecutive experts on rank 1. The existing full slab reference generated the expected outputs. Its gate reference confirmed 474 routes over 20 experts, including the last local expert ID. This exercises the second pass of four pack blocks and three owners with zero routes. The generator `make-spill-golden.py` and `golden-spill-coverage.json` have SHA-256 `7682e89bfbf5b67132aad1d33c28d480bf876ce8ba198dd66e588750a31292c2` and `0093d779c1f72f44e1d75d07184296b9536f00b14f0b11fff52786bbc1916edd`.

The server verified the source manifest and archive SHA-256 `7473e3fa1fa42f8e6941968a0612756913ecd2e460b54467d9382e5a0ab3f0c7`. Suite `p77-20261122T223723Z` waits for Prefill 75 to finish and requires successful Prefill 73 and reference generation. It uses the same four-card lock. All four device replays must pass before the candidate, Prefill 73 and native Serving arms.

## Prefill 78 · Sort the compact source route map in unified buffer

Starting from Prefill 73, build the existing compact source route maps from a stable expert sort. The vector key combines the expert ID and padded input lane. All valid keys are exact FP32 integers. Inactive rows and the two padding lanes per token sort behind all valid routes. One UB scan finds expert run lengths and writes the forward route map and its inverse at the same dense positions used by Prefill 73. The per-owner counts, aligned expert offsets, 192-word plan and both route maps retain their existing meanings and shapes.

Over the 736,800 routes in 1,560 archived single-source rounds, this replaces 1,473,600 scalar route-loop iterations with 736,800. The source also skips the second zeroing of 384 cursors and the local 384-entry return-start table each round. Each round adds a vector sort of a 1,024-element tile, three merges and padding. These are operation counts. Real latency depends on the sort, scalar reads and handoff costs on device.

The single, mixed, native and 79-row host slabs, fixed and online variants, and full native and compact Serving compile at 80 rows. Only `moonep_histogram` changes in the compact device kernels. The generated native and Decode kernels are identical, and the compact orchestration preserves all 291 submit sites, 231 block assignments and communication buffers. The original mixed-source route scan is retained. The same frozen replays, including the 20-expert remote owner from Prefill 77, will check the exact dispatch and return behavior on device.

The server verified both source manifests and archive SHA-256 `b229a7fb1ab0daa549aaadf486d3ee1c14fad6ed9681b1c5a70b93e3d3a29145`. Suite `p78-20261122T225936Z` waits for Prefill 77 to finish and for a successful Prefill 73 control. It uses the original four-card lock and runs the four replays before native, Prefill 73 and candidate Serving in that order.

## Prefill 79 · Return eight compact rows per store with 32 scatter blocks

Use the device-validated Prefill 62 as the control. Its compact single-source owner plan assigns each rank a contiguous BF16 return lane. Return up to eight adjacent rows in each transfer and use 32 return scatter blocks. The stride stays 256 result rows, so a large owner still has the same second pass threshold. Mixed sources use the original 64-block per-route scatter. Expert submissions and launch counts, the 80-row MoE width, communication buffers, serial slot reuse, notifications and TopK accumulation order follow the control.

Over the 1,560 single-source rounds in the three real route captures, the 64-block four-row scatter has 35 idle blocks at the median. The heaviest owner needs 60 four-row stores at P95 and 68 at the maximum. With eight rows and 32 blocks, remote stores fall from 141,309 to 71,829, and scheduled blocks from 399,360 to 199,680. Both cases transmit 558,268 BF16 rows remotely and have 18 rounds with a second store on the heaviest block. The 16-row, 16-block projection gives 37,095 remote writes and also retains that threshold. It uses a 160 KiB tile per active block, compared with 80 KiB in Prefill 79. Its single-source host slab compiles.

The model under `experiments/diag-59` includes `return-scatter-width.py` and `return-scatter-tiles.py`, with script SHA-256 `98f2312bc221f39ea03fb399dfa220de1fc20f7cbb826530d2cf5d0c14b16079` and `e9332d0480bc5e8d68f28172aedf340aea9f37077bf70a342b6f41664a65b5e8`. The accompanying reports hash to `1604f148ac1a67b4c98bc71a9a4116d53da640ad76748d5b27609ec13a553f00` and `546b72b11945780ee21bd488151f2ad36ec48b3b074d64b9813f84a390b56a48`.

The six host slab builds and both full Serving layouts compile. The only changed compact kernel body is `moonep_scatter`. Native and Decode code, all 290 existing compact submit sites, 229 other block assignments and the communication buffers match. The additional emitted submit site is guarded by the zero-block branch, while scatter always uses 32 or 64 blocks. Device validation will include the 79-token, 20-expert remote owner with 474 return rows. That owner requires an additional store on some blocks and a short final tile.

The server verified both source manifests and archive SHA-256 `5eb78c6edb64fbcc3c9a4464495cabab72f2ef4f100985d31c5654b0a68baeb7`. Suite `p79-20261122T230536Z` waits for Prefill 76 to finish and requires the successful Prefill 62 control. It uses the four-card lock to run four replays before candidate, Prefill 62 and native Serving in that order.

The spill, single-source, mixed-source and 79-row tail replays all exited 0 on the four cards. Candidate Serving started at 00:46:25 UTC on November 23 and reached health in 298.119 s.

The candidate exited 0 after 12 warm requests, the long answer and both concurrency waves. Warm Prefill P50 and P95 are 2.109012 and 4.301823 s. The 1,024, 3,070 and 6,142 token P50 values are 0.700420, 2.109012 and 4.236238 s. All 12 requests emitted 13 tokens with the expected stop reason. The candidate `http.json` has SHA-256 `5c3e3e01ea70899c09e006704f24eea6a200c905434f083eb439e68c65eef565`. The Prefill 62 control started at 00:55:42 UTC and matches the candidate's normalized launch command. The native arm and full comparison are pending.

## Prefill 80 · Return sixteen compact rows per store with 16 scatter blocks

Use the same Prefill 62 control and preserve the 256-result-row stride per scatter pass. The single-source transport now writes up to 16 contiguous BF16 rows per tile with 16 blocks. Mixed-source scatter remains per route with 64 blocks. The known real routes project 37,095 remote stores rather than 141,309, and 99,840 scatter blocks rather than 399,360. All 558,268 remote rows are carried with their original BF16 representation. The number of rounds that require a second store at the busiest owner is 18 in both cases. Larger per-block tiles and lower concurrent work can change transfer latency.

Six slabs and both full Serving builds compile. Only the compact device kernel `moonep_scatter` changes. Native and Decode code, the existing 290 compact task submission sites, communication window sizes and all expert launches match. As in Prefill 79, the emitted additional site is a fallback guarded by the zero-block check. The candidate sets 16 or 64 blocks. The heavy-owner numerical replay covers repeated stores and a partial 16-row tail.

The server verified both source manifests and archive SHA-256 `45a76ce0e0336dbe1e7176c4f89af771bec727dfd2bda379b437129934bda315`. Suite `p80-20261122T231149Z` waits for Prefill 79 to finish and requires the verified Prefill 62 control. It executes four replays under the same four-card lock, followed by native, Prefill 62 and candidate Serving in that order.

## Prefill 81 · Publish compact return blocks directly from scatter

Extend Prefill 80's sixteen-block contiguous return scatter. Once a remote block finishes all of its BF16 stores, it publishes the current epoch in its own padded counter on the source. The source waits for those 48 peer counters and its local scatter before the existing TopK-order reduction. Each peer joins its local scatter and output initialization before its existing round release. The release and reuse kernel bodies, epoch progression, communication slot and shared window allocation follow Prefill 80. Mixed and native retain their return protocol.

The 48 conditions fit the installed A5 host-build-graph capacity of 64. A 64-block scatter with the same per-block protocol would require 192 conditions. In a compact single-source round, this replaces four return notification tasks, three peer waits and three empty reduction grids with three dependency-only joins. It sends 48 remote counter writes instead of 12. In the 520-round trace workload this trades 18,720 additional counter writes for 3,640 fewer submissions and 49,920 fewer empty reduction blocks. The physical effect of these changes depends on the device runtime.

The six host slab builds and both full Serving layouts compile. Only `moonep_scatter`, `combine_wait` and `moe_signal_clear` change in the compact device code. Native, Decode, expert kernels and launch sizes, all existing block assignments and communication buffer specifications match Prefill 80. The generated scatter orders the data store, cache maintenance, pipeline barrier and DDR fence before notification. The only added submit site is the compact peer join. The existing release still gates the following round. The remote 474-row replay covers a block with multiple BF16 stores before publishing.

The server verified both source manifests and archive SHA-256 `86a9b9e6ebaae5d0a936a6d4c6538d1a2cce63bf844a5d1effb883e0b18ad773`. Suite `p81-20261122T231831Z` waits for successful Prefill 80, uses the original device lock and checks all four numerical fixtures before candidate, Prefill 80 and native Serving.

## Prefill 82 · Register only the compact return flags that cover each owner

Prefill 81 publishes 16 flags for every remote owner. Compute each owner's required flags from its existing return row count and put four five-bit counts into the unused first-row padding word of the source's return lookup. A nonempty owner publishes once per block that stores rows. An empty owner acknowledges through block zero. The existing release still waits for each rank. The return addresses, BF16 bytes, payload tiles, sixteen-block scatter and original TopK reduction stay the same.

The device waiter reads that local header once before registration. Its statically bounded loop adds conditions only for the specified blocks. The compiler requires a static trip bound and disallows further reads once registration begins. This keeps the maximum at 48 peer conditions and avoids adding a control task or sending an extra plan word over the fabric.

Across 1,560 captured single-source rounds, peer notifications and wait conditions each drop from 74,880 to 37,077, a modeled 50.48% reduction from Prefill 81. The median per round is 24, P95 is 29 and max is 31. These captures have no empty remote owner. The concentrated 474-row replay has two empty remote owners and tests both the acknowledgement and the capped sixteen-block count. The script `experiments/diag-59/return-scatter-active-flags.py` and its JSON report hash to `066143de3f039e9b66d1a4feaa892ea909a4183a725d965e31cf5756b8cb7485` and `8e515779d68af407007d261ddcf02d1fa9cbbbdce025b45f061c10381a5ef8cf`.

Six slabs and both fresh full Serving builds pass host compilation. The compact device changes are confined to `moonep_histogram`, `moonep_scatter` and `combine_wait`. Native, Decode, the expert kernels, communication buffers, task submissions and block assignments match Prefill 81. The generated waiter uses a single local load before the registration loop, and the scatter still orders the output stores before notification.

The server verified both source manifests and archive SHA-256 `227732e5e6892d5742f27c0430d3e91c3cee8d6f0a085a2b6d6d3abb64342a77`. Suite `p82-20261122T232649Z` waits for successful Prefill 81 and uses the original device lock. It runs the four numerical replays and then Serving in native, Prefill 81 and candidate order.

## Matched Serving control audit

Re-read the raw requests and normalized launch commands for Prefills 47, 48, 49, 51, 54, 55, 56, 57, 58, 60, 61, 62 and 64. All 13 suites exited 0. Every matched arm has the same ordered 12 requests, input lengths 1,024, 3,070 and 6,142, and 13 output tokens ending at EOS. The per-experiment checks above establish the 80-row, single-slot settings.

Native P50 across the recorded sessions ranges from 2.084674 to 3.749463 s. Prefills 49 and 57 measured native-relative reductions above 30% in the first respective ordering, while their candidates were 1.04% and 0.53% slower than their same-run MoonEP controls. Repeated Prefill 54 reversed its first-run reduction. Prefill 62 improved all 12 pairs versus native both in its own suite and as the control of Prefill 64, with native-relative P50 reductions of 2.22% and 1.83%. Its own candidate also improved all 12 pairs versus its Prefill 61 control. The follow-up uses Prefill 62 as the base for further MoonEP changes and as the compact arm for the updated length sweep.

`experiments/diag-59/serving-control-audit.py` and `serving-control-audit.json` hash to `7b7b2de7fac88b9b7430a553bb705b9e4076910787d9da0a813cad709b062b70` and `3e03a1136c2dc123a8ee1bb29a94be4ebaca9db4b9afe2282f4caf668e5f54bf`.

## Prefill 83 · Balance direct operand pushes by 16-row work group

Start from device-verified Prefill 62. The sender histogram creates a compact list of the existing expert and 16-row group pairs. The original 56 push blocks stride over that list. Each block reads the group's positions and writes the same FP8 activations, packed MX scale and FP32 weights to the same destination rows. Mixed sources retain the original expert-ID stride. The list occupies one additional 488-word rank-local tensor. The remote row plans, peer flags, round release and communication windows match Prefill 62.

The 1,560 real single-source rounds have 736,800 routes in 99,766 groups. The peak routes per block median, P95 and max change from 63, 101 and 122 to 23, 32 and 40. Peak groups per block P95 falls from seven to two. The median number of idle blocks falls from 22 to zero. Putting full groups first gives P95 of 31 and a higher median of 25. The candidate uses the simpler expert-ordered list with no extra sorting. These are work counts. Parallel stores and source list preparation still need device timing.

The local tool `experiments/diag-59/direct-push-group-balance.py` and JSON report hash to `adb6ff3964817673d3306b99d711ef6880a26d63cdb07f90ebad00ea4a48f720` and `4c0b7c7701723b7b20320a0c8a048e95c1a6755f5fefdcde8022fa9b46b19543`. All six slabs and both full Serving layouts compile. Only the compact histogram and push kernel bodies differ. Code generation confirms unchanged native and Decode kernels, expert blocks, submit sites, 80-row width and one serial slot.

The server verified both source manifests and archive SHA-256 `c4739a8df3d1590ad030dc3fd1227229f9db9504f6f9b2cf1b0ce94b3fae5e8a`. Suite `p83-20261122T233628Z` waits for Sweep 84 to terminate and for the successful Prefill 62 control. It runs the four frozen replays followed by native, Prefill 62 and candidate Serving.

## Sweep 84 · Compare direct expert operand stores through 16,384 tokens

Run the pinned original native implementation first and device-validated Prefill 62 second, each at 80 rows with one serial communication slot. The exact Sweep 52 workload and comparison script are reused. It has four timed requests per length at 1,024, 3,070, 6,142, 8,192, 12,288 and 16,384 tokens. Both arms have a 16,640-token context, 1,024-token chunks, the same warmups, long-answer checks and two concurrency waves. The checksum-verified input is `1b57f626bdd0e59050a5d002865cac95a664412f091c85031fc4d32f1e6e8722`. This provides a long-context test of the current validated base independent of the original compact-first Prefill 23 sweep.

Suite `sweep84-20261122T233409Z` uses the original four-card lock after Prefill 76 terminates and requires a successful Prefill 62. The locked script and suite have SHA-256 `da528910c935765cd71091f4c254dca951b3e125188fac0333bbb6e5598ee9db` and `956351424252c433c59d9b199c748fda0caa3635525dcf27fd7956338c6e542d`.

## Prefill 85 · Build one local metadata packet per direct push group

Extend Prefill 83's 56-block source assignment. The sender histogram writes a 64-byte rank-local packet for each group. It carries the expert ID, aligned destination row, valid count and up to 16 route IDs. Those fields fit unsigned 16 bits at the fixed EP4 80-row shape. The source reads one packet per group. It sends the same FP8 activations, packed MX scales and FP32 weights with the same wire stores and release. Mixed sources retain the original per-expert position storage.

The packet replaces the source's separate route-position stores, mirrored send plan and work list. Across the 1,560 archive rounds these go from 736,800 scalar stores plus 7,800 tile stores to 1,560 packet tile stores. The logical bytes written into those local buffers fall from 8,388,480 to 6,484,864 (22.69%). The packet tensor reserves 28,832 additional bytes of local scratch compared with the separate Prefill 83 work list. Communication window specifications match. The model counts local stores and bytes. Device timing includes the on-core work to build each packet.

Six host slabs and both full Serving layouts compile. Only the compact `moonep_histogram` and `moonep_push` bodies differ. The audit verifies the rank-local packet store and 64-byte loads. All native and Decode kernels, block assignments, task submissions, expert launches, communication buffers, default 80-row width and the serial slot match Prefill 83.

The server verified both source manifests and archive SHA-256 `60f3cbdb2f800fd8d352a0b198bd316a8d38341b1a09933057c1cea55e48fec7`. Suite `p85-20261122T234219Z` waits for successful Prefill 83. The same four-card lock gates the four frozen replays and candidate, Prefill 83 and native Serving.

## Prefill 86 · Store each direct push activation group as one tile

Extend Prefill 85. Load each FP8 input row directly into its final offset in a 16-row on-core tile using the existing `gather_row` operation. Send one tile per compact group at the original aligned destination row and limit its valid extent to the actual group count. The MX scale and FP32 weight transfers keep their original bytes, offsets and group structure. The 56 push blocks, release signals, operand layout and mixed-source path follow the base.

The 1,560 real rounds have 558,268 cross-card activation rows in 75,097 cross-card groups. Runs of adjacent source tokens within groups would issue 415,556 writes even with a 16-row limit. Gathering into the final on-core rows lets the source write 75,097 times (86.55% fewer activation writes). With the two unchanged metadata writes per group, modeled cross-card forward writes drop from 708,462 to 225,291 (68.20%). The valid payload bytes match. Each active block uses an 80 KiB activation tile. The model counts operations without assigning timing to the larger tile or its loads.

The script `experiments/diag-59/direct-push-activation-runs-v2.py` and its report hash to `9b59374c99b0f3e4508ed456e737c3e304c93b0120fb96434bcac4baa36de3cf` and `a25ab42c1e3ce0773771ad317ed9e6241e488c2f7f77f1023ee9c5eccf3a17ed`. Six host slabs and complete native and compact Serving compile. The generated push uses GM loads into activation tile subviews and one valid-row activation store per group. Only `moonep_push` changes. Native, Decode, expert kernels, block assignments, submission order, communication windows, 80-row width and single serial slot match.

The server verified both source manifests and archive SHA-256 `73cb133daee2324b4076e4f61bdaedf5f41abe81121670f5317cdd185ba10b0f`. Suite `p86-20261122T234941Z` waits for a successful Prefill 85. Under the original four-card lock, it runs the four frozen replays and then native, Prefill 85 and candidate Serving.

## Prefill 87 · Gather a compact activation group with one indexed load

Extend Prefill 86 with the A5 `MGATHER` row operation. Reuse the source token indices already derived from the compact group packet to build a 16-entry index tile. Set its valid count to the group's actual row count, gather from the 80-row FP8 source and send the same valid activation group as Prefill 86. Scale and FP32 routing-weight preparation, return and release protocols, storage and 56 push blocks follow Prefill 86.

The archive workload uses 736,800 local row loads for activations in Prefill 86. This prototype replaces those loads with 99,766 indexed gather calls and preserves the bytes and cross-rank store count. The operator has different execution overhead and needs device timing. Six host slabs and both full Serving builds compile. Only compact `moonep_push` differs and its generated code uses one `MGATHER` per group, on the unchanged 80-by-5,120 FP8 source. Native and Decode code, expert kernels, launch assignments, submit sites, communication windows, width and single serial slot match.

The server verified both source manifests and archive SHA-256 `d4cc9a9383339d4f087620eeae4d137b8add30c8f1f42c4fa8d31a38872ee0cc`. Suite `p87-20261122T235131Z` waits for a successful Prefill 86. It runs four numerical replays under the same lock, then candidate, Prefill 86 and native Serving.

## Prefill 88 · Pack exact routing weights once per source round

Extend Prefill 86 independently of the indexed gather. The compact sender histogram loads the FP32 routing table once per source round. It writes each weight beside its route ID into the second half of the existing group packet. The push reinterprets those 64 bytes as 16 FP32 entries and sends them using the original weight store. The group retains the exact valid weights and zero padding. Activation and scale loads, communication bytes and offsets, return publication, mixed-source processing and push block assignment follow Prefill 86.

In the 1,560 archived rounds, 99,766 group loads of the 80-by-6 table become 1,560 round loads. The model counts 191,550,720 logical table bytes before and 2,995,200 after. Enlarging the rank-local packet from 64 to 128 bytes adds 6,484,864 bytes of written packets and 6,385,024 bytes of reads. The sum is 175,685,632 fewer logical local bytes in this subset, with 30,784 extra reserved scratch bytes per rank. The device cost also includes filling the packet in the histogram and the realized cache behavior.

All six slabs and both full Serving layouts compile on the host. The generated histogram's FP32 view shares its on-core address with the unsigned packet and completes the writes before storing it. The sender reads the packet's second 64-byte half at the original weight-store shape. Only compact `moonep_histogram` and `moonep_push` change. Both routing operands come from the same existing gate task. Native, Decode, expert kernels, block assignments, task submissions, communication windows, default width and the single slot match Prefill 86.

The local experiment archive has SHA-256 `3947aac839ca3f4b980d698c4057ab6d58b88a079922c24d5d90d50aec62b0a1`. Server upload is pending explicit transfer authorization. The transfer had written zero archive bytes at the time of the check.

## Prefill 89 · Wait for one return count per remote owner

Starting from Prefill 82, remote scatter blocks add one to a shared counter for their owner after completing their BF16 stores. The source reads the same routing-plan count header and registers one threshold per peer. Empty owners retain the block-zero acknowledgement. The source clears the three counters in its existing round-release task after reduction and before releasing the peers. The same release and reuse dependencies gate every next round.

Across the archived 1,560 rounds, this reduces deferred conditions from 37,077 to 4,680, by 87.38%. Per-round median and P95 go from 24 and 29 to three. The number of remote publications matches Prefill 82. The prototype trades each per-block set for an atomic increment and adds three local resets per round. The device comparison needs to measure the shared-address atomic contention and reset cost.

Six host slabs and both full Serving layouts compile. Only the compact scatter, return wait, round release and initial signal clear kernels change. The generated code orders BF16 stores and fences before the increment and resets a peer's counter before notifying its release. The native and Decode kernels, expert launches, block assignments, submit sites, communication windows, 80-row width and serial slot match Prefill 82. The local package passed shell syntax and patch application checks and hashes to `ee9273ca72f6d6b81b958ebb1164076759228e142cd85c0ff14a24fec0ef3978`. Hardware testing requires a validated Prefill 82 control and authorization to upload the archive.

## Prefill 90 · Load scale rows directly into the compact send tile

Extend Prefill 87's existing sender-plan path. Its generated push loads each source scale row into a temporary tile and issues a second vector move into the prefilled 16-row group. Load directly into the destination row with `gather_row`. The group still initializes to scale value 127, leaving inactive rows unchanged. Packing produces the same MX scale tile for the existing store. The indexed activation gather, FP32 weights, mixed-source path and notifications follow Prefill 87.

The archived 1,560 rounds contain 736,800 routed rows. The candidate removes one 160-byte local tile move per route, or 117,888,000 logical bytes copied on core. Source GM reads and the bytes and offsets of all rank transfers match. Six slabs and both full Serving layouts compile on the host. Only compact `moonep_push` differs. Generated code retains the fill and load ordering, uses one direct row load and emits no sender-plan scale copy. Native and Decode kernels, expert launches, block assignments, task submissions, communication windows, 80-row width and the serial slot match Prefill 87. The local archive passed manifest, patch and shell syntax checks. Its SHA-256 is `22ca0b8b2022342f8a7c614e6d49ccacd5e9a4e56929c23cd8b54336906f91f9`. Device latency is pending.

## Diagnostic 91 · Check an asynchronous compact return transfer

The fixed A5 compiler passed five existing async transfer code-generation tests. A host probe then lowered a dynamic contiguous BF16 region in the existing 480-by-5,120 return-window shape. It emits one `pypto_urma_put_async_bf16` call with the same window argument for the local and peer slices and no UB staging tile. The probe covers an owner publishing its local compact output into its assigned range on the source rank.

The Prefill 80 route model counts 37,095 cross-rank 16-row stores for 1,560 rounds. A single submission per nonempty remote owner's contiguous lane would issue 4,680 calls, while moving the same 558,268 BF16 rows. This compares operation counts at the API boundary. A Serving integration needs the remote owner's exact W2 output to reside in its local return window and must publish the peer flag after the asynchronous task completes. The source's local return, mixed sources, expert launch dimensions, serial slot and round release remain part of the matched comparison. Host lowering establishes the operand and region shape. Runtime completion, contention and Serving latency still need verification.

### Prefill 91 local Serving prototype

Start from Prefill 80. In compact single-source rounds, each non-source owner can use its local 480-row return window as the W2 destination. It has no peer result writes in that role. W2 keeps its original compact active-row order. The owner submits the valid prefix as one URMA operation into the preassigned lane on the source. The existing return notification depends on the selected return task's completion. A zero-row owner takes the same notification path after an empty transport task. The source rank and mixed-source rounds retain the existing scatter. Round release and TopK-order reduction keep their original dependencies.

The reusable activation helper keeps native W13 and SwiGLU together, followed by its original result allocation and W2. This preserves all 109 native kernel bodies, 289 submissions, 230 block assignments and 110 allocation groups. Both sets of 45 Decode kernels match. Compact generation retains all 109 existing kernel bodies and adds only `moonep_return_urma`. Each runtime branch launches the same expert sequence with 32, 64 and 32 blocks. The remote W2 argument uses the existing 480-row window as its backing extent and keeps the 1,920-row activation capacity. The generated graph binds W2 output and URMA to that window, then makes notify wait on the selected completion. Payload-window sizes, default 80-row width and serial single-slot reuse match. The compact communication domain requests the runtime's existing URMA workspace.

All six host slabs and both full Serving layouts compile. Local archive SHA-256 is `d0f059e4cf16c399977a8f24ad1482f40412548ea585211344cfde024033bb81`. Source manifests, patch application and shell syntax passed. The installed server runtime contains the URMA completion adapter header. Hardware correctness, URMA completion and latency require the four-card numerical fixtures and matched Serving benchmark. The archive is staged locally.

The production V4.1 binding uses the existing `tensormap_and_ringbuffer` configuration. Its persistent domain provider forwards the generated workspace options into the installed Simpler device-domain allocator. A metadata-only call to the installed validator accepts A5 device IDs 4–7, logical domain members `(0, 1, 2, 3)` and `workspace_engines=("urma",)` with this runtime. The pinned URMA header limits each work-queue entry to 256 MiB. At 80 rows and TopK six, an owner's entire BF16 lane is at most 4,915,200 bytes, within a single entry at this limit. The adapter binds each completion to the enclosing task. The live domain and provider-reported queue capacity remain part of the device validation.
