# DeepSeek V4.1 A5 EP4 optimization log

## Acceptance

- Reduce the repeatable process-start to HTTP `/health` time for the real checkpoint. Track checkpoint staging, kernel preparation, device initialization, and first-request communicator setup separately.
- After the startup change is validated, tune MoonEP with matching native and candidate EP4 workloads. Use a 30% reduction in median warm prefill time on the fixed workload as the working threshold, calculated as `1 - candidate_seconds / native_seconds`. Keep the same model, prompt tokens, topology, and scheduler settings, and report tail latency and correctness with the result.
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

The normal CLI reduced median warm Prefill by 31.47% and P95 by 30.26% against the locked native control from Prefill 11. The median reductions by prompt length were 31.08%, 31.47% and 31.42% at 1,024, 3,070 and 6,142 tokens. Against the earlier native run's 2.223 s median, the reduction was 25.10%. The acceptance result uses the same-session control with the same checkpoint, sidecar, scheduler and unprofiled runtime settings. The 30% criterion is met on that fixed warm workload with DSpark off. The 256-row setting is opt-in for EP4 compact. Keep the concurrent checks as correctness evidence and use `benchmark-12-comparison.json` for the complete timed samples.
