# Tune V4.1 Prefill expert parallelism on A5

Use the paired Serving and pypto-lib checkouts with a DeepSeek V4.1 Flash checkpoint on TP1, CP1 and DP=EP4 or EP8. Set `PYPTO_LIB_ROOT` to the paired lib checkout. Record both Git revisions, any working-tree changes, the checkpoint, A5 topology, CANN, PyPTO, Simpler and PTOAS versions, device IDs, and the JIT cache directory. Use a new Serving process for each graph variant.

See the [EP4 optimization log](v41-a5-ep4-optimization-log.md) for the startup baseline, prepack workflow, and iteration results.

## Experiment arms

| Arm | CLI backend | Additional environment | Graph behavior |
| --- | --- | --- | --- |
| N | `native` | Start the locked baseline builds | Original full-capacity rounds |
| N1 | `native` | `PYPTO_V41_PREFILL_TRIM_EMPTY_ROUNDS=1` | Native transfer with group-wide empty tail rounds removed |
| F | `moonep` | `PYPTO_V41_PREFILL_MOONEP_LAYOUT=fixed` | Direct transfer into the original expert stride |
| A | `moonep` | `PYPTO_V41_PREFILL_MOONEP_LAYOUT=compact` | Direct transfer into aligned compact expert segments |
| B | `moonep` | `PYPTO_V41_PREFILL_MOONEP_LAYOUT=preplaced` and a replay directory | Replay placement and load missing weight slots before the measured window |
| C | `moonep` | `PYPTO_V41_PREFILL_MOONEP_LAYOUT=planned` and the same replay directory | Preload the same slots and measure the planner, checking its decision against the capture |
| D | `moonep` | `PYPTO_V41_PREFILL_MOONEP_LAYOUT=online` | Plan placement, send missing packed weights and execute the resulting segments |

For EP4 compact, set `PYPTO_V41_PREFILL_MOE_TOKENS` in a fresh worker to
sweep the number of token rows per MoE round. It accepts multiples of 16 from
16 through 256. The compiler caps the width at the configured Prefill capacity
and the scheduler uses the same effective width for the round count. Decode
keeps its 80-row ABI.

The 256-row EP4 benchmark uses `PYPTO_V41_PREFILL_MOE_TOKENS=256` with
`--v41-prefill-ep-backend moonep --dp 4 --ep 4` and a 1,024-token Prefill
budget. Use the same checkpoint sidecar and scheduler settings for the native
comparison.

Select the backend with `--v41-prefill-ep-backend`. The defaults are `native` for Serving and `compact` when MoonEP is selected. F through D trim empty tail rounds. The planner currently selects at most one move per round, using one allocated slot per rank. Its cold-copy penalty is measured in 64-row compute tiles and defaults to 2. Sweep `PYPTO_V41_PREFILL_MOONEP_COPY_COST_TILES` in D and use the capture's value in C. The slot state is retired at the layer boundary. The D path uses a single bank. A copy stores 18,800,640 bytes of packed payload and MX scales. Compact input segments use 32-row alignment. The initial histogram exchange gives every rank the same source-by-expert counts.

## Verify the device path before timing

Use `TASK_DEVICE` from the allocated A5 job and the same scheduler settings for every arm. Start from the V4.1 HTTP regression in `tests/test_deepseek_v41_serving.py`. For comparisons that require the backend flag, set `PYPTO_DSV41_PREFILL_EP_BACKEND` for the test process. Run it with DSpark disabled and enabled. Cover empty source ranks, the 79/80/81-row boundary, and repeated forwards on one worker so each layout exercises window reuse. Check greedy output, logits where available, and the DSpark target hidden. Use the same checkpoint and real prompts for correctness and performance runs.

Run the slab golden from the paired lib checkout before testing online Serving. Repeat with the `native`, `fixed`, `compact` and `online` layouts.

```bash
python models/deepseek_v4_1_flash/prefill_moe.py --platform a5 --ep 4 \
  --device "$TASK_DEVICE" --experts-per-rank 96 --entry slabs --tokens 80 \
  --slab-capacity 128 --num-tokens-per-rank 81,80,79,0 \
  --moonep-layout online --moonep-copy-cost-tiles 0 --route-case hot
```

## Capture and replay routing

Set `PYPTO_V41_PREFILL_ROUTE_TRACE_DIR` to a fresh directory and run the desired chunks. The worker writes one `prefill-NNNNNN.pt` file per forward with rank/stage counts and gate top-6 expert IDs and FP32 weights. D captures also contain the selected expert, destination, moved route count and copy flag per round. Analyze the home-load distribution from the paired lib checkout.

```bash
python -m models.deepseek_v4_1_flash.route_replay /path/to/routes/prefill-000000.pt --tokens 80
```

Set `PYPTO_V41_PREFILL_ROUTE_REPLAY_DIR` in a new worker and submit the same workload in the same chunk order. Use one request stream for reproducible routing captures, or drive the same staged batches when tuning mixed-rank load. Replay validates rank counts, routing bounds, and the copy sequence for B/C. C also reports a mismatch between the device planner and captured placement. The lib's checkpoint-free `prefill_fwd.py` fixture can replay a single capture with `--moonep-route-mode replay --moonep-route-input <file>`, compiled at the same EP, capacity, and slab size. Use `--query-lens` to match the recorded counts.

For B/C, use the device swimlane window after `moonep_replica_wait` to exclude the staging copies from the measured sample. Record those copies separately. Measure the online exposed copy from the D swimlane, where `moonep_replica_copy` can overlap the payload push. Compare A, B and C over matching per-round windows and verify the same decisions and active rows. Keep capture and replay outside formal service measurements.

## Measure full requests

Create a JSON list of request objects with `arrival_s`, `prompt` and `max_tokens`. Warm the service before measuring. Replay the same file for each arm and repeat each arrival-rate point with separate output files.

```bash
python scripts/v41_prefill_bench.py --model dsv41 --label A \
  --workload /path/to/arrivals.json --output /path/to/A.json
```

The client keeps raw samples with client queue time, TTFT, TPOT, request latency, usage and failures. Compare successful input tokens/s at the same P95 TTFT target, checking client queueing, P99 and errors. Run profiling separately. Use the `moonep_histogram`, `moonep_placement`, `moonep_row_plan`, `moonep_push`, `moonep_pack_scale`, `moonep_replica_copy` and `moonep_scatter` tasks to explain observed differences. Sweep offered load and prompt lengths at both EP sizes and record available KV pages for each worker.
