# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Replay a fixed HTTP request arrival trace against a V4.1 serving process."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
import urllib.request

from daily_serving_perf import HTTP, percentile, read_stream


def send_case(url, model, case, arrival, timeout):
    sent = time.monotonic()
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": case["max_tokens"],
    }).encode()
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    result = {"client_queue_s": max(0, sent - arrival)}
    try:
        with HTTP.open(request, timeout=timeout) as response:
            result.update(read_stream(response, arrival))
        output = result["usage"]["completion_tokens"]
        if output > 1:
            result["tpot_s"] = (result["last_token_s"] - result["ttft_s"]) / (output - 1)
    except (OSError, ValueError, RuntimeError) as exc:
        result["error"] = str(exc)
    return result


def replay(url, model, cases, timeout, max_inflight):
    with ThreadPoolExecutor(max_workers=max_inflight) as pool:
        futures = []
        start = time.monotonic()
        for case in cases:
            arrival = start + case["arrival_s"]
            delay = arrival - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            futures.append(pool.submit(send_case, url, model, case, arrival, timeout))
        results = [
            {"request": index, "arrival_s": case["arrival_s"], **future.result()}
            for index, (case, future) in enumerate(zip(cases, futures, strict=True))
        ]
        wall_s = time.monotonic() - start
    successful = [result for result in results if "error" not in result]
    summary = {
        "requests": len(results),
        "errors": len(results) - len(successful),
        "wall_s": wall_s,
        "input_tokens_per_s": sum(item["usage"]["prompt_tokens"] for item in successful) / wall_s,
        "output_tokens_per_s": sum(item["usage"]["completion_tokens"] for item in successful) / wall_s,
    }
    if successful:
        ttft = [item["ttft_s"] for item in successful]
        for p in (50, 95, 99):
            summary[f"ttft_p{p}_s"] = percentile(ttft, p / 100)
        tpot = [item["tpot_s"] for item in successful if "tpot_s" in item]
        if tpot:
            summary["tpot_p50_s"] = percentile(tpot, 0.5)
    return {"summary": summary, "requests": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-inflight", type=int, default=64)
    args = parser.parse_args()
    cases = json.loads(args.workload.read_text())
    if not isinstance(cases, list) or not cases or any(
        not isinstance(case, dict)
        or not isinstance(case.get("prompt"), str)
        or not case["prompt"]
        or type(case.get("max_tokens")) is not int
        or case["max_tokens"] <= 0
        or type(case.get("arrival_s")) not in (int, float)
        or not math.isfinite(case["arrival_s"])
        or case["arrival_s"] < 0
        for case in cases
    ):
        parser.error("workload must contain requests with prompt, positive max_tokens and nonnegative arrival_s")
    if args.timeout <= 0 or args.max_inflight <= 0:
        parser.error("timeout and max-inflight must be positive")
    cases.sort(key=lambda case: case["arrival_s"])
    report = replay(args.url, args.model, cases, args.timeout, args.max_inflight)
    report.update(label=args.label, model=args.model, workload=str(args.workload))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))
    if report["summary"]["errors"] == len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
