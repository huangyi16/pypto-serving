# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build a reusable TP1/EP4 or TP1/EP8 DeepSeek V4.1 weight sidecar."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import struct
import tempfile
import time

import torch

from pypto_serving.model.deepseek_v41.npu_runner import SegmentTopology
from pypto_serving.model.deepseek_v41.weight_loader import (
    _FLOAT8,
    _FORMAT,
    DecodeWeightBank,
    DecodeWeightLayout,
    iter_decode_weight_slices,
    plan_layers,
    source_fingerprint,
)

_DTYPE_CODES = {
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
}
_FLOAT8_NAMES = {dtype: name for name, dtype in _FLOAT8.items()}


def write_sidecar(output, tensors, fingerprint):
    """Write the safetensors header and stream existing tensor storage to disk."""
    float8_fields = {}
    header = {"__metadata__": {
        "format": _FORMAT,
        "source_fingerprint": fingerprint,
    }}
    offset = 0
    for name, tensor in tensors.items():
        dtype = tensor.dtype
        if dtype in _FLOAT8_NAMES:
            float8_fields[name] = _FLOAT8_NAMES[dtype]
            dtype = torch.uint8
        end = offset + tensor.numel() * tensor.element_size()
        header[name] = {
            "dtype": _DTYPE_CODES[dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, end],
        }
        offset = end
    header["__metadata__"]["float8_fields"] = json.dumps(float8_fields)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(struct.pack("<Q", len(encoded)))
            stream.write(encoded)
            chunk_bytes = 64 << 20
            for tensor in tensors.values():
                payload = memoryview(tensor.view(torch.uint8).numpy()).cast("B")
                for start in range(0, len(payload), chunk_bytes):
                    stream.write(payload[start:start + chunk_bytes])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_sidecar(model_dir, output, ranks):
    """Pack the same weight slices used by the ordinary Serving fallback."""
    raw = json.loads((model_dir / "config.json").read_text())
    layers = plan_layers(raw)
    layout = DecodeWeightLayout(
        len(layers),
        sum(layer.kv_source == layer.layer_id for layer in layers),
        sum(layer.index_source == layer.layer_id for layer in layers),
        sum(layer.mode == "c2a_full" for layer in layers),
        world=ranks,
    )
    topology = SegmentTopology.for_decode(dp=ranks)
    fingerprint = source_fingerprint(model_dir, layout)
    bank = DecodeWeightBank(layout)
    started = time.perf_counter()
    for item in iter_decode_weight_slices(model_dir, topology, parallel_experts=True, bank=bank):
        bank.append(item)
        if item.name == "hc_attn_fn":
            print(f"Packed layer {item.slot + 1}/{layout.layers} in {time.perf_counter() - started:.1f}s",
                  flush=True)
    write_sidecar(output, bank.finish(bank.tensors), fingerprint)
    print(f"Wrote {output} ({output.stat().st_size / 2**30:.2f} GiB) "
          f"in {time.perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--ranks", type=int, choices=(4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error(f"{args.output} already exists; pass --force to replace it")
    logging.basicConfig(level=logging.INFO)
    if torch.get_num_threads() < 8:
        torch.set_num_threads(8)
    build_sidecar(args.model_dir, args.output, args.ranks)


if __name__ == "__main__":
    main()
