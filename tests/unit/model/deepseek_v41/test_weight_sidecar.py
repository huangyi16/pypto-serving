# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise the V4.1 sidecar writer through the production mmap reader."""

import torch

from pypto_serving.model.deepseek_v41.weight_loader import (
    _ALL,
    DecodeWeightBank,
    DecodeWeightLayout,
    DecodeWeightSlice,
    load_prepacked_bank,
    source_fingerprint,
)
from pypto_serving.tools.prepack_deepseek_v41 import write_sidecar


def test_prepacked_ep4_bank_preserves_tensor_bytes(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"weight":"model-01.safetensors"}}')
    (tmp_path / "model-01.safetensors").touch()
    output = tmp_path / "ep4.safetensors"
    layout = DecodeWeightLayout(2, 1, 2, 1, world=4)
    bank = DecodeWeightBank(layout)
    dtypes = {
        "gate_weight": torch.bfloat16,
        "wq_a": torch.float8_e4m3fn,
        "wq_a_scale": torch.float8_e8m0fnu,
        "routed_w1": torch.int8,
        "hc_attn_fn": torch.float32,
    }
    for name in sorted(_ALL):
        for slot in range(layout.slots(name)):
            value = torch.arange(32, dtype=torch.uint8).reshape(4, 8) + slot
            bank.append(DecodeWeightSlice(name, slot, value.view(dtypes.get(name, torch.uint8))))
    expected = bank.finish(_ALL)

    write_sidecar(output, expected, source_fingerprint(tmp_path, layout))
    mapped = load_prepacked_bank(tmp_path, layout, _ALL, path=output)

    assert mapped.prepacked
    assert mapped.allocated_bytes == bank.allocated_bytes
    for name, original in expected.items():
        actual = mapped.tensors[name]
        assert actual.shape == original.shape
        assert actual.dtype == original.dtype
        assert torch.equal(actual.view(torch.uint8), original.view(torch.uint8)), name
    assert load_prepacked_bank(tmp_path, DecodeWeightLayout(2, 1, 2, 1, world=8),
                               _ALL, path=output) is None
