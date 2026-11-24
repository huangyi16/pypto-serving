# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import ctypes

import pytest
import torch

from pypto_serving.model.common.moe_offload import ExpertBacking, ROUTED_FIELDS


def test_mapped_rows_preserve_rank_layer_expert_and_field_bytes():
    layers, ranks, experts = 3, 2, 4
    weights = torch.arange(layers * ranks * experts * 32).byte().view(layers, ranks, experts, 32)
    scales = torch.arange(layers * ranks * experts * 16).short().view(layers, ranks, experts, 16)
    fields = tuple({"w": weights[:, rank], "scale": scales[:, rank]} for rank in range(ranks))
    backing = ExpertBacking(fields, budget_bytes=191)
    assert backing.capacity == 2

    windows = [bytearray(layers * experts * backing.expert_bytes) for _ in range(ranks)]
    bases = [ctypes.addressof(ctypes.c_char.from_buffer(window)) for window in windows]
    descriptors = backing.descriptors(bases)
    for rank, window in enumerate(windows):
        backing.initialize(rank, memoryview(window))
        for layer in range(layers):
            for field, tensor in enumerate(fields[rank].values()):
                start, stride, size, slot_stride = descriptors[rank, layer, field].tolist()
                for expert in range(experts):
                    assert ctypes.string_at(start + expert * stride, size) == tensor[layer, expert].numpy().tobytes()
                assert slot_stride == tensor[layer, 0].nbytes


def test_v4_offload_accepts_ep2_and_packs_gate_down_up():
    from argparse import Namespace

    from pypto_serving.cli.main import _validate_expert_offload
    from pypto_serving.model.deepseek.expert_offload import routed_expert_backing

    args = Namespace(platform="a2a3", speculative_config=None, num_speculative_tokens=None)
    with pytest.raises(ValueError, match="--ep 1, 2 or 4"):
        _validate_expert_offload(Namespace(**vars(args), expert_parallel_size=8), "deepseek_v4", 4096)
    _validate_expert_offload(Namespace(**vars(args), expert_parallel_size=2), "deepseek_v4", 1 << 20)

    layers, ranks, experts = 2, 2, 3
    hidden, intermediate = 32, 64
    rows = layers * experts

    def packed(tail, dtype):
        return torch.arange(ranks * rows * int(torch.tensor(tail).prod()), dtype=dtype).reshape(ranks, rows, *tail)

    tensors = {
        "routed_w1": packed((intermediate, hidden), torch.int8),
        "routed_w1_scale": packed((intermediate,), torch.float32),
        "routed_w2": packed((hidden, intermediate), torch.int8),
        "routed_w2_scale": packed((hidden,), torch.float32),
        "routed_w3": packed((intermediate, hidden), torch.int8),
        "routed_w3_scale": packed((intermediate,), torch.float32),
    }
    backing = routed_expert_backing(tensors, layers, budget_bytes=1 << 20)
    assert tuple(backing.by_rank[0]) == ROUTED_FIELDS
    assert tuple(backing.strides) == ROUTED_FIELDS
    strides = [tensors[name][0, 0].nbytes for name in ROUTED_FIELDS]
    assert list(backing.strides.values()) == strides
    assert backing.by_rank[1]["routed_w2"].shape == (layers, experts, hidden, intermediate)
    descriptors = backing.descriptors([1000, 2000])
    assert descriptors.shape == (ranks, layers, 6, 4)
    assert descriptors[0, 0, :, 1].tolist() == strides


def test_budget_must_fit_one_packed_expert():
    fields = ({"w": torch.empty(2, 2, 32, dtype=torch.uint8)},)
    with pytest.raises(ValueError, match="at least 32 bytes"):
        ExpertBacking(fields, 31)
    assert ExpertBacking(fields, 65536).capacity == 4
