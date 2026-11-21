# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host staging for group-wide V4.1 prefill MoE round limits."""

import pytest
import torch

from pypto_serving.model.deepseek_v41.npu_runner import PrefillMetadataResources, SegmentTopology
from pypto_serving.model.deepseek_v41.task_args import ForwardStep, RequestSlice


@pytest.mark.parametrize(("world", "width", "rounds_for_81", "capacity_rounds"), (
    (4, 80, 2, 4),
    (8, 80, 2, 4),
    (4, 256, 1, 1),
))
@pytest.mark.parametrize("trim", (False, True))
def test_prefill_moe_rounds_share_one_limit_across_ranks_and_stages(world, width, rounds_for_81, capacity_rounds, trim):
    topology = SegmentTopology.for_prefill(dp=world, max_tokens=256)
    rope = (torch.ones(256, 2), torch.zeros(256, 2))
    metadata = PrefillMetadataResources(
        topology, max_seq_len=256, max_requests=1, window_pages=2, c2a_pages=1,
        c1a_pages=2, rope_tables=rope, compressed_rope_tables=rope,
        full_hidden_size=4, engram_k=1, moe_tokens=width, trim_empty_moe_rounds=trim,
    )
    arguments = metadata.full_arguments()
    assert arguments["moe_rounds"].is_shared()
    pages = {"window": (0, 1), "c2a": (0,), "c1a": (0, 1)}
    for epoch, (count, expected) in enumerate(((81, rounds_for_81), (80, 1), (79, 1), (0, 0)), start=1):
        requests = (RequestSlice("req", world - 1, 0, 0, (1,) * count, count, pages),) if count else ()
        embeddings = torch.ones(count, 4, dtype=torch.bfloat16)
        metadata.stage_full(ForwardStep("prefill", requests, epoch), embeddings)
        torch.testing.assert_close(
            arguments["moe_rounds"],
            torch.full((world, 2), expected if trim else capacity_rounds, dtype=torch.int32),
        )
        torch.testing.assert_close(arguments["counts"][-1], torch.full((2,), count, dtype=torch.int32))
        assert not arguments["counts"][:-1].any()
