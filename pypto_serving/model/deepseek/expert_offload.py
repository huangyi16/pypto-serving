# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rank-local mapped expert geometry and packed W8A8 views."""

import torch

from pypto_serving.model.common.moe_offload import ExpertBacking, ROUTED_FIELDS, expert_slot_capacity


def v4_expert_slots(budget, config_data, *, layers, ranks, hidden, n_routed_experts,
                    prefill_tokens, decode_tokens, max_batch_size):
    local_experts = n_routed_experts // ranks
    intermediate = int(config_data.get("moe_intermediate_size", 2048))
    expert_bytes = 3 * hidden * intermediate + 4 * (2 * intermediate + hidden)
    expert_slots = expert_slot_capacity(
        budget, expert_bytes, layers, local_experts,
    )
    topk = int(config_data.get("num_experts_per_tok", 6))
    prefill_slots = min(local_experts, prefill_tokens * topk)
    if expert_slots < prefill_slots:
        raise ValueError(
            f"DeepSeek V4 prefill needs {prefill_slots} expert slots per rank "
            f"({prefill_slots * expert_bytes} bytes) for the compiled "
            f"capacity of {prefill_tokens} tokens per rank"
        )
    # One planner wave must fit every distinct expert that may receive a route.
    max_tokens_per_rank = min(decode_tokens, max_batch_size)
    required_slots = min(local_experts, ranks * max_tokens_per_rank * topk)
    if expert_slots < required_slots:
        raise ValueError(f"DeepSeek V4 decode needs {required_slots} expert slots per rank "
                         f"({required_slots * expert_bytes} bytes) for the configured batch size")
    return expert_slots


def routed_expert_backing(
    tensors: dict[str, torch.Tensor], layers: int, budget_bytes: int,
) -> ExpertBacking:
    """Expose layer and expert axes on each rank's packed W8A8 fields."""
    ranks, rows = tensors[ROUTED_FIELDS[0]].shape[:2]
    local_experts = rows // layers
    return ExpertBacking(tuple(
        {
            name: tensors[name][rank].view(layers, local_experts, *tensors[name].shape[2:])
            for name in ROUTED_FIELDS
        }
        for rank in range(ranks)
    ), budget_bytes)
