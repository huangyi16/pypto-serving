# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise graph selection and ABI assembly through the real lib entries."""

import importlib
import inspect
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from pypto_serving.model.common.moe_offload import ROUTED_FIELDS
from pypto_serving.model.deepseek_v41 import npu_runner as runner
from pypto_serving.model.deepseek_v41.offload_runner import V41OffloadExecution
from pypto_serving.model.deepseek_v41.weight_loader import DecodeWeightBank


@pytest.fixture
def kernel_root():
    root = Path(os.environ.get("PYPTO_LIB_ROOT", Path(__file__).resolve().parents[4] / "pypto-lib"))
    kernel_dir = root / "models" / "deepseek_v4_1_flash"
    names = {path.stem for path in kernel_dir.glob("*.py")}
    package = "models.deepseek_v4_1_flash"
    saved = {name: module for name, module in sys.modules.items()
             if name in names or name == package or name.startswith(package + ".")}
    old_path, old_argv = sys.path[:], sys.argv[:]
    for name in saved:
        sys.modules.pop(name)
    try:
        yield root
    finally:
        for name in tuple(sys.modules):
            if name in names or name == package or name.startswith(package + "."):
                sys.modules.pop(name)
        sys.modules.update(saved)
        sys.path[:], sys.argv = old_path, old_argv


@pytest.fixture
def a5_graph_backend():
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    if not (cann / "asc/include/basic_api").is_dir():
        pytest.skip("V4.1 graph imports require CANN/AscendC headers")
    from pypto import backend

    previous = backend.get_backend_type() if backend.is_backend_configured() else None
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    try:
        yield
    finally:
        backend.reset_for_testing()
        if previous is not None:
            backend.set_backend_type(previous)


@pytest.mark.parametrize("mapped,dp,speculative", [(False, 4, 0), (True, 2, 0), (False, 4, 5)])
def test_execution_selects_graphs_and_generation_head(kernel_root, a5_graph_backend, mapped, dp, speculative):
    execution = V41OffloadExecution() if mapped else runner.V41ResidentExecution()
    topology = runner.selected_decode_topology(
        kernel_root, dp_size=dp, supported_dp_sizes=execution.supported_dp_sizes)
    runner.load_segment_modules(kernel_root, topology)
    config = sys.modules["config"]
    text = asdict(config.FLASH)
    options = SimpleNamespace(expert_cache_bytes=8 << 30)
    runtime = SimpleNamespace(num_speculative_tokens=speculative)
    execution.prepare(kernel_root, topology, runtime, options, text, text["num_hidden_layers"])
    compiled = {}

    class SpecializingCompiler:
        def compile(self, name, entry, **kwargs):
            program = entry.specialize(**kwargs)
            assert program is not None
            if mapped:
                assert {*ROUTED_FIELDS, "expert_state"} <= set(entry.output_param_names)
            compiled[name] = (entry, program)
            return program

    forwards = execution.compile(
        SpecializingCompiler(), kernel_root, topology,
        runner.SegmentTopology.for_prefill(dp=dp, max_tokens=128),
        num_speculative_tokens=speculative,
    )
    assert forwards.decode_counts_are_tensor
    DecodeWeightBank.validate_abi(forwards.decode_names)
    expected = {"expert_state", "expert_descriptors"}
    assert expected <= set(forwards.decode_names)
    assert expected <= set(forwards.prefill_names)
    assert "engram_ids" in forwards.decode_names and "engram_lookup" not in forwards.decode_names
    assert "engram_ids" in forwards.prefill_names and "engram_lookup" not in forwards.prefill_names
    engram = sys.modules["engram"]
    assert engram.ENGRAM_ROWS >= max(config.FLASH.engram_num_embeddings)
    assert engram.ENGRAM_SHARD_ROWS * topology.world == engram.ENGRAM_ROWS
    assert not forwards.abi.PACK_ROUTED_WEIGHTS
    assert forwards.abi.__name__.endswith(".decode_fwd")
    assert (forwards.decode_head is None) == mapped
    assert ("v41_generation_head" in compiled) != mapped
    assert ("v41_decode_generate" in compiled) == mapped
    assert bool(forwards.dspark_programs) == bool(speculative)
    assert bool(forwards.dspark_param_names) == bool(speculative)


@pytest.mark.parametrize("world,rows,bytes_per_rank", [(4, 642, 23088), (8, 322, 15408)])
def test_resident_decode_expert_parameter_contract(kernel_root, a5_graph_backend, world, rows, bytes_per_rank):
    import pypto.language as pl

    runner.load_segment_modules(kernel_root, runner.SegmentTopology.for_decode(dp=world))
    module = importlib.import_module("models.deepseek_v4_1_flash.decode_fwd")
    parameters = inspect.signature(module.l3_decode_fwd._func).parameters
    state = parameters["expert_state"].annotation
    descriptors = parameters["expert_descriptors"].annotation
    assert tuple(state.shape) == (world, rows, 6)
    assert state.dtype == pl.INT32
    assert tuple(descriptors.shape) == (world, 40, 6, 4)
    assert descriptors.dtype == pl.INT64
    assert rows * 6 * 4 + 40 * 6 * 4 * 8 == bytes_per_rank
    assert module.RESIDENT_EXPERT_SLOTS == 1
    for entry in (module.decode_encoder, module.decode_decoder, module.decode_fwd):
        signature = inspect.signature(entry._func)
        for name in ("expert_state", "expert_descriptors", "expert_slots", "offload"):
            assert signature.parameters[name].default is inspect.Parameter.empty
    assert "expert_context" not in sys.modules


@pytest.mark.parametrize("world,bytes_per_rank", [(4, 23088), (8, 15408)])
def test_resident_expert_buffers_are_owned_until_close(kernel_root, world, bytes_per_rank):
    from pypto.runtime import DeviceTensor

    runner.load_segment_modules(kernel_root, runner.SegmentTopology.for_decode(dp=world))

    class AllocationWorker:
        def __init__(self):
            self.allocated = []
            self.freed = []

        def alloc_tensor(self, shape, dtype, *, worker_id):
            tensor = DeviceTensor(0x10000 * (len(self.allocated) + 1), tuple(shape), dtype)
            self.allocated.append(tensor)
            return tensor

        def free_stacked_tensor(self, tensor):
            self.freed.extend(tensor.shards)

    worker = AllocationWorker()
    resources = runner.ResidentExpertPlaceholders(worker, world=world, layers=40, experts=384 // world)
    assert len(worker.allocated) == 2 * world
    assert resources.allocated_bytes == world * bytes_per_rank
    assert set(resources.arguments) == {"expert_state", "expert_descriptors"}
    assert not worker.freed
    resources.close()
    assert {id(t) for t in worker.freed} == {id(t) for t in worker.allocated}
    assert not resources.arguments


@pytest.mark.parametrize("mapped,dp", [(False, 4), (False, 8), (True, 1), (True, 2), (True, 4), (True, 8)])
def test_execution_configures_topology_and_expert_geometry(kernel_root, mapped, dp):
    execution = V41OffloadExecution() if mapped else runner.V41ResidentExecution()
    topology = runner.selected_decode_topology(
        kernel_root, dp_size=dp, supported_dp_sizes=execution.supported_dp_sizes)
    runner.load_segment_modules(kernel_root, topology)
    config = sys.modules["config"]
    text = asdict(config.FLASH)
    local_experts = text["n_routed_experts"] // dp
    matrix = text["hidden_size"] * text["moe_intermediate_size"]
    budget = local_experts * 3 * (matrix // 2 + matrix // 32)
    options = SimpleNamespace(expert_cache_bytes=budget)
    runtime = SimpleNamespace(num_speculative_tokens=0)
    execution.prepare(kernel_root, topology, runtime, options, text, text["num_hidden_layers"])
    assert config.EP_SIZE == dp
    assert not hasattr(config, "EXPERT_SLOTS")
    if mapped:
        assert sys.modules["expert_context"].EXPERT_SLOTS == local_experts
        assert execution.expert_slots == local_experts
        original_argv = sys.argv[:]
        execution.prepare(kernel_root, topology, runtime, options, text, text["num_hidden_layers"])
        assert sys.argv == original_argv
        with pytest.raises(ValueError, match="fresh model worker"):
            V41OffloadExecution().prepare(
                kernel_root, topology, runtime, SimpleNamespace(expert_cache_bytes=2 * budget),
                text, text["num_hidden_layers"])
        assert sys.argv == original_argv
        assert execution.expert_cache_bytes == budget
        with pytest.raises(ValueError, match="non-speculative"):
            execution.prepare(kernel_root, topology, SimpleNamespace(num_speculative_tokens=5),
                              options, text, text["num_hidden_layers"])
    else:
        assert "expert_context" not in sys.modules
        with pytest.raises(ValueError, match="DP/EP"):
            runner.selected_decode_topology(
                kernel_root, dp_size=2, supported_dp_sizes=execution.supported_dp_sizes)


def test_prefill_uses_required_expert_parameters(kernel_root, a5_graph_backend, monkeypatch):
    runner.load_segment_modules(kernel_root, runner.SegmentTopology.for_decode(dp=4))
    module = importlib.import_module("models.deepseek_v4_1_flash.prefill_fwd")
    signature = inspect.signature(module.prefill_fwd._func)
    for name in ("expert_state", "expert_descriptors", "expert_slots", "offload"):
        assert signature.parameters[name].default is inspect.Parameter.empty
    assert {"expert_state", "expert_descriptors"} <= set(module.l3_prefill_fwd.param_names)
    assert "expert_descriptors" not in module.l3_prefill_fwd.output_param_names
    assert "expert_state" in module.l2_prefill_fwd.output_param_names
    banks = {"routed_w1", "routed_w1_scale", "routed_w2", "routed_w2_scale", "routed_w3", "routed_w3_scale"}
    assert banks.isdisjoint(module.l2_prefill_fwd.output_param_names)
    assert banks.isdisjoint(module.l3_prefill_fwd.output_param_names)
    monkeypatch.setattr(sys, "argv", ["offload", "--expert-slots", "384"])
    offload = importlib.import_module("models.deepseek_v4_1_flash.offload_prefill_fwd")
    assert banks | {"expert_state"} <= set(offload.l2_prefill_fwd.output_param_names)
    assert banks | {"expert_state"} <= set(offload.l3_prefill_fwd.output_param_names)
    assert "expert_descriptors" not in offload.l3_prefill_fwd.output_param_names


@pytest.mark.parametrize("ep,cp,capacity,local,moe", [
    (1, 1, 128, 128, 128), (2, 2, 128, 64, 64),
    (4, 1, 4096, 4096, 1024), (8, 4, 4096, 1024, 1024),
])
@pytest.mark.parametrize("slots", [0, 384])
def test_prefill_geometry_does_not_depend_on_cache_slots(
    kernel_root, a5_graph_backend, monkeypatch, ep, cp, capacity, local, moe, slots,
):
    monkeypatch.syspath_prepend(str(kernel_root / "models" / "deepseek_v4_1_flash"))
    monkeypatch.setattr(sys, "argv", ["prefill", "--ep", str(ep), "--cp", str(cp),
                                    "--capacity", str(capacity), "--expert-slots", str(slots)])
    config = importlib.import_module("config")
    assert not hasattr(config, "EXPERT_SLOTS")
    geometry = importlib.import_module("prefill_fwd")
    assert geometry.RANKS == ep
    assert geometry.CP_SIZE == cp
    assert geometry.DP_SIZE == ep // cp
    assert geometry.CAPACITY == capacity
    assert geometry.LOCAL_CAPACITY == local
    assert config.MOE_TOKENS == moe


@pytest.mark.parametrize("args", [[], ["--expert-slots", "0"], ["--expert-slots", "-1"]])
def test_offload_context_requires_positive_slot_count(kernel_root, monkeypatch, capsys, args):
    runner.load_segment_modules(kernel_root, runner.SegmentTopology.for_decode(dp=1))
    monkeypatch.setattr(sys, "argv", ["offload", *args])
    with pytest.raises(SystemExit) as exc:
        importlib.import_module("expert_context")
    assert exc.value.code == 2
    assert "--expert-slots" in capsys.readouterr().err
    assert not hasattr(sys.modules["config"], "EXPERT_SLOTS")


def test_executor_checks_the_selected_bindings_device_capability():
    from pypto_serving.model.deepseek_v41.npu_executor import DeepSeekV41PyptoExecutor

    groups = (SimpleNamespace(num_partitions=2, max_requests_per_partition=1),)
    bindings = SimpleNamespace(cache_groups=groups, supported_dp_sizes=(4, 8))
    executor = DeepSeekV41PyptoExecutor(device_ids=(0, 1), bindings=bindings, expert_cache_bytes=1)
    record = SimpleNamespace(runtime=SimpleNamespace(kv_cache_groups=groups, max_batch_size=2))
    with pytest.raises(ValueError, match="device count in "):
        executor.register_model("v41", record)


@pytest.mark.parametrize("row_stride", [None, 6])
def test_engram_stages_hash_ids_with_request_history_and_padding(row_stride):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    from pypto_serving.model.deepseek_v41.engram import HostEngram
    from pypto_serving.model.deepseek_v41.task_args import ForwardStep, RequestSlice

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3}, unk_token="[UNK]"))
    config = dict(engram_layer_ids=[1, 14], engram_max_ngram_size=3, engram_n_heads=2,
                  engram_vocab_size=7, engram_num_embeddings=[40, 80], engram_head_dim=256,
                  engram_compressed_vocab_size=4, engram_pad_token_id=0, hc_mult=4, hidden_size=64)
    # Hash staging must not read checkpoint tables or allocate projection weights.
    resources = runner.HostEngramResources.__new__(runner.HostEngramResources)
    resources.host = HostEngram(None, tokenizer, config)
    resources.capacity = 16
    resources.hash_columns = resources.host.layout.hash_columns
    reference = HostEngram(None, tokenizer, config)
    topology = runner.SegmentTopology(tp=1, dp=2, local_capacity=16)
    target = torch.empty((2, 2 * 16 * resources.hash_columns, 1), dtype=torch.int32)
    requests = (
        RequestSlice("a", 0, 0, 0, (1, 2), 2, {}),
        RequestSlice("b", 0, 1, 0, (3,), 1, {}),
        RequestSlice("c", 1, 0, 0, (2,), 1, {}),
    )
    resources.stage_ids(target, ForwardStep("prefill", requests, 0), topology, row_stride=row_stride)
    rows = target.view(2, 2, 16, resources.hash_columns)
    second = 2 if row_stride is None else row_stride
    for request, start in zip(requests, (0, second, 0)):
        expected = reference.hash_ids(request.token_ids, request.start, request_id=request.request_id)
        torch.testing.assert_close(rows[request.partition, :, start:start + len(request.token_ids)],
                                   expected.to(torch.int32), atol=0, rtol=0)
    assert (rows[0, :, second + 1:] == -1).all()
    assert (rows[1, :, 1:] == -1).all()
    if row_stride is not None:
        assert (rows[0, :, 2:second] == -1).all()
    continuation = RequestSlice("a", 0, 0, 2, (3,), 2, {})
    resources.stage_ids(target, ForwardStep("decode", (continuation,), 1), topology)
    expected = reference.hash_ids((3,), 2, request_id="a")
    torch.testing.assert_close(rows[0, :, :1], expected.to(torch.int32), atol=0, rtol=0)
    assert (rows[0, :, 1:] == -1).all()
    assert (rows[1] == -1).all()
    resources.release_request("a")
    assert "a" not in resources.host.history
