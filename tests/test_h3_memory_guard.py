import importlib.util
from pathlib import Path
import sys

import pytest
import torch


def load_chunk_ops():
    path = Path(__file__).parents[1] / "chunk_ops.py"
    spec = importlib.util.spec_from_file_location("h3_memory_guard_chunk_ops_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32769, 32), (2, 17000, 32)])
def test_chunked_core_mlp_matches_original(dtype, shape):
    from comfy.ldm.minimax.model import MLP

    ops = load_chunk_ops()
    torch.manual_seed(2718)
    model = MLP(32, 48, dtype=dtype, device="cpu", operations=torch.nn)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    x = torch.randn(shape, dtype=dtype)
    original = MLP.forward
    expected = original(model, x)
    result = ops.make_chunked_forward(original)(model, x)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert result.device == expected.device
    torch.testing.assert_close(result, expected, rtol=0.02, atol=0.002)


@pytest.mark.parametrize("shape,requires_grad", [((16384, 8), False), ((33000, 8), True)])
def test_fallback_calls_original_with_the_same_tensor(shape, requires_grad):
    ops = load_chunk_ops()
    calls = []

    def original(self, x):
        calls.append(x)
        return x

    x = torch.randn(shape, requires_grad=requires_grad)
    assert ops.make_chunked_forward(original)(None, x) is x
    assert calls == [x]


def test_noncontiguous_input_and_chunk_bound():
    ops = load_chunk_ops()
    calls = []

    def original(self, x):
        calls.append(x.shape)
        return x * 2

    x = torch.randn(17000, 2, 8).transpose(0, 1)
    result = ops.make_chunked_forward(original)(None, x)
    torch.testing.assert_close(result, x * 2)
    assert all(shape[0] <= ops.CHUNK_TOKENS for shape in calls)


def test_compatibility():
    from comfy.ldm.minimax.model import MLP

    ops = load_chunk_ops()
    assert ops.is_compatible_forward(MLP.forward)

    class DifferentMLP:
        def forward(self, x):
            return x + 1

    assert not ops.is_compatible_forward(DifferentMLP.forward)


def small_patcher():
    from comfy.ldm.minimax.model import MLP
    from comfy.model_patcher import ModelPatcher

    root = torch.nn.Module()
    root.diffusion_model = torch.nn.Module()
    root.diffusion_model.mlp = MLP(8, 12, dtype=torch.float32, device="cpu", operations=torch.nn)
    for parameter in root.parameters():
        parameter.requires_grad_(False)
    return ModelPatcher(root, torch.device("cpu"), torch.device("cpu"))


def test_model_patch_is_scoped_and_updates_without_stacking():
    from comfy.ldm.minimax.model import MLP

    ops = load_chunk_ops()
    source = small_patcher()
    native = MLP.forward
    mlp = source.model.diffusion_model.mlp
    first = ops.configure_model(source, 16384)
    second = ops.configure_model(first, 32768)
    path = "diffusion_model.mlp.forward"
    assert source.object_patches == {}
    assert mlp.forward.__func__ is native
    assert MLP.forward is native
    assert first.object_patches[path]._h3_memory_guard_chunk_tokens == 16384
    assert second.object_patches[path]._h3_memory_guard_chunk_tokens == 32768
    assert second.object_patches[path].__wrapped__ is native
    x = torch.randn(33000, 8)
    expected = native(mlp, x)
    second.patch_model(load_weights=False)
    try:
        torch.testing.assert_close(mlp(x), expected)
    finally:
        second.unpatch_model(unpatch_weights=False)
    assert mlp.forward.__func__ is native


def test_node_keeps_lora_fc1_path():
    ops = load_chunk_ops()
    source = small_patcher()
    mlp = source.model.diffusion_model.mlp
    calls = []
    handle = mlp.fc1.register_forward_hook(lambda module, args, output: calls.append(args[0].shape[0]))
    try:
        patched = ops.configure_model(source, 16384)
        patched.get_model_object("diffusion_model.mlp.forward")(torch.randn(33000, 8))
        assert calls == [16384, 16384, 232]
    finally:
        handle.remove()


def test_non_h3_model_fails_clearly():
    from comfy.model_patcher import ModelPatcher

    ops = load_chunk_ops()
    root = torch.nn.Module()
    root.diffusion_model = torch.nn.Linear(8, 8)
    source = ModelPatcher(root, torch.device("cpu"), torch.device("cpu"))
    with pytest.raises(ValueError, match="MiniMax H3"):
        ops.configure_model(source, 16384)


def test_plugin_registers_node_without_global_patch():
    from comfy.ldm.minimax.model import MLP

    native = MLP.forward
    path = Path(__file__).parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location("h3_memory_guard_plugin_test", path)
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    try:
        spec.loader.exec_module(plugin)
        node = plugin.NODE_CLASS_MAPPINGS["MiniMaxH3FFNChunkTokens"]
        assert node.INPUT_TYPES()["required"]["chunk_tokens"][1]["default"] == 16384
        assert node.RETURN_TYPES == ("MODEL",)
        assert node().configure(small_patcher(), 32768)[0].object_patches
        assert MLP.forward is native
    finally:
        for name in list(sys.modules):
            if name == spec.name or name.startswith(spec.name + "."):
                del sys.modules[name]
