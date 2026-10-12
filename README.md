# ComfyUI H3 FFN Chunk Tokens

An independent ComfyUI MODEL patch node for native MiniMax H3. It limits the
number of tokens processed in each FFN call to reduce intermediate activation
memory. This package does not include models or modify ComfyUI core files.

## Install

Copy this repository folder into `ComfyUI/custom_nodes/` and restart ComfyUI
after any running generation finishes. Search for:

**MiniMax H3 FFN 分块设置（Token）** (`MiniMaxH3FFNChunkTokens`)

No additional runtime dependencies are required beyond a compatible ComfyUI
installation with native MiniMax H3 support. Remove the earlier global
`h3-memory-guard` plugin before installing this separate package; do not install
both copies or stack this node with another FFN chunking node.

## Use

Connect `MODEL loader → LoRA / model patches → this node → guider / sampler`.
The node outputs a configured MODEL. It does not change workflows automatically.
Each model branch can use a separate node and chunk size.

- Default `chunk_tokens`: **16384**.
- This is the maximum tokens **per chunk**, not the number of chunks.
- For 50000 tokens, 16384 gives four chunks; 32768 gives two.
- Inputs at or below the limit, or with `requires_grad=True`, use the original
  forward function.
- Leading dimensions are flattened into tokens; the final feature dimension
  and output shape are preserved.
- Every chunk invokes the existing `fc1 → linear_input_act(fc2, ..., "swiglu")`
  path, including existing weight/LoRA operations.
- The node preserves weights, dtype and device; it does not change resolution,
  frame count or model precision. Floating-point rounding can still differ.
- Smaller chunks can reduce FFN peak memory but may add overhead. Larger chunks
  are not guaranteed to be faster. This does not cap total GPU memory or solve
  attention/VAE memory exhaustion.

The patch is scoped through `ModelPatcher.clone()` and `add_object_patch()`.
Loading the plugin does **not** replace the global `MLP.forward`. Applying this
node again on a branch replaces this package's previous chunk setting rather
than wrapping it twice. Bypass the node in ComfyUI to use the incoming MODEL.

## Compatibility and validation

The supported native `comfy.ldm.minimax.model.MLP.forward` is:

```python
def forward(self, x):
    return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")
```

Other implementations or incompatible FFN forward patches produce a clear
error instead of silently substituting different model math. Compatibility is
based on this implementation, not a blanket claim for all ComfyUI versions.

Development validation: ComfyUI 0.39.2, PyTorch 2.13 + CUDA 13.0,
RTX PRO 6000 Blackwell 96 GB. Fourteen CPU tests cover numeric output, dtype,
shape, non-contiguous inputs, gradient/small-input fallback, model patch/unpatch,
branch isolation, reconfiguration and node registration without a global patch.

A separate BF16 FFN microbenchmark used 32769 input tokens, hidden width 5376
and FFN width 14336. Additional allocated peak memory was approximately
3024 MiB unchunked versus 1848 MiB with 16384-token chunks. Maximum absolute
output difference was 0.00390625; mean absolute difference was 0.00012284.
These measurements were from the earlier global wrapper with the same chunk
calculation, not a full node-based video benchmark. They exclude model weights
and do not establish a speed/quality guarantee for complete workflows.
The full-video trial was interrupted before video saving; complete output
quality and comparative full-workflow speed remain unverified. No 8 GB GPU
qualification is claimed.

To run tests, use ComfyUI's Python environment with `pytest` installed and the
ComfyUI source directory on `PYTHONPATH`:

```sh
python -m pytest /path/to/this/plugin/tests -q
```

## 中文说明

安装到 `ComfyUI/custom_nodes/`，等待任务结束后重启。搜索
**MiniMax H3 FFN 分块设置（Token）**，接在模型／LoRA 后、引导器前。
默认 16384，填写的是每块最多的 token 数，不是分块数量。

这是独立节点，不会自动改工作流，也不会加载后全局修改模型。多采流程可
给各个模型分支分别设置。分块只减少 FFN 中间张量的峰值，不保证 8GB 显卡
能生成特定分辨率／时长的视频。权重、dtype、设备和输出形状保持不变，但
浮点舍入可能存在微小差异。不要与其他 FFN 分块节点叠加。

如果之前安装过全局 `h3-memory-guard`，先移除旧插件，避免加载两份节点。

## Related work

FFN chunking is an existing technique, not a new algorithm claimed by this
package. Related MiniMax H3 implementations include
[KJNodes](https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/minimax_nodes.py)
and [T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/main/docs/H3_MEMORY_NODES_EXP.md).
This plugin delegates model computation to ComfyUI and does not redistribute
those projects' implementations or model weights.

## License

MIT. Model weights and ComfyUI retain their own licenses.
