import ast
import functools
import inspect
import logging
import math
import textwrap
from types import MethodType

from comfy.ldm.minimax.model import MLP


CHUNK_TOKENS = 16384


def make_chunked_forward(original_forward, chunk_tokens=CHUNK_TOKENS):
    @functools.wraps(original_forward)
    def chunked_forward(self, x):
        token_count = math.prod(x.shape[:-1])
        if x.requires_grad or token_count <= chunk_tokens:
            return original_forward(self, x)

        tokens = x.reshape(token_count, x.shape[-1])
        part = original_forward(self, tokens[:chunk_tokens])
        output = part.new_empty((token_count, part.shape[-1]))
        output[:chunk_tokens].copy_(part)
        del part
        for start in range(chunk_tokens, token_count, chunk_tokens):
            end = min(start + chunk_tokens, token_count)
            part = original_forward(self, tokens[start:end])
            output[start:end].copy_(part)
            del part
        return output.reshape(*x.shape[:-1], output.shape[-1])

    chunked_forward._h3_memory_guard_chunk_tokens = chunk_tokens
    return chunked_forward


def is_compatible_forward(original):
    expected = ast.parse('def forward(self, x):\n    return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")')
    try:
        actual = ast.parse(textwrap.dedent(inspect.getsource(original)))
    except (OSError, TypeError, SyntaxError):
        return False
    return ast.dump(actual) == ast.dump(expected)


def configure_model(model, chunk_tokens):
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    if not is_compatible_forward(MLP.forward):
        raise ValueError("Incompatible MiniMax H3 MLP.forward; expected fc1 -> SwiGLU/fc2")
    diffusion = model.get_model_object("diffusion_model")
    patched = model.clone()
    count = 0
    for name, module in diffusion.named_modules():
        if not isinstance(module, MLP):
            continue
        path = f"diffusion_model.{name}.forward" if name else "diffusion_model.forward"
        previous = model.get_model_object(path)
        original = previous.__func__ if isinstance(previous, MethodType) else previous
        if hasattr(original, "_h3_memory_guard_chunk_tokens"):
            original = original.__wrapped__
        if not is_compatible_forward(original):
            raise ValueError(f"Incompatible MiniMax H3 FFN patch at {path}; remove the other FFN chunk node")
        patched.add_object_patch(path, MethodType(make_chunked_forward(original, chunk_tokens), module))
        count += 1
    if count == 0:
        raise ValueError("This node requires a native MiniMax H3 model")
    logging.info("[h3-memory-guard] MODEL FFN configured; chunk_tokens=%d, modules=%d", chunk_tokens, count)
    return patched
