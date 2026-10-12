import logging

from .chunk_ops import CHUNK_TOKENS, configure_model


class MiniMaxH3FFNChunkTokens:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "chunk_tokens": ("INT", {"default": CHUNK_TOKENS, "min": 1, "max": 2147483647, "step": 1024,
                                     "tooltip": "每块最多处理的 token 数。更小通常更省 FFN 显存；不超过此数或输入需要梯度时使用原版计算。"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "configure"
    CATEGORY = "model_patches/memory"

    def configure(self, model, chunk_tokens):
        return (configure_model(model, chunk_tokens),)


NODE_CLASS_MAPPINGS = {"MiniMaxH3FFNChunkTokens": MiniMaxH3FFNChunkTokens}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3FFNChunkTokens": "MiniMax H3 FFN 分块设置（Token）"}

logging.info("[h3-memory-guard] FFN token node loaded; default chunk_tokens=%d; global patch disabled", CHUNK_TOKENS)
