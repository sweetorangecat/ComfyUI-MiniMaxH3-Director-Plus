"""Optional adapter for the third-party Viggle H3 action-replacement nodes.

The adapter deliberately stops at ``VIGGLE_COND_SET``.  Viggle's chunked
sampler and model have a different type contract from native H3 REF2VA, so
pretending they are ordinary H3 conditioning would produce invalid workflows.
"""

from __future__ import annotations


def _viggle_class(name):
    try:
        import nodes as comfy_nodes

        mapping = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})
        cls = mapping.get(name)
        if cls is None:
            raise KeyError(name)
        return cls
    except (ImportError, KeyError, AttributeError) as exc:
        raise RuntimeError(
            "动作替换需要安装并启用 ComfyUI-Viggle-Animate-H3 "
            "（节点包 comfyui-viggle-animate-h3）"
        ) from exc


def _execute_node(name, **kwargs):
    cls = _viggle_class(name)
    runner = getattr(cls, "execute", None)
    if runner is None:
        runner = getattr(cls(), "execute", None)
    if runner is None:
        raise RuntimeError(f"Viggle 节点 {name} 没有可执行接口，请更新节点包")
    try:
        return runner(**kwargs)
    except TypeError as exc:
        raise RuntimeError(f"Viggle 节点 {name} 输入契约不匹配，请更新节点包：{exc}") from exc


class MiniMaxH3ActionReplacementConditioning:
    """Convert source motion frames and a target identity image to Viggle H3 conditions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cond_video": ("IMAGE", {"tooltip": "原视频解码后的动作帧"}),
                "ref_image": ("IMAGE", {"tooltip": "目标人物身份/外观参考图"}),
                "text_cond": ("TEXT_COND",),
                "vae": ("VAE",),
                "anchor_mode": (["five_frame_anchor", "first_frame"], {"default": "five_frame_anchor"}),
            },
            "optional": {
                "source_audio": ("AUDIO", {"tooltip": "原视频音轨，原样传给下游合成"}),
                "director_guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",),
            },
        }

    RETURN_TYPES = ("VIGGLE_COND_SET", "AUDIO", "STRING")
    RETURN_NAMES = ("Viggle动作条件", "原视频音轨", "状态")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3 导演台 Plus/动作替换"

    def build(
        self,
        cond_video,
        ref_image,
        text_cond,
        vae,
        anchor_mode="five_frame_anchor",
        source_audio=None,
        director_guide=None,
    ):
        if director_guide and director_guide.get("mode") not in {None, "ACTION_REPLACE"}:
            raise ValueError("动作替换适配节点只能连接 ACTION_REPLACE 导演指南")
        result = _execute_node(
            "ViggleAnimateConditioningWindowed",
            cond_video=cond_video,
            ref_image=ref_image,
            text_cond=text_cond,
            vae=vae,
            anchor_mode=anchor_mode,
        )
        unpacked = getattr(result, "result", result)
        if isinstance(unpacked, dict):
            cond_set = unpacked.get("cond_set") or unpacked.get("conditioning")
        elif isinstance(unpacked, (tuple, list)):
            cond_set = unpacked[0] if unpacked else None
        else:
            cond_set = unpacked
        if cond_set is None:
            raise RuntimeError("Viggle 条件节点没有返回 VIGGLE_COND_SET")
        status = "动作替换条件已生成；原视频音轨默认保留"
        return (cond_set, source_audio, status)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ActionReplacementConditioning": MiniMaxH3ActionReplacementConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ActionReplacementConditioning": "H3 动作替换条件适配（Viggle）",
}
