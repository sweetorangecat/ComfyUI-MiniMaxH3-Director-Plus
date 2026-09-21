"""Optional upstream Qwen base/tail integration, isolated from video execution."""

BASE_MODEL = "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors"
TAIL_MODEL = "qwen3vl_32b_h3_generation_tail_50_63_int8_convrot.safetensors"
SYSTEM_PROMPT = "Draft a MiniMax H3 video prompt. Preserve all supplied dialogue verbatim, reference numbers, speaker bindings and timing. Do not invent dialogue. Return only the draft."


def _registry():
    import nodes as comfy_nodes
    return getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})


def _model_choices(tail=False):
    try:
        import folder_paths
        names = folder_paths.get_filename_list("text_encoders")
    except ImportError:
        names = []
    names = [name for name in names if name.lower().endswith('.safetensors') and ("generation_tail_50_63" in name.lower()) == tail
             and (tail or ("h3" in name.lower() and "qwen" in name.lower()))]
    return sorted(set(names + [TAIL_MODEL if tail else BASE_MODEL]))


def _require_model(name):
    import folder_paths
    if not folder_paths.get_full_path("text_encoders", name):
        raise FileNotFoundError(f"Missing text encoder model: {name}. Place it under ComfyUI/models/text_encoders and refresh.")


class MiniMaxH3LocalPromptDraft:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "enabled": ("BOOLEAN", {"default": False}),
            "base_model": (_model_choices(),),
            "tail_model": (_model_choices(tail=True),),
            "prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            "system_prompt": ("STRING", {"default": SYSTEM_PROMPT, "multiline": True, "dynamicPrompts": False}),
            "max_new_tokens": ("INT", {"default": 512, "min": 32, "max": 4096}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
        }, "optional": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("draft", "report", "original_prompt")
    FUNCTION = "generate"
    CATEGORY = "MiniMax H3 导演台 Plus"
    OUTPUT_NODE = True

    def generate(self, enabled, base_model, tail_model, prompt, system_prompt, max_new_tokens, seed, image=None):
        if not enabled:
            return {"ui": {"text": ["本地提示词草稿已关闭。"]},
                    "result": (prompt, "disabled: no models loaded", prompt)}
        mappings = _registry()
        generator_class = mappings.get("H3QwenVLGenerateText")
        loader_class = mappings.get("CLIPLoader")
        if generator_class is None or loader_class is None:
            raise RuntimeError("请安装 ComfyUI-H3-Qwen3VL-TextGen 并重启 ComfyUI：https://github.com/ethanfel/ComfyUI-H3-Qwen3VL-TextGen")
        if "generation_tail_50_63" in base_model.lower():
            raise ValueError("generation tail cannot be used as the base encoder")
        if "generation_tail_50_63" not in tail_model.lower():
            raise ValueError("tail_model must contain generation_tail_50_63")
        _require_model(base_model)
        _require_model(tail_model)
        # Reuse the upstream defaults and its finally-based tail cleanup.
        # Fail on a new required socket rather than guessing how to supply it.
        kwargs = {}
        supplied = {"clip", "tail_clip", "prompt", "system_prompt", "max_new_tokens", "seed"}
        for name, spec in generator_class.INPUT_TYPES().get("required", {}).items():
            if name in supplied:
                continue
            options = spec[1] if len(spec) > 1 else {}
            if "default" in options:
                kwargs[name] = options["default"]
            elif isinstance(spec[0], (list, tuple)) and spec[0]:
                kwargs[name] = spec[0][0]
            else:
                raise RuntimeError(f"Unsupported TextGen required input: {name}; update the Director Plus adapter.")
        clip, = loader_class().load_clip(clip_name=base_model, type="minimax", device="default")
        kwargs.update(clip=clip, tail_clip={"tail_name": tail_model}, prompt=prompt, sampling="deterministic",
                      system_prompt=system_prompt, max_new_tokens=max_new_tokens, seed=seed)
        if image is not None:
            kwargs["image"] = image
        generator = generator_class()
        result = getattr(generator, generator.FUNCTION)(**kwargs)
        draft, report = result[0], result[4]
        return {"ui": {"text": [draft, report]}, "result": (draft, report, prompt)}
