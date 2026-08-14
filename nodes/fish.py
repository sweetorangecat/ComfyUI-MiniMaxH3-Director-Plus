"""Mode-aware Fish S2 voice generation for Director Plus."""

from __future__ import annotations


def fish_voice_clone_node():
    try:
        import nodes as comfy_nodes

        node_class = comfy_nodes.NODE_CLASS_MAPPINGS.get("FishS2VoiceCloneTTS")
    except (AttributeError, ImportError) as exc:
        raise RuntimeError("Fish S2 节点不可用，请安装并重启 ComfyUI-fish-audio-s2") from exc
    if node_class is None:
        raise RuntimeError("Fish S2 节点不可用，请安装并重启 ComfyUI-fish-audio-s2")
    return node_class()


class MiniMaxH3FishVoiceBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",),
                "reference_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("Fish生成的新对白",)
    FUNCTION = "generate"
    CATEGORY = "MiniMax H3 导演台 Plus"

    def generate(self, guide, reference_audio):
        if guide.get("voice_mode") != "fish_lock":
            return (None,)
        model_path = guide.get("fish_model_path", "s2-pro-w4a16 (auto download)")
        text = guide.get("target_dialogue", "")
        reference_text = guide.get("reference_transcript", "")
        if reference_audio is None:
            raise ValueError("Fish 高级音色锁定需要上传音色参考 1")
        if not str(text or "").strip():
            raise ValueError("Fish 高级音色锁定需要填写新的目标对白")
        try:
            return fish_voice_clone_node().generate(
                model_path=model_path,
                text=str(text).strip(),
                reference_audio=reference_audio,
                language="auto",
                device="auto",
                precision="auto",
                attention="auto",
                max_new_tokens=0,
                chunk_length=200,
                temperature=0.8,
                top_p=0.8,
                repetition_penalty=1.1,
                seed=0,
                keep_model_loaded=True,
                compile_model=False,
                reference_text=str(reference_text or "").strip(),
            )
        except Exception as exc:
            raise RuntimeError(f"Fish S2 生成失败，未回退到 H3 原生音色：{exc}") from exc
