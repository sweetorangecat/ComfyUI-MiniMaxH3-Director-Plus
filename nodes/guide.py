"""Thin adapters to native MiniMax H3 conditioning nodes."""

from __future__ import annotations

from copy import copy


def native_node(name):
    try:
        from comfy_extras import nodes_minimax_h3
        return getattr(nodes_minimax_h3, name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"缺少 ComfyUI 原生节点 {name}，请更新 ComfyUI 后重试。"
        ) from exc


class MiniMaxH3DirectorPlusGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",),
            },
            "optional": {
                "generated_voice_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("正向条件", "初始Latent")
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3 导演台 Plus"

    def apply(self, clip, video_vae, audio_vae, guide, generated_voice_audio=None):
        state = copy(guide)
        if state.get("voice_mode") == "fish_lock":
            if generated_voice_audio is None:
                raise ValueError("Fish 高级音色锁定尚未生成目标对白音频")
            state["ref_audios"] = {"ref_audio_1": generated_voice_audio}

        if state["resolved_backend"] == "fl2va_model":
            return native_node("MiniMaxH3ImageToVideo").execute(
                clip,
                video_vae,
                state["prompt"],
                state["width"],
                state["height"],
                state["length"],
                state.get("first_frame"),
                state.get("last_frame"),
            )

        return native_node("MiniMaxH3ReferenceToVideo").execute(
            clip,
            video_vae,
            audio_vae,
            state["prompt"],
            state["width"],
            state["height"],
            state["length"],
            state.get("ref_image_size", "match"),
            state.get("ref_images", {}),
            state.get("ref_videos", {}),
            state.get("ref_video_audios", {}),
            state.get("ref_audios", {}),
        )


class MiniMaxH3ModelRouter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",)},
            "optional": {
                "fl2va_model": ("MODEL", {"lazy": True}),
                "ref2va_model": ("MODEL", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("当前模型",)
    FUNCTION = "select"
    CATEGORY = "MiniMax H3 导演台 Plus"

    def check_lazy_status(self, guide, fl2va_model=None, ref2va_model=None):
        selected = "ref2va_model" if guide["resolved_backend"] == "ref2va_model" else "fl2va_model"
        return [selected] if locals()[selected] is None else []

    def select(self, guide, fl2va_model=None, ref2va_model=None):
        selected = ref2va_model if guide["resolved_backend"] == "ref2va_model" else fl2va_model
        if selected is None:
            label = "REF2VA" if guide["resolved_backend"] == "ref2va_model" else "FL2VA"
            raise ValueError(f"缺少 {label} 模型连接")
        return (selected,)
