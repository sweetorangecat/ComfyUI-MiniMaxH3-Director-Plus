"""Thin adapters to native MiniMax H3 conditioning nodes."""

from __future__ import annotations

from .performance import memory_policy, preset_values

import logging

LOGGER = logging.getLogger("MiniMaxH3.DirectorPlus.Guide")


def normalize_h3_reference_audio(audio):
    """Normalize reference waveforms to the stereo contract used by H3's audio VAE."""
    if not isinstance(audio, dict):
        return audio
    waveform = audio.get("waveform")
    shape = getattr(waveform, "shape", ())
    if len(shape) != 3 or int(shape[1]) == 2:
        return audio
    if int(shape[1]) == 1:
        normalized_waveform = waveform.repeat(1, 2, 1)
    else:
        normalized_waveform = waveform.mean(dim=1, keepdim=True).repeat(1, 2, 1)
    normalized = audio.copy()
    normalized["waveform"] = normalized_waveform
    return normalized


def native_node(name):
    try:
        from comfy_extras import nodes_minimax_h3
        return getattr(nodes_minimax_h3, name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"缺少 ComfyUI 原生节点 {name}，请更新 ComfyUI 后重试。"
        ) from exc


def _route_low_vram_inputs(clip, video_vae, audio_vae, guide):
    """Keep native patcher devices; LOW_VRAM handles dynamic CPU offload."""
    preset_values(guide.get("performance_preset", "quality"))
    return clip, video_vae, audio_vae


def attach_stage2_keyframe_latents(cond, video_vae, state):
    """Re-encode anchored keyframe images on the exact second-stage pixel grid.

    官方 ImageToVideo 节点把首尾帧按第一阶段原生分辨率 VAE 编码。二采在放大后的
    网格上重采样，此前对条件 latent 做 trilinear 插值会把首尾帧条件推离 VAE
    流形——实测锚定帧（首帧）出现砖块状重影伪影，其余帧完全正常。这里用源图在
    第二阶段精确像素网格上重新 VAE 编码，保持与官方单阶段一致的流形语义；
    二采侧优先使用 ``latent_stage2``，缺失时回退插值并告警。
    """
    if not state.get("two_stage_enabled"):
        return cond
    first_frame = state.get("first_frame")
    last_frame = state.get("last_frame")
    if first_frame is None and last_frame is None:
        return cond
    width = int(state.get("second_stage_width") or 0)
    height = int(state.get("second_stage_height") or 0)
    if width < 32 or height < 32:
        return cond
    if (width, height) == (int(state.get("width") or 0), int(state.get("height") or 0)):
        return cond

    import comfy.utils

    reencoded = []
    for entry in cond or []:
        payload = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else None
        if not isinstance(payload, dict):
            continue
        keyframes = payload.get("minimax_keyframes")
        if not isinstance(keyframes, list):
            continue
        for keyframe in keyframes:
            if not isinstance(keyframe, dict) or keyframe.get("latent") is None:
                continue
            if "latent_stage2" in keyframe:
                continue
            index = int(keyframe.get("resolved_frame_index") or 0)
            image = first_frame if index == 0 else last_frame
            if image is None:
                continue
            # 官方语义：首帧几何锚点直接拉伸，尾帧跟随者等比覆盖裁剪
            crop = "disabled" if index == 0 else "center"
            resized = comfy.utils.common_upscale(
                image[:1][..., :3].movedim(-1, 1), width, height, "lanczos", crop
            ).movedim(1, -1)
            keyframe["latent_stage2"] = video_vae.encode(resized)
            reencoded.append("首帧" if index == 0 else "尾帧")
    if reencoded:
        LOGGER.info(
            "[H3 guide] 已按第二阶段网格 %sx%s 重编码锚定%s条件（官方网格语义）",
            width,
            height,
            "+".join(reencoded),
        )
    return cond


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
        state = guide.copy()
        if state.get("voice_mode") == "fish_lock":
            if generated_voice_audio is None:
                raise ValueError("Fish 高级音色锁定尚未生成目标对白音频")
            state["ref_audios"] = {"ref_audio_1": generated_voice_audio}

        clip, video_vae, audio_vae = _route_low_vram_inputs(
            clip, video_vae, audio_vae, state
        )
        with memory_policy(state):
            if state["resolved_backend"] == "fl2va_model":
                result = native_node("MiniMaxH3ImageToVideo").execute(
                    clip,
                    video_vae,
                    state["prompt"],
                    state["width"],
                    state["height"],
                    state["length"],
                    state.get("first_frame"),
                    state.get("last_frame"),
                )
                unpacked = getattr(result, "result", result)
                if isinstance(unpacked, (tuple, list)) and unpacked:
                    attach_stage2_keyframe_latents(unpacked[0], video_vae, state)
                return result

            state["ref_audios"] = {
                name: normalize_h3_reference_audio(audio)
                for name, audio in (state.get("ref_audios") or {}).items()
            }
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
