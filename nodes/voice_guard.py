"""Voice reference sample quality gate for H3 native voice cloning.

H3's official Ref2VA spec bounds each standalone reference audio clip to
2–15 seconds, and community production experience puts the stable timbre
lock window at 5–10 seconds of clean single-speaker speech.  This module
measures every loaded voice sample before H3 sampling starts, so a bad
sample fails (or warns) before the expensive generation instead of after.
"""

from __future__ import annotations

import math


H3_REF_AUDIO_MIN_SECONDS = 2.0
H3_REF_AUDIO_MAX_SECONDS = 15.0
H3_REF_AUDIO_RECOMMENDED_SECONDS = 5.0

_SILENCE_THRESHOLD = 1e-3
_CLIP_THRESHOLD = 0.999
_QUIET_DBFS = -45.0
_SILENCE_RATIO_LIMIT = 0.5
_CLIP_RATIO_LIMIT = 0.01


def analyze_voice_reference(audio, index=1):
    """Measure one loaded voice sample.

    Returns {"errors": [...], "warnings": [...], "duration": float|None}.
    Unmeasurable payloads (non-tensor waveforms, missing sample rate) are
    skipped silently so placeholder dictionaries never break the build.
    """
    label = f"音色参考 {index}"
    result = {"errors": [], "warnings": [], "duration": None}
    if not isinstance(audio, dict):
        return result
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    shape = getattr(waveform, "shape", None)
    if shape is None or len(shape) != 3 or not sample_rate:
        return result
    try:
        duration = float(shape[-1]) / float(sample_rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return result
    result["duration"] = duration

    if duration < H3_REF_AUDIO_MIN_SECONDS:
        result["errors"].append(
            f"{label} 仅 {duration:.1f} 秒，低于 H3 官方 2 秒下限，模型无法锁定音色；"
            "请提供 5–10 秒干净人声样本"
        )
    elif duration > H3_REF_AUDIO_MAX_SECONDS:
        result["errors"].append(
            f"{label} 长 {duration:.1f} 秒，超过 H3 官方单路 15 秒上限；"
            "请裁剪出 5–10 秒精华人声段"
        )
    elif duration < H3_REF_AUDIO_RECOMMENDED_SECONDS:
        result["warnings"].append(
            f"{label} 仅 {duration:.1f} 秒；H3 官方允许 2–15 秒，"
            "但实测 5–10 秒干净单人声的音色锁定最稳，过短样本容易漂移成通用音色"
        )

    try:
        mono = waveform[0]
        if int(shape[1]) > 1:
            mono = mono.mean(dim=0)
        else:
            mono = mono[0]
        magnitude = mono.abs()
        silence_ratio = float((magnitude < _SILENCE_THRESHOLD).float().mean())
        clip_ratio = float((magnitude >= _CLIP_THRESHOLD).float().mean())
        rms = float(mono.pow(2).mean().sqrt())
        dbfs = 20.0 * math.log10(max(rms, 1e-8))
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        return result

    if dbfs < _QUIET_DBFS:
        result["warnings"].append(
            f"{label} 音量过低（约 {dbfs:.0f} dBFS），音色特征不足；请提高录音电平后重录"
        )
    if silence_ratio > _SILENCE_RATIO_LIMIT:
        result["warnings"].append(
            f"{label} 静音占比 {silence_ratio:.0%}，有效人声太少；请裁剪到实际说话片段"
        )
    if clip_ratio > _CLIP_RATIO_LIMIT:
        result["warnings"].append(
            f"{label} 削波样本占比 {clip_ratio:.1%}，失真会污染音色；请用未爆音的录音"
        )
    return result
