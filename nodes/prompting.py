"""Prompt helpers for H3 reference-only audio routing."""

from __future__ import annotations


FORBIDDEN_AUDIO_MARKERS = ("fully_" + "copy", "partially_" + "copy", "audio " + "reuse")


def is_structured_reference_prompt(detail):
    """Detect a prompt that already follows the official R2V section layout."""
    text = str(detail or "")
    return "subject_definitions:" in text and "detailed_description:" in text


def _audio_binding_line(index, subject):
    owner = f"，对应角色“{subject}”" if subject else f"，对应提示词中的音色参考 {index}"
    return f"<Audio {index}> 仅作为人物音色与表达方式的 reference{owner}，不复制原音频信号。"


def _transcript_line(index, transcript):
    return f"<Audio {index}> 的样本原文是“{transcript}”（仅用于音色对齐，不复制其内容）。"


def _augment_structured_prompt(detail, audio_count, audio_names, transcripts):
    """Pass a structured prompt through, only filling in missing audio bindings.

    Re-wrapping a professionally structured prompt would duplicate the
    subject_definitions/retention sections and dilute the author's explicit
    reference-to-role assignments, which is exactly what the official R2V
    prompt guide warns against.  Instead we keep the author's text verbatim
    and append only the pieces that are actually missing.
    """
    text = str(detail or "").strip()
    names = list(audio_names or [])
    additions = []
    for index in range(1, audio_count + 1):
        if f"<Audio {index}>" in text:
            continue
        subject = str(names[index - 1]).strip() if index <= len(names) else ""
        additions.append(_audio_binding_line(index, subject))
    for index, transcript in enumerate(transcripts, 1):
        if transcript and index <= audio_count and transcript not in text:
            additions.append(_transcript_line(index, transcript))
    if not additions:
        return text
    return text + "\n\naudio_reference_notes:\n" + "\n".join(additions)


def _timestamp(seconds):
    seconds = float(seconds)
    if seconds.is_integer():
        return f"00:{int(seconds):02d}"
    return f"00:{seconds:04.1f}"


def build_reference_prompt(
    mode,
    detail,
    duration,
    has_first=False,
    has_last=False,
    has_audio=False,
    extra_reference_count=0,
    audio_count=0,
    audio_names=None,
    voice_gender="auto",
    audio_transcripts=None,
):
    transcripts = [str(item or "").strip() for item in (audio_transcripts or [])]
    resolved_audio_count = max(int(bool(has_audio)), min(3, int(audio_count or 0)))
    if is_structured_reference_prompt(detail):
        prompt = _augment_structured_prompt(
            detail, resolved_audio_count, audio_names, transcripts
        )
        if any(marker in prompt for marker in FORBIDDEN_AUDIO_MARKERS):
            raise ValueError("提示词包含不允许的音频复制语义")
        return prompt

    definitions = []
    retention = []

    picture_number = 1
    if has_first:
        definitions.append("<Picture 1> 是 00:00 的起始画面参考。")
        retention.append("<Picture 1>: fully_preserved - 保持起始构图和人物身份。")
        picture_number = 2
    if has_last:
        definitions.append(
            f"<Picture {picture_number}> 是 {_timestamp(duration)} 的结束画面参考。"
        )
        retention.append(
            f"<Picture {picture_number}>: fully_preserved - 保持结束构图和人物身份。"
        )
        picture_number += 1
    for _ in range(max(0, int(extra_reference_count))):
        definitions.append(f"<Picture {picture_number}> 是角色或场景的额外 reference 图片。")
        retention.append(f"<Picture {picture_number}>: reference - 保持身份、服装与空间连续性。")
        picture_number += 1
    if mode == "REF2VA":
        # REF2VA has no implicit keyframe semantics: the prompt chooses which
        # numbered picture, if any, is used at 00:00 or at the ending timestamp.
        total = int(bool(has_first)) + int(bool(has_last)) + max(0, int(extra_reference_count))
        definitions = [
            f"<Picture {index}> is a generic reference image; do not auto-assign first or last frame."
            for index in range(1, total + 1)
        ]
        retention = [
            f"<Picture {index}>: reference - preserve identity, clothing, and spatial continuity."
            for index in range(1, total + 1)
        ]

    names = list(audio_names or [])
    gender = str(voice_gender or "auto").strip().lower()
    gender_instruction = {
        "male": "Preserve the reference speaker's gender (male) and vocal register; do not feminize the voice.",
        "female": "Preserve the reference speaker's gender (female) and vocal register; do not masculinize the voice.",
        "neutral": "Preserve the reference speaker's gender-neutral vocal register and timbre without introducing a gender shift.",
        "auto": "Preserve the reference speaker's apparent gender and vocal register; do not shift the voice to another gender.",
    }.get(gender, "Preserve the reference speaker's apparent gender and vocal register; do not shift the voice to another gender.")
    for index in range(1, resolved_audio_count + 1):
        subject = str(names[index - 1]).strip() if index <= len(names) else ""
        definitions.append(_audio_binding_line(index, subject))
        retention.append(f"<Audio {index}>: reference - 保持音色参考 {index} 的音色与表达方式。")
    for index, transcript in enumerate(transcripts, 1):
        if transcript and index <= resolved_audio_count:
            definitions.append(_transcript_line(index, transcript))
    if resolved_audio_count:
        retention.append(gender_instruction)

    endpoint_note = ""
    if mode == "REF2VA" and (has_first or has_last):
        endpoint_note = (
            "REF2VA does not treat upload order as first or last frame. "
            "Specify <Picture N> as the 00:00 first frame or ending frame in the prompt."
        )
    elif has_first or has_last:
        endpoint_note = (
            "这些图片是 REF2VA 的提示词约束，不是 FL2VA 硬端点；"
            "生成过程保持身份、构图和动作方向连续。"
        )

    summary_type = "[reference generation + audio reference]" if has_audio else "[reference generation]"
    prompt = "\n".join(
        [
            "subject_definitions:",
            *definitions,
            "",
            "summary:",
            f"{summary_type} 使用已标注的图片和音色参考生成 {mode} 视频。",
            "",
            "retention_analysis:",
            *retention,
            "",
            "detailed_description:",
            endpoint_note,
            str(detail or "").strip(),
            "",
            "overall_soundscape:",
            "按照画面动作生成同步环境声；音色参考只用于人物说话特征。" if has_audio else "按照画面动作生成同步环境声。",
            "",
            "non_diegetic_music:",
            "N/A",
        ]
    ).strip()
    if any(marker in prompt for marker in FORBIDDEN_AUDIO_MARKERS):
        raise ValueError("提示词包含不允许的音频复制语义")
    return prompt
