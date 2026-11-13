import re

import pytest

from nodes.prompting import (
    build_reference_prompt,
    ensure_visual_text_policy,
    without_voice_references,
)


def test_ref2va_pictures_are_generic_references_until_prompt_assigns_timing():
    prompt = build_reference_prompt(
        mode="REF2VA",
        detail="让 <Picture 2> 作为 00:00 首帧，<Picture 1> 作为角色参考。",
        duration=5,
        has_first=True,
        has_last=True,
        extra_reference_count=1,
        has_audio=False,
    )

    assert "<Picture 1>" in prompt and "<Picture 2>" in prompt and "<Picture 3>" in prompt
    assert "do not auto-assign first or last frame" in prompt
    assert "<Picture 1> 是 00:00" not in prompt
    assert "<Picture 2> 是 00:05" not in prompt
    assert "Specify <Picture N> as the 00:00 first frame or ending frame" in prompt


def test_reference_prompt_uses_only_reference_audio_semantics():
    prompt = build_reference_prompt(
        mode="I2VA",
        detail="角色说：你好。",
        duration=5,
        has_first=True,
        has_last=False,
        has_audio=True,
    )

    assert "<Audio 1>" in prompt
    assert "reference" in prompt
    assert "音色与表达方式" in prompt
    assert "fully_copy" not in prompt
    assert "partially_copy" not in prompt
    assert "audio reuse" not in prompt


def test_fl2va_reference_prompt_labels_both_endpoint_images():
    prompt = build_reference_prompt(
        mode="FL2VA",
        detail="人物从室内走向门外。",
        duration=5,
        has_first=True,
        has_last=True,
        has_audio=False,
    )

    assert "<Picture 1>" in prompt
    assert "00:00" in prompt
    assert "<Picture 2>" in prompt
    assert "00:05" in prompt
    assert "提示词约束" in prompt


def test_last_only_reference_uses_picture_one():
    prompt = build_reference_prompt(
        mode="L2VA",
        detail="画面结束在雪山。",
        duration=8,
        has_first=False,
        has_last=True,
        has_audio=False,
    )

    assert "<Picture 1>" in prompt
    assert "00:08" in prompt
    assert "<Picture 2>" not in prompt


def test_reference_prompt_numbers_multiple_audio_samples():
    prompt = build_reference_prompt(
        mode="REF2VA",
        detail="两个人物分别说话。",
        duration=5,
        has_audio=True,
        audio_count=2,
    )

    assert "<Audio 1>" in prompt
    assert "<Audio 2>" in prompt
    assert "音色参考 1" in prompt
    assert "音色参考 2" in prompt
    assert "<Audio 3>" not in prompt


def test_reference_prompt_explicitly_keeps_spoken_dialogue_out_of_video_pixels():
    prompt = build_reference_prompt(
        mode="REF2VA",
        detail="人物使用 <Audio 1> 的音色说：<d>[Chinese] 我还要睡觉。</d>",
        duration=5,
        has_audio=True,
        audio_count=1,
    )

    assert "visual_text_policy:" in prompt
    assert "Speech is audio-only" in prompt
    assert "subtitles" in prompt.lower()
    assert "speech bubbles" in prompt.lower()
    assert "reference images or storyboards" in prompt


def test_structured_reference_prompt_also_receives_visible_text_policy():
    detail = (
        "subject_definitions:\n"
        "<Audio 1> is the voice-timbre reference for the woman.\n\n"
        "summary:\n"
        "[reference generation + audio reference] one shot.\n\n"
        "detailed_description:\n"
        "The woman speaks one short line.\n"
    )

    prompt = build_reference_prompt(
        mode="REF2VA",
        detail=detail,
        duration=5,
        has_audio=True,
        audio_count=1,
    )

    assert "visual_text_policy:" in prompt
    assert "Speech is audio-only" in prompt


def test_reference_prompt_can_lock_male_voice_register():
    prompt = build_reference_prompt(
        mode="REF2VA",
        detail="橘总说新的中文台词。",
        duration=5,
        has_audio=True,
        audio_count=1,
        audio_names=("橘总",),
        voice_gender="male",
    )

    assert "male" in prompt.lower()
    assert "preserve the reference speaker's gender" in prompt.lower()
    assert "do not feminize" in prompt.lower()


def test_structured_prompt_passes_through_without_rewrapping():
    detail = (
        "subject_definitions:\n"
        "<Subject 1> 橘总 is the cat from <Picture 2>.\n"
        "<Audio 1> is the voice-timbre reference for 橘总; transfer timbre only.\n\n"
        "summary:\n"
        "[reference generation + audio reference] custom authored content.\n\n"
        "detailed_description:\n"
        "橘总 says a line.\n"
    )

    prompt = build_reference_prompt(
        mode="REF2VA",
        detail=detail,
        duration=5,
        has_first=True,
        has_last=True,
        has_audio=True,
        extra_reference_count=3,
        audio_count=1,
        audio_names=("橘总",),
    )

    assert detail.strip() in prompt
    assert prompt.count("subject_definitions:") == 1
    assert "generic reference image" not in prompt
    assert "visual_text_policy:" in prompt


def test_structured_prompt_applies_explicit_female_voice_constraint():
    detail = (
        "subject_definitions:\n"
        "<Audio 1> is the voice-timbre reference for 南宫婉.\n"
        "<Audio 2> is the voice-timbre reference for 南宫阙.\n\n"
        "summary:\n"
        "[reference generation + audio reference] two speakers.\n\n"
        "detailed_description:\n"
        "南宫婉 speaks first, then 南宫阙.\n"
    )

    prompt = build_reference_prompt(
        mode="REF2VA",
        detail=detail,
        duration=5,
        has_audio=True,
        audio_count=2,
        voice_gender="female",
    )

    assert "Preserve every referenced speaker's gender as female" in prompt
    assert "do not masculinize the voice" in prompt


def test_structured_prompt_fills_only_missing_audio_binding():
    detail = (
        "subject_definitions:\n"
        "<Subject 1> 橘总 is the cat from <Picture 1>.\n\n"
        "detailed_description:\n"
        "橘总 says a line.\n"
    )

    prompt = build_reference_prompt(
        mode="REF2VA",
        detail=detail,
        duration=5,
        has_first=True,
        has_audio=True,
        audio_count=1,
        audio_names=("橘总",),
    )

    assert detail.strip() in prompt
    assert prompt.count("subject_definitions:") == 1
    assert "audio_reference_notes:" in prompt
    assert '<Audio 1> is the voice-timbre and delivery reference for role “橘总”' in prompt


def test_reference_transcript_is_injected_for_timbre_alignment():
    prompt = build_reference_prompt(
        mode="REF2VA",
        detail="橘总说新的中文台词。",
        duration=5,
        has_audio=True,
        audio_count=1,
        audio_names=("橘总",),
        audio_transcripts=("江湖规矩，先来后到。",),
    )

    assert "样本原文是“江湖规矩，先来后到。”" in prompt
    assert "仅用于音色对齐" in prompt


def test_structured_prompt_keeps_author_text_and_appends_transcript():
    detail = (
        "subject_definitions:\n"
        "<Audio 1> is the voice-timbre reference for 橘总.\n\n"
        "detailed_description:\n"
        "橘总 says a line.\n"
    )

    prompt = build_reference_prompt(
        mode="REF2VA",
        detail=detail,
        duration=5,
        has_audio=True,
        audio_count=1,
        audio_transcripts=("江湖规矩，先来后到。",),
    )

    assert detail.strip() in prompt
    assert '<Audio 1> sample transcript: “江湖规矩，先来后到。”' in prompt
    assert prompt.count("is the voice-timbre and delivery reference") == 0  # binding already present


@pytest.mark.parametrize("voice_reference", [True, False])
def test_structured_ref2va_keeps_mandarin_only_in_authored_dialogue(voice_reference):
    dialogue = "<d>先把你留下。</d>"
    detail = (
        "subject_definitions:\n"
        "Nangong Wan <Subject 1> is the woman from <Picture 1>. "
        "<Audio 1> is her voice reference only.\n\n"
        "summary:\nOne continuous fifteen-second office shot.\n\n"
        "retention_analysis:\nPreserve identity and the original three swatches.\n\n"
        "detailed_description:\n"
        f"She places the blue-gray swatch, says {dialogue}, then rests quietly.\n\n"
        "overall_soundscape:\nQuiet office ambience and the one authored line.\n\n"
        "non_diegetic_music:\nNone."
    )
    routed_detail = detail if voice_reference else without_voice_references(detail)

    prompt = ensure_visual_text_policy(build_reference_prompt(
        mode="REF2VA", detail=routed_detail, duration=15,
        has_audio=voice_reference, audio_count=int(voice_reference),
    ))

    assert prompt.startswith(routed_detail)
    assert prompt.count(dialogue) == 1
    assert ("<Audio 1>" in prompt) is voice_reference
    assert not re.search(r"[\u4e00-\u9fff]", prompt.replace(dialogue, ""))
    assert prompt.count("subject_definitions:") == 1
    assert prompt.count("visual_text_policy:") == 1
    assert "Speech is audio-only" in prompt
    assert "no subtitles, captions, transcript text, speech bubbles" in prompt
    assert ensure_visual_text_policy(prompt) == prompt


def test_shared_visual_text_policy_uses_english_control_prose():
    prompt = ensure_visual_text_policy("One silent office shot.")

    assert prompt.startswith("One silent office shot.\n\nvisual_text_policy:")
    assert not re.search(r"[\u4e00-\u9fff]", prompt)
    assert "only when the user explicitly requests it" in prompt
    assert "Speech is audio-only" in prompt
    assert "no subtitles, captions, transcript text, speech bubbles" in prompt
    assert "reference images or storyboards" in prompt
    assert "lower third and margins" in prompt


def test_structured_audio_notes_use_english_controls_and_keep_literal_data_and_slot_order():
    detail = (
        "subject_definitions:\n"
        "<Audio 1> is the woman's voice reference only.\n\n"
        "detailed_description:\nThe woman speaks, then the man answers."
    )
    name = " 南宫阙 / Nangong Que "
    transcripts = (" 我是南宫婉。 ", "江湖规矩， 先来后到。")

    prompt = ensure_visual_text_policy(build_reference_prompt(
        mode="REF2VA", detail=detail, duration=15,
        has_audio=True, audio_count=3, audio_names=("Nangong Wan", name),
        audio_transcripts=transcripts,
    ))

    assert prompt.startswith(detail)
    notes = prompt.split("audio_reference_notes:\n", 1)[1].split("\n\nvisual_text_policy:", 1)[0]
    lines = notes.splitlines()
    assert lines[0].startswith("<Audio 2>")
    assert lines[1].startswith("<Audio 3>")
    assert f'for role “{name}”' in lines[0]
    assert "for speaker 3 in the prompt" in lines[1]
    assert f'<Audio 1> sample transcript: “{transcripts[0]}”' in lines[2]
    assert f'<Audio 2> sample transcript: “{transcripts[1]}”' in lines[3]
    assert "for timbre alignment only; do not speak or copy this transcript" in notes
    assert "do not copy the original audio signal" in notes
    assert "<Audio 4>" not in prompt
    assert "<d>" not in notes
    controls = notes
    for literal in (name, *transcripts):
        controls = controls.replace(literal, "")
    assert not re.search(r"[\u4e00-\u9fff]", controls)


def test_structured_prompt_keeps_custom_visual_policy_verbatim_and_idempotent():
    detail = (
        "subject_definitions:\nOne woman.\n\n"
        "detailed_description:\n作者原文：<d>你好。</d>\n\n"
        "visual_text_policy:\n仅保留用户明确指定的门牌文字。"
    )

    prompt = build_reference_prompt(mode="REF2VA", detail=detail, duration=15)

    assert prompt == detail
    assert ensure_visual_text_policy(prompt) == detail
    assert ensure_visual_text_policy(ensure_visual_text_policy(prompt)) == detail
