from nodes.prompting import build_reference_prompt


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

    assert prompt == detail.strip()
    assert prompt.count("subject_definitions:") == 1
    assert "generic reference image" not in prompt


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
    assert "<Audio 1> 仅作为人物音色与表达方式的 reference，对应角色“橘总”" in prompt


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
    assert "样本原文是“江湖规矩，先来后到。”" in prompt
    assert prompt.count("<Audio 1> 仅作为") == 0  # binding already present
