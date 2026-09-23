import json
import re
import pytest
from pathlib import Path

from nodes.director import MiniMaxH3DirectorPlus
from nodes import guide as guide_module
from tools.validate_workflow import validate_workflow


@pytest.mark.parametrize("mode", ["T2VA", "REF2VA"])
def test_voice_mode_none_keeps_dialogue_without_missing_audio_references(mode):
    from tools.configure_three_voice_example import PROMPT
    state, *_ = MiniMaxH3DirectorPlus().build(
        mode=mode, prompt=PROMPT, duration=15, width=1344, height=768,
        voice_mode="none", ref_image_size="match", performance_preset="稳定质量",
        timeline_data="{}", target_dialogue="", reference_transcript="",
        voice_reference_audio_file="unused-one.wav",
        voice_reference_audio_2_file="unused-two.wav",
        voice_reference_audio_3_file="unused-three.wav",
    )
    output = state["prompt"]
    assert "<Audio" not in output
    assert "voice reference" not in output
    assert "reference recording" not in output
    assert re.findall(r"<d>.*?</d>", output) == re.findall(r"<d>.*?</d>", PROMPT)
    assert re.findall(r"00:\d+\.\d+-00:\d+\.\d+", output) == re.findall(r"00:\d+\.\d+-00:\d+\.\d+", PROMPT)
    for i in range(1, 4):
        assert f"<Subject {i}> (S{i})" in output
    assert state["ref_audios"] == {}
    assert "<Audio 3>" in PROMPT


def test_shipped_example_forwards_all_three_named_audio_slots(monkeypatch):
    path = next((Path(__file__).resolve().parents[1] / "examples").glob("U11-*.json"))
    workflow = json.loads(path.read_text(encoding="utf-8"))
    validate_workflow(workflow, require_sections=True)
    director = next(n for n in workflow["nodes"] if n["type"] == "MiniMaxH3DirectorPlus")
    values = director["widgets_values_named"]
    assert values["voice_mode"] == "h3_reference"
    assert list(values.values()) == director["widgets_values"]
    prompt = values["prompt"]
    definitions, detail = prompt.split("detailed_description:")
    roles = ("橘总", "苏小满", "阿哈")
    for i, role in enumerate(roles, 1):
        assert values[f"voice_reference_name_{i}"] == role
        for section in (definitions, detail):
            assert f"<Audio {i}>" in section
            assert f"<Subject {i}>" in section
            assert f"(S{i})" in section
    intervals = re.findall(r"00:(\d+\.\d+)-00:(\d+\.\d+).*?<d>.*?</d>", detail)
    assert len(intervals) == 3
    assert all(float(a) < float(b) for a, b in intervals)
    assert all(float(left[1]) <= float(right[0]) for left, right in zip(intervals, intervals[1:]))
    assert "No dialogue" not in prompt
    audios = [{"waveform": object(), "sample_rate": 32000} for _ in range(3)]
    state, *_ = MiniMaxH3DirectorPlus().build(
        mode="REF2VA", prompt=prompt, duration=15, width=1344, height=768,
        voice_mode=values["voice_mode"], ref_image_size="match",
        performance_preset="参考高清（原生20步）", timeline_data="{}",
        target_dialogue="", reference_transcript="",
        **{f"voice_reference_name_{i}": role for i, role in enumerate(roles, 1)},
        voice_reference_audio=audios[0], voice_reference_audio_2=audios[1],
        voice_reference_audio_3=audios[2],
    )
    calls = []
    class NativeReference:
        @staticmethod
        def execute(**kwargs):
            calls.append(kwargs)
            return "conditioning", "latent"
    monkeypatch.setattr(guide_module, "native_node", lambda _: NativeReference)
    guide_module.MiniMaxH3DirectorPlusGuide().apply("clip", "video", "audio", state)
    assert list(calls[0]["ref_audios"]) == [f"ref_audio_{i}" for i in range(1, 4)]
    assert all(calls[0]["ref_audios"][f"ref_audio_{i}"] is audios[i-1] for i in range(1, 4))
    assert state["voice_reference_names"] == list(roles)
