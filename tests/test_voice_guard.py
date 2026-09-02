import pytest
import torch

from nodes.voice_guard import analyze_voice_reference


def _audio(seconds, amplitude=0.3, sample_rate=32000, channels=1):
    length = int(seconds * sample_rate)
    waveform = torch.full((1, channels, length), float(amplitude))
    return {"waveform": waveform, "sample_rate": sample_rate}


def test_clean_six_second_sample_passes_without_warnings():
    report = analyze_voice_reference(_audio(6.0))

    assert report["errors"] == []
    assert report["warnings"] == []
    assert report["duration"] == pytest.approx(6.0)


def test_sub_two_second_sample_is_rejected_per_official_limit():
    report = analyze_voice_reference(_audio(1.0))

    assert any("官方 2 秒下限" in error for error in report["errors"])


def test_over_fifteen_second_sample_is_rejected_per_official_limit():
    report = analyze_voice_reference(_audio(16.0))

    assert any("15 秒上限" in error for error in report["errors"])


def test_two_to_five_second_sample_warns_about_recommended_window():
    report = analyze_voice_reference(_audio(3.0))

    assert report["errors"] == []
    assert any("5–10 秒" in warning for warning in report["warnings"])


def test_mostly_silent_sample_warns():
    report = analyze_voice_reference(_audio(6.0, amplitude=0.0))

    assert any("静音占比" in warning for warning in report["warnings"])
    assert any("音量过低" in warning for warning in report["warnings"])


def test_clipped_sample_warns():
    audio = _audio(6.0, amplitude=0.3)
    audio["waveform"][0, 0, : 32000 // 10] = 1.0  # 10% hard clipping

    report = analyze_voice_reference(audio)

    assert any("削波" in warning for warning in report["warnings"])


def test_stereo_sample_is_measured_after_mixdown():
    report = analyze_voice_reference(_audio(6.0, channels=2))

    assert report["errors"] == []
    assert report["warnings"] == []


def test_unmeasurable_placeholder_is_skipped_silently():
    report = analyze_voice_reference({"waveform": object(), "sample_rate": 32000})

    assert report == {"errors": [], "warnings": [], "duration": None}
