import pytest

from nodes.schema import PUBLIC_API_KEYS, RequestError, normalize_request, public_schema


def test_public_schema_lists_every_public_api_key():
    assert set(PUBLIC_API_KEYS) <= set(public_schema()["properties"])


def test_public_schema_exposes_fish_model_choice():
    schema = public_schema()
    assert "fish_model_path" in PUBLIC_API_KEYS
    assert schema["properties"]["fish_model_path"]["中文名称"] == "Fish S2 模型"


class TensorLikeImage:
    def __bool__(self):
        raise RuntimeError("ambiguous tensor boolean")

    def __len__(self):
        return 1


def test_tensor_like_image_is_accepted_without_boolean_coercion():
    request = normalize_request({"mode": "I2VA", "first_image": TensorLikeImage()})
    assert request["resolved_backend"] == "fl2va_model"


def test_fl2va_without_voice_keeps_fl_backend():
    request = normalize_request({
        "mode": "FL2VA",
        "first_image": "opening.png",
        "last_image": "closing.png",
    })

    assert request["resolved_backend"] == "fl2va_model"


def test_i2va_with_h3_reference_uses_ref_backend():
    request = normalize_request({
        "mode": "I2VA",
        "first_image": "opening.png",
        "voice_mode": "h3_reference",
        "voice_reference_audio": "voice.wav",
    })

    assert request["resolved_backend"] == "ref2va_model"


def test_fl2va_with_fish_lock_uses_ref_backend():
    request = normalize_request({
        "mode": "FL2VA",
        "first_image": "opening.png",
        "last_image": "closing.png",
        "voice_mode": "fish_lock",
        "voice_reference_audio": "voice.wav",
        "target_dialogue": "我们回家。",
    })

    assert request["resolved_backend"] == "ref2va_model"


def test_fish_lock_requires_target_dialogue():
    with pytest.raises(RequestError, match="目标对白"):
        normalize_request({
            "mode": "I2VA",
            "first_image": "opening.png",
            "voice_mode": "fish_lock",
            "voice_reference_audio": "voice.wav",
        })


def test_i2va_requires_first_image():
    with pytest.raises(RequestError, match="首帧图片"):
        normalize_request({"mode": "I2VA"})


def test_reference_mode_never_accepts_copy_semantics():
    with pytest.raises(RequestError, match="音色模式"):
        normalize_request({"mode": "REF2VA", "voice_mode": "fully_copy"})


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("稳定质量", "quality"),
        ("极速4步", "fast_4step"),
        ("参考图加速", "reference_fast"),
        ("低显存", "low_vram"),
        ("自定义", "custom"),
    ],
)
def test_chinese_performance_presets_normalize_to_stable_keys(preset, expected):
    request = normalize_request({"mode": "T2VA", "performance_preset": preset})

    assert request["performance_preset"] == expected


def test_duration_uses_h3_native_four_to_fifteen_second_range():
    assert normalize_request({"mode": "T2VA", "duration": 4})["duration"] == 4
    assert normalize_request({"mode": "T2VA", "duration": 15})["duration"] == 15
    with pytest.raises(RequestError, match="4 到 15"):
        normalize_request({"mode": "T2VA", "duration": 3})


def test_fish_voice_mode_rejects_unused_secondary_reference_audio():
    with pytest.raises(RequestError, match="Fish"):
        normalize_request({
            "mode": "T2VA",
            "voice_mode": "fish_lock",
            "voice_reference_audios": ["voice-1", "voice-2"],
            "target_dialogue": "对白",
        })


def test_reference_limits_match_h3_native_caps():
    assert len(normalize_request({"mode": "REF2VA", "references": list(range(9))})["references"]) == 9
    with pytest.raises(RequestError, match="最多支持 9 张"):
        normalize_request({"mode": "REF2VA", "references": list(range(10))})
    with pytest.raises(RequestError, match="最多支持 3 路"):
        normalize_request({"mode": "REF2VA", "voice_reference_audios": list(range(4))})
