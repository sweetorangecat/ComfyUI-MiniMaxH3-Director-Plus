from pathlib import Path


from nodes.two_stage_assets import (
    FL_STAGE1_LORA,
    FL_STAGE2_LORA,
    LATENT_UPSCALER_MODEL,
    REF_STAGE_LORA,
    dependency_report,
    resolve_two_stage_route,
)


def _touch(root: Path, relative: str):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_route_is_locked_to_backend_and_excludes_fish():
    assert resolve_two_stage_route(
        {"resolved_backend": "fl2va_model", "voice_mode": "none"}
    ) == "trained_latent_fl"
    assert resolve_two_stage_route(
        {"resolved_backend": "ref2va_model", "voice_mode": "h3_reference"}
    ) == "trained_latent_ref"
    assert resolve_two_stage_route(
        {"resolved_backend": "ref2va_model", "voice_mode": "fish_lock"}
    ) == "bypass"


def test_fl_dependency_report_requires_only_fl_assets(tmp_path):
    report = dependency_report(
        tmp_path,
        "trained_latent_fl",
        node_mappings={},
    )

    assert report["ready"] is False
    assert report["missing"] == [
        "MinimaxH3LatentUpscaler3D",
        LATENT_UPSCALER_MODEL,
        FL_STAGE1_LORA,
        FL_STAGE2_LORA,
    ]
    assert REF_STAGE_LORA not in report["missing"]
    assert "H3SigmaRefiner" not in report["missing"]


def test_reference_dependency_report_requires_refiner_and_ref_lora(tmp_path):
    report = dependency_report(
        tmp_path,
        "trained_latent_ref",
        node_mappings={},
    )

    assert report["ready"] is False
    assert report["missing"] == [
        "MinimaxH3LatentUpscaler3D",
        "H3SigmaRefiner",
        LATENT_UPSCALER_MODEL,
        REF_STAGE_LORA,
    ]
    assert FL_STAGE1_LORA not in report["missing"]
    assert FL_STAGE2_LORA not in report["missing"]


def test_installed_legacy_upscaler_id_is_accepted(tmp_path):
    _touch(tmp_path, f"models/latent_upscale_models/{LATENT_UPSCALER_MODEL}")
    _touch(tmp_path, f"models/loras/{FL_STAGE1_LORA}")
    _touch(tmp_path, f"models/loras/{FL_STAGE2_LORA}")

    report = dependency_report(
        tmp_path,
        "trained_latent_fl",
        node_mappings={"MinimaxH3LatentUpscalerNode3D": object()},
    )

    assert report["ready"] is True
    assert report["missing"] == []
    assert report["upscaler_node_id"] == "MinimaxH3LatentUpscalerNode3D"


def test_bypass_route_has_no_required_assets(tmp_path):
    report = dependency_report(tmp_path, "bypass", node_mappings={})
    assert report["ready"] is True
    assert report["missing"] == []
