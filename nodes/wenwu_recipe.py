"""Versioned YZ金鱼 author policy, independent of Torch and model loading.

The four motion nodes in the archived graph were bypassed. Selecting combat
explicitly opts into them here. Canonical speed 1.1 and Avatar global_stream
are documented execution adaptations, not a claim of byte-identical replay.
"""
from dataclasses import dataclass
import importlib
import math


@dataclass(frozen=True)
class AuthorAsset:
    category: str
    filename: str
    strength: float | None = None


AUTHOR_RECIPE_ID = "yz-jinyu-wenwu"
AUTHOR_RECIPE_VERSION = 1
AUTHOR_LORAS = (
    AuthorAsset("loras", "minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors", .8),
    AuthorAsset("loras", "MysticXXX_MMH3-V1.safetensors", .35),
    AuthorAsset("loras", "H3_Combat_V2.safetensors", .65),
    AuthorAsset("loras", "Motion_Repair_V2.safetensors", .35),
    AuthorAsset("loras", "H3_Fight_Flow_V1.safetensors", .35),
    AuthorAsset("loras", "H3_speed_slider_1.1.safetensors", 3.0),
)
AUTHOR_UPSCALER = AuthorAsset("latent_upscale_models", "h3_upscaler_lms_v0.1.safetensors")
AUTHOR_ASSETS = (*AUTHOR_LORAS, AUTHOR_UPSCALER)


@dataclass(frozen=True)
class AuthorRecipe:
    id: str = AUTHOR_RECIPE_ID
    version: int = AUTHOR_RECIPE_VERSION
    scheduler: str = "simple"
    sampler: str = "euler"
    steps: int = 12
    transition_step: int = 6
    cfg: float = 1.0
    lowres_scale: float = .6
    video_shift: float = 12.0
    audio_shift: float = 3.0
    rho: float = 0.0
    w_min: float = .5
    w_max: float = 1.0
    assets: tuple[AuthorAsset, ...] = AUTHOR_ASSETS


AUTHOR_RECIPE = AuthorRecipe()


def is_author_recipe(settings):
    return (settings or {}).get("community_lora_profile") == "combat"


def author_recipe_metadata():
    recipe = AUTHOR_RECIPE
    return {
        "id": recipe.id, "version": recipe.version,
        "name": "文武双全（作者配方）", "scheduler": recipe.scheduler, "sampler": recipe.sampler,
        "steps": recipe.steps, "transition_step": recipe.transition_step, "cfg": recipe.cfg,
        "lowres_scale": recipe.lowres_scale, "video_shift": recipe.video_shift,
        "audio_shift": recipe.audio_shift, "rho": recipe.rho, "w_min": recipe.w_min, "w_max": recipe.w_max,
        "upscaler_model": AUTHOR_UPSCALER.filename,
        "canonical_speed_model": AUTHOR_LORAS[-1].filename,
        "canonical_speed_identity": "public_canonical_not_verified_byte_identical_to_archive",
        "execution_adaptation": "modern_avatar_native_sol_attn_global_stream_learned_only",
        "archived_highres_tiling": False, "highres_tiling_mode": "global_stream",
        "motion_nodes_archived_bypassed": True,
        "archive_sha256": "2fd8ba1bb77d29a5ed2348127982d32ffc3ac2c1db46a105fa0128cfd35535ed",
    }


def resolve_author_assets(resolver):
    """Inspect all seven exact files, retaining registered names and all gaps."""
    missing, loras = [], []
    upscaler = None
    for asset in AUTHOR_ASSETS:
        registered = resolver(asset.category, asset.filename)
        if registered is None:
            missing.append(asset.filename)
        if asset.category == "loras":
            loras.append({"name": registered, "filename": asset.filename, "strength": asset.strength})
        else:
            upscaler = registered
    return {
        "ready": not missing, "missing": missing, "loras": loras,
        "upscaler_model": upscaler,
        "required_assets": [asset.filename for asset in AUTHOR_ASSETS],
        "lora_recipe": {
            "kind": AUTHOR_RECIPE_ID, "version": AUTHOR_RECIPE_VERSION,
            "lora_name": loras[0]["name"], "lora_strength": AUTHOR_LORAS[0].strength,
            "detail_name": loras[1]["name"], "detail_strength": AUTHOR_LORAS[1].strength,
        },
    }


def require_author_assets(resolver):
    report = resolve_author_assets(resolver)
    if not report["ready"]:
        raise ValueError("文武双全作者配方缺少完整权重：" + "、".join(report["missing"]))
    return report


def validate_author_upscaler(name):
    """Allow registered subdirectories, but never a substitute learned lifter."""
    if str(name or "").replace("\\", "/").rsplit("/", 1)[-1] != AUTHOR_UPSCALER.filename:
        raise ValueError("文武双全作者配方必须使用 " + AUTHOR_UPSCALER.filename)


def validate_manual_loras(rows):
    """Reject conflicting upstream patches before V5 reconciliation can act."""
    known = {asset.filename.casefold() for asset in AUTHOR_LORAS}
    known.update({"motion_repair.safetensors", "动作i连续性修复lora.safetensors", "动作连续性修复lora.safetensors"})
    conflicts = []
    for row in rows or ():
        name = str(row.get("name", ""))
        base = name.replace("\\", "/").rsplit("/", 1)[-1].casefold()
        if (base in known or base.startswith("mysticxxx_mmh3-v5")
                or base.startswith("minimax_h3_dmad_") or base.startswith("dmad_minimax_h3_")
                or base.startswith("minimax_h3_turbo_") or base.startswith("minimax_h3_ref2v_turbo_")):
            conflicts.append(name)
    if conflicts:
        raise ValueError("文武双全作者配方与附加 LoRA 冲突，请关闭这些手动项目后重试：" + "、".join(conflicts))


def apply_author_shift(model, guide):
    """Normalize a real model clone, including a downstream saved manual shift."""
    if not is_author_recipe(guide):
        return model
    sampling = model.get_model_object("model_sampling")
    if _author_shift_matches(sampling):
        return model
    shift = importlib.import_module("comfy_extras.nodes_minimax_h3").MiniMaxH3SigmaShift
    result = shift.execute(model, AUTHOR_RECIPE.video_shift, AUTHOR_RECIPE.audio_shift)
    return getattr(result, "result", result)[0]


def _author_shift_matches(sampling):
    return all(math.isclose(float(getattr(sampling, key, float("nan"))), value)
               for key, value in (("shift", AUTHOR_RECIPE.video_shift), ("audio_shift", AUTHOR_RECIPE.audio_shift)))


def validate_author_shift(model):
    try:
        if _author_shift_matches(model.get_model_object("model_sampling")):
            return
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    raise ValueError("文武双全作者配方需要实际模型 video/audio shift=12/3")
