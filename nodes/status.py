"""Local capability reporting for Director Plus."""

from __future__ import annotations

import json
import importlib
from pathlib import Path


def _directory_names(path):
    if not path.is_dir():
        return set()
    return {item.name.lower() for item in path.iterdir() if item.is_dir()}


def _has_matching_file(path, *needles):
    if not path.is_dir():
        return False
    lowered = tuple(value.lower() for value in needles)
    return any(
        all(value in item.name.lower() for value in lowered)
        for item in path.rglob("*")
        if item.is_file()
    )


def _rtx_vsr_status():
    """Probe the active ComfyUI Python environment without loading a model."""
    try:
        module = importlib.import_module("nvvfx")
    except (ImportError, ModuleNotFoundError) as exc:
        return {
            "node_available": True,
            "dependency_available": False,
            "message": f"缺少 nvidia-vfx/nvvfx，请在当前 ComfyUI Python 环境安装后重启：{exc}",
        }
    video_super_res = getattr(module, "VideoSuperRes", None)
    if video_super_res is None:
        return {
            "node_available": True,
            "dependency_available": False,
            "message": "已找到 nvvfx，但缺少 VideoSuperRes；请检查 NVIDIA Broadcast SDK 与 DaSiWa 依赖。",
        }
    return {
        "node_available": True,
        "dependency_available": True,
        "message": "RTX VSR 依赖已就绪",
    }


def detect_capabilities(comfy_root):
    root = Path(comfy_root)
    models = root / "models"
    custom_nodes = root / "custom_nodes"
    node_names = _directory_names(custom_nodes)
    diffusion = models / "diffusion_models"
    loras = models / "loras"
    fish_models = models / "fishaudioS2"

    fish_node = "comfyui-fish-audio-s2" in node_names
    fish_model = any(item.is_file() for item in fish_models.rglob("*")) if fish_models.is_dir() else False

    return {
        "version": "1.0",
        "models": {
            "fl2va": _has_matching_file(diffusion, "minimax", "h3", "fl2va"),
            "ref2va": _has_matching_file(diffusion, "minimax", "h3", "ref2va"),
            "video_vae": _has_matching_file(models / "vae", "minimax", "h3", "video", "vae"),
            "audio_vae": _has_matching_file(models / "vae", "minimax", "h3", "audio", "vae"),
            "text_encoder": _has_matching_file(models / "text_encoders", "minimax", "h3"),
        },
        "acceleration": {
            "lightx2v_4step": _has_matching_file(loras, "minimax", "h3", "lightx2v"),
            "h3_turbo_4step": _has_matching_file(loras, "minimax", "h3", "turbo"),
            "gguf": "comfyui-gguf" in node_names or (models / "gguf").is_dir(),
            "solattn": any("solattn" in name for name in node_names),
            "sage_attention": any("sage" in name or "kjnodes" in name for name in node_names),
            "memory_efficient_sage": any("sage" in name or "kjnodes" in name for name in node_names),
            "easycache": any("easycache" in name or "dasiwa" in name for name in node_names),
            "first_block_cache": any("minimaxh3-firstblockcache" in name or "minimax-h3-blockcache" in name for name in node_names),
            "uniblockswap": any("uniblockswap" in name for name in node_names),
        },
        "postprocess": {
            "rife": any("frame-interpolation" in name for name in node_names),
            "ltx_upscale": "comfyui-ltxvideo" in node_names,
            "video_helper_suite": "comfyui-videohelpersuite" in node_names,
            "rtx_vsr": _rtx_vsr_status(),
        },
        "fish_s2": {
            "node_available": fish_node,
            "model_available": fish_model,
            "available": fish_node and fish_model,
        },
    }


def status_summary(status):
    fish = status["fish_s2"]
    if fish["available"]:
        fish_text = "Fish S2：节点与模型均已就绪"
    elif fish["node_available"]:
        fish_text = "Fish S2：节点已安装，模型未就绪"
    else:
        fish_text = "Fish S2：节点未安装，保持关闭即可"
    models = status["models"]
    h3_text = f"H3 模型：FL2VA {'可用' if models['fl2va'] else '缺失'} / REF2VA {'可用' if models['ref2va'] else '缺失'}"
    rtx_text = f"RTX VSR：{status['postprocess']['rtx_vsr']['message']}"
    return f"{h3_text}\n{fish_text}\n{rtx_text}"


class MiniMaxH3DirectorPlusStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("能力状态JSON", "中文状态摘要")
    FUNCTION = "report"
    CATEGORY = "MiniMax H3 导演台 Plus"

    def report(self):
        comfy_root = Path(__file__).resolve().parents[3]
        status = detect_capabilities(comfy_root)
        return json.dumps(status, ensure_ascii=False, indent=2), status_summary(status)
