"""Patch a versioned ComfyUI API prompt template."""

from __future__ import annotations

from copy import deepcopy

try:
    from ..nodes.performance import PRESET_LABELS, preset_values
    from ..nodes.schema import allowed_performance_presets
except ImportError:
    from nodes.performance import PRESET_LABELS, preset_values
    from nodes.schema import allowed_performance_presets


CONTROLLER_TITLE = "快速设置 / API 入参"
ASSET_TITLES = {
    "first_image": "API 首帧图片",
    "last_image": "API 尾帧图片",
}
VOICE_TITLES = ("API 音色参考音频1", "API 音色参考音频2", "API 音色参考音频3")


class TemplateError(ValueError):
    pass


def _node_by_title(prompt, title):
    for node_id, node in prompt.items():
        if node.get("_meta", {}).get("title") == title:
            return str(node_id), node
    return None, None


def patch_template(template, request):
    prompt = deepcopy(template)
    controller_id, controller = _node_by_title(prompt, CONTROLLER_TITLE)
    if controller is None or controller.get("class_type") != "MiniMaxH3DirectorPlus":
        raise TemplateError("API 模板缺少‘快速设置 / API 入参’控制节点")

    controller_inputs = controller.setdefault("inputs", {})
    public_controller_fields = (
        "mode", "prompt", "duration", "width", "height", "voice_mode",
        "aspect_ratio", "resolution_preset", "custom_width", "custom_height",
        "ref_image_size", "performance_preset", "target_dialogue", "reference_transcript",
        "fish_model_path", "postprocess_mode", "rtx_quality", "ai_upscale_model",
    )
    for name in public_controller_fields:
        if name in request:
            controller_inputs[name] = request[name]
    voice_names = list(request.get("voice_reference_names") or [])
    for index in range(1, 4):
        controller_inputs[f"voice_reference_name_{index}"] = voice_names[index - 1] if index <= len(voice_names) else ""

    for field, title in ASSET_TITLES.items():
        loader_id, loader = _node_by_title(prompt, title)
        value = request.get(field)
        if value:
            if loader is None:
                raise TemplateError(f"API 模板缺少素材节点：{title}")
            input_name = "audio" if field == "voice_reference_audio" else "image"
            loader.setdefault("inputs", {})[input_name] = value
            controller_inputs[field] = [loader_id, 0]
        else:
            controller_inputs.pop(field, None)
            if loader_id is not None:
                prompt.pop(loader_id, None)

    audio_references = list(request.get("voice_reference_audios") or [])
    if request.get("voice_reference_audio") and not audio_references:
        audio_references = [request["voice_reference_audio"]]
    for index, title in enumerate(VOICE_TITLES, 1):
        field = "voice_reference_audio" if index == 1 else f"voice_reference_audio_{index}"
        loader_id, loader = _node_by_title(prompt, title)
        if index == 1 and loader is None:
            loader_id, loader = _node_by_title(prompt, "API 音色参考音频")
        value = audio_references[index - 1] if index <= len(audio_references) else None
        if value:
            if loader is None:
                raise TemplateError(f"API 模板缺少素材节点：{title}")
            loader.setdefault("inputs", {})["audio"] = value
            controller_inputs[field] = [loader_id, 0]
        else:
            controller_inputs.pop(field, None)
            if loader_id is not None:
                prompt.pop(loader_id, None)

    bridge_id, bridge = _node_by_title(prompt, "API Fish S2 音色桥接")
    guide_id, guide = _node_by_title(prompt, "API H3 原生指南")
    if request.get("voice_mode") == "fish_lock":
        if bridge is None or guide is None:
            raise TemplateError("API 模板缺少 Fish S2 音色桥接")
        first_voice_id, _ = _node_by_title(prompt, VOICE_TITLES[0])
        if first_voice_id is None:
            raise TemplateError("Fish S2 需要音色参考音频1")
        bridge.setdefault("inputs", {})["guide"] = [controller_id, 0]
        bridge["inputs"]["reference_audio"] = [first_voice_id, 0]
        guide.setdefault("inputs", {})["generated_voice_audio"] = [bridge_id, 0]
    else:
        if bridge_id is not None:
            prompt.pop(bridge_id, None)
        if guide is not None:
            guide.setdefault("inputs", {}).pop("generated_voice_audio", None)

    references = list(request.get("references") or [])
    for index in range(1, 10):
        field = f"reference_image_{index}"
        title = f"API 参考图{index}"
        loader_id, loader = _node_by_title(prompt, title)
        value = references[index - 1] if index <= len(references) else None
        if value:
            if loader is None:
                raise TemplateError(f"API 模板缺少素材节点：{title}")
            loader.setdefault("inputs", {})["image"] = value
            controller_inputs[field] = [loader_id, 0]
        else:
            controller_inputs.pop(field, None)
            if loader_id is not None:
                prompt.pop(loader_id, None)

    seed_node_id, seed_node = _node_by_title(prompt, "API 随机种子")
    if seed_node is not None and "seed" in request:
        seed_node.setdefault("inputs", {})["noise_seed"] = int(request["seed"])
    _, scheduler_node = _node_by_title(prompt, "API 调度器")
    if scheduler_node is not None and "performance_preset" in request:
        backend = "ref2va_model" if request.get("mode") == "REF2VA" or request.get("voice_mode") != "none" else "fl2va_model"
        requested_preset = request["performance_preset"]
        preset_name = PRESET_LABELS.get(requested_preset, requested_preset)
        if (
            request.get("mode")
            and preset_name != "custom"
            and preset_name not in allowed_performance_presets(request["mode"], request.get("voice_mode", "none"))
        ):
            requested_preset = "quality"
        steps_input = scheduler_node.setdefault("inputs", {}).get("steps")
        # The generated API template routes steps through PerformancePreset so
        # an acceleration failure can downgrade 4 steps to a safe count. Keep
        # a literal-step fallback for older user-supplied templates.
        if not (isinstance(steps_input, list) and len(steps_input) == 2):
            scheduler_node["inputs"]["steps"] = preset_values(requested_preset, backend=backend)["steps"]

    return prompt
