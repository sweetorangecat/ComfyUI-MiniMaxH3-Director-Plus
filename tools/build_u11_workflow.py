"""Deterministically derive the clean U11 workflow from U10."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import uuid

try:
    from .validate_workflow import validate_workflow
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validate_workflow import validate_workflow


PLUGIN_ID = "ComfyUI-MiniMaxH3-Director-Plus"


def _find_node(nodes, *, node_type=None, title=None):
    for node in nodes:
        if node_type is not None and node.get("type") == node_type:
            return node
        if title is not None and node.get("title") == title:
            return node
    return None


def _link_value(link):
    if not isinstance(link, dict):
        return link
    if "value" in link:
        return link["value"]
    if "origin_id" in link:
        return [link.get("id"), link.get("origin_id"), link.get("origin_slot"), link.get("target_id"), link.get("target_slot"), link.get("type")]
    return link


def _set_link_field(raw, index, value):
    if isinstance(raw, dict) and "origin_id" in raw:
        names = ("id", "origin_id", "origin_slot", "target_id", "target_slot", "type")
        raw[names[index]] = value
    else:
        raw[index] = value


def _new_id_factory(workflow):
    ids = [int(node["id"]) for node in workflow.get("nodes", []) if isinstance(node.get("id"), int)]
    for subgraph in workflow.get("definitions", {}).get("subgraphs", []) or []:
        ids.extend(int(node["id"]) for node in subgraph.get("nodes", []) if isinstance(node.get("id"), int))
    current = max([int(workflow.get("last_node_id", 0)), *ids], default=0)

    def allocate():
        nonlocal current
        current += 1
        workflow["last_node_id"] = current
        return current

    return allocate


def _new_link_factory(workflow):
    existing = [int(_link_value(link)[0]) for link in workflow.get("links", []) if _link_value(link)]
    current = max([int(workflow.get("last_link_id", 0)), *existing], default=0)

    def allocate():
        nonlocal current
        current += 1
        workflow["last_link_id"] = current
        return current

    return allocate


def _socket(name, data_type, link=None, widget=False, shape=None):
    value = {"name": name, "type": data_type, "link": link}
    if widget:
        value["widget"] = {"name": name}
    if shape is not None:
        value["shape"] = shape
    return value


def _output(name, data_type, links=None):
    return {"name": name, "type": data_type, "links": links}


def _properties(node_name, core=False):
    return {
        "Node name for S&R": node_name,
        "cnr_id": "comfy-core" if core else PLUGIN_ID,
        "ue_properties": {"widget_ue_connectable": {}, "version": "7.8", "input_ue_unconnectable": {}},
    }


def _replace_director(node):
    node.update({
        "type": "MiniMaxH3DirectorPlus",
        "title": "快速设置 / API 入参",
        "size": [1350, 1510],
        "flags": {"collapsed": False},
        "mode": 0,
        "inputs": [
            _socket("width", "INT", widget=True),
            _socket("height", "INT", widget=True),
            _socket("first_image_file", "COMBO", widget=True),
            _socket("last_image_file", "COMBO", widget=True),
            _socket("voice_reference_audio_file", "COMBO", widget=True),
            _socket("voice_reference_audio_2_file", "COMBO", widget=True),
            _socket("voice_reference_audio_3_file", "COMBO", widget=True),
            _socket("reference_image_1_file", "COMBO", widget=True),
            _socket("reference_image_2_file", "COMBO", widget=True),
            _socket("reference_image_3_file", "COMBO", widget=True),
            _socket("reference_image_4_file", "COMBO", widget=True),
            _socket("reference_image_5_file", "COMBO", widget=True),
            _socket("reference_image_6_file", "COMBO", widget=True),
            _socket("reference_image_7_file", "COMBO", widget=True),
            _socket("reference_image_8_file", "COMBO", widget=True),
            _socket("reference_image_9_file", "COMBO", widget=True),
        ],
        "outputs": [
            _output("导演指南", "MINIMAX_H3_DIRECTOR_PLUS_GUIDE"),
            _output("帧数", "INT"),
            _output("最终提示词", "STRING"),
            _output("实际后端", "STRING"),
            _output("警告", "STRING"),
            _output("Fish音色样本", "AUDIO"),
            _output("Fish目标对白", "STRING"),
            _output("FL2VA硬端点", "BOOLEAN"),
            _output("噪音种子", "INT"),
        ],
        "properties": _properties("MiniMaxH3DirectorPlus"),
        "widgets_values": [
            "FL2VA", "", 5, 1344, 768, "16:9", "0.83 MP", 16, 9, 0, "randomize",
            "none", "s2-pro-w4a16 (auto download)", "match", "稳定质量", "native", "HIGH", "auto",
            "{\"version\":1,\"items\":[]}", "", "", "", "", "",
            "", "", "", "", "", "", "", "", "", "", "", "", "", "",
        ],
        "color": "#24353d",
        "bgcolor": "#344b55",
    })


def _router_node(node_id, pos):
    return {
        "id": node_id, "type": "MiniMaxH3ModelRouter", "title": "模型自动路由",
        "pos": pos, "size": [270, 110], "flags": {"collapsed": True}, "order": 100,
        "mode": 0,
        "inputs": [_socket("guide", "MINIMAX_H3_DIRECTOR_PLUS_GUIDE"), _socket("fl2va_model", "MODEL", shape=7), _socket("ref2va_model", "MODEL", shape=7)],
        "outputs": [_output("当前模型", "MODEL")], "properties": _properties("MiniMaxH3ModelRouter"), "widgets_values": [],
        "color": "#24353d", "bgcolor": "#344b55",
    }


def _acceleration_node(node_id, pos):
    return {
        "id": node_id, "type": "MiniMaxH3AccelerationRouter", "title": "兼容加速路由",
        "pos": pos, "size": [300, 110], "flags": {"collapsed": True}, "order": 101, "mode": 0,
        "inputs": [_socket("model", "MODEL"), _socket("guide", "MINIMAX_H3_DIRECTOR_PLUS_GUIDE")],
        "outputs": [_output("加速后模型", "MODEL"), _output("加速说明", "STRING"), _output("加速成功", "BOOLEAN")],
        "properties": _properties("MiniMaxH3AccelerationRouter"), "widgets_values": [], "color": "#24353d", "bgcolor": "#344b55",
    }


def _performance_node(node_id, pos):
    return {
        "id": node_id, "type": "MiniMaxH3PerformancePreset", "title": "性能预设应用",
        "pos": pos, "size": [300, 140], "flags": {"collapsed": True}, "order": 102, "mode": 0,
        "inputs": [_socket("guide", "MINIMAX_H3_DIRECTOR_PLUS_GUIDE"), _socket("acceleration_ready", "BOOLEAN")],
        "outputs": [_output("采样步数", "INT"), _output("启用Sage", "BOOLEAN"), _output("启用缓存", "BOOLEAN"), _output("预设说明", "STRING")],
        "properties": _properties("MiniMaxH3PerformancePreset"), "widgets_values": [], "color": "#24353d", "bgcolor": "#344b55",
    }


def _status_node(node_id, pos):
    return {
        "id": node_id, "type": "MiniMaxH3DirectorPlusStatus", "title": "本机能力状态",
        "pos": pos, "size": [280, 100], "flags": {"collapsed": True}, "order": 103, "mode": 0,
        "inputs": [], "outputs": [_output("能力状态JSON", "STRING"), _output("中文状态摘要", "STRING")],
        "properties": _properties("MiniMaxH3DirectorPlusStatus"), "widgets_values": [], "color": "#24353d", "bgcolor": "#344b55",
    }


def _load_image(node_id, title, pos):
    return {
        "id": node_id, "type": "LoadImage", "title": title, "pos": pos, "size": [300, 330], "flags": {}, "order": 104, "mode": 0,
        "inputs": [_socket("image", "COMBO", widget=True), _socket("upload", "IMAGEUPLOAD", widget=True)],
        "outputs": [_output("IMAGE", "IMAGE"), _output("MASK", "MASK")], "properties": _properties("LoadImage", core=True),
        "widgets_values": ["", "image"],
    }


def _load_audio(node_id, title, pos):
    return {
        "id": node_id, "type": "LoadAudio", "title": title, "pos": pos, "size": [320, 180], "flags": {}, "order": 105, "mode": 0,
        "inputs": [_socket("audio", "COMBO", widget=True), _socket("audioUI", "AUDIO_UI", widget=True), _socket("upload", "AUDIOUPLOAD", widget=True)],
        "outputs": [_output("AUDIO", "AUDIO")], "properties": _properties("LoadAudio", core=True), "widgets_values": ["", None, None],
        "color": "#322", "bgcolor": "#533",
    }


def _note_node(node_id, title, pos, size, text):
    return {
        "id": node_id, "type": "MarkdownNote", "title": title, "pos": pos, "size": size,
        "flags": {}, "order": 110, "mode": 0, "inputs": [], "outputs": [],
        "properties": {"Node name for S&R": "MarkdownNote"}, "widgets_values": [text],
        "color": "#23343d", "bgcolor": "#1d2a31",
    }


def _fish_node(node_id, pos):
    return {
        "id": node_id, "type": "MiniMaxH3FishVoiceBridge", "title": "Fish S2 高级音色锁定（自动旁路）",
        "pos": pos, "size": [320, 150], "flags": {"collapsed": True}, "order": 106, "mode": 0,
        "inputs": [_socket("guide", "MINIMAX_H3_DIRECTOR_PLUS_GUIDE"), _socket("reference_audio", "AUDIO")],
        "outputs": [_output("Fish生成的新对白", "AUDIO")], "properties": _properties("MiniMaxH3FishVoiceBridge"),
        "widgets_values": [],
        "color": "#3d2d3f", "bgcolor": "#503b52",
    }


def _append_input(node, name, data_type, widget=False):
    for index, item in enumerate(node.get("inputs", [])):
        if item.get("name") == name:
            return index
    node.setdefault("inputs", []).append(_socket(name, data_type, widget=widget))
    return len(node["inputs"]) - 1


def _upgrade_stream_output_inputs(workflow, output):
    """Normalize output socket order and remap links by socket name.

    U10's combine node exposes ``images, audio, frame_rate`` while the U11
    node inserts the required ``guide`` input and follows ComfyUI's declared
    order ``images, guide, frame_rate, audio``.  Reusing the old numeric
    target slots would silently connect the FLOAT FPS stream to ``audio``.
    """
    old_inputs = list(output.get("inputs", []))
    old_names = {index: item.get("name") for index, item in enumerate(old_inputs)}
    desired = (
        ("images", "IMAGE"),
        ("guide", "MINIMAX_H3_DIRECTOR_PLUS_GUIDE"),
        ("frame_rate", "FLOAT"),
        ("audio", "AUDIO"),
    )
    existing = {item.get("name"): item for item in old_inputs}
    output["inputs"] = [
        {**existing.get(name, _socket(name, data_type)), "name": name, "type": data_type}
        for name, data_type in desired
    ]
    new_slots = {item["name"]: index for index, item in enumerate(output["inputs"])}

    rewritten = []
    for raw in workflow.get("links", []):
        link = _link_value(raw)
        if link[3] != output["id"]:
            rewritten.append(raw)
            continue
        old_name = old_names.get(int(link[4]))
        if old_name not in new_slots:
            # Drop links to sockets that do not exist on the new output node.
            continue
        _set_link_field(raw, 4, new_slots[old_name])
        rewritten.append(raw)
    workflow["links"] = rewritten


def _add_link(workflow, allocate_link, source, source_slot, target, target_slot, data_type):
    link_id = allocate_link()
    workflow.setdefault("links", []).append([link_id, source["id"], source_slot, target["id"], target_slot, data_type])
    return link_id


def _rewire_existing(workflow, predicate, *, source=None, source_slot=None, target=None, target_slot=None, data_type=None):
    for raw in workflow.get("links", []):
        link = _link_value(raw)
        if predicate(link):
            if source is not None:
                link[1] = source["id"]
            if source_slot is not None:
                link[2] = source_slot
            if target is not None:
                link[3] = target["id"]
            if target_slot is not None:
                link[4] = target_slot
            if data_type is not None:
                link[5] = data_type
            return link[0]
    return None


def _remove_links(workflow, predicate):
    workflow["links"] = [raw for raw in workflow.get("links", []) if not predicate(_link_value(raw))]


def _remove_node_inputs(workflow, node, names):
    old_inputs = list(node.get("inputs", []))
    kept_indexes = [index for index, item in enumerate(old_inputs) if item.get("name") not in names]
    index_map = {old_index: new_index for new_index, old_index in enumerate(kept_indexes)}
    rewritten = []
    for raw in workflow.get("links", []):
        link = _link_value(raw)
        if link[3] != node["id"]:
            rewritten.append(raw)
            continue
        old_slot = int(link[4])
        if old_slot not in index_map:
            continue
        _set_link_field(raw, 4, index_map[old_slot])
        rewritten.append(raw)
    workflow["links"] = rewritten
    node["inputs"] = [old_inputs[index] for index in kept_indexes]


def _remove_subgraph_inputs(subgraph, names):
    old_inputs = list(subgraph.get("inputs", []))
    kept_indexes = [index for index, item in enumerate(old_inputs) if item.get("name") not in names]
    index_map = {old_index: new_index for new_index, old_index in enumerate(kept_indexes)}
    rewritten = []
    for raw in subgraph.get("links", []):
        link = _link_value(raw)
        if link[1] != -10:
            rewritten.append(raw)
            continue
        old_slot = int(link[2])
        if old_slot not in index_map:
            continue
        _set_link_field(raw, 2, index_map[old_slot])
        rewritten.append(raw)
    subgraph["links"] = rewritten
    subgraph["inputs"] = [old_inputs[index] for index in kept_indexes]
    for item in subgraph["inputs"]:
        item["linkIds"] = []
    for raw in rewritten:
        link = _link_value(raw)
        if link[1] == -10 and 0 <= int(link[2]) < len(subgraph["inputs"]):
            subgraph["inputs"][int(link[2])]["linkIds"].append(link[0])


def _remove_subgraph_widget_values(node, subgraph, names):
    values = list(node.get("widgets_values", []))
    widget_input_count = min(len(values), len(subgraph.get("inputs", [])))
    remove_indexes = {
        index
        for index, item in enumerate(subgraph.get("inputs", [])[:widget_input_count])
        if item.get("name") in names
    }
    node["widgets_values"] = [value for index, value in enumerate(values) if index not in remove_indexes]


def _remove_legacy_resolution_calculator(subgraph):
    """Remove the disconnected DaSiWa resolution widget from the old panel."""
    legacy_ids = {
        node.get("id")
        for node in subgraph.get("nodes", [])
        if node.get("type") == "DaSiWa_ResolutionScaleCalculator"
    }
    if not legacy_ids:
        return
    subgraph["nodes"] = [node for node in subgraph.get("nodes", []) if node.get("id") not in legacy_ids]
    subgraph["links"] = [
        raw for raw in subgraph.get("links", [])
        if _link_value(raw)[1] not in legacy_ids and _link_value(raw)[3] not in legacy_ids
    ]


def _color_guard_node(node_id, pos):
    return {
        "id": node_id, "type": "MiniMaxH3ColorGuard", "title": "曝光与色彩连续性保护",
        "pos": pos, "size": [420, 220], "flags": {}, "order": 104, "mode": 0,
        "inputs": [
            _socket("images", "IMAGE"),
            _socket("guide", "MINIMAX_H3_DIRECTOR_PLUS_GUIDE"),
            _socket("enabled", "BOOLEAN", widget=True),
            _socket("strength", "FLOAT", widget=True),
        ],
        "outputs": [_output("色彩稳定帧", "IMAGE"), _output("色彩保护说明", "STRING")],
        "properties": _properties("MiniMaxH3ColorGuard"),
        "widgets_values": [True, 1.0],
        "color": "#2a363b", "bgcolor": "#3f5159",
    }


def _strip_model_path_prefixes(value):
    if isinstance(value, str):
        for prefix in ("MiniMaxH3/", "minimax/"):
            if value.startswith(prefix):
                return value[len(prefix):]
        return value
    if isinstance(value, list):
        return [_strip_model_path_prefixes(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_model_path_prefixes(item) for key, item in value.items()}
    return value


def _rebuild_socket_links(nodes, links):
    node_map = {node.get("id"): node for node in nodes}
    for node in nodes:
        for item in node.get("inputs", []):
            item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = None
    for raw in links:
        link = _link_value(raw)
        if len(link) < 6:
            continue
        source = node_map.get(link[1])
        target = node_map.get(link[3])
        if source is not None and 0 <= int(link[2]) < len(source.get("outputs", [])):
            output = source["outputs"][int(link[2])]
            output["links"] = list(output.get("links") or []) + [link[0]]
        if target is not None and 0 <= int(link[4]) < len(target.get("inputs", [])):
            target["inputs"][int(link[4])]["link"] = link[0]


def _disable_legacy_acceleration_switches(subgraph):
    """Detach obsolete frontend switches from the old, dead H3 model branches.

    U11 applies acceleration on the routed model in MiniMaxH3AccelerationRouter.
    Leaving these target links lets the frontend activate disconnected KJ nodes,
    which can execute twice or fail on direct MiniMaxH3Model wrappers.
    """
    legacy_types = {
        "EasyCache",
        "PathchSageAttentionKJ",
        "MiniMaxH3MemoryEfficientSageAttentionPatch",
    }
    legacy_ids = {
        node.get("id")
        for node in subgraph.get("nodes", [])
        if node.get("type") in legacy_types
    }
    switch_ids = {
        node.get("id")
        for node in subgraph.get("nodes", [])
        if node.get("type") == "DaSiWa_NodeStatusSwitch"
    }
    removed = set()
    kept_links = []
    for raw in subgraph.get("links", []):
        link = _link_value(raw)
        if len(link) >= 6 and link[3] in switch_ids and link[4] >= 1 and link[1] in legacy_ids:
            removed.add(link[0])
            continue
        kept_links.append(raw)
    subgraph["links"] = kept_links
    for node in subgraph.get("nodes", []):
        if node.get("type") != "DaSiWa_NodeStatusSwitch":
            continue
        for socket in node.get("inputs", []):
            if socket.get("name", "").startswith("target_") and socket.get("link") in removed:
                socket["link"] = None


def _upgrade_subgraphs(workflow):
    for subgraph in workflow.get("definitions", {}).get("subgraphs", []) or []:
        _remove_subgraph_inputs(
            subgraph,
            {"resolution_preset", "aspect_preset_when_not_image", "swap_aspect_when_not_image"},
        )
        exposed_inputs = subgraph.setdefault("inputs", [])
        generated_slot = next((index for index, item in enumerate(exposed_inputs) if item.get("name") == "generated_voice_audio"), None)
        if generated_slot is None:
            generated_slot = len(exposed_inputs)
            exposed_inputs.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{subgraph.get('id')}:generated_voice_audio")),
                "name": "generated_voice_audio",
                "type": "AUDIO",
                "linkIds": [],
                "label": "Fish生成的新对白",
            })
        next_link_id = max((_link_value(raw)[0] for raw in subgraph.get("links", [])), default=0) + 1
        for exposed in subgraph.get("inputs", []) or []:
            if exposed.get("name") == "guide":
                exposed["type"] = "MINIMAX_H3_DIRECTOR_PLUS_GUIDE"
        for node in subgraph.get("nodes", []):
            if node.get("type") not in {"MiniMaxH3DirectorGuide", "MiniMaxH3DirectorPlusGuide"}:
                continue
            node["type"] = "MiniMaxH3DirectorPlusGuide"
            node["properties"] = _properties("MiniMaxH3DirectorPlusGuide")
            old = {item.get("name"): item for item in node.get("inputs", [])}
            node["inputs"] = [
                {**old.get("clip", _socket("clip", "CLIP")), "name": "clip", "type": "CLIP"},
                {**old.get("vae", _socket("video_vae", "VAE")), "name": "video_vae", "type": "VAE"},
                {**old.get("audio_vae", _socket("audio_vae", "VAE")), "name": "audio_vae", "type": "VAE"},
                {**old.get("guide", _socket("guide", "MINIMAX_H3_DIRECTOR_PLUS_GUIDE")), "name": "guide", "type": "MINIMAX_H3_DIRECTOR_PLUS_GUIDE"},
                _socket("generated_voice_audio", "AUDIO", shape=7),
            ]
            for raw in subgraph.get("links", []):
                link = _link_value(raw)
                if link[3] != node["id"]:
                    continue
                if link[4] == 2:
                    _set_link_field(raw, 4, 3)
                    _set_link_field(raw, 5, "MINIMAX_H3_DIRECTOR_PLUS_GUIDE")
                elif link[4] == 3:
                    _set_link_field(raw, 4, 2)
            generated_input = node["inputs"][4]
            generated_link = next((
                _link_value(raw)[0] for raw in subgraph.get("links", [])
                if _link_value(raw)[1] == -10 and _link_value(raw)[2] == generated_slot and _link_value(raw)[3] == node["id"]
            ), None)
            if generated_link is None:
                generated_link = next_link_id
                next_link_id += 1
                subgraph.setdefault("links", []).append({
                    "id": generated_link, "origin_id": -10, "origin_slot": generated_slot,
                    "target_id": node["id"], "target_slot": 4, "type": "AUDIO",
                })
            generated_input["link"] = generated_link
            exposed_inputs[generated_slot].setdefault("linkIds", []).append(generated_link)

        guide_slot = next((index for index, item in enumerate(exposed_inputs) if item.get("name") == "guide"), None)
        if guide_slot is not None:
            sampler = next((item for item in subgraph.get("nodes", []) if item.get("type") == "KSamplerSelect"), None)
            if sampler is not None:
                sampler["type"] = "MiniMaxH3SamplerRouter"
                sampler["title"] = "H3 实际采样器自动路由"
                sampler["properties"] = _properties("MiniMaxH3SamplerRouter")
                sampler["inputs"] = [
                    {"name": "sampler_name", "type": "COMBO", "widget": {"name": "sampler_name"}, "link": next((
                        _link_value(raw)[0] for raw in subgraph.get("links", [])
                        if _link_value(raw)[1] == -10 and _link_value(raw)[2] == 1 and _link_value(raw)[3] == sampler.get("id")
                    ), None)},
                    {"name": "guide", "type": "MINIMAX_H3_DIRECTOR_PLUS_GUIDE", "link": None},
                ]
                sampler["outputs"] = [{"name": "实际采样器", "type": "SAMPLER", "links": [5595, 5831]}]
                guide_link = next((
                    _link_value(raw)[0] for raw in subgraph.get("links", [])
                    if _link_value(raw)[1] == -10 and _link_value(raw)[2] == guide_slot and _link_value(raw)[3] == sampler.get("id")
                ), None)
                if guide_link is None:
                    guide_link = next_link_id
                    next_link_id += 1
                    subgraph.setdefault("links", []).append({
                        "id": guide_link, "origin_id": -10, "origin_slot": guide_slot,
                        "target_id": sampler["id"], "target_slot": 1,
                        "type": "MINIMAX_H3_DIRECTOR_PLUS_GUIDE",
                    })
                sampler["inputs"][1]["link"] = guide_link
                if guide_link not in exposed_inputs[guide_slot].setdefault("linkIds", []):
                    exposed_inputs[guide_slot]["linkIds"].append(guide_link)

            # Keep one sampler socket layout while routing the selected preset
            # through the self-contained U15 two-stage node. It automatically
            # bypasses the second stage for every other performance preset.
            for sampler in [
                item for item in subgraph.get("nodes", [])
                if item.get("type") == "SamplerCustomAdvanced"
            ]:
                old_inputs = {item.get("name"): item for item in sampler.get("inputs", [])}
                sampler["type"] = "MiniMaxH3TwoStageSampler"
                sampler["title"] = "H3 U15 二阶段 Latent 细化采样（自动旁路）"
                sampler["properties"] = _properties("MiniMaxH3TwoStageSampler")
                sampler["inputs"] = [
                    {**old_inputs.get("noise", _socket("noise", "NOISE")), "name": "noise", "type": "NOISE"},
                    {**old_inputs.get("guider", _socket("guider", "GUIDER")), "name": "guider", "type": "GUIDER"},
                    {**old_inputs.get("sampler", _socket("sampler", "SAMPLER")), "name": "sampler", "type": "SAMPLER"},
                    {**old_inputs.get("sigmas", _socket("sigmas", "SIGMAS")), "name": "sigmas", "type": "SIGMAS"},
                    {**old_inputs.get("latent_image", _socket("latent_image", "LATENT")), "name": "latent_image", "type": "LATENT"},
                    _socket("guide", "MINIMAX_H3_DIRECTOR_PLUS_GUIDE"),
                ]
                sampler["outputs"] = [
                    _output("输出Latent", "LATENT"),
                    _output("去噪Latent", "LATENT"),
                ]
                guide_link = next((
                    _link_value(raw)[0] for raw in subgraph.get("links", [])
                    if _link_value(raw)[1] == -10
                    and _link_value(raw)[2] == guide_slot
                    and _link_value(raw)[3] == sampler.get("id")
                ), None)
                if guide_link is None:
                    guide_link = next_link_id
                    next_link_id += 1
                    subgraph.setdefault("links", []).append({
                        "id": guide_link,
                        "origin_id": -10,
                        "origin_slot": guide_slot,
                        "target_id": sampler["id"],
                        "target_slot": 5,
                        "type": "MINIMAX_H3_DIRECTOR_PLUS_GUIDE",
                    })
                else:
                    for raw in subgraph.get("links", []):
                        link = _link_value(raw)
                        if link[0] == guide_link:
                            _set_link_field(raw, 4, 5)
                            _set_link_field(raw, 5, "MINIMAX_H3_DIRECTOR_PLUS_GUIDE")
                sampler["inputs"][5]["link"] = guide_link
                if guide_link not in exposed_inputs[guide_slot].setdefault("linkIds", []):
                    exposed_inputs[guide_slot]["linkIds"].append(guide_link)
        _disable_legacy_acceleration_switches(subgraph)
        node_ids = {item.get("id") for item in subgraph.get("nodes", [])}
        subgraph["links"] = [
            raw for raw in subgraph.get("links", [])
            if (
                (_link_value(raw)[1] in node_ids or (isinstance(_link_value(raw)[1], int) and _link_value(raw)[1] < 0))
                and (_link_value(raw)[3] in node_ids or (isinstance(_link_value(raw)[3], int) and _link_value(raw)[3] < 0))
            )
        ]
        _rebuild_socket_links(subgraph.get("nodes", []), subgraph.get("links", []))


def build_workflow(source):
    workflow = deepcopy(source)
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "minimax-h3-director-plus-u11"))
    workflow["revision"] = 0
    nodes = workflow.setdefault("nodes", [])
    allocate_node = _new_id_factory(workflow)
    allocate_link = _new_link_factory(workflow)

    director = _find_node(nodes, node_type="MiniMaxH3Director") or _find_node(nodes, node_type="MiniMaxH3DirectorPlus")
    if director is None:
        raise ValueError("U10 中找不到 MiniMaxH3Director")
    old_director_id = director["id"]
    _replace_director(director)
    settings = _find_node(nodes, title="Settings")
    output = next((node for node in nodes if "EnhancedVideoCombine" in str(node.get("type"))), None)
    if output is not None:
        _upgrade_stream_output_inputs(workflow, output)
        output["type"] = "MiniMaxH3StreamingVideoCombine"
        output["properties"] = _properties("MiniMaxH3StreamingVideoCombine")
        output["title"] = "预览与输出"
        values = list(output.get("widgets_values", []))
        if len(values) >= 3:
            values[1] = "H.264"
            values[2] = "MP4"
            output["widgets_values"] = values

    router = _router_node(allocate_node(), [620, 2100])
    acceleration = _acceleration_node(allocate_node(), [920, 2100])
    performance = _performance_node(allocate_node(), [1260, 2100])
    status = _status_node(allocate_node(), [1600, 2100])
    fish = _fish_node(allocate_node(), [260, 2320])
    color_guard = _color_guard_node(allocate_node(), [1510, 1800])
    materials_note = _note_node(
        allocate_node(), "导演与素材区", [620, 2290], [1280, 300],
        "## 导演与素材区\n\n1. 先选择模式，导演台会自动显示所需素材上传框。\n2. REF2VA 图片 reference 最多 9 张；提示词用 <Picture 1> 到 <Picture 9> 引用。\n3. H3 原生音色最多 3 路；角色名与 <Audio 1> 到 <Audio 3> 一一对应。\n4. 无需创建或连接任何图片、音频加载节点。",
    )
    acceleration_note = _note_node(
        allocate_node(), "加速与后处理说明", [-400, 2470], [950, 270],
        "## 已整合能力\n\n- 极速4步：T2VA/FL2VA 使用官方 H3 Turbo，REF2VA/音色参考使用官方 Ref2VA Turbo + 原生 Euler。\n- 参考图加速：实际路由模型应用 SageAttention + ComfyUI 原生 EasyCache。\n- 低显存：分辨率档位代表最终输出目标；4/8/12/15 秒分别最多按约 1.00/0.50/0.30/0.26 MP 原生生成，再在最终编码阶段按小块 CPU 放大。\n- 目标档位包含 2K QHD（2560×1440）和 4K UHD（3840×2160）；4K 不会创建整段 4K 图像批次，直接流式写入 FFmpeg。\n- RTX 30 系列使用 BF16 兼容的 Sage FP16 PV 内核；不再激活旧的死分支。\n- 原 U10 的 RIFE 插帧、RTX/模型超分、水印与高级保存继续保留。\n- LTX 二段超分属于独立高显存链路，保留为可选扩展而非默认执行。",
    )
    nodes.extend([router, acceleration, performance, status, fish, color_guard, materials_note, acceleration_note])

    if output is not None:
        image_link = next((
            _link_value(raw) for raw in workflow.get("links", [])
            if _link_value(raw)[3] == output["id"] and _link_value(raw)[4] == 0 and _link_value(raw)[5] == "IMAGE"
        ), None)
        if image_link is not None:
            image_link[3] = color_guard["id"]
            image_link[4] = 0
            image_link[5] = "IMAGE"
            _add_link(workflow, allocate_link, color_guard, 0, output, 0, "IMAGE")
        _add_link(workflow, allocate_link, director, 0, color_guard, 1, "MINIMAX_H3_DIRECTOR_PLUS_GUIDE")
        _add_link(workflow, allocate_link, director, 0, output, 1, "MINIMAX_H3_DIRECTOR_PLUS_GUIDE")

    if settings is not None:
        legacy_resolution_names = {
            "resolution_preset", "aspect_preset_when_not_image", "swap_aspect_when_not_image",
        }
        settings_subgraph = next((
            subgraph for subgraph in workflow.get("definitions", {}).get("subgraphs", []) or []
            if subgraph.get("id") == settings.get("type")
        ), None)
        if settings_subgraph is not None:
            _remove_subgraph_widget_values(settings, settings_subgraph, legacy_resolution_names)
            _remove_legacy_resolution_calculator(settings_subgraph)
        _remove_node_inputs(
            workflow,
            settings,
            legacy_resolution_names,
        )
        for item in settings.get("inputs", []):
            if item.get("name") == "guide":
                item["type"] = "MINIMAX_H3_DIRECTOR_PLUS_GUIDE"

        _rewire_existing(workflow, lambda link: link[1] == settings["id"] and link[3] == old_director_id and link[5] == "MODEL" and link[2] == 6, target=router, target_slot=1)
        _rewire_existing(workflow, lambda link: link[1] == settings["id"] and link[3] == old_director_id and link[5] == "MODEL" and link[2] == 7, target=router, target_slot=2)
        _remove_links(
            workflow,
            lambda link: link[1] == settings["id"] and link[3] == old_director_id and link[5] == "INT" and link[2] in {4, 5},
        )
        _rewire_existing(workflow, lambda link: link[1] == old_director_id and link[3] == settings["id"] and link[5] == "MINIMAX_H3_DIRECTOR_GUIDE", source_slot=0, data_type="MINIMAX_H3_DIRECTOR_PLUS_GUIDE")
        _rewire_existing(workflow, lambda link: link[1] == old_director_id and link[3] == settings["id"] and link[5] == "BOOLEAN", source_slot=7)

    lora = next((node for node in nodes if node.get("type") == "DaSiWa_LTX2LoraLoader"), None)
    if lora is not None:
        _rewire_existing(workflow, lambda link: link[3] == lora["id"] and link[5] == "MODEL", source=acceleration, source_slot=0)
        lora["title"] = "附加 LoRA（保持 None 即关闭）"

    _add_link(workflow, allocate_link, director, 0, router, 0, "MINIMAX_H3_DIRECTOR_PLUS_GUIDE")
    _add_link(workflow, allocate_link, router, 0, acceleration, 0, "MODEL")
    _add_link(workflow, allocate_link, director, 0, acceleration, 1, "MINIMAX_H3_DIRECTOR_PLUS_GUIDE")
    _add_link(workflow, allocate_link, director, 0, performance, 0, "MINIMAX_H3_DIRECTOR_PLUS_GUIDE")
    _add_link(workflow, allocate_link, acceleration, 2, performance, 1, "BOOLEAN")
    _add_link(workflow, allocate_link, director, 0, fish, 0, "MINIMAX_H3_DIRECTOR_PLUS_GUIDE")
    _add_link(workflow, allocate_link, director, 5, fish, 1, "AUDIO")

    if settings is not None:
        for source_slot, name, data_type in ((0, "steps", "INT"), (1, "enabled", "BOOLEAN"), (2, "enabled_1", "BOOLEAN")):
            target_slot = _append_input(settings, name, data_type, widget=True)
            _add_link(workflow, allocate_link, performance, source_slot, settings, target_slot, data_type)
        generated_slot = _append_input(settings, "generated_voice_audio", "AUDIO")
        _add_link(workflow, allocate_link, fish, 0, settings, generated_slot, "AUDIO")
        noise_slot = _append_input(settings, "noise_seed", "INT")
        _add_link(workflow, allocate_link, director, 8, settings, noise_slot, "INT")

    _upgrade_subgraphs(workflow)
    workflow = _strip_model_path_prefixes(workflow)
    nodes = workflow["nodes"]
    _rebuild_socket_links(nodes, workflow.get("links", []))

    workflow["groups"] = [
        {"id": "u11-settings", "title": "快速设置", "bounding": [-430, 400, 550, 1590], "color": "#5c7580", "font_size": 24, "flags": {}},
        {"id": "u11-director", "title": "导演控制台", "bounding": [120, 400, 1370, 1590], "color": "#4e788a", "font_size": 24, "flags": {}},
        {"id": "u11-output", "title": "预览与输出", "bounding": [1500, 400, 900, 1850], "color": "#5c7580", "font_size": 24, "flags": {}},
        {"id": "u11-assets", "title": "自动素材与加速", "bounding": [-430, 2030, 2560, 620], "color": "#4c6f62", "font_size": 24, "flags": {}},
    ]
    workflow.setdefault("extra", {})["u11_director_plus"] = {
        "version": "1.2",
        "source": "U10-DaSiWa-MiniMaxH3-MythicAlchemy-v12导演台.json",
        "voice_semantics": "reference_only",
    }
    return workflow


def build_api_template():
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}, "_meta": {"title": "API FL2VA 模型"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}, "_meta": {"title": "API REF2VA 模型"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax", "device": "default"}, "_meta": {"title": "API H3 文本编码器"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}, "_meta": {"title": "API 视频 VAE"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}, "_meta": {"title": "API 音频 VAE"}},
        "10": {
            "class_type": "MiniMaxH3DirectorPlus",
            "inputs": {
                "mode": "FL2VA", "prompt": "", "duration": 5, "width": 1344, "height": 768,
                "aspect_ratio": "16:9", "resolution_preset": "0.83 MP", "custom_width": 16, "custom_height": 9,
                "seed": 0,
                "voice_mode": "none", "ref_image_size": "match", "performance_preset": "稳定质量",
                "fish_model_path": "s2-pro-w4a16 (auto download)", "timeline_data": "{\"version\":1,\"items\":[]}", "target_dialogue": "", "reference_transcript": "",
                "postprocess_mode": "native", "rtx_quality": "HIGH", "ai_upscale_model": "auto",
                "voice_reference_name_1": "", "voice_reference_name_2": "", "voice_reference_name_3": "",
                "first_image": ["11", 0], "last_image": ["12", 0],
                **{f"reference_image_{index}": [str(30 + index), 0] for index in range(1, 10)},
            },
            "_meta": {"title": "快速设置 / API 入参"},
        },
        "11": {"class_type": "LoadImage", "inputs": {"image": "first.png"}, "_meta": {"title": "API 首帧图片"}},
        "12": {"class_type": "LoadImage", "inputs": {"image": "last.png"}, "_meta": {"title": "API 尾帧图片"}},
        "13": {"class_type": "LoadAudio", "inputs": {"audio": "voice_1.wav"}, "_meta": {"title": "API 音色参考音频1"}},
        "25": {"class_type": "LoadAudio", "inputs": {"audio": "voice_2.wav"}, "_meta": {"title": "API 音色参考音频2"}},
        "26": {"class_type": "LoadAudio", "inputs": {"audio": "voice_3.wav"}, "_meta": {"title": "API 音色参考音频3"}},
        **{
            str(30 + index): {"class_type": "LoadImage", "inputs": {"image": f"reference_{index}.png"}, "_meta": {"title": f"API 参考图{index}"}}
            for index in range(1, 10)
        },
        "14": {"class_type": "MiniMaxH3ModelRouter", "inputs": {"guide": ["10", 0], "fl2va_model": ["1", 0], "ref2va_model": ["2", 0]}, "_meta": {"title": "API 模型路由"}},
        "15": {"class_type": "MiniMaxH3AccelerationRouter", "inputs": {"model": ["14", 0], "guide": ["10", 0]}, "_meta": {"title": "API 兼容加速"}},
        "28": {"class_type": "MiniMaxH3PerformancePreset", "inputs": {"guide": ["10", 0], "acceleration_ready": ["15", 2]}, "_meta": {"title": "API 性能预设应用"}},
        "27": {"class_type": "MiniMaxH3FishVoiceBridge", "inputs": {"guide": ["10", 0], "reference_audio": ["13", 0]}, "_meta": {"title": "API Fish S2 音色桥接"}},
        "16": {"class_type": "MiniMaxH3DirectorPlusGuide", "inputs": {"clip": ["3", 0], "video_vae": ["4", 0], "audio_vae": ["5", 0], "guide": ["10", 0], "generated_voice_audio": ["27", 0]}, "_meta": {"title": "API H3 原生指南"}},
        "17": {"class_type": "BasicGuider", "inputs": {"model": ["15", 0], "conditioning": ["16", 0]}, "_meta": {"title": "API 采样引导"}},
        "18": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}, "_meta": {"title": "API 随机种子"}},
        "19": {"class_type": "MiniMaxH3SamplerRouter", "inputs": {"sampler_name": "res_multistep", "guide": ["10", 0]}, "_meta": {"title": "API H3 实际采样器路由"}},
        "20": {"class_type": "BasicScheduler", "inputs": {"model": ["15", 0], "scheduler": "simple", "steps": ["28", 0], "denoise": 1.0}, "_meta": {"title": "API 调度器"}},
        "21": {"class_type": "MiniMaxH3TwoStageSampler", "inputs": {"noise": ["18", 0], "guider": ["17", 0], "sampler": ["19", 0], "sigmas": ["20", 0], "latent_image": ["16", 1], "guide": ["10", 0]}, "_meta": {"title": "API H3 U15 二阶段 Latent 细化采样（自动旁路）"}},
        "22": {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["4", 0]}, "_meta": {"title": "API 视频解码"}},
        "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["21", 1], "vae": ["5", 0]}, "_meta": {"title": "API 音频解码"}},
        "29": {"class_type": "MiniMaxH3ColorGuard", "inputs": {"images": ["22", 0], "guide": ["10", 0], "enabled": True, "strength": 1.0}, "_meta": {"title": "曝光与色彩连续性保护"}},
        "24": {"class_type": "MiniMaxH3StreamingVideoCombine", "inputs": {"images": ["29", 0], "guide": ["10", 0], "frame_rate": 24.0, "codec": "H.264", "container": "MP4", "bit_depth": "Auto", "quality": 20, "log_level": "Standard", "pingpong": False, "save_metadata": True, "filename_prefix": "DirectorPlus", "save_output": True, "pass_frames": False, "crop_to_audio": False, "audio_codec": "Auto", "audio_bitrate": "192k", "save_first_frame": False, "save_last_frame": False, "audio": ["23", 0]}, "_meta": {"title": "预览与输出"}},
    }


def main():
    parser = argparse.ArgumentParser(description="从 U10 构建 U11 MiniMax H3 导演台 Plus")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    built = build_workflow(source)
    validate_workflow(built, require_sections=True)
    args.output.write_text(json.dumps(built, ensure_ascii=False, indent=2), encoding="utf-8")
    api_template = build_api_template()
    template_path = Path(__file__).resolve().parents[1] / "templates" / "u11_api.json"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(json.dumps(api_template, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"U11 已生成：{args.output}")
    print(f"API 模板已生成：{template_path}")


if __name__ == "__main__":
    main()
