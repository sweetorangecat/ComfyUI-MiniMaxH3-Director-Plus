"""Build project-owned FaceRefine examples and append a postprocess branch to U11."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


PLUGIN = "ComfyUI-MiniMaxH3-Director-Plus"
MARKER = "director_plus_face_refine"
NODE_TYPES = {
    "H3FaceTrackCrop": "MiniMaxH3FaceTrackCrop",
    "H3FaceStitch": "MiniMaxH3FaceStitch",
    "H3InjectVideoLatent": "MiniMaxH3FaceInjectVideoLatent",
    "H3PerFrameDenoise": "MiniMaxH3FacePerFrameDenoise",
    "MiniMaxH3NativeAudioLock": "MiniMaxH3FaceAudioLock",
}
PROMPT = (
    "subject_definitions:\n<Subject 1> is the same person shown in <Picture 1>. "
    "The picture is an identity reference cropped from the source video.\n\n"
    "summary:\nRefine existing facial detail while retaining the person's identity, "
    "facial proportions, age, expression, gaze, head pose and original lighting.\n\n"
    "retention_analysis:\n<Subject 1>: fully_preserved.\n\n"
    "detailed_description:\nPreserve the original action and mouth timing. Resolve natural "
    "skin detail, eyes and eyelashes without changing makeup, facial accessories or "
    "hairstyle. No new action or dialogue. No plastic skin or exaggerated sharpening.\n\n"
    "overall_soundscape:\nRetain the source audio.\n\nnon_diegetic_music:\nNo additional music."
)


def _layout(nodes, x=0, y=0):
    # Each column is a stage; large tracker and prompt widgets get their own space.
    positions = {
        1: (0, 0, 330), 2: (0, 430, 960), 115: (0, 1490, 350), 116: (0, 1940, 440),
        108: (520, 0, 220), 12: (520, 320, 220), 113: (520, 640, 160),
        109: (520, 900, 220), 4: (520, 1220, 170), 5: (520, 1490, 170),
        112: (1040, 0, 200), 9: (1040, 300, 720), 10: (1040, 1120, 190),
        13: (1040, 1410, 190), 31: (1040, 1700, 420),
        15: (1560, 0, 160), 16: (1560, 260, 210), 17: (1560, 570, 180),
        14: (1560, 850, 160), 18: (1560, 1110, 230), 19: (1560, 1440, 180),
        22: (2080, 0, 440), 114: (2080, 540, 220), 23: (2080, 860, 600),
    }
    for node in nodes:
        px, py, height = positions[node["id"]]
        node["pos"] = [x + px, y + py]
        node["size"] = [440, height]


def make_example(template, side="right", preview=False):
    workflow = deepcopy(template)
    for node in workflow["nodes"]:
        node["type"] = NODE_TYPES.get(node["type"], node["type"])
        if node["type"].startswith("MiniMaxH3Face"):
            node["properties"] = {"Node name for S&R": node["type"], "cnr_id": PLUGIN}
    by_id = {node["id"]: node for node in workflow["nodes"]}
    by_id[1]["widgets_values"]["video"] = "input/source.mp4"
    by_id[1]["title"] = "01 读取待修复视频"
    by_id[2]["widgets_values"][3:6] = [768, 768, "manual"]
    by_id[2]["widgets_values"][2] = 2.5
    by_id[2]["widgets_values"][10] = False
    by_id[2]["widgets_values"][12] = side + "_most"
    by_id[2]["title"] = "02 跟踪" + ("右侧人物" if side == "right" else "左侧人物")
    by_id[9]["widgets_values"][0] = PROMPT
    by_id[12]["title"] = "人脸修复 8 步 LoRA"
    by_id[13]["title"] = "原音轨锁定（本项目内置）"
    by_id[16]["widgets_values"] = ["simple", 8, 0.4]
    by_id[22]["title"] = "合成回原分辨率"
    by_id[23]["title"] = "保存修复版（原帧率与原音轨）"
    by_id[23]["widgets_values"]["filename_prefix"] = "video/FaceRefine/" + side
    by_id[116]["widgets_values"] = [
        "人脸修复由当前 Director Plus 项目提供，无需安装独立 FaceRefine 插件。\n"
        "768×768，8 步，denoise=0.4；select 选择左/右人物，连续跟踪。\n"
        "先用 tracking-only 示例检查跟踪。身份识别默认关闭，可另装依赖后启用。\n"
        "合成保持输入分辨率、帧数和原音轨；1080p 原片输出仍为 1080p。\n"
        "高清角色参考图可替换 ImageFromBatch 到参考图输入的连线。\n"
        "RTX 5090 已跑通；这段双人物近景实测变软，不能保证更清晰。\n"
        "主工作流默认禁用本分支，需按原片对照后决定是否启用。"
    ]
    if preview:
        keep = {1, 2, 115, 116}
        workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] in keep]
        workflow["links"] = [edge for edge in workflow["links"] if edge[1] in keep and edge[3] in keep]
        _clean_links(workflow)
    _layout(workflow["nodes"])
    workflow["groups"] = []
    workflow["extra"] = {"ds": {"scale": 0.5, "offset": [40, 60]}, MARKER: {
        "subject": side, "gpu_inference_verified": False,
        "backend_commit": "d8521d14fe0d721d80cd9417fff5a559cbc21aba",
    }}
    return workflow


def _clean_links(workflow):
    valid = {edge[0] for edge in workflow["links"]}
    for node in workflow["nodes"]:
        for item in node.get("inputs", []):
            if item.get("link") not in valid:
                item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = [ident for ident in item.get("links") or [] if ident in valid]


def integrate(main, face):
    workflow = deepcopy(main)
    nodes = workflow["nodes"]
    by_id = {node["id"]: node for node in nodes}
    # Reuse the project-owned branch already present in the U11 workflow. This
    # keeps its latent/model wiring intact while replacing only the old
    # save-and-reread boundary.
    branch = [node for node in nodes if node.get("properties", {}).get(MARKER)]
    old_switches = [node for node in branch if node.get("type") == "MiniMaxH3FaceRefineSwitch"]
    obsolete_types = {"VHS_LoadVideoPath", "VHS_VideoCombine", "VHS_VideoInfoLoaded"}
    obsolete = {node["id"] for node in branch if node.get("type") in obsolete_types}
    obsolete.update(node["id"] for node in old_switches)
    if obsolete:
        workflow["nodes"] = [node for node in nodes if node["id"] not in obsolete]
        workflow["links"] = [edge for edge in workflow["links"] if edge[1] not in obsolete and edge[3] not in obsolete]
        nodes = workflow["nodes"]
        by_id = {node["id"]: node for node in nodes}
    workflow["groups"] = [group for group in workflow.get("groups", []) if group.get("title") != "FaceRefine 人脸修复"]
    _clean_links(workflow)
    saver = next(node for node in workflow["nodes"] if node["type"] == "MiniMaxH3StreamingVideoCombine")
    branch = [node for node in workflow["nodes"] if node.get("properties", {}).get(MARKER)]
    if not branch:
        raise ValueError("主工作流缺少项目内置 FaceRefine 分支，请先导入旧版 FaceRefine 集成工作流")
    face_track = next(node for node in branch if node.get("type") == "MiniMaxH3FaceTrackCrop")
    face_stitch = next(node for node in branch if node.get("type") == "MiniMaxH3FaceStitch")
    audio_lock = next(node for node in branch if node.get("type") == "MiniMaxH3FaceAudioLock")
    color_guard = next(node for node in workflow["nodes"] if node.get("type") == "MiniMaxH3ColorGuard")
    director = next(node for node in workflow["nodes"] if node.get("type") == "MiniMaxH3DirectorPlus")
    director_values = director.setdefault("widgets_values", [])
    if not director_values or director_values[-1] != "off":
        director_values.append("off")
    director.setdefault("widgets_values_named", {})["face_refine_mode"] = "off"

    def remove_target_links(target_id, target_slot):
        workflow["links"] = [edge for edge in workflow["links"] if not (edge[3] == target_id and edge[4] == target_slot)]

    def add_link(source, source_slot, target, target_slot, data_type):
        ident = max([edge[0] for edge in workflow["links"]] + [0]) + 1
        workflow["links"].append([ident, source["id"], source_slot, target["id"], target_slot, data_type])
        return ident

    def source_link(target_id, target_slot):
        return next((edge for edge in workflow["links"] if edge[3] == target_id and edge[4] == target_slot), None)

    # Original decoded frames are the tracking/stitch base. The original H3
    # audio remains authoritative and never passes through a video file.
    remove_target_links(face_track["id"], 0)
    remove_target_links(face_stitch["id"], 0)
    remove_target_links(audio_lock["id"], 3)
    add_link(color_guard, 0, face_track, 0, "IMAGE")
    add_link(color_guard, 0, face_stitch, 0, "IMAGE")
    audio_edge = source_link(saver["id"], 2)
    if audio_edge is None:
        raise ValueError("主工作流最终输出没有原始音频连接")
    workflow["links"].append([max([edge[0] for edge in workflow["links"]] + [0]) + 1,
                               audio_edge[1], audio_edge[2], audio_lock["id"], 3, "AUDIO"])

    for node in branch:
        # An active preview is its own execution root: it would evaluate the
        # tracker even when the final lazy selector requests only original frames.
        node["mode"] = 2 if node["type"] == "PreviewImage" else 0
        node.setdefault("properties", {})[MARKER] = True

    selector_id = max(node["id"] for node in workflow["nodes"] if isinstance(node.get("id"), int)) + 1
    selector = {
        "id": selector_id, "type": "MiniMaxH3FaceRefineSwitch", "title": "人脸修复统一开关（懒加载）",
        "pos": [saver["pos"][0] + saver["size"][0] + 80, saver["pos"][1]], "size": [420, 150], "flags": {},
        "order": max(node.get("order", 0) for node in workflow["nodes"]) + 1, "mode": 0,
        "inputs": [
            {"name": "guide", "type": "MINIMAX_H3_DIRECTOR_PLUS_GUIDE", "link": None},
            {"name": "original_images", "type": "IMAGE", "link": None, "shape": 7},
            {"name": "refined_images", "type": "IMAGE", "link": None, "shape": 7},
        ],
        "outputs": [{"name": "images", "type": "IMAGE", "links": []}],
        "properties": {"Node name for S&R": "MiniMaxH3FaceRefineSwitch", "cnr_id": PLUGIN, MARKER: True},
        "widgets_values": [], "color": "#3f789e", "bgcolor": "#31566f",
    }
    workflow["nodes"].append(selector)
    for target, slot in ((selector, 0),):
        remove_target_links(target["id"], slot)
    add_link(director, 0, selector, 0, "MINIMAX_H3_DIRECTOR_PLUS_GUIDE")
    add_link(color_guard, 0, selector, 1, "IMAGE")
    add_link(face_stitch, 0, selector, 2, "IMAGE")
    remove_target_links(saver["id"], 0)
    add_link(selector, 0, saver, 0, "IMAGE")
    for order, node in enumerate(workflow["nodes"]):
        node["order"] = order
    origin_x = max(node["pos"][0] + node["size"][0] for node in branch) + 300
    workflow["groups"].append({"title": "FaceRefine 人脸修复", "bounding": [origin_x - 40, -100, 2600, 2580],
                               "color": "#3f789e", "font_size": 24, "flags": {}})
    workflow["last_node_id"] = max(node["id"] for node in workflow["nodes"] if isinstance(node["id"], int))
    workflow["last_link_id"] = max([edge[0] for edge in workflow["links"]] + [0])
    workflow.setdefault("extra", {})[MARKER] = {"subject": "right", "gpu_inference_verified": False,
                                               "input": "original_decoded_frames", "enabled_by_default": False}
    # Reconstruct serialized socket metadata after rewiring. ComfyUI validates
    # both sides of every edge, so stale output link ids are not harmless.
    for node in workflow["nodes"]:
        for item in node.get("inputs", []):
            item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = []
    for edge in workflow["links"]:
        ident, source_id, source_slot, target_id, target_slot, _ = edge
        source = next(node for node in workflow["nodes"] if node["id"] == source_id)
        target = next(node for node in workflow["nodes"] if node["id"] == target_id)
        target["inputs"][target_slot]["link"] = ident
        target_links = source["outputs"][source_slot].setdefault("links", [])
        if ident not in target_links:
            target_links.append(ident)
    return workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-directory", type=Path, help="One-time import of the previous trial deliverables")
    parser.add_argument("--workflow", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    examples = root / "examples" / "face_refine"
    template_path = ((args.trial_directory or examples) / "FaceRefine-right-1080p.json")
    template = json.loads(template_path.read_text("utf-8"))
    examples.mkdir(parents=True, exist_ok=True)
    for side in ("right", "left"):
        for preview in (False, True):
            result = make_example(template, side, preview)
            suffix = "tracking-only" if preview else "1080p"
            path = examples / f"FaceRefine-{side}-{suffix}.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", "utf-8")
    path = args.workflow or next((root / "examples").glob("U11-*.json"))
    main_workflow = json.loads(path.read_text("utf-8"))
    result = integrate(main_workflow, make_example(template))
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"Updated {path.name}: {len(result['nodes'])} nodes, {len(result['links'])} links")


if __name__ == "__main__":
    main()
