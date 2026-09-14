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
        "生成效果和耗时尚未经过 GPU 实测；脸型变化时可降低 denoise 至 0.3。"
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
    removed = {node["id"] for node in workflow["nodes"] if node.get("properties", {}).get(MARKER)}
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] not in removed]
    workflow["links"] = [edge for edge in workflow["links"] if edge[1] not in removed and edge[3] not in removed]
    workflow["groups"] = [group for group in workflow.get("groups", []) if group.get("title") != "FaceRefine 人脸修复"]
    _clean_links(workflow)
    saver = next(node for node in workflow["nodes"] if node["type"] == "MiniMaxH3StreamingVideoCombine")
    branch = deepcopy(face)
    origin_x = max(node["pos"][0] + node["size"][0] for node in workflow["nodes"]) + 300
    _layout(branch["nodes"], origin_x, 0)
    subgraphs = workflow.get("definitions", {}).get("subgraphs", [])
    all_nodes = workflow["nodes"] + [node for graph in subgraphs for node in graph.get("nodes", [])]
    first_id = max(node["id"] for node in all_nodes if isinstance(node["id"], int)) + 1
    mapping = {node["id"]: first_id + index for index, node in enumerate(branch["nodes"])}
    first_link = max([edge[0] for edge in workflow["links"]] + [0]) + 1
    link_mapping = {edge[0]: first_link + index for index, edge in enumerate(branch["links"])}
    for node in branch["nodes"]:
        node["id"] = mapping[node["id"]]
        node.setdefault("properties", {})[MARKER] = True
        for input_ in node.get("inputs", []):
            if input_.get("link") is not None:
                input_["link"] = link_mapping[input_["link"]]
        for output in node.get("outputs", []):
            output["links"] = [link_mapping[ident] for ident in output.get("links") or []]
    for edge in branch["links"]:
        edge[0], edge[1], edge[3] = link_mapping[edge[0]], mapping[edge[1]], mapping[edge[3]]
    reader = next(node for node in branch["nodes"] if node["type"] == "VHS_LoadVideoPath")
    reader["title"] = "自动读取 U11 原版输出"
    reader["widgets_values"]["video"] = ""
    link_id = first_link + len(branch["links"])
    reader.setdefault("inputs", []).append({"name": "video", "type": "STRING",
                                           "widget": {"name": "video"}, "link": link_id})
    saver["outputs"][1].setdefault("links", []).append(link_id)
    branch["links"].append([link_id, saver["id"], 1, reader["id"], len(reader["inputs"]) - 1, "STRING"])
    workflow["nodes"].extend(branch["nodes"])
    workflow["links"].extend(branch["links"])
    for order, node in enumerate(workflow["nodes"]):
        node["order"] = order
    workflow["groups"].append({"title": "FaceRefine 人脸修复", "bounding": [origin_x - 40, -100, 2600, 2580],
                               "color": "#3f789e", "font_size": 24, "flags": {}})
    workflow["last_node_id"] = max(node["id"] for node in workflow["nodes"] if isinstance(node["id"], int))
    workflow["last_link_id"] = link_id
    workflow.setdefault("extra", {})[MARKER] = {"subject": "right", "gpu_inference_verified": False,
                                               "input": "saved_output_filename"}
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
