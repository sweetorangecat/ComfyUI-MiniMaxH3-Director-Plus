"""Structural and layout validation for ComfyUI workflow JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class WorkflowError(ValueError):
    pass


def _link_value(link):
    if not isinstance(link, dict):
        return link
    if "value" in link:
        return link["value"]
    if "origin_id" in link:
        return [link.get("id"), link.get("origin_id"), link.get("origin_slot"), link.get("target_id"), link.get("target_slot"), link.get("type")]
    return link


def _visible(node):
    if node.get("type") == "Label (rgthree)":
        return False
    return node.get("mode", 0) == 0 and not node.get("flags", {}).get("hidden", False)


def _rectangle(node):
    pos = node.get("pos")
    size = node.get("size")
    if not isinstance(pos, list) or not isinstance(size, list) or len(pos) < 2 or len(size) < 2:
        return None
    return float(pos[0]), float(pos[1]), float(pos[0]) + float(size[0]), float(pos[1]) + float(size[1])


def _overlap(first, second, padding=2):
    return not (
        first[2] <= second[0] + padding
        or second[2] <= first[0] + padding
        or first[3] <= second[1] + padding
        or second[3] <= first[1] + padding
    )


def _validate_scope(name, nodes, links, check_layout=True):
    ids = [node.get("id") for node in nodes]
    if len(ids) != len(set(ids)):
        raise WorkflowError(f"{name} 存在重复节点 ID")
    node_map = {node.get("id"): node for node in nodes}
    link_ids = []
    for raw in links or []:
        link = _link_value(raw)
        if not isinstance(link, list) or len(link) < 6:
            raise WorkflowError(f"{name} 存在无效连线记录")
        link_id, source_id, source_slot, target_id, target_slot, _ = link[:6]
        link_ids.append(link_id)
        if source_id not in node_map and not (isinstance(source_id, int) and source_id < 0):
            raise WorkflowError(f"{name} 连线 {link_id} 指向不存在的节点")
        if target_id not in node_map and not (isinstance(target_id, int) and target_id < 0):
            raise WorkflowError(f"{name} 连线 {link_id} 指向不存在的节点")
        source_outputs = node_map.get(source_id, {}).get("outputs", [])
        target_inputs = node_map.get(target_id, {}).get("inputs", [])
        if source_id in node_map and not 0 <= int(source_slot) < len(source_outputs):
            raise WorkflowError(f"{name} 连线 {link_id} 的输出槽无效")
        if target_id in node_map and not 0 <= int(target_slot) < len(target_inputs):
            raise WorkflowError(f"{name} 连线 {link_id} 的输入槽无效")
    if len(link_ids) != len(set(link_ids)):
        raise WorkflowError(f"{name} 存在重复连线 ID")

    if check_layout:
        visible = [node for node in nodes if _visible(node) and _rectangle(node)]
        for index, first in enumerate(visible):
            for second in visible[index + 1:]:
                if _overlap(_rectangle(first), _rectangle(second)):
                    first_name = first.get("title") or first.get("type") or first.get("id")
                    second_name = second.get("title") or second.get("type") or second.get("id")
                    raise WorkflowError(f"{name} 可见节点重叠：{first_name} / {second_name}")


def validate_workflow(workflow, require_sections=False):
    if not isinstance(workflow, dict):
        raise WorkflowError("工作流必须是 JSON 对象")
    nodes = workflow.get("nodes", [])
    links = workflow.get("links", [])
    _validate_scope("主画布", nodes, links, check_layout=True)

    subgraphs = workflow.get("definitions", {}).get("subgraphs", []) or []
    subgraph_ids = [item.get("id") for item in subgraphs]
    if len(subgraph_ids) != len(set(subgraph_ids)):
        raise WorkflowError("存在重复子图 ID")
    for subgraph in subgraphs:
        for slot in (subgraph.get("inputs", []) or []) + (subgraph.get("outputs", []) or []):
            if "displayName" in slot:
                raise WorkflowError(
                    f"子图 {subgraph.get('name') or subgraph.get('id')} 端口不得包含 displayName，请使用 label"
                )
        _validate_scope(f"子图 {subgraph.get('name') or subgraph.get('id')}", subgraph.get("nodes", []), subgraph.get("links", []), check_layout=False)

    if require_sections:
        titles = {node.get("title") for node in nodes}
        required = {"快速设置 / API 入参", "导演与素材区", "预览与输出"}
        missing = required - titles
        if missing:
            raise WorkflowError(f"缺少主要中文区域：{', '.join(sorted(missing))}")
    return {
        "nodes": len(nodes),
        "links": len(links),
        "subgraphs": len(subgraphs),
        "visible_overlaps": 0,
    }


def main():
    parser = argparse.ArgumentParser(description="校验 MiniMax H3 U11 工作流")
    parser.add_argument("workflow", type=Path)
    args = parser.parse_args()
    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    report = validate_workflow(workflow, require_sections=True)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
