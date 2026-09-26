"""Retire the marked U11 face branch, preserving prompts, audio and link IDs."""

from copy import deepcopy


def remove_face_refine(source):
    workflow = deepcopy(source)
    nodes = workflow.get("nodes", [])
    if not any(n["type"] == "MiniMaxH3DirectorPlus" for n in nodes):
        return workflow
    branch = {n["id"] for n in nodes if n.get("properties", {}).get("director_plus_face_refine")}
    if not branch:
        return workflow
    originals = {}
    for node in nodes:
        if node["id"] in branch and node["type"] == "MiniMaxH3FaceRefineSwitch":
            original = next(i for i in node["inputs"] if i["name"] == "original_images")
            edge = next(e for e in workflow["links"] if e[0] == original["link"])
            if edge[1] not in branch:
                originals[node["id"]] = edge
    links = []
    for edge in workflow["links"]:
        if edge[3] in branch:
            continue
        if edge[1] in branch:
            if edge[1] not in originals:
                raise ValueError("Unknown face branch output; cannot safely reconnect")
            original = originals[edge[1]]
            edge = [edge[0], original[1], original[2], *edge[3:]]
        links.append(edge)
    workflow["nodes"] = [n for n in nodes if n["id"] not in branch]
    workflow["links"] = links
    for node in workflow["nodes"]:
        for slot, socket in enumerate(node.get("inputs", [])):
            socket["link"] = next((e[0] for e in links if e[3:5] == [node["id"], slot]), None)
        for slot, socket in enumerate(node.get("outputs", [])):
            socket["links"] = [e[0] for e in links if e[1:3] == [node["id"], slot]]
        if node["type"] == "MiniMaxH3DirectorPlus":
            named = node.get("widgets_values_named", {})
            if "face_refine_mode" in named:
                named["face_refine_mode"] = "off"
    workflow["groups"] = [g for g in workflow.get("groups", []) if g.get("title") != "FaceRefine 人脸修复"]
    workflow.get("extra", {}).pop("director_plus_face_refine", None)
    return workflow
