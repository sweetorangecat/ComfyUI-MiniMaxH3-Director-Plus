import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_workflow_has_no_face_branch_and_preserves_output():
    workflow = json.loads(next((ROOT / "examples").glob("U11-*.json")).read_text("utf-8"))
    assert_edges(workflow)
    assert not any(n.get("properties", {}).get("director_plus_face_refine") or
                   n["type"].startswith("MiniMaxH3Face") for n in workflow["nodes"])
    assert not any("FaceRefine" in g.get("title", "") for g in workflow.get("groups", []))
    by_id = {n["id"]: n for n in workflow["nodes"]}
    output = next(n for n in workflow["nodes"] if n["type"] == "MiniMaxH3StreamingVideoCombine")
    image = next(i for i in output["inputs"] if i["name"] == "images")
    edge = next(e for e in workflow["links"] if e[0] == image["link"])
    assert by_id[edge[1]]["type"] == "MiniMaxH3ColorGuard"
    assert next(i for i in output["inputs"] if i["name"] == "audio")["link"] is not None


def assert_edges(workflow):
    by_id = {node["id"]: node for node in workflow["nodes"]}
    assert len(by_id) == len(workflow["nodes"])
    edges = {link[0]: link for link in workflow["links"]}
    assert len(edges) == len(workflow["links"])
    subgraphs = {graph["id"] for graph in workflow.get("definitions", {}).get("subgraphs", [])}
    predecessors = {ident: set() for ident, node in by_id.items() if node["type"] not in subgraphs}
    for ident, source, slot, target, target_slot, kind in edges.values():
        output = by_id[source]["outputs"][slot]
        input_ = by_id[target]["inputs"][target_slot]
        assert ident in output["links"]
        assert input_["link"] == ident
        assert kind == output["type"] == input_["type"]
        # A serialized subgraph has independently produced output ports. Treating
        # it as one executable node invents cycles in the original U11 graph.
        if target in predecessors and source in predecessors:
            predecessors[target].add(source)
    for node in by_id.values():
        for input_ in node.get("inputs", []):
            assert input_.get("link") is None or input_["link"] in edges
        for output in node.get("outputs", []):
            assert all(ident in edges for ident in output.get("links") or [])
    while predecessors:
        ready = {ident for ident, deps in predecessors.items() if not deps}
        assert ready, "Workflow contains a cycle"
        predecessors = {ident: deps - ready for ident, deps in predecessors.items() if ident not in ready}


def test_face_examples_preserve_audio_fps_and_sampling_recipe():
    files = list((ROOT / "examples" / "face_refine").glob("*.json"))
    assert len(files) == 4, "Integrated left/right workflows and previews are missing"
    from tools.validate_workflow import validate_workflow
    for path in files:
        workflow = json.loads(path.read_text("utf-8"))
        assert_edges(workflow)
        validate_workflow(workflow)
        types = {node["type"] for node in workflow["nodes"]}
        assert not types.intersection({"H3FaceTrackCrop", "H3FaceStitch", "H3InjectVideoLatent",
                                       "H3PerFrameDenoise", "MiniMaxH3NativeAudioLock"})
        tracker = next(node for node in workflow["nodes"] if node["type"] == "MiniMaxH3FaceTrackCrop")
        assert tracker["widgets_values"][3:6] == [768, 768, "manual"]
        assert tracker["widgets_values"][12] == ("left_most" if "left" in path.name else "right_most")
        if "tracking-only" in path.name:
            assert "SamplerCustomAdvanced" not in types
            continue
        by_id = {node["id"]: node for node in workflow["nodes"]}
        assert by_id[112]["type"] == "LoadImage"
        assert not any(node["type"] == "ImageFromBatch" for node in workflow["nodes"])
        saver = next(node for node in workflow["nodes"] if node["type"] == "VHS_VideoCombine")
        for input_name, source_type, slot in [("audio", "VHS_LoadVideoPath", 2),
                                               ("frame_rate", "VHS_VideoInfoLoaded", 0)]:
            input_ = next(item for item in saver["inputs"] if item["name"] == input_name)
            edge = next(edge for edge in workflow["links"] if edge[0] == input_["link"])
            assert by_id[edge[1]]["type"] == source_type and edge[2] == slot
        scheduler = next(node for node in workflow["nodes"] if node["type"] == "BasicScheduler")
        assert scheduler["widgets_values"] == ["simple", 8, 0.4]


def test_remove_face_branch_preserves_media_and_is_idempotent():
    from tools.remove_face_refine import remove_face_refine
    graph = {"nodes": [
        {"id": 1, "type": "MiniMaxH3DirectorPlus", "widgets_values": ["user prompt", "reference.png"]},
        {"id": 2, "type": "MiniMaxH3ColorGuard", "outputs": [{"links": [10]}]},
        {"id": 3, "type": "MiniMaxH3FaceRefineSwitch", "properties": {"director_plus_face_refine": True},
         "inputs": [{"name": "original_images", "link": 10}], "outputs": [{"links": [11]}]},
        {"id": 4, "type": "MiniMaxH3StreamingVideoCombine", "inputs": [{"link": 11}, {"link": 12}]},
        {"id": 5, "type": "VAEDecodeAudio", "outputs": [{"links": [12]}]},
    ], "links": [[10, 2, 0, 3, 0, "IMAGE"], [11, 3, 0, 4, 0, "IMAGE"], [12, 5, 0, 4, 1, "AUDIO"]],
       "groups": [{"title": "FaceRefine 人脸修复"}, {"title": "Output"}]}
    result = remove_face_refine(graph)
    assert len(graph["nodes"]) == 5
    assert [n["id"] for n in result["nodes"]] == [1, 2, 4, 5]
    assert result["links"] == [[11, 2, 0, 4, 0, "IMAGE"], [12, 5, 0, 4, 1, "AUDIO"]]
    assert result["nodes"][0] == graph["nodes"][0]
    assert result["groups"] == [{"title": "Output"}]
    assert result == remove_face_refine(result)
