import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_workflow_uses_one_lazy_face_refine_switch_before_final_output():
    workflow = json.loads(next((ROOT / "examples").glob("U11-*.json")).read_text("utf-8"))
    by_id = {node["id"]: node for node in workflow["nodes"]}
    switches = [node for node in workflow["nodes"] if node["type"] == "MiniMaxH3FaceRefineSwitch"]
    assert len(switches) == 1
    switch = switches[0]
    sources = {}
    for input_ in switch["inputs"]:
        edge = next(edge for edge in workflow["links"] if edge[0] == input_["link"])
        sources[input_["name"]] = by_id[edge[1]]["type"]
    assert sources == {
        "guide": "MiniMaxH3DirectorPlus",
        "original_images": "MiniMaxH3ColorGuard",
        "refined_images": "MiniMaxH3FaceStitch",
    }
    output = next(node for node in workflow["nodes"] if node["type"] == "MiniMaxH3StreamingVideoCombine")
    image_input = next(item for item in output["inputs"] if item["name"] == "images")
    image_edge = next(edge for edge in workflow["links"] if edge[0] == image_input["link"])
    assert by_id[image_edge[1]]["type"] == "MiniMaxH3FaceRefineSwitch"
    assert not any(node["type"] in {"VHS_LoadVideoPath", "VHS_VideoCombine"} for node in workflow["nodes"])
    branch = [node for node in workflow["nodes"] if node.get("properties", {}).get("director_plus_face_refine")]
    assert branch and all(node["mode"] == 0 for node in branch)


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
        saver = next(node for node in workflow["nodes"] if node["type"] == "VHS_VideoCombine")
        for input_name, source_type, slot in [("audio", "VHS_LoadVideoPath", 2),
                                               ("frame_rate", "VHS_VideoInfoLoaded", 0)]:
            input_ = next(item for item in saver["inputs"] if item["name"] == input_name)
            edge = next(edge for edge in workflow["links"] if edge[0] == input_["link"])
            assert by_id[edge[1]]["type"] == source_type and edge[2] == slot
        scheduler = next(node for node in workflow["nodes"] if node["type"] == "BasicScheduler")
        assert scheduler["widgets_values"] == ["simple", 8, 0.4]


def test_integration_is_repeatable_and_graph_remains_connected():
    assert importlib.util.find_spec("tools.integrate_face_refine") is not None, "Integration tool missing"
    from tools.integrate_face_refine import integrate
    main = json.loads(next((ROOT / "examples").glob("U11-*.json")).read_text("utf-8"))
    face = json.loads((ROOT / "examples/face_refine/FaceRefine-right-1080p.json").read_text("utf-8"))
    result = integrate(main, face)
    assert result == integrate(result, face)
    assert_edges(result)
