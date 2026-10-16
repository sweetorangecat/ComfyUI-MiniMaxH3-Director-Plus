from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.sampling_links import ScheduleLinkError, align_h3_scheduler_models, scope_scheduler_mismatches
from tools.validate_workflow import WorkflowError, validate_workflow


def graph(link_format="array"):
    nodes = [
        {"id": 1, "type": "UNETLoader", "outputs": [{"name": "MODEL", "links": [1, 2]}]},
        {"id": 2, "type": "MiniMaxH3SigmaShift", "inputs": [{"name": "model", "link": 1}], "outputs": [{"name": "MODEL", "links": [3]}], "widgets_values": [10.0, 5.0]},
        {"id": 3, "type": "MiniMaxH3SchedulerRouter", "inputs": [{"name": "model", "link": 2}], "outputs": [{"name": "SIGMAS", "links": [4]}]},
        {"id": 4, "type": "BasicGuider", "inputs": [{"name": "model", "link": 3}], "outputs": [{"name": "GUIDER", "links": [5]}]},
        {"id": 5, "type": "MiniMaxH3TwoStageSampler", "inputs": [{"name": "guider", "link": 5}, {"name": "sigmas", "link": 4}], "outputs": []},
    ]
    links = [[1, 1, 0, 2, 0, "MODEL"], [2, 1, 0, 3, 0, "MODEL"], [3, 2, 0, 4, 0, "MODEL"], [4, 3, 0, 5, 1, "SIGMAS"], [5, 4, 0, 5, 0, "GUIDER"]]
    if link_format == "dict":
        links = [dict(zip(("id", "origin_id", "origin_slot", "target_id", "target_slot", "type"), link)) for link in links]
    elif link_format == "value":
        links = [{"value": link} for link in links]
    return {"nodes": nodes, "links": links, "extra": {"prompt": "全片无旁白", "seed": 705894478236066, "duration": 15}}


@pytest.mark.parametrize("link_format", ["array", "dict", "value"])
def test_validator_catches_unshifted_scheduler_and_repair_preserves_request(link_format):
    workflow = graph(link_format)
    before = deepcopy(workflow)
    with pytest.raises(WorkflowError, match="模型与引导器不一致"):
        validate_workflow(workflow)
    changes = align_h3_scheduler_models(workflow)
    assert len(changes) == 1
    assert changes[0]["old_source"] == [1, 0]
    assert changes[0]["new_source"] == [2, 0]
    assert workflow["extra"] == before["extra"]
    assert workflow["nodes"][1]["widgets_values"] == [10.0, 5.0]
    assert workflow["links"][2:] == before["links"][2:]
    assert workflow["nodes"][0]["outputs"][0]["links"] == [1]
    assert workflow["nodes"][1]["outputs"][0]["links"] == [3, 2]
    validate_workflow(workflow)
    once = deepcopy(workflow)
    assert align_h3_scheduler_models(workflow) == []
    assert workflow == once


def test_subgraph_repair_removes_obsolete_exposed_input_link():
    scope = graph("dict")
    scope["id"] = "settings"
    scope["inputs"] = [{"name": "model", "linkIds": [1, 2]}]
    scope["nodes"] = [n for n in scope["nodes"] if n["id"] != 1]
    for link in scope["links"]:
        if link["origin_id"] == 1:
            link["origin_id"] = -10
    workflow = {"nodes": [], "links": [], "definitions": {"subgraphs": [scope]}}
    assert len(align_h3_scheduler_models(workflow)) == 1
    assert scope["inputs"][0]["linkIds"] == [1]
    assert scope["links"][1]["origin_id"] == 2
    validate_workflow(workflow)


def test_shared_scheduler_with_conflicting_models_is_not_partially_repaired():
    workflow = graph()
    workflow["nodes"].extend([
        {"id": 6, "type": "BasicGuider", "inputs": [{"name": "model", "link": 6}], "outputs": [{"name": "GUIDER", "links": [7]}]},
        {"id": 7, "type": "MiniMaxH3TwoStageSampler", "inputs": [{"name": "guider", "link": 7}, {"name": "sigmas", "link": 8}], "outputs": []},
    ])
    workflow["links"].extend([[6, 1, 0, 6, 0, "MODEL"], [7, 6, 0, 7, 0, "GUIDER"], [8, 3, 0, 7, 1, "SIGMAS"]])
    before = deepcopy(workflow)
    with pytest.raises(ScheduleLinkError, match="shared by different guider models"):
        align_h3_scheduler_models(workflow)
    assert workflow == before


def test_upgrader_repairs_shift_model_links_in_existing_example():
    from tools.build_u11_workflow import _upgrade_subgraphs
    source = json.loads(next((Path(__file__).resolve().parents[1] / "examples").glob("U11-*.json")).read_text(encoding="utf-8"))
    settings = next(g for g in source["definitions"]["subgraphs"] if any(n.get("id") == 2590 for n in g["nodes"]))
    # Recreate the original legacy input regardless of the checked-in example's state.
    first = next(link for link in settings["links"] if link.get("target_id") == 2590 and link.get("target_slot") == 0)
    first["origin_id"], first["origin_slot"] = -10, 14
    _upgrade_subgraphs(source)
    for scope in [source, *source.get("definitions", {}).get("subgraphs", [])]:
        assert scope_scheduler_mismatches(scope.get("nodes", []), scope.get("links", [])) == []
    validate_workflow(source)


def test_checked_in_example_matches_each_sampler_guider_model():
    workflow = json.loads(next((Path(__file__).resolve().parents[1] / "examples").glob("U11-*.json")).read_text(encoding="utf-8"))
    for scope in [workflow, *workflow.get("definitions", {}).get("subgraphs", [])]:
        assert scope_scheduler_mismatches(scope.get("nodes", []), scope.get("links", [])) == []
