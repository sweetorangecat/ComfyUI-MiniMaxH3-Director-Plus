import pytest

from tools.build_u11_workflow import build_workflow, _disable_legacy_acceleration_switches
from tools.validate_workflow import WorkflowError, validate_workflow


def test_validator_rejects_overlapping_visible_nodes():
    workflow = {
        "nodes": [
            {"id": 1, "type": "A", "pos": [0, 0], "size": [200, 100], "mode": 0},
            {"id": 2, "type": "B", "pos": [100, 50], "size": [200, 100], "mode": 0},
        ],
        "links": [],
        "groups": [],
    }
    with pytest.raises(WorkflowError, match="重叠"):
        validate_workflow(workflow)


def test_legacy_acceleration_switches_do_not_control_dead_subgraph_nodes():
    subgraph = {
        "nodes": [
            {"id": 1, "type": "EasyCache", "inputs": [{"name": "model", "link": None}]},
            {"id": 2, "type": "DaSiWa_NodeStatusSwitch", "inputs": [
                {"name": "enabled", "link": None},
                {"name": "target_01", "link": 11},
            ]},
        ],
        "links": [{"id": 11, "origin_id": 1, "origin_slot": 0, "target_id": 2, "target_slot": 1, "type": "*"}],
    }

    _disable_legacy_acceleration_switches(subgraph)

    switch = next(node for node in subgraph["nodes"] if node["id"] == 2)
    assert switch["inputs"][1]["link"] is None
    assert subgraph["links"] == []


def test_validator_rejects_read_only_subgraph_display_name():
    workflow = {
        "nodes": [],
        "links": [],
        "groups": [],
        "definitions": {
            "subgraphs": [
                {
                    "id": "settings-subgraph",
                    "inputs": [
                        {
                            "id": "generated-audio",
                            "name": "generated_voice_audio",
                            "type": "AUDIO",
                            "linkIds": [],
                            "displayName": "Fish generated dialogue",
                        }
                    ],
                    "outputs": [],
                    "nodes": [],
                    "links": [],
                }
            ]
        },
    }

    with pytest.raises(WorkflowError, match="displayName"):
        validate_workflow(workflow)


def test_built_workflow_has_three_primary_sections():
    source = {
        "last_node_id": 10,
        "last_link_id": 20,
        "nodes": [
            {"id": 1, "type": "Settings", "title": "Settings", "pos": [0, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
            {"id": 2, "type": "MiniMaxH3Director", "title": "", "pos": [400, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {"id": 3, "type": "DaSiWa_EnhancedVideoCombine", "title": "", "pos": [1250, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
        ],
        "links": [],
        "groups": [],
        "definitions": {"subgraphs": []},
    }
    built = build_workflow(source)
    titles = {node.get("title") for node in built["nodes"]}
    assert {"快速设置 / API 入参", "导演与素材区", "预览与输出"} <= titles


def test_visual_workflow_embeds_media_uploads_in_director():
    source = {
        "last_node_id": 10,
        "last_link_id": 20,
        "nodes": [
            {"id": 1, "type": "Settings", "title": "Settings", "pos": [0, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
            {"id": 2, "type": "MiniMaxH3Director", "title": "", "pos": [400, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {"id": 3, "type": "DaSiWa_EnhancedVideoCombine", "title": "", "pos": [1250, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
        ],
        "links": [],
        "groups": [],
        "definitions": {"subgraphs": []},
    }
    built = build_workflow(source)
    director = next(node for node in built["nodes"] if node["type"] == "MiniMaxH3DirectorPlus")
    input_names = {item["name"] for item in director["inputs"]}
    assert "first_image_file" in input_names
    assert "voice_reference_audio_file" in input_names
    assert "voice_reference_audio_2_file" in input_names
    assert "voice_reference_audio_3_file" in input_names
    assert "reference_image_9_file" in input_names
    assert not any(node["type"] in {"LoadImage", "LoadAudio"} for node in built["nodes"])


def test_built_workflow_keeps_duration_and_resolution_widgets():
    source = {
        "last_node_id": 10,
        "last_link_id": 20,
        "nodes": [
            {"id": 1, "type": "Settings", "title": "Settings", "pos": [0, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
            {"id": 2, "type": "MiniMaxH3Director", "title": "", "pos": [400, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {"id": 3, "type": "DaSiWa_EnhancedVideoCombine", "title": "", "pos": [1250, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
        ],
        "links": [],
        "groups": [],
        "definitions": {"subgraphs": []},
    }
    built = build_workflow(source)
    director = next(node for node in built["nodes"] if node["type"] == "MiniMaxH3DirectorPlus")
    input_names = {item["name"] for item in director["inputs"]}
    assert {"width", "height"} <= input_names
    assert director["widgets_values"][:5] == ["FL2VA", "", 5, 1344, 768]
    assert director["widgets_values"][15:17] == ["native", "HIGH"]


def test_built_settings_removes_legacy_resolution_calculator():
    source = {
        "last_node_id": 10,
        "last_link_id": 20,
        "nodes": [
            {"id": 1, "type": "settings-subgraph", "title": "Settings", "pos": [0, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
            {"id": 2, "type": "MiniMaxH3Director", "title": "", "pos": [400, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {"id": 3, "type": "DaSiWa_EnhancedVideoCombine", "title": "", "pos": [1250, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
        ],
        "links": [],
        "groups": [],
        "definitions": {"subgraphs": [{
            "id": "settings-subgraph",
            "inputs": [],
            "outputs": [],
            "nodes": [{"id": 2531, "type": "DaSiWa_ResolutionScaleCalculator", "inputs": [], "outputs": []}],
            "links": [],
        }]},
    }

    built = build_workflow(source)
    subgraph = built["definitions"]["subgraphs"][0]

    assert not any(node["type"] == "DaSiWa_ResolutionScaleCalculator" for node in subgraph["nodes"])


def test_api_template_has_clean_output_prefix_and_all_reference_slots():
    from tools.build_u11_workflow import build_api_template

    prompt = build_api_template()
    controller = prompt["10"]["inputs"]
    titles = {node.get("_meta", {}).get("title") for node in prompt.values()}
    assert all(f"API 参考图{index}" in titles for index in range(1, 10))
    assert all(f"API 音色参考音频{index}" in titles for index in range(1, 4))
    assert prompt["24"]["inputs"]["filename_prefix"] == "DirectorPlus"
    assert "reference_image_9" in controller
    assert controller["motion_smoothing"] == "off"
    assert controller["audio_loudness"] == "auto"


def test_generated_director_appends_safe_output_defaults_without_shifting_timeline():
    source = {
        "last_node_id": 3,
        "last_link_id": 0,
        "nodes": [
            {"id": 1, "type": "Settings", "title": "Settings", "pos": [0, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
            {"id": 2, "type": "MiniMaxH3Director", "title": "", "pos": [400, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {"id": 3, "type": "DaSiWa_EnhancedVideoCombine", "title": "", "pos": [1250, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
        ],
        "links": [],
        "groups": [],
        "definitions": {"subgraphs": []},
    }

    director = next(
        node for node in build_workflow(source)["nodes"]
        if node["type"] == "MiniMaxH3DirectorPlus"
    )

    assert director["widgets_values"][18] == '{"version":1,"items":[]}'
    assert director["widgets_values"][-2:] == ["off", "auto"]


def test_api_template_scopes_low_vram_policy_around_sampling():
    from tools.build_u11_workflow import build_api_template

    sampler = build_api_template()["21"]

    assert sampler["class_type"] == "MiniMaxH3TwoStageSampler"
    assert sampler["inputs"]["guide"] == ["10", 0]
    assert sampler["inputs"]["video_vae"] == ["4", 0]


def test_builder_routes_color_guard_through_streaming_output():
    source = {
        "last_node_id": 3,
        "last_link_id": 21,
        "nodes": [
            {
                "id": 1, "type": "Settings", "title": "Settings", "pos": [0, 0],
                "size": [300, 300], "mode": 0, "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [21]}],
            },
            {"id": 2, "type": "MiniMaxH3Director", "title": "", "pos": [400, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {
                "id": 3, "type": "DaSiWa_EnhancedVideoCombine", "title": "", "pos": [1250, 0],
                "size": [300, 300], "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": 21}], "outputs": [],
            },
        ],
        "links": [[21, 1, 0, 3, 0, "IMAGE"]],
        "groups": [],
        "definitions": {"subgraphs": []},
    }

    built = build_workflow(source)
    director = next(node for node in built["nodes"] if node["type"] == "MiniMaxH3DirectorPlus")
    guard = next(node for node in built["nodes"] if node["type"] == "MiniMaxH3ColorGuard")
    output = next(node for node in built["nodes"] if node["type"] == "MiniMaxH3StreamingVideoCombine")

    assert any(link[1:5] == [guard["id"], 0, output["id"], 0] for link in built["links"])
    assert any(link[1:5] == [director["id"], 0, output["id"], 1] for link in built["links"])


def test_builder_preserves_audio_and_frame_rate_slots_when_upgrading_output():
    source = {
        "last_node_id": 3,
        "last_link_id": 12,
        "nodes": [
            {
                "id": 1,
                "type": "Settings",
                "title": "Settings",
                "pos": [0, 0],
                "size": [300, 300],
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "CLIP", "type": "CLIP", "links": []},
                    {"name": "IMAGE", "type": "IMAGE", "links": [10]},
                    {"name": "FLOAT", "type": "FLOAT", "links": [11]},
                    {"name": "AUDIO", "type": "AUDIO", "links": [12]},
                ],
            },
            {"id": 2, "type": "MiniMaxH3Director", "title": "", "pos": [400, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {
                "id": 3,
                "type": "DaSiWa_EnhancedVideoCombine",
                "title": "",
                "pos": [1250, 0],
                "size": [300, 300],
                "mode": 0,
                "inputs": [
                    {"name": "images", "type": "IMAGE", "link": 10},
                    {"name": "audio", "type": "AUDIO", "link": 12},
                    {"name": "frame_rate", "type": "FLOAT", "link": 11},
                ],
                "outputs": [],
            },
        ],
        "links": [
            [10, 1, 1, 3, 0, "IMAGE"],
            [11, 1, 2, 3, 2, "FLOAT"],
            [12, 1, 3, 3, 1, "AUDIO"],
        ],
        "groups": [],
        "definitions": {"subgraphs": []},
    }

    built = build_workflow(source)
    output = next(node for node in built["nodes"] if node["type"] == "MiniMaxH3StreamingVideoCombine")
    slots = {item["name"]: index for index, item in enumerate(output["inputs"])}

    assert slots["audio"] > slots["frame_rate"]
    assert any(link[1:5] == [1, 3, output["id"], slots["audio"]] and link[5] == "AUDIO" for link in built["links"])
    assert any(link[1:5] == [1, 2, output["id"], slots["frame_rate"]] and link[5] == "FLOAT" for link in built["links"])


def test_builder_removes_old_resolution_links_and_model_directory_prefixes():
    source = {
        "last_node_id": 10,
        "last_link_id": 22,
        "nodes": [
            {"id": 1, "type": "Settings", "title": "Settings", "pos": [0, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": [
                {"name": f"out{index}", "type": "INT" if index in {4, 5} else "ANY", "links": []}
                for index in range(8)
            ], "widgets_values": ["minimax/minimax_h3_fl2va_pruned_int8_convrot.safetensors"]},
            {"id": 2, "type": "MiniMaxH3Director", "title": "", "pos": [400, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {"id": 3, "type": "DaSiWa_EnhancedVideoCombine", "title": "", "pos": [1250, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
        ],
        "links": [[21, 1, 4, 2, 0, "INT"], [22, 1, 5, 2, 1, "INT"]],
        "groups": [],
        "definitions": {"subgraphs": []},
    }
    built = build_workflow(source)
    director = next(node for node in built["nodes"] if node["type"] == "MiniMaxH3DirectorPlus")
    assert director["inputs"][0]["link"] is None
    assert director["inputs"][1]["link"] is None
    serialized = str(built)
    assert "minimax/minimax_h3" not in serialized
    assert "MiniMaxH3/" not in serialized


def test_builder_connects_fish_bridge_through_settings_to_all_guides():
    subgraph = {
        "id": "settings-subgraph",
        "inputs": [{"id": "guide-input", "name": "guide", "type": "MINIMAX_H3_DIRECTOR_GUIDE", "linkIds": [31]}],
        "outputs": [],
        "nodes": [
            {
                "id": 100,
                "type": "MiniMaxH3DirectorGuide",
                "inputs": [
                    {"name": "clip", "type": "CLIP", "link": None},
                    {"name": "vae", "type": "VAE", "link": None},
                    {"name": "guide", "type": "MINIMAX_H3_DIRECTOR_GUIDE", "link": 31},
                    {"name": "audio_vae", "type": "VAE", "link": None},
                ],
                "outputs": [],
            },
            {
                "id": 101,
                "type": "MiniMaxH3DirectorGuide",
                "inputs": [
                    {"name": "clip", "type": "CLIP", "link": None},
                    {"name": "vae", "type": "VAE", "link": None},
                    {"name": "guide", "type": "MINIMAX_H3_DIRECTOR_GUIDE", "link": None},
                    {"name": "audio_vae", "type": "VAE", "link": None},
                ],
                "outputs": [],
            },
        ],
        "links": [{"id": 31, "origin_id": -10, "origin_slot": 0, "target_id": 100, "target_slot": 2, "type": "MINIMAX_H3_DIRECTOR_GUIDE"}],
    }
    source = {
        "last_node_id": 101,
        "last_link_id": 31,
        "nodes": [
            {"id": 1, "type": "settings-subgraph", "title": "Settings", "pos": [0, 0], "size": [300, 300], "mode": 0, "inputs": [{"name": "guide", "type": "MINIMAX_H3_DIRECTOR_GUIDE", "link": None}], "outputs": []},
            {"id": 2, "type": "MiniMaxH3Director", "title": "", "pos": [400, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {"id": 3, "type": "DaSiWa_EnhancedVideoCombine", "title": "", "pos": [1250, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
        ],
        "links": [],
        "groups": [],
        "definitions": {"subgraphs": [subgraph]},
    }

    built = build_workflow(source)
    director = next(node for node in built["nodes"] if node["type"] == "MiniMaxH3DirectorPlus")
    bridge = next(node for node in built["nodes"] if node["type"] == "MiniMaxH3FishVoiceBridge")
    settings = next(node for node in built["nodes"] if node.get("title") == "Settings")
    generated_slot = next(index for index, item in enumerate(settings["inputs"]) if item["name"] == "generated_voice_audio")

    assert any(link[1:5] == [director["id"], 5, bridge["id"], 1] for link in built["links"])
    assert any(link[1:5] == [bridge["id"], 0, settings["id"], generated_slot] for link in built["links"])

    upgraded = built["definitions"]["subgraphs"][0]
    exposed_slot = next(index for index, item in enumerate(upgraded["inputs"]) if item["name"] == "generated_voice_audio")
    targets = {(link["target_id"], link["target_slot"]) for link in upgraded["links"] if link["origin_id"] == -10 and link["origin_slot"] == exposed_slot}
    assert targets == {(100, 4), (101, 4)}
    upgraded_guides = [node for node in upgraded["nodes"] if node["type"] == "MiniMaxH3DirectorPlusGuide"]
    assert all(node["inputs"][4]["link"] is not None for node in upgraded_guides)


def test_performance_preset_does_not_override_manual_postprocessing_switches():
    source = {
        "last_node_id": 10,
        "last_link_id": 20,
        "nodes": [
            {"id": 1, "type": "Settings", "title": "Settings", "pos": [0, 0], "size": [300, 300], "mode": 0, "inputs": [{"name": "enabled_2", "type": "BOOLEAN", "link": None}], "outputs": []},
            {"id": 2, "type": "MiniMaxH3Director", "title": "", "pos": [400, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {"id": 3, "type": "DaSiWa_EnhancedVideoCombine", "title": "", "pos": [1250, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
        ],
        "links": [],
        "groups": [],
        "definitions": {"subgraphs": []},
    }

    built = build_workflow(source)
    performance = next(node for node in built["nodes"] if node["type"] == "MiniMaxH3PerformancePreset")
    acceleration = next(node for node in built["nodes"] if node["type"] == "MiniMaxH3AccelerationRouter")
    settings = next(node for node in built["nodes"] if node.get("title") == "Settings")
    interpolation_slot = next(index for index, item in enumerate(settings["inputs"]) if item["name"] == "enabled_2")

    assert not any(link[1:5] == [performance["id"], 3, settings["id"], interpolation_slot] for link in built["links"])
    assert any(link[1:5] == [acceleration["id"], 2, performance["id"], 1] for link in built["links"])


def test_built_settings_uses_h3_sampler_router_for_fast_mode():
    source = {
        "last_node_id": 10,
        "last_link_id": 20,
        "nodes": [
            {"id": 1, "type": "Settings", "title": "Settings", "pos": [0, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
            {"id": 2, "type": "MiniMaxH3Director", "title": "", "pos": [400, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {"id": 3, "type": "DaSiWa_EnhancedVideoCombine", "title": "", "pos": [1250, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
        ],
        "links": [], "groups": [], "definitions": {"subgraphs": []},
    }
    built = build_workflow(source)
    # Minimal source has no embedded Settings subgraph; API template still uses
    # the backend-aware sampler router.
    from tools.build_u11_workflow import build_api_template
    assert build_api_template()["19"]["class_type"] == "MiniMaxH3SamplerRouter"


def test_api_template_streams_final_target_into_output_node():
    from tools.build_u11_workflow import build_api_template

    output = build_api_template()["24"]

    assert output["class_type"] == "MiniMaxH3StreamingVideoCombine"
    assert output["inputs"]["images"] == ["29", 0]
    assert output["inputs"]["guide"] == ["10", 0]


def test_api_template_uses_safe_video_vae_decode():
    from tools.build_u11_workflow import build_api_template

    assert build_api_template()["22"]["class_type"] == "MiniMaxH3SafeVAEDecode"


def test_builder_replaces_video_vae_decode_with_safe_decoder():
    source = {
        "last_node_id": 3,
        "last_link_id": 0,
        "nodes": [
            {"id": 1, "type": "VAEDecode", "title": "视频解码", "pos": [0, 0], "size": [230, 60], "mode": 0, "inputs": [], "outputs": []},
            {"id": 2, "type": "MiniMaxH3Director", "title": "", "pos": [400, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {"id": 3, "type": "DaSiWa_EnhancedVideoCombine", "title": "", "pos": [1250, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
        ],
        "links": [],
        "groups": [],
        "definitions": {"subgraphs": []},
    }
    built = build_workflow(source)
    decoder = next(node for node in built["nodes"] if node["id"] == 1)
    assert decoder["type"] == "MiniMaxH3SafeVAEDecode"


def test_builder_makes_director_the_only_resolution_and_seed_control_surface():
    subgraph_inputs = [
        {"id": "fps", "name": "value_3", "type": "FLOAT", "linkIds": [101]},
        {"id": "resolution", "name": "resolution_preset", "type": "COMBO", "linkIds": [102]},
        {"id": "aspect", "name": "aspect_preset_when_not_image", "type": "COMBO", "linkIds": [103]},
        {"id": "swap", "name": "swap_aspect_when_not_image", "type": "BOOLEAN", "linkIds": [104]},
        {"id": "seed", "name": "noise_seed", "type": "INT", "linkIds": [105]},
        {"id": "guide", "name": "guide", "type": "MINIMAX_H3_DIRECTOR_GUIDE", "linkIds": [106]},
    ]
    subgraph = {
        "id": "settings-subgraph",
        "inputs": subgraph_inputs,
        "outputs": [],
        "nodes": [
            {
                "id": 100,
                "type": "RandomNoise",
                "inputs": [{"name": "noise_seed", "type": "INT", "link": 105}],
                "outputs": [{"name": "NOISE", "type": "NOISE", "links": None}],
                "widgets_values": [1, "randomize"],
            }
        ],
        "links": [
            {"id": 101, "origin_id": -10, "origin_slot": 0, "target_id": 100, "target_slot": 0, "type": "FLOAT"},
            {"id": 102, "origin_id": -10, "origin_slot": 1, "target_id": 100, "target_slot": 0, "type": "COMBO"},
            {"id": 103, "origin_id": -10, "origin_slot": 2, "target_id": 100, "target_slot": 0, "type": "COMBO"},
            {"id": 104, "origin_id": -10, "origin_slot": 3, "target_id": 100, "target_slot": 0, "type": "BOOLEAN"},
            {"id": 105, "origin_id": -10, "origin_slot": 4, "target_id": 100, "target_slot": 0, "type": "INT"},
        ],
    }
    settings_inputs = [
        {"name": item["name"], "type": item["type"], "link": None}
        for item in subgraph_inputs
        if item["name"] != "noise_seed"
    ]
    source = {
        "last_node_id": 100,
        "last_link_id": 106,
        "nodes": [
            {
                "id": 1,
                "type": "settings-subgraph",
                "title": "Settings",
                "pos": [0, 0],
                "size": [300, 300],
                "mode": 0,
                "inputs": settings_inputs,
                "outputs": [],
                "widgets_values": [24, "0.65 MP - Balanced", "3:4 - Photo", False, 99, "guide-value"],
            },
            {"id": 2, "type": "MiniMaxH3Director", "title": "", "pos": [400, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {"id": 3, "type": "DaSiWa_EnhancedVideoCombine", "title": "", "pos": [1250, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
        ],
        "links": [],
        "groups": [],
        "definitions": {"subgraphs": [subgraph]},
    }

    built = build_workflow(source)
    director = next(node for node in built["nodes"] if node["type"] == "MiniMaxH3DirectorPlus")
    settings = next(node for node in built["nodes"] if node.get("title") == "Settings")
    upgraded = built["definitions"]["subgraphs"][0]
    removed = {"resolution_preset", "aspect_preset_when_not_image", "swap_aspect_when_not_image"}

    assert removed.isdisjoint(item["name"] for item in settings["inputs"])
    assert removed.isdisjoint(item["name"] for item in upgraded["inputs"])
    assert [item["name"] for item in upgraded["inputs"]][:3] == ["value_3", "noise_seed", "guide"]
    assert settings["widgets_values"] == [24, 99, "guide-value"]

    seed_output = next(index for index, item in enumerate(director["outputs"]) if item["name"] == "噪音种子")
    noise_input = next(index for index, item in enumerate(settings["inputs"]) if item["name"] == "noise_seed")
    assert any(link[1:5] == [director["id"], seed_output, settings["id"], noise_input] for link in built["links"])


def test_builder_forces_mp4_h264_output_for_u11():
    source = {
        "last_node_id": 3,
        "last_link_id": 0,
        "nodes": [
            {"id": 1, "type": "MiniMaxH3DirectorPlus", "title": "", "pos": [0, 0], "size": [800, 500], "mode": 0, "inputs": [], "outputs": []},
            {"id": 2, "type": "Settings", "title": "Settings", "pos": [0, 600], "size": [300, 300], "mode": 0, "inputs": [], "outputs": []},
            {"id": 3, "type": "DaSiWa_EnhancedVideoCombine", "title": "", "pos": [1000, 0], "size": [300, 300], "mode": 0, "inputs": [], "outputs": [], "widgets_values": [24, "Auto", "Auto", "Auto", 20, "Standard", False, True, "video/%date:hhmmss%", True, False, False, False, "Auto", "192k", False, False, ""]},
        ],
        "links": [], "groups": [], "definitions": {"subgraphs": []},
    }

    built = build_workflow(source)
    output = next(node for node in built["nodes"] if node["type"] == "MiniMaxH3StreamingVideoCombine")
    assert output["widgets_values"][1:3] == ["H.264", "MP4"]
