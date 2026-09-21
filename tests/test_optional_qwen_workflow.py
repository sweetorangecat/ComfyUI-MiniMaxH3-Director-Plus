from copy import deepcopy
from tools.add_optional_qwen import add_optional_qwen, update_example_defaults
from tools.validate_workflow import validate_workflow


def test_optional_branch_is_disabled_unconnected_and_idempotent():
    workflow = {'last_node_id': 1, 'nodes': [{'id': 1, 'type': 'Existing', 'pos': [0, 0], 'size': [300, 300], 'inputs': [], 'outputs': []}], 'links': []}
    original = deepcopy(workflow)
    add_optional_qwen(workflow)
    node = workflow['nodes'][-1]
    assert node['widgets_values'][0] is False
    assert workflow['nodes'][0] == original['nodes'][0]
    assert workflow['links'] == original['links']
    assert all(output['links'] is None for output in node['outputs'])
    expected = deepcopy(workflow)
    add_optional_qwen(workflow)
    assert workflow == expected
    validate_workflow(workflow)


def test_example_defaults_disable_only_extra_loras():
    workflow = {'nodes': [{'type': 'MiniMaxH3AccelerationRouter', 'widgets_values': ['全部自动叠加', '注入新噪声（U22 同配方）']}]}
    update_example_defaults(workflow)
    assert workflow['nodes'][0]['widgets_values'] == ['全部关闭', '注入新噪声（U22 同配方）']
