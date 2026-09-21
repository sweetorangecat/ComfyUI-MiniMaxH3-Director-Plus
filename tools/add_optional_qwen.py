"""Idempotently add a disabled local drafting branch without changing video links."""
import argparse
import json
from pathlib import Path

BASE = "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors"
TAIL = "qwen3vl_32b_h3_generation_tail_50_63_int8_convrot.safetensors"


def add_optional_qwen(workflow):
    if any(n['type'] == 'MiniMaxH3LocalPromptDraft' for n in workflow['nodes']):
        return workflow
    nodes = workflow['nodes']
    node_id = max(workflow.get('last_node_id', 0), *(n['id'] for n in nodes)) + 1
    x = max(n['pos'][0] + n['size'][0] for n in nodes) + 120
    nodes.append({
        'id': node_id, 'type': 'MiniMaxH3LocalPromptDraft',
        'title': '可选：本地提示词草稿（默认关闭，预览后使用）',
        'pos': [x, 0], 'size': [640, 850], 'flags': {}, 'order': len(nodes), 'mode': 0,
        'inputs': [{'name': 'image', 'type': 'IMAGE', 'link': None}],
        'outputs': [{'name': name, 'type': 'STRING', 'links': None} for name in ['draft', 'report', 'original_prompt']],
        'properties': {'Node name for S&R': 'MiniMaxH3LocalPromptDraft'},
        'widgets_values': [False, BASE, TAIL, '',
            'Draft a MiniMax H3 video prompt. Preserve supplied dialogue verbatim, reference numbers, speaker bindings and timing. Do not invent dialogue. Return only the draft.', 512, 0],
    })
    workflow['last_node_id'] = node_id
    return workflow


def update_example_defaults(workflow):
    scopes = [workflow, *workflow.get('definitions', {}).get('subgraphs', [])]
    for scope in scopes:
        for node in scope.get('nodes', []):
            if node['type'] == 'MiniMaxH3AccelerationRouter':
                values = node.get('widgets_values', [])
                if values:
                    values[0] = '全部关闭'
            if node['type'] == 'MarkdownNote' and node.get('title') == '加速与后处理说明':
                node['widgets_values'] = ['480p / 768p H3 / 1080p FHD / 2K QHD / 4K UHD\n\n480p 智能单采允许 4–15 秒，8GB 峰值与画质需实测。768p 是最终输出尺寸，不代表原生采样细节；1080p 低显存限制为 4–6 秒。\n\n额外社区 LoRA 默认关闭。可选 Qwen 本地提示词草稿默认关闭，不自动改写导演台。接入说明：docs/Qwen本地草稿与480p.md']
    return workflow


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workflow', type=Path)
    args = parser.parse_args()
    data = json.loads(args.workflow.read_text(encoding='utf-8'))
    add_optional_qwen(data)
    update_example_defaults(data)
    args.workflow.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
