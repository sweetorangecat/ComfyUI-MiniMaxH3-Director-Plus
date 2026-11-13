"""Read-only U11 plan/binding checks. Never freezes files or creates a take."""
from copy import deepcopy
import json
import re

from .long_video_plan import (target_frame_count, validate_contract, plan_windows,
                              protected_intervals, validate_window_overrides, compile_window)
from .long_video_graph import roles, apply_window_reference_policy
from .long_video_references import snapshot_bindings, SOURCE_FILES_META, PICTURE_SLOTS
from .long_video_av import conditioning_geometry


def read_contract(raw, total_frames):
    """Accept the canonical U11 export only, retaining all producer facts."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError('剧情 JSON 无法解析：' + exc.msg) from exc
    if not isinstance(raw, dict):
        raise ValueError('U11 scene_contract must be a JSON object（JSON 对象）')
    if not raw:
        raise ValueError('请导入 H3 技能导出的 U11 scene_contract JSON；默认 {} 不是剧情。示例仅供格式参考，不会自动排队。')
    if not {'schema_version', 'fps', 'beats'}.issubset(raw):
        raise ValueError('请使用 H3 技能的 U11 单语 scene_contract 导出；source_beats、分镜规划和提示词包不能直接作为运行合同。')
    story_frames = raw.get('total_frames')
    if type(story_frames) is int and story_frames != total_frames:
        raise ValueError(f'剧情为 {story_frames / 24:g} 秒，目标为 {total_frames / 24:g} 秒；'
                         '请明确采用剧情时长，或重新导出匹配合同。不会拉伸剧情或删改对白。')
    return validate_contract(raw, total_frames)


def validate_graph_shape(graph):
    if not isinstance(graph, dict) or not graph:
        raise ValueError('prompt must be the actual nonempty U11 API graph object')
    for key, node in graph.items():
        if (not isinstance(key, str) or not isinstance(node, dict)
                or not isinstance(node.get('class_type'), str) or not isinstance(node.get('inputs'), dict)):
            raise ValueError('Each prompt node requires string class_type and object inputs')
        if '_meta' in node and not isinstance(node['_meta'], dict):
            raise ValueError(f'prompt node {key} _meta metadata must be an object when provided')
        meta = node.get('_meta', {})
        if SOURCE_FILES_META in meta:
            source_files = meta[SOURCE_FILES_META]
            if (not isinstance(source_files, dict)
                    or any(field not in PICTURE_SLOTS or not isinstance(value, str)
                           for field, value in source_files.items())):
                raise ValueError(f'{SOURCE_FILES_META} metadata must be an object of original picture slots and filename strings')
        if node['class_type'] == 'MiniMaxH3DirectorPlus':
            for field in ('mode','sampling_method','performance_preset','postprocess_mode',
                          'vosr2_profile','voice_mode','face_refine_mode','motion_smoothing'):
                if field in node['inputs'] and not isinstance(node['inputs'][field], str):
                    raise ValueError(f'Director {field} must be a string')


def validate_config_shape(raw):
    if not isinstance(raw, dict):
        raise ValueError('config must be a JSON object')
    if 'enabled' in raw and type(raw['enabled']) is not bool:
        raise ValueError('config.enabled must be a boolean')
    name = raw.get('project_name', 'h3-native-av')
    reserved = {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}
    # Mirror the journal's name grammar without creating/resolving a directory.
    if (not isinstance(name, str) or not re.fullmatch(r'[\w-][\w .-]{0,99}', name)
            or name.endswith(('.', ' ')) or name.split('.')[0].upper() in reserved):
        raise ValueError('config.project_name must be one safe directory name')
    stop = raw.get('stop_condition', 'target')
    if not isinstance(stop, str) or stop not in {'target','manual'}:
        raise ValueError('config.stop_condition must be target or manual')


def preflight(base_graph, raw, *, check_encoder=True):
    """Validate all real plan windows and the production reference wrapper.

    Future prompts use a conspicuous temporary sentinel solely to exercise
    static clock/reference checks. It never escapes this function, persists to
    disk or substitutes for the actual AV observation required at runtime.
    """
    validate_config_shape(raw)
    validate_graph_shape(base_graph)
    total = target_frame_count(raw.get('target_seconds', 30))
    contract = read_contract(raw.get('scene_contract'), total)
    graph, director, _, _, output = roles(base_graph)
    from .long_video_media import resolve_encoding, encoder_capabilities
    encoding = resolve_encoding(graph[output]['inputs'])
    plan = plan_windows(total, protected_intervals=protected_intervals(contract))
    validate_window_overrides(contract, plan)
    settings = graph[director]['inputs']
    inventory = snapshot_bindings(settings, graph)
    windows = []
    for index, window in enumerate(plan):
        temporary_state = 'STATIC PREFLIGHT ONLY: actual incoming AV observation required at runtime; unobserved, unapproved.' if index else ''
        compiled = compile_window(contract, window, temporary_state)
        active = deepcopy(graph)
        active_settings = active[director]['inputs']
        active_settings['prompt'] = compiled['prompt']
        trace = apply_window_reference_policy(active_settings, {**compiled, 'window': window}, active, preflight=True)
        geometry = conditioning_geometry(window, compiled['forward_frames'],
                                         plan[index - 1]['sample_frames'] if index else None)
        windows.append({'index': index, 'start_seconds': window['new_start'] / 24,
            'end_seconds': window['new_end'] / 24,
            'net_new_frames': window['new_end'] - window['new_start'],
            'sample_seconds': window['sample_frames'] / 24,
            'reference_bindings': trace, 'dialogue_ids': compiled['dialogue_ids'],
            'forward_review_required': bool(compiled['forward_frames']), 'forward_geometry': geometry})
    model_fields = {'schema_version','fps','total_frames','subject_definitions','summary',
                    'retention_analysis','scale_contract','overall_soundscape','non_diegetic_music',
                    'beats','dialogues','window_overrides','reference_selection','voice_policy','parent_reference'}
    facts = {key: deepcopy(value) for key, value in contract.items() if key not in model_fields}
    warnings = []
    if contract.get('not_ready_for_video') is True:
        warnings.append('来源明确标记 not_ready_for_video: true；预检不授予资产或内容审批。')
    if contract.get('approved') is False or contract.get('mounted') is False:
        warnings.append('来源仍明确标记 approved:false 或 mounted:false；这些生产者事实保持不变。')
    for asset in contract.get('reference_assets', []) if isinstance(contract.get('reference_assets', []), list) else []:
        if isinstance(asset, dict) and (asset.get('approved') is False or asset.get('mounted') is False):
            warnings.append(f'{asset.get("slot", "参考资产")}：来源仍为未审批或未挂载；实际图接线清单不能改写这项来源事实。')
    warnings.append('静态预检不读取权重、不哈希资产、不排队；实际文件、依赖和模型在普通排队时核验。续窗需要实际已审 AV 状态。')
    capability = encoder_capabilities() if check_encoder else {'checked': False}
    return {'total_frames': total, 'total_seconds': total / 24, 'window_count': len(windows),
            'windows': windows, 'binding_inventory': inventory, 'source_facts': facts,
            'warnings': warnings, 'encoding': encoding, 'encoder_capability': capability,
            'route': 'REF2VA / 质量优先二采样 / SelfLift / VOSR2 manual_baseline',
            'incoming_state': 'runtime_observation_required',
            'static_future_checks': 'synthetic_state_only_never_persisted_or_approved',
            'assets_approved': False, 'media_approved': False, 'queued': False}
