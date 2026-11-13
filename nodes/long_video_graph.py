"""Compile physical windows from the user's actual U11 API graph.

No model, LoRA, image or voice slot is substituted by the long-video path.
"""
from copy import deepcopy
import json
import math

from .long_video_references import apply_reference_policy, PICTURE_SLOTS, SOURCE_FILES_META


OUTPUT_TYPES = {"MiniMaxH3StreamingVideoCombine", "MiniMaxH3U11Output"}
REQUESTED_VOICE_META = 'h3_long_video_requested_voice'


def adapt_long_voice(settings, metadata):
    """Native-only adapter; the short Director/Fish policy remains unchanged."""
    if settings.get('voice_mode', 'none') == 'none':
        metadata.pop(REQUESTED_VOICE_META, None)
    elif settings.get('voice_mode') == 'fish_lock':
        requested = {key: deepcopy(settings.get(key, '')) for key in
                     ('voice_mode', 'target_dialogue', 'reference_transcript', 'fish_model_path')}
        if metadata.get(REQUESTED_VOICE_META, requested) != requested:
            raise ValueError('Original Fish voice provenance changed during native adaptation')
        metadata[REQUESTED_VOICE_META] = requested
        settings['voice_mode'] = 'h3_reference'
        for field in ('voice_reference_audio_2', 'voice_reference_audio_3'):
            settings.pop(field, None)
            settings.pop(field + '_file', None)
    return settings


def original_voice_settings(graph, director):
    settings = deepcopy(graph[director]['inputs'])
    if settings.get('voice_mode', 'none') != 'none':
        settings.update(deepcopy(graph[director].get('_meta', {}).get(REQUESTED_VOICE_META, {})))
    return settings


class _ReferenceShapeProbe:
    """Exercise the Director's crop/filter decisions without loading an asset."""
    shape = (1, 100, 100, 3)

    def __getitem__(self, key):
        return object()


def _picture_preprocessing(prompt, slot, source, picture_id=None):
    from .director import _sanitize_h3_composite_refs
    image = _ReferenceShapeProbe()
    first = image if slot == 'first_image' else None
    refs = [image] if slot.startswith('reference_image_') else []
    actual_first, actual_refs, _ = _sanitize_h3_composite_refs(
        prompt, first, refs, source if first is not None else '', [source] if refs else [],
        reference_picture_ids=[picture_id] if refs and picture_id is not None else None)
    return {'cropped': first is not None and actual_first is not image,
            'dropped': bool(refs) and not actual_refs}


def _validate_picture_relocation(prompt, original, chosen):
    for local, original_id in enumerate(chosen):
        record = original[original_id - 1]
        before = _picture_preprocessing(prompt, record['slot'], record['preprocessing_source'], original_id)
        after = _picture_preprocessing(prompt, PICTURE_SLOTS[local], record['preprocessing_source'], local + 1)
        if before['dropped']:
            raise ValueError(f'Picture {original_id} is an auto-dropped storyboard reference; use reviewed '
                             'individual reference images instead of selecting a filtered planning sheet')
        if before != after:
            raise ValueError(f'Picture {original_id} relocation changes Director crop/filter preprocessing; '
                             'use reviewed individual reference images or preserve compatible Director roles')


def apply_window_reference_policy(settings, request, graph, *, parent_store=None, preflight=False):
    """Production binding handshake; validate actual Director relocation policy.

    Call this same wrapper before fingerprint/create_take and during graph
    compilation. The pure helper's default static trace is not a production
    preprocessing check and cannot substitute for this immutable take snapshot.
    """
    from .long_video_parent import apply_parent_overlay, retain_parent_instruction
    overlay = apply_parent_overlay(settings, request, graph, parent_store=parent_store, preflight=preflight)
    trace = apply_reference_policy(settings, request, graph,
                                   preprocessing_validator=_validate_picture_relocation)
    if request.get('parent_reference') is not None and len(trace['selection']['pictures']) > 9:
        raise ValueError('parent_reference cannot exceed the Director maximum of 9 active pictures, including the first window')
    if request.get('reference_selection') is not None:
        trace['preprocessing_validation'] = 'director_policy_checked'
    if overlay is not None:
        local = trace['maps']['pictures'][str(overlay['original_picture_id'])]
        destination = trace['selected']['pictures'][local - 1]['destination']
        actual = _picture_preprocessing(settings['prompt'], destination, '', local)
        if actual['cropped'] or actual['dropped']:
            raise ValueError('Parent native image would be cropped or dropped by the actual Director preprocessing policy')
        trace['parent_reference'] = {**overlay, 'local_picture_id': local, 'destination': destination}
        trace['preprocessing_validation'] = 'director_policy_checked'
        settings['prompt'] = retain_parent_instruction(settings['prompt'], local)
    return trace


def link(value, graph):
    return (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
            and type(value[1]) is int and value[0] in graph)


def ancestors(graph, root):
    keep, visiting = set(), set()

    def visit(node_id):
        if node_id in visiting:
            raise ValueError("U11 graph contains a cycle")
        if node_id in keep:
            return
        visiting.add(node_id)
        for value in graph[node_id]["inputs"].values():
            if link(value, graph):
                visit(value[0])
        visiting.remove(node_id)
        keep.add(node_id)
    visit(root)
    result = {key: deepcopy(value) for key, value in graph.items() if key in keep}
    # Comfy decorates its live original PROMPT with cache markers, including
    # NaN from output IS_CHANGED. These are not recipe data and cannot be
    # hashed or written into the strict project JSON journal.
    for node in result.values():
        node.pop('is_changed', None)
    return result


def one(graph, class_type):
    choices = [key for key, node in graph.items() if node.get("class_type") == class_type]
    if len(choices) != 1:
        raise ValueError(f"U11 long video requires one {class_type}; found {len(choices)}")
    return choices[0]


def roles(graph):
    graph = deepcopy(graph)
    output_ids = [key for key, node in graph.items() if node.get("class_type") in OUTPUT_TYPES]
    if len(output_ids) != 1:
        raise ValueError("U11 long video requires one connected final U11 output")
    output = output_ids[0]
    # The original U11 subgraph contains REF2VA and FL2VA samplers behind
    # native lazy switches. Freeze only its actual REF2VA route, as execution
    # does; retaining both branches would falsely find two active samplers.
    directors = [key for key,node in graph.items() if node.get('class_type') == 'MiniMaxH3DirectorPlus']
    if len(directors) == 1:
        director_id = directors[0]
        fish_ids = {key for key,node in graph.items() if node.get('class_type') == 'MiniMaxH3FishVoiceBridge'}
        for node in graph.values():
            inputs = node['inputs']
            for field in ('generated_voice_audio', 'audio_override'):
                value = inputs.get(field)
                if link(value, graph) and value[0] in fish_ids:
                    inputs.pop(field)
            if node['class_type'] == 'ComfySwitchNode' and inputs.get('switch') == [director_id, 7]:
                inputs['switch'] = False  # Director output 7 is "use FL2VA".
                inputs.pop('on_true', None)
            if node['class_type'] == 'MiniMaxH3ModelRouter':
                inputs.pop('fl2va_model', None)
    settings = graph[directors[0]]['inputs'] if len(directors) == 1 else {}
    if len(directors) == 1:
        adapt_long_voice(settings, graph[directors[0]].setdefault('_meta', {}))
    if settings.get('voice_mode', 'none') == 'none':
        for key in list(settings):
            if key.startswith('voice_reference_audio'):
                settings.pop(key)
    source_files = {field: settings[field + '_file'] for field in PICTURE_SLOTS if field + '_file' in settings}
    if source_files:
        # A connected image wins loading, but the Director still reads its file
        # widget as a crop/filter hint. Retain that hint for explicit selection
        # checks while preserving legacy no-selection input cleanup below.
        graph[directors[0]].setdefault('_meta', {})[SOURCE_FILES_META] = source_files
    for key in list(settings):
        if key.endswith('_file') and key[:-5] in settings and link(settings[key[:-5]], graph):
            settings.pop(key)
    relevant = ancestors(graph, output)
    director = one(relevant, "MiniMaxH3DirectorPlus")
    sampler = one(relevant, "MiniMaxH3TwoStageSampler")
    guide = one(relevant, "MiniMaxH3DirectorPlusGuide")
    settings = relevant[director]["inputs"]
    if settings.get("community_lora_profile") == "combat":
        from .sampling_experiments import validate_sampling_options
        normalized = dict(settings, performance_preset={"质量优先二采样": "quality_two_stage"}.get(
            settings.get("performance_preset"), settings.get("performance_preset")))
        settings.update(validate_sampling_options(normalized))
    if (settings.get("mode") != "REF2VA" or settings.get("sampling_method", "standard") != "selflift"
            or settings.get("performance_preset") not in {"quality_two_stage", "质量优先二采样"}):
        raise ValueError("长视频当前使用已验证的 REF2VA / 质量优先二采样 / SelfLift 路线，请在原导演台选择这些设置")
    if settings.get("postprocess_mode") != "vosr2" or settings.get("vosr2_profile", "legacy") != "manual_baseline":
        raise ValueError("长视频当前保留实测 VOSR2 manual_baseline，请在原导演台选择该质量配置")
    if settings.get("voice_mode", "none") not in {"none", "h3_reference"}:
        raise ValueError("长视频目前接受无音色或 H3 原生参考音色；Fish/整段音轨复用需要独立长片验证")
    if settings.get("face_refine_mode", "off") != "off" or settings.get("motion_smoothing", "off") not in {"off", "auto"}:
        raise ValueError("长视频的 clean AV 交接不能使用修脸或插帧结果")
    return relevant, director, sampler, guide, output


def build_window_graph(base, request, stage="native", *, parent_store=None):
    if stage not in {"native", "finish", "preview"}:
        raise ValueError("Unknown long-video graph stage")
    if any(key.startswith('h3long_') for key in base):
        raise ValueError('U11 graph uses reserved h3long_ node IDs')
    graph, director, sampler, conditioning, output = roles(base)
    if (original_voice_settings(graph, director).get('voice_mode') == 'fish_lock'
            and request.get('target_audio') is None):
        raise ValueError('Original long Fish requires prepared complete target audio before native execution')
    window = request["window"]
    settings = graph[director]["inputs"]
    settings.update(prompt=request["prompt"], duration=max(4, math.ceil(window["sample_frames"] / 24)),
                    seed=request["seed"])
    bindings = apply_window_reference_policy(settings, {**request, 'stage': stage}, graph, parent_store=parent_store)
    if 'reference_bindings' in request and request['reference_bindings'] != bindings:
        raise ValueError('Actual reference bindings no longer match the immutable take snapshot')
    graph[director].setdefault('_meta', {})['h3_long_video_reference_bindings'] = bindings
    # Automatic original-field graphs retain linked upload names only as
    # preprocessing hints. roles() removed them before asset hashing so a
    # stale hint cannot become a second file asset. Restore the same hint at
    # the physical Director boundary; legacy explicit-selection rules remain.
    metadata = graph[director]['_meta']
    if (metadata.get('h3_long_video_entry') == 'director_duration'
            and request.get('reference_selection') is None):
        for field, source in metadata.get(SOURCE_FILES_META, {}).items():
            if source and link(settings.get(field), graph):
                settings[field + '_file'] = source
    # Runtime consumes the exact immutable take specification. Binding trace is
    # node metadata; callers may snapshot the helper result before saving a take.
    payload = json.dumps({**request, 'stage': stage}, ensure_ascii=False, allow_nan=False)
    revision = request.get('asset_revision')
    if revision:
        # Built-in file loaders cache by name, not weight bytes. Introduce a
        # stable dependency on frozen native identity so same-name replacement
        # reloads weights and every downstream LoRA, while unchanged windows
        # continue to share the loader cache. Re-evaluate hidden uploads too.
        bindings = [(director, 'prompt')]
        bindings.extend((node_id, field) for node_id, node in graph.items()
                        for field in ('unet_name', 'clip_name', 'vae_name')
                        if isinstance(node['inputs'].get(field), str))
        for index, (node_id, field) in enumerate(bindings):
            name = f'h3long_asset_{index}'
            graph[name] = {'class_type': 'MiniMaxH3LongVideoAssetName', 'inputs': {
                'value': graph[node_id]['inputs'][field], 'revision': revision}}
            graph[node_id]['inputs'][field] = [name, 0]
        # Keep LoRA selections literal: the router inspects them via PROMPT.
        # Their MODEL ancestry already includes the revised UNET loader.
        loaders = {node_id: f'h3long_loader_{revision}_{node_id}'
                   for node_id, node in graph.items()
                   if any(field in node['inputs'] for field in
                          ('unet_name', 'clip_name', 'vae_name', 'lora_name', 'stack_data'))}
        # Comfy caches node instances separately by ID/class. In particular,
        # LoraLoader keeps loaded_lora tensors on its instance by file path.
        # A weight revision needs both new output and new instance identity.
        for node_id, node in graph.items():
            if node_id in loaders:
                node.setdefault('_meta', {})['h3_long_video_source_node_id'] = node_id
            for field, value in node['inputs'].items():
                if link(value, graph) and value[0] in loaders:
                    node['inputs'][field] = [loaders[value[0]], value[1]]
        graph = {loaders.get(node_id, node_id): node for node_id, node in graph.items()}
    old_guide = [director, 0]
    for node in graph.values():
        for name, value in list(node["inputs"].items()):
            if value == old_guide:
                node["inputs"][name] = ["h3long_window", 0]
        if node["class_type"] == "RandomNoise":
            node["inputs"]["noise_seed"] = request["seed"]
    graph["h3long_window"] = {"class_type": "MiniMaxH3LongVideoWindow", "inputs": {
        "guide": old_guide, "request_json": payload}}
    if stage == "finish":
        graph[sampler] = {"class_type": "MiniMaxH3LongVideoLoadAV", "inputs": {"request_json": payload}}
        graph[output]["class_type"] = "MiniMaxH3LongVideoExport"
        graph[output]["inputs"].pop("long_video", None)
        graph[output]["inputs"]["request_json"] = payload
        return ancestors(graph, output)
    source_input = graph[sampler]["inputs"]["latent_image"]
    graph["h3long_context"] = {"class_type": "MiniMaxH3LongVideoContext", "inputs": {
        "latent": source_input, "guide": ["h3long_window", 0], "request_json": payload,
        "model": graph[sampler]["inputs"]["second_model"]}}
    if stage == 'native' and request.get('target_audio') is not None and not request.get('repair'):
        graph['h3long_target_audio'] = {'class_type': 'MiniMaxH3LongVideoTargetAudio', 'inputs': {'request_json': payload}}
        graph['h3long_audio_encode'] = {'class_type': 'VAEEncodeAudio', 'inputs': {
            'audio': ['h3long_target_audio', 0], 'vae': graph[conditioning]['inputs']['audio_vae']}}
        graph['h3long_context']['inputs']['target_audio'] = ['h3long_audio_encode', 0]
    graph[sampler]["inputs"]["latent_image"] = ["h3long_context", 0]
    if stage == "preview":
        graph[sampler] = {"class_type": "MiniMaxH3LongVideoLoadAV", "inputs": {"request_json": payload}}
        prepared = [sampler, 0]
    else:
        prepared = ["h3long_context", 0]
    inputs = graph[conditioning]["inputs"]
    graph["h3long_native"] = {"class_type": "MiniMaxH3LongVideoCheckpoint", "inputs": {
        "latent": [sampler, 0], "prepared": prepared, "guide": ["h3long_window", 0],
        "video_vae": inputs["video_vae"], "audio_vae": inputs["audio_vae"],
        "request_json": payload}}
    return ancestors(graph, "h3long_native")


def generation_settings(base):
    graph, director, sampler, _, _ = roles(base)
    graph = ancestors(graph, sampler)
    settings = graph[director]["inputs"]
    # H3's native SelfLift canvas depends on aspect and requested canvas, not
    # on the final SR size. Preserve its derived canvas in the generation key.
    from .sampling_experiments import plan_selflift_dimensions
    width, height = output_canvas(settings)
    canvas = plan_selflift_dimensions(width, height, 13, 32, 28,
                                    community_lora_profile=settings.get("community_lora_profile", "off"))
    for key in ("width", "height", "resolution_preset", "postprocess_mode", "vosr2_profile",
                "rtx_quality", "ai_upscale_model", "audio_loudness", "duration", "prompt", "seed"):
        settings.pop(key, None)
    graph[director]["inputs"]["native_canvas"] = [canvas["second_stage_width"], canvas["second_stage_height"]]
    for node in graph.values():
        node.pop("_meta", None)
    return graph


def output_canvas(settings):
    from .resolution import calculate_resolution
    return calculate_resolution(settings.get('resolution_preset', '0.83 MP'),
                                settings.get('aspect_ratio', '16:9'),
                                int(settings.get('custom_width', 16)),
                                int(settings.get('custom_height', 9)))


def freeze_assets(graph, folder_paths):
    """Hash the files really referenced by this graph, including model weights."""
    from pathlib import Path
    from .longfilm_media import sha256_file
    records, cache = {}, {}

    def record(label, path):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError('Referenced asset is missing: ' + str(path))
        identity = str(path)
        if identity not in cache:
            cache[identity] = {'path': identity, 'sha256': sha256_file(path), 'bytes': path.stat().st_size}
        records[label] = cache[identity]

    for node_id, node in graph.items():
        categories = {'unet_name': 'diffusion_models', 'clip_name': 'text_encoders',
                      'vae_name': 'vae', 'lora_name': 'loras'}
        for key, value in node["inputs"].items():
            if 'lora' in node['class_type'].lower() and key in {'stack_data','lora_stack','loras'}:
                try:
                    stack = json.loads(value) if isinstance(value, str) else value
                except (TypeError, ValueError):
                    stack = None
                if isinstance(stack, list):
                    for index, entry in enumerate(stack):
                        if not isinstance(entry, dict) or entry.get('on', True) is False:
                            continue
                        name = entry.get('lora') or entry.get('lora_name')
                        if not name or str(name).lower() == 'none':
                            continue
                        file = folder_paths.get_full_path('loras', name)
                        if not file:
                            raise ValueError('Referenced LoRA is missing: ' + str(name))
                        record(f'{node_id}:{key}:{index}', file)
            if not isinstance(value, str) or not value or value.lower() == "none":
                continue
            path = None
            if key in categories:
                path = folder_paths.get_full_path(categories[key], value)
                if not path:
                    raise ValueError('Referenced model is missing: ' + value)
            elif key.endswith("_file") or (node["class_type"] in {"LoadImage", "LoadAudio"} and key in {"image", "audio"}):
                path = folder_paths.get_annotated_filepath(value)
            if path:
                record(f'{node_id}:{key}', path)
    return records
