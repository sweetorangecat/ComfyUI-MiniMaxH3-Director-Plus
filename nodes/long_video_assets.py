"""Read-only provenance for models loaded implicitly by the U11 route.

Choice inspection never imports Torch, constructs a model, downloads a file or
requires a VOSR2 plugin. Hashing is a separate operation so queue guards can
recheck model registration and candidate installation cheaply.

The documented VOSR2 bundle paths below were verified against
ylchen333/ComfyUI-VOSR2 loader.py at 6a21810b3f4f1a0ffc477c55330db0ac03c671f0
and the installed TE-Speed-VOSR2 README (which additionally accepts DINO .pth).
The legacy compiled loader's exact priority is opaque: by default all installed
documented candidates are hashed. A backend can supply its actual component
paths with bundle_resolver; none of these helpers executes the backend loader.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import re

from .additional_loras import mystic_v5_selection, upstream_loras
from .sampling_experiments import motion_lora_chain, validate_sampling_options
from .wenwu_recipe import AUTHOR_ASSETS, AUTHOR_RECIPE_ID, AUTHOR_RECIPE_VERSION, is_author_recipe, validate_manual_loras
from .two_stage_assets import (
    SELFLIFT_DMAD_LORA_CANDIDATES, SELFLIFT_DMAD_LORA_STRENGTH,
    SELFLIFT_MYSTIC_LORA, SELFLIFT_MYSTIC_LORA_STRENGTH,
    SELFLIFT_UPSCALER_MODEL_CANDIDATES, V4_TURBO_LORA, V4_TURBO_LORA_PRUNED,
)


def freeze_fish_backend(requested_model, folder_paths, *, node_source=None, speech_source=None):
    """Inspect installed files only; never call resolve_model_path/load_engine."""
    from .long_video_audio import file_identity
    name = re.sub(r' \(auto download\)$', '', str(requested_model))
    if not re.fullmatch(r'[\w.-]+', name) or name in {'.', '..'}:
        raise ValueError('Fish requires one installed model directory name')
    directory = Path(folder_paths.models_dir) / 'fishaudioS2' / name
    if not directory.is_dir():
        raise ValueError('Fish model is not installed; native audio preparation never downloads weights: ' + name)
    weights = sorted(path for path in directory.rglob('*')
                     if path.is_file() and path.suffix.lower() in {'.safetensors', '.pt', '.pth', '.ckpt', '.bin'})
    if not weights or not (directory / 'config.json').is_file():
        raise ValueError('Fish requires actual installed model weights and config.json')
    decoder = next((root / candidate for root in (directory, directory.parent)
                    for candidate in ('codec.pth', 'firefly-gan-vq-fsq-8x1024-21hz-generator.pth', 'decoder.pth', 'vocoder.pth')
                    if (root / candidate).is_file()), None)
    if decoder is None:
        raise ValueError('Fish requires an installed codec/decoder checkpoint')
    model_files = sorted(set([*weights, decoder, *[path for path in directory.rglob('*')
                        if path.is_file() and path.suffix.lower() in {'.json', '.jinja', '.model', '.tiktoken'}]]))
    if node_source is None:
        import inspect
        import nodes as comfy_nodes
        node_type = getattr(comfy_nodes, 'NODE_CLASS_MAPPINGS', {}).get('FishS2VoiceCloneTTS')
        if node_type is None:
            raise ValueError('FishS2VoiceCloneTTS backend is not installed')
        node_source = inspect.getsourcefile(node_type)
    if not node_source or not Path(node_source).is_file():
        raise ValueError('Fish requires actual installed backend code identity')
    if speech_source is None:
        import importlib.util
        package = importlib.util.find_spec('fish_speech')
        if package is None or not package.submodule_search_locations:
            raise ValueError('Installed fish_speech code is missing; model provenance cannot be verified')
        speech_source = next(iter(package.submodule_search_locations))
    code_roots = {'plugin': Path(node_source).parent.parent, 'fish_speech': Path(speech_source)}
    codes = {f'{label}/{file.relative_to(root).as_posix()}': file_identity(file)
             for label, root in code_roots.items() for file in sorted(root.rglob('*.py')) if file.is_file()}
    if not codes or not any(name.startswith('fish_speech/') for name in codes):
        raise ValueError('Fish requires actual backend/inference code identities')
    return {'backend': 'FishS2VoiceCloneTTS', 'requested_model': str(requested_model),
            'resolved_model': {'name': name, 'path': str(directory.resolve()),
                               'weights': [{**file_identity(path), 'name': path.name} for path in model_files]},
            'code': {name: identity['sha256'] for name, identity in codes.items()}, 'code_files': list(codes.values()),
            'parameters': {'language': 'auto', 'device': 'auto', 'precision': 'auto', 'attention': 'auto',
                           'max_new_tokens': 0, 'chunk_length': 200, 'temperature': .8, 'top_p': .8,
                           'repetition_penalty': 1.1, 'keep_model_loaded': False, 'compile_model': False}}


def freeze_audio_assets(graph, contract, windows, folder_paths):
    from .long_video_audio import file_identity, recipe_for_window
    from .long_video_graph import original_voice_settings
    from .long_video_references import snapshot_bindings
    director = next(key for key, node in graph.items() if node.get('class_type') == 'MiniMaxH3DirectorPlus')
    settings = original_voice_settings(graph, director)
    originals = {}
    for binding in snapshot_bindings(graph[director]['inputs'], graph)['audios']:
        selected = binding['binding']
        if 'file' in selected:
            source = selected['file']
        else:
            node_id, socket = selected['link']
            node = graph[node_id]
            if node['class_type'] != 'LoadAudio' or socket != 0 or not isinstance(node['inputs'].get('audio'), str):
                raise ValueError('Prepared speech requires a directly identifiable original LoadAudio/file binding; transformed audio needs a supplied complete target')
            source = node['inputs']['audio']
        originals[binding['id']] = file_identity(folder_paths.get_annotated_filepath(source))
    recipes = [recipe_for_window(contract, window, settings) for window in windows]
    needs_fish = any(recipe and recipe['kind'] == 'fish' and any(
        line.get('audio_slot') == 1 and line['id'] not in recipe.get('speech_sources', {})
        for line in contract['dialogues'] if window['new_start'] <= line['start_frame'] < line['end_frame'] <= window['new_end'])
        for recipe, window in zip(recipes, windows))
    backend = freeze_fish_backend(settings.get('fish_model_path') or 's2-pro-w4a16 (auto download)', folder_paths) if needs_fish else None
    return {'original_sources': originals, 'fish_backend': backend}


def _settings(graph):
    matches = [node['inputs'] for node in graph.values()
               if node.get('class_type') == 'MiniMaxH3DirectorPlus']
    if len(matches) != 1:
        raise ValueError('Implicit U11 asset provenance requires one director')
    return matches[0]


def _registered_name(registry, category, name):
    """Mirror two_stage_assets.resolve_registered_model_name with an injected registry."""
    try:
        if registry.get_full_path(category, name):
            return name
        for candidate in registry.get_filename_list(category):
            candidate = str(candidate)
            if Path(candidate).name == Path(name).name and registry.get_full_path(category, candidate):
                return candidate
    except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError):
        return None
    return None


def _choice(category, names, registry, resolver, strength=None):
    candidates = []
    for name in names:
        registered = resolver(category, name)
        path = registry.get_full_path(category, registered) if registered is not None else None
        path = str(Path(path).resolve()) if path and Path(path).is_file() else None
        candidates.append({'name': name, 'registered_name': registered, 'path': path})
    selected = next((item for item in candidates if item['path']), None)
    result = {'category': category, 'candidates': candidates,
              'selected': [deepcopy(selected)] if selected else [], 'policy': 'first_registered'}
    if strength is not None:
        result['strength'] = float(strength)
    return result


def native_asset_choices(graph, folder_paths, *, resolver=None, environ=None):
    """Inspect the SelfLift core, explicit motion profile and effective Mystic.

    Pass the original retained API graph, before per-window graph rewriting.
    resolver(category, name) may inject the runtime's registered-model lookup.
    Missing optional candidates are recorded; installing one changes choices.
    The existing route's dependency preflight remains responsible for readiness.
    """
    settings = dict(_settings(graph))
    if is_author_recipe(settings):
        selected = dict(settings, performance_preset={"质量优先二采样": "quality_two_stage"}.get(
            settings.get("performance_preset"), settings.get("performance_preset")))
        settings.update(validate_sampling_options(selected))
    if settings.get('sampling_method', 'standard') != 'selflift':
        raise ValueError('Implicit native asset provenance currently follows the U11 SelfLift route')
    environ = os.environ if environ is None else environ
    detail_on = str(environ.get('MMH3_DETAIL_LORAS', '1')).strip().lower() in {'1', 'true', 'on', 'yes'}
    resolver = resolver or (lambda category, name: _registered_name(folder_paths, category, name))
    profile = settings.get('community_lora_profile', 'legacy')
    result = {'version': 1, 'scope': 'native', 'sampling_method': 'selflift',
              'community_lora_profile': profile, 'detail_loras_enabled': detail_on,
              'selections': {}, 'missing': []}
    selections = result['selections']
    if is_author_recipe(settings):
        result.update(author_recipe_id=AUTHOR_RECIPE_ID, author_recipe_version=AUTHOR_RECIPE_VERSION,
                      detail_loras_enabled=True)
        for asset in AUTHOR_ASSETS:
            selections['implicit:author:' + asset.filename] = _choice(
                asset.category, (asset.filename,), folder_paths, resolver, asset.strength)
        routers = [node_id for node_id, node in graph.items()
                   if node.get('class_type') == 'MiniMaxH3AccelerationRouter']
        if len(routers) > 1:
            raise ValueError('Implicit LoRA provenance requires one active acceleration router')
        rows = upstream_loras(graph, routers[0]) if routers else []
        validate_manual_loras(rows)
        for index, row in enumerate(rows):
            selections[f'implicit:upstream:{index}'] = _choice(
                'loras', (row['name'],), folder_paths, resolver, row['strength'])
        result['missing'] = [label for label, choice in selections.items() if not choice['selected']]
        return result
    core = _choice('loras', (*SELFLIFT_DMAD_LORA_CANDIDATES, V4_TURBO_LORA_PRUNED, V4_TURBO_LORA),
                   folder_paths, resolver)
    selected = core['selected'][0]['name'] if core['selected'] else None
    core['strength'] = SELFLIFT_DMAD_LORA_STRENGTH if selected in SELFLIFT_DMAD_LORA_CANDIDATES else 1.0
    core['recipe'] = 'u22_dmad_mystic' if selected in SELFLIFT_DMAD_LORA_CANDIDATES else 'v4_mystic_fallback' if selected else 'missing'
    selections['implicit:selflift:lora'] = core
    selections['implicit:selflift:upscaler'] = _choice(
        'latent_upscale_models', SELFLIFT_UPSCALER_MODEL_CANDIDATES, folder_paths, resolver)
    # The actual compatible/global route is established at sampling time.
    # Conservatively freeze this optional conditional asset as well, so its
    # installation/removal or replacement cannot silently change later windows.
    from .selflift_lifter import MODEL_NAME
    optional_lifter = _choice('latent_upscale_models', (MODEL_NAME,), folder_paths, resolver)
    optional_lifter.update(optional=True, policy='cover_optional_compatible_global_lifter')
    selections['implicit:selflift:optional_2d_lifter'] = optional_lifter
    if detail_on:
        selections['implicit:selflift:detail'] = _choice(
            'loras', (SELFLIFT_MYSTIC_LORA,), folder_paths, resolver, SELFLIFT_MYSTIC_LORA_STRENGTH)
    for index, (names, strength) in enumerate(motion_lora_chain(profile)):
        selections[f'implicit:motion:{index}'] = _choice('loras', tuple(names), folder_paths, resolver, strength)

    routers = [node_id for node_id, node in graph.items()
               if node.get('class_type') == 'MiniMaxH3AccelerationRouter']
    if len(routers) > 1:
        raise ValueError('Implicit LoRA provenance requires one active acceleration router')
    rows = upstream_loras(graph, routers[0]) if routers else []
    # These are loaded upstream even when SelfLift later restores a clean parent.
    # Include both requested and effective files; graph-only inspection cannot
    # certify a ModelPatcher parent restoration or successful weight application.
    for index, row in enumerate(rows):
        selections[f'implicit:upstream:{index}'] = _choice(
            'loras', (row['name'],), folder_paths, resolver, row['strength'])
    backend = settings.get('resolved_backend') or ('ref2va_model' if settings.get('mode') == 'REF2VA' else 'fl2va_model')
    mystic = mystic_v5_selection(rows, backend)
    result['mystic_v5'] = mystic
    if mystic:
        selections['implicit:mystic:effective'] = _choice(
            'loras', (mystic['effective_name'],), folder_paths, resolver, mystic['strength'])
        result['mystic_policy'] = 'cover_effective_v5_and_selflift_u22_parent_restoration'
    result['missing'] = [label for label, choice in selections.items()
                         if not choice['selected'] and not choice.get('optional')]
    return result


def _documented_vosr2_components(folder_paths, model_bundle):
    root = Path(folder_paths.models_dir) / 'vosr2' / model_bundle
    return {
        'args': [root / 'args.json'],
        'dit': [root / 'clean_weights' / 'ema_model.safetensors',
                root / 'checkpoints' / 'ema_model.safetensors', root / 'ema_model.safetensors'],
        'vae_config': [root / 'Qwen-Image-vae-2d' / 'config.json'],
        'vae': [root / 'Qwen-Image-vae-2d' / 'diffusion_pytorch_model.safetensors'],
        'vision': [root / 'dinov2_vitl14.safetensors', root / 'dinov2_vitl14_pretrain.pth'],
    }


def _bypass_vosr2(settings, source_size, bypass):
    if bypass is not None:
        if type(bypass) is not bool:
            raise ValueError('The actual VOSR2 bypass flag must be boolean')
        return bypass
    if settings.get('postprocess_mode') != 'vosr2':
        return True
    if settings.get('fhd_vosr2_is_redundant') or settings.get('postprocess_path') == 'balanced_fhd_downscale':
        return True
    if source_size is None:
        return False  # A target preset alone does not prove the actual near-FHD bypass.
    from .long_video_graph import output_canvas
    from .vram_budget import is_redundant_fhd_vosr2
    return is_redundant_fhd_vosr2('vosr2', *output_canvas(settings), *source_size)


def export_asset_choices(graph, folder_paths, *, source_size=None, bypass=None, bundle_resolver=None):
    """Inspect actual VOSR2 bundle inputs, separate from generation provenance.

    source_size is the actual native canvas; bypass can carry the director's
    already-resolved decision. A bypass does not inspect plugin/model paths.
    The optional bundle_resolver(folder_paths, bundle_name) returns a mapping
    from component names to ordered candidate paths for a verified backend.
    All existing candidates are covered so opaque legacy priority cannot hide
    a consumed file. Missing files remain explicit without auto-downloading.
    """
    settings = _settings(graph)
    bypassed = _bypass_vosr2(settings, source_size, bypass)
    result = {'version': 1, 'scope': 'export', 'bypassed': bypassed,
              'postprocess_mode': settings.get('postprocess_mode'), 'selections': {}, 'missing': []}
    if bypassed:
        return result
    from .vosr2_profiles import resolve_vosr2_profile
    # All currently authorized profiles use this same concrete bundle.
    bundle = resolve_vosr2_profile('manual_baseline')['loader']['model_bundle']
    result['model_bundle'] = bundle
    result['path_contract'] = 'backend_resolver' if bundle_resolver else 'documented_vosr2_bundle_v1'
    components = (bundle_resolver or _documented_vosr2_components)(folder_paths, bundle)
    if not isinstance(components, dict) or not components:
        raise ValueError('VOSR2 backend asset resolver must return nonempty component candidate paths')
    for label, paths in sorted(components.items()):
        if (not isinstance(label, str) or not label or not isinstance(paths, (list, tuple)) or not paths):
            raise ValueError('VOSR2 components require names and nonempty lists of actual candidate paths')
        candidates = []
        for path in paths:
            if not isinstance(path, (str, Path)) or not str(path):
                raise ValueError('VOSR2 candidate paths must be filesystem paths')
            file = Path(path).resolve()
            candidates.append({'name': str(file), 'path': str(file) if file.is_file() else None})
        key = 'implicit:vosr2:' + label
        result['selections'][key] = {'category': 'vosr2', 'candidates': candidates,
                                     'selected': [deepcopy(item) for item in candidates if item['path']],
                                     'policy': 'cover_all_installed_documented_candidates'}
        if not result['selections'][key]['selected']:
            result['missing'].append(key)
    return result


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_asset_choices(choices, *, hasher=None):
    """Hash existing selected files; return compatible records plus choice metadata.

    Feed records into the matching native/export asset map, and feed choices
    into the same scope's fingerprint. Keep the scopes separate. Existing file
    stat guards cover records; re-running the choice inspector detects newly
    installed optional candidates, deleted files and resolver order changes.
    """
    choices = deepcopy(choices)
    records, cache = {}, {}
    hash_file = hasher or _sha256_file
    for label, choice in choices['selections'].items():
        for index, selected in enumerate(choice['selected']):
            file = Path(selected['path']).resolve()
            path = str(file)
            if path not in cache:
                if not file.is_file():
                    raise ValueError('Selected implicit model asset disappeared: ' + path)
                before = file.stat()
                digest = hash_file(file)
                after = file.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                    raise ValueError('Implicit model asset changed while hashing: ' + path)
                cache[path] = {'path': path, 'sha256': digest, 'bytes': after.st_size}
            records[label + (f':{index}' if len(choice['selected']) > 1 else '')] = deepcopy(cache[path])
    return {'records': records, 'choices': choices}


def freeze_native_assets(graph, folder_paths, *, resolver=None, environ=None, hasher=None):
    return freeze_asset_choices(native_asset_choices(graph, folder_paths, resolver=resolver, environ=environ), hasher=hasher)


def freeze_export_assets(graph, folder_paths, *, source_size=None, bypass=None, bundle_resolver=None, hasher=None):
    return freeze_asset_choices(export_asset_choices(graph, folder_paths, source_size=source_size,
                               bypass=bypass, bundle_resolver=bundle_resolver), hasher=hasher)
