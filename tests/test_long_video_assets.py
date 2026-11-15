"""Real small model files exercise implicit selection and scoped fingerprints."""
import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from nodes.additional_loras import MYSTIC_V5_FL, MYSTIC_V5_REF
from nodes.sampling_experiments import MOTION_LORA_CANDIDATES
from nodes.two_stage_assets import (
    SELFLIFT_DMAD_LORA_CANDIDATES, SELFLIFT_UPSCALER_MODEL_CANDIDATES,
    SELFLIFT_MYSTIC_LORA, V4_TURBO_LORA_PRUNED,
)

ROOT = Path(__file__).resolve().parents[1]


def module():
    assert (ROOT / 'nodes/long_video_assets.py').is_file(), 'Implicit asset provenance is missing'
    return importlib.import_module('nodes.long_video_assets')


class Registry:
    def __init__(self, root):
        self.models_dir = str(root)
        self.entries = {'loras': {}, 'latent_upscale_models': {}}

    def install(self, category, name, content=b'weights'):
        path = Path(self.models_dir) / category / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        self.entries[category][name] = str(path)
        return path

    def get_full_path(self, category, name):
        path = self.entries.get(category, {}).get(name)
        return path if path and Path(path).is_file() else None

    def get_filename_list(self, category):
        return list(self.entries.get(category, {}))


@pytest.fixture
def registry(tmp_path):
    value = Registry(tmp_path)
    value.install('loras', 'vendor/' + SELFLIFT_DMAD_LORA_CANDIDATES[0])
    value.install('loras', 'vendor/' + SELFLIFT_MYSTIC_LORA)
    value.install('latent_upscale_models', 'vendor/' + SELFLIFT_UPSCALER_MODEL_CANDIDATES[0])
    return value


def graph(profile='continuity', extra=None, preset='2K QHD'):
    result = {
        'base': {'class_type': 'UNETLoader', 'inputs': {}},
        'director': {'class_type': 'MiniMaxH3DirectorPlus', 'inputs': {
            'mode': 'REF2VA', 'sampling_method': 'selflift', 'performance_preset': 'quality_two_stage',
            'community_lora_profile': profile, 'postprocess_mode': 'vosr2',
            'vosr2_profile': 'manual_baseline', 'resolution_preset': preset, 'aspect_ratio': '16:9',
        }},
        'router': {'class_type': 'MiniMaxH3AccelerationRouter', 'inputs': {'model': ['base', 0]}},
    }
    if extra:
        result['extra'] = {'class_type': 'DaSiWa_LTX2LoraLoader', 'inputs': {
            'model': ['base', 0], 'model_type': 'Basic',
            'stack_data': json.dumps([{'on': True, 'lora': extra, 'str': 0.45}]),
        }}
        result['router']['inputs']['model'] = ['extra', 0]
    return result


def paths(snapshot):
    return {Path(record['path']).name for record in snapshot['records'].values()}


def test_author_freezes_seven_exact_assets_and_strengths_without_optional_env(registry):
    from test_wenwu_recipe import LORAS, UPSCALER
    for name, _ in LORAS:
        registry.install('loras', 'author/' + name)
    registry.install('latent_upscale_models', 'author/' + UPSCALER)
    original = graph('combat')
    original['director']['inputs']['sampling_method'] = 'standard'
    assets = module()
    frozen = assets.freeze_native_assets(original, registry, environ={'MMH3_DETAIL_LORAS': '0', 'MMH3_EXTRA_LORAS': '0'})
    assert paths(frozen) == {name for name, _ in LORAS} | {UPSCALER}
    choices = frozen['choices']['selections']
    actual = {Path(item['selected'][0]['name']).name: item['strength'] for item in choices.values() if item['category'] == 'loras'}
    assert actual == dict(LORAS)
    assert frozen['choices']['author_recipe_id']
    assert original['director']['inputs']['sampling_method'] == 'standard'
    registry.install('loras', 'author/' + LORAS[-1][0], b'updated speed')
    assert assets.freeze_native_assets(original, registry, environ={})['records'] != frozen['records']


def test_author_fingerprint_tracks_installation_of_exact_fro_without_accepting_plain_dmad(registry):
    assets = module()
    absent = assets.native_asset_choices(graph('combat'), registry, environ={})
    assert any('minimax_h3_dmad_full_dareties_v4_step600_comfy_fro' in label for label in absent['missing'])
    registry.install('loras', 'new/minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors')
    present = assets.native_asset_choices(graph('combat'), registry, environ={})
    assert absent != present


def test_selected_optional_motion_weights_and_actual_bytes_are_frozen(registry):
    assets = module()
    motion = registry.install('loras', 'vendor/' + MOTION_LORA_CANDIDATES[0], b'motion-A')
    snapshot = assets.freeze_native_assets(graph(), registry, environ={})
    assert MOTION_LORA_CANDIDATES[0] in paths(snapshot)
    record = next(value for value in snapshot['records'].values() if value['path'] == str(motion.resolve()))
    assert record == {'path': str(motion.resolve()), 'sha256': hashlib.sha256(b'motion-A').hexdigest(), 'bytes': 8}
    motion.write_bytes(b'motion-B')
    changed = assets.freeze_native_assets(graph(), registry, environ={})
    assert changed['records'] != snapshot['records']
    assert changed['choices'] == snapshot['choices']


def test_optional_2d_lifter_installation_and_weight_bytes_are_frozen_only_for_ordinary_route(registry):
    from nodes.selflift_lifter import MODEL_NAME
    assets = module()
    absent = assets.freeze_native_assets(graph('off'), registry, environ={})
    author = assets.freeze_native_assets(graph('combat'), registry, environ={})
    model = registry.install('latent_upscale_models', 'vendor/' + MODEL_NAME, b'2d-A')
    present = assets.freeze_native_assets(graph('off'), registry, environ={})
    assert MODEL_NAME in paths(present)
    assert present['choices'] != absent['choices']
    assert not present['choices']['missing']
    model.write_bytes(b'2d-B')
    changed = assets.freeze_native_assets(graph('off'), registry, environ={})
    assert changed['records'] != present['records']
    assert assets.freeze_native_assets(graph('combat'), registry, environ={}) == author


def test_installation_and_first_registered_candidate_changes_are_visible_without_hashing(registry):
    assets = module()
    absent = assets.native_asset_choices(graph(), registry, environ={})
    registry.install('loras', 'pack/' + MOTION_LORA_CANDIDATES[1])
    fallback = assets.native_asset_choices(graph(), registry, environ={})
    assert absent != fallback
    registry.install('loras', 'other/' + MOTION_LORA_CANDIDATES[0])
    preferred = assets.native_asset_choices(graph(), registry, environ={})
    assert preferred != fallback
    snapshot = assets.freeze_native_assets(graph(), registry, environ={})
    assert MOTION_LORA_CANDIDATES[0] in paths(snapshot)
    assert MOTION_LORA_CANDIDATES[1] not in paths(snapshot)


def test_combat_replaces_old_optional_motion_with_exact_author_chain_while_off_stays_ordinary(registry):
    assets = module()
    registry.install('loras', MOTION_LORA_CANDIDATES[0])
    registry.install('loras', 'H3_Combat_V2.safetensors')
    registry.install('loras', 'Motion_Repair_V2.safetensors')
    combat = assets.freeze_native_assets(graph('combat'), registry, environ={})
    off = assets.freeze_native_assets(graph('off'), registry, environ={})
    assert {'H3_Combat_V2.safetensors', 'Motion_Repair_V2.safetensors'} <= paths(combat)
    assert MOTION_LORA_CANDIDATES[0] not in paths(combat)
    assert 'H3_Combat_V2.safetensors' not in paths(off)
    assert MOTION_LORA_CANDIDATES[0] not in paths(off)
    assert SELFLIFT_MYSTIC_LORA in paths(off)


def test_core_fallback_choices_track_new_dmad_and_lms_installations(tmp_path):
    assets = module()
    registry = Registry(tmp_path)
    registry.install('loras', 'pack/' + V4_TURBO_LORA_PRUNED)
    registry.install('loras', SELFLIFT_MYSTIC_LORA)
    registry.install('latent_upscale_models', SELFLIFT_UPSCALER_MODEL_CANDIDATES[2])
    fallback = assets.freeze_native_assets(graph('off'), registry, environ={})
    assert V4_TURBO_LORA_PRUNED in paths(fallback)
    registry.install('loras', SELFLIFT_DMAD_LORA_CANDIDATES[0])
    registry.install('latent_upscale_models', SELFLIFT_UPSCALER_MODEL_CANDIDATES[0])
    preferred = assets.freeze_native_assets(graph('off'), registry, environ={})
    assert fallback['choices'] != preferred['choices']
    assert SELFLIFT_DMAD_LORA_CANDIDATES[0] in paths(preferred)
    assert V4_TURBO_LORA_PRUNED not in paths(preferred)


def test_ref_route_freezes_effective_mystic_variant_even_when_upstream_requests_fl(registry):
    assets = module()
    registry.install('loras', 'vendor/' + MYSTIC_V5_FL, b'FL variant')
    effective = registry.install('loras', 'vendor/' + MYSTIC_V5_REF, b'REF variant')
    original_graph = graph(extra='vendor/' + MYSTIC_V5_FL)
    saved_graph = copy.deepcopy(original_graph)
    before = assets.freeze_native_assets(original_graph, registry, environ={})
    assert MYSTIC_V5_REF in paths(before)
    assert MYSTIC_V5_FL in paths(before)
    assert SELFLIFT_MYSTIC_LORA in paths(before)  # SelfLift may restore its clean parent and U22 detail.
    effective.write_bytes(b'new REF variant')
    assert assets.freeze_native_assets(original_graph, registry, environ={})['records'] != before['records']
    assert original_graph == saved_graph


def test_missing_effective_variant_is_recorded_and_installation_changes_choices(registry):
    assets = module()
    registry.install('loras', MYSTIC_V5_FL)
    before = assets.freeze_native_assets(graph(extra=MYSTIC_V5_FL), registry, environ={})
    assert any('mystic' in key for key in before['choices']['missing'])
    registry.install('loras', MYSTIC_V5_REF)
    after = assets.freeze_native_assets(graph(extra=MYSTIC_V5_FL), registry, environ={})
    assert before['choices'] != after['choices']


def test_detail_environment_is_a_real_selection_input(registry):
    assets = module()
    on = assets.freeze_native_assets(graph('off'), registry, environ={'MMH3_DETAIL_LORAS': '1'})
    off = assets.freeze_native_assets(graph('off'), registry, environ={'MMH3_DETAIL_LORAS': '0'})
    assert on['choices'] != off['choices']
    assert SELFLIFT_MYSTIC_LORA in paths(on)
    assert SELFLIFT_MYSTIC_LORA not in paths(off)


def install_bundle(registry):
    root = Path(registry.models_dir) / 'vosr2' / 'VOSR2'
    for name in ['args.json', 'checkpoints/ema_model.safetensors', 'Qwen-Image-vae-2d/config.json',
                 'Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors', 'dinov2_vitl14.safetensors']:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(name.encode())
    return root


def test_actual_implicit_vosr2_weights_and_configs_belong_only_to_export_fingerprint(registry):
    assets = module()
    root = install_bundle(registry)
    native = assets.freeze_native_assets(graph('off'), registry, environ={})
    export = assets.freeze_export_assets(graph('off'), registry)
    assert len(export['records']) == 5
    assert not export['choices']['missing']
    for name in ['checkpoints/ema_model.safetensors', 'Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors',
                 'dinov2_vitl14.safetensors', 'args.json', 'Qwen-Image-vae-2d/config.json']:
        path = root / name
        prior = assets.freeze_export_assets(graph('off'), registry)
        path.write_bytes(path.read_bytes() + b'changed')
        assert assets.freeze_export_assets(graph('off'), registry)['records'] != prior['records']
        assert assets.freeze_native_assets(graph('off'), registry, environ={}) == native


def test_documented_dit_and_vision_fallback_candidates_are_tracked(registry):
    assets = module()
    root = install_bundle(registry)
    prior = assets.freeze_export_assets(graph('off'), registry)
    extra = root / 'clean_weights/ema_model.safetensors'
    extra.parent.mkdir()
    extra.write_bytes(b'clean')
    changed = assets.freeze_export_assets(graph('off'), registry)
    assert changed['choices'] != prior['choices']
    # Legacy opaque loaders do not expose a safe priority: cover every installed
    # documented weight candidate instead of claiming an unverified winner.
    assert str(extra.resolve()) in {value['path'] for value in changed['records'].values()}
    vision = root / 'dinov2_vitl14.safetensors'
    vision.rename(root / 'dinov2_vitl14_pretrain.pth')
    fallback = assets.freeze_export_assets(graph('off'), registry)
    assert 'dinov2_vitl14_pretrain.pth' in paths(fallback)


def test_missing_bundle_is_represented_without_downloading_or_loading_models(registry):
    assets = module()
    before = assets.freeze_export_assets(graph('off'), registry)
    assert before['records'] == {}
    assert before['choices']['missing']
    assert not (Path(registry.models_dir) / 'vosr2').exists()
    install_bundle(registry)
    after = assets.export_asset_choices(graph('off'), registry)
    assert before['choices'] != after


def test_existing_fhd_bypass_never_requires_vosr2_paths_or_plugin():
    assets = module()

    class NoModelAccess:
        @property
        def models_dir(self):
            pytest.fail('A bypass must not inspect or require VOSR2')

    g = graph('off', preset='1080p FHD')
    out = assets.freeze_export_assets(g, NoModelAccess(), source_size=(1920, 1056))
    assert out['records'] == {}
    assert out['choices']['bypassed'] is True
    assert out['choices']['missing'] == []
    assert assets.freeze_export_assets(g, NoModelAccess(), bypass=True)['records'] == {}


def test_bundle_resolver_can_report_the_actual_external_backend_layout(registry, tmp_path):
    assets = module()
    path = tmp_path / 'external-actual-model.bin'
    path.write_bytes(b'actual')
    calls = []

    def resolver(folder_paths, name):
        calls.append(name)
        return {'actual_model': [str(path)]}

    out = assets.freeze_export_assets(graph('off'), registry, bundle_resolver=resolver)
    assert calls == ['VOSR2']
    assert paths(out) == {'external-actual-model.bin'}


def test_custom_native_resolver_is_used_for_effective_candidate_selection(registry):
    assets = module()
    path = registry.install('loras', 'external/Motion_Repair.safetensors')
    calls = []

    def resolver(category, name):
        calls.append((category, name))
        return 'external/Motion_Repair.safetensors' if name == MOTION_LORA_CANDIDATES[0] else None

    out = assets.freeze_native_assets(graph(), registry, resolver=resolver, environ={})
    assert ('loras', MOTION_LORA_CANDIDATES[0]) in calls
    assert str(path.resolve()) in {value['path'] for value in out['records'].values()}


def test_choices_are_json_serializable_lightweight_and_hashes_deduplicate(registry):
    assets = module()
    install_bundle(registry)
    choices = assets.export_asset_choices(graph('off'), registry)
    json.dumps(choices, allow_nan=False)
    calls = []

    def hasher(path):
        calls.append(str(path))
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    out = assets.freeze_asset_choices(choices, hasher=hasher)
    assert len(calls) == len(set(calls)) == 5
    assert out['choices'] == choices
