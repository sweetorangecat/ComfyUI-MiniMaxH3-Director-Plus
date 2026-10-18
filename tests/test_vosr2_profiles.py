"""Contracts for explicit, measurable VOSR2 configurations."""
import copy
import json
import sys
import types

import pytest


def test_legacy_profile_keeps_existing_settings_and_is_default():
    from nodes.vosr2_profiles import VOSR2_PROFILE_NAMES, resolve_vosr2_profile
    assert VOSR2_PROFILE_NAMES == ('legacy', 'manual_baseline', 'experimental_32gb')
    profile = resolve_vosr2_profile()
    assert profile['name'] == 'legacy'
    assert profile['loader'] == dict(model_bundle='VOSR2', precision='auto',
        memory_policy='auto', torch_compile=False, vae_encode_amp=False)
    assert profile['settings'] == dict(quality_profile='speed', tile_strategy='auto',
        tile_size=512, tile_overlap=32, vae_tile_size=1024, vae_tile_overlap=32,
        image_batch=1, frame_batch=2, dino_batch=2, temporal_cache=False,
        cache_threshold=0.003, cache_refresh=4, memory_policy='auto', color_alignment='wavelet')
    assert profile['video'] == dict(frame_batch=2, temporal_cache=False,
        cache_threshold=0.003, cache_refresh=4)
    profile['settings']['frame_batch'] = 99
    assert resolve_vosr2_profile()['settings']['frame_batch'] == 2


def test_manual_baseline_is_fixed_tile_without_cache_or_compile():
    from nodes.vosr2_profiles import resolve_vosr2_profile
    p = resolve_vosr2_profile('manual_baseline')
    assert p['settings']['quality_profile'] == 'manual'
    assert p['settings']['tile_strategy'] == 'tiled'
    assert p['settings']['tile_size'] == 512
    assert p['settings']['vae_tile_size'] == 1024
    assert p['settings']['memory_policy'] == p['loader']['memory_policy'] == 'staged'
    assert p['loader']['torch_compile'] is False
    assert p['video']['temporal_cache'] is False
    assert p['video']['frame_batch'] == p['settings']['frame_batch'] == 2


@pytest.mark.parametrize('total,free', [(None,None),(32,None),(None,28),(24,23),
    (32,12),(32,float('nan')),(float('inf'),28),(32,33),(True,28),(32,'28')])
def test_experimental_profile_requires_valid_measured_total_and_free(total, free):
    from nodes.vosr2_profiles import resolve_vosr2_profile
    with pytest.raises(ValueError, match='experimental_32gb'):
        resolve_vosr2_profile('experimental_32gb', total_vram_gb=total, free_vram_gb=free)


def test_experimental_profile_is_only_selected_explicitly():
    from nodes.vosr2_profiles import resolve_vosr2_profile
    assert resolve_vosr2_profile(total_vram_gb=32, free_vram_gb=28)['name'] == 'legacy'
    p = resolve_vosr2_profile('experimental_32gb', total_vram_gb=32, free_vram_gb=28)
    assert p['experimental'] is True
    assert p['settings']['frame_batch'] == p['video']['frame_batch'] == 4
    assert p['settings']['dino_batch'] == 4
    assert p['settings']['tile_size'] == 512
    assert p['settings']['temporal_cache'] is p['loader']['torch_compile'] is False


@pytest.mark.parametrize('profile', ['fastest', '', None])
def test_unknown_profile_is_rejected(profile):
    from nodes.vosr2_profiles import resolve_vosr2_profile
    with pytest.raises(ValueError, match='VOSR2'):
        resolve_vosr2_profile(profile)


@pytest.fixture
def vosr_adapter(monkeypatch):
    import torch
    from nodes import two_stage_assets, video_finish
    seen = {'events': []}
    fake = types.ModuleType('nodes.stream_output')
    fake._release_comfy_models_before_video_sr = lambda: seen['events'].append('release')
    fake._unwrap_node_result = lambda value: value[0]
    fake._resize_cpu_chunk = lambda images, w, h, method: torch.zeros(len(images),h,w,3)
    fake._iter_video_sr_frame_chunks = lambda *args, **kwargs: iter(())
    monkeypatch.setitem(sys.modules, 'nodes.stream_output', fake)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(video_finish, '_vosr2_memory_snapshot',
        lambda: seen['events'].append('memory') or {'total_vram_gb':32, 'free_vram_gb':28}, raising=False)
    class Loader:
        FUNCTION = 'load'
        def load(self, **kwargs):
            seen['events'].append('load'); seen['loader'] = kwargs
            return ({},)
    class Settings:
        FUNCTION = 'make'
        def make(self, **kwargs):
            seen['settings'] = kwargs
            return (kwargs,)
    class Video:
        FUNCTION = 'upscale'
        def upscale(self, **kwargs):
            seen['video'] = kwargs
            if seen.get('fail'):
                raise RuntimeError('simulated OOM')
            images = kwargs['images']
            return (torch.zeros(len(images) - seen.get('lost_frames', 0),
                images.shape[1]*kwargs['scale'],images.shape[2]*kwargs['scale'],3),)
    monkeypatch.setattr(two_stage_assets, '_comfy_node_mappings', lambda: {
        'TESpeedVOSR2Loader':Loader,'TESpeedVOSR2Settings':Settings,'TESpeedVOSR2Video':Video})
    return video_finish, seen


@pytest.mark.parametrize('profile', ['legacy','manual_baseline','experimental_32gb'])
def test_adapter_forwards_profile_and_measures_result(vosr_adapter, profile):
    import torch
    from nodes.vosr2_profiles import resolve_vosr2_profile
    module, seen = vosr_adapter
    diagnostic = {}
    images = torch.zeros(3,8,12,3,dtype=torch.float16)
    result = module._vosr2(images,18,12,42,profile=profile,diagnostics=diagnostic)
    expected = resolve_vosr2_profile(profile,total_vram_gb=32,free_vram_gb=28)
    assert seen['loader'] == expected['loader']
    assert seen['settings'] == expected['settings']
    assert all(seen['video'][key] == value for key,value in expected['video'].items())
    assert seen['video']['seed'] == 42
    assert seen['video']['images'].dtype == torch.float32
    assert result.shape == (3,16,24,3)
    assert seen['events'][:3] == ['release','memory','load']
    assert diagnostic['status'] == 'completed'
    assert diagnostic['profile'] == profile
    assert diagnostic['requested_config'] == expected
    assert diagnostic['output_shape'] == [3,16,24,3]
    assert diagnostic['elapsed_seconds'] >= 0
    assert diagnostic['memory_before']['free_vram_gb'] == 28
    json.dumps(diagnostic, allow_nan=False)


def test_experimental_rejects_low_free_memory_before_loading(vosr_adapter, monkeypatch):
    import torch
    module, seen = vosr_adapter
    monkeypatch.setattr(module, '_vosr2_memory_snapshot', lambda: {'total_vram_gb':32,'free_vram_gb':12})
    diagnostic = {}
    with pytest.raises(ValueError, match='experimental_32gb'):
        module._vosr2(torch.zeros(2,8,12,3),24,16,42,'experimental_32gb',diagnostic)
    assert 'loader' not in seen
    assert diagnostic['status'] == 'failed'


def test_failed_adapter_never_records_completed_timing(vosr_adapter):
    import torch
    module, seen = vosr_adapter
    seen['fail'] = True
    diagnostic = {}
    with pytest.raises(RuntimeError, match='simulated OOM'):
        module._vosr2(torch.zeros(2,8,12,3),24,16,42,diagnostics=diagnostic)
    assert diagnostic['status'] == 'failed'
    assert diagnostic['error_type'] == 'RuntimeError'
    assert 'output_shape' not in diagnostic


def test_adapter_rejects_changed_frame_count(vosr_adapter):
    import torch
    module, seen = vosr_adapter
    seen['lost_frames'] = 1
    with pytest.raises(RuntimeError, match='帧数'):
        module._vosr2(torch.zeros(3,8,12,3),24,16,42)


def test_adapter_rejects_undersized_reconstruction(vosr_adapter, monkeypatch):
    import torch
    module, seen = vosr_adapter
    monkeypatch.setattr(module, 'vosr2_input_size', lambda *args: (12,8,1))
    diagnostics = {}
    with pytest.raises(RuntimeError, match='最低分辨率'):
        module._vosr2(torch.zeros(3,8,12,3),24,16,42,diagnostics=diagnostics)
    assert diagnostics['status'] == 'failed'


def test_finish_exposes_diagnostics_without_changing_image_output(vosr_adapter,monkeypatch):
    import torch
    module, seen = vosr_adapter
    monkeypatch.setattr(module,'output_size',lambda w,h,preset:(18,12))
    result = module.MiniMaxH3VideoFinish().finish(torch.zeros(3,8,12,3),vosr2_profile='manual_baseline',benchmark_run='run-1')
    assert result['result'][0].shape == (3,16,24,3)
    assert result['ui']['vosr2_diagnostics'][0]['profile'] == 'manual_baseline'
    inputs = module.MiniMaxH3VideoFinish.INPUT_TYPES()
    assert list(inputs['required']) == ['images','resolution','method','seed']
    assert inputs['optional']['vosr2_profile'][1]['default'] == 'legacy'
    ordinary = module.MiniMaxH3VideoFinish().finish(torch.zeros(3,8,12,3))
    assert isinstance(ordinary, tuple)
    assert ordinary[0].shape == (3,16,24,3)


def test_finish_preserves_source_lineage_and_adds_measured_reconstruction(vosr_adapter,monkeypatch):
    import torch
    from nodes import execution_report as evidence
    module, seen = vosr_adapter
    monkeypatch.setattr(module,'output_size',lambda w,h,preset:(18,12))
    source = torch.zeros(3,8,12,3)
    report = evidence.new_report({'mode':'T2VA'})
    with evidence.stage_timer(report,'sampling',steps=8):
        pass
    evidence.attach_report(source,report)
    output = module.MiniMaxH3VideoFinish().finish(source,vosr2_profile='manual_baseline')[0]
    result = evidence.report_from(output)
    assert result is not None
    assert [stage['stage'] for stage in result['stages']] == ['sampling','reconstruction']
    assert result['stages'][0]['origin'] == 'upstream_artifact'
    assert result['stages'][1]['status'] == 'completed'
    assert result['reconstruction']['profile'] == 'manual_baseline'
    assert len(evidence.report_from(source)['stages']) == 1


def test_benchmark_builds_same_short_clip_and_seed_for_each_profile():
    from tools.benchmark_vosr2 import build_workflow
    first = build_workflow('/server/input/clip.mp4','legacy',seed=42,fps=24,duration=5,
        resolution='1080p',run_id='run-a',output_prefix='comparisons')
    second = build_workflow('/server/input/clip.mp4','manual_baseline',seed=42,fps=24,duration=5,
        resolution='1080p',run_id='run-b',output_prefix='comparisons')
    assert first['1'] == second['1']
    assert first['1']['class_type'] == 'VHS_LoadVideoPath'
    assert first['1']['inputs']['frame_load_cap'] == 120
    assert first['1']['inputs']['force_rate'] == 24
    assert first['2']['inputs']['seed'] == second['2']['inputs']['seed'] == 42
    assert first['2']['inputs']['vosr2_profile'] == 'legacy'
    assert second['2']['inputs']['vosr2_profile'] == 'manual_baseline'
    assert first['2']['inputs']['benchmark_run'] != second['2']['inputs']['benchmark_run']
    assert first['3']['inputs']['frame_rate'] == 24
    assert first['3']['inputs']['images'] == ['2',0]


@pytest.mark.parametrize('duration,fps', [(6,24),(0,24),(float('nan'),24),(5,0),(5,121)])
def test_benchmark_rejects_unbounded_clips(duration, fps):
    from tools.benchmark_vosr2 import build_workflow
    with pytest.raises(ValueError):
        build_workflow('/clip.mp4','legacy',seed=42,fps=fps,duration=duration,
            resolution='1080p',run_id='r',output_prefix='test')


def test_benchmark_summary_keeps_report_output_references_and_failure():
    from tools.benchmark_vosr2 import summarize_history
    history = {'status':{'status_str':'success','completed':True}, 'outputs':{
        '2':{'vosr2_diagnostics':[{'profile':'legacy','status':'completed','elapsed_seconds':2.3}]},
        '3':{'gifs':[{'filename':'legacy_00001.mp4','subfolder':'comparison','type':'output'}]}}}
    summary = summarize_history(history,profile='legacy',prompt_id='p',wall_seconds=8.5)
    assert summary['status'] == 'completed'
    assert summary['vosr2']['elapsed_seconds'] == 2.3
    assert summary['output_files'][0]['filename'] == 'legacy_00001.mp4'
    assert summary['api_wall_seconds'] == 8.5
    broken = copy.deepcopy(history)
    broken['status'] = {'status_str':'error','completed':False,'messages':[['execution_error',{'exception_message':'OOM'}]]}
    assert summarize_history(broken,profile='legacy',prompt_id='p',wall_seconds=8.5)['status'] == 'failed'


def test_benchmark_does_not_call_cached_history_a_measurement():
    from tools.benchmark_vosr2 import summarize_history
    history = {'status': {'status_str':'success', 'completed':True}, 'outputs':{
        '3':{'gifs':[{'filename':'clip.mp4','subfolder':'','type':'output'}]}}}
    assert summarize_history(history,profile='legacy',prompt_id='p',wall_seconds=0.1)['status'] == 'incomplete'


def test_benchmark_api_execution_writes_workflow_history_and_summary(tmp_path, monkeypatch):
    from tools import benchmark_vosr2 as benchmark
    calls = []
    def request_json(url, path, payload=None):
        calls.append((path,payload))
        if path == '/prompt':
            return {'prompt_id':'p-1'}
        if path == '/history/p-1':
            return {'p-1': {'status': {'status_str':'success','completed':True}, 'outputs':{
                '2':{'vosr2_diagnostics':[{'profile':'legacy','status':'completed','elapsed_seconds':2}]},
                '3':{'gifs':[{'filename':'result.mp4','subfolder':'benchmark','type':'output'}]}}}}
        raise AssertionError(path)
    monkeypatch.setattr(benchmark,'request_json',request_json)
    workflow = benchmark.build_workflow('/clip.mp4','legacy',seed=42,fps=24,duration=5,
        resolution='1080p',run_id='run-1',output_prefix='benchmark')
    summary = benchmark.run_workflow('http://localhost:8188', workflow, tmp_path, 'legacy', timeout=5)
    assert summary['status'] == 'completed'
    assert calls[0][0] == '/prompt'
    assert calls[0][1]['prompt'] == workflow
    assert (tmp_path/'workflow.json').exists()
    assert (tmp_path/'history.json').exists()
    assert json.loads((tmp_path/'summary.json').read_text())['prompt_id'] == 'p-1'
    assert 'filename=result.mp4' in summary['output_files'][0]['view_url']


def test_benchmark_records_server_rejection(tmp_path, monkeypatch):
    from tools import benchmark_vosr2 as benchmark
    monkeypatch.setattr(benchmark,'request_json', lambda *args: {
        'error':'validation failed','node_errors':{'2':'VOSR2 unavailable'}})
    result = benchmark.run_workflow('http://localhost:8188',{},tmp_path,'legacy')
    assert result['status'] == 'failed'
    assert 'validation failed' in result['error']
    assert result['prompt_id'] is None
    assert json.loads((tmp_path/'summary.json').read_text())['status'] == 'failed'


def test_benchmark_dry_run_rotates_profiles_without_network(tmp_path,monkeypatch):
    from tools import benchmark_vosr2 as benchmark
    def unexpected_request(*args):
        raise AssertionError('dry run cannot contact server')
    monkeypatch.setattr(benchmark, 'request_json', unexpected_request)
    assert benchmark.main(['--video','/clip.mp4','--repeat','2','--dry-run',
                           '--output-dir',str(tmp_path)]) == 0
    summary = json.loads(next(tmp_path.glob('*/summary.json')).read_text())
    assert [row['profile'] for row in summary['runs']] == [
        'legacy','manual_baseline','manual_baseline','legacy']
    assert all(row['status'] == 'planned' for row in summary['runs'])
