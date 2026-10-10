import pytest

from nodes.resolution import calculate_resolution
from nodes.smart_1080p import resolve_smart_1080p_plan
from nodes.director import MiniMaxH3DirectorPlus


@pytest.mark.parametrize('aspect, expected', [('16:9', (854, 480)), ('9:16', (480, 854)), ('1:1', (480, 480))])
def test_480p_output_dimensions(aspect, expected):
    assert calculate_resolution('480p', aspect) == expected


@pytest.mark.parametrize('backend', ['fl2va_model', 'ref2va_model'])
@pytest.mark.parametrize('seconds', [5, 10, 15])
def test_480p_does_not_run_expensive_two_stage(backend, seconds):
    plan = resolve_smart_1080p_plan(backend, seconds, 8, 6.9, two_stage_ready=True,
        seedvr2_ready=True, target_width=854, target_height=480, target_preset='480p')
    assert plan['performance_preset'] == 'low_vram'
    assert plan['two_stage_route'] == 'bypass'
    assert plan['postprocess_mode'] == 'ai_upscale'
    assert plan['max_duration'] == 15


@pytest.mark.parametrize('aspect, expected', [('16:9', (854, 480)), ('9:16', (480, 854))])
def test_director_keeps_480p_target(monkeypatch, aspect, expected):
    monkeypatch.setattr('nodes.director._cuda_memory_gb', lambda: (8, 6.9))
    monkeypatch.setattr('nodes.director._trained_two_stage_dependency_report', lambda *a: {'ready': True})
    monkeypatch.setattr('nodes.director._seedvr2_dependency_report', lambda: {'ready': False, 'missing': []})
    monkeypatch.setattr('nodes.director.resolve_upscale_model_name', lambda *a, **kw: 'RealESRGAN_x2plus.pth')
    guide, *_ = MiniMaxH3DirectorPlus().build(mode='T2VA', prompt='A quiet room', duration=10,
        width=expected[0], height=expected[1], aspect_ratio=aspect, resolution_preset='480p',
        performance_preset='智能画质（自动适配）', timeline_data='{}', voice_mode='none',
        ref_image_size='match', target_dialogue='', reference_transcript='')
    assert (guide['target_width'], guide['target_height']) == expected
    assert guide['resolved_two_stage_route'] == 'bypass'
    assert guide['native_width'] * guide['native_height'] < 854 * 480 * 1.1
