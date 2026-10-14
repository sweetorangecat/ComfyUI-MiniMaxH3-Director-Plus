import pytest

from nodes.schema import normalize_request, public_schema, visible_postprocess_modes


def test_experiments_default_off_and_are_public():
    request = normalize_request({'mode': 'T2VA'})
    assert request['sampling_method'] == 'standard'
    assert request['community_lora_profile'] == 'off'
    properties = public_schema()['properties']
    assert properties['sampling_method']['enum'] == ['standard', 'selflift']
    assert properties['community_lora_profile']['default'] == 'off'


@pytest.mark.parametrize('preset', ['quality', 'quality_sage', 'quality_two_stage', 'low_vram'])
@pytest.mark.parametrize('engine', ['vosr2', 'video_sr', 'ai_upscale'])
def test_curated_sr_is_selectable_on_every_visible_sampling_preset(preset, engine):
    request = normalize_request({'mode': 'REF2VA', 'performance_preset': preset,
                                 'postprocess_mode': engine})
    assert request['postprocess_mode'] == engine
    assert visible_postprocess_modes(preset) == ('vosr2', 'video_sr', 'ai_upscale')


@pytest.mark.parametrize('changes', [
    {'sampling_method': 'unknown'}, {'community_lora_profile': 'unknown'},
    {'sampling_method': 'selflift', 'performance_preset': 'low_vram'},
    {'sampling_method': 'selflift', 'mode': 'T2VA'},
    {'sampling_method': 'selflift', 'duration': 16},
    {'community_lora_profile': 'combat', 'performance_preset': 'quality'},
])
def test_api_rejects_incompatible_experimental_settings(changes):
    request = dict(mode='REF2VA', performance_preset='quality_two_stage',
                   postprocess_mode='vosr2', duration=5)
    request.update(changes)
    with pytest.raises(ValueError):
        normalize_request(request)


@pytest.mark.parametrize('preset,size', [
    ('480p', (848, 480)), ('768p H3', (1344, 768)),
    ('1080p FHD', (1920, 1080)), ('2K QHD', (2560, 1440)),
    ('4K UHD', (3840, 2160)),
])
@pytest.mark.parametrize('engine', ['vosr2', 'video_sr', 'ai_upscale'])
def test_smart_output_preserves_selected_engine(preset, size, engine):
    from nodes.smart_1080p import resolve_smart_1080p_plan
    plan = resolve_smart_1080p_plan('ref2va_model', 5, 32, 29,
        target_width=size[0], target_height=size[1], target_preset=preset,
        two_stage_ready=True, seedvr2_ready=True, postprocess_mode=engine)
    assert plan['postprocess_mode'] == engine


def test_smart_ai_upscale_does_not_require_seedvr2():
    from nodes.smart_1080p import resolve_smart_1080p_plan
    plan = resolve_smart_1080p_plan('ref2va_model', 5, 32, 29,
        target_width=3840, target_height=2160, target_preset='4K UHD',
        two_stage_ready=True, seedvr2_ready=False, postprocess_mode='ai_upscale')
    assert plan['postprocess_mode'] == 'ai_upscale'


def test_smart_selected_seedvr2_cannot_fall_back_on_low_vram():
    from nodes.smart_1080p import resolve_smart_1080p_plan
    with pytest.raises(ValueError, match='SeedVR2'):
        resolve_smart_1080p_plan('ref2va_model', 5, 8, 7,
            target_width=1920, target_height=1080, target_preset='1080p FHD',
            seedvr2_ready=False, postprocess_mode='video_sr')
