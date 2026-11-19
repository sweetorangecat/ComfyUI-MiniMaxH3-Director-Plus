from copy import deepcopy
from types import SimpleNamespace

import pytest

from nodes import short_quality_reference as quality
from nodes.long_video_shots import build_shot_graph


def settings(**changes):
    return dict(mode='REF2VA', sampling_method='selflift',
        performance_preset='quality_two_stage', community_lora_profile='continuity',
        resolution_preset='1080p FHD', voice_mode='h3_reference',
        postprocess_mode='vosr2', **changes)


def actual(**changes):
    return {'short_quality_reference': quality.PROFILE_ID,
        'first_stage_width':1152, 'first_stage_height':768,
        'second_stage_width':1632, 'second_stage_height':1088,
        'selflift_lowres_scale':.7, 'selflift_transition_step':6,
        'selflift_cfg':1.0, 'selflift_rho':0.0,
        'selflift_highres_tiling_mode':'global_stream', 'sol_attn_applied':True,
        'selflift_lifter':'trained_2d_tier3',
        'selflift_lifter_evidence':{'tier':3,'checkpoint_sha256':quality.LIFTER_SHA256},
        **changes}


def produced(**changes):
    from nodes.execution_report import attach_report
    image=SimpleNamespace(shape=(360,1088,1632,3))
    evidence={**actual(), 'steps':10,'sigma_count':11,'sampler':'euler','scheduler':'beta',
              'video_shift':12.0,'audio_shift':3.0,'width':1632,'height':1088,
              'sampling_actual_method':'selflift',
              'two_stage_first_executed_steps':6,'two_stage_second_executed_steps':4,
              **changes}
    attach_report(image, {'version':1,'current_plan':{'short_quality_reference':quality.PROFILE_ID,
        'width':1632,'height':1088},'sampling_settings':evidence,
        'stages':[{'stage':'sampling','status':'completed','steps':10,
                   'path':'SelfLiftAvatarH3Sampler'}]})
    return image


def test_accepted_portrait_geometry_uses_the_actual_measured_prefix():
    plan = quality.plan(1620,1080,15,32,30)
    assert (plan['second_stage_width'],plan['second_stage_height']) == (1632,1088)
    assert (plan['first_stage_width'],plan['first_stage_height']) == (1152,768)
    assert plan['selflift_lowres_scale_requested'] == .7
    assert plan['short_quality_reference'] == quality.PROFILE_ID
    assert plan['final_scale'] == 1


@pytest.mark.parametrize('canvas',[(1920,1080),(1080,1920),(1620,1080),(1760,1184),(1440,1440),(1824,1152),(1152,1824)])
def test_fhd_rounding_retains_the_requested_canvas_without_silent_reduction(canvas):
    plan = quality.plan(*canvas,15,32,30)
    assert plan['second_stage_width'] >= canvas[0]
    assert plan['second_stage_height'] >= canvas[1]
    assert plan['second_stage_width'] % 32 == plan['second_stage_height'] % 32 == 0


@pytest.mark.parametrize('total',[8,16,24])
def test_smaller_hardware_retains_its_existing_planner(total):
    assert quality.plan(1620,1080,15,total,total-1) is None


def test_insufficient_available_memory_cannot_silently_reduce_a_qualified_grid():
    with pytest.raises(ValueError, match='24GB'):
        quality.plan(1620,1080,15,32,23)


def test_metadata_requires_an_actual_independent_clip_request():
    graph={'10':{'_meta':{'h3_long_video_entry':'independent_shots',
        'h3_short_quality_reference':quality.PROFILE_ID}}}
    assert quality.requested(graph,'10')
    assert not quality.requested(graph,None)
    graph['10']['_meta']['h3_long_video_entry']='director_duration'
    assert not quality.requested(graph,'10')
    assert not quality.requested(None,None)


@pytest.mark.parametrize('changes',[
    {'sampling_method':'standard'}, {'performance_preset':'low_vram'},
    {'community_lora_profile':'combat'}, {'resolution_preset':'768p H3'},
    {'mode':'FL2VA'}, {'voice_mode':'fish_lock'}, {'aspect_ratio':'21:9'},
])
def test_other_original_modes_do_not_implicitly_adopt_the_reference(changes):
    request=settings();request.update(changes)
    assert not quality.eligible(request)


@pytest.mark.parametrize('voice',['none','h3_reference'])
def test_native_voice_choice_does_not_change_the_spatial_reference(voice):
    request=settings();request['voice_mode']=voice
    assert quality.eligible(request)


def test_internal_graph_uses_original_references_and_the_existing_memory_control():
    config=settings();config.update(prompt='full film',duration=30,seed=8,
        reference_image_1=['image',0],voice_reference_audio_file='original.wav')
    base={'director':{'class_type':'MiniMaxH3DirectorPlus','inputs':config},
          'image':{'class_type':'LoadImage','inputs':{'image':'original.png'}},
          'accel':{'class_type':'MiniMaxH3AccelerationRouter','inputs':{
              'guide':['director',0],'attention_chunks':'自动（按显存）'}},
          'sampler':{'class_type':'MiniMaxH3TwoStageSampler','inputs':{
              'guide':['director',0],'model':['accel',0]}},
          'output':{'class_type':'MiniMaxH3U11Output','inputs':{
              'images':['sampler',0],'guide':['director',0]}}}
    before=deepcopy(base)
    request={'prompt':'this paragraph','seed':1009,'project_name':'film',
             'take_id':'take1','window':{'duration':15,'dialogue':[]}}
    result=build_shot_graph(base,request)
    assert base==before
    assert quality.requested(result,'director')
    assert result['director']['class_type']=='MiniMaxH3LongVideoQualityDirector'
    assert result['director']['inputs']['reference_image_1']==['image',0]
    assert result['director']['inputs']['voice_reference_audio_file']=='original.wav'
    assert result['accel']['inputs']['attention_chunks']=='自动（按显存）'
    assert not any(node['class_type']=='MiniMaxH3AddGuide' for node in result.values())


@pytest.mark.parametrize('changes',[
    {'selflift_lowres_scale':.6}, {'selflift_transition_step':5},
    {'selflift_rho':.6}, {'selflift_highres_tiling_mode':'spatial'},
    {'sol_attn_applied':False}, {'selflift_lifter':'upstream_learned'},
    {'selflift_lifter_evidence':{'tier':3,'checkpoint_sha256':'changed'}},
])
def test_publication_rejects_a_changed_actual_recipe(changes):
    with pytest.raises(ValueError, match='基准'):
        quality.validate_output(actual(), produced(**changes))


def test_publication_records_actual_geometry_and_rejects_a_small_result():
    guide=actual()
    result=quality.validate_output(guide,produced())
    assert result['profile']==quality.PROFILE_ID
    assert result['canvas']==[1632,1088]
    assert result['content_review']=='pending'
    with pytest.raises(ValueError,match='基准'):
        quality.validate_output(guide,SimpleNamespace(shape=(360,768,1152,3)))
    assert quality.validate_output({},SimpleNamespace(shape=(360,768,1152,3))) is None


@pytest.mark.parametrize('changes',[
    {'steps':1}, {'sampler':'res_multistep'}, {'scheduler':'simple'},
    {'selflift_lowres_scale':.6}, {'two_stage_first_executed_steps':1},
    {'sol_attn_applied':False}, {'width':1152,'height':768},
    {'sampling_actual_method':'standard'},
])
def test_old_mutable_guide_evidence_cannot_approve_a_different_image_producer(changes):
    with pytest.raises(ValueError,match='基准'):
        quality.validate_output(actual(),produced(**changes))


def test_missing_or_failed_producer_report_is_not_an_accepted_reference():
    from nodes.execution_report import report_from
    with pytest.raises(ValueError,match='基准'):
        quality.validate_output(actual(),SimpleNamespace(shape=(360,1088,1632,3)))
    image=produced()
    report_from(image)['stages'][0]['status']='failed'
    with pytest.raises(ValueError,match='基准'):
        quality.validate_output(actual(),image)


@pytest.mark.parametrize('reference,control,expected',[(True,'自动（按显存）',16),
    (True,'8（推荐）',16),(False,'自动（按显存）',None),(False,'8（推荐）',8)])
def test_memory_control_changes_only_after_the_high_grid_actually_qualifies(monkeypatch,reference,control,expected):
    from nodes import performance
    guide={'short_quality_reference':quality.PROFILE_ID if reference else None}
    monkeypatch.setattr(performance,'upstream_loras',lambda *args:[])
    monkeypatch.setattr(performance,'_restore_selflift_natural_texture_base',lambda model,guide:model)
    monkeypatch.setattr(performance,'_apply_acceleration',lambda model,guide:guide)
    result=performance.MiniMaxH3AccelerationRouter().apply(object(),guide,attention_chunks=control)
    assert result['minimax_head_chunks_ui']==expected


@pytest.mark.parametrize('first,current',[(True,False),(False,True)])
def test_a_project_cannot_mix_reference_and_compatibility_segments(first,current):
    reference=quality.validate_output(actual(),produced())
    previous=[{'short_quality_reference':reference}] if first else [{}]
    with pytest.raises(ValueError,match='基准'):
        quality.validate_project_reference(previous,reference if current else None)


def test_same_project_keeps_the_first_accepted_geometry_and_lifter():
    reference=quality.validate_output(actual(),produced())
    quality.validate_project_reference([],reference)
    quality.validate_project_reference([{'short_quality_reference':reference}],deepcopy(reference))
    quality.validate_project_reference([{}],None)
    changed=deepcopy(reference);changed['canvas']=[1920,1088]
    with pytest.raises(ValueError,match='基准'):
        quality.validate_project_reference([{'short_quality_reference':reference}],changed)


def test_complete_film_rejects_mixed_profiles_even_at_identical_final_dimensions():
    from nodes.long_video_shot_service import _delivery_canvas
    reference=quality.validate_output(actual(),produced())
    record={'technical':{'width':1632,'height':1088},'short_quality_reference':reference}
    assert _delivery_canvas({'export':{'target':[1620,1080]}},[record,deepcopy(record)])==(1632,1088)
    with pytest.raises(ValueError,match='基准'):
        _delivery_canvas({'export':{'target':[1620,1080]}},[record,{'technical':record['technical']}])


def test_redo_first_clip_checks_the_still_valid_accepted_later_clips():
    from nodes.long_video_shots import shot_identity
    reference=quality.validate_output(actual(),produced())
    plan=[{'index':i,'prompt':f'shot{i}'} for i in range(3)]
    config={'generation_key':'generation1','seed':42}
    windows=[]
    for window in plan:
        seed,key=shot_identity(config,window,0)
        windows.append({'window':window,'seed':seed,'fingerprint':key,
                        'status':'accepted','short_quality_reference':reference})
    state={'config':config,'plan':plan,'windows':windows,'content_attempts':{'0':1}}
    peers=quality.reference_peers(state,0)
    assert len(peers)==2
    with pytest.raises(ValueError,match='基准'):
        quality.validate_project_reference(peers,None)
    # A complete generation change makes these later clips ineligible for
    # reuse; they must not block starting the new rendering recipe.
    state['config']={**config,'generation_key':'generation2'}
    assert quality.reference_peers(state,0)==[]
