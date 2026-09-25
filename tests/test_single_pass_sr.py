import pytest
from nodes.smart_1080p import resolve_smart_1080p_plan

@pytest.mark.parametrize('backend',['fl2va_model','ref2va_model'])
@pytest.mark.parametrize('voice',['none','h3_reference'])
@pytest.mark.parametrize('method',['vosr2','video_sr'])
def test_original_fhd_workflow_is_single_pass(backend,voice,method):
    p=resolve_smart_1080p_plan(backend,15,32,28,seedvr2_ready=True,two_stage_ready=True,target_width=1920,target_height=1080,voice_mode=voice,target_preset='1080p FHD',postprocess_mode=method)
    assert p['two_stage_route']=='bypass'
    assert p['dimension_plan'] is None
    assert p['performance_preset'] not in {'quality_two_stage','low_vram_two_stage'}
    assert p['postprocess_mode']==method

def test_old_fhd_auto_selection_migrates_to_vosr2():
    p=resolve_smart_1080p_plan('ref2va_model',15,24,23,two_stage_ready=True,target_preset='1080p FHD',postprocess_mode='ai_upscale')
    assert p['postprocess_mode']=='vosr2'
    assert p['two_stage_route']=='bypass'

def test_768p_no_second_pass():
    p=resolve_smart_1080p_plan('ref2va_model',15,32,28,two_stage_ready=True,target_preset='768p H3')
    assert p['two_stage_route']=='bypass'

@pytest.mark.parametrize('mode,voice',[('T2VA','none'),('REF2VA','h3_reference')])
@pytest.mark.parametrize('method',['vosr2','video_sr'])
@pytest.mark.parametrize('aspect,width,height',[('16:9',1920,1080),('9:16',1080,1920)])
def test_director_to_output_single_pass_contract(monkeypatch,mode,voice,method,aspect,width,height):
    import torch
    from nodes import director, two_stage_assets, stream_output
    monkeypatch.setattr(director,'_probe_vram_after_prefree',lambda:(32,28))
    monkeypatch.setattr(director,'_cuda_memory_gb',lambda:(32,28))
    def forbid_two_stage(*args):
        raise AssertionError('1080p must not inspect two-stage assets')
    monkeypatch.setattr(director,'_trained_two_stage_dependency_report',forbid_two_stage)
    monkeypatch.setattr(director,'_seedvr2_dependency_report',lambda:{'ready':True,'missing':[],'available_dit':['seedvr2_ema_3b_fp8_e4m3fn.safetensors']})
    monkeypatch.setattr(two_stage_assets,'_comfy_node_mappings',lambda:{n:object for n in ('TESpeedVOSR2Loader','TESpeedVOSR2Settings','TESpeedVOSR2Video')})
    kwargs={}
    if voice=='h3_reference':
        wave=torch.sin(torch.arange(48000*3).float()*0.04).reshape(1,1,-1)*0.2
        kwargs['voice_reference_audio']={'waveform':wave,'sample_rate':48000}
    guide,*_=director.MiniMaxH3DirectorPlus().build(mode=mode,prompt='A person says hello.',duration=5,width=width,height=height,aspect_ratio=aspect,resolution_preset='1080p FHD',voice_mode=voice,ref_image_size='match',performance_preset='智能画质（自动适配）',postprocess_mode=method,timeline_data='{}',target_dialogue='',reference_transcript='',**kwargs)
    assert guide['two_stage_enabled'] is False
    assert guide['resolved_two_stage_route']=='bypass'
    assert guide['second_stage_width']==guide['first_stage_width']
    assert (guide['target_width'],guide['target_height'])==(width,height)
    assert stream_output._resolve_postprocess_path(guide,guide['native_width'],guide['native_height'])=='video_sr'
    assert guide['upscale_method']==('vosr2' if method=='vosr2' else 'seedvr2')
    if method=='vosr2':assert guide['video_sr_plan']=={'engine':'vosr2'}
    else:assert '3b' in guide['video_sr_plan']['dit_model']
    if voice=='h3_reference':assert len(guide['ref_audios'])==1

def test_output_vosr_dispatch_never_loads_seedvr(monkeypatch):
    import torch
    from nodes import stream_output,video_finish
    seen={}
    def fake_vosr(images,w,h,seed):
        seen['shape']=images.shape
        return torch.zeros(len(images),h,w,3)
    def forbidden():raise AssertionError('VOSR2 must not load SeedVR2')
    monkeypatch.setattr(stream_output,'resolve_seedvr2_callables',forbidden)
    monkeypatch.setattr(video_finish,'_vosr2',fake_vosr)
    out=torch.cat(list(stream_output._iter_video_sr_frame_chunks(torch.zeros(3,72,128,3),192,108,plan={'engine':'vosr2'})))
    assert out.shape==(3,108,192,3)
    assert seen['shape'][0]==3
