import pytest
from nodes.video_finish import output_size, vosr2_input_size, validate_sr_result

@pytest.mark.parametrize('preset,expected', [('480p',(854,480)),('768p',(1366,768)),('1080p',(1920,1080))])
def test_exact_landscape(preset,expected):
    assert output_size(1920,1080,preset)==expected

def test_portrait_and_no_stretch():
    assert output_size(1080,1920,'1080p')==(1080,1920)
    assert output_size(1344,768,'1080p')==(1890,1080)

def test_vosr_does_not_generate_above_target():
    assert vosr2_input_size(1344,768,1890,1080)==(945,540,2)
    assert vosr2_input_size(1920,1080,854,480)==(854,480,1)

def test_frame_loss_is_error():
    import torch
    with pytest.raises(RuntimeError,match='帧数'):
        validate_sr_result(torch.zeros(2,8,8,3),3)

def test_vosr_adapter_uses_bounded_size_and_no_temporal_cache(monkeypatch):
    import sys, types, torch
    from nodes import two_stage_assets, video_finish
    seen={}
    def resize(images,w,h,method):
        return torch.zeros(images.shape[0],h,w,3)
    fake=types.ModuleType('nodes.stream_output')
    fake._resize_cpu_chunk=resize
    fake._release_comfy_models_before_video_sr=lambda:None
    fake._unwrap_node_result=lambda x:x[0]
    monkeypatch.setitem(sys.modules,'nodes.stream_output',fake)
    class Loader:
        FUNCTION='load'
        def load(self,**kw):return ({},)
    class Settings:
        FUNCTION='make'
        def make(self,**kw):seen['settings']=kw;return (kw,)
    class Video:
        FUNCTION='upscale'
        def upscale(self,**kw):
            seen['video']=kw
            im=kw['images'];return (torch.zeros(len(im),im.shape[1]*kw['scale'],im.shape[2]*kw['scale'],3),)
    monkeypatch.setattr(two_stage_assets,'_comfy_node_mappings',lambda:{'TESpeedVOSR2Loader':Loader,'TESpeedVOSR2Settings':Settings,'TESpeedVOSR2Video':Video})
    result=video_finish._vosr2(torch.zeros(3,72,128,3),192,108,42)
    assert result.shape==(3,108,192,3)
    assert seen['video']['scale']==2
    assert seen['video']['images'].shape==(3,54,96,3)
    assert seen['video']['temporal_cache'] is False
    assert seen['settings']['frame_batch']==2

def test_workflow_preserves_fps_and_audio():
    import json
    from pathlib import Path
    from tools.validate_workflow import validate_workflow
    d=json.loads(Path('examples/H3-VideoFinish-480p-768p-1080p.json').read_text(encoding='utf-8'))
    validate_workflow(d)
    assert [3,1,2,4,1,'AUDIO'] in d['links']
    assert [5,3,0,4,2,'FLOAT'] in d['links']
    assert d['nodes'][0]['widgets_values']['force_rate']==0
