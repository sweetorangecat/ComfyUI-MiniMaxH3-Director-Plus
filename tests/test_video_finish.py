import pytest
from nodes.video_finish import output_size, vosr2_input_size, validate_sr_result

@pytest.mark.parametrize('preset,expected', [('480p',(1920,1080)),('768p',(1920,1080)),('1080p',(1920,1080))])
def test_exact_landscape(preset,expected):
    assert output_size(1920,1080,preset)==expected

def test_portrait_and_no_stretch():
    assert output_size(1080,1920,'1080p')==(1080,1920)
    assert output_size(1344,768,'1080p')==(1890,1080)

def test_vosr_preserves_source_and_treats_target_as_minimum():
    assert vosr2_input_size(1344,768,1890,1080)==(1890,1080,1)
    assert vosr2_input_size(1344,768,1920,1098)==(1920,1098,1)
    assert vosr2_input_size(768,1344,1098,1920)==(1098,1920,1)
    assert vosr2_input_size(1920,1080,854,480)==(1920,1080,1)
    assert vosr2_input_size(832,448,1428,768)==(1428,768,1)
    assert vosr2_input_size(960,540,1920,1080)==(960,540,2)

def test_sr_output_size_preserves_source_aspect_and_minimum():
    from nodes.video_finish import sr_output_size
    assert sr_output_size(832,448,1344,768,'vosr2') == (1428,768)
    assert sr_output_size(1920,1088,2560,1440,'vosr2') == (2560,1452)
    assert sr_output_size(832,448,1344,768,'seedvr2') == (1428,768)
    assert sr_output_size(1920,1080,854,480,'seedvr2') == (1920,1080)

def test_frame_loss_is_error():
    import torch
    with pytest.raises(RuntimeError,match='帧数'):
        validate_sr_result(torch.zeros(2,8,8,3),3)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_reconstruction_is_rejected(value):
    import torch
    images = torch.zeros(2,8,8,3)
    images[-1,-1,-1,-1] = value
    with pytest.raises(RuntimeError, match='非有限'):
        validate_sr_result(images,2)


def test_fractional_canvas_is_not_rounded_again_at_backend_handoff():
    from nodes.video_finish import sr_output_size
    # Exact integer arithmetic: 2400/2240 * 2240 rounds above 2400 in float.
    for source, target in [((2240,1280),(2400,1350)), ((1344,768),(1920,1080))]:
        first = sr_output_size(*source,*target,'vosr2')
        width, height, scale = vosr2_input_size(*source,*first)
        assert (width * scale, height * scale) == first
        assert first[0] >= target[0] and first[1] >= target[1]


def test_fractional_preparation_keeps_sub_8bit_detail_and_all_frames():
    import torch
    from nodes.video_finish import _prepare_vosr2_input
    # Both constants round to the same 8-bit value. AI input must retain them.
    source = torch.stack([torch.full((8,12,3), value) for value in (0.5001, 0.5002)])
    prepared = _prepare_vosr2_input(source,18,12)
    assert prepared.shape == (2,12,18,3)
    assert prepared.dtype == torch.float32
    assert torch.allclose(prepared[0],torch.full((12,18,3),0.5001),atol=1e-6)
    assert torch.allclose(prepared[1],torch.full((12,18,3),0.5002),atol=1e-6)

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
    source=torch.zeros(3,72,128,3,dtype=torch.float16)
    result=video_finish._vosr2(source,192,108,42)
    assert result.shape==(3,108,192,3)
    assert seen['video']['scale']==1
    assert seen['video']['images'].shape==(3,108,192,3)
    assert seen['video']['images'].dtype==torch.float32
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
