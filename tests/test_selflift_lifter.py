"""The reviewed 2D lifter must preserve Avatar's complete AV sampling contract."""
import hashlib
import inspect
from pathlib import Path
import sys
import types

import pytest
import torch

from nodes import selflift_lifter as lifter


class Avatar:
    def sample(self, low_res_model, positive, negative, vae, latent_image, sampler,
               sigmas, seed, cfg, transition_step, lowres_scale, rho, w_min, w_max,
               upscaler_model, high_res_model=None, highres_tiling=False,
               tiling_mode='auto', tiling_tiles=2, tiling_axis='auto',
               highres_tiling_mode='spatial', highres_token_chunk=4096,
               highres_weight_cache_gb=4., unique_id=None):
        raise AssertionError('2D weights must never reach the upstream 3D loader')


def producer(low_res_model, positive, negative, vae, latent_image, sampler, sigmas,
             seed, cfg, transition_step, lowres_scale, rho, w_min, w_max,
             latent_upsample, latent_lifter=None, high_res_model=None,
             highres_tiling=False, tiling_mode='auto', tiling_tiles=2,
             tiling_axis='auto', on_tiling_plan=None, highres_tiling_mode='spatial',
             highres_token_chunk=4096, highres_weight_cache_gb=4.,
             spatial_anchor_mode='off', tiling_overlap=None, diagnostics_out=None):
    producer.received = locals()
    return latent_image


def arguments():
    return dict(low_res_model=object(), high_res_model=object(), positive=object(),
                negative=object(), vae=object(), latent_image={'samples': object(),
                'noise_mask': object(), '_h3_long_video_context': {'mode': 'repair'}},
                sampler=object(), sigmas=object(), seed=111330200, cfg=1.,
                transition_step=6, lowres_scale=.6, rho=0., w_min=.5, w_max=1.,
                upscaler_model=lifter.MODEL_NAME, highres_tiling=True,
                highres_tiling_mode='global_stream')


def test_optional_selection_requires_explicit_producer_and_exact_registered_model(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], 'progressive_sample', producer, raising=False)
    seen = []
    def resolve(category, name):
        seen.append((category, name))
        return 'vendor/' + name
    assert lifter.preferred_model(Avatar().sample, resolver=resolve) == 'vendor/' + lifter.MODEL_NAME
    assert seen == [('latent_upscale_models', lifter.MODEL_NAME)]
    assert lifter.preferred_model(Avatar().sample, resolver=lambda *_: None) is None


def test_legacy_producer_is_not_selected_even_when_model_exists(monkeypatch):
    monkeypatch.delattr(sys.modules[__name__], 'progressive_sample', raising=False)
    assert lifter.preferred_model(Avatar().sample, resolver=lambda *_: lifter.MODEL_NAME) is None
    monkeypatch.setattr(sys.modules[__name__], 'progressive_sample', lambda **kwargs: None, raising=False)
    assert lifter.preferred_model(Avatar().sample, resolver=lambda *_: lifter.MODEL_NAME) is None


def test_2d_call_preserves_actual_av_masks_and_sampling_inputs(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], 'progressive_sample', producer, raising=False)
    kwargs = arguments()
    diagnostics = {}
    result = lifter.run(Avatar().sample, kwargs, diagnostics=diagnostics)
    received = producer.received
    assert result['samples'] is kwargs['latent_image']['samples']
    assert received['latent_image'] is kwargs['latent_image']
    assert received['latent_image']['noise_mask'] is kwargs['latent_image']['noise_mask']
    for key, value in kwargs.items():
        if key != 'upscaler_model':
            assert received[key] is value or received[key] == value
    assert callable(received['latent_lifter'])
    assert received['latent_upsample'] == 'nearest'
    assert received['spatial_anchor_mode'] == 'off'
    assert received['tiling_overlap'] is None
    assert received['diagnostics_out'] is diagnostics


@pytest.mark.parametrize('key,value', [('rho', .6), ('highres_tiling', False),
    ('highres_tiling_mode', 'spatial'), ('upscaler_model', 'other.safetensors')])
def test_unreviewed_routes_do_not_reinterpret_2d_weights(monkeypatch, key, value):
    monkeypatch.setattr(sys.modules[__name__], 'progressive_sample', producer, raising=False)
    kwargs = arguments()
    kwargs[key] = value
    with pytest.raises(ValueError):
        lifter.run(Avatar().sample, kwargs)


def test_pinned_inference_source_has_the_reviewed_identity():
    assert hashlib.sha256(lifter.SOURCE_PATH.read_bytes()).hexdigest() == lifter.SOURCE_SHA256


def test_checkpoint_identity_rejects_a_renamed_or_modified_model(tmp_path):
    path = tmp_path / lifter.MODEL_NAME
    path.write_bytes(b'wrong checkpoint under the expected name')
    with pytest.raises(ValueError, match='checkpoint|权重'):
        lifter.verify_checkpoint(path)


def fake_runtime(monkeypatch, result_kind):
    events = []
    class Model:
        def __call__(self, x, **kwargs):
            events.append(('forward', kwargs))
            if result_kind == 'raise':
                raise RuntimeError('model failure')
            result = torch.nn.functional.interpolate(x, size=(x.shape[2], *kwargs['target_hw']), mode='nearest')
            if result_kind == 'bad_shape':
                result = result[:, :-1]
            if result_kind == 'nan':
                result.fill_(float('nan'))
            return result
        def to(self, device):
            events.append(('offload', device))
    source = types.SimpleNamespace(load_model=lambda *args: Model(),
        _make_norm_tensors=lambda device, dtype: (torch.zeros(1,24,1,1,1,device=device,dtype=dtype),
                                                torch.ones(1,24,1,1,1,device=device,dtype=dtype)))
    mm = types.SimpleNamespace(get_torch_device=lambda: torch.device('cpu'),
        intermediate_device=lambda: torch.device('cpu'), soft_empty_cache=lambda: events.append(('empty',)))
    monkeypatch.setattr(lifter, '_inference_source', lambda: source)
    monkeypatch.setattr(lifter, '_model_path', lambda name: Path(name))
    monkeypatch.setattr(lifter, 'verify_checkpoint', lambda path: lifter.MODEL_SHA256)
    monkeypatch.setattr(lifter, '_model_management', lambda: mm)
    return events


@pytest.mark.parametrize('kind', ['valid', 'bad_shape', 'nan', 'raise'])
def test_lift_validates_geometry_finite_pixels_and_always_offloads(monkeypatch, kind):
    events = fake_runtime(monkeypatch, kind)
    values = torch.arange(1*24*3*2*3).reshape(1,24,3,2,3).float() / 100
    if kind == 'valid':
        evidence = {}
        result = lifter.lift(values, (4,6), lifter.MODEL_NAME, evidence=evidence)
        assert result.shape == (1,24,3,4,6)
        assert result.dtype == torch.float32 and result.device.type == 'cpu'
        expected = torch.nn.functional.interpolate(values.half(), size=(3,4,6), mode='nearest').float()
        torch.testing.assert_close(result, expected)
        assert events[0][1]['tier'] == 3
        assert evidence['tier'] == 3
        assert events[0][1]['chunk_frames'] == 32
    else:
        with pytest.raises((ValueError, RuntimeError)):
            lifter.lift(values, (4,6), lifter.MODEL_NAME)
    assert events[-2:] == [('offload', 'cpu'), ('empty',)]
