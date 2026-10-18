import copy
import sys
import types

import pytest
import torch

from nodes import performance, schema


def test_balanced_rejects_latent_second_pass():
    with pytest.raises(schema.RequestError, match='h3_two_stage'):
        schema.normalize_request({'mode': 'T2VA', 'performance_preset': 'balanced_8step',
                                  'postprocess_mode': 'h3_two_stage'})


@pytest.mark.parametrize('mode', schema.MODES)
@pytest.mark.parametrize('voice', schema.VOICE_MODES)
def test_balanced_available_in_each_mode_without_changing_voice_route(mode, voice):
    assert 'balanced_8step' in schema.allowed_performance_presets(mode, voice)
    assert 'balanced_8step' in schema.visible_performance_presets(mode, voice)


@pytest.mark.parametrize('backend,prefix', [('fl2va_model', 'fl2v'), ('ref2va_model', 'ref2v')])
def test_balanced_contract_uses_separate_768p_weights(backend, prefix):
    guide = {'performance_preset': 'balanced_8step', 'resolved_backend': backend}
    plan = performance.acceleration_plan(guide)
    assert plan['lora_name'] == f'minimax_h3_{prefix}_turbo_8step_v1.0_768p_comfyui_bf16.safetensors'
    assert plan['lora_strength'] == 1
    assert plan['route'] == 'bypass'
    assert plan['use_turbo_lora']
    assert performance.preset_values('balanced_8step')['steps'] == 8
    assert performance.scheduler_plan(guide)['scheduler'] == 'simple'
    assert performance.sampler_name_for_guide(guide, 'res_multistep') == 'euler'


def test_balanced_missing_adapter_does_not_silently_fallback(monkeypatch):
    monkeypatch.setattr(performance, '_load_lightx2v_lora', lambda *a, **k: (_ for _ in ()).throw(performance.H3LoRAApplicationError('missing')))
    with pytest.raises(RuntimeError, match='8.*missing'):
        performance.MiniMaxH3AccelerationRouter().apply(object(), {
            'performance_preset': 'balanced_8step', 'resolved_backend': 'ref2va_model'})


class Model:
    def __init__(self, video=10., audio=5.):
        self.sampling = types.SimpleNamespace(shift=video, audio_shift=audio)
        self.model_options = {}

    def clone(self):
        return copy.deepcopy(self)

    def get_model_object(self, name):
        return self.sampling


@pytest.fixture
def recipe_runtime(monkeypatch):
    class Shift:
        @staticmethod
        def execute(model, shift_video, shift_audio):
            result = model.clone()
            result.sampling = types.SimpleNamespace(shift=shift_video, audio_shift=shift_audio)
            return (result,)

    def sample_euler():
        pass

    def simple_scheduler(model_sampling, steps):
        q = torch.linspace(1, 0, steps + 1)
        return model_sampling.shift * q / (1 + (model_sampling.shift - 1) * q)

    monkeypatch.setitem(sys.modules, 'comfy_extras.nodes_minimax_h3', types.SimpleNamespace(MiniMaxH3SigmaShift=Shift))
    monkeypatch.setitem(sys.modules, 'comfy.samplers', types.SimpleNamespace(simple_scheduler=simple_scheduler))
    monkeypatch.setitem(sys.modules, 'comfy.k_diffusion.sampling', types.SimpleNamespace(sample_euler=sample_euler))
    monkeypatch.setattr(performance, '_load_lightx2v_lora', lambda model, *a, **k: model.clone())
    return types.SimpleNamespace(sampler_function=sample_euler, extra_options={}), simple_scheduler


def test_balanced_shift_owned_by_recipe_even_in_legacy_manual_shift_graph(recipe_runtime):
    from nodes.sampling_recipes import prepare_balanced_guider, validate_balanced_sampling
    sampler, schedule = recipe_runtime
    guide = {'performance_preset': 'balanced_8step', 'resolved_backend': 'ref2va_model'}
    original = Model()
    patched, _, ready, _ = performance.MiniMaxH3AccelerationRouter().apply(original, guide)
    assert ready
    assert (patched.sampling.shift, patched.sampling.audio_shift) == (6., 3.)
    assert (original.sampling.shift, original.sampling.audio_shift) == (10., 5.)
    # Saved workflows have a manual SigmaShift after acceleration. Recipe must
    # fix both scheduler and guider clones without altering the saved model.
    patched.sampling = types.SimpleNamespace(shift=11., audio_shift=4.)
    class Guider:
        def __init__(self, model):
            self.model_patcher = model
            self.model_options = model.model_options
            self.original_conds = {'positive': object()}
    original_guider = Guider(patched)
    guider = prepare_balanced_guider(original_guider, guide)
    assert guider is not original_guider
    assert guider.original_conds is original_guider.original_conds
    assert guider.model_options is guider.model_patcher.model_options
    assert original_guider.model_options is patched.model_options
    assert (guider.model_patcher.sampling.shift, guider.model_patcher.sampling.audio_shift) == (6., 3.)
    assert patched.sampling.audio_shift == 4.
    sigmas = schedule(guider.model_patcher.sampling, 8)
    validate_balanced_sampling(guider.model_patcher, sampler, sigmas, guide)
    with pytest.raises(ValueError, match='sigma'):
        validate_balanced_sampling(guider.model_patcher, sampler, sigmas * .9, guide)
    with pytest.raises(ValueError, match='Euler'):
        validate_balanced_sampling(guider.model_patcher, object(), sigmas, guide)
    with pytest.raises(ValueError, match='LoRA'):
        prepare_balanced_guider(Guider(Model()), guide)


def test_balanced_scheduler_ignores_stale_step_widget_and_normalizes_model(recipe_runtime, monkeypatch):
    _, schedule = recipe_runtime
    received = []
    class Scheduler:
        @staticmethod
        def execute(model, scheduler, steps, denoise):
            received.append((model.sampling.shift, model.sampling.audio_shift, scheduler, steps))
            return (schedule(model.sampling, steps),)
    monkeypatch.setitem(sys.modules, 'comfy_extras.nodes_custom_sampler', types.SimpleNamespace(BasicScheduler=Scheduler))
    guide = {'performance_preset': 'balanced_8step', 'resolved_backend': 'fl2va_model'}
    model = performance.MiniMaxH3AccelerationRouter().apply(Model(), guide)[0]
    model.sampling.audio_shift = 5.
    sigmas = performance.MiniMaxH3SchedulerRouter().route(model, 20, guide)[0]
    assert received == [(6., 3., 'simple', 8)]
    assert len(sigmas) == 9


def test_legacy_quality_keeps_twenty_steps():
    assert performance.preset_values('quality_sage')['steps'] == 20
    assert not performance.acceleration_plan({'performance_preset': 'quality_sage'})['use_turbo_lora']
