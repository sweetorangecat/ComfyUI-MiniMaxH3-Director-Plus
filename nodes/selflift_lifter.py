"""Reviewed 2D lifting through Avatar's explicit progressive-sampler interface.

Only the learned spatial lift changes. Avatar retains ownership of its AV
noise masks, sampling schedule, global attention and temporal continuation.
Torch and the pinned inference module are imported only when actually used.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import logging
from pathlib import Path
import sys

LOGGER = logging.getLogger('MiniMaxH3.DirectorPlus')
MODEL_NAME = 'minimax_h3_latent_upscaler_2d_conv_v1_fp16.safetensors'
MODEL_SIZE = 364308928
MODEL_SHA256 = '6e0f35692784ba2376704a48a90fb07f90aef76ef21200d5525d6627a54fe909'
SOURCE_SHA256 = '8c95ea2c09f474464640570b67b61565ac6d8aa38c579b0662a5e080e26157b8'
SOURCE_PATH = Path(__file__).parent / 'vendor' / 'minimax_h3_latent_upscaler_2d.py'
_SOURCE_MODULE = '_h3_director_reviewed_latent2d'


def _producer(function):
    module = sys.modules.get(getattr(function, '__module__', ''))
    producer = getattr(module, 'progressive_sample', None)
    if not callable(producer):
        return None
    names = inspect.signature(producer).parameters
    required = {'low_res_model', 'latent_image', 'latent_upsample', 'latent_lifter',
                'highres_tiling_mode', 'spatial_anchor_mode', 'tiling_overlap', 'diagnostics_out'}
    if not required.issubset(names):
        return None
    if any(names[name].kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY) for name in required):
        return None
    return producer


def preferred_model(function, *, resolver=None):
    """Return an optional exact 2D asset only for a compatible Avatar producer."""
    if _producer(function) is None:
        return None
    if resolver is None:
        from .two_stage_assets import resolve_registered_model_name
        resolver = resolve_registered_model_name
    return resolver('latent_upscale_models', MODEL_NAME)


def verify_checkpoint(path):
    path = Path(path)
    if path.stat().st_size != MODEL_SIZE:
        raise ValueError('SelfLift 2D checkpoint 权重大小与已验证版本不一致')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError('SelfLift 2D checkpoint 权重 SHA256 与已验证版本不一致')
    return digest


def _inference_source():
    module = sys.modules.get(_SOURCE_MODULE)
    if module is not None:
        return module
    if hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('SelfLift 2D inference source 与已验证版本不一致')
    spec = importlib.util.spec_from_file_location(_SOURCE_MODULE, SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SOURCE_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_SOURCE_MODULE, None)
        raise
    return module


def _model_path(name):
    import folder_paths
    return Path(folder_paths.get_full_path_or_raise('latent_upscale_models', name))


def _model_management():
    import comfy.model_management
    return comfy.model_management


def lift(values, target_hw, model_name, *, evidence=None):
    """Use the author's normalization, tier1 and full temporal chunk context."""
    import torch
    if (not isinstance(values, torch.Tensor) or values.ndim != 5 or values.shape[1] != 24
            or any(size <= 0 for size in values.shape)):
        raise ValueError('SelfLift 2D requires a nonempty [B,24,T,H,W] video latent')
    if (len(target_hw) != 2 or any(isinstance(v, bool) or int(v) != v or v <= 0 for v in target_hw)
            or any(int(v) < original for v, original in zip(target_hw, values.shape[-2:]))):
        raise ValueError('SelfLift 2D requires an upscaled positive target geometry')
    target_hw = tuple(map(int, target_hw))
    digest = verify_checkpoint(_model_path(model_name))
    source = _inference_source()
    mm = _model_management()
    device, dtype = mm.get_torch_device(), torch.float16
    model = source.load_model(model_name, device, 'fp16')
    try:
        mean, std = source._make_norm_tensors(device, dtype)
        with torch.inference_mode():
            normalized = (values.to(device=device, dtype=dtype) - mean) / std
            output = model(normalized, scale=sum(v / old for v, old in zip(target_hw, values.shape[-2:])) / 2,
                           target_hw=target_hw, enable_chunking=True, chunk_frames=32, tier=1)
            output = (output * std + mean).float().to(mm.intermediate_device())
        if tuple(output.shape) != (*values.shape[:-2], *target_hw):
            raise RuntimeError('SelfLift 2D changed non-spatial latent geometry')
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError('SelfLift 2D returned nonfinite latent values')
        if evidence is not None:
            evidence.update(model=model_name, checkpoint_sha256=digest,
                            source_sha256=SOURCE_SHA256, tier=1, temporal_chunk_frames=32)
        LOGGER.info('[H3 SelfLift lifter] trained_2d tier1 checkpoint=%s sha256=%s %s -> %s',
                    model_name, digest, tuple(values.shape), tuple(output.shape))
        return output
    finally:
        model.to('cpu')
        mm.soft_empty_cache()


def run(function, kwargs, *, diagnostics=None, evidence=None):
    """Call the unchanged progressive sampler without the incompatible 3D loader."""
    producer = _producer(function)
    if producer is None:
        raise ValueError('SelfLift Avatar does not expose the reviewed latent_lifter interface')
    bound = inspect.signature(function).bind(**kwargs)
    bound.apply_defaults()
    params = dict(bound.arguments)
    model_name = params.pop('upscaler_model')
    if Path(str(model_name).replace('\\', '/')).name != MODEL_NAME:
        raise ValueError('SelfLift 2D adapter requires its exact reviewed checkpoint')
    if (not params.get('highres_tiling') or params.get('highres_tiling_mode') != 'global_stream'
            or params.get('rho') != 0.):
        raise ValueError('SelfLift 2D adapter requires the reviewed global_stream/rho0 route')
    params.pop('unique_id', None)
    params.pop('upscaler_unload', None)
    params.update(latent_upsample='nearest',
                  latent_lifter=lambda z, hw: lift(z, hw, model_name, evidence=evidence),
                  spatial_anchor_mode='off', tiling_overlap=None,
                  diagnostics_out={} if diagnostics is None else diagnostics)
    inspect.signature(producer).bind(**params)
    return producer(**params)
