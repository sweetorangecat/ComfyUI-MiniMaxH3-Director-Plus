"""Optional decoded-video finishing, independent of H3 latent sampling."""
from __future__ import annotations

import math
import logging

LOGGER = logging.getLogger(__name__)


def output_size(width, height, preset):
    short = {'480p': 480, '768p': 768, '1080p': 1080}[preset]
    if min(width, height) <= 0:
        raise ValueError('输入视频尺寸必须大于零')
    scale = short / min(width, height)
    return tuple(max(2, round(v * scale / 2) * 2) for v in (width, height))


def vosr2_input_size(width, height, target_width, target_height):
    scale = max(1, math.ceil(max(target_width / width, target_height / height)))
    return math.ceil(target_width / scale), math.ceil(target_height / scale), scale


def validate_sr_result(result, frame_count):
    import torch
    if not isinstance(result, torch.Tensor) or result.ndim != 4 or result.shape[-1] != 3:
        raise RuntimeError('视频超分返回了无效 IMAGE 张量')
    if result.shape[0] != frame_count:
        raise RuntimeError('视频超分改变了帧数，已停止以避免音画不同步')
    return result


def _vosr2(images, width, height, seed):
    from .two_stage_assets import _comfy_node_mappings, _resolve_upscaler_callable
    from .stream_output import _release_comfy_models_before_video_sr, _unwrap_node_result, _resize_cpu_chunk
    mappings = _comfy_node_mappings()
    ids = ('TESpeedVOSR2Loader', 'TESpeedVOSR2Settings', 'TESpeedVOSR2Video')
    missing = [name for name in ids if name not in mappings]
    if missing:
        raise RuntimeError('缺少 TE-Speed-VOSR2 节点，请安装并重启 ComfyUI：' + ', '.join(missing))
    _release_comfy_models_before_video_sr()
    loader, settings_node, video = [_resolve_upscaler_callable(mappings[name]) for name in ids]
    model = _unwrap_node_result(loader(model_bundle='VOSR2', precision='auto', memory_policy='auto', torch_compile=False, vae_encode_amp=False))
    settings = _unwrap_node_result(settings_node(
        quality_profile='speed', tile_strategy='auto', tile_size=512, tile_overlap=32,
        vae_tile_size=1024, vae_tile_overlap=32, image_batch=1, frame_batch=2,
        dino_batch=2, temporal_cache=False, cache_threshold=0.003, cache_refresh=4,
        memory_policy='auto', color_alignment='wavelet'))
    iw, ih, scale = vosr2_input_size(images.shape[2], images.shape[1], width, height)
    prepared = _resize_cpu_chunk(images, iw, ih, method='lanczos')
    return _unwrap_node_result(video(model=model, images=prepared, scale=scale, seed=int(seed),
        settings=settings, frame_batch=2, temporal_cache=False, cache_threshold=0.003, cache_refresh=4))


class MiniMaxH3VideoFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'images': ('IMAGE',),
            'resolution': (['480p', '768p', '1080p'], {'default': '1080p'}),
            'method': (['resize', 'VOSR2', 'SeedVR2'], {'default': 'VOSR2'}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'finish'
    CATEGORY = 'MiniMax H3 导演台 Plus'

    def finish(self, images, resolution='1080p', method='VOSR2', seed=42):
        import torch
        import time
        from .stream_output import _resize_cpu_chunk, _iter_video_sr_frame_chunks
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[-1] != 3:
            raise ValueError('需要非空 RGB 视频帧')
        width, height = output_size(images.shape[2], images.shape[1], resolution)
        start = time.perf_counter()
        if method == 'resize':
            result = images
        elif method == 'VOSR2':
            result = _vosr2(images, width, height, seed)
        elif method == 'SeedVR2':
            import folder_paths
            from .video_sr import seedvr2_dependency_report, resolve_seedvr2_fhd_plan
            report = seedvr2_dependency_report(folder_paths.base_path)
            if not report['ready']:
                raise RuntimeError('SeedVR2 未就绪：' + ', '.join(report['missing']))
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
            plan = resolve_seedvr2_fhd_plan(total, report['available_dit'])
            if plan['dit_model'] is None:
                raise RuntimeError('独立 1080p 路线需要 SeedVR2 3B 权重')
            result = torch.cat(list(_iter_video_sr_frame_chunks(images, width, height, seed=seed, plan=plan)), dim=0)
        else:
            raise ValueError('不支持的视频处理方式：' + method)
        validate_sr_result(result, images.shape[0])
        chunks = [_resize_cpu_chunk(result[i:i+4], width, height, method='lanczos') for i in range(0,len(result),4)]
        output = torch.cat(chunks).clamp(0,1)
        LOGGER.info('[H3 video finish] %s %s -> %sx%s, %s frames, %.1fs; no H3 second pass', method, resolution, width, height, len(output), time.perf_counter()-start)
        return (output,)
