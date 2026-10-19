"""Optional decoded-video finishing, independent of H3 latent sampling."""
from __future__ import annotations

import math
import logging
import time
from fractions import Fraction

LOGGER = logging.getLogger(__name__)


def output_size(width, height, preset):
    short = {'480p': 480, '768p': 768, '1080p': 1080}[preset]
    if min(width, height) <= 0:
        raise ValueError('输入视频尺寸必须大于零')
    scale = max(1, short / min(width, height))
    return tuple(max(2, round(v * scale / 2) * 2) for v in (width, height))


def vosr2_input_size(width, height, target_width, target_height):
    """Adapt an already aspect-resolved canvas to TE's integer-scale API.

    Fractional factors are evaluated by VOSR2's native 1x restoration at the
    required canvas, not by a larger integer SR pass followed by downsampling.
    Resolve aspect exactly once with sr_output_size before calling this helper.
    """
    if min(width, height, target_width, target_height) <= 0:
        raise ValueError('视频超分尺寸必须大于零')
    target_width, target_height = max(width, target_width), max(height, target_height)
    if (target_width % width == 0 and target_height % height == 0
            and target_width // width == target_height // height):
        return width, height, target_width // width
    return target_width, target_height, 1


def sr_output_size(width, height, target_width, target_height, engine):
    """Preserve source pixels/aspect; the selected output is a lower bound."""
    if min(width, height, target_width, target_height) <= 0:
        raise ValueError('视频超分尺寸必须大于零')
    scale = max(Fraction(1), Fraction(target_width, width), Fraction(target_height, height))
    return tuple(math.ceil(v * scale / 2) * 2 for v in (width, height))


def validate_sr_result(result, frame_count, *, check_finite=True):
    import torch
    if not isinstance(result, torch.Tensor) or result.ndim != 4 or result.shape[-1] != 3:
        raise RuntimeError('视频超分返回了无效 IMAGE 张量')
    if result.shape[0] != frame_count:
        raise RuntimeError('视频超分改变了帧数，已停止以避免音画不同步')
    if check_finite:
        for start in range(0, frame_count, 4):
            if not torch.isfinite(result[start:start + 4]).all().item():
                raise RuntimeError('视频超分返回非有限像素（NaN/Inf），已停止以避免输出损坏画面')
    return result


def _prepare_vosr2_input(images, width, height):
    """Keep floating-point detail and bound interpolation scratch to 64 MiB.

    TE processes a whole clip in one invocation. Preallocate its input rather
    than concatenate resized chunks, which would temporarily double host RAM.
    """
    import torch
    import torch.nn.functional as F
    if (images.shape[2], images.shape[1]) == (width, height):
        return images.detach().to(device='cpu', dtype=torch.float32)
    output = torch.empty((len(images), height, width, 3), device='cpu', dtype=torch.float32)
    chunk_frames = max(1, (64 * 1024 * 1024) // (width * height * 3 * 4))
    for start in range(0, len(images), chunk_frames):
        chunk = images[start:start + chunk_frames].detach().to(device='cpu', dtype=torch.float32)
        resized = F.interpolate(chunk.movedim(-1, 1), size=(height, width),
                                mode='bicubic', align_corners=False, antialias=True)
        output[start:start + chunk_frames].copy_(resized.movedim(1, -1).clamp(0, 1))
    return output


def _vosr2_cuda_device():
    import torch
    if not torch.cuda.is_available():
        return None
    try:
        import comfy.model_management as model_management
        device = model_management.get_torch_device()
        return device if device.type == 'cuda' else None
    except (ImportError, AttributeError):
        return torch.cuda.current_device()


def _vosr2_memory_snapshot():
    """Device-wide free memory and this process's allocator counters, in GiB."""
    import torch
    try:
        device = _vosr2_cuda_device()
        if device is None:
            return {'available': False}
        free, total = torch.cuda.mem_get_info(device)
        return {
            'available': True, 'device': str(device),
            'device_name': torch.cuda.get_device_name(device),
            'free_vram_gb': free / 1024**3, 'total_vram_gb': total / 1024**3,
            'allocated_gb': torch.cuda.memory_allocated(device) / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved(device) / 1024**3,
            # Do not reset shared CUDA counters used by other Comfy nodes.
            'process_peak_allocated_gb': torch.cuda.max_memory_allocated(device) / 1024**3,
            'process_peak_reserved_gb': torch.cuda.max_memory_reserved(device) / 1024**3,
        }
    except Exception as exc:
        return {'available': False, 'measurement_error': type(exc).__name__}


def _vosr2(images, width, height, seed, profile='legacy', diagnostics=None):
    from .two_stage_assets import _comfy_node_mappings, _resolve_upscaler_callable
    from .stream_output import _release_comfy_models_before_video_sr, _unwrap_node_result
    from .vosr2_profiles import VOSR2_PROFILE_NAMES, resolve_vosr2_profile
    import torch
    report = diagnostics if diagnostics is not None else {}
    report.clear()
    report.update(profile=profile, status='running', seed=int(seed),
                  input_shape=list(images.shape), requested_size=[width, height],
                  timing_scope='model release, load, settings and VOSR2 reconstruction; excludes final resize/encoding')
    start = time.perf_counter()
    try:
        if profile not in VOSR2_PROFILE_NAMES:
            raise ValueError(f'未知 VOSR2 配置：{profile!r}')
        mappings = _comfy_node_mappings()
        ids = ('TESpeedVOSR2Loader', 'TESpeedVOSR2Settings', 'TESpeedVOSR2Video')
        missing = [name for name in ids if name not in mappings]
        if missing:
            raise RuntimeError('缺少 TE-Speed-VOSR2 节点，请安装并重启 ComfyUI：' + ', '.join(missing))
        _release_comfy_models_before_video_sr()
        device = _vosr2_cuda_device()
        if device is not None:
            torch.cuda.synchronize(device)
        report['memory_before'] = _vosr2_memory_snapshot()
        config = resolve_vosr2_profile(profile,
            free_vram_gb=report['memory_before'].get('free_vram_gb'),
            total_vram_gb=report['memory_before'].get('total_vram_gb'))
        report['requested_config'] = config
        LOGGER.info('[H3 VOSR2] profile=%s settings=%s', profile, config)
        loader, settings_node, video = [_resolve_upscaler_callable(mappings[name]) for name in ids]
        model = _unwrap_node_result(loader(**config['loader']))
        settings = _unwrap_node_result(settings_node(**config['settings']))
        # Report upstream's returned scalar settings when exposed, not presumed defaults.
        upstream = settings if isinstance(settings, dict) else vars(settings) if hasattr(settings, '__dict__') else {}
        report['upstream_settings'] = {key: value for key, value in upstream.items()
            if key in config['settings'] and isinstance(value, (str, int, float, bool, type(None)))
            and not (isinstance(value, float) and not math.isfinite(value))}
        # width/height are resolved once by the caller; rounding aspect again
        # would grow the canvas at each adapter boundary.
        source_width, source_height = images.shape[2], images.shape[1]
        input_width, input_height, scale = vosr2_input_size(source_width, source_height, width, height)
        report.update(scale=max(input_width * scale / source_width, input_height * scale / source_height),
                      backend_scale=scale, processing_input_size=[input_width, input_height],
                      sizing_strategy='target_canvas_1x' if (input_width, input_height) != (source_width, source_height)
                      else 'native_integer')
        LOGGER.info('[H3 VOSR2] %sx%s -> %sx%s, effective_scale=%.6f backend_scale=%s strategy=%s',
                    source_width, source_height, input_width * scale, input_height * scale,
                    report['scale'], scale, report['sizing_strategy'])
        images = _prepare_vosr2_input(images, input_width, input_height)
        result = _unwrap_node_result(video(model=model, images=images, scale=scale,
            seed=int(seed), settings=settings, **config['video']))
        validate_sr_result(result, images.shape[0])
        if result.shape[2] < width or result.shape[1] < height:
            raise RuntimeError('VOSR2 输出尺寸低于请求的最低分辨率')
        if device is not None:
            torch.cuda.synchronize(device)
        report.update(status='completed', output_shape=list(result.shape))
        return result
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:500])
        raise
    finally:
        report['elapsed_seconds'] = round(time.perf_counter() - start, 6)
        report['memory_after'] = _vosr2_memory_snapshot()
        LOGGER.info('[H3 VOSR2] profile=%s status=%s elapsed=%.3fs memory=%s',
                    profile, report['status'], report['elapsed_seconds'], report['memory_after'])


class MiniMaxH3VideoFinish:
    @classmethod
    def INPUT_TYPES(cls):
        from .vosr2_profiles import VOSR2_PROFILE_NAMES
        return {'required': {
            'images': ('IMAGE',),
            'resolution': (['480p', '768p', '1080p'], {'default': '1080p'}),
            'method': (['resize', 'VOSR2', 'SeedVR2'], {'default': 'VOSR2'}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 0xffffffffffffffff}),
        }, 'optional': {
            'vosr2_profile': (list(VOSR2_PROFILE_NAMES), {'default': 'legacy'}),
            'benchmark_run': ('STRING', {'default': '', 'advanced': True,
                                        'tooltip': '对比工具的唯一运行标识，避免复用缓存计时；日常留空'}),
        }}

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'finish'
    CATEGORY = 'MiniMax H3 导演台 Plus'

    def finish(self, images, resolution='1080p', method='VOSR2', seed=42,
               vosr2_profile='legacy', benchmark_run=''):
        import torch
        from .stream_output import _resize_cpu_chunk, _iter_video_sr_frame_chunks
        from .execution_report import attach_report, new_report, reconstruction_diagnostics, stage_timer
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[-1] != 3:
            raise ValueError('需要非空 RGB 视频帧')
        width, height = output_size(images.shape[2], images.shape[1], resolution)
        if method in {'VOSR2', 'SeedVR2'}:
            width, height = sr_output_size(images.shape[2], images.shape[1], width, height, method.lower())
        start = time.perf_counter()
        diagnostics = {}
        evidence = new_report(source=images)
        with stage_timer(evidence, 'reconstruction', path=method, profile=vosr2_profile,
                         includes_nested_stages=True,
                         scope='standalone finish, including model release/load and final CPU resize'):
            if method == 'resize':
                result = images
            elif method == 'VOSR2':
                result = _vosr2(images, width, height, seed, profile=vosr2_profile, diagnostics=diagnostics)
            elif method == 'SeedVR2':
                import folder_paths
                from .video_sr import seedvr2_dependency_report, resolve_seedvr2_fhd_plan
                dependency = seedvr2_dependency_report(folder_paths.base_path)
                if not dependency['ready']:
                    raise RuntimeError('SeedVR2 未就绪：' + ', '.join(dependency['missing']))
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
                plan = resolve_seedvr2_fhd_plan(total, dependency['available_dit'])
                if plan['dit_model'] is None:
                    raise RuntimeError('独立 1080p 路线需要 SeedVR2 3B 权重')
                result = torch.cat(list(_iter_video_sr_frame_chunks(images, width, height, seed=seed, plan=plan)), dim=0)
            else:
                raise ValueError('不支持的视频处理方式：' + method)
            validate_sr_result(result, images.shape[0], check_finite=method != 'VOSR2')
            chunks = [_resize_cpu_chunk(result[i:i+4], width, height, method='lanczos') for i in range(0,len(result),4)]
            output = torch.cat(chunks).clamp(0,1)
        if diagnostics:
            evidence['reconstruction'] = reconstruction_diagnostics(diagnostics)
        attach_report(output, evidence)
        LOGGER.info('[H3 video finish] %s %s -> %sx%s, %s frames, %.1fs; no H3 second pass', method, resolution, width, height, len(output), time.perf_counter()-start)
        if benchmark_run:
            diagnostics.update(benchmark_run=str(benchmark_run)[:100],
                               finish_elapsed_seconds=round(time.perf_counter()-start, 6),
                               finished_output_shape=list(output.shape))
            return {'ui': {'vosr2_diagnostics': [diagnostics]}, 'result': (output,)}
        return (output,)
