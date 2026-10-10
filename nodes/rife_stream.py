"""Memory-bounded RIFE 2x interpolation for H3 output frames."""

from __future__ import annotations

import sys

import folder_paths
import torch
import torch.nn.functional as F
from comfy import model_management
from comfy.ldm.common_dit import pad_to_patch_size
from comfy_extras.nodes_frame_interpolation import FrameInterpolationModelLoader


DEFAULT_RIFE_MODEL = "rife_v4.26.safetensors"


class RifeProcessingError(RuntimeError):
    pass


def _node_output(result, index=0):
    result = getattr(result, "result", result)
    return result[index]


def probe_rife_capability(model_name=DEFAULT_RIFE_MODEL):
    """Fail fast when the selected ComfyUI frame-interpolation model is absent."""
    path = folder_paths.get_full_path("frame_interpolation", model_name)
    if not path:
        raise RuntimeError(
            "缺少 RIFE 运动平滑模型；请安装到 "
            f"models/frame_interpolation/{model_name}"
        )
    return path


def smoothed_frame_count(source_count, multiplier=2):
    source_count = max(0, int(source_count))
    multiplier = max(1, int(multiplier))
    if source_count < 2 or multiplier == 1:
        return source_count
    return (source_count - 1) * multiplier + 1


def _scene_change_score(first, second):
    """Return a cheap low-resolution luma difference in the range 0..1."""
    pair = torch.stack((first[..., :3], second[..., :3])).to(
        device="cpu", dtype=torch.float32
    ).movedim(-1, 1)
    target_height = min(32, int(pair.shape[-2]))
    target_width = min(32, int(pair.shape[-1]))
    if tuple(pair.shape[-2:]) != (target_height, target_width):
        pair = F.interpolate(pair, size=(target_height, target_width), mode="area")
    luma = (
        pair[:, 0] * 0.2126
        + pair[:, 1] * 0.7152
        + pair[:, 2] * 0.0722
    )
    return float((luma[1] - luma[0]).abs().mean().item())


class RifeFrameProcessor:
    """Load one core RIFE patcher and interpolate one adjacent pair at a time."""

    def __init__(self, model_name=DEFAULT_RIFE_MODEL):
        probe_rife_capability(model_name)
        self._model_management = model_management
        self.patcher = _node_output(FrameInterpolationModelLoader.execute(model_name))
        self.device = self.patcher.load_device
        self.dtype = self.patcher.model_dtype()
        self.model = self.patcher.model
        self.align = int(getattr(self.model, "pad_align", 1))
        self._next_features = None
        self._closed = False

    def _prepare(self, frame):
        image = frame[..., :3].unsqueeze(0).movedim(-1, 1).to(
            device=self.device,
            dtype=self.dtype,
        )
        if self.align > 1:
            image = pad_to_patch_size(
                image,
                (self.align, self.align),
                padding_mode="reflect",
            )
        return image

    def process(self, first, second):
        height, width = int(first.shape[0]), int(first.shape[1])
        activation_mem = self.model.memory_used_forward(
            (2, height, width, 3),
            self.dtype,
        )
        self._model_management.load_models_gpu(
            [self.patcher],
            memory_required=activation_mem,
        )
        image0 = self._prepare(first)
        image1 = self._prepare(second)
        cache = {
            "img0": self._next_features
            if self._next_features is not None
            else self.model.extract_features(image0),
            "img1": self.model.extract_features(image1),
        }
        self._next_features = cache["img1"].detach()
        midpoint = self.model(image0, image1, timestep=0.5, cache=cache)
        return (
            midpoint[0, :, :height, :width]
            .movedim(0, -1)
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .clamp(0.0, 1.0)
            .contiguous()
        )

    def reset(self):
        """Drop the adjacent-frame feature cache after a real scene cut."""
        self._next_features = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._next_features = None
        errors = []
        try:
            self._model_management.unload_model_and_clones(self.patcher)
        except (AttributeError, RuntimeError) as exc:
            errors.append(exc)
        try:
            self._model_management.soft_empty_cache()
        except (AttributeError, RuntimeError) as exc:
            errors.append(exc)
        if errors:
            raise RifeProcessingError(f"RIFE 资源释放失败：{errors[0]}") from errors[0]


def iter_rife_frames(
    images,
    processor_factory=RifeFrameProcessor,
    model_name=DEFAULT_RIFE_MODEL,
    scene_cut_threshold=0.35,
    pingpong=False,
):
    """Yield source/midpoint/source frames without materializing the 2x clip."""
    if len(images) == 0:
        return

    def source_frames():
        yield from (images[index] for index in range(len(images)))
        if pingpong:
            yield from (images[index] for index in range(len(images) - 2, 0, -1))

    frames = source_frames()
    previous = next(frames)
    try:
        processor = processor_factory(model_name)
    except Exception as exc:
        raise RifeProcessingError(f"RIFE 模型加载失败：{exc}") from exc
    try:
        yield previous.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for current in frames:
            if _scene_change_score(previous, current) >= float(scene_cut_threshold):
                reset = getattr(processor, "reset", None)
                if callable(reset):
                    reset()
                midpoint = previous.detach().to(device="cpu", dtype=torch.float32).clone()
            else:
                try:
                    midpoint = processor.process(previous, current)
                except Exception as exc:
                    raise RifeProcessingError(f"RIFE 插帧失败：{exc}") from exc
            yield midpoint
            yield current.detach().to(device="cpu", dtype=torch.float32).contiguous()
            previous = current
    finally:
        active_error = sys.exc_info()[1]
        try:
            processor.close()
        except Exception:
            if active_error is None or isinstance(active_error, GeneratorExit):
                raise
