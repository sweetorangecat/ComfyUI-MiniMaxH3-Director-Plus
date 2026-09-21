"""Director Plus face refinement using a bundled, lazily loaded MIT backend."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def detector_names():
    import folder_paths

    names = set()
    for category in ("ultralytics_bbox", "ultralytics"):
        try:
            names.update(folder_paths.get_filename_list(category))
        except (KeyError, FileNotFoundError):
            pass
    root = Path(folder_paths.models_dir) / "ultralytics"
    if root.is_dir():
        names.update(path.relative_to(root).as_posix() for path in root.rglob("*.pt"))
    return sorted(names) or ["bbox/face_yolov8m.pt"]


def detector_path(name):
    import folder_paths

    for category in ("ultralytics_bbox", "ultralytics"):
        try:
            path = folder_paths.get_full_path(category, name)
        except (KeyError, FileNotFoundError):
            path = None
        if path and Path(path).is_file():
            return path
    root = (Path(folder_paths.models_dir) / "ultralytics").resolve()
    for candidate in (root / name, root / "bbox" / name):
        candidate = candidate.resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"FaceRefine 人脸检测模型不存在：{name}。请放入 models/ultralytics/bbox。")


def _load_detector(name):
    from ._vendor import h3_face_refine as backend

    path = detector_path(name)
    if path not in backend._DETECTOR_CACHE:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "FaceRefine 缺少 ultralytics；请在 ComfyUI 环境安装 requirements-facerefine.txt。"
            ) from error
        backend._DETECTOR_CACHE[path] = YOLO(path)
    return backend._DETECTOR_CACHE[path]


def _backend():
    from ._vendor import h3_face_refine

    # The upstream model registry assumes Impact Subpack; use our own discovery.
    h3_face_refine._detector_list = detector_names
    h3_face_refine._load_detector = _load_detector
    return h3_face_refine


class _FaceNode:
    CATEGORY = "MiniMax H3 导演台 Plus/人脸修复"
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return deepcopy(getattr(_backend(), cls.BACKEND).INPUT_TYPES())

    def run(self, **kwargs):
        return getattr(_backend(), self.BACKEND)().run(**kwargs)


class MiniMaxH3FaceTrackCrop(_FaceNode):
    BACKEND = "H3FaceTrackCrop"
    RETURN_TYPES = ("IMAGE", "H3FACEXFORM", "IMAGE", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("crops", "transform", "preview", "report", "canvas_w", "canvas_h", "frame_count")

    def run(self, **kwargs):
        images = kwargs["images"]
        if images.ndim != 4 or not len(images) or images.shape[-1] < 3:
            raise ValueError("FaceRefine 需要非空 NHWC 视频帧。")
        return super().run(**kwargs)


class MiniMaxH3FaceStitch(_FaceNode):
    BACKEND = "H3FaceStitch"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    def run(self, **kwargs):
        base = kwargs["base_images"]
        crops = kwargs["refined_crops"]
        transform = kwargs["transform"]
        if base.ndim != 4 or tuple(transform["src_size"]) != (base.shape[2], base.shape[1]):
            raise ValueError("FaceRefine 合成原片尺寸与跟踪 source 尺寸不一致。")
        if crops.ndim != 4 or not len(crops):
            raise ValueError("FaceRefine 修复帧不能为空。")
        if tuple(transform["canvas"]) != (crops.shape[2], crops.shape[1]):
            raise ValueError("FaceRefine 修复帧尺寸与跟踪画布不一致。")
        if any(index < 0 or index >= len(base) for index in transform.get("source", [])):
            raise ValueError("FaceRefine 跟踪帧索引超出原片范围。")
        return super().run(**kwargs)


class MiniMaxH3FaceInjectVideoLatent(_FaceNode):
    BACKEND = "H3InjectVideoLatent"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("av_latent", "report")


class MiniMaxH3FacePerFrameDenoise(_FaceNode):
    BACKEND = "H3PerFrameDenoise"
    RETURN_TYPES = ("LATENT", "STRING", "MODEL")
    RETURN_NAMES = ("av_latent", "report", "model")


class MiniMaxH3FaceTransformInfo(_FaceNode):
    BACKEND = "H3FaceTransformInfo"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True


class MiniMaxH3FaceAudioLock:
    CATEGORY = _FaceNode.CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("MODEL", "LATENT", "AUDIO")
    RETURN_NAMES = ("model", "av_latent", "original_audio")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "av_latent": ("LATENT",),
                             "audio_vae": ("VAE",), "audio": ("AUDIO",)}}

    def run(self, model, av_latent, audio_vae, audio):
        import torch
        import torch.nn.functional as F
        from comfy.nested_tensor import NestedTensor

        samples = av_latent.get("samples")
        if not getattr(samples, "is_nested", False):
            raise ValueError("FaceRefine 音轨锁定需要 H3 联合 AV latent。")
        members = list(samples.unbind())
        if len(members) < 2:
            raise ValueError("H3 AV latent 缺少音频流。")
        waveform = audio.get("waveform")
        rate = int(audio.get("sample_rate", 0))
        if waveform is None or waveform.ndim != 3 or waveform.shape[0] != 1 or not waveform.shape[-1] or rate <= 0:
            raise ValueError("FaceRefine 需要单段非空原始音轨和有效采样率。")
        vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if rate != vae_rate:
            from torchaudio.functional import resample
            waveform = resample(waveform, rate, vae_rate)
        encoded = audio_vae.encode(waveform.movedim(1, -1))
        target = members[1]
        if encoded.shape[:-1] != target.shape[:-1]:
            raise ValueError("FaceRefine 音频 VAE 输出与 H3 音频 latent 形状不匹配。")
        encoded = encoded[..., :target.shape[-1]]
        encoded = F.pad(encoded, (0, target.shape[-1] - encoded.shape[-1]))
        members[1] = encoded.to(device=target.device, dtype=target.dtype)
        masks = [torch.zeros_like(member) for member in members]
        masks[0] = torch.ones_like(members[0])
        result = dict(av_latent, samples=NestedTensor(members), noise_mask=NestedTensor(masks))
        patched = model.clone()
        options = dict(patched.model_options.get("transformer_options", {}))
        options["minimax_h3_lock_audio_clean"] = True
        patched.model_options["transformer_options"] = options
        return patched, result, audio


class MiniMaxH3FaceRefineSwitch:
    CATEGORY = _FaceNode.CATEGORY
    FUNCTION = "select"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",)},
            "optional": {
                "original_images": ("IMAGE", {"lazy": True}),
                "refined_images": ("IMAGE", {"lazy": True}),
            },
        }

    @staticmethod
    def _selected(guide):
        return "refined_images" if guide.get("face_refine_mode", "off") == "auto" else "original_images"

    def check_lazy_status(self, guide, original_images=None, refined_images=None):
        selected = self._selected(guide)
        return [selected] if locals()[selected] is None else []

    def select(self, guide, original_images=None, refined_images=None):
        selected_name = self._selected(guide)
        selected = refined_images if selected_name == "refined_images" else original_images
        if selected is None:
            raise ValueError("人脸修复输出未连接" if selected_name == "refined_images" else "原始视频帧未连接")
        return (selected,)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    MiniMaxH3FaceTrackCrop, MiniMaxH3FaceStitch, MiniMaxH3FaceInjectVideoLatent,
    MiniMaxH3FacePerFrameDenoise, MiniMaxH3FaceTransformInfo, MiniMaxH3FaceAudioLock,
    MiniMaxH3FaceRefineSwitch,
)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3FaceTrackCrop": "H3 人脸跟踪与裁剪",
    "MiniMaxH3FaceStitch": "H3 人脸修复合成回原片",
    "MiniMaxH3FaceInjectVideoLatent": "H3 人脸原帧重绘注入",
    "MiniMaxH3FacePerFrameDenoise": "H3 人脸逐帧重绘强度",
    "MiniMaxH3FaceTransformInfo": "H3 人脸跟踪报告",
    "MiniMaxH3FaceAudioLock": "H3 人脸修复原音轨锁定",
    "MiniMaxH3FaceRefineSwitch": "H3 人脸修复开关（懒加载）",
}
