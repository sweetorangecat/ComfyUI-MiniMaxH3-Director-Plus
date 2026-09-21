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
        backend = _backend()
        if kwargs.get("cut_detection", "none") != "none":
            try:
                backend._make_cut_detector(kwargs.get("cut_threshold", 3.0))
            except ImportError as error:
                raise RuntimeError("FaceRefine 切镜检测需要 scenedetect，请安装 requirements-facerefine.txt。") from error
        reference = kwargs.get("identity_reference")
        if reference is not None:
            if not kwargs.get("identity_track", True):
                raise ValueError("连接身份参考时必须启用 identity_track，避免参考被忽略。")
            try:
                embedder = backend._make_embedder(kwargs.get("identity_model", "insightface"),
                                                 kwargs.get("identity_clip_vision"))
                detector = _load_detector(kwargs["detector"]) if getattr(embedder, "needs_detector", True) else None
                anchor = embedder.embed_reference(reference[:1], detector, kwargs.get("confidence", 0.35))
            except Exception as error:
                raise RuntimeError("FaceRefine 身份匹配初始化失败；请检查所选身份模型依赖。") from error
            if anchor is None:
                raise ValueError("FaceRefine 身份参考未识别出有效人脸，请使用该人物清晰的单人参考图。")
        result = super().run(**kwargs)
        if reference is not None:
            report = result[3]
            if any(message in report for message in (
                "unavailable, tracking by continuity", "absent-shot detection failed",
                "needs a usable identity anchor; ignored",
            )):
                raise RuntimeError("FaceRefine 身份跟踪退化，已停止后续重绘。请检查跟踪报告：" + report)
        return result


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


class MiniMaxH3FaceRefineInputs:
    CATEGORY = _FaceNode.CATEGORY
    FUNCTION = "prepare"
    RETURN_TYPES = ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE", "IMAGE")
    RETURN_NAMES = ("refine_guide", "identity_reference")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",),
                "identity_picture": ("INT", {"default": 1, "min": 1, "max": 12,
                    "tooltip": "要修复的人物在导演台中的 Picture 编号；一次只修复此人。请选择单人清晰参考。"}),
            },
            "optional": {"identity_override": ("IMAGE", {
                "tooltip": "可接同一人物的单人脸部参考用于跟踪；保留导演台原有图片编号。无普通参考图的模式必须连接。"})},
        }

    def prepare(self, guide, identity_picture, identity_override=None):
        state = guide.copy()
        refs = dict(guide.get("ref_images") or {})
        key = f"ref_image_{identity_picture}"
        if not refs:
            if identity_override is None or identity_picture != 1:
                raise ValueError("此模式没有普通角色参考图，请连接人脸修复 identity_override，并选择 Picture 1。")
            refs[key] = identity_override
        if key not in refs:
            raise ValueError(f"人脸修复选择的 <Picture {identity_picture}> 不存在，请核对导演台参考图编号。")
        identity = identity_override if identity_override is not None else refs[key]
        state["ref_images"] = refs
        return state, identity


class MiniMaxH3FaceRefineConditioning:
    CATEGORY = _FaceNode.CATEGORY
    FUNCTION = "apply"
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("conditioning", "av_latent")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",), "vae": ("VAE",), "audio_vae": ("VAE",),
            "guide": ("MINIMAX_H3_DIRECTOR_PLUS_GUIDE",),
            "width": ("INT", {"forceInput": True}),
            "height": ("INT", {"forceInput": True}),
            "length": ("INT", {"forceInput": True}),
        }}

    def apply(self, clip, vae, audio_vae, guide, width, height, length):
        from .guide import native_node, normalize_h3_reference_audio
        from .performance import memory_policy

        with memory_policy(guide):
            return native_node("MiniMaxH3ReferenceToVideo").execute(
                clip=clip, vae=vae, audio_vae=audio_vae, prompt=guide["prompt"],
                width=width, height=height, length=length,
                ref_image_size=guide.get("ref_image_size", "match"),
                ref_images=guide.get("ref_images", {}),
                ref_videos=guide.get("ref_videos", {}),
                ref_video_audios=guide.get("ref_video_audios", {}),
                ref_audios={key: normalize_h3_reference_audio(value)
                            for key, value in (guide.get("ref_audios") or {}).items()},
            )


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    MiniMaxH3FaceTrackCrop, MiniMaxH3FaceStitch, MiniMaxH3FaceInjectVideoLatent,
    MiniMaxH3FacePerFrameDenoise, MiniMaxH3FaceTransformInfo, MiniMaxH3FaceAudioLock,
    MiniMaxH3FaceRefineSwitch, MiniMaxH3FaceRefineInputs, MiniMaxH3FaceRefineConditioning,
)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3FaceTrackCrop": "H3 人脸跟踪与裁剪",
    "MiniMaxH3FaceStitch": "H3 人脸修复合成回原片",
    "MiniMaxH3FaceInjectVideoLatent": "H3 人脸原帧重绘注入",
    "MiniMaxH3FacePerFrameDenoise": "H3 人脸逐帧重绘强度",
    "MiniMaxH3FaceTransformInfo": "H3 人脸跟踪报告",
    "MiniMaxH3FaceAudioLock": "H3 人脸修复原音轨锁定",
    "MiniMaxH3FaceRefineSwitch": "H3 人脸修复开关（懒加载）",
    "MiniMaxH3FaceRefineInputs": "H3 人脸修复参考（原始角色图）",
    "MiniMaxH3FaceRefineConditioning": "H3 人脸修复条件（沿用导演台）",
}
