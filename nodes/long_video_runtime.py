"""Disk-backed native AV nodes for the existing U11 SelfLift and VOSR2 path.

Only clean native tensors continue a scene. Technical validation never grants a
content review; enhancement consumes only explicitly accepted delivery frames.
"""
from __future__ import annotations

from copy import deepcopy
import json
from math import ceil
import os
import re
import uuid
import wave

from .long_video_av import conditioning_geometry, prepare_av, slice_fingerprint, streams, validate_result
from .long_video_plan import FPS, audio_ticks, video_steps, validate_window
from .long_video_store import ProjectStore
from .stream_output import MiniMaxH3StreamingVideoCombine


GUIDE = "MINIMAX_H3_DIRECTOR_PLUS_GUIDE"
CATEGORY = "MiniMax H3 导演台 Plus/原生长视频"


def _request(raw):
    try:
        request = json.loads(raw) if isinstance(raw, str) else deepcopy(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid long-video request JSON") from exc
    if not isinstance(request, dict) or request.get("stage") not in {"native", "preview", "finish"}:
        raise ValueError("Long-video request requires native, preview or finish stage")
    if not isinstance(request.get("project_name"), str) or not request["project_name"].strip():
        raise ValueError("Long-video request requires project_name")
    if not isinstance(request.get("take_id"), str) or not re.fullmatch(r"[\w-]{1,128}", request["take_id"]):
        raise ValueError("Invalid long-video take_id")
    request["window"] = validate_window(request.get("window"))
    return request


def _bound_store(request):
    import folder_paths
    store = ProjectStore(folder_paths.get_output_directory(), request["project_name"])
    saved = json.loads(store.path(f"takes/{request['take_id']}/request.json").read_text(encoding="utf-8"))
    transient = {"stage", "export_fingerprint"}
    if ({k: v for k, v in request.items() if k not in transient}
            != {k: v for k, v in saved.items() if k not in transient}):
        raise ValueError("Long-video request differs from the immutable take request")
    ProjectStore.active(store.read(), request["take_id"])
    from .long_video_parent import verify_parent_reference
    verify_parent_reference(store, request)
    return store


def _nested(values):
    from comfy.nested_tensor import NestedTensor
    return NestedTensor(values)


def _cpu_latent(latent):
    result = dict(latent)
    result["samples"] = _nested(tuple(t.detach().to(device="cpu").contiguous().clone() for t in streams(latent)))
    result.pop("noise_mask", None)
    return result


def _save_tensors(path, tensors):
    from safetensors.torch import save
    path.parent.mkdir(parents=True, exist_ok=True)
    data = save({k: v.detach().to(device="cpu").contiguous().clone() for k, v in tensors.items()})
    try:
        with path.open("xb") as dest:
            dest.write(data)
            dest.flush()
            os.fsync(dest.fileno())
    except FileExistsError as exc:
        raise ValueError("Refusing to overwrite immutable AV tensors") from exc


def _take_record(store, take_id):
    state = store.read()
    record = next((item for item in [*state["windows"], *state.get("superseded", [])]
                   if item["take_id"] == take_id), None)
    if record is None:
        raise ValueError("Unknown immutable clean take")
    return record


def _read_clean(store, take_id, record, artifacts):
    from safetensors.torch import load_file
    import torch
    store.verify(artifacts)
    request = json.loads(store.path(artifacts["request"]["name"]).read_text(encoding="utf-8"))
    if request["take_id"] != take_id or request["window"] != record["window"]:
        raise ValueError("Clean AV request provenance differs from the journal")
    values = tuple(load_file(str(store.path(artifacts["clean_" + name]["name"])), device="cpu")["samples"]
                   for name in ("video", "audio"))
    result = {"samples": _nested(values)}
    video, audio = streams(result)
    frames = validate_window(request["window"])["sample_frames"]
    if video.shape[2] != video_steps(frames) or audio.shape[-1] != audio_ticks(frames):
        raise ValueError("Saved clean AV geometry differs from its immutable window")
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in values):
        raise ValueError("Saved clean AV contains nonfinite values")
    return result


def _require_global_sampling(guide):
    if (guide.get("sampling_actual_method") != "selflift"
            or guide.get("selflift_highres_tiling_mode") != "global_stream"
            or not guide.get("selflift_highres_tiling")
            or guide.get("sampling_route") != "SelfLiftAvatarH3Sampler"
            or guide.get("attention_backend") != "sol_attn" or not guide.get("sol_attn_applied")):
        raise RuntimeError("The actual native sampler did not execute SelfLift Avatar global_stream with Sol-Attn")


def load_clean(store, take_id):
    """Load one verified CPU take; recover its durable receipt after a crash."""
    record = _take_record(store, take_id)
    artifacts = record.get("artifacts", {})
    if {"clean_video", "clean_audio", "request"}.issubset(artifacts):
        return _read_clean(store, take_id, record, artifacts)
    path = store.path(f"takes/{take_id}/clean_receipt.json")
    if not path.is_file():
        raise ValueError("The take has no recoverable clean AV checkpoint")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    _require_global_sampling(receipt["actual_sampling"])
    clean = _read_clean(store, take_id, record, receipt["artifacts"])
    baseline, prepared_artifacts = _load_prepared(store, take_id)
    if (any(receipt["artifacts"].get(key) != value for key, value in prepared_artifacts.items())
            or receipt["context"] != baseline["_h3_long_video_context"]
            or validate_result(clean, baseline, require_change=True) != receipt["tensor_validation"]):
        raise ValueError("Clean receipt differs from independent AV validation evidence")
    receipt["artifacts"]["clean_receipt"] = store.artifact(path)
    store.record_clean(take_id, receipt)
    return clean


def _load_prepared(store, take_id):
    from safetensors.torch import load_file
    path = store.path(f"takes/{take_id}/prepared_context.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    store.verify({**receipt["artifacts"], "request": receipt["request"]})
    values = [load_file(str(store.path(receipt["artifacts"]["prepared_" + name]["name"])), device="cpu")
              for name in ("video", "audio")]
    prepared = {"samples": _nested(tuple(value["samples"] for value in values)),
                "noise_mask": _nested(tuple(value["noise_mask"] for value in values)),
                "_h3_long_video_context": receipt["context"]}
    artifacts = {**receipt["artifacts"], "request": receipt["request"],
                 "prepared_context": store.artifact(path)}
    return prepared, artifacts


def _source(store, request):
    window = request["window"]
    forward = request.get('forward_frames', 0)
    conditioning_geometry(window, forward)
    parent = request.get("parent")
    source = None
    if window["context_frames"]:
        if not isinstance(parent, dict) or not parent.get("take_id") or not parent.get("history_sha256"):
            raise ValueError("Continuation requires its actual parent take and history SHA-256")
        record = ProjectStore.active(store.read(), parent["take_id"])
        previous = record["window"]
        if (record["status"] != "accepted" or previous["index"] != window["index"] - 1
                or previous["new_end"] != window["new_start"]
                or previous["local_new_end"] != window["source_endpoint"]):
            raise ValueError("Continuation parent must be the current accepted adjacent window")
        store.verify_accepted(record)
        if forward:
            if not parent.get('forward_review'):
                raise ValueError('Forward AV requires an explicit immutable actual source-tail review')
            proof = store.verify_forward_review(record, parent['forward_review'])
            if (proof['requirement']['next_window'] != window
                    or proof['requirement']['forward_frames'] != forward):
                raise ValueError('Forward source review geometry differs from the immutable request')
        source = load_clean(store, parent["take_id"])
        parent_history_sha256 = slice_fingerprint(source, window["source_endpoint"], window["context_frames"], forward)
        if parent_history_sha256 != parent["history_sha256"]:
            raise ValueError("Actual parent history SHA-256 differs from the immutable request")
    elif parent is not None:
        raise ValueError("A first window cannot have a continuation parent")
    repair = request.get("repair")
    if repair is not None:
        if not isinstance(repair, dict) or not repair.get("source_take"):
            raise ValueError("A bounded repair requires an immutable source_take")
        original = _take_record(store, repair["source_take"])
        if original["window"] != window:
            raise ValueError("Repair source must use the same physical and delivery window")
        if original.get('target_audio') != request.get('target_audio'):
            raise ValueError('Target-conditioned repair must inherit the failed take complete audio; changed audio requires a new base generation')
        repair_source = load_clean(store, repair["source_take"])
        if window["context_frames"]:
            # Repair locks this existing prefix, so it must also match the
            # actual history consumed from the current accepted parent.
            head = window["context_frames"]
            if slice_fingerprint(repair_source, head, head, forward) != parent_history_sha256:
                raise ValueError("Repair source protected AV history differs from the actual current parent; regenerate this window")
        source = repair_source
    return source


def _accepted_prefix(store, request):
    cursor = 0
    records = store.read()["windows"]
    for index in range(request["window"]["index"] + 1):
        record = records[index]
        if (record["status"] != "accepted" or record["window"]["index"] != index
                or record["window"]["new_start"] != cursor):
            raise ValueError("Enhancement requires accepted contiguous native windows")
        store.verify_accepted(record)
        cursor = record["window"]["new_end"]


def _node_value(value):
    value = getattr(value, "result", value)
    if isinstance(value, dict) and "result" in value:
        value = value["result"]
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def _decode_native(latent, video_vae, audio_vae):
    from .vae_decode import MiniMaxH3SafeVAEDecode
    from .two_stage_assets import _comfy_node_mappings
    images = MiniMaxH3SafeVAEDecode().decode(vae=video_vae, samples=latent)[0]
    factory = _comfy_node_mappings().get("VAEDecodeAudio")
    if not callable(factory):
        raise RuntimeError("Native VAEDecodeAudio node is not registered")
    node = factory()
    function = getattr(node, "execute", None) or getattr(node, "decode", None)
    if not callable(function):
        raise RuntimeError("Native VAEDecodeAudio has no decode interface")
    audio = _node_value(function(vae=audio_vae, samples=latent))
    return images, audio


def _publish_native(store, take_id, pending):
    """Recover publication even if a process stops between either file/link."""
    for key, artifact in pending["artifacts"].items():
        final = store.path(artifact["name"])
        staged = pending["staged"][key]
        if not final.exists():
            store.verify({key: staged})
            try:
                os.link(store.path(staged["name"]), final)
            except FileExistsError:
                pass
        store.verify({key: artifact})
    receipt = {key: value for key, value in pending.items() if key != "staged"}
    relative = f"takes/{take_id}/native_receipt.json"
    path = store.path(relative)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != receipt:
            raise ValueError("Immutable native receipt differs from staged media")
    else:
        store.write_once(relative, receipt)
    receipt["artifacts"] = {**receipt["artifacts"], "native_receipt": store.artifact(path)}
    record = ProjectStore.active(store.read(), take_id)
    if record["status"] not in {"awaiting_review", "accepted", "rejected"}:
        store.record_native(take_id, receipt)
    for artifact in pending["staged"].values():
        store.path(artifact["name"]).unlink(missing_ok=True)


class MiniMaxH3LongVideoWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"guide": (GUIDE,), "request_json": ("STRING",)}}

    RETURN_TYPES = (GUIDE,)
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, guide, request_json):
        request = _request(request_json)
        result = dict(guide)
        window = request["window"]
        result.update(length=window["sample_frames"], duration=window["sample_frames"] / FPS,
                      long_video_window=deepcopy(window))
        return (result,)


class MiniMaxH3LongVideoContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "guide": (GUIDE,),
                             "request_json": ("STRING",), "model": ("MODEL",)},
                "optional": {"target_audio": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(self, latent, guide, request_json, model, target_audio=None):
        request = _request(request_json)
        if request["stage"] != "native":
            raise ValueError("AV context preparation requires the native stage")
        if (guide.get("sampling_method") != "selflift" or guide.get("attention_backend") != "sol_attn"
                or not guide.get("sol_attn_applied")):
            raise RuntimeError("Native long-video AV requires SelfLift with applied Sol-Attn")
        if not callable(getattr(getattr(model, "model", model), "_denoise_mask_conds", None)):
            raise RuntimeError("Native H3 model does not support both AV denoise masks")
        store = _bound_store(request)
        locked = request.get('target_audio') is not None
        if locked:
            _validated_target(request)
        if not locked and target_audio is not None:
            raise ValueError('Encoded target audio must be declared in the immutable native request')
        prepared = prepare_av(_cpu_latent(latent), request["window"], source=_source(store, request),
                              repair=request.get("repair"), forward_frames=request.get('forward_frames', 0),
                              target_audio=target_audio, target_audio_locked=locked)
        if locked:
            prepared['_h3_long_video_context']['target_audio_sha256'] = request['target_audio']['sha256']
        video, audio = streams(prepared)
        vm, am = prepared["noise_mask"].unbind()
        base = f"takes/{request['take_id']}"
        artifacts = {}
        for name, sample, mask in (("video", video, vm), ("audio", audio, am)):
            path = store.path(f"{base}/prepared_{name}.safetensors")
            _save_tensors(path, {"samples": sample, "noise_mask": mask})
            artifacts["prepared_" + name] = store.artifact(path)
        store.write_once(f"{base}/prepared_context.json", {
            "context": prepared["_h3_long_video_context"], "artifacts": artifacts,
            "request": store.artifact(store.path(f"{base}/request.json")),
        })
        return (prepared,)


def _validated_target(request):
    from .long_video_audio import validate_target_audio
    return validate_target_audio(request.get('target_audio'), request['window'],
                                 request.get('target_dialogues', []), request.get('target_audio_original_sources', {}),
                                 request.get('forward_frames', 0))


class MiniMaxH3LongVideoTargetAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'request_json': ('STRING',)}}

    RETURN_TYPES = ('AUDIO',)
    FUNCTION = 'load'
    CATEGORY = CATEGORY

    def load(self, request_json):
        from .long_video_audio import read_wave
        request = _request(request_json)
        if request['stage'] != 'native' or request.get('repair'):
            raise ValueError('Target waveform loading belongs only to a new native base generation')
        _bound_store(request)
        return (read_wave(_validated_target(request)),)


class MiniMaxH3LongVideoCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "prepared": ("LATENT",),
                             "guide": (GUIDE,), "video_vae": ("VAE",), "audio_vae": ("VAE",),
                             "request_json": ("STRING",)}}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("clean_av", "native_preview")
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save(self, latent, prepared, guide, video_vae, audio_vae, request_json):
        from .long_video_media import encode_preview, write_delivery_audio
        request = _request(request_json)
        store = _bound_store(request)
        take_id = request["take_id"]
        base = f"takes/{take_id}"
        if request["stage"] == "native":
            _require_global_sampling(guide)
            baseline, artifacts = _load_prepared(store, take_id)
            # `prepared` is deliberately not trusted: it may alias sampler input.
            clean = _cpu_latent(latent)
            validation = validate_result(clean, baseline, require_change=True)
            for name, value in zip(("video", "audio"), streams(clean)):
                path = store.path(f"{base}/clean_{name}.safetensors")
                _save_tensors(path, {"samples": value})
                artifacts["clean_" + name] = store.artifact(path)
            actual = {key: deepcopy(value) for key, value in guide.items()
                      if key.startswith(("sampling_", "selflift_", "sol_attn_"))
                      or key in {"attention_backend", "length", "width", "height"}}
            receipt = {"artifacts": artifacts, "tensor_validation": validation,
                       "context": baseline["_h3_long_video_context"], "actual_sampling": actual,
                       "content_review": "pending"}
            clean_receipt = store.write_once(f"{base}/clean_receipt.json", receipt)
            receipt["artifacts"]["clean_receipt"] = store.artifact(clean_receipt)
            store.record_clean(take_id, receipt)
        elif request["stage"] == "preview":
            clean = load_clean(store, take_id)
        else:
            raise ValueError("Checkpoint only accepts native or preview stage")
        native, delivery = store.path(f"{base}/native.mp4"), store.path(f"{base}/delivery.wav")
        pending_path = store.path(f"{base}/native_pending.json")
        if pending_path.exists():
            _publish_native(store, take_id, json.loads(pending_path.read_text(encoding="utf-8")))
        else:
            if native.exists() or delivery.exists():
                raise ValueError("Refusing to overwrite an immutable native preview")
            images, audio = _decode_native(clean, video_vae, audio_vae)
            if images.ndim != 4 or len(images) != request["window"]["sample_frames"]:
                raise ValueError("Native decode frame count differs from the physical sample window")
            staged = {"native": store.path(f"{base}/.native-{uuid.uuid4().hex}.mp4"),
                      "delivery_audio": store.path(f"{base}/.delivery-{uuid.uuid4().hex}.wav")}
            durable = False
            try:
                audio_range = write_delivery_audio(staged["delivery_audio"], audio, request["window"])
                encode_preview(staged["native"], images, audio)
                temporary = {key: store.artifact(path) for key, path in staged.items()}
                final_paths = {"native": native, "delivery_audio": delivery}
                artifacts = {key: {**artifact, "name": final_paths[key].relative_to(store.root).as_posix()}
                             for key, artifact in temporary.items()}
                pending = {"artifacts": artifacts, "staged": temporary, "audio_sample_range": audio_range,
                           "content_review": "pending"}
                store.write_once(f"{base}/native_pending.json", pending)
                durable = True
                _publish_native(store, take_id, pending)
            finally:
                if not durable:
                    for path in staged.values():
                        path.unlink(missing_ok=True)
        asset = {"filename": native.name, "subfolder": native.parent.relative_to(store.output_root).as_posix(),
                 "type": "output", "format": "video/mp4"}
        return {"ui": {"gifs": [asset], "text": ["Native AV saved; content review pending"]},
                "result": (clean, str(native))}


class MiniMaxH3LongVideoLoadAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"request_json": ("STRING",)}}

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("clean_av", "denoised_av")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def load(self, request_json):
        request = _request(request_json)
        store = _bound_store(request)
        if request["stage"] == "finish":
            _accepted_prefix(store, request)
        elif request["stage"] != "preview":
            raise ValueError("Saved AV loading requires preview or finish stage")
        latent = load_clean(store, request["take_id"])
        # The real two-stage sampler emits the same nested AV on both outputs.
        # Existing separate/decode nodes still receive that exact contract.
        return latent, latent


def _delivery_audio(store, record):
    import numpy as np
    import torch
    artifact = record["artifacts"]["delivery_audio"]
    store.verify({"delivery_audio": artifact})
    with wave.open(str(store.path(artifact["name"])), "rb") as source:
        rate, channels, width, count = source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getnframes()
        expected = record["audio_sample_range"]
        if width != 2 or (rate, channels, count) != (expected["sample_rate"], expected["channels"], expected["count"]):
            raise ValueError("Saved delivery PCM differs from its exact audio sample index")
        raw = source.readframes(count)
    values = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).T.copy()
    return {"waveform": torch.from_numpy(values).unsqueeze(0).float() / 32767, "sample_rate": rate}


def _commit_export(store, request, receipt, path):
    with store.transaction() as journal:
        current = ProjectStore.active(journal, request["take_id"])
        if current["status"] != "accepted" or current["artifacts"] != receipt["source_artifacts"]:
            raise ValueError("Native source approval changed during enhancement")
        store.verify_accepted(current)
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != receipt:
                raise ValueError("Immutable export receipt differs from the completed export")
        else:
            store.write_once(path.relative_to(store.root), receipt)
        persisted = {**receipt, "receipt": store.artifact(path)}
        current.setdefault("exports", {})[request["export_fingerprint"]] = persisted
    return persisted


class MiniMaxH3LongVideoExport(MiniMaxH3StreamingVideoCombine):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = deepcopy(super().INPUT_TYPES())
        inputs["required"]["request_json"] = ("STRING",)
        return inputs

    CATEGORY = CATEGORY

    def combine(
        self, images, guide, frame_rate, codec, container, bit_depth, quality, log_level,
        pingpong, save_metadata, filename_prefix, save_output, pass_frames, crop_to_audio,
        audio_codec, audio_bitrate, save_first_frame, save_last_frame, request_json, audio=None,
        audio_override=None, prompt=None, extra_pnginfo=None,
    ):
        from .long_video_media import inspect_movie
        request = _request(request_json)
        if request["stage"] != "finish":
            raise ValueError("Long-video enhancement requires the finish stage")
        export_key = request.get("export_fingerprint")
        if not isinstance(export_key, str) or not re.fullmatch(r"[a-f0-9]{64}", export_key):
            raise ValueError("An immutable export_fingerprint is required")
        if (float(frame_rate) != FPS or pingpong or crop_to_audio or not save_output
                or guide.get("postprocess_mode") != "vosr2" or guide.get("vosr2_profile") != "manual_baseline"
                or guide.get("face_refine_mode", "off") != "off" or guide.get("motion_smoothing", "off") != "off"):
            raise ValueError("Long-video export requires unchanged 24 FPS frames and VOSR2 manual_baseline")
        store = _bound_store(request)
        _accepted_prefix(store, request)
        record = ProjectStore.active(store.read(), request["take_id"])
        window = request["window"]
        frames = window["new_end"] - window["new_start"]
        base = f"takes/{request['take_id']}/exports/{export_key}"
        receipt_path = store.path(f"{base}/receipt.json")
        # DaSiWa's current MP4 remux uses a millisecond movie clock. Preserve
        # its measured quantization here; final assembly uses exact PCM + PTS.
        tolerance = max(1, ceil(record["audio_sample_range"]["sample_rate"] / 1000))
        cached = record.get("exports", {}).get(export_key)
        if not cached and receipt_path.is_file():
            completed = json.loads(receipt_path.read_text(encoding="utf-8"))
            store.verify({"video": completed["video"], "audio": completed["audio"]})
            if (completed["take_id"] != request["take_id"] or completed["export_fingerprint"] != export_key
                    or completed["source_artifacts"] != record["artifacts"] or completed["frames"] != frames
                    or completed["audio"] != record["artifacts"]["delivery_audio"]
                    or inspect_movie(store.path(completed["video"]["name"]), frames,
                                     audio_sample_tolerance=tolerance) != completed["media_validation"]):
                raise ValueError("Completed export receipt does not bind this accepted native delivery")
            cached = _commit_export(store, request, completed, receipt_path)
        if cached:
            store.verify({"video": cached["video"], "audio": cached["audio"], "receipt": cached["receipt"]})
            if cached["source_artifacts"] != record["artifacts"]:
                raise ValueError("Export source differs from the current accepted native source")
            return {"ui": {"text": ["Reused immutable accepted-window export"]},
                    "result": (images[:0], str(store.path(cached["video"]["name"])))}
        if images.ndim != 4 or len(images) != window["sample_frames"]:
            raise ValueError("Export requires the complete decoded physical native window")
        delivered = images[window["local_new_start"]:window["local_new_end"]]
        sound = _delivery_audio(store, record)
        state = dict(guide)
        state.update(length=len(delivered), duration=len(delivered) / FPS, audio_loudness="original")
        prefix = store.path(f"{base}/enhanced").relative_to(store.output_root).as_posix()
        result = super().combine(
            images=delivered, guide=state, frame_rate=frame_rate, codec=codec, container=container,
            bit_depth=bit_depth, quality=quality, log_level=log_level, pingpong=pingpong,
            save_metadata=save_metadata, filename_prefix=prefix, save_output=save_output,
            pass_frames=pass_frames, crop_to_audio=crop_to_audio, audio_codec=audio_codec,
            audio_bitrate=audio_bitrate, save_first_frame=save_first_frame, save_last_frame=save_last_frame,
            audio=sound, audio_override=None, prompt=prompt, extra_pnginfo=extra_pnginfo,
        )
        output = result["result"][1]
        video_artifact = store.artifact(output)
        validation = inspect_movie(store.path(video_artifact["name"]), len(delivered),
                                   audio_sample_tolerance=tolerance)
        receipt = {"take_id": request["take_id"], "export_fingerprint": export_key,
                   "frames": len(delivered), "fps": FPS, "video": video_artifact,
                   "audio": deepcopy(record["artifacts"]["delivery_audio"]),
                   "source_artifacts": deepcopy(record["artifacts"]), "media_validation": validation,
                   "temporal_edits": "contiguous approved net-new frames only", "content_review": "accepted_native_source"}
        _commit_export(store, request, receipt, receipt_path)
        result.setdefault("ui", {})["long_video_export"] = [str(receipt_path)]
        return result
