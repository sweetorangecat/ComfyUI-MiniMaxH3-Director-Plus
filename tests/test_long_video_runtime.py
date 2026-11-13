"""CPU contract tests at the runtime node, disk journal and native-node seams."""
import importlib
import json
from pathlib import Path
import sys
import types
import wave

import pytest
import torch

from nodes.long_video_plan import audio_ticks, plan_windows, video_steps
from nodes.long_video_store import ProjectStore, fingerprint


class Nested:
    is_nested = True

    def __init__(self, values):
        self.values = tuple(values)

    def unbind(self):
        return self.values


def runtime():
    return importlib.import_module("nodes.long_video_runtime")


def latent(frames=328, offset=0):
    video = torch.arange(4 * video_steps(frames) * 2 * 3, dtype=torch.float32).reshape(1, 4, -1, 2, 3)
    audio = torch.arange(8 * 2 * audio_ticks(frames), dtype=torch.float32).reshape(1, 8, 2, -1)
    return {"samples": Nested((video + offset, audio + offset))}


def guide(**changes):
    result = dict(sampling_method="selflift", mode="REF2VA", performance_preset="quality_two_stage",
                  attention_backend="sol_attn", sol_attn_applied=True, length=345,
                  postprocess_mode="vosr2", vosr2_profile="manual_baseline",
                  face_refine_mode="off", motion_smoothing="off")
    result.update(changes)
    return result


def request(store, window=None, **changes):
    spec = dict(stage="native", window=window or plan_windows(720)[0], prompt="official Ref2VA prompt",
                seed=7, fingerprint=fingerprint({"seed": 7}), parent=None, repair=None)
    spec.update(changes)
    return store.create_take(spec)


def silent_target(store):
    from nodes.long_video_audio import file_identity, physical_samples
    window = plan_windows(720)[0]
    path = store.path('complete-silent-target.wav')
    with wave.open(str(path), 'wb') as sound:
        sound.setnchannels(2)
        sound.setsampwidth(2)
        sound.setframerate(32000)
        sound.writeframes(b'\0' * physical_samples(window) * 4)
    return {'kind': 'complete_wav', **file_identity(path), 'sample_rate': 32000, 'channels': 2,
            'samples': physical_samples(window), 'window': window, 'sources': [], 'utterances': [],
            'provenance': {'method': 'Supplied complete intentional silence.', 'seed': None}}


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", types.SimpleNamespace(NestedTensor=Nested))
    monkeypatch.setattr(importlib.import_module("nodes"), "NODE_CLASS_MAPPINGS", {"VAEDecodeAudio": AudioDecode}, raising=False)
    return ProjectStore(tmp_path, "scene")


def model():
    return types.SimpleNamespace(model=types.SimpleNamespace(_denoise_mask_conds=lambda *args: {}))


class VideoVAE:
    output_device = torch.device("cpu")
    device = torch.device("cpu")

    def vae_output_dtype(self):
        return torch.float32

    def decode(self, samples):
        frames = 5 + 17 * ((samples.shape[2] - 2) // 5)
        return torch.linspace(0, 1, frames).view(-1, 1, 1, 1).expand(-1, 2, 4, 3).clone()


class AudioDecode:
    @classmethod
    def execute(cls, vae, samples):
        audio = samples["samples"].unbind()[-1]
        count = int(audio.shape[-1]) * 800
        return ({"waveform": torch.ones(1, 2, count) * .05, "sample_rate": 32000},)


def sampled_guide(**changes):
    return guide(sampling_actual_method="selflift", selflift_highres_tiling=True,
                 selflift_highres_tiling_mode="global_stream", selflift_node_id="SelfLiftAvatarH3Sampler",
                 sampling_route="SelfLiftAvatarH3Sampler", **changes)


def sample(prepared):
    for value, mask in zip(prepared["samples"].unbind(), prepared["noise_mask"].unbind()):
        value.add_(mask.expand_as(value) * .25)
    return prepared


def native_take(store, window=None, **changes):
    req = request(store, window, **changes)
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(latent(req["window"]["sample_frames"]), guide(), json.dumps(req), model())[0]
    result = sample(prepared)
    runtime().MiniMaxH3LongVideoCheckpoint().save(result, prepared, sampled_guide(), VideoVAE(), object(), json.dumps(req))
    return req, result


def export_options():
    from nodes.stream_output import MiniMaxH3StreamingVideoCombine
    return {key: definition[1]["default"] for key, definition in MiniMaxH3StreamingVideoCombine.INPUT_TYPES()["required"].items()
            if len(definition) > 1 and "default" in definition[1]}


def test_window_uses_exact_planned_native_length_without_rounding_seconds():
    req = dict(project_name="scene", take_id="take", window=plan_windows(720)[2], stage="native")
    old = guide()
    result = runtime().MiniMaxH3LongVideoWindow().apply(old, json.dumps(req))[0]
    assert result["length"] == 277
    assert result["duration"] == 277 / 24
    assert result["long_video_window"] == req["window"]
    assert old["length"] == 345


def test_context_persists_independent_cpu_baseline_and_two_masks(environment):
    from safetensors.torch import load_file
    req = request(environment)
    target = latent()
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(target, guide(), json.dumps(req), model())[0]
    video, audio = prepared["samples"].unbind()
    vm, am = prepared["noise_mask"].unbind()
    video.add_(9)
    audio.add_(11)
    vm.zero_()
    am.zero_()
    saved_v = load_file(str(environment.path(f"takes/{req['take_id']}/prepared_video.safetensors")))
    saved_a = load_file(str(environment.path(f"takes/{req['take_id']}/prepared_audio.safetensors")))
    assert torch.equal(saved_v["samples"], target["samples"].unbind()[0])
    assert torch.equal(saved_a["samples"], target["samples"].unbind()[1])
    assert bool((saved_v["noise_mask"] == 1).all()) and bool((saved_a["noise_mask"] == 1).all())
    assert saved_v["samples"].device.type == saved_a["samples"].device.type == "cpu"
    with pytest.raises(ValueError, match="immutable"):
        runtime().MiniMaxH3LongVideoContext().prepare(target, guide(), json.dumps(req), model())


def test_context_installs_optional_encoded_target_and_persists_full_audio_lock(environment):
    manifest = silent_target(environment)
    req = request(environment, target_audio=manifest, target_dialogues=[], target_audio_original_sources={})
    target = latent()
    encoded = {'samples': torch.full_like(target['samples'].unbind()[1], .375)}
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(
        target, guide(), json.dumps(req), model(), target_audio=encoded)[0]
    assert torch.all(prepared['samples'].unbind()[1] == .375)
    assert torch.all(prepared['noise_mask'].unbind()[1] == 0)
    saved, _ = runtime()._load_prepared(environment, req['take_id'])
    assert torch.equal(saved['samples'].unbind()[1], encoded['samples'])
    assert saved['_h3_long_video_context']['target_audio_sha256'] == manifest['sha256']
    assert saved['_h3_long_video_context']['method'] == 'target_audio_conditioned_native_video'


def test_context_requires_target_latent_only_when_immutable_request_declares_it(environment):
    req = request(environment, target_audio=silent_target(environment))
    with pytest.raises(ValueError, match='target|Target'):
        runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(req), model())


def test_target_loader_checks_immutable_waveform_before_official_vae_encoding(environment):
    manifest = silent_target(environment)
    req = request(environment, target_audio=manifest, target_dialogues=[], target_audio_original_sources={})
    loaded = runtime().MiniMaxH3LongVideoTargetAudio().load(json.dumps(req))[0]
    assert loaded['waveform'].shape == (1, 2, 437600) and loaded['sample_rate'] == 32000
    with open(manifest['path'], 'r+b') as stream:
        stream.seek(-2, 2)
        stream.write(b'\1\0')
    with pytest.raises(ValueError, match='hash|SHA'):
        runtime().MiniMaxH3LongVideoTargetAudio().load(json.dumps(req))


def test_checkpoint_saves_clean_before_native_decode_and_keeps_review_pending(environment, monkeypatch):
    pytest.importorskip("av")
    monkeypatch.setattr(importlib.import_module("nodes"), "NODE_CLASS_MAPPINGS", {"VAEDecodeAudio": AudioDecode}, raising=False)
    req = request(environment)
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(req), model())[0]
    result = sample(prepared)
    for mask in prepared["noise_mask"].unbind():
        mask.zero_()  # A sampler may mutate the object also wired to `prepared`.
    runtime().MiniMaxH3LongVideoCheckpoint().save(result, prepared, sampled_guide(), VideoVAE(), object(), json.dumps(req))
    record = environment.read()["windows"][0]
    assert record["status"] == "awaiting_review" and environment.read()["completed_frames"] == 0
    assert record["tensor_validation"]["open_video_changed"]
    assert record["tensor_validation"]["content_review"] == "pending"
    environment.verify(record["artifacts"])
    loaded = runtime().load_clean(environment, req["take_id"])
    assert torch.equal(loaded["samples"].unbind()[0], result["samples"].unbind()[0])
    assert "noise_mask" not in loaded
    with wave.open(str(environment.path(record["artifacts"]["delivery_audio"]["name"])), "rb") as sound:
        assert sound.getnframes() == 392000
    import av
    with av.open(str(environment.path(record["artifacts"]["native"]["name"]))) as media:
        assert sum(1 for _ in media.decode(video=0)) == 328


def test_continuation_consumes_only_verified_accepted_history_and_detects_aliased_damage(environment):
    from nodes.long_video_av import slice_fingerprint
    parent, original = native_take(environment)
    environment.review(parent["take_id"], True, "hand remains on plate", "full native motion and audio checked")
    window = plan_windows(720)[1]
    parent_hash = slice_fingerprint(original, 294, 39)
    req = request(environment, window, parent={"take_id": parent["take_id"], "history_sha256": parent_hash})
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(req), model())[0]
    pv, pa = prepared["samples"].unbind()
    sv, sa = original["samples"].unbind()
    assert torch.equal(pv[:, :, :12], sv[:, :, 75:87])
    assert torch.equal(pa[..., :65], sa[..., 425:490])
    vm, am = prepared["noise_mask"].unbind()
    assert bool((vm[:, :, :12] == 0).all()) and bool((am[..., :65] == 0).all())
    result = sample(prepared)
    pv[:, :, 0].add_(100)
    vm.fill_(1)
    with pytest.raises(ValueError, match="protected clean video"):
        runtime().MiniMaxH3LongVideoCheckpoint().save(result, prepared, sampled_guide(), VideoVAE(), object(), json.dumps(req))
    assert not environment.path(f"takes/{req['take_id']}/clean_video.safetensors").exists()


def test_failed_decode_recovers_full_av_without_sampling_and_finish_requires_review(environment):
    class BrokenVideoVAE(VideoVAE):
        def decode(self, samples):
            raise RuntimeError("decode interrupted")

    req = request(environment)
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(req), model())[0]
    result = sample(prepared)
    with pytest.raises(RuntimeError, match="decode interrupted"):
        runtime().MiniMaxH3LongVideoCheckpoint().save(result, prepared, sampled_guide(), BrokenVideoVAE(), object(), json.dumps(req))
    assert environment.read()["windows"][0]["status"] == "sampled"
    preview = {**req, "stage": "preview"}
    loaded, denoised = runtime().MiniMaxH3LongVideoLoadAV().load(json.dumps(preview))
    assert [list(x.shape) for x in loaded["samples"].unbind()] == [[1, 4, 97, 2, 3], [1, 8, 2, 547]]
    assert denoised["samples"].is_nested
    finish = {**req, "stage": "finish", "export_fingerprint": "a" * 64}
    with pytest.raises(ValueError, match="accepted"):
        runtime().MiniMaxH3LongVideoLoadAV().load(json.dumps(finish))
    runtime().MiniMaxH3LongVideoCheckpoint().save(loaded, loaded, guide(), VideoVAE(), object(), json.dumps(preview))
    environment.review(req["take_id"], True, "actual final pose", "native delivery, sound, free boundary and tail checked")
    full, also_full = runtime().MiniMaxH3LongVideoLoadAV().load(json.dumps(finish))
    assert full["samples"].is_nested and also_full["samples"].is_nested
    assert full["samples"].unbind()[0].shape[2] == 97  # Never slice compressed video by RGB frame index.


def test_encoder_interruption_leaves_clean_av_recoverable_without_partial_preview(environment, monkeypatch):
    import nodes.long_video_media as media
    req = request(environment)
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(req), model())[0]
    result = sample(prepared)
    with monkeypatch.context() as failure:
        def interrupted(*args):
            raise RuntimeError("encoder interrupted")
        failure.setattr(media, "encode_preview", interrupted)
        with pytest.raises(RuntimeError, match="encoder interrupted"):
            runtime().MiniMaxH3LongVideoCheckpoint().save(result, prepared, sampled_guide(), VideoVAE(), object(), json.dumps(req))
    assert environment.read()["windows"][0]["status"] == "sampled"
    assert not environment.path(f"takes/{req['take_id']}/delivery.wav").exists()
    preview = {**req, "stage": "preview"}
    runtime().MiniMaxH3LongVideoCheckpoint().save(result, result, guide(), VideoVAE(), object(), json.dumps(preview))
    assert environment.read()["windows"][0]["status"] == "awaiting_review"


def test_clean_receipt_recovers_sampling_if_journal_update_was_interrupted(environment, monkeypatch):
    req = request(environment)
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(req), model())[0]
    result = sample(prepared)
    with monkeypatch.context() as failure:
        def interrupted(*args):
            raise RuntimeError("journal interrupted")
        failure.setattr(ProjectStore, "record_clean", interrupted)
        with pytest.raises(RuntimeError, match="journal interrupted"):
            runtime().MiniMaxH3LongVideoCheckpoint().save(result, prepared, sampled_guide(), VideoVAE(), object(), json.dumps(req))
    assert environment.read()["windows"][0]["status"] == "queued"
    loaded = runtime().load_clean(environment, req["take_id"])
    assert torch.equal(loaded["samples"].unbind()[1], result["samples"].unbind()[1])
    assert environment.read()["windows"][0]["status"] == "sampled"
    environment.verify(environment.read()["windows"][0]["artifacts"])


def test_bounded_repair_uses_superseded_clean_take_and_preserves_all_audio(environment):
    original_request, original = native_take(environment)
    environment.review(original_request["take_id"], False, "", "mouth movement needs native repair")
    environment.request_repair(0, [[20, 24]], lock_audio=True)
    req = request(environment, repair={"source_take": original_request["take_id"],
                                      "video_ranges": [[20, 24]], "lock_audio": True})
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(req), model())[0]
    with pytest.raises(ValueError, match="Repair did not change"):
        runtime().MiniMaxH3LongVideoCheckpoint().save(prepared, prepared, sampled_guide(), VideoVAE(), object(), json.dumps(req))
    repaired = sample(prepared)
    runtime().MiniMaxH3LongVideoCheckpoint().save(repaired, prepared, sampled_guide(), VideoVAE(), object(), json.dumps(req))
    saved = runtime().load_clean(environment, req["take_id"])
    video, audio = saved["samples"].unbind()
    ov, oa = original["samples"].unbind()
    assert torch.equal(audio, oa)
    assert torch.equal(video[:, :, :20], ov[:, :, :20]) and torch.equal(video[:, :, 24:], ov[:, :, 24:])
    assert not torch.equal(video[:, :, 20:24], ov[:, :, 20:24])
    assert environment.read()["windows"][0]["status"] == "awaiting_review"


@pytest.mark.parametrize("parent_change", ["video_history", "audio_history", "outside_history", "unchanged"])
def test_continuation_repair_binds_its_locked_prefix_to_actual_current_parent(environment, parent_change):
    from nodes.long_video_av import slice_fingerprint
    old_parent, old_av = native_take(environment)
    environment.review(old_parent["take_id"], True, "old observed state", "full native review")
    window = plan_windows(720)[1]
    old_child, child_av = native_take(environment, window,
        parent={"take_id": old_parent["take_id"], "history_sha256": slice_fingerprint(old_av, 294, 39)})
    environment.review(old_child["take_id"], True, "old child state", "full continuation review")
    old_records = environment.read()["windows"]
    environment.invalidate_from(0, "Replace the parent before repairing its child")

    new_parent = request(environment, seed=8, fingerprint=fingerprint({"seed": 8}))
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(new_parent), model())[0]
    parent_av = sample(prepared)
    video, audio = parent_av["samples"].unbind()
    if parent_change == "video_history":
        video[:, :, 75, 0, 0].add_(1)
    elif parent_change == "audio_history":
        audio[..., 425].add_(1)
    elif parent_change == "outside_history":
        video[:, :, 0].add_(1)
        audio[..., 0].add_(1)
    runtime().MiniMaxH3LongVideoCheckpoint().save(parent_av, prepared, sampled_guide(), VideoVAE(), object(), json.dumps(new_parent))
    environment.review(new_parent["take_id"], True, "new observed state", "new parent and contact reviewed")
    current_parent = environment.read()["windows"][0]
    immutable_sources = {f"{record['take_id']}:{key}": artifact
                         for record in [*old_records, current_parent]
                         for key, artifact in record["artifacts"].items()}
    repair = request(environment, window,
        parent={"take_id": new_parent["take_id"], "history_sha256": slice_fingerprint(parent_av, 294, 39)},
        repair={"source_take": old_child["take_id"], "video_ranges": [[20, 24]], "lock_audio": True})
    repair_request = environment.artifact(environment.path(f"takes/{repair['take_id']}/request.json"))
    if parent_change in {"video_history", "audio_history"}:
        with pytest.raises(ValueError, match="Repair source protected AV history.*current parent"):
            runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(repair), model())
        assert not environment.path(f"takes/{repair['take_id']}/prepared_context.json").exists()
    else:
        fixed = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(repair), model())[0]
        fixed_video, fixed_audio = fixed["samples"].unbind()
        assert torch.equal(fixed_video[:, :, :12], video[:, :, 75:87])
        assert torch.equal(fixed_audio[..., :65], audio[..., 425:490])
        runtime().MiniMaxH3LongVideoCheckpoint().save(sample(fixed), fixed, sampled_guide(), VideoVAE(), object(), json.dumps(repair))
        assert torch.equal(runtime().load_clean(environment, repair["take_id"])["samples"].unbind()[1], child_av["samples"].unbind()[1])
    environment.verify({**immutable_sources, "repair_request": repair_request})


def test_partial_preview_publication_resumes_from_staged_media_without_decoding(environment, monkeypatch):
    import os
    req = request(environment)
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(req), model())[0]
    result = sample(prepared)
    original_link = os.link
    with monkeypatch.context() as failure:
        def interrupted(source, destination):
            if Path(destination).name == "delivery.wav":
                raise RuntimeError("publication interrupted")
            return original_link(source, destination)
        failure.setattr(os, "link", interrupted)
        with pytest.raises(RuntimeError, match="publication interrupted"):
            runtime().MiniMaxH3LongVideoCheckpoint().save(result, prepared, sampled_guide(), VideoVAE(), object(), json.dumps(req))
    assert environment.path(f"takes/{req['take_id']}/native.mp4").is_file()
    assert not environment.path(f"takes/{req['take_id']}/delivery.wav").exists()

    class NoDecode(VideoVAE):
        def decode(self, samples):
            pytest.fail("already encoded staged media was decoded again")

    preview = {**req, "stage": "preview"}
    runtime().MiniMaxH3LongVideoCheckpoint().save(result, result, guide(), NoDecode(), object(), json.dumps(preview))
    assert environment.read()["windows"][0]["status"] == "awaiting_review"
    environment.verify(environment.read()["windows"][0]["artifacts"])


def test_runtime_rejects_changed_request_and_unbound_native_acceptance(environment):
    req, result = native_take(environment)
    changed = {**req, "stage": "preview", "seed": req["seed"] + 1}
    with pytest.raises(ValueError, match="immutable take request"):
        runtime().MiniMaxH3LongVideoLoadAV().load(json.dumps(changed))
    environment.review(req["take_id"], True, "actual pose", "full native review")
    record = environment.read()["windows"][0]
    environment.path(record["review"]["name"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        runtime().MiniMaxH3LongVideoLoadAV().load(json.dumps({**req, "stage": "finish"}))


@pytest.mark.parametrize("field,value", [("sampling_actual_method", "standard"),
    ("selflift_highres_tiling_mode", "spatial"), ("sol_attn_applied", False),
    ("sampling_route", "SelfLiftH3Sampler")])
def test_checkpoint_refuses_unexecuted_native_recipe_before_writing_clean(environment, field, value):
    req = request(environment)
    prepared = runtime().MiniMaxH3LongVideoContext().prepare(latent(), guide(), json.dumps(req), model())[0]
    result = sample(prepared)
    invalid = sampled_guide()
    invalid[field] = value
    with pytest.raises(RuntimeError, match="actual native sampler"):
        runtime().MiniMaxH3LongVideoCheckpoint().save(result, prepared, invalid, VideoVAE(), object(), json.dumps(req))
    assert not environment.path(f"takes/{req['take_id']}/clean_video.safetensors").exists()


def test_export_enhances_only_approved_decoded_net_new_frames_and_records_immutable_receipt(environment, monkeypatch):
    from nodes.long_video_av import slice_fingerprint
    from nodes.long_video_media import encode_preview
    from nodes.stream_output import MiniMaxH3StreamingVideoCombine
    parent, original = native_take(environment)
    environment.review(parent["take_id"], True, "actual pose", "full native review")
    req, second = native_take(environment, plan_windows(720)[1],
                              parent={"take_id": parent["take_id"], "history_sha256": slice_fingerprint(original, 294, 39)})
    environment.review(req["take_id"], True, "actual continuing pose", "full continuation review")
    captured = {}

    def enhance_and_encode(self, **kwargs):
        captured.update(kwargs)
        path = Path(environment.output_root) / (kwargs["filename_prefix"] + ".mp4")
        encode_preview(path, kwargs["images"], kwargs["audio"])
        return {"ui": {}, "result": (kwargs["images"][:0], str(path))}

    monkeypatch.setattr(MiniMaxH3StreamingVideoCombine, "combine", enhance_and_encode)
    finish = {**req, "stage": "finish", "export_fingerprint": "e" * 64}
    images = VideoVAE().decode(second["samples"].unbind()[0])
    options = export_options()
    runtime().MiniMaxH3LongVideoExport().combine(images=images, guide=guide(), request_json=json.dumps(finish), **options)
    assert torch.equal(captured["images"], images[39:294])
    assert captured["audio"]["waveform"].shape == (1, 2, 340000)
    assert captured["guide"]["audio_loudness"] == "original"
    receipt = environment.read()["windows"][1]["exports"]["e" * 64]
    assert receipt["frames"] == 255 and receipt["media_validation"]["all_pts_exact"]
    assert receipt["media_validation"]["audio_duration_tolerance_samples"] == 32
    environment.verify({"video": receipt["video"], "audio": receipt["audio"], "receipt": receipt["receipt"]})
    # Same immutable export is reused and never sent through enhancement twice.
    monkeypatch.setattr(MiniMaxH3StreamingVideoCombine, "combine", lambda *args, **kwargs: pytest.fail("export was regenerated"))
    runtime().MiniMaxH3LongVideoExport().combine(images=images, guide=guide(), request_json=json.dumps(finish), **options)


def test_export_receipt_recovers_without_repeating_enhancement_after_journal_failure(environment, monkeypatch):
    from nodes.long_video_media import encode_preview
    from nodes.stream_output import MiniMaxH3StreamingVideoCombine
    import nodes.long_video_store as journal
    req, clean = native_take(environment)
    environment.review(req["take_id"], True, "actual final pose", "full native review")
    finish = {**req, "stage": "finish", "export_fingerprint": "f" * 64}
    images = VideoVAE().decode(clean["samples"].unbind()[0])

    def enhance(self, **kwargs):
        path = Path(environment.output_root) / (kwargs["filename_prefix"] + ".mp4")
        encode_preview(path, kwargs["images"], kwargs["audio"])
        return {"ui": {}, "result": (kwargs["images"][:0], str(path))}

    monkeypatch.setattr(MiniMaxH3StreamingVideoCombine, "combine", enhance)
    write_state = journal.atomic_write_json
    with monkeypatch.context() as failure:
        def interrupted(path, value):
            if value["windows"][0].get("exports"):
                raise RuntimeError("export journal interrupted")
            return write_state(path, value)
        failure.setattr(journal, "atomic_write_json", interrupted)
        with pytest.raises(RuntimeError, match="export journal interrupted"):
            runtime().MiniMaxH3LongVideoExport().combine(images=images, guide=guide(), request_json=json.dumps(finish), **export_options())
    assert not environment.read()["windows"][0]["exports"]
    monkeypatch.setattr(MiniMaxH3StreamingVideoCombine, "combine", lambda *args, **kwargs: pytest.fail("enhancement was repeated"))
    runtime().MiniMaxH3LongVideoExport().combine(images=images, guide=guide(), request_json=json.dumps(finish), **export_options())
    assert environment.read()["windows"][0]["exports"]["f" * 64]["frames"] == 294
