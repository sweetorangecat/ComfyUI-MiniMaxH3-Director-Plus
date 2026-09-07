import json
from pathlib import Path
import pytest
from nodes import longfilm


def story():
    return json.loads(Path('templates/u11_longfilm_story.json').read_text(encoding='utf-8'))


def test_runtime_node_exposes_loaded_models_and_output():
    cls = longfilm.MiniMaxH3LongFilm61
    fields = cls.INPUT_TYPES()['required']
    assert {'model', 'clip', 'video_vae', 'audio_vae', 'target', 'seed'} <= fields.keys()
    assert cls.OUTPUT_NODE is True
    assert cls.FUNCTION == 'generate'


def test_missing_continuation_capability_is_actionable():
    with pytest.raises(RuntimeError, match='Infinite.*v1.4'):
        longfilm.require_capabilities({})


def test_fingerprint_is_stable_and_sensitive_to_content():
    assert longfilm.content_fingerprint({'b': 2, 'a': 1}) == longfilm.content_fingerprint({'a': 1, 'b': 2})
    assert longfilm.content_fingerprint({'seed': 1}) != longfilm.content_fingerprint({'seed': 2})


def test_checkpoint_verifies_every_artifact_and_parent(tmp_path):
    a = tmp_path / 'a.mkv'; a.write_bytes(b'video')
    b = tmp_path / 'b.safetensors'; b.write_bytes(b'latent')
    record = {'fingerprint': 'expected', 'artifacts': {
        'video': {'name': a.name, 'sha256': longfilm.media.sha256_file(a)},
        'latent': {'name': b.name, 'sha256': longfilm.media.sha256_file(b)},
    }}
    assert longfilm.checkpoint_valid(tmp_path, record, 'expected')
    assert not longfilm.checkpoint_valid(tmp_path, record, 'changed-prefix')
    b.write_bytes(b'corrupt')
    assert not longfilm.checkpoint_valid(tmp_path, record, 'expected')


def test_checkpoint_rejects_external_artifact(tmp_path):
    outside = tmp_path / 'outside'; outside.write_bytes(b'video')
    project = tmp_path / 'project'; project.mkdir()
    record = {'fingerprint': 'f', 'artifacts': {'video': {'name': '../outside', 'sha256': longfilm.media.sha256_file(outside)}, 'latent': {'name': '../outside', 'sha256': longfilm.media.sha256_file(outside)}}}
    assert not longfilm.checkpoint_valid(project, record, 'f')

class FixtureBackend:
    total_vram_gb = 32
    generation_config = {'model': 'fixture-v1', 'width': 16, 'height': 16}
    def __init__(self): self.rendered = []; self.exports = []
    def render_chunk(self, *, chunk, prompt, seed, previous_latent, handover):
        self.rendered.append((chunk['index'], previous_latent, handover))
        return {'latent': 'clean-' + str(chunk['index']), 'images': chunk['visible_frames'], 'audio': None, 'handover': {'available': True, 'end': chunk['visible_end']}}
    def export_segment(self, source, destination, target):
        self.exports.append(target); destination.write_bytes(source.read_bytes() + target.encode())


@pytest.fixture
def disk_backend(monkeypatch):
    monkeypatch.setattr(longfilm.media, 'save_av_latent', lambda path, latent: Path(path).write_text(latent))
    monkeypatch.setattr(longfilm.media, 'load_av_latent', lambda path: Path(path).read_text())
    monkeypatch.setattr(longfilm.media, 'encode_segment', lambda path, images, audio, fps: Path(path).write_text(str(images)))
    def join(paths, output_path, expected_frames, fps):
        assert expected_frames == 1464
        Path(output_path).write_bytes(b''.join(Path(p).read_bytes() for p in paths))
        return {'frames': 1464, 'fps': 24}
    monkeypatch.setattr(longfilm.media, 'join_segments', join)
    return FixtureBackend()


def test_run_project_resumes_and_target_change_only_reexports(tmp_path, disk_backend):
    first = longfilm.run_project(tmp_path, story(), disk_backend, '1080p', 4)
    count = len(disk_backend.rendered)
    assert first['status'] == 'complete' and count > 1
    assert disk_backend.rendered[1][1] == 'clean-0'
    longfilm.run_project(tmp_path, story(), disk_backend, '1080p', 4)
    assert len(disk_backend.rendered) == count
    assert len(disk_backend.exports) == count
    longfilm.run_project(tmp_path, story(), disk_backend, '2K', 4)
    assert len(disk_backend.rendered) == count
    assert len(disk_backend.exports) == count * 2


def test_corrupt_checkpoint_regenerates_following_chain(tmp_path, disk_backend):
    longfilm.run_project(tmp_path, story(), disk_backend, '1080p', 4)
    count = len(disk_backend.rendered)
    (tmp_path / 'chunk-0001.safetensors').write_bytes(b'corrupt')
    longfilm.run_project(tmp_path, story(), disk_backend, '1080p', 4)
    assert [index for index, _, _ in disk_backend.rendered[count:]] == list(range(1, count))


def test_cancel_only_own_project_at_segment_boundary(tmp_path, disk_backend):
    (tmp_path / 'cancel.request').write_text('cancel')
    result = longfilm.run_project(tmp_path, story(), disk_backend, '1080p', 4)
    assert result['status'] == 'cancelled'
    assert not disk_backend.rendered


def test_failure_keeps_prior_checkpoint(tmp_path, disk_backend):
    original = disk_backend.render_chunk
    def failing(**kwargs):
        if kwargs['chunk']['index'] == 1: raise RuntimeError('GPU fixture failure')
        return original(**kwargs)
    disk_backend.render_chunk = failing
    with pytest.raises(RuntimeError, match='GPU fixture failure'):
        longfilm.run_project(tmp_path, story(), disk_backend, '1080p', 4)
    state = json.loads((tmp_path / 'state.json').read_text())
    assert state['status'] == 'failed' and len(state['chunks']) == 1


def test_project_lock_rejects_concurrent_run(tmp_path, disk_backend):
    (tmp_path / 'run.lock').write_text('active')
    with pytest.raises(RuntimeError, match='already running'):
        longfilm.run_project(tmp_path, story(), disk_backend, '1080p', 4)
    assert not disk_backend.rendered


def test_failed_run_releases_project_lock(tmp_path, disk_backend):
    def fail(**kwargs): raise RuntimeError('fail')
    disk_backend.render_chunk = fail
    with pytest.raises(RuntimeError):
        longfilm.run_project(tmp_path, story(), disk_backend, '1080p', 4)
    assert not (tmp_path / 'run.lock').exists()
