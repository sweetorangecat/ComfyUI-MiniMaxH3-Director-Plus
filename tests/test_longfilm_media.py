import importlib.util
import sys
import types
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('longfilm_media', Path(__file__).parents[1] / 'nodes/longfilm_media.py')
media = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(media)


def test_disk_helpers(tmp_path):
    p = media.safe_project_dir(tmp_path, 'film-01')
    assert p == tmp_path / 'h3_longfilm' / 'film-01'
    media.atomic_write_json(p / 'state.json', {'ok': True})
    assert len(media.sha256_file(p / 'state.json')) == 64
    for name in ('../escape', '.', '', 'a/b', 'a\\b', 'CON', 'a:', 'a.'):
        with pytest.raises(ValueError):
            media.safe_project_dir(tmp_path, name)


def test_latent_roundtrip_clean_only(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('safetensors')
    class NestedTensor:
        def __init__(self, tensors): self.tensors = list(tensors)
        def unbind(self): return self.tensors
    monkeypatch.setitem(sys.modules, 'comfy', types.ModuleType('comfy'))
    stub = types.ModuleType('comfy.nested_tensor')
    stub.NestedTensor = NestedTensor
    monkeypatch.setitem(sys.modules, 'comfy.nested_tensor', stub)
    latent = {'samples': NestedTensor([torch.ones(1, 2, 3), torch.zeros(1, 4)]), 'noise_mask': torch.ones(1)}
    path = tmp_path / 'clean.safetensors'
    media.save_av_latent(path, latent)
    result = media.load_av_latent(path)
    assert set(result) == {'samples'}
    assert all(torch.equal(a, b) for a, b in zip(result['samples'].unbind(), latent['samples'].unbind()))


def test_streaming_join_exact_frames_audio(tmp_path):
    torch = pytest.importorskip('torch')
    av = pytest.importorskip('av')
    paths = []
    for index, count in enumerate((3, 4)):
        path = tmp_path / f'{index}.mkv'
        frames = torch.full((count, 16, 16, 3), 0.2 + index * 0.6)
        audio = {'waveform': torch.full((1, 1, count * 1000), 0.1 + index * 0.2), 'sample_rate': 24000}
        media.encode_segment(path, frames, audio)
        paths.append(path)
    result = tmp_path / 'final.mkv'
    info = media.join_segments(paths, result, expected_frames=7)
    assert info['frames'] == 7
    assert info['audio_samples'] == 7000
    with av.open(str(result)) as container:
        means = [f.to_ndarray(format='rgb24').mean() for f in container.decode(video=0)]
    assert max(means[:3]) < min(means[3:])
    with av.open(str(result)) as container:
        import numpy as np
        waveform = np.concatenate([f.to_ndarray().reshape(-1) for f in container.decode(audio=0)])
    assert abs(float(waveform[:3000].mean()) / 32768 - .1) < .001
    assert abs(float(waveform[3000:].mean()) / 32768 - .3) < .001
    with pytest.raises(ValueError, match='frames'):
        media.join_segments(paths, tmp_path / 'bad.mkv', expected_frames=8)
    assert not (tmp_path / 'bad.mkv').exists()
