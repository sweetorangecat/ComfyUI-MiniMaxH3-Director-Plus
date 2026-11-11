from fractions import Fraction
import wave

import pytest
import torch

from nodes.long_video_media import encode_preview, write_delivery_audio, assemble, inspect_movie


def test_streamed_assembly_has_exact_pts_and_audio_timeline(tmp_path):
    pytest.importorskip('av')
    parts = []
    cursor = 0
    for index, count in enumerate((5, 7, 11)):
        video = tmp_path / f'{index}.mp4'
        wav = tmp_path / f'{index}.wav'
        frames = torch.full((count, 18, 32, 3), 0.2 + index * 0.2)
        sound = {'waveform': torch.ones(1, 2, 50000) * 0.05, 'sample_rate': 32000}
        encode_preview(video, frames, sound)
        window = {'context_frames': 0, 'new_start': cursor, 'new_end': cursor + count}
        info = write_delivery_audio(wav, sound, window)
        parts.append({'video': str(video), 'audio': str(wav), 'frames': count})
        cursor += count
    output = tmp_path / 'complete.mp4'
    report = assemble(parts, output, cursor, (32, 18))
    assert report['frames'] == 23
    assert report['all_pts_exact'] and report['full_decode_pass']
    assert report['audio_duration'] == pytest.approx(23 / 24, abs=1 / 32000)
    assert report['source_audio_samples'] == round(23 * 32000 / 24)


def test_audio_uses_cumulative_global_sample_bounds(tmp_path):
    audio = {'waveform': torch.arange(100000).view(1, 1, -1) / 100000, 'sample_rate': 32000}
    path = tmp_path / 'delivery.wav'
    report = write_delivery_audio(path, audio, {'context_frames': 39, 'new_start': 294, 'new_end': 295})
    assert report['count'] == round(295 * 32000 / 24) - round(294 * 32000 / 24)
    with wave.open(str(path), 'rb') as f:
        assert f.getnframes() == report['count']
    with pytest.raises(ValueError, match='audio'):
        write_delivery_audio(tmp_path / 'short.wav', audio, {'context_frames': 39, 'new_start': 294, 'new_end': 500})
