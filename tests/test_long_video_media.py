from fractions import Fraction
import wave

import pytest
import torch

import nodes.long_video_media as media
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


@pytest.mark.parametrize('field,value', [('codec', 'AV1'), ('container', 'WebM'),
    ('bit_depth', '10-bit'), ('audio_codec', 'Opus'), ('quality', True), ('audio_bitrate', '999k')])
def test_long_encoding_unsupported_choice_rejected_before_generation(field, value):
    assert hasattr(media, 'resolve_encoding'), 'long output encoding validation is missing'
    with pytest.raises(ValueError):
        media.resolve_encoding({field: value})


def test_encoding_resolves_auto_and_honors_two_stage_quality_and_audio():
    assert hasattr(media, 'resolve_encoding'), 'long output encoding validation is missing'
    resolved = media.resolve_encoding({'codec': 'Auto', 'container': 'Auto', 'bit_depth': 'Auto',
        'audio_codec': 'Auto', 'quality': 11, 'audio_bitrate': '96k'})
    assert (resolved['codec'], resolved['container'], resolved['bit_depth'], resolved['audio_codec']) == (
        'H.264', 'MP4', '8-bit', 'AAC')
    assert resolved['crf'] == 11 and resolved['audio_bitrate'] == 96000
    assert media.resolve_encoding({'quality': 25})['crf'] == 16


def test_real_encoder_uses_bt709_matrix_and_limited_range_and_tags(tmp_path):
    av = pytest.importorskip('av')
    assert hasattr(media, 'resolve_encoding'), 'long output encoding validation is missing'
    import numpy as np
    # Large flat regions expose a 601 conversion falsely tagged as 709.
    colors = torch.tensor([[0.9,0.1,0.1], [0.1,0.8,0.2], [0.1,0.2,0.9]])
    frames = torch.stack([torch.cat([c.expand(32,32,3) for c in colors], dim=1)] * 5)
    sound = {'waveform': torch.zeros(1, 2, 16000), 'sample_rate': 32000}
    source, audio = tmp_path / 'native.mp4', tmp_path / 'native.wav'
    encoding = media.resolve_encoding({'quality': 9, 'audio_bitrate': '96k'})
    encode_preview(source, frames, sound, encoding=encoding)
    write_delivery_audio(audio, sound, {'context_frames': 0, 'new_start': 0, 'new_end': 5})
    output = tmp_path / 'complete.mp4'
    report = assemble([{'video': str(source), 'audio': str(audio), 'frames': 5}], output, 5, (96,32), encoding=encoding)
    assert report['encoding']['crf'] == 9
    assert report['encoding']['audio_bitrate'] == 96000
    with av.open(str(output)) as movie:
        ctx = movie.streams.video[0].codec_context
        assert (ctx.colorspace, ctx.color_primaries, ctx.color_trc, ctx.color_range) == (1,1,1,1)
        frame = next(movie.decode(video=0))
        y = np.asarray(frame.to_ndarray(format='yuv420p'))[:32]
        observed = [float(y[8:24, x+8:x+24].mean()) for x in (0,32,64)]
        expected = [16 + 219 * float(c @ torch.tensor([.2126,.7152,.0722])) for c in colors]
        assert np.max(np.abs(np.array(observed)-expected)) < 3
        rgb = frame.to_ndarray(format='rgb24')
        assert np.max(np.abs(np.array([rgb[16,x] for x in (16,48,80)])/255 - colors.numpy())) < .035


def test_seam_contact_is_bounded_cached_and_uses_actual_net_new_frames(tmp_path):
    pytest.importorskip('av')
    assert hasattr(media, 'seam_contact_sheet'), 'real bounded seam evidence is missing'
    from PIL import Image
    import hashlib
    segments = []
    for role, color, bounds in [('parent', .2, [5,9]), ('child', .8, [3,7])]:
        path = tmp_path / f'{role}.mp4'
        frames = torch.full((12,18,32,3), color)
        frames[0] = 0  # The child context must not be mistaken for its first new frames.
        encode_preview(path, frames, {'waveform': torch.zeros(1,1,18000), 'sample_rate': 32000})
        segments.append({'role': role, 'take_id': role, 'path': str(path), 'local_frames': bounds,
            'film_seconds': [10,10+4/24], 'artifact': {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}})
    output = media.seam_contact_sheet(segments, tmp_path / 'seams')
    assert Image.open(output).size[0] <= 1600
    # The requested child new frames are bright; its black context frame is excluded.
    assert min(Image.open(output).getpixel((10,225))) > 180
    assert max(Image.open(output).getpixel((10,35))) < 80
    first_mtime = output.stat().st_mtime_ns
    assert media.seam_contact_sheet(segments, tmp_path / 'seams') == output
    assert output.stat().st_mtime_ns == first_mtime


def test_actual_encoder_quality_changes_noise_reconstruction(tmp_path):
    av = pytest.importorskip('av')
    import numpy as np
    generator = torch.Generator().manual_seed(88)
    images = torch.rand((8,32,32,1),generator=generator).expand(-1,-1,-1,3)
    audio = {'waveform': torch.zeros(1,1,16000),'sample_rate':32000}
    errors, sizes = [], []
    for quality in (5,16):
        path = tmp_path / f'q{quality}.mp4'
        encode_preview(path,images,audio,encoding=media.resolve_encoding({'quality':quality,'audio_bitrate':'64k'}))
        with av.open(str(path)) as movie:
            decoded = np.stack([frame.to_ndarray(format='rgb24') for frame in movie.decode(video=0)])/255
        errors.append(float(np.mean((decoded-images.numpy())**2)))
        sizes.append(path.stat().st_size)
    assert errors[0] < errors[1] and sizes[0] > sizes[1]
