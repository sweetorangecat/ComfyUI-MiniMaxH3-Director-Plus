"""Real media checks for preserving short-clip pictures in a long export."""
import hashlib
from pathlib import Path

import pytest
import torch

import nodes.long_video_media as media


def _hashes(path, *, packets=False):
    av = pytest.importorskip('av')
    with av.open(str(path)) as movie:
        if packets:
            return [hashlib.sha256(bytes(p)).hexdigest()
                    for p in movie.demux(video=0) if p.size]
        return [hashlib.sha256(f.to_ndarray(format='rgb24').tobytes()).hexdigest()
                for f in movie.decode(video=0)]


def _parts(tmp_path, monkeypatch, counts=(17, 19), widths=(96, 96), bframes=(2, 2)):
    pytest.importorskip('av')
    parts = []
    cursor = 0
    original_stream = media._video_stream
    for index, (count, width, reorder) in enumerate(zip(counts, widths, bframes)):
        texture = torch.rand((64, width, 3), generator=torch.Generator().manual_seed(810 + index))
        images = torch.stack([torch.roll(texture, shifts=i, dims=1) for i in range(count)])
        time = torch.arange(64000) / 32000
        waveform = torch.stack([torch.sin(2 * torch.pi * 220 * time),
                                torch.cos(2 * torch.pi * 330 * time)])[None] * .05
        audio = {'waveform': waveform, 'sample_rate': 32000}
        video, wav = tmp_path / f'clip{index}.mp4', tmp_path / f'clip{index}.wav'

        def source_stream(*args, **kwargs):
            stream = original_stream(*args, **kwargs)
            stream.options = {**stream.options,
                'x264-params': f'bframes={reorder}:b-adapt=0:open-gop=0'}
            return stream

        with monkeypatch.context() as patch:
            patch.setattr(media, '_video_stream', source_stream)
            media.encode_preview(video, images, audio)
        media.write_delivery_audio(wav, audio, {
            'context_frames': 0, 'new_start': cursor, 'new_end': cursor+count})
        parts.append({'video': str(video), 'audio': str(wav), 'frames': count})
        cursor += count
    return parts


def test_matching_clips_preserve_every_picture_and_compressed_video_packet(tmp_path, monkeypatch):
    parts = _parts(tmp_path, monkeypatch)
    inputs = {p['video']: hashlib.sha256(Path(p['video']).read_bytes()).hexdigest() for p in parts}
    frames = [value for p in parts for value in _hashes(p['video'])]
    packets = [value for p in parts for value in _hashes(p['video'], packets=True)]
    output = tmp_path / 'long.mp4'
    report = media.assemble(parts, output, 36, (96, 64))
    assert _hashes(output) == frames
    assert _hashes(output, packets=True) == packets
    assert report['video_assembly_mode'] == 'packet_copy'
    assert report['actual_video_encoder'] == 'copy'
    assert report['all_pts_exact'] and report['full_decode_pass']
    assert report['source_audio_samples'] == round(36*32000/24)
    assert report['audio_duration'] == pytest.approx(36/24, abs=1/32000)
    for path, digest in inputs.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


def test_required_spatial_crop_keeps_existing_encoding_path(tmp_path, monkeypatch):
    parts = _parts(tmp_path, monkeypatch, widths=(100, 100))
    output = tmp_path / 'cropped.mp4'
    report = media.assemble(parts, output, 36, (96, 64))
    assert report['video_assembly_mode'] == 'decode_reencode'
    assert report['actual_video_encoder'] == 'libx264'
    assert (report['width'], report['height'], report['frames']) == (96, 64, 36)
    assert report['all_pts_exact'] and report['full_decode_pass']


def test_different_h264_reorder_parameters_use_existing_encoding_path(tmp_path, monkeypatch):
    parts = _parts(tmp_path, monkeypatch, bframes=(2, 0))
    output = tmp_path / 'mixed.mp4'
    report = media.assemble(parts, output, 36, (96, 64))
    assert report['video_assembly_mode'] == 'decode_reencode'
    assert report['all_pts_exact'] and report['full_decode_pass']


def test_incomplete_clip_never_replaces_existing_output(tmp_path, monkeypatch):
    parts = _parts(tmp_path, monkeypatch)
    parts[-1]['frames'] += 1
    output = tmp_path / 'keep.mp4'
    output.write_bytes(b'previous-user-output')
    with pytest.raises(ValueError, match='lost delivery frames'):
        media.assemble(parts, output, 37, (96, 64))
    assert output.read_bytes() == b'previous-user-output'
    assert not list(tmp_path.glob('.*.mp4'))


def test_source_timing_changed_after_preflight_never_replaces_output(tmp_path, monkeypatch):
    av = pytest.importorskip('av')
    import nodes.long_video_packet_copy as packet_copy
    parts = _parts(tmp_path, monkeypatch)
    original_plan = packet_copy.plan_copy

    def changed_after_scan(*args, **kwargs):
        plan = original_plan(*args, **kwargs)
        assert plan is not None
        rewritten = tmp_path / 'changed.mp4'
        with av.open(parts[1]['video']) as source, av.open(str(rewritten), 'w') as dest:
            original = source.streams.video[0]
            video = dest.add_stream_from_template(original)
            video.time_base = original.time_base
            index = 0
            for packet in source.demux(original):
                if not packet.size:
                    continue
                if index == 1:
                    packet.dts += 1
                packet.stream = video
                dest.mux(packet)
                index += 1
        rewritten.replace(Path(parts[1]['video']))
        return plan

    monkeypatch.setattr(packet_copy, 'plan_copy', changed_after_scan)
    output = tmp_path / 'keep.mp4'
    output.write_bytes(b'previous-user-output')
    with pytest.raises(ValueError, match='changed after packet-copy preflight'):
        media.assemble(parts, output, 36, (96, 64))
    assert output.read_bytes() == b'previous-user-output'
    assert not list(tmp_path.glob('.*.mp4'))


def test_concealed_corrupt_idr_never_replaces_existing_output(tmp_path, monkeypatch):
    av = pytest.importorskip('av')
    parts = _parts(tmp_path, monkeypatch)
    rewritten = tmp_path / 'corrupt.mp4'
    with av.open(parts[1]['video']) as source, av.open(str(rewritten), 'w') as dest:
        original = source.streams.video[0]
        extra = original.codec_context.extradata
        length_size = (extra[4] & 3) + 1
        video = dest.add_stream_from_template(original)
        video.time_base = original.time_base
        index = 0
        for packet in source.demux(original):
            if not packet.size:
                continue
            if index == 0:
                payload, changed, cursor = bytes(packet), bytearray(), 0
                while cursor < len(payload):
                    size = int.from_bytes(payload[cursor:cursor+length_size], 'big')
                    cursor += length_size
                    nal = payload[cursor:cursor+size]
                    cursor += size
                    if nal[0] & 31 == 5:
                        nal = nal[:max(8, len(nal)//3)]
                    changed.extend(len(nal).to_bytes(length_size, 'big'))
                    changed.extend(nal)
                modified = av.Packet(bytes(changed))
                modified.pts, modified.dts = packet.pts, packet.dts
                modified.duration, modified.time_base = packet.duration, packet.time_base
                modified.is_keyframe = packet.is_keyframe
                packet = modified
            packet.stream = video
            dest.mux(packet)
            index += 1
    rewritten.replace(Path(parts[1]['video']))
    with av.open(parts[1]['video']) as source:
        assert next(source.decode(video=0)).is_corrupt, 'Fixture must exercise concealed damage'
    output = tmp_path / 'keep.mp4'
    output.write_bytes(b'previous-user-output')
    with pytest.raises(ValueError, match='Corrupt decoded video frame'):
        media.assemble(parts, output, 36, (96, 64))
    assert output.read_bytes() == b'previous-user-output'
    assert not list(tmp_path.glob('.*.mp4'))
