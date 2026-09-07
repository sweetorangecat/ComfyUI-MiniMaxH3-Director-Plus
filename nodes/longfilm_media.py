"""Disk checkpoints and bounded-memory, exact-length AV assembly.

Heavy media/torch dependencies are imported only at their point of use.
The lossless audio master uses Matroska, avoiding per-segment AAC priming.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from fractions import Fraction


def _temporary(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.stem + '-', suffix=path.suffix, dir=path.parent)
    os.close(fd)
    return Path(name)


def atomic_write_json(path, data):
    path = Path(path)
    temporary = _temporary(path)
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_project_dir(output_root, project_name):
    name = str(project_name)
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    if not re.fullmatch(r'[\w-][\w .-]{0,99}', name) or name.endswith(('.', ' ')) or name.split('.')[0].upper() in reserved:
        raise ValueError('project_name must be one safe directory name')
    root = Path(output_root).resolve()
    base = root / 'h3_longfilm'
    target = base / name
    for candidate in (base, target):
        # Reject junctions/symlinks, including ones resolving within the root.
        if candidate.is_symlink() or (hasattr(candidate, 'is_junction') and candidate.is_junction()):
            raise ValueError('Project directory cannot use symlinks or junctions')
        if not candidate.resolve().is_relative_to(root) or (candidate == target and not candidate.resolve().is_relative_to(base.resolve())):
            raise ValueError('Project directory escapes output root')
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_av_latent(path, latent):
    """Persist the caller's clean denoised x0 AV samples, stripping noise masks."""
    from safetensors.torch import save_file
    samples = latent['samples']
    if not hasattr(samples, 'unbind'):
        raise ValueError('Expected nested video/audio latent samples')
    parts = list(samples.unbind())
    if len(parts) != 2:
        raise ValueError('Expected exactly video and audio latent samples')
    tensors = {name: value.detach().to(device='cpu').contiguous().clone() for name, value in zip(('video', 'audio'), parts)}
    temporary = _temporary(path)
    try:
        save_file(tensors, str(temporary), metadata={'format': 'h3-clean-av-x0-v1'})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_av_latent(path):
    from safetensors import safe_open
    from comfy.nested_tensor import NestedTensor
    with safe_open(str(path), framework='pt', device='cpu') as source:
        if source.metadata().get('format') != 'h3-clean-av-x0-v1' or set(source.keys()) != {'video', 'audio'}:
            raise ValueError('Unsupported clean AV checkpoint')
        return {'samples': NestedTensor([source.get_tensor('video'), source.get_tensor('audio')])}


def probe_video(path):
    import av
    with av.open(str(path)) as source:
        if len(source.streams.video) != 1 or len(source.streams.audio) != 1:
            raise ValueError('Expected one video and one audio stream')
        video, audio = source.streams.video[0], source.streams.audio[0]
        info = dict(frames=0, width=video.width, height=video.height, fps=float(video.average_rate),
                    audio_samples=0, sample_rate=audio.codec_context.sample_rate,
                    channels=audio.codec_context.channels)
        for packet in source.demux(video, audio):
            for frame in packet.decode():
                if isinstance(frame, av.VideoFrame): info['frames'] += 1
                else: info['audio_samples'] += frame.samples
        return info


def _streams(container, width, height, fps, sample_rate, channels):
    video = container.add_stream('libx264', rate=Fraction(fps))
    video.width, video.height, video.pix_fmt = width, height, 'yuv420p'
    video.options = {'crf': '16', 'preset': 'medium'}
    audio = container.add_stream('pcm_s16le', rate=sample_rate)
    audio.layout = {1: 'mono', 2: 'stereo'}.get(channels, '')
    return video, audio


def _mux(container, stream, frame):
    for packet in stream.encode(frame): container.mux(packet)


def encode_segment(path, images, audio, fps=24):
    import av
    import numpy as np
    if fps <= 0 or images.ndim != 4 or images.shape[-1] != 3 or len(images) == 0:
        raise ValueError('Expected nonempty BHWC RGB images and positive fps')
    waveform = audio['waveform']
    rate = int(audio['sample_rate'])
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2) or rate <= 0:
        raise ValueError('Expected one mono/stereo audio batch and positive sample rate')
    count, height, width, _ = images.shape
    if width % 2 or height % 2:
        raise ValueError('H264 yuv420p requires even dimensions')
    expected_samples = round(count * rate / fps)
    if waveform.shape[-1] != expected_samples:
        raise ValueError(f'Audio must contain exactly {expected_samples} samples for {count} frames')
    temporary = _temporary(path)
    try:
        with av.open(str(temporary), 'w', format='matroska') as dest:
            video, sound = _streams(dest, width, height, fps, rate, waveform.shape[1])
            offset = 0
            for index, image in enumerate(images):
                pixels = (image.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
                frame.pts, frame.time_base = index, Fraction(1, fps)
                _mux(dest, video, frame)
                end = round((index + 1) * rate / fps)
                values = waveform[0, :, offset:end].detach().cpu().clamp(-1, 1).numpy()
                pcm = np.round(values.T.reshape(1, -1) * 32767).astype(np.int16)
                frame = av.AudioFrame.from_ndarray(pcm, format='s16', layout=sound.layout.name)
                frame.sample_rate, frame.pts, frame.time_base = rate, offset, Fraction(1, rate)
                _mux(dest, sound, frame)
                offset = end
            _mux(dest, video, None)
            _mux(dest, sound, None)
        info = probe_video(temporary)
        if info['frames'] != count or info['audio_samples'] != expected_samples:
            raise ValueError('Encoded segment lost frames or audio samples')
        os.replace(temporary, path)
        return info
    finally:
        temporary.unlink(missing_ok=True)


def join_segments(paths, output_path, expected_frames=1464, fps=24):
    import av
    paths = list(paths)
    if not paths:
        raise ValueError('No segments to join')
    infos = [probe_video(path) for path in paths]
    first = infos[0]
    if sum(info['frames'] for info in infos) != expected_frames:
        raise ValueError('Segment frames do not match expected frames')
    for info in infos:
        if any(info[key] != first[key] for key in ('width', 'height', 'sample_rate', 'channels')) or abs(info['fps'] - fps) > .001:
            raise ValueError('Segment AV formats must match')
    samples = sum(info['audio_samples'] for info in infos)
    if samples != round(expected_frames * first['sample_rate'] / fps):
        raise ValueError('Audio samples do not match expected duration')
    temporary = _temporary(output_path)
    try:
        with av.open(str(temporary), 'w', format='matroska') as dest:
            video, audio = _streams(dest, first['width'], first['height'], fps, first['sample_rate'], first['channels'])
            video_pts = audio_pts = 0
            for path in paths:
                with av.open(str(path)) as source:
                    for packet in source.demux():
                        for frame in packet.decode():
                            if isinstance(frame, av.VideoFrame):
                                frame.pts, frame.time_base = video_pts, Fraction(1, fps)
                                video_pts += 1
                                _mux(dest, video, frame)
                            elif isinstance(frame, av.AudioFrame):
                                frame.pts, frame.time_base = audio_pts, Fraction(1, first['sample_rate'])
                                audio_pts += frame.samples
                                _mux(dest, audio, frame)
            _mux(dest, video, None)
            _mux(dest, audio, None)
        result = probe_video(temporary)
        if result['frames'] != expected_frames or result['audio_samples'] != samples:
            raise ValueError('Final output lost frames or audio samples')
        os.replace(temporary, output_path)
        return result
    finally:
        temporary.unlink(missing_ok=True)
