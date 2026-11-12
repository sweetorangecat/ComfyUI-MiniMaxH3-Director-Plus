"""One-window previews and file-indexed assembly, with exact frame/sample clocks."""
from fractions import Fraction
from math import lcm
import os
from pathlib import Path
import uuid
import wave
import hashlib
import json

FPS = 24


def resolve_encoding(settings):
    """The bounded complete-film path: honest H264/MP4/8-bit/AAC only."""
    if not isinstance(settings, dict):
        raise ValueError('Long-video encoding settings must be an object')
    for field, allowed in (
            ('codec', {'Auto', 'H.264'}), ('container', {'Auto', 'MP4'}),
            ('bit_depth', {'Auto', '8-bit'}), ('audio_codec', {'Auto', 'AAC'})):
        value = settings.get(field, 'Auto')
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(f'长片完整文件当前仅支持 H.264 / MP4 / 8-bit / AAC；{field}={value!r} 不支持，请在原输出节点明确选择。')
    quality = settings.get('quality', 20)
    bitrate = settings.get('audio_bitrate', '320k')
    if type(quality) is not int or not 0 <= quality <= 51:
        raise ValueError('quality must be an integer CRF in 0..51')
    if not isinstance(bitrate, str) or bitrate not in {'64k','96k','128k','160k','192k','256k','320k'}:
        raise ValueError('audio_bitrate must be one of 64k..320k U11 choices')
    try:
        rate = float(settings.get('frame_rate', 24))
    except (TypeError, ValueError) as exc:
        raise ValueError('Long-video delivery requires 24 fps') from exc
    if (rate != FPS or settings.get('pingpong', False) or settings.get('crop_to_audio', False)
            or not settings.get('save_output', True)):
        raise ValueError('长片完整文件要求 24 fps、保存输出；不支持来回播放或按音频裁帧。')
    return {'codec': 'H.264', 'encoder': 'libx264', 'container': 'MP4', 'bit_depth': '8-bit',
            'pixel_format': 'yuv420p', 'requested_quality': quality, 'crf': min(quality, 16),
            'audio_codec': 'AAC', 'audio_bitrate': int(bitrate[:-1]) * 1000,
            'color_matrix': 'BT.709', 'color_primaries': 'BT.709', 'color_transfer': 'BT.709',
            'color_range': 'limited', 'rgb_input_range': 'full'}


def encoder_capabilities():
    """No models, downloads, files or queue work; require real encoder support."""
    try:
        import av
        from av.video.reformatter import VideoReformatter
        av.Codec('libx264', 'w')
        av.Codec('aac', 'w')
        if 'dst_color_range' not in (VideoReformatter.reformat.__doc__ or ''):
            raise ValueError('PyAV lacks explicit RGB/YUV color range conversion')
    except (ImportError, ValueError, RuntimeError) as exc:
        raise ValueError('长片需要支持 libx264/AAC 与显式色域/范围转换的 PyAV；参见 requirements-long-video.txt。不会自动安装或下载。' + str(exc)) from exc
    return {'checked': True, 'pyav_version': av.__version__, 'video_encoder': 'libx264',
            'audio_encoder': 'aac', 'explicit_bt709_conversion': True}


def _encoding(value):
    if value is None:
        return resolve_encoding({})
    if not isinstance(value, dict):
        raise ValueError('Resolved long-video encoding must be an object')
    # Revalidate even an injected assembly recipe; never silently downgrade it.
    candidate = dict(value, quality=value.get('requested_quality', value.get('crf', 20)),
                     audio_bitrate=str(value.get('audio_bitrate', 320000) // 1000) + 'k')
    result = resolve_encoding(candidate)
    if result != value:
        raise ValueError('Assembly requires the validated, resolved long-video encoding recipe')
    return result


def _bt709(frame):
    """Convert actual full-range RGB with a 709 matrix before tagging it."""
    from av.video.reformatter import VideoReformatter
    result = VideoReformatter().reformat(frame, format='yuv420p',
        src_colorspace='ITU709', dst_colorspace='ITU709',
        src_color_range='JPEG', dst_color_range='MPEG')
    result.colorspace = result.color_primaries = result.color_trc = 1
    result.color_range = 1
    return result


def _mux(dest, stream, frame):
    for packet in stream.encode(frame):
        dest.mux(packet)


def _video_stream(dest, width, height, encoding=None):
    encoding = _encoding(encoding)
    stream = dest.add_stream('libx264', rate=FPS)
    stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
    stream.options = {'crf': str(encoding['crf']), 'preset': 'medium'}
    stream.time_base = Fraction(1, FPS)
    stream.codec_context.colorspace = stream.codec_context.color_primaries = stream.codec_context.color_trc = 1
    stream.codec_context.color_range = 1
    return stream


def _audio_stream(dest, rate, channels, encoding=None):
    encoding = _encoding(encoding)
    if channels not in (1, 2):
        raise ValueError('Long-video audio must be mono or stereo')
    stream = dest.add_stream('aac', rate=rate)
    stream.layout = 'mono' if channels == 1 else 'stereo'
    stream.bit_rate = encoding['audio_bitrate']
    return stream


def _audio_frame(values, rate, offset, layout):
    import av
    import numpy as np
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(values, dtype=np.float32), format='fltp', layout=layout)
    frame.sample_rate, frame.pts, frame.time_base = rate, offset, Fraction(1, rate)
    return frame


def encode_preview(path, images, audio, *, encoding=None):
    """Encode just one physical window. No RGB survives beyond this job."""
    import av
    import numpy as np
    encoding = _encoding(encoding)
    path = Path(path)
    temporary = path.with_name('.' + uuid.uuid4().hex + '.mp4')
    path.parent.mkdir(parents=True, exist_ok=True)
    values = audio['waveform']
    if values.ndim != 3 or values.shape[0] != 1 or images.ndim != 4 or not len(images):
        raise ValueError('Invalid preview video/audio geometry')
    rate, channels = int(audio['sample_rate']), int(values.shape[1])
    try:
        with av.open(str(temporary), 'w', options={'movie_timescale': str(lcm(FPS, rate))}) as dest:
            video = _video_stream(dest, int(images.shape[2]), int(images.shape[1]), encoding)
            sound = _audio_stream(dest, rate, channels, encoding)
            for i, image in enumerate(images):
                frame = av.VideoFrame.from_ndarray(np.round(image.detach().float().cpu().clamp(0, 1).numpy() * 255).astype('uint8'), format='rgb24')
                frame = _bt709(frame)
                frame.pts, frame.time_base = i, Fraction(1, FPS)
                _mux(dest, video, frame)
            _mux(dest, video, None)
            limit = min(values.shape[-1], round(len(images) * rate / FPS))
            for i in range(0, limit, 4096):
                chunk = values[0, :, i:min(i + 4096, limit)].detach().float().cpu().numpy()
                _mux(dest, sound, _audio_frame(chunk, rate, i, sound.layout.name))
            _mux(dest, sound, None)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_delivery_audio(path, audio, window):
    """Use cumulative global sample endpoints, avoiding per-window drift."""
    import numpy as np
    values, rate = audio['waveform'], int(audio['sample_rate'])
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] not in (1, 2) or rate <= 0:
        raise ValueError('Invalid delivery audio geometry')
    a, b = window['new_start'], window['new_end']
    start = round(Fraction(window['context_frames'] * rate, FPS))
    ga, gb = round(Fraction(a * rate, FPS)), round(Fraction(b * rate, FPS))
    count = gb - ga
    if count <= 0 or start + count > values.shape[-1]:
        raise ValueError('Not enough actual audio for the delivery interval')
    chunk = values[0, :, start:start + count].detach().float().cpu().clamp(-1, 1).numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as dest:
        dest.setnchannels(values.shape[1])
        dest.setsampwidth(2)
        dest.setframerate(rate)
        dest.writeframes(np.round(chunk.T * 32767).astype('<i2').tobytes())
    return {'local_start': start, 'local_end': start + count, 'global_start': ga,
            'global_end': gb, 'count': count, 'sample_rate': rate, 'channels': int(values.shape[1])}


def inspect_movie(path, total_frames, target=None, *, audio_sample_tolerance=1):
    """Full decode is technical evidence, separate from any content review."""
    import av
    count, pts_exact = 0, True
    with av.open(str(path)) as source:
        if len(source.streams.video) != 1 or len(source.streams.audio) != 1:
            raise ValueError('Final video must have one video and one audio track')
        v, a = source.streams.video[0], source.streams.audio[0]
        rate = a.codec_context.sample_rate
        duration = Fraction(a.duration or 0) * a.time_base
        width, height = v.width, v.height
        for frame in source.decode(video=0):
            pts_exact &= frame.pts is not None and frame.pts * frame.time_base == Fraction(count, FPS)
            count += 1
    with av.open(str(path)) as source:
        decoded_samples = sum(f.samples for f in source.decode(audio=0))
    if type(audio_sample_tolerance) is not int or audio_sample_tolerance < 1:
        raise ValueError('Audio tolerance must be an explicit positive sample count')
    expected_samples = round(Fraction(total_frames * rate, FPS))
    delta = duration * rate - expected_samples
    if (count != total_frames or not pts_exact or (target and (width, height) != tuple(target))
            or abs(delta) > audio_sample_tolerance or decoded_samples < expected_samples):
        raise ValueError(f'Final media differs from plan: frames={count}/{total_frames}, '
                         f'pts_exact={pts_exact}, canvas={(width, height)}/{target}, '
                         f'audio_duration={duration}, expected={Fraction(total_frames, FPS)}, '
                         f'sample_rate={rate}')
    return {'frames': count, 'fps': FPS, 'width': width, 'height': height,
            'seconds': count / FPS, 'all_pts_exact': bool(pts_exact), 'full_decode_pass': True,
            'audio_duration': float(duration), 'sample_rate': rate,
            'audio_duration_delta_samples': float(delta),
            'audio_duration_tolerance_samples': audio_sample_tolerance,
            'decoded_aac_samples_including_codec_padding': decoded_samples,
            'content_review': 'not_granted_by_technical_validation'}


def assemble(parts, output, total_frames, target, *, encoding=None):
    """Decode/re-encode one frame and bounded PCM blocks, in source order.

    No frame skipping, speed change, interpolation or padding is permitted.
    Exact target aspect uses the same center crop for every frame.
    """
    import av
    import numpy as np
    encoding = _encoding(encoding)
    if not parts or sum(x['frames'] for x in parts) != total_frames:
        raise ValueError('Assembly index does not cover the target')
    width, height = map(int, target)
    output = Path(output)
    temporary = output.with_name('.' + uuid.uuid4().hex + '.mp4')
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(parts[0]['audio']), 'rb') as first:
        rate, channels = first.getframerate(), first.getnchannels()
    cursor = samples = 0
    try:
        # MP4's default 1000 Hz movie clock rounds edit-list audio endpoints
        # to milliseconds. Use a common clock for whole frames and PCM samples.
        with av.open(str(temporary), 'w', options={
                'movflags': '+faststart', 'movie_timescale': str(lcm(FPS, rate))}) as dest:
            video, sound = _video_stream(dest, width, height, encoding), _audio_stream(dest, rate, channels, encoding)
            for item in parts:
                local = 0
                with av.open(str(item['video'])) as source:
                    for frame in source.decode(video=0):
                        if local >= item['frames']:
                            raise ValueError('An enhanced source has unexpected extra frames')
                        # Trim only uniform spatial rounding. Temporal ranges
                        # were fixed by the approved native window, before SR.
                        if frame.width < width or frame.height < height:
                            raise ValueError('Enhanced source is smaller than the output canvas')
                        image = frame.to_ndarray(format='rgb24')
                        x, y = (frame.width - width) // 2, (frame.height - height) // 2
                        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image[y:y+height, x:x+width]), format='rgb24')
                        frame = _bt709(frame)
                        frame.pts, frame.time_base = cursor, Fraction(1, FPS)
                        _mux(dest, video, frame)
                        cursor += 1
                        local += 1
                if local != item['frames']:
                    raise ValueError('An enhanced source lost delivery frames')
                with wave.open(str(item['audio']), 'rb') as source:
                    if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (rate, channels, 2):
                        raise ValueError('Source audio formats differ')
                    expected = round(Fraction(cursor * rate, FPS)) - samples
                    if source.getnframes() != expected:
                        raise ValueError('PCM index differs from the cumulative global clock')
                    while True:
                        raw = source.readframes(4096)
                        if not raw:
                            break
                        data = np.frombuffer(raw, '<i2').reshape(-1, channels).T.astype('float32') / 32767
                        _mux(dest, sound, _audio_frame(data, rate, samples, sound.layout.name))
                        samples += data.shape[1]
            _mux(dest, video, None)
            _mux(dest, sound, None)
        report = inspect_movie(temporary, total_frames, target)
        report.update(source_audio_samples=samples, source_parts=len(parts), encoding=encoding,
                      temporal_edits='none; contiguous approved net-new ranges only')
        os.replace(temporary, output)
        return report
    finally:
        temporary.unlink(missing_ok=True)


def seam_contact_sheet(segments, directory):
    """At most three small rows of four actual frames; no full-film RGB load.

    Source hashes and actual end-exclusive ranges key an immutable cache. This
    optional explicit review request hashes MP4 files, never clean AV tensors.
    """
    import av
    from PIL import Image, ImageDraw
    if not isinstance(segments, list) or not 2 <= len(segments) <= 3:
        raise ValueError('Seam evidence requires two or three actual source segments')
    identity = []
    for item in segments:
        a, b = item['local_frames']
        if type(a) is not int or type(b) is not int or not 0 <= a < b or b-a > 8:
            raise ValueError('Seam evidence is limited to eight frames per actual range')
        digest = hashlib.sha256()
        with Path(item['path']).open('rb') as source:
            for block in iter(lambda: source.read(1024*1024), b''):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != item['artifact']['sha256']:
            raise ValueError('Actual seam preview source no longer matches its immutable artifact')
        identity.append({'sha256': actual, 'range': [a,b], 'role': item['role'],
                         'take_id': item['take_id'], 'film_seconds': item['film_seconds']})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / (key + '.jpg')
    if output.is_file():
        return output
    cells = max(b-a for a,b in (item['local_frames'] for item in segments))
    sheet = Image.new('RGB', (cells*256, len(segments)*190), '#161a20')
    draw = ImageDraw.Draw(sheet)
    for row, item in enumerate(segments):
        a, b = item['local_frames']
        top = row*190
        draw.text((6,top+3), f'{item["role"]} {item["take_id"][:28]} local [{a},{b}) film {item["film_seconds"][0]:.3f}-{item["film_seconds"][1]:.3f}s', fill='white')
        found = set()
        with av.open(item['path']) as source:
            stream = source.streams.video[0]
            source.seek(int(Fraction(a,FPS) / stream.time_base), stream=stream, backward=True)
            for frame in source.decode(video=0):
                if frame.pts is None:
                    raise ValueError('Actual preview has no frame time evidence')
                index = round(frame.pts*frame.time_base*FPS)
                if index >= b:
                    break
                if index < a:
                    continue
                found.add(index)
                image = frame.to_image()
                image.thumbnail((252,140))
                sheet.paste(image, ((index-a)*256,top+27))
                draw.text(((index-a)*256+5,top+171), f'f{index} = {index/FPS:.3f}s', fill='white')
        if found != set(range(a,b)):
            raise ValueError('Actual seam preview does not contain every requested frame')
    temporary = directory / ('.'+uuid.uuid4().hex+'.jpg')
    try:
        sheet.save(temporary, quality=92)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
