"""Conservative H.264 packet concatenation, without changing source pictures."""
from fractions import Fraction
import hashlib
from struct import pack

FPS = 24


def _signature(stream, target):
    context = stream.codec_context
    extra = bytes(context.extradata or b'')
    if (context.name != 'h264' or (stream.width, stream.height) != tuple(target)
            or stream.average_rate != FPS or context.format is None
            or context.format.name != 'yuv420p'
            or (context.colorspace, context.color_primaries, context.color_trc, context.color_range) != (1, 1, 1, 1)
            or stream.sample_aspect_ratio not in (None, Fraction(1))
            or len(extra) < 7 or extra[0] != 1):
        return None
    return (context.name, stream.width, stream.height, context.format.name,
            context.profile, extra)


def _closed_start(payload, extra):
    length_size = (extra[4] & 3) + 1
    cursor, slices = 0, set()
    while cursor < len(payload):
        if cursor + length_size > len(payload):
            return False
        size = int.from_bytes(payload[cursor:cursor+length_size], 'big')
        cursor += length_size
        if not size or cursor + size > len(payload):
            return False
        kind = payload[cursor] & 31
        if 1 <= kind <= 5:
            slices.add(kind)
        cursor += size
    return slices == {5}


def _hash_packet(digest, payload):
    digest.update(len(payload).to_bytes(8, 'big'))
    digest.update(payload)


def _hash_timing(digest, packet, clock):
    flags = [bool(getattr(packet, key, False)) for key in
             ('is_keyframe', 'is_corrupt', 'is_discard', 'is_disposable', 'is_trusted')]
    digest.update(pack('>qqqqq?????', packet.pts, packet.dts, packet.duration,
                       clock.numerator, clock.denominator, *flags))


def _scan(source, stream, count, extra):
    if type(count) is not int or count <= 0 or stream.frames not in (0, count):
        return None
    positions, first_dts = set(), None
    digest = hashlib.sha256()
    timing = hashlib.sha256()
    packets = 0
    for packet in source.demux(stream):
        if not packet.size:
            continue
        if packet.pts is None or packet.dts is None or not packet.duration:
            return None
        clock = Fraction(packet.time_base or stream.time_base)
        position = packet.pts * clock * FPS
        if ((Fraction(1, FPS) / clock).denominator != 1 or position.denominator != 1
                or not 0 <= position < count or position in positions
                or packet.duration * clock != Fraction(1, FPS)):
            return None
        payload = bytes(packet)
        current_dts = packet.dts * clock
        if first_dts is None:
            if position != 0 or not packet.is_keyframe or not _closed_start(payload, extra):
                return None
            first_dts = current_dts
            if first_dts > 0 or (first_dts * FPS).denominator != 1:
                return None
        if current_dts != first_dts + Fraction(packets, FPS):
            return None
        positions.add(int(position))
        _hash_packet(digest, payload)
        _hash_timing(timing, packet, clock)
        packets += 1
    if packets != count or len(positions) != count:
        return None
    return {'frames': count, 'first_dts': first_dts,
            'time_base': Fraction(stream.time_base), 'packet_sha256': digest.hexdigest(),
            'packet_timing_sha256': timing.hexdigest()}


def plan_copy(parts, target):
    """Return a fully scanned compatible plan, or retain normal encoding."""
    import av
    output_type = getattr(av.container, 'OutputContainer', None)
    if (not callable(getattr(output_type, 'add_stream_from_template', None))
            or not hasattr(av.VideoFrame, 'is_corrupt')):
        return None
    plan = {'target': tuple(target), 'parts': []}
    signature = delay = None
    for item in parts:
        with av.open(str(item['video'])) as source:
            if len(source.streams.video) != 1:
                return None
            stream = source.streams.video[0]
            actual = _signature(stream, target)
            if actual is None or (signature is not None and actual != signature):
                return None
            evidence = _scan(source, stream, item['frames'], actual[-1])
            if evidence is None or (delay is not None and evidence['first_dts'] != delay):
                return None
            signature, delay = actual, evidence['first_dts']
            evidence.update(path=str(item['video']), signature=actual)
            plan['parts'].append(evidence)
    return plan if plan['parts'] else None


def create_stream(destination, plan):
    import av
    first = plan['parts'][0]
    with av.open(first['path']) as source:
        stream = destination.add_stream_from_template(source.streams.video[0])
        stream.time_base = first['time_base']
    return stream


def mux_part(destination, stream, evidence, frame_offset):
    """Keep compressed bytes and reorder timing; offset the complete clip."""
    import av
    digest, timing, count = hashlib.sha256(), hashlib.sha256(), 0
    with av.open(evidence['path']) as source:
        original = source.streams.video[0]
        if _signature(original, evidence['signature'][1:3]) != evidence['signature']:
            raise ValueError('Video source changed after packet-copy preflight')
        for packet in source.demux(original):
            if not packet.size:
                continue
            clock = Fraction(packet.time_base or original.time_base)
            shift = Fraction(frame_offset, FPS) / clock
            if shift.denominator != 1 or packet.pts is None or packet.dts is None:
                raise ValueError('Video packet clock changed after packet-copy preflight')
            _hash_packet(digest, bytes(packet))
            _hash_timing(timing, packet, clock)
            pts, dts = packet.pts + int(shift), packet.dts + int(shift)
            packet.stream = stream
            packet.pts, packet.dts, packet.time_base = pts, dts, clock
            destination.mux(packet)
            count += 1
    if (count != evidence['frames'] or digest.hexdigest() != evidence['packet_sha256']
            or timing.hexdigest() != evidence['packet_timing_sha256']):
        raise ValueError('Video packets changed after packet-copy preflight')
