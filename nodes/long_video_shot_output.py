"""Publish the original short output and PCM as one immutable editorial take."""
from copy import deepcopy
import json

from .stream_output import MiniMaxH3StreamingVideoCombine
from .long_video_store import ProjectStore, fingerprint
from .long_video_media import encode_preview, inspect_movie, write_delivery_audio
from .execution_report import attach_report, report_from


class MiniMaxH3LongVideoShotOutput(MiniMaxH3StreamingVideoCombine):
    @classmethod
    def INPUT_TYPES(cls):
        fields = deepcopy(super().INPUT_TYPES())
        fields['required']['request_json'] = ('STRING',)
        return fields

    def combine(self, images, guide, request_json, **kwargs):
        import folder_paths
        request = json.loads(request_json)
        store = ProjectStore(folder_paths.get_output_directory(), request['project_name'])
        take_id = request['take_id']
        saved = json.loads(store.path(f'takes/{take_id}/request.json').read_text(encoding='utf-8'))
        if {key: value for key, value in request.items() if key != 'export_fingerprint'} != saved:
            raise ValueError('Short clip output differs from the immutable generation request')
        state = store.read()
        store.validate_parent_binding(state, saved)
        ProjectStore.active(state, take_id)
        if store.path(f'takes/{take_id}/shot_receipt.json').exists():
            raise ValueError('Refusing to overwrite an immutable short clip')
        window = saved['window']
        count = window['new_end'] - window['new_start']
        if images.ndim != 4 or len(images) < count:
            raise ValueError('Original short sampler did not produce the complete delivery interval')
        source_frames = len(images)
        source_report = report_from(images)
        images = images[:count]
        # Tensor slicing drops its attached provenance. Preserve the actual
        # sampler/decode/ColorGuard evidence without changing source pixels.
        if source_report:
            attach_report(images, source_report)
        from .short_quality_reference import validate_output
        quality_reference = validate_output(guide, images)
        artifacts, audio_index = {}, None
        original_audio = None
        guide = dict(guide)

        def capture(sound):
            # This receives the actual post-cleanup/reuse PCM from the original
            # short output. It neither replaces native speech nor changes gain.
            nonlocal audio_index, original_audio
            if sound is None:
                raise ValueError('H3 independent clips require the actual generated or selected audio')
            audio_index = write_delivery_audio(store.path(f'takes/{take_id}/delivery.wav'), sound, window)
            trimmed = dict(sound)
            trimmed['waveform'] = sound['waveform'][..., :audio_index['count']]
            original_audio = trimmed
            return trimmed

        guide['_h3_delivery_audio_callback'] = capture
        result = super().combine(images=images, guide=guide, **kwargs)
        output = result['result'][1]
        if audio_index is None:
            raise RuntimeError('Original short output did not publish its actual PCM audio')
        technical = inspect_movie(output, count, audio_sample_tolerance=max(1, audio_index['sample_rate'] // 1000))
        native = store.path(f'takes/{take_id}/native.mp4')
        encode_preview(native, images, original_audio, encoding=state['config']['export']['encoding'])
        artifacts.update(native=store.artifact(native), preview=store.artifact(output),
                         delivery_audio=store.artifact(store.path(f'takes/{take_id}/delivery.wav')))
        # Three full-size pre-SR probes make skin/hair/color inspection possible
        # without treating an enhanced/encoded frame as the native source.
        from PIL import Image
        import numpy as np
        for frame in sorted({0, count // 2, count - 1}):
            path = store.path(f'takes/{take_id}/before_sr_f{frame}.png')
            data = images[frame].detach().float().cpu().clamp(0, 1).numpy()
            Image.fromarray(np.round(data * 255).astype('uint8')).save(path)
            artifacts[f'before_sr_{frame}'] = store.artifact(path)
        receipt = {'artifacts': artifacts, 'request_sha256': fingerprint(saved),
                   'sampling_route': 'original_short_graph', 'source_frames': source_frames,
                   'delivered_frames': count, 'audio_index': audio_index, 'technical': technical,
                   'content_review': 'pending'}
        if quality_reference is not None:
            receipt['short_quality_reference'] = quality_reference
        path = store.write_once(f'takes/{take_id}/shot_receipt.json', receipt)
        receipt['artifacts']['shot_receipt'] = store.artifact(path)
        store.record_native(take_id, receipt)
        result.setdefault('ui', {})['h3_independent_shot'] = [str(path)]
        return result
