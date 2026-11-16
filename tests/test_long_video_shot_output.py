import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest
import torch

import nodes.stream_output as output
from nodes.execution_report import attach_report, report_from
from nodes.long_video_shot_output import MiniMaxH3LongVideoShotOutput
from nodes.long_video_media import encode_preview
from test_long_video_shot_service import setup
from test_stream_output import _fake_dasiwa


def fixture(tmp_path, monkeypatch):
    store, backend, runner = setup(tmp_path)
    runner.step()
    graph = backend.submitted[0][1]
    inputs = next(n['inputs'] for n in graph.values() if n['class_type'] == 'MiniMaxH3LongVideoShotOutput')
    inputs = dict(inputs)
    for key in ('images', 'guide', 'audio', 'audio_override'):
        inputs.pop(key, None)
    captured = []
    module = _fake_dasiwa([])
    import folder_paths
    monkeypatch.setattr(folder_paths, 'get_output_directory', lambda: str(tmp_path))

    def save_path(prefix, root, width, height):
        path = Path(root) / prefix
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path.parent), path.name, 1, str(path.parent.relative_to(root)), None

    module.folder_paths = SimpleNamespace(get_output_directory=lambda: str(tmp_path),
        get_temp_directory=lambda: str(tmp_path), get_save_image_path=save_path)

    def audio_file(audio):
        captured.append(audio)
        return ('unused-test-pcm.wav', audio['sample_rate'], audio['waveform'].shape[1]), 15.0

    def encode(*args):
        frames = torch.cat(list(args[6]()))
        encode_preview(args[7], frames, captured[-1])
        return 'libx264'

    module._audio_file = audio_file
    module._encode_with_available_encoder = encode
    monkeypatch.setattr(output, '_dasiwa_video_module', lambda: module)
    monkeypatch.setattr(output, '_probe_working_video_encoders', lambda _: None)
    return store, inputs, captured


def test_real_short_output_trims_exact_frames_and_its_actual_pcm_before_encoding(tmp_path, monkeypatch):
    pytest.importorskip('av')
    store, inputs, captured = fixture(tmp_path, monkeypatch)
    guide = {'postprocess_path': 'native_bypass', 'target_width': 32, 'target_height': 18,
             'voice_mode': 'none', 'audio_loudness': 'original'}
    images = torch.full((362, 18, 32, 3), .3)
    # Mark the sampling surplus so it cannot silently enter the delivery.
    images[360:] = 1
    provenance = {'version': 1, 'plan': {'seed': 123},
        'stages': [{'stage': 'sampling', 'status': 'completed', 'seconds': 1.25}],
        'sampling_settings': {'sampler': 'SelfLiftAvatarH3Sampler', 'steps': 10},
        'color_guard': {'method': 'scene_reference_v1', 'strength': .5}}
    attach_report(images, provenance)
    sound = {'waveform': torch.linspace(-.2, .2, 500000).reshape(1,1,-1), 'sample_rate':32000}
    result = MiniMaxH3LongVideoShotOutput().combine(images, guide, audio=sound, **inputs)
    record = store.read()['windows'][0]
    assert record['status'] == 'awaiting_review'
    assert record['technical']['frames'] == 360
    assert record['source_frames'] == 362 and record['delivered_frames'] == 360
    assert record['audio_index']['count'] == 480000
    assert torch.equal(captured[-1]['waveform'], sound['waveform'][...,:480000])
    with wave.open(str(store.path(record['artifacts']['delivery_audio']['name']))) as pcm:
        assert pcm.getnframes() == 480000
        samples = __import__('numpy').frombuffer(pcm.readframes(480000), dtype='<i2')
    assert abs(samples[-1]/32767 - float(sound['waveform'][0,0,479999])) < 1/32767
    assert Path(result['result'][1]).is_file()
    report = result['ui']['execution_report'][0]
    assert any(stage['stage'] == 'sampling' and stage['seconds'] == 1.25
               and stage['origin'] == 'upstream_artifact' for stage in report['stages'])
    assert report['sampling_settings']['sampler'] == 'SelfLiftAvatarH3Sampler'
    assert report['sampling_settings']['steps'] == 10
    assert report['color_guard']['method'] == 'scene_reference_v1'
    assert report_from(images) == provenance
    assert set(k for k in record['artifacts'] if k.startswith('before_sr_')) == {'before_sr_0','before_sr_180','before_sr_359'}
    with pytest.raises(ValueError, match='immutable short clip'):
        MiniMaxH3LongVideoShotOutput().combine(images, guide, audio=sound, **inputs)


def test_changed_immutable_request_rejected_before_original_output(tmp_path, monkeypatch):
    store, inputs, _ = fixture(tmp_path, monkeypatch)
    request = json.loads(inputs['request_json'])
    request['seed'] += 1
    inputs['request_json'] = json.dumps(request)
    with pytest.raises(ValueError, match='immutable generation request'):
        MiniMaxH3LongVideoShotOutput().combine(torch.zeros(362,18,32,3), {}, **inputs)
    assert not store.path(f"takes/{request['take_id']}/shot_receipt.json").exists()
