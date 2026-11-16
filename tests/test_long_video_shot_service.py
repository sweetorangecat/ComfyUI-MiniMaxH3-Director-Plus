from copy import deepcopy
import json

import pytest

from nodes.long_video_shot_service import ShotCoordinator, configure_shots
from nodes.long_video_shots import automatic_shots
from nodes.long_video_store import ProjectStore, fingerprint


def setup(tmp_path):
    base = json.loads((__import__('pathlib').Path(__file__).parents[1] / 'templates/u11_api.json').read_text(encoding='utf-8'))
    from test_long_video_shots import prompt
    base['10']['inputs'].update(duration=45, prompt=prompt(), mode='REF2VA', voice_mode='h3_reference',
                              reference_image_1_file='identity.png', voice_reference_audio_file='voice.wav',
                              motion_smoothing='off', postprocess_mode='vosr2')
    active, config = automatic_shots(base, '10', 45)
    config['project_name'] = 'independent'
    store = ProjectStore(tmp_path, config['project_name'])
    configure_shots(store, active, config, asset_freezer=lambda _: {}, encoder_checker=lambda: {})
    backend = Backend()
    runner = ShotCoordinator(store, backend, assembler=assemble)
    return store, backend, runner


class Backend:
    def __init__(self):
        self.submitted, self.states = [], {}

    def submit(self, prompt_id, graph):
        self.submitted.append((prompt_id, deepcopy(graph)))
        self.states[prompt_id] = 'queued'

    def status(self, prompt_id):
        return self.states.get(prompt_id, 'missing')


def assemble(parts, output, frames, target, *, encoding):
    assert sum(item['frames'] for item in parts) == frames
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b'final movie')
    return {'frames': frames, 'parts': len(parts), 'encoding': encoding}


def complete(store, backend):
    state = store.read()
    job = state['active_job']
    record = next(w for w in state['windows'] if w['take_id'] == job['take_id'])
    artifacts = {}
    for name in ('native', 'preview', 'delivery_audio'):
        path = store.path(f"takes/{record['take_id']}/{name}.bin")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{record['take_id']} {name}".encode())
        artifacts[name] = store.artifact(path)
    saved = json.loads(store.path(f"takes/{record['take_id']}/request.json").read_text(encoding='utf-8'))
    receipt = {'artifacts': artifacts, 'sampling_route': 'original_short_graph', 'request_sha256': fingerprint(saved),
               'content_review': 'pending', 'technical': {'frames': record['window']['local_new_end'],
                    'width': state['config']['export']['target'][0], 'height': state['config']['export']['target'][1]}}
    path = store.write_once(f"takes/{record['take_id']}/shot_receipt.json", receipt)
    receipt['artifacts']['shot_receipt'] = store.artifact(path)
    store.record_native(record['take_id'], receipt)
    backend.states[job['prompt_id']] = 'completed'


def accept(store):
    record = next(w for w in store.read()['windows'] if w['status'] == 'awaiting_review')
    store.review(record['take_id'], True, 'Actual hands rest on desk; lips closed at the edit.',
                 'Synthetic lifecycle fixture only, no visual acceptance claimed.')


def test_each_independent_clip_uses_original_short_graph_and_waits_for_review(tmp_path):
    store, backend, runner = setup(tmp_path)
    assert runner.step() is True
    submitted = backend.submitted[0][1]
    assert any(n['class_type'] == 'MiniMaxH3LongVideoShotOutput' for n in submitted.values())
    assert not any(n['class_type'] == 'MiniMaxH3LongVideoContext' for n in submitted.values())
    assert runner.step() is True and len(backend.submitted) == 1
    complete(store, backend)
    assert runner.step() is False
    assert len(backend.submitted) == 1 and store.read()['status'] == 'awaiting_review'
    accept(store)
    assert runner.step() is True
    assert len(backend.submitted) == 2
    assert store.read()['windows'][1]['parent'] is None


def test_three_accepted_clips_assemble_exactly_once_and_resume_reuses_them(tmp_path):
    store, backend, runner = setup(tmp_path)
    for index in range(3):
        assert runner.step() is True
        complete(store, backend)
        assert runner.step() is False
        accept(store)
    assert runner.step() is False
    state = store.read()
    assert state['status'] == 'complete' and state['completed_frames'] == 1080
    assert state['assembly']['frames'] == 1080
    assert len(backend.submitted) == 3
    assert runner.step() is False
    assert store.read()['output'] == state['output']
    assert len(backend.submitted) == 3


def test_independent_retry_does_not_invalidate_later_accepted_clips(tmp_path):
    store, backend, runner = setup(tmp_path)
    for index in range(3):
        runner.step(); complete(store, backend); runner.step(); accept(store)
    runner.step()
    before = store.read()['windows']
    store.request_retry(1, replace_accepted=True, take_id=before[1]['take_id'])
    assert store.read()['windows'][0] == before[0]
    assert store.read()['windows'][2] == before[2]
    assert runner.step() is True
    assert len(backend.submitted) == 4
    assert store.read()['windows'][2] == before[2]


def test_stop_keeps_completed_clip_and_resume_does_not_duplicate_queue_work(tmp_path):
    store, backend, runner = setup(tmp_path)
    runner.step(); store.stop()
    assert runner.step() is True
    complete(store, backend)
    assert runner.step() is False
    assert store.read()['status'] == 'stopped'
    store.resume(); accept(store)
    assert runner.step() is True
    assert len(backend.submitted) == 2


def test_corrupted_clip_is_not_reused_or_assembled(tmp_path):
    store, backend, runner = setup(tmp_path)
    runner.step(); complete(store, backend); runner.step(); accept(store)
    record = store.read()['windows'][0]
    store.path(record['artifacts']['native']['name']).write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        runner.step()
    assert len(backend.submitted) == 1


def test_restart_recovers_published_shot_receipt_before_retrying(tmp_path):
    store, backend, runner = setup(tmp_path)
    runner.step(); complete(store, backend)
    with store.transaction() as state:
        state['windows'][0].update(status='queued', artifacts={})
    assert runner.step() is False
    assert store.read()['windows'][0]['status'] == 'awaiting_review'
    assert len(backend.submitted) == 1


def test_independent_assembly_preserves_original_short_output_canvas_above_preset(tmp_path):
    store, backend, runner = setup(tmp_path)
    for index in range(3):
        runner.step(); complete(store, backend); runner.step()
        with store.transaction() as state:
            state['windows'][index]['technical'].update(width=1920, height=1098)
        accept(store)
    observed = []
    def assembly(parts, output, frames, target, **kwargs):
        observed.append(target)
        return assemble(parts, output, frames, target, **kwargs)
    runner.assembler = assembly
    assert runner.step() is False
    assert observed == [(1920, 1098)]


def test_new_implicit_model_selection_cannot_change_a_resumed_clip_silently(tmp_path, monkeypatch):
    store, _, runner = setup(tmp_path)
    with store.transaction() as state:
        state['config']['asset_choices'] = {'native': 'old', 'export': 'old'}
    monkeypatch.setattr('nodes.long_video_shot_service._choices', lambda _: {'native': 'new', 'export': 'old'})
    with pytest.raises(ValueError, match='model selection changed'):
        runner.step()
