import json
from pathlib import Path

import pytest

from nodes.longfilm_plan import validate_story, plan_chunks, compile_prompt


def story():
    return json.loads((Path(__file__).parents[1] / 'templates/u11_longfilm_story.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('vram,maximum', [(32,362),(8,124)])
def test_exact_timeline_and_native_endpoints(vram, maximum):
    chunks = plan_chunks(story(), vram)
    assert sum(c['visible_frames'] for c in chunks) == 1464
    assert chunks[0]['visible_start'] == 0
    assert chunks[-1]['visible_end'] == 1464
    for i,c in enumerate(chunks):
        assert (c['endpoint_frames'] - 39) % 51 == 0
        assert (c['sample_frames'] - 5) % 17 == 0
        assert c['sample_frames'] <= maximum
        assert c['context_frames'] == (39 if i else 0)
        assert c['tail_reserve_frames'] >= 17
        if i:
            assert c['visible_start'] == chunks[i-1]['visible_end']


def test_dialogue_boundary_and_compilation():
    s = story()
    s['cards'][1]['beats'][0]['dialogue'] = {'text':'我们继续走吧。','start_frame':230,'end_frame':260}
    chunks = plan_chunks(s)
    assert chunks[0]['visible_end'] == 230
    prompt = json.loads(compile_prompt(s,chunks[1]))
    assert set(prompt) == {'integrated_multimodal_description','overall_soundscape','non_diegetic_music'}
    assert '我们继续走吧。' in prompt['integrated_multimodal_description']
    assert 'frames 39–69' in prompt['integrated_multimodal_description']
    assert s['global_setting'] in prompt['integrated_multimodal_description']
    assert 'Picture' not in json.dumps(prompt)


def test_impossible_low_memory_dialogue_is_clear():
    s = story()
    s['cards'][1]['beats'][0]['dialogue'] = {'text':'长句','start_frame':220,'end_frame':300}
    with pytest.raises(ValueError, match='dialogue.*card-2'):
        plan_chunks(s,8)


def test_invalid_schema_rejected():
    for key,value in [('fps',25),('total_frames',1465),('mode','REF2VA'),('voice_mode','fish_lock')]:
        s = story(); s[key] = value
        with pytest.raises(ValueError): validate_story(s)
    s = story(); s['cards'][0]['beats'][0]['start_frame'] = 1
    with pytest.raises(ValueError): validate_story(s)


def test_prompt_contains_only_current_beats():
    s = story()
    s['cards'][-1]['beats'][0]['action'] = 'A unique final action.'
    assert 'unique final action' not in compile_prompt(s,plan_chunks(s)[0])
