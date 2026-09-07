"""Deterministic frame-domain story validation, native AV planning and Base IR."""
import copy
import json
import math
import re

FPS = 24
TOTAL_FRAMES = 1464
CARD_LENGTHS = [216] * 6 + [168]
CONTEXT_FRAMES = 39


def _integer(value, label):
    if type(value) is not int:
        raise ValueError(f'{label} must be an integer frame count')
    return value


def _text(value, label, english=True):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be nonempty text')
    if english and re.search(r'[\u3400-\u9fff]', value):
        raise ValueError(f'{label} must use English execution text; put Chinese notes in review_zh')
    if english and re.search(r'\bPicture\s*\d', value, re.I):
        raise ValueError(f'{label}: Picture references are not supported by this continuation schema')


def _interval(item, lo, hi, label):
    a = _integer(item.get('start_frame'), label + '.start_frame')
    b = _integer(item.get('end_frame'), label + '.end_frame')
    if not lo <= a < b <= hi:
        raise ValueError(f'{label}: frame interval must lie inside [{lo}, {hi})')
    return a,b


def validate_story(story):
    """Return an isolated canonical dictionary; all intervals are global [start,end)."""
    if isinstance(story,str):
        try: story = json.loads(story)
        except json.JSONDecodeError as exc: raise ValueError(f'Story JSON: {exc.msg}') from exc
    if not isinstance(story,dict): raise ValueError('Story must be a JSON object')
    s = copy.deepcopy(story)
    for key,value in [('schema_version',1),('fps',FPS),('total_frames',TOTAL_FRAMES)]:
        if type(s.get(key)) is not int or s[key] != value: raise ValueError(f'{key} must be {value}')
    if s.get('mode') not in ('T2VA','I2VA'): raise ValueError('Long film supports T2VA or I2VA only')
    if s.get('voice_mode') != 'none': raise ValueError('Long film requires voice_mode none')
    for key in ('global_setting','overall_soundscape','non_diegetic_music'): _text(s.get(key),key)
    cards = s.get('cards')
    if not isinstance(cards,list) or len(cards) != 7: raise ValueError('Story requires seven editorial cards')
    cursor = 0
    for i,(card,length) in enumerate(zip(cards,CARD_LENGTHS)):
        if not isinstance(card,dict): raise ValueError(f'Card {i+1} must be an object')
        label = f'card-{i+1}'
        card.setdefault('id',label)
        if not isinstance(card['id'],str): raise ValueError(f'{label}.id must be text')
        a,b = _interval(card,cursor,cursor+length,label)
        if (a,b) != (cursor,cursor+length): raise ValueError(f'{label} must span frames {cursor}–{cursor+length}')
        for key in ('opening_state','ending_state','camera'): _text(card.get(key),f'{label}.{key}')
        if 'soundscape' in card: _text(card['soundscape'],label+'.soundscape')
        beats = card.get('beats')
        if not isinstance(beats,list) or not beats: raise ValueError(f'{label} requires continuous beats')
        beat_cursor = a
        for j,beat in enumerate(beats):
            name = f'{label}.beat-{j+1}'
            if not isinstance(beat,dict): raise ValueError(f'{name} must be an object')
            x,y = _interval(beat,a,b,name)
            if x != beat_cursor: raise ValueError(f'{name}: beats must cover the card without gaps or overlap')
            _text(beat.get('action'),name+'.action')
            if type(beat.get('indivisible',False)) is not bool: raise ValueError(f'{name}.indivisible must be boolean')
            if 'dialogue' in beat:
                d = beat['dialogue']
                if not isinstance(d,dict): raise ValueError(f'{name}.dialogue must be an object')
                _interval(d,x,y,name+'.dialogue'); _text(d.get('text'),name+'.dialogue.text',False)
            beat_cursor = y
        if beat_cursor != b: raise ValueError(f'{label}: beats must cover the entire card')
        cursor = b
    return s


def plan_chunks(story,total_vram_gb=32):
    """Native endpoints use 39+51n; reserve 34 frames (two temporal grid steps)."""
    s = validate_story(story)
    if isinstance(total_vram_gb,bool) or not isinstance(total_vram_gb,(int,float)) or not math.isfinite(total_vram_gb) or total_vram_gb <= 0:
        raise ValueError('total_vram_gb must be a positive finite number')
    endpoint_cap = 90 if total_vram_gb <= 8 else 141 if total_vram_gb <= 16 else 243
    protected = []
    for card in s['cards']:
        for beat in card['beats']:
            if beat.get('indivisible'): protected.append((beat['start_frame'],beat['end_frame'],f'indivisible action in {card["id"]}'))
            if 'dialogue' in beat:
                d = beat['dialogue']; protected.append((d['start_frame'],d['end_frame'],f'dialogue in {card["id"]}'))
    chunks = []; start = 0
    while start < TOTAL_FRAMES:
        head = CONTEXT_FRAMES if chunks else 0
        end = min(TOTAL_FRAMES,start+endpoint_cap-head)
        while True:
            conflicts = [(a,b,label) for a,b,label in protected if a < end < b]
            if not conflicts: break
            end = min(a for a,b,label in conflicts)
            if end <= start:
                raise ValueError(f'Cannot fit {conflicts[0][2]} in hardware chunk budget of {endpoint_cap-head} new frames; shorten or move that interval')
        visible = end-start
        endpoint = 39 + 51 * max(1,math.ceil((head+visible-39)/51))
        chunks.append(dict(index=len(chunks),visible_start=start,visible_end=end,visible_frames=visible,
                           context_frames=head,context_start=max(0,start-head),context_end=start,
                           local_visible_start=head,local_visible_end=head+visible,
                           endpoint_frames=endpoint,sample_frames=endpoint+34,tail_reserve_frames=34,
                           alignment_padding_frames=endpoint-head-visible,fps=FPS))
        start = end
    return chunks


def compile_prompt(story,chunk):
    """Render only intersecting beats into the three official Base IR text fields."""
    s = validate_story(story)
    start,end,head = chunk['visible_start'],chunk['visible_end'],chunk['context_frames']
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= TOTAL_FRAMES or head not in (0,39):
        raise ValueError('Invalid physical chunk interval')
    lines = [s['global_setting'], 'All frame intervals below are local, end-exclusive, at 24 fps.']
    if head: lines.append('Frames 0–39 are protected existing audiovisual context. Continue its ongoing movement and sound without restarting the action.')
    sounds = [s['overall_soundscape']]
    for card in s['cards']:
        if card['start_frame'] >= end or card['end_frame'] <= start: continue
        lines.append('Camera: '+card['camera'])
        if 'soundscape' in card: sounds.append(card['soundscape'])
        for beat in card['beats']:
            a,b = max(start,beat['start_frame']),min(end,beat['end_frame'])
            if a >= b: continue
            phase = 'Continue the ongoing portion' if a > beat['start_frame'] else 'Begin this portion'
            lines.append(f'Local frames {a-start+head}–{b-start+head}: {phase}: {beat["action"]}')
            if b < beat['end_frame']: lines.append('This is an intermediate portion; keep the action in progress, without completing or restarting it.')
            if 'dialogue' in beat:
                d = beat['dialogue']
                if d['start_frame'] < end and d['end_frame'] > start:
                    if not start <= d['start_frame'] < d['end_frame'] <= end: raise ValueError('Chunk cuts dialogue; re-plan before compiling')
                    lines.append(f'Local frames {d["start_frame"]-start+head}–{d["end_frame"]-start+head}: Say exactly: '+json.dumps(d['text'],ensure_ascii=False))
    return json.dumps(dict(integrated_multimodal_description='\n'.join(lines),overall_soundscape=' '.join(dict.fromkeys(sounds)),non_diegetic_music=s['non_diegetic_music']),ensure_ascii=False)
