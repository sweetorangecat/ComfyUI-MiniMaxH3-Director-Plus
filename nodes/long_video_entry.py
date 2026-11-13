"""The original Director's duration selects short output or native AV continuation."""
from copy import deepcopy
import math

from .stream_output import MiniMaxH3StreamingVideoCombine
from .long_video_graph import ancestors, adapt_long_voice, original_voice_settings
from .long_video_service import launch
from .long_video_preflight import preflight
from .long_video_store import fingerprint


def contract_from_prompt(*args, **kwargs):
    # The text compiler is needed only for automatic long videos.
    from .long_video_prompt import contract_from_prompt as compile_prompt
    return compile_prompt(*args, **kwargs)


def _director_target(prompt, unique_id):
    if not isinstance(prompt, dict) or str(unique_id) not in prompt:
        return None
    graph = ancestors(prompt, str(unique_id))
    choices = [(key, node['inputs']) for key, node in graph.items()
               if node.get('class_type') == 'MiniMaxH3DirectorPlus']
    if not choices:
        return None
    if len(choices) != 1:
        raise ValueError('原 U11 输出需要连接一个导演台，无法判断多个导演台的总时长。')
    key, settings = choices[0]
    seconds = settings.get('duration', 5)
    if isinstance(seconds, list):
        # Old custom short graphs may supply an INT link. Preserve their lazy
        # path; automatic long targets use the original editable duration.
        return None
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds < 4 or int(seconds) != seconds):
        raise ValueError('视频总时长请输入至少 4 秒的整数。')
    return graph, key, int(seconds)


def _automatic_recipe(target):
    graph, director, seconds = target
    settings = graph[director]['inputs']
    adapt_long_voice(settings, graph[director].setdefault('_meta', {}))
    if settings.get('community_lora_profile') == 'combat':
        from .schema import PERFORMANCE_PRESETS
        from .sampling_experiments import validate_sampling_options
        selected = dict(settings, performance_preset=PERFORMANCE_PRESETS.get(
            settings.get('performance_preset'), settings.get('performance_preset')))
        settings.update(validate_sampling_options(selected))
    if settings.get('voice_mode', 'none') not in {'none', 'h3_reference'}:
        raise ValueError('超过 15 秒的原生续拍目前支持“不使用音色”或“H3 原生音色参考”；'
                         '请在原导演台选择对应音色方式。')
    # Hidden uploads from an inactive mode must not become new references just
    # because the long-video engine uses Ref2VA. Match the short-mode policy.
    mode = settings.get('mode', 'REF2VA')
    keep = {'REF2VA': {'first_image', 'last_image', *(f'reference_image_{i}' for i in range(1, 10))},
            'FL2VA': {'first_image', 'last_image'}, 'I2VA': {'first_image'},
            'L2VA': {'last_image'}, 'T2VA': set()}.get(mode)
    if keep is None:
        raise ValueError('请在原导演台选择有效的生成模式。')
    from .long_video_references import PICTURE_SLOTS
    for slot in PICTURE_SLOTS:
        if slot not in keep:
            settings.pop(slot, None)
            settings.pop(slot + '_file', None)
    if seconds > 15:
        from .long_video_references import snapshot_bindings
        if len(snapshot_bindings(settings, graph)['pictures']) > 8:
            raise ValueError('Native continuity needs one existing picture slot: use at most 8 static images '
                             'so the generated parent can occupy the ninth continuity slot')
    graph[director].setdefault('_meta', {})['h3_long_video_entry'] = 'director_duration'
    _prepare_long_pictures(settings, graph)
    project = 'u11-' + fingerprint({'graph': graph, 'seconds': seconds})[:24]
    if settings.get('voice_mode', 'none') == 'none':
        from .prompting import without_voice_references
        settings['prompt'] = without_voice_references(settings.get('prompt', ''))
    script = deepcopy(contract_from_prompt(settings.get('prompt', ''), seconds, settings=settings))
    if seconds > 15:
        script['parent_reference'] = {'mode': 'append', 'frame_policy': 'delivery_last'}
    original_voice = original_voice_settings(graph, director)
    if original_voice.get('voice_mode') == 'fish_lock':
        from .long_video_audio import recipe_for_window
        try:
            script['target_audio'] = recipe_for_window(script, {'new_start': 0, 'new_end': seconds * 24}, original_voice)
        except ValueError as exc:
            raise ValueError('Fish 音色长视频需要完整目标音频：' + str(exc)) from exc
    settings.update(mode='REF2VA', performance_preset='质量优先二采样',
                    sampling_method='selflift', postprocess_mode='vosr2',
                    vosr2_profile='manual_baseline', face_refine_mode='off',
                    motion_smoothing='off')
    config = {'enabled': True, 'target_seconds': seconds, 'project_name': project,
              'scene_contract': script, 'stop_condition': 'target',
              'entry_mode': 'director_duration'}
    # Ordinary queueing performs the same static checks as the old separate
    # button, before asset hashing, model loading or any internal queue job.
    preflight(graph, config)
    return graph, config


def _prepare_long_pictures(settings, graph):
    """Keep flattened Picture identities while making optional endpoints legal."""
    from .long_video_references import PICTURE_SLOTS, snapshot_bindings
    from .long_video_graph import _picture_preprocessing
    records = snapshot_bindings(settings, graph)['pictures']
    compact = any(record['slot'] in {'first_image','last_image'} for record in records)
    for index, record in enumerate(records):
        before = _picture_preprocessing(settings.get('prompt',''), record['slot'],
                                         record['preprocessing_source'], record['id'])
        if before['dropped']:
            raise ValueError(f'Picture {record["id"]} 是故事板文件：请在原提示词中明确它的故事板职责、'
                             '阅读顺序及动作时间段，避免把未声明的规划图送入视觉条件。')
        if compact:
            after = _picture_preprocessing(settings.get('prompt',''), PICTURE_SLOTS[index],
                                            record['preprocessing_source'], record['id'])
            if before != after:
                raise ValueError(f'Picture {record["id"]} 的复合图处理会因长片参考位置变化而改变；'
                                 '请使用已确认的单张参考画面。')
    if compact:
        for field in PICTURE_SLOTS:
            settings.pop(field,None); settings.pop(field+'_file',None)
        for field,record in zip(PICTURE_SLOTS,records):
            binding=record['binding']
            settings[field+('_file' if 'file' in binding else '')]=deepcopy(next(iter(binding.values())))
            if 'link' in binding and record['preprocessing_source']:
                settings[field+'_file']=record['preprocessing_source']


class MiniMaxH3LongVideoAssetName:
    """Make frozen asset identity part of Comfy's loader ancestry signature."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'value': ('STRING',), 'revision': ('STRING',)}}

    # File selectors may be COMBO or STRING sockets. This internal node always
    # returns the same string and supports either declared selector type.
    RETURN_TYPES = ('*',)
    FUNCTION = 'resolve'
    CATEGORY = 'MiniMax H3 导演台 Plus/原生长视频'

    def resolve(self, value, revision):
        return (value,)


class MiniMaxH3LongVideoConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'enabled': ('BOOLEAN', {'default': False, 'tooltip': '开启后使用原 U11 长片面板；关闭时保持原短片路径。'}),
            'target_seconds': ('FLOAT', {'default': 30.0, 'min': 0.05, 'step': 1.0,
                'tooltip': '总时长（秒）：30/60/90 为便捷选择，也可输入其他正的有限秒数；按 24 FPS 舍入，不缩放剧情。'}),
            'project_name': ('STRING', {'default': 'h3-native-av', 'tooltip': '同一项目名从已保存检查点恢复。'}),
            'scene_contract': ('STRING', {'multiline': True, 'default': '{}',
                'tooltip': '导入 H3 技能的 U11 单语 scene_contract JSON；默认 {} 不是剧情。先预检，再普通排队。'}),
            'stop_condition': (['target','manual'], {'default': 'target',
                'tooltip': 'target 在目标时长结束；manual 到已编排剧情末尾暂停，可增加剧情和时长继续。'}),
        }}

    RETURN_TYPES = ('MINIMAX_H3_LONG_VIDEO',)
    RETURN_NAMES = ('长视频配置',)
    FUNCTION = 'configure'
    CATEGORY = 'MiniMax H3 导演台 Plus'

    def configure(self, enabled, target_seconds, project_name, scene_contract, stop_condition='target'):
        if enabled:
            from .long_video_plan import target_frame_count
            from .long_video_preflight import read_contract
            read_contract(scene_contract, target_frame_count(target_seconds))
        return ({'enabled': bool(enabled), 'target_seconds': target_seconds,
                 'project_name': project_name, 'scene_contract': scene_contract,
                 'stop_condition': stop_condition},)


class MiniMaxH3U11Output(MiniMaxH3StreamingVideoCombine):
    @classmethod
    def INPUT_TYPES(cls):
        fields = deepcopy(super().INPUT_TYPES())
        for section, key in [('required','images'),('required','guide'),
                             ('optional','audio'),('optional','audio_override')]:
            type_info = fields[section][key]
            fields[section][key] = (type_info[0], {**(type_info[1] if len(type_info)>1 else {}), 'lazy': True})
        fields['optional']['long_video'] = ('MINIMAX_H3_LONG_VIDEO',)
        fields['hidden']['unique_id'] = 'UNIQUE_ID'
        return fields

    def check_lazy_status(self, long_video=None, images=None, guide=None,
                          prompt=None, unique_id=None, **kwargs):
        if long_video and long_video.get('enabled'):
            return []
        target = _director_target(prompt, unique_id)
        if target and target[2] > 15:
            return []
        needed = [key for key,value in [('images',images),('guide',guide)] if value is None]
        needed.extend(key for key in ('audio','audio_override') if key in kwargs and kwargs[key] is None)
        return needed

    def combine(self, images=None, guide=None, long_video=None, unique_id=None, **kwargs):
        prompt = kwargs.get('prompt')
        explicit = bool(long_video and long_video.get('enabled'))
        target = None if explicit else _director_target(prompt, unique_id)
        if not explicit and not (target and target[2] > 15):
            return super().combine(images=images, guide=guide, **kwargs)
        if not isinstance(prompt, dict) or str(unique_id) not in prompt:
            raise ValueError('Long-video entry requires the actual U11 prompt graph')
        graph, config = ((ancestors(prompt, str(unique_id)), long_video) if explicit
                         else _automatic_recipe(target))
        state = launch(graph, config)
        state = dict(state)
        directors = [key for key, value in graph.items() if value.get('class_type') == 'MiniMaxH3DirectorPlus']
        if len(directors) == 1:
            state['director_id'] = directors[0]
        import torch
        return {'ui': {'h3_long_video': [state]},
                'result': (torch.empty((0, 1, 1, 3)), state.get('output', ''))}
