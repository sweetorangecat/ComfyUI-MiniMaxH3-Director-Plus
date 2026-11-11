"""Optional long-video entry on the existing U11 output, short branch unchanged."""
from copy import deepcopy

from .stream_output import MiniMaxH3StreamingVideoCombine
from .long_video_graph import ancestors
from .long_video_service import launch


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
            'enabled': ('BOOLEAN', {'default': False}),
            'target_seconds': ('FLOAT', {'default': 30.0, 'min': 0.05, 'step': 1.0}),
            'project_name': ('STRING', {'default': 'h3-native-av'}),
            'scene_contract': ('STRING', {'multiline': True, 'default': '{}'}),
            'stop_condition': (['target','manual'], {'default': 'target',
                'tooltip': 'target 在目标时长结束；manual 到已编排剧情末尾暂停，可增加剧情和时长继续。'}),
        }}

    RETURN_TYPES = ('MINIMAX_H3_LONG_VIDEO',)
    RETURN_NAMES = ('长视频配置',)
    FUNCTION = 'configure'
    CATEGORY = 'MiniMax H3 导演台 Plus'

    def configure(self, enabled, target_seconds, project_name, scene_contract, stop_condition='target'):
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

    def check_lazy_status(self, long_video=None, images=None, guide=None, **kwargs):
        if long_video and long_video.get('enabled'):
            return []
        needed = [key for key,value in [('images',images),('guide',guide)] if value is None]
        needed.extend(key for key in ('audio','audio_override') if key in kwargs and kwargs[key] is None)
        return needed

    def combine(self, images=None, guide=None, long_video=None, unique_id=None, **kwargs):
        if not long_video or not long_video.get('enabled'):
            return super().combine(images=images, guide=guide, **kwargs)
        prompt = kwargs.get('prompt')
        if not isinstance(prompt, dict) or str(unique_id) not in prompt:
            raise ValueError('Long-video entry requires the actual U11 prompt graph')
        state = launch(ancestors(prompt, str(unique_id)), long_video)
        import torch
        return {'ui': {'h3_long_video': [state]},
                'result': (torch.empty((0, 1, 1, 3)), state.get('output', ''))}
