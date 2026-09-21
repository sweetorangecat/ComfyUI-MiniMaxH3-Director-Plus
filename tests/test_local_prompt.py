import pytest
from nodes.local_prompt import MiniMaxH3LocalPromptDraft


def test_disabled_draft_never_loads_models(monkeypatch):
    def unexpected():
        raise AssertionError('disabled node loaded a dependency')
    monkeypatch.setattr('nodes.local_prompt._registry', unexpected)
    result = MiniMaxH3LocalPromptDraft().generate(False, 'missing', 'missing', '原文 <Audio 1>', '', 128, 0)
    assert result['result'][0] == '原文 <Audio 1>'


def test_enabled_draft_explains_missing_dependency(monkeypatch):
    monkeypatch.setattr('nodes.local_prompt._registry', lambda: {})
    with pytest.raises(RuntimeError, match='ComfyUI-H3-Qwen3VL-TextGen'):
        MiniMaxH3LocalPromptDraft().generate(True, 'base', 'tail', 'test', '', 128, 0)


def test_tail_cannot_be_selected_as_encoder(monkeypatch):
    monkeypatch.setattr('nodes.local_prompt._registry', lambda: {'H3QwenVLGenerateText': object, 'CLIPLoader': object})
    with pytest.raises(ValueError, match='tail'):
        MiniMaxH3LocalPromptDraft().generate(True, 'generation_tail_50_63.safetensors', 'tail', 'test', '', 128, 0)


def test_enabled_draft_preserves_original_prompt_and_uses_tail(monkeypatch):
    calls = []
    class Loader:
        def load_clip(self, **kwargs):
            calls.append(kwargs)
            return ('CLIP',)
    class Generator:
        FUNCTION = 'generate_text'
        @classmethod
        def INPUT_TYPES(cls):
            return {'required': {'sampling': (['deterministic', 'sample'],), 'max_new_tokens': ('INT', {'default': 512})}}
        def generate_text(self, **kwargs):
            calls.append(kwargs)
            return ('草稿', 'raw', 'chat', 'system', 'report')
    monkeypatch.setattr('nodes.local_prompt._registry', lambda: {'CLIPLoader': Loader, 'H3QwenVLGenerateText': Generator})
    monkeypatch.setattr('nodes.local_prompt._require_model', lambda name: None)
    result = MiniMaxH3LocalPromptDraft().generate(True, 'base.safetensors', 'generation_tail_50_63.safetensors', '原文', '系统', 128, 42)
    assert result['result'] == ('草稿', 'report', '原文')
    assert calls[0]['type'] == 'minimax'
    assert calls[1]['tail_clip'] == {'tail_name': 'generation_tail_50_63.safetensors'}
    assert calls[1]['prompt'] == '原文'
    assert calls[1]['max_new_tokens'] == 128
    assert calls[1]['sampling'] == 'deterministic'
