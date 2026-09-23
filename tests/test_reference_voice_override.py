"""Exercise the director's post-planning guard without importing ComfyUI."""
import ast
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("switch,expected", [(None, "quality_two_stage"), ("1", "quality_two_stage"), ("0", "ref_quality_native")])
def test_director_does_not_undo_default_reference_redraw(monkeypatch, switch, expected):
    if switch is None:
        monkeypatch.delenv("MMH3_REF_VOICE_TWO_STAGE", raising=False)
    else:
        monkeypatch.setenv("MMH3_REF_VOICE_TWO_STAGE", switch)
    tree = ast.parse((Path(__file__).parents[1] / "nodes/director.py").read_text(encoding="utf-8"))
    guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and "MMH3_REF_VOICE_TWO_STAGE" in ast.unparse(node.test)]
    assert len(guards) == 1
    request = {"performance_preset": "quality_two_stage", "warnings": []}
    scope = dict(os=os, voice_mode="h3_reference", request=request,
                 smart_vram=(31.4, 30.7), TWO_STAGE_PERFORMANCE_PRESETS={"quality_two_stage"})
    exec(compile(ast.Module(body=guards, type_ignores=[]), "director_guard", "exec"), scope)
    assert request["performance_preset"] == expected
