import json
from pathlib import Path

import pytest

from nodes.schema import RequestError, normalize_request


def test_action_replace_is_not_a_supported_generation_mode():
    with pytest.raises(RequestError, match="不支持的生成模式"):
        normalize_request({"mode": "ACTION_REPLACE"})


def test_action_replace_adapter_and_examples_are_removed():
    assert not Path("nodes/action_replace.py").exists()
    assert not list(Path("examples").glob("*动作替换*"))

    for path in Path("examples").glob("*.json"):
        payload = path.read_text(encoding="utf-8")
        assert "ACTION_REPLACE" not in payload
        assert "Viggle" not in payload
