import json
from pathlib import Path
from nodes.longfilm import MiniMaxH3LongFilm61

def test_longfilm_node_returns_61s_plan():
    story=json.loads(Path("templates/u11_longfilm_story.json").read_text(encoding="utf-8"))
    out=MiniMaxH3LongFilm61().plan(json.dumps(story),32.0,"demo")
    plan=json.loads(out[0]); prompts=json.loads(out[1])
    assert plan["total_seconds"]==61 and plan["total_frames"]==1464
    assert sum(c["visible_frames"] for c in plan["chunks"])==1464
    assert len(prompts)==len(plan["chunks"])
