"""61-second long-film planning node with deterministic segment validation."""
from __future__ import annotations
import json
from .longfilm_plan import validate_story, plan_chunks, compile_prompt

class MiniMaxH3LongFilm61:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"story_json": ("STRING", {"multiline": True, "default": "{}"}), "total_vram_gb": ("FLOAT", {"default": 32.0, "min": 1.0, "max": 256.0}), "project_name": ("STRING", {"default": "h3-61s"})}}
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("规划JSON", "分段提示词JSON", "项目名")
    FUNCTION = "plan"
    CATEGORY = "MiniMax H3/长片"
    def plan(self, story_json, total_vram_gb, project_name):
        story = validate_story(story_json)
        chunks = plan_chunks(story, total_vram_gb)
        prompts = [json.loads(compile_prompt(story, c)) for c in chunks]
        return (json.dumps({"total_seconds": 61, "total_frames": 1464, "chunks": chunks}, ensure_ascii=False), json.dumps(prompts, ensure_ascii=False), str(project_name))

NODE_CLASS_MAPPINGS = {"MiniMaxH3LongFilm61": MiniMaxH3LongFilm61}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3LongFilm61": "MiniMax H3 61秒长片（分段续拍规划）"}
