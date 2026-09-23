"""Configure the U11 example for sequential, explicitly bound dialogue."""

from copy import deepcopy


PROMPT = """subject_definitions:
橘总 <Subject 1> (S1) is the orange cat from <Picture 1>. <Audio 1> is his voice reference only.
苏小满 <Subject 2> (S2) is the young woman from <Picture 2>. <Audio 2> is her voice reference only.
阿哈 <Subject 3> (S3) is the husky from <Picture 3>. <Audio 3> is his voice reference only.
<Picture 4> is the furnished warm apartment. <Picture 5> is the unique copper bell.

summary:
Generate a 15-second live-action comedy in one continuous medium-wide shot with all three characters visible.

retention_analysis:
Preserve identities, outfits, screen positions and the single bell. Use each audio solely for its assigned speaker's timbre. Do not reproduce the reference recording's words. Only the current speaker moves their mouth; the others listen silently.

detailed_description:
00:00.000-00:01.000 Establish the living room and the silent bell on the table.
00:01.000-00:04.000 橘总 <Subject 1> (S1) uses <Audio 1>, looks at the bell and says: <d>[Chinese] 这铃铛，谁都不许碰。</d>
00:04.000-00:05.000 A silent reaction; no speaker change within a line.
00:05.000-00:08.000 苏小满 <Subject 2> (S2) uses <Audio 2>, looks at the cat and says: <d>[Chinese] 那你刚才伸手干什么？</d>
00:08.000-00:09.000 The cat withdraws his paw silently.
00:09.000-00:12.000 阿哈 <Subject 3> (S3) uses <Audio 3>, looks at the cat and says: <d>[Chinese] 他是在检查自己的规矩。</d>
00:12.000-00:15.000 All three react silently; no additional speech.

overall_soundscape:
Clear Mandarin dialogue, one speaker at a time, no overlapping voices, no off-screen voices, no barking or meowing. Quiet room tone and subtle cloth movement. No bell ring.

non_diegetic_music:
N/A"""


def configure_three_voice_example(workflow):
    result = deepcopy(workflow)
    node = next(n for n in result["nodes"] if n["type"] == "MiniMaxH3DirectorPlus")
    named = node["widgets_values_named"]
    keys = list(named)
    updates = {"mode": "REF2VA", "prompt": PROMPT, "duration": 15,
               "voice_mode": "h3_reference", "voice_gender": "auto",
               "voice_reference_name_1": "橘总", "voice_reference_name_2": "苏小满",
               "voice_reference_name_3": "阿哈"}
    for key, value in updates.items():
        node["widgets_values"][keys.index(key)] = value
        named[key] = value
    for note in result["nodes"]:
        if note.get("type") == "MarkdownNote" and note.get("title") == "导演与素材区":
            note["widgets_values"] = [
                "三人对白示例：必须在导演台上传三份真实音频。\n"
                "Audio 1 = 橘总 / S1；Audio 2 = 苏小满 / S2；Audio 3 = 阿哈 / S3。\n"
                "上传文件槽直接加载音频，外部 AUDIO 接口可以不接线。\n"
                "每份建议 5–10 秒，仅含对应角色的干净单人声，无音乐、串话。\n"
                "参考图 1–3 对应三人，4 为房间，5 为铃铛。素材未随示例附带。"
            ]
    return result


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="将 U11 示例配置为三路音色依次对白")
    parser.add_argument("workflow", type=Path)
    args = parser.parse_args()
    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    configured = configure_three_voice_example(workflow)
    args.workflow.write_text(json.dumps(configured, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
