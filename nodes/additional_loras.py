"""Read only the LoRA loaders on the acceleration router's MODEL input path.

The existing DaSiWa node owns loading. Hidden PROMPT/UNIQUE_ID inputs let us
avoid adding the automatic Mystic V1 a second time when that node selects V5.
These records describe requested adapters, not verified weight application.
"""

import json
import math


MYSTIC_V5_REF = "MysticXXX_MMH3-V5-ref2va.safetensors"
MYSTIC_V5_FL = "MysticXXX_MMH3-V5.safetensors"
MYSTIC_V1 = "MysticXXX_MMH3-V1.safetensors"


def lora_basename(name):
    return str(name).replace("\\", "/").rsplit("/", 1)[-1].casefold()


def upstream_loras(prompt, unique_id):
    """Return active, literal selections from connected known LoRA loaders."""
    if not isinstance(prompt, dict) or unique_id is None:
        return []
    node = prompt.get(str(unique_id), {})
    link = node.get("inputs", {}).get("model")
    visited, stacks = set(), []
    while isinstance(link, (list, tuple)) and len(link) == 2 and link[1] == 0:
        node_id = str(link[0])
        if node_id in visited:
            break
        visited.add(node_id)
        node = prompt.get(node_id, {})
        inputs = node.get("inputs", {})
        kind = node.get("class_type")
        if kind == "DaSiWa_LTX2LoraLoader":
            raw = inputs.get("stack_data", "[]")
            if not isinstance(raw, str):
                break  # A linked dynamic string is not evidence of its value.
            try:
                rows = json.loads(raw)
            except (ValueError, TypeError):
                break
            if not isinstance(rows, list):
                break
            selected = [
                (row.get("lora"), row.get("str", 1), row.get("vs", 1))
                for row in rows if isinstance(row, dict) and row.get("on")
            ]
            model_type = inputs.get("model_type", "Basic")
        elif kind in {"LoraLoaderModelOnly", "LoraLoader"}:
            selected = [(inputs.get("lora_name"), inputs.get("strength_model", 1), 1)]
            model_type = "Basic"
        else:
            break
        stack = []
        for name, strength, multiplier in selected:
            if not isinstance(name, str) or name in {"", "None"}:
                continue
            try:
                strength = float(strength) * float(multiplier)
            except (ValueError, TypeError):
                continue
            if math.isfinite(strength) and strength != 0:
                stack.append({"name": name, "strength": strength, "model_type": model_type})
        stacks.append(stack)
        link = inputs.get("model")
    return [row for stack in reversed(stacks) for row in stack]


def mystic_v5_override(rows, backend):
    """Validate mode-specific Mystic selections before applying acceleration."""
    mystic = [row for row in rows if lora_basename(row["name"]) in {
        lora_basename(MYSTIC_V5_REF), lora_basename(MYSTIC_V5_FL), lora_basename(MYSTIC_V1),
    }]
    v5 = [row for row in mystic if lora_basename(row["name"]) != lora_basename(MYSTIC_V1)]
    if not v5:
        return None
    if len(mystic) != 1:
        raise ValueError("附加 LoRA 请只保留一项 Mystic，不能重复叠加 V1/V5 或两份 V5")
    row = v5[0]
    if row["model_type"] != "Basic":
        raise ValueError("H3 的 Mystic V5 请在原附加 LoRA 节点选择 Basic 模式")
    expected = MYSTIC_V5_REF if backend == "ref2va_model" else MYSTIC_V5_FL
    if lora_basename(row["name"]) != lora_basename(expected):
        mode = "REF2VA" if backend == "ref2va_model" else "FL2VA"
        raise ValueError(f"当前 {mode} 模型需要附加 LoRA {expected}，请切换对应版本")
    return row["name"]
