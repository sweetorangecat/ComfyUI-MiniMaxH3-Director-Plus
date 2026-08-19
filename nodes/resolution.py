"""Aspect-aware resolution calculation for MiniMax H3."""

from __future__ import annotations

import math


ASPECTS = {
    "1:1": (1, 1),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "8:5": (8, 5),
    "5:8": (5, 8),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "21:9": (21, 9),
    "9:21": (9, 21),
}

MEGAPIXELS = {
    "0.20 MP": 0.20,
    "0.26 MP": 0.26,
    "0.30 MP": 0.30,
    "0.36 MP": 0.36,
    "0.40 MP": 0.40,
    "0.50 MP": 0.50,
    "0.52 MP": 0.52,
    "0.60 MP": 0.60,
    "0.65 MP": 0.65,
    "0.70 MP": 0.70,
    "0.80 MP": 0.80,
    "0.83 MP": 0.83,
    "0.90 MP": 0.90,
    "1.00 MP": 1.00,
    "1.05 MP": 1.05,
    "1.10 MP": 1.10,
    "1.20 MP": 1.20,
    "1.30 MP": 1.30,
    "1.35 MP": 1.35,
    "1.40 MP": 1.40,
    "1.50 MP": 1.50,
    "1.55 MP": 1.55,
    "1.60 MP": 1.60,
    "1.65 MP": 1.65,
    "1.70 MP": 1.70,
    "1.75 MP": 1.75,
    "1.80 MP": 1.80,
    "1.90 MP": 1.90,
    "2.00 MP": 2.00,
    "2.10 MP": 2.10,
    "2K QHD": 3.6864,
    "4K UHD": 8.2944,
}

EXACT_OUTPUT_TARGETS = {
    ("2K QHD", "16:9"): (2560, 1440),
    ("2K QHD", "9:16"): (1440, 2560),
    ("4K UHD", "16:9"): (3840, 2160),
    ("4K UHD", "9:16"): (2160, 3840),
}

# MiniMax H3's native ComfyUI nodes document a 768px short edge and a
# 768x1344 pixel-area cap. Larger values are output targets for a postprocess
# stage, not safe DiT sampling canvases (especially for 15-second clips).
H3_NATIVE_SHORT_EDGE = 768
H3_NATIVE_MAX_PIXELS = H3_NATIVE_SHORT_EDGE * 1344


def _snap(value, divisor=32):
    return max(divisor, int(round(value / divisor)) * divisor)


def calculate_resolution(preset, aspect, custom_width=16, custom_height=9):
    megapixels = MEGAPIXELS.get(preset)
    if megapixels is None:
        raise ValueError(f"不支持的分辨率档位：{preset}")

    if aspect == "CUSTOM":
        ratio = (custom_width, custom_height)
        if min(ratio) <= 0:
            raise ValueError("自定义比例的宽和高必须大于 0")
    else:
        ratio = ASPECTS.get(aspect)
        if ratio is None:
            raise ValueError(f"不支持的画面比例：{aspect}")

    exact_target = EXACT_OUTPUT_TARGETS.get((preset, aspect))
    if exact_target is not None:
        return exact_target

    area = megapixels * 1024 * 1024
    width = math.sqrt(area * ratio[0] / ratio[1])
    height = math.sqrt(area * ratio[1] / ratio[0])
    return _snap(width), _snap(height)


def h3_native_canvas(aspect, custom_width=16, custom_height=9):
    """Return the official H3 native canvas for an aspect ratio."""
    if aspect == "CUSTOM":
        ratio = (int(custom_width), int(custom_height))
        if min(ratio) <= 0:
            raise ValueError("自定义比例的宽和高必须大于 0")
    else:
        ratio = ASPECTS.get(aspect)
        if ratio is None:
            raise ValueError(f"不支持的画面比例：{aspect}")

    ratio_value = ratio[0] / ratio[1]
    if ratio_value >= 1.0:
        nominal_width = H3_NATIVE_SHORT_EDGE * ratio_value
        nominal_height = H3_NATIVE_SHORT_EDGE
    else:
        nominal_width = H3_NATIVE_SHORT_EDGE
        nominal_height = H3_NATIVE_SHORT_EDGE / ratio_value

    area = nominal_width * nominal_height
    if area > H3_NATIVE_MAX_PIXELS:
        scale = (H3_NATIVE_MAX_PIXELS / area) ** 0.5
        nominal_width *= scale
        nominal_height *= scale
    return _snap(nominal_width), _snap(nominal_height)


class MiniMaxH3ResolutionPlus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "分辨率档位": (list(MEGAPIXELS), {"default": "0.83 MP"}),
                "画面比例": ([*ASPECTS, "CUSTOM"], {"default": "16:9"}),
                "自定义宽比": ("INT", {"default": 16, "min": 1, "max": 8192}),
                "自定义高比": ("INT", {"default": 9, "min": 1, "max": 8192}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("宽度", "高度")
    FUNCTION = "calculate"
    CATEGORY = "MiniMax H3 导演台 Plus"

    def calculate(self, 分辨率档位, 画面比例, 自定义宽比, 自定义高比):
        return calculate_resolution(分辨率档位, 画面比例, 自定义宽比, 自定义高比)
