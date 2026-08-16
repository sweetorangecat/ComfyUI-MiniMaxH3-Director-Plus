"""MiniMax H3 Director Plus custom nodes."""

if __package__:
    from .api import register_routes
    from .nodes.director import MiniMaxH3DirectorPlus
    from .nodes.fish import MiniMaxH3FishVoiceBridge
    from .nodes.guide import MiniMaxH3DirectorPlusGuide, MiniMaxH3ModelRouter
    from .nodes.performance import (
        MiniMaxH3AccelerationRouter,
        MiniMaxH3MemoryAwareSampler,
        MiniMaxH3PerformancePreset,
        MiniMaxH3SamplerRouter,
    )
    from .nodes.resolution import MiniMaxH3ResolutionPlus
    from .nodes.status import MiniMaxH3DirectorPlusStatus
    from .nodes.color_guard import MiniMaxH3ColorGuard
    from .nodes.upscale import MiniMaxH3VideoUpscale

    NODE_CLASS_MAPPINGS = {
        "MiniMaxH3DirectorPlus": MiniMaxH3DirectorPlus,
        "MiniMaxH3FishVoiceBridge": MiniMaxH3FishVoiceBridge,
        "MiniMaxH3DirectorPlusGuide": MiniMaxH3DirectorPlusGuide,
        "MiniMaxH3ModelRouter": MiniMaxH3ModelRouter,
        "MiniMaxH3PerformancePreset": MiniMaxH3PerformancePreset,
        "MiniMaxH3AccelerationRouter": MiniMaxH3AccelerationRouter,
        "MiniMaxH3MemoryAwareSampler": MiniMaxH3MemoryAwareSampler,
        "MiniMaxH3SamplerRouter": MiniMaxH3SamplerRouter,
        "MiniMaxH3ResolutionPlus": MiniMaxH3ResolutionPlus,
        "MiniMaxH3DirectorPlusStatus": MiniMaxH3DirectorPlusStatus,
        "MiniMaxH3ColorGuard": MiniMaxH3ColorGuard,
        "MiniMaxH3VideoUpscale": MiniMaxH3VideoUpscale,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "MiniMaxH3DirectorPlus": "MiniMax H3 导演台 Plus",
        "MiniMaxH3FishVoiceBridge": "H3 Fish S2 音色桥接",
        "MiniMaxH3DirectorPlusGuide": "H3 导演指南应用",
        "MiniMaxH3ModelRouter": "H3 模型自动选择",
        "MiniMaxH3PerformancePreset": "H3 性能预设应用",
        "MiniMaxH3AccelerationRouter": "H3 兼容加速模型路由",
        "MiniMaxH3MemoryAwareSampler": "H3 低显存采样保护",
        "MiniMaxH3ResolutionPlus": "H3 横竖比例与分辨率",
        "MiniMaxH3DirectorPlusStatus": "H3 能力与模型状态",
        "MiniMaxH3ColorGuard": "H3 曝光与色彩连续性保护",
        "MiniMaxH3VideoUpscale": "H3 低显存 CPU 分块放大",
    }

    register_routes()
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
