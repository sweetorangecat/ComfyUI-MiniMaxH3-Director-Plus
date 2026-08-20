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
    from .nodes.two_stage import MiniMaxH3TwoStageSampler
    from .nodes.resolution import MiniMaxH3ResolutionPlus
    from .nodes.status import MiniMaxH3DirectorPlusStatus
    from .nodes.color_guard import MiniMaxH3ColorGuard
    from .nodes.upscale import MiniMaxH3VideoUpscale
    from .nodes.stream_output import MiniMaxH3StreamingVideoCombine
    from .nodes.vae_decode import MiniMaxH3SafeVAEDecode

    NODE_CLASS_MAPPINGS = {
        "MiniMaxH3DirectorPlus": MiniMaxH3DirectorPlus,
        "MiniMaxH3FishVoiceBridge": MiniMaxH3FishVoiceBridge,
        "MiniMaxH3DirectorPlusGuide": MiniMaxH3DirectorPlusGuide,
        "MiniMaxH3ModelRouter": MiniMaxH3ModelRouter,
        "MiniMaxH3PerformancePreset": MiniMaxH3PerformancePreset,
        "MiniMaxH3AccelerationRouter": MiniMaxH3AccelerationRouter,
        "MiniMaxH3MemoryAwareSampler": MiniMaxH3MemoryAwareSampler,
        "MiniMaxH3TwoStageSampler": MiniMaxH3TwoStageSampler,
        "MiniMaxH3SamplerRouter": MiniMaxH3SamplerRouter,
        "MiniMaxH3ResolutionPlus": MiniMaxH3ResolutionPlus,
        "MiniMaxH3DirectorPlusStatus": MiniMaxH3DirectorPlusStatus,
        "MiniMaxH3ColorGuard": MiniMaxH3ColorGuard,
        "MiniMaxH3VideoUpscale": MiniMaxH3VideoUpscale,
        "MiniMaxH3StreamingVideoCombine": MiniMaxH3StreamingVideoCombine,
        "MiniMaxH3SafeVAEDecode": MiniMaxH3SafeVAEDecode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "MiniMaxH3DirectorPlus": "MiniMax H3 导演台 Plus",
        "MiniMaxH3FishVoiceBridge": "H3 Fish S2 音色桥接",
        "MiniMaxH3DirectorPlusGuide": "H3 导演指南应用",
        "MiniMaxH3ModelRouter": "H3 模型自动选择",
        "MiniMaxH3PerformancePreset": "H3 性能预设应用",
        "MiniMaxH3AccelerationRouter": "H3 兼容加速模型路由",
        "MiniMaxH3MemoryAwareSampler": "H3 低显存采样保护",
        "MiniMaxH3TwoStageSampler": "H3 U09 图像空间二采重绘（自动旁路）",
        "MiniMaxH3ResolutionPlus": "H3 横竖比例与分辨率",
        "MiniMaxH3DirectorPlusStatus": "H3 能力与模型状态",
        "MiniMaxH3ColorGuard": "H3 曝光与色彩连续性保护",
        "MiniMaxH3VideoUpscale": "H3 低显存 CPU 分块放大",
        "MiniMaxH3StreamingVideoCombine": "H3 低显存流式放大与 MP4 输出",
        "MiniMaxH3SafeVAEDecode": "H3 安全视频 VAE 解码（GPU计算 / CPU帧缓存 FP16）",
    }

    register_routes()
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
