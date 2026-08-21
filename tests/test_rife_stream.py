import importlib

import pytest
import torch


def _module():
    return importlib.import_module("nodes.rife_stream")


def test_rife_iterator_is_lazy_and_preserves_pairwise_order():
    rife_stream = _module()
    calls = []
    closed = []

    class FakeProcessor:
        def __init__(self, model_name):
            assert model_name == "rife_v4.26.safetensors"

        def process(self, first, second):
            calls.append((first.clone(), second.clone()))
            return (first + second) / 2

        def close(self):
            closed.append(True)

    images = torch.stack([
        torch.full((2, 3, 3), value, dtype=torch.float32)
        for value in (0.0, 0.1, 0.2, 0.3)
    ])
    frames = rife_stream.iter_rife_frames(
        images,
        processor_factory=FakeProcessor,
        scene_cut_threshold=1.0,
    )

    first = next(frames)
    assert torch.equal(first, images[0])
    assert calls == []
    midpoint = next(frames)
    assert torch.allclose(midpoint, torch.full_like(midpoint, 0.05))
    assert len(calls) == 1

    output = [first, midpoint, *list(frames)]
    values = [round(float(frame.mean()), 3) for frame in output]
    assert values == [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    assert len(calls) == 3
    assert closed == [True]


def test_rife_iterator_bypasses_real_scene_cut_without_model_inference():
    rife_stream = _module()

    class FailingProcessor:
        def __init__(self, model_name):
            assert model_name == "rife_v4.26.safetensors"

        def process(self, first, second):
            raise AssertionError("硬切不得调用 RIFE")

        def close(self):
            pass

    images = torch.stack([torch.zeros(2, 3, 3), torch.ones(2, 3, 3)])
    output = list(rife_stream.iter_rife_frames(
        images,
        processor_factory=FailingProcessor,
        scene_cut_threshold=0.1,
    ))

    assert len(output) == 3
    assert torch.equal(output[0], images[0])
    assert torch.equal(output[1], images[0])
    assert torch.equal(output[2], images[1])


def test_rife_iterator_resets_feature_cache_after_scene_cut():
    rife_stream = _module()
    events = []

    class FakeProcessor:
        def __init__(self, model_name):
            pass

        def process(self, first, second):
            events.append("process")
            return (first + second) / 2

        def reset(self):
            events.append("reset")

        def close(self):
            pass

    images = torch.stack([
        torch.zeros(2, 3, 3),
        torch.ones(2, 3, 3),
        torch.full((2, 3, 3), 0.95),
    ])

    list(rife_stream.iter_rife_frames(
        images,
        processor_factory=FakeProcessor,
        scene_cut_threshold=0.2,
    ))

    assert events == ["reset", "process"]


@pytest.mark.parametrize(("source_count", "expected"), [(0, 0), (1, 1), (4, 7)])
def test_smoothed_frame_count_matches_two_x_timeline(source_count, expected):
    assert _module().smoothed_frame_count(source_count, 2) == expected


def test_rife_dependency_probe_reports_expected_model_path(monkeypatch):
    rife_stream = _module()
    monkeypatch.setattr(rife_stream.folder_paths, "get_full_path", lambda category, name: None)

    with pytest.raises(RuntimeError, match=r"models/frame_interpolation/rife_v4\.26\.safetensors"):
        rife_stream.probe_rife_capability("rife_v4.26.safetensors")


def test_rife_close_reports_cleanup_failure_and_still_empties_cache():
    rife_stream = _module()
    emptied = []

    class FakeManagement:
        @staticmethod
        def unload_model_and_clones(patcher):
            raise RuntimeError("卸载失败")

        @staticmethod
        def soft_empty_cache():
            emptied.append(True)

    processor = rife_stream.RifeFrameProcessor.__new__(rife_stream.RifeFrameProcessor)
    processor._model_management = FakeManagement()
    processor.patcher = object()
    processor._next_features = object()
    processor._closed = False

    with pytest.raises(RuntimeError, match="RIFE 资源释放失败"):
        processor.close()

    assert emptied == [True]
    assert processor._next_features is None
