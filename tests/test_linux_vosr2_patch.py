"""Exercise patched compatibility code against an independent scalar-tile path."""
import types
import torch
import pytest

from tools.patch_linux_vosr2 import patch_engine_source, patch_nodes_source


SCALAR_LOOP = '''            for hi, wi in keys:
                he, we = hi + lt, wi + lt
                u_tile = self._dit_call(
                    lq_latent[:, :, hi:he, wi:we], z[:, :, hi:he, wi:we],
                    t_cur, t_nxt, tile_fea[(hi, wi)],
                )
                u_acc[:, :, hi:he, wi:we] += u_tile * g_weight
                w_acc[:, :, hi:he, wi:we] += g_weight
'''

SOURCE = '''class VOSREngine:
    def _flow(self, lq_latent, z, settings):
        b = lq_latent.shape[0]
        keys = [(0, 0), (0, 2), (0, 4), (2, 0), (2, 2)]
        lt = 2
        g_weight = torch.ones(1, 1, 2, 2)
        tile_fea = {k: [torch.full((b, 1, 1), float(i))] for i, k in enumerate(keys)}
        t_cur, t_nxt = torch.tensor(1.), torch.tensor(0.)
        u_acc, w_acc = torch.zeros_like(lq_latent), torch.zeros_like(lq_latent)
        for step_i in range(1):
SCALAR_LOOP
        return u_acc, w_acc
    def upscale_batch(self, images, scale, settings, seed, temporal_cache=False,
                      cache_threshold=0.003, cache_refresh=4, is_video=False):
        for i in range(len(images)):
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.calls += 1
        return images
'''.replace('SCALAR_LOOP\n', SCALAR_LOOP)


def engine(source):
    namespace = {'torch': torch}
    exec(compile(source, '<compat-fixture>', 'exec'), namespace)
    result = namespace['VOSREngine']()
    result.batches = []
    def dit(self, x, z, t, nt, features):
        self.batches.append(len(x))
        return x * 0.3 + z * 0.7 + features[0].view(len(x), 1, 1, 1)
    result._dit_call = types.MethodType(dit, result)
    return result


@pytest.mark.parametrize('frames', [1, 2])
@pytest.mark.parametrize('batch', [1, 2, 4, 8])
def test_tile_batch_preserves_tile_order_features_and_partial_tail(frames, batch):
    original, patched = engine(SOURCE), engine(patch_engine_source(SOURCE))
    gen = torch.Generator().manual_seed(42)
    x, z = [torch.rand((frames, 3, 4, 6), generator=gen) for _ in range(2)]
    expected = original._flow(x, z, {})
    actual = patched._flow(x, z, {'image_batch': batch})
    assert all(torch.equal(a, e) for a, e in zip(actual, expected))
    assert len(patched.batches) == (5 + batch - 1) // batch
    assert sum(patched.batches) == 5 * frames


def test_video_keeps_allocator_cache_but_image_cleanup_is_preserved(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: calls.append('flush'))
    value = engine(patch_engine_source(SOURCE))
    value.device = types.SimpleNamespace(type='cuda')
    value.calls = 0
    value.upscale_batch([0, 1, 2], 1, {}, 42, is_video=True)
    assert value.calls == 3 and calls == []
    value.upscale_batch([0], 1, {}, 42, is_video=False)
    assert calls == ['flush']


def test_patch_is_idempotent_and_refuses_unrecognized_engine():
    updated = patch_engine_source(SOURCE)
    assert patch_engine_source(updated) == updated
    with pytest.raises(ValueError, match='unrecognized'):
        patch_engine_source('class Unknown: pass\n')
    nodes = 'def clear_engines():\n    return "released"\nclass TESpeedVOSR2Video:\n    pass\n'
    updated_nodes = patch_nodes_source(nodes)
    assert patch_nodes_source(updated_nodes) == updated_nodes
    scope = {}
    exec(updated_nodes, scope)
    assert scope['TESpeedVOSR2Video'].H3_TILE_BATCH_SUPPORTED is True
    assert scope['TESpeedVOSR2Video'].H3_RELEASE_ENGINES() == 'released'
