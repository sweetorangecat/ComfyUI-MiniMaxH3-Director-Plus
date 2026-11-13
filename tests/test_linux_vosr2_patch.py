"""Exercise patched compatibility code against an independent scalar-tile path."""
import types
import torch
import torch.nn.functional as F
import pytest

from tools.patch_linux_vosr2 import main, patch_engine_source, patch_nodes_source


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
    def _venc_features(self, lq_bchw):
        with torch.no_grad():
            raw = (0.5 * lq_bchw + 0.5) * 255.0
            x = F.interpolate(raw, size=(DINOV2_SIZE, DINOV2_SIZE), mode="bicubic",
                              align_corners=False).clamp(0.0, 1.0)
            mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            x = ((x - mean) / std).to(self.dtype)
            return [self.venc.forward_intermediate_layer(x, layer) for layer in self.layer_list]
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
    namespace = {'torch': torch, 'F': F, 'DINOV2_SIZE': 8,
                 'IMAGENET_MEAN': (0.485, 0.456, 0.406),
                 'IMAGENET_STD': (0.229, 0.224, 0.225)}
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


@pytest.mark.parametrize('already_batched', [False, True])
def test_dino_preprocessing_keeps_rgb_variation_instead_of_clipping_to_white(already_batched):
    # Independent expected RGB / ImageNet contract from cswry/VOSR
    # preprocess_raw_image: raw255 /255, resize/clamp01, then normalize.
    source = SOURCE
    if already_batched:
        from tools.patch_linux_vosr2 import BATCH_LOOP, VIDEO_FLUSH, FLUSH
        source = source.replace(SCALAR_LOOP, BATCH_LOOP).replace(FLUSH, VIDEO_FLUSH)
    value = engine(patch_engine_source(source))
    value.dtype, value.layer_list = torch.float32, [17]
    captured = []
    value.venc = types.SimpleNamespace(
        forward_intermediate_layer=lambda image, layer: captured.append(image.clone()) or image)
    rgb = torch.tensor([0., 1./255, .1, .3, .5, .7, .9, 1.]).view(1, 1, 2, 4).expand(1, 3, 2, 4)
    value._venc_features(rgb * 2 - 1)
    resized = F.interpolate(rgb, size=(8, 8), mode='bicubic', align_corners=False).clamp(0, 1)
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    torch.testing.assert_close(captured[0], (resized - mean) / std, rtol=1e-6, atol=1e-6)


def test_patch_refuses_unknown_dino_preprocessing_even_if_batch_loop_is_recognized():
    source = SOURCE.replace('raw = (0.5 * lq_bchw + 0.5) * 255.0', 'raw = lq_bchw')
    with pytest.raises(ValueError, match='DINO'):
        patch_engine_source(source)


def test_apply_repairs_previously_batched_engine_and_preserves_original_backup(tmp_path):
    from tools.patch_linux_vosr2 import BATCH_LOOP, VIDEO_FLUSH, FLUSH
    root = tmp_path / 'TE-Speed-VOSR2-linux-compat'
    root.mkdir()
    path = root / 'vosr_engine.py'
    original = SOURCE.replace(SCALAR_LOOP, BATCH_LOOP).replace(FLUSH, VIDEO_FLUSH)
    path.write_bytes(original.encode('utf-8'))
    nodes = root / '__init__.py'
    nodes.write_bytes(b'def clear_engines():\n    pass\nclass TESpeedVOSR2Video:\n    pass\n')
    main([str(root)])
    assert path.read_text(encoding='utf-8') == original  # Dry run is read-only.
    assert not list(root.glob('*.bak'))
    main([str(root), '--apply'])
    actual = path.read_bytes()
    assert actual.decode('utf-8') == patch_engine_source(original)
    backups = sorted(root.glob('*.bak'))
    assert len(backups) == 2
    original_bytes = original.encode('utf-8')
    assert any(p.read_bytes() == original_bytes for p in backups)
    main([str(root), '--apply'])
    assert path.read_bytes() == actual and sorted(root.glob('*.bak')) == backups


def test_unknown_features_abort_before_any_files_are_written(tmp_path):
    root = tmp_path / 'TE-Speed-VOSR2-linux-compat'
    root.mkdir()
    path = root / 'vosr_engine.py'
    path.write_bytes(SOURCE.replace('raw = (0.5 * lq_bchw + 0.5) * 255.0', 'raw = lq_bchw').encode('utf-8'))
    nodes = root / '__init__.py'
    nodes.write_bytes(b'def clear_engines():\n    pass\nclass TESpeedVOSR2Video:\n    pass\n')
    original = {p: p.read_bytes() for p in (path, nodes)}
    with pytest.raises(ValueError, match='DINO'):
        main([str(root), '--apply'])
    assert all(p.read_bytes() == data for p, data in original.items())
    assert not list(root.glob('*.bak'))
