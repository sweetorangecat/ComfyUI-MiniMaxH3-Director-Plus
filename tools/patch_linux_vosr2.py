"""Opt-in patch for the inspected TE-Speed-VOSR2 Linux compatibility shim.

No weights, tile geometry or per-frame random seeds change. Unknown source
layouts are refused. Run without --apply to validate the patch first.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

MARKER = '# H3 compatibility tile batching v1'
SCALAR_LOOP = '''            for hi, wi in keys:
                he, we = hi + lt, wi + lt
                u_tile = self._dit_call(
                    lq_latent[:, :, hi:he, wi:we], z[:, :, hi:he, wi:we],
                    t_cur, t_nxt, tile_fea[(hi, wi)],
                )
                u_acc[:, :, hi:he, wi:we] += u_tile * g_weight
                w_acc[:, :, hi:he, wi:we] += g_weight
'''
BATCH_LOOP = '''            # H3 compatibility tile batching v1
            tile_batch = max(1, min(8, int(settings.get("image_batch") or 1)))
            for tile_start in range(0, len(keys), tile_batch):
                positions = keys[tile_start:tile_start + tile_batch]
                tile_lq = torch.cat([lq_latent[:, :, hi:hi + lt, wi:wi + lt]
                                     for hi, wi in positions], dim=0)
                tile_z = torch.cat([z[:, :, hi:hi + lt, wi:wi + lt]
                                    for hi, wi in positions], dim=0)
                features = [torch.cat([tile_fea[key][layer] for key in positions], dim=0)
                            for layer in range(len(tile_fea[positions[0]]))]
                predictions = self._dit_call(tile_lq, tile_z, t_cur, t_nxt, features)
                for (hi, wi), u_tile in zip(positions, predictions.split(b, dim=0)):
                    u_acc[:, :, hi:hi + lt, wi:wi + lt] += u_tile * g_weight
                    w_acc[:, :, hi:hi + lt, wi:wi + lt] += g_weight
                del tile_lq, tile_z, features, predictions, u_tile
'''
FLUSH = '''            if self.device.type == "cuda":
                torch.cuda.empty_cache()
'''
VIDEO_FLUSH = '''            if self.device.type == "cuda" and not is_video:
                torch.cuda.empty_cache()
'''


def patch_engine_source(source):
    if MARKER in source:
        if source.count(BATCH_LOOP) != 1 or source.count(VIDEO_FLUSH) != 1:
            raise ValueError('unrecognized partially patched engine')
        return source
    tree = ast.parse(source)
    engine = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VOSREngine'), None)
    methods = {n.name for n in engine.body if isinstance(n, ast.FunctionDef)} if engine else set()
    if not {'_flow', 'upscale_batch'} <= methods or source.count(SCALAR_LOOP) != 1 or source.count(FLUSH) != 1:
        raise ValueError('unrecognized Linux compatibility engine; inspect before patching')
    result = source.replace(SCALAR_LOOP, BATCH_LOOP).replace(FLUSH, VIDEO_FLUSH)
    compile(result, '<patched-vosr-engine>', 'exec')
    return result


def patch_nodes_source(source):
    declaration = ('TESpeedVOSR2Video.H3_TILE_BATCH_SUPPORTED = True\n'
                   'TESpeedVOSR2Video.H3_RELEASE_ENGINES = staticmethod(clear_engines)')
    if declaration in source:
        return source
    tree = ast.parse(source)
    if not any(isinstance(n, ast.ClassDef) and n.name == 'TESpeedVOSR2Video' for n in tree.body):
        raise ValueError('unrecognized Linux compatibility node layer')
    has_release = any(
        isinstance(n, ast.FunctionDef) and n.name == 'clear_engines'
        or isinstance(n, ast.ImportFrom) and any((a.asname or a.name) == 'clear_engines' for a in n.names)
        for n in tree.body)
    if not has_release or 'TESpeedVOSR2Video.H3_TILE_BATCH_SUPPORTED' in source:
        raise ValueError('unrecognized or partially patched compatibility cleanup')
    result = source.rstrip() + '\n\n' + MARKER + '\n' + declaration + '\n'
    compile(result, '<patched-vosr-nodes>', 'exec')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    root = args.directory.resolve(strict=True)
    if root.name != 'TE-Speed-VOSR2-linux-compat':
        parser.error('Expected the explicitly named Linux compatibility plugin directory')
    prepared = []
    for filename, transform in [('vosr_engine.py', patch_engine_source), ('__init__.py', patch_nodes_source)]:
        path = root / filename
        original = path.read_bytes()
        updated = transform(original.decode('utf-8')).encode('utf-8')
        compile(updated, str(path), 'exec')
        prepared.append((path, original, updated))
        print(filename, 'unchanged' if updated == original else 'patch ready', hashlib.sha256(original).hexdigest())
    if not args.apply:
        return
    written = []
    try:
        for path, original, updated in prepared:
            if original == updated:
                continue
            if path.read_bytes() != original:
                raise RuntimeError(f'Source changed during preparation: {path.name}')
            backup = path.with_name(path.name + '.h3-' + hashlib.sha256(original).hexdigest()[:12] + '.bak')
            if backup.exists() and backup.read_bytes() != original:
                raise RuntimeError('Backup collision')
            if not backup.exists():
                backup.write_bytes(original)
            temp = path.with_name(path.name + '.h3-tmp')
            temp.write_bytes(updated)
            temp.replace(path)
            written.append((path, original))
    except Exception:
        for path, original in reversed(written):
            path.write_bytes(original)
        raise
    print('Applied. Restart ComfyUI only after its queue is idle.')


if __name__ == '__main__':
    main()
