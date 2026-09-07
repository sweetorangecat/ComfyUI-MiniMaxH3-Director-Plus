"""Dedicated 61-second AV continuation output and disk checkpoint orchestration."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from . import longfilm_media as media
from .longfilm_plan import validate_story, plan_chunks, compile_prompt

INFINITE_NODES = ('H3ContinuousStartV14', 'H3ContinuousContinueV14', 'H3ContinuousAnalyzeHandoverV14')


def require_capabilities(registry=None):
    if registry is None:
        import nodes
        registry = getattr(nodes, 'NODE_CLASS_MAPPINGS', {})
    missing = [name for name in INFINITE_NODES if name not in registry]
    if missing:
        raise RuntimeError('61 秒长片需要 Herrgotts H3 Infinite v1.4；缺少节点：' + ', '.join(missing) + '。请安装依赖并重启 ComfyUI。')
    return {name: registry[name] for name in INFINITE_NODES}


def content_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def checkpoint_valid(project, record, fingerprint):
    if not isinstance(record, dict) or record.get('fingerprint') != fingerprint:
        return False
    artifacts = record.get('artifacts')
    if not isinstance(artifacts, dict) or not {'video', 'latent'} <= artifacts.keys():
        return False
    root = Path(project).resolve()
    try:
        for artifact in artifacts.values():
            name = artifact['name']
            if not isinstance(name, str) or Path(name).name != name:
                return False
            path = root / name
            if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
                return False
            if media.sha256_file(path) != artifact['sha256']:
                return False
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return True


class MiniMaxH3LongFilm61:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',), 'clip': ('CLIP',), 'video_vae': ('VAE',), 'audio_vae': ('VAE',),
            'project_name': ('STRING', {'default': 'h3-61s'}),
            'story_json': ('STRING', {'multiline': True, 'default': '{}'}),
            'target': (['1080p', '2K'], {'default': '1080p'}),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
        }, 'optional': {'first_image': ('IMAGE',)}, 'hidden': {'prompt': 'PROMPT'}}
    RETURN_TYPES = ('STRING', 'STRING', 'STRING')
    RETURN_NAMES = ('成片路径', '运行状态JSON', '项目名')
    OUTPUT_NODE = True
    FUNCTION = 'generate'
    CATEGORY = 'MiniMax H3/长片'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Disk resume and project-local cancellation are evaluated on every queue.
        return float('nan')

    def plan(self, story_json, total_vram_gb, project_name):
        story = validate_story(story_json)
        chunks = plan_chunks(story, total_vram_gb)
        prompts = [json.loads(compile_prompt(story, c)) for c in chunks]
        return (json.dumps({'total_seconds': 61, 'total_frames': 1464, 'chunks': chunks}, ensure_ascii=False), json.dumps(prompts, ensure_ascii=False), str(project_name))

    def generate(self, model, clip, video_vae, audio_vae, project_name, story_json, target, seed, first_image=None, prompt=None):
        story = validate_story(story_json)
        if target not in ('1080p', '2K'):
            raise ValueError('target must be 1080p or 2K')
        if story['mode'] == 'I2VA' and first_image is None:
            raise ValueError('I2VA 长片需要连接首帧图片')
        if story['mode'] == 'T2VA' and first_image is not None:
            raise ValueError('T2VA 长片不接受首帧；请在方案中选择 I2VA')
        capabilities = require_capabilities()
        try:
            from .longfilm_backend import LongFilmBackend
        except ImportError as exc:
            raise RuntimeError('61 秒运行后端尚未安装完整，无法开始生成。') from exc
        backend = LongFilmBackend(model=model, clip=clip, video_vae=video_vae, audio_vae=audio_vae,
                                  capabilities=capabilities, first_image=first_image, prompt=prompt)
        import folder_paths
        project = media.safe_project_dir(folder_paths.get_output_directory(), project_name)
        result = run_project(project, story, backend, target, seed)
        return (str(result['output']), json.dumps(result, ensure_ascii=False), str(project_name))


NODE_CLASS_MAPPINGS = {'MiniMaxH3LongFilm61': MiniMaxH3LongFilm61}
NODE_DISPLAY_NAME_MAPPINGS = {'MiniMaxH3LongFilm61': 'MiniMax H3 61秒长片（分段续拍）'}


def run_project(project, story, backend, target, seed):
    """Render one chunk at a time, storing clean AV and bounded native video.

    Backend owns GPU policy and target export. Its generation_config must identify
    model/CLIP/VAE assets, parameters and initial-image content. Target/SR belong
    only in export_config so changing delivery resolution does not resample H3.
    """
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    state_path = project / 'state.json'
    cancel_path = project / 'cancel.request'
    lock_path = project / 'run.lock'
    try:
        fd = lock_path.open('x', encoding='utf-8')
        fd.write('active')
        fd.close()
    except FileExistsError as exc:
        raise RuntimeError('61 秒长片项目 already running') from exc
    story = validate_story(story)
    chunks = plan_chunks(story, backend.total_vram_gb)
    generation = content_fingerprint({'story': story, 'chunks': chunks, 'seed': seed,
                                      'config': backend.generation_config})
    try:
        previous_state = json.loads(state_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        previous_state = {}
    prior = previous_state.get('chunks', []) if previous_state.get('generation') == generation else []
    state = {'schema_version': 1, 'generation': generation, 'status': 'running',
             'total_frames': 1464, 'completed_frames': 0, 'chunks': [], 'output': ''}
    invalid_chain = False
    previous_latent = handover = None
    prefix = generation
    exports = []
    try:
        media.atomic_write_json(state_path, state)
        for chunk in chunks:
            if cancel_path.exists():
                cancel_path.unlink(missing_ok=True)
                state['status'] = 'cancelled'
                media.atomic_write_json(state_path, state)
                return state
            index = chunk['index']
            prompt = compile_prompt(story, chunk)
            fingerprint = content_fingerprint({'generation': generation, 'prefix': prefix, 'chunk': chunk, 'prompt': prompt})
            old = prior[index] if index < len(prior) else None
            valid = not invalid_chain and checkpoint_valid(project, old, fingerprint)
            video_path = project / f'chunk-{index:04d}.mkv'
            latent_path = project / f'chunk-{index:04d}.safetensors'
            if valid:
                record = old
                video_path = project / record['artifacts']['video']['name']
                latent_path = project / record['artifacts']['latent']['name']
            else:
                invalid_chain = True
                result = backend.render_chunk(chunk=chunk, prompt=prompt, seed=(int(seed) + index) % (2**64),
                                              previous_latent=previous_latent, handover=handover)
                if not isinstance(result.get('handover'), dict) or not result['handover'].get('available'):
                    raise ValueError(f'Chunk {index + 1} has no safe audiovisual handover')
                media.save_av_latent(latent_path, result['latent'])
                media.encode_segment(video_path, result['images'], result['audio'], fps=24)
                record = {'fingerprint': fingerprint, 'handover': result['handover'], 'artifacts': {
                    'video': {'name': video_path.name, 'sha256': media.sha256_file(video_path)},
                    'latent': {'name': latent_path.name, 'sha256': media.sha256_file(latent_path)},
                }}
                del result
            state['chunks'].append(record)
            state['completed_frames'] = chunk['visible_end']
            media.atomic_write_json(state_path, state)
            # Load only the immediately preceding native clean x0; RGB is never accumulated.
            previous_latent = media.load_av_latent(latent_path)
            handover = record['handover']
            prefix = content_fingerprint({'fingerprint': fingerprint, 'artifacts': record['artifacts'], 'handover': handover})
            export_key = content_fingerprint({'video': record['artifacts']['video']['sha256'], 'target': target,
                                               'config': getattr(backend, 'export_config', {})})
            export_path = project / f'export-{index:04d}-{export_key[:16]}.mkv'
            export_record = record.get('exports', {}).get(export_key)
            if not (export_record and export_path.is_file() and media.sha256_file(export_path) == export_record['sha256']):
                backend.export_segment(video_path, export_path, target)
                record.setdefault('exports', {})[export_key] = {'name': export_path.name, 'sha256': media.sha256_file(export_path)}
                media.atomic_write_json(state_path, state)
            exports.append(export_path)
        if cancel_path.exists():
            cancel_path.unlink(missing_ok=True)
            state['status'] = 'cancelled'
        else:
            final_path = project / f'film-61s-{target}.mkv'
            state['media'] = media.join_segments(exports, final_path, expected_frames=1464, fps=24)
            state['output'] = str(final_path)
            state['status'] = 'complete'
        media.atomic_write_json(state_path, state)
        return state
    except Exception as exc:
        state['status'] = 'failed'
        state['error'] = str(exc)
        media.atomic_write_json(state_path, state)
        raise
    finally:
        lock_path.unlink(missing_ok=True)

