"""Compare short clips via a running ComfyUI API, without importing custom nodes.

The video path belongs to the server. No model is imported into this process.
Only explicit profile choices are submitted; results never change saved defaults.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid

PROFILES = ('legacy', 'manual_baseline', 'experimental_32gb')


def build_workflow(video, profile, *, seed, fps, duration, resolution, run_id, output_prefix):
    if profile not in PROFILES:
        raise ValueError(f'Unknown VOSR2 profile: {profile}')
    if not math.isfinite(duration) or not 0 < duration <= 5:
        raise ValueError('Comparison duration must be > 0 and <= 5 seconds')
    if not math.isfinite(fps) or not 1 <= fps <= 60:
        raise ValueError('Comparison FPS must be between 1 and 60')
    frame_cap = math.floor(duration * fps)
    if frame_cap < 1:
        raise ValueError('Comparison clip must contain at least one frame')
    if resolution not in ('480p', '768p', '1080p'):
        raise ValueError('Unsupported resolution')
    return {
        '1': {'class_type': 'VHS_LoadVideoPath', 'inputs': {
            'video': video, 'force_rate': fps, 'custom_width': 0, 'custom_height': 0,
            'frame_load_cap': frame_cap, 'skip_first_frames': 0, 'select_every_nth': 1}},
        '2': {'class_type': 'MiniMaxH3VideoFinish', 'inputs': {
            'images': ['1', 0], 'resolution': resolution, 'method': 'VOSR2',
            'seed': seed, 'vosr2_profile': profile, 'benchmark_run': run_id}},
        '3': {'class_type': 'VHS_VideoCombine', 'inputs': {
            'images': ['2', 0], 'audio': ['1', 2], 'frame_rate': fps, 'loop_count': 0,
            'filename_prefix': f'{output_prefix}/{run_id}_{profile}',
            'format': 'video/h264-mp4', 'pix_fmt': 'yuv420p', 'crf': 18,
            'save_metadata': True, 'trim_to_audio': False, 'pingpong': False,
            'save_output': True}},
    }


def request_json(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = Request(url.rstrip('/') + path, data=data,
                      headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read(8000).decode('utf-8', errors='replace')
        raise RuntimeError(f'ComfyUI HTTP {exc.code}: {detail}') from exc


def summarize_history(history, *, profile, prompt_id, wall_seconds):
    outputs = history.get('outputs', {})
    reports = outputs.get('2', {}).get('vosr2_diagnostics', [])
    report = reports[0] if reports and isinstance(reports[0], dict) else None
    files = []
    for output in outputs.values():
        for key in ('gifs', 'images', 'videos'):
            for item in output.get(key, []):
                if isinstance(item, dict) and item.get('filename'):
                    files.append({field: item[field] for field in
                                  ('filename', 'subfolder', 'type', 'fullpath') if field in item})
    server_status = history.get('status', {})
    if server_status.get('status_str') == 'error':
        status = 'failed'
    elif (server_status.get('completed') and report and report.get('status') == 'completed'
          and report.get('profile') == profile and files):
        status = 'completed'
    else:
        status = 'incomplete'
    return {'profile': profile, 'prompt_id': prompt_id, 'status': status,
            'api_wall_seconds': round(wall_seconds, 6),
            'api_timing_scope': 'submission, queue wait, decode, finish, encode and history polling',
            'vosr2': report, 'output_files': files,
            'server_status': server_status}


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run_workflow(url, workflow, directory, profile, *, timeout=3600):
    """Persist evidence even on failures; timeout leaves the server job alone."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / 'workflow.json', workflow)
    start = time.perf_counter()
    prompt_id = None
    try:
        response = request_json(url, '/prompt', {'prompt': workflow, 'client_id': str(uuid.uuid4())})
        prompt_id = response.get('prompt_id')
        if not prompt_id or response.get('node_errors'):
            raise RuntimeError('ComfyUI rejected workflow: ' + json.dumps(response, ensure_ascii=False))
        _write_json(directory / 'submission.json', response)
        while time.perf_counter() - start < timeout:
            history = request_json(url, '/history/' + str(prompt_id)).get(str(prompt_id))
            if history is not None:
                _write_json(directory / 'history.json', history)
                summary = summarize_history(history, profile=profile, prompt_id=prompt_id,
                                            wall_seconds=time.perf_counter()-start)
                for item in summary['output_files']:
                    query = {k: item[k] for k in ('filename','subfolder','type') if k in item}
                    item['view_url'] = url.rstrip('/') + '/view?' + urlencode(query)
                break
            time.sleep(1)
        else:
            raise TimeoutError('Timed out; this tool has NOT cancelled the ComfyUI job. Inspect its queue before retrying.')
    except Exception as exc:
        summary = {'profile': profile, 'prompt_id': prompt_id,
                   'status': 'timeout' if isinstance(exc, TimeoutError) else 'failed',
                   'error_type': type(exc).__name__, 'error': str(exc)[:8000],
                   'api_wall_seconds': round(time.perf_counter()-start,6), 'output_files': []}
    _write_json(directory / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8188', help='Running ComfyUI base URL')
    parser.add_argument('--video', required=True, help='Absolute input video path on the ComfyUI server')
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=['legacy','manual_baseline'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fps', type=float, default=24)
    parser.add_argument('--duration', type=float, default=5, help='At most five seconds')
    parser.add_argument('--resolution', choices=['480p','768p','1080p'], default='1080p')
    parser.add_argument('--repeat', type=int, default=1, help='Rotate profile order for successive rounds')
    parser.add_argument('--output-dir', default='benchmark-results', help='Local JSON evidence directory')
    parser.add_argument('--server-output-prefix', default='vosr2-comparison')
    parser.add_argument('--timeout', type=float, default=3600)
    parser.add_argument('--dry-run', action='store_true', help='Write workflows without contacting ComfyUI')
    args = parser.parse_args(argv)
    if not 1 <= args.repeat <= 20 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('repeat must be 1..20 and timeout must be finite and > 0')
    if not 0 <= args.seed <= 0xffffffffffffffff:
        parser.error('seed must be an unsigned 64-bit integer')
    session = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    directory = Path(args.output_dir) / session
    # Validate inputs before writing files or submitting work.
    try:
        build_workflow(args.video,args.profiles[0],seed=args.seed,fps=args.fps,
                       duration=args.duration,resolution=args.resolution,run_id=session,
                       output_prefix=args.server_output_prefix)
    except ValueError as exc:
        parser.error(str(exc))
    directory.mkdir(parents=True, exist_ok=False)
    evidence = {'session':session, 'arguments':vars(args), 'runs':[],
                'quality_assessment':'Manual review required; no winner is selected.',
                'memory_note':'Free memory is device-wide. process_peak_* is a process lifetime counter, not a per-profile peak.',
                'warmup_note':'First run may include cold loading; keep all timings and compare repeated rotated rounds.'}
    if not args.dry_run:
        try:
            evidence['server_system_stats'] = request_json(args.url, '/system_stats')
            object_info = request_json(args.url, '/object_info/MiniMaxH3VideoFinish')
            optional = object_info.get('MiniMaxH3VideoFinish',{}).get('input',{}).get('optional',{})
            if 'vosr2_profile' not in optional or 'benchmark_run' not in optional:
                raise RuntimeError('Restart ComfyUI with the updated MiniMaxH3VideoFinish before benchmarking.')
        except Exception as exc:
            evidence.update(status='failed', error=str(exc))
            _write_json(directory/'summary.json',evidence)
            print(f'Preflight failed: {exc}\nEvidence: {directory}')
            return 1
    for repeat in range(args.repeat):
        offset = repeat % len(args.profiles)
        order = args.profiles[offset:] + args.profiles[:offset]
        for position, profile in enumerate(order):
            run_id = f'{session}-r{repeat+1}-{position+1}'
            run_directory = directory / f'r{repeat+1}-{position+1}-{profile}'
            workflow = build_workflow(args.video,profile,seed=args.seed,fps=args.fps,
                duration=args.duration,resolution=args.resolution,run_id=run_id,
                output_prefix=args.server_output_prefix)
            if args.dry_run:
                run_directory.mkdir()
                _write_json(run_directory/'workflow.json',workflow)
                result = {'profile':profile,'status':'planned'}
            else:
                print(f'Running {profile}, round {repeat+1}/{args.repeat}', flush=True)
                result = run_workflow(args.url,workflow,run_directory,profile,timeout=args.timeout)
                print(f"  {result['status']}: {result.get('api_wall_seconds')}s", flush=True)
            result['evidence_directory'] = str(run_directory.resolve())
            evidence['runs'].append(result)
            _write_json(directory/'summary.json',evidence)
            if result['status'] == 'timeout':
                print(f'Stopped after timeout; inspect server queue. Evidence: {directory}')
                return 1
    print(f'Evidence: {directory.resolve()}')
    return 0 if all(r['status'] in ('completed','planned') for r in evidence['runs']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
