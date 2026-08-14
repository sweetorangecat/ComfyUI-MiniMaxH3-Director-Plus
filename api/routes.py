"""Simplified HTTP routes backed by ComfyUI's standard prompt queue."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .template import patch_template

try:
    from ..nodes.resolution import calculate_resolution
    from ..nodes.schema import PERFORMANCE_PRESETS, RequestError, normalize_request, public_schema
    from ..nodes.status import detect_capabilities
except ImportError:  # Tests import this directory as a top-level package.
    from nodes.resolution import calculate_resolution
    from nodes.schema import PERFORMANCE_PRESETS, RequestError, normalize_request, public_schema
    from nodes.status import detect_capabilities


SCHEMA_VERSION = "1.0"
ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".mov",
    ".wav", ".mp3", ".m4a", ".flac",
}
_ROUTES_REGISTERED = False


def error_payload(code, message):
    return {"ok": False, "error": {"code": code, "message": message}}


def asset_destination(input_root, filename):
    name = str(filename or "").strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name or Path(name).name != name:
        raise ValueError("文件名必须是单个安全文件名，不能包含目录")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式：{suffix or '无扩展名'}")
    safe_name = "".join(character if character.isalnum() or character in "._-() " else "_" for character in name).strip(" .")
    if not safe_name:
        raise ValueError("文件名无效")
    return Path(input_root) / "h3-director-plus" / safe_name


def prepare_generation(raw, template):
    request = normalize_request(raw)
    width, height = calculate_resolution(
        request.get("resolution_preset", "0.83 MP"),
        request.get("aspect_ratio", "16:9"),
        int(request.get("custom_width", 16)),
        int(request.get("custom_height", 9)),
    )
    request["width"] = width
    request["height"] = height
    labels = {internal: label for label, internal in PERFORMANCE_PRESETS.items() if label in PERFORMANCE_PRESETS and not label.isascii()}
    template_request = dict(request)
    template_request["performance_preset"] = labels.get(request["performance_preset"], request["performance_preset"])
    return request, patch_template(template, template_request)


async def validate_generation(raw, template, validator):
    """Validate a patched prompt without placing it in the execution queue."""
    normalized, prompt = prepare_generation(raw, template)
    prompt_id = str(uuid.uuid4())
    valid = await validator(prompt_id, prompt, None)
    return {
        "valid": bool(valid[0]),
        "resolved_backend": normalized["resolved_backend"],
        "prompt_id": prompt_id,
        "error": valid[1],
        "node_errors": valid[3],
        "prompt": prompt,
    }


def _template_path():
    return Path(__file__).resolve().parents[1] / "templates" / "u11_api.json"


def load_api_template():
    return json.loads(_template_path().read_text(encoding="utf-8"))


async def _queue_prompt(prompt, client_id=None):
    import execution
    from server import PromptServer

    server = PromptServer.instance
    prompt_id = str(uuid.uuid4())
    server.node_replace_manager.apply_replacements(prompt)
    valid = await execution.validate_prompt(prompt_id, prompt, None)
    if not valid[0]:
        return None, {"error": valid[1], "node_errors": valid[3]}
    number = server.number
    server.number += 1
    extra_data = {"create_time": int(time.time() * 1000)}
    if client_id:
        extra_data["client_id"] = client_id
    server.prompt_queue.put((number, prompt_id, prompt, extra_data, valid[2], {}))
    return {"prompt_id": prompt_id, "number": number, "node_errors": valid[3]}, None


async def schema_handler(request):
    from aiohttp import web
    return web.json_response(public_schema())


async def status_handler(request):
    from aiohttp import web
    import folder_paths
    comfy_root = Path(folder_paths.__file__).resolve().parent
    return web.json_response(detect_capabilities(comfy_root))


async def assets_handler(request):
    from aiohttp import web
    import folder_paths

    try:
        post = await request.post()
        field = post.get("asset") or post.get("file")
        if field is None or not getattr(field, "file", None):
            return web.json_response(error_payload("missing_asset", "缺少 asset 文件字段"), status=400)
        destination = asset_destination(folder_paths.get_input_directory(), post.get("filename") or field.filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as output:
            while True:
                block = field.file.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
    except ValueError as exc:
        return web.json_response(error_payload("invalid_asset", str(exc)), status=400)
    except Exception as exc:
        return web.json_response(error_payload("upload_failed", f"上传失败：{exc}"), status=400)
    return web.json_response({"ok": True, "asset": f"h3-director-plus/{destination.name}"})


async def generate_handler(request):
    from aiohttp import web

    try:
        raw = await request.json()
        normalized, prompt = prepare_generation(raw, load_api_template())
        queued, error = await _queue_prompt(prompt, raw.get("client_id"))
    except (RequestError, ValueError, json.JSONDecodeError) as exc:
        return web.json_response(error_payload("invalid_request", str(exc)), status=400)
    except FileNotFoundError:
        return web.json_response(error_payload("template_missing", "U11 API 模板尚未安装"), status=503)
    if error:
        return web.json_response({"ok": False, **error, "resolved_backend": normalized["resolved_backend"]}, status=400)
    return web.json_response({"ok": True, "resolved_backend": normalized["resolved_backend"], **queued})


async def validate_handler(request):
    from aiohttp import web
    import execution

    try:
        raw = await request.json()
        result = await validate_generation(raw, load_api_template(), execution.validate_prompt)
    except (RequestError, ValueError, json.JSONDecodeError) as exc:
        return web.json_response(error_payload("invalid_request", str(exc)), status=400)
    except FileNotFoundError:
        return web.json_response(error_payload("template_missing", "U11 API 模板尚未安装"), status=503)
    return web.json_response(result, status=200 if result["valid"] else 400)


async def job_handler(request):
    from aiohttp import web
    from server import PromptServer

    prompt_id = request.match_info.get("prompt_id", "")
    server = PromptServer.instance
    history = server.prompt_queue.get_history(prompt_id=prompt_id)
    running, pending = server.prompt_queue.get_current_queue_volatile()
    state = "completed" if history else "running" if any(item[1] == prompt_id for item in running) else "pending" if any(item[1] == prompt_id for item in pending) else "not_found"
    status = 404 if state == "not_found" else 200
    return web.json_response({"ok": status == 200, "prompt_id": prompt_id, "state": state, "history": history}, status=status)


def register_routes():
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True
    try:
        from server import PromptServer
        server = PromptServer.instance
        if server is None:
            return False
        routes = server.routes
        routes.get("/h3-director-plus/schema")(schema_handler)
        routes.get("/h3-director-plus/status")(status_handler)
        routes.post("/h3-director-plus/assets")(assets_handler)
        routes.post("/h3-director-plus/generate")(generate_handler)
        routes.post("/h3-director-plus/validate")(validate_handler)
        routes.get("/h3-director-plus/jobs/{prompt_id}")(job_handler)
    except (ImportError, AttributeError):
        return False
    _ROUTES_REGISTERED = True
    return True
