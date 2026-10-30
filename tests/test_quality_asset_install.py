import hashlib
from urllib.error import HTTPError

import pytest

from tools import install_u24_assets as installer


def test_verified_download_is_rejected_before_replacing_existing_model(tmp_path, monkeypatch):
    destination = tmp_path / "models" / "loras" / "test.safetensors"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing valid model")
    asset = dict(name="test.safetensors", category="loras", url="https://example.invalid/model",
                 sha256=hashlib.sha256(b"correct new weights").hexdigest())
    monkeypatch.setattr(installer, "_download", lambda url, path: path.write_bytes(b"login page"))
    with pytest.raises(RuntimeError, match="SHA256"):
        installer.install(asset, tmp_path, force=True)
    assert destination.read_bytes() == b"existing valid model"
    assert not destination.with_suffix(".safetensors.partial").exists()


def test_cached_model_must_match_the_author_hash(tmp_path, monkeypatch):
    destination = tmp_path / "models" / "loras" / "test.safetensors"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"wrong model")
    asset = dict(name="test.safetensors", category="loras", url="https://example.invalid/model",
                 sha256=hashlib.sha256(b"correct weights").hexdigest())
    monkeypatch.setattr(installer, "_download", lambda *args: pytest.fail("do not overwrite cached files"))
    with pytest.raises(RuntimeError, match="SHA256"):
        installer.install(asset, tmp_path, force=False)


def test_valid_model_is_installed_in_the_existing_minimax_folder(tmp_path, monkeypatch):
    payload = b"verified model weights"
    digest = hashlib.sha256(payload).hexdigest()
    asset = dict(name="minimax/test.safetensors", category="loras", url="https://example.invalid/model", sha256=digest)
    monkeypatch.setattr(installer, "_download", lambda url, path: path.write_bytes(payload))
    path, actual = installer.install(asset, tmp_path, force=False)
    assert path == tmp_path / "models" / "loras" / "minimax" / "test.safetensors"
    assert path.read_bytes() == payload and actual == digest


def test_author_download_403_uses_pinned_mirror_with_same_hash(tmp_path, monkeypatch):
    payload = b"author weights distributed by mirror"
    asset = dict(name="test.safetensors", category="loras", url="https://author.invalid/model",
                 mirrors=("https://mirror.invalid/pinned/model",), sha256=hashlib.sha256(payload).hexdigest())
    calls = []

    def download(url, path):
        calls.append(url)
        if url == asset["url"]:
            raise HTTPError(url, 403, "Forbidden", None, None)
        path.write_bytes(payload)

    monkeypatch.setattr(installer, "_download", download)
    path, digest = installer.install(asset, tmp_path, force=False)
    assert calls == [asset["url"], *asset["mirrors"]]
    assert path.read_bytes() == payload and digest == asset["sha256"]


def test_wrong_hash_from_mirror_cannot_replace_existing_model(tmp_path, monkeypatch):
    destination = tmp_path / "models" / "loras" / "test.safetensors"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"original")
    asset = dict(name="test.safetensors", category="loras", url="https://author.invalid/model",
                 mirrors=("https://mirror.invalid/pinned/model",), sha256=hashlib.sha256(b"author weights").hexdigest())

    def download(url, path):
        if url == asset["url"]:
            raise HTTPError(url, 403, "Forbidden", None, None)
        path.write_bytes(b"wrong mirror data")

    monkeypatch.setattr(installer, "_download", download)
    with pytest.raises(RuntimeError, match="SHA256"):
        installer.install(asset, tmp_path, force=True)
    assert destination.read_bytes() == b"original"
    assert not destination.with_suffix(".safetensors.partial").exists()
