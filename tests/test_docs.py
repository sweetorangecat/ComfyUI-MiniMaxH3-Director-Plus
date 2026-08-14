from pathlib import Path

from nodes.schema import PUBLIC_API_KEYS


DOCS = Path("docs")


def test_docs_cover_required_workflows():
    usage = (DOCS / "使用说明.md").read_text(encoding="utf-8")
    for text in ("I2VA", "FL2VA", "REF2VA", "H3 原生音色参考", "Fish S2", "提示词约束", "硬首尾帧"):
        assert text in usage


def test_api_docs_cover_every_public_key():
    api = (DOCS / "API说明.md").read_text(encoding="utf-8")
    for key in PUBLIC_API_KEYS:
        assert f"`{key}`" in api


def test_docs_explain_runtime_status_and_output_retrieval():
    api = (DOCS / "API说明.md").read_text(encoding="utf-8")
    troubleshooting = (DOCS / "故障排查.md").read_text(encoding="utf-8")
    assert "/h3-director-plus/status" in api
    assert "history" in api
    assert "模型未就绪" in troubleshooting

