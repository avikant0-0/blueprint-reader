import importlib

from fastapi.testclient import TestClient


def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("BR_JOBS_DIR", str(tmp_path / "jobs"))
    for var in ("BR_DETECTOR_WEIGHTS", "BR_CLASSIFIER_WEIGHTS", "VLM_API_KEY", "VLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    import api.main as main
    return TestClient(importlib.reload(main).app)


def test_analyze_roundtrip(drawing_set, tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    assert client.get("/api/health").json()["methods"]["schedule_extractor"] == "text_table_baseline"
    with open(drawing_set, "rb") as fh:
        res = client.post("/api/analyze", files={"file": ("set.pdf", fh, "application/pdf")})
    assert res.status_code == 200
    body = res.json()
    assert body["pages_processed"] == 4 and len(body["schedule"]["rows"]) == 4
    assert client.get(body["pages"][0]["image_url"]).status_code == 200
    assert "Blueprint Reader" in client.get("/").text


def test_rejects_non_pdf(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    assert client.post("/api/analyze", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 400
    assert client.post("/api/analyze", files={"file": ("a.pdf", b"not a pdf", "application/pdf")}).status_code == 400
