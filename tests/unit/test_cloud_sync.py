from dataclasses import replace
import json
from pathlib import Path
import sys
import types
import numpy as np
import pytest
import requests

from nutriscan.ai.detector import ComponentNutrition, DetectionItem, ScanAnalysis
from nutriscan.core.config import load_config
from nutriscan.db.cloud_sync import (
    CloudSync,
    HttpRestTransport,
    MqttTransport,
    SupabaseTransport,
    TransportError,
)
from nutriscan.db.database import Database


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sample_analysis() -> ScanAnalysis:
    return ScanAnalysis(
        detection_count=1,
        detections=(DetectionItem("Nasi", 0.95, (10.0, 10.0, 90.0, 90.0)),),
        components={
            "Nasi": ComponentNutrition(
                class_name="Nasi",
                area_px=1000,
                proportion=1.0,
                estimated_grams=250.0,
                nutrition={"energi_kcal": 450.0, "protein_g": 7.5, "lemak_g": 0.75, "karbohidrat_g": 99.5},
            )
        },
        weight_total_g=250.0,
        total_nutrition={"energi_kcal": 450.0, "protein_g": 7.5, "lemak_g": 0.75, "karbohidrat_g": 99.5},
        has_occlusion=False,
        timing_ms={"total": 42.0},
    )


def _temp_cfg(tmp_path: Path):
    base = load_config(_repo_root() / "data" / "config.example.json")
    return replace(
        base,
        storage=replace(
            base.storage,
            db_path=tmp_path / "nutriscan.db",
            capture_dir=tmp_path / "captures",
            save_annotated_image=True,
        ),
    )


def test_cloud_sync_local_only_saves_db_and_image(tmp_path: Path):
    cfg = _temp_cfg(tmp_path)
    db = Database(cfg.storage.db_path)
    db.init_db()
    sync = CloudSync(cfg, db, transports=[])

    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    payload = sync.record_and_sync(_sample_analysis(), annotated_frame=frame)

    assert payload["scan_id"]
    assert payload["image_filename"].endswith("_hasil.jpg")
    assert db.get_queue_stats()["local_only"] == 1
    assert len(list(cfg.storage.capture_dir.glob("*.jpg"))) == 1


def test_cloud_sync_offline_retry_and_recovery(tmp_path: Path):
    cfg = _temp_cfg(tmp_path)
    cfg = replace(cfg, sync=replace(cfg.sync, retry_base_delay_s=5.0, max_retries=0))
    db = Database(cfg.storage.db_path)
    db.init_db()

    sent_records = []

    class FlakyTransport:
        name = "flaky"
        should_fail = True

        def send(self, payload, _image_path):
            if self.should_fail:
                raise TransportError("Wi-Fi putus")
            sent_records.append(payload["scan_id"])

    transport = FlakyTransport()
    sync = CloudSync(cfg, db, transports=[transport])

    payload = sync.record_and_sync(_sample_analysis())
    assert db.get_queue_stats()["pending"] == 1
    assert sent_records == []

    # Recover internet and flush after retry delay
    transport.should_fail = False
    res = sync.sync_pending(now=1e12)
    assert res["sent"] == 1
    assert res["failed"] == 0
    assert sent_records == [payload["scan_id"]]
    assert db.get_queue_stats()["synced"] == 1


def test_http_rest_transport_multipart_and_json(tmp_path: Path, monkeypatch):
    cfg = _temp_cfg(tmp_path)
    img = tmp_path / "plate.jpg"
    img.write_bytes(b"fakejpeg")
    monkeypatch.setenv("NUTRISCAN_HTTP_TOKEN", "tok-abc")

    captured_calls = []

    class FakeResp:
        def __init__(self, status_code=200, text="ok"):
            self.status_code = status_code
            self.text = text

    def fake_post(url, **kwargs):
        captured_calls.append((url, kwargs))
        return FakeResp(201)

    monkeypatch.setattr(requests, "post", fake_post)

    # Multipart mode
    t_multi = HttpRestTransport(replace(cfg.sync.http, enabled=True, mode="multipart", include_image=True))
    t_multi.send({"scan_id": "s1", "berat_total_g": 200.0}, img)
    assert captured_calls[0][1]["headers"]["Authorization"] == "Bearer tok-abc"
    assert "files" in captured_calls[0][1]

    # JSON mode
    t_json = HttpRestTransport(replace(cfg.sync.http, enabled=True, mode="json", include_image=True))
    t_json.send({"scan_id": "s2", "berat_total_g": 200.0}, img)
    assert "image_base64" in captured_calls[1][1]["json"]

    # Non-2xx raises TransportError
    monkeypatch.setattr(requests, "post", lambda *_a, **_kw: FakeResp(500, "err"))
    with pytest.raises(TransportError, match="500"):
        t_json.send({"scan_id": "s3"}, None)


def test_supabase_and_mqtt_transports_mocked(tmp_path: Path, monkeypatch):
    cfg = _temp_cfg(tmp_path)

    # Supabase mocked client
    upserted = []

    class FakeTable:
        def upsert(self, row, on_conflict=None):
            upserted.append((row, on_conflict))
            return types.SimpleNamespace(execute=lambda: {"ok": True})

    fake_client = types.SimpleNamespace(table=lambda _name: FakeTable())
    supa = SupabaseTransport(replace(cfg.sync.supabase, enabled=True), client=fake_client)
    supa.send(
        {
            "scan_id": "supa-1",
            "device_id": "PI-01",
            "location": "dapur",
            "timestamp_utc": "2026-11-25T10:00:00Z",
            "total_gizi": {"energi_kcal": 300.0},
        },
        None,
    )
    assert upserted[0][0]["scan_uuid"] == "supa-1"
    assert upserted[0][1] == "scan_uuid"

    # MQTT mocked module
    published = []

    class FakeMqttInfo:
        def wait_for_publish(self, timeout=None):
            return True

    class FakeMqttClient:
        def __init__(self, callback_api_version=None):
            self.ver = callback_api_version

        def username_pw_set(self, u, p):
            self.auth = (u, p)

        def connect(self, host, port, keepalive=60):
            self.target = (host, port)

        def loop_start(self):
            pass

        def publish(self, topic, payload=None, qos=0):
            published.append((topic, json.loads(payload), qos))
            return FakeMqttInfo()

        def loop_stop(self):
            pass

        def disconnect(self):
            pass

    fake_mqtt_mod = types.SimpleNamespace(
        Client=FakeMqttClient,
        CallbackAPIVersion=types.SimpleNamespace(VERSION2=2),
    )
    monkeypatch.setitem(sys.modules, "paho", types.SimpleNamespace(mqtt=types.SimpleNamespace(client=fake_mqtt_mod)))
    monkeypatch.setitem(sys.modules, "paho.mqtt", types.SimpleNamespace(client=fake_mqtt_mod))
    monkeypatch.setitem(sys.modules, "paho.mqtt.client", fake_mqtt_mod)

    mqtt_t = MqttTransport(replace(cfg.sync.mqtt, enabled=True), device_id="PI-01")
    mqtt_t.send({"scan_id": "mq-1", "berat_total_g": 123.0}, None)
    assert published[0][0] == "nutriscan/PI-01/telemetry"
    assert published[0][1]["scan_id"] == "mq-1"
