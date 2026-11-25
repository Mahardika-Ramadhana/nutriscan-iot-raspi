from pathlib import Path

from nutriscan.db.database import Database


def _sample_payload(scan_id: str) -> dict:
    return {
        "schema_version": "1.0",
        "scan_id": scan_id,
        "device_id": "NUTRISCAN-PI-01",
        "location": "dapur-01",
        "timestamp_utc": "2026-11-25T10:00:00Z",
        "berat_total_g": 300.0,
        "jumlah_deteksi": 1,
        "deteksi": [],
        "komponen": {"Nasi": {"luas_px": 100, "proporsi": 1.0, "estimasi_gram": 300.0}},
        "total_gizi": {
            "energi_kcal": 540.0,
            "protein_g": 9.0,
            "lemak_g": 0.9,
            "karbohidrat_g": 119.4,
        },
    }


def test_database_lifecycle_and_queue_transitions(tmp_path: Path):
    db = Database(tmp_path / "nutriscan.db")
    db.init_db()

    db.save_scan(_sample_payload("uuid-1"), image_path=tmp_path / "img1.jpg", sync_status="pending")
    db.save_scan(_sample_payload("uuid-2"), sync_status="local_only")

    stats = db.get_queue_stats()
    assert stats["pending"] == 1
    assert stats["local_only"] == 1
    assert stats["total"] == 2

    due = db.fetch_pending_scans(now=100.0)
    assert [row["scan_uuid"] for row in due] == ["uuid-1"]

    # Retry pushes next_retry_at into future
    db.mark_retry("uuid-1", error="network timeout", next_retry_at=200.0)
    assert db.fetch_pending_scans(now=150.0) == []
    assert len(db.fetch_pending_scans(now=250.0)) == 1

    # Sync marks completed
    db.mark_synced("uuid-1", synced_at="2026-11-25T10:01:00Z")
    assert db.get_queue_stats()["synced"] == 1
    assert len(db.get_recent_scans(limit=10)) == 2
    assert len(db.get_all_scans()) == 2

    db.clear_scans()
    assert db.get_queue_stats()["total"] == 0
