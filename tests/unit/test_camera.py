from pathlib import Path
import sys
import types
import numpy as np
from PIL import Image
import pytest

from nutriscan.core.config import CameraConfig
from nutriscan.hardware.camera import (
    CameraError,
    CameraManager,
    MockCameraBackend,
    OpenCVCameraBackend,
)


def _cam_cfg(**overrides) -> CameraConfig:
    defaults = dict(
        backend="mock",
        camera_ids=(0,),
        width=160,
        height=120,
        use_mjpeg=True,
        warmup_s=0.0,
        flush_frames=2,
        mock_image_path=None,
    )
    defaults.update(overrides)
    return CameraConfig(**defaults)


def test_mock_camera_synthetic_and_file(tmp_path: Path):
    mgr = CameraManager(_cam_cfg())
    with mgr:
        frame = mgr.capture_frame()
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8

    sample_path = tmp_path / "sample.jpg"
    Image.new("RGB", (80, 60), color=(255, 0, 0)).save(sample_path)
    file_mgr = CameraManager(_cam_cfg(mock_image_path=sample_path))
    with file_mgr:
        loaded = file_mgr.capture_frame()
        assert loaded.shape == (60, 80, 3)
        # RGB red (255,0,0) -> BGR (0,0,255)
        assert loaded[0, 0, 2] > 200
        assert loaded[0, 0, 0] < 30


def test_mock_camera_missing_file_raises(tmp_path: Path):
    backend = MockCameraBackend(64, 64, tmp_path / "missing.jpg")
    with pytest.raises(CameraError, match="tidak ditemukan"):
        backend.open()


def test_camera_auto_falls_back_from_picamera2_to_opencv(monkeypatch):
    # Ensure picamera2 missing, provide fake cv2 module
    monkeypatch.setitem(sys.modules, "picamera2", None)

    class FakeCap:
        def __init__(self, idx):
            self.idx = idx
            self.opened = True
            self.props = {}

        def isOpened(self):
            return self.opened

        def set(self, prop, val):
            self.props[prop] = val

        def read(self):
            return True, np.full((120, 160, 3), 42, dtype=np.uint8)

        def release(self):
            self.opened = False

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=FakeCap,
        VideoWriter_fourcc=lambda *chars: 1196444237,
        CAP_PROP_FOURCC=1,
        CAP_PROP_FRAME_WIDTH=2,
        CAP_PROP_FRAME_HEIGHT=3,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    mgr = CameraManager(_cam_cfg(backend="auto", flush_frames=1), sleeper=lambda _: None)
    with mgr:
        assert mgr.active_backend_name == "opencv:0"
        frame = mgr.capture_frame()
        assert frame.shape == (120, 160, 3)


def test_opencv_auto_reconnect_on_drop(monkeypatch):
    attempts = {"count": 0}

    class FlakyCap:
        def __init__(self, _idx):
            attempts["count"] += 1
            self.reads = 0

        def isOpened(self):
            return True

        def set(self, _p, _v):
            pass

        def read(self):
            self.reads += 1
            # First instance fails on read after flush
            if attempts["count"] == 1 and self.reads > 2:
                return False, None
            return True, np.zeros((10, 10, 3), dtype=np.uint8)

        def release(self):
            pass

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=FlakyCap,
        VideoWriter_fourcc=lambda *_: 1,
        CAP_PROP_FOURCC=1,
        CAP_PROP_FRAME_WIDTH=2,
        CAP_PROP_FRAME_HEIGHT=3,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    backend = OpenCVCameraBackend(0, 10, 10, flush_frames=2, sleeper=lambda _: None)
    backend.open()
    frame = backend.capture()
    assert attempts["count"] == 2
    assert frame.shape == (10, 10, 3)
    backend.close()


def test_picamera2_backend_and_capture_reopen(monkeypatch):
    class FakePiCam:
        def create_still_configuration(self, main):
            return main

        def configure(self, _cfg):
            pass

        def start(self):
            pass

        def capture_array(self, _name):
            return np.full((60, 80, 3), 77, dtype=np.uint8)

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "picamera2", types.SimpleNamespace(Picamera2=FakePiCam))
    mgr = CameraManager(_cam_cfg(backend="picamera2", warmup_s=0.01), sleeper=lambda _: None)
    with mgr:
        assert mgr.active_backend_name == "picamera2"
        frame = mgr.capture_frame()
        assert frame.shape == (60, 80, 3)

    # Verify manager fails loudly when no backend can open
    monkeypatch.setitem(sys.modules, "picamera2", None)
    monkeypatch.setitem(sys.modules, "cv2", None)
    fail_mgr = CameraManager(_cam_cfg(backend="auto"), sleeper=lambda _: None)
    with pytest.raises(CameraError, match="Tidak ada backend kamera"):
        fail_mgr.init_camera()

