"""Camera capture manager supporting PiCamera2, OpenCV MJPEG, and mock simulation."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Callable, Protocol

import numpy as np

from nutriscan.core.config import CameraConfig
from nutriscan.core.logger import get_logger

logger = get_logger("CameraManager")


class CameraError(RuntimeError):
    """Raised when camera initialization or frame capture fails."""


class CameraBackend(Protocol):
    """Protocol implemented by every camera backend."""

    def open(self) -> None: ...

    def capture(self) -> np.ndarray: ...

    def close(self) -> None: ...


class PiCamera2Backend:
    """Raspberry Pi CSI Camera backend via picamera2 (BGR byte order for OpenCV/YOLO)."""

    def __init__(
        self,
        width: int,
        height: int,
        warmup_s: float = 1.5,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._width = width
        self._height = height
        self._warmup_s = warmup_s
        self._sleeper = sleeper
        self._cam = None

    def open(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise CameraError("picamera2 tidak terpasang (pasang sudo apt install python3-picamera2)") from exc

        try:
            cam = Picamera2()
            # Format RGB888 pada picamera2 menghasilkan urutan byte BGR yang kompatibel dengan OpenCV/YOLO
            cam.configure(
                cam.create_still_configuration(
                    main={"size": (self._width, self._height), "format": "RGB888"}
                )
            )
            cam.start()
            if self._warmup_s > 0:
                self._sleeper(self._warmup_s)
            self._cam = cam
        except Exception as exc:
            raise CameraError(f"Gagal membuka PiCamera2: {exc}") from exc

    def capture(self) -> np.ndarray:
        if self._cam is None:
            raise CameraError("PiCamera2 belum dibuka")
        try:
            frame = self._cam.capture_array("main")
        except Exception as exc:
            raise CameraError(f"Gagal mengambil frame PiCamera2: {exc}") from exc
        if frame is None or frame.size == 0:
            raise CameraError("PiCamera2 mengembalikan frame kosong")
        return frame

    def close(self) -> None:
        if self._cam is not None:
            try:
                self._cam.stop()
            except Exception:
                pass
            try:
                self._cam.close()
            finally:
                self._cam = None


class OpenCVCameraBackend:
    """USB UVC Webcam backend via OpenCV VideoCapture with MJPEG & auto-reconnect."""

    def __init__(
        self,
        camera_id: int,
        width: int,
        height: int,
        use_mjpeg: bool = True,
        flush_frames: int = 4,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._camera_id = camera_id
        self._width = width
        self._height = height
        self._use_mjpeg = use_mjpeg
        self._flush_frames = flush_frames
        self._sleeper = sleeper
        self._cap = None

    def _configure_capture(self, cv2_mod) -> object:
        cap = cv2_mod.VideoCapture(self._camera_id)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"Gagal membuka kamera USB index {self._camera_id}")
        if self._use_mjpeg:
            cap.set(cv2_mod.CAP_PROP_FOURCC, cv2_mod.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv2_mod.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2_mod.CAP_PROP_FRAME_HEIGHT, self._height)
        return cap

    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise CameraError("OpenCV (cv2) tidak terpasang") from exc
        self._cap = self._configure_capture(cv2)
        logger.info("Kamera USB %d dibuka (%dx%d, mjpeg=%s)", self._camera_id, self._width, self._height, self._use_mjpeg)

    def capture(self) -> np.ndarray:
        if self._cap is None:
            raise CameraError("OpenCVCameraBackend belum dibuka")

        for _ in range(self._flush_frames):
            self._cap.read()
        ok, frame = self._cap.read()
        if ok and frame is not None:
            return frame

        logger.warning("[Fail-Safe] Kamera USB %d terputus, mencoba auto-reconnect...", self._camera_id)
        self.close()
        self._sleeper(1.0)
        self.open()
        for _ in range(self._flush_frames):
            self._cap.read()
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError(f"Kamera USB {self._camera_id} gagal membaca frame setelah reconnect")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None


class MockCameraBackend:
    """Mock camera generating synthetic BGR frames or reading a sample image file."""

    def __init__(
        self,
        width: int,
        height: int,
        mock_image_path: Path | None = None,
    ) -> None:
        self._width = width
        self._height = height
        self._mock_image_path = mock_image_path
        self._is_open = False

    def open(self) -> None:
        if self._mock_image_path is not None and not self._mock_image_path.is_file():
            raise CameraError(f"File mock_image_path tidak ditemukan: {self._mock_image_path}")
        self._is_open = True

    def capture(self) -> np.ndarray:
        if not self._is_open:
            raise CameraError("MockCameraBackend belum dibuka")
        if self._mock_image_path is not None:
            return self._load_image(self._mock_image_path)

        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        frame[:, :] = (40, 50, 60)
        # Gambar 3 area warna kontras mewakili Nasi, Ayam Goreng, Sawi pada nampan
        h, w = self._height, self._width
        frame[int(h * 0.15) : int(h * 0.55), int(w * 0.1) : int(w * 0.5)] = (235, 235, 235)
        frame[int(h * 0.15) : int(h * 0.55), int(w * 0.55) : int(w * 0.9)] = (60, 120, 190)
        frame[int(h * 0.6) : int(h * 0.88), int(w * 0.25) : int(w * 0.75)] = (70, 175, 80)
        return frame

    @staticmethod
    def _load_image(image_path: Path) -> np.ndarray:
        try:
            import cv2

            img = cv2.imread(str(image_path))
            if img is not None:
                return img
        except ImportError:
            pass
        from PIL import Image

        with Image.open(image_path) as pil_img:
            rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
            return rgb[:, :, ::-1].copy()

    def close(self) -> None:
        self._is_open = False


class CameraManager:
    """High-level camera facade with auto PiCamera2->OpenCV fallback and recovery."""

    def __init__(
        self,
        config: CameraConfig,
        environment: str = "development",
        backend: CameraBackend | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._environment = environment.lower()
        self._backend: CameraBackend | None = backend
        self._sleeper = sleeper
        self.active_backend_name: str = "custom" if backend is not None else "uninitialized"

    def __enter__(self) -> CameraManager:
        self.init_camera()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release_all()

    def _build_and_open_backend(self) -> tuple[CameraBackend, str]:
        mode = self._config.backend
        if mode == "mock":
            backend = MockCameraBackend(
                self._config.width,
                self._config.height,
                self._config.mock_image_path,
            )
            backend.open()
            return backend, "mock"

        candidates = ["picamera2", "opencv"] if mode == "auto" else [mode]
        errors: list[str] = []
        for candidate in candidates:
            if candidate == "picamera2":
                try:
                    backend = PiCamera2Backend(
                        self._config.width,
                        self._config.height,
                        self._config.warmup_s,
                        sleeper=self._sleeper,
                    )
                    backend.open()
                    return backend, "picamera2"
                except CameraError as exc:
                    errors.append(f"picamera2: {exc}")
                    logger.warning("PiCamera2 tidak dapat digunakan (%s); mencoba fallback...", exc)
            elif candidate == "opencv":
                for cam_id in self._config.camera_ids:
                    try:
                        backend = OpenCVCameraBackend(
                            camera_id=cam_id,
                            width=self._config.width,
                            height=self._config.height,
                            use_mjpeg=self._config.use_mjpeg,
                            flush_frames=self._config.flush_frames,
                            sleeper=self._sleeper,
                        )
                        backend.open()
                        return backend, f"opencv:{cam_id}"
                    except CameraError as exc:
                        errors.append(f"opencv[{cam_id}]: {exc}")

        raise CameraError("Tidak ada backend kamera yang berhasil dibuka: " + "; ".join(errors))

    def init_camera(self) -> None:
        """Open configured camera backend (auto tries PiCamera2 then OpenCV USB)."""
        if self._backend is not None:
            self._backend.open()
            return
        self._backend, self.active_backend_name = self._build_and_open_backend()
        logger.info("CameraManager aktif menggunakan backend: %s", self.active_backend_name)

    def capture_frame(self) -> np.ndarray:
        """Capture a single BGR frame, attempting one full reopen on transient failure."""
        if self._backend is None:
            self.init_camera()
        assert self._backend is not None
        try:
            return self._backend.capture()
        except CameraError as first_err:
            logger.warning("Capture gagal (%s), mencoba re-inisialisasi kamera...", first_err)
            self.release_all()
            self._backend, self.active_backend_name = self._build_and_open_backend()
            return self._backend.capture()

    def release_all(self) -> None:
        """Release active camera handle safely."""
        if self._backend is not None:
            try:
                self._backend.close()
            finally:
                self._backend = None
