from pathlib import Path
import sys
import types
import numpy as np
import pytest

from nutriscan.ai.detector import NutriScanDetector
from nutriscan.ai.ncnn_engine import (
    NcnnSegmentEngine,
    decode_ncnn_seg_outputs,
    letterbox_bgr,
    nms_boxes_numpy,
    read_ncnn_metadata,
    render_segmentation_overlay,
)
from nutriscan.core.config import ModelConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_read_ncnn_metadata_for_640_and_416_models():
    imgsz_640, names_640 = read_ncnn_metadata(_repo_root() / "models" / "best_640_ncnn_model")
    imgsz_416, names_416 = read_ncnn_metadata(_repo_root() / "models" / "best_416_ncnn_model")

    assert imgsz_640 == 640
    assert imgsz_416 == 416
    assert len(names_640) == 39
    assert len(names_416) == 39
    assert names_640[20] == "Nasi"
    assert names_416[3] == "Ayam Goreng"


def test_letterbox_bgr_shapes_and_immutability():
    frame_hd = np.full((720, 1280, 3), 42, dtype=np.uint8)
    frame_hd_before = frame_hd.copy()
    lb_hd, gain_hd, (left_hd, top_hd) = letterbox_bgr(frame_hd, 640)

    assert np.array_equal(frame_hd, frame_hd_before)
    assert lb_hd.shape == (640, 640, 3)
    assert gain_hd == pytest.approx(0.5)
    assert left_hd == 0
    assert top_hd == 140
    # Top padding border must be (114, 114, 114)
    assert tuple(lb_hd[0, 0]) == (114, 114, 114)
    assert tuple(lb_hd[320, 320]) == (42, 42, 42)

    frame_odd = np.full((1279, 721, 3), 99, dtype=np.uint8)
    lb_odd, _gain_odd, _pad_odd = letterbox_bgr(frame_odd, 416)
    assert lb_odd.shape == (416, 416, 3)


def test_nms_boxes_numpy_class_aware():
    boxes = np.array(
        [
            [10.0, 10.0, 100.0, 100.0],  # class 20, highest conf
            [12.0, 12.0, 98.0, 98.0],    # class 20, heavy overlap -> suppressed
            [12.0, 12.0, 98.0, 98.0],    # class 3, heavy overlap -> kept (different class)
        ],
        dtype=np.float32,
    )
    scores = np.array([0.92, 0.75, 0.85], dtype=np.float32)
    cls_ids = np.array([20, 20, 3], dtype=np.int32)

    keep = nms_boxes_numpy(boxes, scores, cls_ids, iou_threshold=0.7)
    assert list(keep) == [0, 2]


def test_decode_ncnn_seg_outputs_scales_boxes_and_crops_masks():
    # 720x1280 frame letterboxed to 640x640 -> gain=0.5, pad=(0, 140)
    out0 = np.zeros((75, 8400), dtype=np.float32)
    # Anchor 10: Nasi (class 20), box at x:[100..300], y:[200..360] in 640x640 -> cx=200, cy=280, w=200, h=160
    out0[0:4, 10] = [200.0, 280.0, 200.0, 160.0]
    out0[4 + 20, 10] = 0.91
    out0[4 + 39 + 0, 10] = 3.0  # proto channel 0 positive weight

    # Anchor 20: Ayam Goreng (class 3), box at x:[360..520], y:[220..380]
    out0[0:4, 20] = [440.0, 300.0, 160.0, 160.0]
    out0[4 + 3, 20] = 0.84
    out0[4 + 39 + 1, 20] = 3.0  # proto channel 1 positive weight

    out1 = np.full((32, 160, 160), -1.0, dtype=np.float32)
    out1[0, :, :] = 2.0
    out1[1, :, :] = 2.0

    cls_ids, confs, boxes, masks = decode_ncnn_seg_outputs(
        out0=out0,
        out1=out1,
        orig_shape=(720, 1280),
        imgsz=640,
        gain=0.5,
        pad=(0, 140),
        num_classes=39,
        conf_threshold=0.35,
    )

    assert len(cls_ids) == 2
    assert set(cls_ids.tolist()) == {20, 3}
    assert confs[0] >= confs[1]
    assert masks is not None
    assert masks.shape == (2, 720, 1280)
    assert masks.dtype == bool

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        assert 0.0 <= x1 < x2 <= 1280.0
        assert 0.0 <= y1 < y2 <= 720.0
        ys, xs = np.where(masks[idx])
        assert len(ys) > 0
        assert xs.min() >= int(np.floor(x1))
        assert xs.max() < int(np.ceil(x2))
        assert ys.min() >= int(np.floor(y1))
        assert ys.max() < int(np.ceil(y2))


def test_render_segmentation_overlay_returns_new_uint8_frame():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    snapshot = frame.copy()
    masks = np.zeros((1, 120, 160), dtype=bool)
    masks[0, 20:80, 30:100] = True

    rendered = render_segmentation_overlay(
        frame=frame,
        cls_ids=np.array([20], dtype=np.int32),
        confs=np.array([0.92], dtype=np.float32),
        boxes_xyxy=np.array([[30.0, 20.0, 100.0, 80.0]], dtype=np.float32),
        masks_bool=masks,
        class_names={20: "Nasi"},
    )

    assert np.array_equal(frame, snapshot)
    assert rendered is not frame
    assert rendered.shape == (120, 160, 3)
    assert rendered.dtype == np.uint8
    assert rendered.sum() > 0


def test_detector_uses_native_ncnn_without_importing_ultralytics(monkeypatch):
    captured_opt: dict[str, object] = {}

    class PoisonUltralytics:
        def __getattr__(self, name):
            raise AssertionError(f"ultralytics must not be imported/used (accessed {name})")

    class FakeMat:
        def __init__(self, arr: np.ndarray):
            self._arr = np.asarray(arr)

        def clone(self):
            return FakeMat(self._arr.copy())

        def __array__(self, dtype=None):
            return self._arr.astype(dtype) if dtype else self._arr

    class FakeOpt:
        def __init__(self):
            self.use_vulkan_compute = True
            self.use_fp16_packed = True
            self.use_fp16_storage = True
            self.use_fp16_arithmetic = True
            self.use_bf16_storage = True
            self.lightmode = False
            self.openmp_blocktime = 20
            self.num_threads = 4

    class FakeExtractor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def input(self, name, mat):
            assert name == "in0"
            arr = np.asarray(mat)
            assert arr.shape == (3, 416, 416)

        def extract(self, name):
            if name == "out0":
                arr = np.zeros((75, 3549), dtype=np.float32)
                arr[0:4, 5] = [208.0, 208.0, 160.0, 160.0]
                arr[4 + 20, 5] = 0.93
                arr[4 + 39, 5] = 4.0
                return 0, FakeMat(arr)
            if name == "out1":
                arr = np.zeros((32, 104, 104), dtype=np.float32)
                arr[0, :, :] = 2.0
                return 0, FakeMat(arr)
            raise AssertionError(f"Unexpected blob: {name}")

    class FakeNet:
        def __init__(self):
            self.opt = FakeOpt()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def load_param(self, path):
            assert Path(path).name == "model.ncnn.param"
            captured_opt["use_vulkan_compute"] = self.opt.use_vulkan_compute
            captured_opt["use_fp16_packed"] = self.opt.use_fp16_packed
            captured_opt["use_fp16_storage"] = self.opt.use_fp16_storage
            captured_opt["use_fp16_arithmetic"] = self.opt.use_fp16_arithmetic
            captured_opt["use_bf16_storage"] = self.opt.use_bf16_storage
            return 0

        def load_model(self, path):
            assert Path(path).name == "model.ncnn.bin"
            return 0

        def create_extractor(self):
            return FakeExtractor()

    fake_ncnn = types.SimpleNamespace(Net=FakeNet, Mat=FakeMat)
    monkeypatch.setitem(sys.modules, "ncnn", fake_ncnn)
    monkeypatch.setitem(sys.modules, "ultralytics", PoisonUltralytics())

    cfg = ModelConfig(
        variant="416",
        path=_repo_root() / "models" / "best_416_ncnn_model",
        imgsz=416,
        conf_threshold=0.35,
        nutrition_csv=_repo_root() / "data" / "classes_nutritions.csv",
        warmup=True,
        mock_mode=False,
    )
    detector = NutriScanDetector(cfg)
    detector.load_model()

    assert isinstance(detector._ncnn_engine, NcnnSegmentEngine)
    assert captured_opt == {
        "use_vulkan_compute": False,
        "use_fp16_packed": False,
        "use_fp16_storage": False,
        "use_fp16_arithmetic": False,
        "use_bf16_storage": False,
    }

    analysis, annotated = detector.analyze(np.zeros((240, 320, 3), dtype=np.uint8), weight_total_g=200.0)
    assert analysis.detection_count == 1
    assert "Nasi" in analysis.components
    assert analysis.components["Nasi"].estimated_grams == pytest.approx(200.0)
    assert analysis.total_nutrition is not None
    assert analysis.total_nutrition["energi_kcal"] == pytest.approx(360.0)
    assert annotated is not None and annotated.shape == (240, 320, 3)
