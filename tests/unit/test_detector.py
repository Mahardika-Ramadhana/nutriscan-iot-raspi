from pathlib import Path
import sys
import types
import numpy as np
import pytest

from nutriscan.ai.detector import (
    DetectorError,
    NutriScanDetector,
    calculate_iou,
    compute_mask_nutrition,
    load_nutrition_table,
    read_ncnn_metadata_imgsz,
)
from nutriscan.core.config import ModelConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_load_real_nutrition_csv_all_39_classes():
    csv_path = _repo_root() / "data" / "classes_nutritions.csv"
    table = load_nutrition_table(csv_path)

    assert len(table) == 39
    assert table["Nasi"]["energi_kcal"] == pytest.approx(180.0)
    assert table["Nasi"]["karbohidrat_g"] == pytest.approx(39.8)
    assert table["Ayam Goreng"]["protein_g"] == pytest.approx(35.7)
    # Row with messy trailing blanks must still preserve macro values cleanly
    assert table["Bakso Saus BBQ"]["energi_kcal"] == pytest.approx(140.0)
    assert table["Bakso Saus BBQ"]["lemak_g"] == pytest.approx(6.0)


def test_read_ncnn_metadata_imgsz_for_both_models():
    assert read_ncnn_metadata_imgsz(_repo_root() / "models" / "best_640_ncnn_model") == 640
    assert read_ncnn_metadata_imgsz(_repo_root() / "models" / "best_416_ncnn_model") == 416


def test_compute_mask_nutrition_highest_conf_wins_overlap():
    # 10x10 mask: class 0 covers 0:6 (60 px), class 1 covers 4:10 (60 px). Overlap 4:6 (20 px)
    # Class 1 has higher confidence (0.9 vs 0.6), so class 1 keeps 60 px and class 0 keeps 40 px!
    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[0, 0:6, :] = True
    masks[1, 4:10, :] = True
    cls_ids = np.array([0, 1], dtype=int)
    confs = np.array([0.6, 0.9], dtype=float)
    names = {0: "Nasi", 1: "Ayam Goreng"}
    table = load_nutrition_table(_repo_root() / "data" / "classes_nutritions.csv")

    components, totals = compute_mask_nutrition(cls_ids, confs, masks, names, 500.0, table)

    assert components["Nasi"].area_px == 40
    assert components["Nasi"].proportion == pytest.approx(0.4)
    assert components["Nasi"].estimated_grams == pytest.approx(200.0)

    assert components["Ayam Goreng"].area_px == 60
    assert components["Ayam Goreng"].proportion == pytest.approx(0.6)
    assert components["Ayam Goreng"].estimated_grams == pytest.approx(300.0)

    # Energi: (200/100 * 180) + (300/100 * 283) = 360 + 849 = 1209 kcal
    assert totals is not None
    assert totals["energi_kcal"] == pytest.approx(1209.0)


def test_calculate_iou_and_mock_detector_analysis():
    assert calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert calculate_iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)

    cfg = ModelConfig(
        variant="640",
        path=_repo_root() / "models" / "best_640_ncnn_model",
        imgsz=640,
        conf_threshold=0.35,
        nutrition_csv=_repo_root() / "data" / "classes_nutritions.csv",
        warmup=False,
        mock_mode=True,
    )
    detector = NutriScanDetector(cfg)
    detector.load_model()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    analysis, annotated = detector.analyze(frame, weight_total_g=350.0)

    assert annotated is not None
    assert analysis.detection_count == 3
    assert set(analysis.components.keys()) == {"Nasi", "Ayam Goreng", "Sawi"}
    assert analysis.total_nutrition is not None
    assert analysis.total_nutrition["energi_kcal"] > 0


def test_detector_rejects_mismatched_imgsz():
    cfg = ModelConfig(
        variant="640",
        path=_repo_root() / "models" / "best_640_ncnn_model",
        imgsz=416,  # mismatched with best_640_ncnn_model/metadata.yaml
        conf_threshold=0.35,
        nutrition_csv=_repo_root() / "data" / "classes_nutritions.csv",
        warmup=False,
        mock_mode=False,
    )
    detector = NutriScanDetector(cfg)
    with pytest.raises(DetectorError, match="Ketidakcocokan imgsz"):
        detector.load_model()


def test_detector_real_mode_with_fake_ultralytics(monkeypatch):
    class FakeTensor:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    class FakeBoxes:
        def __init__(self):
            self.cls = FakeTensor([20])
            self.conf = FakeTensor([0.92])
            self.xyxy = FakeTensor([[10.0, 10.0, 50.0, 50.0]])

        def __len__(self):
            return 1

    class FakeResult:
        names = {20: "Nasi"}
        speed = {"preprocess": 2.0, "inference": 35.0, "postprocess": 3.0}
        boxes = FakeBoxes()
        masks = types.SimpleNamespace(data=FakeTensor(np.ones((1, 20, 20), dtype=float)))

        def plot(self):
            return np.full((20, 20, 3), 255, dtype=np.uint8)

    class FakeYOLO:
        def __init__(self, path, task="segment"):
            self.path = path
            self.task = task

        def predict(self, *_args, **_kwargs):
            return [FakeResult()]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
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
    analysis, annotated = detector.analyze(np.zeros((20, 20, 3), dtype=np.uint8), weight_total_g=200.0)
    assert analysis.detection_count == 1
    assert analysis.components["Nasi"].estimated_grams == pytest.approx(200.0)
    assert annotated is not None
