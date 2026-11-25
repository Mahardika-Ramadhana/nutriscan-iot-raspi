"""YOLO26n-seg NCNN detector, mask-proportion weight splitter, and nutrition calculator."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from nutriscan.ai.ncnn_engine import (
    NcnnSegmentEngine,
    read_ncnn_metadata,
    render_segmentation_overlay,
)
from nutriscan.core.config import ModelConfig
from nutriscan.core.logger import get_logger

logger = get_logger("NutriScanDetector")

# Pemetaan kolom CSV sumber (nilai gizi per 100 g) -> key JSON keluaran (dari yolo/detect.py)
NUTRITION_COLUMNS: dict[str, str] = {
    "energi": "energi_kcal",
    "protein": "protein_g",
    "lemak": "lemak_g",
    "karbohidrat": "karbohidrat_g",
}


class DetectorError(RuntimeError):
    """Raised when model loading, metadata validation, or inference fails."""


@dataclass(frozen=True)
class DetectionItem:
    """Single detected food item instance."""

    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kelas": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
        }


@dataclass(frozen=True)
class ComponentNutrition:
    """Aggregated pixel proportion, estimated grams, and nutrition per food class."""

    class_name: str
    area_px: int
    proportion: float
    estimated_grams: float | None
    nutrition: dict[str, float] | None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "luas_px": self.area_px,
            "proporsi": self.proportion,
        }
        if self.estimated_grams is not None:
            data["estimasi_gram"] = self.estimated_grams
        if self.nutrition is not None:
            data["gizi"] = dict(self.nutrition)
        return data


@dataclass(frozen=True)
class ScanAnalysis:
    """Immutable analysis output matching yolo/detect.py JSON structure."""

    detection_count: int
    detections: tuple[DetectionItem, ...]
    components: dict[str, ComponentNutrition]
    weight_total_g: float | None
    total_nutrition: dict[str, float] | None
    has_occlusion: bool
    timing_ms: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "jumlah_deteksi": self.detection_count,
            "deteksi": [d.to_dict() for d in self.detections],
            "komponen": {name: comp.to_dict() for name, comp in self.components.items()},
            "berat_total_g": self.weight_total_g,
            "total_gizi": dict(self.total_nutrition) if self.total_nutrition else None,
            "has_occlusion": self.has_occlusion,
            "waktu_ms": dict(self.timing_ms),
        }


def load_nutrition_table(csv_path: Path | str) -> dict[str, dict[str, float | None]]:
    """Parse classes_nutritions.csv into {class_name: {energi_kcal: ..., ...}}.

    Kolom gizi makro (energi, protein, lemak, karbohidrat) berada di awal baris
    sehingga tetap terbaca akurat meskipun kolom mikronutrien di belakang bergeser.
    """
    path = Path(csv_path)
    if not path.is_file():
        return {}

    table: dict[str, dict[str, float | None]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        try:
            header = [h.strip().lower() for h in next(reader)]
        except StopIteration:
            return {}

        if "class_name" not in header:
            raise DetectorError(f"Kolom 'class_name' tidak ditemukan di {path}")

        col_indices = {
            col: header.index(col)
            for col in NUTRITION_COLUMNS
            if col in header
        }
        name_idx = header.index("class_name")

        for raw_row in reader:
            row = [cell.strip() for cell in raw_row]
            if len(row) < len(NUTRITION_COLUMNS) + 2 or name_idx >= len(row):
                continue
            class_name = row[name_idx]
            if not class_name:
                continue

            values: dict[str, float | None] = {}
            for src_col, out_key in NUTRITION_COLUMNS.items():
                idx = col_indices.get(src_col)
                if idx is None or idx >= len(row):
                    values[out_key] = None
                    continue
                try:
                    values[out_key] = float(row[idx].replace(",", "."))
                except ValueError:
                    values[out_key] = None
            table[class_name] = values

    return table


def read_ncnn_metadata_imgsz(model_path: Path) -> int | None:
    """Read exported imgsz from NCNN metadata.yaml if present."""
    try:
        imgsz, _names = read_ncnn_metadata(model_path)
    except yaml.YAMLError as exc:
        raise DetectorError(f"Gagal membaca metadata NCNN {model_path / 'metadata.yaml'}: {exc}") from exc
    return imgsz


def calculate_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Calculate Intersection over Union for occlusion warnings (from DICA)."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_area = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter_area == 0.0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def _check_occlusion(detections: list[DetectionItem], threshold: float = 0.5) -> bool:
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if calculate_iou(detections[i].bbox_xyxy, detections[j].bbox_xyxy) > threshold:
                return True
    return False


def compute_mask_nutrition(
    cls_ids: np.ndarray,
    confs: np.ndarray,
    masks: np.ndarray | None,
    class_names: dict[int, str],
    weight_total_g: float | None,
    nutrition_table: dict[str, dict[str, float | None]],
) -> tuple[dict[str, ComponentNutrition], dict[str, float] | None]:
    """Resolve overlapping masks (highest confidence wins) and split total weight & nutrition."""
    if masks is None or len(masks) == 0:
        return {}, None

    label_map = np.full(masks.shape[1:], -1, dtype=np.int16)
    for idx in np.argsort(confs):
        label_map[masks[idx]] = int(cls_ids[idx])

    ids, counts = np.unique(label_map[label_map >= 0], return_counts=True)
    total_px = int(counts.sum())
    if total_px <= 0:
        return {}, None

    components: dict[str, ComponentNutrition] = {}
    total_nutrition: dict[str, float] = {}

    for cls_id, px_count in zip(ids, counts):
        class_name = class_names[int(cls_id)]
        proportion_raw = float(px_count / total_px)
        est_grams = round(proportion_raw * weight_total_g, 1) if weight_total_g is not None else None

        comp_nutrition: dict[str, float] | None = None
        if est_grams is not None and nutrition_table and class_name in nutrition_table:
            comp_nutrition = {}
            for nutrient_key, per_100g in nutrition_table[class_name].items():
                if per_100g is None:
                    continue
                val = est_grams / 100.0 * per_100g
                comp_nutrition[nutrient_key] = round(val, 2)
                total_nutrition[nutrient_key] = round(total_nutrition.get(nutrient_key, 0.0) + val, 2)

        components[class_name] = ComponentNutrition(
            class_name=class_name,
            area_px=int(px_count),
            proportion=round(proportion_raw, 4),
            estimated_grams=est_grams,
            nutrition=comp_nutrition,
        )

    return components, (total_nutrition or None)


class NutriScanDetector:
    """NCNN YOLO26n-seg inference engine integrated with Load Cell weight telemetry."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model = None
        self._ncnn_engine: NcnnSegmentEngine | None = None
        self._nutrition_table: dict[str, dict[str, float | None]] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def nutrition_table(self) -> dict[str, dict[str, float | None]]:
        return self._nutrition_table

    def _load_ultralytics_model(self, model_path: Path) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError("Library ncnn maupun ultralytics tidak terpasang di environment ini") from exc

        t0 = time.perf_counter()
        self._model = YOLO(str(model_path), task="segment")
        if self._config.warmup:
            dummy = np.zeros((self._config.imgsz, self._config.imgsz, 3), dtype=np.uint8)
            self._model.predict(dummy, imgsz=self._config.imgsz, verbose=False)
        self._loaded = True
        logger.info(
            "Model Ultralytics '%s' (imgsz=%d) siap dalam %.2f s",
            model_path.name,
            self._config.imgsz,
            time.perf_counter() - t0,
        )

    def load_model(self) -> None:
        """Load nutrition CSV, validate NCNN metadata, load native NCNN or YOLO model, and warm up."""
        self._nutrition_table = load_nutrition_table(self._config.nutrition_csv)
        if not self._nutrition_table:
            logger.warning("Tabel nutrisi kosong atau tidak ditemukan: %s", self._config.nutrition_csv)
        else:
            logger.info("Tabel nutrisi dimuat (%d kelas makanan)", len(self._nutrition_table))

        if self._config.mock_mode:
            self._loaded = True
            logger.info("NutriScanDetector dimuat dalam MODE MOCK.")
            return

        model_path = self._config.path
        if not model_path.exists():
            raise DetectorError(f"Model NCNN tidak ditemukan: {model_path}")

        meta_imgsz = read_ncnn_metadata_imgsz(model_path)
        if meta_imgsz is not None and meta_imgsz != self._config.imgsz:
            raise DetectorError(
                f"Ketidakcocokan imgsz: config={self._config.imgsz}, "
                f"metadata NCNN ({model_path.name})={meta_imgsz}"
            )

        is_ncnn_dir = (model_path / "model.ncnn.param").is_file() and (model_path / "model.ncnn.bin").is_file()
        if is_ncnn_dir:
            try:
                t0 = time.perf_counter()
                engine = NcnnSegmentEngine(model_path, self._config.imgsz)
                engine.load()
                if self._config.warmup:
                    dummy = np.zeros((self._config.imgsz, self._config.imgsz, 3), dtype=np.uint8)
                    engine.predict(dummy, conf_threshold=self._config.conf_threshold)
                self._ncnn_engine = engine
                self._loaded = True
                logger.info(
                    "Model native NCNN '%s' (imgsz=%d) siap dalam %.2f s",
                    model_path.name,
                    self._config.imgsz,
                    time.perf_counter() - t0,
                )
                return
            except ImportError as exc:
                logger.warning("Paket 'ncnn' tidak tersedia (%s), fallback ke Ultralytics...", exc)

        self._load_ultralytics_model(model_path)

    def _analyze_ncnn(
        self,
        frame: np.ndarray,
        weight_total_g: float | None,
    ) -> tuple[ScanAnalysis, np.ndarray]:
        assert self._ncnn_engine is not None
        pred = self._ncnn_engine.predict(frame, conf_threshold=self._config.conf_threshold)
        detections: list[DetectionItem] = []
        for c, p, b in zip(pred.cls_ids, pred.confs, pred.boxes_xyxy):
            detections.append(
                DetectionItem(
                    class_name=pred.class_names.get(int(c), str(int(c))),
                    confidence=round(float(p), 3),
                    bbox_xyxy=tuple(round(float(v), 1) for v in b),  # type: ignore[arg-type]
                )
            )

        components, total_nutrition = compute_mask_nutrition(
            cls_ids=pred.cls_ids,
            confs=pred.confs,
            masks=pred.masks_bool,
            class_names=pred.class_names,
            weight_total_g=weight_total_g,
            nutrition_table=self._nutrition_table,
        )
        analysis = ScanAnalysis(
            detection_count=len(detections),
            detections=tuple(detections),
            components=components,
            weight_total_g=weight_total_g,
            total_nutrition=total_nutrition,
            has_occlusion=_check_occlusion(detections),
            timing_ms=pred.timing_ms,
        )
        annotated = render_segmentation_overlay(
            frame=frame,
            cls_ids=pred.cls_ids,
            confs=pred.confs,
            boxes_xyxy=pred.boxes_xyxy,
            masks_bool=pred.masks_bool,
            class_names=pred.class_names,
        )
        return analysis, annotated

    def analyze(
        self,
        frame: np.ndarray,
        weight_total_g: float | None = None,
    ) -> tuple[ScanAnalysis, np.ndarray | None]:
        """Run segmentation on a BGR frame and compute gram/nutrition split."""
        if not self._loaded:
            self.load_model()
        if frame is None or frame.size == 0:
            raise DetectorError("Frame input kosong untuk inferensi")

        if self._config.mock_mode:
            return self._analyze_mock(frame, weight_total_g)
        if self._ncnn_engine is not None:
            return self._analyze_ncnn(frame, weight_total_g)

        assert self._model is not None
        t0 = time.perf_counter()
        res = self._model.predict(
            frame,
            imgsz=self._config.imgsz,
            conf=self._config.conf_threshold,
            retina_masks=False,
            verbose=False,
        )[0]
        wall_ms = (time.perf_counter() - t0) * 1000.0

        detections: list[DetectionItem] = []
        components: dict[str, ComponentNutrition] = {}
        total_nutrition: dict[str, float] | None = None

        if res.boxes is not None and len(res.boxes):
            cls_arr = res.boxes.cls.cpu().numpy().astype(int)
            confs_arr = res.boxes.conf.cpu().numpy()
            boxes_arr = res.boxes.xyxy.cpu().numpy()
            names = res.names

            for c, p, b in zip(cls_arr, confs_arr, boxes_arr):
                detections.append(
                    DetectionItem(
                        class_name=str(names[int(c)]),
                        confidence=round(float(p), 3),
                        bbox_xyxy=tuple(round(float(v), 1) for v in b),  # type: ignore[arg-type]
                    )
                )

            if res.masks is not None:
                masks_bool = res.masks.data.cpu().numpy() > 0.5
                components, total_nutrition = compute_mask_nutrition(
                    cls_ids=cls_arr,
                    confs=confs_arr,
                    masks=masks_bool,
                    class_names={int(k): str(v) for k, v in names.items()},
                    weight_total_g=weight_total_g,
                    nutrition_table=self._nutrition_table,
                )

        timing = {k: round(float(v), 1) for k, v in res.speed.items()}
        timing["total"] = round(wall_ms, 1)

        analysis = ScanAnalysis(
            detection_count=len(detections),
            detections=tuple(detections),
            components=components,
            weight_total_g=weight_total_g,
            total_nutrition=total_nutrition,
            has_occlusion=_check_occlusion(detections),
            timing_ms=timing,
        )
        annotated = res.plot()
        return analysis, annotated

    def _analyze_mock(
        self,
        frame: np.ndarray,
        weight_total_g: float | None,
    ) -> tuple[ScanAnalysis, np.ndarray]:
        """Deterministic synthetic segmentation using real nutrition CSV math."""
        t0 = time.perf_counter()
        mock_classes = {0: "Nasi", 1: "Ayam Goreng", 2: "Sawi"}
        h_mask, w_mask = 100, 100
        masks = np.zeros((3, h_mask, w_mask), dtype=bool)
        # 50% Nasi, 30% Ayam Goreng, 20% Sawi (total 5000 piksel aktif)
        masks[0, 0:50, 0:50] = True
        masks[1, 0:30, 50:100] = True
        masks[2, 50:70, 0:50] = True

        cls_ids = np.array([0, 1, 2], dtype=int)
        confs = np.array([0.94, 0.91, 0.87], dtype=float)
        components, total_nutrition = compute_mask_nutrition(
            cls_ids=cls_ids,
            confs=confs,
            masks=masks,
            class_names=mock_classes,
            weight_total_g=weight_total_g,
            nutrition_table=self._nutrition_table,
        )

        fh, fw = frame.shape[:2]
        detections = (
            DetectionItem("Nasi", 0.94, (round(fw * 0.1, 1), round(fh * 0.15, 1), round(fw * 0.5, 1), round(fh * 0.55, 1))),
            DetectionItem("Ayam Goreng", 0.91, (round(fw * 0.55, 1), round(fh * 0.15, 1), round(fw * 0.9, 1), round(fh * 0.55, 1))),
            DetectionItem("Sawi", 0.87, (round(fw * 0.25, 1), round(fh * 0.6, 1), round(fw * 0.75, 1), round(fh * 0.88, 1))),
        )
        wall_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        analysis = ScanAnalysis(
            detection_count=len(detections),
            detections=detections,
            components=components,
            weight_total_g=weight_total_g,
            total_nutrition=total_nutrition,
            has_occlusion=False,
            timing_ms={"preprocess": 0.5, "inference": wall_ms, "postprocess": 0.2, "total": wall_ms},
        )
        return analysis, frame.copy()
