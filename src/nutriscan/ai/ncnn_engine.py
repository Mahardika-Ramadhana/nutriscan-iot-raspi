"""Pure-NCNN + NumPy inference engine for YOLO26n-seg without PyTorch/Ultralytics."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
import yaml

import numpy as np

from nutriscan.core.logger import get_logger

logger = get_logger("NcnnEngine")

_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (56, 189, 248),
    (16, 185, 129),
    (245, 158, 11),
    (244, 63, 94),
    (168, 85, 247),
    (34, 211, 238),
    (250, 204, 21),
    (96, 165, 250),
    (52, 211, 153),
    (251, 146, 60),
)


@dataclass(frozen=True)
class NcnnPrediction:
    """Immutable segmentation prediction output from NcnnSegmentEngine."""

    cls_ids: np.ndarray
    confs: np.ndarray
    boxes_xyxy: np.ndarray
    masks_bool: np.ndarray | None
    class_names: dict[int, str]
    timing_ms: dict[str, float]


def read_ncnn_metadata(model_path: Path) -> tuple[int | None, dict[int, str]]:
    """Read imgsz and class names mapping from NCNN metadata.yaml."""
    meta_file = model_path / "metadata.yaml"
    if not meta_file.is_file():
        return None, {}

    meta = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        return None, {}

    raw_imgsz = meta.get("imgsz")
    imgsz = int(raw_imgsz[0]) if isinstance(raw_imgsz, list) and raw_imgsz else None

    raw_names = meta.get("names")
    names: dict[int, str] = {}
    if isinstance(raw_names, dict):
        names = {int(k): str(v) for k, v in raw_names.items()}
    elif isinstance(raw_names, list):
        names = {idx: str(val) for idx, val in enumerate(raw_names)}
    return imgsz, names


def _resize_uint8_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image

        rgb = frame[:, :, ::-1]
        resized_rgb = Image.fromarray(rgb).resize((width, height), Image.BILINEAR)
        return np.asarray(resized_rgb, dtype=np.uint8)[:, :, ::-1].copy()


def _resize_float_map(grid: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(grid, (width, height), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image

        pil_img = Image.fromarray(grid.astype(np.float32), mode="F")
        return np.asarray(pil_img.resize((width, height), Image.BILINEAR), dtype=np.float32)


def letterbox_bgr(
    frame: np.ndarray,
    imgsz: int,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize BGR image preserving aspect ratio and pad to (imgsz, imgsz, 3) cleanly."""
    h, w = frame.shape[:2]
    gain = min(imgsz / float(h), imgsz / float(w))
    new_w = max(1, int(round(w * gain)))
    new_h = max(1, int(round(h * gain)))

    resized = frame if (w == new_w and h == new_h) else _resize_uint8_bgr(frame, new_w, new_h)
    dw = (imgsz - new_w) / 2.0
    dh = (imgsz - new_h) / 2.0
    left = max(0, min(imgsz - new_w, int(round(dw - 0.1))))
    top = max(0, min(imgsz - new_h, int(round(dh - 0.1))))

    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, float(gain), (left, top)


def preprocess_for_ncnn(
    frame: np.ndarray,
    imgsz: int,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Letterbox BGR image and convert to contiguous RGB float32 CHW scaled to [0, 1]."""
    letterboxed, gain, pad = letterbox_bgr(frame, imgsz)
    rgb = letterboxed[:, :, ::-1]
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    chw *= np.float32(1.0 / 255.0)
    return np.ascontiguousarray(chw, dtype=np.float32), gain, pad


def nms_boxes_numpy(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    cls_ids: np.ndarray,
    iou_threshold: float = 0.7,
    max_det: int = 300,
) -> np.ndarray:
    """Class-aware greedy Non-Maximum Suppression implemented in pure NumPy."""
    if len(boxes_xyxy) == 0:
        return np.empty((0,), dtype=np.int64)

    offsets = cls_ids.astype(np.float32) * 7680.0
    shifted = boxes_xyxy.astype(np.float32) + offsets[:, None]
    x1, y1, x2, y2 = shifted[:, 0], shifted[:, 1], shifted[:, 2], shifted[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-scores, kind="mergesort")

    keep: list[int] = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        ious = np.where(union > 0.0, inter / np.maximum(union, 1e-6), 0.0)
        order = rest[ious <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def _normalize_ncnn_blobs(
    out0: np.ndarray,
    out1: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    det = np.asarray(out0, dtype=np.float32)
    proto = np.asarray(out1, dtype=np.float32)
    if det.ndim == 3 and det.shape[0] == 1:
        det = det[0]
    if proto.ndim == 4 and proto.shape[0] == 1:
        proto = proto[0]
    expected_ch = 4 + num_classes + proto.shape[0]
    if det.ndim == 2 and det.shape[0] != expected_ch and det.shape[1] == expected_ch:
        det = det.T
    return det, proto


def _decode_masks_elementwise(
    coeffs: np.ndarray,
    proto: np.ndarray,
    boxes_orig: np.ndarray,
    orig_shape: tuple[int, int],
    imgsz: int,
    gain: float,
    pad: tuple[int, int],
) -> np.ndarray | None:
    if len(boxes_orig) == 0 or proto.ndim != 3:
        return None
    orig_h, orig_w = orig_shape
    _nm, mh, mw = proto.shape
    scale_x = mw / float(imgsz)
    scale_y = mh / float(imgsz)
    new_w = int(round(orig_w * gain))
    new_h = int(round(orig_h * gain))

    p_left = max(0, int(np.floor(pad[0] * scale_x)))
    p_top = max(0, int(np.floor(pad[1] * scale_y)))
    p_right = min(mw, max(p_left + 1, int(np.ceil((pad[0] + new_w) * scale_x))))
    p_bottom = min(mh, max(p_top + 1, int(np.ceil((pad[1] + new_h) * scale_y))))
    proto_valid = proto[:, p_top:p_bottom, p_left:p_right]

    masks = np.zeros((len(boxes_orig), orig_h, orig_w), dtype=bool)
    for idx in range(len(boxes_orig)):
        # Pure elementwise reduction avoids OpenBLAS cblas_sgemm calls on ARMv8.0
        logits = np.sum(coeffs[idx, :, None, None] * proto_valid, axis=0, dtype=np.float32)
        resized = _resize_float_map(logits, orig_w, orig_h)
        ix1 = max(0, min(orig_w, int(np.floor(boxes_orig[idx, 0]))))
        iy1 = max(0, min(orig_h, int(np.floor(boxes_orig[idx, 1]))))
        ix2 = max(ix1, min(orig_w, int(np.ceil(boxes_orig[idx, 2]))))
        iy2 = max(iy1, min(orig_h, int(np.ceil(boxes_orig[idx, 3]))))
        if ix2 > ix1 and iy2 > iy1:
            masks[idx, iy1:iy2, ix1:ix2] = resized[iy1:iy2, ix1:ix2] > 0.0
    return masks


def decode_ncnn_seg_outputs(
    out0: np.ndarray,
    out1: np.ndarray,
    orig_shape: tuple[int, int],
    imgsz: int,
    gain: float,
    pad: tuple[int, int],
    num_classes: int,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.7,
    max_det: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Decode YOLO26n-seg NCNN outputs into classes, confidences, xyxy boxes, and masks."""
    det, proto = _normalize_ncnn_blobs(out0, out1, num_classes)
    cls_scores = det[4 : 4 + num_classes, :]
    best_cls = np.argmax(cls_scores, axis=0)
    best_conf = cls_scores[best_cls, np.arange(det.shape[1])]

    keep_mask = best_conf >= conf_threshold
    if not np.any(keep_mask):
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            None,
        )

    cand = det[:, keep_mask]
    cand_cls = best_cls[keep_mask].astype(np.int32)
    cand_conf = best_conf[keep_mask].astype(np.float32)

    cx, cy, bw, bh = cand[0], cand[1], cand[2], cand[3]
    boxes_lb = np.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], axis=1)
    keep_idx = nms_boxes_numpy(boxes_lb, cand_conf, cand_cls, iou_threshold=iou_threshold, max_det=max_det)

    cls_ids = cand_cls[keep_idx]
    confs = cand_conf[keep_idx]
    boxes_kept = boxes_lb[keep_idx].copy()
    coeffs = cand[4 + num_classes :, keep_idx].T

    orig_h, orig_w = orig_shape
    boxes_kept[:, [0, 2]] = np.clip((boxes_kept[:, [0, 2]] - pad[0]) / gain, 0.0, float(orig_w))
    boxes_kept[:, [1, 3]] = np.clip((boxes_kept[:, [1, 3]] - pad[1]) / gain, 0.0, float(orig_h))

    masks = _decode_masks_elementwise(coeffs, proto, boxes_kept, orig_shape, imgsz, gain, pad)
    return cls_ids, confs, boxes_kept, masks


def render_segmentation_overlay(
    frame: np.ndarray,
    cls_ids: np.ndarray,
    confs: np.ndarray,
    boxes_xyxy: np.ndarray,
    masks_bool: np.ndarray | None,
    class_names: dict[int, str],
) -> np.ndarray:
    """Render masks, bounding boxes, and class labels onto a new BGR uint8 copy."""
    canvas = frame.copy()
    if len(cls_ids) == 0:
        return canvas

    if masks_bool is not None:
        for idx, cls_id in enumerate(cls_ids):
            mask = masks_bool[idx]
            if not np.any(mask):
                continue
            color = np.asarray(_PALETTE_BGR[int(cls_id) % len(_PALETTE_BGR)], dtype=np.float32)
            canvas[mask] = (canvas[mask].astype(np.float32) * 0.55 + color * 0.45).astype(np.uint8)

    try:
        import cv2
    except ImportError:
        cv2 = None

    for idx, cls_id in enumerate(cls_ids):
        color = _PALETTE_BGR[int(cls_id) % len(_PALETTE_BGR)]
        x1, y1, x2, y2 = [int(round(float(v))) for v in boxes_xyxy[idx]]
        label = f"{class_names.get(int(cls_id), str(cls_id))} {float(confs[idx]):.2f}"
        if cv2 is not None:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            text_y = max(18, y1 - 6)
            cv2.putText(canvas, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        else:
            from PIL import Image, ImageDraw

            rgb_img = Image.fromarray(canvas[:, :, ::-1])
            draw = ImageDraw.Draw(rgb_img)
            rgb_color = (color[2], color[1], color[0])
            draw.rectangle([x1, y1, x2, y2], outline=rgb_color, width=2)
            draw.text((x1 + 2, max(0, y1 - 14)), label, fill=rgb_color)
            canvas = np.asarray(rgb_img, dtype=np.uint8)[:, :, ::-1].copy()

    return canvas


def _configure_arm_safe_net_options(net: object, num_threads: int) -> None:
    """Disable FP16/BF16/Vulkan and cap threads before loading NCNN model weights."""
    opt = getattr(net, "opt", None)
    if opt is None:
        return
    for attr, val in (
        ("use_vulkan_compute", False),
        ("use_fp16_packed", False),
        ("use_fp16_storage", False),
        ("use_fp16_arithmetic", False),
        ("use_bf16_storage", False),
        ("lightmode", True),
        ("openmp_blocktime", 0),
        ("num_threads", num_threads),
    ):
        if hasattr(opt, attr):
            setattr(opt, attr, val)


class NcnnSegmentEngine:
    """Thread-safe native NCNN runner configured strictly for Cortex-A72 stability."""

    def __init__(self, model_path: Path, imgsz: int, class_names: dict[int, str] | None = None) -> None:
        self._model_path = model_path
        self._imgsz = imgsz
        self._class_names = dict(class_names or {})
        self._net = None
        self._ncnn = None
        self._lock = threading.Lock()

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._class_names)

    def load(self) -> None:
        """Load NCNN param/bin with Cortex-A72 safe Net options."""
        param_file = self._model_path / "model.ncnn.param"
        bin_file = self._model_path / "model.ncnn.bin"
        if not param_file.is_file() or not bin_file.is_file():
            raise FileNotFoundError(f"File model NCNN tidak lengkap di {self._model_path}")

        meta_imgsz, meta_names = read_ncnn_metadata(self._model_path)
        if not self._class_names and meta_names:
            self._class_names = meta_names
        if meta_imgsz is not None and meta_imgsz != self._imgsz:
            raise ValueError(f"Ketidakcocokan imgsz metadata ({meta_imgsz}) vs config ({self._imgsz})")

        import ncnn

        raw_threads = os.environ.get("NUTRISCAN_NCNN_THREADS", os.environ.get("OMP_NUM_THREADS", "2"))
        try:
            threads = max(1, min(4, int(raw_threads)))
        except ValueError:
            threads = 2

        net = ncnn.Net()
        _configure_arm_safe_net_options(net, threads)
        if net.load_param(str(param_file)) != 0:
            raise RuntimeError(f"Gagal memuat parameter NCNN: {param_file}")
        if net.load_model(str(bin_file)) != 0:
            raise RuntimeError(f"Gagal memuat bobot NCNN: {bin_file}")

        self._ncnn = ncnn
        self._net = net
        logger.info("Native NcnnSegmentEngine siap (%s, imgsz=%d, threads=%d)", self._model_path.name, self._imgsz, threads)

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.7,
    ) -> NcnnPrediction:
        """Run full preprocess -> NCNN forward -> NumPy decode & mask generation."""
        if self._net is None or self._ncnn is None:
            raise RuntimeError("NcnnSegmentEngine belum dimuat; panggil load() lebih dulu")

        t0 = time.perf_counter()
        chw, gain, pad = preprocess_for_ncnn(frame, self._imgsz)
        t_pre = time.perf_counter()

        with self._lock:
            in_mat = self._ncnn.Mat(chw).clone()
            with self._net.create_extractor() as ex:
                ex.input("in0", in_mat)
                ret0, mat0 = ex.extract("out0")
                ret1, mat1 = ex.extract("out1")
                if ret0 != 0 or ret1 != 0:
                    raise RuntimeError(f"Ekstraksi NCNN gagal (out0={ret0}, out1={ret1})")
                out0 = np.array(mat0, dtype=np.float32, copy=True)
                out1 = np.array(mat1, dtype=np.float32, copy=True)
        t_inf = time.perf_counter()

        num_classes = len(self._class_names) or 39
        cls_ids, confs, boxes, masks = decode_ncnn_seg_outputs(
            out0=out0,
            out1=out1,
            orig_shape=frame.shape[:2],
            imgsz=self._imgsz,
            gain=gain,
            pad=pad,
            num_classes=num_classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        t_post = time.perf_counter()

        timing_ms = {
            "preprocess": round((t_pre - t0) * 1000.0, 1),
            "inference": round((t_inf - t_pre) * 1000.0, 1),
            "postprocess": round((t_post - t_inf) * 1000.0, 1),
            "total": round((t_post - t0) * 1000.0, 1),
        }
        return NcnnPrediction(
            cls_ids=cls_ids,
            confs=confs,
            boxes_xyxy=boxes,
            masks_bool=masks,
            class_names=self._class_names,
            timing_ms=timing_ms,
        )
