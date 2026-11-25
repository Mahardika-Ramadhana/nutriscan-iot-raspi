"""Typed JSON configuration loader, validator, and environment secret helper."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Any

from nutriscan.core.logger import get_logger

logger = get_logger("Config")

VALID_ENVIRONMENTS = ("development", "production", "mock")
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
VALID_MODEL_VARIANTS = ("416", "640")
VALID_CAMERA_BACKENDS = ("auto", "picamera2", "opencv", "mock")
VALID_LOADCELL_BACKENDS = ("auto", "adafruit_hx711", "lgpio", "rpi_gpio", "mock")
VALID_GAINS = (128, 64, 32)
VALID_HTTP_MODES = ("multipart", "json")


class ConfigError(ValueError):
    """Raised when config file loading or validation fails."""


@dataclass(frozen=True)
class DeviceConfig:
    id: str
    location: str
    log_level: str


@dataclass(frozen=True)
class ModelConfig:
    variant: str
    path: Path
    imgsz: int
    conf_threshold: float
    nutrition_csv: Path
    warmup: bool
    mock_mode: bool


@dataclass(frozen=True)
class CameraConfig:
    backend: str
    camera_ids: tuple[int, ...]
    width: int
    height: int
    use_mjpeg: bool
    warmup_s: float
    flush_frames: int
    mock_image_path: Path | None


@dataclass(frozen=True)
class HardwarePinsConfig:
    pin_loadcell_dt: int
    pin_loadcell_sck: int
    pin_tombol: int
    enable_physical_button: bool


@dataclass(frozen=True)
class LoadCellMockConfig:
    enabled: bool
    weights_g: tuple[float, ...]
    static_weight_g: float | None
    noise_g: float
    hold_seconds_per_step: float


@dataclass(frozen=True)
class LoadCellConfig:
    backend: str
    gain: int
    reference_unit: float
    tare_offset: float
    auto_tare_on_boot: bool
    tare_samples: int
    read_samples: int
    sample_interval_s: float
    deadband_g: float
    trigger_threshold_g: float
    removal_threshold_g: float
    stability_tolerance_g: float
    stability_duration_s: float
    removal_confirm_s: float
    cooldown_min_s: float
    mock: LoadCellMockConfig


@dataclass(frozen=True)
class StorageConfig:
    db_path: Path
    capture_dir: Path
    save_raw_image: bool
    save_annotated_image: bool
    send_empty_detections: bool


@dataclass(frozen=True)
class SupabaseSyncConfig:
    enabled: bool
    url: str
    table: str
    url_env: str
    key_env: str


@dataclass(frozen=True)
class HttpSyncConfig:
    enabled: bool
    endpoint: str
    mode: str
    include_image: bool
    timeout_s: float
    token_env: str
    verify_tls: bool


@dataclass(frozen=True)
class MqttSyncConfig:
    enabled: bool
    broker: str
    port: int
    topic: str
    qos: int
    use_tls: bool
    include_image_base64: bool
    username_env: str
    password_env: str
    connect_timeout_s: float


@dataclass(frozen=True)
class SyncConfig:
    flush_interval_s: float
    retry_base_delay_s: float
    retry_max_delay_s: float
    max_retries: int
    supabase: SupabaseSyncConfig
    http: HttpSyncConfig
    mqtt: MqttSyncConfig


@dataclass(frozen=True)
class WebConfig:
    enabled: bool
    host: str
    port: int
    api_key_env: str


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    config_path: Path
    environment: str
    device: DeviceConfig
    model: ModelConfig
    camera: CameraConfig
    hardware: HardwarePinsConfig
    loadcell: LoadCellConfig
    storage: StorageConfig
    sync: SyncConfig
    web: WebConfig


def default_project_root() -> Path:
    """Return the root directory of nutriscan-iot-raspi."""
    return Path(__file__).resolve().parents[3]


def load_dotenv_file(dotenv_path: Path) -> None:
    """Load key=value pairs from a .env file into os.environ without overwriting."""
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def get_secret(env_var: str, fallback: str = "") -> str:
    """Resolve a credential from environment variables, falling back only to non-empty config."""
    if env_var:
        env_val = os.environ.get(env_var, "").strip()
        if env_val:
            return env_val
    return fallback.strip()


def _resolve_project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name in ("data", "config"):
        return resolved.parent.parent
    return resolved.parent


def _resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (project_root / candidate).resolve()


def _require_mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"Bagian konfigurasi '{section}' harus berupa object JSON")
    return value


def _require_str(mapping: dict[str, Any], key: str, section: str) -> str:
    val = mapping.get(key)
    if not isinstance(val, str) or not val.strip():
        raise ConfigError(f"'{section}.{key}' harus berupa string tidak kosong")
    return val.strip()


def _require_int(mapping: dict[str, Any], key: str, section: str) -> int:
    val = mapping.get(key)
    if isinstance(val, bool) or not isinstance(val, int):
        raise ConfigError(f"'{section}.{key}' harus berupa integer")
    return int(val)


def _require_num(mapping: dict[str, Any], key: str, section: str) -> float:
    val = mapping.get(key)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ConfigError(f"'{section}.{key}' harus berupa angka")
    return float(val)


def _require_bool(mapping: dict[str, Any], key: str, section: str) -> bool:
    val = mapping.get(key)
    if not isinstance(val, bool):
        raise ConfigError(f"'{section}.{key}' harus berupa boolean")
    return val


def _parse_device(raw: dict[str, Any]) -> DeviceConfig:
    sec = _require_mapping(raw.get("device"), "device")
    log_level = _require_str(sec, "log_level", "device").upper()
    if log_level not in VALID_LOG_LEVELS:
        raise ConfigError(f"'device.log_level' harus salah satu dari {VALID_LOG_LEVELS}")
    return DeviceConfig(
        id=_require_str(sec, "id", "device"),
        location=_require_str(sec, "location", "device"),
        log_level=log_level,
    )


def _parse_model(raw: dict[str, Any], project_root: Path) -> ModelConfig:
    sec = _require_mapping(raw.get("model"), "model")
    variant = str(sec.get("variant", "640")).strip()
    path_override = sec.get("path")
    if path_override is None:
        if variant not in VALID_MODEL_VARIANTS:
            raise ConfigError(f"'model.variant' harus salah satu dari {VALID_MODEL_VARIANTS}")
        model_path = _resolve_path(project_root, f"models/best_{variant}_ncnn_model")
    else:
        if not isinstance(path_override, str) or not path_override.strip():
            raise ConfigError("'model.path' harus null atau path string tidak kosong")
        model_path = _resolve_path(project_root, path_override.strip())

    imgsz = _require_int(sec, "imgsz", "model")
    if imgsz <= 0:
        raise ConfigError("'model.imgsz' harus > 0")

    conf = _require_num(sec, "conf_threshold", "model")
    if not (0.0 < conf <= 1.0):
        raise ConfigError("'model.conf_threshold' harus berada di (0.0, 1.0]")

    csv_path = _resolve_path(project_root, _require_str(sec, "nutrition_csv", "model"))
    return ModelConfig(
        variant=variant,
        path=model_path,
        imgsz=imgsz,
        conf_threshold=conf,
        nutrition_csv=csv_path,
        warmup=_require_bool(sec, "warmup", "model"),
        mock_mode=_require_bool(sec, "mock_mode", "model"),
    )


def _parse_camera(raw: dict[str, Any], project_root: Path) -> CameraConfig:
    sec = _require_mapping(raw.get("camera"), "camera")
    backend = _require_str(sec, "backend", "camera").lower()
    if backend not in VALID_CAMERA_BACKENDS:
        raise ConfigError(f"'camera.backend' harus salah satu dari {VALID_CAMERA_BACKENDS}")

    raw_ids = sec.get("camera_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ConfigError("'camera.camera_ids' harus list integer tidak kosong")
    cam_ids: list[int] = []
    for cid in raw_ids:
        if isinstance(cid, bool) or not isinstance(cid, int) or cid < 0:
            raise ConfigError("Setiap item 'camera.camera_ids' harus integer >= 0")
        cam_ids.append(int(cid))

    width = _require_int(sec, "width", "camera")
    height = _require_int(sec, "height", "camera")
    warmup_s = _require_num(sec, "warmup_s", "camera")
    flush_frames = _require_int(sec, "flush_frames", "camera")
    if width <= 0 or height <= 0 or warmup_s < 0 or flush_frames < 0:
        raise ConfigError("Parameter resolusi/warmup/flush kamera tidak valid")

    raw_mock_img = sec.get("mock_image_path")
    mock_image_path: Path | None = None
    if raw_mock_img is not None:
        if not isinstance(raw_mock_img, str) or not raw_mock_img.strip():
            raise ConfigError("'camera.mock_image_path' harus null atau string tidak kosong")
        mock_image_path = _resolve_path(project_root, raw_mock_img.strip())

    return CameraConfig(
        backend=backend,
        camera_ids=tuple(cam_ids),
        width=width,
        height=height,
        use_mjpeg=_require_bool(sec, "use_mjpeg", "camera"),
        warmup_s=warmup_s,
        flush_frames=flush_frames,
        mock_image_path=mock_image_path,
    )


def _parse_hardware(raw: dict[str, Any]) -> HardwarePinsConfig:
    sec = _require_mapping(raw.get("hardware"), "hardware")
    dt = _require_int(sec, "pin_loadcell_dt", "hardware")
    sck = _require_int(sec, "pin_loadcell_sck", "hardware")
    btn = _require_int(sec, "pin_tombol", "hardware")
    for pin_name, pin_val in (("pin_loadcell_dt", dt), ("pin_loadcell_sck", sck), ("pin_tombol", btn)):
        if not (0 <= pin_val <= 27):
            raise ConfigError(f"'hardware.{pin_name}' harus BCM GPIO valid (0..27)")
    if len({dt, sck, btn}) != 3:
        raise ConfigError("'pin_loadcell_dt', 'pin_loadcell_sck', dan 'pin_tombol' tidak boleh sama")
    return HardwarePinsConfig(
        pin_loadcell_dt=dt,
        pin_loadcell_sck=sck,
        pin_tombol=btn,
        enable_physical_button=_require_bool(sec, "enable_physical_button", "hardware"),
    )


def _parse_loadcell(raw: dict[str, Any]) -> LoadCellConfig:
    sec = _require_mapping(raw.get("loadcell"), "loadcell")
    backend = _require_str(sec, "backend", "loadcell").lower()
    if backend not in VALID_LOADCELL_BACKENDS:
        raise ConfigError(f"'loadcell.backend' harus salah satu dari {VALID_LOADCELL_BACKENDS}")

    gain = _require_int(sec, "gain", "loadcell")
    if gain not in VALID_GAINS:
        raise ConfigError(f"'loadcell.gain' harus salah satu dari {VALID_GAINS}")

    reference_unit = _require_num(sec, "reference_unit", "loadcell")
    if reference_unit == 0:
        raise ConfigError("'loadcell.reference_unit' tidak boleh 0")

    tare_offset = _require_num(sec, "tare_offset", "loadcell")
    auto_tare = _require_bool(sec, "auto_tare_on_boot", "loadcell")
    tare_samples = _require_int(sec, "tare_samples", "loadcell")
    read_samples = _require_int(sec, "read_samples", "loadcell")
    if tare_samples < 1 or read_samples < 1:
        raise ConfigError("'tare_samples' dan 'read_samples' harus >= 1")

    sample_interval_s = _require_num(sec, "sample_interval_s", "loadcell")
    deadband_g = _require_num(sec, "deadband_g", "loadcell")
    trigger_threshold_g = _require_num(sec, "trigger_threshold_g", "loadcell")
    removal_threshold_g = _require_num(sec, "removal_threshold_g", "loadcell")
    stability_tolerance_g = _require_num(sec, "stability_tolerance_g", "loadcell")
    stability_duration_s = _require_num(sec, "stability_duration_s", "loadcell")
    removal_confirm_s = _require_num(sec, "removal_confirm_s", "loadcell")
    cooldown_min_s = _require_num(sec, "cooldown_min_s", "loadcell")

    if sample_interval_s <= 0 or stability_duration_s <= 0:
        raise ConfigError("'sample_interval_s' dan 'stability_duration_s' harus > 0")
    if deadband_g < 0 or stability_tolerance_g < 0 or removal_confirm_s < 0 or cooldown_min_s < 0:
        raise ConfigError("Nilai deadband/tolerance/cooldown loadcell harus >= 0")
    if trigger_threshold_g <= 0:
        raise ConfigError("'loadcell.trigger_threshold_g' harus > 0")
    if not (0 <= removal_threshold_g < trigger_threshold_g):
        raise ConfigError("'loadcell.removal_threshold_g' harus di [0, trigger_threshold_g)")

    mock_sec = _require_mapping(sec.get("mock", {}), "loadcell.mock")
    raw_weights = mock_sec.get("weights_g", [0.0, 350.0, 0.0])
    if not isinstance(raw_weights, list) or not raw_weights:
        raise ConfigError("'loadcell.mock.weights_g' harus list angka tidak kosong")
    weights: list[float] = []
    for item in raw_weights:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConfigError("'loadcell.mock.weights_g' harus berisi angka")
        weights.append(float(item))

    raw_static = mock_sec.get("static_weight_g")
    static_weight: float | None = None
    if raw_static is not None:
        if isinstance(raw_static, bool) or not isinstance(raw_static, (int, float)):
            raise ConfigError("'loadcell.mock.static_weight_g' harus null atau angka")
        static_weight = float(raw_static)

    noise_g = float(mock_sec.get("noise_g", 0.0))
    hold_s = float(mock_sec.get("hold_seconds_per_step", 2.0))
    if noise_g < 0 or hold_s <= 0:
        raise ConfigError("'loadcell.mock.noise_g' >= 0 dan 'hold_seconds_per_step' > 0 wajib terpenuhi")

    return LoadCellConfig(
        backend=backend,
        gain=gain,
        reference_unit=reference_unit,
        tare_offset=tare_offset,
        auto_tare_on_boot=auto_tare,
        tare_samples=tare_samples,
        read_samples=read_samples,
        sample_interval_s=sample_interval_s,
        deadband_g=deadband_g,
        trigger_threshold_g=trigger_threshold_g,
        removal_threshold_g=removal_threshold_g,
        stability_tolerance_g=stability_tolerance_g,
        stability_duration_s=stability_duration_s,
        removal_confirm_s=removal_confirm_s,
        cooldown_min_s=cooldown_min_s,
        mock=LoadCellMockConfig(
            enabled=bool(mock_sec.get("enabled", False)),
            weights_g=tuple(weights),
            static_weight_g=static_weight,
            noise_g=noise_g,
            hold_seconds_per_step=hold_s,
        ),
    )


def _parse_storage(raw: dict[str, Any], project_root: Path) -> StorageConfig:
    sec = _require_mapping(raw.get("storage"), "storage")
    return StorageConfig(
        db_path=_resolve_path(project_root, _require_str(sec, "db_path", "storage")),
        capture_dir=_resolve_path(project_root, _require_str(sec, "capture_dir", "storage")),
        save_raw_image=_require_bool(sec, "save_raw_image", "storage"),
        save_annotated_image=_require_bool(sec, "save_annotated_image", "storage"),
        send_empty_detections=_require_bool(sec, "send_empty_detections", "storage"),
    )


def _parse_sync(raw: dict[str, Any]) -> SyncConfig:
    sec = _require_mapping(raw.get("sync"), "sync")
    supa_sec = _require_mapping(sec.get("supabase"), "sync.supabase")
    http_sec = _require_mapping(sec.get("http"), "sync.http")
    mqtt_sec = _require_mapping(sec.get("mqtt"), "sync.mqtt")

    flush_interval_s = _require_num(sec, "flush_interval_s", "sync")
    retry_base_delay_s = _require_num(sec, "retry_base_delay_s", "sync")
    retry_max_delay_s = _require_num(sec, "retry_max_delay_s", "sync")
    max_retries = _require_int(sec, "max_retries", "sync")
    if flush_interval_s <= 0 or retry_base_delay_s <= 0 or retry_max_delay_s < retry_base_delay_s or max_retries < 0:
        raise ConfigError("Parameter interval/retry sync tidak valid")

    supa_cfg = SupabaseSyncConfig(
        enabled=_require_bool(supa_sec, "enabled", "sync.supabase"),
        url=str(supa_sec.get("url", "")).strip(),
        table=_require_str(supa_sec, "table", "sync.supabase"),
        url_env=_require_str(supa_sec, "url_env", "sync.supabase"),
        key_env=_require_str(supa_sec, "key_env", "sync.supabase"),
    )

    http_mode = _require_str(http_sec, "mode", "sync.http").lower()
    if http_mode not in VALID_HTTP_MODES:
        raise ConfigError(f"'sync.http.mode' harus salah satu dari {VALID_HTTP_MODES}")
    http_timeout_s = _require_num(http_sec, "timeout_s", "sync.http")
    if http_timeout_s <= 0:
        raise ConfigError("'sync.http.timeout_s' harus > 0")
    http_cfg = HttpSyncConfig(
        enabled=_require_bool(http_sec, "enabled", "sync.http"),
        endpoint=_require_str(http_sec, "endpoint", "sync.http"),
        mode=http_mode,
        include_image=_require_bool(http_sec, "include_image", "sync.http"),
        timeout_s=http_timeout_s,
        token_env=_require_str(http_sec, "token_env", "sync.http"),
        verify_tls=_require_bool(http_sec, "verify_tls", "sync.http"),
    )

    mqtt_port = _require_int(mqtt_sec, "port", "sync.mqtt")
    mqtt_qos = _require_int(mqtt_sec, "qos", "sync.mqtt")
    mqtt_timeout = _require_num(mqtt_sec, "connect_timeout_s", "sync.mqtt")
    if not (1 <= mqtt_port <= 65535):
        raise ConfigError("'sync.mqtt.port' harus di antara 1..65535")
    if mqtt_qos not in (0, 1, 2):
        raise ConfigError("'sync.mqtt.qos' harus 0, 1, atau 2")
    if mqtt_timeout <= 0:
        raise ConfigError("'sync.mqtt.connect_timeout_s' harus > 0")
    mqtt_cfg = MqttSyncConfig(
        enabled=_require_bool(mqtt_sec, "enabled", "sync.mqtt"),
        broker=_require_str(mqtt_sec, "broker", "sync.mqtt"),
        port=mqtt_port,
        topic=_require_str(mqtt_sec, "topic", "sync.mqtt"),
        qos=mqtt_qos,
        use_tls=_require_bool(mqtt_sec, "use_tls", "sync.mqtt"),
        include_image_base64=_require_bool(mqtt_sec, "include_image_base64", "sync.mqtt"),
        username_env=_require_str(mqtt_sec, "username_env", "sync.mqtt"),
        password_env=_require_str(mqtt_sec, "password_env", "sync.mqtt"),
        connect_timeout_s=mqtt_timeout,
    )

    return SyncConfig(
        flush_interval_s=flush_interval_s,
        retry_base_delay_s=retry_base_delay_s,
        retry_max_delay_s=retry_max_delay_s,
        max_retries=max_retries,
        supabase=supa_cfg,
        http=http_cfg,
        mqtt=mqtt_cfg,
    )


def _parse_web(raw: dict[str, Any]) -> WebConfig:
    sec = _require_mapping(raw.get("web"), "web")
    port = _require_int(sec, "port", "web")
    if not (1 <= port <= 65535):
        raise ConfigError("'web.port' harus di antara 1..65535")
    return WebConfig(
        enabled=_require_bool(sec, "enabled", "web"),
        host=_require_str(sec, "host", "web"),
        port=port,
        api_key_env=_require_str(sec, "api_key_env", "web"),
    )


def load_config(
    config_path: Path | str | None = None,
    allow_example_fallback: bool = False,
) -> AppConfig:
    """Load, validate, and resolve paths from JSON configuration."""
    if config_path is None:
        root = default_project_root()
        target = root / "data" / "config.json"
    else:
        target = Path(config_path).resolve()
        root = _resolve_project_root(target)

    if not target.is_file():
        example_path = target.parent / "config.example.json"
        if allow_example_fallback and example_path.is_file():
            logger.warning(
                "File %s belum ada; memakai fallback sementara %s.",
                target,
                example_path,
            )
            target = example_path
        else:
            raise ConfigError(
                f"File konfigurasi tidak ditemukan: {target}. "
                "Salin data/config.example.json menjadi data/config.json terlebih dahulu."
            )

    load_dotenv_file(root / ".env")

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Format JSON konfigurasi rusak di {target}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Root konfigurasi {target} harus berupa JSON object")

    environment = str(raw.get("environment", "development")).strip().lower()
    if environment not in VALID_ENVIRONMENTS:
        raise ConfigError(f"'environment' harus salah satu dari {VALID_ENVIRONMENTS}")

    return AppConfig(
        project_root=root,
        config_path=target,
        environment=environment,
        device=_parse_device(raw),
        model=_parse_model(raw, root),
        camera=_parse_camera(raw, root),
        hardware=_parse_hardware(raw),
        loadcell=_parse_loadcell(raw),
        storage=_parse_storage(raw, root),
        sync=_parse_sync(raw),
        web=_parse_web(raw),
    )


def with_mock_overrides(cfg: AppConfig) -> AppConfig:
    """Return a new immutable AppConfig configured for hardware-free simulation."""
    return replace(
        cfg,
        environment="mock",
        model=replace(cfg.model, mock_mode=True),
        camera=replace(cfg.camera, backend="mock"),
        hardware=replace(cfg.hardware, enable_physical_button=False),
        loadcell=replace(
            cfg.loadcell,
            backend="mock",
            mock=replace(cfg.loadcell.mock, enabled=True),
        ),
    )
