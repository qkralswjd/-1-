"""
config.py
---------
설정 파일(config.json) 로드 및 저장을 담당한다.
런타임 중 ROI 변경 등을 저장할 때도 이 모듈을 사용한다.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


@dataclass
class ROIConfig:
    x: int = 100
    y: int = 200
    width: int = 800
    height: int = 500


@dataclass
class CaptureConfig:
    capture_fps: int = 30
    detection_fps: int = 10


@dataclass
class DetectionConfig:
    match_threshold: float = 0.75
    nms_overlap_threshold: float = 0.3


@dataclass
class TrackingConfig:
    tracking_distance: int = 80
    missing_frames_tolerance: int = 3


@dataclass
class AttackConfig:
    attack_cooldown: float = 0.5
    click_duration: float = 0.05


@dataclass
class ControllerConfig:
    type: str = "pico"
    port: str = "COM3"
    baudrate: int = 115200
    enabled: bool = False


@dataclass
class ReferencePoint:
    x: int = 960
    y: int = 540


@dataclass
class DebugConfig:
    show_confidence: bool = True
    show_ids: bool = True
    show_fps: bool = True
    show_roi: bool = True


@dataclass
class Config:
    roi: ROIConfig = field(default_factory=ROIConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    reference_point: ReferencePoint = field(default_factory=ReferencePoint)
    debug: DebugConfig = field(default_factory=DebugConfig)


def _dict_to_dataclass(cls, data: dict):
    """dict를 dataclass로 재귀 변환한다."""
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return data
    fieldtypes = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, val in data.items():
        if key in fieldtypes:
            # 타입 힌트가 문자열인 경우 eval 없이 처리
            field_cls = cls.__dataclass_fields__[key].default_factory
            # 중첩 dataclass 처리
            import sys
            module = sys.modules[cls.__module__]
            nested_cls = getattr(module, cls.__dataclass_fields__[key].type
                                 if isinstance(cls.__dataclass_fields__[key].type, str)
                                 else cls.__dataclass_fields__[key].type.__name__,
                                 None)
            if nested_cls and hasattr(nested_cls, '__dataclass_fields__') and isinstance(val, dict):
                kwargs[key] = _dict_to_dataclass(nested_cls, val)
            else:
                kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str = CONFIG_PATH) -> Config:
    """JSON 파일에서 설정을 로드한다. 파일이 없으면 기본값을 반환한다."""
    if not os.path.exists(path):
        print(f"[Config] 설정 파일을 찾을 수 없습니다. 기본값을 사용합니다: {path}")
        return Config()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = Config()
        if "roi" in data:
            cfg.roi = ROIConfig(**data["roi"])
        if "capture" in data:
            cfg.capture = CaptureConfig(**data["capture"])
        if "detection" in data:
            cfg.detection = DetectionConfig(**data["detection"])
        if "tracking" in data:
            cfg.tracking = TrackingConfig(**data["tracking"])
        if "attack" in data:
            cfg.attack = AttackConfig(**data["attack"])
        if "controller" in data:
            cfg.controller = ControllerConfig(**data["controller"])
        if "reference_point" in data:
            cfg.reference_point = ReferencePoint(**data["reference_point"])
        if "debug" in data:
            cfg.debug = DebugConfig(**data["debug"])

        print(f"[Config] 설정 로드 완료: {path}")
        return cfg

    except Exception as e:
        print(f"[Config] 설정 로드 실패: {e} → 기본값 사용")
        return Config()


def save_config(cfg: Config, path: str = CONFIG_PATH) -> None:
    """현재 설정을 JSON 파일에 저장한다."""
    import dataclasses
    data = dataclasses.asdict(cfg)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[Config] 설정 저장 완료: {path}")
    except Exception as e:
        print(f"[Config] 설정 저장 실패: {e}")


def save_roi(cfg: Config, x: int, y: int, width: int, height: int,
             path: str = CONFIG_PATH) -> None:
    """ROI만 업데이트해서 저장한다. 마우스 드래그로 ROI 지정 후 호출한다."""
    cfg.roi.x = x
    cfg.roi.y = y
    cfg.roi.width = width
    cfg.roi.height = height
    save_config(cfg, path)
    print(f"[Config] ROI 저장: x={x}, y={y}, w={width}, h={height}")
