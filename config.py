"""
config.py
---------
설정 파일(config.json) 로드 및 저장을 담당한다.
"""

import json
import os
from dataclasses import dataclass, field

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
    detection_fps: int = 15


@dataclass
class MotionConfig:
    """움직임 감지 설정."""
    diff_threshold: int = 20          # 픽셀 밝기 차이 임계값 (낮을수록 예민)
    min_area: int = 800               # 움직임 최소 면적 (px²) - 노이즈 제거
    max_area: int = 40000             # 움직임 최대 면적 (px²) - 큰 배경 제거
    min_width: int = 20               # 바운딩박스 최소 너비
    max_width: int = 300              # 바운딩박스 최대 너비
    min_height: int = 20              # 바운딩박스 최소 높이
    max_height: int = 300             # 바운딩박스 최대 높이
    dilate_iterations: int = 3        # 팽창 반복 횟수 (윤곽선 연결)
    history_frames: int = 3           # 배경 비교에 사용할 프레임 수
    nms_overlap_threshold: float = 0.3


@dataclass
class TemplateConfig:
    """템플릿 매칭 설정 (현재 미사용, 나중을 위해 유지)."""
    match_threshold: float = 0.75
    nms_overlap_threshold: float = 0.3


@dataclass
class DetectionConfig:
    mode: str = "motion"              # "motion" 또는 "template"
    motion: MotionConfig = field(default_factory=MotionConfig)
    template: TemplateConfig = field(default_factory=TemplateConfig)


@dataclass
class TrackingConfig:
    tracking_distance: int = 100
    missing_frames_tolerance: int = 5


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
class ExclusionZone:
    """고정 제외 영역 (나무, 배경 오브젝트 등)."""
    name: str = ""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0


@dataclass
class PlayerExclusionConfig:
    """플레이어 위치 제외 영역 설정."""
    enabled: bool = True
    x: int = 972
    y: int = 390
    radius: int = 100


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
    player_exclusion: PlayerExclusionConfig = field(default_factory=PlayerExclusionConfig)
    exclusion_zones: list = field(default_factory=list)
    debug: DebugConfig = field(default_factory=DebugConfig)


def load_config(path: str = CONFIG_PATH) -> Config:
    if not os.path.exists(path):
        print(f"[Config] 설정 파일 없음 → 기본값 사용: {path}")
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
            d = data["detection"]
            motion_cfg = MotionConfig(**(d.get("motion", {})))
            template_cfg = TemplateConfig(**(d.get("template", {})))
            cfg.detection = DetectionConfig(
                mode=d.get("mode", "motion"),
                motion=motion_cfg,
                template=template_cfg
            )
        if "tracking" in data:
            cfg.tracking = TrackingConfig(**data["tracking"])
        if "attack" in data:
            cfg.attack = AttackConfig(**data["attack"])
        if "controller" in data:
            cfg.controller = ControllerConfig(**data["controller"])
        if "reference_point" in data:
            cfg.reference_point = ReferencePoint(**data["reference_point"])
        if "player_exclusion" in data:
            cfg.player_exclusion = PlayerExclusionConfig(**data["player_exclusion"])
        if "exclusion_zones" in data:
            cfg.exclusion_zones = [ExclusionZone(**z) for z in data["exclusion_zones"]]
        if "debug" in data:
            cfg.debug = DebugConfig(**data["debug"])

        print(f"[Config] 설정 로드 완료: {path}")
        return cfg

    except Exception as e:
        print(f"[Config] 설정 로드 실패: {e} → 기본값 사용")
        return Config()


def save_config(cfg: Config, path: str = CONFIG_PATH) -> None:
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
    cfg.roi.x = x
    cfg.roi.y = y
    cfg.roi.width = width
    cfg.roi.height = height
    save_config(cfg, path)
    print(f"[Config] ROI 저장: x={x}, y={y}, w={width}, h={height}")
