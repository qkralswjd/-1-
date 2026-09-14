"""
config.py
---------
설정 파일(config.json) 로드 및 저장을 담당한다.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List

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
    diff_threshold: int = 35
    min_area: int = 1500
    max_area: int = 40000
    min_width: int = 30
    max_width: int = 250
    min_height: int = 30
    max_height: int = 250
    dilate_iterations: int = 2
    history_frames: int = 5
    nms_overlap_threshold: float = 0.3


@dataclass
class TemplateConfig:
    match_threshold: float = 0.72
    nms_overlap_threshold: float = 0.3
    scales: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    fallback_to_motion: bool = False
    padding: int = 40


@dataclass
class DetectionConfig:
    mode: str = "hybrid"
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
    name: str = ""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0


@dataclass
class PlayerExclusionConfig:
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
class HuntConfig:
    """자동 사냥 설정."""
    # 이동 클릭 가능 범위 (화면 좌표)
    move_x_min: int = 600
    move_x_max: int = 1300
    move_y_min: int = 300
    move_y_max: int = 600
    # 타이밍
    move_wait_sec: float = 1.5          # 이동 후 멈춤 대기 시간
    attack_cooldown_sec: float = 0.5    # 공격 쿨다운
    no_monster_timeout_sec: float = 3.0 # 몬스터 없으면 N초 후 이동
    # 드래그 설정 (자동공격용)
    drag_dx: int = 5                    # 드래그 x 거리
    drag_dy: int = 0                    # 드래그 y 거리
    drag_hold_ms: int = 80              # 클릭 유지 시간(ms)


@dataclass
class MovementConfig:
    """이동/멈춤 감지 설정."""
    move_threshold: float = 20.0     # 이 값 이상이면 이동 중 판정
    still_frames_required: int = 3   # N프레임 연속 멈춤이어야 탐지 시작
    sample_scale: float = 0.25       # 비교 프레임 축소 비율 (빠른 연산)


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
    movement: MovementConfig = field(default_factory=MovementConfig)
    hunt: HuntConfig = field(default_factory=HuntConfig)


def load_config(path: str = CONFIG_PATH) -> Config:
    if not os.path.exists(path):
        print(f"[Config] 파일 없음 → 기본값 사용")
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
            cfg.detection = DetectionConfig(
                mode=d.get("mode", "hybrid"),
                motion=MotionConfig(**d.get("motion", {})),
                template=TemplateConfig(**d.get("template", {})),
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
        if "movement" in data:
            cfg.movement = MovementConfig(**data["movement"])
        if "hunt" in data:
            cfg.hunt = HuntConfig(**data["hunt"])
        print(f"[Config] 로드 완료: mode={cfg.detection.mode}, "
              f"threshold={cfg.detection.motion.diff_threshold}, "
              f"match={cfg.detection.template.match_threshold}")
        return cfg
    except Exception as e:
        print(f"[Config] 로드 실패: {e} → 기본값 사용")
        import traceback; traceback.print_exc()
        return Config()


def save_config(cfg: Config, path: str = CONFIG_PATH) -> None:
    import dataclasses
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(cfg), f, indent=4, ensure_ascii=False)
        print(f"[Config] 저장 완료")
    except Exception as e:
        print(f"[Config] 저장 실패: {e}")


def save_roi(cfg: Config, x: int, y: int, width: int, height: int,
             path: str = CONFIG_PATH) -> None:
    cfg.roi.x, cfg.roi.y = x, y
    cfg.roi.width, cfg.roi.height = width, height
    save_config(cfg, path)
