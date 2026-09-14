"""
detector.py
-----------
세 가지 감지 모드를 지원한다.

  1. motion   : 배경 차분(Motion Detection) + 크기 필터
  2. template : 전체 ROI에서 템플릿 매칭
  3. hybrid   : Motion으로 후보 영역 추출 → Template로 확인 (권장)

동작 원리 (hybrid 기준):
  1. 최근 N개 프레임 평균 → 배경
  2. 현재 프레임과 배경 diff → 이진화 → dilate → contour
  3. 크기 필터로 후보 ROI 추출
  4. 각 후보 ROI에서 모든 템플릿 매칭 실행
  5. match_threshold 이상인 것만 최종 탐지로 채택
  6. NMS로 중복 제거
"""

import cv2
import numpy as np
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque


@dataclass
class MonsterDetection:
    """탐지 결과 하나. 모든 좌표는 원본 화면(전체 화면) 기준."""
    id: int
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    confidence: float
    template_name: str = "motion"


# ======================================================================
# 공통 유틸
# ======================================================================

def _calc_iou(a: MonsterDetection, b: MonsterDetection) -> float:
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.width, b.y + b.height
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _apply_nms_list(detections: List[MonsterDetection],
                    nms_threshold: float) -> List[MonsterDetection]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[MonsterDetection] = []
    for det in detections:
        if not any(_calc_iou(det, k) > nms_threshold for k in kept):
            kept.append(det)
    return kept


def _filter_zones(detections: List[MonsterDetection], zones: list) -> List[MonsterDetection]:
    if not zones:
        return detections
    filtered = []
    for det in detections:
        in_zone = any(
            zone.x <= det.center_x <= zone.x + zone.width and
            zone.y <= det.center_y <= zone.y + zone.height
            for zone in zones
        )
        if not in_zone:
            filtered.append(det)
    return filtered


def _filter_player(detections: List[MonsterDetection],
                   px: int, py: int, radius: int) -> List[MonsterDetection]:
    import math
    return [d for d in detections
            if math.sqrt((d.center_x - px)**2 + (d.center_y - py)**2) > radius]


# ======================================================================
# Motion Detector
# ======================================================================

class MotionDetector:
    """움직임 감지 기반 탐지기."""

    def __init__(self, diff_threshold=30, min_area=1500, max_area=40000,
                 min_width=30, max_width=250, min_height=30, max_height=250,
                 dilate_iterations=2, history_frames=5, nms_overlap_threshold=0.3):
        self._diff_threshold = diff_threshold
        self._min_area = min_area
        self._max_area = max_area
        self._min_width = min_width
        self._max_width = max_width
        self._min_height = min_height
        self._max_height = max_height
        self._dilate_iterations = dilate_iterations
        self._nms_threshold = nms_overlap_threshold
        self._history: deque = deque(maxlen=history_frames)
        self._background: Optional[np.ndarray] = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._fps_ticks: List[float] = []
        self._detection_fps: float = 0.0
        print(f"[MotionDetector] threshold={diff_threshold}, "
              f"area={min_area}~{max_area}, size={min_width}x{min_height}~{max_width}x{max_height}")

    def detect(self, roi_frame, roi_offset_x, roi_offset_y) -> List[MonsterDetection]:
        candidates = self._get_candidates(roi_frame, roi_offset_x, roi_offset_y)
        result = _apply_nms_list(candidates, self._nms_threshold)
        self._tick_fps()
        return result

    def get_candidate_regions(self, roi_frame, roi_offset_x, roi_offset_y) -> List[MonsterDetection]:
        """HybridDetector 내부에서 호출 - NMS 없이 후보만 반환."""
        return self._get_candidates(roi_frame, roi_offset_x, roi_offset_y)

    def _get_candidates(self, roi_frame, roi_offset_x, roi_offset_y) -> List[MonsterDetection]:
        if roi_frame is None:
            return []
        gray = cv2.GaussianBlur(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        self._history.append(gray.astype(np.float32))
        if len(self._history) < 2:
            return []
        self._background = np.mean(np.stack(list(self._history), axis=0), axis=0).astype(np.uint8)
        diff = cv2.absdiff(gray, self._background)
        _, thresh = cv2.threshold(diff, self._diff_threshold, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, self._kernel, iterations=self._dilate_iterations)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for c in contours:
            d = self._contour_to_det(c, roi_offset_x, roi_offset_y)
            if d:
                result.append(d)
        return result

    def _contour_to_det(self, contour, ox, oy) -> Optional[MonsterDetection]:
        area = cv2.contourArea(contour)
        if not (self._min_area <= area <= self._max_area):
            return None
        rx, ry, rw, rh = cv2.boundingRect(contour)
        if not (self._min_width <= rw <= self._max_width):
            return None
        if not (self._min_height <= rh <= self._max_height):
            return None
        sx, sy = rx + ox, ry + oy
        return MonsterDetection(
            id=-1, x=sx, y=sy, width=rw, height=rh,
            center_x=sx + rw // 2, center_y=sy + rh // 2,
            confidence=round(min(area / self._max_area, 1.0), 3),
            template_name="motion"
        )

    def reset(self):
        self._history.clear()
        self._background = None
        print("[MotionDetector] 배경 초기화")

    def filter_zones(self, detections, zones):
        return _filter_zones(detections, zones)

    def filter_player(self, detections, px, py, radius):
        return _filter_player(detections, px, py, radius)

    def _tick_fps(self):
        now = time.time()
        self._fps_ticks.append(now)
        if len(self._fps_ticks) > 30:
            self._fps_ticks.pop(0)
        if len(self._fps_ticks) >= 2:
            elapsed = self._fps_ticks[-1] - self._fps_ticks[0]
            if elapsed > 0:
                self._detection_fps = round((len(self._fps_ticks) - 1) / elapsed, 1)

    @property
    def detection_fps(self):
        return self._detection_fps

    def get_debug_mask(self, roi_frame):
        if roi_frame is None or self._background is None:
            return None
        gray = cv2.GaussianBlur(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        diff = cv2.absdiff(gray, self._background)
        _, thresh = cv2.threshold(diff, self._diff_threshold, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, self._kernel, iterations=self._dilate_iterations)
        return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)


# ======================================================================
# Template Detector
# ======================================================================

class TemplateDetector:
    """템플릿 매칭 탐지기."""

    def __init__(self, templates_dir: str, match_threshold=0.70,
                 nms_overlap_threshold=0.3, scales=(0.8, 1.0, 1.2)):
        self._threshold = match_threshold
        self._nms_threshold = nms_overlap_threshold
        self._scales = scales
        self._templates: List[Tuple[str, np.ndarray]] = []
        self._fps_ticks: List[float] = []
        self._detection_fps: float = 0.0
        self._load_templates(templates_dir)

    def _load_templates(self, templates_dir: str):
        self._templates.clear()
        if not os.path.isdir(templates_dir):
            print(f"[TemplateDetector] 폴더 없음: {templates_dir}")
            return
        for fname in sorted(os.listdir(templates_dir)):
            if not fname.lower().endswith(".png"):
                continue
            img = cv2.imdecode(
                np.fromfile(os.path.join(templates_dir, fname), dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self._templates.append((fname, img))
        print(f"[TemplateDetector] {len(self._templates)}개 로드")

    def reload_templates(self, templates_dir: str):
        self._load_templates(templates_dir)

    @property
    def template_count(self):
        return len(self._templates)

    def detect(self, roi_frame, roi_offset_x, roi_offset_y) -> List[MonsterDetection]:
        if roi_frame is None or not self._templates:
            return []
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        all_dets = []
        for tname, tmpl in self._templates:
            for scale in self._scales:
                all_dets.extend(self._match(gray_roi, tmpl, tname, scale,
                                            roi_offset_x, roi_offset_y))
        result = _apply_nms_list(all_dets, self._nms_threshold)
        self._tick_fps()
        return result

    def match_in_region(self, roi_frame, region_x, region_y, region_w, region_h,
                        roi_offset_x, roi_offset_y, padding=40) -> List[MonsterDetection]:
        """motion 후보 영역 주변에서만 템플릿 매칭 실행."""
        if roi_frame is None or not self._templates:
            return []
        h_roi, w_roi = roi_frame.shape[:2]
        x1 = max(0, region_x - roi_offset_x - padding)
        y1 = max(0, region_y - roi_offset_y - padding)
        x2 = min(w_roi, region_x - roi_offset_x + region_w + padding)
        y2 = min(h_roi, region_y - roi_offset_y + region_h + padding)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return []
        crop = roi_frame[y1:y2, x1:x2]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        cx_off = roi_offset_x + x1
        cy_off = roi_offset_y + y1
        all_dets = []
        for tname, tmpl in self._templates:
            for scale in self._scales:
                all_dets.extend(self._match(gray_crop, tmpl, tname, scale, cx_off, cy_off))
        return _apply_nms_list(all_dets, self._nms_threshold)

    def _match(self, gray_src, tmpl_gray, tname, scale, ox, oy) -> List[MonsterDetection]:
        th, tw = tmpl_gray.shape[:2]
        sh, sw = gray_src.shape[:2]
        nw, nh = max(1, int(tw * scale)), max(1, int(th * scale))
        if nw > sw or nh > sh:
            return []
        tmpl = cv2.resize(tmpl_gray, (nw, nh)) if scale != 1.0 else tmpl_gray
        try:
            res = cv2.matchTemplate(gray_src, tmpl, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            return []
        locs = np.where(res >= self._threshold)
        dets = []
        for ry, rx in zip(*locs):
            conf = float(res[ry, rx])
            sx, sy = int(rx) + ox, int(ry) + oy
            dets.append(MonsterDetection(
                id=-1, x=sx, y=sy, width=nw, height=nh,
                center_x=sx + nw // 2, center_y=sy + nh // 2,
                confidence=round(conf, 3), template_name=tname
            ))
        return dets

    def reset(self):
        pass

    def filter_zones(self, detections, zones):
        return _filter_zones(detections, zones)

    def filter_player(self, detections, px, py, radius):
        return _filter_player(detections, px, py, radius)

    def _tick_fps(self):
        now = time.time()
        self._fps_ticks.append(now)
        if len(self._fps_ticks) > 30:
            self._fps_ticks.pop(0)
        if len(self._fps_ticks) >= 2:
            elapsed = self._fps_ticks[-1] - self._fps_ticks[0]
            if elapsed > 0:
                self._detection_fps = round((len(self._fps_ticks) - 1) / elapsed, 1)

    @property
    def detection_fps(self):
        return self._detection_fps


# ======================================================================
# Hybrid Detector
# ======================================================================

class HybridDetector:
    """
    하이브리드 탐지기: Motion으로 후보 추출 → Template로 확인.

    fallback_to_motion=False: 템플릿 확인된 것만 탐지 (나무 오탐 완전 제거)
    fallback_to_motion=True : 템플릿 실패해도 motion 결과 사용
    """

    def __init__(self, motion: MotionDetector, template: TemplateDetector,
                 nms_overlap_threshold=0.3, fallback_to_motion=False, padding=40):
        self._motion = motion
        self._template = template
        self._nms_threshold = nms_overlap_threshold
        self._fallback = fallback_to_motion
        self._padding = padding
        self._fps_ticks: List[float] = []
        self._detection_fps: float = 0.0
        print(f"[HybridDetector] fallback={fallback_to_motion}, "
              f"padding={padding}, templates={template.template_count}개")

    def detect(self, roi_frame, roi_offset_x, roi_offset_y) -> List[MonsterDetection]:
        if roi_frame is None:
            return []

        # 1. Motion 후보 추출
        candidates = self._motion.get_candidate_regions(roi_frame, roi_offset_x, roi_offset_y)
        if not candidates:
            self._tick_fps()
            return []

        confirmed = []
        motion_only = []

        # 2. 각 후보에서 Template 확인
        if self._template.template_count > 0:
            for cand in candidates:
                tmpl_dets = self._template.match_in_region(
                    roi_frame,
                    region_x=cand.x, region_y=cand.y,
                    region_w=cand.width, region_h=cand.height,
                    roi_offset_x=roi_offset_x, roi_offset_y=roi_offset_y,
                    padding=self._padding
                )
                if tmpl_dets:
                    confirmed.extend(tmpl_dets)
                else:
                    motion_only.append(cand)
        else:
            motion_only = candidates

        # 3. fallback 처리
        all_dets = confirmed[:]
        if self._fallback and not confirmed:
            all_dets = motion_only

        result = _apply_nms_list(all_dets, self._nms_threshold)
        self._tick_fps()
        return result

    def reset(self):
        self._motion.reset()

    def filter_zones(self, detections, zones):
        return _filter_zones(detections, zones)

    def filter_player(self, detections, px, py, radius):
        return _filter_player(detections, px, py, radius)

    def reload_templates(self, templates_dir: str):
        self._template.reload_templates(templates_dir)

    def _tick_fps(self):
        now = time.time()
        self._fps_ticks.append(now)
        if len(self._fps_ticks) > 30:
            self._fps_ticks.pop(0)
        if len(self._fps_ticks) >= 2:
            elapsed = self._fps_ticks[-1] - self._fps_ticks[0]
            if elapsed > 0:
                self._detection_fps = round((len(self._fps_ticks) - 1) / elapsed, 1)

    @property
    def detection_fps(self):
        return self._detection_fps


# ======================================================================
# Factory
# ======================================================================

def make_detector(cfg):
    """config.detection.mode에 따라 detector 생성."""
    m = cfg.detection.motion
    t = cfg.detection.template
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")

    motion = MotionDetector(
        diff_threshold=m.diff_threshold,
        min_area=m.min_area, max_area=m.max_area,
        min_width=m.min_width, max_width=m.max_width,
        min_height=m.min_height, max_height=m.max_height,
        dilate_iterations=m.dilate_iterations,
        history_frames=m.history_frames,
        nms_overlap_threshold=m.nms_overlap_threshold,
    )

    mode = cfg.detection.mode

    if mode == "motion":
        print(f"[Detector] 모드: Motion Only")
        return motion

    tmpl = TemplateDetector(
        templates_dir=templates_dir,
        match_threshold=t.match_threshold,
        nms_overlap_threshold=t.nms_overlap_threshold,
        scales=tuple(t.scales),
    )

    if mode == "template":
        print(f"[Detector] 모드: Template Only")
        return tmpl

    elif mode == "hybrid":
        print(f"[Detector] 모드: Hybrid (Motion + Template)")
        return HybridDetector(
            motion=motion, template=tmpl,
            nms_overlap_threshold=m.nms_overlap_threshold,
            fallback_to_motion=t.fallback_to_motion,
            padding=t.padding,
        )

    print(f"[Detector] 알 수 없는 모드 '{mode}' → Motion Only")
    return motion
