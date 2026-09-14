"""
detector.py
-----------
움직임 감지(Motion Detection) + 크기 필터로 몬스터를 탐지한다.

동작 원리:
  1. 최근 N개 프레임의 평균을 배경으로 사용 (BackgroundAverager)
  2. 현재 프레임과 배경의 차이(diff)를 계산
  3. 임계값 이상인 픽셀을 움직임 영역으로 판단
  4. 윤곽선(contour) 추출
  5. 크기 필터: 너무 작거나 너무 큰 것 제거 → 노이즈 & 배경 제거
  6. NMS로 겹치는 박스 제거
  7. 잠깐 멈춘 몬스터 보정: 직전 프레임 탐지 결과 재활용 (missing_frames 연계)

입력: ROI 이미지 (numpy BGR array)
출력: MonsterDetection 리스트 (id=-1, confidence=면적 기반)
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Optional
from collections import deque


@dataclass
class MonsterDetection:
    """
    탐지 결과 하나.
    모든 좌표는 원본 화면(전체 화면) 기준이다.
    """
    id: int
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    confidence: float        # 움직임 면적 기반 (0.0~1.0 정규화)
    template_name: str = "motion"


class MotionDetector:
    """
    움직임 감지 기반 몬스터 탐지기.

    배경 = 최근 history_frames 프레임의 평균 (rolling average)
    diff = 현재 프레임 - 배경
    → threshold → dilate → contours → 크기 필터 → NMS
    """

    def __init__(self,
                 diff_threshold: int = 20,
                 min_area: int = 800,
                 max_area: int = 40000,
                 min_width: int = 20,
                 max_width: int = 300,
                 min_height: int = 20,
                 max_height: int = 300,
                 dilate_iterations: int = 3,
                 history_frames: int = 3,
                 nms_overlap_threshold: float = 0.3):

        self._diff_threshold = diff_threshold
        self._min_area = min_area
        self._max_area = max_area
        self._min_width = min_width
        self._max_width = max_width
        self._min_height = min_height
        self._max_height = max_height
        self._dilate_iterations = dilate_iterations
        self._nms_overlap_threshold = nms_overlap_threshold

        # 배경 프레임 히스토리 (grayscale)
        self._history: deque = deque(maxlen=history_frames)
        self._background: Optional[np.ndarray] = None

        # Detection FPS
        self._fps_ticks: List[float] = []
        self._detection_fps: float = 0.0

        # 팽창 커널
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5))

        print(f"[Detector] MotionDetector 초기화")
        print(f"  diff_threshold={diff_threshold}, "
              f"area={min_area}~{max_area}, "
              f"size={min_width}x{min_height}~{max_width}x{max_height}")

    # ------------------------------------------------------------------
    # 메인 탐지
    # ------------------------------------------------------------------

    def detect(self,
               roi_frame: np.ndarray,
               roi_offset_x: int,
               roi_offset_y: int) -> List[MonsterDetection]:
        """
        roi_frame: ROI 영역의 BGR 이미지
        roi_offset_x/y: ROI의 화면 기준 좌표 (결과 좌표 변환용)
        반환: NMS 적용된 MonsterDetection 리스트
        """
        if roi_frame is None:
            return []

        # 그레이스케일 변환
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        # 노이즈 제거 (가우시안 블러)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 배경 업데이트
        self._update_background(gray)

        # 배경이 아직 준비 안 됐으면 빈 결과
        if self._background is None:
            return []

        # 차이 계산
        diff = cv2.absdiff(gray, self._background)

        # 임계값 적용 → 이진화
        _, thresh = cv2.threshold(
            diff, self._diff_threshold, 255, cv2.THRESH_BINARY)

        # 팽창 (dilate): 작은 구멍 메우고 윤곽선 연결
        dilated = cv2.dilate(
            thresh, self._kernel,
            iterations=self._dilate_iterations)

        # 윤곽선 추출
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 각 윤곽선 → MonsterDetection 변환 + 크기 필터
        detections: List[MonsterDetection] = []
        for contour in contours:
            det = self._contour_to_detection(
                contour, roi_offset_x, roi_offset_y)
            if det is not None:
                detections.append(det)

        # NMS
        result = self._apply_nms(detections)

        # FPS 계산
        self._tick_fps()

        return result

    # ------------------------------------------------------------------
    # 배경 업데이트
    # ------------------------------------------------------------------

    def _update_background(self, gray: np.ndarray):
        """
        히스토리에 현재 프레임을 추가하고
        평균을 배경으로 설정한다.
        """
        self._history.append(gray.astype(np.float32))

        if len(self._history) < 2:
            return

        # 히스토리 평균 → 배경
        stack = np.stack(list(self._history), axis=0)
        self._background = np.mean(stack, axis=0).astype(np.uint8)

    # ------------------------------------------------------------------
    # 윤곽선 → Detection 변환 + 크기 필터
    # ------------------------------------------------------------------

    def _contour_to_detection(self,
                               contour,
                               offset_x: int,
                               offset_y: int) -> Optional[MonsterDetection]:
        """
        윤곽선 하나를 MonsterDetection으로 변환한다.
        크기 조건을 만족하지 않으면 None을 반환한다.
        """
        area = cv2.contourArea(contour)
        if area < self._min_area or area > self._max_area:
            return None

        rx, ry, rw, rh = cv2.boundingRect(contour)

        # 크기 필터
        if rw < self._min_width or rw > self._max_width:
            return None
        if rh < self._min_height or rh > self._max_height:
            return None

        # 화면 기준 좌표 변환
        screen_x = rx + offset_x
        screen_y = ry + offset_y
        center_x = screen_x + rw // 2
        center_y = screen_y + rh // 2

        # confidence = 면적을 max_area로 정규화 (0~1)
        confidence = min(area / self._max_area, 1.0)

        return MonsterDetection(
            id=-1,
            x=screen_x,
            y=screen_y,
            width=rw,
            height=rh,
            center_x=center_x,
            center_y=center_y,
            confidence=round(confidence, 3),
            template_name="motion"
        )

    # ------------------------------------------------------------------
    # NMS
    # ------------------------------------------------------------------

    def _apply_nms(self, detections: List[MonsterDetection]) -> List[MonsterDetection]:
        """confidence 내림차순 정렬 후 IoU 기반 중복 제거."""
        if not detections:
            return []

        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[MonsterDetection] = []

        for det in detections:
            duplicate = False
            for k in kept:
                if self._calc_iou(det, k) > self._nms_overlap_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)

        return kept

    @staticmethod
    def _calc_iou(a: MonsterDetection, b: MonsterDetection) -> float:
        ax1, ay1 = a.x, a.y
        ax2, ay2 = a.x + a.width, a.y + a.height
        bx1, by1 = b.x, b.y
        bx2, by2 = b.x + b.width, b.y + b.height

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        union = a.width * a.height + b.width * b.height - inter
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # 배경 리셋 (ROI 변경 시 호출)
    # ------------------------------------------------------------------

    def reset(self):
        """배경 히스토리를 초기화한다. ROI가 바뀌었을 때 호출한다."""
        self._history.clear()
        self._background = None
        print("[Detector] 배경 히스토리 초기화")

    # ------------------------------------------------------------------
    # FPS
    # ------------------------------------------------------------------

    def _tick_fps(self):
        now = time.time()
        self._fps_ticks.append(now)
        if len(self._fps_ticks) > 30:
            self._fps_ticks.pop(0)
        if len(self._fps_ticks) >= 2:
            elapsed = self._fps_ticks[-1] - self._fps_ticks[0]
            if elapsed > 0:
                self._detection_fps = round(
                    (len(self._fps_ticks) - 1) / elapsed, 1)

    @property
    def detection_fps(self) -> float:
        return self._detection_fps

    # ------------------------------------------------------------------
    # 디버그용: 마스크 이미지 반환
    # ------------------------------------------------------------------

    def get_debug_mask(self,
                       roi_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        현재 움직임 마스크를 BGR 이미지로 반환한다.
        디버그 창에 추가로 표시할 때 사용한다.
        """
        if roi_frame is None or self._background is None:
            return None

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, self._background)
        _, thresh = cv2.threshold(
            diff, self._diff_threshold, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(
            thresh, self._kernel, iterations=self._dilate_iterations)
        return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)


# ------------------------------------------------------------------
# Factory: config에 따라 적절한 Detector 생성
# ------------------------------------------------------------------

def make_detector(cfg) -> MotionDetector:
    """config를 보고 적절한 Detector를 생성해서 반환한다."""
    m = cfg.detection.motion
    return MotionDetector(
        diff_threshold=m.diff_threshold,
        min_area=m.min_area,
        max_area=m.max_area,
        min_width=m.min_width,
        max_width=m.max_width,
        min_height=m.min_height,
        max_height=m.max_height,
        dilate_iterations=m.dilate_iterations,
        history_frames=m.history_frames,
        nms_overlap_threshold=m.nms_overlap_threshold,
    )
