"""
detector.py
-----------
템플릿 매칭 + NMS(중복 제거)만을 담당한다.

입력: ROI 이미지 (numpy BGR array)
출력: MonsterDetection 리스트

ROI 안에서만 탐지를 수행하고,
탐지 결과의 좌표는 ROI 기준이 아닌 원본 화면 기준으로 변환해서 반환한다.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MonsterDetection:
    """
    템플릿 매칭 결과 하나를 나타낸다.
    모든 좌표는 원본 화면(전체 화면) 기준이다.
    """
    id: int                  # 추적기가 부여하는 ID (탐지 시점에는 -1)
    x: int                   # 바운딩 박스 좌측 상단 X (화면 기준)
    y: int                   # 바운딩 박스 좌측 상단 Y (화면 기준)
    width: int               # 바운딩 박스 너비
    height: int              # 바운딩 박스 높이
    center_x: int            # 중심 X (화면 기준)
    center_y: int            # 중심 Y (화면 기준)
    confidence: float        # 매칭 신뢰도 (0.0 ~ 1.0)
    template_name: str = ""  # 어떤 템플릿에서 매칭됐는지


class TemplateDetector:
    """
    templates/ 디렉토리의 이미지들을 로드해서
    ROI 이미지에 대해 템플릿 매칭을 수행한다.
    """

    def __init__(self,
                 templates_dir: str,
                 match_threshold: float = 0.75,
                 nms_overlap_threshold: float = 0.3):
        self._templates_dir = templates_dir
        self._match_threshold = match_threshold
        self._nms_overlap_threshold = nms_overlap_threshold
        self._templates: List[Tuple[str, np.ndarray]] = []  # (이름, 이미지)
        self._fps_counter_ticks: List[float] = []
        self._detection_fps: float = 0.0

        self._load_templates()

    # ------------------------------------------------------------------
    # 템플릿 로드
    # ------------------------------------------------------------------

    def _load_templates(self):
        """templates_dir에서 PNG/JPG 파일을 모두 로드한다."""
        self._templates.clear()

        if not os.path.isdir(self._templates_dir):
            print(f"[Detector] 템플릿 디렉토리 없음: {self._templates_dir}")
            return

        supported = (".png", ".jpg", ".jpeg", ".bmp")
        for fname in sorted(os.listdir(self._templates_dir)):
            if not fname.lower().endswith(supported):
                continue
            path = os.path.join(self._templates_dir, fname)
            img = cv2.imread(path)
            if img is None:
                print(f"[Detector] 템플릿 로드 실패: {path}")
                continue
            name = os.path.splitext(fname)[0]
            self._templates.append((name, img))
            print(f"[Detector] 템플릿 로드: {name} "
                  f"({img.shape[1]}x{img.shape[0]}px)")

        print(f"[Detector] 총 {len(self._templates)}개 템플릿 로드 완료")

    def reload_templates(self):
        """런타임 중 템플릿을 다시 로드한다."""
        self._load_templates()

    @property
    def template_count(self) -> int:
        return len(self._templates)

    # ------------------------------------------------------------------
    # 메인 탐지
    # ------------------------------------------------------------------

    def detect(self,
               roi_frame: np.ndarray,
               roi_offset_x: int,
               roi_offset_y: int) -> List[MonsterDetection]:
        """
        roi_frame: ROI 영역의 BGR 이미지
        roi_offset_x, roi_offset_y: ROI의 화면 기준 좌측 상단 좌표
            (탐지 결과를 화면 기준 좌표로 변환할 때 사용)

        반환: 중복 제거된 MonsterDetection 리스트 (id=-1 상태)
        """
        import time
        if roi_frame is None or len(self._templates) == 0:
            return []

        all_detections: List[MonsterDetection] = []

        for name, template in self._templates:
            detections = self._match_single_template(
                roi_frame, template, name, roi_offset_x, roi_offset_y
            )
            all_detections.extend(detections)

        # NMS로 중복 제거
        result = self._apply_nms(all_detections)

        # Detection FPS 계산
        now = time.time()
        self._fps_counter_ticks.append(now)
        if len(self._fps_counter_ticks) > 20:
            self._fps_counter_ticks.pop(0)
        if len(self._fps_counter_ticks) >= 2:
            elapsed = self._fps_counter_ticks[-1] - self._fps_counter_ticks[0]
            if elapsed > 0:
                self._detection_fps = round(
                    (len(self._fps_counter_ticks) - 1) / elapsed, 1)

        return result

    # ------------------------------------------------------------------
    # 단일 템플릿 매칭
    # ------------------------------------------------------------------

    def _match_single_template(self,
                                roi_frame: np.ndarray,
                                template: np.ndarray,
                                template_name: str,
                                offset_x: int,
                                offset_y: int) -> List[MonsterDetection]:
        """
        하나의 템플릿에 대해 cv2.matchTemplate을 수행하고
        threshold 이상인 모든 위치를 MonsterDetection으로 반환한다.
        (이 단계에서는 아직 NMS를 적용하지 않는다)
        """
        roi_h, roi_w = roi_frame.shape[:2]
        tmpl_h, tmpl_w = template.shape[:2]

        # 템플릿이 ROI보다 크면 탐지 불가
        if tmpl_h > roi_h or tmpl_w > roi_w:
            return []

        try:
            # TM_CCOEFF_NORMED: 조명 변화에 비교적 강한 방식
            result = cv2.matchTemplate(roi_frame, template, cv2.TM_CCOEFF_NORMED)
        except cv2.error as e:
            print(f"[Detector] matchTemplate 오류 ({template_name}): {e}")
            return []

        # threshold 이상인 모든 위치 추출
        locations = np.where(result >= self._match_threshold)
        detections: List[MonsterDetection] = []

        for pt_y, pt_x in zip(*locations):
            confidence = float(result[pt_y, pt_x])
            # ROI 기준 → 화면 기준 변환
            screen_x = int(pt_x) + offset_x
            screen_y = int(pt_y) + offset_y
            center_x = screen_x + tmpl_w // 2
            center_y = screen_y + tmpl_h // 2

            detections.append(MonsterDetection(
                id=-1,
                x=screen_x,
                y=screen_y,
                width=tmpl_w,
                height=tmpl_h,
                center_x=center_x,
                center_y=center_y,
                confidence=confidence,
                template_name=template_name
            ))

        return detections

    # ------------------------------------------------------------------
    # NMS (Non-Maximum Suppression)
    # ------------------------------------------------------------------

    def _apply_nms(self, detections: List[MonsterDetection]) -> List[MonsterDetection]:
        """
        OpenCV의 groupRectangles와 유사한 방식으로
        겹치는 바운딩 박스를 제거한다.

        IoU(Intersection over Union) 기반으로 구현한다.
        confidence가 높은 것을 우선 유지한다.
        """
        if not detections:
            return []

        # confidence 내림차순 정렬
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        kept: List[MonsterDetection] = []

        for det in detections:
            is_duplicate = False
            for kept_det in kept:
                iou = self._calc_iou(det, kept_det)
                if iou > self._nms_overlap_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(det)

        return kept

    @staticmethod
    def _calc_iou(a: MonsterDetection, b: MonsterDetection) -> float:
        """두 바운딩 박스의 IoU를 계산한다."""
        ax1, ay1 = a.x, a.y
        ax2, ay2 = a.x + a.width, a.y + a.height
        bx1, by1 = b.x, b.y
        bx2, by2 = b.x + b.width, b.y + b.height

        # 교차 영역
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        area_a = a.width * a.height
        area_b = b.width * b.height
        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @property
    def detection_fps(self) -> float:
        return self._detection_fps
