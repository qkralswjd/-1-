"""
screen_change_detector.py
--------------------------
화면 전체 변화량을 측정해서 플레이어가 이동 중인지 / 멈춰있는지 판단한다.

원리:
  - 이동 중: 배경이 스크롤 → 전체 화면 픽셀 변화량 매우 큼
  - 멈춤   : 배경 고정 → 전체 화면 픽셀 변화량 거의 0

사용법:
    scd = ScreenChangeDetector(move_threshold=15.0, still_frames_required=3)
    ...
    is_moving = scd.update(frame)  # True=이동중, False=멈춤
    if not is_moving:
        # 탐지 실행
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional


class ScreenChangeDetector:
    """
    프레임 간 평균 픽셀 변화량으로 이동/멈춤을 판단한다.

    Parameters
    ----------
    move_threshold : float
        이 값 이상이면 '이동 중' 판정.
        낮을수록 민감 (작은 움직임도 이동으로), 높을수록 둔감.
        기본값 20.0 → 배경 스크롤 시 보통 50~200, 멈춤 시 1~10 정도.

    still_frames_required : int
        몇 프레임 연속으로 변화량이 threshold 이하여야 '멈춤'으로 확정.
        너무 낮으면 잠깐 멈춰도 탐지 시작, 너무 높으면 반응이 늦음.
        기본값 3.

    sample_scale : float
        비교할 프레임을 얼마나 축소할지 (0 < scale <= 1.0).
        0.25이면 1/4 크기로 줄여서 계산 → 빠름. 기본값 0.25.
    """

    def __init__(self,
                 move_threshold: float = 20.0,
                 still_frames_required: int = 3,
                 sample_scale: float = 0.25):
        self._threshold = move_threshold
        self._still_required = still_frames_required
        self._scale = sample_scale

        self._prev_gray: Optional[np.ndarray] = None
        self._still_count: int = 0          # 연속 멈춤 프레임 수
        self._is_moving: bool = True        # 초기 상태: 이동 중
        self._last_change: float = 0.0      # 마지막 변화량 (디버그용)

    def update(self, frame: np.ndarray) -> bool:
        """
        새 프레임을 입력받아 이동 중 여부를 반환한다.

        Returns
        -------
        bool
            True  = 이동 중 (탐지 스킵)
            False = 멈춤    (탐지 실행)
        """
        if frame is None:
            return self._is_moving

        # 1. 축소 + 그레이스케일 변환 (빠른 연산)
        h, w = frame.shape[:2]
        sw = max(1, int(w * self._scale))
        sh = max(1, int(h * self._scale))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return True  # 첫 프레임은 이동 중으로 처리

        # 2. 프레임 간 평균 픽셀 변화량 계산
        diff = cv2.absdiff(gray, self._prev_gray)
        change = float(np.mean(diff))
        self._last_change = change
        self._prev_gray = gray

        # 3. 이동/멈춤 판정
        if change >= self._threshold:
            # 이동 중 → 멈춤 카운터 리셋
            self._still_count = 0
            self._is_moving = True
        else:
            # 변화량 낮음 → 멈춤 카운터 증가
            self._still_count += 1
            if self._still_count >= self._still_required:
                self._is_moving = False

        return self._is_moving

    @property
    def is_moving(self) -> bool:
        """현재 상태 (True=이동중, False=멈춤)"""
        return self._is_moving

    @property
    def last_change(self) -> float:
        """마지막으로 측정된 픽셀 변화량 (디버그/튜닝용)"""
        return self._last_change

    def reset(self):
        """상태 초기화 (ROI 변경 등 외부 이벤트 시 호출)"""
        self._prev_gray = None
        self._still_count = 0
        self._is_moving = True
        self._last_change = 0.0
