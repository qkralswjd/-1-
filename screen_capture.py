"""
screen_capture.py
-----------------
화면 캡처만을 담당한다.
mss를 사용해서 전체 화면 또는 ROI 영역을 numpy 배열로 반환한다.
캡처 자체는 최대한 가볍게 유지한다.
"""

import time
import numpy as np
from typing import Optional, Tuple
import mss
import mss.tools


class ScreenCapture:
    """
    mss 기반 화면 캡처 클래스.
    전체 화면을 캡처하고, ROI는 별도로 크롭해서 반환한다.
    """

    def __init__(self):
        self._sct = mss.mss()
        self._monitor = self._sct.monitors[0]   # monitors[0] = 전체 화면 가상 모니터
        self._last_frame: Optional[np.ndarray] = None
        self._last_capture_time: float = 0.0
        self._fps_counter = FPSCounter()

    def get_screen_size(self) -> Tuple[int, int]:
        """전체 화면 크기를 (width, height)로 반환한다."""
        m = self._sct.monitors[1]  # monitors[1] = 첫 번째 실제 모니터
        return m["width"], m["height"]

    def capture_full(self) -> Optional[np.ndarray]:
        """
        전체 화면을 캡처해서 BGR numpy 배열로 반환한다.
        실패 시 None을 반환한다.
        """
        try:
            monitor = self._sct.monitors[1]   # 첫 번째 실제 모니터
            sct_img = self._sct.grab(monitor)
            # mss는 BGRA 포맷 → BGR로 변환
            frame = np.array(sct_img)[:, :, :3]  # BGRA → BGR (alpha 제거)
            self._last_frame = frame
            self._last_capture_time = time.time()
            self._fps_counter.tick()
            return frame
        except Exception as e:
            print(f"[ScreenCapture] 캡처 실패: {e}")
            return None

    def capture_roi(self, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
        """
        지정된 ROI 영역만 캡처해서 BGR numpy 배열로 반환한다.
        전체 캡처보다 빠르다.
        """
        try:
            monitor = {
                "top": y,
                "left": x,
                "width": width,
                "height": height
            }
            sct_img = self._sct.grab(monitor)
            frame = np.array(sct_img)[:, :, :3]  # BGRA → BGR
            return frame
        except Exception as e:
            print(f"[ScreenCapture] ROI 캡처 실패: {e}")
            return None

    def crop_roi(self, frame: np.ndarray,
                 x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
        """
        이미 캡처된 전체 화면 frame에서 ROI를 크롭해서 반환한다.
        전체 화면 캡처 후 ROI를 잘라낼 때 사용한다.
        """
        if frame is None:
            return None
        try:
            h, w = frame.shape[:2]
            # 경계 보정
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + width)
            y2 = min(h, y + height)
            if x2 <= x1 or y2 <= y1:
                return None
            return frame[y1:y2, x1:x2].copy()
        except Exception as e:
            print(f"[ScreenCapture] ROI 크롭 실패: {e}")
            return None

    @property
    def fps(self) -> float:
        """현재 캡처 FPS를 반환한다."""
        return self._fps_counter.fps

    @property
    def last_frame(self) -> Optional[np.ndarray]:
        """마지막으로 캡처된 프레임을 반환한다."""
        return self._last_frame

    def release(self):
        """mss 리소스를 해제한다."""
        try:
            self._sct.close()
        except Exception:
            pass


class FPSCounter:
    """
    간단한 FPS 카운터.
    최근 N개의 프레임 시간으로 FPS를 계산한다.
    """

    def __init__(self, window: int = 30):
        self._window = window
        self._times: list = []
        self._fps: float = 0.0

    def tick(self):
        """프레임 하나가 처리될 때마다 호출한다."""
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) >= 2:
            elapsed = self._times[-1] - self._times[0]
            if elapsed > 0:
                self._fps = (len(self._times) - 1) / elapsed

    @property
    def fps(self) -> float:
        return round(self._fps, 1)
