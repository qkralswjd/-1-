"""
overlay.py
----------
화면 표시(시각화)만을 담당한다.

입력: 캡처된 frame + 추적 결과 + 상태 정보
출력: OpenCV imshow 창

색상 규칙:
  - ROI 영역      → 파란색
  - 일반 몬스터   → 초록색
  - 현재 타겟     → 빨간색
  - 기준점(플레이어) → 노란색 십자
  - 텍스트 배경   → 반투명 검은색
"""

import cv2
import numpy as np
from typing import List, Optional

from tracker import TrackedMonster


# ------------------------------------------------------------------
# 색상 상수 (BGR)
# ------------------------------------------------------------------
COLOR_ROI        = (255, 100,   0)   # 파란색
COLOR_MONSTER    = ( 50, 220,  50)   # 초록색
COLOR_TARGET     = (  0,   0, 255)   # 빨간색
COLOR_REFERENCE  = (  0, 220, 220)   # 노란색
COLOR_TEXT_BG    = (  0,   0,   0)   # 검은색 (배경)
COLOR_TEXT       = (255, 255, 255)   # 흰색
COLOR_FPS        = (100, 255, 100)   # 연두색

# 선 굵기 및 폰트
THICKNESS_BOX    = 2
THICKNESS_ROI    = 2
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SMALL = 0.45
FONT_SCALE_NORM  = 0.55
FONT_SCALE_LARGE = 0.7


class Overlay:
    """
    디버그 정보를 프레임 위에 그려서 반환한다.
    실제 imshow는 main.py에서 호출한다.
    """

    def __init__(self, show_confidence: bool = True,
                 show_ids: bool = True,
                 show_fps: bool = True,
                 show_roi: bool = True):
        self._show_confidence = show_confidence
        self._show_ids = show_ids
        self._show_fps = show_fps
        self._show_roi = show_roi

    # ------------------------------------------------------------------
    # 메인 그리기
    # ------------------------------------------------------------------

    def draw(self,
             frame: np.ndarray,
             monsters: List[TrackedMonster],
             target: Optional[TrackedMonster],
             roi_x: int, roi_y: int, roi_w: int, roi_h: int,
             ref_x: int, ref_y: int,
             capture_fps: float,
             detection_fps: float,
             state: str = "",
             player_exclusion=None,
             exclusion_zones=None,
             screen_change: float = 0.0,
             move_threshold: float = 20.0) -> np.ndarray:
        """
        frame에 모든 디버그 정보를 그려서 반환한다.
        원본 frame을 수정하지 않고 복사본에 그린다.
        """
        out = frame.copy()

        # 1. ROI 박스
        if self._show_roi:
            self._draw_roi(out, roi_x, roi_y, roi_w, roi_h)

        # 2. 기준점 (플레이어 위치)
        self._draw_reference(out, ref_x, ref_y)

        # 2-1. 고정 제외 영역 (나무 등) - 빨간 점선 사각형
        if exclusion_zones:
            for zone in exclusion_zones:
                cv2.rectangle(out,
                              (zone.x, zone.y),
                              (zone.x + zone.width, zone.y + zone.height),
                              (0, 0, 180), 1)
                self._put_label(out, f"EXCL:{zone.name}",
                                zone.x + 2, zone.y + 14,
                                font_scale=FONT_SCALE_SMALL,
                                text_color=(0, 0, 200))

        # 2-2. 플레이어 제외 영역
        if player_exclusion and player_exclusion.enabled:
            cv2.circle(out,
                       (player_exclusion.x, player_exclusion.y),
                       player_exclusion.radius,
                       (0, 80, 180), 1)
            self._put_label(out, "PLAYER", 
                            player_exclusion.x - 25,
                            player_exclusion.y - player_exclusion.radius - 5,
                            font_scale=FONT_SCALE_SMALL,
                            text_color=(0, 140, 255))

        # 3. 몬스터 박스
        for monster in monsters:
            is_target = (target is not None and monster.id == target.id)
            self._draw_monster(out, monster, is_target)

        # 4. HUD (FPS, 상태, 몬스터 목록)
        if self._show_fps:
            self._draw_hud(out, monsters, target, capture_fps, detection_fps,
                          state, screen_change, move_threshold)

        return out

    # ------------------------------------------------------------------
    # ROI
    # ------------------------------------------------------------------

    def _draw_roi(self, frame: np.ndarray,
                  x: int, y: int, w: int, h: int):
        """ROI 영역을 파란색 사각형으로 그린다."""
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      COLOR_ROI, THICKNESS_ROI)
        label = "MONSTER ROI"
        self._put_label(frame, label, x + 4, y + 16,
                        font_scale=FONT_SCALE_SMALL,
                        text_color=COLOR_ROI)

    # ------------------------------------------------------------------
    # 기준점
    # ------------------------------------------------------------------

    def _draw_reference(self, frame: np.ndarray, x: int, y: int):
        """플레이어 기준점을 십자 마커로 그린다."""
        size = 12
        cv2.line(frame, (x - size, y), (x + size, y), COLOR_REFERENCE, 2)
        cv2.line(frame, (x, y - size), (x, y + size), COLOR_REFERENCE, 2)
        cv2.circle(frame, (x, y), 4, COLOR_REFERENCE, -1)

    # ------------------------------------------------------------------
    # 몬스터 박스
    # ------------------------------------------------------------------

    def _draw_monster(self, frame: np.ndarray,
                      monster: TrackedMonster,
                      is_target: bool):
        """몬스터 하나의 바운딩 박스와 레이블을 그린다."""
        color = COLOR_TARGET if is_target else COLOR_MONSTER
        x, y, w, h = monster.x, monster.y, monster.width, monster.height

        # 바운딩 박스
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, THICKNESS_BOX)

        # 중심점
        cv2.circle(frame, (monster.center_x, monster.center_y), 3, color, -1)

        # 레이블 조합
        parts = []
        if self._show_ids:
            parts.append(f"#{monster.id}")
        parts.append(monster.template_name if monster.template_name else "Monster")
        if self._show_confidence:
            parts.append(f"{monster.confidence:.2f}")
        if is_target:
            parts.append("TARGET")
        label = " ".join(parts)

        # missing 상태면 반투명하게
        if monster.missing_frames > 0:
            label += f" (miss:{monster.missing_frames})"

        self._put_label(frame, label, x, y - 6,
                        font_scale=FONT_SCALE_SMALL,
                        text_color=color,
                        with_bg=True)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(self, frame: np.ndarray,
                  monsters: List[TrackedMonster],
                  target: Optional[TrackedMonster],
                  capture_fps: float,
                  detection_fps: float,
                  state: str,
                  screen_change: float = 0.0,
                  move_threshold: float = 20.0):
        """화면 좌상단에 FPS, 상태, 몬스터 목록을 표시한다."""
        # 이동/멈춤 상태 색상
        is_moving = (state == "MOVING")
        move_color = (0, 80, 255) if is_moving else (0, 220, 80)  # 빨강 or 초록
        move_label = ">> MOVING <<" if is_moving else "** DETECTING **"

        lines = []
        lines.append(f"FPS: {capture_fps:.1f}  Det FPS: {detection_fps:.1f}")
        lines.append(f"Change: {screen_change:.1f} / thr:{move_threshold:.0f}")
        lines.append(f"State: {state}")
        lines.append(f"Monsters: {len(monsters)}")
        if target:
            lines.append(f"Target: #{target.id}")
        else:
            lines.append("Target: None")
        lines.append("---")
        for m in monsters:
            is_tgt = (target is not None and m.id == target.id)
            tag = "  TARGET" if is_tgt else ""
            lines.append(
                f"#{m.id} {m.template_name}  conf={m.confidence:.2f}{tag}"
            )

        x_start = 8
        y_start = 20
        line_h = 18

        for i, line in enumerate(lines):
            y = y_start + i * line_h
            self._put_label(frame, line, x_start, y,
                            font_scale=FONT_SCALE_SMALL,
                            text_color=COLOR_FPS,
                            with_bg=True)

        # 화면 우상단에 이동/멈춤 상태 크게 표시
        fh, fw = frame.shape[:2]
        self._put_label(frame, move_label,
                        fw - 180, 24,
                        font_scale=FONT_SCALE_LARGE,
                        text_color=move_color,
                        with_bg=True)

    # ------------------------------------------------------------------
    # 유틸
    # ------------------------------------------------------------------

    def _put_label(self, frame: np.ndarray,
                   text: str, x: int, y: int,
                   font_scale: float = FONT_SCALE_SMALL,
                   text_color=(255, 255, 255),
                   with_bg: bool = False):
        """텍스트를 그린다. with_bg=True이면 배경 사각형도 그린다."""
        if y < 0:
            y = 0
        if x < 0:
            x = 0

        thickness = 1
        (tw, th), baseline = cv2.getTextSize(
            text, FONT, font_scale, thickness)

        if with_bg:
            bg_x1 = x - 1
            bg_y1 = y - th - 2
            bg_x2 = x + tw + 2
            bg_y2 = y + baseline + 1

            # 반투명 배경
            sub = frame[max(0, bg_y1):bg_y2, max(0, bg_x1):bg_x2]
            if sub.size > 0:
                rect = np.zeros_like(sub)
                cv2.addWeighted(sub, 0.5, rect, 0.5, 0, sub)
                frame[max(0, bg_y1):bg_y2, max(0, bg_x1):bg_x2] = sub

        cv2.putText(frame, text, (x, y),
                    FONT, font_scale, text_color, thickness, cv2.LINE_AA)


# ------------------------------------------------------------------
# ROI 선택 UI (마우스 드래그)
# ------------------------------------------------------------------

class ROISelector:
    """
    OpenCV 창에서 마우스 드래그로 ROI를 선택한다.
    select() 를 호출하면 사용자가 드래그해서 영역을 지정하고
    (x, y, w, h) 를 반환한다.
    """

    def __init__(self):
        self._dragging = False
        self._start = (0, 0)
        self._end = (0, 0)
        self._done = False
        self._result = None

    def select(self, frame: np.ndarray,
               window_name: str = "ROI 설정 - 드래그로 영역 지정 후 Enter") -> Optional[tuple]:
        """
        frame을 보여주고 사용자가 드래그로 영역을 선택하게 한다.
        Enter 또는 Space로 확정, ESC로 취소.
        반환: (x, y, width, height) 또는 None
        """
        self._dragging = False
        self._start = (0, 0)
        self._end = (0, 0)
        self._done = False
        self._result = None
        self._frame = frame.copy()
        self._draw_frame = frame.copy()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("[ROISelector] 마우스로 드래그해서 ROI를 지정하세요.")
        print("  Enter 또는 Space: 확정 | ESC: 취소")

        while True:
            display = self._draw_frame.copy()

            if self._dragging or self._done:
                x1 = min(self._start[0], self._end[0])
                y1 = min(self._start[1], self._end[1])
                x2 = max(self._start[0], self._end[0])
                y2 = max(self._start[1], self._end[1])
                cv2.rectangle(display, (x1, y1), (x2, y2), COLOR_ROI, 2)
                label = f"ROI: ({x1},{y1}) {x2-x1}x{y2-y1}"
                cv2.putText(display, label, (x1, max(y1 - 6, 16)),
                            FONT, FONT_SCALE_SMALL, COLOR_ROI, 1)

            cv2.putText(display,
                        "Drag to select ROI | Enter=OK | ESC=Cancel",
                        (8, display.shape[0] - 10),
                        FONT, FONT_SCALE_SMALL, COLOR_TEXT, 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key in (13, 32):   # Enter or Space
                if self._done and self._result:
                    break
            elif key == 27:       # ESC
                self._result = None
                break

        cv2.destroyWindow(window_name)
        return self._result

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._done = False
            self._start = (x, y)
            self._end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._dragging:
                self._end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
            self._end = (x, y)
            x1 = min(self._start[0], self._end[0])
            y1 = min(self._start[1], self._end[1])
            x2 = max(self._start[0], self._end[0])
            y2 = max(self._start[1], self._end[1])
            w = x2 - x1
            h = y2 - y1
            if w > 10 and h > 10:
                self._result = (x1, y1, w, h)
                self._done = True
            else:
                self._result = None
