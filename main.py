"""
main.py
-------
전체 실행 흐름을 조율한다.

각 모듈의 로직은 이 파일에 넣지 않고, 해당 모듈을 호출하는 방식으로만 사용한다.

실행 흐름:
  1. 설정 로드
  2. ROI 설정 (마우스 드래그 또는 config 재사용)
  3. 메인 루프:
       - 화면 캡처 (매 프레임)
       - ROI 크롭
       - 움직임 감지 + 크기 필터 (detection_fps 주기로)
       - 추적 업데이트
       - 타겟 선정
       - 공격 (cooldown 제어)
       - 디버그 화면 표시

종료: ESC 키 또는 'q'
"""

import sys
import os
import time
import cv2
import numpy as np
from typing import Optional

# 현재 파일 위치를 sys.path에 추가 (상대 import 없이 사용)
sys.path.insert(0, os.path.dirname(__file__))

from config import load_config, save_roi, Config
from screen_capture import ScreenCapture
from detector import make_detector, MotionDetector, MonsterDetection
from tracker import MonsterTracker, TrackedMonster
from target_selector import select_target
from controller import make_controller, BaseController
from overlay import Overlay, ROISelector


# ------------------------------------------------------------------
# 봇 상태 (단순 문자열로 관리)
# ------------------------------------------------------------------
STATE_SEARCHING        = "SEARCHING"
STATE_TARGET_SELECTED  = "TARGET_SELECTED"
STATE_ATTACKING        = "ATTACKING"
STATE_TARGET_LOST      = "TARGET_LOST"


class MonsterBot:
    """
    메인 봇 클래스.
    각 모듈의 객체를 생성하고 메인 루프를 실행한다.
    """

    def __init__(self):
        # 설정
        self._cfg: Config = load_config()

        # 모듈
        self._capture   = ScreenCapture()
        self._detector: MotionDetector = make_detector(self._cfg)
        self._tracker   = MonsterTracker(
            tracking_distance=self._cfg.tracking.tracking_distance,
            missing_frames_tolerance=self._cfg.tracking.missing_frames_tolerance
        )
        self._overlay   = Overlay(
            show_confidence=self._cfg.debug.show_confidence,
            show_ids=self._cfg.debug.show_ids,
            show_fps=self._cfg.debug.show_fps,
            show_roi=self._cfg.debug.show_roi
        )
        self._controller: BaseController = make_controller(self._cfg)

        # 상태
        self._state     = STATE_SEARCHING
        self._target: Optional[TrackedMonster] = None

        # 타이밍 제어
        self._last_detection_time = 0.0
        self._last_attack_time    = 0.0
        self._detection_interval  = 1.0 / self._cfg.capture.detection_fps
        self._attack_cooldown     = self._cfg.attack.attack_cooldown

    # ------------------------------------------------------------------
    # ROI 설정
    # ------------------------------------------------------------------

    def setup_roi(self) -> bool:
        """
        실행 시작 시 ROI를 설정한다.

        - 기존 config에 ROI가 있으면 그대로 사용할지 물어본다.
        - 새로 지정하려면 'n'을 입력한다.
        - 콘솔에서 'n'을 누르면 화면 캡처 후 ROI 선택 창을 띄운다.
        """
        cfg_roi = self._cfg.roi
        print(f"\n[ROI] 현재 설정된 ROI: x={cfg_roi.x}, y={cfg_roi.y}, "
              f"w={cfg_roi.width}, h={cfg_roi.height}")
        print("[ROI] 이 설정을 사용하려면 Enter, 새로 지정하려면 'n'을 입력하세요.")

        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = ""

        if choice == "n":
            return self._interactive_roi_select()
        else:
            print(f"[ROI] 기존 설정 사용: ({cfg_roi.x}, {cfg_roi.y}, "
                  f"{cfg_roi.width}x{cfg_roi.height})")
            return True

    def _interactive_roi_select(self) -> bool:
        """화면을 캡처해서 ROI 선택 UI를 띄운다."""
        print("[ROI] 화면을 캡처합니다...")
        frame = self._capture.capture_full()
        if frame is None:
            print("[ROI] 화면 캡처 실패")
            return False

        selector = ROISelector()
        result = selector.select(frame)

        if result is None:
            print("[ROI] ROI 선택 취소. 기존 설정을 사용합니다.")
            return True

        x, y, w, h = result
        save_roi(self._cfg, x, y, w, h)
        print(f"[ROI] ROI 설정 완료: x={x}, y={y}, w={w}, h={h}")
        return True

    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------

    def run(self):
        """메인 루프를 실행한다."""
        print("\n" + "="*50)
        print("  Monster Bot - v1.0  (ESC 또는 'q' 종료)")
        print("="*50)
        print(f"  감지 모드: {self._cfg.detection.mode}")
        print(f"  ROI: {self._cfg.roi.x}, {self._cfg.roi.y}, "
              f"{self._cfg.roi.width}x{self._cfg.roi.height}")
        print(f"  Controller: {self._controller.__class__.__name__}")
        print("="*50 + "\n")

        window_name = "Monster Bot - Debug"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        monsters = []

        try:
            while True:
                loop_start = time.time()

                # ── 1. 화면 캡처 ──────────────────────────────────────────
                frame = self._capture.capture_full()
                if frame is None:
                    time.sleep(0.01)
                    continue

                roi = self._cfg.roi

                # ── 2. 탐지 (detection_fps 주기로만 실행) ─────────────────
                now = time.time()
                if now - self._last_detection_time >= self._detection_interval:
                    roi_frame = self._capture.crop_roi(
                        frame, roi.x, roi.y, roi.width, roi.height
                    )
                    if roi_frame is not None:
                        detections = self._detector.detect(
                            roi_frame,
                            roi_offset_x=roi.x,
                            roi_offset_y=roi.y
                        )
                        # 플레이어 위치 제외
                        pe = self._cfg.player_exclusion
                        if pe.enabled:
                            detections = self._detector.filter_player(
                                detections, pe.x, pe.y, pe.radius
                            )
                        # ── 3. 추적 업데이트 ──────────────────────────────
                        monsters = self._tracker.update(detections)
                    self._last_detection_time = now

                # ── 4. 상태 전환 및 타겟 선정 ─────────────────────────────
                self._update_state(monsters)

                # ── 5. 공격 ───────────────────────────────────────────────
                self._try_attack()

                # ── 6. 디버그 표시 ────────────────────────────────────────
                debug_frame = self._overlay.draw(
                    frame,
                    monsters=monsters,
                    target=self._target,
                    roi_x=roi.x, roi_y=roi.y, roi_w=roi.width, roi_h=roi.height,
                    ref_x=self._cfg.reference_point.x,
                    ref_y=self._cfg.reference_point.y,
                    capture_fps=self._capture.fps,
                    detection_fps=self._detector.detection_fps,
                    state=self._state,
                    player_exclusion=self._cfg.player_exclusion
                )

                cv2.imshow(window_name, debug_frame)

                # ── 7. 키 입력 처리 ───────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):   # ESC or q
                    print("[Main] 종료 키 입력")
                    break
                elif key == ord('r'):              # r → ROI 재설정
                    print("[Main] ROI 재설정...")
                    self._interactive_roi_select()
                    self._detector.reset()         # 배경 히스토리 초기화
                elif key == ord('c'):              # c → 추적 초기화
                    print("[Main] 추적 정보 초기화")
                    self._tracker.clear()
                    self._target = None
                    self._state = STATE_SEARCHING
                elif key == ord('p'):              # p → 일시정지 토글
                    self._pause()

                # ── 8. FPS 제한 ───────────────────────────────────────────
                elapsed = time.time() - loop_start
                target_interval = 1.0 / self._cfg.capture.capture_fps
                sleep_time = target_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[Main] KeyboardInterrupt 수신")
        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # 상태 전환
    # ------------------------------------------------------------------

    def _update_state(self, monsters: list):
        """
        현재 몬스터 목록과 기존 타겟 상태를 기반으로
        봇 상태를 갱신하고 타겟을 선택/유지/해제한다.
        """
        ref_x = self._cfg.reference_point.x
        ref_y = self._cfg.reference_point.y

        if self._state == STATE_SEARCHING:
            # 몬스터가 발견되면 타겟 선정
            if monsters:
                new_target = select_target(monsters, ref_x, ref_y)
                if new_target:
                    self._target = new_target
                    self._state = STATE_TARGET_SELECTED
                    print(f"[Bot] Target 선정: #{self._target.id} "
                          f"({self._target.center_x}, {self._target.center_y})")

        elif self._state in (STATE_TARGET_SELECTED, STATE_ATTACKING):
            # 기존 타겟이 아직 살아 있으면 유지
            if self._target is not None:
                live_target = self._tracker.get_monster(self._target.id)
                if live_target is not None:
                    # 타겟 위치 업데이트 (몬스터가 움직였을 수 있음)
                    self._target = live_target
                    self._state = STATE_ATTACKING
                else:
                    # 타겟 사라짐
                    print(f"[Bot] Target #{self._target.id} 소실 → TARGET_LOST")
                    self._state = STATE_TARGET_LOST
            else:
                self._state = STATE_SEARCHING

        elif self._state == STATE_TARGET_LOST:
            # 타겟 초기화 후 다시 탐색
            self._target = None
            self._state = STATE_SEARCHING

    # ------------------------------------------------------------------
    # 공격
    # ------------------------------------------------------------------

    def _try_attack(self):
        """
        ATTACKING 상태이고 cooldown이 지났으면 공격한다.
        매 프레임 클릭하지 않도록 cooldown으로 제한한다.
        """
        if self._state != STATE_ATTACKING:
            return
        if self._target is None:
            return

        now = time.time()
        if now - self._last_attack_time < self._attack_cooldown:
            return

        x, y = self._target.center_x, self._target.center_y
        self._controller.attack(x, y)
        self._last_attack_time = now

    # ------------------------------------------------------------------
    # 일시정지
    # ------------------------------------------------------------------

    def _pause(self):
        """'p' 키 입력 시 일시정지. 다시 'p'를 누르면 재개."""
        print("[Main] 일시정지. 'p'를 누르면 재개합니다.")
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('p') or key == 27:
                print("[Main] 재개")
                break

    # ------------------------------------------------------------------
    # 종료
    # ------------------------------------------------------------------

    def _cleanup(self):
        print("[Main] 정리 중...")
        self._capture.release()
        self._controller.disconnect()
        cv2.destroyAllWindows()
        print("[Main] 종료 완료")


# ------------------------------------------------------------------
# 엔트리 포인트
# ------------------------------------------------------------------

def main():
    bot = MonsterBot()

    # ROI 설정
    if not bot.setup_roi():
        print("[Main] ROI 설정 실패. 프로그램을 종료합니다.")
        sys.exit(1)

    # 메인 루프
    bot.run()


if __name__ == "__main__":
    main()
