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
from screen_change_detector import ScreenChangeDetector
from auto_hunter import AutoHunter


# ------------------------------------------------------------------
# 봇 상태 (단순 문자열로 관리)
# ------------------------------------------------------------------
STATE_SEARCHING        = "SEARCHING"
STATE_TARGET_SELECTED  = "TARGET_SELECTED"
STATE_ATTACKING        = "ATTACKING"
STATE_TARGET_LOST      = "TARGET_LOST"
STATE_MOVING           = "MOVING"


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

        # 이동/멈춤 감지기
        mv = self._cfg.movement
        self._move_detector = ScreenChangeDetector(
            move_threshold=mv.move_threshold,
            still_frames_required=mv.still_frames_required,
            sample_scale=mv.sample_scale,
        )

        # 자동사냥
        self._hunter = AutoHunter(self._cfg)

        # 상태
        self._state     = STATE_MOVING
        self._target: Optional[TrackedMonster] = None

        # 타이밍 제어
        self._last_detection_time = 0.0
        self._last_attack_time    = 0.0
        self._detection_interval  = 1.0 / self._cfg.capture.detection_fps
        self._attack_cooldown     = self._cfg.attack.attack_cooldown

        # 탐지 결과 캐시 (템플릿 탐지는 비쌈 → 결과 재사용)
        self._cached_monsters: list = []
        self._need_redetect: bool = True  # True면 다음 루프에서 탐지 실행

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
        mv = self._cfg.movement
        print(f"  이동감지: threshold={mv.move_threshold}, "
              f"still={mv.still_frames_required}frames")
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

                # ── 2. 이동/멈춤 감지 ──────────────────────────────────
                is_moving = self._move_detector.update(frame)

                if is_moving:
                    # 이동 중 → 탐지 스킵, 캐시 초기화
                    if self._state != STATE_MOVING:
                        print("[Bot] 이동 감지 → 탐지 중단")
                        self._tracker.clear()
                        self._target = None
                        monsters = []
                        self._cached_monsters = []
                        self._need_redetect = True
                        self._detector.reset()
                    self._state = STATE_MOVING
                else:
                    # ── 3. 멈춤 → 탐지 실행 ──────────────────────────────
                    if self._state == STATE_MOVING:
                        print("[Bot] 멈춤 감지 → 탐지 시작")
                        self._state = STATE_SEARCHING
                        self._need_redetect = True  # 멈추면 반드시 재탐지

                    now = time.time()
                    # 재탐지 필요할 때만 실행 (캐시 있으면 스킵)
                    if self._need_redetect and \
                       now - self._last_detection_time >= self._detection_interval:
                        roi_frame = self._capture.crop_roi(
                            frame, roi.x, roi.y, roi.width, roi.height
                        )
                        if roi_frame is not None:
                            detections = self._detector.detect(
                                roi_frame,
                                roi_offset_x=roi.x,
                                roi_offset_y=roi.y
                            )
                            if self._cfg.exclusion_zones:
                                detections = self._detector.filter_zones(
                                    detections, self._cfg.exclusion_zones
                                )
                            pe = self._cfg.player_exclusion
                            if pe.enabled:
                                detections = self._detector.filter_player(
                                    detections, pe.x, pe.y, pe.radius
                                )
                            monsters = self._tracker.update(detections)
                            self._cached_monsters = monsters
                            self._need_redetect = False  # 캐시 완료
                        self._last_detection_time = now
                    else:
                        # 캐시된 결과 재사용 (탐지 스킵 → FPS 유지)
                        monsters = self._cached_monsters

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
                    player_exclusion=self._cfg.player_exclusion,
                    exclusion_zones=self._cfg.exclusion_zones,
                    screen_change=self._move_detector.last_change,
                    move_threshold=self._cfg.movement.move_threshold,
                    bot_enabled=self._hunter.enabled,
                    bot_state=self._hunter.state
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
                elif key == ord('t') or key == ord('T'):  # t → 템플릿 캡처
                    self._capture_template(frame)
                elif key == ord('c'):              # c → 추적 초기화
                    print("[Main] 추적 정보 초기화")
                    self._tracker.clear()
                    self._target = None
                    self._state = STATE_SEARCHING
                elif key == ord('p'):              # p → 일시정지 토글
                    self._pause()
                elif key == ord('b') or key == ord('B'):  # b → 자동사냥 ON/OFF
                    self._toggle_bot()

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
            # 타겟 초기화 후 다시 탐색 → 재탐지 트리거
            self._target = None
            self._cached_monsters = []
            self._need_redetect = True
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
    # 템플릿 캡처 (T 키)
    # ------------------------------------------------------------------

    def _capture_template(self, frame: np.ndarray):
        """
        현재 프레임을 그대로 캡처 창으로 보여주고
        마우스 드래그로 몬스터 영역을 선택해서 templates/에 저장한다.
        메인 루프를 블로킹하지 않고 별도 창에서 처리한다.
        """
        import os

        templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        os.makedirs(templates_dir, exist_ok=True)

        print("\n[Template] 드래그로 몬스터를 선택하세요.")
        print("  Enter/Space: 저장 | R: 다시선택 | ESC: 취소\n")

        window = "템플릿 캡처 - 드래그 후 Enter"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        h, w = frame.shape[:2]
        cv2.resizeWindow(window, min(w, 1400), min(h, 900))

        # 드래그 상태
        drag = {"start": None, "end": None, "dragging": False}

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                drag["start"] = (x, y)
                drag["end"] = (x, y)
                drag["dragging"] = True
            elif event == cv2.EVENT_MOUSEMOVE and drag["dragging"]:
                drag["end"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                drag["end"] = (x, y)
                drag["dragging"] = False

        cv2.setMouseCallback(window, mouse_cb)

        saved_count = 0

        while True:
            display = frame.copy()

            # 선택 영역 그리기
            if drag["start"] and drag["end"]:
                x1 = min(drag["start"][0], drag["end"][0])
                y1 = min(drag["start"][1], drag["end"][1])
                x2 = max(drag["start"][0], drag["end"][0])
                y2 = max(drag["start"][1], drag["end"][1])
                rw, rh = x2 - x1, y2 - y1
                if rw > 5 and rh > 5:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display, f"{rw}x{rh}  Enter=저장",
                                (x1, max(y1 - 8, 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 안내
            cv2.putText(display,
                        f"드래그:선택  Enter:저장  R:초기화  ESC:닫기  저장:{saved_count}개",
                        (8, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF

            # Enter / Space → 저장
            if key in (13, 32):
                if drag["start"] and drag["end"]:
                    x1 = min(drag["start"][0], drag["end"][0])
                    y1 = min(drag["start"][1], drag["end"][1])
                    x2 = max(drag["start"][0], drag["end"][0])
                    y2 = max(drag["start"][1], drag["end"][1])
                    rw, rh = x2 - x1, y2 - y1
                    if rw > 5 and rh > 5:
                        cropped = frame[y1:y2, x1:x2]
                        existing = [f for f in os.listdir(templates_dir)
                                    if f.lower().endswith(".png")]
                        name = f"monster_{len(existing)+1:02d}.png"
                        out_path = os.path.join(templates_dir, name)
                        cv2.imwrite(out_path, cropped)
                        saved_count += 1
                        print(f"[Template] 저장: {name} ({rw}x{rh}px)")

                        # 저장 확인 표시
                        confirm = display.copy()
                        cv2.putText(confirm, f"저장완료! {name}",
                                    (w//2 - 150, h//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (0, 255, 0), 2)
                        cv2.imshow(window, confirm)
                        cv2.waitKey(600)

                        # 선택 초기화 (다음 몬스터 선택 가능)
                        drag["start"] = None
                        drag["end"] = None

            # R → 선택 초기화
            elif key in (ord('r'), ord('R')):
                drag["start"] = None
                drag["end"] = None

            # ESC → 닫기
            elif key == 27:
                break

        cv2.destroyWindow(window)

        if saved_count > 0:
            print(f"[Template] 총 {saved_count}개 저장 완료 → {templates_dir}")
            print("[Template] 'r' 키로 탐지기를 재시작하면 새 템플릿이 적용됩니다.")
        else:
            print("[Template] 저장된 템플릿 없음")

        # 배경 히스토리 초기화 (일시정지 후 재개)
        self._detector.reset()

    # ------------------------------------------------------------------
    # 일시정지
    # ------------------------------------------------------------------

    def _toggle_bot(self):
        """'b' 키로 자동사냥 ON/OFF 토글."""
        if self._hunter.enabled:
            self._hunter.disable()
            print("[Main] 자동사냥 OFF")
        else:
            self._hunter.enable()
            print("[Main] 자동사냥 ON")

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
