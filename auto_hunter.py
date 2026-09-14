"""
auto_hunter.py
--------------
자동 사냥 루프를 담당한다.

흐름:
  1. 멈춤 감지 → 탐지 실행
  2. 몬스터 발견 → 몬스터 좌표로 클릭 드래그 (자동공격)
  3. 몬스터 사라질 때까지 공격 반복
  4. 몬스터 없음 → 무작위 바닥 좌표 클릭으로 이동
  5. 반복

이동 방식:
  - ROI 내 무작위 좌표 클릭 (마우스 왼클릭 = 이동)
  - 플레이어 위치 주변 일정 범위 내에서만 이동

공격 방식:
  - 몬스터 좌표로 클릭 드래그 = 자동공격 발동
  - 몬스터가 tracker에서 사라지면 처치 완료
"""

import random
import time
from typing import Optional, Tuple


class AutoHunter:
    """
    자동 사냥 상태 머신.

    States:
        IDLE      : 대기 (봇 비활성)
        MOVING    : 이동 중 (바닥 클릭 후 도착 대기)
        DETECTING : 멈춤 - 몬스터 탐지 중
        ATTACKING : 몬스터 공격 중
    """

    STATE_IDLE      = "IDLE"
    STATE_MOVING    = "MOVING"
    STATE_DETECTING = "DETECTING"
    STATE_ATTACKING = "ATTACKING"

    def __init__(self, cfg):
        h = cfg.hunt

        # 이동 가능 범위 (화면 좌표)
        self._move_x_min  = h.move_x_min
        self._move_x_max  = h.move_x_max
        self._move_y_min  = h.move_y_min
        self._move_y_max  = h.move_y_max

        # 타이밍
        self._move_wait       = h.move_wait_sec      # 이동 후 멈춤 대기 시간(초)
        self._attack_cooldown = h.attack_cooldown_sec
        self._no_monster_timeout = h.no_monster_timeout_sec  # 몬스터 없으면 N초 후 이동

        # 드래그 설정
        self._drag_dx   = h.drag_dx
        self._drag_dy   = h.drag_dy
        self._drag_hold = h.drag_hold_ms

        self._state: str = self.STATE_IDLE
        self._last_attack_time: float = 0.0
        self._last_move_time: float = 0.0
        self._detecting_since: float = 0.0  # 탐지 시작 시각
        self._enabled: bool = False

    # ------------------------------------------------------------------
    # 활성화 / 비활성화
    # ------------------------------------------------------------------

    def enable(self):
        self._enabled = True
        self._state = self.STATE_DETECTING
        self._detecting_since = time.time()
        print("[AutoHunter] 활성화")

    def disable(self):
        self._enabled = False
        self._state = self.STATE_IDLE
        print("[AutoHunter] 비활성화")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def state(self) -> str:
        return self._state

    # ------------------------------------------------------------------
    # 메인 업데이트 (매 프레임 호출)
    # ------------------------------------------------------------------

    def update(self, monsters: list, is_screen_moving: bool,
               controller) -> Optional[Tuple[int, int]]:
        """
        매 프레임 호출. 상태에 따라 이동/공격 명령을 controller에 전송.

        Parameters
        ----------
        monsters          : 현재 추적 중인 몬스터 리스트
        is_screen_moving  : 화면 이동 감지 결과 (True=이동중)
        controller        : PicoController or DummyController

        Returns
        -------
        공격한 몬스터의 (x, y) 또는 None
        """
        if not self._enabled:
            return None

        now = time.time()

        # 화면이 이동 중이면 → MOVING 상태 유지
        if is_screen_moving:
            self._state = self.STATE_MOVING
            return None

        # 화면 멈춤 → 탐지/공격 상태로
        if self._state == self.STATE_MOVING:
            self._state = self.STATE_DETECTING
            self._detecting_since = now
            print("[AutoHunter] 멈춤 감지 → 탐지 시작")

        # ── DETECTING: 몬스터 탐색 ──────────────────────────────────────
        if self._state == self.STATE_DETECTING:
            if monsters:
                # 몬스터 발견 → 공격 상태로
                self._state = self.STATE_ATTACKING
                print(f"[AutoHunter] 몬스터 발견 {len(monsters)}마리 → 공격 시작")
            else:
                # N초 동안 몬스터 없으면 → 이동
                elapsed = now - self._detecting_since
                if elapsed >= self._no_monster_timeout:
                    self._do_random_move(controller)
                return None

        # ── ATTACKING: 공격 ─────────────────────────────────────────────
        if self._state == self.STATE_ATTACKING:
            if not monsters:
                # 몬스터 전멸 → 이동
                print("[AutoHunter] 몬스터 처치 완료 → 이동")
                self._do_random_move(controller)
                return None

            # cooldown 체크
            if now - self._last_attack_time < self._attack_cooldown:
                return None

            # 가장 가까운 몬스터 공격 (target_selector와 별개로 직접 선택)
            target = monsters[0]
            x, y = target.center_x, target.center_y

            # 클릭 드래그로 자동공격 발동
            controller.click_drag(
                x, y,
                drag_dx=self._drag_dx,
                drag_dy=self._drag_dy,
                hold_ms=self._drag_hold
            )
            self._last_attack_time = now
            print(f"[AutoHunter] 공격: #{target.id} ({x}, {y})")
            return (x, y)

        return None

    # ------------------------------------------------------------------
    # 무작위 이동
    # ------------------------------------------------------------------

    def _do_random_move(self, controller):
        """무작위 바닥 좌표 클릭으로 이동."""
        x = random.randint(self._move_x_min, self._move_x_max)
        y = random.randint(self._move_y_min, self._move_y_max)
        controller.click_move(x, y)
        self._last_move_time = time.time()
        self._state = self.STATE_MOVING
        print(f"[AutoHunter] 이동 클릭: ({x}, {y})")
