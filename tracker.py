"""
tracker.py
----------
몬스터 ID 및 위치 추적을 담당한다.

핵심 기능:
  - 거리 기반 greedy 매칭으로 ID 유지
  - 이동 거리 필터: N프레임 동안 거의 안 움직이면 나무/배경으로 판단 → 제거
    (나무는 같은 자리에서 흔들리지만 실제로 이동하지 않음)
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Deque

from detector import MonsterDetection


@dataclass
class TrackedMonster:
    """추적 중인 몬스터 하나."""
    id: int
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    confidence: float
    template_name: str

    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    missing_frames: int = 0
    age: int = 0

    # 이동 거리 추적 (나무/배경 오탐지 필터용)
    # 최근 N프레임의 중심점 기록
    _position_history: Deque = field(
        default_factory=lambda: deque(maxlen=12), repr=False)
    # 총 누적 이동 거리
    _total_movement: float = field(default=0.0, repr=False)
    # 이동 확인 완료 여부 (한번 이동 확인되면 계속 유지)
    _confirmed_moving: bool = field(default=False, repr=False)

    def update_from_detection(self, det: MonsterDetection):
        """새로운 탐지 결과로 위치와 상태를 업데이트한다."""
        prev_cx, prev_cy = self.center_x, self.center_y

        self.x = det.x
        self.y = det.y
        self.width = det.width
        self.height = det.height
        self.center_x = det.center_x
        self.center_y = det.center_y
        self.confidence = det.confidence
        self.template_name = det.template_name
        self.last_seen = time.time()
        self.missing_frames = 0
        self.age += 1

        # 이동 거리 기록
        move = math.sqrt((det.center_x - prev_cx)**2 + (det.center_y - prev_cy)**2)
        self._total_movement += move
        self._position_history.append((det.center_x, det.center_y))

        # 한번이라도 충분히 움직이면 confirmed
        if move >= 8:
            self._confirmed_moving = True

    def mark_missing(self):
        """이번 프레임에서 탐지되지 않았을 때 호출한다."""
        self.missing_frames += 1
        self.age += 1

    @property
    def is_static(self) -> bool:
        """
        나무/배경 오탐지 여부 판단.
        - age가 충분히 쌓였는데도 총 이동거리가 적으면 정적 오탐지
        - 한번이라도 크게 움직였으면 False (진짜 몬스터)
        """
        if self._confirmed_moving:
            return False
        # age 10 이상 쌓였을 때만 판단 (초반엔 아직 모름)
        if self.age < 10:
            return False
        # 10프레임 동안 총 이동거리가 15px 미만 = 정적
        return self._total_movement < 15.0

    @property
    def is_active(self) -> bool:
        return True


class MonsterTracker:
    """
    매 프레임의 탐지 결과를 받아 TrackedMonster를 업데이트한다.

    알고리즘:
    1. 현재 탐지 결과와 기존 추적 몬스터들 사이의 거리를 계산
    2. tracking_distance 이내 + 가장 가까운 쌍을 매칭
    3. 매칭된 몬스터는 위치 업데이트
    4. 매칭 안 된 탐지 결과는 새 몬스터로 등록
    5. 매칭 안 된 기존 몬스터는 missing_frames 증가
    6. missing_frames > tolerance 이면 삭제
    7. is_static=True 인 몬스터는 나무로 판단하여 삭제
    """

    def __init__(self,
                 tracking_distance: int = 80,
                 missing_frames_tolerance: int = 5):
        self._tracking_distance = tracking_distance
        self._missing_frames_tolerance = missing_frames_tolerance
        self._monsters: Dict[int, TrackedMonster] = {}
        self._next_id: int = 1

    def update(self, detections: List[MonsterDetection]) -> List[TrackedMonster]:
        """새로운 탐지 결과로 추적기를 업데이트하고 살아있는 몬스터 리스트 반환."""
        existing = list(self._monsters.values())

        matched_monster_ids, matched_det_indices = self._match(existing, detections)

        # 1. 매칭된 몬스터 위치 업데이트
        for monster_id, det_idx in zip(matched_monster_ids, matched_det_indices):
            self._monsters[monster_id].update_from_detection(detections[det_idx])
            detections[det_idx].id = monster_id

        # 2. 매칭 안 된 기존 몬스터 → missing 처리
        unmatched_monster_ids = set(m.id for m in existing) - set(matched_monster_ids)
        for mid in unmatched_monster_ids:
            self._monsters[mid].mark_missing()

        # 3. 매칭 안 된 탐지 결과 → 새 몬스터 등록
        unmatched_det_indices = set(range(len(detections))) - set(matched_det_indices)
        for di in unmatched_det_indices:
            self._register_new_monster(detections[di])

        # 4. missing 초과 몬스터 삭제
        to_delete = [
            mid for mid, m in self._monsters.items()
            if m.missing_frames > self._missing_frames_tolerance
        ]
        for mid in to_delete:
            del self._monsters[mid]

        # 5. 정적 오탐지 (나무/배경) 삭제
        static_delete = [
            mid for mid, m in self._monsters.items()
            if m.is_static
        ]
        for mid in static_delete:
            m = self._monsters[mid]
            print(f"[Tracker] #{mid} 정적 오탐지 제거 "
                  f"(age={m.age}, 이동={m._total_movement:.1f}px)")
            del self._monsters[mid]

        return list(self._monsters.values())

    def _match(self, existing: List[TrackedMonster],
               detections: List[MonsterDetection]) -> tuple:
        """거리 기반 greedy 매칭."""
        if not existing or not detections:
            return [], []

        pairs = []
        for mi, monster in enumerate(existing):
            for di, det in enumerate(detections):
                dist = _dist(monster.center_x, monster.center_y,
                             det.center_x, det.center_y)
                if dist <= self._tracking_distance:
                    pairs.append((dist, mi, di))

        pairs.sort(key=lambda p: p[0])

        matched_monster_ids = []
        matched_det_indices = []
        used_monsters = set()
        used_det = set()

        for dist, mi, di in pairs:
            if mi in used_monsters or di in used_det:
                continue
            matched_monster_ids.append(existing[mi].id)
            matched_det_indices.append(di)
            used_monsters.add(mi)
            used_det.add(di)

        return matched_monster_ids, matched_det_indices

    def _register_new_monster(self, det: MonsterDetection):
        new_id = self._next_id
        self._next_id += 1
        monster = TrackedMonster(
            id=new_id,
            x=det.x, y=det.y,
            width=det.width, height=det.height,
            center_x=det.center_x, center_y=det.center_y,
            confidence=det.confidence,
            template_name=det.template_name,
        )
        # 초기 위치 기록
        monster._position_history.append((det.center_x, det.center_y))
        self._monsters[new_id] = monster
        det.id = new_id
        print(f"[Tracker] 새 Monster #{new_id} 등록 "
              f"({det.center_x}, {det.center_y}) conf={det.confidence:.2f}")

    @property
    def monsters(self) -> List[TrackedMonster]:
        return list(self._monsters.values())

    def get_monster(self, monster_id: int) -> Optional[TrackedMonster]:
        return self._monsters.get(monster_id)

    def is_alive(self, monster_id: int) -> bool:
        return monster_id in self._monsters

    def clear(self):
        self._monsters.clear()
        self._next_id = 1
        print("[Tracker] 추적 정보 초기화")

    @property
    def count(self) -> int:
        return len(self._monsters)


def _dist(x1, y1, x2, y2) -> float:
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
