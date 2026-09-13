"""
tracker.py
----------
몬스터 ID 및 위치 추적만을 담당한다.

매 프레임의 MonsterDetection 리스트를 받아서
이전 프레임의 TrackedMonster와 거리를 비교해
같은 몬스터인지 판단하고 ID를 유지한다.

복잡한 Kalman Filter 없이 '중심점 간 거리' 기준으로만 추적한다.
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from detector import MonsterDetection


@dataclass
class TrackedMonster:
    """
    추적 중인 몬스터 하나를 나타낸다.
    MonsterDetection의 정보를 유지하면서 ID와 추적 상태를 추가한다.
    """
    id: int                       # 고유 추적 ID (1부터 시작)
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    confidence: float
    template_name: str

    first_seen: float = field(default_factory=time.time)   # 처음 발견된 시각
    last_seen: float = field(default_factory=time.time)    # 마지막으로 탐지된 시각
    missing_frames: int = 0         # 연속으로 탐지 실패한 프레임 수
    age: int = 0                    # 총 생존 프레임 수 (1 이상)

    def update_from_detection(self, det: MonsterDetection):
        """새로운 탐지 결과로 위치와 상태를 업데이트한다."""
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

    def mark_missing(self):
        """이번 프레임에서 탐지되지 않았을 때 호출한다."""
        self.missing_frames += 1
        self.age += 1

    @property
    def is_active(self) -> bool:
        """추적 중인 상태인지 반환한다 (아직 삭제되지 않았으면 True)."""
        return True  # 존재하면 항상 active (삭제는 Tracker가 결정)


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
    """

    def __init__(self,
                 tracking_distance: int = 80,
                 missing_frames_tolerance: int = 3):
        self._tracking_distance = tracking_distance
        self._missing_frames_tolerance = missing_frames_tolerance
        self._monsters: Dict[int, TrackedMonster] = {}   # id → TrackedMonster
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # 메인 업데이트
    # ------------------------------------------------------------------

    def update(self, detections: List[MonsterDetection]) -> List[TrackedMonster]:
        """
        새로운 탐지 결과로 추적기를 업데이트하고
        현재 살아 있는 TrackedMonster 리스트를 반환한다.
        """
        existing = list(self._monsters.values())

        # 매칭 수행
        matched_monster_ids, matched_det_indices = self._match(
            existing, detections
        )

        # 1. 매칭된 몬스터 위치 업데이트
        for monster_id, det_idx in zip(matched_monster_ids, matched_det_indices):
            self._monsters[monster_id].update_from_detection(detections[det_idx])
            detections[det_idx].id = monster_id  # id 동기화

        # 2. 매칭 안 된 기존 몬스터 → missing 처리
        unmatched_monster_ids = set(m.id for m in existing) - set(matched_monster_ids)
        for mid in unmatched_monster_ids:
            self._monsters[mid].mark_missing()

        # 3. 매칭 안 된 탐지 결과 → 새 몬스터 등록
        unmatched_det_indices = set(range(len(detections))) - set(matched_det_indices)
        for di in unmatched_det_indices:
            self._register_new_monster(detections[di])

        # 4. tolerance 초과 몬스터 삭제
        to_delete = [
            mid for mid, m in self._monsters.items()
            if m.missing_frames > self._missing_frames_tolerance
        ]
        for mid in to_delete:
            print(f"[Tracker] Monster #{mid} 삭제 "
                  f"(missing={self._monsters[mid].missing_frames})")
            del self._monsters[mid]

        return list(self._monsters.values())

    # ------------------------------------------------------------------
    # 매칭 알고리즘
    # ------------------------------------------------------------------

    def _match(self,
               existing: List[TrackedMonster],
               detections: List[MonsterDetection]
               ) -> tuple:
        """
        기존 추적 몬스터와 새 탐지 결과를 거리 기반으로 매칭한다.

        반환: (매칭된 monster_id 리스트, 매칭된 detection index 리스트)
        """
        if not existing or not detections:
            return [], []

        matched_monster_ids = []
        matched_det_indices = []
        used_det = set()

        # 거리 행렬 계산
        # 기존 몬스터 순서대로 가장 가까운 detection을 찾는다
        # (Hungarian algorithm 없이 greedy 방식으로 충분)

        # 모든 (monster, detection) 쌍의 거리를 구해서 거리 오름차순으로 정렬
        pairs = []
        for mi, monster in enumerate(existing):
            for di, det in enumerate(detections):
                dist = self._distance(monster.center_x, monster.center_y,
                                      det.center_x, det.center_y)
                if dist <= self._tracking_distance:
                    pairs.append((dist, mi, di))

        pairs.sort(key=lambda p: p[0])

        used_monsters = set()

        for dist, mi, di in pairs:
            if mi in used_monsters or di in used_det:
                continue
            # 매칭 성사
            matched_monster_ids.append(existing[mi].id)
            matched_det_indices.append(di)
            used_monsters.add(mi)
            used_det.add(di)

        return matched_monster_ids, matched_det_indices

    # ------------------------------------------------------------------
    # 유틸
    # ------------------------------------------------------------------

    def _register_new_monster(self, det: MonsterDetection):
        """새 몬스터를 추적 목록에 등록한다."""
        new_id = self._next_id
        self._next_id += 1

        monster = TrackedMonster(
            id=new_id,
            x=det.x,
            y=det.y,
            width=det.width,
            height=det.height,
            center_x=det.center_x,
            center_y=det.center_y,
            confidence=det.confidence,
            template_name=det.template_name,
        )
        self._monsters[new_id] = monster
        det.id = new_id
        print(f"[Tracker] 새 Monster #{new_id} 등록 "
              f"({det.center_x}, {det.center_y}) conf={det.confidence:.2f}")

    @staticmethod
    def _distance(x1: int, y1: int, x2: int, y2: int) -> float:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # ------------------------------------------------------------------
    # 외부 접근
    # ------------------------------------------------------------------

    @property
    def monsters(self) -> List[TrackedMonster]:
        """현재 추적 중인 모든 몬스터를 반환한다."""
        return list(self._monsters.values())

    def get_monster(self, monster_id: int) -> Optional[TrackedMonster]:
        """ID로 특정 몬스터를 반환한다. 없으면 None."""
        return self._monsters.get(monster_id)

    def is_alive(self, monster_id: int) -> bool:
        """해당 ID의 몬스터가 아직 추적 중인지 확인한다."""
        return monster_id in self._monsters

    def clear(self):
        """모든 추적 정보를 초기화한다."""
        self._monsters.clear()
        self._next_id = 1
        print("[Tracker] 추적 정보 초기화")

    @property
    def count(self) -> int:
        return len(self._monsters)
