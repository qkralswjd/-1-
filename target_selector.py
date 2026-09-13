"""
target_selector.py
------------------
공격할 타겟 몬스터를 선택하는 로직만을 담당한다.

현재는 '기준점에서 가장 가까운 몬스터'만 구현한다.
타겟 선정 로직을 나중에 바꾸려면 이 파일만 수정하면 된다.
"""

import math
from typing import List, Optional

from tracker import TrackedMonster


def select_target(monsters: List[TrackedMonster],
                  ref_x: int,
                  ref_y: int) -> Optional[TrackedMonster]:
    """
    monsters 리스트에서 공격할 타겟 1개를 선택해서 반환한다.
    몬스터가 없으면 None을 반환한다.

    현재 전략: 기준점(ref_x, ref_y)에서 가장 가까운 몬스터.
    기준점은 보통 플레이어 위치 또는 화면 중앙이다.

    나중에 전략을 바꾸고 싶다면:
        - select_closest()   ← 지금 이것
        - select_oldest()    ← 가장 오래 추적된 몬스터
        - select_center()    ← 화면 중앙에 가장 가까운 몬스터
    등의 함수를 추가하고 이 함수에서 선택하면 된다.
    """
    if not monsters:
        return None
    return select_closest(monsters, ref_x, ref_y)


def select_closest(monsters: List[TrackedMonster],
                   ref_x: int,
                   ref_y: int) -> Optional[TrackedMonster]:
    """기준점에서 가장 가까운 몬스터를 반환한다."""
    if not monsters:
        return None

    best: Optional[TrackedMonster] = None
    best_dist = float("inf")

    for m in monsters:
        dist = _dist(m.center_x, m.center_y, ref_x, ref_y)
        if dist < best_dist:
            best_dist = dist
            best = m

    return best


def select_oldest(monsters: List[TrackedMonster]) -> Optional[TrackedMonster]:
    """가장 오래 추적된 몬스터를 반환한다 (age 기준)."""
    if not monsters:
        return None
    return max(monsters, key=lambda m: m.age)


def select_nearest_center(monsters: List[TrackedMonster],
                           screen_width: int,
                           screen_height: int) -> Optional[TrackedMonster]:
    """화면 중앙에서 가장 가까운 몬스터를 반환한다."""
    cx = screen_width // 2
    cy = screen_height // 2
    return select_closest(monsters, cx, cy)


def _dist(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
