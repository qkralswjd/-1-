"""
make_test_template.py
---------------------
실제 게임 스크린샷에서 몬스터 템플릿을 잘라내는 도우미 스크립트.

사용법:
    python make_test_template.py <screenshot_path> [output_name]

실행 후 마우스로 드래그해서 몬스터 영역을 선택하면
templates/ 디렉토리에 저장된다.

또는 --generate 옵션으로 테스트용 더미 템플릿을 자동 생성한다:
    python make_test_template.py --generate
"""

import sys
import os
import cv2
import numpy as np

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


def generate_dummy_templates():
    """
    테스트 목적으로 간단한 도형 템플릿을 생성한다.
    실제 게임 없이 매칭 동작을 검증할 때 사용한다.
    """
    os.makedirs(TEMPLATES_DIR, exist_ok=True)

    templates = [
        ("monster_01", _make_slime_template),
        ("monster_02", _make_orc_template),
        ("monster_03", _make_bat_template),
    ]

    for name, fn in templates:
        img = fn()
        path = os.path.join(TEMPLATES_DIR, f"{name}.png")
        cv2.imwrite(path, img)
        print(f"  생성: {path} ({img.shape[1]}x{img.shape[0]}px)")

    print(f"\n더미 템플릿 {len(templates)}개 생성 완료 → {TEMPLATES_DIR}")
    print("※ 이 템플릿은 테스트용입니다. 실제 게임 화면에서는 실제 몬스터를 캡처하세요.")


def _make_slime_template() -> np.ndarray:
    """슬라임 모양 더미 템플릿 (초록색 원)."""
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)   # 어두운 배경
    cv2.circle(img, (20, 22), 16, (50, 200, 50), -1)   # 초록 몸통
    cv2.circle(img, (14, 18), 4, (200, 200, 255), -1)  # 눈 왼쪽
    cv2.circle(img, (26, 18), 4, (200, 200, 255), -1)  # 눈 오른쪽
    cv2.circle(img, (14, 18), 2, (0, 0, 0), -1)
    cv2.circle(img, (26, 18), 2, (0, 0, 0), -1)
    return img


def _make_orc_template() -> np.ndarray:
    """오크 모양 더미 템플릿 (갈색 사각형)."""
    img = np.zeros((50, 35, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)
    # 몸통
    cv2.rectangle(img, (5, 15), (30, 45), (40, 80, 140), -1)
    # 머리
    cv2.ellipse(img, (17, 12), (12, 14), 0, 0, 360, (40, 80, 140), -1)
    # 눈
    cv2.circle(img, (12, 10), 3, (0, 200, 255), -1)
    cv2.circle(img, (22, 10), 3, (0, 200, 255), -1)
    return img


def _make_bat_template() -> np.ndarray:
    """박쥐 모양 더미 템플릿 (보라색 날개)."""
    img = np.zeros((30, 60, 3), dtype=np.uint8)
    img[:] = (15, 15, 15)
    # 왼쪽 날개
    pts_l = np.array([[5, 25], [20, 5], [30, 15]], np.int32)
    cv2.fillPoly(img, [pts_l], (150, 50, 150))
    # 오른쪽 날개
    pts_r = np.array([[55, 25], [40, 5], [30, 15]], np.int32)
    cv2.fillPoly(img, [pts_r], (150, 50, 150))
    # 몸통
    cv2.ellipse(img, (30, 18), (8, 10), 0, 0, 360, (100, 30, 100), -1)
    return img


def capture_from_screenshot(screenshot_path: str, output_name: str = None):
    """스크린샷에서 마우스 드래그로 템플릿을 잘라낸다."""
    img = cv2.imread(screenshot_path)
    if img is None:
        print(f"이미지 로드 실패: {screenshot_path}")
        return

    print(f"이미지 로드: {screenshot_path}")
    print("마우스로 드래그해서 몬스터 영역을 선택하세요.")
    print("Enter 또는 Space: 저장 | ESC: 취소")

    roi = cv2.selectROI("Template Capture", img, showCrosshair=True)
    cv2.destroyWindow("Template Capture")

    x, y, w, h = [int(v) for v in roi]
    if w == 0 or h == 0:
        print("선택 취소")
        return

    cropped = img[y:y+h, x:x+w]

    os.makedirs(TEMPLATES_DIR, exist_ok=True)

    if output_name is None:
        existing = [f for f in os.listdir(TEMPLATES_DIR) if f.endswith(".png")]
        output_name = f"monster_{len(existing)+1:02d}"

    out_path = os.path.join(TEMPLATES_DIR, f"{output_name}.png")
    cv2.imwrite(out_path, cropped)
    print(f"저장: {out_path} ({w}x{h}px)")


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "--generate":
        print("더미 테스트 템플릿 생성 중...")
        generate_dummy_templates()
    else:
        screenshot_path = sys.argv[1]
        output_name = sys.argv[2] if len(sys.argv) > 2 else None
        capture_from_screenshot(screenshot_path, output_name)
