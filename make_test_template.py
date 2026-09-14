"""
make_test_template.py
---------------------
게임 화면에서 몬스터 템플릿을 캡처해서 저장하는 도구.

실행:
    python make_test_template.py

실행하면 메뉴가 표시된다:
  1. 화면 캡처 후 드래그로 템플릿 추출  ← 주요 기능
  2. 기존 이미지 파일에서 추출
  3. 더미 템플릿 자동 생성 (테스트용)
  4. 저장된 템플릿 목록 보기
  5. 종료
"""

import sys
import os
import time
import cv2
import numpy as np
import mss

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

# -----------------------------------------------------------------------
# 색상 (BGR)
# -----------------------------------------------------------------------
COLOR_BOX    = (0, 255, 0)
COLOR_TEXT   = (255, 255, 255)
COLOR_GUIDE  = (0, 200, 255)
FONT         = cv2.FONT_HERSHEY_SIMPLEX


# -----------------------------------------------------------------------
# 1. 화면 직접 캡처 후 드래그로 템플릿 추출
# -----------------------------------------------------------------------

def capture_from_screen(output_name: str = None):
    """
    3초 카운트다운 후 전체 화면을 캡처하고
    드래그로 몬스터 영역을 선택해서 저장한다.
    """
    print("\n[캡처 모드]")
    print("게임 창을 준비하세요.")
    print("3초 후 화면을 캡처합니다...\n")

    # 3초 카운트다운
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("  캡처!")

    # 화면 캡처
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)[:, :, :3]  # BGRA → BGR

    print(f"  캡처 완료: {frame.shape[1]}x{frame.shape[0]}px")
    print()

    # 드래그로 템플릿 영역 선택
    _select_and_save(frame, output_name, source="screen")


# -----------------------------------------------------------------------
# 2. 파일에서 드래그로 템플릿 추출
# -----------------------------------------------------------------------

def capture_from_file(file_path: str = None, output_name: str = None):
    """기존 이미지 파일을 열고 드래그로 몬스터 영역을 선택해서 저장한다."""
    if file_path is None:
        file_path = input("이미지 파일 경로 입력: ").strip().strip('"')

    img = cv2.imread(file_path)
    if img is None:
        # 한글 경로 대응
        try:
            img = cv2.imdecode(
                np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            pass

    if img is None:
        print(f"파일 로드 실패: {file_path}")
        print("팁: 파일명에 한글이 있으면 영어로 바꿔서 저장해 보세요.")
        return

    print(f"파일 로드 완료: {file_path} ({img.shape[1]}x{img.shape[0]}px)")
    _select_and_save(img, output_name, source="file")


# -----------------------------------------------------------------------
# 공통: 드래그 선택 및 저장
# -----------------------------------------------------------------------

class _DragSelector:
    """마우스 드래그로 영역을 선택하는 헬퍼."""

    def __init__(self):
        self.start = None
        self.end = None
        self.dragging = False
        self.confirmed = False

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.end = (x, y)
            self.dragging = True
            self.confirmed = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end = (x, y)
            self.dragging = False

    def get_rect(self):
        if self.start is None or self.end is None:
            return None
        x1 = min(self.start[0], self.end[0])
        y1 = min(self.start[1], self.end[1])
        x2 = max(self.start[0], self.end[0])
        y2 = max(self.start[1], self.end[1])
        w = x2 - x1
        h = y2 - y1
        if w < 5 or h < 5:
            return None
        return (x1, y1, w, h)


def _select_and_save(img: np.ndarray, output_name: str = None, source: str = ""):
    """
    이미지에서 드래그로 영역을 선택하고 templates/에 저장한다.
    여러 번 반복해서 여러 몬스터를 저장할 수 있다.
    """
    os.makedirs(TEMPLATES_DIR, exist_ok=True)

    window = "템플릿 캡처 - 드래그로 몬스터 선택"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # 화면 크기에 맞게 창 크기 조정
    h, w = img.shape[:2]
    win_w = min(w, 1400)
    win_h = int(win_w * h / w)
    cv2.resizeWindow(window, win_w, win_h)

    selector = _DragSelector()
    cv2.setMouseCallback(window, selector.callback)

    saved_count = 0

    print("─" * 50)
    print("조작 방법:")
    print("  마우스 드래그 : 몬스터 영역 선택")
    print("  Enter / Space : 선택 영역 저장")
    print("  R             : 다시 선택")
    print("  ESC / Q       : 종료")
    print("─" * 50)

    while True:
        display = img.copy()

        # 선택 영역 그리기
        rect = selector.get_rect()
        if rect:
            x, y, rw, rh = rect
            cv2.rectangle(display, (x, y), (x + rw, y + rh), COLOR_BOX, 2)
            label = f"{rw} x {rh}px  →  Enter로 저장"
            cv2.putText(display, label, (x, max(y - 8, 16)),
                        FONT, 0.6, COLOR_BOX, 2, cv2.LINE_AA)

        # 안내 텍스트
        guide_lines = [
            "드래그: 몬스터 선택  |  Enter: 저장  |  R: 다시  |  ESC: 종료",
            f"저장됨: {saved_count}개  |  저장위치: templates/"
        ]
        for i, line in enumerate(guide_lines):
            y_pos = display.shape[0] - 30 + i * 20
            cv2.putText(display, line, (8, y_pos),
                        FONT, 0.5, COLOR_GUIDE, 1, cv2.LINE_AA)

        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF

        # Enter 또는 Space → 저장
        if key in (13, 32):
            rect = selector.get_rect()
            if rect is None:
                print("먼저 영역을 드래그로 선택하세요.")
                continue

            x, y, rw, rh = rect
            cropped = img[y:y + rh, x:x + rw]

            # 저장 이름 결정
            if output_name and saved_count == 0:
                name = output_name
            else:
                existing = [f for f in os.listdir(TEMPLATES_DIR)
                            if f.lower().endswith(".png")]
                name = f"monster_{len(existing) + 1:02d}"

            out_path = os.path.join(TEMPLATES_DIR, f"{name}.png")
            cv2.imwrite(out_path, cropped)
            saved_count += 1
            print(f"  ✅ 저장: {out_path}  ({rw}x{rh}px)")
            print("  계속 드래그해서 다른 몬스터도 저장하거나 ESC로 종료하세요.")

            # 저장 확인 표시
            confirm = display.copy()
            cv2.putText(confirm, f"저장완료: {name}.png",
                        (display.shape[1]//2 - 150, display.shape[0]//2),
                        FONT, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window, confirm)
            cv2.waitKey(800)

        # R → 선택 초기화
        elif key == ord('r') or key == ord('R'):
            selector.start = None
            selector.end = None

        # ESC 또는 Q → 종료
        elif key == 27 or key == ord('q') or key == ord('Q'):
            break

    cv2.destroyWindow(window)

    if saved_count > 0:
        print(f"\n총 {saved_count}개 템플릿 저장 완료 → {TEMPLATES_DIR}")
        print("main.py를 재실행하거나 실행 중이면 't' 키로 템플릿을 재로드하세요.")
    else:
        print("저장된 템플릿 없음.")


# -----------------------------------------------------------------------
# 3. 더미 템플릿 자동 생성
# -----------------------------------------------------------------------

def generate_dummy_templates():
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
    print(f"\n더미 템플릿 {len(templates)}개 생성 완료")


def _make_slime_template():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.circle(img, (20, 22), 16, (50, 200, 50), -1)
    cv2.circle(img, (14, 18), 4, (200, 200, 255), -1)
    cv2.circle(img, (26, 18), 4, (200, 200, 255), -1)
    cv2.circle(img, (14, 18), 2, (0, 0, 0), -1)
    cv2.circle(img, (26, 18), 2, (0, 0, 0), -1)
    return img


def _make_orc_template():
    img = np.zeros((50, 35, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)
    cv2.rectangle(img, (5, 15), (30, 45), (40, 80, 140), -1)
    cv2.ellipse(img, (17, 12), (12, 14), 0, 0, 360, (40, 80, 140), -1)
    cv2.circle(img, (12, 10), 3, (0, 200, 255), -1)
    cv2.circle(img, (22, 10), 3, (0, 200, 255), -1)
    return img


def _make_bat_template():
    img = np.zeros((30, 60, 3), dtype=np.uint8)
    img[:] = (15, 15, 15)
    pts_l = np.array([[5, 25], [20, 5], [30, 15]], np.int32)
    cv2.fillPoly(img, [pts_l], (150, 50, 150))
    pts_r = np.array([[55, 25], [40, 5], [30, 15]], np.int32)
    cv2.fillPoly(img, [pts_r], (150, 50, 150))
    cv2.ellipse(img, (30, 18), (8, 10), 0, 0, 360, (100, 30, 100), -1)
    return img


# -----------------------------------------------------------------------
# 4. 저장된 템플릿 목록
# -----------------------------------------------------------------------

def list_templates():
    if not os.path.isdir(TEMPLATES_DIR):
        print("templates/ 폴더가 없습니다.")
        return
    files = [f for f in sorted(os.listdir(TEMPLATES_DIR))
             if f.lower().endswith(".png")]
    if not files:
        print("저장된 템플릿이 없습니다.")
        return
    print(f"\n저장된 템플릿 ({len(files)}개):")
    for f in files:
        path = os.path.join(TEMPLATES_DIR, f)
        img = cv2.imread(path)
        if img is not None:
            print(f"  {f}  ({img.shape[1]}x{img.shape[0]}px)")
        else:
            print(f"  {f}  (로드 실패)")


# -----------------------------------------------------------------------
# 메인 메뉴
# -----------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  Monster Bot - 템플릿 캡처 도구")
    print("=" * 50)

    while True:
        print()
        print("메뉴:")
        print("  1. 화면 캡처 후 드래그로 템플릿 추출  ← 추천")
        print("  2. 이미지 파일에서 추출")
        print("  3. 더미 템플릿 자동 생성 (테스트용)")
        print("  4. 저장된 템플릿 목록 보기")
        print("  5. 종료")
        print()

        choice = input("선택 (1-5): ").strip()

        if choice == "1":
            name = input("저장할 이름 (Enter=자동): ").strip()
            capture_from_screen(name if name else None)

        elif choice == "2":
            path = input("이미지 파일 경로: ").strip().strip('"')
            name = input("저장할 이름 (Enter=자동): ").strip()
            capture_from_file(path, name if name else None)

        elif choice == "3":
            generate_dummy_templates()

        elif choice == "4":
            list_templates()

        elif choice == "5":
            print("종료합니다.")
            break

        else:
            print("1~5 중에서 선택하세요.")


if __name__ == "__main__":
    main()
