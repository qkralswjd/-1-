# Monster Bot v1.0

실시간 화면 캡처 + 템플릿 매칭 + 몬스터 추적 + 자동 공격 봇 (1차 버전)

---

## 현재 구현된 기능

- [x] 실시간 화면 캡처 (mss)
- [x] 마우스 드래그로 몬스터 탐지 ROI 지정
- [x] ROI 안에서만 템플릿 매칭 (cv2.matchTemplate)
- [x] NMS(IoU 기반) 중복 탐지 제거
- [x] 거리 기반 몬스터 추적 (ID 유지)
- [x] missing_frames 허용 (잠깐 탐지 실패해도 ID 유지)
- [x] 가장 가까운 몬스터 타겟 선정
- [x] 공격 cooldown 제어 (매 프레임 클릭 방지)
- [x] 타겟 소실 시 자동 전환
- [x] Pico HID 컨트롤러 추상화
- [x] 디버그 오버레이 (FPS, 박스, ID, 상태)

---

## 미구현 (2차 이후)

- [ ] OCR (레벨/HP 인식)
- [ ] 포션, 텔레포트
- [ ] 루팅, 아데나 탐지
- [ ] 이동 경로 (웨이포인트)
- [ ] Kalman Filter 추적
- [ ] YOLO 등 AI 객체 탐지

---

## 파일 구조

```
monster_bot/
├── main.py              ← 전체 실행 흐름 조율
├── config.py            ← 설정 로드/저장
├── screen_capture.py    ← 화면 캡처 (mss)
├── detector.py          ← 템플릿 매칭 + NMS
├── tracker.py           ← 몬스터 ID 추적
├── target_selector.py   ← 타겟 선정 로직
├── controller.py        ← 입력 추상화 (Pico HID)
├── overlay.py           ← 화면 표시 (OpenCV)
├── config.json          ← 설정값
├── make_test_template.py← 템플릿 캡처 도구
└── templates/
    ├── monster_01.png
    ├── monster_02.png
    └── ...
```

---

## 설치

```bash
pip install mss opencv-python numpy pyserial
```

---

## 실행

```bash
cd monster_bot
python main.py
```

### 실행 흐름

1. 프로그램 시작
2. ROI 설정 (기존 config 사용 또는 새로 지정)
3. 메인 루프 시작

### 키 조작

| 키 | 동작 |
|----|------|
| `ESC` / `q` | 종료 |
| `r` | ROI 재설정 |
| `t` | 템플릿 재로드 |
| `c` | 추적 정보 초기화 |
| `p` | 일시정지 토글 |

---

## 템플릿 준비

### 더미 템플릿 자동 생성 (테스트용)

```bash
python make_test_template.py --generate
```

### 실제 게임 화면에서 템플릿 캡처

```bash
python make_test_template.py screenshot.png monster_slime
```

실행 후 마우스로 드래그해서 몬스터 영역을 선택한다.

---

## config.json 설명

```json
{
    "roi": {
        "x": 100, "y": 200,
        "width": 800, "height": 500
    },
    "capture": {
        "capture_fps": 30,    // 화면 캡처 FPS
        "detection_fps": 10   // 탐지 실행 FPS (높을수록 CPU 부하 증가)
    },
    "detection": {
        "match_threshold": 0.75,       // 탐지 신뢰도 임계값 (높을수록 엄격)
        "nms_overlap_threshold": 0.3   // 중복 제거 IoU 임계값
    },
    "tracking": {
        "tracking_distance": 80,           // 같은 몬스터로 판단하는 최대 거리(px)
        "missing_frames_tolerance": 3      // 탐지 실패 허용 프레임 수
    },
    "attack": {
        "attack_cooldown": 0.5   // 공격 사이 최소 간격(초)
    },
    "controller": {
        "type": "pico",
        "port": "COM3",
        "baudrate": 115200,
        "enabled": false   // true로 변경하면 실제 Pico에 연결 시도
    },
    "reference_point": {
        "x": 960, "y": 540   // 타겟 선정 기준점 (보통 플레이어 위치)
    }
}
```

---

## 화면 표시 색상

| 색상 | 의미 |
|------|------|
| 파란색 | ROI 영역 |
| 초록색 | 일반 몬스터 |
| 빨간색 | 현재 타겟 |
| 노란색 십자 | 기준점(플레이어) |

---

## 봇 상태

```
SEARCHING       → 몬스터 탐색 중
TARGET_SELECTED → 타겟 선정됨
ATTACKING       → 공격 중 (cooldown 제어)
TARGET_LOST     → 타겟 소실 → 다시 SEARCHING
```

---

## Pico HID 연결 (선택사항)

`config.json`에서:
```json
"controller": {
    "enabled": true,
    "port": "COM3",   // Windows: COM3, Linux: /dev/ttyACM0
    "baudrate": 115200
}
```

Pico 펌웨어는 JSON 명령을 시리얼로 수신한다:
```json
{"cmd": "click", "x": 500, "y": 300, "btn": "left"}
{"cmd": "move",  "x": 500, "y": 300}
{"cmd": "key",   "key": "a"}
```

---

## 테스트 체크리스트

- [ ] 화면이 실시간으로 표시되는가?
- [ ] ROI가 정확하게 지정되는가?
- [ ] ROI 안에서만 몬스터를 찾는가?
- [ ] 같은 몬스터가 중복 박스로 표시되지 않는가?
- [ ] 몬스터가 움직여도 ID가 유지되는가?
- [ ] 여러 몬스터 중 하나만 타겟으로 잡는가?
- [ ] 타겟이 움직이면 공격 좌표도 따라가는가?
- [ ] 타겟이 사라지면 다음 타겟으로 넘어가는가?
- [ ] 탐지가 잠깐 실패해도 ID가 바로 사라지지 않는가?
- [ ] 화면 FPS와 Detection FPS가 서로 방해하지 않는가?
