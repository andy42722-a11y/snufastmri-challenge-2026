# 2026 SNU FastMRI Challenge — Knee MRI Reconstruction

undersampled(부분 수집) 무릎 MRI k-space로부터 원본 이미지를 복원하는 딥러닝 모델을,
GTX 1080 한 장이라는 제약 안에서 **가설–대조군–판정의 통제된 실험**으로 개선한 프로젝트입니다.
최종 모델은 리더보드 **combined SSIM 0.92295** 를 기록했습니다.

> 이 저장소는 실험의 **과정·근거·의사결정**을 문서화한 기록입니다. 대회 및 서버 환경
> 종료로 원본 데이터(약 153GB)와 학습된 가중치(.pt)는 포함하지 않으며, 실제 평가 코드와
> 모델·데이터 파이프라인 코드, 라운드별 결과 수치를 담았습니다.

---

## 목차
- [1. 대회 개요](#1-대회-개요)
- [2. 평가 방법과 채점 지표](#2-평가-방법과-채점-지표)
- [3. 실험 환경과 원칙](#3-실험-환경과-원칙)
- [4. 라운드 요약](#4-라운드-요약)
- [5. 저장소 구조](#5-저장소-구조)
- [6. 핵심 교훈](#6-핵심-교훈)

---

## 1. 대회 개요

**MRI 가속의 문제.** MRI는 k-space(주파수 영역)를 한 줄씩 수집한 뒤 푸리에 변환으로 이미지를
만든다. 촬영을 빠르게 하려면 k-space의 일부 줄만 수집(undersampling)하는데, 그러면 이미지에
앨리어싱(aliasing)이라는 겹침 아티팩트가 생긴다. 이 대회의 과제는 **그렇게 손상된 데이터로부터
원본을 복원하는 딥러닝 모델**을 만드는 것이다. 가속 배율은 두 가지 — acc4(4배), acc8(8배).

**임상적 초점 — 병변(lesion).** 일반 화질 복원과 달리, 이 대회는 무릎 병변(인대 파열, 반월상
연골 손상, 골 병변 등) 부위의 복원을 별도로 채점한다. 배경이 깨끗해도 병변이 뭉개지면 진단에
쓸 수 없기 때문이다. 병변 위치는 fastMRI+ 데이터셋의 bounding box 주석으로 주어진다.
단, **학습 때는 병변 위치를 써도 되지만 추론 때는 볼 수 없다**(box-blind 추론) — 실제 환자를
찍을 때는 병변 위치를 미리 알 수 없기 때문이다.

**베이스라인 모델 — baby-VarNet.** End-to-End Variational Network(E2E-VarNet)의 축소판.
① SensitivityModel이 15개 코일의 감도맵을 추정하고(k-space 중앙 저주파 ACS 영역 사용),
② cascade 블록들이 "데이터 일관성(측정된 k-space는 보존) + U-Net 정규화(미측정 부분을 채움)"를
반복하며, ③ 역푸리에 변환 → 코일 결합(RSS) → 384×384 중앙 크롭으로 이미지를 낸다.

**데이터 구성.**
- train: acc4 85볼륨 + acc8 85볼륨 (총 170개, 슬라이스 4651장). 각 볼륨은 완전 샘플링 k-space와
  정답 이미지(`image_label`), 병변 주석(`annotations`), 정규화 상수(`max`)를 담는다.
- val: acc4 15볼륨 + acc8 15볼륨.
- leaderboard(공개): acc4 40볼륨 + acc8 40볼륨. 별도의 비공개(private) 데이터가 존재.

---

## 2. 평가 방법과 채점 지표

**공식 점수식**

```
Final = 0.5 · SSIM_full + 0.5 · SSIM_bbox + Tiebreaker
```

- **SSIM_full** — 전경 마스크(몸이 있는 영역) 안에서만 평균한 SSIM.
- **SSIM_bbox** — 병변 박스 안에서만 계산한 SSIM. **박스 하나당 1표**로 평균(박스 크기 무관).
- **Tiebreaker** — 추론 속도. 80ms/slice에서 0.0010점, 2000ms/slice에서 0점으로 선형 감소.
  최종 채점 시 같은 추론을 30회 반복해 **최소 시간**을 사용.

**실질적으로는 4분면 각 25%.** 채점 코드는 acc4와 acc8을 각각 따로 채점한 뒤 반반 섞으므로,
점수를 풀어 쓰면 다음과 같다. 이 관점이 이 프로젝트 전체의 판단 기준이 되었다.

```
Final = 0.25·full_acc4 + 0.25·full_acc8 + 0.25·bbox_acc4 + 0.25·bbox_acc8 + Tiebreaker
```

**반영 비율** — 최종 점수 = 공개 리더보드 40% + 비공개(private) 60%. 즉 눈에 보이는 리더보드
점수는 최종의 40%에 불과하며, **일반화(generalization)가 지배적 변수**다.

**실제 채점 함수** ([`code/metrics.py`](code/metrics.py), [`code/recon_eval.py`](code/recon_eval.py))

`recon_eval.py`는 운영진이 그대로 재실행하는 고정 하네스로, 4분면을 이렇게 집계한다:

```python
print("Leaderboard SSIM_full : {:.4f}".format((full4 + full8) / 2))
print("Leaderboard SSIM_bbox : {:.4f}".format((bbox4 + bbox8) / 2))
```

SSIM_full은 전경 마스크 안에서만 평균한다(`metrics.py`):

```python
def ssim_full(ssim, recon_t, target_t, mask_t, data_range):
    ssim_map = ssim(recon_t * mask_t, target_t * mask_t, data_range)
    pad = ssim.win_size // 2
    mask_valid = mask_t[pad:mask_t.shape[0] - pad, pad:mask_t.shape[1] - pad]
    denom = mask_valid.sum()
    if denom <= 0:
        return None
    return ((ssim_map * mask_valid).sum() / denom).item()
```

전경 마스크는 정답 이미지의 형태학 연산으로 만들어진다:

```python
def foreground_mask(target):
    mask = np.zeros(target.shape)
    mask[target > 2e-5] = 1
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=15)
    mask = cv2.erode(mask, kernel, iterations=14)
    return mask
```

SSIM_bbox는 박스별로 잘라 계산하며, win_size(7)보다 작은 박스는 건너뛴다:

```python
def ssim_bbox(ssim, recon_t, target_t, box, data_range):
    win = ssim.win_size
    x0, y0 = max(0, box["x"]), max(0, box["y"])
    x1 = min(target_t.shape[1], box["x"] + box["width"])
    y1 = min(target_t.shape[0], box["y"] + box["height"])
    if (x1 - x0) < win or (y1 - y0) < win:
        return None
    recon_crop = recon_t[y0:y1, x0:x1]
    target_crop = target_t[y0:y1, x0:x1]
    return ssim(recon_crop, target_crop, data_range).mean().item()
```

**컴플라이언스** — 추론 시 이미지 파일 필드(병변 주석, 참고 GRAPPA) 접근 금지. 모든 복원 연산은
`recon_slice()` 내부에서 수행(시간 측정 회피 금지). 채점 하네스(`recon_eval.py`) 수정 금지.
최종 제출은 단일 명령으로 재현 가능해야 하며 소수점 4자리까지 일치해야 함.

---

## 3. 실험 환경과 원칙

- **GPU:** GTX 1080 1장. 실험 하나당 1~2.5일 소요. 순차 실행만 가능해 "무엇을 실험할지"의
  선택이 곧 전략이었다. 반면 리더보드 평가는 약 10분 — 이 비대칭이 여러 설계 결정의 근거가 됨.
- **자동화:** tmux detached 세션으로 학습을 백그라운드 실행. 라운드마다 사전검증 게이트(잘못된
  설정으로 GPU를 낭비하기 전에 차단)와 위험 단계 정지 지점을 두어 관리.
- **핵심 원칙:**
  1. **한 번에 한 변수 + 대조군.** 원인을 명확히 귀속하기 위해.
  2. **리더보드 4분면으로 판정.** 검증셋(val)은 신뢰하지 않음(→ Round 4 참조).
  3. **채택 임계 +0.0010.** SSIM의 지각 분해능(그보다 작은 차이는 사람 눈에 안 보임)에 근거.
  4. **고정 시드(430)** — 재현성 확보. 이 시드 값 자체도 초기 실험을 통해 선택했다.

---

## 4. 라운드 요약

각 라운드의 가설·변경점·결과·판정·근거는 [`docs/`](docs/)에 상세히 있습니다.
Round 5–7은 데이터 분석과 해법 도출 과정을 특히 자세히 기록했습니다.

| 라운드 | 핵심 아이디어 | combined | 판정 |
|---|---|---:|---|
| [R1](docs/round-01-baseline.md) | 용량 스윕 (cascade/chans, 12ep) | 0.9126 | 기준 확립 |
| [R2](docs/round-02-loss-intervention.md) | bbox 가중 / 고주파 loss (12ep) | ≤0.9126 | 보류 |
| [R3](docs/round-03-training-duration.md) | 30ep + 코사인 스케줄러 | 0.9162 | 채택 |
| [R4](docs/round-04-validation-lesson.md) | 50ep 확장 + 검증 신뢰성 | 0.9173 | 채택 |
| [R5](docs/round-05-foreground-loss.md) | 전경 마스크 loss | 0.9179 | 채택 |
| [R6](docs/round-06-mask-augmentation.md) | 마스크 증강 (분포 이동 교정) | 0.92195 | 채택 |
| [**R7**](docs/round-07-lesion-oversampling.md) | 병변 슬라이스 오버샘플링 | **0.92295** | **최종** |

*(각 combined는 해당 라운드 최고 후보 기준. 세부 4분면 수치는 각 문서와 [`results/`](results/) CSV 참조.)*

---

## 5. 저장소 구조

```
README.md                         이 문서
docs/
├── 00-project-summary.md         대회·평가·환경 상세
├── round-01-baseline.md          ~ round-07-lesion-oversampling.md
└── (라운드별: 가설/변경/결과 수치/판정/코드/인사이트)
notes/
├── key-learnings.md              라운드를 관통하는 원칙
└── data-analysis.md              데이터 정밀 분석 기록(마스크 분포, k-space, 병변 통계)
code/
├── metrics.py                    실제 채점 함수 (SSIM_full / SSIM_bbox / foreground_mask)
├── recon_eval.py                 채점+시간측정 하네스
├── varnet.py                     baby-VarNet 모델
├── test_part.py                  추론 인터페이스 (prep_volume / recon_slice)
├── transforms.py, transforms_bbox.py, load_data_bbox.py, train_bbox.py
results/
└── round1~7 결과 CSV (4분면 수치 포함)
```

---

## 6. 핵심 교훈

- **수렴 전에는 비교하지 마라.** 초기 라운드의 "실패"는 아이디어가 아니라 학습 시간 부족 탓이었다.
- **측정 도구를 먼저 의심하라.** 검증셋이 모델 순위를 뒤집는 것을 발견하고 판정 기준을 바꿨다.
- **데이터 분석이 곧 성능이다.** 리더보드와 학습 데이터의 분포 차이를 분석으로 찾아내 가장 큰 개선을 만들었다.
- **한 번에 한 변수 + 대조군.** 모든 개선을 "왜 올랐는지" 설명할 수 있게 설계했다.

자세한 내용은 [`notes/key-learnings.md`](notes/key-learnings.md)를 참고하세요.
