# 2026 SNU FastMRI Challenge — Knee MRI Reconstruction

촬영 시간이 긴 MRI를 빠르게 하기 위해 k-space 데이터의 일부만 수집(undersampling)하고, 그 데이터로부터 원본 이미지를 복원하는 딥러닝 모델을 만드는 대회입니다. 데이터를 약 4분의 1만 수집한 경우와 8분의 1만 수집한 경우 모두에서 복원 품질이 가장 높은 모델을 만드는 것이 목표입니다. 지원받은 GPU(GTX 1080) 한 장으로, 한 번에 하나의 변수만 바꿔 가며 실험해 성능을 개선했습니다. 결국 모델을 키우는 것보다 데이터의 특성을 분석하는 것이 더 큰 개선을 만들었습니다. 학습 데이터와 평가 데이터의 샘플링 패턴이 다르다는 것을 찾아내 교정한 것이 가장 큰 개선이었습니다. 리더보드 combined SSIM은 0.9126에서 0.9230으로 올랐습니다.

> 이 저장소는 실험의 **과정·근거·의사결정**을 문서화한 기록입니다. 대회 및 서버 환경
> 종료로 원본 데이터(약 153GB)와 학습된 가중치(.pt)는 포함하지 않으며, 실제 평가 코드와
> 모델·데이터 파이프라인 코드, 라운드별 결과 수치를 담았습니다.

## 코드 출처

대회 측 베이스라인([LISTatSNU/FastMRI_challenge](https://github.com/LISTatSNU/FastMRI_challenge), 2026_baby_varnet_phase2 브랜치)을 바탕으로 했습니다.

- 베이스라인 그대로: varnet.py, recon_eval.py, metrics.py, test_part.py, transforms.py
- 새로 작성·수정: transforms_bbox.py, load_data_bbox.py, train_bbox.py, round6_mask.py (Round 6 마스크 증강의 마스크 생성 로직)
- 유실된 코드: 학습 루프(`train_part_bbox.py`)와 마스크 증강을 데이터 파이프라인에 연결한 부분은 대회 서버 종료로 남아 있지 않습니다. 연결 방식은 [Round 6 문서](docs/round-06-mask-augmentation.md)에 정리했으며, 이 저장소의 코드만으로는 학습을 그대로 재현할 수 없습니다.

## 팀 구성과 역할

2인 팀으로 참가했습니다. 이 저장소의 Round 1~7은 제가 진행한 부분이고, 그 이후 팀원이 이어서 진행한 작업은 포함하지 않았습니다.

제가 한 일:
- 학습 환경 세팅과 실험 자동화 (tmux, 사전검증 게이트)
- Round 1~7 실험 설계, 실행, 결과 정리와 문서화
- 채점 코드를 분석해 학습 손실과 채점식의 불일치를 찾고 전경 마스크 손실로 교정 (R5)
- 학습·리더보드 마스크 분포를 전수 비교해 분포 이동을 찾고 마스크 증강으로 교정 (R6, 단일 최대 개선 +0.0040)
- 병변 슬라이스 오버샘플링과 교란을 분리하는 대조군 설계 (R7, combined 0.92295)

---

## 목차
- [1. 대회와 채점 요약](#1-대회와-채점-요약)
- [2. 실험 환경과 원칙](#2-실험-환경과-원칙)
- [3. 라운드 요약](#3-라운드-요약)
- [4. 저장소 구조](#4-저장소-구조)
- [5. 핵심 교훈](#5-핵심-교훈)

---

## 1. 대회와 채점 요약

- 과제: 4배(acc4)와 8배(acc8) 가속 조건에서 무릎 MRI를 복원합니다. 병변 부위 복원 품질을 따로 채점하는 것이 특징입니다.
- 채점: 전체(전경) SSIM 50%와 병변 박스 SSIM 50%를 acc4·acc8 반반으로 합쳐, 실질적으로 네 칸이 25%씩입니다. 공개 리더보드는 최종 점수의 40%뿐이라 일반화가 중요했습니다.
- 모델 구조, 데이터 구성, 채점 함수, 대회 규정은 [프로젝트 개요](docs/00-project-summary.md)에 정리했습니다.

---

## 2. 실험 환경과 원칙

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

## 3. 라운드 요약

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

## 4. 저장소 구조

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
├── round6_mask.py                Round 6 마스크 증강의 마스크 생성 로직
results/
└── round1~7 결과 CSV (4분면 수치 포함)
```

---

## 5. 핵심 교훈

- **수렴 전에는 비교하지 마라.** 초기 라운드의 "실패"는 아이디어가 아니라 학습 시간 부족 탓이었다.
- **측정 도구를 먼저 의심하라.** 검증셋이 모델 순위를 뒤집는 것을 발견하고 판정 기준을 바꿨다.
- **데이터 분석이 곧 성능이다.** 리더보드와 학습 데이터의 분포 차이를 분석으로 찾아내 가장 큰 개선을 만들었다.
- **한 번에 한 변수 + 대조군.** 모든 개선을 "왜 올랐는지" 설명할 수 있게 설계했다.

자세한 내용은 [`notes/key-learnings.md`](notes/key-learnings.md)를 참고하세요.
