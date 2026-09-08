# 프로젝트 개요 — 2026 SNU FastMRI Challenge (Phase 2)

## 과제
undersampled 무릎 MRI k-space로부터 원본 이미지를 복원하는 딥러닝 모델을 만들어 리더보드
combined 점수를 최대화한다. 가속 배율은 acc4(4배)와 acc8(8배) 두 가지. 병변(lesion) 부위
복원이 별도로 채점되는 것이 이 대회의 임상적 특징이다.

## 모델 — baby-VarNet
E2E-VarNet의 축소판.
- SensitivityModel: 15개 코일의 감도맵 추정(k-space 중앙 저주파 ACS 영역 사용).
- cascade 블록: 데이터 일관성(DC, 측정된 k-space 보존) + U-Net 정규화(미측정 부분 복원)를 반복.
- 출력: 역푸리에 변환 → 코일 결합(RSS) → 384×384 중앙 크롭.
- 조절 하이퍼파라미터: cascade 수, chans(cascade U-Net 채널), sens_chans(감도맵 U-Net 채널).

## 데이터
- train: acc4 85볼륨 + acc8 85볼륨 = 170볼륨, 슬라이스 4651장.
- val: acc4 15 + acc8 15.
- leaderboard(공개): acc4 40 + acc8 40. 별도 비공개(private) 데이터 존재.
- 각 image 파일: 정답 image_label(384x384), 병변 주석 annotations(bounding box), 정규화 상수 max.
- 각 kspace 파일: 완전 샘플링 k-space(코일 15, 높이 640, 너비 W), 샘플링 마스크 mask(길이 W).
  (W는 볼륨마다 다름: 320, 322, 356, 368, 372, 386, 388, 400, 480 등.)

## 채점
- 공식: Final = 0.5*SSIM_full + 0.5*SSIM_bbox + Tiebreaker
- 실질: acc4/acc8을 따로 채점 후 반반 → 4분면 각 25%:
  Final = 0.25*full_acc4 + 0.25*full_acc8 + 0.25*bbox_acc4 + 0.25*bbox_acc8 + Tiebreaker
- SSIM_full: 전경 마스크 안에서만 평균.
- SSIM_bbox: 병변 박스별 1표 평균(크기 무관, 전경 마스크 미사용).
- Tiebreaker: 추론 속도. 80ms/slice=0.0010, 2000ms/slice=0 선형. 30회 반복 후 최소 시간 사용.
- 반영비: 공개 리더보드 40% + 비공개 60%. 일반화가 지배적 변수.
- 실제 함수는 code/metrics.py, code/recon_eval.py 참조.

## 컴플라이언스
- 추론 시 이미지 파일 필드(병변 주석, 참고 GRAPPA) 접근 금지(box-blind).
- 어노테이션은 학습에서만 사용 가능.
- leaderboard 데이터는 학습 금지(관찰·평가만). 마스크도 train/val 원리에서 생성만 허용.
- 모든 복원 연산은 recon_slice() 내부에서(시간 측정 회피 금지).
- 채점 하네스 recon_eval.py는 수정 금지(운영진이 그대로 재실행).

## 환경
- GPU: GTX 1080 1장(학습 시 약 5.8GB). 순차 실행만 가능, 실험당 1~2.5일.
- VESSL 클라우드, JupyterLab. 브랜치 2026_baby_varnet_phase2.
- 고정 시드 430(이 값도 초기 실험을 통해 선택). 고정 백본(주력): cascade 6, chans 9, sens_chans 8.

## 재현성
- 최종 제출은 단일 bash recon_eval.sh로 재현 가능해야 하며, README대로 재학습 시 소수점 4자리까지 일치.
- 학습·평가 로그, PROGRESS.md, decisions.log, 라운드별 결과 CSV 보존.
