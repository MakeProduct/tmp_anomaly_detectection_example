이 프로젝트는 **시계열 진동 데이터**를 기반으로 특징을 추출한 뒤, 다양한 이상 탐지(Anomaly Detection) 모델을 적용하는 예제를 보여줍니다.

## 파일 구조

```bash
├── main.py # 메인 실행부 (데이터 생성, 윈도우 분할, 특징 추출, 모델 학습, 평가 등)
├── feature_extraction.py # 특징 추출 모듈 (시간영역, 주파수영역, 웨이블릿 변환)
├── models.py # 모델 정의 (Autoencoder, OneClassSVM, IsolationForest)
├── train.py # 모델 학습 함수 정의 (Autoencoder, OneClassSVM, IsolationForest)
├── evaluate.py # 모델 평가 함수 정의
└── README.md # 프로젝트 소개 및 사용 방법
```

## 주요 기능

1. **가상 데이터 생성**:
    - `generate_sin_wave_data` 함수를 통해 여러 개의 사인파 + 작은 노이즈로 구성된 신호를 생성.
2. **윈도우 분할**:
    - `create_windows` 함수를 사용해 시계열 데이터를 일정 크기로 슬라이딩 분할.
3. **특징 추출**:
    - `aggregate_features` 함수로 시간영역(RMS, Peak-to-Peak 등), 주파수영역(FFT 기반 파워 대역비, 피크 주파수 등), 웨이블릿 특징(에너지, 엔트로피 등)을 추출.
4. **스케일링**:
    - scikit-learn의 `StandardScaler`를 사용해 평균 0, 표준편차 1로 정규화.
5. **모델 학습**:
    - **Autoencoder**: PyTorch 기반으로 재구성 오차를 줄이도록 학습.
    - **OneClassSVM**: 정상 데이터만으로 경계를 형성해 이상여부 판별.
    - **IsolationForest**: 트리 기반으로 이상치를 격리.
6. **평가**:
    - Autoencoder는 재구성 오차로 이상/정상(Threshold).
    - OneClassSVM과 IsolationForest는 `predict` 결과(+1/-1)를 0(정상)/1(이상)으로 변환해 사용.

## 사용 방법

1. **필요 라이브러리 설치**:
    
    ```bash
    pip install numpy pandas torch scikit-learn pywt
    ```
    
2. 실행:
    
    ```bash
    python main.py
    ```
    
3. 출력 해석:
Autoencoder: 학습 중 Epoch마다 MSELoss가 줄어드는지 확인.
OneClassSVM / IsolationForest: 각자 predict 결과를 통해 0(정상), 1(이상) 분류 현황 확인.
결과는 === Autoencoder 평가 ===, === OneClassSVM 평가 ===, === Isolation Forest 평가 === 섹션에서 확인 가능.

## 참고 사항

데이터가 전부 정상이라고 가정되어 있을 때, 일정 비율은 노이즈가 커서 이상으로 분류될 수도 있습니다.
Threshold(Autoencoder)와 모델 파라미터(OneClassSVM, IsolationForest)는 상황에 따라 조정해야 합니다.
실제 산업 데이터에 적용할 경우, 해당 데이터의 스펙(센서 종류, 샘플링 레이트 등)에 맞춰 특징 추출 함수를 수정할 수도 있습니다.
GPU가 여러 개 장착된 경우, PyTorch 모델(Autoencoder)는 자동으로 cuda:0 등을 사용합니다. 다중 GPU 병렬 처리는 추가 설정이 필요합니다.

## 출력 결과 예시(noise_scale 변경하며 테스트)

### train_signal과 test_signal이 거의 유사할 때

```bash
## train_signal(noise_scale=0.01), test_signal(noise_scale=0.01)

[Epoch 5/50] Loss: 1.008571
[Epoch 10/50] Loss: 0.953176
[Epoch 15/50] Loss: 0.896685
[Epoch 20/50] Loss: 0.838923
[Epoch 25/50] Loss: 0.789693
[Epoch 30/50] Loss: 0.744735
[Epoch 35/50] Loss: 0.706929
[Epoch 40/50] Loss: 0.674480
[Epoch 45/50] Loss: 0.649653
[Epoch 50/50] Loss: 0.627515
OneClassSVM 모델 학습 완료!
IsolationForest 모델 학습 완료!
=== Autoencoder 평가 ===   threshold=2.79025000333786
재구성 오차 예시(앞 5개): [0.41270095 0.47888365 0.8560506  0.35027066 0.26254258]
이상 판별(0=정상,1=이상) (앞 5개): [0 0 0 0 0]

=== OneClassSVM 평가 ===
결과(0=정상,1=이상) (앞 5개): [0 0 1 0 0]
decision_function 점수(앞 5개): [ 0.0085186   0.15715893 -0.19433773  0.13184959  0.63242967]
=== Isolation Forest 평가 ===
결과(0=정상,1=이상) (앞 5개): [0 0 0 0 0]
decision_function 점수(앞 5개): [0.06302402 0.05583199 0.04048268 0.08050244 0.09157262]
```

### train_signal과 test_signal이 조금 다를 때

```bash
## train_signal(noise_scale=0.01), test_signal(noise_scale=0.02)

[Epoch 5/50] Loss: 1.009172
[Epoch 10/50] Loss: 0.939183
[Epoch 15/50] Loss: 0.854338
[Epoch 20/50] Loss: 0.793880
[Epoch 25/50] Loss: 0.737501
[Epoch 30/50] Loss: 0.691659
[Epoch 35/50] Loss: 0.654859
[Epoch 40/50] Loss: 0.623014
[Epoch 45/50] Loss: 0.601146
[Epoch 50/50] Loss: 0.582984
OneClassSVM 모델 학습 완료!
IsolationForest 모델 학습 완료!
=== Autoencoder 평가 ===   threshold=2.7707815170288086
재구성 오차 예시(앞 5개): [2.926654  2.4856923 3.996944  1.9486561 1.6318251]
이상 판별(0=정상,1=이상) (앞 5개): [1 0 1 0 0]

=== OneClassSVM 평가 ===
결과(0=정상,1=이상) (앞 5개): [1 1 1 1 1]
decision_function 점수(앞 5개): [-0.6519388  -0.65195754 -0.65195782 -0.64222683 -0.65139353]
=== Isolation Forest 평가 ===
결과(0=정상,1=이상) (앞 5개): [1 1 1 0 1]
decision_function 점수(앞 5개): [-0.05038825 -0.07876052 -0.09920099  0.00225615 -0.06109936]
```

### train_signal과 test_signal이 더 많이 다를 때

```bash
## train_signal(noise_scale=0.01), test_signal(noise_scale=0.05)

[Epoch 5/50] Loss: 1.014215
[Epoch 10/50] Loss: 0.951840
[Epoch 15/50] Loss: 0.862534
[Epoch 20/50] Loss: 0.799064
[Epoch 25/50] Loss: 0.749172
[Epoch 30/50] Loss: 0.706610
[Epoch 35/50] Loss: 0.671137
[Epoch 40/50] Loss: 0.640894
[Epoch 45/50] Loss: 0.617053
[Epoch 50/50] Loss: 0.597909
OneClassSVM 모델 학습 완료!
IsolationForest 모델 학습 완료!
=== Autoencoder 평가 ===   threshold=3.969429850578308
재구성 오차 예시(앞 5개): [167.3157   176.765    167.12767   99.978676  84.94456 ]
이상 판별(0=정상,1=이상) (앞 5개): [1 1 1 1 1]

=== OneClassSVM 평가 ===
결과(0=정상,1=이상) (앞 5개): [1 1 1 1 1]
decision_function 점수(앞 5개): [-0.65195877 -0.65195877 -0.65195877 -0.65195877 -0.65195877]
=== Isolation Forest 평가 ===
결과(0=정상,1=이상) (앞 5개): [1 1 1 1 1]
decision_function 점수(앞 5개): [-0.13339344 -0.12443031 -0.13247977 -0.11415685 -0.10088279]
```