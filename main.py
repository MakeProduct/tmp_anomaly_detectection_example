# main.py

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# feature_extraction 모듈에서 전체 특징 추출 함수와 일부 특징만 추출하는 함수를 불러옴
from feature_extraction import (
    aggregate_features,
    aggregate_features_selected
)

# models.py 파일에서 정의한 모델(오토인코더, OneClassSVM, IsolationForest)을 불러옴
from models import SimpleAutoencoder, OC_SVMModel, IForestModel

# train.py 파일에서 정의한 훈련 함수(오토인코더 학습, OneClassSVM 학습, IsolationForest 학습)를 불러옴
from train import train_autoencoder, train_oneclass_svm, train_isolation_forest

# evaluate.py 파일에서 정의한 평가 함수(오토인코더, OneClassSVM, IsolationForest)를 불러옴
from evaluate import evaluate_autoencoder, evaluate_oneclass_svm, evaluate_isolation_forest


def generate_sin_wave_data(num_samples=2000,
                           base_amplitude=0.2,
                           noise_scale=0.01,
                           random_state=42):
    """
    [개선된 사인 파형 데이터 생성 함수]
    - 여러 개의 사인파를 작은 진폭(amplitude)으로 합성하여 큰 변동을 방지
    - 노이즈 또한 scale을 작게 설정해 실제 데이터 변동을 최소화
    - random_state로 매번 동일한 결과 재현 가능

    Args:
        num_samples (int): 총 샘플 수 (시계열 길이)
        base_amplitude (float): 각 사인파의 기본 진폭 (사인파 높이를 얼마로 할지 결정)
        noise_scale (float): 노이즈의 표준편차 (작을수록 노이즈 영향이 작음)
        random_state (int): 난수 시드 (재현성을 위해 고정)

    Returns:
        np.ndarray: (num_samples,) 형태의 합성 신호 (사인파 + 노이즈)
    """
    # 재현성 확보를 위한 시드 고정
    rng = np.random.default_rng(seed=random_state)

    # 사용할 사인파 주파수 설정 (예: 5개 주파수)
    freqs = [1, 3, 5, 7, 9]

    # 시간 축 생성 (0 ~ 2π 범위 균등 분할)
    t = np.linspace(0, 2*np.pi, num_samples)

    # 여러 개 사인파를 작은 진폭으로 합산
    signal = np.zeros_like(t)
    for f in freqs:
        # 각 사인파에 랜덤 위상(phase)을 더해준다 (다양한 모양의 사인파 생성)
        phase = rng.uniform(0, 2*np.pi)
        signal += base_amplitude * np.sin(f * t + phase)

    # 가우시안 노이즈 추가 (noise_scale만큼 표준편차를 가짐)
    noise = rng.normal(loc=0.0, scale=noise_scale, size=num_samples)

    # 사인파 + 노이즈가 결합된 최종 신호
    signal_noisy = signal + noise

    return signal_noisy


def create_windows(signal, window_size=200, step_size=200):
    """
    [신호를 일정 크기의 윈도우로 분할하는 함수]
    
    Args:
        signal (np.ndarray): 전체 시계열 신호 (1차원)
        window_size (int): 한 번에 자를 길이 (샘플 수)
        step_size (int): 윈도우 간 이동 간격 (슬라이딩 윈도우 개념)

    Returns:
        List[np.ndarray]: (samples, 1) 형태를 갖는 여러 윈도우의 리스트
    """
    windows = []
    start = 0
    # start + window_size가 신호 길이를 초과하지 않는 한 계속 분할
    while start + window_size <= len(signal):
        segment = signal[start:start+window_size]
        # (samples,) -> (samples, 1) 형태로 변환
        segment = segment.reshape(-1, 1)
        windows.append(segment)
        # 다음 윈도우 시작점
        start += step_size
    return windows


def main():
    """
    [메인 실행부]
    - 데이터 생성 -> 윈도우 분할 -> 특징 추출 -> 스케일링 -> 모델 학습 -> 평가
    """
    # 0) device 설정 (GPU 사용 가능 여부 확인)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) 학습 데이터 생성 (사인 파형 + 노이즈)
    train_signal = generate_sin_wave_data(
        num_samples=20000,
        base_amplitude=0.2,
        noise_scale=0.01,
        random_state=42  # 시드 고정
    )
    # 2차원 윈도우 분할
    train_windows = create_windows(train_signal, window_size=20, step_size=10)

    # 2) 테스트 데이터 생성 (다른 시드로 생성 -> 조금 다른 위상/노이즈)
    test_signal = generate_sin_wave_data(
        num_samples=20000,
        base_amplitude=0.2,
        noise_scale=0.01,
        random_state=84  # 시드 고정 (42와 다름)
    )
    test_windows = create_windows(test_signal, window_size=20, step_size=10)

    # 3) 특징 추출 (시간영역, 주파수영역, 웨이블릿 영역 등)
    df_train_features = aggregate_features(train_windows, fs=100)
    df_test_features = aggregate_features(test_windows, fs=100)

    print("[Train Features] describe() 결과:")
    print(df_train_features.describe(), "\n")

    # 스케일링(표준화) - 평균 0, 표준편차 1로 변환
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train_features)
    X_test_scaled = scaler.transform(df_test_features)

    # 4) 모델 학습
    # 4-1) Autoencoder (파이토치 기반)
    input_dim = X_train_scaled.shape[1]  # 특징 차원
    autoencoder = SimpleAutoencoder(input_dim=input_dim, latent_dim=64)  # latent_dim 확대
    autoencoder.to(device)  # 모델을 GPU로 이동 (CUDA)

    # 오토인코더 학습
    autoencoder = train_autoencoder(
        model=autoencoder,
        X_train=X_train_scaled,
        device=device,
        num_epochs=50,
        batch_size=32,
        learning_rate=1e-4
    )

    # 4-2) OneClassSVM (scikit-learn 기반, CPU만 사용)
    oc_svm = OC_SVMModel(kernel='rbf', nu=0.1, gamma=0.1)
    train_oneclass_svm(oc_svm, X_train_scaled)

    # 4-3) Isolation Forest (scikit-learn 기반)
    i_forest = IForestModel(
        n_estimators=100, 
        max_samples='auto', 
        contamination='auto', 
        random_state=42
    )
    train_isolation_forest(i_forest, X_train_scaled)

    # 5) 모델 평가
    #   - Autoencoder (재구성 오차 계산 -> threshold로 이상/정상 구분)
    # 학습 데이터 재구성 오차 (threshold 계산용)
    ae_errors, ae_preds = evaluate_autoencoder(autoencoder, X_train_scaled, threshold=9999, device=device)
    mean_err = np.mean(ae_errors)
    std_err = np.std(ae_errors)
    threshold = mean_err + 3 * std_err  # 예: 학습 데이터 평균 + 3*sigma

    # 테스트 데이터에 대해 재평가
    ae_errors, ae_preds = evaluate_autoencoder(autoencoder, X_test_scaled, threshold=threshold, device=device)

    print(f"=== Autoencoder 평가 ===   threshold={threshold}")
    print(f"재구성 오차 예시(앞 5개): {ae_errors[:5]}")
    print(f"이상 판별(0=정상,1=이상) (앞 5개): {ae_preds[:5]}\n")

    #   - OneClassSVM 평가 (결정함수 -> 1/-1 -> 0/1 변환)
    svm_preds, svm_scores = evaluate_oneclass_svm(oc_svm, X_test_scaled)
    print("=== OneClassSVM 평가 ===")
    print(f"결과(0=정상,1=이상) (앞 5개): {svm_preds[:5]}")
    print(f"decision_function 점수(앞 5개): {svm_scores[:5]}")

    #   - Isolation Forest 평가 (+1/-1 -> 0/1 변환)
    preds, scores = evaluate_isolation_forest(i_forest, X_test_scaled)
    print("=== Isolation Forest 평가 ===")
    print(f"결과(0=정상,1=이상) (앞 5개): {preds[:5]}")
    print(f"decision_function 점수(앞 5개): {scores[:5]}")

if __name__ == "__main__":
    main()
