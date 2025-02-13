# evaluate.py

import numpy as np
import torch
import torch.nn as nn

def evaluate_autoencoder(model, X_test, threshold=0.01, device='cpu'):
    """
    [Autoencoder 평가 함수]
    - 주어진 X_test(특징 벡터)에 대해 오토인코더 재구성 오차를 계산
    - threshold보다 크면 '이상(1)', 작으면 '정상(0)'으로 분류

    Args:
        model (nn.Module): 학습된 오토인코더 모델
        X_test (np.ndarray): 테스트 데이터(스케일링된 특징)
        threshold (float): 재구성 오차를 구분하는 임계값
        device (str): 'cuda' 또는 'cpu'

    Returns:
        errors (np.ndarray): 각 샘플별 재구성 오차
        predictions (np.ndarray): 0(정상), 1(이상)
    """
    model.eval()
    # 입력 데이터를 텐서로 변환하고 GPU/CPU로 이동
    tensor_x = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        # 오토인코더에 입력 -> 출력(재구성)
        reconstructed = model(tensor_x)
        # 샘플별 MSE 계산 (reduction='none' -> (batch, input_dim) 형태)
        mse_loss_fn = nn.MSELoss(reduction='none')
        loss_per_sample = mse_loss_fn(reconstructed, tensor_x).mean(dim=1).cpu().numpy()

    errors = loss_per_sample
    # threshold 기준으로 이상/정상 구분
    predictions = (errors > threshold).astype(int)

    return errors, predictions


def evaluate_oneclass_svm(oc_svm, X_test):
    """
    [OneClassSVM 평가 함수]
    - oc_svm.predict(X_test) -> 1(정상), -1(이상)
    - 이를 0(정상), 1(이상)으로 변환
    - decision_function은 점수가 높을수록 정상, 낮을수록 이상

    Args:
        oc_svm (OC_SVMModel): OneClassSVM 래퍼 객체
        X_test (np.ndarray): 테스트 데이터(스케일링된 특징)

    Returns:
        predictions (np.ndarray): 0(정상), 1(이상)
        decision_scores (np.ndarray): 클수록 정상, 작을수록 이상
    """
    # OneClassSVM의 예측 결과 (+1 / -1)
    raw_preds = oc_svm.predict(X_test)
    predictions = np.where(raw_preds == 1, 0, 1)  # +1 -> 0(정상), -1 -> 1(이상)
    decision_scores = oc_svm.decision_function(X_test)
    return predictions, decision_scores


def evaluate_isolation_forest(i_forest, X_test):
    """
    [Isolation Forest 평가 함수]
    - i_forest.predict(X_test) -> +1(정상), -1(이상)
    - 이를 0(정상), 1(이상)으로 변환
    - decision_function은 점수가 작을수록 이상치에 가깝다는 의미

    Args:
        i_forest (IForestModel): Isolation Forest 래퍼 객체
        X_test (np.ndarray): 테스트 데이터(스케일링된 특징)

    Returns:
        predictions (np.ndarray): 0(정상), 1(이상)
        decision_scores (np.ndarray): decision_function 결과값
    """
    raw_preds = i_forest.predict(X_test)  # +1 or -1
    predictions = np.where(raw_preds == 1, 0, 1)  # 0(정상), 1(이상)
    decision_scores = i_forest.decision_function(X_test)
    return predictions, decision_scores
