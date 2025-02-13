# models.py

import torch
import torch.nn as nn
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

class SimpleAutoencoder(nn.Module):
    """
    [PyTorch 기반 간단한 Autoencoder 모델]
    - 입력 차원(input_dim)을 받아 단순 Fully Connected 레이어로 구성
    - latent_dim은 중간 숨은 표현(embedding) 차원
    """
    def __init__(self, input_dim=10, latent_dim=4):
        super(SimpleAutoencoder, self).__init__()
        # 인코더(차원 축소)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
            nn.ReLU()
        )
        # 디코더(차원 복원)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        - encoder -> decoder -> output
        """
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class IForestModel:
    """
    [Isolation Forest 모델 래퍼 클래스]
    - scikit-learn의 IsolationForest를 감싸서 사용
    """
    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto', random_state=42):
        """
        Args:
            n_estimators (int): 트리에 사용할 추정기(트리) 개수
            max_samples (str or int): 각 트리를 학습할 때 사용할 샘플 수
            contamination (str or float): 이상치 비율. 'auto'면 자동 추정
            random_state (int): 난수 시드(재현성)
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state
        )

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        """
        예측 결과:
        +1 : 정상
        -1 : 이상
        """
        return self.model.predict(X)

    def decision_function(self, X):
        """
        점수가 작을수록 이상(Outlier)에 가깝다고 판단
        """
        return self.model.decision_function(X)


class OC_SVMModel:
    """
    [OneClassSVM 모델 래퍼 클래스]
    - scikit-learn의 OneClassSVM을 감싸서 사용
    """
    def __init__(self, kernel='rbf', nu=0.5, gamma='scale'):
        """
        Args:
            kernel (str): 커널 종류 (예: 'rbf', 'linear' 등)
            nu (float): 이상치로 분류할 수 있는 비율 (0~1)
            gamma (str or float): 'scale', 'auto' 또는 숫자
        """
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        """
        반환값:
        +1 : 정상
        -1 : 이상
        """
        return self.model.predict(X)

    def decision_function(self, X):
        """
        점수가 높을수록 normal, 낮을수록 anomaly
        """
        return self.model.decision_function(X)
