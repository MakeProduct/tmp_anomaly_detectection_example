# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def train_autoencoder(model, X_train, device, num_epochs=20, batch_size=32, learning_rate=1e-3):
    """
    [Autoencoder 모델 학습 함수]
    - Autoencoder는 입력과 출력을 동일하게 두고 MSELoss로 학습
    - device: GPU/CPU 설정

    Args:
        model (nn.Module): Autoencoder 모델
        X_train (np.ndarray): 학습 데이터(스케일링된 특징)
        device (torch.device): 'cuda' 또는 'cpu'
        num_epochs (int): 학습 에폭 수
        batch_size (int): 배치 크기
        learning_rate (float): 학습률

    Returns:
        model (nn.Module): 학습이 완료된 Autoencoder
    """
    # 모델을 훈련 모드로 전환
    model.train()
    
    # (numpy -> torch Tensor) 변환
    tensor_x = torch.FloatTensor(X_train)

    # Autoencoder는 입력=출력이므로 label이 따로 필요 없음
    dataset = TensorDataset(tensor_x, tensor_x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 손실함수(MSE)와 옵티마이저(Adam)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_in, _ in dataloader:
            # 배치를 GPU 또는 CPU로 복사
            batch_in = batch_in.to(device)

            # Forward
            reconstructed = model(batch_in)
            loss = criterion(reconstructed, batch_in)

            # Backward & Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch+1) % 5 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")

    return model


def train_oneclass_svm(oc_svm, X_train):
    """
    [OneClassSVM 모델 학습 함수]
    - scikit-learn의 OneClassSVM을 사용해 X_train에 대한 경계를 학습

    Args:
        oc_svm (OC_SVMModel): OneClassSVM 래퍼 객체
        X_train (np.ndarray): 학습 데이터(스케일링된 특징)
    """
    oc_svm.fit(X_train)
    print("OneClassSVM 모델 학습 완료!")


def train_isolation_forest(i_forest, X_train):
    """
    [Isolation Forest 모델 학습 함수]
    - scikit-learn의 IsolationForest를 사용해 X_train에서 이상치를 식별할 수 있는 트리 집합을 학습

    Args:
        i_forest (IForestModel): Isolation Forest 래퍼 객체
        X_train (np.ndarray): 학습 데이터(스케일링된 특징)
    """
    i_forest.fit(X_train)
    print("IsolationForest 모델 학습 완료!")
