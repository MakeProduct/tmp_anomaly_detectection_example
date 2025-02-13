# feature_extraction.py

import numpy as np
import pandas as pd
from scipy.signal import welch
from typing import Dict, List
import pywt  # 웨이블릿 변환을 위해 pywt 라이브러리 사용

def extract_time_features(window: np.ndarray) -> Dict[str, float]:
    """
    [시간영역 특징 추출 함수]
    - window(윈도우 데이터)에서 RMS, Peak-to-Peak, 평균, 표준편차, 첨도(kurtosis), 왜도(skewness) 등을 계산해 반환
    
    Args:
        window (np.ndarray): (samples,) 또는 (samples, channels) 형태의 신호
    
    Returns:
        Dict[str, float]: 시간영역 특징들을 담은 딕셔너리
    """
    if not isinstance(window, np.ndarray):
        raise TypeError("window는 np.ndarray여야 합니다.")
    if window.size == 0:
        raise ValueError("window가 비어 있습니다.")

    # 2차원인 경우 첫 번째 채널만 사용 (단일 채널로 가정)
    sig = window[:, 0] if window.ndim == 2 else window

    # 시간영역 특징 계산
    features = {}
    features['time_rms'] = np.sqrt(np.mean(sig**2))                   # RMS
    features['time_peak2peak'] = np.max(sig) - np.min(sig)            # 피크 투 피크
    features['time_mean'] = np.mean(sig)                              # 평균
    features['time_std'] = np.std(sig)                                # 표준편차

    # 첨도(kurtosis)와 왜도(skewness)
    mean_val = np.mean(sig)
    std_val = np.std(sig) if np.std(sig) != 0 else 1e-9
    features['time_kurtosis'] = np.mean(((sig - mean_val)/std_val)**4)
    features['time_skewness'] = np.mean(((sig - mean_val)/std_val)**3)

    return features


def extract_freq_features(window: np.ndarray, fs: int = 100) -> Dict[str, float]:
    """
    [주파수영역 특징 추출 함수]
    - FFT를 수행하여 주파수 스펙트럼(파워 스펙트럼)을 구하고,
      대역별 파워 및 스펙트럼 첨두값(peak) 등을 계산
    
    Args:
        window (np.ndarray): (samples,) 또는 (samples, channels) 형태의 신호
        fs (int): 샘플링 주파수(Hz)
    
    Returns:
        Dict[str, float]: 주파수영역 특징들을 담은 딕셔너리
    """
    if not isinstance(window, np.ndarray):
        raise TypeError("window는 np.ndarray여야 합니다.")
    if window.size == 0:
        raise ValueError("window가 비어 있습니다.")

    # 2차원인 경우 첫 번째 채널 사용
    sig = window[:, 0] if window.ndim == 2 else window
    N = len(sig)

    # FFT 계산
    fft_result = np.fft.fft(sig)
    freqs_full = np.fft.fftfreq(N, d=1/fs)
    half_idx = N // 2  # 양의 주파수 범위만 사용

    freqs = freqs_full[:half_idx]
    spec = np.abs(fft_result[:half_idx])
    power_spectrum = spec**2  # 진폭^2 = 파워

    # 대역별 파워 계산 (예: 0~10Hz, 10~30Hz, 30Hz 이상)
    band1 = power_spectrum[(freqs >= 0) & (freqs < 10)].sum()
    band2 = power_spectrum[(freqs >= 10) & (freqs < 30)].sum()
    band3 = power_spectrum[(freqs >= 30)].sum()
    total_power = power_spectrum.sum() + 1e-9

    # 대역별 파워 비율
    freq_band1_ratio = band1 / total_power
    freq_band2_ratio = band2 / total_power
    freq_band3_ratio = band3 / total_power

    # 스펙트럼에서 최대 파워(peak)
    freq_peak_power = power_spectrum.max()
    peak_idx = np.argmax(spec)
    peak_freq = freqs[peak_idx]
    peak_amp = spec[peak_idx]

    # 스펙트럼 무게중심, 대역폭
    if spec.sum() == 0:
        spectral_centroid = 0.0
        spectral_bandwidth = 0.0
    else:
        spectral_centroid = np.sum(freqs * spec) / np.sum(spec)
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * spec) / np.sum(spec))

    freq_dict = {
        'freq_band1_ratio': freq_band1_ratio,
        'freq_band2_ratio': freq_band2_ratio,
        'freq_band3_ratio': freq_band3_ratio,
        'freq_peak_power': freq_peak_power,
        'peak_freq': peak_freq,
        'peak_amp': peak_amp,
        'total_power': total_power,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth
    }

    return freq_dict


def wavelet_transform(window: np.ndarray, wavelet: str = 'db4', level: int = 3) -> Dict[str, float]:
    """
    [웨이블릿 변환 기반 특징 추출 함수]
    - 웨이블릿 분해(wavedec)를 통해 다중 해상도 분석 후
      각 레벨별 에너지, 통계량, 엔트로피 등을 계산
    
    Args:
        window (np.ndarray): (samples,) 또는 (samples, channels) 형태의 신호
        wavelet (str): 사용할 웨이블릿 종류 (예: 'db4')
        level (int): 웨이블릿 분해 레벨
    
    Returns:
        Dict[str, float]: 웨이블릿 변환 특징들을 담은 딕셔너리
    """
    if not isinstance(window, np.ndarray):
        raise TypeError("window는 np.ndarray여야 합니다.")
    if window.size == 0:
        raise ValueError("window가 비어 있습니다.")

    # 첫 번째 채널만 사용
    sig = window[:, 0] if window.ndim == 2 else window

    # 웨이블릿 분해 수행
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    # coeffs[0]: 최종 근사 계수(저주파 성분)
    # coeffs[1:]: 디테일 계수(고주파 성분)

    features = {}
    total_energy = 0.0

    # 각 레벨별 에너지 합산
    for i, c in enumerate(coeffs):
        energy = np.sum(c**2)
        total_energy += energy
        features[f'wavelet_energy_level{i}'] = energy

    # 각 레벨별 통계량(평균, 표준편차, 왜도, 첨도)
    for i, c in enumerate(coeffs):
        level_mean = np.mean(c)
        level_std = np.std(c)
        if level_std < 1e-12:
            level_skew, level_kurt = 0.0, 0.0
        else:
            z = (c - level_mean) / level_std
            level_skew = np.mean(z**3)
            level_kurt = np.mean(z**4)

        features[f'wavelet_mean_level{i}'] = level_mean
        features[f'wavelet_std_level{i}'] = level_std
        features[f'wavelet_skew_level{i}'] = level_skew
        features[f'wavelet_kurt_level{i}'] = level_kurt

    # 웨이블릿 엔트로피
    if total_energy < 1e-12:
        features['wavelet_entropy'] = 0.0
    else:
        ent = 0.0
        for i, c in enumerate(coeffs):
            E_i = np.sum(c**2)
            p_i = E_i / total_energy
            if p_i > 1e-12:
                ent -= p_i * np.log2(p_i)
        features['wavelet_entropy'] = ent

    return features


def aggregate_features(windows: List[np.ndarray], fs: int = 100) -> pd.DataFrame:
    """
    [전체 특징 추출 함수]
    - 입력받은 여러 윈도우 각각에 대해
      시간영역, 주파수영역, 웨이블릿 변환 특징을 추출하고 하나의 DataFrame으로 합쳐 반환
    
    Args:
        windows (List[np.ndarray]): 윈도우 리스트
        fs (int): 샘플링 주파수
    
    Returns:
        pd.DataFrame: 각 윈도우의 특징을 행으로 갖는 DataFrame
    """
    if not isinstance(windows, list):
        raise TypeError("windows는 리스트 형태여야 합니다.")
    if len(windows) == 0:
        raise ValueError("windows 리스트가 비어 있습니다.")

    all_features = []
    for w in windows:
        # 시간영역 특징
        f_time = extract_time_features(w)
        # 주파수영역 특징
        f_freq = extract_freq_features(w, fs=fs)
        # 웨이블릿 변환 특징
        f_wave = wavelet_transform(w)

        # 세 종류 특징을 하나의 딕셔너리로 합침
        combined = {**f_time, **f_freq, **f_wave}
        all_features.append(combined)

    return pd.DataFrame(all_features)


def aggregate_features_selected(windows: List[np.ndarray], fs: int = 100) -> pd.DataFrame:
    """
    [일부(5개) 특징만 추출 함수]
    - 예시로 시간영역 2개, 주파수영역 2개, 웨이블릿 1개만 추출
    
    Args:
        windows (List[np.ndarray]): 윈도우 리스트
        fs (int): 샘플링 주파수
    
    Returns:
        pd.DataFrame: 선택된 특징들만 모아서 반환
    """
    if not isinstance(windows, list):
        raise TypeError("windows는 리스트 형태여야 합니다.")
    if len(windows) == 0:
        raise ValueError("windows 리스트가 비어 있습니다.")

    selected_features = []
    for w in windows:
        # 먼저 전체 특징 추출
        full_features = {}
        full_features.update(extract_time_features(w))
        full_features.update(extract_freq_features(w, fs=fs))
        full_features.update(wavelet_transform(w))

        # 여기서는 5개만 선택 (예시)
        sub_dict = {
            'time_rms': full_features['time_rms'],
            'time_skewness': full_features['time_skewness'],
            'freq_peak_power': full_features['freq_peak_power'],
            'freq_band1_ratio': full_features['freq_band1_ratio'],
            'wavelet_entropy': full_features['wavelet_entropy']
        }
        selected_features.append(sub_dict)

    return pd.DataFrame(selected_features)
