import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Optional, Tuple
from utils.logging_config import get_logger

logger = get_logger(__name__)


class LorentzianIndicators:
  """
  Реализация индикаторов из оригинального Machine Learning: Lorentzian Classification
  """

  @staticmethod
  def rescale(series: pd.Series, old_min: float, old_max: float,
              new_min: float, new_max: float) -> pd.Series:
    """Масштабирование значений из одного диапазона в другой"""
    return new_min + (new_max - new_min) * (series - old_min) / (old_max - old_min + 1e-10)

  @staticmethod
  def normalize(series: pd.Series, min_val: float = 0, max_val: float = 1) -> pd.Series:
    """Нормализация с отслеживанием исторических минимумов и максимумов"""
    historic_min = series.expanding().min()
    historic_max = series.expanding().max()
    return min_val + (max_val - min_val) * (series - historic_min) / (historic_max - historic_min + 1e-10)

  @staticmethod
  def n_rsi(close: pd.Series, n1: int = 14, n2: int = 1) -> pd.Series:
    """Нормализованный RSI как в оригинале"""
    rsi = ta.rsi(close, length=n1)
    if n2 > 1:
      rsi = ta.ema(rsi, length=n2)
    return LorentzianIndicators.rescale(rsi, 0, 100, 0, 1)

  @staticmethod
  def n_wt(hlc3: pd.Series, n1: int = 10, n2: int = 11) -> pd.Series:
    """
    Нормализованный WaveTrend Classic
    WaveTrend использует HLC3 (High+Low+Close)/3
    """
    # Первая EMA
    ema1 = ta.ema(hlc3, length=n1)

    # Вторая EMA от абсолютной разницы
    ema2 = ta.ema(abs(hlc3 - ema1), length=n1)

    # CI (Commodity Index)
    ci = (hlc3 - ema1) / (0.015 * ema2 + 1e-10)

    # TCI (Trend Confirmation Index) - основная линия WaveTrend
    wt1 = ta.ema(ci, length=n2)

    # Сигнальная линия
    wt2 = ta.sma(wt1, length=4)

    # Возвращаем разницу (как в оригинале)
    wt_diff = wt1 - wt2

    return LorentzianIndicators.normalize(wt_diff, 0, 1)

  @staticmethod
  def n_cci(high: pd.Series, low: pd.Series, close: pd.Series,
            n1: int = 20, n2: int = 1) -> pd.Series:
    """Нормализованный CCI"""
    cci = ta.cci(high, low, close, length=n1)
    if n2 > 1:
      cci = ta.ema(cci, length=n2)
    return LorentzianIndicators.normalize(cci, 0, 1)

  @staticmethod
  def n_adx(high: pd.Series, low: pd.Series, close: pd.Series, n1: int = 20) -> pd.Series:
    """Нормализованный ADX как в оригинале"""
    # Расчет True Range
    tr = pd.DataFrame({
      'hl': high - low,
      'hc': abs(high - close.shift(1)),
      'lc': abs(low - close.shift(1))
    }).max(axis=1)

    # Directional Movement
    dm_plus = pd.Series(0.0, index=high.index)
    dm_minus = pd.Series(0.0, index=high.index)

    high_diff = high.diff()
    low_diff = -low.diff()

    dm_plus[high_diff > low_diff] = high_diff[high_diff > low_diff].clip(lower=0)
    dm_minus[low_diff > high_diff] = low_diff[low_diff > high_diff].clip(lower=0)

    # Сглаженные значения
    tr_smooth = tr.rolling(window=n1).sum()
    dm_plus_smooth = dm_plus.rolling(window=n1).sum()
    dm_minus_smooth = dm_minus.rolling(window=n1).sum()

    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
    di_minus = 100 * dm_minus_smooth / (tr_smooth + 1e-10)

    # DX и ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = ta.rma(dx, length=n1)

    return LorentzianIndicators.rescale(adx, 0, 100, 0, 1)

  @staticmethod
  def calculate_features(data: pd.DataFrame,
                         f1_config: dict, f2_config: dict, f3_config: dict,
                         f4_config: dict, f5_config: dict) -> pd.DataFrame:
    """
    Рассчитывает все 5 признаков как в оригинальном индикаторе

    Конфигурация признака: {'type': 'RSI', 'paramA': 14, 'paramB': 1}
    """
    features = pd.DataFrame(index=data.index)
    hlc3 = (data['high'] + data['low'] + data['close']) / 3

    # Функция для расчета одного признака
    def calc_feature(config: dict) -> pd.Series:
      feature_type = config['type']
      param_a = config['paramA']
      param_b = config.get('paramB', 1)

      if feature_type == 'RSI':
        return LorentzianIndicators.n_rsi(data['close'], param_a, param_b)
      elif feature_type == 'WT':
        return LorentzianIndicators.n_wt(hlc3, param_a, param_b)
      elif feature_type == 'CCI':
        return LorentzianIndicators.n_cci(data['high'], data['low'], data['close'], param_a, param_b)
      elif feature_type == 'ADX':
        return LorentzianIndicators.n_adx(data['high'], data['low'], data['close'], param_a)
      else:
        raise ValueError(f"Неизвестный тип признака: {feature_type}")

    # Рассчитываем все 5 признаков
    features['f1'] = calc_feature(f1_config)
    features['f2'] = calc_feature(f2_config)
    features['f3'] = calc_feature(f3_config)
    features['f4'] = calc_feature(f4_config)
    features['f5'] = calc_feature(f5_config)

    # Удаляем NaN значения
    features = features.dropna()

    return features


class LorentzianFilters:
  """Фильтры из оригинального индикатора"""

  @staticmethod
  def volatility_filter(data: pd.DataFrame, min_length: int = 1,
                        max_length: int = 10) -> pd.Series:
    """Фильтр волатильности"""
    recent_atr = ta.atr(data['high'], data['low'], data['close'], length=min_length)
    historical_atr = ta.atr(data['high'], data['low'], data['close'], length=max_length)
    return recent_atr > historical_atr

  @staticmethod
  def regime_filter(data: pd.DataFrame, threshold: float = -0.1) -> pd.Series:
    """Фильтр рыночного режима (KLMF)"""
    src = (data['open'] + data['high'] + data['low'] + data['close']) / 4

    # Расчет KLMF (Kalman-Like Moving Filter)
    value1 = pd.Series(0.0, index=data.index)
    value2 = pd.Series(0.0, index=data.index)

    for i in range(1, len(data)):
      value1.iloc[i] = 0.2 * (src.iloc[i] - src.iloc[i - 1]) + 0.8 * value1.iloc[i - 1]
      value2.iloc[i] = 0.1 * (data['high'].iloc[i] - data['low'].iloc[i]) + 0.8 * value2.iloc[i - 1]

    omega = abs(value1 / (value2 + 1e-10))
    alpha = (-omega ** 2 + np.sqrt(omega ** 4 + 16 * omega ** 2)) / 8

    klmf = pd.Series(0.0, index=data.index)
    for i in range(1, len(data)):
      klmf.iloc[i] = alpha.iloc[i] * src.iloc[i] + (1 - alpha.iloc[i]) * klmf.iloc[i - 1]

    # Наклон кривой
    abs_curve_slope = abs(klmf.diff())
    exp_avg_slope = ta.ema(abs_curve_slope, length=200)
    normalized_slope = (abs_curve_slope - exp_avg_slope) / (exp_avg_slope + 1e-10)

    return normalized_slope >= threshold

  @staticmethod
  def adx_filter(data: pd.DataFrame, threshold: int = 20) -> pd.Series:
    """ADX фильтр"""
    adx = ta.adx(data['high'], data['low'], data['close'], length=14)
    if adx is not None and not adx.empty:
      return adx.iloc[:, 0] > threshold
    return pd.Series(True, index=data.index)