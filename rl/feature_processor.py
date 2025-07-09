# rl/feature_processor.py
import asyncio

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import pandas_ta as ta

from utils.logging_config import get_logger
from core.enums import Timeframe

logger = get_logger(__name__)


class RLFeatureProcessor:
  """
  Процессор признаков для RL, интегрированный с feature_engineer проекта
  """

  def __init__(
      self,
      feature_engineer=None,
      config: Dict[str, Any] = None
  ):
    self.feature_engineer = feature_engineer
    self.config = config or {}

    # Скейлеры для нормализации
    self.price_scaler = RobustScaler()
    self.volume_scaler = RobustScaler()
    self.indicator_scaler = StandardScaler()

    # PCA для снижения размерности
    self.use_pca = self.config.get('use_pca', False)
    self.pca = PCA(n_components=self.config.get('pca_components', 50))

    # Кэш признаков
    self.feature_cache = {}

    # Статистика для адаптивной нормализации
    self.rolling_stats = {
      'price_mean': None,
      'price_std': None,
      'volume_mean': None,
      'volume_std': None
    }

    self.is_fitted = False

  def create_rl_features(
      self,
      data: pd.DataFrame,
      symbol: str,
      include_multi_timeframe: bool = True
  ) -> np.ndarray:
    """
    Создает признаки для RL из рыночных данных
    """
    try:
      features_list = []

      # 1. Базовые ценовые признаки
      price_features = self._create_price_features(data)
      features_list.append(price_features)

      # 2. Технические индикаторы
      technical_features = self._create_technical_features(data)
      features_list.append(technical_features)

      # 3. Паттерны и структура рынка
      pattern_features = self._create_pattern_features(data)
      features_list.append(pattern_features)

      # 4. Микроструктурные признаки
      microstructure_features = self._create_microstructure_features(data)
      features_list.append(microstructure_features)

      # 5. Признаки из feature_engineer если доступен
      if self.feature_engineer:
        ml_features = self._get_ml_features(data, symbol)
        if ml_features is not None:
          features_list.append(ml_features)

      # Объединяем все признаки
      all_features = pd.concat(features_list, axis=1)

      # Заполняем пропуски
      all_features = all_features.fillna(method='ffill').fillna(0)

      # Нормализация
      normalized_features = self._normalize_features(all_features)

      # PCA если включено
      if self.use_pca and self.is_fitted:
        normalized_features = self.pca.transform(normalized_features)

      return normalized_features

    except Exception as e:
      logger.error(f"Ошибка создания RL признаков: {e}")
      # Возвращаем базовые признаки в случае ошибки
      return self._create_fallback_features(data)

  def _create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Создает ценовые признаки"""
    features = pd.DataFrame(index=data.index)

    # Логарифмические доходности
    features['log_return'] = np.log(data['close'] / data['close'].shift(1))
    features['log_return_5'] = np.log(data['close'] / data['close'].shift(5))
    features['log_return_20'] = np.log(data['close'] / data['close'].shift(20))

    # Относительное положение цены
    features['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-10)

    # OHLC соотношения
    features['high_low_ratio'] = data['high'] / (data['low'] + 1e-10)
    features['close_open_ratio'] = data['close'] / (data['open'] + 1e-10)

    # Скользящие средние отклонения
    for period in [10, 20, 50]:
      ma = data['close'].rolling(period).mean()
      features[f'price_ma{period}_ratio'] = data['close'] / (ma + 1e-10)

    # Волатильность
    features['volatility_20'] = data['close'].pct_change().rolling(20).std()
    features['volatility_ratio'] = features['volatility_20'] / features['volatility_20'].rolling(50).mean()

    return features

  def _create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Создает технические индикаторы"""
    features = pd.DataFrame(index=data.index)

    # RSI с разными периодами
    for period in [7, 14, 21]:
      rsi = ta.rsi(data['close'], length=period)
      features[f'rsi_{period}'] = rsi / 100  # Нормализация в [0, 1]

    # MACD
    macd = ta.macd(data['close'])
    if macd is not None and not macd.empty:
      features['macd_signal'] = macd.iloc[:, 0] - macd.iloc[:, 1]  # MACD - Signal
      features['macd_histogram'] = macd.iloc[:, 2]

    # Bollinger Bands
    bb = ta.bbands(data['close'], length=20, std=2)
    if bb is not None and not bb.empty:
      features['bb_position'] = (data['close'] - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0] + 1e-10)
      features['bb_width'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / data['close']

    # ATR
    atr = ta.atr(data['high'], data['low'], data['close'], length=14)
    features['atr_ratio'] = atr / data['close']

    # ADX
    adx = ta.adx(data['high'], data['low'], data['close'], length=14)
    if adx is not None and not adx.empty:
      features['adx'] = adx.iloc[:, 0] / 100  # ADX в [0, 1]
      features['adx_di_diff'] = (adx.iloc[:, 1] - adx.iloc[:, 2]) / 100  # DI+ - DI-

    # Stochastic
    stoch = ta.stoch(data['high'], data['low'], data['close'])
    if stoch is not None and not stoch.empty:
      features['stoch_k'] = stoch.iloc[:, 0] / 100
      features['stoch_d'] = stoch.iloc[:, 1] / 100

    return features

  def _create_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Создает признаки паттернов"""
    features = pd.DataFrame(index=data.index)

    # Свечные паттерны
    features['doji'] = (abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-10) < 0.1).astype(int)
    features['hammer'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-10) > 0.6).astype(int)
    features['shooting_star'] = ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-10) > 0.6).astype(
      int)

    # Трендовые признаки
    features['higher_highs'] = (data['high'] > data['high'].shift(1)).astype(int).rolling(5).sum() / 5
    features['lower_lows'] = (data['low'] < data['low'].shift(1)).astype(int).rolling(5).sum() / 5

    # Уровни поддержки/сопротивления
    features['near_resistance'] = self._detect_near_level(data, 'resistance')
    features['near_support'] = self._detect_near_level(data, 'support')

    return features

  def _create_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Создает микроструктурные признаки"""
    features = pd.DataFrame(index=data.index)

    # Объемные признаки
    features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    features['volume_trend'] = ta.sma(data['volume'], 5) / ta.sma(data['volume'], 20)

    # Price-Volume correlation
    features['pv_corr'] = data['close'].pct_change().rolling(20).corr(data['volume'].pct_change())

    # Амплитуда
    features['amplitude'] = (data['high'] - data['low']) / data['close']
    features['amplitude_ratio'] = features['amplitude'] / features['amplitude'].rolling(20).mean()

    # Тиковые данные (если доступны)
    if 'tick_count' in data.columns:
      features['tick_intensity'] = data['tick_count'] / data['tick_count'].rolling(20).mean()

    return features

  def _detect_near_level(self, data: pd.DataFrame, level_type: str) -> pd.Series:
    """Определяет близость к уровням поддержки/сопротивления"""
    window = 20
    threshold = 0.02  # 2% от цены

    if level_type == 'resistance':
      levels = data['high'].rolling(window).max()
    else:
      levels = data['low'].rolling(window).min()

    distance = abs(data['close'] - levels) / data['close']
    return (distance < threshold).astype(float)

  def _get_ml_features(self, data: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    """Получает признаки из ML feature_engineer"""
    try:
      if not self.feature_engineer:
        return None

      # Используем существующий feature_engineer
      features, _ = self.feature_engineer.create_features_and_labels(
        data,
        for_prediction=True
      )

      # Выбираем только числовые колонки
      numeric_features = features.select_dtypes(include=[np.number])

      # Ограничиваем количество признаков
      if numeric_features.shape[1] > 50:
        # Выбираем наиболее важные
        feature_importance = self.feature_engineer.get_feature_importance()
        if feature_importance:
          top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
          )[:50]
          selected_cols = [f[0] for f in top_features if f[0] in numeric_features.columns]
          numeric_features = numeric_features[selected_cols]

      return numeric_features

    except Exception as e:
      logger.error(f"Ошибка получения ML признаков: {e}")
      return None

  def _normalize_features(self, features: pd.DataFrame) -> np.ndarray:
    """Нормализует признаки"""
    try:
      # Разделяем признаки по типам для разной нормализации
      price_cols = [col for col in features.columns if 'price' in col or 'close' in col or 'open' in col]
      volume_cols = [col for col in features.columns if 'volume' in col]
      other_cols = [col for col in features.columns if col not in price_cols + volume_cols]

      normalized_data = []

      # Нормализуем ценовые признаки
      if price_cols:
        price_data = features[price_cols].values
        if self.is_fitted:
          normalized_price = self.price_scaler.transform(price_data)
        else:
          normalized_price = self.price_scaler.fit_transform(price_data)
        normalized_data.append(normalized_price)

      # Нормализуем объемные признаки
      if volume_cols:
        volume_data = features[volume_cols].values
        if self.is_fitted:
          normalized_volume = self.volume_scaler.transform(volume_data)
        else:
          normalized_volume = self.volume_scaler.fit_transform(volume_data)
        normalized_data.append(normalized_volume)

      # Нормализуем остальные признаки
      if other_cols:
        other_data = features[other_cols].values
        if self.is_fitted:
          normalized_other = self.indicator_scaler.transform(other_data)
        else:
          normalized_other = self.indicator_scaler.fit_transform(other_data)
        normalized_data.append(normalized_other)

      # Объединяем все нормализованные данные
      if normalized_data:
        result = np.hstack(normalized_data)
      else:
        result = features.values

      # Обрезаем экстремальные значения
      result = np.clip(result, -5, 5)

      return result

    except Exception as e:
      logger.error(f"Ошибка нормализации признаков: {e}")
      return features.values

  def _create_fallback_features(self, data: pd.DataFrame) -> np.ndarray:
    """Создает минимальный набор признаков в случае ошибки"""
    try:
      features = pd.DataFrame(index=data.index)

      # Минимальные признаки
      features['return'] = data['close'].pct_change()
      features['volume_norm'] = data['volume'] / data['volume'].rolling(20).mean()
      features['rsi'] = ta.rsi(data['close'], length=14) / 100

      # Простая MA
      features['ma_ratio'] = data['close'] / data['close'].rolling(20).mean()

      return features.fillna(0).values

    except:
      # Крайний случай - возвращаем только цены
      return data[['close', 'volume']].fillna(0).values

  def fit(self, historical_data: Dict[str, pd.DataFrame]):
    """Обучает процессор на исторических данных"""
    logger.info("Обучение RL Feature Processor...")

    try:
      all_features = []

      # Собираем признаки со всех символов
      for symbol, data in historical_data.items():
        features = self.create_rl_features(data, symbol, include_multi_timeframe=False)
        all_features.append(features)

      # Объединяем все данные
      combined_features = np.vstack(all_features)

      # Обучаем PCA если включено
      if self.use_pca:
        self.pca.fit(combined_features)
        logger.info(f"PCA обучен, объясненная дисперсия: {self.pca.explained_variance_ratio_.sum():.2%}")

      self.is_fitted = True
      logger.info("Feature Processor успешно обучен")

    except Exception as e:
      logger.error(f"Ошибка обучения Feature Processor: {e}")
      self.is_fitted = False

  async def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Новый публичный метод-оркестратор для полного обогащения данных.
    """
    if df.empty:
      return pd.DataFrame()

    # Шаг 1: Добавляем технические индикаторы синхронно
    logger.info("Добавление технических индикаторов...")
    processed_df = df.groupby('tic', group_keys=False).apply(self._add_technical_indicators)

    # Шаг 2: Асинхронно добавляем ML-признаки (режим рынка, аномалии)
    logger.info("Добавление ML-признаков (режим рынка, аномалии)...")

    tasks = []
    for name, group in processed_df.groupby('tic'):
      tasks.append(self._add_advanced_features_to_group(group))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    final_df_list = [res for res in results if isinstance(res, pd.DataFrame)]

    if not final_df_list:
      logger.error("Не удалось обработать ни одну группу признаков.")
      return pd.DataFrame()

    final_df = pd.concat(final_df_list).sort_values(['date', 'tic']).reset_index(drop=True)
    final_df.bfill(inplace=True).ffill(inplace=True)

    return final_df

  async def _add_advanced_features_to_group(self, df_group: pd.DataFrame) -> pd.DataFrame:
    """
    Асинхронный вспомогательный метод для добавления ML-признаков к группе.
    """
    df = df_group.copy()
    symbol = df['tic'].iloc[0]
    try:
      # --- Режим рынка ---
      regime = await self.market_regime_detector.detect_regime(symbol, df)
      if regime:
        df['market_regime'] = regime.primary_regime.value
      else:
        df['market_regime'] = 0

      # --- Аномалии ---
      # Используем detect_anomalies, как и исправляли ранее
      anomaly_reports = await self.anomaly_detector.detect_anomalies(df, symbol)
      if anomaly_reports:
        df['anomaly_score'] = max(report.severity for report in anomaly_reports)
      else:
        df['anomaly_score'] = 0.0

    except Exception as e:
      logger.error(f"Ошибка добавления ML признаков для {symbol}: {e}")
      df['market_regime'] = 0
      df['anomaly_score'] = 0.0

    return df

  async def process_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
      """
      Новый метод-оркестратор, который асинхронно обрабатывает все признаки
      для полного DataFrame, содержащего несколько тикеров.
      """
      if df.empty:
        return pd.DataFrame()

      logger.info("Асинхронное добавление ML-признаков для всех групп...")

      # Создаем задачи для каждой группы (каждого тикера)
      tasks = []
      for tic, group_df in df.groupby('tic'):
        tasks.append(self._add_features_to_group(group_df, tic))

      # ПРАВИЛЬНЫЙ ВЫЗОВ: Запускаем задачи параллельно с помощью asyncio.gather
      results = await asyncio.gather(*tasks, return_exceptions=True)

      # Собираем обработанные группы обратно в один DataFrame
      processed_groups = [res for res in results if isinstance(res, pd.DataFrame)]

      if not processed_groups:
        raise ValueError("Не удалось обработать признаки ни для одной группы.")

      final_df = pd.concat(processed_groups, ignore_index=True)
      final_df.sort_values(['date', 'tic'], inplace=True)
      final_df.bfill(inplace=True).ffill(inplace=True)  # Заполняем возможные пропуски

      return final_df

  async def _add_features_to_group(self, df_group: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Вспомогательный метод для асинхронного добавления признаков к одной группе.
    """
    df = df_group.copy()
    try:
      # --- Режим рынка ---
      regime = await self.market_regime_detector.detect_regime(symbol, df)
      df['market_regime'] = regime.primary_regime.value if regime else 0

      # --- Аномалии ---
      # Используем detect_anomalies, а не несуществующий calculate_anomaly_score
      anomaly_reports = await self.anomaly_detector.detect_anomalies(df, symbol)
      df['anomaly_score'] = max(r.severity for r in anomaly_reports) if anomaly_reports else 0.0

    except Exception as e:
      logger.error(f"Ошибка добавления ML признаков для {symbol}: {e}")
      df['market_regime'] = 0
      df['anomaly_score'] = 0.0

    return df