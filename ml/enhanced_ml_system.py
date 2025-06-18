# ml/enhanced_ml_system.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import entropy
import joblib
import json

from core.enums import SignalType
from utils.logging_config import get_logger
from ml.anomaly_detector import MarketAnomalyDetector, AnomalyType

logger = get_logger(__name__)


@dataclass
class MLPrediction:
  """Расширенное предсказание с дополнительной информацией"""
  signal_type: SignalType
  probability: float
  confidence: float
  model_agreement: float  # Согласованность моделей в ансамбле
  feature_importance: Dict[str, float]
  risk_assessment: Dict[str, Any]
  metadata: Dict[str, Any]


class AdvancedFeatureEngineer:
  """Продвинутый генератор признаков"""

  def __init__(self):
    self.feature_names = []
    self.feature_stats = {}

  def create_advanced_features(self, data: pd.DataFrame,
                               external_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """
    Создает продвинутые признаки включая межрыночные корреляции
    """
    features = pd.DataFrame(index=data.index)

    # 1. Микроструктурные признаки
    microstructure_features = self._create_microstructure_features(data)
    features = pd.concat([features, microstructure_features], axis=1)

    # 2. Признаки рыночных режимов
    regime_features = self._create_regime_features(data)
    features = pd.concat([features, regime_features], axis=1)

    # 3. Признаки на основе теории информации
    information_features = self._create_information_features(data)
    features = pd.concat([features, information_features], axis=1)

    # 4. Нелинейные взаимодействия
    interaction_features = self._create_interaction_features(data)
    features = pd.concat([features, interaction_features], axis=1)

    # 5. Межрыночные признаки (если есть внешние данные)
    if external_data:
      cross_market_features = self._create_cross_market_features(data, external_data)
      features = pd.concat([features, cross_market_features], axis=1)

    # 6. Временные признаки
    time_features = self._create_time_features(data)
    features = pd.concat([features, time_features], axis=1)

    # 7. Признаки памяти рынка
    memory_features = self._create_memory_features(data)
    features = pd.concat([features, memory_features], axis=1)

    # Сохраняем имена признаков
    self.feature_names = features.columns.tolist()

    # Вычисляем статистику признаков
    self._calculate_feature_statistics(features)

    return features

  def _create_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Микроструктурные признаки рынка"""
    features = pd.DataFrame(index=data.index)

    # Эффективный спред
    features['effective_spread'] = 2 * np.abs(
      data['close'] - (data['high'] + data['low']) / 2
    ) / data['close']

    # Индикатор дисбаланса объема
    volume_ma = data['volume'].rolling(20).mean()
    features['volume_imbalance'] = (data['volume'] - volume_ma) / (volume_ma + 1e-9)

    # Токсичность потока ордеров (приближение)
    returns = data['close'].pct_change()
    features['order_flow_toxicity'] = (
        returns.rolling(10).std() * np.sqrt(data['volume'].rolling(10).mean())
    )

    # Вероятность информированной торговли (PIN approximation)
    buy_volume = data['volume'] * (data['close'] > data['open']).astype(int)
    sell_volume = data['volume'] * (data['close'] <= data['open']).astype(int)

    features['pin_approximation'] = np.abs(
      buy_volume.rolling(20).sum() - sell_volume.rolling(20).sum()
    ) / (buy_volume.rolling(20).sum() + sell_volume.rolling(20).sum() + 1e-9)

    # Realized variance
    features['realized_variance'] = returns.rolling(20).apply(
      lambda x: np.sum(x ** 2)
    )

    # Amihud illiquidity
    features['amihud_illiquidity'] = (
        np.abs(returns) / (data['volume'] * data['close'] + 1e-9)
    ).rolling(20).mean()

    return features

  def _create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Признаки рыночных режимов"""
    features = pd.DataFrame(index=data.index)

    returns = data['close'].pct_change()

    # Режим волатильности (через Markov regime switching approximation)
    vol_short = returns.rolling(10).std()
    vol_long = returns.rolling(50).std()
    features['volatility_regime'] = vol_short / (vol_long + 1e-9)

    # Трендовый режим
    sma_10 = data['close'].rolling(10).mean()
    sma_50 = data['close'].rolling(50).mean()
    features['trend_regime'] = (sma_10 - sma_50) / (sma_50 + 1e-9)

    # Режим моментума
    features['momentum_regime'] = returns.rolling(20).mean() / (
        returns.rolling(20).std() + 1e-9
    )

    # Режим ликвидности
    volume_ratio = data['volume'] / data['volume'].rolling(50).mean()
    spread_ratio = (data['high'] - data['low']) / data['close']
    features['liquidity_regime'] = volume_ratio / (spread_ratio + 1e-9)

    # Вероятности переходов между режимами
    for col in ['volatility_regime', 'trend_regime']:
      if col in features:
        features[f'{col}_change'] = features[col].diff()
        features[f'{col}_persistence'] = features[col].rolling(10).apply(
          lambda x: len(x[x > x.median()]) / len(x)
        )

    return features

  def _create_information_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Признаки на основе теории информации"""
    features = pd.DataFrame(index=data.index)

    returns = data['close'].pct_change()

    # Энтропия доходностей
    features['returns_entropy'] = returns.rolling(50).apply(
      lambda x: self._calculate_shannon_entropy(x)
    )

    # Взаимная информация между объемом и доходностью
    features['volume_return_mi'] = self._rolling_mutual_information(
      data['volume'], returns, window=50
    )

    # Условная энтропия
    features['conditional_entropy'] = self._calculate_conditional_entropy(
      returns, data['volume'], window=50
    )

    # Transfer entropy (упрощенная версия)
    features['transfer_entropy'] = self._calculate_transfer_entropy(
      data['volume'], returns, window=50
    )

    # Сложность Колмогорова (приближение через сжатие)
    features['kolmogorov_complexity'] = returns.rolling(50).apply(
      lambda x: self._estimate_kolmogorov_complexity(x)
    )

    return features

  def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Нелинейные взаимодействия между признаками"""
    features = pd.DataFrame(index=data.index)

    # Базовые признаки для взаимодействий
    returns = data['close'].pct_change()
    volume_norm = data['volume'] / data['volume'].rolling(50).mean()
    volatility = returns.rolling(20).std()

    # Полиномиальные взаимодействия
    features['volume_volatility_interaction'] = volume_norm * volatility
    features['volume_returns_squared'] = volume_norm * returns ** 2

    # Тригонометрические взаимодействия (для захвата циклов)
    time_of_day = pd.to_datetime(data.index).hour + pd.to_datetime(data.index).minute / 60
    features['volume_time_sin'] = volume_norm * np.sin(2 * np.pi * time_of_day / 24)
    features['volatility_time_cos'] = volatility * np.cos(2 * np.pi * time_of_day / 24)

    # Логарифмические взаимодействия
    features['log_volume_returns'] = np.log(volume_norm + 1) * returns

    # Экспоненциальные взаимодействия
    features['exp_decay_volume'] = volume_norm * np.exp(-np.abs(returns) * 10)

    return features

  def _create_cross_market_features(self, data: pd.DataFrame,
                                    external_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Межрыночные признаки"""
    features = pd.DataFrame(index=data.index)

    main_returns = data['close'].pct_change()

    for market_name, market_data in external_data.items():
      if 'close' not in market_data.columns:
        continue

      # Выравниваем индексы
      aligned_data = market_data.reindex(data.index, method='ffill')
      market_returns = aligned_data['close'].pct_change()

      # Корреляция
      features[f'correlation_{market_name}'] = main_returns.rolling(50).corr(market_returns)

      # Бета
      features[f'beta_{market_name}'] = self._rolling_beta(
        main_returns, market_returns, window=50
      )

      # Отношение волатильностей
      main_vol = main_returns.rolling(20).std()
      market_vol = market_returns.rolling(20).std()
      features[f'volatility_ratio_{market_name}'] = main_vol / (market_vol + 1e-9)

      # Lead-lag отношения
      for lag in [1, 5, 10]:
        features[f'lead_lag_{market_name}_{lag}'] = main_returns.rolling(20).corr(
          market_returns.shift(lag)
        )

    return features

  def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Временные признаки"""
    features = pd.DataFrame(index=data.index)

    dt_index = pd.to_datetime(data.index)

    # Время суток (циклическое кодирование)
    hour = dt_index.hour + dt_index.minute / 60
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # День недели
    day_of_week = dt_index.dayofweek
    features['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    features['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    # День месяца
    day_of_month = dt_index.day
    features['month_day_sin'] = np.sin(2 * np.pi * day_of_month / 30)
    features['month_day_cos'] = np.cos(2 * np.pi * day_of_month / 30)

    # Близость к важным событиям (выходные, начало/конец месяца)
    features['is_monday'] = (day_of_week == 0).astype(int)
    features['is_friday'] = (day_of_week == 4).astype(int)
    features['is_month_start'] = (day_of_month <= 3).astype(int)
    features['is_month_end'] = (day_of_month >= 28).astype(int)

    # Торговая сессия
    features['asian_session'] = ((hour >= 0) & (hour < 8)).astype(int)
    features['european_session'] = ((hour >= 8) & (hour < 16)).astype(int)
    features['american_session'] = ((hour >= 16) & (hour < 24)).astype(int)

    return features

  def _create_memory_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Признаки памяти рынка и долгосрочных зависимостей"""
    features = pd.DataFrame(index=data.index)

    returns = data['close'].pct_change()

    # Автокорреляционная функция на разных лагах
    for lag in [1, 5, 10, 20, 50]:
      features[f'autocorr_lag_{lag}'] = returns.rolling(100).apply(
        lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
      )

    # Частичная автокорреляция
    features['partial_autocorr'] = self._rolling_partial_autocorr(returns, lag=5, window=100)

    # Long memory parameter (d) estimation
    features['long_memory_d'] = self._estimate_long_memory(returns, window=100)

    # Mean reversion speed
    features['mean_reversion_speed'] = self._calculate_mean_reversion_speed(
      data['close'], window=50
    )

    # Persistence of shocks
    features['shock_persistence'] = self._calculate_shock_persistence(returns, window=50)

    return features

  # Вспомогательные методы

  @staticmethod
  def _calculate_shannon_entropy(series: pd.Series, bins: int = 10) -> float:
    """Расчет энтропии Шеннона"""
    if len(series.dropna()) < bins:
      return np.nan

    hist, _ = np.histogram(series.dropna(), bins=bins)
    hist = hist / hist.sum()
    hist = hist[hist > 0]

    return -np.sum(hist * np.log2(hist))

  def _rolling_mutual_information(self, x: pd.Series, y: pd.Series,
                                  window: int, bins: int = 10) -> pd.Series:
    """Скользящая взаимная информация"""

    def mi(x_window, y_window):
      if len(x_window) < bins * 2:
        return np.nan

      # Дискретизация
      x_discrete = pd.qcut(x_window, bins, labels=False, duplicates='drop')
      y_discrete = pd.qcut(y_window, bins, labels=False, duplicates='drop')

      # Совместное распределение
      xy_hist = np.histogram2d(x_discrete, y_discrete, bins=bins)[0]
      xy_hist = xy_hist / xy_hist.sum()

      # Маргинальные распределения
      x_hist = xy_hist.sum(axis=1)
      y_hist = xy_hist.sum(axis=0)

      # Взаимная информация
      mi_value = 0
      for i in range(bins):
        for j in range(bins):
          if xy_hist[i, j] > 0 and x_hist[i] > 0 and y_hist[j] > 0:
            mi_value += xy_hist[i, j] * np.log2(
              xy_hist[i, j] / (x_hist[i] * y_hist[j])
            )

      return mi_value

    return pd.Series([
      mi(x.iloc[max(0, i - window):i], y.iloc[max(0, i - window):i])
      for i in range(1, len(x) + 1)
    ], index=x.index)

  def _calculate_conditional_entropy(self, x: pd.Series, y: pd.Series,
                                     window: int, bins: int = 10) -> pd.Series:
    """H(X|Y) = H(X,Y) - H(Y)"""

    def cond_entropy(x_window, y_window):
      if len(x_window) < bins * 2:
        return np.nan

      # Совместная энтропия
      xy_combined = pd.DataFrame({'x': x_window, 'y': y_window})
      joint_entropy = self._calculate_shannon_entropy(
        xy_combined.apply(lambda row: f"{row['x']:.3f}_{row['y']:.3f}", axis=1),
        bins=bins
      )

      # Энтропия Y
      y_entropy = self._calculate_shannon_entropy(y_window, bins=bins)

      return joint_entropy - y_entropy

    return pd.Series([
      cond_entropy(x.iloc[max(0, i - window):i], y.iloc[max(0, i - window):i])
      for i in range(1, len(x) + 1)
    ], index=x.index)

  def _calculate_transfer_entropy(self, x: pd.Series, y: pd.Series,
                                  window: int, lag: int = 1) -> pd.Series:
    """Упрощенная transfer entropy"""

    def te(x_window, y_window):
      if len(x_window) < window:
        return np.nan

      # TE(X->Y) ≈ I(Y_t; X_{t-lag} | Y_{t-lag})
      y_future = y_window.iloc[lag:]
      x_past = x_window.iloc[:-lag]
      y_past = y_window.iloc[:-lag]

      if len(y_future) < 10:
        return np.nan

      # Используем корреляцию как прокси для TE
      corr_xy = np.corrcoef(y_future, x_past)[0, 1]
      corr_yy = np.corrcoef(y_future, y_past)[0, 1]

      return abs(corr_xy) - abs(corr_yy)

    return pd.Series([
      te(x.iloc[max(0, i - window):i], y.iloc[max(0, i - window):i])
      for i in range(1, len(x) + 1)
    ], index=x.index)

  @staticmethod
  def _estimate_kolmogorov_complexity(series: pd.Series) -> float:
    """Оценка сложности Колмогорова через сжатие"""
    if len(series) < 10:
      return np.nan

    # Преобразуем в строку с фиксированной точностью
    string_repr = ''.join([f'{x:.5f}' for x in series.dropna()])

    # Оценка через длину строки (упрощенно)
    # В реальности можно использовать zlib.compress
    unique_chars = len(set(string_repr))
    total_chars = len(string_repr)

    # Нормализованная сложность
    return unique_chars / (total_chars + 1)

  def _rolling_beta(self, y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Скользящая бета"""

    def beta(y_window, x_window):
      if len(y_window) < 10:
        return np.nan

      # Cov(Y,X) / Var(X)
      cov = np.cov(y_window, x_window)[0, 1]
      var = np.var(x_window)

      return cov / (var + 1e-9)

    return pd.Series([
      beta(y.iloc[max(0, i - window):i], x.iloc[max(0, i - window):i])
      for i in range(1, len(y) + 1)
    ], index=y.index)

  def _rolling_partial_autocorr(self, series: pd.Series, lag: int, window: int) -> pd.Series:
    """Скользящая частичная автокорреляция"""

    def pacf(window_data):
      if len(window_data) < lag + 10:
        return np.nan

      # Упрощенный PACF через регрессию
      y = window_data.iloc[lag:]
      X = pd.DataFrame({
        f'lag_{i}': window_data.shift(i).iloc[lag:]
        for i in range(1, lag + 1)
      })

      if X.isnull().any().any():
        return np.nan

      # Коэффициент при последнем лаге в множественной регрессии
      try:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[-1]
      except:
        return np.nan

    return series.rolling(window).apply(pacf)

  def _estimate_long_memory(self, returns: pd.Series, window: int) -> pd.Series:
    """Оценка параметра долгой памяти d"""

    def estimate_d(window_data):
      if len(window_data) < 50:
        return np.nan

      # R/S анализ для оценки H, затем d = H - 0.5
      cumsum = np.cumsum(window_data - window_data.mean())
      R = np.max(cumsum) - np.min(cumsum)
      S = np.std(window_data)

      if S == 0:
        return 0

      # Упрощенная оценка
      n = len(window_data)
      H = np.log(R / S) / np.log(n)
      d = H - 0.5

      return np.clip(d, -0.5, 0.5)

    return returns.rolling(window).apply(estimate_d)

  def _calculate_mean_reversion_speed(self, prices: pd.Series, window: int) -> pd.Series:
    """Скорость возврата к среднему"""

    def ou_speed(window_data):
      if len(window_data) < 20:
        return np.nan

      # Оцениваем через AR(1) модель
      returns = np.diff(np.log(window_data))
      if len(returns) < 2:
        return np.nan

      # Автокорреляция
      autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]

      # Скорость mean reversion ≈ -log(autocorr)
      if autocorr > 0 and autocorr < 1:
        return -np.log(autocorr)
      else:
        return 0

    return prices.rolling(window).apply(ou_speed)

  def _calculate_shock_persistence(self, returns: pd.Series, window: int) -> pd.Series:
    """Персистентность шоков"""

    def persistence(window_data):
      if len(window_data) < 20:
        return np.nan

      # Импульсная функция отклика
      shock_idx = np.argmax(np.abs(window_data))
      if shock_idx >= len(window_data) - 5:
        return np.nan

      # Измеряем затухание после шока
      post_shock = window_data.iloc[shock_idx + 1:shock_idx + 6]
      decay_rate = np.mean(np.abs(post_shock)) / np.abs(window_data.iloc[shock_idx])

      return decay_rate

    return returns.rolling(window).apply(persistence)

  def _calculate_feature_statistics(self, features: pd.DataFrame):
    """Вычисляет статистику признаков для мониторинга"""
    self.feature_stats = {
      'mean': features.mean().to_dict(),
      'std': features.std().to_dict(),
      'null_percentage': (features.isnull().sum() / len(features) * 100).to_dict(),
      'correlation_matrix': features.corr().to_dict()
    }


class EnhancedEnsembleModel:
  """Улучшенная ансамблевая модель с мета-обучением"""

  def __init__(self, anomaly_detector: Optional[MarketAnomalyDetector] = None):
    self.anomaly_detector = anomaly_detector
    self.feature_engineer = AdvancedFeatureEngineer()

    # Базовые модели
    self.models = {
      'rf': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
      ),
      'gb': GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=50,
        random_state=42
      ),
      'xgb': xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
      ),
      'lgb': lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
      )
    }

    # Мета-модель для стекинга
    self.meta_model = xgb.XGBClassifier(
      n_estimators=50,
      learning_rate=0.1,
      max_depth=3,
      random_state=42,
      use_label_encoder=False,
      eval_metric='logloss'
    )

    # Скейлеры для разных групп признаков
    self.scalers = {
      'standard': StandardScaler(),
      'robust': RobustScaler()
    }

    # Селектор признаков
    self.feature_selector = None
    self.selected_features = None

    # Параметры для адаптивного обучения
    self.performance_history = []
    self.feature_importance_history = []
    self.is_fitted = False

  def fit(self, X: pd.DataFrame, y: pd.Series,
          external_data: Optional[Dict[str, pd.DataFrame]] = None,
          optimize_features: bool = True):
    """
    Обучение ансамбля с оптимизацией признаков
    """
    logger.info("Начало обучения Enhanced Ensemble Model...")

    try:
      # 1. Создание продвинутых признаков
      logger.info("Создание продвинутых признаков...")
      X_enhanced = self.feature_engineer.create_advanced_features(X, external_data)

      # 2. Обнаружение и обработка аномалий
      if self.anomaly_detector and self.anomaly_detector.is_fitted:
        logger.info("Проверка на аномалии в обучающих данных...")
        anomalies = self.anomaly_detector.detect_anomalies(X, "training_data")

        # Фильтруем серьезные аномалии
        severe_anomalies = [a for a in anomalies if a.severity > 0.7]
        if severe_anomalies:
          logger.warning(f"Обнаружено {len(severe_anomalies)} серьезных аномалий в данных")
          # Можно добавить логику удаления аномальных точек

      # 3. Подготовка данных
      X_clean = X_enhanced.dropna()
      y_clean = y.loc[X_clean.index]

      if len(X_clean) < 100:
        raise ValueError("Недостаточно данных для обучения после очистки")

      # 4. Оптимизация признаков
      if optimize_features:
        logger.info("Оптимизация набора признаков...")
        X_clean, selected_features = self._optimize_features(X_clean, y_clean)
        self.selected_features = selected_features

      # 5. Масштабирование
      X_scaled = self.scalers['robust'].fit_transform(X_clean)

      # 6. Обучение базовых моделей с кросс-валидацией
      logger.info("Обучение базовых моделей...")
      base_predictions = np.zeros((len(X_scaled), len(self.models)))

      tscv = TimeSeriesSplit(n_splits=5)

      for i, (name, model) in enumerate(self.models.items()):
        logger.info(f"Обучение {name}...")

        # Кросс-валидация для оценки
        scores = cross_val_score(model, X_scaled, y_clean, cv=tscv, scoring='f1_weighted')
        logger.info(f"{name} CV F1-score: {scores.mean():.3f} (+/- {scores.std():.3f})")

        # Обучение на всех данных
        model.fit(X_scaled, y_clean)

        # Out-of-fold предсказания для мета-модели
        for train_idx, val_idx in tscv.split(X_scaled):
          X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
          y_train = y_clean.iloc[train_idx]

          # Клонируем модель для обучения
          model_clone = self._clone_model(model)
          model_clone.fit(X_train, y_train)

          # Предсказания вероятностей
          pred_proba = model_clone.predict_proba(X_val)
          base_predictions[val_idx, i] = pred_proba[:, 1]  # Вероятность класса 1

      # 7. Обучение мета-модели
      logger.info("Обучение мета-модели...")
      self.meta_model.fit(base_predictions, y_clean)

      # 8. Анализ важности признаков
      self._analyze_feature_importance(X_clean)

      self.is_fitted = True
      logger.info("Обучение Enhanced Ensemble Model завершено")

    except Exception as e:
      logger.error(f"Ошибка при обучении модели: {e}")
      raise

  def predict_proba(self, X: pd.DataFrame,
                    external_data: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[np.ndarray, MLPrediction]:
    """
    Предсказание с расширенной информацией
    """
    if not self.is_fitted:
      raise ValueError("Модель не обучена")

    try:
      # 1. Создание признаков
      X_enhanced = self.feature_engineer.create_advanced_features(X, external_data)

      # 2. Проверка на аномалии
      risk_assessment = {'anomaly_detected': False, 'anomaly_type': None, 'severity': 0}

      if self.anomaly_detector:
        anomalies = self.anomaly_detector.detect_anomalies(X, "prediction")
        if anomalies:
          most_severe = max(anomalies, key=lambda x: x.severity)
          risk_assessment = {
            'anomaly_detected': True,
            'anomaly_type': most_severe.anomaly_type.value,
            'severity': most_severe.severity,
            'description': most_severe.description
          }

      # 3. Подготовка данных
      X_clean = X_enhanced.dropna()
      if self.selected_features:
        X_clean = X_clean[self.selected_features]

      X_scaled = self.scalers['robust'].transform(X_clean)

      # 4. Получение предсказаний от базовых моделей
      base_predictions = np.zeros((len(X_scaled), len(self.models)))
      model_predictions = {}

      for i, (name, model) in enumerate(self.models.items()):
        pred_proba = model.predict_proba(X_scaled)
        base_predictions[:, i] = pred_proba[:, 1]
        model_predictions[name] = pred_proba

      # 5. Мета-предсказание
      final_proba = self.meta_model.predict_proba(base_predictions)

      # 6. Расчет согласованности моделей
      model_agreement = 1 - np.std(base_predictions, axis=1).mean()

      # 7. Определение типа сигнала
      probabilities = final_proba[-1]  # Последнее предсказание

      # Трехклассовая классификация: 0=SELL, 1=HOLD, 2=BUY
      if len(probabilities) == 3:
        sell_prob, hold_prob, buy_prob = probabilities
      else:
        # Бинарная классификация
        sell_prob = 1 - probabilities[1]
        buy_prob = probabilities[1]
        hold_prob = 0

      # Определяем сигнал
      if buy_prob > sell_prob and buy_prob > hold_prob:
        signal_type = SignalType.BUY
        probability = buy_prob
      elif sell_prob > buy_prob and sell_prob > hold_prob:
        signal_type = SignalType.SELL
        probability = sell_prob
      else:
        signal_type = SignalType.HOLD
        probability = hold_prob

      # 8. Важность признаков для данного предсказания
      feature_importance = self._get_prediction_feature_importance(X_scaled[-1])

      # 9. Формирование расширенного предсказания
      ml_prediction = MLPrediction(
        signal_type=signal_type,
        probability=probability,
        confidence=probability * model_agreement,
        model_agreement=model_agreement,
        feature_importance=feature_importance,
        risk_assessment=risk_assessment,
        metadata={
          'model_predictions': {k: v[-1].tolist() for k, v in model_predictions.items()},
          'feature_stats': self.feature_engineer.feature_stats
        }
      )

      return final_proba, ml_prediction

    except Exception as e:
      logger.error(f"Ошибка при предсказании: {e}")
      raise

  def _optimize_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
    """Оптимизация набора признаков"""

    # 1. Удаление высококоррелированных признаков
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(
      np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    X_reduced = X.drop(columns=to_drop)

    # 2. Отбор признаков по важности
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_reduced, y)

    importances = pd.Series(rf_selector.feature_importances_, index=X_reduced.columns)
    important_features = importances.nlargest(50).index.tolist()

    # 3. RFE для финального отбора
    rfe = RFE(
      estimator=xgb.XGBClassifier(random_state=42, use_label_encoder=False),
      n_features_to_select=30
    )

    X_important = X_reduced[important_features]
    rfe.fit(X_important, y)

    selected_features = X_important.columns[rfe.support_].tolist()

    logger.info(f"Отобрано {len(selected_features)} признаков из {len(X.columns)}")

    return X[selected_features], selected_features

  def _clone_model(self, model):
    """Клонирование модели с теми же параметрами"""
    from sklearn.base import clone
    return clone(model)

  def _analyze_feature_importance(self, X: pd.DataFrame):
    """Анализ важности признаков"""
    importances = {}

    for name, model in self.models.items():
      if hasattr(model, 'feature_importances_'):
        importances[name] = pd.Series(
          model.feature_importances_,
          index=X.columns
        ).nlargest(20).to_dict()

    self.feature_importance_history.append({
      'timestamp': datetime.now(),
      'importances': importances
    })

  def _get_prediction_feature_importance(self, x: np.ndarray) -> Dict[str, float]:
    """Важность признаков для конкретного предсказания"""
    # Используем SHAP values или похожий подход
    # Здесь упрощенная версия

    if hasattr(self.models['rf'], 'feature_importances_'):
      return dict(zip(
        self.selected_features[:10] if self.selected_features else [],
        self.models['rf'].feature_importances_[:10]
      ))
    return {}

  def save(self, filepath: str):
    """Сохранение модели"""
    model_data = {
      'models': self.models,
      'meta_model': self.meta_model,
      'scalers': self.scalers,
      'selected_features': self.selected_features,
      'feature_engineer': self.feature_engineer,
      'performance_history': self.performance_history,
      'is_fitted': self.is_fitted
    }

    joblib.dump(model_data, filepath)
    logger.info(f"Enhanced model сохранена в {filepath}")

  @classmethod
  def load(cls, filepath: str, anomaly_detector: Optional[MarketAnomalyDetector] = None) -> 'EnhancedEnsembleModel':
    """Загрузка модели"""
    model_data = joblib.load(filepath)

    instance = cls(anomaly_detector)
    instance.models = model_data['models']
    instance.meta_model = model_data['meta_model']
    instance.scalers = model_data['scalers']
    instance.selected_features = model_data['selected_features']
    instance.feature_engineer = model_data['feature_engineer']
    instance.performance_history = model_data['performance_history']
    instance.is_fitted = model_data['is_fitted']

    logger.info(f"Enhanced model загружена из {filepath}")
    return instance