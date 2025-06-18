# ml/anomaly_detector.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from collections import deque

from utils.logging_config import get_logger

warnings.filterwarnings('ignore')
logger = get_logger(__name__)


class AnomalyType(Enum):
  """Типы обнаруженных аномалий"""
  NORMAL = "normal"
  VOLUME_SPIKE = "volume_spike"
  PRICE_SPIKE = "price_spike"
  VOLATILITY_SPIKE = "volatility_spike"
  LIQUIDITY_CRISIS = "liquidity_crisis"
  FLASH_CRASH = "flash_crash"
  PUMP_DUMP = "pump_dump"
  MARKET_MANIPULATION = "market_manipulation"
  TECHNICAL_GLITCH = "technical_glitch"


@dataclass
class AnomalyReport:
  """Отчет об обнаруженной аномалии"""
  timestamp: datetime
  symbol: str
  anomaly_type: AnomalyType
  severity: float  # 0.0 - 1.0
  confidence: float  # 0.0 - 1.0
  description: str
  metrics: Dict[str, float]
  recommended_action: str

  def to_dict(self) -> Dict:
    return {
      'timestamp': self.timestamp.isoformat(),
      'symbol': self.symbol,
      'anomaly_type': self.anomaly_type.value,
      'severity': self.severity,
      'confidence': self.confidence,
      'description': self.description,
      'metrics': self.metrics,
      'recommended_action': self.recommended_action
    }


class MarketAnomalyDetector:
  """
  Комплексный детектор рыночных аномалий с использованием ML
  """

  def __init__(self, lookback_periods: int = 100, contamination: float = 0.05):
    """
    Args:
        lookback_periods: Количество периодов для анализа
        contamination: Ожидаемая доля аномалий в данных
    """
    self.lookback_periods = lookback_periods
    self.contamination = contamination

    # Модели для разных типов аномалий
    self.volume_detector = IsolationForest(
      contamination=contamination,
      random_state=42,
      n_estimators=100
    )

    self.price_detector = IsolationForest(
      contamination=contamination * 0.5,  # Ценовые аномалии реже
      random_state=42,
      n_estimators=100
    )

    self.pattern_detector = IsolationForest(
      contamination=contamination,
      random_state=42,
      n_estimators=150
    )

    # Скейлеры для нормализации
    self.volume_scaler = StandardScaler()
    self.price_scaler = StandardScaler()
    self.pattern_scaler = StandardScaler()

    # PCA для снижения размерности сложных паттернов
    self.pca = PCA(n_components=10)

    # История аномалий для предотвращения ложных срабатываний
    self.anomaly_history = deque(maxlen=1000)

    # Пороги для различных метрик
    self.thresholds = {
      'volume_spike_multiplier': 5.0,
      'price_change_percent': 10.0,
      'volatility_spike_multiplier': 3.0,
      'bid_ask_spread_percent': 2.0,
      'order_book_imbalance': 0.8
    }

    # Статистика
    self.detection_stats = {
      'total_checks': 0,
      'anomalies_detected': 0,
      'by_type': {}
    }

    self.is_fitted = False

  def fit(self, historical_data: pd.DataFrame):
    """
    Обучает детекторы на исторических данных
    """
    logger.info("Обучение детекторов аномалий...")

    try:
      # Извлекаем признаки для обучения
      volume_features = self._extract_volume_features(historical_data)
      price_features = self._extract_price_features(historical_data)
      pattern_features = self._extract_pattern_features(historical_data)

      # Удаляем NaN
      volume_features = volume_features.dropna()
      price_features = price_features.dropna()
      pattern_features = pattern_features.dropna()

      if len(volume_features) < 100:
        logger.warning("Недостаточно данных для обучения детекторов")
        return

      # Обучаем скейлеры и модели
      volume_scaled = self.volume_scaler.fit_transform(volume_features)
      self.volume_detector.fit(volume_scaled)

      price_scaled = self.price_scaler.fit_transform(price_features)
      self.price_detector.fit(price_scaled)

      # Для паттернов используем PCA
      pattern_scaled = self.pattern_scaler.fit_transform(pattern_features)
      pattern_reduced = self.pca.fit_transform(pattern_scaled)
      self.pattern_detector.fit(pattern_reduced)

      self.is_fitted = True
      logger.info("Детекторы аномалий успешно обучены")

    except Exception as e:
      logger.error(f"Ошибка при обучении детекторов: {e}")
      self.is_fitted = False

  def _extract_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Извлекает признаки связанные с объемом"""
    features = pd.DataFrame()

    # Относительный объем
    features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

    # Скорость изменения объема
    features['volume_roc'] = data['volume'].pct_change(5)

    # Объем относительно волатильности
    volatility = data['close'].pct_change().rolling(20).std()
    features['volume_to_volatility'] = data['volume'] / (volatility + 1e-9)

    # Z-score объема
    volume_mean = data['volume'].rolling(50).mean()
    volume_std = data['volume'].rolling(50).std()
    features['volume_zscore'] = (data['volume'] - volume_mean) / (volume_std + 1e-9)

    # Кумулятивный объем
    features['cumulative_volume_ratio'] = (
        data['volume'].rolling(5).sum() /
        data['volume'].rolling(20).sum()
    )

    return features

  def _extract_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Извлекает признаки связанные с ценой"""
    features = pd.DataFrame()

    # Простые изменения цены
    features['price_change_1'] = data['close'].pct_change()
    features['price_change_5'] = data['close'].pct_change(5)
    features['price_change_20'] = data['close'].pct_change(20)

    # Отклонение от скользящих средних
    sma_20 = data['close'].rolling(20).mean()
    sma_50 = data['close'].rolling(50).mean()
    features['price_to_sma20'] = (data['close'] - sma_20) / sma_20
    features['price_to_sma50'] = (data['close'] - sma_50) / sma_50

    # High-Low spread
    features['hl_spread'] = (data['high'] - data['low']) / data['close']

    # Ускорение цены
    price_change = data['close'].pct_change()
    features['price_acceleration'] = price_change.diff()

    # Z-score цены
    price_mean = data['close'].rolling(50).mean()
    price_std = data['close'].rolling(50).std()
    features['price_zscore'] = (data['close'] - price_mean) / (price_std + 1e-9)

    return features

  def _extract_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Извлекает сложные паттерны из данных"""
    features = pd.DataFrame()

    # Автокорреляция
    for lag in [1, 5, 10, 20]:
      features[f'autocorr_{lag}'] = data['close'].rolling(50).apply(
        lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
      )

    # Скошенность и эксцесс распределения доходностей
    returns = data['close'].pct_change()
    features['returns_skew'] = returns.rolling(50).skew()
    features['returns_kurtosis'] = returns.rolling(50).kurt()

    # Энтропия
    features['price_entropy'] = returns.rolling(50).apply(
      lambda x: self._calculate_entropy(x)
    )

    # Фрактальная размерность
    features['fractal_dimension'] = data['close'].rolling(50).apply(
      lambda x: self._calculate_fractal_dimension(x)
    )

    # Персистентность (Hurst exponent)
    features['hurst_exponent'] = data['close'].rolling(100).apply(
      lambda x: self._calculate_hurst_exponent(x)
    )

    # Микроструктурные признаки
    features['price_efficiency'] = self._calculate_price_efficiency(data)

    # Кластеризация волатильности
    volatility = returns.rolling(20).std()
    features['volatility_clustering'] = volatility.rolling(20).apply(
      lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan
    )

    return features

  def detect_anomalies(self, data: pd.DataFrame, symbol: str) -> List[AnomalyReport]:
    """
    Основной метод обнаружения аномалий
    """
    self.detection_stats['total_checks'] += 1
    anomalies = []

    if not self.is_fitted:
      logger.warning("Детекторы не обучены. Используем правила-эвристики.")
      return self._detect_anomalies_heuristic(data, symbol)

    try:
      # Проверяем различные типы аномалий

      # 1. Аномалии объема
      volume_anomalies = self._detect_volume_anomalies(data, symbol)
      anomalies.extend(volume_anomalies)

      # 2. Ценовые аномалии
      price_anomalies = self._detect_price_anomalies(data, symbol)
      anomalies.extend(price_anomalies)

      # 3. Паттерновые аномалии
      pattern_anomalies = self._detect_pattern_anomalies(data, symbol)
      anomalies.extend(pattern_anomalies)

      # 4. Специфические проверки
      specific_anomalies = self._detect_specific_anomalies(data, symbol)
      anomalies.extend(specific_anomalies)

      # Фильтруем дубликаты и слабые аномалии
      anomalies = self._filter_anomalies(anomalies)

      # Обновляем статистику
      if anomalies:
        self.detection_stats['anomalies_detected'] += len(anomalies)
        for anomaly in anomalies:
          anomaly_type = anomaly.anomaly_type.value
          self.detection_stats['by_type'][anomaly_type] = \
            self.detection_stats['by_type'].get(anomaly_type, 0) + 1

      # Сохраняем в историю
      self.anomaly_history.extend(anomalies)

      return anomalies

    except Exception as e:
      logger.error(f"Ошибка при обнаружении аномалий для {symbol}: {e}")
      return []

  def _detect_volume_anomalies(self, data: pd.DataFrame, symbol: str) -> List[AnomalyReport]:
    """Обнаружение аномалий в объеме торгов"""
    anomalies = []

    try:
      # Извлекаем признаки
      features = self._extract_volume_features(data)
      latest_features = features.iloc[-1:]

      if latest_features.isnull().any().any():
        return anomalies

      # Используем ML модель
      scaled_features = self.volume_scaler.transform(latest_features)
      anomaly_score = self.volume_detector.decision_function(scaled_features)[0]
      is_anomaly = self.volume_detector.predict(scaled_features)[0] == -1

      if is_anomaly:
        # Определяем тип аномалии
        volume_ratio = latest_features['volume_ratio'].iloc[0]
        volume_zscore = latest_features['volume_zscore'].iloc[0]

        if volume_ratio > self.thresholds['volume_spike_multiplier']:
          anomaly_type = AnomalyType.VOLUME_SPIKE
          severity = min(volume_ratio / 10, 1.0)
          description = f"Обнаружен аномальный всплеск объема (x{volume_ratio:.1f} от среднего)"
          action = "Осторожно с входом в позицию, возможна манипуляция"
        else:
          anomaly_type = AnomalyType.MARKET_MANIPULATION
          severity = 0.6
          description = "Подозрительная активность в объеме торгов"
          action = "Усилить мониторинг, отложить торговые решения"

        anomaly = AnomalyReport(
          timestamp=datetime.now(),
          symbol=symbol,
          anomaly_type=anomaly_type,
          severity=severity,
          confidence=abs(anomaly_score),
          description=description,
          metrics={
            'volume_ratio': volume_ratio,
            'volume_zscore': volume_zscore,
            'anomaly_score': anomaly_score
          },
          recommended_action=action
        )
        anomalies.append(anomaly)

    except Exception as e:
      logger.error(f"Ошибка при обнаружении объемных аномалий: {e}")

    return anomalies

  def _detect_price_anomalies(self, data: pd.DataFrame, symbol: str) -> List[AnomalyReport]:
    """Обнаружение ценовых аномалий"""
    anomalies = []

    try:
      features = self._extract_price_features(data)
      latest_features = features.iloc[-1:]

      if latest_features.isnull().any().any():
        return anomalies

      # ML предсказание
      scaled_features = self.price_scaler.transform(latest_features)
      anomaly_score = self.price_detector.decision_function(scaled_features)[0]
      is_anomaly = self.price_detector.predict(scaled_features)[0] == -1

      if is_anomaly:
        price_change = latest_features['price_change_5'].iloc[0] * 100
        price_zscore = latest_features['price_zscore'].iloc[0]

        # Классификация типа аномалии
        if abs(price_change) > self.thresholds['price_change_percent']:
          if price_change < -self.thresholds['price_change_percent']:
            anomaly_type = AnomalyType.FLASH_CRASH
            severity = min(abs(price_change) / 20, 1.0)
            description = f"Возможный flash crash! Цена упала на {abs(price_change):.1f}%"
            action = "НЕМЕДЛЕННО закрыть long позиции, проверить стоп-лоссы"
          else:
            anomaly_type = AnomalyType.PRICE_SPIKE
            severity = min(price_change / 20, 1.0)
            description = f"Аномальный рост цены на {price_change:.1f}%"
            action = "Не входить в long позиции, возможен разворот"
        else:
          anomaly_type = AnomalyType.PRICE_SPIKE
          severity = 0.5
          description = "Необычное ценовое движение"
          action = "Усилить контроль рисков"

        anomaly = AnomalyReport(
          timestamp=datetime.now(),
          symbol=symbol,
          anomaly_type=anomaly_type,
          severity=severity,
          confidence=abs(anomaly_score),
          description=description,
          metrics={
            'price_change_5bar': price_change,
            'price_zscore': price_zscore,
            'anomaly_score': anomaly_score
          },
          recommended_action=action
        )
        anomalies.append(anomaly)

    except Exception as e:
      logger.error(f"Ошибка при обнаружении ценовых аномалий: {e}")

    return anomalies

  def _detect_pattern_anomalies(self, data: pd.DataFrame, symbol: str) -> List[AnomalyReport]:
    """Обнаружение аномальных паттернов"""
    anomalies = []

    try:
      features = self._extract_pattern_features(data)
      latest_features = features.iloc[-1:]

      if latest_features.isnull().any().any():
        return anomalies

      # Снижение размерности и предсказание
      scaled_features = self.pattern_scaler.transform(latest_features)
      reduced_features = self.pca.transform(scaled_features)
      anomaly_score = self.pattern_detector.decision_function(reduced_features)[0]
      is_anomaly = self.pattern_detector.predict(reduced_features)[0] == -1

      if is_anomaly:
        # Анализируем конкретные метрики
        hurst = latest_features['hurst_exponent'].iloc[0]
        entropy = latest_features['price_entropy'].iloc[0]

        if hurst < 0.4:  # Антиперсистентность
          anomaly_type = AnomalyType.VOLATILITY_SPIKE
          severity = 0.7
          description = "Обнаружена антиперсистентность - рынок крайне нестабилен"
          action = "Снизить размер позиций, расширить стоп-лоссы"
        elif entropy > 0.9:  # Высокая энтропия
          anomaly_type = AnomalyType.TECHNICAL_GLITCH
          severity = 0.6
          description = "Аномально высокая энтропия - возможны технические сбои"
          action = "Проверить качество данных, временно приостановить торговлю"
        else:
          anomaly_type = AnomalyType.MARKET_MANIPULATION
          severity = 0.5
          description = "Обнаружены подозрительные паттерны в движении цены"
          action = "Усилить фильтрацию сигналов"

        anomaly = AnomalyReport(
          timestamp=datetime.now(),
          symbol=symbol,
          anomaly_type=anomaly_type,
          severity=severity,
          confidence=abs(anomaly_score),
          description=description,
          metrics={
            'hurst_exponent': hurst,
            'price_entropy': entropy,
            'anomaly_score': anomaly_score
          },
          recommended_action=action
        )
        anomalies.append(anomaly)

    except Exception as e:
      logger.error(f"Ошибка при обнаружении паттерновых аномалий: {e}")

    return anomalies

  def _detect_specific_anomalies(self, data: pd.DataFrame, symbol: str) -> List[AnomalyReport]:
    """Обнаружение специфических типов аномалий"""
    anomalies = []

    # Проверка на Pump & Dump
    pump_dump = self._check_pump_dump(data, symbol)
    if pump_dump:
      anomalies.append(pump_dump)

    # Проверка на кризис ликвидности
    liquidity_crisis = self._check_liquidity_crisis(data, symbol)
    if liquidity_crisis:
      anomalies.append(liquidity_crisis)

    return anomalies

  def _check_pump_dump(self, data: pd.DataFrame, symbol: str) -> Optional[AnomalyReport]:
    """Проверка на pump & dump схему"""
    if len(data) < 20:
      return None

    # Анализируем последние 20 баров
    recent_data = data.tail(20)

    # Признаки pump & dump:
    # 1. Резкий рост объема и цены
    # 2. Последующее падение цены при высоком объеме

    volume_spike = recent_data['volume'].max() / recent_data['volume'].mean()
    price_change = (recent_data['close'].max() - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]

    # Проверяем, был ли pump
    if volume_spike > 5 and price_change > 0.15:
      # Проверяем, начался ли dump
      max_price_idx = recent_data['close'].idxmax()
      if max_price_idx != recent_data.index[-1]:  # Максимум не в конце
        price_drop = (recent_data['close'].iloc[-1] - recent_data['close'].max()) / recent_data['close'].max()

        if price_drop < -0.1:  # Падение более 10%
          return AnomalyReport(
            timestamp=datetime.now(),
            symbol=symbol,
            anomaly_type=AnomalyType.PUMP_DUMP,
            severity=0.9,
            confidence=0.8,
            description=f"Обнаружена pump & dump схема! Рост {price_change * 100:.1f}%, затем падение {abs(price_drop) * 100:.1f}%",
            metrics={
              'volume_spike': volume_spike,
              'price_pump': price_change,
              'price_dump': price_drop
            },
            recommended_action="ИЗБЕГАТЬ торговли данным активом! Высокий риск манипуляции"
          )

    return None

  def _check_liquidity_crisis(self, data: pd.DataFrame, symbol: str) -> Optional[AnomalyReport]:
    """Проверка на кризис ликвидности"""
    if len(data) < 10:
      return None

    recent_data = data.tail(10)

    # Признаки кризиса ликвидности:
    # 1. Увеличение спреда high-low
    # 2. Резкие движения цены при малом объеме
    # 3. Разрывы в ценовом графике

    avg_spread = ((recent_data['high'] - recent_data['low']) / recent_data['close']).mean()
    historical_spread = ((data['high'] - data['low']) / data['close']).rolling(50).mean().iloc[-1]

    spread_increase = avg_spread / (historical_spread + 1e-9)

    # Волатильность на единицу объема
    returns_vol = recent_data['close'].pct_change().std()
    volume_avg = recent_data['volume'].mean()
    historical_volume = data['volume'].rolling(50).mean().iloc[-1]

    if spread_increase > 2 and volume_avg < historical_volume * 0.5:
      return AnomalyReport(
        timestamp=datetime.now(),
        symbol=symbol,
        anomaly_type=AnomalyType.LIQUIDITY_CRISIS,
        severity=0.8,
        confidence=0.7,
        description="Возможный кризис ликвидности - широкие спреды при низком объеме",
        metrics={
          'spread_increase': spread_increase,
          'volume_decrease': volume_avg / historical_volume,
          'volatility': returns_vol
        },
        recommended_action="Использовать только лимитные ордера, увеличить проскальзывание"
      )

    return None

  def _detect_anomalies_heuristic(self, data: pd.DataFrame, symbol: str) -> List[AnomalyReport]:
    """Эвристическое обнаружение аномалий без ML"""
    anomalies = []

    if len(data) < 50:
      return anomalies

    # Простые правила для обнаружения аномалий
    latest = data.iloc[-1]

    # 1. Проверка объема
    volume_ma = data['volume'].rolling(20).mean().iloc[-1]
    if latest['volume'] > volume_ma * self.thresholds['volume_spike_multiplier']:
      anomalies.append(AnomalyReport(
        timestamp=datetime.now(),
        symbol=symbol,
        anomaly_type=AnomalyType.VOLUME_SPIKE,
        severity=0.6,
        confidence=0.7,
        description=f"Объем превышает средний в {latest['volume'] / volume_ma:.1f} раз",
        metrics={'volume_ratio': latest['volume'] / volume_ma},
        recommended_action="Проверить новости, возможна важная информация"
      ))

    # 2. Проверка волатильности
    returns = data['close'].pct_change()
    current_vol = returns.tail(10).std()
    historical_vol = returns.rolling(50).std().iloc[-1]

    if current_vol > historical_vol * self.thresholds['volatility_spike_multiplier']:
      anomalies.append(AnomalyReport(
        timestamp=datetime.now(),
        symbol=symbol,
        anomaly_type=AnomalyType.VOLATILITY_SPIKE,
        severity=0.7,
        confidence=0.6,
        description="Волатильность значительно превышает историческую",
        metrics={
          'current_volatility': current_vol,
          'historical_volatility': historical_vol
        },
        recommended_action="Уменьшить размер позиций, расширить стоп-лоссы"
      ))

    return anomalies

  def _filter_anomalies(self, anomalies: List[AnomalyReport]) -> List[AnomalyReport]:
    """Фильтрация и приоритизация аномалий"""
    if not anomalies:
      return []

    # Удаляем слабые аномалии
    filtered = [a for a in anomalies if a.severity >= 0.5 and a.confidence >= 0.5]

    # Сортируем по важности (severity * confidence)
    filtered.sort(key=lambda x: x.severity * x.confidence, reverse=True)

    # Оставляем только топ-3 самых важных
    return filtered[:3]

  # Вспомогательные методы для расчета метрик

  @staticmethod
  def _calculate_entropy(series: pd.Series) -> float:
    """Расчет энтропии Шеннона"""
    if len(series) < 2:
      return np.nan

    # Дискретизируем значения
    hist, _ = np.histogram(series.dropna(), bins=10)
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # Убираем нули

    # Энтропия
    entropy = -np.sum(hist * np.log2(hist))
    return entropy / np.log2(len(hist))  # Нормализуем

  @staticmethod
  def _calculate_fractal_dimension(series: pd.Series) -> float:
    """Упрощенный расчет фрактальной размерности"""
    if len(series) < 10:
      return np.nan

    # Используем метод подсчета ячеек (box-counting)
    n = len(series)
    max_val = series.max()
    min_val = series.min()

    if max_val == min_val:
      return 1.0

    # Нормализуем
    normalized = (series - min_val) / (max_val - min_val)

    # Считаем занятые ячейки для разных масштабов
    dimensions = []
    for scale in [2, 4, 8, 16]:
      if scale > n:
        continue

      boxes = set()
      for i in range(0, n, n // scale):
        chunk = normalized.iloc[i:i + n // scale]
        if len(chunk) > 0:
          box_x = i // (n // scale)
          box_y = int(chunk.mean() * scale)
          boxes.add((box_x, box_y))

      if len(boxes) > 0:
        dimensions.append(np.log(len(boxes)) / np.log(scale))

    return np.mean(dimensions) if dimensions else 1.5

  @staticmethod
  def _calculate_hurst_exponent(series: pd.Series) -> float:
    """Расчет показателя Херста"""
    if len(series) < 20:
      return np.nan

    # Simplified R/S analysis
    lags = range(2, min(20, len(series) // 2))
    tau = []

    for lag in lags:
      # Стандартное отклонение приращений
      std_dev = np.sqrt(np.std(np.subtract(series[lag:].values, series[:-lag].values)))
      tau.append(std_dev)

    # Линейная регрессия в лог-лог координатах
    if len(tau) > 2:
      log_lags = np.log(list(lags))
      log_tau = np.log(tau)
      hurst = np.polyfit(log_lags, log_tau, 1)[0] * 2
      return np.clip(hurst, 0, 1)

    return 0.5

  def _calculate_price_efficiency(self, data: pd.DataFrame) -> pd.Series:
    """Расчет эффективности цены (отклонение от случайного блуждания)"""
    # Variance ratio test
    returns = data['close'].pct_change().dropna()

    def variance_ratio(returns, k):
      if len(returns) < k * 2:
        return np.nan

      # Дисперсия k-периодных доходностей
      k_returns = returns.rolling(k).sum().dropna()
      var_k = k_returns.var()

      # Дисперсия 1-периодных доходностей
      var_1 = returns.var()

      # Variance ratio
      vr = var_k / (k * var_1)
      return vr

    # Рассчитываем для окна
    efficiency = returns.rolling(50).apply(lambda x: variance_ratio(x, 5))
    return efficiency

  def get_statistics(self) -> Dict[str, Any]:
    """Возвращает статистику работы детектора"""
    return {
      'total_checks': self.detection_stats['total_checks'],
      'anomalies_detected': self.detection_stats['anomalies_detected'],
      'detection_rate': (
        self.detection_stats['anomalies_detected'] /
        self.detection_stats['total_checks']
        if self.detection_stats['total_checks'] > 0 else 0
      ),
      'by_type': dict(self.detection_stats['by_type']),
      'is_fitted': self.is_fitted
    }

  def save(self, filepath: str):
    """Сохраняет обученную модель"""
    if not self.is_fitted:
      logger.warning("Модель не обучена, сохранение пропущено")
      return

    model_data = {
      'volume_detector': self.volume_detector,
      'price_detector': self.price_detector,
      'pattern_detector': self.pattern_detector,
      'volume_scaler': self.volume_scaler,
      'price_scaler': self.price_scaler,
      'pattern_scaler': self.pattern_scaler,
      'pca': self.pca,
      'thresholds': self.thresholds,
      'lookback_periods': self.lookback_periods,
      'contamination': self.contamination
    }

    joblib.dump(model_data, filepath)
    logger.info(f"Детектор аномалий сохранен в {filepath}")

  @classmethod
  def load(cls, filepath: str) -> 'MarketAnomalyDetector':
    """Загружает обученную модель"""
    model_data = joblib.load(filepath)

    detector = cls(
      lookback_periods=model_data['lookback_periods'],
      contamination=model_data['contamination']
    )

    detector.volume_detector = model_data['volume_detector']
    detector.price_detector = model_data['price_detector']
    detector.pattern_detector = model_data['pattern_detector']
    detector.volume_scaler = model_data['volume_scaler']
    detector.price_scaler = model_data['price_scaler']
    detector.pattern_scaler = model_data['pattern_scaler']
    detector.pca = model_data['pca']
    detector.thresholds = model_data['thresholds']
    detector.is_fitted = True

    logger.info(f"Детектор аномалий загружен из {filepath}")
    return detector