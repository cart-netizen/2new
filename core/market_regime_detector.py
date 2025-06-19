# core/market_regime_detector.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import deque
import pandas_ta as ta
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from core.data_fetcher import DataFetcher
from core.enums import Timeframe
from utils.logging_config import get_logger

logger = get_logger(__name__)


class MarketRegime(Enum):
  """Типы рыночных режимов"""
  STRONG_TREND_UP = "strong_trend_up"  # Сильный восходящий тренд
  TREND_UP = "trend_up"  # Умеренный восходящий тренд
  WEAK_TREND_UP = "weak_trend_up"  # Слабый восходящий тренд
  RANGING = "ranging"  # Боковик/флэт
  WEAK_TREND_DOWN = "weak_trend_down"  # Слабый нисходящий тренд
  TREND_DOWN = "trend_down"  # Умеренный нисходящий тренд
  STRONG_TREND_DOWN = "strong_trend_down"  # Сильный нисходящий тренд
  VOLATILE = "volatile"  # Высокая волатильность
  QUIET = "quiet"  # Низкая активность
  BREAKOUT = "breakout"  # Пробой уровня
  REVERSAL = "reversal"  # Разворот тренда
  SQUEEZE = "squeeze"  # Сжатие волатильности


@dataclass
class RegimeCharacteristics:
  """Характеристики текущего режима"""
  primary_regime: MarketRegime
  secondary_regime: Optional[MarketRegime]
  confidence: float  # 0.0-1.0

  # Количественные характеристики
  trend_strength: float  # -1.0 to 1.0 (отрицательные = вниз)
  volatility_level: float  # 0.0 to 1.0
  momentum_score: float  # -1.0 to 1.0
  volume_profile: str  # 'increasing', 'decreasing', 'stable', 'spike'

  # Временные характеристики
  regime_duration: timedelta
  time_since_change: timedelta
  expected_duration: Optional[timedelta]

  # Вероятности переходов
  transition_probabilities: Dict[MarketRegime, float]

  # Метаданные
  supporting_indicators: List[str]
  key_levels: Dict[str, float]  # support, resistance, pivot
  timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RegimeParameters:
  """Оптимальные параметры торговли для режима"""
  # Стратегии
  recommended_strategies: List[str]
  avoided_strategies: List[str]

  # Параметры риска
  position_size_multiplier: float
  stop_loss_multiplier: float
  take_profit_multiplier: float
  max_positions: int

  # Параметры входа
  use_limit_orders: bool
  entry_confirmation_required: bool
  min_signal_quality: float

  # Таймфреймы
  primary_timeframe: Timeframe
  confirmation_timeframes: List[Timeframe]

  # Индикаторы
  key_indicators: List[str]
  indicator_settings: Dict[str, Any]

  # Фильтры
  volume_filter_enabled: bool
  volatility_filter_enabled: bool
  correlation_filter_strict: bool


class MarketRegimeDetector:
  """
  Комплексная система определения рыночных режимов
  """

  def __init__(self, data_fetcher: DataFetcher):
    self.data_fetcher = data_fetcher

    # История режимов
    self.regime_history: Dict[str, deque] = {}  # symbol -> deque of RegimeCharacteristics
    self.regime_transitions: Dict[str, List[Tuple[MarketRegime, MarketRegime, datetime]]] = {}

    # Параметры определения режимов
    self.lookback_periods = {
      'short': 20,
      'medium': 50,
      'long': 200
    }

    # Hidden Markov Model для режимов
    self.hmm_models: Dict[str, GaussianMixture] = {}
    self.scaler = StandardScaler()

    # Кэш текущих режимов
    self.current_regimes: Dict[str, RegimeCharacteristics] = {}
    self.cache_ttl = 300  # 5 минут

    # Параметры для каждого режима
    self.regime_parameters = self._initialize_regime_parameters()

    # Статистика
    self.regime_accuracy = {}
    self.false_signals = deque(maxlen=100)

  def _initialize_regime_parameters(self) -> Dict[MarketRegime, RegimeParameters]:
    """Инициализирует оптимальные параметры для каждого режима"""
    return {
      MarketRegime.STRONG_TREND_UP: RegimeParameters(
        recommended_strategies=['Momentum_Spike', 'Dual_Thrust', 'Live_ML_Strategy'],
        avoided_strategies=['Mean_Reversion_BB'],
        position_size_multiplier=1.2,
        stop_loss_multiplier=1.5,  # Шире стопы в тренде
        take_profit_multiplier=2.0,  # Больше целей
        max_positions=5,
        use_limit_orders=False,  # Маркет для входа в тренд
        entry_confirmation_required=False,
        min_signal_quality=0.6,
        primary_timeframe=Timeframe.ONE_HOUR,
        confirmation_timeframes=[Timeframe.FOUR_HOURS],
        key_indicators=['ADX', 'MACD', 'RSI'],
        indicator_settings={'ADX_threshold': 30, 'RSI_overbought': 80},
        volume_filter_enabled=True,
        volatility_filter_enabled=False,
        correlation_filter_strict=False
      ),

      MarketRegime.TREND_UP: RegimeParameters(
        recommended_strategies=['Live_ML_Strategy', 'Ichimoku_Cloud', 'Dual_Thrust'],
        avoided_strategies=['Mean_Reversion_BB'],
        position_size_multiplier=1.0,
        stop_loss_multiplier=1.2,
        take_profit_multiplier=1.5,
        max_positions=4,
        use_limit_orders=False,
        entry_confirmation_required=True,
        min_signal_quality=0.65,
        primary_timeframe=Timeframe.ONE_HOUR,
        confirmation_timeframes=[Timeframe.FIFTEEN_MINUTES],
        key_indicators=['EMA', 'RSI', 'Volume'],
        indicator_settings={'EMA_period': 20, 'RSI_oversold': 40},
        volume_filter_enabled=True,
        volatility_filter_enabled=True,
        correlation_filter_strict=True
      ),

      MarketRegime.RANGING: RegimeParameters(
        recommended_strategies=['Mean_Reversion_BB', 'Live_ML_Strategy'],
        avoided_strategies=['Momentum_Spike', 'Dual_Thrust'],
        position_size_multiplier=0.8,
        stop_loss_multiplier=0.8,  # Узкие стопы в боковике
        take_profit_multiplier=1.0,  # Небольшие цели
        max_positions=3,
        use_limit_orders=True,  # Лимитки для лучшей цены
        entry_confirmation_required=True,
        min_signal_quality=0.7,
        primary_timeframe=Timeframe.THIRTY_MINUTES,
        confirmation_timeframes=[Timeframe.FIVE_MINUTES],
        key_indicators=['BB', 'RSI', 'Stochastic'],
        indicator_settings={'BB_period': 20, 'RSI_period': 14},
        volume_filter_enabled=False,
        volatility_filter_enabled=True,
        correlation_filter_strict=True
      ),

      MarketRegime.VOLATILE: RegimeParameters(
        recommended_strategies=['Live_ML_Strategy'],
        avoided_strategies=['Momentum_Spike', 'Mean_Reversion_BB'],
        position_size_multiplier=0.5,  # Уменьшаем риск
        stop_loss_multiplier=2.0,  # Широкие стопы
        take_profit_multiplier=2.5,  # Большие цели
        max_positions=2,
        use_limit_orders=True,
        entry_confirmation_required=True,
        min_signal_quality=0.8,  # Только лучшие сигналы
        primary_timeframe=Timeframe.FIFTEEN_MINUTES,
        confirmation_timeframes=[Timeframe.FIVE_MINUTES],
        key_indicators=['ATR', 'BB', 'Volume'],
        indicator_settings={'ATR_multiplier': 2.5},
        volume_filter_enabled=True,
        volatility_filter_enabled=False,  # Уже волатильно
        correlation_filter_strict=True
      ),

      MarketRegime.QUIET: RegimeParameters(
        recommended_strategies=[],  # Избегаем торговли
        avoided_strategies=['ALL'],
        position_size_multiplier=0.3,
        stop_loss_multiplier=0.5,
        take_profit_multiplier=0.8,
        max_positions=1,
        use_limit_orders=True,
        entry_confirmation_required=True,
        min_signal_quality=0.9,  # Только исключительные сигналы
        primary_timeframe=Timeframe.FOUR_HOURS,
        confirmation_timeframes=[Timeframe.ONE_DAY],
        key_indicators=['Volume', 'ATR'],
        indicator_settings={'min_volume_threshold': 0.5},
        volume_filter_enabled=True,
        volatility_filter_enabled=True,
        correlation_filter_strict=True
      ),

      MarketRegime.BREAKOUT: RegimeParameters(
        recommended_strategies=['Momentum_Spike', 'Dual_Thrust'],
        avoided_strategies=['Mean_Reversion_BB'],
        position_size_multiplier=1.1,
        stop_loss_multiplier=1.0,
        take_profit_multiplier=2.0,
        max_positions=3,
        use_limit_orders=False,  # Маркет для быстрого входа
        entry_confirmation_required=False,
        min_signal_quality=0.65,
        primary_timeframe=Timeframe.FIFTEEN_MINUTES,
        confirmation_timeframes=[Timeframe.FIVE_MINUTES],
        key_indicators=['Volume', 'RSI', 'MACD'],
        indicator_settings={'volume_spike': 2.0},
        volume_filter_enabled=True,
        volatility_filter_enabled=False,
        correlation_filter_strict=False
      ),

      MarketRegime.SQUEEZE: RegimeParameters(
        recommended_strategies=['Live_ML_Strategy'],
        avoided_strategies=['Momentum_Spike'],
        position_size_multiplier=0.7,
        stop_loss_multiplier=0.6,
        take_profit_multiplier=1.5,
        max_positions=2,
        use_limit_orders=True,
        entry_confirmation_required=True,
        min_signal_quality=0.75,
        primary_timeframe=Timeframe.ONE_HOUR,
        confirmation_timeframes=[Timeframe.FOUR_HOURS],
        key_indicators=['BB', 'Keltner', 'ATR'],
        indicator_settings={'BB_squeeze_threshold': 0.02},
        volume_filter_enabled=False,
        volatility_filter_enabled=True,
        correlation_filter_strict=True
      )
    }

    # Добавляем параметры для остальных режимов
    for regime in [MarketRegime.TREND_DOWN, MarketRegime.STRONG_TREND_DOWN,
                   MarketRegime.WEAK_TREND_UP, MarketRegime.WEAK_TREND_DOWN,
                   MarketRegime.REVERSAL]:
      if regime not in self.regime_parameters:
        # Копируем похожие параметры
        if 'DOWN' in regime.value:
          base = self.regime_parameters.get(MarketRegime.TREND_UP)
          self.regime_parameters[regime] = RegimeParameters(
            recommended_strategies=base.recommended_strategies,
            avoided_strategies=base.avoided_strategies,
            position_size_multiplier=base.position_size_multiplier,
            stop_loss_multiplier=base.stop_loss_multiplier,
            take_profit_multiplier=base.take_profit_multiplier,
            max_positions=base.max_positions,
            use_limit_orders=base.use_limit_orders,
            entry_confirmation_required=base.entry_confirmation_required,
            min_signal_quality=base.min_signal_quality,
            primary_timeframe=base.primary_timeframe,
            confirmation_timeframes=base.confirmation_timeframes,
            key_indicators=base.key_indicators,
            indicator_settings=base.indicator_settings,
            volume_filter_enabled=base.volume_filter_enabled,
            volatility_filter_enabled=base.volatility_filter_enabled,
            correlation_filter_strict=base.correlation_filter_strict
          )
        else:
          self.regime_parameters[regime] = self.regime_parameters[MarketRegime.RANGING]

    return self.regime_parameters

  async def detect_regime(self, symbol: str, market_data: pd.DataFrame,
                          additional_timeframes: Optional[
                            Dict[Timeframe, pd.DataFrame]] = None) -> RegimeCharacteristics:
    """
    Определяет текущий рыночный режим для символа
    """
    logger.debug(f"Определение режима для {symbol}")

    try:
      # Проверяем кэш
      if symbol in self.current_regimes:
        cached = self.current_regimes[symbol]
        if (datetime.now() - cached.timestamp).seconds < self.cache_ttl:
          return cached

      # 1. Извлекаем признаки режима
      features = self._extract_regime_features(market_data)

      # 2. Основное определение режима
      primary_regime = self._determine_primary_regime(features, market_data)

      # 3. Вторичный режим (если есть)
      secondary_regime = self._determine_secondary_regime(features, primary_regime)

      # 4. Расчет уверенности
      confidence = self._calculate_regime_confidence(features, primary_regime)

      # 5. Анализ переходов
      transition_probs = self._calculate_transition_probabilities(symbol, primary_regime)

      # 6. Ключевые уровни
      key_levels = self._identify_key_levels(market_data)

      # 7. Временные характеристики
      regime_duration, time_since_change = self._calculate_regime_timing(symbol, primary_regime)

      # 8. Создаем характеристики режима
      characteristics = RegimeCharacteristics(
        primary_regime=primary_regime,
        secondary_regime=secondary_regime,
        confidence=confidence,
        trend_strength=features['trend_strength'],
        volatility_level=features['volatility_level'],
        momentum_score=features['momentum_score'],
        volume_profile=features['volume_profile'],
        regime_duration=regime_duration,
        time_since_change=time_since_change,
        expected_duration=self._estimate_regime_duration(primary_regime),
        transition_probabilities=transition_probs,
        supporting_indicators=features['supporting_indicators'],
        key_levels=key_levels
      )

      # 9. Сохраняем в историю и кэш
      self._update_regime_history(symbol, characteristics)
      self.current_regimes[symbol] = characteristics

      # 10. Логируем изменение режима
      if self._is_regime_change(symbol, primary_regime):
        logger.info(f"🔄 Изменение режима для {symbol}: {primary_regime.value}")
        logger.info(f"  Уверенность: {confidence:.2f}, Сила тренда: {features['trend_strength']:.2f}")

      return characteristics

    except Exception as e:
      logger.error(f"Ошибка определения режима для {symbol}: {e}")
      return self._get_default_regime()

  def _extract_regime_features(self, data: pd.DataFrame) -> Dict[str, Any]:
    """Извлекает признаки для определения режима"""
    features = {}

    try:
      closes = data['close']
      highs = data['high']
      lows = data['low']
      volumes = data['volume']

      # 1. Тренд
      sma_20 = closes.rolling(20).mean()
      sma_50 = closes.rolling(50).mean()
      sma_200 = closes.rolling(200).mean() if len(closes) >= 200 else sma_50

      # Сила тренда (-1 to 1)
      if len(closes) >= 50:
        price_position = (closes.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
        trend_alignment = 0
        if closes.iloc[-1] > sma_20.iloc[-1] > sma_50.iloc[-1]:
          trend_alignment = 1
        elif closes.iloc[-1] < sma_20.iloc[-1] < sma_50.iloc[-1]:
          trend_alignment = -1

        features['trend_strength'] = np.clip(price_position * 10 + trend_alignment * 0.3, -1, 1)
      else:
        features['trend_strength'] = 0

      # 2. Волатильность
      returns = closes.pct_change()
      current_volatility = returns.rolling(20).std().iloc[-1]
      historical_volatility = returns.rolling(100).std().mean() if len(returns) >= 100 else current_volatility

      features['volatility_level'] = min(current_volatility / (historical_volatility + 1e-9), 2.0) / 2.0

      # 3. Моментум
      rsi = ta.rsi(closes, length=14)
      macd = ta.macd(closes)

      if rsi is not None and not rsi.empty:
        rsi_score = (rsi.iloc[-1] - 50) / 50
      else:
        rsi_score = 0

      if macd is not None and len(macd) > 0:
        macd_hist = macd.iloc[-1, 2] if len(macd.columns) > 2 else 0
        macd_score = np.sign(macd_hist) * min(abs(macd_hist) / closes.iloc[-1] * 100, 1)
      else:
        macd_score = 0

      features['momentum_score'] = (rsi_score + macd_score) / 2

      # 4. Объем
      volume_ma = volumes.rolling(20).mean()
      current_volume_ratio = volumes.iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1

      if current_volume_ratio > 2:
        features['volume_profile'] = 'spike'
      elif current_volume_ratio > 1.2:
        features['volume_profile'] = 'increasing'
      elif current_volume_ratio < 0.8:
        features['volume_profile'] = 'decreasing'
      else:
        features['volume_profile'] = 'stable'

      # 5. ADX для силы тренда
      adx = ta.adx(highs, lows, closes, length=14)
      if adx is not None and not adx.empty:
        features['adx_value'] = adx.iloc[-1, 0]  # ADX value
      else:
        features['adx_value'] = 25

      # 6. Bollinger Bands для определения сжатия
      bb = ta.bbands(closes, length=20, std=2)
      if bb is not None and not bb.empty:
        bb_width = (bb.iloc[-1, 2] - bb.iloc[-1, 0]) / closes.iloc[-1]
        features['bb_squeeze'] = bb_width < 0.02  # Сжатие если ширина < 2%
      else:
        features['bb_squeeze'] = False

      # 7. Поддерживающие индикаторы
      supporting = []
      if features['trend_strength'] > 0.3:
        supporting.append('Uptrend SMA')
      if features['momentum_score'] > 0.3:
        supporting.append('Positive Momentum')
      if features['adx_value'] > 25:
        supporting.append(f"Strong Trend (ADX={features['adx_value']:.0f})")

      features['supporting_indicators'] = supporting

    except Exception as e:
      logger.error(f"Ошибка извлечения признаков режима: {e}")
      features = self._get_default_features()

    return features

  def _determine_primary_regime(self, features: Dict[str, Any], data: pd.DataFrame) -> MarketRegime:
    """Определяет основной режим рынка"""
    trend_strength = features['trend_strength']
    volatility = features['volatility_level']
    momentum = features['momentum_score']
    adx = features.get('adx_value', 25)

    # Проверка на сжатие
    if features.get('bb_squeeze', False) and volatility < 0.3:
      return MarketRegime.SQUEEZE

    # Проверка на высокую волатильность
    if volatility > 1.5:
      return MarketRegime.VOLATILE

    # Проверка на низкую активность
    if volatility < 0.2 and abs(trend_strength) < 0.1:
      return MarketRegime.QUIET

    # Проверка на пробой
    if features.get('volume_profile') == 'spike' and abs(momentum) > 0.7:
      return MarketRegime.BREAKOUT

    # Определение трендовых режимов
    if adx > 40:  # Сильный тренд
      if trend_strength > 0.5:
        return MarketRegime.STRONG_TREND_UP
      elif trend_strength < -0.5:
        return MarketRegime.STRONG_TREND_DOWN
    elif adx > 25:  # Умеренный тренд
      if trend_strength > 0.2:
        return MarketRegime.TREND_UP
      elif trend_strength < -0.2:
        return MarketRegime.TREND_DOWN
    elif adx > 20:  # Слабый тренд
      if trend_strength > 0:
        return MarketRegime.WEAK_TREND_UP
      elif trend_strength < 0:
        return MarketRegime.WEAK_TREND_DOWN

    # По умолчанию - боковик
    return MarketRegime.RANGING

  def _determine_secondary_regime(self, features: Dict[str, Any],
                                  primary: MarketRegime) -> Optional[MarketRegime]:
    """Определяет вторичный режим"""
    # Проверка на потенциальный разворот
    if primary in [MarketRegime.TREND_UP, MarketRegime.STRONG_TREND_UP]:
      if features['momentum_score'] < -0.3:
        return MarketRegime.REVERSAL
    elif primary in [MarketRegime.TREND_DOWN, MarketRegime.STRONG_TREND_DOWN]:
      if features['momentum_score'] > 0.3:
        return MarketRegime.REVERSAL

    # Проверка на формирование сжатия
    if features.get('bb_squeeze', False) and primary == MarketRegime.RANGING:
      return MarketRegime.SQUEEZE

    return None

  def _calculate_regime_confidence(self, features: Dict[str, Any], regime: MarketRegime) -> float:
    """Рассчитывает уверенность в определении режима"""
    confidence = 0.5

    # Факторы, повышающие уверенность
    if regime in [MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN]:
      if features.get('adx_value', 0) > 40:
        confidence += 0.2
      if abs(features['trend_strength']) > 0.7:
        confidence += 0.2
      if features['volume_profile'] in ['increasing', 'spike']:
        confidence += 0.1

    elif regime == MarketRegime.RANGING:
      if features.get('adx_value', 25) < 20:
        confidence += 0.3
      if abs(features['trend_strength']) < 0.2:
        confidence += 0.2

    elif regime == MarketRegime.VOLATILE:
      if features['volatility_level'] > 1.5:
        confidence += 0.3
      if features['volume_profile'] == 'spike':
        confidence += 0.2

    return min(confidence, 1.0)

  def _calculate_transition_probabilities(self, symbol: str,
                                          current_regime: MarketRegime) -> Dict[MarketRegime, float]:
    """Рассчитывает вероятности перехода в другие режимы"""
    # Базовые вероятности перехода
    base_transitions = {
      MarketRegime.STRONG_TREND_UP: {
        MarketRegime.TREND_UP: 0.3,
        MarketRegime.RANGING: 0.1,
        MarketRegime.REVERSAL: 0.1
      },
      MarketRegime.TREND_UP: {
        MarketRegime.STRONG_TREND_UP: 0.2,
        MarketRegime.WEAK_TREND_UP: 0.3,
        MarketRegime.RANGING: 0.2,
        MarketRegime.REVERSAL: 0.1
      },
      MarketRegime.RANGING: {
        MarketRegime.TREND_UP: 0.2,
        MarketRegime.TREND_DOWN: 0.2,
        MarketRegime.BREAKOUT: 0.1,
        MarketRegime.SQUEEZE: 0.1
      },
      MarketRegime.VOLATILE: {
        MarketRegime.RANGING: 0.3,
        MarketRegime.TREND_UP: 0.15,
        MarketRegime.TREND_DOWN: 0.15,
        MarketRegime.QUIET: 0.1
      }
    }

    # Получаем базовые вероятности или создаем равномерные
    if current_regime in base_transitions:
      transitions = base_transitions[current_regime].copy()
    else:
      # Равномерное распределение
      all_regimes = list(MarketRegime)
      transitions = {r: 1.0 / len(all_regimes) for r in all_regimes}

    # Нормализуем вероятности
    total = sum(transitions.values())
    if total > 0:
      transitions = {k: v / total for k, v in transitions.items()}

    return transitions

  def _identify_key_levels(self, data: pd.DataFrame) -> Dict[str, float]:
    """Определяет ключевые уровни поддержки и сопротивления"""
    try:
      highs = data['high']
      lows = data['low']
      closes = data['close']

      # Простой метод: локальные экстремумы
      window = 10

      # Находим локальные максимумы и минимумы
      local_highs = highs.rolling(window, center=True).max() == highs
      local_lows = lows.rolling(window, center=True).min() == lows

      resistance_levels = highs[local_highs].dropna().unique()
      support_levels = lows[local_lows].dropna().unique()

      current_price = closes.iloc[-1]

      # Ближайшие уровни
      nearest_resistance = min([r for r in resistance_levels if r > current_price],
                               default=current_price * 1.05)
      nearest_support = max([s for s in support_levels if s < current_price],
                            default=current_price * 0.95)

      # Pivot point
      last_candle = data.iloc[-1]
      pivot = (last_candle['high'] + last_candle['low'] + last_candle['close']) / 3

      return {
        'resistance': nearest_resistance,
        'support': nearest_support,
        'pivot': pivot,
        'current': current_price
      }

    except Exception as e:
      logger.error(f"Ошибка определения ключевых уровней: {e}")
      return {'resistance': 0, 'support': 0, 'pivot': 0, 'current': 0}

  def _calculate_regime_timing(self, symbol: str,
                                 current_regime: MarketRegime) -> Tuple[timedelta, timedelta]:
      """Рассчитывает временные характеристики режима"""
      if symbol not in self.regime_history:
        return timedelta(0), timedelta(0)

      history = list(self.regime_history[symbol])
      if not history:
        return timedelta(0), timedelta(0)

      # Находим последнее изменение режима
      regime_start = datetime.now()
      time_since_change = timedelta(0)

      for i in range(len(history) - 1, -1, -1):
        if history[i].primary_regime != current_regime:
          if i < len(history) - 1:
            regime_start = history[i + 1].timestamp
            time_since_change = datetime.now() - regime_start
          break

      # Рассчитываем среднюю продолжительность режима
      regime_durations = []
      current_start = None

      for entry in history:
        if entry.primary_regime == current_regime:
          if current_start is None:
            current_start = entry.timestamp
        else:
          if current_start is not None:
            duration = entry.timestamp - current_start
            regime_durations.append(duration)
            current_start = None

      if regime_durations:
        avg_duration = sum(regime_durations, timedelta(0)) / len(regime_durations)
      else:
        avg_duration = timedelta(hours=4)  # По умолчанию 4 часа

      return avg_duration, time_since_change

  def _estimate_regime_duration(self, regime: MarketRegime) -> timedelta:
    """Оценивает ожидаемую продолжительность режима"""
    # Базовые оценки на основе исторических данных
    expected_durations = {
      MarketRegime.STRONG_TREND_UP: timedelta(hours=12),
      MarketRegime.STRONG_TREND_DOWN: timedelta(hours=12),
      MarketRegime.TREND_UP: timedelta(hours=8),
      MarketRegime.TREND_DOWN: timedelta(hours=8),
      MarketRegime.WEAK_TREND_UP: timedelta(hours=4),
      MarketRegime.WEAK_TREND_DOWN: timedelta(hours=4),
      MarketRegime.RANGING: timedelta(hours=6),
      MarketRegime.VOLATILE: timedelta(hours=3),
      MarketRegime.QUIET: timedelta(hours=5),
      MarketRegime.BREAKOUT: timedelta(hours=2),
      MarketRegime.REVERSAL: timedelta(hours=2),
      MarketRegime.SQUEEZE: timedelta(hours=4)
    }

    return expected_durations.get(regime, timedelta(hours=4))

  def _update_regime_history(self, symbol: str, characteristics: RegimeCharacteristics):
    """Обновляет историю режимов"""
    if symbol not in self.regime_history:
      self.regime_history[symbol] = deque(maxlen=1000)
      self.regime_transitions[symbol] = []

    history = self.regime_history[symbol]

    # Проверяем изменение режима
    if history and history[-1].primary_regime != characteristics.primary_regime:
      # Записываем переход
      transition = (
        history[-1].primary_regime,
        characteristics.primary_regime,
        characteristics.timestamp
      )
      self.regime_transitions[symbol].append(transition)

      # Ограничиваем историю переходов
      if len(self.regime_transitions[symbol]) > 100:
        self.regime_transitions[symbol] = self.regime_transitions[symbol][-100:]

    # Добавляем в историю
    history.append(characteristics)

  def _is_regime_change(self, symbol: str, new_regime: MarketRegime) -> bool:
    """Проверяет, изменился ли режим"""
    if symbol not in self.regime_history or len(self.regime_history[symbol]) == 0:
      return True

    last_regime = self.regime_history[symbol][-1].primary_regime
    return last_regime != new_regime

  def _get_default_regime(self) -> RegimeCharacteristics:
    """Возвращает характеристики режима по умолчанию"""
    return RegimeCharacteristics(
      primary_regime=MarketRegime.RANGING,
      secondary_regime=None,
      confidence=0.5,
      trend_strength=0.0,
      volatility_level=0.5,
      momentum_score=0.0,
      volume_profile='stable',
      regime_duration=timedelta(hours=4),
      time_since_change=timedelta(0),
      expected_duration=timedelta(hours=4),
      transition_probabilities={},
      supporting_indicators=[],
      key_levels={'support': 0, 'resistance': 0, 'pivot': 0, 'current': 0}
    )

  def _get_default_features(self) -> Dict[str, Any]:
    """Возвращает признаки по умолчанию"""
    return {
      'trend_strength': 0.0,
      'volatility_level': 0.5,
      'momentum_score': 0.0,
      'volume_profile': 'stable',
      'adx_value': 25,
      'bb_squeeze': False,
      'supporting_indicators': []
    }

  def _rolling_beta(self, returns1: pd.Series, returns2: pd.Series,
                    window: int = 50) -> pd.Series:
    """Рассчитывает скользящую бету между двумя рядами доходностей"""

    def calc_beta(window_data):
      if len(window_data) < 2:
        return np.nan
      cov = np.cov(window_data.iloc[:, 0], window_data.iloc[:, 1])[0, 1]
      var = np.var(window_data.iloc[:, 1])
      return cov / var if var > 0 else 0

    combined = pd.concat([returns1, returns2], axis=1)
    return combined.rolling(window).apply(calc_beta, raw=False)

  def get_regime_parameters(self, symbol: str) -> RegimeParameters:
    """Получает оптимальные параметры для текущего режима"""
    if symbol in self.current_regimes:
      regime = self.current_regimes[symbol].primary_regime
      return self.regime_parameters.get(regime, self.regime_parameters[MarketRegime.RANGING])

    return self.regime_parameters[MarketRegime.RANGING]

  async def analyze_multi_timeframe_regime(self, symbol: str,
                                           timeframes: List[Timeframe]) -> Dict[Timeframe, RegimeCharacteristics]:
    """Анализирует режимы на нескольких таймфреймах"""
    results = {}

    for timeframe in timeframes:
      try:
        # Получаем данные для таймфрейма
        data = await self.data_fetcher.get_historical_candles(
          symbol, timeframe, limit=500
        )

        if not data.empty:
          # Определяем режим
          regime = await self.detect_regime(symbol, data)
          results[timeframe] = regime

      except Exception as e:
        logger.error(f"Ошибка анализа режима для {symbol} на {timeframe.value}: {e}")

    return results

  def get_regime_strength_score(self, characteristics: RegimeCharacteristics) -> float:
    """Рассчитывает силу текущего режима (0-1)"""
    # Факторы, влияющие на силу режима
    confidence_weight = 0.3
    trend_weight = 0.3
    momentum_weight = 0.2
    duration_weight = 0.2

    # Нормализуем продолжительность (чем дольше режим, тем он сильнее)
    duration_hours = characteristics.regime_duration.total_seconds() / 3600
    duration_score = min(duration_hours / 24, 1.0)  # Максимум 1 для режима > 24 часов

    # Рассчитываем общий счет
    strength_score = (
        characteristics.confidence * confidence_weight +
        abs(characteristics.trend_strength) * trend_weight +
        abs(characteristics.momentum_score) * momentum_weight +
        duration_score * duration_weight
    )

    return min(max(strength_score, 0.0), 1.0)

  def should_adapt_strategy(self, symbol: str,
                            previous_regime: Optional[MarketRegime] = None) -> Tuple[bool, str]:
    """Определяет, нужно ли адаптировать стратегию"""
    if symbol not in self.current_regimes:
      return False, "Нет данных о режиме"

    current = self.current_regimes[symbol]

    # Проверка 1: Изменился ли режим
    if previous_regime and current.primary_regime != previous_regime:
      return True, f"Изменение режима: {previous_regime.value} -> {current.primary_regime.value}"

    # Проверка 2: Низкая уверенность в режиме
    if current.confidence < 0.4:
      return True, f"Низкая уверенность в режиме: {current.confidence:.2f}"

    # Проверка 3: Вторичный режим указывает на возможные изменения
    if current.secondary_regime in [MarketRegime.REVERSAL, MarketRegime.BREAKOUT]:
      return True, f"Обнаружен вторичный режим: {current.secondary_regime.value}"

    # Проверка 4: Высокая вероятность перехода
    if current.transition_probabilities:
      max_transition_prob = max(current.transition_probabilities.values())
      if max_transition_prob > 0.7:
        likely_regime = [k for k, v in current.transition_probabilities.items()
                         if v == max_transition_prob][0]
        return True, f"Высокая вероятность перехода в {likely_regime.value}: {max_transition_prob:.2f}"

    return False, "Адаптация не требуется"

  def get_regime_statistics(self, symbol: str) -> Dict[str, Any]:
    """Получает статистику по режимам для символа"""
    if symbol not in self.regime_history:
      return {"error": "Нет истории для символа"}

    history = list(self.regime_history[symbol])
    if not history:
      return {"error": "История пуста"}

    # Подсчет режимов
    regime_counts = {}
    total_duration = {}

    for entry in history:
      regime = entry.primary_regime
      regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1

      if regime.value not in total_duration:
        total_duration[regime.value] = timedelta(0)
      total_duration[regime.value] += entry.regime_duration

    # Переходы
    transition_matrix = {}
    if symbol in self.regime_transitions:
      for from_regime, to_regime, _ in self.regime_transitions[symbol]:
        key = f"{from_regime.value} -> {to_regime.value}"
        transition_matrix[key] = transition_matrix.get(key, 0) + 1

    # Средние характеристики
    avg_confidence = np.mean([e.confidence for e in history])
    avg_trend_strength = np.mean([e.trend_strength for e in history])
    avg_volatility = np.mean([e.volatility_level for e in history])

    return {
      "total_observations": len(history),
      "regime_distribution": regime_counts,
      "average_durations": {
        regime: str(duration / count) if count > 0 else "0:00:00"
        for regime, duration in total_duration.items()
        for count in [regime_counts.get(regime, 1)]
      },
      "transition_matrix": transition_matrix,
      "average_metrics": {
        "confidence": round(avg_confidence, 3),
        "trend_strength": round(avg_trend_strength, 3),
        "volatility_level": round(avg_volatility, 3)
      },
      "current_regime": self.current_regimes[symbol].primary_regime.value if symbol in self.current_regimes else None
    }

  def export_regime_data(self, symbol: str, filepath: str):
    """Экспортирует данные о режимах в файл"""
    try:
      if symbol not in self.regime_history:
        logger.error(f"Нет данных для экспорта по символу {symbol}")
        return

      data = []
      for entry in self.regime_history[symbol]:
        data.append({
          'timestamp': entry.timestamp.isoformat(),
          'primary_regime': entry.primary_regime.value,
          'secondary_regime': entry.secondary_regime.value if entry.secondary_regime else None,
          'confidence': entry.confidence,
          'trend_strength': entry.trend_strength,
          'volatility_level': entry.volatility_level,
          'momentum_score': entry.momentum_score,
          'volume_profile': entry.volume_profile,
          'regime_duration_minutes': entry.regime_duration.total_seconds() / 60,
          'key_support': entry.key_levels.get('support', 0),
          'key_resistance': entry.key_levels.get('resistance', 0)
        })

      df = pd.DataFrame(data)
      df.to_csv(filepath, index=False)
      logger.info(f"Данные о режимах экспортированы в {filepath}")

    except Exception as e:
      logger.error(f"Ошибка экспорта данных о режимах: {e}")