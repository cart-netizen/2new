import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import asyncio

from core.schemas import TradingSignal
from core.enums import SignalType, Timeframe
from core.data_fetcher import DataFetcher
from data.database_manager import AdvancedDatabaseManager
from utils.logging_config import get_logger

logger = get_logger(__name__)


class QualityScore(Enum):
    """Категории качества сигнала"""
    EXCELLENT = "excellent"  # 0.8-1.0
    GOOD = "good"           # 0.6-0.8
    FAIR = "fair"           # 0.4-0.6
    POOR = "poor"           # 0.2-0.4
    UNACCEPTABLE = "unacceptable"  # 0-0.2


@dataclass
class SignalQualityMetrics:
  """Метрики качества сигнала"""
  # Основные метрики
  overall_score: float  # 0.0-1.0
  quality_category: QualityScore

  # Детальные оценки компонентов
  timeframe_alignment_score: float  # Согласованность на разных таймфреймах
  momentum_strength_score: float  # Сила моментума
  market_structure_score: float  # Качество рыночной структуры
  historical_performance_score: float  # Историческая успешность
  volume_confirmation_score: float  # Подтверждение объемом
  volatility_fitness_score: float  # Соответствие волатильности
  trend_alignment_score: float  # Соответствие тренду

  # Дополнительная информация
  confidence_level: float
  risk_reward_ratio: float
  expected_win_rate: float
  signal_strength_percentile: float  # Процентиль силы среди всех сигналов

  # Детали анализа
  strengths: List[str] = field(default_factory=list)
  weaknesses: List[str] = field(default_factory=list)
  recommendations: List[str] = field(default_factory=list)
  timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HistoricalSignalPerformance:
  """Историческая производительность похожих сигналов"""
  total_signals: int
  winning_signals: int
  losing_signals: int
  win_rate: float
  avg_profit: float
  avg_loss: float
  profit_factor: float
  max_consecutive_wins: int
  max_consecutive_losses: int
  avg_holding_time: timedelta


class SignalQualityAnalyzer:
  """
  Комплексная система оценки качества торговых сигналов
  """

  def __init__(self, data_fetcher: DataFetcher, db_manager: AdvancedDatabaseManager):
    self.data_fetcher = data_fetcher
    self.db_manager = db_manager

    # История сигналов для анализа
    self.signal_history = deque(maxlen=1000)
    self.performance_cache = {}

    # Параметры оценки
    self.quality_weights = {
      'timeframe_alignment': 0.20,
      'momentum_strength': 0.15,
      'market_structure': 0.15,
      'historical_performance': 0.20,
      'volume_confirmation': 0.10,
      'volatility_fitness': 0.10,
      'trend_alignment': 0.10
    }

    # Пороги качества
    self.quality_thresholds = {
      QualityScore.EXCELLENT: 0.8,
      QualityScore.GOOD: 0.6,
      QualityScore.FAIR: 0.4,
      QualityScore.POOR: 0.2,
      QualityScore.UNACCEPTABLE: 0.0
    }

    # Статистика для нормализации
    self.signal_statistics = {
      'momentum_values': deque(maxlen=500),
      'volume_ratios': deque(maxlen=500),
      'volatility_values': deque(maxlen=500)
    }

  async def rate_signal_quality(self, signal: TradingSignal,
                                market_data: pd.DataFrame,
                                additional_timeframes: Optional[
                                  Dict[Timeframe, pd.DataFrame]] = None) -> SignalQualityMetrics:
    """
    Оценивает качество торгового сигнала от 0 до 1
    """
    logger.debug(f"Оценка качества сигнала для {signal.symbol}")

    try:
      # 1. Проверка согласованности на разных таймфреймах
      timeframe_score = await self._check_multi_timeframe_alignment(
        signal, market_data, additional_timeframes
      )

      # 2. Оценка силы моментума
      momentum_score = self._calculate_momentum_strength(signal, market_data)

      # 3. Анализ структуры рынка
      structure_score = self._analyze_market_structure(signal, market_data)

      # 4. Историческая производительность похожих сигналов
      historical_score = await self._check_historical_performance(signal, market_data)

      # 5. Подтверждение объемом
      volume_score = self._check_volume_confirmation(signal, market_data)

      # 6. Соответствие текущей волатильности
      volatility_score = self._check_volatility_fitness(signal, market_data)

      # 7. Соответствие общему тренду
      trend_score = self._check_trend_alignment(signal, market_data)

      # Вычисляем общий балл
      scores = {
        'timeframe_alignment': timeframe_score,
        'momentum_strength': momentum_score,
        'market_structure': structure_score,
        'historical_performance': historical_score,
        'volume_confirmation': volume_score,
        'volatility_fitness': volatility_score,
        'trend_alignment': trend_score
      }

      overall_score = sum(
        score * self.quality_weights[component]
        for component, score in scores.items()
      )

      # Определяем категорию качества
      quality_category = self._determine_quality_category(overall_score)

      # Анализ сильных и слабых сторон
      strengths, weaknesses = self._analyze_strengths_weaknesses(scores)

      # Рекомендации
      recommendations = self._generate_recommendations(scores, signal)

      # Расчет дополнительных метрик
      risk_reward_ratio = self._calculate_risk_reward_ratio(signal, market_data)
      expected_win_rate = historical_score * 0.7 + structure_score * 0.3

      # Процентиль силы сигнала
      signal_strength_percentile = self._calculate_signal_percentile(overall_score)

      # Сохраняем в историю
      self._save_to_history(signal, overall_score)

      return SignalQualityMetrics(
        overall_score=overall_score,
        quality_category=quality_category,
        timeframe_alignment_score=timeframe_score,
        momentum_strength_score=momentum_score,
        market_structure_score=structure_score,
        historical_performance_score=historical_score,
        volume_confirmation_score=volume_score,
        volatility_fitness_score=volatility_score,
        trend_alignment_score=trend_score,
        confidence_level=signal.confidence * overall_score,
        risk_reward_ratio=risk_reward_ratio,
        expected_win_rate=expected_win_rate,
        signal_strength_percentile=signal_strength_percentile,
        strengths=strengths,
        weaknesses=weaknesses,
        recommendations=recommendations
      )

    except Exception as e:
      logger.error(f"Ошибка при оценке качества сигнала: {e}")
      # Возвращаем базовую оценку
      return self._create_default_metrics(signal)

  async def _check_multi_timeframe_alignment(self, signal: TradingSignal,
                                             market_data: pd.DataFrame,
                                             additional_timeframes: Optional[Dict[Timeframe, pd.DataFrame]]) -> float:
    """Проверяет согласованность сигнала на разных таймфреймах"""
    if not additional_timeframes:
      # Загружаем дополнительные таймфреймы
      additional_timeframes = {}
      timeframes_to_check = [Timeframe.FIFTEEN_MINUTES, Timeframe.FOUR_HOURS, Timeframe.ONE_DAY]

      for tf in timeframes_to_check:
        try:
          data = await self.data_fetcher.get_historical_candles(
            signal.symbol, tf, limit=100
          )
          if not data.empty:
            additional_timeframes[tf] = data
        except Exception as e:
          logger.warning(f"Не удалось загрузить данные {tf} для {signal.symbol}: {e}")

    alignment_scores = []

    for timeframe, data in additional_timeframes.items():
      if data.empty:
        continue

      # Проверяем направление тренда
      sma_20 = data['close'].rolling(20).mean()
      sma_50 = data['close'].rolling(50).mean()

      if len(data) < 50:
        continue

      current_price = data['close'].iloc[-1]

      # Оценка соответствия
      if signal.signal_type == SignalType.BUY:
        if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
          alignment_scores.append(1.0)  # Полное соответствие
        elif current_price > sma_20.iloc[-1]:
          alignment_scores.append(0.7)  # Частичное соответствие
        elif current_price > sma_50.iloc[-1]:
          alignment_scores.append(0.5)
        else:
          alignment_scores.append(0.2)  # Противоречие
      else:  # SELL
        if current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
          alignment_scores.append(1.0)
        elif current_price < sma_20.iloc[-1]:
          alignment_scores.append(0.7)
        elif current_price < sma_50.iloc[-1]:
          alignment_scores.append(0.5)
        else:
          alignment_scores.append(0.2)

    return np.mean(alignment_scores) if alignment_scores else 0.5

  def _calculate_momentum_strength(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
    """Оценивает силу моментума"""
    try:
      # RSI
      rsi = market_data.get('rsi_14', pd.Series())
      if rsi.empty:
        import pandas_ta as ta
        rsi = ta.rsi(market_data['close'], length=14)

      current_rsi = rsi.iloc[-1] if not rsi.empty else 50

      # MACD
      macd_result = None
      if 'macd' not in market_data.columns:
        import pandas_ta as ta
        macd_result = ta.macd(market_data['close'])

      # Rate of Change
      roc = market_data['close'].pct_change(10).iloc[-1] * 100

      # Оценка для BUY сигналов
      if signal.signal_type == SignalType.BUY:
        rsi_score = min((current_rsi - 30) / 40, 1.0) if current_rsi > 30 else 0
        roc_score = min(roc / 5, 1.0) if roc > 0 else 0
      else:  # SELL
        rsi_score = min((70 - current_rsi) / 40, 1.0) if current_rsi < 70 else 0
        roc_score = min(-roc / 5, 1.0) if roc < 0 else 0

      # MACD гистограмма
      macd_score = 0.5
      if macd_result is not None and len(macd_result) > 0:
        macd_hist = macd_result.iloc[-1, 2] if len(macd_result.columns) > 2 else 0
        if signal.signal_type == SignalType.BUY:
          macd_score = 1.0 if macd_hist > 0 else 0.3
        else:
          macd_score = 1.0 if macd_hist < 0 else 0.3

      # Комбинированная оценка
      momentum_score = (rsi_score * 0.4 + roc_score * 0.3 + macd_score * 0.3)

      # Сохраняем для статистики
      self.signal_statistics['momentum_values'].append(abs(roc))

      return momentum_score

    except Exception as e:
      logger.error(f"Ошибка расчета моментума: {e}")
      return 0.5

  def _analyze_market_structure(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
    """Анализирует структуру рынка"""
    try:
      closes = market_data['close']
      highs = market_data['high']
      lows = market_data['low']

      # 1. Поиск уровней поддержки/сопротивления
      support_resistance_score = self._check_support_resistance_levels(
        signal, closes, highs, lows
      )

      # 2. Анализ свинг-точек
      swing_score = self._analyze_swing_points(highs, lows, signal.signal_type)

      # 3. Проверка паттернов Price Action
      pattern_score = self._check_price_patterns(market_data, signal.signal_type)

      # 4. Анализ структуры тренда (HH, HL, LH, LL)
      trend_structure_score = self._analyze_trend_structure(highs, lows)

      # Комбинированная оценка
      structure_score = (
          support_resistance_score * 0.3 +
          swing_score * 0.2 +
          pattern_score * 0.2 +
          trend_structure_score * 0.3
      )

      return structure_score

    except Exception as e:
      logger.error(f"Ошибка анализа структуры рынка: {e}")
      return 0.5

  def _check_support_resistance_levels(self, signal: TradingSignal,
                                       closes: pd.Series,
                                       highs: pd.Series,
                                       lows: pd.Series) -> float:
    """Проверяет близость к уровням поддержки/сопротивления"""
    current_price = signal.price

    # Находим локальные экстремумы
    window = 20
    local_highs = highs.rolling(window, center=True).max() == highs
    local_lows = lows.rolling(window, center=True).min() == lows

    resistance_levels = highs[local_highs].dropna().unique()
    support_levels = lows[local_lows].dropna().unique()

    # Проверяем близость к уровням
    price_range = highs.max() - lows.min()
    proximity_threshold = price_range * 0.02  # 2% от диапазона

    if signal.signal_type == SignalType.BUY:
      # Для покупки хорошо, если мы отскочили от поддержки
      distances_to_support = [abs(current_price - level) for level in support_levels]
      if distances_to_support:
        min_distance = min(distances_to_support)
        if min_distance < proximity_threshold:
          return 0.9  # Отличная точка входа
        elif min_distance < proximity_threshold * 2:
          return 0.7
      return 0.5
    else:  # SELL
      # Для продажи хорошо, если мы у сопротивления
      distances_to_resistance = [abs(current_price - level) for level in resistance_levels]
      if distances_to_resistance:
        min_distance = min(distances_to_resistance)
        if min_distance < proximity_threshold:
          return 0.9
        elif min_distance < proximity_threshold * 2:
          return 0.7
      return 0.5

  def _analyze_swing_points(self, highs: pd.Series, lows: pd.Series, signal_type: SignalType) -> float:
    """Анализирует свинг-точки (разворотные точки)"""
    # Определяем свинги
    swing_highs = []
    swing_lows = []

    for i in range(2, len(highs) - 2):
      # Swing high: H[i] > H[i-1] и H[i] > H[i-2] и H[i] > H[i+1] и H[i] > H[i+2]
      if (highs.iloc[i] > highs.iloc[i - 1] and highs.iloc[i] > highs.iloc[i - 2] and
          highs.iloc[i] > highs.iloc[i + 1] and highs.iloc[i] > highs.iloc[i + 2]):
        swing_highs.append(i)

      # Swing low
      if (lows.iloc[i] < lows.iloc[i - 1] and lows.iloc[i] < lows.iloc[i - 2] and
          lows.iloc[i] < lows.iloc[i + 1] and lows.iloc[i] < lows.iloc[i + 2]):
        swing_lows.append(i)

    if not swing_highs or not swing_lows:
      return 0.5

    # Анализируем последовательность свингов
    last_swing_high_idx = swing_highs[-1] if swing_highs else 0
    last_swing_low_idx = swing_lows[-1] if swing_lows else 0

    if signal_type == SignalType.BUY:
      # Для покупки хорошо, если последний свинг - это low
      if last_swing_low_idx > last_swing_high_idx:
        return 0.8
      else:
        return 0.4
    else:  # SELL
      # Для продажи хорошо, если последний свинг - это high
      if last_swing_high_idx > last_swing_low_idx:
        return 0.8
      else:
        return 0.4

  def _check_price_patterns(self, market_data: pd.DataFrame, signal_type: SignalType) -> float:
    """Проверяет наличие классических паттернов Price Action"""
    try:
      # Упрощенная проверка нескольких паттернов
      pattern_scores = []

      # 1. Проверка пин-бара
      pin_bar_score = self._check_pin_bar(market_data, signal_type)
      if pin_bar_score > 0:
        pattern_scores.append(pin_bar_score)

      # 2. Проверка поглощения
      engulfing_score = self._check_engulfing(market_data, signal_type)
      if engulfing_score > 0:
        pattern_scores.append(engulfing_score)

      # 3. Проверка доджи
      doji_score = self._check_doji(market_data)
      if doji_score > 0:
        pattern_scores.append(doji_score * 0.7)  # Доджи менее надежен

      return max(pattern_scores) if pattern_scores else 0.5

    except Exception as e:
      logger.error(f"Ошибка проверки паттернов: {e}")
      return 0.5

  def _check_pin_bar(self, data: pd.DataFrame, signal_type: SignalType) -> float:
    """Проверка паттерна пин-бар"""
    last_candle = data.iloc[-1]
    body = abs(last_candle['close'] - last_candle['open'])
    full_range = last_candle['high'] - last_candle['low']

    if full_range == 0:
      return 0

    body_ratio = body / full_range

    if signal_type == SignalType.BUY:
      # Бычий пин-бар: длинная нижняя тень
      lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']
      upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])

      if body_ratio < 0.3 and lower_shadow > body * 2 and upper_shadow < body:
        return 0.9
    else:  # SELL
      # Медвежий пин-бар: длинная верхняя тень
      lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']
      upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])

      if body_ratio < 0.3 and upper_shadow > body * 2 and lower_shadow < body:
        return 0.9

    return 0

  def _check_engulfing(self, data: pd.DataFrame, signal_type: SignalType) -> float:
    """Проверка паттерна поглощения"""
    if len(data) < 2:
      return 0

    current = data.iloc[-1]
    previous = data.iloc[-2]

    if signal_type == SignalType.BUY:
      # Бычье поглощение
      if (previous['close'] < previous['open'] and  # Предыдущая медвежья
          current['close'] > current['open'] and  # Текущая бычья
          current['open'] <= previous['close'] and  # Открытие ниже закрытия предыдущей
          current['close'] >= previous['open']):  # Закрытие выше открытия предыдущей
        return 0.85
    else:  # SELL
      # Медвежье поглощение
      if (previous['close'] > previous['open'] and  # Предыдущая бычья
          current['close'] < current['open'] and  # Текущая медвежья
          current['open'] >= previous['close'] and  # Открытие выше закрытия предыдущей
          current['close'] <= previous['open']):  # Закрытие ниже открытия предыдущей
        return 0.85

    return 0

  def _check_doji(self, data: pd.DataFrame) -> float:
    """Проверка паттерна доджи"""
    last_candle = data.iloc[-1]
    body = abs(last_candle['close'] - last_candle['open'])
    full_range = last_candle['high'] - last_candle['low']

    if full_range == 0:
      return 0

    body_ratio = body / full_range

    # Доджи - очень маленькое тело
    if body_ratio < 0.1:
      return 0.7  # Доджи указывает на нерешительность

    return 0

  def _analyze_trend_structure(self, highs: pd.Series, lows: pd.Series) -> float:
    """Анализирует структуру тренда (HH, HL, LH, LL)"""
    if len(highs) < 20:
      return 0.5

    # Находим последние несколько экстремумов
    recent_highs = []
    recent_lows = []

    for i in range(-20, -1):
      if i - 1 >= -len(highs) and i + 1 < 0:
        if highs.iloc[i] > highs.iloc[i - 1] and highs.iloc[i] > highs.iloc[i + 1]:
          recent_highs.append(highs.iloc[i])
        if lows.iloc[i] < lows.iloc[i - 1] and lows.iloc[i] < lows.iloc[i + 1]:
          recent_lows.append(lows.iloc[i])

    if len(recent_highs) < 2 or len(recent_lows) < 2:
      return 0.5

    # Анализ последовательности
    last_two_highs = recent_highs[-2:]
    last_two_lows = recent_lows[-2:]

    # Восходящий тренд: HH и HL
    if last_two_highs[1] > last_two_highs[0] and last_two_lows[1] > last_two_lows[0]:
      return 0.8
    # Нисходящий тренд: LH и LL
    elif last_two_highs[1] < last_two_highs[0] and last_two_lows[1] < last_two_lows[0]:
      return 0.8
    # Боковик или переход
    else:
      return 0.5

  async def _check_historical_performance(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
    """Проверяет историческую производительность похожих сигналов"""
    try:
      # Получаем похожие исторические сигналы
      similar_signals = await self._get_similar_historical_signals(signal, market_data)

      if not similar_signals or similar_signals.total_signals < 10:
        return 0.5  # Недостаточно данных

      # Оцениваем на основе win rate и profit factor
      win_rate_score = min(similar_signals.win_rate / 0.6, 1.0)  # 60% win rate = максимум

      profit_factor_score = 0.5
      if similar_signals.profit_factor > 2.0:
        profit_factor_score = 1.0
      elif similar_signals.profit_factor > 1.5:
        profit_factor_score = 0.8
      elif similar_signals.profit_factor > 1.2:
        profit_factor_score = 0.6
      elif similar_signals.profit_factor > 1.0:
        profit_factor_score = 0.4
      else:
        profit_factor_score = 0.2

      # Учитываем последовательные проигрыши
      consecutive_loss_penalty = 0
      if similar_signals.max_consecutive_losses > 5:
        consecutive_loss_penalty = 0.2
      elif similar_signals.max_consecutive_losses > 3:
        consecutive_loss_penalty = 0.1

      historical_score = (win_rate_score * 0.6 + profit_factor_score * 0.4) - consecutive_loss_penalty

      return max(0, min(1, historical_score))

    except Exception as e:
      logger.error(f"Ошибка проверки исторической производительности: {e}")
      return 0.5

  async def _get_similar_historical_signals(self, signal: TradingSignal,
                                            market_data: pd.DataFrame) -> Optional[HistoricalSignalPerformance]:
    """Получает статистику по похожим историческим сигналам"""
    try:
      # Определяем характеристики текущего сигнала
      current_rsi = market_data.get('rsi_14', pd.Series()).iloc[-1] if 'rsi_14' in market_data else 50
      current_volatility = market_data['close'].pct_change().rolling(20).std().iloc[-1]

      # Запрашиваем похожие сигналы из БД
      query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN profit_loss > 0 THEN profit_loss ELSE 0 END) as avg_profit,
                AVG(CASE WHEN profit_loss < 0 THEN profit_loss ELSE 0 END) as avg_loss,
                AVG(JULIANDAY(close_timestamp) - JULIANDAY(open_timestamp)) as avg_holding_days
            FROM trades
            WHERE 
                symbol = ? AND 
                strategy = ? AND
                side = ? AND
                status = 'CLOSED' AND
                open_timestamp > datetime('now', '-180 days')
            """

      result = await self.db_manager._execute(
        query,
        (signal.symbol, signal.strategy_name, signal.signal_type.value),
        fetch='one'
      )

      if not result or result['total'] == 0:
        return None

      total_signals = result['total']
      winning_signals = result['wins'] or 0
      losing_signals = result['losses'] or 0
      win_rate = winning_signals / total_signals if total_signals > 0 else 0
      avg_profit = abs(result['avg_profit'] or 0)
      avg_loss = abs(result['avg_loss'] or 0)
      profit_factor = avg_profit / avg_loss if avg_loss > 0 else avg_profit

      return HistoricalSignalPerformance(
        total_signals=total_signals,
        winning_signals=winning_signals,
        losing_signals=losing_signals,
        win_rate=win_rate,
        avg_profit=avg_profit,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        max_consecutive_wins=0,  # Можно добавить расчет
        max_consecutive_losses=0,  # Можно добавить расчет
        avg_holding_time=timedelta(days=result['avg_holding_days'] or 1)
      )

    except Exception as e:
      logger.error(f"Ошибка получения исторических данных: {e}")
      return None

  def _check_volume_confirmation(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
    """Проверяет подтверждение сигнала объемом"""
    try:
      volumes = market_data['volume']

      # Средний объем
      avg_volume = volumes.rolling(20).mean().iloc[-1]
      current_volume = volumes.iloc[-1]

      if avg_volume == 0:
        return 0.5

      volume_ratio = current_volume / avg_volume

      # Сохраняем для статистики
      self.signal_statistics['volume_ratios'].append(volume_ratio)

      # Анализ объема при движении цены
      price_change = market_data['close'].pct_change().iloc[-1]

      # Хороший сигнал: движение цены подтверждается объемом
      if signal.signal_type == SignalType.BUY:
        if price_change > 0 and volume_ratio > 1.5:
          return 0.9  # Отличное подтверждение
        elif price_change > 0 and volume_ratio > 1.2:
          return 0.7
        elif price_change > 0:
          return 0.5  # Движение без объема
        else:
          return 0.3  # Противоречие
      else:  # SELL
        if price_change < 0 and volume_ratio > 1.5:
          return 0.9
        elif price_change < 0 and volume_ratio > 1.2:
          return 0.7
        elif price_change < 0:
          return 0.5
        else:
          return 0.3

    except Exception as e:
      logger.error(f"Ошибка проверки объема: {e}")
      return 0.5

  def _check_volatility_fitness(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
    """Проверяет соответствие сигнала текущей волатильности"""
    try:
      returns = market_data['close'].pct_change()
      current_volatility = returns.rolling(20).std().iloc[-1]

      # Сохраняем для статистики
      self.signal_statistics['volatility_values'].append(current_volatility)

      # ATR
      import pandas_ta as ta
      atr = ta.atr(market_data['high'], market_data['low'], market_data['close'], length=14)
      if atr is not None and not atr.empty:
        current_atr = atr.iloc[-1]
        atr_ratio = current_atr / market_data['close'].iloc[-1]
      else:
        atr_ratio = current_volatility

      # Оценка соответствия
      # Слишком низкая волатильность - плохо для трендовых стратегий
      if current_volatility < 0.001:  # Менее 0.1%
        return 0.3
      # Слишком высокая волатильность - высокий риск
      elif current_volatility > 0.05:  # Более 5%
        return 0.4
      # Оптимальная волатильность
      elif 0.005 < current_volatility < 0.02:  # 0.5-2%
        return 0.9
      else:
        return 0.6

    except Exception as e:
      logger.error(f"Ошибка проверки волатильности: {e}")
      return 0.5

  def _check_trend_alignment(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
    """Проверяет соответствие сигнала общему тренду"""
    try:
      closes = market_data['close']

      # Различные периоды для определения тренда
      sma_10 = closes.rolling(10).mean().iloc[-1]
      sma_20 = closes.rolling(20).mean().iloc[-1]
      sma_50 = closes.rolling(50).mean().iloc[-1]
      sma_200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else sma_50

      current_price = closes.iloc[-1]

      # Подсчет соответствия тренду
      trend_scores = []

      if signal.signal_type == SignalType.BUY:
        # Проверяем восходящую структуру
        if current_price > sma_10:
          trend_scores.append(1.0)
        if sma_10 > sma_20:
          trend_scores.append(0.8)
        if sma_20 > sma_50:
          trend_scores.append(0.7)
        if current_price > sma_200:
          trend_scores.append(0.9)
      else:  # SELL
        # Проверяем нисходящую структуру
        if current_price < sma_10:
          trend_scores.append(1.0)
        if sma_10 < sma_20:
          trend_scores.append(0.8)
        if sma_20 < sma_50:
          trend_scores.append(0.7)
        if current_price < sma_200:
          trend_scores.append(0.9)

      return np.mean(trend_scores) if trend_scores else 0.5

    except Exception as e:
      logger.error(f"Ошибка проверки тренда: {e}")
      return 0.5

  def _determine_quality_category(self, score: float) -> QualityScore:
    """Определяет категорию качества по баллу"""
    for category, threshold in sorted(self.quality_thresholds.items(),
                                      key=lambda x: x[1], reverse=True):
      if score >= threshold:
        return category
    return QualityScore.UNACCEPTABLE

  def _analyze_strengths_weaknesses(self, scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
    """Анализирует сильные и слабые стороны сигнала"""
    strengths = []
    weaknesses = []

    for component, score in scores.items():
      readable_name = component.replace('_', ' ').title()

      if score >= 0.8:
        strengths.append(f"Отличный {readable_name} ({score:.2f})")
      elif score >= 0.6:
        strengths.append(f"Хороший {readable_name} ({score:.2f})")
      elif score <= 0.3:
        weaknesses.append(f"Слабый {readable_name} ({score:.2f})")
      elif score <= 0.5:
        weaknesses.append(f"Недостаточный {readable_name} ({score:.2f})")

    return strengths, weaknesses

  def _generate_recommendations(self, scores: Dict[str, float], signal: TradingSignal) -> List[str]:
    """Генерирует рекомендации по улучшению торговли"""
    recommendations = []

    # Рекомендации по компонентам с низкими оценками
    if scores['timeframe_alignment'] < 0.5:
      recommendations.append("Дождитесь лучшего выравнивания на разных таймфреймах")

    if scores['momentum_strength'] < 0.5:
      recommendations.append("Слабый моментум - рассмотрите уменьшение позиции")

    if scores['volume_confirmation'] < 0.5:
      recommendations.append("Отсутствует подтверждение объемом - будьте осторожны")

    if scores['historical_performance'] < 0.4:
      recommendations.append("Низкая историческая успешность - используйте жесткий стоп-лосс")

    if scores['volatility_fitness'] < 0.5:
      if self.signal_statistics['volatility_values']:
        current_vol = self.signal_statistics['volatility_values'][-1]
        if current_vol > 0.03:
          recommendations.append("Высокая волатильность - уменьшите размер позиции")
        elif current_vol < 0.002:
          recommendations.append("Низкая волатильность - ожидайте медленное движение")

    # Общие рекомендации на основе качества
    overall_score = sum(score * self.quality_weights[comp] for comp, score in scores.items())

    if overall_score < 0.4:
      recommendations.append("⚠️ Низкое качество сигнала - рекомендуется пропустить")
    elif overall_score < 0.6:
      recommendations.append("Среднее качество - используйте половинный размер позиции")
    elif overall_score > 0.8:
      recommendations.append("✅ Высокое качество сигнала - можно использовать полный размер")

    return recommendations

  def _calculate_risk_reward_ratio(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
    """Рассчитывает соотношение риск/прибыль"""
    if not signal.stop_loss or not signal.take_profit:
      return 1.0

    current_price = signal.price

    if signal.signal_type == SignalType.BUY:
      risk = current_price - signal.stop_loss
      reward = signal.take_profit - current_price
    else:  # SELL
      risk = signal.stop_loss - current_price
      reward = current_price - signal.take_profit

    if risk <= 0:
      return 0

    return reward / risk

  def _calculate_signal_percentile(self, score: float) -> float:
    """Рассчитывает процентиль силы сигнала среди всех исторических"""
    if not self.signal_history:
      return 50.0

    scores = [s[1] for s in self.signal_history]
    percentile = (sum(1 for s in scores if s < score) / len(scores)) * 100

    return percentile

  def _save_to_history(self, signal: TradingSignal, score: float):
    """Сохраняет сигнал в историю"""
    self.signal_history.append((signal, score, datetime.now()))

  def _create_default_metrics(self, signal: TradingSignal) -> SignalQualityMetrics:
    """Создает метрики по умолчанию при ошибке"""
    return SignalQualityMetrics(
      overall_score=0.5,
      quality_category=QualityScore.FAIR,
      timeframe_alignment_score=0.5,
      momentum_strength_score=0.5,
      market_structure_score=0.5,
      historical_performance_score=0.5,
      volume_confirmation_score=0.5,
      volatility_fitness_score=0.5,
      trend_alignment_score=0.5,
      confidence_level=signal.confidence * 0.5,
      risk_reward_ratio=1.0,
      expected_win_rate=0.5,
      signal_strength_percentile=50.0,
      strengths=["Базовая оценка"],
      weaknesses=["Не удалось выполнить полный анализ"],
      recommendations=["Используйте с осторожностью"]
    )

  def get_quality_statistics(self) -> Dict[str, Any]:
    """Возвращает статистику по качеству сигналов"""
    if not self.signal_history:
      return {"status": "no_data"}

    scores = [s[1] for s in self.signal_history]
    recent_scores = [s[1] for s in self.signal_history if (datetime.now() - s[2]).days <= 7]

    quality_distribution = {
      category.value: sum(1 for s in scores if self._determine_quality_category(s) == category)
      for category in QualityScore
    }

    return {
      "total_signals_analyzed": len(self.signal_history),
      "average_quality_score": np.mean(scores),
      "recent_average_score": np.mean(recent_scores) if recent_scores else 0,
      "best_score": max(scores),
      "worst_score": min(scores),
      "quality_distribution": quality_distribution,
      "high_quality_percentage": sum(1 for s in scores if s >= 0.7) / len(scores) * 100
    }

