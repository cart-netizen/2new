# strategies/rl_strategy.py

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime

from strategies.base_strategy import BaseStrategy
from core.schemas import TradingSignal
from core.enums import SignalType
from rl.finrl_agent import EnhancedRLAgent
from rl.feature_processor import RLFeatureProcessor
from utils.logging_config import get_logger

logger = get_logger(__name__)


class RLStrategy(BaseStrategy):
  """
  Стратегия на основе обучения с подкреплением
  Интегрирована с системой стратегий проекта
  """

  def __init__(
      self,
      rl_agent: EnhancedRLAgent,
      feature_processor: RLFeatureProcessor,
      data_fetcher,
      config: Dict[str, Any]
  ):
    super().__init__(strategy_name="RL_Strategy")
    self.rl_agent = rl_agent
    self.feature_processor = feature_processor
    self.data_fetcher = data_fetcher
    self.config = config

    # Параметры стратегии
    self.confidence_threshold = config.get('confidence_threshold', 0.6)
    self.use_ml_signals = config.get('use_ml_signals', True)
    self.use_regime_filter = config.get('use_regime_filter', True)
    self.position_sizing_mode = config.get('position_sizing_mode', 'dynamic')

    # История для анализа
    self.signal_history = []
    self.performance_metrics = {
      'total_signals': 0,
      'successful_signals': 0,
      'average_confidence': 0,
      'win_rate': 0
    }

    logger.info(f"Инициализирована RL стратегия с алгоритмом {rl_agent.algorithm}")

  async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
    """
    Генерация торгового сигнала с использованием RL агента
    """
    try:
      if data.empty or len(data) < 100:
        return None

      # 1. Создаем признаки для RL
      features = self.feature_processor.create_rl_features(data, symbol)

      if features is None or features.shape[0] == 0:
        logger.warning(f"Не удалось создать признаки для {symbol}")
        return None

      # Берем последнее состояние
      current_state = features[-1] if len(features.shape) == 2 else features

      # 2. Получаем расширенное состояние с ML сигналами
      if self.use_ml_signals:
        enhanced_state = await self.rl_agent.get_enhanced_state(symbol, data)
      else:
        enhanced_state = None

      # 3. Получаем предсказание от RL агента
      prediction = self.rl_agent.predict_with_analysis(
        current_state,
        data
      )

      # 4. Интерпретируем действие
      action = prediction['action']
      confidence = prediction['confidence']

      # Проверяем порог уверенности
      if confidence < self.confidence_threshold:
        logger.debug(f"Сигнал отклонен для {symbol}: низкая уверенность {confidence:.2f}")
        return None

      # 5. Определяем тип сигнала
      if action == 2:  # BUY
        signal_type = SignalType.BUY
      elif action == 0:  # SELL
        signal_type = SignalType.SELL
      else:  # HOLD
        return None

      # 6. Применяем дополнительные фильтры
      if not await self._apply_filters(symbol, data, signal_type, enhanced_state):
        return None

      # 7. Рассчитываем параметры позиции
      current_price = float(data['close'].iloc[-1])
      position_params = self._calculate_position_parameters(
        current_price,
        signal_type,
        data,
        confidence,
        enhanced_state
      )

      # 8. Создаем сигнал
      signal = TradingSignal(
        symbol=symbol,
        signal_type=signal_type,
        confidence=confidence,
        entry_price=current_price,
        stop_loss=position_params['stop_loss'],
        take_profit=position_params['take_profit'],
        strategy_name=self.strategy_name,
        metadata={
          'rl_action': action,
          'q_values': prediction.get('q_values'),
          'enhanced_state': enhanced_state,
          'market_conditions': prediction.get('market_conditions', {}),
          'position_size_multiplier': position_params['size_multiplier'],
          'risk_reward_ratio': position_params['risk_reward_ratio']
        }
      )

      # 9. Обновляем статистику
      self._update_statistics(signal)

      logger.info(f"RL сигнал сгенерирован для {symbol}: {signal_type.value} с уверенностью {confidence:.2f}")

      return signal

    except Exception as e:
      logger.error(f"Ошибка генерации RL сигнала для {symbol}: {e}", exc_info=True)
      return None

  async def _apply_filters(
        self,
        symbol: str,
        data: pd.DataFrame,
        signal_type: SignalType,
        enhanced_state: Optional[Dict] = None
    ) -> bool:
      """Применяет дополнительные фильтры к сигналу"""

      # 1. Фильтр по режиму рынка
      if self.use_regime_filter and enhanced_state and 'regime_info' in enhanced_state:
        regime = enhanced_state['regime_info'].get('current_regime')

        # Проверяем соответствие сигнала режиму
        if signal_type == SignalType.BUY and regime in ['STRONG_TREND_DOWN', 'TREND_DOWN']:
          logger.debug(f"BUY сигнал отклонен в медвежьем режиме для {symbol}")
          return False
        elif signal_type == SignalType.SELL and regime in ['STRONG_TREND_UP', 'TREND_UP']:
          logger.debug(f"SELL сигнал отклонен в бычьем режиме для {symbol}")
          return False

      # 2. Фильтр по аномалиям
      if enhanced_state and 'anomaly_score' in enhanced_state:
        if enhanced_state['anomaly_score'] > 0.8:
          logger.debug(f"Сигнал отклонен из-за высокой аномальности рынка для {symbol}")
          return False

      # 3. Фильтр по волатильности
      if enhanced_state and 'volatility_forecast' in enhanced_state:
        predicted_vol = enhanced_state['volatility_forecast'].get('predicted_volatility', 0)
        if predicted_vol > self.config.get('max_volatility', 0.05):
          logger.debug(f"Сигнал отклонен из-за высокой прогнозной волатильности для {symbol}")
          return False

      # 4. Проверка объема
      volume_sma = data['volume'].rolling(20).mean().iloc[-1]
      current_volume = data['volume'].iloc[-1]
      if current_volume < volume_sma * 0.5:
        logger.debug(f"Сигнал отклонен из-за низкого объема для {symbol}")
        return False

      return True

  def _calculate_position_parameters(
        self,
        current_price: float,
        signal_type: SignalType,
        data: pd.DataFrame,
        confidence: float,
        enhanced_state: Optional[Dict] = None
    ) -> Dict[str, float]:
      """Рассчитывает параметры позиции"""

      # Базовый ATR для расчета уровней
      atr = data['high'].rolling(14).apply(lambda x: np.mean(np.abs(np.diff(x)))).iloc[-1]
      if pd.isna(atr) or atr == 0:
        atr = current_price * 0.02  # 2% как fallback

      # Динамический размер позиции на основе уверенности
      if self.position_sizing_mode == 'dynamic':
        # Чем выше уверенность, тем больше позиция
        size_multiplier = 0.5 + (confidence - self.confidence_threshold) * 2
        size_multiplier = np.clip(size_multiplier, 0.5, 2.0)
      else:
        size_multiplier = 1.0

      # Расчет стоп-лосса с учетом волатильности
      if enhanced_state and 'volatility_forecast' in enhanced_state:
        vol_multiplier = 1 + enhanced_state['volatility_forecast'].get('predicted_volatility', 0) * 10
      else:
        vol_multiplier = 1.0

      # Адаптивные уровни на основе режима рынка
      if enhanced_state and 'regime_info' in enhanced_state:
        regime = enhanced_state['regime_info'].get('current_regime', 'RANGE_BOUND')

        if 'STRONG_TREND' in regime:
          # В сильном тренде - шире стопы, дальше цели
          sl_multiplier = 2.0 * vol_multiplier
          tp_multiplier = 3.0
        elif 'TREND' in regime:
          sl_multiplier = 1.5 * vol_multiplier
          tp_multiplier = 2.5
        else:  # RANGE_BOUND
          sl_multiplier = 1.0 * vol_multiplier
          tp_multiplier = 1.5
      else:
        sl_multiplier = 1.5 * vol_multiplier
        tp_multiplier = 2.0

      # Расчет уровней
      if signal_type == SignalType.BUY:
        stop_loss = current_price - (atr * sl_multiplier)
        take_profit = current_price + (atr * tp_multiplier * (1 + confidence - 0.5))
      else:  # SELL
        stop_loss = current_price + (atr * sl_multiplier)
        take_profit = current_price - (atr * tp_multiplier * (1 + confidence - 0.5))

      # Risk/Reward ratio
      risk = abs(current_price - stop_loss)
      reward = abs(take_profit - current_price)
      risk_reward_ratio = reward / risk if risk > 0 else 0

      return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'size_multiplier': size_multiplier,
        'risk_reward_ratio': risk_reward_ratio
      }

  def _update_statistics(self, signal: TradingSignal):
      """Обновляет статистику стратегии"""
      self.signal_history.append({
        'timestamp': datetime.now(),
        'symbol': signal.symbol,
        'signal_type': signal.signal_type,
        'confidence': signal.confidence
      })

      # Ограничиваем историю
      if len(self.signal_history) > 100:
        self.signal_history.pop(0)

      # Обновляем метрики
      self.performance_metrics['total_signals'] += 1

      # Средняя уверенность
      confidences = [s['confidence'] for s in self.signal_history]
      self.performance_metrics['average_confidence'] = np.mean(confidences)

  def get_strategy_status(self) -> Dict[str, Any]:
      """Возвращает текущий статус стратегии"""
      return {
        'name': self.strategy_name,
        'algorithm': self.rl_agent.algorithm,
        'is_trained': self.rl_agent.is_trained,
        'total_signals': self.performance_metrics['total_signals'],
        'average_confidence': self.performance_metrics['average_confidence'],
        'recent_signals': len([s for s in self.signal_history if
                               (datetime.now() - s['timestamp']).seconds < 3600]),
        'config': {
          'confidence_threshold': self.confidence_threshold,
          'use_ml_signals': self.use_ml_signals,
          'use_regime_filter': self.use_regime_filter,
          'position_sizing_mode': self.position_sizing_mode
        }
      }

