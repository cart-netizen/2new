# strategies/dual_thrust_strategy.py

import pandas as pd
import pandas_ta as ta
from typing import Optional, Dict, Any
from datetime import datetime

from core.schemas import TradingSignal
from core.enums import SignalType, Timeframe
from strategies.base_strategy import BaseStrategy
from utils.logging_config import get_logger

logger = get_logger(__name__)


class DualThrustStrategy(BaseStrategy):
  """
  Классическая пробойная интрадей-стратегия Dual Thrust.
  Генерирует сигнал, когда цена пробивает диапазон, рассчитанный
  на основе предыдущего дня.
  """

  def __init__(self, config: Dict[str, Any], data_fetcher):
    super().__init__(strategy_name="Dual_Thrust")
    self.data_fetcher = data_fetcher
    # Загружаем коэффициенты K1 и K2 из настроек
    self.k1 = config.get('strategy_settings', {}).get('dual_thrust_k1', 0.5)
    self.k2 = config.get('strategy_settings', {}).get('dual_thrust_k2', 0.5)

  async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
    """
    data: DataFrame с интрадей-данными (например, 1H или 15m)
    """
    if data.empty or len(data) < 2:
      return None

    try:
      # 1. Получаем данные за предыдущий день для расчета диапазона
      # limit=2, чтобы получить вчерашний и позавчерашний день для надежности
      daily_data = await self.data_fetcher.get_historical_candles(symbol, Timeframe.ONE_DAY, limit=2)
      if daily_data is None or len(daily_data) < 2:
        logger.warning(f"[{self.strategy_name}] Недостаточно дневных данных для {symbol}.")
        return None

      prev_day = daily_data.iloc[-2]  # Данные за вчерашний день

      # 2. Рассчитываем диапазон и границы пробоя
      prev_day_range = prev_day['high'] - prev_day['low']
      today_open_price = data['open'].iloc[0]  # Цена открытия текущего дня

      upper_band = today_open_price + (self.k1 * prev_day_range)
      lower_band = today_open_price - (self.k2 * prev_day_range)

      # 3. Проверяем на пробой
      current_price = data['close'].iloc[-1]
      signal_type = SignalType.HOLD

      if current_price > upper_band:
        signal_type = SignalType.BUY
      elif current_price < lower_band:
        signal_type = SignalType.SELL

      if signal_type == SignalType.HOLD:
        return None

      logger.info(
        f"[{self.strategy_name}] Обнаружен сигнал {signal_type.value} для {symbol} по цене {current_price:.4f}")

      # Для пробойной стратегии стоп-лосс логично ставить на цену открытия дня
      # stop_loss = today_open_price
      # Тейк-профит можно установить как симметричный пробой в обратную сторону
      # take_profit = lower_band if signal_type == SignalType.BUY else upper_band

      return TradingSignal(
        signal_type=signal_type,
        symbol=symbol,
        price=current_price,
        confidence=0.75,  # Пробойные сигналы обычно довольно надежны
        strategy_name=self.strategy_name,
        timestamp=datetime.now(),
        # stop_loss=stop_loss,
        # take_profit=take_profit
      )

    except Exception as e:
      logger.error(f"[{self.strategy_name}] Ошибка при генерации сигнала для {symbol}: {e}", exc_info=True)
      return None