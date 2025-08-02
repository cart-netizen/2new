# strategies/momentum_strategy.py
import pandas as pd
import pandas_ta as ta
from typing import Optional
from datetime import datetime, timezone

from config.config_manager import logger
from core.schemas import TradingSignal
from core.enums import SignalType

from strategies.base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
  def __init__(self, strategy_name: str = "Momentum_Spike"):
    super().__init__(strategy_name)

  async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
    if len(data) < 21: return None

    # Расчет всплеска объема и быстрой EMA
    volume_sma = ta.sma(data['volume'], length=20)
    volume_spike = data['volume'].iloc[-1] / (volume_sma.iloc[-1] + 1e-9)

    fast_ema = ta.ema(data['close'], length=9)

    if len(data) >= 2:
      # Проверяем порядок сортировки по индексу или timestamp
      if hasattr(data.index, 'to_timestamp'):
        first_ts = data.index[0]
        second_ts = data.index[1]
        is_desc_order = first_ts > second_ts
      elif 'timestamp' in data.columns:
        first_ts = data['timestamp'].iloc[0]
        second_ts = data['timestamp'].iloc[1]
        is_desc_order = first_ts > second_ts
      else:
        is_desc_order = False  # По умолчанию берем последний

      # Берем актуальную цену в зависимости от порядка сортировки
      last_price = data['close'].iloc[0] if is_desc_order else data['close'].iloc[-1]

      logger.debug(f"🔍 Стратегия {self.strategy_name}: цена для анализа = {last_price}, порядок desc = {is_desc_order}")
    else:
      last_price = data['close'].iloc[-1]  # Fallback для одной записи



    signal_type = SignalType.HOLD

    # Сигнал BUY: сильный всплеск объема И цена выше своей быстрой EMA (подтверждение импульса)
    if volume_spike > 5.0 and last_price > fast_ema.iloc[-1]:
      signal_type = SignalType.BUY

    # Сигнал SELL: сильный всплеск объема И цена ниже своей быстрой EMA
    elif volume_spike > 5.0 and last_price < fast_ema.iloc[-1]:
      signal_type = SignalType.SELL

    if signal_type == SignalType.HOLD:
      return None

    atr = ta.atr(data['high'], data['low'], data['close'], length=14).iloc[-1]
    stop_loss = last_price - 2 * atr if signal_type == SignalType.BUY else last_price + 2 * atr

    return TradingSignal(
      signal_type=signal_type, symbol=symbol, price=last_price,
      confidence=0.80,  # Высокая уверенность из-за объема
      strategy_name=self.strategy_name, timestamp=datetime.now(timezone.utc),
      stop_loss=stop_loss
    )