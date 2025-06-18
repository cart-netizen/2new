# strategies/mean_reversion_strategy.py
import pandas as pd
import pandas_ta as ta
from typing import Optional
from datetime import datetime
from core.schemas import TradingSignal
from core.enums import SignalType
from strategies.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
  def __init__(self, strategy_name: str = "Mean_Reversion_BB"):
    super().__init__(strategy_name)

  async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
    if len(data) < 20: return None
    bbands = ta.bbands(data['close'], length=20, std=2)
    if bbands is None: return None

    last_price = data['close'].iloc[-1]
    lower_band = bbands.iloc[-1, 0]  # BBL_20_2.0
    upper_band = bbands.iloc[-1, 2]  # BBU_20_2.0

    signal_type = SignalType.HOLD
    if last_price < lower_band:
      signal_type = SignalType.BUY
    elif last_price > upper_band:
      signal_type = SignalType.SELL

    if signal_type == SignalType.HOLD:
      return None

    stop_loss = last_price * (1 - 0.05) if signal_type == SignalType.BUY else last_price * (1 + 0.05)
    take_profit = bbands.iloc[-1, 1]  # Средняя линия Боллинджера как цель

    return TradingSignal(
      signal_type=signal_type, symbol=symbol, price=last_price,
      confidence=0.70, strategy_name=self.strategy_name, timestamp=datetime.now(),
      stop_loss=stop_loss, take_profit=take_profit
    )