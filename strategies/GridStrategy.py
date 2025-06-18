# strategies/grid_strategy.py
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Optional, List, Dict, Any
from datetime import datetime

from core.schemas import TradingSignal, GridSignal  # <-- Добавим новый тип сигнала
from core.enums import SignalType
from strategies.base_strategy import BaseStrategy


class GridStrategy(BaseStrategy):
  """
  Адаптивная сеточная стратегия для работы во флэте.
  """

  def __init__(self, config: Dict[str, Any]):
    super().__init__(strategy_name="Grid_Trading")
    self.grid_levels = config.get('strategy_settings', {}).get('grid_levels', 10)
    self.atr_multiplier = config.get('strategy_settings', {}).get('grid_atr_multiplier', 0.5)

  async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[GridSignal]:
    if len(data) < 21: return None

    # 1. Рассчитываем границы (Bollinger Bands) и шаг (ATR)
    bbands = ta.bbands(data['close'], length=20, std=2)
    atr = ta.atr(data['high'], data['low'], data['close'], length=14)
    if bbands is None or atr is None: return None

    upper_band = bbands.iloc[-1, 2]  # BBU_20_2.0
    lower_band = bbands.iloc[-1, 0]  # BBL_20_2.0
    atr_value = atr.iloc[-1]

    # 2. Формируем сетку ордеров
    grid_step = atr_value * self.atr_multiplier
    if grid_step == 0: return None

    buy_prices = [p for p in np.arange(data['close'].iloc[-1] - grid_step, lower_band, -grid_step)]
    sell_prices = [p for p in np.arange(data['close'].iloc[-1] + grid_step, upper_band, grid_step)]

    if not buy_prices or not sell_prices:
      return None

    return GridSignal(
      symbol=symbol,
      buy_orders=[{'price': p, 'qty': 0} for p in buy_prices[:self.grid_levels // 2]],
      sell_orders=[{'price': p, 'qty': 0} for p in sell_prices[:self.grid_levels // 2]]
    )