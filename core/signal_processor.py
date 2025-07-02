from typing import Dict, Any

import pandas as pd

from core.risk_manager import AdvancedRiskManager
from core.schemas import TradingSignal
from core.signal_filter import SignalFilter
from core.signal_quality_analyzer import SignalQualityAnalyzer


class SignalProcessor:
  """Отдельный процессор для верификации сигналов"""

  def __init__(self, risk_manager: AdvancedRiskManager, signal_filter:SignalFilter, signal_quality_analyzer:SignalQualityAnalyzer):
    self.risk_manager = risk_manager

  async def verify_signal(self, signal: TradingSignal, symbol: str, balance: float, market_data: pd.DataFrame) -> Dict[str, Any]:
    return await self.risk_manager.validate_signal(signal, symbol, balance, market_data)