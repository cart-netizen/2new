from typing import Dict, Any

from core.risk_manager import AdvancedRiskManager
from core.schemas import TradingSignal


class SignalProcessor:
  """Отдельный процессор для верификации сигналов"""

  def __init__(self, risk_manager: AdvancedRiskManager):
    self.risk_manager = risk_manager

  async def verify_signal(self, signal: TradingSignal, symbol: str, balance: float) -> Dict[str, Any]:
    return await self.risk_manager.validate_signal(signal, symbol, balance)