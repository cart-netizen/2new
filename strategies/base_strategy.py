from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional
from core.schemas import TradingSignal

class BaseStrategy(ABC):
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    @abstractmethod
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Анализирует данные и генерирует торговый сигнал.
        data: DataFrame со свечами (index=timestamp, columns=['open', 'high', 'low', 'close', 'volume'])
        """
        pass