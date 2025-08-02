from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

from config.config_manager import logger
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

    def get_current_price(self, data: pd.DataFrame) -> float:
        """Универсальный метод получения актуальной цены с учетом сортировки данных"""
        if len(data) < 1:
            raise ValueError("Данные пусты")

        if len(data) >= 2:
            # Проверяем порядок сортировки данных
            if 'timestamp' in data.columns:
                first_ts = data['timestamp'].iloc[0]
                second_ts = data['timestamp'].iloc[1]
                is_desc_order = first_ts > second_ts
            elif hasattr(data.index, 'to_timestamp') or isinstance(data.index, pd.DatetimeIndex):
                first_idx = data.index[0]
                second_idx = data.index[1]
                is_desc_order = first_idx > second_idx
            else:
                is_desc_order = False

            # Берем правильную цену в зависимости от сортировки
            current_price = float(data['close'].iloc[0] if is_desc_order else data['close'].iloc[-1])

            # Логируем для диагностики
            logger.debug(
                f"🔍 {self.strategy_name}: актуальная цена = {current_price}, порядок убывающий = {is_desc_order}")

            return current_price
        else:
            # Если только одна запись
            return float(data['close'].iloc[-1])
        #---------------------------------------------------------------------------------------------
        # Вместо всех проверок и last_price = ... используйте:
        # last_price = self.get_current_price(data)
        #---------------------------------------------------------------------------------------------
