import pandas as pd
from typing import Optional, Dict, List
from utils.logging_config import get_logger
from core.schemas import TradingSignal
from strategies.base_strategy import BaseStrategy

# Импортируем вашу ML стратегию, когда она будет готова в strategies/
# from strategies.ensemble_ml_strategy import EnsembleMLStrategy

logger = get_logger(__name__)


class StrategyManager:
  def __init__(self):
    self.strategies: Dict[str, BaseStrategy] = {}
    # self.active_strategy_name: Optional[str] = None # Если будет выбор активной стратегии

    # Пока что, если EnsembleMLStrategy будет основной, можно ее здесь инициализировать
    # self.default_ml_strategy = EnsembleMLStrategy(...) # Потребуются db_manager, ml_model
    # self.add_strategy(self.default_ml_strategy)
    # self.active_strategy_name = self.default_ml_strategy.strategy_name
    logger.info("StrategyManager инициализирован. Пока нет активных стратегий.")

  def add_strategy(self, strategy: BaseStrategy):
    if strategy.strategy_name in self.strategies:
      logger.warning(f"Стратегия {strategy.strategy_name} уже существует и будет перезаписана.")
    self.strategies[strategy.strategy_name] = strategy
    logger.info(f"Стратегия {strategy.strategy_name} добавлена.")

  # def set_active_strategy(self, strategy_name: str):
  #     if strategy_name in self.strategies:
  #         self.active_strategy_name = strategy_name
  #         logger.info(f"Активная стратегия установлена: {strategy_name}")
  #     else:
  #         logger.error(f"Стратегия {strategy_name} не найдена.")

  async def get_signal(self, symbol: str, data: pd.DataFrame, strategy_name: Optional[str] = None) -> Optional[
    TradingSignal]:
    # if not self.active_strategy_name and not strategy_name:
    #     logger.warning("Активная стратегия не установлена и имя стратегии не указано.")
    #     return None

    # target_strategy_name = strategy_name if strategy_name else self.active_strategy_name

    # Пока используем заглушку, т.к. реальная стратегия требует ML модель и данные
    logger.debug(f"Заглушка: StrategyManager получил запрос на сигнал для {symbol}.")
    if not self.strategies:
      logger.warning("В StrategyManager нет зарегистрированных стратегий.")
      return None

    # Пока будем использовать первую попавшуюся стратегию или указанную по имени
    # В будущем это будет более сложная логика выбора или ансамбля
    active_strategy = None
    if strategy_name and strategy_name in self.strategies:
      active_strategy = self.strategies[strategy_name]
    elif self.strategies:  # Если есть хоть одна стратегия
      active_strategy = next(iter(self.strategies.values()))
      logger.info(f"Используется первая доступная стратегия: {active_strategy.strategy_name}")

    if active_strategy:
      try:
        return await active_strategy.generate_signal(symbol, data)
      except Exception as e:
        logger.error(f"Ошибка при генерации сигнала стратегией {active_strategy.strategy_name} для {symbol}: {e}")
        return None
    else:
      logger.error(f"Стратегия не найдена или не активна.")
      return None