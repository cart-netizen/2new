"""
Быстрый тест RL стратегии для проверки работоспособности
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta

from core.enums import Timeframe
from core.integrated_system import IntegratedTradingSystem
from config.config_manager import ConfigManager
from utils.logging_config import get_logger

logger = get_logger(__name__)


async def quick_rl_test():
  """Быстрый тест RL на минимальных данных"""
  logger.info("⚡ Быстрый тест RL стратегии...")

  # Загружаем конфигурацию
  config_manager = ConfigManager()
  config = config_manager.load_config()

  # Включаем RL
  config['rl_trading']['enabled'] = True
  config['general_settings']['monitoring_interval_seconds'] = 10  # Быстрее для теста

  # Создаем систему
  system = IntegratedTradingSystem(config=config)

  # Тестируем на одном символе
  test_symbol = 'BTCUSDT'

  try:
    # Получаем данные
    data = await system.data_fetcher.get_historical_candles(
      symbol=test_symbol,
      timeframe=Timeframe.ONE_HOUR,
      limit=100
    )

    if data is None or data.empty:
      logger.error("Не удалось получить данные")
      return

    logger.info(f"Получено {len(data)} баров для {test_symbol}")

    # Генерируем сигнал
    rl_strategy = system.strategy_manager.strategies.get('RL_Strategy')

    if not rl_strategy:
      logger.error("RL стратегия не найдена")
      return

    signal = await rl_strategy.generate_signal(test_symbol, data)

    if signal:
      logger.info(f"✅ Сигнал сгенерирован: {signal.signal_type.value}")
      logger.info(f"   Уверенность: {signal.confidence:.2f}")
      logger.info(f"   Цена входа: {signal.entry_price}")
      logger.info(f"   Метаданные: {signal.metadata}")
    else:
      logger.info("❌ Сигнал не сгенерирован (условия не выполнены)")

    # Проверяем статус стратегии
    status = rl_strategy.get_strategy_status()

    logger.info("\n📊 Статус RL стратегии:")
    logger.info(f"   Алгоритм: {status['algorithm']}")
    logger.info(f"   Обучена: {status['is_trained']}")
    logger.info(f"   Всего сигналов: {status['total_signals']}")
    logger.info(f"   Средняя уверенность: {status['average_confidence']:.2f}")

    logger.info("\n✅ Быстрый тест завершен успешно!")

  except Exception as e:
    logger.error(f"Ошибка быстрого теста: {e}", exc_info=True)
  finally:
    # await system.shutdown()

    # Закрываем все соединения
    if system and hasattr(system, 'connector') and system.connector:
      await system.connector.close()
    if system and hasattr(system, 'data_fetcher') and hasattr(system.data_fetcher, 'connector'):
      await system.data_fetcher.connector.close()

if __name__ == "__main__":
  asyncio.run(quick_rl_test())