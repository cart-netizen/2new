import asyncio
import os
import signal  # Для корректной остановки
import sys
from utils.logging_config import setup_logging, get_logger, setup_signal_logger
from config import settings
from core.integrated_system import IntegratedTradingSystem

import warnings
# Попытаемся импортировать конкретный тип предупреждения.
# Если не получится (вдруг его тоже удалят), будем использовать общий DeprecationWarning.
try:
    from pkg_resources import PkgResourcesDeprecationWarning
    warnings.filterwarnings("ignore", category=PkgResourcesDeprecationWarning)
except ImportError:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
# --- БЛОК ФИЛЬТРАЦИИ ПРЕДУПРЕЖДЕНИЙ ---
# Игнорируем предупреждение об устаревшем pkg_resources от pandas_ta
# Фильтруем по тексту, так как это UserWarning, а не DeprecationWarning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
# --- КОНЕЦ БЛОКА --


logger = get_logger(__name__)


async def main():
  setup_logging(settings.LOG_LEVEL)
  setup_signal_logger()
  logger.info("Запуск торгового бота с оптимизациями...")

  trading_system = IntegratedTradingSystem()

  # Обработчик для корректного завершения
  # loop = asyncio.get_event_loop()
  stop_event = asyncio.Event()

  if not os.path.exists("ml_models/anomaly_detector.pkl"):
    logger.info("Обучение детектора аномалий...")
    # Получаем топ символы для обучения
    await trading_system.connector.sync_time()
    symbols = await trading_system.data_fetcher.get_active_symbols_by_volume(10)
    if symbols:
      await trading_system.train_anomaly_detector(symbols[:5], lookback_days=7)

  if not os.path.exists("ml_models/enhanced_model.pkl"):
    logger.info("Обучение Enhanced ML модели...")
    # Получаем топ символы для обучения
    await trading_system.connector.sync_time()
    symbols = await trading_system.data_fetcher.get_active_symbols_by_volume(10)
    if symbols:
      await trading_system.train_enhanced_ml_model(symbols[:5], lookback_days=7)

  def signal_handler():
    logger.info("Получен сигнал остановки. Завершение работы...")
    stop_event.set()

  if sys.platform != 'win32':
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
      loop.add_signal_handler(sig, signal_handler)
  else:
    # Для Windows используем отдельную обработку
    signal.signal(signal.SIGINT, lambda s, f: signal_handler())

  try:
    # Используем оптимизированный запуск
    await trading_system.start_optimized()  # Вместо start()

    while not stop_event.is_set() and trading_system.is_running:
      trading_system.display_balance()
      trading_system.display_active_symbols()

      # Выводим статистику производительности
      if hasattr(trading_system, '_monitoring_cycles') and trading_system._monitoring_cycles % 10 == 0:
        await trading_system._log_performance_stats()

      try:
        await asyncio.wait_for(stop_event.wait(), timeout=60)
      except asyncio.TimeoutError:
        pass

  except asyncio.CancelledError:
    logger.info("Основная задача была отменена.")
  except Exception as e:
    logger.critical(f"Критическая ошибка в основном цикле: {e}", exc_info=True)
  finally:
    logger.info("Начинаем процесс остановки торговой системы...")
    if trading_system.is_running:
      await trading_system.stop()
    logger.info("Торговый бот завершил работу.")


if __name__ == "__main__":
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    logger.info("Приложение прервано пользователем (KeyboardInterrupt).")
  except Exception as e:
    # Логируем неуловленные исключения на верхнем уровне
    setup_logging()  # Убедимся, что логгер настроен
    logger.critical(f"Неуловленное исключение на верхнем уровне: {e}", exc_info=True)