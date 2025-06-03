import asyncio
import signal  # Для корректной остановки
import sys
from utils.logging_config import setup_logging, get_logger
from config import settings
from core.integrated_system import IntegratedTradingSystem

logger = get_logger(__name__)


async def main():
  setup_logging(settings.LOG_LEVEL)
  logger.info("Запуск торгового бота...")

  trading_system = IntegratedTradingSystem()

  # Обработчик для корректного завершения
  # loop = asyncio.get_event_loop()
  stop_event = asyncio.Event()

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
    await trading_system.start()

    while not stop_event.is_set() and trading_system.is_running:
      trading_system.display_balance()
      trading_system.display_active_symbols()

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