import asyncio
import os
import signal  # Для корректной остановки
import sys
from datetime import datetime

import aiohttp

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

async def generate_shadow_trading_reports(trading_system):
  """Генерация и вывод отчетов Shadow Trading"""

  if not trading_system.shadow_trading:
    return

  try:
    # Ежедневный отчет
    daily_report = await trading_system.shadow_trading.generate_daily_report()

    if 'error' not in daily_report:
      logger.info("📊 === ЕЖЕДНЕВНЫЙ ОТЧЕТ SHADOW TRADING ===")

      overall = daily_report.get('overall_performance', {})
      if overall:
        logger.info(f"🎯 Сигналов за день: {overall.get('total_signals', 0)}")
        logger.info(f"✅ Win Rate: {overall.get('win_rate_pct', 0)}%")
        logger.info(f"💰 Общий P&L: {overall.get('total_pnl_pct', 0):+.2f}%")
        logger.info(f"📈 Средняя прибыль: +{overall.get('avg_win_pct', 0)}%")
        logger.info(f"📉 Средний убыток: {overall.get('avg_loss_pct', 0)}%")
        logger.info(f"⚖️ Profit Factor: {overall.get('profit_factor', 0)}")
        logger.info(f"🚫 Отфильтровано: {overall.get('filtered_signals', 0)}")

      # Топ источники
      sources = daily_report.get('performance_by_source', [])
      if sources:
        logger.info("🏆 Лучшие источники сигналов:")
        for source in sources[:3]:
          logger.info(f"  • {source['source']}: WR {source['win_rate_pct']}% "
                      f"({source['total_signals']} сигналов, P&L: {source['total_pnl_pct']:+.1f}%)")

      # Рекомендации
      recommendations = await trading_system.shadow_trading.performance_analyzer.generate_optimization_recommendations(
        1)
      if 'error' not in recommendations:
        high_priority_recs = [r for r in recommendations.get('recommendations', []) if r['priority'] == 'high']
        if high_priority_recs:
          logger.info("🔴 ВАЖНЫЕ РЕКОМЕНДАЦИИ:")
          for rec in high_priority_recs[:2]:  # Топ 2
            logger.info(f"  • {rec['message']}")
            logger.info(f"    💡 {rec['suggested_action']}")

      logger.info("=" * 50)

  except Exception as e:
    logger.error(f"Ошибка генерации отчетов Shadow Trading: {e}")

logger = get_logger(__name__)


async def test_api_connectivity():
  """Тестирует подключение к API и свежесть данных"""
  try:
    logger.info("🧪 Тестирование подключения к Bybit API...")

    # Создаем временный connector для теста
    from core.bybit_connector import BybitConnector
    test_connector = BybitConnector()

    # Тестируем получение времени сервера
    server_time_url = test_connector.base_url + "/v5/market/time"
    async with aiohttp.ClientSession() as session:
      async with session.get(server_time_url) as response:
        if response.status == 200:
          data = await response.json()
          server_timestamp = int(data['result']['timeNano']) // 1_000_000
          server_time = datetime.fromtimestamp(server_timestamp / 1000)
          logger.info(f"✅ Время сервера Bybit: {server_time}")
        else:
          logger.error(f"❌ Ошибка получения времени сервера: {response.status}")

    # Тестируем получение свечей для BTCUSDT
    test_candles = await test_connector.get_kline("BTCUSDT", "60", limit=5)
    if test_candles:
      last_candle_timestamp = int(test_candles[0][0])
      last_candle_time = datetime.fromtimestamp(last_candle_timestamp / 1000)
      current_time = datetime.now()
      age_hours = (current_time - last_candle_time).total_seconds() / 3600

      logger.info(f"✅ Последняя свеча BTCUSDT: {last_candle_time}")
      logger.info(f"✅ Возраст данных: {age_hours:.1f} часов")

      if age_hours > 2:
        logger.error(f"🚨 API возвращает устаревшие данные! Возраст: {age_hours:.1f} часов")
      else:
        logger.info("✅ API возвращает свежие данные")
    else:
      logger.error("❌ Не удалось получить тестовые данные")

    await test_connector.close()

  except Exception as e:
    logger.error(f"Ошибка тестирования API: {e}")

async def main():
  setup_logging(settings.LOG_LEVEL)
  setup_signal_logger()
  logger.info("Запуск торгового бота с оптимизациями...")

  trading_system = IntegratedTradingSystem()
  await test_api_connectivity()
  # Обработчик для корректного завершения
  # loop = asyncio.get_event_loop()
  stop_event = asyncio.Event()

  if not os.path.exists("ml_models/anomaly_detector.pkl"):
    logger.info("Обучение детектора аномалий...")
    # Получаем топ символы для обучения
    await trading_system.connector.sync_time()
    symbols = await trading_system.data_fetcher.get_active_symbols_by_volume(200)
    if symbols:
      await trading_system.train_anomaly_detector(symbols[:100], lookback_days=60)

  if not os.path.exists("ml_models/enhanced_model.pkl"):
    logger.info("Обучение Enhanced ML модели...")
    # Получаем топ символы для обучения
    await trading_system.connector.sync_time()
    symbols = await trading_system.data_fetcher.get_active_symbols_by_volume(200)
    if symbols:
      await trading_system.train_enhanced_ml_model(symbols[:150], lookback_days=60)

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

    # НОВОЕ: Периодическая генерация отчетов Shadow Trading
    async def periodic_shadow_reports():
      while not stop_event.is_set():
        try:
          await asyncio.sleep(3600)  # Каждый час
          await generate_shadow_trading_reports(trading_system)
        except Exception as e:
          logger.error(f"Ошибка периодических отчетов: {e}")

    # Запускаем отчеты в фоне
    asyncio.create_task(periodic_shadow_reports())

    # Основной цикл мониторинга состояния
    report_counter = 0
    while not stop_event.is_set() and trading_system.is_running:
      # Основная работа происходит в фоновых задачах (_monitoring_loop_optimized и _fast_position_monitoring_loop)
      # Здесь только выводим статус каждую минуту

      if report_counter % 5 == 0:  # Каждые 5 минут
        trading_system.display_balance()
        trading_system.display_active_symbols()

      # Выводим статистику производительности
      if hasattr(trading_system, '_monitoring_cycles') and trading_system._monitoring_cycles % 10 == 0:
        await trading_system._log_performance_stats()

      # Периодический отчет Shadow Trading
      if report_counter % 30 == 0:  # Каждые 30 минут
        try:
          await generate_shadow_trading_reports(trading_system)
        except Exception as e:
          logger.error(f"Ошибка генерации отчета: {e}")

      report_counter += 1

      # Ждем 60 секунд или сигнал остановки
      try:
        await asyncio.wait_for(stop_event.wait(), timeout=60)
      except asyncio.TimeoutError:
        continue  # Продолжаем цикл

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