import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from core.schemas import TradingSignal
from data.database_manager import AdvancedDatabaseManager
# from main import logger
from shadow_trading.signal_tracker import FilterReason, SignalTracker, PriceMonitor, PerformanceAnalyzer

from utils.logging_config import get_logger
logger = get_logger(__name__)

class ShadowTradingManager:
  """Главный менеджер системы Shadow Trading"""

  def __init__(self, db_manager: AdvancedDatabaseManager, data_fetcher,  config_path=None):
    self.db_manager = db_manager
    self.data_fetcher = data_fetcher

    # Загружаем конфигурацию Shadow Trading из папки config
    if config_path is None:
      config_path = "config/enhanced_shadow_trading_config.json"  # В папке config

    try:
      import json
      import os

      # Проверяем существование файла
      if not os.path.exists(config_path):
        logger.warning(f"Файл конфигурации Shadow Trading не найден: {config_path}")
        self.shadow_config = self._get_default_config()
      else:
        with open(config_path, 'r', encoding='utf-8') as f:
          full_config = json.load(f)
          self.shadow_config = full_config.get('enhanced_shadow_trading', {})
          logger.info(f"✅ Конфигурация Shadow Trading загружена из {config_path}")

    except Exception as e:
      logger.error(f"Ошибка загрузки конфигурации Shadow Trading: {e}")
      self.shadow_config = self._get_default_config()

    # Инициализация компонентов с настройками
    self.signal_tracker = SignalTracker(db_manager, self.shadow_config)
    self.price_monitor = PriceMonitor(self.signal_tracker, data_fetcher)
    # self.price_monitor = PriceMonitor(self.signal_tracker, data_fetcher, self.shadow_config)
    # self.performance_analyzer = PerformanceAnalyzer(db_manager, self.shadow_config)
    self.performance_analyzer = PerformanceAnalyzer(db_manager)

    # Применяем настройки из конфигурации
    self._apply_config_settings()

    self.is_active = False

  def _apply_config_settings(self):
    """Применяет настройки из конфигурации"""
    # Основные настройки
    self.is_enabled = self.shadow_config.get('enabled', True)

    # Настройки мониторинга
    monitoring_config = self.shadow_config.get('monitoring', {})
    self.price_update_interval = monitoring_config.get('price_update_interval_seconds', 30)
    self.signal_tracking_duration = monitoring_config.get('signal_tracking_duration_hours', 24)
    self.max_concurrent_tracking = monitoring_config.get('max_concurrent_tracking', 1000)

    # Настройки алертов
    alerts_config = self.shadow_config.get('alerts', {})
    self.auto_alerts_enabled = alerts_config.get('enabled', True)
    self.alert_cooldown_minutes = alerts_config.get('cooldown_minutes', 60)

    # Настройки отчетов
    reporting_config = self.shadow_config.get('reporting', {})
    self.auto_reports_enabled = reporting_config.get('auto_reports_enabled', True)
    self.daily_summary_time = reporting_config.get('daily_summary_time_utc', 9)

    # Настройки оптимизации
    optimization_config = self.shadow_config.get('optimization', {})
    self.auto_optimization_enabled = optimization_config.get('auto_optimization_enabled', True)
    self.optimization_frequency_hours = optimization_config.get('optimization_frequency_hours', 24)

    logger.info(f"Shadow Trading настройки применены: enabled={self.is_enabled}")

  def _get_default_config(self):
    """Конфигурация по умолчанию"""
    return {
      "enabled": True,
      "monitoring": {"price_update_interval_seconds": 30},
      "alerts": {"enabled": True},
      "reporting": {"auto_reports_enabled": True},
      "optimization": {"optimization_frequency_hours": 24}
    }

  async def start_shadow_trading(self):
    """Запуск системы Shadow Trading"""
    if self.is_active:
      logger.warning("Shadow Trading уже активен")
      return

    try:
      await self.price_monitor.start_monitoring()
      self.is_active = True
      logger.info("🌟 Shadow Trading система запущена")

    except Exception as e:
      logger.error(f"Ошибка запуска Shadow Trading: {e}")

  async def start_enhanced_monitoring(self):
      """Запуск расширенного мониторинга"""
      await self.start_shadow_trading()  # Ваш существующий метод

      # Запускаем дополнительные фоновые задачи
      if self.auto_reports_enabled:
        asyncio.create_task(self._auto_reporting_loop())

      if self.auto_alerts_enabled:
        asyncio.create_task(self._auto_alert_loop())

      asyncio.create_task(self._periodic_optimization_loop())

      logger.info("🚀 Enhanced Shadow Trading система запущена")

  async def _auto_reporting_loop(self):
    """Автоматическая генерация отчетов"""
    while self.is_active:
      try:
        # Ежедневный отчет в 9:00 UTC
        current_hour = datetime.now().hour
        if current_hour == 9:
          daily_report = await self.generate_daily_report()
          logger.info("📊 Ежедневный автоматический отчет:")
          logger.info(daily_report)

        await asyncio.sleep(3600)  # Ждем час

      except Exception as e:
        logger.error(f"Ошибка в автоматической отчетности: {e}")
        await asyncio.sleep(3600)

  async def _auto_alert_loop(self):
    """Автоматическая проверка алертов"""
    while self.is_active:
      try:
        # Простые алерты каждые 15 минут
        performance = await self.performance_analyzer.get_overall_performance(1)

        if 'error' not in performance:
          win_rate = performance.get('win_rate_pct', 0)
          if win_rate < 40 and performance.get('completed_signals', 0) >= 5:
            logger.warning(f"🚨 SHADOW TRADING ALERT: Низкий Win Rate {win_rate}%")

        await asyncio.sleep(900)  # 15 минут

      except Exception as e:
        logger.error(f"Ошибка в системе алертов: {e}")
        await asyncio.sleep(900)

  async def _periodic_optimization_loop(self):
    """Периодическая оптимизация параметров"""
    while self.is_active:
      try:
        # Простая оптимизация каждые 24 часа
        performance = await self.performance_analyzer.get_overall_performance(7)

        if 'error' not in performance:
          logger.info("🔧 Еженедельный анализ производительности:")
          logger.info(f"  Win Rate: {performance.get('win_rate_pct', 0)}%")
          logger.info(f"  Всего сигналов: {performance.get('total_signals', 0)}")
          logger.info(f"  Profit Factor: {performance.get('profit_factor', 0)}")

        await asyncio.sleep(self.optimization_frequency_hours * 3600)

      except Exception as e:
        logger.error(f"Ошибка в периодической оптимизации: {e}")
        await asyncio.sleep(3600)

  async def force_comprehensive_report(self) -> str:
    """Принудительная генерация комплексного отчета"""
    try:
      # Получаем основные данные
      overall_perf = await self.performance_analyzer.get_overall_performance(7)
      source_perf = await self.performance_analyzer.get_performance_by_source(7)

      # Создаем текстовый отчет
      lines = [
        "=" * 60,
        "📊 КОМПЛЕКСНЫЙ ОТЧЕТ SHADOW TRADING",
        "=" * 60,
        f"Сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "🎯 ОСНОВНЫЕ ПОКАЗАТЕЛИ:"
      ]

      if 'error' not in overall_perf:
        lines.extend([
          f"  • Всего сигналов: {overall_perf.get('total_signals', 0)}",
          f"  • Win Rate: {overall_perf.get('win_rate_pct', 0)}%",
          f"  • Общий P&L: {overall_perf.get('total_pnl_pct', 0):+.2f}%",
          f"  • Profit Factor: {overall_perf.get('profit_factor', 0):.2f}",
          f"  • Отфильтровано: {overall_perf.get('filtered_signals', 0)}",
          ""
        ])

      # Топ источники
      if source_perf:
        lines.append("🏆 ТОП ИСТОЧНИКИ СИГНАЛОВ:")
        for source in source_perf[:3]:
          lines.append(
            f"  • {source['source']}: WR {source['win_rate_pct']}%, "
            f"P&L {source.get('total_pnl_pct', 0):+.1f}% ({source['total_signals']} сигналов)"
          )
        lines.append("")

      lines.extend([
        "=" * 60,
        "Отчет сгенерирован автоматически системой Shadow Trading",
        "=" * 60
      ])

      return "\n".join(lines)

    except Exception as e:
      logger.error(f"Ошибка генерации комплексного отчета: {e}")
      return f"Ошибка генерации отчета: {e}"

  async def stop_shadow_trading(self):
    """Остановка системы"""
    if not self.is_active:
      return

    try:
      await self.price_monitor.stop_monitoring()
      self.is_active = False
      logger.info("🛑 Shadow Trading система остановлена")

    except Exception as e:
      logger.error(f"Ошибка остановки Shadow Trading: {e}")

  async def process_signal(self, signal: TradingSignal, metadata: Dict[str, Any] = None,
                           was_filtered: bool = False, filter_reasons: List[FilterReason] = None) -> str:
    """
    Обработать сигнал в системе Shadow Trading

    Args:
        signal: Торговый сигнал
        metadata: Метаданные сигнала
        was_filtered: Был ли сигнал отфильтрован
        filter_reasons: Причины фильтрации

    Returns:
        signal_id для отслеживания
    """
    try:
      # Начинаем отслеживание сигнала
      signal_id = await self.signal_tracker.track_signal(signal, metadata or {})

      if not signal_id:
        return ""

      # Отмечаем фильтрацию если нужно
      if was_filtered and filter_reasons:
        await self.signal_tracker.mark_signal_filtered(signal_id, filter_reasons)
      else:
        # Добавляем в мониторинг только неотфильтрованные сигналы
        await self.price_monitor.add_signal_for_monitoring(signal_id, signal.symbol)

      return signal_id

    except Exception as e:
      logger.error(f"Ошибка обработки сигнала в Shadow Trading: {e}")
      return ""

  async def generate_daily_report(self) -> Dict[str, Any]:
    """Создать ежедневный отчет"""
    try:
      # Общая производительность
      overall_perf = await self.performance_analyzer.get_overall_performance(days=1)

      # По источникам
      source_perf = await self.performance_analyzer.get_performance_by_source(days=1)

      # Анализ фильтров
      filter_analysis = await self.performance_analyzer.get_filter_analysis(days=1)

      # Почасовая статистика
      hourly_perf = await self.performance_analyzer.get_hourly_performance(days=1)

      report = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'overall_performance': overall_perf,
        'performance_by_source': source_perf,
        'filter_analysis': filter_analysis,
        'hourly_performance': hourly_perf,
        'generated_at': datetime.now().isoformat()
      }

      return report

    except Exception as e:
      logger.error(f"Ошибка создания ежедневного отчета: {e}")
      return {'error': str(e)}

  async def get_signal_details(self, signal_id: str) -> Optional[Dict[str, Any]]:
    """Получить детальную информацию о сигнале"""
    try:
      query = """
                SELECT * FROM signal_analysis WHERE signal_id = ?
            """
      result = await self.db_manager._execute(query, (signal_id,), fetch='one')

      if not result:
        return None

      # Получаем историю цен
      price_query = """
                SELECT price, timestamp, minutes_elapsed 
                FROM price_tracking 
                WHERE signal_id = ? 
                ORDER BY timestamp
            """
      price_history = await self.db_manager._execute(price_query, (signal_id,), fetch='all')

      # Конвертируем в удобный формат
      signal_details = dict(result)
      signal_details['price_history'] = [
        {
          'price': row['price'],
          'timestamp': row['timestamp'],
          'minutes_elapsed': row['minutes_elapsed']
        }
        for row in price_history
      ]

      # Парсим JSON поля
      try:
        signal_details['indicators_triggered'] = json.loads(signal_details['indicators_triggered'] or '[]')
        signal_details['ml_prediction_data'] = json.loads(signal_details['ml_prediction_data'] or '{}')
        signal_details['filter_reasons'] = json.loads(signal_details['filter_reasons'] or '[]')
      except json.JSONDecodeError:
        pass

      return signal_details

    except Exception as e:
      logger.error(f"Ошибка получения деталей сигнала {signal_id}: {e}")
      return None