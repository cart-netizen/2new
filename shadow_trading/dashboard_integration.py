from typing import Dict, Any

from main import logger
from shadow_trading.shadow_trading_manager import ShadowTradingManager


class ShadowTradingDashboard:
  """Интеграция Shadow Trading с dashboard"""

  def __init__(self, shadow_manager: ShadowTradingManager):
    self.shadow_manager = shadow_manager

  def create_performance_summary(self, days: int = 7) -> str:
    """Создать краткую сводку производительности для dashboard"""
    try:
      import asyncio

      # Получаем данные асинхронно
      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)

      overall_perf = loop.run_until_complete(
        self.shadow_manager.performance_analyzer.get_overall_performance(days)
      )

      source_perf = loop.run_until_complete(
        self.shadow_manager.performance_analyzer.get_performance_by_source(days)
      )

      loop.close()

      if 'error' in overall_perf:
        return f"❌ Нет данных Shadow Trading за {days} дней"

      # Создаем краткую сводку
      summary_lines = [
        f"📊 **Shadow Trading за {days} дней:**",
        f"🎯 Сигналов: {overall_perf['total_signals']} (завершено: {overall_perf['completed_signals']})",
        f"✅ Win Rate: {overall_perf['win_rate_pct']}%",
        f"📈 Средняя прибыль: +{overall_perf['avg_win_pct']}%",
        f"📉 Средний убыток: {overall_perf['avg_loss_pct']}%",
        f"⚖️ Profit Factor: {overall_perf['profit_factor']}",
        ""
      ]

      # Топ источников
      if source_perf:
        summary_lines.append("🏆 **Лучшие источники:**")
        for source in source_perf[:3]:
          summary_lines.append(
            f"  • {source['source']}: {source['win_rate_pct']}% "
            f"({source['total_signals']} сигналов)"
          )

      return "\n".join(summary_lines)

    except Exception as e:
      logger.error(f"Ошибка создания сводки для dashboard: {e}")
      return f"❌ Ошибка получения данных Shadow Trading: {e}"

  def create_signal_chart_data(self, days: int = 7) -> Dict[str, Any]:
    """Создать данные для графиков сигналов"""
    try:
      import asyncio

      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)

      hourly_perf = loop.run_until_complete(
        self.shadow_manager.performance_analyzer.get_hourly_performance(days)
      )

      loop.close()

      # Преобразуем в формат для графика
      hours = list(range(24))
      win_rates = []
      signal_counts = []

      for hour in hours:
        if hour in hourly_perf:
          win_rates.append(hourly_perf[hour]['win_rate_pct'])
          signal_counts.append(hourly_perf[hour]['total_signals'])
        else:
          win_rates.append(0)
          signal_counts.append(0)

      return {
        'hours': hours,
        'win_rates': win_rates,
        'signal_counts': signal_counts
      }

    except Exception as e:
      logger.error(f"Ошибка создания данных графика: {e}")
      return {'hours': [], 'win_rates': [], 'signal_counts': []}