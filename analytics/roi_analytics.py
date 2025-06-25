from typing import Dict, Any

from data.database_manager import AdvancedDatabaseManager


class ROIAnalytics:
  """Аналитика эффективности ROI настроек"""

  def __init__(self, db_manager: AdvancedDatabaseManager):
    self.db_manager = db_manager

  async def analyze_roi_performance(self, days: int = 30) -> Dict[str, Any]:
    """Анализ эффективности ROI за последние N дней"""

    # Получаем все закрытые сделки за период
    query = """
      SELECT * FROM trades 
      WHERE status = 'CLOSED' 
      AND close_timestamp >= datetime('now', '-{} days')
      ORDER BY close_timestamp DESC
      """.format(days)

    trades = await self.db_manager._execute(query, fetch='all')

    if not trades:
      return {'error': 'Нет данных за указанный период'}

    # Анализируем результаты
    total_trades = len(trades)
    profitable_trades = len([t for t in trades if t['profit_loss'] > 0])
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0

    total_pnl = sum(t['profit_loss'] for t in trades)
    avg_profit = sum(t['profit_loss'] for t in trades if t['profit_loss'] > 0) / max(profitable_trades, 1)
    avg_loss = sum(t['profit_loss'] for t in trades if t['profit_loss'] < 0) / max(total_trades - profitable_trades, 1)

    # Анализ достижения SL/TP целей
    sl_hits = len([t for t in trades if 'stop_loss_hit' in t.get('exit_reason', '')])
    tp_hits = len([t for t in trades if 'take_profit_hit' in t.get('exit_reason', '')])

    return {
      'period_days': days,
      'total_trades': total_trades,
      'win_rate': win_rate * 100,
      'total_pnl': total_pnl,
      'avg_profit': avg_profit,
      'avg_loss': avg_loss,
      'profit_factor': abs(avg_profit / avg_loss) if avg_loss != 0 else 0,
      'sl_hits': sl_hits,
      'tp_hits': tp_hits,
      'sl_hit_rate': sl_hits / total_trades * 100 if total_trades > 0 else 0,
      'tp_hit_rate': tp_hits / total_trades * 100 if total_trades > 0 else 0,
      'recommendation': self._generate_roi_recommendation(win_rate, avg_profit, avg_loss)
    }

  def _generate_roi_recommendation(self, win_rate: float, avg_profit: float, avg_loss: float) -> str:
    """Генерирует рекомендации по настройке ROI"""

    if win_rate < 0.4:
      return "Рекомендуется увеличить SL или уменьшить TP для повышения винрейта"
    elif win_rate > 0.7:
      return "Высокий винрейт - можно попробовать увеличить TP для большей прибыли"
    elif abs(avg_profit / avg_loss) < 1.5:
      return "Низкий profit factor - рассмотрите увеличение TP относительно SL"
    else:
      return "ROI настройки выглядят сбалансированными"