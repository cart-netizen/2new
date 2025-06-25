# core/adaptive_strategy_selector.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

from core.schemas import TradingSignal
from core.enums import SignalType
from strategies.base_strategy import BaseStrategy
from data.database_manager import AdvancedDatabaseManager
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StrategyPerformance:
  """Метрики производительности стратегии"""
  strategy_name: str
  total_trades: int = 0
  winning_trades: int = 0
  losing_trades: int = 0
  total_profit: float = 0.0
  total_loss: float = 0.0
  win_rate: float = 0.0
  profit_factor: float = 0.0
  sharpe_ratio: float = 0.0
  max_drawdown: float = 0.0
  avg_profit_per_trade: float = 0.0
  last_update: datetime = field(default_factory=datetime.now)

  # Адаптивные веса
  current_weight: float = 1.0
  weight_history: List[float] = field(default_factory=list)

  # Производительность по режимам
  performance_by_regime: Dict[str, Dict[str, float]] = field(default_factory=dict)

  # Скользящие метрики
  recent_trades: deque = field(default_factory=lambda: deque(maxlen=20))
  recent_win_rate: float = 0.0

  def update_metrics(self):
    """Обновляет расчетные метрики"""
    if self.total_trades > 0:
      self.win_rate = self.winning_trades / self.total_trades
      self.avg_profit_per_trade = (self.total_profit - self.total_loss) / self.total_trades

      if self.total_loss > 0:
        self.profit_factor = abs(self.total_profit / self.total_loss)
      else:
        self.profit_factor = float('inf') if self.total_profit > 0 else 0

      # Обновляем скользящий win rate
      if len(self.recent_trades) > 0:
        recent_wins = sum(1 for trade in self.recent_trades if trade['profit'] > 0)
        self.recent_win_rate = recent_wins / len(self.recent_trades)


class AdaptiveStrategySelector:
  """
  Адаптивный селектор стратегий, который динамически управляет
  активными стратегиями и их весами на основе производительности
  """

  def __init__(self, db_manager: AdvancedDatabaseManager,
               min_trades_for_evaluation: int = 10,
               weight_update_interval: timedelta = timedelta(hours=24)):
    self.db_manager = db_manager
    self.min_trades_for_evaluation = min_trades_for_evaluation
    self.weight_update_interval = weight_update_interval

    # Производительность стратегий
    self.strategy_performance: Dict[str, StrategyPerformance] = {}

    # Активные стратегии
    self.active_strategies: Dict[str, bool] = {}

    # Параметры адаптации
    self.adaptation_config = {
      'min_weight': 0.1,
      'max_weight': 2.0,
      'weight_change_rate': 0.1,
      'performance_window': 50,  # trades
      'disable_threshold': 0.3,  # win rate
      'enable_threshold': 0.5,  # win rate
      'regime_weight_bonus': 0.2  # бонус для стратегий в подходящем режиме
    }

    # История адаптаций
    self.adaptation_history = deque(maxlen=1000)

    # Загружаем историческую производительность
    self._load_historical_performance()

  def _load_historical_performance(self):
    """Загружает историческую производительность из БД"""
    try:
      # Получаем закрытые сделки за последние 30 дней
      end_date = datetime.now()
      start_date = end_date - timedelta(days=30)

      query = """
            SELECT strategy_name, open_price, close_price, 
                   profit_loss, status, close_timestamp, metadata
            FROM trades 
            WHERE status = 'CLOSED' 
            AND close_timestamp BETWEEN ? AND ?
            ORDER BY close_timestamp DESC
        """

      # Используем синхронный метод
      trades = self.db_manager.execute_query_sync(
        query,
        (start_date.isoformat(), end_date.isoformat())
      )

      if not trades:
        logger.info("Нет исторических данных для загрузки")
        return

      # Группируем по стратегиям
      for trade in trades:
        strategy_name = trade.get('strategy_name', 'Unknown')
        profit_loss = trade.get('profit_loss', 0)
        metadata_str = trade.get('metadata', '{}')

        # Безопасный парсинг JSON
        try:
          metadata = json.loads(metadata_str) if metadata_str else {}
        except json.JSONDecodeError:
          metadata = {}

        if strategy_name not in self.strategy_performance:
          self.strategy_performance[strategy_name] = StrategyPerformance(strategy_name)

        perf = self.strategy_performance[strategy_name]
        perf.total_trades += 1

        if profit_loss > 0:
          perf.winning_trades += 1
          perf.total_profit += profit_loss
        else:
          perf.losing_trades += 1
          perf.total_loss += abs(profit_loss)

        # Добавляем в recent_trades
        perf.recent_trades.append({
          'profit': profit_loss,
          'timestamp': trade.get('close_timestamp'),
          'regime': metadata.get('regime', 'unknown')
        })

        # Обновляем производительность по режимам
        regime = metadata.get('regime', 'unknown')
        if regime not in perf.performance_by_regime:
          perf.performance_by_regime[regime] = {
            'trades': 0, 'wins': 0, 'profit': 0
          }

        perf.performance_by_regime[regime]['trades'] += 1
        if profit_loss > 0:
          perf.performance_by_regime[regime]['wins'] += 1
        perf.performance_by_regime[regime]['profit'] += profit_loss

      # Обновляем метрики
      for perf in self.strategy_performance.values():
        perf.update_metrics()
        # Инициализируем активность на основе производительности
        self.active_strategies[perf.strategy_name] = perf.win_rate > 0.4

      logger.info(f"Загружена производительность для {len(self.strategy_performance)} стратегий")

    except Exception as e:
      logger.error(f"Ошибка загрузки исторической производительности: {e}")

  def update_strategy_performance(self, strategy_name: str, trade_result: Dict[str, Any]):
    """Обновляет производительность стратегии после закрытия сделки"""
    try:
      if strategy_name not in self.strategy_performance:
        self.strategy_performance[strategy_name] = StrategyPerformance(strategy_name)

      perf = self.strategy_performance[strategy_name]
      profit_loss = trade_result.get('profit_loss', 0)

      # Обновляем базовые метрики
      perf.total_trades += 1
      if profit_loss > 0:
        perf.winning_trades += 1
        perf.total_profit += profit_loss
      else:
        perf.losing_trades += 1
        perf.total_loss += abs(profit_loss)

      # Добавляем в recent_trades
      perf.recent_trades.append({
        'profit': profit_loss,
        'timestamp': datetime.now(),
        'regime': trade_result.get('regime', 'unknown')
      })

      # Обновляем производительность по режимам
      regime = trade_result.get('regime', 'unknown')
      if regime not in perf.performance_by_regime:
        perf.performance_by_regime[regime] = {
          'trades': 0, 'wins': 0, 'profit': 0
        }

      perf.performance_by_regime[regime]['trades'] += 1
      if profit_loss > 0:
        perf.performance_by_regime[regime]['wins'] += 1
      perf.performance_by_regime[regime]['profit'] += profit_loss

      # Пересчитываем метрики
      perf.update_metrics()
      perf.last_update = datetime.now()

      # Проверяем необходимость адаптации
      if self._should_adapt_weights(strategy_name):
        self._adapt_strategy_weight(strategy_name)

      logger.info(f"Обновлена производительность {strategy_name}: "
                  f"WR={perf.win_rate:.2f}, PF={perf.profit_factor:.2f}")

    except Exception as e:
      logger.error(f"Ошибка обновления производительности {strategy_name}: {e}")

  def _should_adapt_weights(self, strategy_name: str) -> bool:
    """Проверяет, нужно ли адаптировать вес стратегии"""
    perf = self.strategy_performance.get(strategy_name)
    if not perf:
      return False

    # Адаптируем после каждых N сделок
    if perf.total_trades % 10 == 0:
      return True

    # Или если прошло достаточно времени
    if datetime.now() - perf.last_update > self.weight_update_interval:
      return True

    return False

  def _adapt_strategy_weight(self, strategy_name: str):
    """Адаптирует вес стратегии на основе производительности"""
    perf = self.strategy_performance[strategy_name]

    # Базовый вес на основе общей производительности
    base_weight = 1.0

    # Корректировка на основе win rate
    if perf.win_rate > 0.6:
      base_weight *= 1.2
    elif perf.win_rate < 0.4:
      base_weight *= 0.8

    # Корректировка на основе profit factor
    if perf.profit_factor > 1.5:
      base_weight *= 1.1
    elif perf.profit_factor < 0.8:
      base_weight *= 0.9

    # Корректировка на основе recent performance
    if perf.recent_win_rate > perf.win_rate * 1.2:
      base_weight *= 1.1  # Улучшается
    elif perf.recent_win_rate < perf.win_rate * 0.8:
      base_weight *= 0.9  # Ухудшается

    # Ограничиваем изменение
    old_weight = perf.current_weight
    change_limit = self.adaptation_config['weight_change_rate']
    new_weight = old_weight + np.clip(base_weight - old_weight, -change_limit, change_limit)

    # Применяем границы
    new_weight = np.clip(new_weight,
                         self.adaptation_config['min_weight'],
                         self.adaptation_config['max_weight'])

    perf.current_weight = new_weight
    perf.weight_history.append(new_weight)

    # Записываем в историю
    self.adaptation_history.append({
      'timestamp': datetime.now(),
      'strategy': strategy_name,
      'old_weight': old_weight,
      'new_weight': new_weight,
      'reason': f"WR={perf.win_rate:.2f}, PF={perf.profit_factor:.2f}"
    })

    logger.info(f"Адаптирован вес {strategy_name}: {old_weight:.2f} -> {new_weight:.2f}")

  def should_activate_strategy(self, strategy_name: str, current_regime: str) -> bool:
    """Определяет, должна ли быть активна стратегия"""
    # Проверяем базовую активность
    if strategy_name not in self.active_strategies:
      self.active_strategies[strategy_name] = True

    if not self.active_strategies[strategy_name]:
      # Проверяем условия для реактивации
      perf = self.strategy_performance.get(strategy_name)
      if perf and perf.recent_win_rate > self.adaptation_config['enable_threshold']:
        self.active_strategies[strategy_name] = True
        logger.info(f"Реактивирована стратегия {strategy_name}")

    # Проверяем производительность в текущем режиме
    if current_regime and strategy_name in self.strategy_performance:
      perf = self.strategy_performance[strategy_name]
      regime_perf = perf.performance_by_regime.get(current_regime, {})

      if regime_perf.get('trades', 0) > 5:
        regime_win_rate = regime_perf.get('wins', 0) / regime_perf.get('trades', 1)
        if regime_win_rate < self.adaptation_config['disable_threshold']:
          logger.warning(f"Стратегия {strategy_name} плохо работает в режиме {current_regime}")
          return False

    return self.active_strategies.get(strategy_name, True)

  def get_strategy_weight(self, strategy_name: str, current_regime: Optional[str] = None) -> float:
    """Получает текущий вес стратегии"""
    if strategy_name not in self.strategy_performance:
      return 1.0

    weight = self.strategy_performance[strategy_name].current_weight

    # Бонус за хорошую производительность в текущем режиме
    if current_regime:
      perf = self.strategy_performance[strategy_name]
      regime_perf = perf.performance_by_regime.get(current_regime, {})

      if regime_perf.get('trades', 0) > 5:
        regime_win_rate = regime_perf.get('wins', 0) / regime_perf.get('trades', 1)
        if regime_win_rate > 0.6:
          weight += self.adaptation_config['regime_weight_bonus']

    return min(weight, self.adaptation_config['max_weight'])

  def disable_poorly_performing_strategies(self):
    """Отключает плохо работающие стратегии"""
    for strategy_name, perf in self.strategy_performance.items():
      if perf.total_trades >= self.min_trades_for_evaluation:
        if perf.recent_win_rate < self.adaptation_config['disable_threshold']:
          if self.active_strategies.get(strategy_name, True):
            self.active_strategies[strategy_name] = False
            logger.warning(f"Отключена стратегия {strategy_name}: "
                           f"recent WR={perf.recent_win_rate:.2f}")

            # Записываем в историю
            self.adaptation_history.append({
              'timestamp': datetime.now(),
              'strategy': strategy_name,
              'action': 'disabled',
              'reason': f"Low recent win rate: {perf.recent_win_rate:.2f}"
            })

  def get_performance_summary(self) -> Dict[str, Any]:
    """Возвращает сводку производительности всех стратегий"""
    summary = {}

    for strategy_name, perf in self.strategy_performance.items():
      summary[strategy_name] = {
        'active': self.active_strategies.get(strategy_name, True),
        'weight': perf.current_weight,
        'total_trades': perf.total_trades,
        'win_rate': round(perf.win_rate, 3),
        'recent_win_rate': round(perf.recent_win_rate, 3),
        'profit_factor': round(perf.profit_factor, 2),
        'total_profit': round(perf.total_profit - perf.total_loss, 2),
        'last_update': perf.last_update.isoformat()
      }

    return summary

  def export_adaptation_history(self, filepath: str):
    """Экспортирует историю адаптаций"""
    try:
      # Создаем директорию если не существует
      import os
      dir_path = os.path.dirname(filepath)
      if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

      df = pd.DataFrame(list(self.adaptation_history))
      df.to_csv(filepath, index=False)
      logger.info(f"История адаптаций экспортирована в {filepath}")
    except Exception as e:
      logger.error(f"Ошибка экспорта истории адаптаций: {e}")