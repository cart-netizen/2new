# rl/portfolio_manager.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioState:
  """Состояние портфеля"""
  total_value: float
  cash: float
  positions: Dict[str, float]  # symbol -> количество
  weights: Dict[str, float]  # symbol -> вес в портфеле
  leverage: float
  unrealized_pnl: float
  realized_pnl: float
  timestamp: datetime


class RLPortfolioManager:
  """
  Менеджер портфеля для RL стратегии
  Управляет распределением капитала и ребалансировкой
  """

  def __init__(
      self,
      initial_capital: float,
      risk_manager=None,
      config: Dict[str, Any] = None
  ):
    self.initial_capital = initial_capital
    self.risk_manager = risk_manager
    self.config = config or {}

    # Текущее состояние портфеля
    self.current_cash = initial_capital
    self.positions = {}  # symbol -> {quantity, entry_price, current_price}
    self.trade_history = []

    # Параметры управления
    self.max_positions = self.config.get('max_positions', 10)
    self.min_position_size = self.config.get('min_position_size', 0.01)  # 1% минимум
    self.max_position_size = self.config.get('max_position_size', 0.2)  # 20% максимум
    self.rebalance_threshold = self.config.get('rebalance_threshold', 0.1)  # 10% отклонение

    # Целевые веса (если заданы)
    self.target_weights = self.config.get('target_weights', {})

    # История состояний
    self.state_history = []
    self._save_state()

  def calculate_position_size(
      self,
      symbol: str,
      signal_confidence: float,
      current_price: float,
      volatility: Optional[float] = None
  ) -> float:
    """
    Рассчитывает размер позиции с учетом риска и уверенности
    """
    try:
      # Базовый размер на основе уверенности
      base_size = self.current_cash * (self.min_position_size +
                                       (self.max_position_size - self.min_position_size) * signal_confidence)

      # Корректировка на основе волатильности
      if volatility and volatility > 0:
        # Чем выше волатильность, тем меньше позиция
        volatility_multiplier = 1 / (1 + volatility * 10)
        base_size *= volatility_multiplier

      # Проверка риск-менеджмента
      if self.risk_manager:
        risk_adjusted_size = self.risk_manager.calculate_position_size(
          symbol=symbol,
          entry_price=current_price,
          stop_loss=current_price * 0.98,  # 2% стоп по умолчанию
          account_balance=self.get_total_value()
        )
        base_size = min(base_size, risk_adjusted_size)

      # Проверка лимитов
      max_allowed = self.current_cash * self.max_position_size
      min_allowed = self.current_cash * self.min_position_size

      position_size = np.clip(base_size, min_allowed, max_allowed)

      # Проверка количества позиций
      if len(self.positions) >= self.max_positions:
        # Если достигнут лимит, уменьшаем размер
        position_size *= 0.5

      return position_size

    except Exception as e:
      logger.error(f"Ошибка расчета размера позиции: {e}")
      return self.current_cash * self.min_position_size

  def open_position(
      self,
      symbol: str,
      quantity: float,
      entry_price: float,
      position_type: str = 'LONG'
  ) -> bool:
    """Открывает новую позицию"""
    try:
      required_capital = quantity * entry_price

      # Проверяем достаточность средств
      if required_capital > self.current_cash:
        logger.warning(f"Недостаточно средств для открытия позиции {symbol}: "
                       f"требуется {required_capital}, доступно {self.current_cash}")
        return False

        # Открываем позицию
      self.positions[symbol] = {
        'quantity': quantity,
        'entry_price': entry_price,
        'current_price': entry_price,
        'position_type': position_type,
        'open_time': datetime.now(),
        'unrealized_pnl': 0
      }

      # Уменьшаем доступные средства
      self.current_cash -= required_capital

      # Записываем в историю
      self.trade_history.append({
        'timestamp': datetime.now(),
        'action': 'OPEN',
        'symbol': symbol,
        'quantity': quantity,
        'price': entry_price,
        'position_type': position_type,
        'cash_after': self.current_cash
      })

      # Сохраняем состояние
      self._save_state()

      logger.info(f"Открыта позиция {position_type} {symbol}: "
                  f"{quantity} @ {entry_price}, использовано {required_capital}")

      return True

    except Exception as e:
      logger.error(f"Ошибка открытия позиции {symbol}: {e}")
      return False

  def close_position(
      self,
      symbol: str,
      exit_price: float,
      partial_quantity: Optional[float] = None
  ) -> Dict[str, float]:
    """Закрывает позицию (полностью или частично)"""
    try:
      if symbol not in self.positions:
        logger.warning(f"Позиция {symbol} не найдена")
        return {'realized_pnl': 0, 'quantity_closed': 0}

      position = self.positions[symbol]

      # Определяем количество для закрытия
      if partial_quantity and partial_quantity < position['quantity']:
        quantity_to_close = partial_quantity
      else:
        quantity_to_close = position['quantity']

      # Рассчитываем PnL
      if position['position_type'] == 'LONG':
        pnl = (exit_price - position['entry_price']) * quantity_to_close
      else:  # SHORT
        pnl = (position['entry_price'] - exit_price) * quantity_to_close

      # Возвращаем средства
      returned_capital = quantity_to_close * exit_price
      self.current_cash += returned_capital

      # Обновляем или удаляем позицию
      if quantity_to_close >= position['quantity']:
        # Полное закрытие
        del self.positions[symbol]
      else:
        # Частичное закрытие
        position['quantity'] -= quantity_to_close

      # Записываем в историю
      self.trade_history.append({
        'timestamp': datetime.now(),
        'action': 'CLOSE',
        'symbol': symbol,
        'quantity': quantity_to_close,
        'price': exit_price,
        'pnl': pnl,
        'cash_after': self.current_cash
      })

      # Сохраняем состояние
      self._save_state()

      logger.info(f"Закрыта позиция {symbol}: {quantity_to_close} @ {exit_price}, PnL: {pnl:.2f}")

      return {
        'realized_pnl': pnl,
        'quantity_closed': quantity_to_close,
        'returned_capital': returned_capital
      }

    except Exception as e:
      logger.error(f"Ошибка закрытия позиции {symbol}: {e}")
      return {'realized_pnl': 0, 'quantity_closed': 0}

  def update_prices(self, current_prices: Dict[str, float]):
    """Обновляет текущие цены и unrealized PnL"""
    for symbol, position in self.positions.items():
      if symbol in current_prices:
        position['current_price'] = current_prices[symbol]

        # Рассчитываем unrealized PnL
        if position['position_type'] == 'LONG':
          position['unrealized_pnl'] = (
              (position['current_price'] - position['entry_price']) *
              position['quantity']
          )
        else:  # SHORT
          position['unrealized_pnl'] = (
              (position['entry_price'] - position['current_price']) *
              position['quantity']
          )

  def get_portfolio_weights(self) -> Dict[str, float]:
    """Возвращает текущие веса активов в портфеле"""
    total_value = self.get_total_value()

    if total_value == 0:
      return {}

    weights = {}

    # Вес наличных
    weights['CASH'] = self.current_cash / total_value

    # Веса позиций
    for symbol, position in self.positions.items():
      position_value = position['quantity'] * position['current_price']
      weights[symbol] = position_value / total_value

    return weights

  def needs_rebalancing(self) -> Tuple[bool, Dict[str, float]]:
    """Проверяет необходимость ребалансировки портфеля"""
    if not self.target_weights:
      return False, {}

    current_weights = self.get_portfolio_weights()
    deviations = {}

    max_deviation = 0
    for symbol, target_weight in self.target_weights.items():
      current_weight = current_weights.get(symbol, 0)
      deviation = abs(current_weight - target_weight)
      deviations[symbol] = current_weight - target_weight
      max_deviation = max(max_deviation, deviation)

    needs_rebalance = max_deviation > self.rebalance_threshold

    return needs_rebalance, deviations

  def calculate_rebalancing_trades(self) -> List[Dict[str, Any]]:
    """Рассчитывает сделки для ребалансировки портфеля"""
    needs_rebalance, deviations = self.needs_rebalancing()

    if not needs_rebalance:
      return []

    total_value = self.get_total_value()
    trades = []

    for symbol, deviation in deviations.items():
      if abs(deviation) < 0.01:  # Игнорируем малые отклонения
        continue

      target_value = self.target_weights[symbol] * total_value
      current_value = self.get_portfolio_weights().get(symbol, 0) * total_value

      value_difference = target_value - current_value

      # Нужно купить
      if value_difference > 0:
        trades.append({
          'symbol': symbol,
          'action': 'BUY',
          'value': value_difference,
          'reason': 'rebalance'
        })
      # Нужно продать
      elif value_difference < 0:
        trades.append({
          'symbol': symbol,
          'action': 'SELL',
          'value': abs(value_difference),
          'reason': 'rebalance'
        })

    return trades

  def get_total_value(self) -> float:
    """Возвращает общую стоимость портфеля"""
    total = self.current_cash

    for position in self.positions.values():
      total += position['quantity'] * position['current_price']

    return total

  def get_unrealized_pnl(self) -> float:
    """Возвращает общий unrealized PnL"""
    return sum(pos['unrealized_pnl'] for pos in self.positions.values())

  def get_realized_pnl(self) -> float:
    """Возвращает общий realized PnL из истории"""
    return sum(
      trade.get('pnl', 0)
      for trade in self.trade_history
      if trade['action'] == 'CLOSE'
    )

  def get_portfolio_metrics(self) -> Dict[str, Any]:
    """Возвращает метрики портфеля"""
    total_value = self.get_total_value()
    unrealized_pnl = self.get_unrealized_pnl()
    realized_pnl = self.get_realized_pnl()

    # Рассчитываем доходность
    total_return = (total_value - self.initial_capital) / self.initial_capital

    # Рассчитываем Sharpe ratio если есть история
    sharpe_ratio = self._calculate_sharpe_ratio()

    # Рассчитываем максимальную просадку
    max_drawdown = self._calculate_max_drawdown()

    return {
      'total_value': total_value,
      'cash': self.current_cash,
      'positions_count': len(self.positions),
      'unrealized_pnl': unrealized_pnl,
      'realized_pnl': realized_pnl,
      'total_pnl': unrealized_pnl + realized_pnl,
      'total_return': total_return,
      'total_return_pct': total_return * 100,
      'sharpe_ratio': sharpe_ratio,
      'max_drawdown': max_drawdown,
      'portfolio_weights': self.get_portfolio_weights(),
      'leverage_used': self._calculate_leverage()
    }

  def _calculate_sharpe_ratio(self, periods: int = 30) -> float:
    """Рассчитывает Sharpe ratio"""
    if len(self.state_history) < periods:
      return 0

    # Получаем последние значения портфеля
    values = [state.total_value for state in self.state_history[-periods:]]

    if len(values) < 2:
      return 0

    # Рассчитываем доходности
    returns = np.diff(values) / values[:-1]

    if len(returns) == 0 or np.std(returns) == 0:
      return 0

    # Annualized Sharpe (предполагаем дневные данные)
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)

    return sharpe

  def _calculate_max_drawdown(self) -> float:
    """Рассчитывает максимальную просадку"""
    if not self.state_history:
      return 0

    values = [state.total_value for state in self.state_history]

    peak = values[0]
    max_dd = 0

    for value in values:
      if value > peak:
        peak = value

      drawdown = (peak - value) / peak
      max_dd = max(max_dd, drawdown)

    return max_dd

  def _calculate_leverage(self) -> float:
    """Рассчитывает используемый leverage"""
    if self.current_cash == 0:
      return 0

    total_position_value = sum(
      pos['quantity'] * pos['current_price']
      for pos in self.positions.values()
    )

    total_value = self.get_total_value()

    if total_value == 0:
      return 0

    return total_position_value / total_value

  def _save_state(self):
    """Сохраняет текущее состояние портфеля"""
    state = PortfolioState(
      total_value=self.get_total_value(),
      cash=self.current_cash,
      positions={s: p['quantity'] for s, p in self.positions.items()},
      weights=self.get_portfolio_weights(),
      leverage=self._calculate_leverage(),
      unrealized_pnl=self.get_unrealized_pnl(),
      realized_pnl=self.get_realized_pnl(),
      timestamp=datetime.now()
    )

    self.state_history.append(state)

    # Ограничиваем размер истории
    if len(self.state_history) > 1000:
      self.state_history.pop(0)

  def export_performance_report(self) -> pd.DataFrame:
    """Экспортирует отчет о производительности"""
    if not self.state_history:
      return pd.DataFrame()

    data = []
    for state in self.state_history:
      data.append({
        'timestamp': state.timestamp,
        'total_value': state.total_value,
        'cash': state.cash,
        'positions_count': len(state.positions),
        'leverage': state.leverage,
        'unrealized_pnl': state.unrealized_pnl,
        'realized_pnl': state.realized_pnl,
        'total_pnl': state.unrealized_pnl + state.realized_pnl
      })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    return df