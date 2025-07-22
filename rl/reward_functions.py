# rl/reward_functions.py

import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseRewardFunction(ABC):
  """Базовый класс для функций вознаграждения"""

  @abstractmethod
  def calculate_reward(self, pnl: float, metrics: Dict[str, Any]) -> float:
    """Рассчитать вознаграждение"""
    pass


class RiskAdjustedRewardFunction(BaseRewardFunction):
  """
  Функция вознаграждения с учетом рисков
  Интегрирована с RiskManager проекта
  """

  def __init__(
      self,
      risk_manager=None,
      risk_free_rate: float = 0.02,
      config: Dict[str, Any] = None
  ):
    self.risk_manager = risk_manager
    self.risk_free_rate = risk_free_rate / 252  # Дневная безрисковая ставка
    self.config = config or {}

    # Веса компонентов вознаграждения
    self.weights = {
      'pnl': self.config.get('pnl_weight', 1.0),
      'sharpe': self.config.get('sharpe_weight', 0.3),
      'risk_penalty': self.config.get('risk_penalty_weight', 0.5),
      'drawdown_penalty': self.config.get('drawdown_penalty_weight', 0.4),
      'consistency_bonus': self.config.get('consistency_bonus_weight', 0.2)
    }

    # История для расчета метрик
    self.returns_history = []
    self.max_portfolio_value = 0

  def calculate_reward(self, pnl: float, metrics: Dict[str, Any]) -> float:
    """
    Расчет вознаграждения с учетом риска

    Args:
        pnl: Прибыль/убыток от последней операции
        metrics: Словарь с метриками риска

    Returns:
        Скорректированное вознаграждение
    """
    try:
      # Обновляем историю
      self.returns_history.append(pnl)
      if len(self.returns_history) > 100:
        self.returns_history.pop(0)

      # Базовое вознаграждение - PnL
      base_reward = pnl * self.weights['pnl']

      # Компонент Sharpe Ratio
      sharpe_component = self._calculate_sharpe_component()

      # Штрафы за риски
      risk_penalty = self._calculate_risk_penalty(metrics)

      # Штраф за просадку
      drawdown_penalty = self._calculate_drawdown_penalty(metrics)

      # Бонус за консистентность
      consistency_bonus = self._calculate_consistency_bonus()

      # Итоговое вознаграждение
      total_reward = (
          base_reward +
          sharpe_component -
          risk_penalty -
          drawdown_penalty +
          consistency_bonus
      )
      # Ограничиваем награду разумными пределами
      total_reward = np.clip(total_reward, -1000, 1000)

      # Проверяем на NaN и Inf
      if np.isnan(total_reward) or np.isinf(total_reward):
        logger.warning(f"Некорректная награда: {total_reward}, заменяем на PnL")
        return pnl

      # Логирование для отладки
      if abs(total_reward) > 100:  # Большие вознаграждения
        logger.debug(f"Reward calculation: base={base_reward:.2f}, "
                     f"sharpe={sharpe_component:.2f}, risk_pen={risk_penalty:.2f}, "
                     f"dd_pen={drawdown_penalty:.2f}, consistency={consistency_bonus:.2f}")

      return total_reward

    except Exception as e:
      logger.error(f"Ошибка расчета вознаграждения: {e}")
      return pnl  # Возвращаем простой PnL в случае ошибки

  def _calculate_sharpe_component(self) -> float:
    """Рассчитывает компонент Sharpe Ratio"""
    if len(self.returns_history) < 5:
      return 0

    returns = np.array(self.returns_history)
    excess_returns = returns - self.risk_free_rate

    if np.std(excess_returns) > 0:
      sharpe = np.mean(excess_returns) / np.std(excess_returns)
      return sharpe * self.weights['sharpe']

    return 0

  def _calculate_risk_penalty(self, metrics: Dict[str, Any]) -> float:
    """Рассчитывает штраф за превышение риска"""
    penalty = 0

    # Штраф за превышение VaR
    if metrics.get('var_exceeded', False):
      penalty += abs(metrics.get('portfolio_value', 10000)) * 0.02

    # Штраф за высокий leverage
    leverage = metrics.get('leverage_ratio', 0)
    if leverage > 5:
      penalty += (leverage - 5) * 100

    # Штраф за концентрацию позиций
    concentration = metrics.get('position_concentration', 0)
    if concentration > 0.3:  # Более 30% в одной позиции
      penalty += (concentration - 0.3) * 1000

    # Штраф за нарушение корреляции
    if metrics.get('correlation_breach', False):
      penalty += 500

    return penalty * self.weights['risk_penalty']

  def _calculate_drawdown_penalty(self, metrics: Dict[str, Any]) -> float:
    """Рассчитывает штраф за просадку"""
    drawdown = metrics.get('drawdown', 0)

    # Прогрессивный штраф
    if drawdown > 0.05:  # 5%
      penalty = 100 * (drawdown - 0.05)
    elif drawdown > 0.10:  # 10%
      penalty = 500 * (drawdown - 0.10)
    elif drawdown > 0.20:  # 20%
      penalty = 2000 * (drawdown - 0.20)
    else:
      penalty = 0

    return penalty * self.weights['drawdown_penalty']

  def _calculate_consistency_bonus(self) -> float:
    """Рассчитывает бонус за консистентную прибыльность"""
    if len(self.returns_history) < 10:
      return 0

    recent_returns = self.returns_history[-10:]
    winning_trades = sum(1 for r in recent_returns if r > 0)
    win_rate = winning_trades / len(recent_returns)

    # Бонус за высокий win rate
    if win_rate > 0.6:
      bonus = (win_rate - 0.6) * 1000
    else:
      bonus = 0

    # Дополнительный бонус за последовательные выигрыши
    consecutive_wins = 0
    for r in reversed(recent_returns):
      if r > 0:
        consecutive_wins += 1
      else:
        break

    if consecutive_wins >= 3:
      bonus += consecutive_wins * 50

    return bonus * self.weights['consistency_bonus']


class SharpeOptimizationReward(BaseRewardFunction):
  """Функция вознаграждения, оптимизирующая Sharpe Ratio"""

  def __init__(self, lookback_period: int = 30):
    self.lookback_period = lookback_period
    self.returns_buffer = []

  def calculate_reward(self, pnl: float, metrics: Dict[str, Any]) -> float:
    """Вознаграждение на основе улучшения Sharpe Ratio"""
    self.returns_buffer.append(pnl)

    if len(self.returns_buffer) > self.lookback_period:
      self.returns_buffer.pop(0)

    if len(self.returns_buffer) < 5:
      return pnl

    # Рассчитываем текущий Sharpe
    returns = np.array(self.returns_buffer)
    current_sharpe = np.mean(returns) / (np.std(returns) + 1e-10)

    # Рассчитываем Sharpe без последней сделки
    prev_returns = returns[:-1]
    prev_sharpe = np.mean(prev_returns) / (np.std(prev_returns) + 1e-10)

    # Вознаграждение за улучшение Sharpe
    sharpe_improvement = current_sharpe - prev_sharpe

    return pnl + sharpe_improvement * 1000


class PortfolioOptimizationReward(BaseRewardFunction):
  """Функция вознаграждения для оптимизации портфеля"""

  def __init__(self, target_weights: Dict[str, float] = None):
    self.target_weights = target_weights or {}

  def calculate_reward(self, pnl: float, metrics: Dict[str, Any]) -> float:
    """Вознаграждение с учетом отклонения от целевых весов портфеля"""
    base_reward = pnl

    # Штраф за отклонение от целевых весов
    if self.target_weights and 'portfolio_weights' in metrics:
      current_weights = metrics['portfolio_weights']

      deviation_penalty = 0
      for asset, target_weight in self.target_weights.items():
        current_weight = current_weights.get(asset, 0)
        deviation = abs(current_weight - target_weight)
        deviation_penalty += deviation * 100

      base_reward -= deviation_penalty

    # Бонус за диверсификацию
    if 'portfolio_weights' in metrics:
      weights = list(metrics['portfolio_weights'].values())
      if weights:
        # Используем энтропию как меру диверсификации
        weights = np.array(weights)
        weights = weights[weights > 0]  # Убираем нулевые веса
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        diversity_bonus = entropy * 100
        base_reward += diversity_bonus

    return base_reward