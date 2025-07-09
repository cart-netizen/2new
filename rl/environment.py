# rl/environment.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from gymnasium import spaces
import logging

from core.enums import Timeframe, SignalType
from core.market_regime_detector import MarketRegime
from ml.feature_engineering import AdvancedFeatureEngineer
from utils.logging_config import get_logger

logger = get_logger(__name__)


class BybitTradingEnvironment(StockTradingEnv):
  """
  Кастомная среда для торговли на Bybit через FinRL
  Интегрирована с существующими компонентами проекта
  """

  def __init__(
      self,
      df: pd.DataFrame,
      data_fetcher,
      market_regime_detector,
      risk_manager,
      shadow_trading_manager,
      feature_engineer: AdvancedFeatureEngineer,
      initial_balance: float = 10000,
      commission_rate: float = 0.001,
      leverage: int = 10,
      max_positions: int = 5,
      config: Dict[str, Any] = None
  ):
    """
    Инициализация среды с интеграцией существующих компонентов
    """
    self.data_fetcher = data_fetcher
    self.market_regime_detector = market_regime_detector
    self.risk_manager = risk_manager
    self.shadow_trading_manager = shadow_trading_manager
    self.feature_engineer = feature_engineer
    self.leverage = leverage
    self.commission_rate = commission_rate
    self.max_positions = max_positions
    self.config = config or {}

    # Дополнительные параметры для Bybit
    self.funding_rate = 0.0001
    self.slippage = 0.0005

    # История для анализа
    self.trade_history = []
    self.regime_history = []
    self.shadow_signals = []

    # Получаем список технических индикаторов
    tech_indicators = self._get_technical_indicators_list()

    # Инициализируем начальные позиции для каждой акции
    num_stocks = len(df.tic.unique()) if 'tic' in df.columns else 1
    num_stock_shares = [0] * num_stocks  # Начальные позиции = 0

    # Инициализация родительского класса с ВСЕМИ требуемыми параметрами
    super().__init__(
      df=df,
      stock_dim=num_stocks,
      hmax=100,
      initial_amount=initial_balance,
      num_stock_shares=num_stock_shares,  # ДОБАВЛЕНО
      buy_cost_pct=self.config.get('buy_cost_pct', commission_rate),
      sell_cost_pct=self.config.get('sell_cost_pct', commission_rate),
      reward_scaling=self.config.get('reward_scaling', 1e-4),  # ДОБАВЛЕНО
      state_space=len(tech_indicators) + 10,
      action_space=3,
      tech_indicator_list=tech_indicators,
      turbulence_threshold=None,
      make_plots=False,
      print_verbosity=0,
      day=0,
      initial=True,
      previous_state=[],
      model_name="BybitRL",
      mode="train",
      iteration=0
    )

    logger.info(f"Инициализирована Bybit среда: баланс={initial_balance}, leverage={leverage}")

  def _get_technical_indicators_list(self) -> List[str]:
    """Получает список технических индикаторов из feature_engineer"""
    # Базовые индикаторы, которые использует ваш проект
    indicators = [
      'rsi', 'macd', 'macd_signal', 'macd_diff',
      'adx', 'cci', 'atr', 'bb_upper', 'bb_lower', 'bb_middle',
      'ema_short', 'ema_long', 'volume_ratio',
      'price_change', 'high_low_ratio'
    ]

    # Добавляем кастомные индикаторы из вашего проекта
    custom_indicators = [
      'regime_score',  # Из MarketRegimeDetector
      'risk_score',  # Из RiskManager
      'ml_signal',  # Из ML моделей
      'shadow_score'  # Из Shadow Trading
    ]

    return indicators + custom_indicators

  def _get_enhanced_state(self) -> np.ndarray:
    """
    Создает расширенное состояние с использованием всех компонентов системы
    """
    try:
      # Базовое состояние из родительского класса
      base_state = super()._get_state()

      # Текущий индекс данных
      current_idx = self.day

      # Получаем текущие данные
      current_data = self.data.iloc[current_idx:current_idx + 1]

      # 1. Режим рынка
      if hasattr(self.market_regime_detector, 'detect_regime'):
        regime = self.market_regime_detector.current_regime
        regime_features = [
          float(regime == MarketRegime.STRONG_TREND_UP),
          float(regime == MarketRegime.TREND_UP),
          float(regime == MarketRegime.RANGE_BOUND),
          float(regime == MarketRegime.TREND_DOWN),
          float(regime == MarketRegime.STRONG_TREND_DOWN),
        ]
      else:
        regime_features = [0, 0, 1, 0, 0]  # По умолчанию - боковик

      # 2. Метрики риска
      if self.risk_manager:
        risk_metrics = {
          'current_drawdown': self.risk_manager.get_current_drawdown(),
          'position_risk': len(self.stocks) / self.max_positions,
          'leverage_usage': self.leverage * sum(self.stocks) / self.amount,
          'daily_var': self.risk_manager.calculate_portfolio_var()
        }
        risk_features = list(risk_metrics.values())
      else:
        risk_features = [0, 0, 0, 0]

      # 3. Shadow Trading сигналы
      if self.shadow_trading_manager:
        shadow_metrics = self._get_shadow_trading_metrics()
        shadow_features = [
          shadow_metrics['missed_profit_ratio'],
          shadow_metrics['shadow_win_rate'],
          shadow_metrics['signal_quality']
        ]
      else:
        shadow_features = [0, 0, 0]

      # 4. Портфельные метрики
      portfolio_features = [
        self.amount / self.initial_amount,  # Относительный баланс
        sum(self.stocks) / max(1, len(self.stocks)),  # Средняя позиция
        len([s for s in self.stocks if s > 0]) / len(self.stocks),  # Доля активных позиций
      ]

      # Объединяем все признаки
      enhanced_features = np.concatenate([
        base_state,
        regime_features,
        risk_features,
        shadow_features,
        portfolio_features
      ])

      return enhanced_features

    except Exception as e:
      logger.error(f"Ошибка создания расширенного состояния: {e}")
      return super()._get_state()

  def _get_shadow_trading_metrics(self) -> Dict[str, float]:
    """Получает метрики из Shadow Trading системы"""
    try:
      if not self.shadow_trading_manager:
        return {'missed_profit_ratio': 0, 'shadow_win_rate': 0.5, 'signal_quality': 0.5}

      # Получаем последние метрики Shadow Trading
      recent_shadows = self.shadow_trading_manager.get_recent_shadow_trades(hours=24)

      if not recent_shadows:
        return {'missed_profit_ratio': 0, 'shadow_win_rate': 0.5, 'signal_quality': 0.5}

      # Рассчитываем метрики
      total_missed_profit = sum(s.get('potential_profit', 0) for s in recent_shadows)
      actual_profit = sum(t.get('profit', 0) for t in self.trade_history[-10:])

      missed_ratio = total_missed_profit / (actual_profit + total_missed_profit + 1e-10)

      # Win rate теневых сделок
      shadow_wins = sum(1 for s in recent_shadows if s.get('potential_profit', 0) > 0)
      shadow_win_rate = shadow_wins / len(recent_shadows) if recent_shadows else 0.5

      # Качество сигналов
      avg_quality = np.mean([s.get('signal_quality', 0.5) for s in recent_shadows])

      return {
        'missed_profit_ratio': missed_ratio,
        'shadow_win_rate': shadow_win_rate,
        'signal_quality': avg_quality
      }

    except Exception as e:
      logger.error(f"Ошибка получения Shadow Trading метрик: {e}")
      return {'missed_profit_ratio': 0, 'shadow_win_rate': 0.5, 'signal_quality': 0.5}

  def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    """
    Выполняет шаг в среде с учетом особенностей Bybit
    """
    # Сохраняем текущее состояние для анализа
    self.regime_history.append(self.market_regime_detector.current_regime if self.market_regime_detector else None)

    # Выполняем базовый шаг
    state, reward, terminated, truncated, info = super().step(actions)

    # Расширяем информацию
    info['regime'] = self.regime_history[-1]
    info['leverage_used'] = self.leverage * sum(self.stocks) / self.amount
    info['position_count'] = len([s for s in self.stocks if s > 0])

    # Применяем кастомную функцию вознаграждения
    if hasattr(self, 'reward_function'):
      risk_metrics = self._calculate_risk_metrics()
      reward = self.reward_function.calculate_reward(reward, risk_metrics)

    # Записываем сделку для анализа
    if any(actions != 0):  # Если были действия
      self.trade_history.append({
        'step': self.day,
        'action': actions,
        'reward': reward,
        'balance': self.amount,
        'positions': self.stocks.copy(),
        'info': info
      })

    # Получаем расширенное состояние
    enhanced_state = self._get_enhanced_state()

    return enhanced_state, reward, terminated, truncated, info

  def _calculate_risk_metrics(self) -> Dict[str, Any]:
    """Рассчитывает метрики риска для текущего состояния"""
    try:
      # Расчет просадки
      if self.trade_history:
        balances = [t['balance'] for t in self.trade_history]
        peak = max(balances)
        current_drawdown = (peak - self.amount) / peak
      else:
        current_drawdown = 0

      # Проверка превышения VaR
      var_exceeded = current_drawdown > 0.1  # 10% VaR

      # Risk/Reward ratio
      if self.trade_history and len(self.trade_history) > 1:
        returns = [
          (self.trade_history[i]['balance'] - self.trade_history[i - 1]['balance']) / self.trade_history[i - 1][
            'balance']
          for i in range(1, len(self.trade_history))
        ]
        avg_win = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
        avg_loss = abs(np.mean([r for r in returns if r < 0])) if any(r < 0 for r in returns) else 1
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0
      else:
        risk_reward_ratio = 0

      return {
        'drawdown': current_drawdown,
        'var_exceeded': var_exceeded,
        'risk_reward_ratio': risk_reward_ratio,
        'leverage_ratio': self.leverage * sum(self.stocks) / self.amount
      }

    except Exception as e:
      logger.error(f"Ошибка расчета метрик риска: {e}")
      return {
        'drawdown': 0,
        'var_exceeded': False,
        'risk_reward_ratio': 1,
        'leverage_ratio': 0
      }

  def reset(self, *, seed=None, options=None):
    """Сброс среды с очисткой истории"""
    self.trade_history = []
    self.regime_history = []
    self.shadow_signals = []

    return super().reset(seed=seed, options=options)

  def render(self, mode='human'):
    """Визуализация состояния среды"""
    if mode == 'human':
      print(f"\n=== Bybit Trading Environment ===")
      print(f"Step: {self.day}")
      print(f"Balance: ${self.amount:,.2f}")
      print(f"Positions: {sum(self.stocks)} contracts")
      print(f"Leverage Used: {self.leverage * sum(self.stocks) / self.amount:.2f}x")
      if self.regime_history:
        print(f"Market Regime: {self.regime_history[-1]}")
      print("================================\n")