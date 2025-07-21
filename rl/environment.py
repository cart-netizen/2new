# rl/environment.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from gymnasium import spaces
import logging
from finrl_wrapper import FinRLCompatibleEnv, FinRLDataProcessor
from rl.finrl_wrapper import FinRLCompatibleEnv, FinRLDataProcessor
from core.enums import Timeframe, SignalType
from core.market_regime_detector import MarketRegime
from ml.feature_engineering import AdvancedFeatureEngineer
from utils.logging_config import get_logger

logger = get_logger(__name__)


def validate_dataframe_for_finrl(df: pd.DataFrame) -> pd.DataFrame:
  """
  Валидирует и корректирует DataFrame для FinRL
  """
  # Проверяем обязательные колонки
  required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
  missing_cols = [col for col in required_cols if col not in df.columns]
  if missing_cols:
    raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

  # Убедимся, что date - это datetime
  df['date'] = pd.to_datetime(df['date'])

  # Сортировка КРИТИЧНА для FinRL
  df = df.sort_values(['date', 'tic']).reset_index(drop=True)

  # Проверяем, что у нас есть данные для всех символов на каждую дату
  unique_dates = df['date'].unique()
  unique_tics = df['tic'].unique()

  expected_rows = len(unique_dates) * len(unique_tics)
  actual_rows = len(df)

  if expected_rows != actual_rows:
    logger.warning(f"Несоответствие количества строк: ожидалось {expected_rows}, получено {actual_rows}")

    # Создаем полную матрицу дата-символ
    full_index = pd.MultiIndex.from_product([unique_dates, unique_tics], names=['date', 'tic'])
    full_df = pd.DataFrame(index=full_index).reset_index()

    # Мержим с существующими данными
    df = full_df.merge(df, on=['date', 'tic'], how='left')

    # Заполняем пропуски по группам
    for tic in unique_tics:
      mask = df['tic'] == tic
      for col in ['open', 'high', 'low', 'close']:
        df.loc[mask, col] = df.loc[mask, col].fillna(method='ffill').fillna(method='bfill')
      df.loc[mask, 'volume'] = df.loc[mask, 'volume'].fillna(0)

  # Финальная проверка типов
  for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[col].isna().any():
      df[col] = df[col].fillna(df[col].mean())

  return df


class BybitTradingEnvironment(FinRLCompatibleEnv):
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
    # Сохраняем компоненты проекта
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
    tech_indicators = self._get_technical_indicators_list(df)

    # Определяем количество символов
    num_stocks = len(df['tic'].unique())
    num_stock_shares = [0] * num_stocks

    # Размерность пространства состояний
    state_space_dim = 1 + num_stocks + num_stocks + len(tech_indicators) * num_stocks

    logger.info(
      f"Создание среды: stocks={num_stocks}, indicators={len(tech_indicators)}, state_space={state_space_dim}")

    # ВАЖНО: FinRL ожидает массивы для комиссий
    buy_cost_pct = [self.config.get('buy_cost_pct', commission_rate)] * num_stocks
    sell_cost_pct = [self.config.get('sell_cost_pct', commission_rate)] * num_stocks

    # Инициализация родительского класса с правильной обработкой данных
    super().__init__(
      df=df,
      stock_dim=num_stocks,
      hmax=100,
      initial_amount=initial_balance,
      num_stock_shares=num_stock_shares,
      buy_cost_pct=buy_cost_pct,  # Теперь массив
      sell_cost_pct=sell_cost_pct,  # Теперь массив
      reward_scaling=self.config.get('reward_scaling', 1e-4),
      state_space=state_space_dim,
      action_space=num_stocks,
      tech_indicator_list=tech_indicators,
      turbulence_threshold=None,
      make_plots=False,
      print_verbosity=1,
      day=0,
      initial=True,
      previous_state=[],
      model_name="BybitRL",
      mode="train",
      iteration=0
    )

    logger.info(f"✅ Среда успешно инициализирована")

  # def _get_technical_indicators_list(self) -> List[str]:
  #   """Получает список технических индикаторов из feature_engineer"""
  #   # Базовые индикаторы, которые использует ваш проект
  #   indicators = [
  #     'rsi', 'macd', 'macd_signal', 'macd_diff',
  #     'adx', 'cci', 'atr', 'bb_upper', 'bb_lower', 'bb_middle',
  #     'ema_short', 'ema_long', 'volume_ratio',
  #     'price_change', 'high_low_ratio'
  #   ]
  #
  #   # Добавляем кастомные индикаторы из вашего проекта
  #   custom_indicators = [
  #     'regime_score',  # Из MarketRegimeDetector
  #     'risk_score',  # Из RiskManager
  #     'ml_signal',  # Из ML моделей
  #     'shadow_score'  # Из Shadow Trading
  #   ]
  #
  #   return indicators + custom_indicators

  def _get_technical_indicators_list(self, df: pd.DataFrame) -> list:
    """
    Получает список технических индикаторов из DataFrame
    """
    # Базовые колонки, которые не являются индикаторами
    base_columns = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume',
                    'timestamp', 'turnover']

    # Колонки, которые мы исключаем из технических индикаторов
    exclude_columns = base_columns + ['market_regime', 'market_regime_numeric']

    # Все остальные числовые колонки считаем техническими индикаторами
    tech_indicators = []
    for col in df.columns:
      if col not in exclude_columns and df[col].dtype in ['float64', 'int64']:
        # Проверяем, что это действительно индикатор (есть для всех символов)
        if df.groupby('tic')[col].count().min() > 0:
          tech_indicators.append(col)

    logger.info(f"Найдены технические индикаторы: {tech_indicators}")
    return tech_indicators

  def _get_enhanced_state(self) -> np.ndarray:
    """
    Создает расширенное состояние с дополнительными признаками
    """
    try:
      # Получаем базовое состояние из текущего состояния среды
      if hasattr(self, 'state') and self.state is not None:
        base_state = np.array(self.state)
      else:
        # Если состояние еще не инициализировано, создаем пустое
        base_state = np.zeros(self.state_space)

      # Дополнительные признаки для улучшения обучения

      # 1. Режим рынка (one-hot encoding)
      if self.market_regime_detector and hasattr(self.market_regime_detector, 'current_regime'):
        regime = self.market_regime_detector.current_regime
        regime_features = [
          1.0 if regime == 'bullish' else 0.0,
          1.0 if regime == 'bearish' else 0.0,
          1.0 if regime == 'sideways' else 0.0,
          1.0 if regime == 'volatile' else 0.0
        ]
      else:
        regime_features = [0, 0, 1, 0]  # По умолчанию sideways

      # 2. Метрики риска
      if self.risk_manager:
        risk_metrics = {
          'current_drawdown': self.risk_manager.get_current_drawdown(),
          'position_risk': len(
            [s for s in base_state[1 + self.stock_dim:1 + 2 * self.stock_dim] if s > 0]) / self.max_positions,
          'leverage_usage': self.leverage * sum(base_state[1 + self.stock_dim:1 + 2 * self.stock_dim]) / base_state[
            0] if base_state[0] > 0 else 0,
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
      current_balance = base_state[0] if len(base_state) > 0 else self.initial_amount
      current_positions = base_state[1 + self.stock_dim:1 + 2 * self.stock_dim] if len(
        base_state) > 1 + 2 * self.stock_dim else np.zeros(self.stock_dim)

      portfolio_features = [
        current_balance / self.initial_amount,  # Относительный баланс
        np.mean(current_positions) if len(current_positions) > 0 else 0,  # Средняя позиция
        len([s for s in current_positions if s > 0]) / len(current_positions) if len(current_positions) > 0 else 0,
        # Доля активных позиций
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
      # В случае ошибки возвращаем базовое состояние
      if hasattr(self, 'state') and self.state is not None:
        return np.array(self.state)
      else:
        return np.zeros(self.state_space)

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
    if self.market_regime_detector:
      self.regime_history.append(self.market_regime_detector.current_regimes)

    # Выполняем базовый шаг родительского класса
    # FinRLCompatibleEnv уже возвращает 5 значений
    state, reward, terminated, truncated, info = super().step(actions)

    # Сохраняем состояние
    self.state = state

    # Безопасное извлечение данных из состояния
    if isinstance(state, np.ndarray) and len(state) >= 1 + 2 * self.stock_dim:
      current_balance = state[0]
      current_positions = state[1 + self.stock_dim:1 + 2 * self.stock_dim]
    else:
      current_balance = self.initial_amount
      current_positions = np.zeros(self.stock_dim)

    # Расширяем информацию
    info['regime'] = self.regime_history[-1] if self.regime_history else None
    info['leverage_used'] = (self.leverage * np.sum(current_positions) / current_balance
                             if current_balance > 0 else 0)
    info['position_count'] = int(np.sum(current_positions > 0))

    # Применяем кастомную функцию вознаграждения если она есть
    if hasattr(self, 'reward_function') and self.reward_function:
      risk_metrics = self._calculate_risk_metrics()
      reward = self.reward_function.calculate_reward(reward, risk_metrics)

    # Записываем сделку для анализа
    if np.any(actions != 0):  # Если были действия
      self.trade_history.append({
        'step': self.day,
        'action': actions.tolist() if hasattr(actions, 'tolist') else list(actions),
        'reward': float(reward),
        'balance': float(current_balance),
        'positions': current_positions.tolist() if hasattr(current_positions, 'tolist') else list(current_positions),
        'info': info.copy()
      })

    return state, reward, terminated, truncated, info

  def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
    """Сброс среды с очисткой истории"""
    # Очищаем историю
    self.trade_history = []
    self.regime_history = []
    self.shadow_signals = []

    # Вызываем родительский reset
    result = super().reset(seed=seed, options=options)

    # Обрабатываем разные форматы возврата
    if isinstance(result, tuple):
      state, info = result
    else:
      state = result
      info = {}

    # Убеждаемся, что state - это numpy array
    if not isinstance(state, np.ndarray):
      state = np.array(state, dtype=np.float32)

    # Сохраняем начальное состояние
    self.state = state

    return state, info

  def _calculate_risk_metrics(self) -> Dict[str, Any]:
    """Рассчитывает метрики риска для текущего состояния"""
    try:
      # Безопасное получение текущего состояния
      if hasattr(self, 'state') and isinstance(self.state, np.ndarray):
        current_state = self.state
      else:
        return {
          'drawdown': 0,
          'var_exceeded': False,
          'risk_reward_ratio': 1,
          'leverage_ratio': 0
        }

      # Безопасное извлечение данных
      if len(current_state) >= 1 + 2 * self.stock_dim:
        current_balance = float(current_state[0])
        current_positions = current_state[1 + self.stock_dim:1 + 2 * self.stock_dim]
      else:
        current_balance = self.initial_amount
        current_positions = np.zeros(self.stock_dim)

      # Расчет просадки
      current_drawdown = 0
      if self.trade_history and len(self.trade_history) > 0:
        balances = [float(t.get('balance', current_balance)) for t in self.trade_history]
        if balances:
          peak = max(balances)
          current_drawdown = (peak - current_balance) / peak if peak > 0 else 0

      # Risk/Reward ratio
      risk_reward_ratio = 1.0
      if self.trade_history and len(self.trade_history) > 1:
        returns = []
        for i in range(1, len(self.trade_history)):
          prev_balance = float(self.trade_history[i - 1].get('balance', 1))
          curr_balance = float(self.trade_history[i].get('balance', 1))
          if prev_balance > 0:
            returns.append((curr_balance - prev_balance) / prev_balance)

        if returns:
          wins = [r for r in returns if r > 0]
          losses = [r for r in returns if r < 0]
          avg_win = np.mean(wins) if wins else 0
          avg_loss = abs(np.mean(losses)) if losses else 1
          risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0

      # Leverage ratio
      leverage_ratio = 0
      if current_balance > 0:
        total_position_value = self.leverage * np.sum(np.abs(current_positions))
        leverage_ratio = total_position_value / current_balance

      return {
        'drawdown': float(current_drawdown),
        'var_exceeded': bool(current_drawdown > 0.1),
        'risk_reward_ratio': float(risk_reward_ratio),
        'leverage_ratio': float(leverage_ratio)
      }

    except Exception as e:
      logger.error(f"Ошибка расчета метрик риска: {e}", exc_info=True)
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
      current_state = self.state if hasattr(self, 'state') else self._get_state()
      current_balance = current_state[0]
      current_positions = current_state[1 + self.stock_dim:1 + 2 * self.stock_dim]

      print(f"\n=== Bybit Trading Environment ===")
      print(f"Step: {self.day}")
      print(f"Balance: ${current_balance:,.2f}")
      print(f"Positions: {sum(current_positions)} contracts")
      print(
        f"Leverage Used: {self.leverage * sum(current_positions) / current_balance:.2f}x" if current_balance > 0 else "Leverage Used: 0.00x")
      if self.regime_history:
        print(f"Market Regime: {self.regime_history[-1]}")
      print("================================\n")