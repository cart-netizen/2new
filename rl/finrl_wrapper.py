import pandas as pd
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class FinRLDataProcessor:
  """
  Процессор данных для правильной подготовки DataFrame для FinRL
  """

  @staticmethod
  def prepare_multistock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает данные множественных акций для FinRL

    FinRL ожидает, что df.loc[day, :] вернет DataFrame со всеми акциями для этого дня,
    а не одну строку.
    """
    # Убеждаемся, что есть необходимые колонки
    required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
      raise ValueError(f"Отсутствуют колонки: {missing_cols}")

    # Сортируем по дате и символу
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)

    # Создаем числовой индекс дня
    unique_dates = df['date'].unique()
    date_to_day = {date: i for i, date in enumerate(unique_dates)}
    df['day'] = df['date'].map(date_to_day)

    # КРИТИЧНО: Устанавливаем day как индекс
    # Это позволит FinRL делать df.loc[day, :] и получать DataFrame со всеми акциями
    df = df.set_index('day')

    logger.info(f"Подготовлены данные: {len(unique_dates)} дней, {df['tic'].nunique()} акций")

    return df


class FinRLCompatibleEnv(StockTradingEnv):
  """
  Совместимая с FinRL среда с правильной обработкой данных
  """

  def __init__(self, df: pd.DataFrame, **kwargs):
    # Подготавливаем данные
    df_processed = FinRLDataProcessor.prepare_multistock_data(df.copy())

    # Переопределяем расчет state_space, если нужно
    if 'state_space' not in kwargs:
      stock_dim = kwargs.get('stock_dim', len(df['tic'].unique()))
      tech_indicators = kwargs.get('tech_indicator_list', [])
      # Правильный расчет: balance + prices + shares + tech_indicators
      kwargs['state_space'] = 1 + stock_dim + stock_dim + len(tech_indicators) * stock_dim

    # Вызываем родительский конструктор с обработанными данными
    super().__init__(df=df_processed, **kwargs)

    # Сохраняем оригинальные данные для reference
    self.original_df = df
    self.unique_dates = df['date'].unique()

    logger.info(f"Среда создана: state_space={self.state_space}, action_space={self.action_space}")

  def reset(self, *, seed=None, options=None) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """Переопределяем reset для совместимости"""
    result = super().reset()

    # Обрабатываем разные форматы возврата
    if isinstance(result, tuple):
      state, info = result
    else:
      state = result
      info = {}

    # Преобразуем в numpy array если нужно
    if isinstance(state, list):
      state = np.array(state, dtype=np.float32)

    logger.info(f"Reset: state shape = {state.shape}, expected = ({self.state_space},)")

    # Возвращаем в формате Gymnasium
    return state, info

  def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
    """Переопределяем step для совместимости с Gymnasium"""
    # FinRL ожидает действия как numpy array
    if not isinstance(action, np.ndarray):
      action = np.array(action)

    # Вызываем родительский step
    state, reward, done, info = super().step(action)

    # Преобразуем state в numpy array если нужно
    if isinstance(state, list):
      state = np.array(state, dtype=np.float32)

    # Gymnasium использует (terminated, truncated) вместо просто done
    terminated = done
    truncated = False

    return state, reward, terminated, truncated, info

  def get_current_date(self) -> pd.Timestamp:
    """Получить текущую дату в симуляции"""
    if self.day < len(self.unique_dates):
      return self.unique_dates[self.day]
    return self.unique_dates[-1]

def create_finrl_compatible_env(
      df: pd.DataFrame,
      initial_amount: float = 10000,
      commission_rate: float = 0.001,
      tech_indicators: List[str] = None
  ) -> FinRLCompatibleEnv:
    """
    Создает среду, совместимую с FinRL
    """
    # Определяем параметры
    stock_dim = len(df['tic'].unique())

    # Технические индикаторы
    if tech_indicators is None:
      tech_indicators = []

    # Фильтруем только существующие индикаторы
    available_indicators = []
    for indicator in tech_indicators:
      if indicator in df.columns:
        available_indicators.append(indicator)
      else:
        logger.warning(f"Индикатор {indicator} отсутствует в данных")

    # Размерность пространства состояний
    # [баланс] + [цены акций] + [количество акций] + [тех. индикаторы]
    state_space = 1 + stock_dim + stock_dim + len(available_indicators) * stock_dim

    logger.info(
      f"Создание среды: stocks={stock_dim}, indicators={len(available_indicators)}, state_space={state_space}")

    # ВАЖНО: FinRL ожидает массивы для buy_cost_pct и sell_cost_pct
    buy_cost_pct = [commission_rate] * stock_dim
    sell_cost_pct = [commission_rate] * stock_dim

    # Создаем среду
    env = FinRLCompatibleEnv(
      df=df,
      stock_dim=stock_dim,
      hmax=100,
      initial_amount=initial_amount,
      num_stock_shares=[0] * stock_dim,
      buy_cost_pct=buy_cost_pct,  # Теперь это массив
      sell_cost_pct=sell_cost_pct,  # Теперь это массив
      reward_scaling=1e-4,
      state_space=state_space,
      action_space=stock_dim,
      tech_indicator_list=available_indicators,
      turbulence_threshold=None,
      make_plots=False,
      print_verbosity=1,  # Включаем вывод для отладки
      day=0,
      initial=True,
      previous_state=[],
      model_name="FinRLCompatible",
      mode="train",
      iteration=0
    )

    return env


# Тестовый код
if __name__ == "__main__":
  import logging

  logging.basicConfig(level=logging.INFO)

  # Создаем тестовые данные
  dates = pd.date_range('2023-01-01', periods=10, freq='D')
  symbols = ['BTCUSDT', 'ETHUSDT']

  data = []
  for date in dates:
    for symbol in symbols:
      base_price = 100 if symbol == 'BTCUSDT' else 50
      data.append({
        'date': date,
        'tic': symbol,
        'open': base_price + np.random.uniform(-2, 2),
        'high': base_price + np.random.uniform(0, 5),
        'low': base_price + np.random.uniform(-5, 0),
        'close': base_price + np.random.uniform(-2, 2),
        'volume': np.random.uniform(1000, 10000),
        # Добавим простой технический индикатор для теста
        'rsi': 50 + np.random.uniform(-20, 20)
      })

  df = pd.DataFrame(data)

  print("=== Тест FinRL Compatible Environment ===\n")
  print(f"Исходные данные shape: {df.shape}")
  print(f"Колонки: {df.columns.tolist()}")
  print(df.head())

  try:
    # Создаем среду с техническими индикаторами
    env = create_finrl_compatible_env(df, tech_indicators=['rsi'])
    print("\n✅ Среда создана успешно!")

    # Проверяем размерности
    print(f"\nРазмерности:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  State space: {env.state_space}")
    print(f"  Stock dim: {env.stock_dim}")

    # Тестируем reset
    state, info = env.reset()  # Теперь правильно распаковываем tuple
    print(f"\nНачальное состояние:")
    print(f"  Тип: {type(state)}")
    print(f"  Shape: {state.shape if hasattr(state, 'shape') else len(state)}")
    print(f"  Длина: {len(state)} (ожидается: {env.state_space})")

    # Если состояние - это правильный numpy array
    if isinstance(state, np.ndarray) and len(state) == env.state_space:
      print(f"\nСтруктура состояния:")
      print(f"  Баланс: {state[0]:.2f}")
      print(f"  Цены: {state[1:1 + env.stock_dim]}")
      print(f"  Позиции: {state[1 + env.stock_dim:1 + 2 * env.stock_dim]}")
      if len(state) > 1 + 2 * env.stock_dim:
        print(f"  Индикаторы RSI: {state[1 + 2 * env.stock_dim:]}")
    else:
      print(f"  Состояние (raw): {state}")

    print(f"  Текущая дата: {env.get_current_date()}")

    # Делаем несколько шагов
    print("\n=== Тестирование шагов ===")
    for i in range(3):
      # Создаем правильное действие для Box action space
      action = env.action_space.sample()  # Случайное действие
      # Или используем конкретные значения
      # action = np.array([0.1, -0.1])  # Купить первую, продать вторую

      print(f"\nДействие: {action}")

      state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated

      print(f"\nШаг {i + 1}:")
      print(f"  Дата: {env.get_current_date()}")
      print(f"  Reward: {reward:.4f}")
      print(f"  Done: {done}")

      if isinstance(state, np.ndarray) and len(state) >= 1 + 2 * env.stock_dim:
        print(f"  Баланс: {state[0]:.2f}")
        print(f"  Позиции: {state[1 + env.stock_dim:1 + 2 * env.stock_dim]}")

      if done:
        print("  Эпизод завершен!")
        break

    print("\n✅ Все тесты пройдены!")

  except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    import traceback

    traceback.print_exc()