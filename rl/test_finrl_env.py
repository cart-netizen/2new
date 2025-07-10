import unittest
import pandas as pd
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

import sys
import os

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFinRLEnvironment(unittest.TestCase):
  """Тесты для проверки интеграции с FinRL"""

  @classmethod
  def setUpClass(cls):
    """Настройка перед всеми тестами"""
    print("\n=== Начало тестирования FinRL Environment ===")

  def setUp(self):
    """Настройка перед каждым тестом"""
    self.df = self.create_test_data()
    self.stock_dim = len(self.df['tic'].unique())

  def create_test_data(self):
    """Создает тестовые данные в формате, который ожидает FinRL"""
    n_days = 50
    symbols = ['BTCUSDT', 'ETHUSDT']

    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='h')

    data = []
    for i, date in enumerate(dates):
      for j, symbol in enumerate(symbols):
        base_price = 100 * (j + 1)
        noise = np.sin(i * 0.1) * 5 + np.random.randn() * 2

        open_price = base_price + noise
        high_price = open_price + np.random.uniform(0, 5)
        low_price = open_price - np.random.uniform(0, 5)
        close_price = open_price + np.random.uniform(-3, 3)
        volume = np.random.uniform(1000, 10000)

        data.append({
          'date': date,
          'tic': symbol,
          'open': open_price,
          'high': high_price,
          'low': low_price,
          'close': close_price,
          'volume': volume
        })

    df = pd.DataFrame(data)
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)

    return df

  def test_data_structure(self):
    """Тест структуры данных"""
    print("\n--- Тест структуры данных ---")

    # Проверяем наличие всех необходимых колонок
    required_columns = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
      self.assertIn(col, self.df.columns, f"Отсутствует колонка {col}")

    # Проверяем количество символов
    unique_symbols = self.df['tic'].unique()
    self.assertEqual(len(unique_symbols), 2, "Должно быть 2 символа")

    # Проверяем, что для каждой даты есть все символы
    for date in self.df['date'].unique()[:5]:
      symbols_for_date = self.df[self.df['date'] == date]['tic'].tolist()
      self.assertEqual(len(symbols_for_date), 2, f"Для даты {date} должно быть 2 символа")

    print(f"✅ Структура данных корректна")
    print(f"   Shape: {self.df.shape}")
    print(f"   Symbols: {unique_symbols}")

  def test_env_creation(self):
    """Тест создания среды"""
    print("\n--- Тест создания среды ---")

    state_space = 1 + 2 * self.stock_dim
    action_space = self.stock_dim

    try:
      env = StockTradingEnv(
        df=self.df,
        stock_dim=self.stock_dim,
        hmax=100,
        initial_amount=10000,
        num_stock_shares=[0] * self.stock_dim,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        reward_scaling=1e-4,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=[],
        turbulence_threshold=None,
        make_plots=False,
        print_verbosity=0,
        day=0,
        initial=True,
        previous_state=[],
        model_name="test",
        mode="train",
        iteration=0
      )

      self.assertIsNotNone(env, "Среда должна быть создана")
      print("✅ Среда создана успешно")

      # Проверяем пространства
      self.assertEqual(env.observation_space.shape[0], state_space)
      self.assertEqual(env.action_space.shape[0], action_space)
      print(f"   Observation space: {env.observation_space}")
      print(f"   Action space: {env.action_space}")

    except Exception as e:
      self.fail(f"Не удалось создать среду: {e}")

  def test_env_reset(self):
    """Тест сброса среды"""
    print("\n--- Тест сброса среды ---")

    env = self._create_env()

    try:
      state = env.reset()
      self.assertIsNotNone(state, "Состояние должно быть возвращено")
      self.assertEqual(len(state), env.observation_space.shape[0])
      print(f"✅ Сброс среды успешен")
      print(f"   Размер состояния: {len(state)}")
      print(f"   Первые элементы состояния: {state[:5]}")

    except Exception as e:
      self.fail(f"Ошибка при сбросе среды: {e}")

  def test_env_step(self):
    """Тест выполнения шагов в среде"""
    print("\n--- Тест выполнения шагов ---")

    env = self._create_env()
    env.reset()

    try:
      # Делаем 10 случайных шагов
      for i in range(10):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        self.assertIsNotNone(state)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

        if i == 0:
          print(f"✅ Шаги выполняются корректно")
          print(f"   Пример reward: {reward:.4f}")

        if done:
          print(f"   Эпизод завершен на шаге {i + 1}")
          break

    except Exception as e:
      self.fail(f"Ошибка при выполнении шага: {e}")

  def test_env_with_technical_indicators(self):
    """Тест среды с техническими индикаторами"""
    print("\n--- Тест с техническими индикаторами ---")

    # Добавляем технические индикаторы
    df_with_indicators = self.df.copy()

    for symbol in df_with_indicators['tic'].unique():
      mask = df_with_indicators['tic'] == symbol
      symbol_data = df_with_indicators[mask].copy()

      # Простой RSI
      close_prices = symbol_data['close'].values
      rsi = self._calculate_simple_rsi(close_prices)
      df_with_indicators.loc[mask, 'rsi'] = rsi

      # Простая скользящая средняя
      sma = symbol_data['close'].rolling(window=10, min_periods=1).mean()
      df_with_indicators.loc[mask, 'sma'] = sma

    tech_indicators = ['rsi', 'sma']
    state_space = 1 + 2 * self.stock_dim + len(tech_indicators) * self.stock_dim

    try:
      env = StockTradingEnv(
        df=df_with_indicators,
        stock_dim=self.stock_dim,
        hmax=100,
        initial_amount=10000,
        num_stock_shares=[0] * self.stock_dim,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        reward_scaling=1e-4,
        state_space=state_space,
        action_space=self.stock_dim,
        tech_indicator_list=tech_indicators,
        turbulence_threshold=None,
        make_plots=False,
        print_verbosity=0,
        day=0,
        initial=True,
        previous_state=[],
        model_name="test",
        mode="train",
        iteration=0
      )

      state = env.reset()
      self.assertEqual(len(state), state_space)
      print(f"✅ Среда с индикаторами создана успешно")
      print(f"   Технические индикаторы: {tech_indicators}")
      print(f"   Размер состояния: {len(state)}")

    except Exception as e:
      self.fail(f"Ошибка при создании среды с индикаторами: {e}")

  def _create_env(self):
    """Вспомогательный метод для создания среды"""
    state_space = 1 + 2 * self.stock_dim

    return StockTradingEnv(
      df=self.df,
      stock_dim=self.stock_dim,
      hmax=100,
      initial_amount=10000,
      num_stock_shares=[0] * self.stock_dim,
      buy_cost_pct=0.001,
      sell_cost_pct=0.001,
      reward_scaling=1e-4,
      state_space=state_space,
      action_space=self.stock_dim,
      tech_indicator_list=[],
      turbulence_threshold=None,
      make_plots=False,
      print_verbosity=0,
      day=0,
      initial=True,
      previous_state=[],
      model_name="test",
      mode="train",
      iteration=0
    )

  def _calculate_simple_rsi(self, prices, period=14):
    """Простой расчет RSI для тестов"""
    deltas = np.diff(prices)
    seed = deltas[:period + 1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 50  # Начальное значение

    for i in range(period, len(prices)):
      delta = deltas[i - 1]
      if delta > 0:
        upval = delta
        downval = 0
      else:
        upval = 0
        downval = -delta

      up = (up * (period - 1) + upval) / period
      down = (down * (period - 1) + downval) / period

      rs = up / down if down != 0 else 100
      rsi[i] = 100 - 100 / (1 + rs)

    return rsi


if __name__ == '__main__':
  # Запускаем тесты с подробным выводом
  unittest.main(verbosity=2)