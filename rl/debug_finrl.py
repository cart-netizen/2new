import pandas as pd
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import inspect


def diagnose_finrl():
  """Диагностика проблемы с FinRL"""
  print("=== Диагностика FinRL ===\n")

  # Проверяем версию и путь
  import finrl
  print(f"FinRL версия: {finrl.__version__ if hasattr(finrl, '__version__') else 'Unknown'}")
  print(f"FinRL путь: {finrl.__file__}\n")

  # Создаем минимальные данные
  dates = pd.date_range('2023-01-01', periods=10, freq='D')
  df = pd.DataFrame({
    'date': dates,
    'tic': 'AAPL',
    'open': 100 + np.random.randn(10),
    'high': 101 + np.random.randn(10),
    'low': 99 + np.random.randn(10),
    'close': 100 + np.random.randn(10),
    'volume': 1000000 + np.random.randint(-100000, 100000, 10)
  })

  print("Тестовые данные:")
  print(df.head())
  print(f"\nТип df: {type(df)}")
  print(f"Тип df.close: {type(df.close)}")
  print(f"Тип df.close.values: {type(df.close.values)}")

  # Пытаемся создать среду с минимальными параметрами
  print("\n--- Попытка создания среды ---")

  try:
    # Смотрим на сигнатуру конструктора
    sig = inspect.signature(StockTradingEnv.__init__)
    print(f"Параметры StockTradingEnv.__init__: {list(sig.parameters.keys())}\n")

    env = StockTradingEnv(
      df=df,
      stock_dim=1,
      hmax=100,
      initial_amount=10000,
      buy_cost_pct=0.001,
      sell_cost_pct=0.001,
      reward_scaling=1e-4,
      state_space=3,
      action_space=1,
      tech_indicator_list=[],
      print_verbosity=1
    )

    print("✅ Среда создана!")

  except TypeError as e:
    print(f"❌ TypeError: {e}")
    print("\nПопробуем с дополнительными параметрами...")

    try:
      env = StockTradingEnv(
        df=df,
        stock_dim=1,
        hmax=100,
        initial_amount=10000,
        num_stock_shares=[0],
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        reward_scaling=1e-4,
        state_space=3,
        action_space=1,
        tech_indicator_list=[],
        turbulence_threshold=None,
        make_plots=False,
        print_verbosity=1,
        day=0,
        initial=True,
        previous_state=[],
        model_name="test",
        mode="train",
        iteration=0
      )
      print("✅ Среда создана с полными параметрами!")

    except Exception as e2:
      print(f"❌ Ошибка с полными параметрами: {e2}")

      # Проверяем внутреннюю структуру
      print("\n--- Анализ ошибки ---")
      import traceback
      traceback.print_exc()

  except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

    # Дополнительная диагностика
    print("\n--- Попытка понять проблему ---")

    # Создаем экземпляр для анализа
    class DebugEnv(StockTradingEnv):
      def _initiate_state(self):
        print(f"В _initiate_state:")
        print(f"  type(self.data): {type(self.data)}")
        if hasattr(self, 'data'):
          print(f"  self.data shape: {getattr(self.data, 'shape', 'No shape')}")
          print(f"  self.data columns: {getattr(self.data, 'columns', 'No columns')}")
          if hasattr(self.data, 'close'):
            print(f"  type(self.data.close): {type(self.data.close)}")
        return super()._initiate_state()

    try:
      debug_env = DebugEnv(
        df=df,
        stock_dim=1,
        hmax=100,
        initial_amount=10000,
        num_stock_shares=[0],
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        reward_scaling=1e-4,
        state_space=3,
        action_space=1,
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
    except Exception as de:
      print(f"\nDebugEnv error: {de}")


def test_multi_stock():
  """Тест с несколькими акциями"""
  print("\n\n=== Тест с несколькими акциями ===\n")

  # Создаем данные для двух акций
  dates = pd.date_range('2023-01-01', periods=10, freq='D')

  data = []
  for date in dates:
    for tic in ['AAPL', 'GOOGL']:
      data.append({
        'date': date,
        'tic': tic,
        'open': 100 + np.random.randn(),
        'high': 101 + np.random.randn(),
        'low': 99 + np.random.randn(),
        'close': 100 + np.random.randn(),
        'volume': 1000000 + np.random.randint(-100000, 100000)
      })

  df = pd.DataFrame(data)
  df = df.sort_values(['date', 'tic']).reset_index(drop=True)

  print("DataFrame с двумя акциями:")
  print(df.head(10))

  try:
    env = StockTradingEnv(
      df=df,
      stock_dim=2,
      hmax=100,
      initial_amount=10000,
      num_stock_shares=[0, 0],
      buy_cost_pct=0.001,
      sell_cost_pct=0.001,
      reward_scaling=1e-4,
      state_space=5,  # 1 (balance) + 2 (prices) + 2 (holdings)
      action_space=2,
      tech_indicator_list=[],
      turbulence_threshold=None,
      make_plots=False,
      print_verbosity=1,
      day=0,
      initial=True,
      previous_state=[],
      model_name="test",
      mode="train",
      iteration=0
    )

    print("\n✅ Среда с несколькими акциями создана!")

    # Тестируем reset
    state = env.reset()
    print(f"Initial state shape: {len(state)}")
    print(f"State: {state}")

  except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

def debug_internal_state():
    """Детальная диагностика внутреннего состояния"""
    print("\n\n=== Детальная диагностика ===\n")

    # Создаем данные для анализа
    dates = pd.date_range('2023-01-01', periods=5, freq='D')

    # Пробуем разные форматы данных
    # Формат 1: Стандартный (который не работает)
    data1 = []
    for date in dates:
      for tic in ['AAPL', 'GOOGL']:
        data1.append({
          'date': date,
          'tic': tic,
          'open': 100,
          'high': 101,
          'low': 99,
          'close': 100,
          'volume': 1000000
        })

    df1 = pd.DataFrame(data1)
    df1 = df1.sort_values(['date', 'tic']).reset_index(drop=True)

    print("Формат 1 - Стандартный:")
    print(df1)
    print(f"\nУникальные даты: {df1['date'].nunique()}")
    print(f"Уникальные символы: {df1['tic'].nunique()}")

    # Создаем кастомный класс для отладки
    class DebugStockTradingEnv(StockTradingEnv):
      def __init__(self, *args, **kwargs):
        print("\n--- В конструкторе DebugStockTradingEnv ---")
        super().__init__(*args, **kwargs)

      def _process_data(self):
        print(f"\nВ _process_data:")
        print(f"  self.df shape: {self.df.shape}")
        print(f"  self.df columns: {self.df.columns.tolist()}")

        # Смотрим, как FinRL обрабатывает данные
        self.data = self.df.loc[self.day, :]
        print(f"  После self.df.loc[{self.day}, :]:")
        print(f"  type(self.data): {type(self.data)}")
        print(f"  self.data:\n{self.data}")

        if isinstance(self.data, pd.DataFrame):
          print(f"  self.data shape: {self.data.shape}")
        elif isinstance(self.data, pd.Series):
          print(f"  self.data index: {self.data.index.tolist()}")

    # Пробуем создать с отладкой
    try:
      print("\n--- Создание среды с отладкой ---")
      env = DebugStockTradingEnv(
        df=df1,
        stock_dim=2,
        hmax=100,
        initial_amount=10000,
        num_stock_shares=[0, 0],
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        reward_scaling=1e-4,
        state_space=5,
        action_space=2,
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
    except Exception as e:
      print(f"Ошибка: {e}")

    # Теперь попробуем понять, что ожидает FinRL
    print("\n--- Анализ ожидаемой структуры ---")

    # Смотрим, как индексируются данные
    print(f"\ndf1.loc[0, :]:")
    print(df1.loc[0, :])
    print(f"\nТип: {type(df1.loc[0, :])}")

    # Если это Series, смотрим на структуру
    test_data = df1.loc[0, :]
    if isinstance(test_data, pd.Series):
      print(f"Index: {test_data.index.tolist()}")
      print(f"Values: {test_data.values}")

      # Проверяем доступ к close
      if 'close' in test_data:
        print(f"\ntest_data['close']: {test_data['close']}")
        print(f"type: {type(test_data['close'])}")


def try_workaround():
  """Пробуем обходное решение"""
  print("\n\n=== Попытка обходного решения ===\n")

  # Может быть, FinRL ожидает другую структуру индексов?
  dates = pd.date_range('2023-01-01', periods=5, freq='D')

  # Создаем данные с мультииндексом
  arrays = []
  for date in dates:
    for tic in ['AAPL', 'GOOGL']:
      arrays.append((date, tic))

  index = pd.MultiIndex.from_tuples(arrays, names=['date', 'tic'])

  df = pd.DataFrame({
    'open': np.random.uniform(99, 101, len(index)),
    'high': np.random.uniform(100, 102, len(index)),
    'low': np.random.uniform(98, 100, len(index)),
    'close': np.random.uniform(99, 101, len(index)),
    'volume': np.random.uniform(900000, 1100000, len(index))
  }, index=index)

  # Сбрасываем индекс
  df = df.reset_index()

  print("DataFrame с мультииндексом (сброшенным):")
  print(df.head(10))

  # Пробуем альтернативный подход - нумерация дней
  df['day'] = df.groupby('date').ngroup()
  print(f"\nДобавлен столбец 'day':")
  print(df[['day', 'date', 'tic', 'close']].head(10))

  # Смотрим, что получается при индексации по day
  print(f"\ndf[df['day'] == 0]:")
  print(df[df['day'] == 0])


# Добавляем вызовы в конец файла
if __name__ == "__main__":
  diagnose_finrl()
  test_multi_stock()
  debug_internal_state()
  try_workaround()


# if __name__ == "__main__":
#   diagnose_finrl()
#   test_multi_stock()