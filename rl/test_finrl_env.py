import pandas as pd
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


# Создаем минимальный тестовый DataFrame
def create_test_data():
  dates = pd.date_range(start='2023-01-01', periods=100, freq='H')

  # Данные для двух символов
  data = []
  for date in dates:
    for tic in ['BTCUSDT', 'ETHUSDT']:
      data.append({
        'date': date,
        'tic': tic,
        'open': np.random.uniform(100, 110),
        'high': np.random.uniform(110, 120),
        'low': np.random.uniform(90, 100),
        'close': np.random.uniform(95, 115),
        'volume': np.random.uniform(1000, 10000)
      })

  df = pd.DataFrame(data)
  return df


# Тестируем
df = create_test_data()
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())

# Пробуем создать среду
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
    state_space=10,
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
  print("✅ Среда создана успешно!")
except Exception as e:
  print(f"❌ Ошибка: {e}")
  import traceback

  traceback.print_exc()