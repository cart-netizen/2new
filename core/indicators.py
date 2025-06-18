import pandas as pd


def crossover_series(x: pd.Series, y: pd.Series) -> pd.Series:
  """Проверяет, пересекла ли серия x серию y снизу вверх."""
  return (x > y) & (x.shift(1) < y.shift(1))


def crossunder_series(x: pd.Series, y: pd.Series) -> pd.Series:
  """Проверяет, пересекла ли серия x серию y сверху вниз."""
  return (x < y) & (x.shift(1) > y.shift(1))