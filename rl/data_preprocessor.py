import pandas as pd
import numpy as np
from typing import List, Dict


def prepare_data_for_finrl(
    raw_data: Dict[str, pd.DataFrame],
    symbols: List[str]
) -> pd.DataFrame:
  """
  Преобразует данные из формата вашего проекта в формат FinRL

  FinRL ожидает DataFrame со следующими колонками:
  - date
  - tic (символ)
  - open, high, low, close, volume
  - технические индикаторы
  """
  all_data = []

  for symbol in symbols:
    if symbol not in raw_data or raw_data[symbol].empty:
      continue

    df = raw_data[symbol].copy()

    # Сбрасываем индекс, чтобы timestamp стал колонкой
    df = df.reset_index()

    # Переименовываем колонки для FinRL
    df['date'] = df['timestamp']
    df['tic'] = symbol

    # Убеждаемся, что все числовые колонки имеют правильный тип
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
      if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    all_data.append(df)

  # Объединяем все данные
  combined_df = pd.concat(all_data, ignore_index=True)

  # Сортируем по дате и символу
  combined_df = combined_df.sort_values(['date', 'tic']).reset_index(drop=True)

  return combined_df