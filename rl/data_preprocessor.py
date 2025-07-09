import pandas as pd
import numpy as np
from typing import List, Dict

from main import logger
from utils.logging_config import get_logger


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

    # Проверяем наличие необходимых колонок
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
      logger.error(f"Отсутствуют колонки для {symbol}: {missing_cols}")
      continue

    # Переименовываем колонки для FinRL
    if 'timestamp' in df.columns:
      df['date'] = pd.to_datetime(df['timestamp'])
    elif 'index' in df.columns:
      df['date'] = pd.to_datetime(df['index'])
    else:
      # Создаем дату из индекса
      df['date'] = pd.to_datetime(df.index)

    df['tic'] = symbol

    # Убеждаемся, что все числовые колонки имеют правильный тип
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
      if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # Проверяем, что это не скаляр
        if isinstance(df[col].iloc[0], (int, float, np.number)):
          pass  # Все хорошо
        else:
          logger.error(f"Колонка {col} содержит нечисловые данные для {symbol}")

    # Убираем дубликаты по дате для каждого символа
    df = df.drop_duplicates(subset=['date'], keep='last')

    all_data.append(df)

  if not all_data:
    raise ValueError("Нет данных для создания FinRL датафрейма")

  # Объединяем все данные
  combined_df = pd.concat(all_data, ignore_index=True)

  # Проверяем, что у нас есть данные
  if combined_df.empty:
    raise ValueError("Объединенный датафрейм пуст")

  # Сортируем по дате и символу
  combined_df = combined_df.sort_values(['date', 'tic']).reset_index(drop=True)

  # Проверяем структуру данных
  logger.info(f"Подготовлено данных для FinRL:")
  logger.info(f"  Символов: {combined_df['tic'].nunique()}")
  logger.info(f"  Записей: {len(combined_df)}")
  logger.info(f"  Период: {combined_df['date'].min()} - {combined_df['date'].max()}")

  # Финальная проверка на NaN значения в критических колонках
  critical_cols = ['open', 'high', 'low', 'close', 'volume']
  for col in critical_cols:
    if col in combined_df.columns:
      nan_count = combined_df[col].isna().sum()
      if nan_count > 0:
        logger.warning(f"Найдено {nan_count} NaN значений в колонке {col}, заполняем нулями")
        combined_df[col] = combined_df[col].fillna(0)

  return combined_df