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
  """
  logger.info(f"Подготовка данных для FinRL. Символы: {symbols}")

  all_data = []

  for symbol in symbols:
    if symbol not in raw_data or raw_data[symbol].empty:
      logger.warning(f"Пропускаем {symbol} - нет данных")
      continue

    df = raw_data[symbol].copy()

    # Сбрасываем индекс
    if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
      df = df.reset_index()

    # Создаем колонку date
    if 'timestamp' in df.columns:
      df['date'] = pd.to_datetime(df['timestamp'])
    elif 'index' in df.columns:
      df['date'] = pd.to_datetime(df['index'])
    else:
      logger.error(f"Не найдена колонка с датой для {symbol}")
      continue

    # Добавляем символ
    df['tic'] = symbol

    # Проверяем наличие всех необходимых колонок
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
      logger.error(f"Отсутствуют колонки для {symbol}: {missing_cols}")
      continue

    # Убеждаемся, что все колонки числовые
    for col in required_cols:
      df[col] = pd.to_numeric(df[col], errors='coerce')

    # Убираем строки с NaN в критических колонках
    df = df.dropna(subset=required_cols)

    if df.empty:
      logger.warning(f"После очистки NaN не осталось данных для {symbol}")
      continue

    all_data.append(df)

  if not all_data:
    raise ValueError("Нет валидных данных для создания FinRL датафрейма")

  # Объединяем все данные
  combined_df = pd.concat(all_data, ignore_index=True)

  # КРИТИЧНО: Убедимся, что у нас есть данные для каждого символа на каждую дату
  # Это требование FinRL!
  unique_dates = combined_df['date'].unique()
  unique_tics = combined_df['tic'].unique()

  logger.info(f"Уникальных дат: {len(unique_dates)}, уникальных символов: {len(unique_tics)}")

  # Создаем полный DataFrame со всеми комбинациями дата-символ
  full_index = pd.MultiIndex.from_product([unique_dates, unique_tics], names=['date', 'tic'])
  full_df = pd.DataFrame(index=full_index).reset_index()

  # Объединяем с реальными данными
  final_df = full_df.merge(
    combined_df,
    on=['date', 'tic'],
    how='left'
  )

  # Заполняем пропуски
  # Группируем по символу и заполняем пропуски
  for tic in unique_tics:
    tic_mask = final_df['tic'] == tic
    for col in ['open', 'high', 'low', 'close']:
      final_df.loc[tic_mask, col] = final_df.loc[tic_mask, col].fillna(method='ffill').fillna(method='bfill')
    # Для volume используем 0 для пропущенных значений
    final_df.loc[tic_mask, 'volume'] = final_df.loc[tic_mask, 'volume'].fillna(0)

  # Финальная проверка и сортировка
  final_df = final_df.sort_values(['date', 'tic']).reset_index(drop=True)

  # Убедимся, что нет NaN в критических колонках
  critical_cols = ['open', 'high', 'low', 'close', 'volume']
  for col in critical_cols:
    if final_df[col].isna().any():
      logger.error(f"Остались NaN в колонке {col}")
      # Крайняя мера - заполняем средним значением
      final_df[col] = final_df[col].fillna(final_df[col].mean())

  logger.info(f"Финальный DataFrame: {len(final_df)} записей")
  logger.info(
    f"Проверка: {len(final_df)} должно равняться {len(unique_dates)} * {len(unique_tics)} = {len(unique_dates) * len(unique_tics)}")

  return final_df