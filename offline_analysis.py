# offline_analysis.py

import asyncio
import pandas as pd
import warnings

# Импортируем необходимые компоненты из нашего бота
from core.data_fetcher import DataFetcher
from core.bybit_connector import BybitConnector
from ml.feature_engineering import feature_engineer, analyze_feature_importance
from config import settings
from utils.logging_config import setup_logging
from core.enums import Timeframe

# Игнорируем будущие предупреждения от pandas, чтобы не мешали
warnings.simplefilter(action='ignore', category=FutureWarning)


async def run_feature_analysis():
  """
  Оффлайн-скрипт для анализа и вывода важности признаков.
  """
  print("--- Запуск анализа важности признаков ---")

  # 1. Инициализация компонентов для загрузки данных
  connector = BybitConnector()
  # Передаем пустой словарь настроек, т.к. они не нужны для этой задачи
  data_fetcher = DataFetcher(connector, settings={})

  # 2. Загружаем большой объем данных для анализа (например, для BTCUSDT)
  # Вы можете изменить символ и количество свечей
  symbol_to_analyze = "SOLUSDT"
  data_limit = 10000  # Берем больше данных для качественного анализа

  print(f"\n[1/3] Загрузка {data_limit} свечей для {symbol_to_analyze}...")
  # Используем мультитаймфрейм функцию, чтобы получить все признаки
  features, labels = await feature_engineer.create_multi_timeframe_features(symbol_to_analyze, data_fetcher)

  if features is None or labels is None:
    print("\n❌ Не удалось создать признаки и метки. Прерывание.")
    await connector.close()
    return

  print(
    f"✅ Данные успешно подготовлены. Размер выборки: {len(features)} наблюдений, {len(features.columns)} признаков.")

  # 3. Запускаем анализ важности
  print("\n[2/3] Расчет важности признаков... (Это может занять несколько минут)")
  importance_df = analyze_feature_importance(features, labels)

  if importance_df.empty:
    print("\n❌ Не удалось рассчитать важность признаков.")
  else:
    print("\n[3/3] Анализ завершен. Топ-20 самых важных признаков:")
    print("-" * 50)
    # Выводим топ-20 признаков с их общим скором
    print(importance_df.head(50)[['combined_score']])
    print(importance_df.tail(50)[['combined_score']])
    print("-" * 50)
    print(
      "\n💡 Рекомендация: Признаки с низкой важностью (например, score < 0.001) можно рассмотреть к удалению из `calculate_technical_indicators` для уменьшения шума и ускорения модели.")

  # 4. Закрываем соединение
  await connector.close()


if __name__ == "__main__":
  # Запускаем асинхронную функцию анализа
  asyncio.run(run_feature_analysis())
