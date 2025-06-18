# tune_and_compare_models.py

import asyncio
import pandas as pd
import warnings

# --- Инструменты для ML ---
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier

# --- Компоненты нашего бота ---
from core.data_fetcher import DataFetcher
from core.bybit_connector import BybitConnector
from ml.feature_engineering import feature_engineer
from ml.lorentzian_classifier import LorentzianClassifier
from config import settings
from utils.logging_config import setup_logging

# Игнорируем предупреждения для чистоты вывода
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


async def run_analysis():
  """
  Оффлайн-скрипт для подбора гиперпараметров и сравнения моделей.
  """
  print("--- Запуск анализа и сравнения моделей ---")

  # 1. Инициализация компонентов
  connector = BybitConnector()
  # Передаем пустой словарь настроек, т.к. они не нужны для этой задачи
  data_fetcher = DataFetcher(connector, settings={})

  # 2. Загрузка большого объема данных для анализа
  symbol_to_analyze = "BTCUSDT"
  print(f"\n[1/4] Загрузка данных для {symbol_to_analyze}...")
  features, labels = await feature_engineer.create_multi_timeframe_features(symbol_to_analyze, data_fetcher)

  if features is None or labels is None:
    print("\n❌ Не удалось создать признаки и метки. Прерывание.")
    await connector.close()
    return

  print(f"✅ Данные успешно подготовлены. Размер выборки: {len(features)} наблюдений.")

  # 3. Подбор гиперпараметров для LorentzianClassifier
  print("\n[2/4] Подбор гиперпараметров для LorentzianClassifier... (Может занять много времени)")

  param_grid = {
    'k_neighbors': [4, 8, 12, 16]  # Сетка значений k, которые нужно проверить
  }
  time_series_cv = TimeSeriesSplit(n_splits=5)

  # n_jobs=-1 использует все ядра CPU
  grid_search = GridSearchCV(LorentzianClassifier(), param_grid, cv=time_series_cv, scoring='accuracy', n_jobs=-1)
  grid_search.fit(features, labels)

  best_params = grid_search.best_params_
  best_score_lc = grid_search.best_score_
  print(f"✅ Подбор завершен. Лучшие параметры: {best_params}, Лучшая точность: {best_score_lc:.4f}")

  # 4. Обучение и оценка LightGBM на тех же данных для сравнения
  print("\n[3/4] Обучение и оценка LightGBM...")

  # Используем тот же TimeSeriesSplit, чтобы взять последнюю тестовую выборку
  train_index, test_index = list(time_series_cv.split(features))[-1]
  X_train, X_test = features.iloc[train_index], features.iloc[test_index]
  y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

  lgbm_model = LGBMClassifier(objective='multiclass', random_state=42)
  lgbm_model.fit(X_train, y_train)
  lgbm_predictions = lgbm_model.predict(X_test)
  accuracy_lgbm = accuracy_score(y_test, lgbm_predictions)
  print(f"✅ Обучение LightGBM завершено. Точность на тестовой выборке: {accuracy_lgbm:.4f}")

  # 5. Итоговый отчет
  print("\n--- [4/4] ИТОГОВЫЙ ОТЧЕТ ---")
  print("=" * 30)
  print(f"Lorentzian Classifier (best params: {best_params}): {best_score_lc:.4f}")
  print(f"LightGBM Classifier:                            {accuracy_lgbm:.4f}")
  print("=" * 30)

  if accuracy_lgbm > best_score_lc:
    print("\n🏆 РЕКОМЕНДАЦИЯ: LightGBM показывает лучшую точность на этих данных.")
  else:
    print("\n🏆 РЕКОМЕНДАЦИЯ: Ваш Lorentzian Classifier с оптимальными параметрами показывает лучший результат!")

  await connector.close()


if __name__ == "__main__":
  setup_logging("WARNING")  # Устанавливаем уровень WARNING, чтобы не видеть лишние INFO логи
  asyncio.run(run_analysis())