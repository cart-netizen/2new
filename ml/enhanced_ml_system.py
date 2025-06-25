# ml/enhanced_ml_system.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, \
  HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import entropy
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils import compute_sample_weight

from core.enums import SignalType
from utils.logging_config import get_logger
from ml.anomaly_detector import MarketAnomalyDetector, AnomalyType

logger = get_logger(__name__)


@dataclass
class MLPrediction:
  """Расширенное предсказание с дополнительной информацией"""
  signal_type: SignalType
  probability: float
  confidence: float
  model_agreement: float  # Согласованность моделей в ансамбле
  feature_importance: Dict[str, float]
  risk_assessment: Dict[str, Any]
  metadata: Dict[str, Any]


class AdvancedFeatureEngineer:
  """Продвинутый генератор признаков"""

  def __init__(self):
    self.feature_names = []
    self.feature_stats = {}
  #
  # def create_advanced_features(self, data: pd.DataFrame,
  #                              external_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
  #   """
  #   Создает продвинутые признаки включая межрыночные корреляции
  #   """
  #   logger.debug("Создание продвинутых признаков с сохранением базовых колонок...")
  #
  #   # Проверяем наличие базовых колонок
  #   base_columns = ['open', 'high', 'low', 'close', 'volume']
  #   available_base = [col for col in base_columns if col in data.columns]
  #
  #   logger.info(f"Доступные базовые колонки: {available_base}")
  #
  #   # Сохраняем оригинальный индекс и базовые данные
  #   original_index = data.index.copy()
  #   original_length = len(data)
  #   # base_data = data[available_base].copy() if available_base else pd.DataFrame(index=original_index)
  #
  #   # Создаем DataFrame для всех признаков, начиная с базовых
  #   # features = data.copy()
  #   if available_base:
  #     features = data[available_base].copy()
  #   else:
  #     features = pd.DataFrame(index=original_index)
  #
  #
  #   # original_index = data.index.copy()
  #   # original_length = len(data)
  #
  #   logger.debug(f"Создание признаков: входной размер={data.shape}, индекс={type(data.index)}")
  #
  #   if not isinstance(data.index, pd.DatetimeIndex):
  #     logger.debug("Преобразуем индекс в datetime для межрыночного анализа...")
  #     try:
  #       # Пытаемся преобразовать существующий индекс
  #       if hasattr(data.index, 'to_datetime'):
  #         data = data.copy()
  #         data.index = pd.to_datetime(data.index)
  #         logger.debug("✅ Индекс успешно преобразован в datetime")
  #       else:
  #         # Создаем искусственный datetime индекс
  #         logger.debug("Создаем искусственный datetime индекс...")
  #         start_date = pd.Timestamp('2024-01-01')
  #         freq = '1H'  # Частота данных
  #         new_index = pd.date_range(start=start_date, periods=len(data), freq=freq)
  #         data = data.copy()
  #         data.index = new_index
  #         logger.debug(f"✅ Создан datetime индекс: {len(data)} периодов с частотой {freq}")
  #     except Exception as index_error:
  #       logger.warning(f"Не удалось создать datetime индекс: {index_error}")
  #       # Продолжаем без datetime индекса
  #   else:
  #     logger.debug("✅ Данные уже имеют datetime индекс")
  #
  #   features = pd.DataFrame(index=data.index)
  #
  #   try:
  #     # 1. Микроструктурные признаки
  #     microstructure_features = self._create_microstructure_features(data)
  #     features = pd.concat([features, microstructure_features], axis=1)
  #
  #     # 2. Признаки рыночных режимов
  #     regime_features = self._create_regime_features(data)
  #     features = pd.concat([features, regime_features], axis=1)
  #
  #     # 3. Признаки на основе теории информации
  #     information_features = self._create_information_features(data)
  #     features = pd.concat([features, information_features], axis=1)
  #
  #     # 4. Нелинейные взаимодействия
  #     interaction_features = self._create_interaction_features(data)
  #     features = pd.concat([features, interaction_features], axis=1)
  #
  #     # 5. Межрыночные признаки (только если индексы совместимы)
  #     if external_data and isinstance(data.index, pd.DatetimeIndex):
  #       try:
  #         cross_market_features = self._create_cross_market_features(data, external_data)
  #         features = pd.concat([features, cross_market_features], axis=1)
  #       except Exception as cross_error:
  #         logger.warning(f"Ошибка межрыночных признаков: {cross_error}")
  #
  #     # 6. Временные признаки
  #     time_features = self._create_time_features(data)
  #     features = pd.concat([features, time_features], axis=1)
  #
  #     # 7. Признаки памяти рынка
  #     memory_features = self._create_memory_features(data)
  #     features = pd.concat([features, memory_features], axis=1)
  #
  #     # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Убеждаемся, что индекс не изменился
  #     if len(features) != original_length:
  #       logger.warning(f"Размер изменился при создании признаков: {original_length} -> {len(features)}")
  #       # Подрезаем или дополняем до оригинального размера
  #       if len(features) > original_length:
  #         features = features.iloc[:original_length]
  #       elif len(features) < original_length:
  #         # Дополняем недостающие строки нулями
  #         missing_rows = original_length - len(features)
  #         missing_index = original_index[-missing_rows:]
  #         missing_data = pd.DataFrame(0, index=missing_index, columns=features.columns)
  #         features = pd.concat([features, missing_data])
  #
  #     # Принудительно восстанавливаем оригинальный индекс
  #     features.index = original_index
  #
  #     logger.debug(
  #       f"Результат создания признаков: размер={features.shape}, индекс сохранен={features.index.equals(original_index)}")
  #
  #     # Сохраняем имена признаков
  #     self.feature_names = features.columns.tolist()
  #
  #     final_base_columns = [col for col in available_base if col in features.columns]
  #     logger.info(f"Сохранено базовых колонок в итоговых признаках: {final_base_columns}")
  #
  #     return features
  #
  #   except Exception as e:
  #     logger.error(f"Ошибка создания признаков: {e}")
  #     # Возвращаем пустой DataFrame с правильным индексом
  #     return pd.DataFrame(index=original_index)

#   def create_advanced_features(self, data: pd.DataFrame,
#                                external_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
#     """
#     Создает продвинутые признаки включая межрыночные корреляции
#     КРИТИЧЕСКИ ВАЖНО: Сохраняет базовые OHLCV колонки
#     """
#     logger.debug("Создание продвинутых признаков с сохранением базовых колонок...")
#
#     # Проверяем наличие базовых колонок
#     base_columns = ['open', 'high', 'low', 'close', 'volume']
#     available_base = [col for col in base_columns if col in data.columns]
#
#     logger.info(f"Доступные базовые колонки: {available_base}")
#
#     # Сохраняем оригинальный индекс и длину
#     original_index = data.index.copy()
#     original_length = len(data)
#
#     # КРИТИЧЕСКИ ВАЖНО: Сохраняем базовые колонки отдельно
#     base_data = data[available_base].copy() if available_base else None
#
#     logger.debug(f"Создание признаков: входной размер={data.shape}, индекс={type(data.index)}")
#
#     if not isinstance(data.index, pd.DatetimeIndex):
#       logger.debug("Преобразуем индекс в datetime для межрыночного анализа...")
#       try:
#         # Пытаемся преобразовать существующий индекс
#         if hasattr(data.index, 'to_datetime'):
#           data = data.copy()
#           data.index = pd.to_datetime(data.index)
#           logger.debug("✅ Индекс успешно преобразован в datetime")
#         else:
#           # Создаем искусственный datetime индекс
#           logger.debug("Создаем искусственный datetime индекс...")
#           start_date = pd.Timestamp('2024-01-01')
#           freq = '1H'  # Частота данных
#           new_index = pd.date_range(start=start_date, periods=len(data), freq=freq)
#           data = data.copy()
#           data.index = new_index
#           logger.debug(f"✅ Создан datetime индекс: {len(data)} периодов с частотой {freq}")
#       except Exception as index_error:
#         logger.warning(f"Не удалось создать datetime индекс: {index_error}")
#         # Продолжаем без datetime индекса
#     else:
#       logger.debug("✅ Данные уже имеют datetime индекс")
#
#     # Список для накопления всех признаков
#     all_features_list = []
#
#     try:
#       # 1. Микроструктурные признаки
#       microstructure_features = self._create_microstructure_features(data)
#       if not microstructure_features.empty:
#         all_features_list.append(microstructure_features)
#
#       # 2. Признаки рыночных режимов
#       regime_features = self._create_regime_features(data)
#       if not regime_features.empty:
#         all_features_list.append(regime_features)
#
#       # 3. Признаки на основе теории информации
#       information_features = self._create_information_features(data)
#       if not information_features.empty:
#         all_features_list.append(information_features)
#
#       # 4. Нелинейные взаимодействия
#       interaction_features = self._create_interaction_features(data)
#       if not interaction_features.empty:
#         all_features_list.append(interaction_features)
#
#       # 5. Межрыночные признаки (только если индексы совместимы)
#       if external_data and isinstance(data.index, pd.DatetimeIndex):
#         try:
#           cross_market_features = self._create_cross_market_features(data, external_data)
#           if not cross_market_features.empty:
#             all_features_list.append(cross_market_features)
#         except Exception as cross_error:
#           logger.warning(f"Ошибка межрыночных признаков: {cross_error}")
#
#       # 6. Временные признаки
#       time_features = self._create_time_features(data)
#       if not time_features.empty:
#         all_features_list.append(time_features)
#
#       # 7. Признаки памяти рынка
#       memory_features = self._create_memory_features(data)
#       if not memory_features.empty:
#         all_features_list.append(memory_features)
# #----------------------------------------------------------------------------------
#       # # КРИТИЧЕСКОЕ ОБЪЕДИНЕНИЕ: Сначала базовые колонки, потом все остальное
#       # if base_data is not None and not base_data.empty:
#       #   # Начинаем с базовых данных
#       #   features = base_data.copy()
#       #
#       #   # Добавляем все дополнительные признаки
#       #   if all_features_list:
#       #     # Объединяем все дополнительные признаки
#       #     additional_features = pd.concat(all_features_list, axis=1)
#       #     # Добавляем их к базовым данным
#       #     features = pd.concat([features, additional_features], axis=1)
#       # else:
#       #   # Если нет базовых данных, объединяем только дополнительные признаки
#       #   if all_features_list:
#       #     features = pd.concat(all_features_list, axis=1)
#       #   else:
#       #     features = pd.DataFrame(index=original_index)
# #----------------------------------------------------------------------------------
#       # КРИТИЧЕСКОЕ ОБЪЕДИНЕНИЕ: Собираем все части в один список
#       all_parts = []
#       if base_data is not None and not base_data.empty:
#         all_parts.append(base_data)
#
#       if all_features_list:
#         all_parts.extend(all_features_list)
#
#       # Выполняем объединение один раз для всех частей
#       if all_parts:
#         features = pd.concat(all_parts, axis=1)
#       else:
#         features = pd.DataFrame(index=original_index)
#
#
#       # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Убеждаемся, что индекс не изменился
#       if len(features) != original_length:
#         logger.warning(f"Размер изменился при создании признаков: {original_length} -> {len(features)}")
#         # Подрезаем или дополняем до оригинального размера
#         if len(features) > original_length:
#           features = features.iloc[:original_length]
#         elif len(features) < original_length:
#           # Дополняем недостающие строки нулями
#           missing_rows = original_length - len(features)
#           missing_index = original_index[-missing_rows:]
#           missing_data = pd.DataFrame(0, index=missing_index, columns=features.columns)
#           features = pd.concat([features, missing_data])
#
#       # Принудительно восстанавливаем оригинальный индекс
#       features.index = original_index
#
#       logger.debug(
#         f"Результат создания признаков: размер={features.shape}, индекс сохранен={features.index.equals(original_index)}")
#
#       # Сохраняем имена признаков
#       self.feature_names = features.columns.tolist()
#
#       # Проверяем наличие базовых колонок в финальных признаках
#       final_base_columns = [col for col in available_base if col in features.columns]
#       logger.info(f"Сохранено базовых колонок в итоговых признаках: {final_base_columns}")
#
#       if not final_base_columns and available_base:
#         logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА: Базовые колонки были потеряны в процессе создания признаков!")
#         logger.error(f"Доступные колонки в результате: {list(features.columns[:10])}")
#
#       return features
#
#     except Exception as e:
#       logger.error(f"Ошибка создания признаков: {e}")
#       # Возвращаем хотя бы базовые данные если они есть
#       if base_data is not None and not base_data.empty:
#         return base_data
#       else:
#         return pd.DataFrame(index=original_index)

  # def create_advanced_features(self, data: pd.DataFrame,
  #                              external_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
  #   """
  #   ОПТИМАЛЬНАЯ ВЕРСИЯ: Создает продвинутые признаки, сохраняя базовые колонки
  #   и избегая удвоения строк.
  #   """
  #   logger.debug("Запуск оптимальной версии create_advanced_features...")
  #
  #   # --- ШАГ 0: УСТРАНЕНИЕ КОРЕННОЙ ПРОБЛЕМЫ - НЕУНИКАЛЬНЫЙ ИНДЕКС ---
  #   if not data.index.is_unique:
  #     logger.warning(
  #       f"Обнаружен неуникальный индекс. Размер до дедупликации: {len(data)}. Выполняется удаление дубликатов...")
  #     data = data.loc[~data.index.duplicated(keep='first')]
  #     logger.info(f"Размер после дедупликации индекса: {len(data)}")
  #
  #   # --- Шаг 1: Надежное сохранение базовых данных и исходного индекса ---
  #   original_index = data.index.copy()
  #   original_length = len(data)
  #   base_columns = ['open', 'high', 'low', 'close', 'volume']
  #   available_base = [col for col in base_columns if col in data.columns]
  #   logger.info(f"Найдены базовые колонки для сохранения: {available_base}")
  #   base_data = data[available_base].copy() if available_base else None
  #
  #   # --- Шаг 2: Обработка индекса для анализа (без изменений) ---
  #   if not isinstance(data.index, pd.DatetimeIndex):
  #     logger.debug("Преобразуем индекс в datetime для межрыночного анализа...")
  #     try:
  #       if hasattr(data.index, 'to_datetime'):
  #         data = data.copy()
  #         data.index = pd.to_datetime(data.index)
  #       else:
  #         start_date = pd.Timestamp('2024-01-01')
  #         freq = '1H'
  #         new_index = pd.date_range(start=start_date, periods=len(data), freq=freq)
  #         data = data.copy()
  #         data.index = new_index
  #       logger.debug("✅ Индекс успешно подготовлен для анализа.")
  #     except Exception as index_error:
  #       logger.warning(f"Не удалось создать datetime индекс: {index_error}")
  #   else:
  #     logger.debug("✅ Данные уже имеют datetime индекс.")
  #
  #   # --- Шаг 3: Генерация всех дополнительных признаков ---
  #   all_features_list = []
  #   try:
  #     # Микроструктурные признаки
  #     microstructure_features = self._create_microstructure_features(data)
  #     if not microstructure_features.empty:
  #       all_features_list.append(microstructure_features)
  #
  #     # Признаки рыночных режимов
  #     regime_features = self._create_regime_features(data)
  #     if not regime_features.empty:
  #       all_features_list.append(regime_features)
  #
  #     # Признаки на основе теории информации
  #     information_features = self._create_information_features(data)
  #     if not information_features.empty:
  #       all_features_list.append(information_features)
  #
  #     # Нелинейные взаимодействия
  #     interaction_features = self._create_interaction_features(data)
  #     if not interaction_features.empty:
  #       all_features_list.append(interaction_features)
  #
  #     # Межрыночные признаки
  #     if external_data and isinstance(data.index, pd.DatetimeIndex):
  #       try:
  #         cross_market_features = self._create_cross_market_features(data, external_data)
  #         if not cross_market_features.empty:
  #           all_features_list.append(cross_market_features)
  #       except Exception as cross_error:
  #         logger.warning(f"Ошибка межрыночных признаков: {cross_error}")
  #
  #     # Временные признаки
  #     time_features = self._create_time_features(data)
  #     if not time_features.empty:
  #       all_features_list.append(time_features)
  #
  #     # Признаки памяти рынка
  #     memory_features = self._create_memory_features(data)
  #     if not memory_features.empty:
  #       all_features_list.append(memory_features)
  #
  #     # --- Шаг 4: ОПТИМАЛЬНОЕ И БЕЗОПАСНОЕ ОБЪЕДИНЕНИЕ ---
  #     all_parts_to_concat = []
  #     if base_data is not None:
  #       all_parts_to_concat.append(base_data)
  #
  #     if all_features_list:
  #       all_parts_to_concat.extend(all_features_list)
  #
  #     if all_parts_to_concat:
  #       features = pd.concat(all_parts_to_concat, axis=1)
  #     else:
  #       # Fallback на случай, если ничего не было сгенерировано
  #       features = pd.DataFrame(index=original_index)
  #
  #     # --- Шаг 5: Финальная проверка (теперь не должна находить расхождений) ---
  #     if len(features) != original_length:
  #       logger.error(f"РАЗМЕР ВСЕ РАВНО ИЗМЕНИЛСЯ: {original_length} -> {len(features)}. ПРОВЕРЬТЕ ЛОГИКУ!")
  #       if len(features) > original_length:
  #         features = features.iloc[:original_length]
  #
  #     features.index = original_index
  #
  #     logger.debug(
  #       f"Результат создания признаков: размер={features.shape}, индекс сохранен={features.index.equals(original_index)}")
  #
  #     # --- Шаг 6: Финальные логи и возврат результата ---
  #     self.feature_names = features.columns.tolist()
  #     final_base_cols_check = [col for col in available_base if col in features.columns]
  #     logger.info(
  #       f"✅ Успешно создано {len(self.feature_names)} признаков. Сохраненные базовые колонки: {final_base_cols_check}")
  #
  #     if len(available_base) > 0 and not final_base_cols_check:
  #       logger.error("КРИТИЧЕСКАЯ ОШИБКА: Базовые колонки были потеряны!")
  #
  #     return features
  #
  #   except Exception as e:
  #     logger.error(f"Критическая ошибка при создании признаков: {e}")
  #     # В случае ошибки возвращаем хотя бы базовые данные, если они есть
  #     return base_data if base_data is not None else pd.DataFrame(index=original_index)

  def create_advanced_features(self, data: pd.DataFrame,
                               external_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """
    ДИАГНОСТИЧЕСКАЯ ВЕРСИЯ: Ищет, какая из вспомогательных функций
    возвращает DataFrame неправильного размера.
    """
    logger.info("=" * 20 + " ЗАПУСК ДИАГНОСТИЧЕСКОЙ ВЕРСИИ " + "=" * 20)

    # --- Шаг 0: Дедупликация индекса (оставляем как лучшую практику) ---
    if not data.index.is_unique:
      logger.warning(f"Обнаружен неуникальный индекс. Размер до: {len(data)}. Дедупликация...")
      data = data.loc[~data.index.duplicated(keep='first')]
      logger.info(f"Размер после дедупликации: {len(data)}")

    original_index = data.index.copy()
    original_length = len(data)
    logger.info(f"Ожидаемый размер для всех наборов признаков: {original_length} строк.")

    base_columns = ['open', 'high', 'low', 'close', 'volume']
    available_base = [col for col in base_columns if col in data.columns]
    base_data = data[available_base].copy() if available_base else None
    if base_data is not None:
      logger.info(f"Проверка базовых данных: shape={base_data.shape}. ✅")

    # --- Вспомогательная функция для проверки и логирования ---
    def check_and_add(feature_df, name, collection):
      if feature_df is not None and not feature_df.empty:
        logger.info(f"--> Проверка набора '{name}': shape={feature_df.shape}")
        if len(feature_df) != original_length:
          logger.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          logger.error(f"!!! НАЙДЕНА ПРОБЛЕМА: НЕПРАВИЛЬНЫЙ РАЗМЕР В '{name}' !!!")
          logger.error(f"!!! Ожидалось: {original_length}, Получено: {len(feature_df)}         !!!")
          logger.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        collection.append(feature_df)
      else:
        logger.info(f"--> Набор '{name}' пуст и будет пропущен.")

    all_features_list = []
    try:
      # --- Последовательная генерация и немедленная проверка каждого набора признаков ---
      check_and_add(self._create_microstructure_features(data), 'microstructure', all_features_list)
      check_and_add(self._create_regime_features(data), 'regime', all_features_list)
      check_and_add(self._create_information_features(data), 'information', all_features_list)
      check_and_add(self._create_interaction_features(data), 'interaction', all_features_list)

      if external_data and isinstance(data.index, pd.DatetimeIndex):
        try:
          cross_features = self._create_cross_market_features(data, external_data)
          check_and_add(cross_features, 'cross_market', all_features_list)
        except Exception as cross_error:
          logger.warning(f"Ошибка в _create_cross_market_features: {cross_error}")

      check_and_add(self._create_time_features(data), 'time', all_features_list)
      check_and_add(self._create_memory_features(data), 'memory', all_features_list)

      # --- Финальное объединение ---
      logger.info("Все наборы признаков сгенерированы. Начинаю финальное объединение...")
      all_parts = [base_data] + all_features_list if base_data is not None else all_features_list

      if all_parts:
        features = pd.concat(all_parts, axis=1)
      else:
        features = pd.DataFrame(index=original_index)

      logger.info(f"Финальный размер DataFrame после concat: {features.shape}")
      return features

    except Exception as e:
      logger.error(f"Критическая ошибка в процессе создания признаков: {e}", exc_info=True)
      return base_data if base_data is not None else pd.DataFrame(index=original_index)

  def _create_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Микроструктурные признаки рынка"""
    features = pd.DataFrame(index=data.index)

    # Эффективный спред
    features['effective_spread'] = 2 * np.abs(
      data['close'] - (data['high'] + data['low']) / 2
    ) / data['close']

    # Индикатор дисбаланса объема
    volume_ma = data['volume'].rolling(20).mean()
    features['volume_imbalance'] = (data['volume'] - volume_ma) / (volume_ma + 1e-9)

    # Токсичность потока ордеров (приближение)
    returns = data['close'].pct_change()
    features['order_flow_toxicity'] = (
        returns.rolling(10).std() * np.sqrt(data['volume'].rolling(10).mean())
    )

    # Вероятность информированной торговли (PIN approximation)
    buy_volume = data['volume'] * (data['close'] > data['open']).astype(int)
    sell_volume = data['volume'] * (data['close'] <= data['open']).astype(int)

    features['pin_approximation'] = np.abs(
      buy_volume.rolling(20).sum() - sell_volume.rolling(20).sum()
    ) / (buy_volume.rolling(20).sum() + sell_volume.rolling(20).sum() + 1e-9)

    # Realized variance
    features['realized_variance'] = returns.rolling(20).apply(
      lambda x: np.sum(x ** 2)
    )

    # Amihud illiquidity
    features['amihud_illiquidity'] = (
        np.abs(returns) / (data['volume'] * data['close'] + 1e-9)
    ).rolling(20).mean()

    return features

  def _create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Признаки рыночных режимов"""
    features = pd.DataFrame(index=data.index)

    returns = data['close'].pct_change()

    # Режим волатильности (через Markov regime switching approximation)
    vol_short = returns.rolling(10).std()
    vol_long = returns.rolling(50).std()
    features['volatility_regime'] = vol_short / (vol_long + 1e-9)

    # Трендовый режим
    sma_10 = data['close'].rolling(10).mean()
    sma_50 = data['close'].rolling(50).mean()
    features['trend_regime'] = (sma_10 - sma_50) / (sma_50 + 1e-9)

    # Режим моментума
    features['momentum_regime'] = returns.rolling(20).mean() / (
        returns.rolling(20).std() + 1e-9
    )

    # Режим ликвидности
    volume_ratio = data['volume'] / data['volume'].rolling(50).mean()
    spread_ratio = (data['high'] - data['low']) / data['close']
    features['liquidity_regime'] = volume_ratio / (spread_ratio + 1e-9)

    # Вероятности переходов между режимами
    for col in ['volatility_regime', 'trend_regime']:
      if col in features:
        features[f'{col}_change'] = features[col].diff()
        features[f'{col}_persistence'] = features[col].rolling(10).apply(
          lambda x: len(x[x > x.median()]) / len(x)
        )

    return features

  def _create_information_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Создает признаки на основе теории информации с защитой от ошибок
    """
    features = pd.DataFrame(index=data.index)

    try:
      # Проверяем наличие необходимых колонок
      required_cols = ['close', 'volume']
      if not all(col in data.columns for col in required_cols):
        logger.warning("Недостаточно данных для информационных признаков")
        return features

      # Вычисляем доходности
      returns = data['close'].pct_change().fillna(0)

      # Убираем бесконечные значения
      returns = returns.replace([np.inf, -np.inf], 0)
      volume_clean = data['volume'].replace([np.inf, -np.inf], data['volume'].median())

      # Проверяем на достаточность данных
      if len(returns) < 100:
        logger.warning("Недостаточно данных для расчета информационных признаков")
        return features

      # Взаимная информация между объемом и доходностью
      logger.debug("Расчет взаимной информации объем-доходность...")
      features['volume_return_mi'] = self._rolling_mutual_information(
        volume_clean, returns, window=50
      )

      # Энтропия доходности (скользящая)
      logger.debug("Расчет скользящей энтропии...")
      features['return_entropy'] = self._rolling_entropy(returns, window=30)

      # Информационное отношение
      if 'high' in data.columns and 'low' in data.columns:
        price_range = (data['high'] - data['low']).fillna(0)
        price_range = price_range.replace([np.inf, -np.inf], 0)
        features['price_range_entropy'] = self._rolling_entropy(price_range, window=30)

      # Заполняем NaN нулями
      features = features.fillna(0)

      logger.debug(f"Создано {len(features.columns)} информационных признаков")

    except Exception as e:
      logger.error(f"Ошибка при создании информационных признаков: {e}")
      # Возвращаем пустой DataFrame в случае ошибки
      features = pd.DataFrame(index=data.index)

    return features

  def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Нелинейные взаимодействия между признаками"""
    features = pd.DataFrame(index=data.index)

    # Базовые признаки для взаимодействий
    returns = data['close'].pct_change()
    volume_norm = data['volume'] / data['volume'].rolling(50).mean()
    volatility = returns.rolling(20).std()

    # Полиномиальные взаимодействия
    features['volume_volatility_interaction'] = volume_norm * volatility
    features['volume_returns_squared'] = volume_norm * returns ** 2

    # Тригонометрические взаимодействия (для захвата циклов)
    time_of_day = pd.to_datetime(data.index).hour + pd.to_datetime(data.index).minute / 60
    features['volume_time_sin'] = volume_norm * np.sin(2 * np.pi * time_of_day / 24)
    features['volatility_time_cos'] = volatility * np.cos(2 * np.pi * time_of_day / 24)

    # Логарифмические взаимодействия
    features['log_volume_returns'] = np.log(volume_norm + 1) * returns

    # Экспоненциальные взаимодействия
    features['exp_decay_volume'] = volume_norm * np.exp(-np.abs(returns) * 10)

    return features

  def _create_cross_market_features(self, data: pd.DataFrame,
                                    external_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """
    Создает признаки межрыночного анализа с исправлением проблем timezone и индексов
    """
    features = pd.DataFrame(index=data.index)

    if not external_data:
      logger.debug("Нет внешних данных для межрыночного анализа")
      return features

    try:
      # Убедимся, что основные данные имеют datetime индекс
      if not isinstance(data.index, pd.DatetimeIndex):
        logger.warning("Основные данные не имеют datetime индекс, пропускаем межрыночный анализ")
        return features

      for market_name, market_data in external_data.items():
        try:
          if market_data is None or market_data.empty:
            continue

          # ======================= НОВОЕ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ =======================
          # Дедупликация индекса внешних данных для предотвращения "взрыва" при соединении
          if hasattr(market_data, 'index') and not market_data.index.is_unique:
            logger.warning(f"Внешние данные '{market_name}' имеют неуникальный индекс. Выполняется дедупликация...")
            market_data = market_data.loc[~market_data.index.duplicated(keep='first')]
          # ==========================================================================

          # Проверяем и исправляем тип индекса внешних данных
          if not isinstance(market_data.index, pd.DatetimeIndex):
            # Пытаемся преобразовать индекс в datetime
            if 'timestamp' in market_data.columns:
              market_data = market_data.set_index('timestamp')
              if not isinstance(market_data.index, pd.DatetimeIndex):
                market_data.index = pd.to_datetime(market_data.index)
            elif hasattr(market_data.index, 'to_datetime'):
              try:
                market_data.index = pd.to_datetime(market_data.index)
              except Exception:
                logger.warning(f"Не удалось преобразовать индекс {market_name} в datetime, пропускаем")
                continue
            else:
              logger.warning(f"Индекс данных {market_name} не является datetime, пропускаем")
              continue

          # ИСПРАВЛЕНИЕ ПРОБЛЕМЫ С TIMEZONE
          # Убираем информацию о часовом поясе из обоих индексов для совместимости
          try:
            if data.index.tz is not None:
              data_index_naive = data.index.tz_localize(None)
            else:
              data_index_naive = data.index

            if market_data.index.tz is not None:
              market_index_naive = market_data.index.tz_localize(None)
            else:
              market_index_naive = market_data.index

            # Создаем временные DataFrame с naive индексами
            data_temp = data.copy()
            data_temp.index = data_index_naive

            market_temp = market_data.copy()
            market_temp.index = market_index_naive

          except Exception as tz_error:
            logger.warning(f"Ошибка обработки timezone для {market_name}: {tz_error}")
            continue

          # Проверяем наличие необходимых колонок
          if 'close' not in market_temp.columns:
            logger.warning(f"Нет колонки 'close' в данных {market_name}")
            continue

          # Выравниваем данные по времени с обработкой ошибок
          try:
            # Используем inner join для безопасного выравнивания
            common_index = data_temp.index.intersection(market_temp.index)

            if len(common_index) < 10:  # Минимум пересечений
              logger.warning(f"Недостаточно общих временных точек с {market_name}: {len(common_index)}")
              continue

            # Берем данные только по общим временным точкам
            main_aligned = data_temp.loc[common_index, 'close'] if 'close' in data_temp.columns else data_temp.loc[
                                                                                                       common_index].iloc[
                                                                                                     :, 0]
            market_aligned = market_temp.loc[common_index, 'close']

            # Убираем NaN значения
            valid_mask = ~(pd.isna(main_aligned) | pd.isna(market_aligned))

            if valid_mask.sum() < 10:
              logger.warning(f"Недостаточно валидных данных для {market_name}")
              continue

            main_clean = main_aligned[valid_mask]
            market_clean = market_aligned[valid_mask]

            # Вычисляем доходности
            main_returns = main_clean.pct_change().fillna(0)
            market_returns = market_clean.pct_change().fillna(0)

            # Корреляция (скользящая)
            window_size = min(30, len(main_returns) // 2)
            if window_size < 5:
              window_size = 5
            correlation = main_returns.rolling(window=window_size).corr(market_returns)

            # Коинтеграция (упрощенная версия)
            spread = main_clean - market_clean
            spread_mean = spread.rolling(window=min(20, len(spread) // 2)).mean()
            spread_std = spread.rolling(window=min(20, len(spread) // 2)).std()
            z_score = (spread - spread_mean) / (spread_std + 1e-8)

            # Относительная сила
            relative_strength = main_clean / (market_clean + 1e-8)
            rs_sma = relative_strength.rolling(window=min(14, len(relative_strength) // 2)).mean()

            # Добавляем признаки в результирующий DataFrame
            # ИСПРАВЛЕНИЕ: Правильное выравнивание индексов
            try:
              # Создаем маппинг между вычисленными признаками и оригинальным индексом данных

              # Для correlation
              if len(correlation) > 0 and not correlation.empty:
                # Берем индексы из common_index, которые соответствуют вычисленным значениям
                valid_corr_idx = correlation.dropna().index
                if len(valid_corr_idx) > 0:
                  # Найдем соответствующие позиции в оригинальном индексе
                  corr_values = correlation.dropna().values
                  # Создаем Series с правильным количеством значений
                  if len(corr_values) <= len(data.index):
                    corr_series = pd.Series(
                      index=data.index,
                      dtype=float,
                      name=f'{market_name.lower()}_correlation'
                    )
                    # Заполняем значениями с конца (последние вычисленные значения)
                    corr_series.iloc[-len(corr_values):] = corr_values
                    corr_series = corr_series.fillna(method='ffill').fillna(0)
                  else:
                    # Если значений больше, берем последние
                    corr_series = pd.Series(
                      corr_values[-len(data.index):],
                      index=data.index,
                      name=f'{market_name.lower()}_correlation'
                    )
                  features[f'{market_name.lower()}_correlation'] = corr_series
                else:
                  features[f'{market_name.lower()}_correlation'] = 0.0
              else:
                features[f'{market_name.lower()}_correlation'] = 0.0

              # Для z_score
              if len(z_score) > 0 and not z_score.empty:
                valid_zscore_idx = z_score.dropna().index
                if len(valid_zscore_idx) > 0:
                  zscore_values = z_score.dropna().values
                  if len(zscore_values) <= len(data.index):
                    zscore_series = pd.Series(
                      index=data.index,
                      dtype=float,
                      name=f'{market_name.lower()}_z_score'
                    )
                    zscore_series.iloc[-len(zscore_values):] = zscore_values
                    zscore_series = zscore_series.fillna(method='ffill').fillna(0)
                  else:
                    zscore_series = pd.Series(
                      zscore_values[-len(data.index):],
                      index=data.index,
                      name=f'{market_name.lower()}_z_score'
                    )
                  features[f'{market_name.lower()}_z_score'] = zscore_series
                else:
                  features[f'{market_name.lower()}_z_score'] = 0.0
              else:
                features[f'{market_name.lower()}_z_score'] = 0.0

              # Для relative_strength
              if len(rs_sma) > 0 and not rs_sma.empty:
                valid_rs_idx = rs_sma.dropna().index
                if len(valid_rs_idx) > 0:
                  rs_values = rs_sma.dropna().values
                  if len(rs_values) <= len(data.index):
                    rs_series = pd.Series(
                      index=data.index,
                      dtype=float,
                      name=f'{market_name.lower()}_relative_strength'
                    )
                    rs_series.iloc[-len(rs_values):] = rs_values
                    rs_series = rs_series.fillna(method='ffill').fillna(1.0)
                  else:
                    rs_series = pd.Series(
                      rs_values[-len(data.index):],
                      index=data.index,
                      name=f'{market_name.lower()}_relative_strength'
                    )
                  features[f'{market_name.lower()}_relative_strength'] = rs_series
                else:
                  features[f'{market_name.lower()}_relative_strength'] = 1.0
              else:
                features[f'{market_name.lower()}_relative_strength'] = 1.0

              logger.debug(f"Создано 3 межрыночных признака для {market_name}")

            except Exception as feature_add_error:
              logger.warning(f"Ошибка добавления признаков для {market_name}: {feature_add_error}")
              # Добавляем нулевые признаки как fallback
              features[f'{market_name.lower()}_correlation'] = 0.0
              features[f'{market_name.lower()}_z_score'] = 0.0
              features[f'{market_name.lower()}_relative_strength'] = 1.0
              continue

          except Exception as align_error:
            logger.warning(f"Ошибка выравнивания данных для {market_name}: {align_error}")
            # Добавляем default значения при ошибке выравнивания
            features[f'{market_name.lower()}_correlation'] = 0.0
            features[f'{market_name.lower()}_z_score'] = 0.0
            features[f'{market_name.lower()}_relative_strength'] = 1.0
            continue

        except Exception as market_error:
          logger.warning(f"Ошибка обработки данных рынка {market_name}: {market_error}")
          # Добавляем default значения при любой ошибке с рынком
          features[f'{market_name.lower()}_correlation'] = 0.0
          features[f'{market_name.lower()}_z_score'] = 0.0
          features[f'{market_name.lower()}_relative_strength'] = 1.0
          continue

      # Финальная проверка и заполнение NaN значений
      features = features.fillna(method='ffill').fillna(0)

      # Убеждаемся, что все признаки имеют правильный тип
      for col in features.columns:
        if features[col].dtype == 'object':
          features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)

      logger.debug(f"Создано {len(features.columns)} межрыночных признаков")

    except Exception as e:
      logger.error(f"Критическая ошибка в создании межрыночных признаков: {e}")
      features = pd.DataFrame(index=data.index)  # Возвращаем пустой DataFrame

    return features

  # Также добавьте вспомогательный метод для безопасного выравнивания данных:

  def _safe_align_data(self, main_data: pd.DataFrame, external_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Безопасное выравнивание данных с разными типами индексов
    """
    try:
      # Убеждаемся, что оба DataFrame имеют datetime индексы
      if not isinstance(main_data.index, pd.DatetimeIndex):
        if 'timestamp' in main_data.columns:
          main_data = main_data.set_index('timestamp')
        main_data.index = pd.to_datetime(main_data.index)

      if not isinstance(external_data.index, pd.DatetimeIndex):
        if 'timestamp' in external_data.columns:
          external_data = external_data.set_index('timestamp')
        external_data.index = pd.to_datetime(external_data.index)

      # Находим пересечение индексов
      common_index = main_data.index.intersection(external_data.index)

      if len(common_index) == 0:
        raise ValueError("Нет общих временных точек")

      # Возвращаем выровненные данные
      return main_data.loc[common_index], external_data.loc[common_index]

    except Exception as e:
      logger.error(f"Ошибка выравнивания данных: {e}")
      raise

  def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Временные признаки"""
    features = pd.DataFrame(index=data.index)

    dt_index = pd.to_datetime(data.index)

    # Время суток (циклическое кодирование)
    hour = dt_index.hour + dt_index.minute / 60
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # День недели
    day_of_week = dt_index.dayofweek
    features['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    features['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    # День месяца
    day_of_month = dt_index.day
    features['month_day_sin'] = np.sin(2 * np.pi * day_of_month / 30)
    features['month_day_cos'] = np.cos(2 * np.pi * day_of_month / 30)

    # Близость к важным событиям (выходные, начало/конец месяца)
    features['is_monday'] = (day_of_week == 0).astype(int)
    features['is_friday'] = (day_of_week == 4).astype(int)
    features['is_month_start'] = (day_of_month <= 3).astype(int)
    features['is_month_end'] = (day_of_month >= 28).astype(int)

    # Торговая сессия
    features['asian_session'] = ((hour >= 0) & (hour < 8)).astype(int)
    features['european_session'] = ((hour >= 8) & (hour < 16)).astype(int)
    features['american_session'] = ((hour >= 16) & (hour < 24)).astype(int)

    return features

  def _create_memory_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Признаки памяти рынка и долгосрочных зависимостей"""
    features = pd.DataFrame(index=data.index)

    returns = data['close'].pct_change()

    # Автокорреляционная функция на разных лагах
    for lag in [1, 5, 10, 20, 50]:
      features[f'autocorr_lag_{lag}'] = returns.rolling(100).apply(
        lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
      )

    # Частичная автокорреляция
    features['partial_autocorr'] = self._rolling_partial_autocorr(returns, lag=5, window=100)

    # Long memory parameter (d) estimation
    features['long_memory_d'] = self._estimate_long_memory(returns, window=100)

    # Mean reversion speed
    features['mean_reversion_speed'] = self._calculate_mean_reversion_speed(
      data['close'], window=50
    )

    # Persistence of shocks
    features['shock_persistence'] = self._calculate_shock_persistence(returns, window=50)

    return features

  # Вспомогательные методы

  @staticmethod
  def _calculate_shannon_entropy(series: pd.Series, bins: int = 10) -> float:
    """Расчет энтропии Шеннона"""
    if len(series.dropna()) < bins:
      return np.nan

    hist, _ = np.histogram(series.dropna(), bins=bins)
    hist = hist / hist.sum()
    hist = hist[hist > 0]

    return -np.sum(hist * np.log2(hist))

  def _rolling_mutual_information(self, x: pd.Series, y: pd.Series, window: int = 50) -> pd.Series:
    """
    Вычисляет скользящую взаимную информацию между двумя сериями с защитой от NaN
    """

    def mi(x_vals, y_vals):
      try:
        # Удаляем NaN значения
        mask = ~(pd.isna(x_vals) | pd.isna(y_vals))
        x_clean = x_vals[mask]
        y_clean = y_vals[mask]

        # Проверяем, что остались данные
        if len(x_clean) < 10:  # Минимум данных для расчета
          return 0.0

        # Проверяем на бесконечные значения
        if not (np.isfinite(x_clean).all() and np.isfinite(y_clean).all()):
          return 0.0

        # Проверяем на постоянные значения
        if x_clean.std() == 0 or y_clean.std() == 0:
          return 0.0

        # Дискретизация
        bins = min(10, len(x_clean) // 5, len(y_clean) // 5)
        if bins < 2:
          bins = 2

        x_discrete = pd.cut(x_clean, bins=bins, labels=False, duplicates='drop')
        y_discrete = pd.cut(y_clean, bins=bins, labels=False, duplicates='drop')

        # Убираем NaN после дискретизации
        discrete_mask = ~(pd.isna(x_discrete) | pd.isna(y_discrete))
        x_discrete = x_discrete[discrete_mask]
        y_discrete = y_discrete[discrete_mask]

        if len(x_discrete) < 5:
          return 0.0

        # Строим гистограмму
        xy_hist = np.histogram2d(x_discrete, y_discrete, bins=bins)[0]

        # Добавляем малое значение для избежания log(0)
        xy_hist = xy_hist + 1e-10

        # Нормализация
        xy_hist = xy_hist / xy_hist.sum()
        x_hist = xy_hist.sum(axis=1)
        y_hist = xy_hist.sum(axis=0)

        # Расчет взаимной информации
        mi_value = 0.0
        for i in range(len(x_hist)):
          for j in range(len(y_hist)):
            if xy_hist[i, j] > 1e-10:
              mi_value += xy_hist[i, j] * np.log2(xy_hist[i, j] / (x_hist[i] * y_hist[j]))

        return max(0.0, mi_value)  # MI не может быть отрицательной

      except Exception as e:
        # В случае любой ошибки возвращаем 0
        return 0.0

    # Применяем функцию скользящим окном
    result = pd.Series(index=x.index, dtype=float)

    for i in range(len(x)):
      start_idx = max(0, i - window)
      end_idx = i + 1

      if end_idx - start_idx < 10:  # Минимум данных
        result.iloc[i] = 0.0
      else:
        x_window = x.iloc[start_idx:end_idx]
        y_window = y.iloc[start_idx:end_idx]
        result.iloc[i] = mi(x_window, y_window)

    return result

  def _calculate_conditional_entropy(self, x: pd.Series, y: pd.Series,
                                     window: int, bins: int = 10) -> pd.Series:
    """H(X|Y) = H(X,Y) - H(Y)"""

    def cond_entropy(x_window, y_window):
      if len(x_window) < bins * 2:
        return np.nan

      # Совместная энтропия
      xy_combined = pd.DataFrame({'x': x_window, 'y': y_window})
      joint_entropy = self._calculate_shannon_entropy(
        xy_combined.apply(lambda row: f"{row['x']:.3f}_{row['y']:.3f}", axis=1),
        bins=bins
      )

      # Энтропия Y
      y_entropy = self._calculate_shannon_entropy(y_window, bins=bins)

      return joint_entropy - y_entropy

    return pd.Series([
      cond_entropy(x.iloc[max(0, i - window):i], y.iloc[max(0, i - window):i])
      for i in range(1, len(x) + 1)
    ], index=x.index)

  def _calculate_transfer_entropy(self, x: pd.Series, y: pd.Series,
                                  window: int, lag: int = 1) -> pd.Series:
    """Упрощенная transfer entropy"""

    def te(x_window, y_window):
      if len(x_window) < window:
        return np.nan

      # TE(X->Y) ≈ I(Y_t; X_{t-lag} | Y_{t-lag})
      y_future = y_window.iloc[lag:]
      x_past = x_window.iloc[:-lag]
      y_past = y_window.iloc[:-lag]

      if len(y_future) < 10:
        return np.nan

      # Используем корреляцию как прокси для TE
      corr_xy = np.corrcoef(y_future, x_past)[0, 1]
      corr_yy = np.corrcoef(y_future, y_past)[0, 1]

      return abs(corr_xy) - abs(corr_yy)

    return pd.Series([
      te(x.iloc[max(0, i - window):i], y.iloc[max(0, i - window):i])
      for i in range(1, len(x) + 1)
    ], index=x.index)

  @staticmethod
  def _estimate_kolmogorov_complexity(series: pd.Series) -> float:
    """Оценка сложности Колмогорова через сжатие"""
    if len(series) < 10:
      return np.nan

    # Преобразуем в строку с фиксированной точностью
    string_repr = ''.join([f'{x:.5f}' for x in series.dropna()])

    # Оценка через длину строки (упрощенно)
    # В реальности можно использовать zlib.compress
    unique_chars = len(set(string_repr))
    total_chars = len(string_repr)

    # Нормализованная сложность
    return unique_chars / (total_chars + 1)

  def _rolling_beta(self, y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Скользящая бета"""

    def beta(y_window, x_window):
      if len(y_window) < 10:
        return np.nan

      # Cov(Y,X) / Var(X)
      cov = np.cov(y_window, x_window)[0, 1]
      var = np.var(x_window)

      return cov / (var + 1e-9)

    return pd.Series([
      beta(y.iloc[max(0, i - window):i], x.iloc[max(0, i - window):i])
      for i in range(1, len(y) + 1)
    ], index=y.index)

  def _rolling_partial_autocorr(self, series: pd.Series, lag: int, window: int) -> pd.Series:
    """Скользящая частичная автокорреляция"""

    def pacf(window_data):
      if len(window_data) < lag + 10:
        return np.nan

      # Упрощенный PACF через регрессию
      y = window_data.iloc[lag:]
      X = pd.DataFrame({
        f'lag_{i}': window_data.shift(i).iloc[lag:]
        for i in range(1, lag + 1)
      })

      if X.isnull().any().any():
        return np.nan

      # Коэффициент при последнем лаге в множественной регрессии
      try:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[-1]
      except:
        return np.nan

    return series.rolling(window).apply(pacf)

  def _estimate_long_memory(self, returns: pd.Series, window: int) -> pd.Series:
    """Оценка параметра долгой памяти d"""

    def estimate_d(window_data):
      if len(window_data) < 50:
        return np.nan

      # R/S анализ для оценки H, затем d = H - 0.5
      cumsum = np.cumsum(window_data - window_data.mean())
      R = np.max(cumsum) - np.min(cumsum)
      S = np.std(window_data)

      if S == 0:
        return 0

      # Упрощенная оценка
      n = len(window_data)
      H = np.log(R / S) / np.log(n)
      d = H - 0.5

      return np.clip(d, -0.5, 0.5)

    return returns.rolling(window).apply(estimate_d)

  def _calculate_mean_reversion_speed(self, prices: pd.Series, window: int) -> pd.Series:
    """Скорость возврата к среднему"""

    def ou_speed(window_data):
      if len(window_data) < 20:
        return np.nan

      # Оцениваем через AR(1) модель
      returns = np.diff(np.log(window_data))
      if len(returns) < 2:
        return np.nan

      # Автокорреляция
      autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]

      # Скорость mean reversion ≈ -log(autocorr)
      if autocorr > 0 and autocorr < 1:
        return -np.log(autocorr)
      else:
        return 0

    return prices.rolling(window).apply(ou_speed)

  def _calculate_shock_persistence(self, returns: pd.Series, window: int) -> pd.Series:
    """Персистентность шоков"""

    def persistence(window_data):
      if len(window_data) < 20:
        return np.nan

      # Импульсная функция отклика
      shock_idx = np.argmax(np.abs(window_data))
      if shock_idx >= len(window_data) - 5:
        return np.nan

      # Измеряем затухание после шока
      post_shock = window_data.iloc[shock_idx + 1:shock_idx + 6]
      decay_rate = np.mean(np.abs(post_shock)) / np.abs(window_data.iloc[shock_idx])

      return decay_rate

    return returns.rolling(window).apply(persistence)

  def _calculate_feature_statistics(self, features: pd.DataFrame):
    """Вычисляет статистику признаков для мониторинга"""
    self.feature_stats = {
      'mean': features.mean().to_dict(),
      'std': features.std().to_dict(),
      'null_percentage': (features.isnull().sum() / len(features) * 100).to_dict(),
      'correlation_matrix': features.corr().to_dict()
    }

  def _rolling_entropy(self, series: pd.Series, window: int = 30) -> pd.Series:
    """
    Вычисляет скользящую энтропию серии с защитой от ошибок
    """

    def entropy(values):
      try:
        # Удаляем NaN и бесконечные значения
        clean_values = values.dropna()
        clean_values = clean_values[np.isfinite(clean_values)]

        if len(clean_values) < 5:
          return 0.0

        # Дискретизация
        bins = min(10, len(clean_values) // 3)
        if bins < 2:
          bins = 2

        hist, _ = np.histogram(clean_values, bins=bins)
        hist = hist[hist > 0]  # Убираем нулевые значения

        if len(hist) == 0:
          return 0.0

        # Нормализация
        prob = hist / hist.sum()

        # Расчет энтропии
        return -np.sum(prob * np.log2(prob))

      except Exception:
        return 0.0

    # Применяем скользящим окном
    return series.rolling(window=window, min_periods=5).apply(entropy, raw=False).fillna(0)


class EnhancedEnsembleModel:
  """Улучшенная ансамблевая модель с мета-обучением"""

  def __init__(self, anomaly_detector: Optional[MarketAnomalyDetector] = None):
    self.anomaly_detector = anomaly_detector
    self.feature_engineer = AdvancedFeatureEngineer()

    # Базовые модели
    self.models = {
      'rf': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        class_weight='balanced',  # ДОБАВЛЕНО: автоматическая балансировка

        random_state=42,
        n_jobs=-1
      ),
      'gb': HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42
      ),
      'xgb': xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        # scale_pos_weight=2,
        random_state=42,
        # use_label_encoder=False,
        eval_metric='mlogloss'
      ),
      'lgb': lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        verbose=-1
      )
    }

    # Мета-модель для стекинга
    self.meta_model = xgb.XGBClassifier(
      n_estimators=50,
      learning_rate=0.1,
      max_depth=3,
      random_state=42,
      use_label_encoder=False,
      eval_metric='logloss'
    )

    # Скейлеры для разных групп признаков
    self.scalers = {
      'standard': StandardScaler(),
      'robust': RobustScaler()
    }

    # Селектор признаков
    self.feature_selector = None
    self.selected_features = None

    # Параметры для адаптивного обучения
    self.performance_history = []
    self.feature_importance_history = []
    self.is_fitted = False

  # def fit(self, X: pd.DataFrame, y: pd.Series,
  #         external_data: Optional[Dict[str, pd.DataFrame]] = None,
  #         optimize_features: bool = True):
  #   """
  #   Обучение ансамбля с оптимизацией признаков и безопасной обработкой индексов
  #   """
  #   logger.info("Начало обучения Enhanced Ensemble Model...")
  #
  #   try:
  #     # 1. Создание продвинутых признаков
  #     logger.info("Создание продвинутых признаков...")
  #     X_enhanced = self.feature_engineer.create_advanced_features(X, external_data)
  #
  #     # 2. Проверка и выравнивание индексов
  #     logger.debug("Проверка соответствия индексов...")
  #
  #     # Находим общие индексы между признаками и метками
  #     logger.info("=== ОТЛАДКА ИНДЕКСОВ ===")
  #     logger.info(f"X_enhanced индекс: {X_enhanced.index[:5]}")
  #     logger.info(f"y индекс: {y.index[:5]}")
  #     logger.info(f"Размеры: X_enhanced={len(X_enhanced)}, y={len(y)}")
  #
  #     # ПРИНУДИТЕЛЬНОЕ ВЫРАВНИВАНИЕ ИНДЕКСОВ
  #     if len(X_enhanced) == len(y):
  #       logger.info("Размеры совпадают, принудительно синхронизируем индексы...")
  #       # Создаем новый числовой индекс
  #       new_index = range(len(X_enhanced))
  #       X_enhanced.index = new_index
  #       y.index = new_index
  #       logger.info("✅ Индексы принудительно синхронизированы")
  #
  #       # Используем синхронизированные данные
  #       X_aligned = X_enhanced
  #       y_aligned = y
  #
  #     else:
  #       logger.warning(f"Размеры не совпадают: X_enhanced={len(X_enhanced)}, y={len(y)}")
  #       # Берем минимальный размер
  #       min_size = min(len(X_enhanced), len(y))
  #       logger.info(f"Обрезаем до минимального размера: {min_size}")
  #
  #       X_aligned = X_enhanced.iloc[:min_size].copy()
  #       y_aligned = y.iloc[:min_size].copy()
  #
  #       # Устанавливаем одинаковые индексы
  #       new_index = range(min_size)
  #       X_aligned.index = new_index
  #       y_aligned.index = new_index
  #
  #     logger.info(f"✅ Финальные размеры: X_aligned={X_aligned.shape}, y_aligned={y_aligned.shape}")
  #     logger.info("=== КОНЕЦ ОТЛАДКИ ИНДЕКСОВ ===")
  #
  #     # ПРОДОЛЖАЕМ с X_aligned и y_aligned вместо common_index
  #     if len(X_aligned) == 0:
  #       raise ValueError("Нет данных после выравнивания")
  #
  #     if len(X_aligned) < 100:
  #       logger.warning(f"Мало общих данных для обучения: {len(X_aligned)} образцов")
  #
  #     # Используем только общие индексы
  #     X_aligned = X_aligned
  #     y_aligned = y_aligned
  #
  #     logger.info(f"Размер выровненных данных: X={X_aligned.shape}, y={y_aligned.shape}")
  #
  #     # 3. Очистка данных от NaN и бесконечных значений
  #     logger.debug("Очистка данных...")
  #
  #     # Заменяем бесконечные значения на NaN
  #     X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan)
  #
  #     # Удаляем строки с NaN в целевой переменной
  #     valid_y_mask = ~pd.isna(y_aligned)
  #     X_clean = X_aligned[valid_y_mask]
  #     y_clean = y_aligned[valid_y_mask]
  #
  #     # Заполняем NaN в признаках медианными значениями
  #     if X_clean.isnull().any().any():
  #       logger.debug("Заполнение пропущенных значений в признаках...")
  #       X_clean = X_clean.fillna(X_clean.median()).fillna(0)
  #
  #     # Финальная проверка на NaN
  #     if X_clean.isnull().any().any() or pd.isna(y_clean).any():
  #       logger.warning("Обнаружены оставшиеся NaN значения, выполняем финальную очистку...")
  #       # Удаляем строки с любыми NaN
  #       final_mask = ~(X_clean.isnull().any(axis=1) | pd.isna(y_clean))
  #       X_clean = X_clean[final_mask]
  #       y_clean = y_clean[final_mask]
  #
  #     logger.info(f"Размер очищенных данных: X={X_clean.shape}, y={y_clean.shape}")
  #
  #     if len(X_clean) < 50:
  #       raise ValueError(f"Недостаточно данных для обучения после очистки: {len(X_clean)} образцов")
  #
  #     # 4. Проверка на аномалии в обучающих данных
  #     if self.anomaly_detector:
  #       logger.info("Проверка на аномалии в обучающих данных...")
  #       try:
  #         # Создаем временный DataFrame для проверки аномалий
  #         temp_data = X.copy()
  #         if 'close' not in temp_data.columns and len(temp_data.columns) > 0:
  #           # Если нет колонки close, используем первую доступную колонку как цену
  #           temp_data['close'] = temp_data.iloc[:, 0]
  #
  #         # Вызываем detect_anomalies с символом
  #         anomalies = self.anomaly_detector.detect_anomalies(temp_data, symbol="TRAINING_DATA")
  #
  #         if anomalies:
  #           logger.warning(f"Обнаружено {len(anomalies)} аномалий в обучающих данных")
  #           # Можем использовать эту информацию для корректировки весов образцов
  #       except Exception as anomaly_error:
  #         logger.warning(f"Ошибка при проверке аномалий: {anomaly_error}")
  #
  #     # 5. Оптимизация признаков с защитой базовых колонок
  #     if optimize_features and len(X_clean.columns) > 10:
  #       logger.info("Оптимизация признаков...")
  #       try:
  #         # ДОБАВИТЬ В НАЧАЛО БЛОКА ОПТИМИЗАЦИИ:
  #
  #         # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Защищаем базовые OHLCV колонки
  #         base_columns = ['open', 'high', 'low', 'close', 'volume']
  #         protected_columns = [col for col in base_columns if col in X_clean.columns]
  #
  #         if len(protected_columns) == 0:
  #           logger.warning("❌ КРИТИЧЕСКАЯ ПРОБЛЕМА: Базовые OHLCV колонки отсутствуют в данных!")
  #           logger.warning(f"Доступные колонки: {list(X_clean.columns[:10])}")
  #           logger.warning("Модель будет работать только с производными признаками")
  #         else:
  #           logger.info(f"✅ Защищенные базовые колонки: {protected_columns}")
  #
  #         # Оптимизируем только дополнительные признаки
  #         feature_columns = [col for col in X_clean.columns if col not in protected_columns]
  #
  #         logger.info(f"Колонки для оптимизации: {len(feature_columns)} из {len(X_clean.columns)}")
  #
  #         if len(feature_columns) > 5:
  #           logger.debug(f"Оптимизация {len(feature_columns)} дополнительных признаков...")
  #
  #           # Удаляем признаки с нулевой дисперсией (только среди дополнительных)
  #           if len(feature_columns) > 0:
  #             feature_data = X_clean[feature_columns]
  #             low_variance_cols = feature_data.columns[feature_data.var() < 1e-8]
  #             if len(low_variance_cols) > 0:
  #               logger.debug(f"Удаление {len(low_variance_cols)} признаков с низкой дисперсией")
  #               feature_columns = [col for col in feature_columns if col not in low_variance_cols]
  #
  #           # Удаляем сильно коррелированные признаки (только среди дополнительных)
  #           if len(feature_columns) > 1:
  #             feature_data = X_clean[feature_columns]
  #             correlation_matrix = feature_data.corr().abs()
  #             upper_tri = correlation_matrix.where(
  #               np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
  #             )
  #
  #             high_corr_cols = [column for column in upper_tri.columns
  #                               if any(upper_tri[column] > 0.95)]
  #             if len(high_corr_cols) > 0:
  #               logger.debug(f"Удаление {len(high_corr_cols)} сильно коррелированных признаков")
  #               feature_columns = [col for col in feature_columns if col not in high_corr_cols]
  #
  #           # Отбор лучших признаков через важность (только дополнительные)
  #           if len(feature_columns) > 20:
  #             try:
  #               feature_data = X_clean[feature_columns]
  #               temp_rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
  #               temp_rf.fit(feature_data, y_clean)
  #
  #               importances = pd.Series(temp_rf.feature_importances_, index=feature_columns)
  #               # Берем топ-15 лучших дополнительных признаков
  #               best_features = importances.nlargest(15).index.tolist()
  #               feature_columns = best_features
  #               logger.debug(f"Отобрано {len(feature_columns)} лучших дополнительных признаков")
  #
  #             except Exception as feat_sel_error:
  #               logger.warning(f"Ошибка отбора признаков по важности: {feat_sel_error}")
  #
  #         # КРИТИЧЕСКИ ВАЖНО: Объединяем защищенные и оптимизированные колонки
  #         final_columns = protected_columns + feature_columns
  #         X_clean = X_clean[final_columns]
  #
  #         logger.info(
  #           f"✅ После оптимизации: {len(protected_columns)} базовых + {len(feature_columns)} дополнительных = {len(final_columns)} признаков")
  #
  #       except Exception as opt_error:
  #         logger.warning(f"Ошибка при оптимизации признаков: {opt_error}")
  #         # При ошибке сохраняем хотя бы базовые колонки если они есть
  #         base_columns = ['open', 'high', 'low', 'close', 'volume']
  #         available_base = [col for col in base_columns if col in X_clean.columns]
  #         if available_base:
  #           logger.warning(f"Fallback: используем только базовые колонки: {available_base}")
  #           X_clean = X_clean[available_base]
  #         else:
  #           logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА: Нет базовых OHLCV колонок!")
  #
  #     # 6. Скалирование признаков
  #     logger.debug("Скалирование признаков...")
  #     try:
  #       # Используем стандартное скалирование
  #       X_scaled = self.scalers['standard'].fit_transform(X_clean)
  #       X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)
  #
  #       # Проверяем результат скалирования
  #       if np.any(~np.isfinite(X_scaled.values)):
  #         logger.warning("Обнаружены нефинитные значения после скалирования, применяем робастное скалирование")
  #         X_scaled = self.scalers['robust'].fit_transform(X_clean)
  #         X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)
  #         X_scaled = X_scaled.fillna(0)  # На всякий случай
  #
  #     except Exception as scale_error:
  #       logger.error(f"Ошибка скалирования: {scale_error}")
  #       # Используем данные без скалирования
  #       X_scaled = X_clean
  #
  #     # 7. Разделение на обучение и валидацию
  #     logger.debug("Разделение данных...")
  #     from sklearn.model_selection import train_test_split
  #
  #     try:
  #       X_train, X_val, y_train, y_val = train_test_split(
  #         X_scaled, y_clean,
  #         test_size=0.2,
  #         random_state=42,
  #         stratify=y_clean if len(np.unique(y_clean)) > 1 else None
  #       )
  #     except Exception as split_error:
  #       logger.warning(f"Ошибка стратифицированного разделения: {split_error}")
  #       # Используем простое разделение
  #       X_train, X_val, y_train, y_val = train_test_split(
  #         X_scaled, y_clean, test_size=0.2, random_state=42
  #       )
  #
  #     # 8. Обучение базовых моделей
  #     logger.info("Обучение базовых моделей...")
  #
  #     for name, model in self.models.items():
  #       try:
  #         logger.debug(f"Обучение модели {name}...")
  #         model.fit(X_train, y_train)
  #
  #         # Проверяем качество на валидации
  #         val_score = model.score(X_val, y_val)
  #         logger.debug(f"Валидационная точность {name}: {val_score:.4f}")
  #
  #       except Exception as model_error:
  #         logger.error(f"Ошибка при обучении модели {name}: {model_error}")
  #         continue
  #
  #     # 9. Обучение мета-модели (стекинг)
  #     logger.info("Обучение мета-модели...")
  #     try:
  #       # Получаем предсказания базовых моделей для мета-обучения
  #       meta_features = []
  #
  #       for name, model in self.models.items():
  #         try:
  #           if hasattr(model, 'predict_proba'):
  #             preds = model.predict_proba(X_train)
  #             if preds.shape[1] > 1:
  #               meta_features.append(preds[:, 1])  # Вероятность положительного класса
  #             else:
  #               meta_features.append(preds[:, 0])
  #           else:
  #             preds = model.predict(X_train)
  #             meta_features.append(preds)
  #         except Exception as pred_error:
  #           logger.warning(f"Ошибка получения предсказаний от {name}: {pred_error}")
  #           continue
  #
  #       if len(meta_features) > 0:
  #         meta_X = np.column_stack(meta_features)
  #         self.meta_model.fit(meta_X, y_train)
  #         logger.info("Мета-модель успешно обучена")
  #       else:
  #         logger.warning("Не удалось получить предсказания для мета-модели")
  #
  #     except Exception as meta_error:
  #       logger.error(f"Ошибка при обучении мета-модели: {meta_error}")
  #
  #
  #
  #     # 10. Сохранение информации о признаках
  #     self.selected_features = list(X_scaled.columns)
  #     self.is_fitted = True
  #
  #     logger.info(f"Обучение завершено успешно. Использовано {len(self.selected_features)} признаков")
  #
  #     try:
  #       from sklearn.metrics import classification_report
  #
  #       # Проверяем, что у нас есть валидационные данные
  #       if len(X_val) > 0 and len(y_val) > 0:
  #         # Получаем финальные предсказания
  #         final_predictions = self.predict(X_val.copy())  # Создаем копию для безопасности
  #
  #         if len(final_predictions) == len(y_val):
  #           report = classification_report(y_val, final_predictions, output_dict=True, zero_division=0)
  #
  #           logger.info(f"Итоговая точность: {report['accuracy']:.4f}")
  #           logger.info(f"F1-score: {report['macro avg']['f1-score']:.4f}")
  #         else:
  #           logger.warning("Размеры предсказаний и меток не совпадают")
  #       else:
  #         logger.warning("Нет валидационных данных для итоговой оценки")
  #
  #     except Exception as eval_error:
  #       logger.warning(f"Ошибка при итоговой оценке: {eval_error}")
  #
  #   except Exception as e:
  #     logger.error(f"Ошибка при обучении модели: {e}")
  #     self.is_fitted = False
  #     raise

  # В методе predict в файле ml/enhanced_ml_system.py, в блоке try замените начало на:

  def fit(self, X: pd.DataFrame, y: pd.Series,
          external_data: Optional[Dict[str, pd.DataFrame]] = None,
          optimize_features: bool = True):
    """
    Обучение ансамбля с оптимизацией признаков, борьбой с дисбалансом (SMOTE)
    и специальной обработкой для XGBoost.
    """
    logger.info("Начало обучения Enhanced Ensemble Model с SMOTE...")

    try:
      # Шаг 1: Создание и выравнивание признаков
      logger.info("Создание продвинутых признаков...")
      X_enhanced = self.feature_engineer.create_advanced_features(X, external_data)

      logger.info("Выравнивание и очистка данных...")
      common_index = X_enhanced.index.intersection(y.index)
      X_aligned = X_enhanced.loc[common_index]
      y_aligned = y.loc[common_index]

      X_clean = X_aligned.replace([np.inf, -np.inf], np.nan)
      X_clean = X_clean.fillna(X_clean.median()).fillna(0)

      # Шаг 2: Оптимизация признаков
      if optimize_features and len(X_clean.columns) > 10:
        # ... (Ваша логика оптимизации признаков остается здесь без изменений) ...
        logger.info("Оптимизация признаков...")
        base_columns = ['open', 'high', 'low', 'close', 'volume']
        protected_columns = [col for col in base_columns if col in X_clean.columns]
        feature_columns = [col for col in X_clean.columns if col not in protected_columns]
        if len(feature_columns) > 1:
          feature_data = X_clean[feature_columns]
          correlation_matrix = feature_data.corr().abs()
          upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
          high_corr_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]
          if len(high_corr_cols) > 0:
            feature_columns = [col for col in feature_columns if col not in high_corr_cols]
        if len(feature_columns) > 20:
          temp_rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
          temp_rf.fit(X_clean[feature_columns], y_aligned)
          importances = pd.Series(temp_rf.feature_importances_, index=feature_columns)
          best_features = importances.nlargest(15).index.tolist()
          feature_columns = best_features
        final_columns = protected_columns + feature_columns
        X_clean = X_clean[final_columns]
        logger.info(f"После оптимизации осталось {len(final_columns)} признаков.")

      self.selected_features = list(X_clean.columns)

      # Шаг 3: Скалирование
      logger.debug("Скалирование признаков...")
      X_scaled = self.scalers['standard'].fit_transform(X_clean)
      X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)

      # Шаг 4: Разделение на обучение и валидацию
      from sklearn.model_selection import train_test_split
      X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_aligned, test_size=0.2, random_state=42, stratify=y_aligned
      )

      # Шаг 5: Применение SMOTE
      logger.info("Применение SMOTE для балансировки классов...")
      smote = SMOTE(random_state=42)
      X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

      # ======================= ИЗМЕНЕНИЕ 1: ЛОГИКА ДЛЯ XGBOOST =======================
      # Шаг 6: Обучение базовых моделей с особой обработкой для XGBoost
      logger.info("Обучение базовых моделей...")
      for name, model in self.models.items():
        logger.debug(f"Обучение модели {name}...")

        if name == 'xgb':
          logger.info("Расчет sample_weight для XGBoost...")
          sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_resampled)
          model.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights, verbose=False)
        else:
          model.fit(X_train_resampled, y_train_resampled)

      # ==============================================================================

      # Шаг 7: Обучение мета-модели
      logger.info("Обучение мета-модели...")
      meta_features = []
      for name, model in self.models.items():
        preds = model.predict_proba(X_train_resampled)[:, 1] if hasattr(model, 'predict_proba') else model.predict(
          X_train_resampled)
        meta_features.append(preds)

      meta_X = np.column_stack(meta_features)
      self.meta_model.fit(meta_X, y_train_resampled)
      self.is_fitted = True
      logger.info(f"Обучение завершено успешно. Использовано {len(self.selected_features)} признаков.")

      # ======================= ИЗМЕНЕНИЕ 2: ВЫВОД ФИНАЛЬНЫХ МЕТРИК =======================
      # Шаг 8: Оценка производительности на валидационных данных
      logger.info("=" * 20 + " ОЦЕНКА МОДЕЛИ НА ВАЛИДАЦИОННЫХ ДАННЫХ " + "=" * 20)

      # Получаем предсказания мета-модели для валидационного набора
      meta_features_val = []
      for name, model in self.models.items():
        preds_val = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
        meta_features_val.append(preds_val)

      meta_X_val = np.column_stack(meta_features_val)
      final_predictions = self.meta_model.predict(meta_X_val)

      # Рассчитываем и выводим метрики
      accuracy = accuracy_score(y_val, final_predictions)
      f1 = f1_score(y_val, final_predictions, average='macro')  # 'macro' лучше для дисбаланса

      logger.info(f"Итоговая точность (Accuracy) на валидации: {accuracy:.4f}")
      logger.info(f"Итоговый F1-score (Macro) на валидации: {f1:.4f}")

      # Печатаем подробный отчет (в консоль для лучшей читаемости)
      print("\n--- Подробный отчет по классам (Classification Report) ---")
      # Предполагая, что 0=SELL, 1=HOLD, 2=BUY. Измените, если у вас другая кодировка.
      target_names = ['SELL', 'HOLD', 'BUY']
      print(classification_report(y_val, final_predictions, target_names=target_names))
      print("----------------------------------------------------------\n")
      # ===================================================================================

    except Exception as e:
      logger.error(f"Критическая ошибка при обучении модели: {e}", exc_info=True)
      self.is_fitted = False
      raise

  def fit_with_hyperparameter_tuning(self, X_train_data: pd.DataFrame, y_train_data: pd.Series,
                                     external_data: Optional[Dict[str, pd.DataFrame]] = None):
    """
    Выполняет полный цикл обучения с подбором гиперпараметров для ключевых моделей.
    """
    logger.info("=" * 20 + " ЗАПУСК ОБУЧЕНИЯ С ПОДБОРОМ ГИПЕРПАРАМЕТРОВ " + "=" * 20)

    # 1. Подготавливаем данные с помощью нового общего метода
    # Передаем X_train_data и y_train_data в правильные аргументы X и y
    X_train_res, y_train_res, X_val, y_val = self._prepare_data_for_training(
      X=X_train_data, y=y_train_data, external_data=external_data, optimize_features=True
    )
    logger.info("Данные для подбора гиперпараметров подготовлены.")

    # 2. Определение сеток параметров для подбора (без изменений)
    param_grids = {
      'lgb': {
        'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 40, 50], 'max_depth': [5, 7, 10]
      },
      'xgb': {
        'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8], 'min_child_weight': [1, 5, 10]
      }
    }

    # 3. Запуск RandomizedSearchCV для выбранных моделей (без изменений)
    tscv = TimeSeriesSplit(n_splits=5)
    models_to_tune = ['lgb', 'xgb']

    for name in models_to_tune:
      if name in self.models:
        logger.info(f"--- Начинается подбор параметров для модели '{name}' ---")
        random_search = RandomizedSearchCV(
          estimator=self.models[name], param_distributions=param_grids[name], n_iter=10,  # Уменьшено для скорости
          cv=tscv, verbose=1, random_state=42, n_jobs=-1, scoring='f1_macro'
        )
        random_search.fit(X_train_res, y_train_res)
        logger.info(f"Лучшие параметры для '{name}': {random_search.best_params_}")
        self.models[name] = random_search.best_estimator_

    # 4. Переобучение остальных моделей и мета-модели
    logger.info("Финальное обучение ансамбля на лучших параметрах...")
    for name, model in self.models.items():
      if name not in models_to_tune:
        model.fit(X_train_res, y_train_res)

    meta_features = [m.predict_proba(X_train_res)[:, 1] for m in self.models.values() if hasattr(m, 'predict_proba')]
    self.meta_model.fit(np.column_stack(meta_features), y_train_res)

    self.is_fitted = True
    logger.info("=" * 20 + " ОБУЧЕНИЕ С ПОДБОРОМ ГИПЕРПАРАМЕТРОВ ЗАВЕРШЕНО " + "=" * 20)

  def _prepare_data_for_training(self, X: pd.DataFrame, y: pd.Series,
                                 external_data: Optional[Dict[str, pd.DataFrame]] = None,
                                 optimize_features: bool = True):
    """
    Приватный метод для полной подготовки данных: создание признаков,
    очистка, оптимизация и балансировка с помощью SMOTE.
    """
    logger.info("Шаг 1: Создание продвинутых признаков...")
    X_enhanced = self.feature_engineer.create_advanced_features(X, external_data)

    logger.info("Шаг 2: Выравнивание и очистка данных...")
    common_index = X_enhanced.index.intersection(y.index)
    X_aligned = X_enhanced.loc[common_index]
    y_aligned = y.loc[common_index]

    X_clean = X_aligned.replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.fillna(X_clean.median()).fillna(0)

    if len(X_clean) < 100:
      logger.warning(f"Мало общих данных для обучения: {len(X_clean)} образцов")

    if optimize_features and len(X_clean.columns) > 10:
      logger.info("Шаг 3: Оптимизация признаков...")
      try:
        base_columns = ['open', 'high', 'low', 'close', 'volume']
        protected_columns = [col for col in base_columns if col in X_clean.columns]
        feature_columns = [col for col in X_clean.columns if col not in protected_columns]

        if len(feature_columns) > 1:
          correlation_matrix = X_clean[feature_columns].corr().abs()
          upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
          high_corr_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]
          if high_corr_cols:
            logger.info(f"Удаление {len(high_corr_cols)} коррелированных признаков (порог > 0.90)")
            feature_columns = [col for col in feature_columns if col not in high_corr_cols]

        if len(feature_columns) > 20:
          temp_rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
          temp_rf.fit(X_clean[feature_columns], y_aligned)
          importances = pd.Series(temp_rf.feature_importances_, index=feature_columns)
          feature_columns = importances.nlargest(15).index.tolist()

        final_columns = protected_columns + feature_columns
        X_clean = X_clean[final_columns]
        logger.info(f"После оптимизации осталось {len(final_columns)} признаков.")
      except Exception as opt_error:
        logger.warning(f"Ошибка при оптимизации признаков: {opt_error}")

    self.selected_features = list(X_clean.columns)

    logger.info("Шаг 4: Скалирование данных...")
    X_scaled = self.scalers['standard'].fit_transform(X_clean)
    X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)

    logger.info("Шаг 5: Разделение и балансировка (SMOTE)...")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
      X_scaled, y_aligned, test_size=0.2, random_state=42, stratify=y_aligned
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"Размер обучающей выборки после SMOTE: {X_train_resampled.shape}")

    return X_train_resampled, y_train_resampled, X_val, y_val

  def predict(self, X: pd.DataFrame, external_data: Optional[Dict[str, pd.DataFrame]] = None) -> np.ndarray:
    """
    Получает предсказания от ансамбля моделей с проверкой базовых колонок
    """
    if not self.is_fitted:
      raise ValueError("Модель не обучена. Вызовите fit() перед предсказанием.")

    try:
      if hasattr(self, 'selected_features') and self.selected_features:
        base_columns = ['open', 'high', 'low', 'close', 'volume']
        has_base_features = any(col in self.selected_features for col in base_columns)

        if not has_base_features:
          logger.warning("⚠️ Модель обучена без базовых OHLCV колонок, используем только производные признаки")

          # Проверяем доступность необходимых производных признаков
          if len(X.columns) < 5:
            logger.error("Недостаточно данных для создания производных признаков")
            return np.ones(len(X), dtype=int)

        else:
          # Стандартная проверка базовых колонок
          required_cols = [col for col in base_columns if col in self.selected_features]
          missing_cols = [col for col in required_cols if col not in X.columns]

          if missing_cols:
            logger.error(f"Отсутствуют необходимые колонки: {missing_cols}")
            return np.ones(len(X), dtype=int)

      # КРИТИЧЕСКАЯ ПРОВЕРКА: Убеждаемся, что есть необходимые колонки
      required_cols = ['open', 'high', 'low', 'close', 'volume']
      missing_cols = [col for col in required_cols if col not in X.columns]

      if missing_cols:
        logger.error(f"Отсутствуют необходимые колонки: {missing_cols}")
        logger.error(f"Доступные колонки: {list(X.columns)}")
        # Возвращаем нейтральные предсказания (класс 1 = HOLD)
        return np.ones(len(X), dtype=int)

      # Проверяем входные данные
      if X.empty:
        logger.warning("Пустые входные данные для предсказания")
        return np.ones(0, dtype=int)

      # 1. Создание признаков
      logger.debug("Создание продвинутых признаков...")
      X_enhanced = self.feature_engineer.create_advanced_features(X, external_data)

      if X_enhanced.empty:
        logger.warning("Не удалось создать признаки для предсказания")
        return np.ones(len(X), dtype=int)

      # 2. Выбор признаков (с проверкой доступности)
      if self.selected_features:
        available_features = [f for f in self.selected_features if f in X_enhanced.columns]

        if len(available_features) < len(self.selected_features) // 2:
          logger.warning(f"Доступно только {len(available_features)} из {len(self.selected_features)} признаков")

        if not available_features:
          logger.error("Нет доступных признаков для предсказания")
          # Используем базовые колонки как fallback
          base_cols = ['open', 'high', 'low', 'close', 'volume']
          fallback_features = [col for col in base_cols if col in X.columns]
          if fallback_features:
            logger.info(f"Используем базовые колонки как fallback: {fallback_features}")
            X_enhanced = X[fallback_features]
          else:
            return np.ones(len(X), dtype=int)
        else:
          X_enhanced = X_enhanced[available_features]
      else:
        logger.warning("Нет информации о выбранных признаках, используем все доступные")

      # 3. Очистка данных
      if X_enhanced.isnull().any().any():
        logger.debug("Обнаружены NaN в признаках, заполняем медианными значениями")
        X_enhanced = X_enhanced.fillna(X_enhanced.median()).fillna(0)

      X_enhanced = X_enhanced.replace([np.inf, -np.inf], 0)

      # 4. Скалирование
      try:
        X_scaled = self.scalers['standard'].transform(X_enhanced)
        X_scaled = pd.DataFrame(X_scaled, columns=X_enhanced.columns, index=X_enhanced.index)
      except Exception as scale_error:
        logger.warning(f"Ошибка скалирования: {scale_error}, используем исходные данные")
        X_scaled = X_enhanced

      # 5. Получение предсказаний от базовых моделей
      predictions = []
      successful_models = []

      for name, model in self.models.items():
        try:
          pred = model.predict(X_scaled)
          predictions.append(pred)
          successful_models.append(name)
          logger.debug(f"Получены предсказания от модели {name}")
        except Exception as e:
          logger.warning(f"Ошибка предсказания от модели {name}: {e}")
          continue

      if not predictions:
        logger.warning("Не удалось получить предсказания ни от одной модели")
        return np.ones(len(X_scaled), dtype=int)

      logger.debug(f"Успешные модели: {successful_models}")

      # 6. Мета-предсказание
      if len(predictions) > 1 and hasattr(self.meta_model, 'predict'):
        try:
          meta_features = np.column_stack(predictions)
          final_prediction = self.meta_model.predict(meta_features)
          logger.debug("Использованы предсказания мета-модели")
          return final_prediction.astype(int)
        except Exception as e:
          logger.warning(f"Ошибка мета-модели: {e}")

      # 7. Простое усреднение как fallback
      ensemble_prediction = np.mean(predictions, axis=0)

      # Преобразуем в классы (0, 1, 2)
      result = np.round(ensemble_prediction).astype(int)
      # Ограничиваем значения диапазоном [0, 2]
      result = np.clip(result, 0, 2)

      logger.debug(f"Получены финальные предсказания: {len(result)} образцов")

      return result

    except Exception as e:
      logger.error(f"Критическая ошибка при предсказании: {e}")
      # Возвращаем нейтральные предсказания (HOLD)
      return np.ones(len(X), dtype=int)

  def predict_proba(self, X: pd.DataFrame, external_data: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[
    np.ndarray, MLPrediction]:
    """
    Получает вероятности классов и детальную информацию о предсказании
    """
    if not self.is_fitted:
      raise ValueError("Модель не обучена. Вызовите fit() перед предсказанием.")

    try:
      # Проверяем входные данные
      if X.empty:
        logger.warning("Пустые входные данные для предсказания")
        neutral_proba = np.full((0, 3), 1 / 3)
        ml_prediction = MLPrediction(
          signal_type=SignalType.HOLD,
          probability=1 / 3,
          confidence=0.3,
          model_agreement=0.0,
          feature_importance={},
          risk_assessment={'anomaly_detected': False},
          metadata={}
        )
        return neutral_proba, ml_prediction

      # Убеждаемся, что есть необходимые колонки
      required_cols = ['open', 'high', 'low', 'close', 'volume']
      missing_cols = [col for col in required_cols if col not in X.columns]

      if missing_cols:
        logger.error(f"Отсутствуют необходимые колонки: {missing_cols}")
        neutral_proba = np.full((len(X), 3), 1 / 3)
        ml_prediction = MLPrediction(
          signal_type=SignalType.HOLD,
          probability=1 / 3,
          confidence=0.1,
          model_agreement=0.0,
          feature_importance={},
          risk_assessment={'anomaly_detected': True, 'error': f'missing columns: {missing_cols}'},
          metadata={'error': 'missing_columns'}
        )
        return neutral_proba, ml_prediction

      # 1. Создание признаков
      X_enhanced = self.feature_engineer.create_advanced_features(X, external_data)

      if X_enhanced.empty:
        logger.warning("Не удалось создать признаки для предсказания")
        neutral_proba = np.full((len(X), 3), 1 / 3)
        ml_prediction = MLPrediction(
          signal_type=SignalType.HOLD,
          probability=1 / 3,
          confidence=0.1,
          model_agreement=0.0,
          feature_importance={},
          risk_assessment={'anomaly_detected': True, 'error': 'feature_creation_failed'},
          metadata={'error': 'feature_creation_failed'}
        )
        return neutral_proba, ml_prediction

      # 2. Выбор признаков
      if self.selected_features:
        available_features = [f for f in self.selected_features if f in X_enhanced.columns]
        if not available_features:
          logger.error("Нет доступных признаков для предсказания")
          neutral_proba = np.full((len(X), 3), 1 / 3)
          ml_prediction = MLPrediction(
            signal_type=SignalType.HOLD,
            probability=1 / 3,
            confidence=0.1,
            model_agreement=0.0,
            feature_importance={},
            risk_assessment={'anomaly_detected': True, 'error': 'no_features'},
            metadata={'error': 'no_features'}
          )
          return neutral_proba, ml_prediction
        X_enhanced = X_enhanced[available_features]

      # 3. КРИТИЧЕСКАЯ ОЧИСТКА ДАННЫХ ОТ NaN И INF
      logger.debug("Очистка данных от NaN и бесконечных значений...")

      # Заменяем бесконечные значения на NaN
      X_enhanced = X_enhanced.replace([np.inf, -np.inf], np.nan)

      # Проверяем процент NaN
      nan_percentage = X_enhanced.isnull().sum().sum() / (X_enhanced.shape[0] * X_enhanced.shape[1])
      if nan_percentage > 0.5:
        logger.warning(f"Слишком много NaN значений: {nan_percentage:.1%}")

      # Заполняем NaN медианными значениями
      X_enhanced_clean = X_enhanced.fillna(X_enhanced.median())

      # Если медиана тоже NaN, заполняем нулями
      X_enhanced_clean = X_enhanced_clean.fillna(0)

      # Финальная проверка на NaN
      if X_enhanced_clean.isnull().any().any():
        logger.error("Остались NaN значения после очистки")
        # Принудительно заполняем нулями
        X_enhanced_clean = X_enhanced_clean.fillna(0)

      # Проверка на бесконечные значения
      if not np.isfinite(X_enhanced_clean.values).all():
        logger.warning("Обнаружены бесконечные значения после очистки")
        X_enhanced_clean = X_enhanced_clean.replace([np.inf, -np.inf], 0)

      # 4. Скалирование
      try:
        X_scaled = self.scalers['standard'].transform(X_enhanced_clean)
        X_scaled = pd.DataFrame(X_scaled, columns=X_enhanced_clean.columns, index=X_enhanced_clean.index)

        # Проверяем результат скалирования
        if not np.isfinite(X_scaled.values).all():
          logger.warning("Скалирование дало нефинитные значения, используем робастное скалирование")
          X_scaled = self.scalers['robust'].transform(X_enhanced_clean)
          X_scaled = pd.DataFrame(X_scaled, columns=X_enhanced_clean.columns, index=X_enhanced_clean.index)
          X_scaled = X_scaled.fillna(0)

      except Exception as scale_error:
        logger.warning(f"Ошибка скалирования: {scale_error}, используем исходные данные")
        X_scaled = X_enhanced_clean

      # 5. Получение вероятностей от базовых моделей
      all_probabilities = []
      model_predictions = {}

      for name, model in self.models.items():
        try:
          # Финальная проверка данных перед подачей в модель
          if X_scaled.isnull().any().any():
            logger.error(f"NaN обнаружены перед моделью {name}")
            continue

          if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_scaled)
            all_probabilities.append(proba)
            model_predictions[name] = proba
            logger.debug(f"Получены вероятности от модели {name}")
          else:
            # Для моделей без predict_proba создаем псевдо-вероятности
            pred = model.predict(X_scaled)
            # Простое преобразование в вероятности
            proba = np.zeros((len(pred), 3))
            for i, p in enumerate(pred):
              class_idx = int(np.clip(p, 0, 2))  # Ограничиваем диапазон 0-2
              proba[i, class_idx] = 0.8  # Высокая уверенность в предсказанном классе
              # Распределяем остальную вероятность
              remaining = 0.2 / 2
              for j in range(3):
                if j != class_idx:
                  proba[i, j] = remaining
            all_probabilities.append(proba)
            model_predictions[name] = proba
            logger.debug(f"Созданы псевдо-вероятности для модели {name}")

        except Exception as e:
          logger.warning(f"Ошибка получения вероятностей от модели {name}: {e}")
          continue

      if not all_probabilities:
        # Fallback: нейтральные вероятности
        logger.warning("Не удалось получить предсказания ни от одной модели")
        neutral_proba = np.full((len(X_scaled), 3), 1 / 3)
        ml_prediction = MLPrediction(
          signal_type=SignalType.HOLD,
          probability=1 / 3,
          confidence=0.3,
          model_agreement=0.0,
          feature_importance={},
          risk_assessment={'anomaly_detected': True, 'error': 'all_models_failed'},
          metadata={'error': 'all_models_failed', 'available_models': list(self.models.keys())}
        )
        return neutral_proba, ml_prediction

      # 6. Усреднение вероятностей
      ensemble_proba = np.mean(all_probabilities, axis=0)

      # 7. Анализ согласованности моделей
      model_agreement = self._calculate_model_agreement(all_probabilities)

      # 8. Определение итогового сигнала
      predicted_class = np.argmax(ensemble_proba[-1])  # Берем последнее предсказание
      max_probability = np.max(ensemble_proba[-1])

      signal_type = SignalType.HOLD
      if predicted_class == 0:
        signal_type = SignalType.SELL
      elif predicted_class == 2:
        signal_type = SignalType.BUY

      # 9. Расчет важности признаков (упрощенный)
      feature_importance = {}
      if hasattr(self.models.get('rf'), 'feature_importances_'):
        try:
          importances = self.models['rf'].feature_importances_
          feature_names = X_enhanced_clean.columns
          feature_importance = dict(zip(feature_names, importances))
          # Берем топ-10
          feature_importance = dict(sorted(feature_importance.items(),
                                           key=lambda x: x[1], reverse=True)[:10])
        except Exception:
          pass

      # 10. Создание MLPrediction с обязательным metadata
      ml_prediction = MLPrediction(
        signal_type=signal_type,
        probability=float(max_probability),
        confidence=float(max_probability * model_agreement),
        model_agreement=float(model_agreement),
        feature_importance=feature_importance,
        risk_assessment={
          'anomaly_detected': False,
          'volatility_regime': 'normal',
          'market_stress': False
        },
        metadata={
          'models_used': list(model_predictions.keys()),
          'features_count': len(X_enhanced_clean.columns),
          'data_quality': 'good' if nan_percentage < 0.1 else 'poor',
          'ensemble_size': len(all_probabilities)
        }
      )

      return ensemble_proba, ml_prediction

    except Exception as e:
      logger.error(f"Ошибка при получении вероятностей: {e}")
      # Fallback
      neutral_proba = np.full((len(X), 3), 1 / 3)
      ml_prediction = MLPrediction(
        signal_type=SignalType.HOLD,
        probability=1 / 3,
        confidence=0.1,
        model_agreement=0.0,
        feature_importance={},
        risk_assessment={'anomaly_detected': True, 'error': str(e)},
        metadata={'error': str(e), 'fallback': True}
      )
      return neutral_proba, ml_prediction

  def fit_with_diagnostics(self, X: pd.DataFrame, y: pd.Series,
                           external_data: Optional[Dict[str, pd.DataFrame]] = None,
                           optimize_features: bool = True,
                           verbose: bool = True):
    """
    Обучение с диагностикой и подробным отчетом
    """
    logger.info("Запуск обучения с диагностикой...")

    # Предварительная диагностика
    if verbose:
      print("🔍 Предварительная диагностика данных...")
      diagnosis = self.diagnose_training_issues(X, y)

      if diagnosis['overall_status'] in ['ТРЕБУЕТ_ВНИМАНИЯ']:
        print("⚠️ Обнаружены проблемы с данными:")
        for issue in diagnosis.get('issues_found', []):
          print(f"   • {issue}")

        response = input("Продолжить обучение? (y/n): ")
        if response.lower() != 'y':
          print("Обучение отменено.")
          return

    # Основное обучение
    try:
      self.fit(X, y, external_data, optimize_features)

      if verbose:
        # Постобучающая диагностика
        print("\n✅ Обучение завершено!")
        self.print_training_report(X, y, diagnosis if verbose else None)

        # Проверка здоровья модели
        health = self.get_model_health_status()
        print(f"\n🏥 Здоровье модели: {health['overall_health']} ({health['health_percentage']:.1f}%)")

        if health.get('issues'):
          print("❗ Обнаруженные проблемы:")
          for issue in health['issues']:
            print(f"   • {issue}")

    except Exception as e:
      logger.error(f"Ошибка при обучении с диагностикой: {e}")
      if verbose:
        print(f"❌ Ошибка обучения: {e}")
      raise

  def _calculate_model_agreement(self, all_probabilities: List[np.ndarray]) -> float:
    """
    Вычисляет согласованность между моделями
    """
    try:
      if len(all_probabilities) < 2:
        return 1.0

      # Берем предсказания для последнего наблюдения
      last_predictions = [proba[-1] for proba in all_probabilities]

      # Вычисляем стандартное отклонение между предсказаниями моделей
      std_across_models = np.std(last_predictions, axis=0)
      avg_std = np.mean(std_across_models)

      # Преобразуем в меру согласованности (чем меньше разброс, тем выше согласованность)
      agreement = max(0.0, 1.0 - (avg_std * 3))  # Масштабируем

      return agreement

    except Exception:
      return 0.5  # Средняя согласованность по умолчанию

  def optimize_features(self, X: pd.DataFrame, y: pd.Series,
                        protect_base_columns: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Оптимизация набора признаков с защитой базовых OHLCV колонок

    Args:
        X: DataFrame с признаками
        y: Целевая переменная
        protect_base_columns: Защищать ли базовые OHLCV колонки от удаления

    Returns:
        Tuple[оптимизированные_признаки, список_выбранных_признаков]
    """
    logger.info("Запуск оптимизации признаков...")

    try:
      # Защищаем базовые колонки
      base_columns = []
      if protect_base_columns:
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        base_columns = [col for col in base_cols if col in X.columns]
        logger.info(f"Защищенные базовые колонки: {base_columns}")

      # Определяем колонки для оптимизации
      optimization_columns = [col for col in X.columns if col not in base_columns]
      logger.info(f"Колонки для оптимизации: {len(optimization_columns)}")

      if len(optimization_columns) == 0:
        logger.info("Нет колонок для оптимизации, возвращаем исходные данные")
        return X, list(X.columns)

      # Работаем с колонками для оптимизации
      X_opt = X[optimization_columns].copy()

      # 1. Удаление колонок с низкой дисперсией
      logger.debug("Удаление признаков с низкой дисперсией...")
      low_var_threshold = 1e-8
      low_variance_mask = X_opt.var() > low_var_threshold
      X_opt = X_opt.loc[:, low_variance_mask]
      removed_low_var = len(optimization_columns) - len(X_opt.columns)
      if removed_low_var > 0:
        logger.debug(f"Удалено {removed_low_var} признаков с низкой дисперсией")

      # 2. Удаление высококоррелированных признаков
      if len(X_opt.columns) > 1:
        logger.debug("Удаление высококоррелированных признаков...")
        corr_matrix = X_opt.corr().abs()

        # Создаем маску верхнего треугольника
        upper_tri = corr_matrix.where(
          np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Находим пары с корреляцией > 0.95
        high_corr_pairs = []
        for column in upper_tri.columns:
          for index in upper_tri.index:
            if pd.notna(upper_tri.loc[index, column]) and upper_tri.loc[index, column] > 0.95:
              high_corr_pairs.append((index, column, upper_tri.loc[index, column]))

        # Удаляем один признак из каждой пары
        to_drop_corr = set()
        for col1, col2, corr_val in high_corr_pairs:
          if col1 not in to_drop_corr and col2 not in to_drop_corr:
            # Удаляем признак с меньшей дисперсией
            if X_opt[col1].var() < X_opt[col2].var():
              to_drop_corr.add(col1)
            else:
              to_drop_corr.add(col2)

        if to_drop_corr:
          X_opt = X_opt.drop(columns=list(to_drop_corr))
          logger.debug(f"Удалено {len(to_drop_corr)} высококоррелированных признаков")

      # 3. Отбор по важности (если признаков слишком много)
      max_features = 30  # Максимальное количество дополнительных признаков
      if len(X_opt.columns) > max_features:
        logger.debug(f"Отбор {max_features} лучших признаков из {len(X_opt.columns)}...")

        try:
          # Используем Random Forest для оценки важности
          rf_selector = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Для работы с несбалансированными данными
          )

          rf_selector.fit(X_opt, y)

          # Получаем важности и отбираем лучшие
          importances = pd.Series(rf_selector.feature_importances_, index=X_opt.columns)
          best_features = importances.nlargest(max_features).index.tolist()
          X_opt = X_opt[best_features]

          logger.debug(f"Отобрано {len(best_features)} признаков по важности")

        except Exception as importance_error:
          logger.warning(f"Ошибка при отборе по важности: {importance_error}")
          # При ошибке берем первые max_features колонок
          X_opt = X_opt.iloc[:, :max_features]

      # 4. Финальный отбор с помощью RFE (если признаков все еще много)
      if len(X_opt.columns) > 20:
        logger.debug("Финальный отбор с помощью RFE...")
        try:
          from sklearn.feature_selection import RFE

          rfe_estimator = xgb.XGBClassifier(
            n_estimators=50,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
          )

          rfe = RFE(estimator=rfe_estimator, n_features_to_select=20)
          rfe.fit(X_opt, y)

          selected_by_rfe = X_opt.columns[rfe.support_].tolist()
          X_opt = X_opt[selected_by_rfe]

          logger.debug(f"RFE отобрал {len(selected_by_rfe)} финальных признаков")

        except Exception as rfe_error:
          logger.warning(f"Ошибка RFE: {rfe_error}")

      # 5. Объединяем базовые и оптимизированные признаки
      if base_columns:
        final_columns = base_columns + list(X_opt.columns)
        X_final = pd.concat([X[base_columns], X_opt], axis=1)
      else:
        final_columns = list(X_opt.columns)
        X_final = X_opt

      logger.info(
        f"Оптимизация завершена: {len(base_columns)} базовых + {len(X_opt.columns)} оптимизированных = {len(final_columns)} итоговых признаков")

      return X_final, final_columns

    except Exception as e:
      logger.error(f"Критическая ошибка при оптимизации признаков: {e}")
      # При критической ошибке возвращаем исходные данные
      return X, list(X.columns)

  def _clone_model(self, model):
    """Клонирование модели с теми же параметрами"""
    from sklearn.base import clone
    return clone(model)

  def _analyze_feature_importance(self, X: pd.DataFrame):
    """Анализ важности признаков"""
    importances = {}

    for name, model in self.models.items():
      if hasattr(model, 'feature_importances_'):
        importances[name] = pd.Series(
          model.feature_importances_,
          index=X.columns
        ).nlargest(20).to_dict()

    self.feature_importance_history.append({
      'timestamp': datetime.now(),
      'importances': importances
    })

  def _get_prediction_feature_importance(self, x: np.ndarray) -> Dict[str, float]:
    """Важность признаков для конкретного предсказания"""
    # Используем SHAP values или похожий подход
    # Здесь упрощенная версия

    if hasattr(self.models['rf'], 'feature_importances_'):
      return dict(zip(
        self.selected_features[:10] if self.selected_features else [],
        self.models['rf'].feature_importances_[:10]
      ))
    return {}

  def save(self, filepath: str):
    """Сохранение модели"""
    model_data = {
      'models': self.models,
      'meta_model': self.meta_model,
      'scalers': self.scalers,
      'selected_features': self.selected_features,
      'feature_engineer': self.feature_engineer,
      'performance_history': self.performance_history,
      'is_fitted': self.is_fitted
    }

    joblib.dump(model_data, filepath)
    logger.info(f"Enhanced model сохранена в {filepath}")

  @classmethod
  def load(cls, filepath: str, anomaly_detector: Optional[MarketAnomalyDetector] = None) -> 'EnhancedEnsembleModel':
    """Загрузка модели"""
    model_data = joblib.load(filepath)

    instance = cls(anomaly_detector)
    instance.models = model_data['models']
    instance.meta_model = model_data['meta_model']
    instance.scalers = model_data['scalers']
    instance.selected_features = model_data['selected_features']
    instance.feature_engineer = model_data['feature_engineer']
    instance.performance_history = model_data['performance_history']
    instance.is_fitted = model_data['is_fitted']

    logger.info(f"Enhanced model загружена из {filepath}")
    return instance

  def analyze_data_quality(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Анализ качества данных для диагностики проблем (ИСПРАВЛЕННАЯ ВЕРСИЯ)
    """
    try:
      analysis = {
        'data_shape': X.shape,
        'target_distribution': y.value_counts().to_dict(),
        'missing_values': {},
        'infinite_values': {},
        'zero_variance_columns': [],
        'data_types': {}
      }

      # ИСПРАВЛЕНИЕ: Безопасная обработка missing values
      try:
        analysis['missing_values'] = X.isnull().sum().to_dict()
      except Exception as e:
        logger.warning(f"Ошибка анализа пропущенных значений: {e}")
        analysis['missing_values'] = {}

      # ИСПРАВЛЕНИЕ: Безопасная обработка infinite values
      try:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
          analysis['infinite_values'] = np.isinf(X[numeric_cols]).sum().to_dict()
        else:
          analysis['infinite_values'] = {}
      except Exception as e:
        logger.warning(f"Ошибка анализа бесконечных значений: {e}")
        analysis['infinite_values'] = {}

      # ИСПРАВЛЕНИЕ: Безопасная обработка variance
      try:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
          zero_var_cols = []
          for col in numeric_cols:
            try:
              if X[col].var() == 0:
                zero_var_cols.append(col)
            except:
              continue
          analysis['zero_variance_columns'] = zero_var_cols
        else:
          analysis['zero_variance_columns'] = []
      except Exception as e:
        logger.warning(f"Ошибка анализа дисперсии: {e}")
        analysis['zero_variance_columns'] = []

      # ИСПРАВЛЕНИЕ: Безопасная обработка типов данных
      try:
        # Преобразуем dtype в строки для JSON совместимости
        analysis['data_types'] = {col: str(dtype) for col, dtype in X.dtypes.to_dict().items()}
      except Exception as e:
        logger.warning(f"Ошибка анализа типов данных: {e}")
        analysis['data_types'] = {}

      # Анализ дисбаланса классов
      try:
        class_counts = y.value_counts()
        total_samples = len(y)
        analysis['class_balance'] = {
          'counts': class_counts.to_dict(),
          'percentages': (class_counts / total_samples * 100).to_dict(),
          'imbalance_ratio': float(class_counts.max() / class_counts.min()) if class_counts.min() > 0 else float('inf')
        }
      except Exception as e:
        logger.warning(f"Ошибка анализа баланса классов: {e}")
        analysis['class_balance'] = {
          'counts': {},
          'percentages': {},
          'imbalance_ratio': 1.0
        }

      # Рекомендации
      recommendations = []

      try:
        if analysis['class_balance']['imbalance_ratio'] > 10:
          recommendations.append("Сильный дисбаланс классов - используйте class_weight='balanced'")

        if any(count > 0 for count in analysis['missing_values'].values()):
          recommendations.append("Обнаружены пропущенные значения - требуется импутация")

        if any(count > 0 for count in analysis['infinite_values'].values()):
          recommendations.append("Обнаружены бесконечные значения - требуется очистка")

        if len(analysis['zero_variance_columns']) > 0:
          recommendations.append(f"Обнаружено {len(analysis['zero_variance_columns'])} колонок с нулевой дисперсией")
      except Exception as e:
        logger.warning(f"Ошибка создания рекомендаций: {e}")

      analysis['recommendations'] = recommendations

      return analysis

    except Exception as e:
      logger.error(f"Критическая ошибка в analyze_data_quality: {e}")
      # Возвращаем минимальную структуру при ошибке
      return {
        'data_shape': getattr(X, 'shape', (0, 0)),
        'target_distribution': {},
        'missing_values': {},
        'infinite_values': {},
        'zero_variance_columns': [],
        'data_types': {},
        'class_balance': {'counts': {}, 'percentages': {}, 'imbalance_ratio': 1.0},
        'recommendations': ['Ошибка анализа данных'],
        'error': str(e)
      }

  def diagnose_training_issues(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Комплексная диагностика проблем обучения (ИСПРАВЛЕННАЯ ВЕРСИЯ)
    """
    logger.info("Запуск диагностики проблем обучения...")

    diagnosis = {
      'timestamp': datetime.now().isoformat(),
      'issues_found': [],
      'warnings': [],
      'recommendations': [],
      'data_analysis': {},
      'overall_status': 'НЕИЗВЕСТНО',  # ИСПРАВЛЕНИЕ: Добавляем значение по умолчанию
      'severity_score': 0
    }

    try:
      # 1. Анализ качества данных
      try:
        data_quality = self.analyze_data_quality(X, y)
        diagnosis['data_analysis'] = data_quality
      except Exception as quality_error:
        logger.error(f"Ошибка анализа качества данных: {quality_error}")
        diagnosis['data_analysis'] = {'error': str(quality_error)}
        diagnosis['issues_found'].append(f"Ошибка анализа данных: {quality_error}")

      # 2. Проверка размера данных
      try:
        min_samples_per_class = 50
        class_counts = y.value_counts()

        for class_label, count in class_counts.items():
          if count < min_samples_per_class:
            diagnosis['issues_found'].append(
              f"Недостаточно образцов для класса {class_label}: {count} (минимум {min_samples_per_class})"
            )
      except Exception as size_error:
        logger.warning(f"Ошибка проверки размера данных: {size_error}")
        diagnosis['warnings'].append(f"Не удалось проверить размер данных: {size_error}")

      # 3. Проверка дисбаланса классов
      try:
        if 'class_balance' in diagnosis['data_analysis']:
          imbalance_ratio = diagnosis['data_analysis']['class_balance'].get('imbalance_ratio', 1.0)
          if imbalance_ratio > 10:
            diagnosis['issues_found'].append(
              f"Критический дисбаланс классов: {imbalance_ratio:.1f}:1"
            )
            diagnosis['recommendations'].append("Используйте SMOTE или class_weight='balanced'")
      except Exception as balance_error:
        logger.warning(f"Ошибка проверки баланса классов: {balance_error}")

      # 4. Проверка признаков
      try:
        if 'zero_variance_columns' in diagnosis['data_analysis']:
          zero_var_cols = diagnosis['data_analysis']['zero_variance_columns']
          if len(zero_var_cols) > 0:
            diagnosis['warnings'].append(
              f"Найдено {len(zero_var_cols)} признаков с нулевой дисперсией"
            )
      except Exception as feature_error:
        logger.warning(f"Ошибка проверки признаков: {feature_error}")

      # 5. Проверка корреляций (БЕЗОПАСНАЯ ВЕРСИЯ)
      try:
        if len(X.columns) > 1:
          # ИСПРАВЛЕНИЕ: Работаем только с числовыми колонками
          numeric_cols = X.select_dtypes(include=[np.number]).columns
          if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr().abs()
            high_corr_pairs = []

            for i in range(len(corr_matrix.columns)):
              for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if pd.notna(corr_val) and corr_val > 0.95:
                  high_corr_pairs.append(
                    (corr_matrix.columns[i], corr_matrix.columns[j], corr_val)
                  )

            if len(high_corr_pairs) > 0:
              diagnosis['warnings'].append(
                f"Найдено {len(high_corr_pairs)} пар высококоррелированных признаков"
              )
              diagnosis['recommendations'].append("Удалите дублирующие признаки")
      except Exception as corr_error:
        logger.warning(f"Ошибка проверки корреляций: {corr_error}")

      # 6. Проверка размера выборки
      try:
        total_samples = len(X)
        num_features = len(X.columns)

        if total_samples < num_features * 10:
          diagnosis['warnings'].append(
            f"Мало образцов относительно признаков: {total_samples} образцов, {num_features} признаков"
          )
          diagnosis['recommendations'].append("Увеличьте объем данных или уменьшите количество признаков")
      except Exception as sample_error:
        logger.warning(f"Ошибка проверки размера выборки: {sample_error}")

      # 7. Итоговая оценка
      try:
        severity_score = len(diagnosis['issues_found']) * 2 + len(diagnosis['warnings'])

        if severity_score == 0:
          diagnosis['overall_status'] = 'ОТЛИЧНО'
        elif severity_score <= 2:
          diagnosis['overall_status'] = 'ХОРОШО'
        elif severity_score <= 5:
          diagnosis['overall_status'] = 'УДОВЛЕТВОРИТЕЛЬНО'
        else:
          diagnosis['overall_status'] = 'ТРЕБУЕТ_ВНИМАНИЯ'

        diagnosis['severity_score'] = severity_score
      except Exception as score_error:
        logger.warning(f"Ошибка расчета итоговой оценки: {score_error}")
        diagnosis['overall_status'] = 'ОШИБКА_ОЦЕНКИ'
        diagnosis['severity_score'] = 999

      logger.info(f"Диагностика завершена. Статус: {diagnosis['overall_status']}")

      return diagnosis

    except Exception as e:
      logger.error(f"Критическая ошибка при диагностике: {e}")
      diagnosis['error'] = str(e)
      diagnosis['overall_status'] = 'КРИТИЧЕСКАЯ_ОШИБКА'
      diagnosis['issues_found'].append(f"Критическая ошибка диагностики: {e}")
      return diagnosis

  def print_training_report(self, X: pd.DataFrame, y: pd.Series,
                              diagnosis: Optional[Dict] = None) -> None:
      """
      Выводит подробный отчет о процессе обучения
      """
      print("\n" + "=" * 60)
      print("📊 ОТЧЕТ ОБ ОБУЧЕНИИ ENHANCED ML МОДЕЛИ")
      print("=" * 60)

      # Базовая информация
      print(f"📈 Размер данных: {X.shape[0]} образцов, {X.shape[1]} признаков")
      print(f"🎯 Распределение классов:")

      class_counts = y.value_counts().sort_index()
      class_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

      for class_idx, count in class_counts.items():
        percentage = count / len(y) * 100
        class_name = class_names.get(class_idx, f'Class_{class_idx}')
        print(f"   {class_name}: {count} ({percentage:.1f}%)")

      # Информация о признаках
      if hasattr(self, 'selected_features') and self.selected_features:
        print(f"🔧 Выбранные признаки: {len(self.selected_features)}")

        # Показываем важные признаки
        if hasattr(self.models.get('rf'), 'feature_importances_'):
          importances = self.models['rf'].feature_importances_
          feature_importance = list(zip(self.selected_features, importances))
          feature_importance.sort(key=lambda x: x[1], reverse=True)

          print("🏆 Топ-5 важных признаков:")
          for i, (feature, importance) in enumerate(feature_importance[:5], 1):
            print(f"   {i}. {feature}: {importance:.4f}")

      # Диагностика
      if diagnosis:
        print(f"\n🔍 Статус диагностики: {diagnosis.get('overall_status', 'НЕИЗВЕСТНО')}")

        if diagnosis.get('issues_found'):
          print("❌ Критические проблемы:")
          for issue in diagnosis['issues_found']:
            print(f"   • {issue}")

        if diagnosis.get('warnings'):
          print("⚠️  Предупреждения:")
          for warning in diagnosis['warnings']:
            print(f"   • {warning}")

        if diagnosis.get('recommendations'):
          print("💡 Рекомендации:")
          for rec in diagnosis['recommendations']:
            print(f"   • {rec}")

      # Информация о моделях
      print(f"\n🤖 Статус моделей:")
      for name, model in self.models.items():
        status = "✅ Обучена" if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_') else "❌ Не обучена"
        print(f"   {name}: {status}")

      meta_status = "✅ Обучена" if hasattr(self.meta_model, 'feature_importances_') else "❌ Не обучена"
      print(f"   meta_model: {meta_status}")

      print(f"\n🔧 Общий статус: {'✅ ГОТОВА К РАБОТЕ' if self.is_fitted else '❌ ТРЕБУЕТ ОБУЧЕНИЯ'}")
      print("=" * 60)

  def create_feature_importance_report(self) -> str:
    """
    Создает детальный отчет о важности признаков
    """
    if not self.is_fitted:
      return "Модель не обучена. Нет данных о важности признаков."

    report_lines = [
      "📊 ОТЧЕТ О ВАЖНОСТИ ПРИЗНАКОВ",
      "=" * 50
    ]

    # Собираем важности от всех моделей
    all_importances = {}

    for model_name, model in self.models.items():
      if hasattr(model, 'feature_importances_') and self.selected_features:
        importances = dict(zip(self.selected_features, model.feature_importances_))
        all_importances[model_name] = importances

    if not all_importances:
      return "Нет доступной информации о важности признаков."

    # Вычисляем среднюю важность
    if len(all_importances) > 1:
      avg_importances = {}
      for feature in self.selected_features:
        feature_importances = [imp.get(feature, 0) for imp in all_importances.values()]
        avg_importances[feature] = np.mean(feature_importances)

      # Сортируем по убыванию важности
      sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)

      report_lines.append("🎯 ТОП-15 ВАЖНЫХ ПРИЗНАКОВ (среднее по моделям):")
      for i, (feature, importance) in enumerate(sorted_features[:15], 1):
        report_lines.append(f"  {i:2d}. {feature:<35} {importance:.4f}")

    # Детализация по моделям
    report_lines.extend([
      "",
      "🔬 ДЕТАЛИЗАЦИЯ ПО МОДЕЛЯМ:",
      "-" * 30
    ])

    for model_name, importances in all_importances.items():
      report_lines.append(f"\n{model_name.upper()}:")
      sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
      for i, (feature, importance) in enumerate(sorted_imp[:10], 1):
        report_lines.append(f"  {i:2d}. {feature:<30} {importance:.4f}")

    return "\n".join(report_lines)

  def monitor_prediction_quality(self, predictions: np.ndarray,
                                 true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Мониторинг качества предсказаний в реальном времени
    """
    monitoring_data = {
      'timestamp': datetime.now().isoformat(),
      'prediction_stats': {},
      'quality_metrics': {},
      'alerts': []
    }

    try:
      # Анализ распределения предсказаний
      unique, counts = np.unique(predictions, return_counts=True)
      pred_distribution = dict(zip(unique, counts))

      monitoring_data['prediction_stats'] = {
        'total_predictions': len(predictions),
        'distribution': pred_distribution,
        'distribution_percentages': {
          k: v / len(predictions) * 100 for k, v in pred_distribution.items()
        }
      }

      # Проверка на аномальные паттерны
      total_preds = len(predictions)

      # Слишком много одного класса
      for class_label, count in pred_distribution.items():
        percentage = count / total_preds * 100
        if percentage > 80:
          monitoring_data['alerts'].append(
            f"⚠️ Слишком много предсказаний класса {class_label}: {percentage:.1f}%"
          )

      # Если есть истинные метки, вычисляем метрики
      if true_labels is not None and len(true_labels) == len(predictions):
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        accuracy = accuracy_score(true_labels, predictions)
        monitoring_data['quality_metrics']['accuracy'] = float(accuracy)

        # Матрица ошибок
        cm = confusion_matrix(true_labels, predictions)
        monitoring_data['quality_metrics']['confusion_matrix'] = cm.tolist()

        # Детальный отчет по классам
        class_report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
        monitoring_data['quality_metrics']['classification_report'] = class_report

        # Алерты по качеству
        if accuracy < 0.6:
          monitoring_data['alerts'].append(
            f"🚨 Низкая точность: {accuracy:.2%}"
          )

      # Проверка на паттерны деградации
      if hasattr(self, 'prediction_history'):
        self.prediction_history.append({
          'timestamp': datetime.now(),
          'predictions': predictions.copy(),
          'distribution': pred_distribution
        })

        # Ограничиваем историю
        if len(self.prediction_history) > 100:
          self.prediction_history = self.prediction_history[-100:]

        # Анализ трендов (если есть достаточно истории)
        if len(self.prediction_history) >= 10:
          recent_distributions = [
            h['distribution'] for h in self.prediction_history[-10:]
          ]

          # Проверяем, не меняется ли радикально распределение
          for class_label in [0, 1, 2]:
            recent_percentages = [
              d.get(class_label, 0) / sum(d.values()) * 100
              for d in recent_distributions
            ]

            if len(recent_percentages) >= 5:
              trend_change = abs(recent_percentages[-1] - recent_percentages[0])
              if trend_change > 30:  # Изменение больше 30%
                monitoring_data['alerts'].append(
                  f"📈 Резкое изменение в предсказаниях класса {class_label}: {trend_change:.1f}%"
                )
      else:
        self.prediction_history = []

      return monitoring_data

    except Exception as e:
      logger.error(f"Ошибка мониторинга предсказаний: {e}")
      monitoring_data['error'] = str(e)
      return monitoring_data

  def get_model_health_status(self) -> Dict[str, Any]:
    """
    Проверка "здоровья" модели
    """
    health_status = {
      'timestamp': datetime.now().isoformat(),
      'overall_health': 'UNKNOWN',
      'components': {},
      'issues': [],
      'recommendations': []
    }

    try:
      # Проверка состояния обучения
      health_status['components']['training_status'] = 'OK' if self.is_fitted else 'NOT_TRAINED'

      if not self.is_fitted:
        health_status['issues'].append("Модель не обучена")
        health_status['recommendations'].append("Выполните обучение модели")

      # Проверка моделей
      working_models = 0
      total_models = len(self.models)

      for name, model in self.models.items():
        try:
          # Простая проверка: есть ли у модели атрибуты обученной модели
          if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_') or hasattr(model, 'tree_'):
            working_models += 1
            health_status['components'][f'model_{name}'] = 'OK'
          else:
            health_status['components'][f'model_{name}'] = 'NOT_TRAINED'
        except Exception as e:
          health_status['components'][f'model_{name}'] = f'ERROR: {str(e)}'

      # Проверка мета-модели
      try:
        if hasattr(self.meta_model, 'feature_importances_') or hasattr(self.meta_model, 'coef_'):
          health_status['components']['meta_model'] = 'OK'
        else:
          health_status['components']['meta_model'] = 'NOT_TRAINED'
      except Exception as e:
        health_status['components']['meta_model'] = f'ERROR: {str(e)}'

      # Проверка скейлеров
      scalers_ok = 0
      for name, scaler in self.scalers.items():
        try:
          if hasattr(scaler, 'mean_') or hasattr(scaler, 'center_'):
            scalers_ok += 1
            health_status['components'][f'scaler_{name}'] = 'OK'
          else:
            health_status['components'][f'scaler_{name}'] = 'NOT_FITTED'
        except Exception as e:
          health_status['components'][f'scaler_{name}'] = f'ERROR: {str(e)}'

      # Проверка feature_engineer
      try:
        if hasattr(self.feature_engineer, 'feature_names') and self.feature_engineer.feature_names:
          health_status['components']['feature_engineer'] = 'OK'
        else:
          health_status['components']['feature_engineer'] = 'NO_FEATURES'
      except Exception as e:
        health_status['components']['feature_engineer'] = f'ERROR: {str(e)}'

      # Проверка selected_features
      if hasattr(self, 'selected_features') and self.selected_features:
        health_status['components']['selected_features'] = f'OK ({len(self.selected_features)} features)'
      else:
        health_status['components']['selected_features'] = 'NO_FEATURES'
        health_status['issues'].append("Нет информации о выбранных признаках")

      # Общая оценка здоровья
      total_components = len(health_status['components'])
      ok_components = len([v for v in health_status['components'].values() if v == 'OK' or v.startswith('OK')])

      health_percentage = ok_components / total_components if total_components > 0 else 0

      if health_percentage >= 0.9:
        health_status['overall_health'] = 'EXCELLENT'
      elif health_percentage >= 0.7:
        health_status['overall_health'] = 'GOOD'
      elif health_percentage >= 0.5:
        health_status['overall_health'] = 'FAIR'
      else:
        health_status['overall_health'] = 'POOR'

      health_status['health_percentage'] = health_percentage * 100

      # Рекомендации на основе проблем
      if working_models < total_models // 2:
        health_status['issues'].append(f"Работает только {working_models} из {total_models} моделей")
        health_status['recommendations'].append("Переобучите проблемные модели")

      if scalers_ok == 0:
        health_status['issues'].append("Ни один скейлер не обучен")
        health_status['recommendations'].append("Переобучите модель полностью")

      return health_status

    except Exception as e:
      logger.error(f"Ошибка проверки здоровья модели: {e}")
      health_status['error'] = str(e)
      health_status['overall_health'] = 'ERROR'
      return health_status

