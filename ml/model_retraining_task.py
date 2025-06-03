import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import logging
import pickle
import os
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from aiogram.dispatcher.middlewares import manager
from logger_setup import setup_logging

from data.database_manager import AdvancedDatabaseManager
from strategies.base_strategy import BaseStrategy
from strategies.ensemble_ml_strategy import EnsembleMLStrategy

# from TG_bot import telegram_bot

# Импорты для ML и статистики
try:
  from sklearn.model_selection import TimeSeriesSplit, cross_val_score
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
  from sklearn.ensemble import RandomForestClassifier
  import joblib

  SKLEARN_AVAILABLE = True
except ImportError:
  SKLEARN_AVAILABLE = False
  logging.warning("sklearn не доступен. Некоторые функции переобучения будут ограничены.")

# Импорты из нашей системы (предполагается структура проекта)
from .feature_engineering import create_features_and_labels, analyze_feature_importance, get_feature_statistics, \
  validate_data_quality

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics(BaseStrategy):
  """Класс для хранения метрик производительности модели"""
  accuracy: float
  precision: float
  recall: float
  f1_score: float
  cross_val_mean: float
  cross_val_std: float
  feature_count: int
  training_samples: int
  training_time_seconds: float
  timestamp: datetime
  model_version: str

  def to_dict(self) -> Dict:
    """Конвертация в словарь для сериализации"""
    result = asdict(self)
    result['timestamp'] = self.timestamp.isoformat()
    return result

  @classmethod
  def from_dict(cls, data: Dict) -> 'ModelPerformanceMetrics':
    """Создание из словаря"""
    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
    return cls(**data)


@dataclass
class RetrainingConfig:
  """Конфигурация для переобучения модели"""
  min_data_points: int = 1000
  max_data_points: int = 50000
  retraining_interval_hours: float = 24.0
  performance_threshold: float = 0.55
  min_improvement: float = 0.02
  backup_models_count: int = 5
  cross_validation_folds: int = 5
  feature_selection_threshold: float = 0.01
  auto_feature_selection: bool = True
  adaptive_thresholds: bool = True
  market_regime_detection: bool = True
  ensemble_models: bool = True
  parallel_processing: bool = True
  max_workers: int = 4



class ModelRetrainingManager:
  """
  Продвинутый менеджер для управления переобучением моделей машинного обучения
  с поддержкой адаптивного обучения, мониторинга производительности и backup системы
  """
  def __init__(self,
               model_save_path: str = "ml_models/",
               config: Optional[RetrainingConfig] = None,
               data_fetcher=None,
               ml_model=None,
               db_manager=None):
    """
    Инициализация менеджера переобучения

    Args:
        model_save_path: Путь для сохранения моделей
        config: Конфигурация переобучения
        data_fetcher: Объект для получения данных
        ml_model: ML модель для переобучения
        db_manager: Менеджер базы данных
    """
    self.config = config or RetrainingConfig()
    self.model_save_path = Path(model_save_path)
    self.model_save_path.mkdir(exist_ok=True, parents=True)

    # Создаем подпапки
    (self.model_save_path / "backups").mkdir(exist_ok=True)
    (self.model_save_path / "performance_logs").mkdir(exist_ok=True)
    (self.model_save_path / "feature_importance").mkdir(exist_ok=True)

    self.data_fetcher = data_fetcher
    self.ml_model = ml_model
    self.db_manager = db_manager

    # Состояние системы
    self.is_running = False
    self.last_retrain_time = None
    self.current_model_version = "v1.0.0"
    self.performance_history: List[ModelPerformanceMetrics] = []
    self.feature_importance_history: List[Dict] = []

    # Очередь задач и пул потоков
    self.task_queue = asyncio.Queue()
    self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)

    # Блокировки для потокобезопасности
    self.model_lock = threading.Lock()
    self.performance_lock = threading.Lock()

    # Callbacks
    self.on_retrain_complete: Optional[Callable] = None
    self.on_performance_decline: Optional[Callable] = None

    # Загружаем историю производительности
    self._load_performance_history()

    logger.info(f"ModelRetrainingManager инициализирован с конфигурацией: {self.config}")

  def calculate_vpt_manual(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    VPT = Previous VPT + Volume * (Close - Previous Close) / Previous Close
    """
    vpt = pd.Series(index=close.index, dtype=float)
    vpt.iloc[0] = 0

    for i in range(1, len(close)):
        if pd.notna(close.iloc[i-1]) and close.iloc[i-1] != 0:
            price_change_pct = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
            vpt.iloc[i] = vpt.iloc[i-1] + volume.iloc[i] * price_change_pct
        else:
            vpt.iloc[i] = vpt.iloc[i-1]

    return vpt

  def calculate_vwap_manual(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Ручная реализация VWAP (Volume Weighted Average Price)
    """
    try:
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap

    except Exception as e:
        logger.error(f"Ошибка расчета VWAP: {e}")
        return close.copy()

  def safe_indicator_calculation(func, *args, **kwargs):
    """
    Обертка для безопасного вызова функций индикаторов
    """
    try:
        result = func(*args, **kwargs)
        if result is None or (hasattr(result, 'empty') and result.empty):
            return None
        return result
    except Exception as e:
        logger.warning(f"Ошибка в индикаторе {func.__name__}: {e}")
        return None

  def validate_dataframe_for_indicators(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Проверяет, подходит ли DataFrame для расчета индикаторов
    """
    try:
        # Проверка наличия колонок
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Отсутствуют колонки: {missing_cols}"

        # Проверка размера данных
        if len(df) < 50:
            return False, f"Недостаточно данных: {len(df)} строк (требуется минимум 50)"

        # Проверка на NaN значения
        nan_counts = df[required_cols].isnull().sum()
        high_nan_cols = [col for col, count in nan_counts.items() if count > len(df) * 0.1]
        if high_nan_cols:
            return False, f"Слишком много NaN в колонках: {high_nan_cols}"

        # Проверка логичности данных
        if (df['high'] < df['low']).any():
            return False, "Обнаружены некорректные данные: high < low"

        if (df['high'] < df['close']).any() or (df['low'] > df['close']).any():
            return False, "Обнаружены некорректные данные: close вне диапазона high-low"

        if (df['volume'] < 0).any():
            return False, "Обнаружены отрицательные объемы"

        return True, "Данные прошли валидацию"

    except Exception as e:
        return False, f"Ошибка валидации: {e}"

  def get_available_indicators() -> Dict[str, List[str]]:
    """
    Возвращает список доступных индикаторов по категориям
    """
    indicators = {
        'trend': ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 'ema_50', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0'],
        'momentum': ['rsi', 'rsi_30', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'willr', 'cci'],
        'volume': ['obv', 'ad', 'volume_sma', 'vpt', 'vwap', 'volume_ratio'],
        'volatility': ['atr', 'true_range', 'bb_width'],
        'directional': ['ADX_14', 'DMP_14', 'DMN_14', 'psar', 'AROONU_14', 'AROOND_14'],
        'custom': ['price_position', 'price_change', 'price_change_abs', 'hl_spread', 'close_position']
    }
    return indicators

  def _load_performance_history(self):
    """Загружает историю производительности из файла"""
    try:
      history_file = self.model_save_path / "performance_logs" / "performance_history.json"
      if history_file.exists():
        with open(history_file, 'r') as f:
          data = json.load(f)
          self.performance_history = [
            ModelPerformanceMetrics.from_dict(item) for item in data
          ]
        logger.info(f"Загружена история производительности: {len(self.performance_history)} записей")
    except Exception as e:
      logger.error(f"Ошибка загрузки истории производительности: {e}")
      self.performance_history = []

  def _save_performance_history(self):
    """Сохраняет историю производительности в файл"""
    try:
      history_file = self.model_save_path / "performance_logs" / "performance_history.json"
      with open(history_file, 'w') as f:
        json.dump([metric.to_dict() for metric in self.performance_history], f, indent=2)
    except Exception as e:
      logger.error(f"Ошибка сохранения истории производительности: {e}")

  def _generate_model_version(self) -> str:
    """Генерирует новую версию модели"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"v{timestamp}"

  async def _get_training_data(self, symbols: List[str],
                               timeframe: str = '1h',
                               limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Получает данные для обучения модели

    Args:
        symbols: Список символов для получения данных
        timeframe: Таймфрейм данных
        limit: Количество свечей

    Returns:
        DataFrame с данными или None в случае ошибки
    """
    try:
      if not self.data_fetcher:
        logger.error("DataFetcher не предоставлен")
        return None

      all_data = {}

      for symbol in symbols:
        try:
          data = await self.data_fetcher.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=min(limit, self.config.max_data_points)
          )

          if not data.empty:
            all_data[symbol] = data
            logger.info(f"Получены данные для {symbol}: {len(data)} записей")
          else:
            logger.warning(f"Пустые данные для символа {symbol}")

        except Exception as e:
          logger.error(f"Ошибка получения данных для {symbol}: {e}")
          continue

      if not all_data:
        logger.error("Не удалось получить данные ни для одного символа")
        return None

      # Объединяем данные всех символов
      combined_data = pd.concat(all_data.values(), keys=all_data.keys(), names=['symbol', 'index'])
      combined_data = combined_data.reset_index(level=0)  # symbol становится колонкой

      logger.info(f"Общий размер обучающих данных: {len(combined_data)} записей")
      return combined_data

    except Exception as e:
      logger.error(f"Ошибка при получении обучающих данных: {e}", exc_info=True)
      return None

  def _validate_data_quality(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[bool, str]:
    """
    Валидирует качество данных для обучения

    Returns:
        Tuple[is_valid, error_message]
    """
    try:
      # Проверка размера данных
      if len(features) < self.config.min_data_points:
        return False, f"Недостаточно данных: {len(features)} < {self.config.min_data_points}"

      # Проверка на пропущенные значения
      missing_features = features.isnull().sum().sum()
      missing_labels = labels.isnull().sum()

      if missing_features > len(features) * 0.05:  # Более 5% пропущенных значений
        return False, f"Слишком много пропущенных значений в признаках: {missing_features}"

      if missing_labels > 0:
        return False, f"Пропущенные значения в метках: {missing_labels}"

      # Проверка распределения классов
      class_distribution = labels.value_counts()
      min_class_size = class_distribution.min()

      if min_class_size < len(labels) * 0.05:  # Менее 5% для любого класса
        return False, f"Несбалансированные классы: {class_distribution.to_dict()}"

      # Проверка на константные признаки
      constant_features = features.nunique() == 1
      if constant_features.any():
        const_count = constant_features.sum()
        logger.warning(f"Найдено {const_count} константных признаков, они будут удалены")

      # Проверка корреляции признаков (для выявления дублирующих)
      if len(features.columns) > 1:
        corr_matrix = features.corr().abs()
        upper_triangle = corr_matrix.where(
          np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr_pairs = (upper_triangle > 0.95).sum().sum()

        if high_corr_pairs > len(features.columns) * 0.1:
          logger.warning(f"Найдено {high_corr_pairs} пар признаков с высокой корреляцией")

      return True, "Данные прошли валидацию"

    except Exception as e:
      return False, f"Ошибка валидации данных: {e}"

  def _preprocess_features(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Предобработка признаков перед обучением

    Returns:
        Tuple[processed_features, processed_labels]
    """
    try:
      processed_features = features.copy()
      processed_labels = labels.copy()

      # Удаляем константные признаки
      constant_features = processed_features.nunique() == 1
      if constant_features.any():
        const_cols = processed_features.columns[constant_features].tolist()
        processed_features = processed_features.drop(columns=const_cols)
        logger.info(f"Удалено {len(const_cols)} константных признаков")

      # Удаляем признаки с высокой корреляцией
      if self.config.auto_feature_selection and len(processed_features.columns) > 1:
        corr_matrix = processed_features.corr().abs()
        upper_triangle = corr_matrix.where(
          np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [column for column in upper_triangle.columns
                   if any(upper_triangle[column] > 0.95)]

        if to_drop:
          processed_features = processed_features.drop(columns=to_drop)
          logger.info(f"Удалено {len(to_drop)} признаков с высокой корреляцией")

      # Обработка выбросов (опционально)
      if self.config.adaptive_thresholds:
        for col in processed_features.select_dtypes(include=[np.number]).columns:
          Q1 = processed_features[col].quantile(0.01)
          Q3 = processed_features[col].quantile(0.99)
          processed_features[col] = processed_features[col].clip(lower=Q1, upper=Q3)

      # Убеждаемся, что индексы совпадают
      common_index = processed_features.index.intersection(processed_labels.index)
      processed_features = processed_features.loc[common_index]
      processed_labels = processed_labels.loc[common_index]

      logger.info(f"Предобработка завершена. Финальный размер: {processed_features.shape}")
      return processed_features, processed_labels

    except Exception as e:
      logger.error(f"Ошибка предобработки признаков: {e}", exc_info=True)
      return features, labels

  def _train_model(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[bool, Optional[ModelPerformanceMetrics]]:
    """
    Обучает модель и возвращает метрики производительности

    Returns:
        Tuple[success, performance_metrics]
    """
    try:
      start_time = time.time()

      with self.model_lock:
        # Обучаем основную модель
        if hasattr(self.ml_model, 'fit'):
          self.ml_model.fit(features, labels)
        else:
          logger.error("ML модель не имеет метода fit()")
          return False, None

        # Проверяем, что модель обучена
        if hasattr(self.ml_model, 'is_fitted'):
          if not self.ml_model.is_fitted:
            logger.error("Модель не была обучена корректно")
            return False, None

      training_time = time.time() - start_time

      # Оцениваем производительность
      if SKLEARN_AVAILABLE:
        metrics = self._evaluate_model_performance(features, labels, training_time)
        return True, metrics
      else:
        # Базовая оценка без sklearn
        predictions = self.ml_model.predict(features) if hasattr(self.ml_model, 'predict') else labels
        accuracy = (predictions == labels).mean() if len(predictions) == len(labels) else 0.5

        metrics = ModelPerformanceMetrics(
          accuracy=accuracy,
          precision=0.0,
          recall=0.0,
          f1_score=0.0,
          cross_val_mean=0.0,
          cross_val_std=0.0,
          feature_count=len(features.columns),
          training_samples=len(features),
          training_time_seconds=training_time,
          timestamp=datetime.now(),
          model_version=self.current_model_version
        )
        return True, metrics

    except Exception as e:
      logger.error(f"Ошибка обучения модели: {e}", exc_info=True)
      return False, None

  def _evaluate_model_performance(self, features: pd.DataFrame, labels: pd.Series,
                                  training_time: float) -> Optional[ModelPerformanceMetrics]:
    """
    Оценивает производительность модели с использованием кросс-валидации
    """
    try:
      if not SKLEARN_AVAILABLE:
        logger.warning("sklearn недоступен для полной оценки модели")
        return None

      # Создаем временную модель для кросс-валидации (если основная модель не поддерживает sklearn интерфейс)
      if hasattr(self.ml_model, 'predict') and hasattr(self.ml_model, 'fit'):
        eval_model = self.ml_model
      else:
        # Используем RandomForest как fallback
        eval_model = RandomForestClassifier(n_estimators=100, random_state=42)
        eval_model.fit(features, labels)

      # Предсказания для расчета метрик
      predictions = eval_model.predict(features)

      # Базовые метрики
      accuracy = accuracy_score(labels, predictions)
      precision = precision_score(labels, predictions, average='weighted', zero_division=0)
      recall = recall_score(labels, predictions, average='weighted', zero_division=0)
      f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

      # Кросс-валидация (только если у нас достаточно данных)
      cv_scores = np.array([0.0])
      if len(features) >= self.config.cross_validation_folds * 50:  # Минимум 50 образцов на фолд
        try:
          tscv = TimeSeriesSplit(n_splits=self.config.cross_validation_folds)
          cv_scores = cross_val_score(eval_model, features, labels, cv=tscv, scoring='accuracy')
        except Exception as cv_error:
          logger.warning(f"Ошибка кросс-валидации: {cv_error}")
          cv_scores = np.array([accuracy])  # Используем обычную точность

      metrics = ModelPerformanceMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        cross_val_mean=cv_scores.mean(),
        cross_val_std=cv_scores.std(),
        feature_count=len(features.columns),
        training_samples=len(features),
        training_time_seconds=training_time,
        timestamp=datetime.now(),
        model_version=self.current_model_version
      )

      logger.info(
        f"Метрики модели: Accuracy={accuracy:.3f}, F1={f1:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
      return metrics

    except Exception as e:
      logger.error(f"Ошибка оценки производительности модели: {e}", exc_info=True)
      return None

  def _should_accept_new_model(self, new_metrics: ModelPerformanceMetrics) -> Tuple[bool, str]:
    """
    Определяет, следует ли принять новую модель на основе метрик производительности

    Returns:
        Tuple[should_accept, reason]
    """
    try:
      if not self.performance_history:
        return True, "Первая обученная модель"

      # Получаем последние метрики
      last_metrics = self.performance_history[-1]

      # Проверяем минимальный порог производительности
      if new_metrics.accuracy < self.config.performance_threshold:
        return False, f"Низкая точность: {new_metrics.accuracy:.3f} < {self.config.performance_threshold}"

      # Проверяем улучшение по сравнению с предыдущей моделью
      improvement = new_metrics.accuracy - last_metrics.accuracy

      if improvement < self.config.min_improvement:
        # Проверяем другие метрики
        f1_improvement = new_metrics.f1_score - last_metrics.f1_score
        cv_improvement = new_metrics.cross_val_mean - last_metrics.cross_val_mean

        if f1_improvement >= self.config.min_improvement * 0.5 or cv_improvement >= self.config.min_improvement * 0.5:
          return True, f"Улучшение по F1 или CV: F1_diff={f1_improvement:.3f}, CV_diff={cv_improvement:.3f}"

        return False, f"Недостаточное улучшение: accuracy_diff={improvement:.3f}, требуется >= {self.config.min_improvement}"

      return True, f"Хорошее улучшение: accuracy_diff={improvement:.3f}"

    except Exception as e:
      logger.error(f"Ошибка проверки критериев принятия модели: {e}")
      return False, f"Ошибка проверки: {e}"

  def _backup_current_model(self):
    """Создает резервную копию текущей модели"""
    try:
      current_model_path = self.model_save_path / "current_model.pkl"
      if not current_model_path.exists():
        return

      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      backup_path = self.model_save_path / "backups" / f"model_backup_{timestamp}.pkl"

      shutil.copy2(current_model_path, backup_path)

      # Удаляем старые backup'ы, если их слишком много
      backup_files = sorted(
        (self.model_save_path / "backups").glob("model_backup_*.pkl"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
      )

      if len(backup_files) > self.config.backup_models_count:
        for old_backup in backup_files[self.config.backup_models_count:]:
          old_backup.unlink()
          logger.info(f"Удален старый backup: {old_backup.name}")

      logger.info(f"Создан backup модели: {backup_path.name}")

    except Exception as e:
      logger.error(f"Ошибка создания backup модели: {e}")

  def _save_model(self, metrics: ModelPerformanceMetrics):
    """Сохраняет обученную модель и её метрики"""
    try:
      # Создаем backup текущей модели
      self._backup_current_model()

      # Сохраняем новую модель
      model_path = self.model_save_path / "current_model.pkl"

      with self.model_lock:
        if hasattr(self.ml_model, 'save_model'):
          self.ml_model.save_model(str(model_path))
        else:
          # Универсальное сохранение через pickle/joblib
          try:
            if SKLEARN_AVAILABLE:
              joblib.dump(self.ml_model, model_path)
            else:
              with open(model_path, 'wb') as f:
                pickle.dump(self.ml_model, f)
          except Exception as save_error:
            logger.error(f"Ошибка сохранения модели: {save_error}")
            return

      # Сохраняем метрики
      metrics_path = self.model_save_path / "performance_logs" / f"metrics_{self.current_model_version}.json"
      with open(metrics_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)

      # Обновляем историю производительности
      with self.performance_lock:
        self.performance_history.append(metrics)
        self._save_performance_history()

      logger.info(f"Модель {self.current_model_version} сохранена с точностью {metrics.accuracy:.3f}")

    except Exception as e:
      logger.error(f"Ошибка сохранения модели: {e}", exc_info=True)

  def _analyze_and_save_feature_importance(self, features: pd.DataFrame, labels: pd.Series):
    """Анализирует и сохраняет важность признаков"""
    try:
      # Используем функцию из feature_engineering.py
      importance_df = analyze_feature_importance(features, labels)

      if not importance_df.empty:
        # Сохраняем важность признаков
        importance_path = self.model_save_path / "feature_importance" / f"importance_{self.current_model_version}.csv"
        importance_df.to_csv(importance_path)

        # Добавляем в историю
        importance_dict = {
          'timestamp': datetime.now().isoformat(),
          'model_version': self.current_model_version,
          'top_features': importance_df.head(20).to_dict()
        }
        self.feature_importance_history.append(importance_dict)

        # Сохраняем историю важности признаков
        history_path = self.model_save_path / "feature_importance" / "importance_history.json"
        with open(history_path, 'w') as f:
          json.dump(self.feature_importance_history, f, indent=2)

        logger.info(f"Анализ важности признаков сохранен для модели {self.current_model_version}")

    except Exception as e:
      logger.error(f"Ошибка анализа важности признаков: {e}")

  async def retrain_model(self, symbols: List[str],
                          timeframe: str = '1h',
                          limit: int = 1000,
                          force_retrain: bool = False) -> Tuple[bool, str]:
    """
    Основная функция переобучения модели

    Args:
        symbols: Список символов для получения данных
        timeframe: Таймфрейм данных
        limit: Количество свечей для получения
        force_retrain: Принудительное переобучение, игнорируя интервалы

    Returns:
        Tuple[success, message]
    """
    try:
      logger.info(f"Начинаем переобучение модели для символов: {symbols}")

      # Проверяем, нужно ли переобучение (если не принудительное)
      if not force_retrain and self.last_retrain_time:
        time_since_last = datetime.now() - self.last_retrain_time
        if time_since_last.total_seconds() < self.config.retraining_interval_hours * 3600:
          remaining_time = self.config.retraining_interval_hours * 3600 - time_since_last.total_seconds()
          return False, f"Переобучение еще не требуется. Осталось {remaining_time / 3600:.1f} часов"

      # 1. Получаем данные
      logger.info("Получение обучающих данных...")
      raw_data = await self._get_training_data(symbols, timeframe, limit)

      if raw_data is None or raw_data.empty:
        return False, "Не удалось получить данные для обучения"

      # 2. Создаем признаки и метки
      logger.info("Создание признаков и меток...")
      features, labels = create_features_and_labels(raw_data)

      if features is None or labels is None:
        return False, "Не удалось создать признаки и метки"

      validation_results = validate_data_quality(features, labels)
      if not validation_results['is_valid']:
        logger.error(f"Data validation failed: {validation_results['errors']}")
        return False, f"Валидация данных не пройдена: {validation_results['errors']}"

      # 3. Валидируем качество данных
      logger.info("Валидация качества данных...")
      is_valid, validation_message = self._validate_data_quality(features, labels)

      if not is_valid:
        return False, f"Данные не прошли валидацию: {validation_message}"

      logger.info(f"Данные валидированы: {validation_message}")

      # 4. Предобрабатываем данные
      logger.info("Предобработка признаков...")
      processed_features, processed_labels = self._preprocess_features(features, labels)

      # 5. Обучаем модель
      logger.info("Обучение модели...")
      self.current_model_version = self._generate_model_version()

      training_success, metrics = self._train_model(processed_features, processed_labels)

      if not training_success or metrics is None:
        return False, "Не удалось обучить модель"

      # 6. Проверяем, стоит ли принимать новую модель
      should_accept, accept_reason = self._should_accept_new_model(metrics)

      if not should_accept:
        logger.info(f"Новая модель отклонена: {accept_reason}")
        return False, f"Модель отклонена: {accept_reason}"

      # 7. Сохраняем модель и метрики
      logger.info(f"Принимаем новую модель: {accept_reason}")
      self._save_model(metrics)

      # 8. Анализируем важность признаков
      self._analyze_and_save_feature_importance(processed_features, processed_labels)

      # 9. Обновляем время последнего переобучения
      self.last_retrain_time = datetime.now()

      # 10. Вызываем callback, если установлен
      if self.on_retrain_complete:
        try:
          if asyncio.iscoroutinefunction(self.on_retrain_complete):
            await self.on_retrain_complete(metrics)
          else:
            self.on_retrain_complete(metrics)
        except Exception as callback_error:
          logger.error(f"Ошибка callback on_retrain_complete: {callback_error}")

        success_message = (
          f"Модель успешно переобучена! "
          f"Версия: {self.current_model_version}, "
          f"Точность: {metrics.accuracy:.3f}, "
          f"F1: {metrics.f1_score:.3f}, "
          f"Время обучения: {metrics.training_time_seconds:.1f}с"
        )

        logger.info(success_message)
        return True, success_message

    except Exception as e:
        logger.error(f"Критическая ошибка переобучения модели: {e}", exc_info=True)
        return False, f"Критическая ошибка: {e}"

  async def schedule_retraining(self, symbols: List[str],
                                  timeframe: str = '1h',
                                  limit: int = 1000):
    """
    Планирует периодическое переобучение модели

      Args:
          symbols: Список символов для обучения
          timeframe: Таймфрейм данных
          limit: Количество свечей
    """
    try:
        logger.info(f"Запуск планировщика переобучения с интервалом {self.config.retraining_interval_hours} часов")

        while self.is_running:
          try:
            # Переобучаем модель
            success, message = await self.retrain_model(symbols, timeframe, limit)

            if success:
              logger.info(f"Плановое переобучение успешно: {message}")
            else:
              logger.warning(f"Плановое переобучение неудачно: {message}")

            # Ожидаем следующий интервал
            await asyncio.sleep(self.config.retraining_interval_hours * 3600)

          except asyncio.CancelledError:
            logger.info("Планировщик переобучения отменен")
            break
          except Exception as e:
            logger.error(f"Ошибка в планировщике переобучения: {e}", exc_info=True)
            # Ждем меньше времени при ошибке
            await asyncio.sleep(300)  # 5 минут

    except Exception as e:
        logger.error(f"Критическая ошибка планировщика переобучения: {e}", exc_info=True)

  def start_scheduled_retraining(self, symbols: List[str],
                                   timeframe: str = '1h',
                                   limit: int = 1000) -> asyncio.Task:
      """
      Запускает планировщик переобучения в фоне

      Returns:
          Task объект для управления планировщиком
      """
      if self.is_running:
        logger.warning("Планировщик переобучения уже запущен")
        return None

      self.is_running = True
      task = asyncio.create_task(self.schedule_retraining(symbols, timeframe, limit))
      logger.info("Планировщик переобучения запущен в фоне")
      return task

  async def stop_scheduled_retraining(self):
      """Останавливает планировщик переобучения"""
      self.is_running = False
      logger.info("Планировщик переобучения остановлен")

  def get_performance_history(self, last_n: Optional[int] = None) -> List[ModelPerformanceMetrics]:
      """
      Возвращает историю производительности моделей

      Args:
          last_n: Количество последних записей (None для всех)

      Returns:
          Список метрик производительности
      """
      with self.performance_lock:
        if last_n is None:
          return self.performance_history.copy()
        return self.performance_history[-last_n:] if self.performance_history else []

  def get_current_performance(self) -> Optional[ModelPerformanceMetrics]:
      """Возвращает текущие метрики производительности"""
      with self.performance_lock:
        return self.performance_history[-1] if self.performance_history else None

  def get_performance_trend(self, metric: str = 'accuracy', periods: int = 5) -> Dict[str, float]:
      """
      Анализирует тренд производительности модели

      Args:
          metric: Метрика для анализа ('accuracy', 'f1_score', 'precision', 'recall')
          periods: Количество периодов для анализа

      Returns:
          Словарь с трендовой информацией
      """
      try:
        with self.performance_lock:
          if len(self.performance_history) < 2:
            return {'trend': 'insufficient_data', 'change': 0.0, 'periods_analyzed': 0}

          recent_metrics = self.performance_history[-periods:] if len(
            self.performance_history) >= periods else self.performance_history

          values = [getattr(m, metric, 0) for m in recent_metrics]

          if len(values) < 2:
            return {'trend': 'insufficient_data', 'change': 0.0, 'periods_analyzed': len(values)}

          # Простой линейный тренд
          x = np.arange(len(values))
          z = np.polyfit(x, values, 1)
          slope = z[0]

          # Определяем направление тренда
          if abs(slope) < 0.001:
            trend = 'stable'
          elif slope > 0:
            trend = 'improving'
          else:
            trend = 'declining'

          change = values[-1] - values[0]

          return {
            'trend': trend,
            'slope': slope,
            'change': change,
            'current_value': values[-1],
            'periods_analyzed': len(values),
            'values': values
          }

      except Exception as e:
        logger.error(f"Ошибка анализа тренда производительности: {e}")
        return {'trend': 'error', 'change': 0.0, 'periods_analyzed': 0}

  def should_trigger_emergency_retraining(self) -> Tuple[bool, str]:
      """
      Определяет, нужно ли экстренное переобучение на основе производительности

      Returns:
          Tuple[should_retrain, reason]
      """
      try:
        current_metrics = self.get_current_performance()
        if not current_metrics:
          return False, "Нет данных о производительности"

        # Проверяем критическое снижение производительности
        if current_metrics.accuracy < self.config.performance_threshold * 0.8:
          return True, f"Критически низкая точность: {current_metrics.accuracy:.3f}"

        # Анализируем тренд
        trend_info = self.get_performance_trend('accuracy', 3)

        if trend_info['trend'] == 'declining' and abs(trend_info['change']) > 0.05:
          return True, f"Значительное снижение производительности: {trend_info['change']:.3f}"

        # Проверяем кросс-валидацию
        if current_metrics.cross_val_mean > 0 and current_metrics.cross_val_mean < self.config.performance_threshold * 0.9:
          return True, f"Низкая кросс-валидация: {current_metrics.cross_val_mean:.3f}"

        return False, "Производительность в норме"

      except Exception as e:
        logger.error(f"Ошибка проверки необходимости экстренного переобучения: {e}")
        return False, f"Ошибка проверки: {e}"

  async def check_and_trigger_emergency_retraining(self, symbols: List[str],
                                                     timeframe: str = '1h',
                                                     limit: int = 1000) -> Tuple[bool, str]:
      """
      Проверяет и при необходимости запускает экстренное переобучение

      Returns:
          Tuple[was_triggered, message]
      """
      should_retrain, reason = self.should_trigger_emergency_retraining()

      if not should_retrain:
        return False, reason

      logger.warning(f"Запуск экстренного переобучения: {reason}")

      # Уведомляем callback о снижении производительности
      if self.on_performance_decline:
        try:
          if asyncio.iscoroutinefunction(self.on_performance_decline):
            await self.on_performance_decline(reason)
          else:
            self.on_performance_decline(reason)
        except Exception as callback_error:
          logger.error(f"Ошибка callback on_performance_decline: {callback_error}")

      # Запускаем экстренное переобучение
      success, message = await self.retrain_model(symbols, timeframe, limit, force_retrain=True)

      return success, f"Экстренное переобучение: {message}"

  def export_performance_report(self, filepath: Optional[str] = None) -> str:
      """
      Экспортирует отчет о производительности в JSON

      Args:
          filepath: Путь для сохранения (опционально)

      Returns:
          Путь к сохраненному файлу
      """
      try:
        if filepath is None:
          timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
          filepath = self.model_save_path / "performance_logs" / f"performance_report_{timestamp}.json"

        report = {
          'generation_time': datetime.now().isoformat(),
          'total_models_trained': len(self.performance_history),
          'current_model_version': self.current_model_version,
          'config': asdict(self.config),
          'performance_history': [m.to_dict() for m in self.performance_history],
          'feature_importance_history': self.feature_importance_history,
          'trends': {
            'accuracy': self.get_performance_trend('accuracy'),
            'f1_score': self.get_performance_trend('f1_score'),
            'precision': self.get_performance_trend('precision'),
            'recall': self.get_performance_trend('recall')
          }
        }

        with open(filepath, 'w') as f:
          json.dump(report, f, indent=2)

        logger.info(f"Отчет о производительности экспортирован: {filepath}")
        return str(filepath)

      except Exception as e:
        logger.error(f"Ошибка экспорта отчета о производительности: {e}")
        return ""

  def cleanup_old_files(self, days_to_keep: int = 30):
      """
      Очищает старые файлы моделей и логов

      Args:
          days_to_keep: Количество дней для хранения файлов
      """
      try:
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_timestamp = cutoff_date.timestamp()

        # Очищаем старые backup'ы
        backup_dir = self.model_save_path / "backups"
        if backup_dir.exists():
          for backup_file in backup_dir.glob("model_backup_*.pkl"):
            if backup_file.stat().st_mtime < cutoff_timestamp:
              backup_file.unlink()
              logger.info(f"Удален старый backup: {backup_file.name}")

        # Очищаем старые логи производительности
        logs_dir = self.model_save_path / "performance_logs"
        if logs_dir.exists():
          for log_file in logs_dir.glob("metrics_*.json"):
            if log_file.stat().st_mtime < cutoff_timestamp:
              log_file.unlink()
              logger.info(f"Удален старый лог: {log_file.name}")

          for report_file in logs_dir.glob("performance_report_*.json"):
            if report_file.stat().st_mtime < cutoff_timestamp:
              report_file.unlink()
              logger.info(f"Удален старый отчет: {report_file.name}")

        # Очищаем старые файлы важности признаков
        importance_dir = self.model_save_path / "feature_importance"
        if importance_dir.exists():
          for importance_file in importance_dir.glob("importance_*.csv"):
            if importance_file.stat().st_mtime < cutoff_timestamp:
              importance_file.unlink()
              logger.info(f"Удален старый файл важности: {importance_file.name}")

        logger.info(f"Очистка старых файлов завершена (старше {days_to_keep} дней)")

      except Exception as e:
        logger.error(f"Ошибка очистки старых файлов: {e}")

  def get_model_info(self) -> Dict[str, Any]:
      """
      Возвращает информацию о текущем состоянии модели

      Returns:
          Словарь с информацией о модели
      """
      try:
        current_metrics = self.get_current_performance()
        trend_info = self.get_performance_trend('accuracy')

        info = {
          'current_version': self.current_model_version,
          'is_running': self.is_running,
          'last_retrain_time': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
          'models_trained_count': len(self.performance_history),
          'current_performance': current_metrics.to_dict() if current_metrics else None,
          'performance_trend': trend_info,
          'next_scheduled_retrain': None,
          'emergency_retrain_needed': self.should_trigger_emergency_retraining()[0],
          'config': asdict(self.config)
        }

        # Рассчитываем время следующего планового переобучения
        if self.last_retrain_time and self.is_running:
          next_retrain = self.last_retrain_time + timedelta(hours=self.config.retraining_interval_hours)
          info['next_scheduled_retrain'] = next_retrain.isoformat()

        return info

      except Exception as e:
        logger.error(f"Ошибка получения информации о модели: {e}")
        return {'error': str(e)}

  async def __aenter__(self):
      """Async context manager вход"""
      return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
      """Async context manager выход"""
      await self.stop_scheduled_retraining()
      if hasattr(self, 'thread_pool'):
        self.thread_pool.shutdown(wait=True)

    # Вспомогательные функции для интеграции

async def create_retraining_manager_from_config(config_path: str,
                                                    data_fetcher,
                                                    ml_model,
                                                    db_manager=None) -> ModelRetrainingManager:
      """
      Создает менеджер переобучения из конфигурационного файла

      Args:
          config_path: Путь к JSON файлу с конфигурацией
          data_fetcher: Объект для получения данных
          ml_model: ML модель
          db_manager: Менеджер базы данных (опционально)

      Returns:
          Настроенный ModelRetrainingManager
      """
      try:
        with open(config_path, 'r') as f:
          config_data = json.load(f)

        # Создаем конфигурацию из словаря
        config = RetrainingConfig(**config_data.get('retraining', {}))

        # Создаем менеджер
        manager = ModelRetrainingManager(
          model_save_path=config_data.get('model_save_path', 'ml_models/'),
          config=config,
          data_fetcher=data_fetcher,
          ml_model=ml_model,
          db_manager=db_manager
        )

        logger.info(f"Менеджер переобучения создан из конфигурации: {config_path}")
        return manager

      except Exception as e:
        logger.error(f"Ошибка создания менеджера из конфигурации: {e}")
        raise

async def setup_retraining_callbacks(manager: ModelRetrainingManager,
                                         telegram_bot=None,
                                         email_notifier=None,
                                         webhook_url: Optional[str] = None):
      """
      Настраивает callback'и для уведомлений о переобучении

      Args:
          manager: Менеджер переобучения
          telegram_bot: Telegram бот для уведомлений
          email_notifier: Email уведомитель
          webhook_url: URL для webhook уведомлений
      """

async def on_retrain_complete(metrics: ModelPerformanceMetrics, webhook_url=None, email_notifier=None):
        """Callback при завершении переобучения"""
        message = (
          f"🤖 Модель переобучена!\n"
          f"📊 Версия: {metrics.model_version}\n"
          f"🎯 Точность: {metrics.accuracy:.3f}\n"
          f"📈 F1-мера: {metrics.f1_score:.3f}\n"
          f"⏱ Время: {metrics.training_time_seconds:.1f}с"
        )

        # Отправляем уведомления
        if telegram_bot:
          try:
            await telegram_bot.send_message(message)
          except Exception as e:
            logger.error(f"Ошибка отправки Telegram уведомления: {e}")

        if email_notifier:
          try:
            await email_notifier.send_email(
              subject="Модель переобучена",
              message=message
            )
          except Exception as e:
            logger.error(f"Ошибка отправки email уведомления: {e}")

        if webhook_url:
          try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
              await session.post(webhook_url, json={
                'event': 'model_retrained',
                'metrics': metrics.to_dict(),
                'message': message
              })
          except Exception as e:
            logger.error(f"Ошибка отправки webhook: {e}")

async def on_performance_decline(reason: str):
    """Callback при снижении производительности"""
    message = f"⚠️ Снижение производительности модели: {reason}"

    # Отправляем предупреждения
    if telegram_bot:
      try:
        await telegram_bot.send_message(message)
      except Exception as e:
        logger.error(f"Ошибка отправки Telegram предупреждения: {e}")

    # Устанавливаем callback'и
    manager.on_retrain_complete = on_retrain_complete
    manager.on_performance_decline = on_performance_decline

    logger.info("Callback'и для уведомлений настроены")

  # Пример использования в main.py

async def integrate_retraining_with_main_bot(main_bot_instance,
                                                 data_fetcher,
                                                 ml_model,
                                                 config_path: str = "config/retraining_config.json"):
      """
      Интегрирует систему переобучения с основным ботом

      Args:
          main_bot_instance: Экземпляр основного торгового бота
          data_fetcher: Объект для получения данных
          ml_model: ML модель
          config_path: Путь к конфигурации переобучения
      """
      try:
        # Создаем менеджер переобучения
        retraining_manager = await create_retraining_manager_from_config(
          config_path, data_fetcher, ml_model
        )

        # Настраиваем уведомления (если доступны в боте)
        if hasattr(main_bot_instance, 'telegram_bot'):
          await setup_retraining_callbacks(
            retraining_manager,
            telegram_bot=main_bot_instance.telegram_bot
          )

        # Добавляем методы для взаимодействия с переобучением в основной бот
        main_bot_instance.retraining_manager = retraining_manager

        # Добавляем метод для проверки необходимости экстренного переобучения
        async def check_model_performance():
          symbols = main_bot_instance.monitoring_service.get_active_symbols()
          if symbols:
            triggered, message = await retraining_manager.check_and_trigger_emergency_retraining(symbols)
            if triggered:
              logger.info(f"Экстренное переобучение выполнено: {message}")

        # Интегрируем проверку в основной цикл бота
        main_bot_instance.check_model_performance = check_model_performance

        # Запускаем планировщик переобучения
        symbols = main_bot_instance.monitoring_service.get_active_symbols()
        if symbols:
          retraining_task = retraining_manager.start_scheduled_retraining(symbols)
          main_bot_instance.retraining_task = retraining_task

        logger.info("Система переобучения успешно интегрирована с основным ботом")
        return retraining_manager

      except Exception as e:
        logger.error(f"Ошибка интеграции системы переобучения: {e}", exc_info=True)
        raise


class MLFeedbackLoop:
  """Цикл обратной связи для улучшения ML модели"""

  def __init__(self, db_manager: AdvancedDatabaseManager, ml_strategy: EnsembleMLStrategy):
    self.db_manager = db_manager
    self.ml_strategy = ml_strategy

  async def update_model_with_results(self, symbol: str):
    """Обновляем модель на основе результатов сделок"""
    try:
      # Получаем результаты недавних сделок
      cursor = self.db_manager.conn.cursor()
      today = datetime.datetime.now() - datetime.timedelta(days=7)

      cursor.execute("""
              SELECT profit_loss, confidence, metadata 
              FROM trades 
              WHERE symbol = ? AND status = 'CLOSED' 
              AND close_timestamp >= ?
          """, (symbol, today))

      results = cursor.fetchall()
      if results:
        # Преобразуем в список словарей для удобства
        results_dicts = []
        for row in results:
          results_dicts.append({
            'profit_loss': row[0],
            'confidence': row[1],
            'metadata': json.loads(row[2]) if row[2] else {}
          })

        # Обновляем модель
        await self.ml_strategy.retrain_with_feedback(symbol, results_dicts)
      else:
        print(f"ℹ️ Нет данных для обновления модели {symbol}")

    except Exception as e:
      print(f"❌ Ошибка получения данных для обратной связи: {e}")



if __name__ == "__main__":
      # Пример использования для тестирования
      async def test_retraining_manager():
        import sys
        sys.path.append('.')


        setup_logging("INFO")

        # Создаем тестовые данные
        test_data = pd.DataFrame({
          'close': np.random.randn(1000).cumsum() + 100,
          'volume': np.random.randn(1000) * 1000 + 10000,
          'high': np.random.randn(1000) + 101,
          'low': np.random.randn(1000) + 99,
          'open': np.random.randn(1000) + 100
        })

        # Мок объекты
        class MockDataFetcher:
          async def get_historical_data(self, symbol, timeframe, limit):
            return test_data.head(limit)

        class MockMLModel:
          def __init__(self):
            self.is_fitted = False

          def fit(self, X, y):
            self.is_fitted = True

          def predict(self, X):
            return np.random.choice([0, 1], size=len(X))

        # Создаем менеджер
        config = RetrainingConfig(
          min_data_points=100,
          retraining_interval_hours=0.01,  # 36 секунд для теста
          performance_threshold=0.4
        )

        data_fetcher = MockDataFetcher()
        ml_model = MockMLModel()

        manager = ModelRetrainingManager(
          model_save_path="test_models/",
          config=config,
          data_fetcher=data_fetcher,
          ml_model=ml_model
        )

        # Тестируем переобучение
        success, message = await manager.retrain_model(['BTCUSDT'], '1h', 500, force_retrain=True)
        print(f"Результат переобучения: {success}, {message}")

        # Проверяем информацию о модели
        info = manager.get_model_info()
        print(f"Информация о модели: {json.dumps(info, indent=2, default=str)}")

        # Экспортируем отчет
        report_path = manager.export_performance_report()
        print(f"Отчет сохранен: {report_path}")

        # Очищаем тестовые файлы
        manager.cleanup_old_files(0)

# Запускаем тест
asyncio.run(test_retraining_manager())