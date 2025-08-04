import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional, Dict, List, Tuple
import joblib
from utils.logging_config import get_logger
import numba

logger = get_logger(__name__)


@numba.jit(nopython=True)
def lorentzian_distance_fast(x1: np.ndarray, x2: np.ndarray) -> float:
  """Быстрый расчет Lorentzian distance"""
  return np.sum(np.log1p(np.abs(x1 - x2)))


class EnhancedLorentzianClassifier(BaseEstimator, ClassifierMixin):
  """
  Улучшенная реализация Lorentzian Classifier с алгоритмом из оригинального индикатора
  """

  def __init__(self,
               k_neighbors: int = 8,
               max_bars_back: int = 2000,
               feature_count: int = 5,
               use_dynamic_exits: bool = True,
               filters: Optional[Dict] = None):
    """
    Args:
        k_neighbors: Количество ближайших соседей
        max_bars_back: Максимальное количество исторических баров
        feature_count: Количество признаков (2-5)
        use_dynamic_exits: Использовать динамические выходы
        filters: Настройки фильтров
    """
    self.k_neighbors = k_neighbors
    self.max_bars_back = max_bars_back
    self.feature_count = feature_count
    self.use_dynamic_exits = use_dynamic_exits

    # Настройки фильтров по умолчанию
    self.filters = filters or {
      'use_volatility_filter': True,
      'use_regime_filter': True,
      'use_adx_filter': False,
      'regime_threshold': -0.1,
      'adx_threshold': 20
    }

    # Внутренние переменные
    self.is_fitted = False
    self.training_data = []
    self.training_labels = []
    self.predictions_array = []
    self.distances_array = []

  # def fit(self, X: pd.DataFrame, y: pd.Series):
  #   """
  #   Обучение модели с использованием алгоритма ANN из оригинала
  #   """
  #   logger.info(f"Обучение Enhanced Lorentzian Classifier на {len(X)} примерах")
  #
  #   # Проверяем количество признаков
  #   if X.shape[1] != self.feature_count:
  #     raise ValueError(f"Ожидалось {self.feature_count} признаков, получено {X.shape[1]}")
  #
  #   # Ограничиваем размер обучающей выборки
  #   if len(X) > self.max_bars_back:
  #     X = X.tail(self.max_bars_back)
  #     y = y.tail(self.max_bars_back)
  #
  #   self.training_data = X.values
  #   self.training_labels = y.values
  #   self.feature_names = X.columns.tolist()
  #
  #   self.is_fitted = True
  #   logger.info("Модель успешно обучена")
  #
  #   return self

  def fit(self, X: pd.DataFrame, y: pd.Series):
    """
    Обучение модели с использованием алгоритма ANN из оригинала
    """
    # Проверяем входные данные
    if X.empty or y.empty:
      raise ValueError("Входные данные не могут быть пустыми")

    if len(X) != len(y):
      raise ValueError(f"Размерности не совпадают: X={len(X)}, y={len(y)}")

    logger.info(f"Обучение Enhanced Lorentzian Classifier на {len(X)} примерах")

    # Проверяем количество признаков
    if X.shape[1] != self.feature_count:
      logger.warning(f"Ожидалось {self.feature_count} признаков, получено {X.shape[1]}. Корректируем...")
      self.feature_count = X.shape[1]

    # Ограничиваем размер обучающей выборки
    if len(X) > self.max_bars_back:
      logger.info(f"Обрезаем данные с {len(X)} до {self.max_bars_back} записей")
      X = X.tail(self.max_bars_back)
      y = y.tail(self.max_bars_back)

    # Проверяем на NaN и бесконечные значения
    if X.isnull().any().any():
      logger.warning("Обнаружены NaN в обучающих данных, заполняем медианными значениями")
      X = X.fillna(X.median()).fillna(0)

    if not np.isfinite(X.values).all():
      logger.warning("Обнаружены бесконечные значения, заменяем нулями")
      X = X.replace([np.inf, -np.inf], 0)

    self.training_data = X.values
    self.training_labels = y.values
    self.feature_names = X.columns.tolist()

    self.is_fitted = True
    logger.info("✅ Модель успешно обучена")

    return self

  def incremental_update(self, new_X: np.ndarray, new_y: int, symbol: str = None):
      """
      Инкрементальное обновление модели новыми данными
      Эмулирует поведение TradingView индикатора
      """
      if not self.is_fitted:
        raise ValueError("Модель должна быть обучена перед инкрементальным обновлением")

      # Добавляем новые данные в обучающий набор
      self.training_data = np.vstack([self.training_data, new_X])
      self.training_labels = np.append(self.training_labels, new_y)

      # Ограничиваем размер по max_bars_back
      if len(self.training_data) > self.max_bars_back:
        # Удаляем старые данные (FIFO)
        remove_count = len(self.training_data) - self.max_bars_back
        self.training_data = self.training_data[remove_count:]
        self.training_labels = self.training_labels[remove_count:]

      # Обновляем статистику для быстрого поиска
      self._update_search_statistics()

      return True

  def _update_search_statistics(self):
    """Обновляет внутренние структуры для ускорения поиска"""
    # Предвычисляем квартили для оптимизации поиска
    if hasattr(self, '_distance_cache'):
      self._distance_cache.clear()

    # Обновляем индексы для быстрого доступа
    self._sample_indices = np.arange(0, len(self.training_data), 4)

    logger.debug(f"Обновлены структуры поиска: {len(self.training_data)} образцов, "
                 f"{len(self._sample_indices)} индексов")

  def predict(self, X: pd.DataFrame) -> np.ndarray:
    """
    Предсказание с использованием Approximate Nearest Neighbors алгоритма
    """
    if not self.is_fitted:
      raise ValueError("Модель не обучена. Вызовите fit() перед predict()")

    predictions = []

    for idx in range(len(X)):
      # Текущая точка для предсказания
      current_point = X.iloc[idx].values

      # Применяем алгоритм ANN из оригинала
      prediction = self._ann_predict(current_point)
      predictions.append(prediction)

    return np.array(predictions)

  def _ann_predict(self, x_new: np.ndarray) -> int:
    """
    Approximate Nearest Neighbors предсказание с алгоритмом из оригинала
    """
    distances = []
    predictions = []
    last_distance = -1.0

    # Итерация через обучающие данные с шагом 4 (как в оригинале)
    for i in range(0, len(self.training_data), 4):
      # Вычисляем Lorentzian distance
      d = lorentzian_distance_fast(x_new, self.training_data[i])

      # Условие из оригинала: d >= lastDistance
      if d >= last_distance:
        last_distance = d
        distances.append(d)
        predictions.append(self.training_labels[i])

        # Ограничиваем размер массива k соседями
        if len(predictions) > self.k_neighbors:
          # Обновляем lastDistance как в оригинале (нижние 75%)
          quartile_idx = int(self.k_neighbors * 3 / 4)
          sorted_distances = sorted(distances)
          last_distance = sorted_distances[quartile_idx] if quartile_idx < len(sorted_distances) else last_distance

          # Удаляем старые значения
          distances.pop(0)
          predictions.pop(0)

    # Суммируем голоса для предсказания
    if predictions:
      # Подсчет голосов для каждого класса
      unique_classes = list(set(predictions))
      class_votes = {cls: predictions.count(cls) for cls in unique_classes}

      # Возвращаем класс с максимальным количеством голосов
      prediction = max(class_votes, key=class_votes.get)

      # В оригинале: prediction > 0 => LONG, prediction < 0 => SHORT
      # Преобразуем в наши метки: 0=HOLD, 1=BUY, 2=SELL
      if prediction > 0:
        return 1  # BUY
      elif prediction < 0:
        return 2  # SELL
      else:
        return 0  # HOLD

    return 0  # По умолчанию HOLD

  def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
    """
    Вероятности предсказаний на основе взвешенного голосования
    """
    if not self.is_fitted:
      raise ValueError("Модель не обучена")

    probabilities = []

    for idx in range(len(X)):
      current_point = X.iloc[idx].values
      proba = self._ann_predict_proba(current_point)
      probabilities.append(proba)

    return np.array(probabilities)

  def _ann_predict_proba(self, x_new: np.ndarray) -> np.ndarray:
    """
    Вычисление вероятностей с использованием взвешенного голосования
    """
    distances = []
    predictions = []
    last_distance = -1.0

    # Собираем k ближайших соседей
    for i in range(0, len(self.training_data), 4):
      d = lorentzian_distance_fast(x_new, self.training_data[i])

      if d >= last_distance:
        last_distance = d
        distances.append(d)
        predictions.append(self.training_labels[i])

        if len(predictions) > self.k_neighbors:
          quartile_idx = int(self.k_neighbors * 3 / 4)
          sorted_distances = sorted(distances)
          last_distance = sorted_distances[quartile_idx] if quartile_idx < len(sorted_distances) else last_distance
          distances.pop(0)
          predictions.pop(0)

    # Взвешенное голосование
    class_weights = {0: 0.0, 1: 0.0, 2: 0.0}  # HOLD, BUY, SELL
    total_weight = 0.0

    for i, pred in enumerate(predictions):
      # Вес обратно пропорционален расстоянию
      weight = 1.0 / (1.0 + distances[i])

      # Преобразование из оригинальных меток
      if pred > 0:
        class_weights[1] += weight  # BUY
      elif pred < 0:
        class_weights[2] += weight  # SELL
      else:
        class_weights[0] += weight  # HOLD

      total_weight += weight

    # Нормализация вероятностей
    if total_weight > 0:
      probabilities = [
        class_weights[0] / total_weight,  # HOLD
        class_weights[1] / total_weight,  # BUY
        class_weights[2] / total_weight  # SELL
      ]
    else:
      probabilities = [1.0, 0.0, 0.0]  # По умолчанию HOLD

    return np.array(probabilities)

  def apply_filters(self, data: pd.DataFrame, predictions: np.ndarray,
                    filters_data: Dict[str, pd.Series]) -> np.ndarray:
    """
    Применение фильтров к предсказаниям

    Args:
        data: Исходные данные
        predictions: Предсказания модели
        filters_data: Словарь с результатами фильтров

    Returns:
        Отфильтрованные предсказания
    """
    filtered_predictions = predictions.copy()

    # Получаем результаты всех фильтров
    volatility_pass = filters_data.get('volatility', pd.Series(True, index=data.index))
    regime_pass = filters_data.get('regime', pd.Series(True, index=data.index))
    adx_pass = filters_data.get('adx', pd.Series(True, index=data.index))

    # Комбинированный фильтр
    filter_mask = volatility_pass & regime_pass & adx_pass

    # Применяем фильтр: где False, ставим HOLD (0)
    filtered_predictions[~filter_mask.values[:len(predictions)]] = 0

    return filtered_predictions

  def save_model(self, filepath: str):
    """Сохранение модели"""
    if not self.is_fitted:
      raise ValueError("Нельзя сохранить необученную модель")

    model_data = {
      'training_data': self.training_data,
      'training_labels': self.training_labels,
      'feature_names': self.feature_names,
      'k_neighbors': self.k_neighbors,
      'max_bars_back': self.max_bars_back,
      'feature_count': self.feature_count,
      'filters': self.filters,
      'is_fitted': self.is_fitted
    }

    joblib.dump(model_data, filepath)
    logger.info(f"Модель сохранена в {filepath}")

  def load_model(self, filepath: str):
    """Загрузка модели"""
    model_data = joblib.load(filepath)

    self.training_data = model_data['training_data']
    self.training_labels = model_data['training_labels']
    self.feature_names = model_data['feature_names']
    self.k_neighbors = model_data['k_neighbors']
    self.max_bars_back = model_data['max_bars_back']
    self.feature_count = model_data['feature_count']
    self.filters = model_data['filters']
    self.is_fitted = model_data['is_fitted']

    logger.info(f"Модель загружена из {filepath}")


def create_lorentzian_labels(df: pd.DataFrame, future_bars: int = 4,
                             threshold_percent: float = 0.0) -> pd.Series:
  """
  Создание меток как в оригинальном индикаторе

  В оригинале:
  - Смотрим на 4 бара вперед
  - Если цена через 4 бара выше текущей => BUY (1)
  - Если цена через 4 бара ниже текущей => SELL (-1)
  - Иначе => NEUTRAL (0)
  """
  labels = []

  for i in range(len(df)):
    if i + future_bars >= len(df):
      labels.append(0)  # NEUTRAL для последних баров
      continue

    current_price = df['close'].iloc[i]
    future_price = df['close'].iloc[i + future_bars]

    price_change = (future_price - current_price) / current_price * 100

    if price_change > threshold_percent:
      labels.append(1)  # Оригинальный LONG => наш BUY
    elif price_change < -threshold_percent:
      labels.append(-1)  # Оригинальный SHORT => наш SELL
    else:
      labels.append(0)  # NEUTRAL

  return pd.Series(labels, index=df.index)