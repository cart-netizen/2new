from pathlib import Path

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
               max_bars_back: int = 5000,
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
      'k_neighbors': self.k_neighbors,
      'max_bars_back': self.max_bars_back,
      'feature_count': self.feature_count,
      'use_dynamic_exits': self.use_dynamic_exits,
      'filters': self.filters,
      'is_fitted': self.is_fitted,
      'predictions_array': getattr(self, 'predictions_array', []),
      'distances_array': getattr(self, 'distances_array', [])
    }

    with open(filepath, 'wb') as f:
      joblib.dump(model_data, f)

    logger.info(f"Enhanced Lorentzian модель сохранена: {filepath}")
    return True

  def load_model(self, filepath: str):
    """Загрузка модели"""
    if not Path(filepath).exists():
      logger.error(f"Файл модели не найден: {filepath}")
      return False

    with open(filepath, 'rb') as f:
      model_data = joblib.load(f)

      # Восстанавливаем состояние модели
    self.training_data = model_data['training_data']
    self.training_labels = model_data['training_labels']
    self.k_neighbors = model_data['k_neighbors']
    self.max_bars_back = model_data['max_bars_back']
    self.feature_count = model_data['feature_count']
    self.use_dynamic_exits = model_data['use_dynamic_exits']
    self.filters = model_data['filters']
    self.is_fitted = model_data['is_fitted']
    self.predictions_array = model_data.get('predictions_array', [])
    self.distances_array = model_data.get('distances_array', [])

    logger.info(f"Enhanced Lorentzian модель загружена: {filepath}")
    return True


# def create_lorentzian_labels(df: pd.DataFrame, future_bars: int = 4,
#                              threshold_percent: float = 0.0) -> pd.Series:
#   """
#   Создание меток как в оригинальном индикаторе
#
#   В оригинале:
#   - Смотрим на 4 бара вперед
#   - Если цена через 4 бара выше текущей => BUY (1)
#   - Если цена через 4 бара ниже текущей => SELL (-1)
#   - Иначе => NEUTRAL (0)
#   """
#   labels = []
#
#   for i in range(len(df)):
#     if i + future_bars >= len(df):
#       labels.append(0)  # NEUTRAL для последних баров
#       continue
#
#     current_price = df['close'].iloc[i]
#     future_price = df['close'].iloc[i + future_bars]
#
#     price_change = (future_price - current_price) / current_price * 100
#
#     if price_change > threshold_percent:
#       labels.append(1)  # Оригинальный LONG => наш BUY
#     elif price_change < -threshold_percent:
#       labels.append(-1)  # Оригинальный SHORT => наш SELL
#     else:
#       labels.append(0)  # NEUTRAL
#
#   return pd.Series(labels, index=df.index)

def create_lorentzian_labels(data: pd.DataFrame, future_bars: int = 4, threshold_percent: float = 0.85) -> pd.Series:
    """
    Создает метки для обучения Lorentzian классификатора (улучшенная версия)

    Args:
        data: DataFrame с OHLCV данными
        future_bars: Количество баров вперед для анализа (как в оригинале TradingView)
        threshold_percent: Процентный порог для определения сигналов (0.85% как в оригинале)

    Returns:
        Series с метками: 0=HOLD, 1=BUY, 2=SELL
    """
    if len(data) < future_bars + 1:
      return pd.Series(index=data.index, dtype=int)

    labels = pd.Series(0, index=data.index, dtype=int)  # По умолчанию HOLD

    # Рассчитываем ATR для адаптивного порога
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift(1))
    low_close = np.abs(data['low'] - data['close'].shift(1))
    atr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

    for i in range(len(data) - future_bars):
      current_close = data['close'].iloc[i]

      # Анализируем будущие цены
      future_highs = data['high'].iloc[i + 1:i + future_bars + 1]
      future_lows = data['low'].iloc[i + 1:i + future_bars + 1]

      if len(future_highs) == 0 or len(future_lows) == 0:
        continue

      # Используем ATR для адаптивного порога
      current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else current_close * 0.01
      adaptive_threshold = max(current_close * (threshold_percent / 100), current_atr * 0.5)

      max_future_high = future_highs.max()
      min_future_low = future_lows.min()

      # Потенциальная прибыль для BUY
      buy_profit = max_future_high - current_close
      # Потенциальная прибыль для SELL
      sell_profit = current_close - min_future_low

      # Определяем метку на основе максимальной потенциальной прибыли
      if buy_profit > adaptive_threshold and buy_profit > sell_profit:
        labels.iloc[i] = 1  # BUY
      elif sell_profit > adaptive_threshold and sell_profit > buy_profit:
        labels.iloc[i] = 2  # SELL
      # Иначе остается 0 (HOLD)

    # Логируем статистику меток
    buy_count = (labels == 1).sum()
    sell_count = (labels == 2).sum()
    hold_count = (labels == 0).sum()
    total = len(labels)

    logger.info(f"Метки созданы: BUY={buy_count} ({buy_count / total * 100:.1f}%), "
                f"SELL={sell_count} ({sell_count / total * 100:.1f}%), "
                f"HOLD={hold_count} ({hold_count / total * 100:.1f}%)")

    return labels


def create_balanced_lorentzian_labels(data: pd.DataFrame,
                                      future_bars: int = 4,
                                      threshold_percent: float = 0.85,
                                      volatility_adaptive: bool = True,
                                      min_signal_ratio: float = 0.15) -> pd.Series:
  """
  Создание сбалансированных меток для Lorentzian классификатора

  Args:
      data: DataFrame с OHLCV данными
      future_bars: Количество баров в будущее для анализа
      threshold_percent: Базовый порог для сигналов (в процентах)
      volatility_adaptive: Адаптировать пороги к волатильности
      min_signal_ratio: Минимальная доля торговых сигналов
  """
  import pandas as pd
  import numpy as np
  # from ta import add_all_ta_features

  logger.info(f"Создание меток с параметрами: future_bars={future_bars}, "
              f"threshold={threshold_percent}, adaptive={volatility_adaptive}")

  # Рассчитываем технические индикаторы для дополнительного анализа
  df_enhanced = data.copy()

  # Добавляем ATR для адаптивных порогов
  high_low = df_enhanced['high'] - df_enhanced['low']
  high_close = np.abs(df_enhanced['high'] - df_enhanced['close'].shift())
  low_close = np.abs(df_enhanced['low'] - df_enhanced['close'].shift())
  ranges = pd.concat([high_low, high_close, low_close], axis=1)
  true_range = np.max(ranges, axis=1)
  df_enhanced['atr'] = true_range.rolling(window=14).mean()

  # Добавляем RSI
  delta = df_enhanced['close'].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / loss
  df_enhanced['rsi'] = 100 - (100 / (1 + rs))

  # Добавляем объемные индикаторы
  df_enhanced['volume_sma'] = df_enhanced['volume'].rolling(window=20).mean()
  df_enhanced['volume_ratio'] = df_enhanced['volume'] / df_enhanced['volume_sma']

  labels = []

  for i in range(len(df_enhanced)):
    if i + future_bars >= len(df_enhanced):
      labels.append(0)  # HOLD для последних баров
      continue

    current_price = df_enhanced.iloc[i]['close']
    current_atr = df_enhanced.iloc[i]['atr'] if pd.notna(df_enhanced.iloc[i]['atr']) else current_price * 0.02
    current_rsi = df_enhanced.iloc[i]['rsi'] if pd.notna(df_enhanced.iloc[i]['rsi']) else 50
    volume_ratio = df_enhanced.iloc[i]['volume_ratio'] if pd.notna(df_enhanced.iloc[i]['volume_ratio']) else 1.0

    # Адаптивные пороги на основе волатильности
    if volatility_adaptive:
      # Базовый порог адаптируется к ATR
      volatility_factor = current_atr / current_price
      dynamic_threshold = max(threshold_percent / 100, volatility_factor * 1.5)
    else:
      dynamic_threshold = threshold_percent / 100

    # Анализ будущих цен
    future_slice = df_enhanced.iloc[i + 1:i + future_bars + 1]
    future_highs = future_slice['high']
    future_lows = future_slice['low']
    future_closes = future_slice['close']

    # Максимальная и минимальная цена в будущем
    max_future_high = future_highs.max()
    min_future_low = future_lows.min()
    final_close = future_closes.iloc[-1]

    # Потенциальная прибыль/убыток
    max_upside = (max_future_high - current_price) / current_price
    max_downside = (current_price - min_future_low) / current_price
    final_return = (final_close - current_price) / current_price

    # Улучшенная логика принятия решений

    # 1. Анализ силы движения
    strong_upside = max_upside > dynamic_threshold * 1.5
    strong_downside = max_downside > dynamic_threshold * 1.5

    # 2. Анализ направления тренда
    trend_up = final_return > dynamic_threshold * 0.5
    trend_down = final_return < -dynamic_threshold * 0.5

    # 3. Фильтры качества сигналов
    volume_confirmation = volume_ratio > 1.2  # Повышенный объем

    # RSI фильтры (избегаем перекупленности/перепроданности)
    rsi_not_overbought = current_rsi < 75
    rsi_not_oversold = current_rsi > 25

    # 4. Принятие решения с множественными критериями
    buy_conditions = [
      strong_upside,
      trend_up,
      max_upside > max_downside * 1.5,  # Соотношение риск/прибыль
      rsi_not_overbought,
      final_return > 0  # Финальный результат положительный
    ]

    sell_conditions = [
      strong_downside,
      trend_down,
      max_downside > max_upside * 1.5,  # Соотношение риск/прибыль
      rsi_not_oversold,
      final_return < 0  # Финальный результат отрицательный
    ]

    # Требуем выполнения минимум 3 из 5 условий
    buy_score = sum(buy_conditions)
    sell_score = sum(sell_conditions)

    if buy_score >= 3 and volume_confirmation:
      labels.append(1)  # BUY
    elif sell_score >= 3 and volume_confirmation:
      labels.append(2)  # SELL
    else:
      labels.append(0)  # HOLD

  # Постобработка: балансировка классов
  labels_series = pd.Series(labels, index=df_enhanced.index)

  # Подсчет текущего распределения
  class_counts = labels_series.value_counts()
  total_samples = len(labels_series)

  buy_ratio = class_counts.get(1, 0) / total_samples
  sell_ratio = class_counts.get(2, 0) / total_samples
  hold_ratio = class_counts.get(0, 0) / total_samples

  logger.info(f"Исходное распределение: BUY={buy_ratio:.3f}, SELL={sell_ratio:.3f}, HOLD={hold_ratio:.3f}")

  # Если торговых сигналов слишком мало, снижаем пороги
  total_trading_signals = buy_ratio + sell_ratio
  if total_trading_signals < min_signal_ratio:
    logger.warning(f"Мало торговых сигналов ({total_trading_signals:.3f}), применяем корректировку")

    # Повторный проход с более мягкими критериями
    corrected_labels = []

    for i, original_label in enumerate(labels):
      if original_label == 0 and i + future_bars < len(df_enhanced):  # Только для HOLD
        current_price = df_enhanced.iloc[i]['close']
        future_slice = df_enhanced.iloc[i + 1:i + future_bars + 1]
        final_close = future_slice['close'].iloc[-1]
        final_return = (final_close - current_price) / current_price

        # Более мягкие критерии
        soft_threshold = dynamic_threshold * 0.7

        if final_return > soft_threshold:
          corrected_labels.append(1)  # BUY
        elif final_return < -soft_threshold:
          corrected_labels.append(2)  # SELL
        else:
          corrected_labels.append(0)  # HOLD
      else:
        corrected_labels.append(original_label)

    labels_series = pd.Series(corrected_labels, index=df_enhanced.index)

  # Финальная статистика
  final_counts = labels_series.value_counts()
  buy_final = final_counts.get(1, 0)
  sell_final = final_counts.get(2, 0)
  hold_final = final_counts.get(0, 0)

  logger.info(f"Метки созданы: BUY={buy_final} ({buy_final / total_samples * 100:.1f}%), "
              f"SELL={sell_final} ({sell_final / total_samples * 100:.1f}%), "
              f"HOLD={hold_final} ({hold_final / total_samples * 100:.1f}%)")

  return labels_series

def apply_label_smoothing(labels: pd.Series, window: int = 3) -> pd.Series:
    """
    Применяет сглаживание меток для уменьшения шума
    """
    smoothed_labels = labels.copy()

    for i in range(window, len(labels) - window):
      current_window = labels[i - window:i + window + 1]

      # Если большинство меток в окне одинаковые, применяем их
      value_counts = current_window.value_counts()
      if len(value_counts) > 0:
        majority_label = value_counts.index[0]
        majority_count = value_counts.iloc[0]

        # Применяем мажоритарную метку если она составляет > 60% окна
        if majority_count / len(current_window) > 0.6:
          smoothed_labels.iloc[i] = majority_label

    return smoothed_labels


def create_multi_timeframe_labels(data: pd.DataFrame,
                                  timeframes: List[int] = [2, 4, 8],
                                  weights: List[float] = [0.5, 0.3, 0.2]) -> pd.Series:
  """
  Создает метки на основе множественных таймфреймов
  """
  all_labels = []

  for tf, weight in zip(timeframes, weights):
    tf_labels = create_balanced_lorentzian_labels(
      data,
      future_bars=tf,
      threshold_percent=0.85,
      volatility_adaptive=True
    )
    all_labels.append((tf_labels, weight))

  # Взвешенное голосование
  final_labels = []

  for i in range(len(data)):
    weighted_votes = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL

    for labels, weight in all_labels:
      if i < len(labels):
        weighted_votes[labels.iloc[i]] += weight

    # Выбираем класс с максимальным весом
    final_label = max(weighted_votes.items(), key=lambda x: x[1])[0]
    final_labels.append(final_label)

  return pd.Series(final_labels, index=data.index)