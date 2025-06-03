import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import Optional, Tuple
import joblib
import os
import copy
from sklearn.base import clone as sk_clone
from utils.logging_config import get_logger

logger = get_logger(__name__)


class LorentzianClassifier(BaseEstimator, ClassifierMixin):
  """
  Реализация Lorentzian Classifier для торговых сигналов.
  Использует Lorentzian distance для классификации.
  """

  def __init__(self, k_neighbors=8, max_lookback=2000, feature_weights=None):
    self.k_neighbors = k_neighbors
    self.max_lookback = max_lookback
    self.feature_weights = feature_weights or {}
    self.scaler = StandardScaler()
    self.is_fitted = False
    self.X_train = None
    self.y_train = None
    self.feature_names = None

    logger.info(f"LorentzianClassifier инициализирован с k={k_neighbors}, max_lookback={max_lookback}")

  def _lorentzian_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Вычисляет Lorentzian расстояние между двумя векторами признаков.
    Lorentzian distance = sum(log(1 + abs(x1[i] - x2[i])))
    """
    if len(x1) != len(x2):
      raise ValueError("Векторы должны иметь одинаковую размерность")

    # Применяем веса признаков если они заданы
    weights = np.ones(len(x1))
    if self.feature_names:
      for i, feature in enumerate(self.feature_names):
        if feature in self.feature_weights:
          weights[i] = self.feature_weights[feature]

    distance = np.sum(weights * np.log1p(np.abs(x1 - x2)))
    return distance

  def _prepare_features(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[list]]:
    """
    Подготавливает признаки из DataFrame.
    Возвращает массив признаков и список их названий.
    """
    # Базовые технические индикаторы
    feature_columns = []

    # RSI
    if 'rsi' in df.columns:
      feature_columns.append('rsi')

    # MACD
    if 'macd' in df.columns:
      feature_columns.append('macd')
    if 'macd_signal' in df.columns:
      feature_columns.append('macd_signal')
    if 'macd_histogram' in df.columns:
      feature_columns.append('macd_histogram')

    # Moving Averages
    if 'sma_20' in df.columns:
      feature_columns.append('sma_20')
    if 'ema_12' in df.columns:
      feature_columns.append('ema_12')

    # Bollinger Bands
    if 'bb_upper' in df.columns:
      feature_columns.append('bb_upper')
    if 'bb_lower' in df.columns:
      feature_columns.append('bb_lower')
    if 'bb_percent' in df.columns:
      feature_columns.append('bb_percent')

    # Volatility indicators
    if 'atr' in df.columns:
      feature_columns.append('atr')
    if 'volatility' in df.columns:
      feature_columns.append('volatility')

    # Price-based features
    if 'close' in df.columns and 'open' in df.columns:
      df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
      feature_columns.append('price_change_pct')

    if 'volume' in df.columns:
      feature_columns.append('volume')

    if not feature_columns:
      logger.error("Не найдено подходящих признаков в DataFrame")
      return None, None

    # Отфильтровываем только существующие колонки
    available_features = [col for col in feature_columns if col in df.columns]

    if not available_features:
      logger.error(f"Ни один из требуемых признаков не найден в DataFrame. Доступные колонки: {df.columns.tolist()}")
      return None, None

    features_df = df[available_features].copy()

    # Удаляем NaN значения
    features_df = features_df.dropna()

    if features_df.empty:
      logger.warning("После удаления NaN не осталось данных")
      return None, None

    logger.info(f"Подготовлено {len(available_features)} признаков: {available_features}")
    return features_df.values, available_features

  def fit(self, X_df: pd.DataFrame, y: pd.Series):
    """
    Обучает модель на исторических данных.
    """
    logger.info(f"Начало обучения LorentzianClassifier на {len(X_df)} примерах...")

    X_prepared, feature_names = self._prepare_features(X_df)
    if X_prepared is None:
      logger.error("Не удалось подготовить признаки для обучения")
      return self

    # Синхронизируем y с подготовленными данными
    valid_indices = X_df.dropna(subset=[col for col in feature_names if col in X_df.columns]).index
    y_aligned = y.loc[valid_indices]

    if len(X_prepared) != len(y_aligned):
      logger.error(f"Размерности X ({len(X_prepared)}) и y ({len(y_aligned)}) не совпадают")
      return self

    # Ограничиваем размер обучающей выборки для эффективности
    if len(X_prepared) > self.max_lookback:
      X_prepared = X_prepared[-self.max_lookback:]
      y_aligned = y_aligned.tail(self.max_lookback)
      logger.info(f"Обучающая выборка ограничена до {self.max_lookback} примеров")

    # Масштабируем признаки
    X_scaled = self.scaler.fit_transform(X_prepared)

    # Сохраняем обучающие данные
    self.X_train = X_scaled
    self.y_train = y_aligned.values
    self.feature_names = feature_names
    self.is_fitted = True

    logger.info("Модель LorentzianClassifier успешно обучена")
    return self

  def predict(self, X_df_new: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Делает предсказания для новых данных.
    """
    if not self.is_fitted:
      logger.error("Модель не обучена. Вызовите fit() перед predict()")
      return None

    X_prepared, _ = self._prepare_features(X_df_new)
    if X_prepared is None:
      logger.warning("Не удалось подготовить признаки для предсказания")
      return np.array([])

    X_scaled = self.scaler.transform(X_prepared)
    predictions = []

    for x_new in X_scaled:
      # Вычисляем расстояния до всех обучающих примеров
      distances = []
      for i, x_train in enumerate(self.X_train):
        dist = self._lorentzian_distance(x_new, x_train)
        distances.append((dist, self.y_train[i]))

      # Сортируем по расстоянию и берем k ближайших соседей
      distances.sort(key=lambda x: x[0])
      k_nearest = distances[:self.k_neighbors]

      # Голосование с весами (обратно пропорциональными расстоянию)
      class_votes = {}
      for dist, label in k_nearest:
        weight = 1.0 / (1.0 + dist)  # Вес обратно пропорционален расстоянию
        if label not in class_votes:
          class_votes[label] = 0
        class_votes[label] += weight

      # Выбираем класс с максимальным весом
      predicted_class = max(class_votes, key=class_votes.get)
      predictions.append(predicted_class)

    logger.debug(f"Предсказания: {predictions}")
    return np.array(predictions)

  def predict_proba(self, X_df_new: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Возвращает вероятности принадлежности к каждому классу.
    """
    if not self.is_fitted:
      logger.error("Модель не обучена")
      return None

    X_prepared, _ = self._prepare_features(X_df_new)
    if X_prepared is None:
      return np.array([])

    X_scaled = self.scaler.transform(X_prepared)
    probabilities = []

    # Получаем уникальные классы
    unique_classes = np.unique(self.y_train)

    for x_new in X_scaled:
      distances = []
      for i, x_train in enumerate(self.X_train):
        dist = self._lorentzian_distance(x_new, x_train)
        distances.append((dist, self.y_train[i]))

      distances.sort(key=lambda x: x[0])
      k_nearest = distances[:self.k_neighbors]

      # Вычисляем вероятности для каждого класса
      class_weights = {cls: 0.0 for cls in unique_classes}
      total_weight = 0.0

      for dist, label in k_nearest:
        weight = 1.0 / (1.0 + dist)
        class_weights[label] += weight
        total_weight += weight

      # Нормализуем веса до вероятностей
      class_probs = [class_weights[cls] / total_weight for cls in unique_classes]
      probabilities.append(class_probs)

    return np.array(probabilities)

  def save_model(self, filepath: str):
    """
    Сохраняет обученную модель в файл.
    """
    if not self.is_fitted:
      logger.error("Нельзя сохранить необученную модель")
      return False

    try:
      model_data = {
        'X_train': self.X_train,
        'y_train': self.y_train,
        'scaler': self.scaler,
        'feature_names': self.feature_names,
        'k_neighbors': self.k_neighbors,
        'max_lookback': self.max_lookback,
        'feature_weights': self.feature_weights,
        'is_fitted': self.is_fitted
      }
      joblib.dump(model_data, filepath)
      logger.info(f"Модель сохранена в {filepath}")
      return True
    except Exception as e:
      logger.error(f"Ошибка при сохранении модели: {e}")
      return False

  def load_model(self, filepath: str):
    """
    Загружает обученную модель из файла.
    """
    if not os.path.exists(filepath):
      logger.error(f"Файл модели не найден: {filepath}")
      return False

    try:
      model_data = joblib.load(filepath)
      self.X_train = model_data['X_train']
      self.y_train = model_data['y_train']
      self.scaler = model_data['scaler']
      self.feature_names = model_data['feature_names']
      self.k_neighbors = model_data['k_neighbors']
      self.max_lookback = model_data['max_lookback']
      self.feature_weights = model_data['feature_weights']
      self.is_fitted = model_data['is_fitted']

      logger.info(f"Модель загружена из {filepath}")
      return True
    except Exception as e:
      logger.error(f"Ошибка при загрузке модели: {e}")
      return False

  def clone(self):
    """Создает копию текущей модели"""
    new_model = LorentzianClassifier(k_neighbors=self.k_neighbors)
    # Копируем все существующие атрибуты
    for attr_name in ['classifier', 'scaler', 'feature_columns', 'accuracy', 'is_fitted']:
      if hasattr(self, attr_name):
        value = getattr(self, attr_name)

        # Для объектов, которые нужно клонировать
        if attr_name == 'classifier' and value is not None:
          setattr(new_model, attr_name, sk_clone(value))
        elif attr_name == 'scaler' and value is not None:
          setattr(new_model, attr_name, copy.deepcopy(value))
        elif attr_name == 'feature_columns' and value is not None:
          setattr(new_model, attr_name, copy.deepcopy(value))
        else:
          setattr(new_model, attr_name, value)
    return new_model

def create_training_labels(df: pd.DataFrame,
                           future_bars: int = 5,
                           profit_threshold: float = 0.01) -> pd.Series:
  """
  Создает метки для обучения на основе будущих движений цены.

  Args:
      df: DataFrame с данными OHLCV
      future_bars: Количество баров в будущее для анализа
      profit_threshold: Порог прибыли для генерации сигналов (в долях)

  Returns:
      Series с метками: 0 - держать, 1 - покупать, 2 - продавать
  """
  labels = []

  for i in range(len(df)):
    if i + future_bars >= len(df):
      # Для последних баров ставим "держать"
      labels.append(0)
      continue

    current_price = df.iloc[i]['close']
    future_prices = df.iloc[i + 1:i + future_bars + 1]['close']

    max_future_price = future_prices.max()
    min_future_price = future_prices.min()

    # Вычисляем потенциальную прибыль/убыток
    upside_potential = (max_future_price - current_price) / current_price
    downside_risk = (current_price - min_future_price) / current_price

    if upside_potential > profit_threshold and upside_potential > downside_risk:
      labels.append(1)  # BUY
    elif downside_risk > profit_threshold and downside_risk > upside_potential:
      labels.append(2)  # SELL
    else:
      labels.append(0)  # HOLD

  return pd.Series(labels, index=df.index)


# Пример использования
if __name__ == '__main__':
  from logger_setup import setup_logging

  setup_logging("INFO")

  # Генерируем тестовые данные
  np.random.seed(42)
  n_samples = 1000

  test_data = pd.DataFrame({
    'close': np.cumsum(np.random.randn(n_samples) * 0.01) + 100,
    'volume': np.random.randint(1000, 10000, n_samples),
    'rsi': np.random.randint(20, 80, n_samples),
    'macd': np.random.randn(n_samples) * 0.1,
    'atr': np.random.rand(n_samples) * 2 + 0.5
  })

  # Создаем метки для обучения
  y_labels = create_training_labels(test_data, future_bars=5, profit_threshold=0.02)

  # Разделяем на обучающую и тестовую выборки
  split_idx = int(0.8 * len(test_data))
  X_train = test_data[:split_idx]
  y_train = y_labels[:split_idx]
  X_test = test_data[split_idx:]
  y_test = y_labels[split_idx:]

  # Обучаем модель
  model = LorentzianClassifier(k_neighbors=8)
  model.fit(X_train, y_train)

  # Делаем предсказания
  predictions = model.predict(X_test)
  if predictions is not None and len(predictions) > 0:
    accuracy = accuracy_score(y_test[:len(predictions)], predictions)
    logger.info(f"Точность модели: {accuracy:.4f}")

    # Сохраняем модель
    model.save_model("trained_lorentzian_model.pkl")