import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import asyncio
import joblib
from datetime import datetime
import warnings

from .lorentzian_classifier import LorentzianClassifier, create_training_labels
from utils.logging_config import get_logger
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


class ModelTrainer:
  """
  Класс для обучения и валидации ML моделей для торговых стратегий.
  """

  def __init__(self, data_path: Optional[str] = None, models_dir: str = "trained_models"):
    self.data_path = data_path
    self.models_dir = Path(models_dir)
    self.models_dir.mkdir(exist_ok=True)
    self.models = {}
    self.training_history = []

  def prepare_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет технические индикаторы к историческим данным.
    """
    if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
      logger.error("DataFrame должен содержать OHLCV колонки")
      return df

    logger.info("Расчет технических индикаторов...")

    # RSI
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['rsi_9'] = ta.rsi(df['close'], length=9)
    df['rsi_21'] = ta.rsi(df['close'], length=21)

    # MACD
    macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd_df is not None:
      df['macd'] = macd_df.iloc[:, 0]  # MACD line
      df['macd_signal'] = macd_df.iloc[:, 1]  # Signal line
      df['macd_histogram'] = macd_df.iloc[:, 2]  # Histogram

    # Moving Averages
    df['sma_20'] = ta.sma(df['close'], length=20)
    df['sma_50'] = ta.sma(df['close'], length=50)
    df['ema_12'] = ta.ema(df['close'], length=12)
    df['ema_26'] = ta.ema(df['close'], length=26)

    # Bollinger Bands
    bb_df = ta.bbands(df['close'], length=20, std=2)
    if bb_df is not None:
      df['bb_upper'] = bb_df.iloc[:, 0]
      df['bb_middle'] = bb_df.iloc[:, 1]
      df['bb_lower'] = bb_df.iloc[:, 2]
      df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
      df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # ATR (Average True Range)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # Stochastic
    stoch_df = ta.stoch(df['high'], df['low'], df['close'])
    if stoch_df is not None:
      df['stoch_k'] = stoch_df.iloc[:, 0]
      df['stoch_d'] = stoch_df.iloc[:, 1]

    # Williams %R
    df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)

    # Volume indicators
    df['volume_sma'] = ta.sma(df['volume'], length=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Price-based features
    df['price_change'] = df['close'].pct_change()
    df['price_change_abs'] = df['price_change'].abs()
    df['high_low_pct'] = (df['high'] - df['low']) / df['close']
    df['open_close_pct'] = (df['close'] - df['open']) / df['open']

    # Volatility
    df['volatility'] = df['close'].rolling(window=20).std()
    df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=50).mean()

    logger.info(
      f"Добавлено {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']])} технических индикаторов")

    return df

  def create_features_and_labels(self,
                                 df: pd.DataFrame,
                                 future_bars: int = 5,
                                 profit_threshold: float = 0.015,
                                 loss_threshold: float = 0.01) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Создает признаки и метки для обучения модели.

    Args:
        df: DataFrame с OHLCV данными и индикаторами
        future_bars: Количество баров для анализа будущих движений
        profit_threshold: Минимальный порог прибыли для BUY сигнала
        loss_threshold: Минимальный порог убытка для SELL сигнала

    Returns:
        Tuple с признаками и метками
    """
    logger.info(f"Создание признаков и меток (future_bars={future_bars}, profit_threshold={profit_threshold})")

    # Добавляем технические индикаторы если их нет
    if 'rsi' not in df.columns:
      df = self.prepare_technical_indicators(df)

    # Создаем дополнительные признаки
    df = self._add_advanced_features(df)

    # Создаем метки
    labels = self._create_advanced_labels(df, future_bars, profit_threshold, loss_threshold)

    # Убираем строки с NaN
    feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    features_df = df[feature_columns].copy()

    # Синхронизируем индексы
    valid_idx = features_df.dropna().index.intersection(labels.dropna().index)
    features_clean = features_df.loc[valid_idx]
    labels_clean = labels.loc[valid_idx]

    logger.info(f"Создано {len(features_clean)} обучающих примеров с {len(feature_columns)} признаками")
    logger.info(f"Распределение классов: {labels_clean.value_counts().to_dict()}")

    return features_clean, labels_clean

  def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет продвинутые признаки для улучшения качества модели.
    """
    # Momentum features
    df['momentum_3'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1

    # RSI дивергенция
    df['rsi_change'] = df['rsi'].diff()
    df['price_change_3'] = df['close'].pct_change(3)

    # Trend strength
    df['trend_strength'] = (df['close'] / df['sma_20'] - 1) if 'sma_20' in df.columns else 0

    # Volume-price trend
    if 'volume' in df.columns:
      df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
      df['vpt_sma'] = df['vpt'].rolling(window=10).mean()

    # Support/Resistance levels
    df['local_high'] = df['high'].rolling(window=10, center=True).max()
    df['local_low'] = df['low'].rolling(window=10, center=True).min()
    df['distance_to_high'] = (df['close'] - df['local_high']) / df['close']
    df['distance_to_low'] = (df['close'] - df['local_low']) / df['close']

    return df

  def _create_advanced_labels(self,
                              df: pd.DataFrame,
                              future_bars: int,
                              profit_threshold: float,
                              loss_threshold: float) -> pd.Series:
    """
    Создает улучшенные метки учитывающие не только направление движения,
    но и силу тренда и временные рамки.
    """
    labels = []

    for i in range(len(df)):
      if i + future_bars >= len(df):
        labels.append(0)  # HOLD для последних баров
        continue

      current_price = df.iloc[i]['close']
      future_prices = df.iloc[i + 1:i + future_bars + 1]['close']

      # Анализируем максимальную прибыль/убыток
      max_profit = (future_prices.max() - current_price) / current_price
      max_loss = (current_price - future_prices.min()) / current_price

      # Анализируем финальную цену
      final_price = future_prices.iloc[-1]
      final_return = (final_price - current_price) / current_price

      # Логика принятия решений
      if max_profit > profit_threshold and final_return > profit_threshold * 0.5:
        # Сильный BUY сигнал: высокий потенциал прибыли и положительный финальный результат
        labels.append(1)
      elif max_loss > loss_threshold and final_return < -loss_threshold * 0.5:
        # Сильный SELL сигнал: высокий риск потерь и отрицательный финальный результат
        labels.append(2)
      else:
        # HOLD: неопределенная ситуация
        labels.append(0)

    return pd.Series(labels, index=df.index)

  def train_lorentzian_model(self,
                             df: pd.DataFrame,
                             model_params: Optional[Dict] = None,
                             validation_method: str = 'time_series',
                             test_size: float = 0.2) -> Dict[str, Any]:
    """
    Обучает Lorentzian Classifier и проводит валидацию.

    Args:
        df: DataFrame с историческими данными
        model_params: Параметры модели
        validation_method: Метод валидации ('simple', 'time_series', 'cross_val')
        test_size: Размер тестовой выборки

    Returns:
        Словарь с результатами обучения и метриками
    """
    if model_params is None:
      model_params = {
        'k_neighbors': 8,
        'max_lookback': 2000,
        'feature_weights': {
          'rsi': 1.2,
          'macd': 1.1,
          'bb_percent': 1.0,
          'atr': 0.9,
          'volume_ratio': 0.8
        }
      }

    logger.info("Начало обучения Lorentzian модели...")

    # Подготавливаем данные
    X, y = self.create_features_and_labels(df)

    if X.empty or y.empty:
      logger.error("Не удалось создать обучающие данные")
      return {'success': False, 'error': 'No training data created'}

    # Создаем модель
    model = LorentzianClassifier(**model_params)

    results = {}

    try:
      if validation_method == 'simple':
        results = self._simple_validation(model, X, y, test_size)
      elif validation_method == 'time_series':
        results = self._time_series_validation(model, X, y, test_size)
      elif validation_method == 'cross_val':
        results = self._cross_validation(model, X, y)
      else:
        logger.error(f"Неизвестный метод валидации: {validation_method}")
        return {'success': False, 'error': f'Unknown validation method: {validation_method}'}

      # Сохраняем лучшую модель
      if results.get('success', False):
        model_name = f"lorentzian_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = self.models_dir / f"{model_name}.pkl"

        if results['final_model'].save_model(str(model_path)):
          results['model_path'] = str(model_path)
          results['model_name'] = model_name
          self.models[model_name] = results['final_model']

          # Сохраняем историю обучения
          training_record = {
            'timestamp': datetime.now(),
            'model_type': 'LorentzianClassifier',
            'model_name': model_name,
            'params': model_params,
            'validation_method': validation_method,
            'metrics': results.get('metrics', {})
          }
          self.training_history.append(training_record)

          logger.info(f"Модель {model_name} успешно обучена и сохранена")

    except Exception as e:
      logger.error(f"Ошибка при обучении модели: {e}")
      results = {'success': False, 'error': str(e)}

    return results

  def _simple_validation(self, model: LorentzianClassifier, X: pd.DataFrame, y: pd.Series, test_size: float) -> Dict:
    """
    Простая валидация с разделением на train/test.
    """
    logger.info("Выполняется простая валидация...")

    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Обучаем модель
    model.fit(X_train, y_train)

    # Предсказания
    y_pred = model.predict(X_test)

    if y_pred is None:
      return {'success': False, 'error': 'Model prediction failed'}

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    logger.info(f"Точность модели: {accuracy:.4f}")

    return {
      'success': True,
      'final_model': model,
      'metrics': {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'train_size': len(X_train),
        'test_size': len(X_test)
      }
    }

  def _time_series_validation(self, model: LorentzianClassifier, X: pd.DataFrame, y: pd.Series,
                              test_size: float) -> Dict:
    """
    Валидация с учетом временной структуры данных.
    """
    logger.info("Выполняется временная валидация...")

    # Разделяем по времени (последние данные для теста)
    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Обучаем модель
    model.fit(X_train, y_train)

    # Предсказания на тестовой выборке
    y_pred = model.predict(X_test)

    if y_pred is None:
      return {'success': False, 'error': 'Model prediction failed'}

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Дополнительная оценка стабильности во времени
    stability_scores = self._calculate_stability_scores(model, X_test, y_test)

    logger.info(f"Точность временной валидации: {accuracy:.4f}")

    return {
      'success': True,
      'final_model': model,
      'metrics': {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'stability_scores': stability_scores,
        'train_size': len(X_train),
        'test_size': len(X_test)
      }
    }

  def _cross_validation(self, model: LorentzianClassifier, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
    """
    Кросс-валидация с TimeSeriesSplit.
    """
    logger.info(f"Выполняется кросс-валидация с {n_splits} фолдами...")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    best_model = None
    best_score = 0

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
      logger.info(f"Обработка фолда {fold + 1}/{n_splits}")

      X_train_fold = X.iloc[train_idx]
      X_test_fold = X.iloc[test_idx]
      y_train_fold = y.iloc[train_idx]
      y_test_fold = y.iloc[test_idx]

      # Создаем новую модель для каждого фолда
      fold_model = LorentzianClassifier(
        k_neighbors=model.k_neighbors,
        max_lookback=model.max_lookback,
        feature_weights=model.feature_weights
      )

      fold_model.fit(X_train_fold, y_train_fold)
      y_pred = fold_model.predict(X_test_fold)

      if y_pred is not None:
        score = accuracy_score(y_test_fold, y_pred)
        scores.append(score)

        # Сохраняем лучшую модель
        if score > best_score:
          best_score = score
          best_model = fold_model

        logger.info(f"Фолд {fold + 1} точность: {score:.4f}")

    if not scores:
      return {'success': False, 'error': 'Cross-validation failed'}

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    logger.info(f"Средняя точность кросс-валидации: {mean_score:.4f} ± {std_score:.4f}")

    return {
      'success': True,
      'final_model': best_model,
      'metrics': {
        'cv_scores': scores,
        'mean_cv_score': mean_score,
        'std_cv_score': std_score,
        'best_fold_score': best_score,
        'total_samples': len(X)
      }
    }

  def _calculate_stability_scores(self, model: LorentzianClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """
    Рассчитывает метрики стабильности модели во времени.
    """
    window_size = len(X_test) // 5  # Разбиваем на 5 окон
    stability_scores = []

    for i in range(0, len(X_test) - window_size, window_size):
      window_X = X_test.iloc[i:i + window_size]
      window_y = y_test.iloc[i:i + window_size]

      y_pred = model.predict(window_X)
      if y_pred is not None and len(y_pred) > 0:
        score = accuracy_score(window_y, y_pred)
        stability_scores.append(score)

    if stability_scores:
      return {
        'window_scores': stability_scores,
        'stability_mean': np.mean(stability_scores),
        'stability_std': np.std(stability_scores),
        'stability_min': np.min(stability_scores),
        'stability_max': np.max(stability_scores)
      }
    else:
      return {'error': 'Could not calculate stability scores'}

  def evaluate_model_performance(self, model_name: str, test_data: pd.DataFrame) -> Optional[Dict]:
    """
    Оценивает производительность обученной модели на новых данных.
    """
    if model_name not in self.models:
      logger.error(f"Модель {model_name} не найдена")
      return None

    model = self.models[model_name]

    # Подготавливаем тестовые данные
    X_test, y_test = self.create_features_and_labels(test_data)

    if X_test.empty:
      logger.error("Не удалось подготовить тестовые данные")
      return None

    # Предсказания
    y_pred = model.predict(X_test)
    if y_pred is None:
      logger.error("Модель не смогла сделать предсказания")
      return None

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(f"Производительность модели {model_name}: точность = {accuracy:.4f}")

    return {
      'model_name': model_name,
      'accuracy': accuracy,
      'classification_report': report,
      'predictions': y_pred.tolist(),
      'actual': y_test.values.tolist()
    }

  def load_trained_model(self, model_path: str, model_name: Optional[str] = None) -> bool:
    """
    Загружает обученную модель из файла.
    """
    if not Path(model_path).exists():
      logger.error(f"Файл модели не найден: {model_path}")
      return False

    try:
      model = LorentzianClassifier()
      if model.load_model(model_path):
        name = model_name or Path(model_path).stem
        self.models[name] = model
        logger.info(f"Модель {name} успешно загружена")
        return True
    except Exception as e:
      logger.error(f"Ошибка при загрузке модели: {e}")

    return False

  def get_available_models(self) -> List[str]:
    """
    Возвращает список доступных обученных моделей.
    """
    return list(self.models.keys())

  def get_training_history(self) -> List[Dict]:
    """
    Возвращает историю обучения моделей.
    """
    return self.training_history

  def create_model_report(self, model_name: str) -> Optional[str]:
    """
    Создает отчет о модели.
    """
    if model_name not in self.models:
      logger.error(f"Модель {model_name} не найдена")
      return None

    # Найдем запись в истории обучения
    training_record = None
    for record in self.training_history:
      if record['model_name'] == model_name:
        training_record = record
        break

    if not training_record:
      return f"Модель {model_name} загружена, но нет записи об обучении"

    report = f"""
Отчет о модели: {model_name}
{'=' * 50}

Дата обучения: {training_record['timestamp']}
Тип модели: {training_record['model_type']}
Метод валидации: {training_record['validation_method']}

Параметры модели:
{self._format_dict(training_record['params'])}

Метрики производительности:
{self._format_dict(training_record['metrics'])}
        """

    return report.strip()

  def _format_dict(self, d: Dict, indent: int = 2) -> str:
    """
    Форматирует словарь для красивого вывода.
    """
    lines = []
    for key, value in d.items():
      if isinstance(value, dict):
        lines.append(f"{' ' * indent}{key}:")
        lines.append(self._format_dict(value, indent + 2))
      elif isinstance(value, list):
        lines.append(f"{' ' * indent}{key}: {len(value)} элементов")
      else:
        lines.append(f"{' ' * indent}{key}: {value}")
    return '\n'.join(lines)

  def cleanup_old_models(self, keep_latest: int = 5):
    """
    Удаляет старые файлы моделей, оставляя только последние.
    """
    model_files = list(self.models_dir.glob("*.pkl"))
    if len(model_files) <= keep_latest:
      return

    # Сортируем по времени создания
    model_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)

    # Удаляем старые файлы
    for model_file in model_files[keep_latest:]:
      try:
        model_file.unlink()
        logger.info(f"Удален старый файл модели: {model_file}")
      except Exception as e:
        logger.error(f"Ошибка при удалении файла {model_file}: {e}")



# Пример использования
if __name__ == '__main__':
  from logger_setup import setup_logging
  import os

  setup_logging("INFO")

  # Создаем тренер
  trainer = ModelTrainer()

  # Генерируем тестовые данные
  np.random.seed(42)
  n_samples = 2000

  dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
  test_data = pd.DataFrame({
    'timestamp': dates,
    'open': np.random.randn(n_samples).cumsum() + 100,
    'high': np.random.randn(n_samples).cumsum() + 102,
    'low': np.random.randn(n_samples).cumsum() + 98,
    'close': np.random.randn(n_samples).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, n_samples)
  })

  # Обучаем модель
  results = trainer.train_lorentzian_model(
    test_data,
    validation_method='time_series',
    test_size=0.2
  )

  if results.get('success'):
    logger.info("Обучение завершено успешно!")
    logger.info(f"Точность: {results['metrics']['accuracy']:.4f}")

    # Создаем отчет
    if 'model_name' in results:
      report = trainer.create_model_report(results['model_name'])
      print(report)
  else:
    logger.error(f"Ошибка обучения: {results.get('error', 'Unknown error')}")