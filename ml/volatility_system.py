# ============================================================================
# СИСТЕМА ПРОГНОЗИРОВАНИЯ ВОЛАТИЛЬНОСТИ
# Комплексная реализация для торгового бота на Bybit
# ============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor # Будем использовать LightGBM как основную регрессионную модель
import joblib

from utils.logging_config import get_logger

warnings.filterwarnings('ignore')
logger = get_logger(__name__)


# ============================================================================
# 1. БАЗОВЫЕ СТРУКТУРЫ И ЭНУМЫ
# ============================================================================

class VolatilityRegime(Enum):
  """Режимы волатильности"""
  VERY_LOW = "very_low"  # < 10-й перцентиль
  LOW = "low"  # 10-25-й перцентиль
  NORMAL = "normal"  # 25-75-й перцентиль
  HIGH = "high"  # 75-90-й перцентиль
  VERY_HIGH = "very_high"  # > 90-й перцентиль


class ModelType(Enum):
  """Типы моделей для прогнозирования"""

  LIGHTGBM = "lightgbm"



@dataclass
class VolatilityPrediction:
  """Структура прогноза волатильности"""
  predicted_volatility: float
  current_volatility: float
  volatility_regime: VolatilityRegime
  confidence: float
  prediction_horizon: int  # periods ahead
  model_used: str
  timestamp: datetime

  @property
  def volatility_change_ratio(self) -> float:
    """Коэффициент изменения волатильности"""
    if self.current_volatility == 0:
      return 1.0
    return self.predicted_volatility / self.current_volatility


@dataclass
class VolatilityFeatures:
  """Признаки для модели волатильности"""
  # Базовые волатильности
  volatility_5: float
  volatility_10: float
  volatility_20: float
  volatility_50: float

  # Относительные волатильности
  vol_ratio_5_20: float
  vol_ratio_10_20: float
  vol_ratio_20_50: float

  # Скользящие средние волатильности
  vol_sma_5: float
  vol_sma_10: float
  vol_ema_5: float
  vol_ema_10: float

  # Статистические меры
  vol_std_20: float
  vol_skew_20: float
  vol_kurt_20: float

  # Ценовые факторы
  price_change_1: float
  price_change_5: float
  price_change_20: float

  # Объемные факторы
  volume_volatility_20: float
  volume_price_correlation_20: float

  # Технические индикаторы
  rsi_14: float
  bb_position: float  # позиция цены в полосах Боллинджера
  atr_14: float

  # Временные факторы
  hour_of_day: int
  day_of_week: int
  is_weekend: bool


# ============================================================================
# 2. РАСЧЕТ ПРИЗНАКОВ ВОЛАТИЛЬНОСТИ
# ============================================================================

class VolatilityFeatureCalculator:
  """Класс для расчета признаков волатильности"""

  def __init__(self):
    self.scaler = StandardScaler()
    self.fitted = False

  def calculate_volatility(self, prices: pd.Series, window: int,
                           method: str = 'std') -> pd.Series:
    """
    Расчет волатильности различными методами

    Args:
        prices: Серия цен
        window: Окно для расчета
        method: Метод расчета ('std', 'parkinson', 'garman_klass', 'rogers_satchell')
    """
    if method == 'std':
      returns = prices.pct_change()
      return returns.rolling(window=window).std() * np.sqrt(252)  # Аннуализированная

    elif method == 'parkinson':
      # Метод Паркинсона (требует high, low)
      # Здесь упрощенная версия
      returns = prices.pct_change()
      return returns.rolling(window=window).std() * np.sqrt(252)

    # Можно добавить другие методы расчета волатильности

  def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Извлечение всех признаков для модели волатильности

    Args:
        data: DataFrame с OHLCV данными
    """
    df = data.copy()

    # Расчет различных волатильностей
    for window in [5, 10, 20, 50]:
      df[f'volatility_{window}'] = self.calculate_volatility(df['close'], window)

    # Относительные волатильности
    df['vol_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
    df['vol_ratio_10_20'] = df['volatility_10'] / df['volatility_20']
    df['vol_ratio_20_50'] = df['volatility_20'] / df['volatility_50']

    # Скользящие средние волатильности
    df['vol_sma_5'] = df['volatility_20'].rolling(5).mean()
    df['vol_sma_10'] = df['volatility_20'].rolling(10).mean()
    df['vol_ema_5'] = df['volatility_20'].ewm(span=5).mean()
    df['vol_ema_10'] = df['volatility_20'].ewm(span=10).mean()

    # Статистические характеристики волатильности
    df['vol_std_20'] = df['volatility_20'].rolling(20).std()
    df['vol_skew_20'] = df['volatility_20'].rolling(20).skew()
    df['vol_kurt_20'] = df['volatility_20'].rolling(20).kurt()

    # Ценовые изменения
    df['price_change_1'] = df['close'].pct_change()
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_20'] = df['close'].pct_change(20)

    # Объемные факторы
    if 'volume' in df.columns:
      df['volume_volatility_20'] = df['volume'].rolling(20).std()
      df['volume_price_correlation_20'] = df['volume'].rolling(20).corr(df['close'])
    else:
      df['volume_volatility_20'] = 0
      df['volume_price_correlation_20'] = 0

    # Технические индикаторы
    df['rsi_14'] = self._calculate_rsi(df['close'], 14)
    bb_upper, bb_lower = self._calculate_bollinger_bands(df['close'], 20, 2)
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    df['atr_14'] = self._calculate_atr(df, 14)

    # Временные факторы
    df['hour_of_day'] = df.index.hour if hasattr(df.index, 'hour') else 0
    df['day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    return df

  def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
    """Расчет RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

  def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20,
                                 num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
    """Расчет полос Боллинджера"""
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

  def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Расчет Average True Range"""
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


# ============================================================================
# 3. МОДЕЛЬ ПРОГНОЗИРОВАНИЯ ВОЛАТИЛЬНОСТИ
# ============================================================================

class VolatilityPredictor:
  """Основной класс для прогнозирования волатильности"""

  def __init__(self, model_type: ModelType = ModelType.LIGHTGBM,
               prediction_horizon: int = 5):
    self.model_type = model_type
    self.prediction_horizon = prediction_horizon
    self.models = {}
    self.scalers = {}
    self.feature_importance = {}
    self.is_fitted = False
    self.feature_calculator = VolatilityFeatureCalculator()

    # Инициализация моделей
    self._initialize_models()

    # Для определения режимов волатильности
    self.volatility_percentiles = {}

  def _initialize_models(self):
    """Инициализация различных моделей"""
    self.models = {
      # 'random_forest': RandomForestRegressor(
      #   n_estimators=100,
      #   max_depth=10,
      #   min_samples_split=5,
      #   min_samples_leaf=2,
      #   random_state=42
      # ),
      # 'gradient_boosting': GradientBoostingRegressor(
      #   n_estimators=100,
      #   max_depth=6,
      #   learning_rate=0.1,
      #   random_state=42
      # ),
      # 'xgboost': xgb.XGBRegressor(
      #   n_estimators=100,
      #   max_depth=6,
      #   learning_rate=0.1,
      #   random_state=42
      # ),
      'lightgbm': lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
      ),
      # 'ridge': Ridge(alpha=1.0),
      # 'lasso': Lasso(alpha=0.1)
    }



    # Скейлеры для каждой модели
    for model_name in self.models.keys():
      self.scalers[model_name] = StandardScaler()

  def prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Подготовка данных для обучения

    Args:
        data: DataFrame с OHLCV данными

    Returns:
        X: признаки
        y: целевая переменная (будущая волатильность)
    """
    # Извлекаем признаки
    df_features = self.feature_calculator.extract_features(data)

    # Определяем признаки для модели
    feature_columns = [
      'volatility_5', 'volatility_10', 'volatility_20', 'volatility_50',
      'vol_ratio_5_20', 'vol_ratio_10_20', 'vol_ratio_20_50',
      'vol_sma_5', 'vol_sma_10', 'vol_ema_5', 'vol_ema_10',
      'vol_std_20', 'vol_skew_20', 'vol_kurt_20',
      'price_change_1', 'price_change_5', 'price_change_20',
      'volume_volatility_20', 'volume_price_correlation_20',
      'rsi_14', 'bb_position', 'atr_14',
      'hour_of_day', 'day_of_week', 'is_weekend'
    ]

    # Создаем целевую переменную (будущая волатильность)
    target_col = f'volatility_{self.prediction_horizon}_ahead'
    df_features[target_col] = df_features['volatility_20'].shift(-self.prediction_horizon)

    # Убираем NaN значения
    df_clean = df_features.dropna()

    X = df_clean[feature_columns]
    y = df_clean[target_col]

    return X, y

  # def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
  #   """
  #   Обучение моделей на уже подготовленных и разделенных данных.
  #   """
  #   if len(X_train) < 100:
  #     raise ValueError("Недостаточно данных для обучения (минимум 100 наблюдений)")
  #
  #   # Сохраняем эталонные признаки
  #   self.feature_names_in_ = X_train.columns.tolist()
  #   model_scores = {}
  #
  #   # Обучение каждой модели
  #   for model_name, model in self.models.items():
  #     try:
  #       # Масштабирование признаков
  #       X_train_scaled = self.scalers[model_name].fit_transform(X_train)
  #       X_test_scaled = self.scalers[model_name].transform(X_test)
  #
  #       # Обучение модели
  #       model.fit(X_train_scaled, y_train)
  #
  #       # Предсказание на тестовой выборке
  #       y_pred = model.predict(X_test_scaled)
  #
  #       # Расчет метрик
  #       mse = mean_squared_error(y_test, y_pred)
  #       mae = mean_absolute_error(y_test, y_pred)
  #       r2 = r2_score(y_test, y_pred)
  #
  #       model_scores[model_name] = {
  #         'mse': mse, 'mae': mae, 'r2': r2, 'rmse': np.sqrt(mse)
  #       }
  #
  #       # Сохранение важности признаков
  #       if hasattr(model, 'feature_importances_'):
  #         self.feature_importance[model_name] = dict(zip(X_train.columns, model.feature_importances_))
  #
  #       print(f"Модель {model_name}: R² = {r2:.4f}, RMSE = {np.sqrt(mse):.6f}")
  #
  #     except Exception as e:
  #       print(f"Ошибка при обучении модели {model_name}: {e}")
  #       model_scores[model_name] = {'error': str(e)}
  #
  #   # Определение перцентилей для режимов волатильности на всей выборке y
  #   full_y = pd.concat([y_train, y_test])
  #   self.volatility_percentiles = {
  #     'very_low': np.percentile(full_y, 10),
  #     'low': np.percentile(full_y, 25),
  #     'normal_high': np.percentile(full_y, 75),
  #     'high': np.percentile(full_y, 90)
  #   }
  #
  #   self.is_fitted = True
  #   return model_scores

  def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Обучение моделей на уже подготовленных и разделенных данных.
    """
    # Сохраняем имена признаков для последующей проверки
    self.feature_names_in_ = X_train.columns.tolist()
    self.feature_names = self.feature_names_in_  # Для совместимости с get_prediction

    model_scores = {}

    # Обучение каждой модели
    for model_name, model in self.models.items():
      try:
        X_train_scaled = self.scalers[model_name].fit_transform(X_train)
        X_test_scaled = self.scalers[model_name].transform(X_test)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Расчет метрик
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_scores[model_name] = {'r2': r2, 'rmse': np.sqrt(mse)}

        if hasattr(model, 'feature_importances_'):
          self.feature_importance[model_name] = dict(zip(X_train.columns, model.feature_importances_))
      except Exception as e:
        model_scores[model_name] = {'error': str(e)}

    # Определение перцентилей
    full_y = pd.concat([y_train, y_test])
    self.volatility_percentiles = {
      'very_low': np.percentile(full_y, 10),
      'low': np.percentile(full_y, 25),
      'normal_high': np.percentile(full_y, 75),
      'high': np.percentile(full_y, 90)
    }

    # Сохраняем также основной скейлер для использования в get_prediction
    self.scaler = self.scalers.get('lightgbm', list(self.scalers.values())[0])

    self.is_fitted = True

    # Логируем информацию об обучении
    logger.info(f"Предиктор волатильности обучен с {len(self.feature_names)} признаками")
    logger.info(f"Результаты моделей: {model_scores}")

    return model_scores

  def predict(self, data: pd.DataFrame,
              use_ensemble: bool = True) -> VolatilityPrediction:
    """
    Прогнозирование волатильности

    Args:
        data: Текущие данные
        use_ensemble: Использовать ансамбль моделей

    Returns:
        Прогноз волатильности
    """
    if not self.is_fitted:
      raise ValueError("Модель не обучена. Сначала вызовите fit()")

    # Извлекаем признаки
    df_features = self.feature_calculator.extract_features(data)

    # Берем последнее наблюдение
    last_features = df_features.iloc[-1]

    feature_columns = [
      'volatility_5', 'volatility_10', 'volatility_20', 'volatility_50',
      'vol_ratio_5_20', 'vol_ratio_10_20', 'vol_ratio_20_50',
      'vol_sma_5', 'vol_sma_10', 'vol_ema_5', 'vol_ema_10',
      'vol_std_20', 'vol_skew_20', 'vol_kurt_20',
      'price_change_1', 'price_change_5', 'price_change_20',
      'volume_volatility_20', 'volume_price_correlation_20',
      'rsi_14', 'bb_position', 'atr_14',
      'hour_of_day', 'day_of_week', 'is_weekend'
    ]

    X = last_features[feature_columns].values.reshape(1, -1)
    current_volatility = last_features['volatility_20']

    predictions = {}

    # Получаем предсказания от всех моделей
    for model_name, model in self.models.items():
      try:
        X_scaled = self.scalers[model_name].transform(X)
        pred = model.predict(X_scaled)[0]
        predictions[model_name] = pred
      except Exception as e:
        print(f"Ошибка при предсказании модели {model_name}: {e}")
        continue

    if not predictions:
      raise ValueError("Не удалось получить предсказания ни от одной модели")

    # Выбираем финальное предсказание
    if use_ensemble and len(predictions) > 1:
      # Простое усреднение (можно улучшить взвешенным усреднением)
      predicted_volatility = np.mean(list(predictions.values()))
      model_used = "ensemble"

      # Расчет уверенности на основе согласованности моделей
      pred_std = np.std(list(predictions.values()))
      confidence = max(0.1, 1.0 - (pred_std / predicted_volatility))
    else:
      # Используем лучшую модель (например, XGBoost по умолчанию)
      best_model = 'xgboost' if 'xgboost' in predictions else list(predictions.keys())[0]
      predicted_volatility = predictions[best_model]
      model_used = best_model
      confidence = 0.7  # Базовая уверенность

    # Определяем режим волатильности
    volatility_regime = self._determine_volatility_regime(predicted_volatility)

    # --- НОВАЯ СТРОКА ЛОГИРОВАНИЯ ---
    logger.info(
      f"Прогноз волатильности: Текущая=, Предсказанная=, Режим=''")
    # --- КОНЕЦ НОВОЙ СТРОКИ ---

    return VolatilityPrediction(
      predicted_volatility=predicted_volatility,
      current_volatility=current_volatility,
      volatility_regime=volatility_regime,
      confidence=confidence,
      prediction_horizon=self.prediction_horizon,
      model_used=model_used,
      timestamp=datetime.now()
    )

  def _determine_volatility_regime(self, volatility: float) -> VolatilityRegime:
    """Определение режима волатильности"""
    if volatility <= self.volatility_percentiles['very_low']:
      return VolatilityRegime.VERY_LOW
    elif volatility <= self.volatility_percentiles['low']:
      return VolatilityRegime.LOW
    elif volatility <= self.volatility_percentiles['normal_high']:
      return VolatilityRegime.NORMAL
    elif volatility <= self.volatility_percentiles['high']:
      return VolatilityRegime.HIGH
    else:
      return VolatilityRegime.VERY_HIGH


# ============================================================================
# 4. ИНТЕГРАЦИЯ С RISK MANAGER
# ============================================================================

class EnhancedRiskManager:
  """Улучшенный Risk Manager с учетом прогнозов волатильности"""

  def __init__(self, base_stop_loss: float = 0.02, base_take_profit: float = 0.04):
    self.base_stop_loss = base_stop_loss
    self.base_take_profit = base_take_profit
    self.volatility_predictor = None

    # Коэффициенты для адаптации на основе волатильности
    self.volatility_multipliers = {
      VolatilityRegime.VERY_LOW: {
        'stop_loss': 0.7,  # Уменьшаем SL при низкой волатильности
        'take_profit': 0.8,  # Уменьшаем TP
        'position_size': 1.2  # Увеличиваем размер позиции
      },
      VolatilityRegime.LOW: {
        'stop_loss': 0.85,
        'take_profit': 0.9,
        'position_size': 1.1
      },
      VolatilityRegime.NORMAL: {
        'stop_loss': 1.0,
        'take_profit': 1.0,
        'position_size': 1.0
      },
      VolatilityRegime.HIGH: {
        'stop_loss': 1.3,
        'take_profit': 1.4,
        'position_size': 0.8
      },
      VolatilityRegime.VERY_HIGH: {
        'stop_loss': 1.6,  # Увеличиваем SL при высокой волатильности
        'take_profit': 1.8,  # Увеличиваем TP
        'position_size': 0.6  # Уменьшаем размер позиции
      }
    }

  def set_volatility_predictor(self, predictor: VolatilityPredictor):
    """Установка предиктора волатильности"""
    self.volatility_predictor = predictor

  def calculate_adaptive_levels(self,
                                current_price: float,
                                signal_type: str,
                                market_data: pd.DataFrame,
                                base_position_size: float = 1.0) -> Dict[str, float]:
    """
    Расчет адаптивных уровней SL/TP и размера позиции

    Args:
        current_price: Текущая цена
        signal_type: Тип сигнала (BUY/SELL)
        market_data: Рыночные данные
        base_position_size: Базовый размер позиции

    Returns:
        Словарь с адаптированными параметрами
    """
    result = {
      'stop_loss': self.base_stop_loss,
      'take_profit': self.base_take_profit,
      'position_size': base_position_size,
      'volatility_regime': 'NORMAL',
      'confidence': 0.5
    }

    # Если предиктор волатильности недоступен, возвращаем базовые значения
    if self.volatility_predictor is None or not self.volatility_predictor.is_fitted:
      return result

    try:
      # Получаем прогноз волатильности
      vol_prediction = self.volatility_predictor.predict(market_data)

      # Получаем мультипликаторы для данного режима волатильности
      multipliers = self.volatility_multipliers[vol_prediction.volatility_regime]

      # Адаптируем параметры
      adapted_sl = self.base_stop_loss * multipliers['stop_loss']
      adapted_tp = self.base_take_profit * multipliers['take_profit']
      adapted_size = base_position_size * multipliers['position_size']

      # Дополнительная корректировка на основе изменения волатильности
      vol_change_factor = vol_prediction.volatility_change_ratio

      if vol_change_factor > 1.5:  # Ожидается резкий рост волатильности
        adapted_sl *= 1.2
        adapted_tp *= 1.3
        adapted_size *= 0.9
      elif vol_change_factor < 0.7:  # Ожидается снижение волатильности
        adapted_sl *= 0.9
        adapted_tp *= 0.9
        adapted_size *= 1.1

      # Учитываем уверенность модели
      confidence_factor = vol_prediction.confidence
      if confidence_factor < 0.6:
        # При низкой уверенности возвращаемся к базовым значениям
        adapted_sl = (adapted_sl + self.base_stop_loss) / 2
        adapted_tp = (adapted_tp + self.base_take_profit) / 2
        adapted_size = (adapted_size + base_position_size) / 2

      result.update({
        'stop_loss': adapted_sl,
        'take_profit': adapted_tp,
        'position_size': adapted_size,
        'volatility_regime': vol_prediction.volatility_regime.value,
        'confidence': vol_prediction.confidence,
        'predicted_volatility': vol_prediction.predicted_volatility,
        'current_volatility': vol_prediction.current_volatility,
        'volatility_change_ratio': vol_prediction.volatility_change_ratio
      })

    except Exception as e:
      print(f"Ошибка при адаптации уровней: {e}")
      # Возвращаем базовые значения при ошибке

    return result


# ============================================================================
# 5. СИСТЕМА МОНИТОРИНГА И ОБНОВЛЕНИЯ МОДЕЛИ
# ============================================================================

class VolatilityModelMonitor:
  """Класс для мониторинга качества модели и автоматического переобучения"""

  def __init__(self, predictor: VolatilityPredictor,
               retrain_threshold: float = 0.1):
    self.predictor = predictor
    self.retrain_threshold = retrain_threshold  # R² должен быть выше этого значения
    self.prediction_history = []
    self.actual_values = []
    self.last_retrain_time = datetime.now()
    self.retrain_interval = timedelta(days=7)  # Переобучение раз в неделю

  def add_prediction_result(self, prediction: VolatilityPrediction,
                            actual_volatility: float):
    """Добавление результата предсказания для мониторинга"""
    self.prediction_history.append({
      'timestamp': prediction.timestamp,
      'predicted': prediction.predicted_volatility,
      'actual': actual_volatility,
      'regime': prediction.volatility_regime.value,
      'confidence': prediction.confidence
    })

    # Ограничиваем историю последними 1000 записями
    if len(self.prediction_history) > 1000:
      self.prediction_history = self.prediction_history[-1000:]

  def calculate_model_performance(self) -> Dict[str, float]:
    """Расчет текущего качества модели"""
    if len(self.prediction_history) < 10:
      return {'insufficient_data': True}

    predicted = [p['predicted'] for p in self.prediction_history]
    actual = [p['actual'] for p in self.prediction_history]

    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    return {
      'mse': mse,
      'mae': mae,
      'r2': r2,
      'rmse': np.sqrt(mse),
      'sample_size': len(predicted)
    }

  def should_retrain(self) -> bool:
    """Проверка необходимости переобучения модели"""
    # Проверка по времени
    time_check = datetime.now() - self.last_retrain_time > self.retrain_interval

    # Проверка по качеству
    performance = self.calculate_model_performance()
    if 'r2' in performance:
      quality_check = performance['r2'] < self.retrain_threshold
    else:
      quality_check = False

    return time_check or quality_check

  def retrain_model(self, new_data: pd.DataFrame) -> Dict[str, float]:
      """Переобучение модели на новых данных"""
      try:
        print(f"Начинаем переобучение модели: {datetime.now()}")

        # Переобучаем модель
        scores = self.predictor.fit(new_data)

        # Обновляем время последнего переобучения
        self.last_retrain_time = datetime.now()

        # Очищаем историю предсказаний
        self.prediction_history = []

        print(f"Переобучение завершено: {datetime.now()}")
        return scores

      except Exception as e:
        print(f"Ошибка при переобучении: {e}")
        return {'error': str(e)}


# ============================================================================
# 6. ОСНОВНОЙ КЛАСС СИСТЕМЫ ПРОГНОЗИРОВАНИЯ
# ============================================================================

class VolatilityPredictionSystem:
  """Главный класс системы прогнозирования волатильности"""

  def __init__(self,
               model_type: ModelType = ModelType.LIGHTGBM,
               prediction_horizon: int = 5,
               auto_retrain: bool = True):
    """
    Инициализация системы

    Args:
        model_type: Тип модели для прогнозирования
        prediction_horizon: Горизонт прогнозирования (в периодах)
        auto_retrain: Автоматическое переобучение
    """
    self.predictor = VolatilityPredictor(model_type, prediction_horizon)
    self.risk_manager = EnhancedRiskManager()
    self.monitor = VolatilityModelMonitor(self.predictor)
    self.auto_retrain = auto_retrain

    # Подключаем предиктор к риск-менеджеру
    self.risk_manager.set_volatility_predictor(self.predictor)

    # История данных для переобучения
    self.data_history = pd.DataFrame()
    self.max_history_size = 10000  # Максимальный размер истории

  # def initialize(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[
  #   str, Any]:
  #   """
  #   ФИНАЛЬНАЯ ВЕРСИЯ: Инициализация системы с уже подготовленными и разделенными
  #   данными для обучения и тестирования.
  #   """
  #   try:
  #     # Проверяем качество данных
  #     if len(X_train) < 200:
  #       raise ValueError("Недостаточно данных для обучения (минимум 200 записей)")
  #
  #     # Обучаем модель, передавая ей и обучающую, и тестовую выборки
  #     scores = self.predictor.fit(X_train, y_train, X_test, y_test)
  #
  #     # Сохраняем историю данных (полный набор до разделения)
  #     self.data_history = pd.concat([X_train, X_test])
  #
  #     return {
  #       'status': 'success',
  #       'model_scores': scores,
  #       'train_data_points': len(X_train),
  #       'test_data_points': len(X_test),
  #       'features_count': len(X_train.columns)
  #     }
  #
  #   except Exception as e:
  #     return {
  #       'status': 'error',
  #       'error': str(e)
  #     }

  # def initialize(self, historical_data: pd.DataFrame) -> Dict[str, any]:
  #   """
  #   Инициализация системы с историческими данными.
  #   Теперь этот метод управляет всем процессом.
  #   """
  #   try:
  #     if len(historical_data) < 200:
  #       raise ValueError("Недостаточно исторических данных (минимум 200 записей)")
  #
  #     # Обучаем предиктор, он сам подготовит данные
  #     scores = self.predictor.fit(historical_data)
  #
  #     return {
  #       'status': 'success',
  #       'model_scores': scores,
  #       'data_points': len(historical_data)
  #     }
  #   except Exception as e:
  #     return {'status': 'error', 'error': str(e)}

  def initialize(self, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """
    Инициализация системы с уже подготовленными и разделенными данными.
    """
    try:
      if len(X_train) < 200:
        raise ValueError("Недостаточно данных для обучения")

      # Обучаем предиктор
      scores = self.predictor.fit(X_train, y_train, X_test, y_test)

      return {'status': 'success', 'model_scores': scores}
    except Exception as e:
      return {'status': 'error', 'error': str(e)}


  def update_data(self, new_data: pd.DataFrame):
    """Обновление данных системы"""
    # Добавляем новые данные к истории
    self.data_history = pd.concat([self.data_history, new_data], ignore_index=True)

    # Ограничиваем размер истории
    if len(self.data_history) > self.max_history_size:
      self.data_history = self.data_history.tail(self.max_history_size)

    # Проверяем необходимость переобучения
    if self.auto_retrain and self.monitor.should_retrain():
      self.monitor.retrain_model(self.data_history)

  # def get_volatility_prediction(self, current_data: pd.DataFrame) -> VolatilityPrediction:
  #   """
  #   Получение прогноза волатильности
  #
  #   Args:
  #       current_data: Текущие рыночные данные
  #
  #   Returns:
  #       Прогноз волатильности
  #   """
  #   if not self.predictor.is_fitted:
  #     raise ValueError("Система не инициализирована. Вызовите initialize() сначала")
  #
  #   return self.predictor.predict(current_data)

  def get_prediction(self, market_data: pd.DataFrame) -> Optional[VolatilityPrediction]:
    """Получает прогноз волатильности"""
    if not self.predictor.is_fitted:
      logger.warning("Предиктор волатильности не обучен")
      return None

    try:
      # Создаем признаки
      features = self._create_features(market_data)
      if features is None or features.empty:
        return None

      # Проверяем количество признаков
      expected_features = self.scaler.n_features_in_
      actual_features = len(features.columns)

      if actual_features != expected_features:
        logger.warning(
          f"Несоответствие признаков: получено {actual_features}, "
          f"ожидалось {expected_features}. Используем только совпадающие."
        )

        # Пытаемся использовать только те признаки, которые есть в обученной модели
        if hasattr(self, 'feature_names'):
          common_features = [f for f in self.feature_names if f in features.columns]
          if len(common_features) > 0:
            features = features[common_features]
            # Добавляем недостающие признаки как нули
            for missing_feature in self.feature_names:
              if missing_feature not in features.columns:
                features[missing_feature] = 0
            # Переупорядочиваем колонки
            features = features[self.feature_names]
          else:
            logger.error("Нет общих признаков между моделью и данными")
            return None
        else:
          logger.error("Список признаков модели не сохранен")
          return None

      # Масштабируем признаки
      features_scaled = self.scaler.transform(features)

      # Получаем предсказание
      prediction = self.predictor.predict(features_scaled)[0]

      # Определяем режим волатильности
      regime = self._classify_volatility_regime(prediction)

      return VolatilityPrediction(
        timestamp=datetime.now(),
        symbol=market_data.index[-1] if hasattr(market_data.index[-1], '__str__') else "UNKNOWN",
        predicted_volatility=prediction,
        confidence=0.8,  # Можно добавить расчет уверенности
        volatility_regime=regime,
        horizon_minutes=60
      )

    except Exception as e:
      logger.error(f"Ошибка при получении прогноза волатильности: {e}")
      return None

  def _classify_volatility_regime(self, volatility: float) -> VolatilityRegime:
    """Классифицирует режим волатильности на основе значения"""
    # Пороги можно настроить под конкретный рынок
    if volatility < 0.005:  # 0.5%
      return VolatilityRegime.VERY_LOW
    elif volatility < 0.01:  # 1%
      return VolatilityRegime.LOW
    elif volatility < 0.02:  # 2%
      return VolatilityRegime.NORMAL
    elif volatility < 0.03:  # 3%
      return VolatilityRegime.HIGH
    else:
      return VolatilityRegime.VERY_HIGH

  def get_adaptive_risk_parameters(self,
                                   current_price: float,
                                   signal_type: str,
                                   market_data: pd.DataFrame,
                                   base_position_size: float = 1.0) -> Dict[str, float]:
    """
    Получение адаптивных параметров риска

    Args:
        current_price: Текущая цена
        signal_type: Тип сигнала
        market_data: Рыночные данные
        base_position_size: Базовый размер позиции

    Returns:
        Адаптированные параметры риска
    """
    return self.risk_manager.calculate_adaptive_levels(
      current_price, signal_type, market_data, base_position_size
    )

  def add_prediction_feedback(self, prediction: VolatilityPrediction,
                              actual_volatility: float):
    """Добавление обратной связи для мониторинга качества"""
    self.monitor.add_prediction_result(prediction, actual_volatility)

  def get_system_status(self) -> Dict[str, any]:
    """Получение статуса системы"""
    performance = self.monitor.calculate_model_performance()

    return {
      'is_fitted': self.predictor.is_fitted,
      'model_type': self.predictor.model_type.value,
      'prediction_horizon': self.predictor.prediction_horizon,
      'data_history_size': len(self.data_history),
      'predictions_count': len(self.monitor.prediction_history),
      'last_retrain': self.monitor.last_retrain_time,
      'model_performance': performance,
      'should_retrain': self.monitor.should_retrain(),
      'feature_importance': self.predictor.feature_importance
    }


# ============================================================================
# 7. УТИЛИТЫ И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def create_sample_data(n_points: int = 1000) -> pd.DataFrame:
  """Создание тестовых данных для демонстрации"""
  np.random.seed(42)

  # Генерируем синтетические данные
  dates = pd.date_range(start='2023-01-01', periods=n_points, freq='1H')

  # Базовый тренд
  base_price = 100
  trend = np.cumsum(np.random.normal(0, 0.001, n_points))

  # Волатильность с режимами
  volatility_regimes = np.random.choice([0.5, 1.0, 1.5, 2.0], n_points,
                                        p=[0.3, 0.4, 0.2, 0.1])

  # Генерируем цены
  returns = np.random.normal(0, 0.01, n_points) * volatility_regimes
  prices = base_price * np.exp(np.cumsum(returns + trend))

  # Создаем OHLCV данные
  data = pd.DataFrame({
    'timestamp': dates,
    'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
    'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_points))),
    'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_points))),
    'close': prices,
    'volume': np.random.lognormal(10, 0.5, n_points)
  })

  data.set_index('timestamp', inplace=True)
  return data


def analyze_volatility_patterns(data: pd.DataFrame, predictor: VolatilityPredictor) -> Dict[str, any]:
  """Анализ паттернов волатильности"""
  if not predictor.is_fitted:
    raise ValueError("Модель не обучена")

  # Извлекаем признаки
  features_df = predictor.feature_calculator.extract_features(data)

  # Анализ корреляций
  vol_features = [col for col in features_df.columns if 'volatility' in col or 'vol_' in col]
  correlation_matrix = features_df[vol_features].corr()

  # Анализ важности признаков
  importance_analysis = {}
  for model_name, importance in predictor.feature_importance.items():
    # Топ-10 важных признаков
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    importance_analysis[model_name] = sorted_importance[:10]

  # Статистика режимов волатильности
  volatility_20 = features_df['volatility_20'].dropna()
  regime_stats = {
    'mean': volatility_20.mean(),
    'std': volatility_20.std(),
    'percentiles': {
      '10': np.percentile(volatility_20, 10),
      '25': np.percentile(volatility_20, 25),
      '50': np.percentile(volatility_20, 50),
      '75': np.percentile(volatility_20, 75),
      '90': np.percentile(volatility_20, 90)
    }
  }

  return {
    'correlation_matrix': correlation_matrix,
    'feature_importance': importance_analysis,
    'volatility_statistics': regime_stats,
    'data_points': len(features_df)
  }


# ============================================================================
# 8. ДЕМОНСТРАЦИОННЫЙ ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

def demonstration_example():
  """Демонстрация использования системы прогнозирования волатильности"""
  print("=== ДЕМОНСТРАЦИЯ СИСТЕМЫ ПРОГНОЗИРОВАНИЯ ВОЛАТИЛЬНОСТИ ===\n")

  # 1. Создание тестовых данных
  print("1. Создание тестовых данных...")
  data = create_sample_data(2000)
  print(f"Создано {len(data)} точек данных")
  print(f"Период: {data.index[0]} - {data.index[-1]}\n")

  # 2. Инициализация системы
  print("2. Инициализация системы...")
  system = VolatilityPredictionSystem(
    model_type=ModelType.LIGHTGBM,
    prediction_horizon=5,
    auto_retrain=True
  )

  # Разделяем данные на обучающие и тестовые
  split_point = int(len(data) * 0.8)
  train_data = data[:split_point]
  test_data = data[split_point:]

  # 3. Обучение системы
  print("3. Обучение системы...")
  init_result = system.initialize(train_data)

  if init_result['status'] == 'success':
    print("✓ Система успешно инициализирована")
    print(f"Обучено на {init_result['data_points']} точках данных")
    print(f"Использовано {init_result['features_count']} признаков")

    # Показываем качество моделей
    print("\nКачество моделей:")
    for model_name, scores in init_result['model_scores'].items():
      if 'error' not in scores:
        print(f"  {model_name}: R² = {scores['r2']:.4f}, RMSE = {scores['rmse']:.6f}")
  else:
    print(f"✗ Ошибка инициализации: {init_result['error']}")
    return

  # 4. Тестирование прогнозов
  print("\n4. Тестирование прогнозов...")
  predictions = []

  for i in range(min(50, len(test_data) - 10)):  # Тестируем на 50 точках
    current_data = data[:split_point + i + 1]  # Данные до текущего момента

    try:
      # Получаем прогноз
      prediction = system.get_prediction(current_data)
      predictions.append(prediction)

      # Получаем адаптивные параметры риска
      risk_params = system.get_adaptive_risk_parameters(
        current_price=current_data['close'].iloc[-1],
        signal_type='BUY',
        market_data=current_data,
        base_position_size=1.0
      )

      if i % 10 == 0:  # Показываем каждый 10-й прогноз
        print(f"\nПрогноз {i + 1}:")
        print(f"  Текущая волатильность: {prediction.current_volatility:.6f}")
        print(f"  Прогнозируемая волатильность: {prediction.predicted_volatility:.6f}")
        print(f"  Режим: {prediction.volatility_regime.value}")
        print(f"  Уверенность: {prediction.confidence:.2f}")
        print(f"  Адаптивный SL: {risk_params['stop_loss']:.4f}")
        print(f"  Адаптивный TP: {risk_params['take_profit']:.4f}")
        print(f"  Размер позиции: {risk_params['position_size']:.2f}")

    except Exception as e:
      print(f"Ошибка при прогнозировании {i}: {e}")

  # 5. Анализ результатов
  print(f"\n5. Анализ результатов ({len(predictions)} прогнозов)...")

  if predictions:
    # Распределение по режимам волатильности
    regime_counts = {}
    for pred in predictions:
      regime = pred.volatility_regime.value
      regime_counts[regime] = regime_counts.get(regime, 0) + 1

    print("\nРаспределение по режимам волатильности:")
    for regime, count in regime_counts.items():
      percentage = (count / len(predictions)) * 100
      print(f"  {regime}: {count} ({percentage:.1f}%)")

    # Средняя уверенность
    avg_confidence = np.mean([p.confidence for p in predictions])
    print(f"\nСредняя уверенность модели: {avg_confidence:.3f}")

    # Изменения волатильности
    vol_changes = [p.volatility_change_ratio for p in predictions]
    avg_change = np.mean(vol_changes)
    print(f"Средний коэффициент изменения волатильности: {avg_change:.3f}")

  # 6. Статус системы
  print("\n6. Статус системы:")
  status = system.get_system_status()
  print(f"  Модель обучена: {status['is_fitted']}")
  print(f"  Тип модели: {status['model_type']}")
  print(f"  Горизонт прогнозирования: {status['prediction_horizon']}")
  print(f"  Размер истории данных: {status['data_history_size']}")
  print(f"  Количество прогнозов: {status['predictions_count']}")

  # 7. Анализ паттернов
  print("\n7. Анализ паттернов волатильности...")
  try:
    patterns = analyze_volatility_patterns(data, system.predictor)
    print(f"  Средняя волатильность: {patterns['volatility_statistics']['mean']:.6f}")
    print(f"  Стандартное отклонение: {patterns['volatility_statistics']['std']:.6f}")

    print("\n  Перцентили волатильности:")
    for p, value in patterns['volatility_statistics']['percentiles'].items():
      print(f"    {p}%: {value:.6f}")

  except Exception as e:
    print(f"  Ошибка анализа паттернов: {e}")

  print("\n=== ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА ===")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ ДЛЯ ЗАПУСКА ДЕМОНСТРАЦИИ
# ============================================================================

if __name__ == "__main__":
  # Запуск демонстрации
  demonstration_example()

  print("\n" + "=" * 60)
  print("СИСТЕМА ПРОГНОЗИРОВАНИЯ ВОЛАТИЛЬНОСТИ ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
  print("=" * 60)

  # Краткое руководство по использованию
  print("""
  КРАТКОЕ РУКОВОДСТВО ПО ИСПОЛЬЗОВАНИЮ:

  1. Инициализация системы:
     system = VolatilityPredictionSystem()
     system.initialize(historical_data)

  2. Получение прогноза:
     prediction = system.get_volatility_prediction(current_data)

  3. Адаптивные параметры риска:
     risk_params = system.get_adaptive_risk_parameters(
         current_price, signal_type, market_data
     )

  4. Мониторинг системы:
     status = system.get_system_status()

  Для интеграции с торговым ботом используйте методы:
  - get_volatility_prediction() для получения прогнозов
  - get_adaptive_risk_parameters() для адаптации SL/TP
  - add_prediction_feedback() для улучшения модели
      """)

  # # Инициализация
  # system = VolatilityPredictionSystem()
  # system.initialize(historical_data)
  #
  # # Получение прогноза
  # prediction = system.get_volatility_prediction(current_data)
  #
  # # Адаптивные параметры риска
  # risk_params = system.get_adaptive_risk_parameters(
  #   current_price=1000.0,
  #   signal_type='BUY',
  #   market_data=current_data
  # )