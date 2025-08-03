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
from sklearn.linear_model import LogisticRegression
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


  def create_advanced_features(self, data: pd.DataFrame,
                               external_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """
    ДИАГНОСТИЧЕСКАЯ ВЕРСИЯ: Ищет, какая из вспомогательных функций
    возвращает DataFrame неправильного размера.
    """
    # logger.info("=" * 20 + " ЗАПУСК ДИАГНОСТИЧЕСКОЙ ВЕРСИИ " + "=" * 20)

    # --- Шаг 0: Дедупликация индекса (оставляем как лучшую практику) ---
    if not data.index.is_unique:
      logger.warning(f"Обнаружен неуникальный индекс. Размер до: {len(data)}. Дедупликация...")
      data = data.loc[~data.index.duplicated(keep='first')]
      logger.info(f"Размер после дедупликации: {len(data)}")

    original_index = data.index.copy()
    original_length = len(data)
    # logger.info(f"Ожидаемый размер для всех наборов признаков: {original_length} строк.")

    base_columns = ['open', 'high', 'low', 'close', 'volume']
    available_base = [col for col in base_columns if col in data.columns]
    base_data = data[available_base].copy() if available_base else None
    # if base_data is not None:
    #   pass
      # logger.info(f"Проверка базовых данных: shape={base_data.shape}. ✅")

    # --- Вспомогательная функция для проверки и логирования ---
    def check_and_add(feature_df, name, collection):
      if feature_df is not None and not feature_df.empty:
        # logger.info(f"--> Проверка набора '{name}': shape={feature_df.shape}")
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
      # logger.info("Все наборы признаков сгенерированы. Начинаю финальное объединение...")
      all_parts = [base_data] + all_features_list if base_data is not None else all_features_list

      if all_parts:
        features = pd.concat(all_parts, axis=1)
      else:
        features = pd.DataFrame(index=original_index)

      # logger.info(f"Финальный размер DataFrame после concat: {features.shape}")
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
    self.market_filter = MarketLogicFilter()
    self.temporal_manager = TemporalDataManager()
    self.use_temporal_management = True
    self.use_market_filters = True  # Флаг для включения/отключения фильтров

    self.balancing_method = 'smote'  # 'smote', 'class_weight', 'none', 'adaptive'
    # Базовые модели
    self.models = {
      'rf': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        # class_weight='balanced',  # ДОБАВЛЕНО: автоматическая балансировка

        random_state=42,
        n_jobs=-1
      ),
      'gb': HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=20,
        # class_weight='balanced',
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
        # class_weight='balanced',
        random_state=42,
        verbose=-1
      )
    }

    # Мета-модель для стекинга
    # self.meta_model = xgb.XGBClassifier(
    #   n_estimators=50,
    #   learning_rate=0.1,
    #   max_depth=3,
    #   random_state=42,
    #   use_label_encoder=False,
    #   eval_metric='logloss'
    # )
    self.meta_model = LogisticRegression(
      C=1.0,
      class_weight='balanced',
      random_state=42,
      max_iter=1000,
      solver='liblinear'  # Лучше работает с малыми данными
    )
    self.backup_meta_model = xgb.XGBClassifier(
      n_estimators=30,  # Уменьшено для предотвращения переобучения
      learning_rate=0.05,  # Более консервативный learning rate
      max_depth=2,  # Ограничиваем глубину
      min_child_weight=5,  # Увеличиваем для регуляризации
      subsample=0.8,
      colsample_bytree=0.8,
      reg_alpha=0.1,  # L1 регуляризация
      reg_lambda=0.1,  # L2 регуляризация
      random_state=42,
      eval_metric='mlogloss'
    )
    # Параметры мета-обучения
    self.meta_model_config = {
      'use_cross_validation': True,
      'cv_folds': 3,
      'validation_split': 0.2,
      'min_samples_for_meta': 100,
      'feature_selection': True,
      'ensemble_weights': True
    }

    # Статистики мета-модели
    self.meta_model_stats = {
      'training_accuracy': None,
      'validation_accuracy': None,
      'feature_importance': None,
      'is_reliable': False,
      'fallback_reason': None
    }

    # Скейлеры для разных групп признаков
    self.scaler = RobustScaler()  # RobustScaler более устойчив к выбросам в финансовых данных
    self.backup_scaler = StandardScaler()  # Резервный скейлер
    self.scaler_type_used = None  # Отслеживаем какой скейлер используется

    # Информация о признаках для валидации
    self.training_features = None  # Список признаков, использованных при обучении
    self.feature_statistics = None  # Статистики признаков для валидации

    # Селектор признаков
    self.feature_selector = None
    self.selected_features = None

    # Параметры для адаптивного обучения
    self.performance_history = []
    self.feature_importance_history = []
    self.is_fitted = False

  def _train_meta_model_safely(self, X_train_resampled: pd.DataFrame, y_train_resampled: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series) -> bool:
      """
      Безопасное обучение мета-модели с валидацией и фолбэк механизмом
      """
      logger.info("Обучение мета-модели с валидацией...")

      try:
        # 1. Получение предсказаний базовых моделей для обучения мета-модели
        meta_features_train, feature_names = self._extract_meta_features(X_train_resampled, 'train')
        meta_features_val, _ = self._extract_meta_features(X_val, 'validation')

        if meta_features_train is None or len(meta_features_train) == 0:
          logger.warning("Не удалось извлечь мета-признаки, мета-модель не будет обучена")
          self.meta_model_stats['fallback_reason'] = 'no_meta_features'
          return False

        # 2. Проверка достаточности данных
        min_samples = self.meta_model_config['min_samples_for_meta']
        if len(meta_features_train) < min_samples:
          logger.warning(f"Недостаточно данных для мета-модели: {len(meta_features_train)} < {min_samples}")
          self.meta_model_stats['fallback_reason'] = 'insufficient_data'
          return False

        # 3. Селекция признаков для мета-модели (если включена)
        if self.meta_model_config['feature_selection']:
          meta_features_train, meta_features_val = self._select_meta_features(
            meta_features_train, meta_features_val, y_train_resampled
          )

        # 4. Обучение мета-модели с кросс-валидацией
        if self.meta_model_config['use_cross_validation']:
          cv_scores = self._train_with_cross_validation(meta_features_train, y_train_resampled)
          logger.info(f"CV scores мета-модели: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

          if cv_scores.mean() < 0.4:  # Если мета-модель работает хуже случайного
            logger.warning("Мета-модель показывает плохие результаты на кросс-валидации")
            self.meta_model_stats['fallback_reason'] = 'poor_cv_performance'
            return False

        # 5. Финальное обучение мета-модели
        logger.info("Финальное обучение мета-модели...")
        # Проверяем, что мета-модель инициализирована
        if not hasattr(self, 'meta_model') or self.meta_model is None:
          # Инициализируем мета-модель если она отсутствует

          self.meta_model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            solver='liblinear'
          )
          logger.info("Мета-модель инициализирована автоматически")

        self.meta_model.fit(meta_features_train, y_train_resampled)

        # 6. Валидация на отложенной выборке
        if len(meta_features_val) > 0:
          val_predictions = self.meta_model.predict(meta_features_val)
          val_accuracy = accuracy_score(y_val, val_predictions)

          self.meta_model_stats['validation_accuracy'] = val_accuracy
          logger.info(f"Точность мета-модели на валидации: {val_accuracy:.4f}")

          # Проверяем, что мета-модель лучше случайного выбора
          random_baseline = 1.0 / len(np.unique(y_val))
          if val_accuracy < random_baseline * 1.1:  # Должна быть хотя бы на 10% лучше случайного
            logger.warning(f"Мета-модель {val_accuracy:.4f} не лучше базовой линии {random_baseline:.4f}")
            self.meta_model_stats['fallback_reason'] = 'poor_validation_performance'
            return False

        # 7. Анализ важности признаков мета-модели
        if hasattr(self.meta_model, 'coef_'):
          feature_importance = dict(zip(feature_names, abs(self.meta_model.coef_[0])))
          self.meta_model_stats['feature_importance'] = feature_importance
          logger.debug(f"Важность мета-признаков: {feature_importance}")

        # 8. Обучение резервной мета-модели для сравнения
        try:
          self.backup_meta_model.fit(meta_features_train, y_train_resampled)
          if len(meta_features_val) > 0:
            backup_predictions = self.backup_meta_model.predict(meta_features_val)
            backup_accuracy = accuracy_score(y_val, backup_predictions)

            # Если резервная модель лучше, используем ее
            if backup_accuracy > val_accuracy * 1.05:  # На 5% лучше
              logger.info(f"Резервная мета-модель лучше: {backup_accuracy:.4f} vs {val_accuracy:.4f}")
              self.meta_model = self.backup_meta_model
              self.meta_model_stats['validation_accuracy'] = backup_accuracy

        except Exception as backup_error:
          logger.warning(f"Ошибка обучения резервной мета-модели: {backup_error}")

        self.meta_model_stats['is_reliable'] = True
        logger.info("Мета-модель успешно обучена и валидирована")
        return True

      except Exception as e:
        logger.error(f"Критическая ошибка обучения мета-модели: {e}")
        self.meta_model_stats['fallback_reason'] = f'training_error: {str(e)}'
        return False

  def _extract_meta_features(self, X: pd.DataFrame, stage: str) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Извлекает мета-признаки из предсказаний базовых моделей
    """
    try:
      meta_features = []
      feature_names = []

      for name, model in self.models.items():
        try:
          if hasattr(model, 'predict_proba'):
            # Используем вероятности как признаки
            proba = model.predict_proba(X)

            # Добавляем все вероятности классов
            for class_idx in range(proba.shape[1]):
              meta_features.append(proba[:, class_idx])
              feature_names.append(f'{name}_proba_class_{class_idx}')

            # Добавляем максимальную вероятность и энтропию
            max_proba = np.max(proba, axis=1)
            entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)

            meta_features.extend([max_proba, entropy])
            feature_names.extend([f'{name}_max_proba', f'{name}_entropy'])

          else:
            # Для моделей без predict_proba используем предсказания классов
            pred = model.predict(X)
            # Преобразуем в one-hot encoding
            for class_val in [0, 1, 2]:
              class_indicator = (pred == class_val).astype(float)
              meta_features.append(class_indicator)
              feature_names.append(f'{name}_pred_class_{class_val}')

          logger.debug(f"Извлечены мета-признаки от модели {name} для {stage}")

        except Exception as model_error:
          logger.warning(f"Ошибка извлечения мета-признаков от {name}: {model_error}")
          continue

      if not meta_features:
        return None, []

      # Объединяем все признаки
      meta_features_array = np.column_stack(meta_features)

      # Проверяем на NaN и inf
      if not np.isfinite(meta_features_array).all():
        logger.warning("Обнаружены нефинитные значения в мета-признаках")
        meta_features_array = np.nan_to_num(meta_features_array, nan=0.0, posinf=1.0, neginf=0.0)

      logger.debug(f"Извлечено {meta_features_array.shape[1]} мета-признаков для {stage}")
      return meta_features_array, feature_names

    except Exception as e:
      logger.error(f"Ошибка извлечения мета-признаков: {e}")
      return None, []

  def _select_meta_features(self, meta_features_train: np.ndarray, meta_features_val: np.ndarray,
                            y_train: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Селекция наиболее важных мета-признаков
    """
    try:
      from sklearn.feature_selection import SelectKBest, f_classif

      # Выбираем лучшие признаки на основе F-статистики
      k_best = min(10, meta_features_train.shape[1])  # Максимум 10 признаков
      selector = SelectKBest(score_func=f_classif, k=k_best)

      meta_train_selected = selector.fit_transform(meta_features_train, y_train)
      meta_val_selected = selector.transform(meta_features_val)

      selected_features = selector.get_support(indices=True)
      logger.info(f"Выбрано {len(selected_features)} мета-признаков из {meta_features_train.shape[1]}")

      return meta_train_selected, meta_val_selected

    except Exception as e:
      logger.warning(f"Ошибка селекции мета-признаков: {e}, используем все признаки")
      return meta_features_train, meta_features_val

  def _train_with_cross_validation(self, X: np.ndarray, y: pd.Series) -> np.ndarray:
    """
    Обучение мета-модели с кросс-валидацией
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    cv_folds = self.meta_model_config['cv_folds']
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_scores = cross_val_score(self.meta_model, X, y, cv=cv, scoring='accuracy')
    return cv_scores

  def _safe_scaling(self, X: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
      """
      Безопасное скалирование с обработкой ошибок и выбором оптимального скейлера
      """
      try:
        if is_training:
          # При обучении выбираем оптимальный скейлер
          scaler_to_use = self._choose_optimal_scaler(X)
          self.scaler = scaler_to_use
          self.scaler_type_used = type(scaler_to_use).__name__

          logger.info(f"Выбран скейлер: {self.scaler_type_used}")

          # Обучаем скейлер
          X_scaled = self.scaler.fit_transform(X)
          X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

          # Сохраняем статистики для проверки
          self.feature_statistics = {
            'mean': X.mean().to_dict(),
            'std': X.std().to_dict(),
            'min': X.min().to_dict(),
            'max': X.max().to_dict(),
            'outlier_percentage': self._calculate_outlier_percentage(X)
          }

        else:
          # При предсказании используем уже обученный скейлер
          if self.scaler is None:
            raise ValueError("Скейлер не обучен. Сначала вызовите fit()")

          X_scaled = self.scaler.transform(X)
          X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Проверяем результат скалирования
        if not np.isfinite(X_scaled.values).all():
          logger.warning(f"Скалирование дало нефинитные значения с {self.scaler_type_used}")

          if is_training:
            # При обучении пробуем резервный скейлер
            logger.info("Переключаемся на резервный скейлер...")
            self.scaler = self.backup_scaler
            self.scaler_type_used = type(self.scaler).__name__
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
          else:
            # При предсказании заполняем проблемные значения
            X_scaled = X_scaled.fillna(0)
            X_scaled = X_scaled.replace([np.inf, -np.inf], 0)

        logger.debug(f"Скалирование завершено. Форма: {X_scaled.shape}, Скейлер: {self.scaler_type_used}")
        return X_scaled

      except Exception as e:
        logger.error(f"Критическая ошибка скалирования: {e}")

        if is_training:
          # При обучении возвращаем исходные данные с предупреждением
          logger.warning("Возвращаем данные без скалирования")
          self.scaler = None
          self.scaler_type_used = 'none'
          return X.fillna(0)
        else:
          # При предсказании используем простую нормализацию
          logger.warning("Применяем простую нормализацию")
          return self._simple_normalization(X)

  def _choose_optimal_scaler(self, X: pd.DataFrame) -> object:
    """
    Выбирает оптимальный скейлер на основе анализа данных
    """
    # Анализируем данные
    outlier_percentage = self._calculate_outlier_percentage(X)
    skewness = X.skew().abs().mean()

    logger.debug(f"Анализ данных для выбора скейлера:")
    logger.debug(f"  Процент выбросов: {outlier_percentage:.2f}%")
    logger.debug(f"  Средняя асимметрия: {skewness:.3f}")

    # Правила выбора скейлера
    if outlier_percentage > 10 or skewness > 2:
      logger.debug("  Выбран RobustScaler (много выбросов или высокая асимметрия)")
      return RobustScaler()
    else:
      logger.debug("  Выбран StandardScaler (данные относительно нормальные)")
      return StandardScaler()

  def _calculate_outlier_percentage(self, X: pd.DataFrame) -> float:
    """
    Рассчитывает процент выбросов в данных
    """
    try:
      numeric_cols = X.select_dtypes(include=[np.number]).columns
      if len(numeric_cols) == 0:
        return 0.0

      total_values = 0
      outlier_count = 0

      for col in numeric_cols:
        if X[col].nunique() <= 1:  # Пропускаем константные колонки
          continue

        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR > 0:  # Проверяем, что IQR не равен нулю
          lower_bound = Q1 - 1.5 * IQR
          upper_bound = Q3 + 1.5 * IQR

          col_outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
          outlier_count += col_outliers
          total_values += len(X[col].dropna())

      return (outlier_count / total_values * 100) if total_values > 0 else 0.0

    except Exception as e:
      logger.warning(f"Ошибка расчета выбросов: {e}")
      return 0.0

  def _simple_normalization(self, X: pd.DataFrame) -> pd.DataFrame:
    """
    Простая нормализация для случаев, когда стандартное скалирование не работает
    """
    try:
      X_norm = X.copy()

      for col in X_norm.select_dtypes(include=[np.number]).columns:
        col_std = X_norm[col].std()
        if col_std > 0:
          X_norm[col] = (X_norm[col] - X_norm[col].mean()) / col_std
        else:
          X_norm[col] = 0

      return X_norm.fillna(0)

    except Exception as e:
      logger.error(f"Ошибка простой нормализации: {e}")
      return X.fillna(0)

  # =================== НОВЫЙ МЕТОД ДЛЯ ПРОВЕРКИ СООТВЕТСТВИЯ ПРИЗНАКОВ ===================

  def _validate_feature_consistency(self, X: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Проверяет и обеспечивает соответствие признаков между обучением и предсказанием
    """
    if is_training:
      # При обучении сохраняем список признаков
      self.training_features = list(X.columns)
      logger.info(f"Сохранено {len(self.training_features)} признаков для обучения")
      return X

    # При предсказании проверяем соответствие
    if self.training_features is None:
      logger.warning("Информация о признаках обучения отсутствует")
      return X

    current_features = list(X.columns)

    # Находим отсутствующие и дополнительные признаки
    missing_features = set(self.training_features) - set(current_features)
    extra_features = set(current_features) - set(self.training_features)

    if missing_features:
      logger.warning(f"Отсутствуют признаки: {list(missing_features)[:10]}...")  # Показываем первые 10

      # Добавляем отсутствующие признаки с нулевыми значениями
      for feature in missing_features:
        X[feature] = 0.0

    if extra_features:
      logger.warning(f"Дополнительные признаки будут удалены: {list(extra_features)[:10]}...")

      # Удаляем дополнительные признаки
      X = X.drop(columns=extra_features)

    # Приводим к нужному порядку колонок
    X = X.reindex(columns=self.training_features, fill_value=0.0)

    logger.debug(f"Признаки приведены к соответствию: {X.shape}")
    return X

  def _get_optimal_balancing_strategy(self, y: pd.Series) -> str:
      """
      Определяет оптимальную стратегию балансировки на основе анализа данных
      """
      class_counts = y.value_counts()
      total_samples = len(y)

      # Рассчитываем коэффициент дисбаланса
      min_class_count = class_counts.min()
      max_class_count = class_counts.max()
      imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')

      logger.info(f"Анализ дисбаланса классов:")
      logger.info(f"  Распределение: {class_counts.to_dict()}")
      logger.info(f"  Коэффициент дисбаланса: {imbalance_ratio:.2f}")

      # Определяем стратегию
      if imbalance_ratio <= 2:
        strategy = 'none'
        logger.info("  Рекомендация: дисбаланс незначительный, балансировка не нужна")
      elif imbalance_ratio <= 5:
        strategy = 'class_weight'
        logger.info("  Рекомендация: умеренный дисбаланс, используем class_weight")
      elif imbalance_ratio <= 20:
        strategy = 'smote'
        logger.info("  Рекомендация: сильный дисбаланс, используем SMOTE")
      else:
        strategy = 'adaptive'
        logger.info("  Рекомендация: критический дисбаланс, используем адаптивный подход")

      return strategy

  def _apply_balancing_strategy(self, X_train: pd.DataFrame, y_train: pd.Series,
                                strategy: str) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Применяет выбранную стратегию балансировки
    """
    balancing_info = {'strategy': strategy, 'original_shape': X_train.shape}

    if strategy == 'none':
      logger.info("Балансировка не применяется")
      return X_train, y_train, balancing_info

    elif strategy == 'smote':
      logger.info("Применение SMOTE...")
      try:
        # Используем консервативные параметры SMOTE
        smote = SMOTE(
          random_state=42,
          k_neighbors=min(5, len(y_train.value_counts().min()) - 1),  # Адаптивное количество соседей
          sampling_strategy='auto'  # Балансирует только до размера мажоритарного класса
        )
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        balancing_info['resampled_shape'] = X_resampled.shape
        balancing_info['class_distribution_after'] = pd.Series(y_resampled).value_counts().to_dict()

        logger.info(f"SMOTE завершен: {X_train.shape} -> {X_resampled.shape}")
        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled), balancing_info

      except Exception as e:
        logger.warning(f"Ошибка SMOTE: {e}, переключаемся на class_weight")
        strategy = 'class_weight'

    if strategy == 'class_weight':
      logger.info("Применение взвешивания классов...")
      # Обновляем модели с class_weight='balanced'
      for name, model in self.models.items():
        if hasattr(model, 'class_weight'):
          model.set_params(class_weight='balanced')
        elif name == 'xgb':
          # Для XGBoost вычисляем scale_pos_weight
          class_counts = y_train.value_counts()
          if len(class_counts) == 2:  # Бинарная классификация
            scale_pos_weight = class_counts[0] / class_counts[1]
            model.set_params(scale_pos_weight=scale_pos_weight)

      balancing_info['method'] = 'class_weight_balanced'
      return X_train, y_train, balancing_info

    elif strategy == 'adaptive':
      logger.info("Применение адаптивного подхода...")
      # Комбинируем умеренный SMOTE с взвешиванием
      try:
        # Сначала умеренный SMOTE (не до полного баланса)
        smote = SMOTE(
          random_state=42,
          sampling_strategy={
            cls: min(count * 2, y_train.value_counts().max())
            for cls, count in y_train.value_counts().items()
            if count < y_train.value_counts().max() * 0.5
          }
        )
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # Затем добавляем взвешивание для оставшегося дисбаланса
        for name, model in self.models.items():
          if hasattr(model, 'class_weight'):
            model.set_params(class_weight='balanced')

        balancing_info['method'] = 'smote_plus_class_weight'
        balancing_info['resampled_shape'] = X_resampled.shape

        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled), balancing_info

      except Exception as e:
        logger.error(f"Ошибка адаптивного подхода: {e}, используем только class_weight")
        return self._apply_balancing_strategy(X_train, y_train, 'class_weight')

    # Fallback
    return X_train, y_train, balancing_info

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
      self.feature_names_ = self.selected_features.copy()
      logger.info(f"✅ Сохранены названия признаков для обучения: {len(self.feature_names_)} признаков")

      # Шаг 3: Скалирование
      # logger.debug("Скалирование признаков...")
      # X_scaled = self.scalers['standard'].fit_transform(X_clean)
      # X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)

      logger.debug("Валидация и скалирование признаков...")
      X_validated = self._validate_feature_consistency(X_clean, is_training=True)
      X_scaled = self._safe_scaling(X_validated, is_training=True)

      # Шаг 4: Разделение на обучение и валидацию
      from sklearn.model_selection import train_test_split
      X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_aligned, test_size=0.2, random_state=42, stratify=y_aligned
      )

      # Шаг 5: Применение SMOTE
      # logger.info("Применение SMOTE для балансировки классов...")
      # smote = SMOTE(random_state=42)
      # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
      logger.info("Определение оптимальной стратегии балансировки...")
      balancing_strategy = self._get_optimal_balancing_strategy(y_train)
      X_train_resampled, y_train_resampled, balancing_info = self._apply_balancing_strategy(
        X_train, y_train, balancing_strategy
      )

      logger.info(f"Балансировка завершена: {balancing_info}")

      # Шаг 6: Обучение базовых моделей (БЕЗ ДОПОЛНИТЕЛЬНОГО ВЗВЕШИВАНИЯ)
      # logger.info("Обучение базовых моделей...")
      # for name, model in self.models.items():
      #   logger.debug(f"Обучение модели {name}...")
      #
      #   if name == 'xgb':
      #     logger.info("Расчет sample_weight для XGBoost...")
      #     sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_resampled)
      #     model.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights, verbose=False)
      #   else:
      #     model.fit(X_train_resampled, y_train_resampled)
      logger.info("Обучение базовых моделей...")
      for name, model in self.models.items():
        try:
          logger.info(f"Начинаем обучение модели {name}...")
          model.fit(X_train_resampled, y_train_resampled)
          logger.info(f"✅ Модель {name} успешно обучена")

          # Проверяем что модель действительно обучилась
          if hasattr(model, 'predict'):
            test_pred = model.predict(X_train_resampled[:1])
            logger.debug(f"Тестовое предсказание {name}: {test_pred}")

        except Exception as model_error:
          logger.error(f"❌ ОШИБКА обучения модели {name}: {model_error}")
          # Удаляем проблемную модель из словаря
          if name in self.models:
            del self.models[name]
          continue

        # Убираем дополнительное взвешивание для XGBoost
        # если уже применена балансировка на уровне данных или модели
        model.fit(X_train_resampled, y_train_resampled)

        # Логируем информацию о модели
        if hasattr(model, 'class_weight') and model.class_weight is not None:
          logger.debug(f"  Модель {name} использует class_weight: {model.class_weight}")
        else:
          logger.debug(f"  Модель {name} обучена на сбалансированных данных")

      # ==============================================================================

      # Шаг 7: Обучение мета-модели
      # logger.info("Обучение мета-модели...")
      # meta_features = []
      # for name, model in self.models.items():
      #   preds = model.predict_proba(X_train_resampled)[:, 1] if hasattr(model, 'predict_proba') else model.predict(
      #     X_train_resampled)
      #   meta_features.append(preds)
      #
      # meta_X = np.column_stack(meta_features)
      # self.meta_model.fit(meta_X, y_train_resampled)
      logger.info("Обучение мета-модели с валидацией...")
      meta_model_success = self._train_meta_model_safely(
        X_train_resampled, y_train_resampled, X_val, y_val
      )

      if not meta_model_success:
        logger.warning(f"Мета-модель не обучена: {self.meta_model_stats.get('fallback_reason', 'unknown')}")
        logger.info("Система будет работать только с базовыми моделями")
        self.meta_model = None  # Отключаем мета-модель

      if not hasattr(self, 'feature_names_'):
        self.feature_names_ = self.selected_features.copy() if hasattr(self, 'selected_features') else []
        logger.info(f"✅ Сохранены названия признаков для обучения: {len(self.feature_names_)} признаков")

      self.is_fitted = True
      # self.training_feature_info = {
      #   'feature_names': list(X.columns),
      #   'feature_count': len(X.columns),
      #   'training_timestamp': datetime.now().isoformat(),
      #   'data_shape': X.shape
      # }
      # logger.info(f"Сохранена информация о {len(X.columns)} признаках обучения")
      self.training_feature_info = {
        'feature_names': list(X_train_resampled.columns),
        'feature_count': len(X_train_resampled.columns),
        'training_timestamp': datetime.now().isoformat(),
        'data_shape': X_train_resampled.shape,
        'selected_features': self.selected_features.copy() if hasattr(self, 'selected_features') else []
      }
      logger.info(f"✅ Сохранена информация о {len(X_train_resampled.columns)} признаках обучения")
      logger.info(f"Обучение завершено успешно. Использовано {len(self.selected_features)} признаков.")

      # ======================= ИЗМЕНЕНИЕ 2: ВЫВОД ФИНАЛЬНЫХ МЕТРИК =======================
      # Шаг 8: Оценка производительности на валидационных данных
      logger.info("=" * 20 + " ОЦЕНКА МОДЕЛИ НА ВАЛИДАЦИОННЫХ ДАННЫХ " + "=" * 20)

      # Безопасная проверка мета-модели для валидации
      if hasattr(self, 'meta_model') and self.meta_model is not None:
        try:
          # Получаем предсказания мета-модели для валидационного набора
          meta_features_val = []
          for name, model in self.models.items():
            preds_val = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
            meta_features_val.append(preds_val)

          if meta_features_val:
            meta_X_val = np.column_stack(meta_features_val)
            final_predictions = self.meta_model.predict(meta_X_val)

            # Рассчитываем и выводим метрики
            accuracy = accuracy_score(y_val, final_predictions)
            f1 = f1_score(y_val, final_predictions, average='macro')

            logger.info(f"Итоговая точность (Accuracy) на валидации: {accuracy:.4f}")
            logger.info(f"Итоговый F1-score (Macro) на валидации: {f1:.4f}")
          else:
            logger.warning("Не удалось получить мета-признаки для валидации")

        except Exception as meta_eval_error:
          logger.warning(f"Ошибка оценки мета-модели: {meta_eval_error}")
          logger.info("Продолжаем без оценки мета-модели")
      else:
        logger.info("Мета-модель недоступна, оценка производится только по базовым моделям")
        # Используем простое голосование базовых моделей для оценки
        try:
          base_predictions = []
          for name, model in self.models.items():
            pred = model.predict(X_val)
            base_predictions.append(pred)

          if base_predictions:
            # Мажоритарное голосование
            ensemble_pred = np.round(np.mean(base_predictions, axis=0)).astype(int)
            accuracy = accuracy_score(y_val, ensemble_pred)
            f1 = f1_score(y_val, ensemble_pred, average='macro')

            logger.info(f"Точность базового ансамбля на валидации: {accuracy:.4f}")
            logger.info(f"F1-score базового ансамбля на валидации: {f1:.4f}")
        except Exception as base_eval_error:
          logger.warning(f"Ошибка оценки базовых моделей: {base_eval_error}")

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
      if hasattr(self, 'selected_features') and self.selected_features:
        self.training_features = self.selected_features.copy()
        logger.info(f"Сохранена информация о {len(self.training_features)} признаках обучения")
      else:
        logger.warning("Информация о выбранных признаках отсутствует")


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

    logger.info("Обучение мета-модели...")
    meta_model_success = self._train_meta_model_safely(
      X_train_res, y_train_res, X_val, y_val
    )

    if not meta_model_success:
      logger.warning(f"Мета-модель не обучена: {self.meta_model_stats.get('fallback_reason', 'unknown')}")
      logger.info("Система будет работать только с базовыми моделями")
      # Не устанавливаем meta_model в None, так как ее может не быть

    if not hasattr(self, 'feature_names_'):
      self.feature_names_ = self.selected_features.copy() if hasattr(self, 'selected_features') else []
      logger.info(f"✅ Сохранены названия признаков для обучения: {len(self.feature_names_)} признаков")

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
    X_scaled = self._safe_scaling(X_clean, is_training=True)
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
        X_scaled = self._safe_scaling(X_enhanced, is_training=False)
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
      # if len(predictions) > 1 and hasattr(self.meta_model, 'predict'):
      #   try:
      #     meta_features = np.column_stack(predictions)
      #     final_prediction = self.meta_model.predict(meta_features)
      #     logger.debug("Использованы предсказания мета-модели")
      #     return final_prediction.astype(int)
      #   except Exception as e:
      #     logger.warning(f"Ошибка мета-модели: {e}")

      use_meta_model = (
          len(predictions) > 1 and
          hasattr(self, 'meta_model') and
          self.meta_model is not None and
          hasattr(self.meta_model, 'predict') and
          getattr(self, 'meta_model_stats', {}).get('is_reliable', False)
      )

      if use_meta_model:
        try:
          # Извлекаем мета-признаки для предсказания
          meta_features, _ = self._extract_meta_features(X, 'prediction')

          if meta_features is not None and len(meta_features) > 0:
            meta_prediction = self.meta_model.predict(meta_features)

            # Дополнительная валидация мета-предсказания
            if self._validate_meta_prediction(meta_prediction, predictions):
              logger.debug("Использованы предсказания надежной мета-модели")
              return meta_prediction.astype(int)
            else:
              logger.warning("Мета-предсказание не прошло валидацию, используем ансамбль")
          else:
            logger.warning("Не удалось извлечь мета-признаки для предсказания")

        except Exception as e:
          logger.warning(f"Ошибка мета-модели при предсказании: {e}")

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

  def _validate_meta_prediction(self, meta_prediction: np.ndarray, base_predictions: List[np.ndarray]) -> bool:
    """
    Валидирует предсказания мета-модели на согласованность с базовыми моделями
    """
    try:
      # Получаем мажоритарное предсказание базовых моделей
      ensemble_prediction = np.round(np.mean(base_predictions, axis=0)).astype(int)

      # Проверяем, что мета-предсказание не слишком сильно отличается
      agreement_ratio = np.mean(meta_prediction == ensemble_prediction)

      # Если согласованность меньше 70%, считаем мета-предсказание ненадежным
      min_agreement = 0.7
      is_valid = agreement_ratio >= min_agreement

      if not is_valid:
        logger.debug(f"Низкая согласованность мета-модели с ансамблем: {agreement_ratio:.3f}")

      return is_valid

    except Exception as e:
      logger.warning(f"Ошибка валидации мета-предсказания: {e}")
      return False

  def predict_proba(self, X: pd.DataFrame, external_data: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[
    np.ndarray, MLPrediction]:
    """
    Получает вероятности классов и детальную информацию о предсказании
    """
    if not self.is_fitted:
      raise ValueError("Модель не обучена. Вызовите fit() перед предсказанием.")


    try:
      # 0. Валидация свежести данных (НОВЫЙ БЛОК)
      if self.use_temporal_management and hasattr(self, 'temporal_manager'):
        data_validation = self.temporal_manager.validate_data_freshness(X, 'current_symbol')

        if not data_validation['is_fresh']:
          logger.warning("Данные не являются свежими:")
          for warning in data_validation['warnings']:
            logger.warning(f"  - {warning}")

          # Можно добавить предупреждение в метаданные
          stale_data_warning = {
            'data_age_minutes': data_validation.get('data_age_minutes'),
            'warnings': data_validation['warnings'],
            'recommendations': data_validation['recommendations']
          }
        else:
          stale_data_warning = None

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

      # 4. Проверка соответствия признаков и безопасное скалирование
      try:
        X_validated = self._validate_feature_consistency(X_enhanced_clean, is_training=False)
        X_scaled = self._safe_scaling(X_validated, is_training=False)

        logger.debug(f"Данные подготовлены для предсказания: {X_scaled.shape}")

      except Exception as prep_error:
        logger.error(f"Ошибка подготовки данных для предсказания: {prep_error}")
        # Fallback: используем исходные данные с базовой обработкой
        X_scaled = X_enhanced_clean.fillna(0).replace([np.inf, -np.inf], 0)
        logger.warning("Используются данные с минимальной обработкой")

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

      # # 6. Усреднение вероятностей
      # ensemble_proba = np.mean(all_probabilities, axis=0)
      #
      # # 7. Анализ согласованности моделей
      # model_agreement = self._calculate_model_agreement(all_probabilities)
      #
      # # 8. Определение итогового сигнала (ИСПРАВЛЕННАЯ ВЕРСИЯ)
      # # Используем среднее по всем предсказаниям вместо только последнего
      # ensemble_proba_mean = np.mean(ensemble_proba, axis=0)
      # predicted_class = np.argmax(ensemble_proba_mean)
      # max_probability = np.max(ensemble_proba_mean)

      # 6. Умная агрегация вероятностей с взвешиванием
      logger.debug(f"Агрегация предсказаний от {len(all_probabilities)} моделей...")

      # Проверяем консистентность размеров
      ensemble_proba = self._aggregate_probabilities_safely(all_probabilities, model_predictions)

      # 6.5. Применение временных весов (НОВЫЙ БЛОК)
      if self.use_temporal_management and hasattr(self, 'temporal_manager'):
        try:
          weighted_probabilities = []
          for proba in all_probabilities:
            weighted_proba = self.temporal_manager.apply_temporal_weights(X_enhanced_clean, proba)
            weighted_probabilities.append(weighted_proba)

          if weighted_probabilities:
            # Пересчитываем ансамблевые вероятности с учетом временных весов
            ensemble_proba = self._aggregate_probabilities_safely(weighted_probabilities, model_predictions)
            logger.debug("Применены временные веса к предсказаниям")

        except Exception as temporal_error:
          logger.warning(f"Ошибка применения временных весов: {temporal_error}")

      # 7. Анализ согласованности моделей
      model_agreement = self._calculate_model_agreement(all_probabilities)
      confidence_boost = self._calculate_confidence_boost(model_agreement)

      # 8. Определение итогового сигнала (ИСПРАВЛЕННАЯ ВЕРСИЯ)
      # Используем агрегированные вероятности для ВСЕХ образцов, не только последнего
      if len(ensemble_proba.shape) == 1:
        # Если у нас только один образец
        final_proba = ensemble_proba
      else:
        # Если несколько образцов, используем взвешенное среднее последних N образцов
        window_size = min(5, len(ensemble_proba))  # Используем последние 5 образцов или меньше
        weights = np.exp(np.linspace(0, 1, window_size))  # Экспоненциальные веса (больший вес последним)
        weights = weights / weights.sum()

        recent_probas = ensemble_proba[-window_size:]
        final_proba = np.average(recent_probas, axis=0, weights=weights)

      predicted_class = np.argmax(final_proba)
      max_probability = np.max(final_proba)
      # ensemble_proba_mean = np.mean(ensemble_proba, axis=0)

      # Применяем буст уверенности на основе согласованности моделей
      adjusted_probability = min(max_probability * confidence_boost, 1.0)

      logger.debug(f"Финальные вероятности: {final_proba}")
      logger.debug(
        f"Predicted class: {predicted_class}, confidence: {max_probability:.3f} -> {adjusted_probability:.3f}")

      # ИСПРАВЛЕННАЯ ЛОГИКА КЛАССИФИКАЦИИ:
      # 0 = SELL, 1 = HOLD, 2 = BUY
      # Определение сигнала с учетом скорректированной уверенности
      signal_type = SignalType.HOLD  # По умолчанию

      # Минимальный порог уверенности для торговых сигналов
      min_trading_confidence = 0.4

      if predicted_class == 0 and adjusted_probability >= min_trading_confidence:
        signal_type = SignalType.SELL
      elif predicted_class == 1:
        signal_type = SignalType.HOLD  # Явно HOLD
      elif predicted_class == 2 and adjusted_probability >= min_trading_confidence:
        signal_type = SignalType.BUY
      else:
        # Если уверенность недостаточна, переходим на HOLD
        signal_type = SignalType.HOLD
        logger.debug(f"Уверенность {adjusted_probability:.3f} ниже порога {min_trading_confidence}, используем HOLD")

      # Финальная проверка: убеждаемся что торговый сигнал доминирует над HOLD
      if signal_type != SignalType.HOLD:
        hold_probability = final_proba[1]  # Вероятность HOLD (класс 1)
        signal_probability = final_proba[predicted_class]

        # Если HOLD почти такой же вероятный, лучше не торговать
        if hold_probability >= signal_probability * 0.8:  # HOLD составляет >80% от сигнала
          logger.debug(f"HOLD вероятность {hold_probability:.3f} слишком близка к сигналу {signal_probability:.3f}")
          signal_type = SignalType.HOLD
          adjusted_probability = hold_probability

      logger.debug(f"Итоговый сигнал: {signal_type.value}, финальная уверенность: {adjusted_probability:.3f}")

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

      # Рассчитываем процент NaN для метаданных (ДОБАВЛЕНО)
      try:
        nan_percentage = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
      except:
        nan_percentage = 0.0

      # 10. Создание базовых метаданных (ИСПРАВЛЕНИЕ)
      metadata = {
        'models_used': list(model_predictions.keys()) if model_predictions else [],
        'features_count': len(X_enhanced_clean.columns),
        'data_quality': 'good' if nan_percentage < 0.1 else 'poor',
        'ensemble_size': len(all_probabilities),
        'model_agreement': model_agreement,
        'signal_adjustments': []
      }

      # 11. Применение фильтров рыночной логики (ИСПРАВЛЕННЫЙ БЛОК)
      if self.use_market_filters and hasattr(self, 'market_filter'):
        try:
          # Применяем фильтры только к последнему (актуальному) предсказанию
          filtered_signal, filtered_confidence, filter_info = self.market_filter.apply_market_filters(
            signal_type, adjusted_probability, X_enhanced_clean, 'current_symbol'
          )

          # Обновляем сигнал и уверенность
          if filtered_signal != signal_type:
            logger.info(f"Сигнал изменен фильтрами: {signal_type.value} -> {filtered_signal.value}")
            metadata['signal_adjustments'].append(f"signal_changed_{signal_type.value}_to_{filtered_signal.value}")
            signal_type = filtered_signal

          if abs(filtered_confidence - adjusted_probability) > 0.05:
            logger.info(
              f"Уверенность скорректирована фильтрами: {adjusted_probability:.3f} -> {filtered_confidence:.3f}")
            metadata['signal_adjustments'].append(
              f"confidence_adjusted_{adjusted_probability:.3f}_to_{filtered_confidence:.3f}")
            adjusted_probability = filtered_confidence

          # Добавляем информацию о фильтрах в метаданные
          metadata['market_filters'] = {
            'filters_applied': filter_info.get('filters_applied', []),
            'adjustments_made': filter_info.get('adjustments_made', []),
            'market_conditions': filter_info.get('market_conditions', {})
          }

        except Exception as filter_error:
          logger.warning(f"Ошибка применения рыночных фильтров: {filter_error}")
          metadata['market_filters'] = {'error': str(filter_error)}
          # Продолжаем без фильтров при ошибке



      # 12. Создание финального MLPrediction с обновленными метаданными
      ml_prediction = MLPrediction(
        signal_type=signal_type,
        probability=float(max_probability),
        confidence=float(adjusted_probability),
        model_agreement=float(model_agreement),
        feature_importance=feature_importance,
        risk_assessment={
          'anomaly_detected': False,
          'volatility_regime': 'normal',
          'market_stress': False
        },
        metadata=metadata
      )
      # 12.5. Корректировка на основе реального времени (НОВЫЙ БЛОК)
      if self.use_temporal_management and hasattr(self, 'temporal_manager'):
        try:
          # Получаем контекст реального времени
          real_time_context = self.temporal_manager.get_real_time_context(X_enhanced_clean, 'current_symbol')

          # Корректируем ML предсказание
          ml_prediction = self.temporal_manager.adjust_prediction_for_real_time(
            ml_prediction, real_time_context
          )

          # Обновляем метаданные
          if stale_data_warning:
            ml_prediction.metadata['data_freshness_warning'] = stale_data_warning

          ml_prediction.metadata['real_time_context'] = real_time_context

        except Exception as rt_error:
          logger.warning(f"Ошибка корректировки в реальном времени: {rt_error}")

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

  def _aggregate_probabilities_safely(self, all_probabilities: List[np.ndarray],
                                        model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
      """
      Безопасная агрегация вероятностей с проверкой размеров и весов моделей
      """
      if not all_probabilities:
        raise ValueError("Нет вероятностей для агрегации")

      # Проверяем консистентность размеров
      target_shape = all_probabilities[0].shape
      consistent_probabilities = []
      model_weights = {}

      for i, proba in enumerate(all_probabilities):
        if proba.shape == target_shape:
          consistent_probabilities.append(proba)
          # Вычисляем вес модели на основе ее "уверенности"
          confidence_score = np.mean(np.max(proba, axis=1))  # Средняя максимальная вероятность
          model_weights[i] = confidence_score
        else:
          logger.warning(f"Несоответствие размеров вероятностей: {proba.shape} vs {target_shape}")

      if not consistent_probabilities:
        raise ValueError("Нет консистентных вероятностей для агрегации")

      # Нормализуем веса
      total_weight = sum(model_weights.values())
      if total_weight > 0:
        normalized_weights = [model_weights.get(i, 0) / total_weight for i in range(len(consistent_probabilities))]
      else:
        normalized_weights = [1.0 / len(consistent_probabilities)] * len(consistent_probabilities)

      # Взвешенное среднее
      ensemble_proba = np.average(consistent_probabilities, axis=0, weights=normalized_weights)

      logger.debug(f"Агрегированы {len(consistent_probabilities)} предсказаний с весами: {normalized_weights}")
      return ensemble_proba

  def _calculate_confidence_boost(self, model_agreement: float) -> float:
    """
    Рассчитывает буст уверенности на основе согласованности моделей
    """
    # Если модели согласны, увеличиваем уверенность
    # Если не согласны, снижаем уверенность

    if model_agreement >= 0.8:
      boost = 1.2  # Высокая согласованность - увеличиваем уверенность на 20%
    elif model_agreement >= 0.6:
      boost = 1.1  # Средняя согласованность - небольшой буст
    elif model_agreement >= 0.4:
      boost = 1.0  # Низкая согласованность - без изменений
    else:
      boost = 0.8  # Очень низкая согласованность - снижаем уверенность

    logger.debug(f"Model agreement: {model_agreement:.3f}, confidence boost: {boost:.2f}")
    return boost

  def _calculate_model_agreement(self, all_probabilities: List[np.ndarray]) -> float:
    """
    Рассчитывает согласованность между моделями (УЛУЧШЕННАЯ ВЕРСИЯ)
    """
    if len(all_probabilities) < 2:
      return 1.0  # Если модель одна, согласованность максимальная

    try:
      # Проверяем размеры
      target_shape = all_probabilities[0].shape
      valid_probabilities = [p for p in all_probabilities if p.shape == target_shape]

      if len(valid_probabilities) < 2:
        return 0.5  # Минимальная согласованность если размеры не совпадают

      # Получаем предсказанные классы от каждой модели
      predicted_classes = [np.argmax(proba, axis=1) for proba in valid_probabilities]

      # Рассчитываем согласованность как долю случаев, когда модели согласны
      total_predictions = len(predicted_classes[0])
      agreement_count = 0

      for i in range(total_predictions):
        # Смотрим предсказания всех моделей для i-го образца
        sample_predictions = [pred_class[i] for pred_class in predicted_classes]

        # Подсчитываем самый частый класс
        most_common_class = max(set(sample_predictions), key=sample_predictions.count)
        most_common_count = sample_predictions.count(most_common_class)

        # Согласованность = доля моделей, предсказавших самый частый класс
        sample_agreement = most_common_count / len(sample_predictions)
        agreement_count += sample_agreement

      overall_agreement = agreement_count / total_predictions

      # Дополнительно учитываем согласованность вероятностей (не только классов)
      probability_agreement = self._calculate_probability_agreement(valid_probabilities)

      # Комбинированная метрика
      final_agreement = 0.7 * overall_agreement + 0.3 * probability_agreement

      return final_agreement

    except Exception as e:
      logger.warning(f"Ошибка расчета согласованности: {e}")
      return 0.5  # Средняя согласованность при ошибке

  def _calculate_probability_agreement(self, probabilities: List[np.ndarray]) -> float:
    """
    Рассчитывает согласованность на уровне вероятностей
    """
    try:
      if len(probabilities) < 2:
        return 1.0

      # Вычисляем среднее расстояние между распределениями вероятностей
      total_distance = 0
      comparisons = 0

      for i in range(len(probabilities)):
        for j in range(i + 1, len(probabilities)):
          # Используем Jensen-Shannon расстояние для сравнения распределений
          distance = self._jensen_shannon_distance(probabilities[i], probabilities[j])
          total_distance += distance
          comparisons += 1

      if comparisons == 0:
        return 1.0

      avg_distance = total_distance / comparisons
      # Преобразуем расстояние в согласованность (чем меньше расстояние, тем выше согласованность)
      agreement = max(0, 1 - avg_distance)

      return agreement

    except Exception as e:
      logger.warning(f"Ошибка расчета согласованности вероятностей: {e}")
      return 0.5

  def _jensen_shannon_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Рассчитывает Jensen-Shannon расстояние между двумя распределениями вероятностей
    """
    try:
      # Берем среднее по всем образцам для каждого распределения
      p1_mean = np.mean(p1, axis=0)
      p2_mean = np.mean(p2, axis=0)

      # Нормализуем чтобы получить корректные вероятности
      p1_mean = p1_mean / p1_mean.sum()
      p2_mean = p2_mean / p2_mean.sum()

      # Избегаем log(0)
      p1_mean = np.clip(p1_mean, 1e-10, 1.0)
      p2_mean = np.clip(p2_mean, 1e-10, 1.0)

      # Jensen-Shannon расстояние
      m = 0.5 * (p1_mean + p2_mean)
      js_div = 0.5 * np.sum(p1_mean * np.log(p1_mean / m)) + 0.5 * np.sum(p2_mean * np.log(p2_mean / m))
      js_distance = np.sqrt(js_div)

      return js_distance

    except Exception as e:
      logger.warning(f"Ошибка расчета JS расстояния: {e}")
      return 1.0  # Максимальное расстояние при ошибке

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

  # def _calculate_model_agreement(self, all_probabilities: List[np.ndarray]) -> float:
  #   """
  #   Вычисляет согласованность между моделями
  #   """
  #   try:
  #     if len(all_probabilities) < 2:
  #       return 1.0
  #
  #     # Берем предсказания для последнего наблюдения
  #     last_predictions = [proba[-1] for proba in all_probabilities]
  #
  #     # Вычисляем стандартное отклонение между предсказаниями моделей
  #     std_across_models = np.std(last_predictions, axis=0)
  #     avg_std = np.mean(std_across_models)
  #
  #     # Преобразуем в меру согласованности (чем меньше разброс, тем выше согласованность)
  #     agreement = max(0.0, 1.0 - (avg_std * 3))  # Масштабируем
  #
  #     return agreement
  #
  #   except Exception:
  #     return 0.5  # Средняя согласованность по умолчанию

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
      # 'meta_model': self.meta_model,
      'meta_model': getattr(self, 'meta_model', None),
      'meta_model_stats': getattr(self, 'meta_model_stats', {}),
      'scaler': self.scaler,
      'scaler_type_used': getattr(self, 'scaler_type_used', 'standard'),
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
    # instance.meta_model = model_data['meta_model']
    instance.meta_model = model_data.get('meta_model', None)
    instance.meta_model_stats = model_data.get('meta_model_stats', {})
    instance.scaler = model_data.get('scaler', StandardScaler())
    instance.scaler_type_used = model_data.get('scaler_type_used', 'standard')
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
        if hasattr(self, 'meta_model') and self.meta_model is not None:
          # Проверяем что мета-модель действительно обучена
          meta_is_fitted = (
              hasattr(self.meta_model, 'coef_') or
              hasattr(self.meta_model, 'feature_importances_') or
              (hasattr(self.meta_model, '_check_is_fitted') and
               getattr(self, 'meta_model_stats', {}).get('is_reliable', False))
          )

          if meta_is_fitted:
            health_status['components']['meta_model'] = 'OK'
          else:
            health_status['components']['meta_model'] = 'NOT_TRAINED'
        else:
          health_status['components']['meta_model'] = 'NOT_AVAILABLE'
      except Exception as e:
        health_status['components']['meta_model'] = f'ERROR: {str(e)}'

      # Проверка скейлеров
      scalers_ok = 0
      scaler_details = {}

      try:
        # Проверка основного скейлера
        if hasattr(self, 'scaler') and self.scaler is not None:
          scaler_type = getattr(self, 'scaler_type_used', type(self.scaler).__name__)

          # Проверяем разные типы скейлеров
          is_fitted = False
          if hasattr(self.scaler, 'mean_'):  # StandardScaler
            is_fitted = True
            scaler_details['mean_features'] = len(self.scaler.mean_)
          elif hasattr(self.scaler, 'center_'):  # RobustScaler
            is_fitted = True
            scaler_details['center_features'] = len(self.scaler.center_)
          elif hasattr(self.scaler, 'scale_'):  # Общий атрибут
            is_fitted = True
            scaler_details['scale_features'] = len(self.scaler.scale_)

          if is_fitted:
            scalers_ok = 1
            health_status['components'][f'scaler_{scaler_type}'] = 'OK'
            logger.debug(f"Скейлер {scaler_type} обучен и готов к использованию")
          else:
            health_status['components'][f'scaler_{scaler_type}'] = 'NOT_FITTED'
            logger.warning(f"Скейлер {scaler_type} присутствует, но не обучен")
        else:
          health_status['components']['scaler_main'] = 'MISSING'
          logger.error("Основной скейлер отсутствует")

        # Проверка резервного скейлера
        if hasattr(self, 'backup_scaler') and self.backup_scaler is not None:
          backup_fitted = (hasattr(self.backup_scaler, 'mean_') or
                           hasattr(self.backup_scaler, 'center_') or
                           hasattr(self.backup_scaler, 'scale_'))

          health_status['components']['scaler_backup'] = 'OK' if backup_fitted else 'NOT_FITTED'
          scaler_details['backup_available'] = True
        else:
          health_status['components']['scaler_backup'] = 'NOT_AVAILABLE'
          scaler_details['backup_available'] = False

        # Добавляем детали в метаданные
        health_status['scaler_details'] = scaler_details

      except Exception as e:
        health_status['components']['scaler_system'] = f'ERROR: {str(e)}'
        logger.error(f"Критическая ошибка проверки скейлеров: {e}")
        scalers_ok = 0

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
        health_status['issues'].append("Скейлер не обучен или недоступен")
        health_status['recommendations'].append("Переобучите модель полностью")
      elif not hasattr(self, 'scaler') or self.scaler is None:
        health_status['issues'].append("Отсутствует основной скейлер")
        health_status['recommendations'].append("Проверьте инициализацию модели")

      return health_status

    except Exception as e:
      logger.error(f"Ошибка проверки здоровья модели: {e}")
      health_status['error'] = str(e)
      health_status['overall_health'] = 'ERROR'
      return health_status


# =================== НОВЫЙ КЛАСС ДЛЯ РЫНОЧНЫХ ФИЛЬТРОВ ===================

class MarketLogicFilter:
  """
  Класс для применения фильтров здравого смысла к ML предсказаниям
  """

  def __init__(self):
    self.trend_window = 20
    self.volatility_window = 14
    self.volume_window = 10
    self.rsi_oversold = 30
    self.rsi_overbought = 70
    self.min_trend_strength = 0.6
    self.max_volatility_ratio = 3.0

  def apply_market_filters(self, signal_type: SignalType, confidence: float,
                           market_data: pd.DataFrame, symbol: str) -> Tuple[SignalType, float, Dict]:
    """
    Применяет фильтры рыночной логики к ML сигналу
    """
    logger.debug(f"Применение рыночных фильтров для {symbol}, исходный сигнал: {signal_type.value}")

    filter_results = {
      'original_signal': signal_type.value,
      'original_confidence': confidence,
      'filters_applied': [],
      'adjustments_made': [],
      'final_signal': signal_type.value,
      'final_confidence': confidence,
      'market_conditions': {}
    }

    try:
      # 1. Анализ рыночных условий
      market_conditions = self._analyze_market_conditions(market_data)
      filter_results['market_conditions'] = market_conditions

      # 2. Фильтр тренда
      signal_type, confidence = self._apply_trend_filter(
        signal_type, confidence, market_conditions, filter_results
      )

      # 3. Фильтр волатильности
      signal_type, confidence = self._apply_volatility_filter(
        signal_type, confidence, market_conditions, filter_results
      )

      # 4. Фильтр RSI (перекупленность/перепроданность)
      signal_type, confidence = self._apply_rsi_filter(
        signal_type, confidence, market_conditions, filter_results
      )

      # 5. Фильтр объема
      signal_type, confidence = self._apply_volume_filter(
        signal_type, confidence, market_conditions, filter_results
      )

      # 6. Фильтр поддержки/сопротивления
      signal_type, confidence = self._apply_support_resistance_filter(
        signal_type, confidence, market_conditions, filter_results
      )

      # 7. Режимный фильтр (трендовый/боковой рынок)
      signal_type, confidence = self._apply_regime_filter(
        signal_type, confidence, market_conditions, filter_results
      )

      filter_results['final_signal'] = signal_type.value
      filter_results['final_confidence'] = confidence

      # Логируем результаты фильтрации
      if signal_type.value != filter_results['original_signal']:
        logger.info(
          f"Сигнал для {symbol} изменен фильтрами: {filter_results['original_signal']} -> {signal_type.value}")

      if abs(confidence - filter_results['original_confidence']) > 0.1:
        logger.info(
          f"Уверенность для {symbol} скорректирована: {filter_results['original_confidence']:.3f} -> {confidence:.3f}")

      return signal_type, confidence, filter_results

    except Exception as e:
      logger.error(f"Ошибка применения рыночных фильтров для {symbol}: {e}")
      return signal_type, confidence, filter_results

  def _analyze_market_conditions(self, data: pd.DataFrame) -> Dict:
    """
    Анализирует текущие рыночные условия
    """
    try:
      conditions = {}

      # БЕЗОПАСНОЕ ИЗВЛЕЧЕНИЕ ДАННЫХ
      if len(data) < 5:
        logger.warning("Недостаточно данных для анализа рыночных условий")
        return conditions

      # Безопасное извлечение ценовых данных
      try:
        close_prices = data['close'].dropna().values
        high_prices = data['high'].dropna().values
        low_prices = data['low'].dropna().values
        volumes = data['volume'].dropna().values if 'volume' in data.columns else None
      except Exception as extraction_error:
        logger.warning(f"Ошибка извлечения базовых данных: {extraction_error}")
        return conditions

      if len(close_prices) < 5:
        logger.warning("Недостаточно валидных ценовых данных")
        return conditions

      current_price = close_prices[-1]

      # 1. БЕЗОПАСНЫЙ АНАЛИЗ ТРЕНДА
      try:
        trend_window = min(self.trend_window, len(close_prices))
        if trend_window >= 5:
          trend_prices = close_prices[-trend_window:]
          trend_sma = np.mean(trend_prices)
          price_vs_trend = (current_price - trend_sma) / trend_sma

          # ИСПРАВЛЕННАЯ линейная регрессия
          x = np.arange(len(trend_prices))  # Используем фактическую длину
          y = trend_prices

          if len(x) == len(y) and len(x) > 1:  # Проверяем совпадение размеров
            trend_slope = np.polyfit(x, y, 1)[0]
            trend_slope_normalized = trend_slope / current_price

            conditions['trend'] = {
              'direction': 'up' if trend_slope_normalized > 0.001 else 'down' if trend_slope_normalized < -0.001 else 'sideways',
              'strength': abs(trend_slope_normalized) * 1000,
              'price_vs_ma': price_vs_trend,
              'slope': trend_slope_normalized
            }
          else:
            logger.warning(f"Размеры не совпадают для тренда: x={len(x)}, y={len(y)}")
      except Exception as trend_error:
        logger.warning(f"Ошибка анализа тренда: {trend_error}")

      # 2. БЕЗОПАСНЫЙ АНАЛИЗ ВОЛАТИЛЬНОСТИ
      try:
        vol_window = min(self.volatility_window, len(close_prices))
        if vol_window >= 5:
          vol_prices = close_prices[-vol_window:]
          if len(vol_prices) > 1:
            returns = np.diff(vol_prices) / vol_prices[:-1]  # ИСПРАВЛЕНО: правильная индексация
            current_volatility = np.std(returns) * np.sqrt(252)

            # Сравниваем с исторической волатильностью
            hist_window = min(vol_window * 2, len(close_prices))
            if hist_window > vol_window:
              hist_prices = close_prices[-hist_window:-vol_window]
              if len(hist_prices) > 1:
                hist_returns = np.diff(hist_prices) / hist_prices[:-1]
                historical_volatility = np.std(hist_returns) * np.sqrt(252)
                volatility_ratio = current_volatility / (historical_volatility + 1e-8)
              else:
                volatility_ratio = 1.0
            else:
              volatility_ratio = 1.0

            conditions['volatility'] = {
              'current': current_volatility,
              'ratio_to_historical': volatility_ratio,
              'regime': 'high' if volatility_ratio > 1.5 else 'low' if volatility_ratio < 0.7 else 'normal'
            }
      except Exception as vol_error:
        logger.warning(f"Ошибка анализа волатильности: {vol_error}")

      # 3. БЕЗОПАСНЫЙ РАСЧЕТ RSI
      try:
        if len(close_prices) >= 14:
          rsi = self._calculate_rsi(close_prices, 14)
          conditions['rsi'] = {
            'value': rsi,
            'condition': 'oversold' if rsi < self.rsi_oversold else 'overbought' if rsi > self.rsi_overbought else 'neutral'
          }
      except Exception as rsi_error:
        logger.warning(f"Ошибка расчета RSI: {rsi_error}")

      # 4. БЕЗОПАСНЫЙ АНАЛИЗ ОБЪЕМА
      try:
        if volumes is not None and len(volumes) >= self.volume_window:
          vol_window = min(self.volume_window, len(volumes))
          current_volume = volumes[-1]
          avg_volume = np.mean(volumes[-vol_window:])
          volume_ratio = current_volume / (avg_volume + 1e-8)

          conditions['volume'] = {
            'current': current_volume,
            'ratio_to_average': volume_ratio,
            'condition': 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.5 else 'normal'
          }
      except Exception as volume_error:
        logger.warning(f"Ошибка анализа объема: {volume_error}")

      # 5. Поддержка и сопротивление
      if len(close_prices) >= 50:
        support_resistance = self._find_support_resistance(high_prices, low_prices, close_prices)
        conditions['support_resistance'] = support_resistance

      # 6. Рыночный режим
      if len(close_prices) >= 50:
        market_regime = self._determine_market_regime(close_prices, high_prices, low_prices)
        conditions['market_regime'] = market_regime

      return conditions

    except Exception as e:
      logger.error(f"Ошибка анализа рыночных условий: {e}")
      return {}

  def _apply_trend_filter(self, signal_type: SignalType, confidence: float,
                          market_conditions: Dict, filter_results: Dict) -> Tuple[SignalType, float]:
    """
    Фильтр на основе тренда
    """
    if 'trend' not in market_conditions:
      return signal_type, confidence

    trend_info = market_conditions['trend']
    trend_direction = trend_info['direction']
    trend_strength = trend_info['strength']

    filter_results['filters_applied'].append('trend_filter')

    # Не торгуем против сильного тренда
    if trend_strength > self.min_trend_strength:
      if signal_type == SignalType.BUY and trend_direction == 'down':
        logger.debug(f"BUY сигнал против нисходящего тренда (сила: {trend_strength:.3f})")
        confidence *= 0.5  # Снижаем уверенность
        filter_results['adjustments_made'].append('reduced_confidence_against_downtrend')

        if confidence < 0.4:  # Если уверенность стала слишком низкой
          signal_type = SignalType.HOLD
          filter_results['adjustments_made'].append('changed_to_hold_weak_against_trend')

      elif signal_type == SignalType.SELL and trend_direction == 'up':
        logger.debug(f"SELL сигнал против восходящего тренда (сила: {trend_strength:.3f})")
        confidence *= 0.5
        filter_results['adjustments_made'].append('reduced_confidence_against_uptrend')

        if confidence < 0.4:
          signal_type = SignalType.HOLD
          filter_results['adjustments_made'].append('changed_to_hold_weak_against_trend')

    # Усиливаем сигналы по тренду
    elif trend_strength > 0.3:  # Умеренный тренд
      if (signal_type == SignalType.BUY and trend_direction == 'up') or \
          (signal_type == SignalType.SELL and trend_direction == 'down'):
        confidence = min(confidence * 1.2, 1.0)  # Повышаем уверенность
        filter_results['adjustments_made'].append('boosted_confidence_with_trend')

    return signal_type, confidence

  def _apply_volatility_filter(self, signal_type: SignalType, confidence: float,
                               market_conditions: Dict, filter_results: Dict) -> Tuple[SignalType, float]:
    """
    Фильтр на основе волатильности
    """
    if 'volatility' not in market_conditions:
      return signal_type, confidence

    vol_info = market_conditions['volatility']
    vol_ratio = vol_info['ratio_to_historical']
    vol_regime = vol_info['regime']

    filter_results['filters_applied'].append('volatility_filter')

    # В периоды экстремальной волатильности снижаем уверенность
    if vol_ratio > self.max_volatility_ratio:
      logger.debug(f"Экстремальная волатильность обнаружена: {vol_ratio:.2f}x")
      confidence *= 0.6
      filter_results['adjustments_made'].append('reduced_confidence_high_volatility')

      # При очень высокой волатильности переходим на HOLD
      if vol_ratio > 5.0 and signal_type != SignalType.HOLD:
        signal_type = SignalType.HOLD
        filter_results['adjustments_made'].append('changed_to_hold_extreme_volatility')

    # В периоды низкой волатильности тоже снижаем уверенность (возможен прорыв)
    elif vol_ratio < 0.3:
      logger.debug(f"Аномально низкая волатильность: {vol_ratio:.2f}x")
      confidence *= 0.8
      filter_results['adjustments_made'].append('reduced_confidence_low_volatility')

    return signal_type, confidence

  def _apply_rsi_filter(self, signal_type: SignalType, confidence: float,
                        market_conditions: Dict, filter_results: Dict) -> Tuple[SignalType, float]:
    """
    Фильтр на основе RSI
    """
    if 'rsi' not in market_conditions:
      return signal_type, confidence

    rsi_info = market_conditions['rsi']
    rsi_value = rsi_info['value']
    rsi_condition = rsi_info['condition']

    filter_results['filters_applied'].append('rsi_filter')

    # Не покупаем в перекупленности, не продаем в перепроданности
    if signal_type == SignalType.BUY and rsi_condition == 'overbought':
      logger.debug(f"BUY сигнал при перекупленности (RSI: {rsi_value:.1f})")
      confidence *= 0.4
      filter_results['adjustments_made'].append('reduced_confidence_overbought')

      if rsi_value > 80:  # Экстремальная перекупленность
        signal_type = SignalType.HOLD
        filter_results['adjustments_made'].append('changed_to_hold_extreme_overbought')

    elif signal_type == SignalType.SELL and rsi_condition == 'oversold':
      logger.debug(f"SELL сигнал при перепроданности (RSI: {rsi_value:.1f})")
      confidence *= 0.4
      filter_results['adjustments_made'].append('reduced_confidence_oversold')

      if rsi_value < 20:  # Экстремальная перепроданность
        signal_type = SignalType.HOLD
        filter_results['adjustments_made'].append('changed_to_hold_extreme_oversold')

    # Усиливаем сигналы в правильном направлении
    elif signal_type == SignalType.BUY and rsi_condition == 'oversold':
      confidence = min(confidence * 1.3, 1.0)
      filter_results['adjustments_made'].append('boosted_confidence_buy_oversold')

    elif signal_type == SignalType.SELL and rsi_condition == 'overbought':
      confidence = min(confidence * 1.3, 1.0)
      filter_results['adjustments_made'].append('boosted_confidence_sell_overbought')

    return signal_type, confidence

  def _apply_volume_filter(self, signal_type: SignalType, confidence: float,
                           market_conditions: Dict, filter_results: Dict) -> Tuple[SignalType, float]:
    """
    Фильтр на основе объема
    """
    if 'volume' not in market_conditions:
      return signal_type, confidence

    volume_info = market_conditions['volume']
    volume_condition = volume_info['condition']
    volume_ratio = volume_info['ratio_to_average']

    filter_results['filters_applied'].append('volume_filter')

    # Низкий объем снижает уверенность в сигнале
    if volume_condition == 'low':
      logger.debug(f"Низкий объем обнаружен: {volume_ratio:.2f}x")
      confidence *= 0.7
      filter_results['adjustments_made'].append('reduced_confidence_low_volume')

    # Высокий объем усиливает сигнал
    elif volume_condition == 'high' and signal_type != SignalType.HOLD:
      logger.debug(f"Высокий объем подтверждает сигнал: {volume_ratio:.2f}x")
      confidence = min(confidence * 1.2, 1.0)
      filter_results['adjustments_made'].append('boosted_confidence_high_volume')

    return signal_type, confidence

  def _apply_support_resistance_filter(self, signal_type: SignalType, confidence: float,
                                       market_conditions: Dict, filter_results: Dict) -> Tuple[SignalType, float]:
    """
    Фильтр на основе уровней поддержки и сопротивления
    """
    if 'support_resistance' not in market_conditions:
      return signal_type, confidence

    sr_info = market_conditions['support_resistance']
    filter_results['filters_applied'].append('support_resistance_filter')

    # Если цена близко к сопротивлению, не покупаем
    if signal_type == SignalType.BUY and sr_info.get('near_resistance', False):
      logger.debug("BUY сигнал близко к сопротивлению")
      confidence *= 0.6
      filter_results['adjustments_made'].append('reduced_confidence_near_resistance')

    # Если цена близко к поддержке, не продаем
    elif signal_type == SignalType.SELL and sr_info.get('near_support', False):
      logger.debug("SELL сигнал близко к поддержке")
      confidence *= 0.6
      filter_results['adjustments_made'].append('reduced_confidence_near_support')

    return signal_type, confidence

  def _apply_regime_filter(self, signal_type: SignalType, confidence: float,
                           market_conditions: Dict, filter_results: Dict) -> Tuple[SignalType, float]:
    """
    Фильтр на основе рыночного режима
    """
    if 'market_regime' not in market_conditions:
      return signal_type, confidence

    regime_info = market_conditions['market_regime']
    regime_type = regime_info.get('type', 'unknown')

    filter_results['filters_applied'].append('regime_filter')

    # В боковом рынке снижаем уверенность в трендовых сигналах
    if regime_type == 'sideways' and signal_type != SignalType.HOLD:
      logger.debug("Трендовый сигнал в боковом рынке")
      confidence *= 0.7
      filter_results['adjustments_made'].append('reduced_confidence_sideways_market')

    # В трендовом рынке усиливаем сигналы по тренду
    elif regime_type == 'trending':
      trend_direction = market_conditions.get('trend', {}).get('direction', 'unknown')
      if (signal_type == SignalType.BUY and trend_direction == 'up') or \
          (signal_type == SignalType.SELL and trend_direction == 'down'):
        confidence = min(confidence * 1.15, 1.0)
        filter_results['adjustments_made'].append('boosted_confidence_trending_market')

    return signal_type, confidence

  def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
    """
    Рассчитывает RSI
    """
    try:
      if len(prices) < period + 1:
        return 50.0  # Нейтральное значение

      deltas = np.diff(prices)
      gains = np.where(deltas > 0, deltas, 0)
      losses = np.where(deltas < 0, -deltas, 0)

      avg_gain = np.mean(gains[-period:])
      avg_loss = np.mean(losses[-period:])

      if avg_loss == 0:
        return 100.0

      rs = avg_gain / avg_loss
      rsi = 100 - (100 / (1 + rs))

      return float(rsi)

    except Exception as e:
      logger.warning(f"Ошибка расчета RSI: {e}")
      return 50.0

  def _find_support_resistance(self, highs: np.ndarray, lows: np.ndarray,
                               closes: np.ndarray) -> Dict:
    """
    Находит уровни поддержки и сопротивления
    """
    try:
      current_price = closes[-1]

      # Простой алгоритм поиска локальных экстремумов
      window = 5
      resistance_levels = []
      support_levels = []

      # Ищем локальные максимумы (сопротивления)
      for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
          resistance_levels.append(highs[i])

      # Ищем локальные минимумы (поддержки)
      for i in range(window, len(lows) - window):
        if lows[i] == min(lows[i - window:i + window + 1]):
          support_levels.append(lows[i])

      # Находим ближайшие уровни
      resistance_levels = sorted(resistance_levels, reverse=True)
      support_levels = sorted(support_levels, reverse=True)

      nearest_resistance = None
      nearest_support = None

      for level in resistance_levels:
        if level > current_price:
          nearest_resistance = level
          break

      for level in support_levels:
        if level < current_price:
          nearest_support = level
          break

      # Проверяем близость к уровням (в пределах 2%)
      threshold = 0.02
      near_resistance = (nearest_resistance is not None and
                         abs(current_price - nearest_resistance) / current_price < threshold)
      near_support = (nearest_support is not None and
                      abs(current_price - nearest_support) / current_price < threshold)

      return {
        'nearest_resistance': nearest_resistance,
        'nearest_support': nearest_support,
        'near_resistance': near_resistance,
        'near_support': near_support,
        'resistance_levels': resistance_levels[:3],  # Топ 3
        'support_levels': support_levels[:3]
      }

    except Exception as e:
      logger.warning(f"Ошибка поиска поддержек/сопротивлений: {e}")
      return {}

  def _determine_market_regime(self, closes: np.ndarray, highs: np.ndarray,
                               lows: np.ndarray) -> Dict:
    """
    Определяет рыночный режим (трендовый/боковой)
    """
    try:
      # Используем ADX-подобный подход для определения режима
      period = 20
      if len(closes) < period:
        return {'type': 'unknown'}

      # Рассчитываем направленное движение
      high_diff = np.diff(highs[-period:])
      low_diff = -np.diff(lows[-period:])

      # True Range
      tr1 = highs[1:] - lows[1:]
      tr2 = np.abs(highs[1:] - closes[:-1])
      tr3 = np.abs(lows[1:] - closes[:-1])
      true_range = np.maximum(tr1, np.maximum(tr2, tr3))

      # Направленные индексы
      plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
      minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

      # Сглаживание
      atr = np.mean(true_range[-14:])
      plus_di = 100 * np.mean(plus_dm[-14:]) / atr if atr > 0 else 0
      minus_di = 100 * np.mean(minus_dm[-14:]) / atr if atr > 0 else 0

      # ADX
      dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0

      # Определяем режим
      if dx > 25:
        regime_type = 'trending'
        strength = 'strong' if dx > 40 else 'moderate'
      else:
        regime_type = 'sideways'
        strength = 'weak'

      return {
        'type': regime_type,
        'strength': strength,
        'adx_value': dx,
        'plus_di': plus_di,
        'minus_di': minus_di
      }

    except Exception as e:
      logger.warning(f"Ошибка определения рыночного режима: {e}")
      return {'type': 'unknown'}

class TemporalDataManager:
    """
    Класс для управления временными аспектами данных и обеспечения актуальности
    """

    def __init__(self):
      self.max_data_age_minutes = 65  # Максимальный возраст данных в минутах
      self.required_recent_bars = 5  # Минимум недавних баров для анализа
      self.temporal_weights = True  # Использовать временные веса
      self.real_time_adjustments = True  # Использовать корректировки в реальном времени

    def _calculate_real_data_age(self, last_timestamp: pd.Timestamp, timeframe: str = '1h') -> float:
      """
      Рассчитывает реальный возраст данных с учетом таймфрейма
      """
      try:
        current_time_utc = pd.Timestamp.now(tz='UTC')

        # ДОБАВЬТЕ ЭТО ЛОГИРОВАНИЕ ДЛЯ ДИАГНОСТИКИ:
        logger.debug(f"=== ДИАГНОСТИКА ВОЗРАСТА ДАННЫХ ===")
        logger.debug(f"Исходный timestamp: {last_timestamp}")
        logger.debug(f"Тип timestamp: {type(last_timestamp)}")
        logger.debug(f"Timezone timestamp: {getattr(last_timestamp, 'tz', 'нет')}")
        logger.debug(f"Текущее время UTC: {current_time_utc}")

        # Определяем интервал таймфрейма в минутах
        timeframe_minutes = {
          '1m': 1, '5m': 5, '15m': 15, '30m': 30,
          '1h': 60, '4h': 240, '1d': 1440
        }

        interval_minutes = timeframe_minutes.get(timeframe, 60)  # По умолчанию час

        # Корректируем время последней свечи
        # Для часовых свечей: если timestamp = 09:00, то свеча закрылась в 10:00
        last_candle_close_time = last_timestamp + pd.Timedelta(minutes=interval_minutes)

        # Рассчитываем реальный возраст
        real_age = current_time_utc - last_candle_close_time
        real_age_minutes = real_age.total_seconds() / 60

        logger.debug(f"Коррекция времени для {timeframe}:")
        logger.debug(f"  Timestamp свечи: {last_timestamp}")
        logger.debug(f"  Время закрытия свечи: {last_candle_close_time}")
        logger.debug(f"  Текущее время: {current_time_utc}")
        logger.debug(f"  Реальный возраст: {real_age_minutes:.1f} мин")

        return max(0, real_age_minutes)  # Не может быть отрицательным

      except Exception as e:
        logger.warning(f"Ошибка корректировки возраста данных: {e}")
        # Fallback к обычному расчету
        return (current_time_utc - last_timestamp).total_seconds() / 60

    def validate_data_freshness(self, data: pd.DataFrame, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
      validation_result = {
        'is_fresh': False,
        'last_update': None,
        'data_age_minutes': None,
        'real_age_minutes': None,  # Добавляем реальный возраст
        'warnings': [],
        'recommendations': []
      }

      try:
        if data.empty:
          validation_result['warnings'].append("Данные отсутствуют")
          return validation_result

        # ИСПРАВЛЕННАЯ ОБРАБОТКА ВРЕМЕННЫХ МЕТОК С УЧЕТОМ UTC
        last_timestamp = None

        # 1. Ищем временные метки
        if 'timestamp' in data.columns:
          try:
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: API Bybit возвращает данные от новых к старым
            # Поэтому берем ПЕРВЫЙ элемент (самый свежий), а не последний

            # Сначала проверяем порядок сортировки
            if len(data) >= 2:
              first_ts = pd.to_datetime(data['timestamp'].iloc[0])
              second_ts = pd.to_datetime(data['timestamp'].iloc[1])

              if first_ts > second_ts:
                # Данные отсортированы от новых к старым (правильно для Bybit)
                last_timestamp = first_ts  # Берем первый = самый свежий
                logger.debug(f"🔍 Данные отсортированы новые→старые, берем первый: {first_ts}")
              else:
                # Данные отсортированы от старых к новым
                last_timestamp = pd.to_datetime(data['timestamp'].iloc[-1])  # Берем последний
                logger.debug(f"🔍 Данные отсортированы старые→новые, берем последний: {last_timestamp}")
            else:
              # Если только одна запись
              last_timestamp = pd.to_datetime(data['timestamp'].iloc[-1])

            logger.debug(f"🔍 Выбранный timestamp для проверки свежести: {last_timestamp}")

          except Exception as e:
            logger.debug(f"Ошибка парсинга timestamp колонки: {e}")

        if last_timestamp is None and hasattr(data.index, 'to_pydatetime'):
          try:
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: проверяем сортировку индекса
            if len(data) >= 2:
              first_idx = data.index[0]
              second_idx = data.index[1]

              # Конвертируем в datetime для сравнения
              first_dt = pd.to_datetime(first_idx) if not isinstance(first_idx, pd.Timestamp) else first_idx
              second_dt = pd.to_datetime(second_idx) if not isinstance(second_idx, pd.Timestamp) else second_idx

              if first_dt > second_dt:
                # Индекс отсортирован от новых к старым
                target_timestamp = data.index[0]  # Берем первый = самый свежий
                logger.debug(f"🔍 Индекс отсортирован новые→старые, берем первый: {first_dt}")
              else:
                # Индекс отсортирован от старых к новым
                target_timestamp = data.index[-1]  # Берем последний = самый свежий
                logger.debug(f"🔍 Индекс отсортирован старые→новые, берем последний: {second_dt}")
            else:
              target_timestamp = data.index[-1]

            if hasattr(target_timestamp, 'to_pydatetime'):
              last_timestamp = target_timestamp.to_pydatetime()
            last_timestamp = pd.Timestamp(last_timestamp)

            logger.debug(f"🔍 Выбранный timestamp из индекса: {last_timestamp}")

          except Exception as e:
            logger.debug(f"Ошибка использования индекса как timestamp: {e}")

        if last_timestamp is None:
          validation_result['warnings'].append("Временные метки не найдены")
          return validation_result

        # ЗАМЕНИТЕ блок "2. КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: ПРИВЕДЕНИЕ К UTC":
        try:
          current_time_utc = pd.Timestamp.now(tz='UTC')

          # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Проверяем формат timestamp
          logger.debug(f"Исходный timestamp: {last_timestamp}, тип: {type(last_timestamp)}")

          # Обработка разных форматов timestamp
          if isinstance(last_timestamp, (int, float)):
            # Проверяем, в миллисекундах ли timestamp
            if last_timestamp > 1e12:
              last_timestamp_utc = pd.Timestamp(last_timestamp / 1000, unit='s', tz='UTC')
            else:
              last_timestamp_utc = pd.Timestamp(last_timestamp, unit='s', tz='UTC')
          elif isinstance(last_timestamp, str):
            # Парсим строку и добавляем UTC
            last_timestamp_utc = pd.Timestamp(last_timestamp, tz='UTC')
          else:
            # Обрабатываем pandas Timestamp
            if hasattr(last_timestamp, 'tz') and last_timestamp.tz is not None:
              last_timestamp_utc = last_timestamp.astimezone(current_time_utc.tz)
            else:
              last_timestamp_utc = pd.Timestamp(last_timestamp).tz_localize('UTC')

          # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: Если данные слишком старые, возможно проблема с API
          time_diff_hours = abs((current_time_utc - last_timestamp_utc).total_seconds() / 3600)
          if time_diff_hours > 24:
            logger.warning(f"🚨 КРИТИЧЕСКИ УСТАРЕВШИЕ ДАННЫЕ: {time_diff_hours:.1f} часов!")
            logger.warning(f"Последний timestamp: {last_timestamp_utc}")
            logger.warning(f"Текущее время: {current_time_utc}")
            logger.warning("Возможные причины: проблемы с API биржи, неправильный формат данных, сетевые проблемы")

            # Принудительно устанавливаем максимальный возраст
            real_age_minutes = 999999
            raw_age_minutes = time_diff_hours * 60
          else:
            # Рассчитываем РЕАЛЬНЫЙ возраст с учетом таймфрейма
            real_age_minutes = self._calculate_real_data_age(last_timestamp_utc, timeframe)

            # Также сохраняем "сырой" возраст для отладки
            raw_age_minutes = (current_time_utc - last_timestamp_utc).total_seconds() / 60

          validation_result['data_age_minutes'] = raw_age_minutes
          validation_result['real_age_minutes'] = real_age_minutes

          logger.debug(
            f"Возраст данных для {symbol}: сырой={raw_age_minutes:.1f}мин, реальный={real_age_minutes:.1f}мин")

        except Exception as age_error:
          logger.warning(f"Ошибка расчета возраста данных: {age_error}")
          real_age_minutes = 999999

          # Проверяем свежесть по РЕАЛЬНОМУ возрасту
        if real_age_minutes <= self.max_data_age_minutes:
          validation_result['is_fresh'] = True
        else:
          validation_result['warnings'].append(
            f"Данные устарели: {real_age_minutes:.1f} мин (макс: {self.max_data_age_minutes})"
          )

        return validation_result

      except Exception as e:
        logger.error(f"Критическая ошибка валидации: {e}")
        return validation_result

    def apply_temporal_weights(self, data: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
      """
      Применяет временные веса к предсказаниям (больший вес последним данным)
      """
      try:
        if not self.temporal_weights or len(predictions.shape) != 2:
          return predictions

        n_samples = predictions.shape[0]

        # Создаем экспоненциальные веса (больший вес последним наблюдениям)
        weights = np.exp(np.linspace(0, 1, n_samples))
        weights = weights / weights.sum()

        # Применяем веса к каждому классу
        weighted_predictions = predictions * weights.reshape(-1, 1)

        logger.debug(f"Применены временные веса к {n_samples} предсказаниям")
        return weighted_predictions

      except Exception as e:
        logger.warning(f"Ошибка применения временных весов: {e}")
        return predictions

    def get_real_time_context(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
      """
      Получает контекст реального времени (ИСПРАВЛЕННАЯ ВЕРСИЯ)
      """
      context = {
        'price_momentum': 0.0,
        'recent_volatility': 0.0,
        'volume_trend': 'neutral',
        'market_phase': 'unknown',
        'confidence_adjustment': 1.0
      }

      try:
        if len(data) < 10:
          return context

        # БЕЗОПАСНОЕ ИЗВЛЕЧЕНИЕ ДАННЫХ
        close_prices = data['close'].dropna().values
        volumes = data['volume'].dropna().values if 'volume' in data.columns else None

        if len(close_prices) < 10:
          return context

        # 1. ИСПРАВЛЕННЫЙ АНАЛИЗ ЦЕНОВОГО МОМЕНТУМА
        try:
          recent_bars = min(5, len(close_prices) // 2)
          if recent_bars >= 2:
            recent_avg = np.mean(close_prices[-recent_bars:])

            # ИСПРАВЛЕНО: проверяем наличие достаточных данных
            if len(close_prices) >= recent_bars * 2:
              previous_avg = np.mean(close_prices[-recent_bars * 2:-recent_bars])

              if previous_avg > 0:
                momentum = (recent_avg - previous_avg) / previous_avg
                context['price_momentum'] = momentum
        except Exception as momentum_error:
          logger.warning(f"Ошибка расчета моментума: {momentum_error}")

        # 2. БЕЗОПАСНЫЙ АНАЛИЗ ВОЛАТИЛЬНОСТИ
        try:
          if len(close_prices) >= 10:
            vol_window = min(10, len(close_prices))
            vol_prices = close_prices[-vol_window:]

            if len(vol_prices) > 1:
              recent_returns = np.diff(vol_prices) / vol_prices[:-1]
              context['recent_volatility'] = np.std(recent_returns)
        except Exception as vol_error:
          logger.warning(f"Ошибка расчета волатильности: {vol_error}")

        # 3. БЕЗОПАСНЫЙ АНАЛИЗ ОБЪЕМА
        try:
          if volumes is not None and len(volumes) >= 10:
            vol_window = min(10, len(volumes))
            if vol_window >= 5:
              recent_vol_avg = np.mean(volumes[-5:])
              if vol_window >= 10:
                previous_vol_avg = np.mean(volumes[-10:-5])

                if previous_vol_avg > 0:
                  vol_ratio = recent_vol_avg / previous_vol_avg
                  if vol_ratio > 1.2:
                    context['volume_trend'] = 'increasing'
                  elif vol_ratio < 0.8:
                    context['volume_trend'] = 'decreasing'
                  else:
                    context['volume_trend'] = 'stable'
        except Exception as volume_error:
          logger.warning(f"Ошибка анализа объема: {volume_error}")

        # 4. БЕЗОПАСНОЕ ОПРЕДЕЛЕНИЕ ФАЗЫ РЫНКА
        try:
          if len(close_prices) >= 20:
            short_ma = np.mean(close_prices[-5:])
            long_ma = np.mean(close_prices[-20:])

            if long_ma > 0:
              if short_ma > long_ma * 1.01:
                context['market_phase'] = 'bullish'
              elif short_ma < long_ma * 0.99:
                context['market_phase'] = 'bearish'
              else:
                context['market_phase'] = 'neutral'
        except Exception as phase_error:
          logger.warning(f"Ошибка определения фазы рынка: {phase_error}")

        # 5. БЕЗОПАСНАЯ КОРРЕКТИРОВКА УВЕРЕННОСТИ
        try:
          confidence_factors = []

          # Проверяем моментум
          momentum = context.get('price_momentum', 0)
          if abs(momentum) > 0.005:
            confidence_factors.append(1.1)
          else:
            confidence_factors.append(0.95)

          # Проверяем волатильность
          volatility = context.get('recent_volatility', 0)
          if 0.01 < volatility < 0.05:
            confidence_factors.append(1.05)
          else:
            confidence_factors.append(0.9)

          # Проверяем объем
          volume_trend = context.get('volume_trend', 'neutral')
          if volume_trend == 'increasing':
            confidence_factors.append(1.1)
          elif volume_trend == 'decreasing':
            confidence_factors.append(0.9)
          else:
            confidence_factors.append(1.0)

          if confidence_factors:
            context['confidence_adjustment'] = np.mean(confidence_factors)

        except Exception as conf_error:
          logger.warning(f"Ошибка корректировки уверенности: {conf_error}")
          context['confidence_adjustment'] = 1.0

        logger.debug(f"Контекст реального времени для {symbol}: {context}")
        return context

      except Exception as e:
        logger.error(f"Критическая ошибка получения контекста реального времени для {symbol}: {e}")
        return context

    def adjust_prediction_for_real_time(self, prediction: 'MLPrediction',
                                        real_time_context: Dict[str, Any]) -> 'MLPrediction':
      """
      Корректирует предсказание на основе контекста реального времени
      """
      try:
        if not self.real_time_adjustments:
          return prediction

        # Корректируем уверенность
        original_confidence = prediction.confidence
        adjusted_confidence = original_confidence * real_time_context.get('confidence_adjustment', 1.0)
        adjusted_confidence = np.clip(adjusted_confidence, 0.1, 1.0)

        # Корректируем сигнал на основе моментума
        momentum = real_time_context.get('price_momentum', 0.0)
        market_phase = real_time_context.get('market_phase', 'unknown')

        adjusted_signal = prediction.signal_type

        # Если моментум сильно противоречит сигналу, снижаем уверенность
        if prediction.signal_type == SignalType.BUY and momentum < -0.01:  # -1%
          adjusted_confidence *= 0.7
          logger.debug("BUY сигнал против отрицательного моментума")
        elif prediction.signal_type == SignalType.SELL and momentum > 0.01:  # +1%
          adjusted_confidence *= 0.7
          logger.debug("SELL сигнал против положительного моментума")

        # Если уверенность стала слишком низкой, переходим на HOLD
        if adjusted_confidence < 0.3 and adjusted_signal != SignalType.HOLD:
          adjusted_signal = SignalType.HOLD
          adjusted_confidence = 0.3
          logger.debug("Сигнал изменен на HOLD из-за низкой скорректированной уверенности")

        # Создаем скорректированное предсказание
        adjusted_prediction = MLPrediction(
          signal_type=adjusted_signal,
          probability=prediction.probability,
          confidence=adjusted_confidence,
          model_agreement=prediction.model_agreement,
          feature_importance=prediction.feature_importance,
          risk_assessment=prediction.risk_assessment,
          metadata={
            **prediction.metadata,
            'real_time_adjustments': {
              'original_confidence': original_confidence,
              'confidence_adjustment_factor': real_time_context.get('confidence_adjustment', 1.0),
              'momentum_impact': momentum,
              'market_phase': market_phase,
              'volume_trend': real_time_context.get('volume_trend', 'neutral')
            }
          }
        )

        if abs(adjusted_confidence - original_confidence) > 0.05:
          logger.info(
            f"Уверенность скорректирована в реальном времени: {original_confidence:.3f} -> {adjusted_confidence:.3f}")

        return adjusted_prediction

      except Exception as e:
        logger.error(f"Ошибка корректировки предсказания в реальном времени: {e}")
        return prediction

