#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
СКРИПТ ОБУЧЕНИЯ МОДЕЛИ ПРОГНОЗИРОВАНИЯ ВОЛАТИЛЬНОСТИ
====================================================

Этот скрипт предназначен для обучения и валидации системы прогнозирования 
волатильности на исторических данных с биржи Bybit.

Основные функции:
- Загрузка данных с Bybit API
- Предобработка и очистка данных
- Обучение множественных моделей
- Валидация и сравнение моделей
- Сохранение обученных моделей
- Генерация отчетов о качестве

Автор: Trading Bot System
Дата: 2025
"""

import os
import sys
import json
import pickle
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Импорт нашей системы волатильности
from ml.volatility_system import (
    VolatilityPredictionSystem,
    VolatilityPredictor,
    ModelType, logger, create_sample_data
)

# ============================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ============================================================================

def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Настройка системы логирования"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    
    return logging.getLogger(__name__)

# ============================================================================
# ЗАГРУЗКА ДАННЫХ С BYBIT
# ============================================================================

class BybitDataLoader:
    """Класс для загрузки данных с Bybit"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.bybit.com"
        
    def fetch_kline_data(self, 
                        symbol: str = "BTCUSDT",
                        interval: str = "60",  # 1 час
                        start_time: datetime = None,
                        end_time: datetime = None,
                        limit: int = 1000) -> pd.DataFrame:
        """
        Загрузка исторических данных свечей
        
        Args:
            symbol: Торговая пара
            interval: Интервал (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            start_time: Начальное время
            end_time: Конечное время
            limit: Максимальное количество записей
        """
        try:
            import requests
            
            # Если времена не указаны, берем последние данные
            if end_time is None:
                end_time = datetime.now()
            if start_time is None:
                start_time = end_time - timedelta(days=30)
            
            # Конвертируем в timestamp
            start_timestamp = int(start_time.timestamp() * 1000)
            end_timestamp = int(end_time.timestamp() * 1000)
            
            url = f"{self.base_url}/v5/market/kline"
            params = {
                'category': 'spot',
                'symbol': symbol,
                'interval': interval,
                'start': start_timestamp,
                'end': end_timestamp,
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['retCode'] == 0:
                klines = data['result']['list']
                
                # Преобразуем в DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                ])
                
                # Конвертируем типы данных
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
                df[numeric_cols] = df[numeric_cols].astype(float)
                
                # Сортируем по времени
                df = df.sort_values('timestamp').reset_index(drop=True)
                df.set_index('timestamp', inplace=True)
                
                return df
            else:
                raise Exception(f"Ошибка API: {data['retMsg']}")
                
        except ImportError:
            logging.warning("Модуль requests не найден. Используем тестовые данные.")
            return self._generate_test_data(symbol, start_time, end_time)
        except Exception as e:
            logging.error(f"Ошибка загрузки данных: {e}")
            logging.info("Используем тестовые данные")
            return self._generate_test_data(symbol, start_time, end_time)
    
    def _generate_test_data(self, symbol: str, start_time: datetime, 
                          end_time: datetime) -> pd.DataFrame:
        """Генерация тестовых данных при недоступности API"""
        days_diff = (end_time - start_time).days
        n_points = max(100, days_diff * 24)  # По часу
        
        data = create_sample_data(n_points)
        data.index = pd.date_range(start=start_time, end=end_time, periods=n_points)
        
        return data


# ============================================================================
# УЛУЧШЕННЫЙ КЛАСС ПРЕДОБРАБОТКИ ДАННЫХ
# ============================================================================

class ImprovedDataPreprocessor:
    """Улучшенный класс для предобработки данных с обработкой NaN"""

    def __init__(self, imputation_strategy: str = 'median'):
        self.logger = logging.getLogger(__name__)
        self.imputation_strategy = imputation_strategy
        self.scaler = RobustScaler()  # Более устойчив к выбросам
        self.imputer = None
        self.feature_columns = None

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Улучшенная очистка данных от аномалий и пропусков"""
        df = data.copy()
        initial_len = len(df)

        self.logger.info(f"Начальное количество записей: {initial_len}")

        # Удаляем полностью пустые строки
        df = df.dropna(how='all')
        self.logger.info(f"После удаления пустых строк: {len(df)}")

        # Удаляем дубликаты по индексу
        df = df[~df.index.duplicated(keep='first')]
        self.logger.info(f"После удаления дубликатов: {len(df)}")

        # Проверяем основные OHLC колонки
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

        # Удаляем строки где все OHLC значения NaN
        df = df.dropna(subset=required_cols, how='all')
        self.logger.info(f"После удаления строк без OHLC: {len(df)}")

        # Удаляем строки с нулевыми объемами (если есть volume)
        if 'volume' in df.columns:
            initial_vol_len = len(df)
            df = df[(df['volume'] > 0) | df['volume'].isna()]
            if len(df) < initial_vol_len:
                self.logger.info(f"Удалено {initial_vol_len - len(df)} строк с нулевым объемом")

        # Базовая валидация OHLC (только для не-NaN значений)
        valid_mask = (
            (df['high'].isna() | df['open'].isna() | (df['high'] >= df['open'])) &
            (df['high'].isna() | df['close'].isna() | (df['high'] >= df['close'])) &
            (df['low'].isna() | df['open'].isna() | (df['low'] <= df['open'])) &
            (df['low'].isna() | df['close'].isna() | (df['low'] <= df['close'])) &
            (df['high'].isna() | df['low'].isna() | (df['high'] >= df['low']))
        )
        df = df[valid_mask]
        self.logger.info(f"После валидации OHLC: {len(df)}")

        # Удаляем экстремальные выбросы (более 50% изменения за период)
        if len(df) > 1:
            returns = df['close'].pct_change()
            outlier_mask = (returns.abs() < 0.5) | returns.isna()
            df = df[outlier_mask]
            self.logger.info(f"После удаления выбросов: {len(df)}")

        # Сортируем по времени
        df = df.sort_index()

        removed_count = initial_len - len(df)
        self.logger.info(f"Итого удалено {removed_count} записей из {initial_len}")

        return df

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Добавление технических индикаторов с обработкой NaN"""
        df = data.copy()

        try:
            # Базовые индикаторы цены
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['price_range'] = (df['high'] - df['low']) / df['close']
            df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

            # Скользящие средние (более короткие периоды для уменьшения NaN)
            for period in [5, 10, 20]:
                df[f'sma_{period}'] = df['close'].rolling(period, min_periods=max(1, period // 2)).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period, min_periods=max(1, period // 2)).mean()

                # Отношения к скользящим средним
                df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
                df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}']

            # Волатильность на разных периодах
            for period in [5, 10, 20]:
                min_periods = max(2, period // 3)
                df[f'volatility_{period}'] = df['returns'].rolling(period, min_periods=min_periods).std()
                df[f'high_low_vol_{period}'] = ((df['high'] - df['low']) / df['close']).rolling(period,
                                                                                                min_periods=min_periods).mean()

            # Простые индикаторы моментума
            for period in [3, 5, 10]:
                df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
                df[f'roc_{period}'] = df['close'].pct_change(period)

            # RSI (упрощенная версия)
            period = 14
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period, min_periods=period // 2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=period // 2).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Индикаторы объема (если доступны)
            if 'volume' in df.columns:
                df['volume_ma'] = df['volume'].rolling(10, min_periods=3).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']

                # VWAP (упрощенный)
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                df['vwap'] = (typical_price * df['volume']).rolling(20, min_periods=5).sum() / df['volume'].rolling(20,
                                                                                                                    min_periods=5).sum()
                df['price_to_vwap'] = df['close'] / df['vwap']

            # Время-основанные признаки
            df['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
            df['day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0

            self.logger.info(f"Добавлено технических индикаторов. Размерность: {df.shape}")

        except Exception as e:
            self.logger.error(f"Ошибка при создании технических индикаторов: {e}")
            # В случае ошибки возвращаем хотя бы базовые индикаторы
            df['returns'] = df['close'].pct_change()
            df['volatility_5'] = df['returns'].rolling(5, min_periods=2).std()
            df['volatility_20'] = df['returns'].rolling(20, min_periods=5).std()

        return df

    def prepare_features_with_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков с импутацией пропущенных значений"""
        df = data.copy()

        # Определяем признаки для использования (исключаем некоторые базовые колонки)
        exclude_cols = ['open', 'high', 'low', 'turnover'] if 'turnover' in df.columns else ['open', 'high', 'low']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        self.logger.info(f"Используемые признаки ({len(feature_cols)}): {feature_cols[:10]}...")

        # Проверяем количество NaN по колонкам
        nan_counts = df[feature_cols].isna().sum()
        problematic_cols = nan_counts[nan_counts > len(df) * 0.5].index.tolist()

        if problematic_cols:
            self.logger.warning(f"Колонки с >50% NaN будут удалены: {problematic_cols}")
            feature_cols = [col for col in feature_cols if col not in problematic_cols]

        # Сохраняем список признаков
        self.feature_columns = feature_cols
        df_features = df[feature_cols].copy()

        # Удаляем строки где слишком много NaN (>70% признаков)
        max_nan_per_row = len(feature_cols) * 0.7
        valid_rows = df_features.isna().sum(axis=1) <= max_nan_per_row
        df_features = df_features[valid_rows]

        self.logger.info(f"После фильтрации строк с множественными NaN: {len(df_features)} строк")

        if len(df_features) == 0:
            raise ValueError("Не осталось валидных строк после фильтрации NaN")

        # Импутация пропущенных значений
        if self.imputation_strategy == 'knn' and len(df_features) > 50:
            # KNN импутер для больших датасетов
            self.imputer = KNNImputer(n_neighbors=min(5, len(df_features) // 10), weights='distance')
        else:
            # Простая импутация медианой/средним
            strategy = 'median' if self.imputation_strategy in ['median', 'knn'] else 'mean'
            self.imputer = SimpleImputer(strategy=strategy)

        # Применяем импутацию
        df_imputed = pd.DataFrame(
            self.imputer.fit_transform(df_features),
            index=df_features.index,
            columns=df_features.columns
        )

        # Проверяем результат
        remaining_nan = df_imputed.isna().sum().sum()
        if remaining_nan > 0:
            self.logger.warning(f"После импутации осталось {remaining_nan} NaN значений")
            # Заполняем оставшиеся NaN нулями
            df_imputed = df_imputed.fillna(0)

        # Заменяем inf значения
        df_imputed = df_imputed.replace([np.inf, -np.inf], 0)

        self.logger.info(f"Импутация завершена. Финальная размерность: {df_imputed.shape}")

        return df_imputed

    def prepare_volatility_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Подготовка данных для обучения системы волатильности

        Returns:
            X_train, y_train, X_test, y_test - уже разделенные данные
        """
        logger.info("Подготовка данных для системы волатильности...")

        try:
            # Создаем временный предиктор только для подготовки данных
            temp_predictor = VolatilityPredictor(prediction_horizon=5)

            # Подготавливаем данные (получаем X, y)
            X, y = temp_predictor.prepare_training_data(data)

            logger.info(f"Подготовлено {len(X)} записей с {len(X.columns)} признаками")

            # Разделяем на train/test (80/20)
            split_point = int(len(X) * 0.8)

            X_train = X.iloc[:split_point]
            y_train = y.iloc[:split_point]
            X_test = X.iloc[split_point:]
            y_test = y.iloc[split_point:]

            logger.info(f"Разделение данных: train={len(X_train)}, test={len(X_test)}")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"Ошибка подготовки данных волатильности: {e}")
            raise


    def fit_scaler(self, data: pd.DataFrame) -> pd.DataFrame:
        """Обучение и применение масштабирования"""
        if self.feature_columns is None:
            raise ValueError("Сначала нужно подготовить признаки с импутацией")

        # Применяем масштабирование
        scaled_data = self.scaler.fit_transform(data)

        return pd.DataFrame(
            scaled_data,
            index=data.index,
            columns=data.columns
        )

    def transform_scaler(self, data: pd.DataFrame) -> pd.DataFrame:
        """Применение обученного масштабирования"""
        if self.scaler is None:
            raise ValueError("Масштабировщик не обучен")

        scaled_data = self.scaler.transform(data)

        return pd.DataFrame(
            scaled_data,
            index=data.index,
            columns=data.columns
        )

# ============================================================================
# КЛАСС ОБУЧЕНИЯ МОДЕЛИ
# ============================================================================

class ImprovedVolatilityModelTrainer:
    """Основной класс для обучения модели волатильности"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.preprocessor = ImprovedDataPreprocessor(
            imputation_strategy=config.get('imputation_strategy', 'median')
        )

        # Создаем директории
        os.makedirs(config.get('model_dir', 'models'), exist_ok=True)
        os.makedirs(config.get('reports_dir', 'reports'), exist_ok=True)
        os.makedirs(config.get('plots_dir', 'plots'), exist_ok=True)

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Загрузка и улучшенная подготовка данных"""
        self.logger.info("Загрузка данных...")

        # Здесь должна быть загрузка данных (упрощенная версия)
        try:
            # Попытка загрузки с Bybit или использование тестовых данных
            from ml.volatility_system import create_sample_data

            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.config.get('data_days', 365))
            n_points = self.config.get('data_days', 365) * 24  # Часовые данные

            data = create_sample_data(n_points)
            data.index = pd.date_range(start=start_time, end=end_time, periods=n_points)

            self.logger.info(f"Загружено {len(data)} записей с {data.index[0]} по {data.index[-1]}")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки данных: {e}")
            raise

            # Улучшенная предобработка
        self.logger.info("Предобработка данных...")
        data = self.preprocessor.clean_data(data)
        data = self.preprocessor.add_technical_indicators(data)

        # Подготовка признаков с импутацией
        data = self.preprocessor.prepare_features_with_imputation(data)

        # Масштабирование
        data = self.preprocessor.fit_scaler(data)

        self.logger.info(f"Финальная предобработка завершена: {data.shape}")

        return data

    # def prepare_volatility_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    #     """
    #     Подготовка данных для обучения системы волатильности
    #
    #     Returns:
    #         X_train, y_train, X_test, y_test - уже разделенные данные
    #     """
    #     self.logger.info("Подготовка данных для системы волатильности...")
    #
    #     try:
    #         # Создаем временный предиктор только для подготовки данных
    #         from ml.volatility_system import VolatilityPredictor
    #         temp_predictor = VolatilityPredictor(prediction_horizon=5)
    #
    #         # Подготавливаем данные (получаем X, y)
    #         X, y = temp_predictor.prepare_training_data(data)
    #
    #         self.logger.info(f"Подготовлено {len(X)} записей с {len(X.columns)} признаками")
    #
    #         # Разделяем на train/test (80/20)
    #         split_point = int(len(X) * 0.8)
    #
    #         X_train = X.iloc[:split_point]
    #         y_train = y.iloc[:split_point]
    #         X_test = X.iloc[split_point:]
    #         y_test = y.iloc[split_point:]
    #
    #         self.logger.info(f"Разделение данных: train={len(X_train)}, test={len(X_test)}")
    #
    #         return X_train, y_train, X_test, y_test
    #
    #     except Exception as e:
    #         self.logger.error(f"Ошибка подготовки данных волатильности: {e}")
    #         raise

    def train_volatility_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Обучение моделей волатильности"""
        results = {}

        try:
            # Подготавливаем данные
            X_train, y_train, X_test, y_test = self.preprocessor.prepare_volatility_data(data)

            # Список моделей для обучения
            models_to_train = ['random_forest', 'gradient_boosting', 'xgboost']

            for model_name in models_to_train:
                self.logger.info(f"  Обучение модели {model_name}...")

                try:
                    # Создаем систему волатильности для каждой модели
                    model_type_map = {
                        'random_forest': ModelType.RANDOM_FOREST,
                        'gradient_boosting': ModelType.GRADIENT_BOOSTING,
                        'xgboost': ModelType.XGBOOST
                    }

                    system = VolatilityPredictionSystem(
                        model_type=model_type_map[model_name],
                        prediction_horizon=5,
                        auto_retrain=False
                    )

                    # Инициализируем систему с подготовленными данными
                    init_result = system.initialize(X_train, y_train, X_test, y_test)

                    if init_result['status'] == 'success':
                        self.logger.info(f"    ✓ Модель {model_name} успешно обучена")

                        # Сохраняем результаты
                        results[model_name] = {
                            'status': 'success',
                            'model_scores': init_result['model_scores'],
                            'system': system  # Сохраняем обученную систему
                        }

                        # Выводим метрики
                        if model_name in init_result['model_scores']:
                            scores = init_result['model_scores'][model_name]
                            if 'error' not in scores:
                                self.logger.info(f"    Метрики: R²={scores['r2']:.4f}, RMSE={scores['rmse']:.6f}")
                            else:
                                self.logger.error(f"    Ошибка в метриках: {scores['error']}")
                    else:
                        self.logger.error(
                            f"    ✗ Ошибка инициализации {model_name}: {init_result.get('error', 'Unknown error')}")
                        results[model_name] = {
                            'status': 'error',
                            'error': init_result.get('error', 'Unknown error')
                        }

                except Exception as e:
                    self.logger.error(f"    ✗ Исключение при обучении {model_name}: {e}")
                    results[model_name] = {
                        'status': 'error',
                        'error': str(e)
                    }

            # Создаем финальную ensemble систему
            self.logger.info("  Создание ensemble модели...")
            try:
                ensemble_system = VolatilityPredictionSystem(
                    model_type=ModelType.ENSEMBLE,
                    prediction_horizon=5,
                    auto_retrain=True
                )

                ensemble_result = ensemble_system.initialize(X_train, y_train, X_test, y_test)

                if ensemble_result['status'] == 'success':
                    self.logger.info("    ✓ Ensemble модель успешно создана")
                    results['ensemble'] = {
                        'status': 'success',
                        'model_scores': ensemble_result['model_scores'],
                        'system': ensemble_system
                    }

                    # Выводим общую статистику по ensemble
                    self.logger.info(f"    Ensemble использует {len(ensemble_result['model_scores'])} моделей")
                else:
                    self.logger.error(f"    ✗ Ошибка создания ensemble: {ensemble_result.get('error')}")

            except Exception as e:
                self.logger.error(f"    ✗ Исключение при создании ensemble: {e}")

        except Exception as e:
            self.logger.error(f"Критическая ошибка в train_volatility_models: {e}")

        return results
    
    def _validate_model(self, system: VolatilityPredictionSystem, 
                       test_data: pd.DataFrame) -> Dict[str, float]:
        """Валидация модели на тестовых данных"""
        predictions = []
        actual_values = []
        
        # Получаем предсказания для каждой точки тестовых данных
        for i in range(len(test_data) - system.predictor.prediction_horizon):
            try:
                # Данные до текущего момента
                current_data = test_data[:i+1]
                
                if len(current_data) < 50:  # Минимум данных для предсказания
                    continue
                
                # Получаем предсказание
                prediction = system.get_prediction(current_data)
                
                # Фактическое значение через prediction_horizon периодов
                future_idx = i + system.predictor.prediction_horizon
                if future_idx < len(test_data):
                    # Рассчитываем фактическую волатильность
                    future_data = test_data[i:future_idx+20]  # +20 для расчета волатильности
                    if len(future_data) >= 20:
                        actual_vol = future_data['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
                        
                        if not np.isnan(actual_vol) and not np.isnan(prediction.predicted_volatility):
                            predictions.append(prediction.predicted_volatility)
                            actual_values.append(actual_vol)
                            
                            # Добавляем обратную связь
                            system.add_prediction_feedback(prediction, actual_vol)
                
            except Exception as e:
                continue
        
        if len(predictions) < 10:
            return {'error': 'Недостаточно валидных предсказаний'}
        
        # Рассчитываем метрики
        mse = mean_squared_error(actual_values, predictions)
        mae = mean_absolute_error(actual_values, predictions)
        r2 = r2_score(actual_values, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse),
            'samples': len(predictions),
            'predictions': predictions,
            'actual': actual_values
        }

    def select_best_models(self) -> Dict[str, str]:
        """Выбор лучших моделей"""
        best_models = {}

        if not self.results:
            self.logger.warning("Нет результатов для выбора лучших моделей")
            return best_models

        best_r2 = -float('inf')
        best_model = None

        for model_name, model_results in self.results.items():
            if model_results.get('status') == 'success':
                # Получаем метрики из model_scores
                model_scores = model_results.get('model_scores', {})

                # Ищем R² в разных возможных местах
                r2 = None
                if model_name in model_scores and 'r2' in model_scores[model_name]:
                    r2 = model_scores[model_name]['r2']
                elif 'r2' in model_scores:
                    r2 = model_scores['r2']

                if r2 is not None and r2 > best_r2:
                    best_r2 = r2
                    best_model = model_name

        if best_model:
            best_models['volatility_5p'] = best_model  # Добавляем ключ для совместимости
            self.logger.info(f"Лучшая модель: {best_model} (R² = {best_r2:.4f})")

        return best_models

    def save_models(self, results: Dict[str, Any], models_dir: str = None):
        """Сохранение обученных моделей волатильности"""

        if models_dir is None:
            models_dir = self.config.get('model_dir', 'models')

        models_dir = os.path.join(models_dir, 'volatility')

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        saved_models = {}

        for model_name, result in results.items():
            if result.get('status') == 'success' and 'system' in result:
                try:
                    model_path = os.path.join(models_dir, f"{model_name}_system.pkl")

                    # Сохраняем всю систему
                    joblib.dump(result['system'], model_path)

                    saved_models[model_name] = {
                        'path': model_path,
                        'model_scores': result.get('model_scores', {}),
                        'saved_at': datetime.now().isoformat()
                    }

                    self.logger.info(f"  ✓ Модель {model_name} сохранена: {model_path}")

                except Exception as e:
                    self.logger.error(f"  ✗ Ошибка сохранения {model_name}: {e}")
                    saved_models[model_name] = {'error': str(e)}

        # Сохраняем метаинформацию
        meta_path = os.path.join(models_dir, "models_meta.json")
        try:
            with open(meta_path, 'w') as f:
                json.dump(saved_models, f, indent=2, default=str)
            self.logger.info(f"  ✓ Метаинформация сохранена: {meta_path}")
        except Exception as e:
            self.logger.error(f"  ✗ Ошибка сохранения метаинформации: {e}")

        return saved_models


    # def save_models(self, best_models: Dict[str, str]):
    #     """Сохранение обученных моделей"""
    #     model_dir = self.config.get('model_dir', 'models')
    #
    #     for horizon_key, best_model_name in best_models.items():
    #         try:
    #             system = self.results[horizon_key][best_model_name]['system']
    #
    #             # Путь для сохранения
    #             model_path = os.path.join(model_dir, f"{horizon_key}_{best_model_name}.pkl")
    #
    #             # Сохраняем систему
    #             with open(model_path, 'wb') as f:
    #                 pickle.dump(system, f)
    #
    #             self.logger.info(f"Модель {horizon_key}_{best_model_name} сохранена в {model_path}")
    #
    #         except Exception as e:
    #             self.logger.error(f"Ошибка сохранения модели {horizon_key}_{best_model_name}: {e}")
    
    def generate_report(self, best_models: Dict[str, str]):
        """Генерация отчета о результатах обучения"""
        report_path = os.path.join(
            self.config.get('reports_dir', 'reports'),
            f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'best_models': best_models,
            'detailed_results': {}
        }
        
        # Добавляем детальные результаты
        for horizon_key, horizon_results in self.results.items():
            report['detailed_results'][horizon_key] = {}
            
            for model_name, model_results in horizon_results.items():
                # Убираем объект системы из отчета (не сериализуется в JSON)
                clean_results = {k: v for k, v in model_results.items() if k != 'system'}
                
                # Очищаем массивы предсказаний (слишком большие для JSON)
                if 'validation_scores' in clean_results:
                    val_scores = clean_results['validation_scores'].copy()
                    if 'predictions' in val_scores:
                        val_scores['predictions'] = f"Array of {len(val_scores['predictions'])} predictions"
                    if 'actual' in val_scores:
                        val_scores['actual'] = f"Array of {len(val_scores['actual'])} actual values"
                    clean_results['validation_scores'] = val_scores
                
                report['detailed_results'][horizon_key][model_name] = clean_results
        
        # Сохраняем отчет
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Отчет сохранен в {report_path}")
        
        return report_path
    
    def create_plots(self, best_models: Dict[str, str]):
        """Создание графиков результатов"""
        plots_dir = self.config.get('plots_dir', 'plots')
        
        # Настройка стиля графиков
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # График сравнения моделей по R²
        self._plot_model_comparison()
        
        # Графики предсказаний vs фактических значений
        for horizon_key, best_model_name in best_models.items():
            try:
                val_scores = self.results[horizon_key][best_model_name]['validation_scores']
                if 'predictions' in val_scores and 'actual' in val_scores:
                    self._plot_predictions_vs_actual(
                        val_scores['predictions'], 
                        val_scores['actual'],
                        f"{horizon_key}_{best_model_name}",
                        plots_dir
                    )
            except Exception as e:
                self.logger.error(f"Ошибка создания графика для {horizon_key}: {e}")
    
    def _plot_model_comparison(self):
        """График сравнения качества моделей"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Сравнение качества моделей прогнозирования волатильности', fontsize=16)
        
        metrics = ['r2', 'rmse', 'mae', 'mse']
        metric_names = ['R² Score', 'RMSE', 'MAE', 'MSE']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            data_for_plot = []
            labels = []
            
            for horizon_key, horizon_results in self.results.items():
                for model_name, model_results in horizon_results.items():
                    val_scores = model_results.get('validation_scores', {})
                    if metric in val_scores:
                        data_for_plot.append(val_scores[metric])
                        labels.append(f"{horizon_key.split('_')[1]}p_{model_name}")
            
            if data_for_plot:
                ax.bar(range(len(data_for_plot)), data_for_plot)
                ax.set_title(metric_name)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.get('plots_dir', 'plots'), 'model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"График сравнения моделей сохранен в {plot_path}")
    
    def _plot_predictions_vs_actual(self, predictions: List[float], actual: List[float],
                                  model_name: str, plots_dir: str):
        """График предсказанных vs фактических значений"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(actual, predictions, alpha=0.6)
        ax1.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', lw=2)
        ax1.set_xlabel('Фактическая волатильность')
        ax1.set_ylabel('Предсказанная волатильность')
        ax1.set_title(f'Предсказания vs Факт: {model_name}')
        ax1.grid(True, alpha=0.3)
        
        # Временной ряд
        ax2.plot(actual, label='Фактическая', alpha=0.7)
        ax2.plot(predictions, label='Предсказанная', alpha=0.7)
        ax2.set_xlabel('Время')
        ax2.set_ylabel('Волатильность')
        ax2.set_title(f'Временной ряд: {model_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f'predictions_{model_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def validate_data_quality(self, data: pd.DataFrame) -> bool:
        """Валидация качества данных перед обучением"""
        # Проверка на NaN
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            self.logger.error(f"Найдено {nan_count} NaN значений в подготовленных данных")
            return False

        # Проверка на inf
        inf_count = np.isinf(data.values).sum()
        if inf_count > 0:
            self.logger.error(f"Найдено {inf_count} бесконечных значений")
            return False

        # Проверка размерности
        if len(data) < 100:
            self.logger.error(f"Недостаточно данных: {len(data)} < 100")
            return False

        if data.shape[1] < 3:
            self.logger.error(f"Недостаточно признаков: {data.shape[1]} < 3")
            return False

        self.logger.info(f"✓ Данные прошли валидацию: {data.shape}")
        return True

# ============================================================================
# КОНФИГУРАЦИЯ И ЗАПУСК
# ============================================================================

def load_config(config_path: str) -> Dict:
    """Загрузка конфигурации из файла"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Конфигурация по умолчанию
        return {
            "symbol": "BTCUSDT",
            "interval": "60",
            "data_days": 365,
            "train_ratio": 0.8,
            "prediction_horizons": [1, 3, 5, 10],
            "model_dir": "models",
            "reports_dir": "reports",
            "plots_dir": "plots",
            "log_level": "INFO"
        }

def create_default_config(config_path: str):
    """Создание файла конфигурации по умолчанию"""
    default_config = {
        "symbol": "BTCUSDT",
        "interval": "60",
        "data_days": 365,
        "train_ratio": 0.8,
        "prediction_horizons": [1, 3, 5, 10],
        "model_dir": "models",
        "reports_dir": "reports", 
        "plots_dir": "plots",
        "log_level": "INFO",
        "api_key": "",
        "api_secret": ""
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)
    
    print(f"Создан файл конфигурации: {config_path}")
    print("Отредактируйте его при необходимости и запустите скрипт снова.")


def main():
    """
    ФИНАЛЬНАЯ ВЕРСИЯ: Главная функция, которая корректно использует
    класс ImprovedVolatilityModelTrainer для управления всем процессом.
    """
    # ... (код парсинга аргументов и настройки логирования без изменений)
    parser = argparse.ArgumentParser(description='Обучение модели прогнозирования волатильности')
    parser.add_argument('--config', '-c', default='config.json', help='Путь к файлу конфигурации')
    parser.add_argument('--create-config', action='store_true', help='Создать файл конфигурации')
    parser.add_argument('--log-file', help='Файл для логов')
    args = parser.parse_args()

    if args.create_config:
        create_default_config(args.config)
        return 0

    config = load_config(args.config)
    logger = setup_logging(config.get('log_level', 'INFO'), args.log_file)

    logger.info("=== ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ ПРОГНОЗИРОВАНИЯ ВОЛАТИЛЬНОСТИ ===")

    try:
        # --- ИНИЦИАЛИЗАЦИЯ И ИСПОЛЬЗОВАНИЕ ТРЕНЕРА ---

        # 1. Создаем экземпляр нашего тренера
        trainer = ImprovedVolatilityModelTrainer(config)

        # 2. Этап 1: Загрузка и полная подготовка данных (теперь это делает тренер)
        logger.info("\n--- ЭТАП 1: ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ ---")
        # Этот метод загрузит сырые данные и полностью их обработает, создав все признаки
        processed_data = trainer.load_and_prepare_data()

        # 3. Валидация качества уже обработанных данных
        if not trainer.validate_data_quality(processed_data):
            logger.error("Данные не прошли валидацию. Обучение прервано.")
            return 1

        # 4. Этап 2: Обучение моделей на полностью готовых данных
        logger.info("\n--- ЭТАП 2: ОБУЧЕНИЕ МОДЕЛЕЙ ---")
        # Передаем уже полностью обработанные данные
        training_results = trainer.train_volatility_models(processed_data)

        if not training_results:
            logger.error("Не удалось обучить ни одной модели.")
            return 1

        trainer.results = training_results  # Сохраняем результаты для отчетов

        # 5. Этап 3: Выбор лучших моделей
        logger.info("\n--- ЭТАП 3: ВЫБОР ЛУЧШИХ МОДЕЛЕЙ ---")
        best_models = trainer.select_best_models()

        if not best_models:
            logger.error("Не удалось выбрать лучшие модели.")
            return 1

        # 6. Этап 4: Сохранение моделей
        logger.info("\n--- ЭТАП 4: СОХРАНЕНИЕ МОДЕЛЕЙ ---")
        trainer.save_models(training_results)  # Передаем полные результаты

        # 7. Этапы 5 и 6: Генерация отчета и графиков
        logger.info("\n--- ЭТАП 5: СОЗДАНИЕ ОТЧЕТА ---")
        report_path = trainer.generate_report(best_models)
        logger.info("\n--- ЭТАП 6: СОЗДАНИЕ ГРАФИКОВ ---")
        trainer.create_plots(best_models)

        logger.info("\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО ===")
        logger.info(f"Отчет сохранен: {report_path}")

    except Exception as e:
        logger.error(f"\nКритическая ошибка во время обучения: {e}", exc_info=True)
        return 1

    return 0


def run_quick_test():
    """Быстрый тест улучшенной системы"""
    logger = setup_logging('INFO')
    logger.info("=== БЫСТРЫЙ ТЕСТ УЛУЧШЕННОЙ СИСТЕМЫ ===")

    try:
        config = {'imputation_strategy': 'median', 'data_days': 30}
        trainer = ImprovedVolatilityModelTrainer(config)

        # Тест предобработки
        data = trainer.load_and_prepare_data()

        if trainer.validate_data_quality(data):
            logger.info("✓ Тест предобработки пройден успешно")
            return 0
        else:
            logger.error("✗ Тест предобработки провален")
            return 1

    except Exception as e:
        logger.error(f"✗ Ошибка теста: {e}")
        return 1


def load_config(config_path: str) -> Dict:
    """Загрузка конфигурации"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "symbol": "BTCUSDT",
        "interval": "60",
        "data_days": 365,
        "train_ratio": 0.8,
        "prediction_horizons": [1, 3, 5, 10],
        "model_dir": "models",
        "reports_dir": "reports",
        "plots_dir": "plots",
        "log_level": "INFO",
        "imputation_strategy": "median"
    }


def create_default_config(config_path: str):
    """Создание конфигурации по умолчанию"""
    config = load_config("")  # Получаем дефолтную конфигурацию
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Конфигурация создана: {config_path}")


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Настройка логирования"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

    return logging.getLogger(__name__)


if __name__ == "__main__":
    sys.exit(main())


# def main():
#     """Главная функция скрипта"""
#     parser = argparse.ArgumentParser(description='Обучение модели прогнозирования волатильности')
#     parser.add_argument('--config', '-c', default='config.json',
#                         help='Путь к файлу конфигурации')
#     parser.add_argument('--create-config', action='store_true',
#                         help='Создать файл конфигурации по умолчанию')
#     parser.add_argument('--log-file', help='Файл для логов')
#
#     args = parser.parse_args()
#
#     # Создание конфигурации если требуется
#     if args.create_config:
#         create_default_config(args.config)
#         return
#
#     # Загрузка конфигурации
#     config = load_config(args.config)
#
#     # Настройка логирования
#     logger = setup_logging(config.get('log_level', 'INFO'), args.log_file)
#
#     logger.info("=== ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ ПРОГНОЗИРОВАНИЯ ВОЛАТИЛЬНОСТИ ===")
#     logger.info(f"Конфигурация загружена из: {args.config}")
#     logger.info(f"Символ: {config.get('symbol', 'BTCUSDT')}")
#     logger.info(f"Период данных: {config.get('data_days', 365)} дней")
#     logger.info(f"Горизонты прогнозирования: {config.get('prediction_horizons', [1, 3, 5, 10])}")
#
#     try:
#         # Инициализация тренера
#         trainer = VolatilityModelTrainer(config)
#
#         # Этап 1: Загрузка и подготовка данных
#         logger.info("\n--- ЭТАП 1: ЗАГРУЗКА ДАННЫХ ---")
#         data = trainer.load_and_prepare_data()
#
#         if len(data) < 100:
#             logger.error("Недостаточно данных для обучения (минимум 100 записей)")
#             return
#
#         # Этап 2: Обучение моделей
#         logger.info("\n--- ЭТАП 2: ОБУЧЕНИЕ МОДЕЛЕЙ ---")
#         training_results = trainer.train_models(data)
#
#         if not training_results:
#             logger.error("Не удалось обучить ни одной модели")
#             return
#
#         # Этап 3: Выбор лучших моделей
#         logger.info("\n--- ЭТАП 3: ВЫБОР ЛУЧШИХ МОДЕЛЕЙ ---")
#         best_models = trainer.select_best_models()
#
#         if not best_models:
#             logger.error("Не удалось найти подходящие модели")
#             return
#
#         # Этап 4: Сохранение моделей
#         logger.info("\n--- ЭТАП 4: СОХРАНЕНИЕ МОДЕЛЕЙ ---")
#         trainer.save_models(best_models)
#
#         # Этап 5: Генерация отчета
#         logger.info("\n--- ЭТАП 5: СОЗДАНИЕ ОТЧЕТА ---")
#         report_path = trainer.generate_report(best_models)
#
#         # Этап 6: Создание графиков
#         logger.info("\n--- ЭТАП 6: СОЗДАНИЕ ГРАФИКОВ ---")
#         trainer.create_plots(best_models)
#
#         # Финальная сводка
#         logger.info("\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО ===")
#         logger.info(f"Обучено моделей: {len(best_models)}")
#         logger.info(f"Отчет сохранен: {report_path}")
#         logger.info(f"Модели сохранены в: {config.get('model_dir', 'models')}")
#         logger.info(f"Графики сохранены в: {config.get('plots_dir', 'plots')}")
#
#         # Краткая сводка по лучшим моделям
#         logger.info("\nЛучшие модели:")
#         for horizon_key, model_name in best_models.items():
#             horizon = horizon_key.split('_')[1]
#             val_scores = trainer.results[horizon_key][model_name]['validation_scores']
#             r2 = val_scores.get('r2', 'N/A')
#             rmse = val_scores.get('rmse', 'N/A')
#             logger.info(f"  {horizon} периодов: {model_name} (R²={r2:.4f}, RMSE={rmse:.4f})")
#
#         # Рекомендации по использованию
#         logger.info("\nРекомендации:")
#         logger.info("1. Используйте модели с R² > 0.3 для практического применения")
#         logger.info("2. Регулярно переобучайте модели на новых данных")
#         logger.info("3. Комбинируйте прогнозы разных горизонтов для лучшего результата")
#
#     except KeyboardInterrupt:
#         logger.info("\nОбучение прервано пользователем")
#     except Exception as e:
#         logger.error(f"\nОшибка во время обучения: {e}")
#         logger.exception("Детали ошибки:")
#         return 1
#
#     return 0
#
#
# def run_quick_test():
#     """Быстрый тест системы с демо-данными"""
#     logger = setup_logging('INFO')
#     logger.info("=== БЫСТРЫЙ ТЕСТ СИСТЕМЫ ===")
#
#     try:
#         # Создаем тестовые данные
#         test_data = create_sample_data(1000)
#         logger.info(f"Создано {len(test_data)} тестовых записей")
#
#         # Инициализируем простую систему
#         system = VolatilityPredictionSystem(
#             model_type=ModelType.RANDOM_FOREST,
#             prediction_horizon=5
#         )
#
#         # Обучаем на тестовых данных
#         result = system.initialize(test_data[:800])  # 80% для обучения
#
#         if result['status'] == 'success':
#             logger.info("✓ Система успешно инициализирована")
#
#             # Тестируем предсказания
#             test_predictions = []
#             for i in range(800, 950):  # Тестируем на оставшихся данных
#                 pred = system.get_volatility_prediction(test_data[:i])
#                 test_predictions.append(pred.predicted_volatility)
#
#             logger.info(f"✓ Получено {len(test_predictions)} предсказаний")
#             logger.info(f"  Средняя предсказанная волатильность: {np.mean(test_predictions):.4f}")
#             logger.info(f"  Стандартное отклонение: {np.std(test_predictions):.4f}")
#
#             # Анализ паттернов волатильности
#             patterns = analyze_volatility_patterns(test_data)
#             logger.info("✓ Анализ паттернов волатильности:")
#             logger.info(f"  Средняя волатильность: {patterns['mean_volatility']:.4f}")
#             logger.info(f"  Режим волатильности: {patterns['current_regime'].value}")
#
#             logger.info("=== ТЕСТ ПРОЙДЕН УСПЕШНО ===")
#
#         else:
#             logger.error(f"✗ Ошибка инициализации: {result['error']}")
#             return 1
#
#     except Exception as e:
#         logger.error(f"✗ Ошибка во время теста: {e}")
#         return 1
#
#     return 0
#
#
# if __name__ == "__main__":
#     # Проверяем аргументы командной строки
#     if len(sys.argv) > 1 and sys.argv[1] == '--quick-test':
#         sys.exit(run_quick_test())
#     else:
#         sys.exit(main())


# # Создание конфигурации по умолчанию
# python volatility_training_script.py --create-config
#
# # Запуск обучения
# python volatility_training_script.py --config config.json
#
# # Быстрый тест системы
# python volatility_training_script.py --quick-test
#
# # Обучение с логированием в файл
# python volatility_training_script.py --log-file training.log