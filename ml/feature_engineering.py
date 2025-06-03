import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    Продвинутый класс для создания признаков и меток для машинного обучения
    в торговых системах с поддержкой множественных символов и таймфреймов
    """

    def __init__(self,
                 lookback_periods: List[int] = [5, 10, 20, 50],
                 prediction_horizon: int = 3,
                 volatility_window: int = 20,
                 use_robust_scaling: bool = True,
                 adaptive_thresholds: bool = True):
        """
        Инициализация генератора признаков

        Args:
            lookback_periods: Периоды для расчета скользящих средних и других индикаторов
            prediction_horizon: Горизонт предсказания (сколько периодов вперед)
            volatility_window: Окно для расчета волатильности
            use_robust_scaling: Использовать RobustScaler вместо StandardScaler
            adaptive_thresholds: Использовать адаптивные пороги для создания меток
        """
        self.lookback_periods = lookback_periods
        self.prediction_horizon = prediction_horizon
        self.volatility_window = volatility_window
        self.adaptive_thresholds = adaptive_thresholds
        self.scaler = RobustScaler() if use_robust_scaling else StandardScaler()
        self.is_fitted = False

    @staticmethod
    def calculate_vpt_manual(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Ручной расчет Volume Price Trend (VPT)
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

    @staticmethod
    def calculate_ad_manual(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Ручной расчет Accumulation/Distribution Line
        """
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_multiplier = money_flow_multiplier.fillna(0)
        money_flow_volume = money_flow_multiplier * volume
        ad_line = money_flow_volume.cumsum()
        return ad_line

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет технических индикаторов с обработкой ошибок
        """
        data = df.copy()

        # Убеждаемся, что у нас есть необходимые колонки
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                if col == 'volume':
                    data[col] = 1.0  # Заглушка для объема
                else:
                    data[col] = data['close']  # Заглушка для OHLC

        try:
            # === ТРЕНДОВЫЕ ИНДИКАТОРЫ ===

            # Скользящие средние различных типов
            for period in self.lookback_periods:
                try:
                    data[f'sma_{period}'] = ta.sma(data['close'], length=period)
                    data[f'ema_{period}'] = ta.ema(data['close'], length=period)
                    data[f'wma_{period}'] = ta.wma(data['close'], length=period)

                    # Отношения цены к скользящим средним
                    data[f'price_to_sma_{period}'] = data['close'] / data[f'sma_{period}']
                    data[f'price_to_ema_{period}'] = data['close'] / data[f'ema_{period}']
                except Exception as e:
                    logger.warning(f"Ошибка при расчете скользящих средних для периода {period}: {e}")

            # MACD семейство
            try:
                macd_data = ta.macd(data['close'])
                if macd_data is not None and not macd_data.empty:
                    data = pd.concat([data, macd_data], axis=1)
            except Exception as e:
                logger.warning(f"Ошибка при расчете MACD: {e}")

            # Bollinger Bands
            try:
                bb_data = ta.bbands(data['close'], length=20, std=2)
                if bb_data is not None and not bb_data.empty:
                    data = pd.concat([data, bb_data], axis=1)
                    # Дополнительные признаки на основе Bollinger Bands
                    if 'BBL_20_2.0' in data.columns and 'BBU_20_2.0' in data.columns:
                        data['bb_position'] = (data['close'] - data['BBL_20_2.0']) / (data['BBU_20_2.0'] - data['BBL_20_2.0'])
                        data['bb_squeeze'] = (data['BBU_20_2.0'] - data['BBL_20_2.0']) / data['close']
            except Exception as e:
                logger.warning(f"Ошибка при расчете Bollinger Bands: {e}")

            # === ОСЦИЛЛЯТОРЫ ===

            # RSI с различными периодами
            for period in [7, 14, 21]:
                try:
                    data[f'rsi_{period}'] = ta.rsi(data['close'], length=period)
                except Exception as e:
                    logger.warning(f"Ошибка при расчете RSI для периода {period}: {e}")

            # Stochastic
            try:
                stoch_data = ta.stoch(data['high'], data['low'], data['close'])
                if stoch_data is not None and not stoch_data.empty:
                    data = pd.concat([data, stoch_data], axis=1)
            except Exception as e:
                logger.warning(f"Ошибка при расчете Stochastic: {e}")

            # Williams %R
            try:
                data['willr'] = ta.willr(data['high'], data['low'], data['close'])
            except Exception as e:
                logger.warning(f"Ошибка при расчете Williams %R: {e}")

            # CCI (Commodity Channel Index)
            try:
                data['cci'] = ta.cci(data['high'], data['low'], data['close'])
            except Exception as e:
                logger.warning(f"Ошибка при расчете CCI: {e}")

            # === ВОЛАТИЛЬНОСТЬ ===

            # Average True Range
            try:
                data['atr'] = ta.atr(data['high'], data['low'], data['close'])
                data['atr_ratio'] = data['atr'] / data['close']
            except Exception as e:
                logger.warning(f"Ошибка при расчете ATR: {e}")

            # Историческая волатильность
            for period in [10, 20, 30]:
                try:
                    returns = data['close'].pct_change()
                    data[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
                except Exception as e:
                    logger.warning(f"Ошибка при расчете волатильности для периода {period}: {e}")

            # === ОБЪЕМНЫЕ ИНДИКАТОРЫ ===

            # Volume-based indicators
            if 'volume' in data.columns and data['volume'].sum() > 0:
                try:
                    # On-Balance Volume
                    data['obv'] = ta.obv(data['close'], data['volume'])
                except Exception as e:
                    logger.warning(f"Ошибка при расчете OBV: {e}")

                try:
                    # Volume-Price Trend - используем ручной расчет
                    data['vpt'] = self.calculate_vpt_manual(data['close'], data['volume'])
                except Exception as e:
                    logger.warning(f"Ошибка при расчете VPT: {e}")

                try:
                    # Accumulation/Distribution Line - пробуем стандартную функцию
                    data['ad'] = ta.ad(data['high'], data['low'], data['close'], data['volume'])
                except Exception:
                    try:
                        # Если не работает, используем ручной расчет
                        data['ad'] = self.calculate_ad_manual(data['high'], data['low'], data['close'], data['volume'])
                    except Exception as e:
                        logger.warning(f"Ошибка при расчете A/D Line: {e}")

                # Volume Moving Averages
                for period in [10, 20]:
                    try:
                        data[f'volume_sma_{period}'] = ta.sma(data['volume'], length=period)
                        data[f'volume_ratio_{period}'] = data['volume'] / data[f'volume_sma_{period}']
                    except Exception as e:
                        logger.warning(f"Ошибка при расчете объемных индикаторов для периода {period}: {e}")

            # === ЦЕНОВЫЕ ПАТТЕРНЫ ===

            # Gaps
            data['gap_up'] = (data['open'] > data['close'].shift(1)).astype(int)
            data['gap_down'] = (data['open'] < data['close'].shift(1)).astype(int)
            data['gap_size'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)

            # Price ranges
            data['high_low_ratio'] = data['high'] / data['low']
            data['body_size'] = abs(data['close'] - data['open']) / data['open']
            data['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['close']
            data['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['close']

            # === ВРЕМЕННЫЕ ПРИЗНАКИ ===

            if 'timestamp' in data.columns:
                try:
                    data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
                    data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
                    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
                except Exception as e:
                    logger.warning(f"Ошибка при создании временных признаков: {e}")
            elif data.index.name == 'timestamp' or hasattr(data.index, 'hour'):
                try:
                    data['hour'] = data.index.hour
                    data['day_of_week'] = data.index.dayofweek
                    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
                except Exception as e:
                    logger.warning(f"Ошибка при создании временных признаков из индекса: {e}")

            # === ПРОИЗВОДНЫЕ ПРИЗНАКИ ===

            # Скорость изменения цены
            for period in [1, 3, 5, 10]:
                try:
                    data[f'roc_{period}'] = ta.roc(data['close'], length=period)
                except Exception as e:
                    logger.warning(f"Ошибка при расчете ROC для периода {period}: {e}")

            # Momentum
            for period in [5, 10, 20]:
                try:
                    data[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
                except Exception as e:
                    logger.warning(f"Ошибка при расчете Momentum для периода {period}: {e}")

            # Взаимодействие индикаторов
            try:
                if 'rsi_14' in data.columns and 'cci' in data.columns:
                    data['rsi_cci_combined'] = (data['rsi_14'] / 100 + data['cci'] / 200) / 2
            except Exception as e:
                logger.warning(f"Ошибка при создании комбинированных индикаторов: {e}")

            # === ФРАКТАЛЬНЫЕ И НЕЛИНЕЙНЫЕ ПРИЗНАКИ ===

            # Hurst Exponent (упрощенная версия)
            def calculate_hurst_simple(prices, window=20):
                """Упрощенный расчет показателя Херста"""
                if len(prices) < window:
                    return np.nan

                lags = range(2, min(window // 2, 10))
                try:
                    tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
                    if len(tau) < 2 or any(t <= 0 for t in tau):
                        return np.nan
                    poly = np.polyfit(np.log(lags), np.log(tau), 1)
                    return poly[0] * 2.0
                except (ValueError, np.linalg.LinAlgError):
                    return np.nan

            # Применяем Hurst к скользящему окну
            hurst_values = []
            for i in range(len(data)):
                start_idx = max(0, i - 50)
                if i - start_idx >= 20:
                    prices = data['close'].iloc[start_idx:i + 1].values
                    hurst = calculate_hurst_simple(prices)
                else:
                    hurst = np.nan
                hurst_values.append(hurst)

            data['hurst_exponent'] = hurst_values

            feature_count = len([col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']])
            logger.info(f"Создано {feature_count} технических индикаторов")

        except Exception as e:
            logger.error(f"Общая ошибка при расчете технических индикаторов: {e}")

        return data

    def create_lagged_features(self, data: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Создание лаговых признаков
        """
        result = data.copy()

        for col in feature_cols:
            if col in data.columns:
                for lag in range(1, 6):  # Лаги от 1 до 5 периодов
                    result[f'{col}_lag_{lag}'] = data[col].shift(lag)

        return result

    def create_rolling_features(self, data: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Создание скользящих статистических признаков
        """
        result = data.copy()

        windows = [5, 10, 20]

        for col in feature_cols:
            if col in data.columns:
                for window in windows:
                    try:
                        # Скользящие статистики
                        result[f'{col}_rolling_mean_{window}'] = data[col].rolling(window=window).mean()
                        result[f'{col}_rolling_std_{window}'] = data[col].rolling(window=window).std()
                        result[f'{col}_rolling_min_{window}'] = data[col].rolling(window=window).min()
                        result[f'{col}_rolling_max_{window}'] = data[col].rolling(window=window).max()

                        # Позиция текущего значения относительно скользящего окна
                        rolling_min = data[col].rolling(window=window).min()
                        rolling_max = data[col].rolling(window=window).max()
                        result[f'{col}_position_in_range_{window}'] = (
                            (data[col] - rolling_min) / (rolling_max - rolling_min + 1e-10)
                        )
                    except Exception as e:
                        logger.warning(f"Ошибка при создании скользящих признаков для {col}, окно {window}: {e}")

        return result

    def create_cross_sectional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Создание кросс-секционных признаков (взаимодействие между индикаторами)
        """
        result = data.copy()

        # Важные пары индикаторов для взаимодействия
        indicator_pairs = [
            ('rsi_14', 'willr'),
            ('sma_10', 'sma_20'),
            ('ema_10', 'ema_20'),
            ('atr', 'volatility_20'),
        ]

        for ind1, ind2 in indicator_pairs:
            if ind1 in data.columns and ind2 in data.columns:
                try:
                    # Отношение
                    result[f'{ind1}_to_{ind2}_ratio'] = data[ind1] / (data[ind2] + 1e-10)

                    # Разность
                    result[f'{ind1}_minus_{ind2}'] = data[ind1] - data[ind2]

                    # Корреляция на скользящем окне
                    result[f'{ind1}_{ind2}_correlation'] = (
                        data[ind1].rolling(window=20).corr(data[ind2])
                    )
                except Exception as e:
                    logger.warning(f"Ошибка при создании кросс-секционных признаков для {ind1}/{ind2}: {e}")

        return result

    def create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Создание признаков рыночных режимов
        """
        result = data.copy()

        try:
            # Волатильность режимы
            if 'volatility_20' in data.columns:
                vol_data = data['volatility_20'].dropna()
                if len(vol_data) > 0:
                    vol_quantiles = vol_data.quantile([0.33, 0.66])
                    result['volatility_regime'] = pd.cut(
                        data['volatility_20'],
                        bins=[-np.inf, vol_quantiles[0.33], vol_quantiles[0.66], np.inf],
                        labels=[0, 1, 2]
                    ).astype(float)

            # Трендовые режимы на основе скользящих средних
            if 'sma_10' in data.columns and 'sma_50' in data.columns:
                result['trend_regime'] = np.where(
                    data['sma_10'] > data['sma_50'], 1, 0  # 1 = восходящий тренд, 0 = нисходящий
                )

            # Momentum режимы
            if 'rsi_14' in data.columns:
                result['momentum_regime'] = pd.cut(
                    data['rsi_14'],
                    bins=[0, 30, 70, 100],
                    labels=[0, 1, 2]  # 0 = oversold, 1 = neutral, 2 = overbought
                ).astype(float)

        except Exception as e:
            logger.warning(f"Ошибка при создании режимных признаков: {e}")

        return result

    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """
        Создание меток для классификации с улучшенной балансировкой

        Returns:
            Метки: 0 = продавать, 1 = держать, 2 = покупать
        """
        if 'close' not in data.columns:
            raise ValueError("Колонка 'close' не найдена в данных")

        # Рассчитываем будущую доходность
        future_returns = data['close'].shift(-self.prediction_horizon) / data['close'] - 1

        if self.adaptive_thresholds:
            # Рассчитываем динамические пороги на основе волатильности
            if 'atr' in data.columns and data['atr'].notna().sum() > 0:
                volatility_threshold = data['atr'] / data['close']
            else:
                # Используем историческую волатильность
                returns = data['close'].pct_change()
                volatility_threshold = returns.rolling(window=self.volatility_window).std()

            # Создаем адаптивные пороги - делаем их менее строгими для лучшего баланса классов
            upper_threshold = volatility_threshold * 0.75  # Уменьшили с 1.5 до 0.75
            lower_threshold = -volatility_threshold * 0.75  # Уменьшили с 1.5 до 0.75
        else:
            # Фиксированные пороги на основе квантилей
            returns_clean = future_returns.dropna()
            if len(returns_clean) > 0:
                upper_threshold = returns_clean.quantile(0.7)  # Увеличили с 0.75 до 0.7
                lower_threshold = returns_clean.quantile(0.3)  # Уменьшили с 0.25 до 0.3
            else:
                upper_threshold = 0.01
                lower_threshold = -0.01

        # Создаем метки
        labels = pd.Series(1, index=data.index)  # По умолчанию HOLD
        labels[future_returns > upper_threshold] = 2  # BUY
        labels[future_returns < lower_threshold] = 0  # SELL

        # Проверяем распределение классов и корректируем при необходимости
        class_counts = labels.value_counts()
        total_samples = len(labels)

        # Если какой-то класс составляет менее 5% от общего количества, корректируем пороги
        min_class_ratio = 0.05
        for class_label, count in class_counts.items():
            if count / total_samples < min_class_ratio:
                logger.warning(f"Класс {class_label} составляет менее {min_class_ratio*100}% данных ({count}/{total_samples})")

                # Делаем пороги еще менее строгими
                if self.adaptive_thresholds:
                    upper_threshold = volatility_threshold * 0.5
                    lower_threshold = -volatility_threshold * 0.5
                else:
                    upper_threshold = returns_clean.quantile(0.65)
                    lower_threshold = returns_clean.quantile(0.35)

                # Пересоздаем метки
                labels = pd.Series(1, index=data.index)
                labels[future_returns > upper_threshold] = 2
                labels[future_returns < lower_threshold] = 0
                break

        return labels

    def clean_and_validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Очистка и валидация данных
        """
        # Удаляем строки с бесконечными значениями
        data = data.replace([np.inf, -np.inf], np.nan)

        # Заполняем пропущенные значения
        # Для индикаторов используем forward fill, затем backward fill
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].ffill().bfill()

        # Удаляем колонки с более чем 50% пропущенных значений
        threshold = len(data) * 0.5
        data = data.dropna(axis=1, thresh=threshold)

        # Удаляем строки с пропущенными значениями
        data = data.dropna()

        return data

    def select_numeric_features(self, data: pd.DataFrame) -> List[str]:
        """
        Выбор только числовых признаков для нормализации
        """
        # Исключаем исходные колонки OHLCV и timestamp
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']

        # Получаем все колонки
        all_cols = set(data.columns)

        # Убираем исключенные колонки
        candidate_cols = all_cols - set(exclude_cols)

        # Проверяем, какие колонки действительно числовые
        numeric_cols = []
        for col in candidate_cols:
            try:
                # Пытаемся преобразовать колонку в числовой тип
                pd.to_numeric(data[col], errors='raise')
                numeric_cols.append(col)
            except (ValueError, TypeError):
                logger.warning(f"Колонка '{col}' содержит нечисловые данные и будет исключена из признаков")
                continue

        logger.info(f"Выбрано {len(numeric_cols)} числовых признаков из {len(all_cols)} общих колонок")
        return numeric_cols

    def create_features_and_labels(self, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Основная функция для создания признаков и меток

        Args:
            data: DataFrame с историческими данными (OHLCV)

        Returns:
            Tuple[features_df, labels_series] или (None, None) в случае ошибки
        """
        try:
            if data is None or data.empty:
                logger.warning("Переданы пустые данные")
                return None, None

            logger.info(f"Начинаем обработку данных. Размер входных данных: {data.shape}")

            # Проверяем наличие необходимых колонок
            if 'close' not in data.columns:
                logger.error("Колонка 'close' не найдена в данных")
                return None, None

            # 1. Расчет технических индикаторов
            logger.info("Расчет технических индикаторов...")
            data_with_indicators = self.calculate_technical_indicators(data)

            # 2. Выбираем основные признаки для дальнейшей обработки
            feature_columns = [col for col in data_with_indicators.columns
                             if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]

            # 3. Создание лаговых признаков
            logger.info("Создание лаговых признаков...")
            important_features = ['rsi_14', 'sma_20', 'ema_10', 'atr']
            # Проверяем наличие MACD
            if any('macd' in col.lower() for col in data_with_indicators.columns):
                macd_cols = [col for col in data_with_indicators.columns if 'macd' in col.lower()]
                important_features.extend(macd_cols[:1])  # Добавляем первую найденную MACD колонку

            important_features = [f for f in important_features if f in data_with_indicators.columns]

            data_with_lags = self.create_lagged_features(data_with_indicators, important_features)

            # 4. Создание скользящих статистических признаков
            logger.info("Создание скользящих статистических признаков...")
            data_with_rolling = self.create_rolling_features(data_with_lags, important_features)

            # 5. Создание кросс-секционных признаков
            logger.info("Создание кросс-секционных признаков...")
            data_with_cross = self.create_cross_sectional_features(data_with_rolling)

            # 6. Создание признаков рыночных режимов
            logger.info("Создание признаков рыночных режимов...")
            data_with_regimes = self.create_regime_features(data_with_cross)

            # 7. Очистка и валидация данных
            logger.info("Очистка и валидация данных...")
            clean_data = self.clean_and_validate_data(data_with_regimes)

            if clean_data.empty:
                logger.warning("После очистки данные стали пустыми")
                return None, None

            # 8. Создание меток
            logger.info("Создание меток...")
            labels = self.create_labels(clean_data)

            # 9. Выбор только числовых признаков
            logger.info("Выбор числовых признаков для нормализации...")
            numeric_feature_cols = self.select_numeric_features(clean_data)

            if not numeric_feature_cols:
                logger.error("Не найдено числовых признаков для обучения модели")
                return None, None

            features_df = clean_data[numeric_feature_cols].copy()

            # 10. Дополнительная проверка на числовые типы
            logger.info("Дополнительная проверка и преобразование типов данных...")
            for col in features_df.columns:
                try:
                    features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Не удалось преобразовать колонку '{col}' в числовой тип: {e}")
                    features_df = features_df.drop(columns=[col])

            # Удаляем колонки, которые стали полностью NaN после преобразования
            features_df = features_df.dropna(axis=1, how='all')

            if features_df.empty:
                logger.error("После преобразования типов не осталось признаков")
                return None, None

            # 11. Нормализация признаков
            logger.info("Нормализация признаков...")
            try:
                if not self.is_fitted:
                    features_scaled = pd.DataFrame(
                        self.scaler.fit_transform(features_df),
                        columns=features_df.columns,
                        index=features_df.index
                    )
                    self.is_fitted = True
                else:
                    features_scaled = pd.DataFrame(
                        self.scaler.transform(features_df),
                        columns=features_df.columns,
                        index=features_df.index
                    )
            except Exception as e:
                logger.error(f"Ошибка при нормализации признаков: {e}")
                return None, None

            # Убеждаемся, что признаки и метки имеют одинаковый индекс
            common_index = features_scaled.index.intersection(labels.index)
            features_final = features_scaled.loc[common_index]
            labels_final = labels.loc[common_index]

            # Проверяем балансировку классов
            class_distribution = labels_final.value_counts()
            logger.info(f"Создано {features_final.shape[1]} признаков для {features_final.shape[0]} наблюдений")
            logger.info(f"Распределение меток: {class_distribution.to_dict()}")

            # Дополнительная проверка на сильный дисбаланс классов
            total_samples = len(labels_final)
            min_class_size = class_distribution.min()
            min_class_ratio = min_class_size / total_samples

            if min_class_ratio < 0.05:  # Если минимальный класс менее 5%
                logger.warning(f"Обнаружен сильный дисбаланс классов. Минимальный класс: {min_class_ratio:.2%}")

                # Пытаемся улучшить баланс, создавая более мягкие пороги
                returns = clean_data['close'].pct_change()
                returns_clean = returns.dropna()

                if len(returns_clean) > 10:
                    # Используем более мягкие квантили
                    upper_threshold = returns_clean.quantile(0.6)
                    lower_threshold = returns_clean.quantile(0.4)

                    future_returns = clean_data['close'].shift(-self.prediction_horizon) / clean_data['close'] - 1
                    labels_balanced = pd.Series(1, index=clean_data.index)
                    labels_balanced[future_returns > upper_threshold] = 2
                    labels_balanced[future_returns < lower_threshold] = 0

                    # Обновляем метки если баланс улучшился
                    new_distribution = labels_balanced.value_counts()
                    new_min_ratio = new_distribution.min() / len(labels_balanced)

                    if new_min_ratio > min_class_ratio:
                        labels_final = labels_balanced.loc[common_index]
                        logger.info(f"Улучшенное распределение меток: {labels_final.value_counts().to_dict()}")

            return features_final, labels_final

        except Exception as e:
            logger.error(f"Ошибка при создании признаков и меток: {e}", exc_info=True)
            return None, None


# Глобальный экземпляр для использования в main.py
feature_engineer = AdvancedFeatureEngineer()


def create_features_and_labels(data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Функция-обертка для использования в main.py

    Args:
        data: DataFrame с историческими данными или словарь DataFrames для разных символов

    Returns:
        Tuple[features_df, labels_series] или (None, None) в случае ошибки
    """
    try:
        # Если передан словарь с данными для разных символов
        if isinstance(data, dict):
            all_features = []
            all_labels = []

            for symbol, symbol_data in data.items():
                logger.info(f"Обработка данных для символа: {symbol}")
                features, labels = feature_engineer.create_features_and_labels(symbol_data)

                if features is not None and labels is not None:
                    # Добавляем информацию о символе как числовой признак (закодированный)
                    symbol_hash = hash(symbol) % 1000  # Простое числовое кодирование символа
                    features[f'symbol_encoded'] = symbol_hash
                    all_features.append(features)
                    all_labels.append(labels)

            if all_features:
                # Объединяем данные всех символов
                combined_features = pd.concat(all_features, axis=0, sort=False)
                combined_labels = pd.concat(all_labels, axis=0)

                # Заполняем пропущенные значения символьных признаков
                if 'symbol_encoded' in combined_features.columns:
                    combined_features['symbol_encoded'] = combined_features['symbol_encoded'].fillna(0)

                logger.info(
                    f"Объединено данных: {combined_features.shape[0]} наблюдений, {combined_features.shape[1]} признаков")
                return combined_features, combined_labels
            else:
                logger.warning("Не удалось обработать данные ни для одного символа")
                return None, None

        # Если передан обычный DataFrame
        elif isinstance(data, pd.DataFrame):
            return feature_engineer.create_features_and_labels(data)

        else:
            logger.error(f"Неподдерживаемый тип данных: {type(data)}")
            return None, None

    except Exception as e:
        logger.error(f"Ошибка в create_features_and_labels: {e}", exc_info=True)
        return None, None


# Дополнительные утилиты для анализа признаков
def analyze_feature_importance(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """
    Анализ важности признаков
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import mutual_info_classif

        # Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(features, labels)
        rf_importance = pd.Series(rf.feature_importances_, index=features.columns, name='rf_importance')

        # Mutual Information
        mi_scores = mutual_info_classif(features, labels, random_state=42)
        mi_importance = pd.Series(mi_scores, index=features.columns, name='mutual_info')

        # Объединяем результаты
        importance_df = pd.concat([rf_importance, mi_importance], axis=1)
        importance_df['combined_score'] = (importance_df['rf_importance'] + importance_df['mutual_info']) / 2

        return importance_df.sort_values('combined_score', ascending=False)

    except ImportError:
        logger.warning("sklearn не доступен для анализа важности признаков")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Ошибка при анализе важности признаков: {e}")
        return pd.DataFrame()


def get_feature_statistics(features: pd.DataFrame) -> Dict:
    """
    Получение статистики по признакам
    """
    stats = {
        'total_features': len(features.columns),
        'total_observations': len(features),
        'missing_values': features.isnull().sum().sum(),
        'feature_types': {
            'technical_indicators': len([c for c in features.columns if
                                       any(indicator in c.lower() for indicator in ['rsi', 'macd', 'sma', 'ema', 'bb'])]),
            'lagged_features': len([c for c in features.columns if 'lag_' in c]),
            'rolling_features': len([c for c in features.columns if 'rolling_' in c]),
            'cross_sectional': len([c for c in features.columns if '_to_' in c or '_minus_' in c]),
            'regime_features': len([c for c in features.columns if 'regime' in c]),
        },
        'memory_usage_mb': features.memory_usage(deep=True).sum() / 1024 / 1024
    }

    return stats


def validate_data_quality(features: pd.DataFrame, labels: pd.Series) -> Dict[str, any]:
    """
    Валидация качества данных для машинного обучения

    Returns:
        Словарь с результатами валидации и рекомендациями
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }

    try:
        # Проверка размера данных
        if len(features) < 100:
            validation_results['errors'].append(f"Недостаточно данных для обучения: {len(features)} < 100")
            validation_results['is_valid'] = False
        elif len(features) < 500:
            validation_results['warnings'].append(f"Малый размер выборки: {len(features)}. Рекомендуется > 500")

        # Проверка баланса классов
        class_counts = labels.value_counts()
        total_samples = len(labels)
        min_class_ratio = class_counts.min() / total_samples

        if min_class_ratio < 0.01:  # Менее 1%
            validation_results['errors'].append(f"Критический дисбаланс классов: {class_counts.to_dict()}")
            validation_results['is_valid'] = False
        elif min_class_ratio < 0.05:  # Менее 5%
            validation_results['warnings'].append(f"Сильный дисбаланс классов: {class_counts.to_dict()}")
            validation_results['recommendations'].append("Рассмотрите использование SMOTE или изменение порогов классификации")

        # Проверка на пропущенные значения
        missing_ratio = features.isnull().sum().sum() / (len(features) * len(features.columns))
        if missing_ratio > 0.1:
            validation_results['warnings'].append(f"Высокий процент пропущенных значений: {missing_ratio:.2%}")

        # Проверка на константные признаки
        constant_features = []
        for col in features.columns:
            if features[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            validation_results['warnings'].append(f"Найдены константные признаки: {len(constant_features)}")
            validation_results['recommendations'].append("Удалите константные признаки перед обучением")

        # Проверка на сильно коррелированные признаки
        if len(features.columns) > 1:
            corr_matrix = features.corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

            if high_corr_pairs:
                validation_results['warnings'].append(f"Найдены сильно коррелированные признаки: {len(high_corr_pairs)} пар")
                validation_results['recommendations'].append("Рассмотрите удаление дублирующих признаков")

        # Проверка масштаба признаков
        feature_scales = features.std()
        if feature_scales.max() / feature_scales.min() > 1000:
            validation_results['warnings'].append("Большие различия в масштабах признаков")
            validation_results['recommendations'].append("Убедитесь, что признаки нормализованы")

    except Exception as e:
        validation_results['errors'].append(f"Ошибка при валидации: {e}")
        validation_results['is_valid'] = False

    return validation_results