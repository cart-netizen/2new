import asyncio
from datetime import datetime

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

from core.enums import Timeframe

logger = logging.getLogger(__name__)
def crossover_series(x: pd.Series, y: pd.Series) -> pd.Series:
    """Проверяет, пересекла ли серия x серию y снизу вверх."""
    return (x > y) & (x.shift(1) < y.shift(1))

def crossunder_series(x: pd.Series, y: pd.Series) -> pd.Series:
    """Проверяет, пересекла ли серия x серию y сверху вниз."""
    return (x < y) & (x.shift(1) > y.shift(1))

class AdvancedFeatureEngineer:
    """
    Продвинутый класс для создания признаков и меток для машинного обучения
    в торговых системах с поддержкой множественных символов и таймфреймов
    """

    def __init__(self,
                 lookback_periods: List[int] = [5, 10, 20, 50],
                 prediction_horizon: int = 5,
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
        # self.prediction_horizon = prediction_horizon
        self.volatility_window = volatility_window
        self.adaptive_thresholds = adaptive_thresholds
        # self.scaler = RobustScaler() if use_robust_scaling else StandardScaler()
        self.is_fitted = False
        self.feature_names_in_ = []
        self.prediction_horizon = prediction_horizon
        self.scaler = RobustScaler(quantile_range=(25.0, 75.0)) if use_robust_scaling else StandardScaler()
        self.is_fitted = False

    def reset_scaler(self):
        """Сброс scaler для переобучения с нуля"""
        self.scaler = StandardScaler()
        self.is_fitted = False
        if hasattr(self, 'feature_names_in_'):
            delattr(self, 'feature_names_in_')

        logger.info("Scaler сброшен для нового обучения")

    @staticmethod
    def calculate_mfi_manual(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                             length: int = 14) -> pd.Series:
        """
        Ручной, надежный расчет Money Flow Index (MFI).
        """
        # 1. Рассчитываем типичную цену
        typical_price = (high + low + close) / 3

        # 2. Рассчитываем денежный поток (Raw Money Flow)
        money_flow = typical_price * volume

        # 3. Определяем положительные и отрицательные денежные потоки
        price_diff = typical_price.diff(1)

        positive_flow = money_flow.where(price_diff > 0, 0)
        negative_flow = money_flow.where(price_diff < 0, 0)

        # 4. Суммируем потоки за заданный период
        positive_mf_sum = positive_flow.rolling(window=length, min_periods=1).sum()
        negative_mf_sum = negative_flow.rolling(window=length, min_periods=1).sum()

        # 5. Рассчитываем Money Flow Ratio (MFR) с защитой от деления на ноль
        money_flow_ratio = positive_mf_sum / (negative_mf_sum + 1e-9)  # +1e-9 для избежания деления на ноль

        # 6. Рассчитываем MFI по стандартной формуле
        mfi = 100 - (100 / (1 + money_flow_ratio))

        return mfi

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
        logger.info("Расчет оптимизированного набора технических индикаторов...")

        # --- ШАГ 1: АГРЕССИВНАЯ ОЧИСТКА ДАННЫХ ---
        required_cols = ['open', 'high', 'low', 'close', 'volume']

        # Проверяем наличие колонок
        if not all(col in data.columns for col in required_cols):
            logger.error(f"Отсутствуют необходимые OHLCV колонки. Пропускаем расчет.")
            return pd.DataFrame()  # Возвращаем пустой DataFrame

        # Принудительно преобразуем все ключевые колонки в числовой формат.
        # Все, что не может быть преобразовано, станет NaN (Not a Number).
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # УЛУЧШЕННАЯ ОЧИСТКА: более мягкий подход
        initial_rows = len(data)

        # Сначала заполняем NaN методом forward/backward fill
        data[required_cols] = data[required_cols].fillna(method='ffill').fillna(method='bfill')

        # Удаляем только строки, где ВСЕ основные колонки NaN
        data = data.dropna(subset=required_cols, how='all')

        # Дополнительно удаляем строки где цена закрытия NaN (критично)
        data = data.dropna(subset=['close'])

        if len(data) < initial_rows:
            logger.warning(f"Удалено {initial_rows - len(data)} строк с невалидными данными.")

        # КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: снижаем минимальный порог
        if data.empty or len(data) < 20:  # Было 50, делаем 20
            logger.error(
                f"После очистки не осталось достаточно данных для расчета индикаторов. Осталось: {len(data)} строк")
            return pd.DataFrame()

        logger.info(f"После очистки данных осталось {len(data)} строк для расчета индикаторов")

        # # Убеждаемся, что у нас есть необходимые колонки
        # required_cols = ['open', 'high', 'low', 'close', 'volume']
        # for col in required_cols:
        #     if col not in data.columns:
        #         if col == 'volume':
        #             data[col] = 1.0  # Заглушка для объема
        #         else:
        #             data[col] = data['close']  # Заглушка для OHLC

        try:
            # === ТРЕНДОВЫЕ ИНДИКАТОРЫ ===
            try:
                # === ГРУППА 1: Базовые индикаторы и волатильность (из прошлых версий) ===
                # data.ta.rsi(length=14, append=True)
                # data.ta.ema(length=50, append=True)
                # data.ta.ema(length=200, append=True)
                # data.ta.adx(length=14, append=True)
                # data.ta.atr(length=14, append=True)
                # RSI
                rsi_14 = ta.rsi(data['close'], length=14)
                if rsi_14 is not None:
                    data['RSI_14'] = rsi_14

                # EMA - ключевые периоды
                ema_50 = ta.ema(data['close'], length=50)
                if ema_50 is not None:
                    data['EMA_50'] = ema_50

                ema_200 = ta.ema(data['close'], length=200)
                if ema_200 is not None:
                    data['EMA_200'] = ema_200

                # ADX
                adx_14 = ta.adx(data['high'], data['low'], data['close'], length=14)
                if adx_14 is not None and not adx_14.empty:
                    data['ADX_14'] = adx_14.iloc[:, 0]  # Основное значение ADX

                # ATR
                atr_14 = ta.atr(data['high'], data['low'], data['close'], length=14)
                if atr_14 is not None:
                    data['ATR_14'] = atr_14
                    data['atr_ratio'] = atr_14 / (data['close'] + 1e-9)

                logger.debug("Базовые индикаторы рассчитаны")
                # data['atr_ratio'] = data['atr'] / data['close']
                # Рассчитываем ATR и ATR_Ratio в одном блоке
                # atr = ta.atr(data['high'], data['low'], data['close'], length=14)
                # if atr is not None and not atr.isnull().all():
                #     data['atr'] = atr
                #     data['atr_ratio'] = atr / data['close']
                if 'atr' in data.columns:
                    data['atr_ratio'] = data['atr'] / (data['close'] + 1e-9)


                # === ГРУППА 2: Компоненты из "MFI + RSI + EMA Dynamic Signals" ===
                logger.debug("Расчет признаков из 'MFI + RSI + EMA'...")
                data['mfi_14'] = self.calculate_mfi_manual(data['high'], data['low'], data['close'], data['volume'], length=14)
                ema_fast = ta.ema(data['close'], length=9)
                ema_slow = ta.ema(data['close'], length=21)
                data['ema_fast_9'] = ema_fast
                data['ema_slow_21'] = ema_slow
                # Признак "близости к пересечению"
                data['ema_proximity_percent'] = abs((ema_fast - ema_slow) / ema_slow) * 100
                # Признак факта пересечения (1 = было, 0 = не было)
                # --- ИСПОЛЬЗУЕМ ВАШИ НОВЫЕ ФУНКЦИИ ---
                data['ema_9_21_crossover'] = crossover_series(ema_fast, ema_slow).astype(int)
                data['ema_9_21_crossunder'] = crossunder_series(ema_fast, ema_slow).astype(int)
                # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

                # === ГРУППА 3: Компоненты из "RSITrend" ===
                logger.debug("Расчет признаков из 'RSITrend'...")
                # Рассчитываем Hull Moving Average (HMA)
                hma_fast = ta.hma(data['close'], length=30)
                # Для сигнала нам нужно сравнить текущее значение с предыдущим сдвинутым
                hma_slow = hma_fast.shift(2)
                data['hma_fast_30'] = hma_fast
                data['hma_slow_30_shifted'] = hma_slow
                # Сигнал тренда по HMA (1 = вверх, 0 = вниз)
                data['hma_trend_signal'] = (hma_fast > hma_slow).astype(int)

                # Рассчитываем RSI от HMA (сглаженный RSI)
                if hma_fast is not None and not hma_fast.isnull().all():
                    data['rsi_of_hma'] = ta.rsi(hma_fast, length=14)

                # === ГРУППА 4: Анализ всплесков объема (уже был, но проверяем) ===
                logger.debug("Расчет признаков всплеска объема...")
                volume_sma = ta.sma(data['volume'], length=20)
                data['volume_spike_ratio'] = data['volume'] / (volume_sma + 1e-9)

                feature_count = len([col for col in data.columns if col not in df.columns])
                logger.info(f"Создано {feature_count} НОВЫХ технических индикаторов.")

            except Exception as e:
                logger.error(f"Общая ошибка при расчете индикаторов: {e}", exc_info=True)

            try:
                # Parabolic SAR (для динамического стоп-лосса)
                psar_data = ta.psar(data['high'], data['low'], data['close'])
                if psar_data is not None:
                    # Нас интересует основная линия PSAR
                    psar_col = next((col for col in psar_data.columns if
                                     'PSAR' in col and 'PSARl' not in col and 'PSARs' not in col), None)
                    if psar_col:
                        data['psar'] = psar_data[psar_col]

            except Exception as e:
                logger.warning(f"Ошибка при расчете осцилляторов и волатильности: {e}")

            # Скользящие средние различных типов
            for period in self.lookback_periods:
                try:
                    # Создаем консистентные имена
                    sma_val = ta.sma(data['close'], length=period)
                    if sma_val is not None:
                        data[f'SMA_{period}'] = sma_val
                        data[f'price_to_sma_{period}'] = data['close'] / (sma_val + 1e-9)

                    ema_val = ta.ema(data['close'], length=period)
                    if ema_val is not None:
                        data[f'EMA_{period}'] = ema_val
                        data[f'price_to_ema_{period}'] = data['close'] / (ema_val + 1e-9)

                    wma_val = ta.wma(data['close'], length=period)
                    if wma_val is not None:
                        data[f'WMA_{period}'] = wma_val

                except Exception as e:
                    logger.warning(f"Ошибка при расчете скользящих средних для периода {period}: {e}")

            # MACD семейство
            try:
                macd_data = ta.macd(data['close'], fast=12, slow=26, signal=9)
                if macd_data is not None and not macd_data.empty:
                    # Явно именуем колонки для консистентности
                    macd_data.columns = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
                    data = pd.concat([data, macd_data], axis=1)

                    # Дополнительные признаки на основе MACD
                    macd_values = data['MACD_12_26_9']
                    if len(macd_values) > 20:  # Изменяем с 0 на 20
                        # Статистические характеристики MACD
                        data['MACD_12_26_9_rolling_mean_20'] = macd_values.rolling(20, min_periods=5).mean()
                        data['MACD_12_26_9_rolling_max_20'] = macd_values.rolling(20, min_periods=5).max()
                        data['MACD_12_26_9_rolling_min_20'] = macd_values.rolling(20, min_periods=5).min()
                        data['MACD_12_26_9_rolling_std_20'] = data['MACD_12_26_9'].rolling(20, min_periods=5).std()

                        # Позиция в диапазоне
                        rolling_max = data['MACD_12_26_9_rolling_max_20']
                        rolling_min = data['MACD_12_26_9_rolling_min_20']
                        data['MACD_12_26_9_position_in_range_20'] = (
                            (macd_values - rolling_min) / (rolling_max - rolling_min + 1e-9)
                        )
                    else:
                        # Создаем признаки с нейтральными значениями для консистентности
                        data['MACD_12_26_9_rolling_mean_20'] = 0.0
                        data['MACD_12_26_9_rolling_max_20'] = 0.0
                        data['MACD_12_26_9_rolling_min_20'] = 0.0
                        data['MACD_12_26_9_position_in_range_20'] = 0.5
                        data['MACD_12_26_9_rolling_std_20'] = 0.0

                    logger.debug("MACD и дополнительные признаки рассчитаны")

            except Exception as e:
                logger.warning(f"Ошибка при расчете MACD: {e}")


                # === НОВЫЕ ПРИЗНАКИ ИЗ RSITrend ===
            try:
                # Hull Moving Average
                hma_fast = ta.hma(data['close'], length=30)
                if hma_fast is not None and not hma_fast.isnull().all():
                    data['rsi_of_hma'] = ta.rsi(hma_fast, length=14)
                else:
                    data['rsi_of_hma'] = 50.0  # Нейтральное значение
            except Exception as e:
                logger.warning(f"Ошибка при расчете rsi_of_hma: {e}")
                data['rsi_of_hma'] = 50.0

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

            # ++ ДОБАВЛЯЕМ ПРИЗНАКИ ГЛОБАЛЬНОГО ТРЕНДА ++
            try:
                # АДАПТИВНЫЙ РАСЧЕТ EMA_200: используем доступную длину
                available_length = len(data)

                if available_length >= 200:
                    ema_period = 200
                elif available_length >= 100:
                    ema_period = 100
                    logger.info(f"Используем EMA_{ema_period} вместо EMA_200 (недостаточно данных)")
                elif available_length >= 50:
                    ema_period = 50
                    logger.info(f"Используем EMA_{ema_period} вместо EMA_200 (недостаточно данных)")
                else:
                    ema_period = min(20, available_length - 1)
                    logger.info(f"Используем EMA_{ema_period} вместо EMA_200 (критически мало данных)")

                ema_200 = ta.ema(data['close'], length=ema_period)

                if ema_200 is not None and not ema_200.isnull().all():
                    # Отношение цены к долгосрочной EMA показывает позицию в тренде
                    data['price_to_ema_200_ratio'] = data['close'] / ema_200
                    # Также сохраняем саму EMA_200 как признак
                    data['EMA_200'] = ema_200
                    logger.debug(f"EMA_{ema_period} успешно рассчитана")
                else:
                    logger.warning(f"EMA_{ema_period} не удалось рассчитать, заполняем нейтральными значениями")
                    data['price_to_ema_200_ratio'] = 1.0
                    data['EMA_200'] = data['close']

                # АДАПТИВНЫЙ РАСЧЕТ ADX: используем доступную длину
                if available_length >= 50:
                    adx_period = 50
                elif available_length >= 30:
                    adx_period = 30
                else:
                    adx_period = min(14, available_length - 1)

                adx_long = ta.adx(data['high'], data['low'], data['close'], length=adx_period)
                if adx_long is not None and not adx_long.empty and adx_long.shape[1] > 0:
                    data['adx_long_term'] = adx_long.iloc[:, 0]
                    logger.debug(f"ADX_{adx_period} успешно рассчитан")
                else:
                    logger.warning(f"ADX_{adx_period} не удалось рассчитать, заполняем нулями")
                    data['adx_long_term'] = 20.0  # Нейтральное значение для ADX

            except Exception as e:
                logger.error(f"Ошибка при расчете признаков глобального тренда: {e}")
                # Устанавливаем значения по умолчанию в случае ошибки
                data['price_to_ema_200_ratio'] = 1.0
                data['EMA_200'] = data['close']
                data['adx_long_term'] = 20.0


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

            # # Williams %R
            # try:
            #     data['willr'] = ta.willr(data['high'], data['low'], data['close'])
            # except Exception as e:
            #     logger.warning(f"Ошибка при расчете Williams %R: {e}")

            # Aroon Oscillator
            try:
                aroon_data = ta.aroon(data['high'], data['low'], length=14)
                if aroon_data is not None and not aroon_data.empty:
                    data = pd.concat([data, aroon_data], axis=1)
            except Exception as e:
                logger.warning(f"Ошибка при расчете Aroon: {e}")

            # # CCI (Commodity Channel Index)
            # try:
            #     data['cci'] = ta.cci(data['high'], data['low'], data['close'])
            # except Exception as e:
            #     logger.warning(f"Ошибка при расчете CCI: {e}")

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
            # data['gap_up'] = (data['open'] > data['close'].shift(1)).astype(int)
            # data['gap_down'] = (data['open'] < data['close'].shift(1)).astype(int)
            # # data['gap_size'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)

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
        #
        # # Важные пары индикаторов для взаимодействия
        # indicator_pairs = [
        #     ('rsi_14', 'willr'),
        #     ('sma_10', 'sma_20'),
        #     ('ema_10', 'ema_20'),
        #     ('atr', 'volatility_20'),
        # ]
        #
        # for ind1, ind2 in indicator_pairs:
        #     if ind1 in data.columns and ind2 in data.columns:
        #         try:
        #             # Отношение
        #             result[f'{ind1}_to_{ind2}_ratio'] = data[ind1] / (data[ind2] + 1e-10)
        #
        #             # Разность
        #             result[f'{ind1}_minus_{ind2}'] = data[ind1] - data[ind2]
        #
        #             # Корреляция на скользящем окне
        #             result[f'{ind1}_{ind2}_correlation'] = (
        #                 data[ind1].rolling(window=20).corr(data[ind2])
        #             )
        #         except Exception as e:
        #             logger.warning(f"Ошибка при создании кросс-секционных признаков для {ind1}/{ind2}: {e}")

        return result

    def create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Создание признаков рыночных режимов
        """
        result = data.copy()

        # try:
        #     # Волатильность режимы
        #     if 'volatility_20' in data.columns:
        #         vol_data = data['volatility_20'].dropna()
        #         if len(vol_data) > 0:
        #             vol_quantiles = vol_data.quantile([0.33, 0.66])
        #             result['volatility_regime'] = pd.cut(
        #                 data['volatility_20'],
        #                 bins=[-np.inf, vol_quantiles[0.33], vol_quantiles[0.66], np.inf],
        #                 labels=[0, 1, 2]
        #             ).astype(float)
        #
        #     # Трендовые режимы на основе скользящих средних
        #     if 'sma_10' in data.columns and 'sma_50' in data.columns:
        #         result['trend_regime'] = np.where(
        #             data['sma_10'] > data['sma_50'], 1, 0  # 1 = восходящий тренд, 0 = нисходящий
        #         )
        #
        #     # Momentum режимы
        #     if 'rsi_14' in data.columns:
        #         result['momentum_regime'] = pd.cut(
        #             data['rsi_14'],
        #             bins=[0, 30, 70, 100],
        #             labels=[0, 1, 2]  # 0 = oversold, 1 = neutral, 2 = overbought
        #         ).astype(float)
        #
        # except Exception as e:
        #     logger.warning(f"Ошибка при создании режимных признаков: {e}")

        return result

    # def create_labels(self, data: pd.DataFrame) -> pd.Series:
    #     """
    #     Создает метки для классификации с использованием "Метода трех барьеров".
    #     Это золотой стандарт в финансовом ML.
    #
    #     Returns:
    #         Метки: 0 = SELL (касание стоп-лосса), 1 = HOLD (касание временного барьера), 2 = BUY (касание тейк-профита)
    #     """
    #     if 'close' not in data.columns:
    #         raise ValueError("Колонка 'close' не найдена в данных")
    #
    #     logger.info("Создание меток с помощью метода трех барьеров...")
    #
    #     # 1. Настройка барьеров
    #     # Рассчитываем волатильность (через ATR) для установки динамических барьеров
    #     if 'atr' not in data.columns or data['atr'].isnull().all():
    #         data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)
    #
    #     # Заполняем возможные пропуски в ATR
    #     data['atr'] = data['atr'].ffill().bfill()
    #
    #     # Динамические барьеры: тейк-профит будет в 2 раза дальше стоп-лосса
    #     tp_multiplier = 2.0
    #     sl_multiplier = 1.0
    #
    #     take_profit_level = data['atr'] / data['close'] * tp_multiplier
    #     stop_loss_level = data['atr'] / data['close'] * sl_multiplier
    #
    #     # Вертикальный барьер (максимальное время удержания сделки)
    #     time_barrier_periods = self.prediction_horizon * 2  # Например, 10 свечей
    #
    #     # 2. Основной цикл для вычисления меток
    #     labels = pd.Series(1, index=data.index)  # По умолчанию 1 (HOLD)
    #
    #     for i in range(len(data) - time_barrier_periods):
    #         entry_price = data['close'].iloc[i]
    #
    #         # Итерируемся по будущим свечам внутри временного окна
    #         for j in range(1, time_barrier_periods + 1):
    #             future_price = data['close'].iloc[i + j]
    #
    #             # Проверяем касание верхнего барьера (Take Profit)
    #             if (future_price - entry_price) / entry_price >= take_profit_level.iloc[i]:
    #                 labels.iloc[i] = 2  # BUY
    #                 break  # Выходим из внутреннего цикла, барьер найден
    #
    #             # Проверяем касание нижнего барьера (Stop Loss)
    #             elif (entry_price - future_price) / entry_price >= stop_loss_level.iloc[i]:
    #                 labels.iloc[i] = 0  # SELL
    #                 break  # Выходим из внутреннего цикла, барьер найден
    #
    #         # Если ни один из боковых барьеров не был коснут, метка остается 1 (HOLD)
    #
    #     logger.info(f"Распределение меток (Triple Barrier): {labels.value_counts().to_dict()}")
    #     return labels
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """
        Улучшенное создание меток с более агрессивными порогами
        """
        if 'close' not in data.columns:
            raise ValueError("Колонка 'close' не найдена в данных")

        logger.info("Создание меток с улучшенными параметрами...")

        # Используем более короткий горизонт для лучшей предсказуемости
        prediction_horizon = 5  # вместо 10

        # Рассчитываем будущую доходность
        future_returns = data['close'].shift(-prediction_horizon) / data['close'] - 1

        # Используем ATR для адаптивных порогов
        if 'atr' not in data.columns:
            data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)

        # Нормализованный ATR
        atr_pct = data['atr'] / data['close']

        # Адаптивные пороги на основе волатильности
        # Делаем пороги более агрессивными для лучшего баланса классов
        buy_threshold = atr_pct * 0.5  # Снижено с 2.0
        sell_threshold = -atr_pct * 0.5  # Снижено с 1.0

        # Создаем метки
        labels = pd.Series(1, index=data.index)  # По умолчанию HOLD

        # Более агрессивная классификация
        labels[future_returns > buy_threshold] = 2  # BUY
        labels[future_returns < sell_threshold] = 0  # SELL

        # Дополнительная балансировка через квантили
        if labels.value_counts().min() / len(labels) < 0.15:
            # Используем квантили для лучшего баланса
            upper_quantile = future_returns.quantile(0.7)
            lower_quantile = future_returns.quantile(0.3)

            labels = pd.Series(1, index=data.index)
            labels[future_returns >= upper_quantile] = 2
            labels[future_returns <= lower_quantile] = 0

        class_distribution = labels.value_counts()
        logger.info(f"Распределение меток: {class_distribution.to_dict()}")
        logger.info(f"Баланс классов: {(class_distribution / len(labels) * 100).round(1).to_dict()}%")

        return labels
    # def create_labels(self, data: pd.DataFrame) -> pd.Series:
    #     """
    #     Создание меток для классификации с улучшенной балансировкой
    #
    #     Returns:
    #         Метки: 0 = продавать, 1 = держать, 2 = покупать
    #     """
    #     if 'close' not in data.columns:
    #         raise ValueError("Колонка 'close' не найдена в данных")
    #
    #     # Рассчитываем будущую доходность
    #     future_returns = data['close'].shift(-self.prediction_horizon) / data['close'] - 1
    #
    #     if self.adaptive_thresholds:
    #         # Рассчитываем динамические пороги на основе волатильности
    #         if 'atr' in data.columns and data['atr'].notna().sum() > 0:
    #             volatility_threshold = data['atr'] / data['close'] if 'atr' in data.columns else 0.01
    #         else:
    #             # Используем историческую волатильность
    #             returns = data['close'].pct_change()
    #             volatility_threshold = returns.rolling(window=self.volatility_window).std()
    #
    #         # Создаем адаптивные пороги - делаем их менее строгими для лучшего баланса классов
    #         upper_threshold = volatility_threshold * 0.75  # Уменьшили с 1.5 до 0.75
    #         lower_threshold = -volatility_threshold * 0.75  # Уменьшили с 1.5 до 0.75
    #     else:
    #         # Фиксированные пороги на основе квантилей
    #         returns_clean = future_returns.dropna()
    #         if len(returns_clean) > 0:
    #             upper_threshold = returns_clean.quantile(0.7)  # Увеличили с 0.75 до 0.7
    #             lower_threshold = returns_clean.quantile(0.3)  # Уменьшили с 0.25 до 0.3
    #         else:
    #             upper_threshold = 0.01
    #             lower_threshold = -0.01
    #
    #     # Создаем метки
    #     labels = pd.Series(1, index=data.index)  # По умолчанию HOLD
    #     labels[future_returns > upper_threshold] = 2  # BUY
    #     labels[future_returns < lower_threshold] = 0  # SELL
    #
    #     # Проверяем распределение классов и корректируем при необходимости
    #     class_counts = labels.value_counts()
    #     total_samples = len(labels)
    #
    #     # Если какой-то класс составляет менее 5% от общего количества, корректируем пороги
    #     min_class_ratio = 0.05
    #     for class_label, count in class_counts.items():
    #         if count / total_samples < min_class_ratio:
    #             logger.warning(f"Класс {class_label} составляет менее {min_class_ratio*100}% данных ({count}/{total_samples})")
    #
    #             # Делаем пороги еще менее строгими
    #             if self.adaptive_thresholds:
    #                 upper_threshold = volatility_threshold * 0.5
    #                 lower_threshold = -volatility_threshold * 0.5
    #             else:
    #                 upper_threshold = returns_clean.quantile(0.65)
    #                 lower_threshold = returns_clean.quantile(0.35)
    #
    #             # Пересоздаем метки
    #             labels = pd.Series(1, index=data.index)
    #             labels[future_returns > upper_threshold] = 2
    #             labels[future_returns < lower_threshold] = 0
    #             break
    #
    #
    #
    #     return labels

    def clean_and_validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Очистка и валидация данных
        """
        # Удаляем строки с бесконечными значениями
        data = data.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)

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

    def create_features_and_labels(self, data: pd.DataFrame, for_prediction: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
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
            logger.info("Создание меток для обучения...")
            labels = self.create_labels(clean_data)

            # 9. Выбор только числовых признаков
            logger.info("Выбор числовых признаков для нормализации...")
            numeric_feature_cols = self.select_numeric_features(clean_data)
            features_df = clean_data[numeric_feature_cols].copy()


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

            # 10. Выравнивание признаков и меток ДО нормализации
            logger.info("Выравнивание признаков и меток по временному индексу...")
            common_index = features_df.index.intersection(labels.dropna().index)

            if common_index.empty:
                logger.error("Нет общих индексов после создания меток. Проверьте логику create_labels.")
                return None, None

            features_aligned = features_df.loc[common_index]
            labels_aligned = labels.loc[common_index]

            # 11. Нормализация признаков
            logger.info("Нормализация признаков...")
            try:
                if not self.is_fitted:
                    # При первом обучении запоминаем имена признаков
                    self.feature_names_in_ = features_aligned.columns.tolist()
                    features_scaled = pd.DataFrame(
                        self.scaler.fit_transform(features_aligned),
                        columns=features_aligned.columns,
                        index=features_aligned.index
                    )
                    self.is_fitted = True
                    logger.info(f"Scaler обучен на {len(self.feature_names_in_)} признаках")
                else:
                    # При последующих вызовах приводим признаки к эталонному набору
                    current_columns = set(features_aligned.columns)
                    expected_columns = set(self.feature_names_in_)

                    # Находим различия
                    missing_cols = expected_columns - current_columns
                    extra_cols = current_columns - expected_columns

                    if missing_cols or extra_cols:
                        logger.warning(f"Обнаружены различия в признаках:")
                        if missing_cols:
                            logger.warning(f"  Недостающие: {list(missing_cols)}")
                            # Добавляем недостающие признаки с нейтральными значениями
                            for col in missing_cols:
                                if 'EMA_200' in col or 'ema' in col.lower():
                                    features_aligned[col] = features_aligned.get('close', 0.0)
                                elif 'ratio' in col:
                                    features_aligned[col] = 1.0
                                elif 'rsi' in col.lower():
                                    features_aligned[col] = 50.0
                                elif 'adx' in col.lower():
                                    features_aligned[col] = 20.0
                                else:
                                    features_aligned[col] = 0.0

                        if extra_cols:
                            logger.warning(f"  Лишние: {list(extra_cols)}")

                    # Приводим к правильному порядку колонок
                    features_aligned = features_aligned.reindex(columns=self.feature_names_in_, fill_value=0.0)

                    # Теперь нормализуем
                    features_scaled = pd.DataFrame(
                        self.scaler.transform(features_aligned),
                        columns=self.feature_names_in_,
                        index=features_aligned.index
                    )
            except Exception as e:
                logger.error(f"Ошибка при нормализации признаков: {e}")
                return None, None

            if for_prediction:
                # Если нам нужны только признаки для предсказания, возвращаем их
                logger.debug(
                    f"Создано {features_scaled.shape[1]} признаков для {features_scaled.shape[0]} наблюдений (режим предсказания).")
                return features_scaled, None
            else:
                # --- Этот блок выполняется только при обучении (когда for_prediction=False) ---
                logger.info("Создание меток для обучения...")

                # 8. Создание меток
                labels = self.create_labels(clean_data)

                # Выравнивание признаков и меток по индексам
                common_index = features_scaled.index.intersection(labels.index)
                features_final = features_scaled.loc[common_index]
                labels_final = labels.loc[common_index]

                # Логика проверки и улучшения баланса классов
                class_distribution = labels_final.value_counts()
                logger.info(f"Создано {features_final.shape[1]} признаков для {features_final.shape[0]} наблюдений")
                logger.info(f"Распределение меток: {class_distribution.to_dict()}")

                total_samples = len(labels_final)
                if total_samples > 0:
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
#-----------------------------------------------
    def _add_volume_spike_feature(self, data: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
        """Добавляет признак для анализа всплесков объема."""
        df = data.copy()
        try:
            volume_sma = ta.sma(df['volume'], length=20)
            # Добавляем малое число, чтобы избежать деления на ноль, если объем был нулевым
            df[f'volume_spike_ratio{suffix}'] = df['volume'] / (volume_sma + 1e-9)
        except Exception as e:
            logger.warning(f"Не удалось рассчитать всплеск объема для суффикса '{suffix}': {e}")
        return df

    def _calculate_secondary_indicators(self, data: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """
        Рассчитывает УПРОЩЕННЫЙ набор индикаторов с ЗАЩИТОЙ от ошибок.
        """
        if data.empty or len(data) < 10:  # Снижаем минимальный порог с 20 до 10
            logger.warning(f"Недостаточно данных для расчета индикаторов{suffix}: {len(data)} строк")
            # Возвращаем DataFrame с нейтральными значениями
            result_df = pd.DataFrame(index=data.index if not data.empty else [0])
            result_df[f'price_to_ema_200{suffix}'] = 1.0
            result_df[f'rsi_14{suffix}'] = 50.0
            result_df[f'volume_spike_ratio{suffix}'] = 1.0
            return result_df

        df = data.copy()

        # Сначала рассчитываем всплеск объема
        df = self._add_volume_spike_feature(df, suffix)

        try:
            # АДАПТИВНЫЕ ПЕРИОДЫ в зависимости от количества данных
            available_length = len(df)

            # Для EMA: адаптируем период
            if available_length >= 200:
                ema_period = 200
            elif available_length >= 100:
                ema_period = 100
            elif available_length >= 50:
                ema_period = 50
            else:
                ema_period = min(20, available_length - 5)

            # Для RSI: адаптируем период
            rsi_period = min(14, max(5, available_length // 3))

            ema_200 = ta.ema(df['close'], length=ema_period)
            rsi_14 = ta.rsi(df['close'], length=rsi_period)

            # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ: ПРОВЕРЯЕМ РЕЗУЛЬТАТ ПЕРЕД ИСПОЛЬЗОВАНИЕМ ---
            if ema_200 is not None and not ema_200.isnull().all():
                df[f'price_to_ema_200{suffix}'] = df['close'] / ema_200
            else:
                # Если рассчитать не удалось, создаем колонку с нейтральным значением
                logger.warning(f"EMA_{ema_period} не удалось рассчитать для {suffix}, заполняем единицами")
                df[f'price_to_ema_200{suffix}'] = 1.0

            if rsi_14 is not None and not rsi_14.isnull().all():
                df[f'rsi_14{suffix}'] = rsi_14
            else:
                # Нейтральное значение для RSI - 50
                logger.warning(f"RSI_{rsi_period} не удалось рассчитать для {suffix}, заполняем нейтральным значением")
                df[f'rsi_14{suffix}'] = 50.0

            logger.debug(f"Индикаторы{suffix} рассчитаны с EMA_{ema_period} и RSI_{rsi_period}")

        except Exception as e:
            logger.warning(f"Не удалось рассчитать индикаторы для {suffix}: {e}")
            # Создаем колонки с нейтральными значениями в случае любой ошибки
            df[f'price_to_ema_200{suffix}'] = 1.0
            df[f'rsi_14{suffix}'] = 50.0

        # Выбираем только созданные колонки для мержа
        indicator_cols = [col for col in df.columns if suffix in col]
        return df[indicator_cols]

    # def _create_primary_features(self, data: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Создает ПОЛНЫЙ набор признаков для основного таймфрейма (1H).
    #     Мы используем вашу существующую функцию.
    #     """
    #     df = self._add_volume_spike_feature(data.copy())  # Добавляем анализ объема
    #     return self.calculate_technical_indicators(df)  # Вызываем вашу основную функцию расчета

    def _create_primary_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Создает ПОЛНЫЙ набор признаков для основного таймфрейма (1H).
        """
        df = self._add_volume_spike_feature(data.copy())

        # Сначала вызываем вашу основную функцию расчета
        df_with_indicators = self.calculate_technical_indicators(df)

        # --- ДОБАВЛЯЕМ НЕДОСТАЮЩИЙ AROON ---
        try:
            aroon_data = ta.aroon(df_with_indicators['high'], df_with_indicators['low'], length=14)
            if aroon_data is not None and not aroon_data.empty:
                # Явно переименовываем колонки, чтобы избежать конфликтов
                aroon_data.columns = [f"{col.upper()}_1H" for col in aroon_data.columns]
                df_with_indicators = pd.concat([df_with_indicators, aroon_data], axis=1)
                logger.debug("Индикатор Aroon успешно добавлен в основной набор признаков.")
        except Exception as e:
            logger.warning(f"Ошибка при расчете Aroon для основного таймфрейма: {e}")
        # --- КОНЕЦ БЛОКА ---

        return df_with_indicators

    def _final_preparation(self, features: pd.DataFrame, labels: Optional[pd.Series] = None) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        ФИНАЛЬНАЯ ВЕРСИЯ: Очистка, нормализация и ГАРАНТИРОВАННОЕ ВЫРАВНИВАНИЕ КОЛОНОК.
        """
        # --- БЛОК ОЧИСТКИ ---
        numeric_feature_cols = self.select_numeric_features(features)
        if not numeric_feature_cols: return None, None

        features_df = features[numeric_feature_cols].copy()
        features_df = features_df.ffill().bfill().fillna(0)

        # --- БЛОК НОРМАЛИЗАЦИИ И ВЫРАВНИВАНИЯ ---
        try:
            if not self.is_fitted:
                # Первое обучение: обучаем и сохраняем эталонный список колонок
                logger.info("Первый запуск: обучение нормализатора и сохранение эталонных признаков...")
                scaled_data = self.scaler.fit_transform(features_df)
                self.is_fitted = True
                # Сохраняем имена признаков в том порядке, в котором они были при обучении
                self.feature_names_in_ = features_df.columns.tolist()
                features_scaled = pd.DataFrame(scaled_data, columns=self.feature_names_in_, index=features_df.index)
                logger.info(
                    f"Скейлер обучен на {len(self.feature_names_in_)} признаках: {self.feature_names_in_[:10]}...")
            else:
                # Последующие запуски: приводим колонки к эталону
                current_columns = set(features_df.columns)
                expected_columns = set(self.feature_names_in_)

                # Находим различия
                missing_cols = expected_columns - current_columns
                extra_cols = current_columns - expected_columns

                if missing_cols or extra_cols:
                    logger.warning(f"Обнаружены различия в признаках:")
                    if missing_cols:
                        logger.warning(f"  Недостающие: {list(missing_cols)}")
                        # Добавляем недостающие признаки, заполняя медианными значениями
                        for col in missing_cols:
                            # Используем медианное значение из обучающей выборки или 0
                            features_df[col] = 0.0

                    if extra_cols:
                        logger.warning(f"  Лишние: {list(extra_cols)}")
                        # Удаляем лишние признаки
                        features_df = features_df.drop(columns=list(extra_cols))

                # Приводим к правильному порядку колонок
                features_df = features_df.reindex(columns=self.feature_names_in_, fill_value=0.0)

                # Теперь нормализуем
                scaled_data = self.scaler.transform(features_df)
                features_scaled = pd.DataFrame(scaled_data, columns=self.feature_names_in_, index=features_df.index)

        except Exception as e:
            logger.error(f"Ошибка нормализации: {e}")
            logger.error(f"Текущие признаки: {features_df.columns.tolist()}")
            if hasattr(self, 'feature_names_in_'):
                logger.error(f"Эталонные признаки: {self.feature_names_in_}")
            # Возвращаем исходные данные без нормализации в крайнем случае
            features_scaled = features_df

        if labels is None:
            return features_scaled, None

        common_index = features_scaled.index.intersection(labels.index)
        features_final = features_scaled.loc[common_index]
        labels_final = labels.loc[common_index]
        return features_final, labels_final

    async def create_multi_timeframe_features(self, symbol: str, data_fetcher) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        ГЛАВНАЯ ФУНКЦИЯ: Создает признаки, объединяя данные с нескольких таймфреймов.
        """
        try:
            logger.info(f"МТА для {symbol}: Загрузка данных для всех таймфреймов...")
            timeframes_to_fetch = {
                '15m': Timeframe.FIFTEEN_MINUTES, '30m': Timeframe.THIRTY_MINUTES,
                '1h': Timeframe.ONE_HOUR, '4h': Timeframe.FOUR_HOURS, '1d': Timeframe.ONE_DAY
            }
            tasks = {name: data_fetcher.get_historical_candles(symbol, tf, limit=2000) for name, tf in
                     timeframes_to_fetch.items()}
            all_data_dict = dict(zip(tasks.keys(), await asyncio.gather(*tasks.values())))

            df_primary = all_data_dict.get('1h')
            if df_primary is None or df_primary.empty or len(df_primary) < 200:
                logger.warning(f"Недостаточно основных данных (1H) для {symbol}")
                return None, None

            logger.info(f"МТА для {symbol}: Создание основного набора признаков (1H)...")
            features = self._create_primary_features(df_primary)

            for tf_name, df_other in all_data_dict.items():
                if tf_name == '1h' or df_other is None or df_other.empty:
                    continue

                logger.info(f"МТА для {symbol}: Расчет и добавление признаков с {tf_name}...")
                indicators_other_tf = self._calculate_secondary_indicators(df_other, suffix=f"_{tf_name}")

                features = pd.merge_asof(
                    features.sort_index(), indicators_other_tf.sort_index(),
                    on='timestamp', direction='backward'
                )

            labels = self.create_labels(features)
            return self._final_preparation(features, labels)

        except Exception as e:
            logger.error(f"Ошибка при создании мультитаймфрейм-признаков для {symbol}: {e}", exc_info=True)
            return None, None

#-----------------------------------------------

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

        # --- ИСПРАВЛЕНИЕ ЗДЕСЬ: ПРОВЕРКА ПЕРЕД ДЕЛЕНИЕМ ---
        # Проверка баланса классов
        class_counts = labels.value_counts()
        total_samples = len(labels)

        if total_samples == 0:
            validation_results['errors'].append("После обработки не осталось меток для анализа баланса классов.")
            validation_results['is_valid'] = False
            return validation_results

        min_class_ratio = class_counts.min() / total_samples

        if min_class_ratio < 0.01:
            validation_results['errors'].append(f"Критический дисбаланс классов: {class_counts.to_dict()}")
            validation_results['is_valid'] = False
        elif min_class_ratio < 0.05:
            validation_results['warnings'].append(f"Сильный дисбаланс классов: {class_counts.to_dict()}")
            validation_results['recommendations'].append(
                "Рассмотрите использование SMOTE или изменение порогов классификации")

        # Проверка на пропущенные значения
        denominator = len(features) * len(features.columns)
        if denominator > 0:
            missing_ratio = features.isnull().sum().sum() / denominator
            if missing_ratio > 0.1:
                validation_results['warnings'].append(f"Высокий процент пропущенных значений: {missing_ratio:.2%}")

        # Проверка на константные признаки
        constant_features = [col for col in features.columns if features[col].nunique() <= 1]
        if constant_features:
            validation_results['warnings'].append(f"Найдены константные признаки: {len(constant_features)}")
            validation_results['recommendations'].append("Удалите константные признаки перед обучением")

        # Проверка масштаба признаков
        feature_scales = features.std()
        min_std = feature_scales.min()
        if min_std > 1e-9:  # Проверяем, что минимальное ст. отклонение не равно нулю
            if feature_scales.max() / min_std > 1000:
                validation_results['warnings'].append("Большие различия в масштабах признаков")
                validation_results['recommendations'].append("Убедитесь, что признаки нормализованы")

    except Exception as e:
        validation_results['errors'].append(f"Ошибка при валидации: {e}")
        validation_results['is_valid'] = False

    return validation_results

def create_regression_target(self, data: pd.DataFrame, target_col: str = 'volatility_20') -> Optional[pd.Series]:
        """
        Создает целевую переменную для регрессионной модели (предсказание будущей волатильности).
        """
        if target_col not in data.columns:
            logger.warning(f"Целевая колонка '{target_col}' не найдена в данных для создания регрессионной метки.")
            return None

        # Сдвигаем данные волатильности назад, чтобы на текущей свече предсказывать будущее значение
        return data[target_col].shift(-self.prediction_horizon)


class UnifiedFeatureEngineer:
    """
    Единый генератор признаков для всех стратегий
    Обеспечивает консистентность признаков между стратегиями
    """

    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.feature_cache = {}
        self.cache_ttl = 300  # 5 минут

    async def get_unified_features(self, symbol: str, data: pd.DataFrame,
                                   data_fetcher=None, include_multiframe: bool = True) -> pd.DataFrame:
        """
        Создает унифицированный набор признаков для всех стратегий

        Args:
            symbol: Торговый символ
            data: Основные данные (OHLCV)
            data_fetcher: Для мультитаймфреймовых признаков
            include_multiframe: Включать ли мультитаймфреймовые признаки

        Returns:
            DataFrame с полным набором признаков
        """
        cache_key = f"{symbol}_{len(data)}_{data.index[-1]}"

        # Проверяем кэш
        if cache_key in self.feature_cache:
            cached_time, cached_features = self.feature_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_features

        try:
            # 1. Базовые технические индикаторы
            features = self.feature_engineer.add_technical_indicators(data.copy())

            # 2. Лаговые признаки
            feature_cols = [col for col in features.columns
                            if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
            features = self.feature_engineer.create_lagged_features(features, feature_cols[:10])

            # 3. Кросс-секционные признаки
            features = self.feature_engineer.create_cross_sectional_features(features)

            # 4. Режимные признаки
            features = self.feature_engineer.create_regime_features(features)

            # 5. Мультитаймфреймовые признаки (если нужно и доступно)
            if include_multiframe and data_fetcher:
                try:
                    mtf_features, _ = await self.feature_engineer.create_multi_timeframe_features(
                        symbol, data_fetcher
                    )
                    if mtf_features is not None and not mtf_features.empty:
                        # Объединяем признаки
                        features = features.merge(
                            mtf_features,
                            left_index=True,
                            right_index=True,
                            how='left'
                        )
                except Exception as e:
                    logger.warning(f"Не удалось создать мультитаймфреймовые признаки: {e}")

            # 6. Нормализация признаков
            features = self._normalize_features(features)

            # 7. Удаление NaN
            features = features.fillna(method='ffill').fillna(0)

            # Кэшируем результат
            self.feature_cache[cache_key] = (datetime.now(), features)

            # Ограничиваем размер кэша
            if len(self.feature_cache) > 100:
                oldest_key = min(self.feature_cache.keys(),
                                 key=lambda k: self.feature_cache[k][0])
                del self.feature_cache[oldest_key]

            logger.debug(f"Создано {len(features.columns)} унифицированных признаков для {symbol}")
            return features

        except Exception as e:
            logger.error(f"Ошибка создания унифицированных признаков: {e}")
            return data

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Нормализует признаки для консистентности"""
        # Определяем типы признаков
        price_features = [col for col in features.columns if any(
            x in col for x in ['price', 'sma', 'ema', 'high', 'low', 'close', 'open']
        )]

        ratio_features = [col for col in features.columns if any(
            x in col for x in ['ratio', 'percent', 'pct', 'rsi', 'cci']
        )]

        # Нормализуем ценовые признаки относительно текущей цены
        if 'close' in features.columns:
            current_price = features['close'].iloc[-1]
            for col in price_features:
                if col != 'close' and col in features.columns:
                    features[f'{col}_norm'] = features[col] / current_price

        # Ограничиваем экстремальные значения
        for col in features.columns:
            if features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Winsorize на уровне 1% и 99%
                lower = features[col].quantile(0.01)
                upper = features[col].quantile(0.99)
                features[col] = features[col].clip(lower=lower, upper=upper)

        return features

    def get_feature_importance_ranking(self) -> List[str]:
        """Возвращает ранжированный список важности признаков"""
        # Это может быть обновлено на основе результатов ML моделей
        return [
            'rsi_14', 'macd_histogram', 'bb_percent', 'atr_ratio',
            'volume_ratio', 'momentum_10', 'adx', 'mfi',
            'trend_strength', 'volatility_20'
        ]


# Создаем глобальный экземпляр
unified_feature_engineer = UnifiedFeatureEngineer()
