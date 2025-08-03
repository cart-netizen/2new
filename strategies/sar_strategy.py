# strategies/stop_and_reverse_strategy.py

# import asyncio
import json

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

from core.schemas import TradingSignal, SignalType
from strategies.base_strategy import BaseStrategy
from core.enums import Timeframe
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SARSignalComponents:
    """Компоненты SAR сигнала для детальной аналитики"""
    psar_trigger: bool = False
    rsi_divergence: bool = False
    macd_divergence: bool = False
    rsi_extreme_zone: bool = False
    mfi_extreme_zone: bool = False
    aroon_confirmation: bool = False
    macro_trend_alignment: bool = False
    hma_rsi_trend_signal: bool = False
    aroon_oscillator_signal: bool = False
    ichimoku_confirmation: bool = False
    chop_filter_passed: bool = False
    adx_filter_passed: bool = False
    atr_filter_passed: bool = False
    shadow_system_score: float = 0.0
    total_score: int = 0


class StopAndReverseStrategy(BaseStrategy):
    """
    Профессиональная реализация стратегии Stop-and-Reverse 
    согласно исследованию с многоуровневой фильтрацией и подтверждениями
    """

    def __init__(self, config: Dict[str, Any], data_fetcher=None):
        super().__init__(strategy_name="Stop_and_Reverse")
        
        # Загружаем конфигурацию SAR из config.json
        self.sar_config = config.get('stop_and_reverse_strategy', {})
        self.data_fetcher = data_fetcher
        # Фильтры режимов (Уровень 1-3 из исследования)
        self.chop_period = self.sar_config.get('chop_period', 14)
        self.chop_threshold = self.sar_config.get('chop_threshold', 40)
        self.adx_period = self.sar_config.get('adx_period', 14) 
        self.adx_threshold = self.sar_config.get('adx_threshold', 25)
        self.atr_period = self.sar_config.get('atr_period', 14)
        self.atr_multiplier = self.sar_config.get('atr_multiplier', 1.25)
        
        # Основной генератор сигналов - PSAR
        self.psar_start = self.sar_config.get('psar_start', 0.02)
        self.psar_step = self.sar_config.get('psar_step', 0.02) 
        self.psar_max = self.sar_config.get('psar_max', 0.2)
        
        # Подтверждения
        self.rsi_period = self.sar_config.get('rsi_period', 14)
        self.rsi_overbought = self.sar_config.get('rsi_overbought', 70)
        self.rsi_oversold = self.sar_config.get('rsi_oversold', 30)
        self.macd_fast = self.sar_config.get('macd_fast', 12)
        self.macd_slow = self.sar_config.get('macd_slow', 26)
        self.macd_signal = self.sar_config.get('macd_signal', 9)
        self.mfi_period = self.sar_config.get('mfi_period', 14)
        self.mfi_overbought = self.sar_config.get('mfi_overbought', 80)
        self.mfi_oversold = self.sar_config.get('mfi_oversold', 20)
        self.aroon_period = self.sar_config.get('aroon_period', 25)
        
        # Макро-фильтр
        self.ema_short = self.sar_config.get('ema_short', 50)
        self.ema_long = self.sar_config.get('ema_long', 200)
        
        # HMA RSI Trend (новый индикатор)
        self.hma_fast_period = self.sar_config.get('hma_fast_period', 14)
        self.hma_slow_period = self.sar_config.get('hma_slow_period', 28)
        self.hma_rsi_period = self.sar_config.get('hma_rsi_period', 14)
        self.hma_adx_threshold = self.sar_config.get('hma_adx_threshold', 20)
        
        # Ichimoku Cloud настройки
        self.ichimoku_conversion = self.sar_config.get('ichimoku_conversion', 9)
        self.ichimoku_base = self.sar_config.get('ichimoku_base', 26)
        self.ichimoku_span_b = self.sar_config.get('ichimoku_span_b', 52)
        self.ichimoku_displacement = self.sar_config.get('ichimoku_displacement', 26)
        
        # Система оценок сигналов
        self.min_signal_score = self.sar_config.get('min_signal_score', 4)
        self.psar_score_weight = self.sar_config.get('psar_score_weight', 1)
        self.divergence_score_weight = self.sar_config.get('divergence_score_weight', 2)
        self.extreme_zone_score_weight = self.sar_config.get('extreme_zone_score_weight', 1)
        self.aroon_score_weight = self.sar_config.get('aroon_score_weight', 1)
        self.macro_alignment_score_weight = self.sar_config.get('macro_alignment_score_weight', 1)
        
        # Защита от быстрых разворотов
        self.commission_rate = 0.00085  # 0.075%
        self.min_profit_protection = self.commission_rate * 2 * 4  # 3 комиссии
        
        # Shadow system интеграция
        self.use_shadow_system = self.sar_config.get('use_shadow_system', True)
        self.shadow_weight = self.sar_config.get('shadow_weight', 0.5)
        
        # Символы для мониторинга
        self.monitored_symbols: Dict[str, Dict] = {}
        self.min_daily_volume_usd = self.sar_config.get('min_daily_volume_usd', 1_000_000)
        self.last_symbol_update = datetime.min
        self.symbol_update_interval = timedelta(hours=1)
        
        # ML модели интеграция (для быстрого включения)
        self.use_ml_confirmation = self.sar_config.get('use_ml_confirmation', True)
        self.ml_weight = self.sar_config.get('ml_weight', 1.0)
        
        # Текущие позиции стратегии
        self.current_positions: Dict[str, Dict] = {}
        self.data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.cache_ttl = timedelta(minutes=5)  # TTL для кэша данных

        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_profit_per_trade': 0.0,
            'recent_trades': [],
            'recent_win_rate': 0.0,
            'last_update': datetime.now()
        }

        # Адаптивные параметры PSAR
        self.acceleration = self.psar_start  # Текущее ускорение
        self.max_acceleration = self.psar_max  # Максимальное ускорение
        self.sensitivity = self.sar_config.get('sensitivity', 0.8)  # Чувствительность к сигналам
        self.confidence_threshold = self.sar_config.get('confidence_threshold', 0.7)  # Порог уверенности

        # Параметры риск-менеджмента
        self.stop_loss_atr_multiplier = self.sar_config.get('stop_loss_atr_multiplier', 1.5)
        self.take_profit_atr_multiplier = self.sar_config.get('take_profit_atr_multiplier', 3.0)

        # История адаптации параметров
        self.parameter_history = []

        # Текущие рыночные данные (для адаптации)
        self.current_data = None
        self.market_regime = 'unknown'
        self.trend_strength = 0.5

        # Время последнего сигнала
        self.last_signal_time = datetime.min
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'component_effectiveness': {
                'psar_trigger': {'total': 0, 'wins': 0, 'avg_profit': 0.0},
                'rsi_divergence': {'total': 0, 'wins': 0, 'avg_profit': 0.0},
                'macd_divergence': {'total': 0, 'wins': 0, 'avg_profit': 0.0},
                'aroon_confirmation': {'total': 0, 'wins': 0, 'avg_profit': 0.0},
                'trend_alignment': {'total': 0, 'wins': 0, 'avg_profit': 0.0},
                'momentum_strength': {'total': 0, 'wins': 0, 'avg_profit': 0.0}
            },
            'parameter_adjustments': [],
            'last_update': datetime.now()
        }

        # Словарь параметров PSAR для динамической корректировки
        self.psar_params = {
            'start': self.psar_start,
            'step': self.psar_step,
            'max': self.psar_max,
            'acceleration_factor': self.psar_start,
            'sensitivity': self.sensitivity,
            'confidence_threshold': self.confidence_threshold,
            'last_modified': datetime.now()
        }
        
        logger.info(f"✅ Stop-and-Reverse стратегия инициализирована с конфигурацией: {self.sar_config}")

    async def should_trade_symbol(self, symbol: str, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Трехуровневый фильтр режимов из исследования:
        1. Choppiness Index < 40
        2. ADX > 25  
        3. ATR волатильность выше среднего
        """
        logger.debug(f"Проверка символа {symbol} для SAR стратегии...")
        try:
            if len(data) < max(self.chop_period, self.adx_period, self.atr_period, 100):
                return False, "Недостаточно исторических данных"
            
            # Уровень 1: Фильтр рваности (Choppiness Filter)
            chop = self._calculate_choppiness_index(data, self.chop_period)
            if chop is None or chop > self.chop_threshold:
                return False, f"Высокая рваность рынка: CHOP={chop:.2f} > {self.chop_threshold}"
            
            # Уровень 2: Фильтр силы тренда (Trend Strength Filter)
            adx_data = ta.adx(data['high'], data['low'], data['close'], length=self.adx_period)
            if adx_data is None or adx_data.empty:
                return False, "Не удалось рассчитать ADX"
            
            current_adx = adx_data.iloc[-1, 0] if len(adx_data.columns) > 0 else None
            if current_adx is None or current_adx < self.adx_threshold:
                return False, f"Слабый тренд: ADX={current_adx:.2f} < {self.adx_threshold}"
            
            # Уровень 3: Фильтр волатильности (Volatility Filter)
            atr_14 = ta.atr(data['high'], data['low'], data['close'], length=self.atr_period)
            atr_100 = ta.atr(data['high'], data['low'], data['close'], length=100)
            
            if atr_14 is None or atr_100 is None or atr_14.empty or atr_100.empty:
                return False, "Не удалось рассчитать ATR"
            
            current_atr = atr_14.iloc[-1]
            avg_atr = atr_100.iloc[-1]
            
            if current_atr < avg_atr * self.atr_multiplier:
                return False, f"Низкая волатильность: ATR={current_atr:.6f} < {avg_atr * self.atr_multiplier:.6f}"
            
            return True, f"Фильтры пройдены: CHOP={chop:.2f}, ADX={current_adx:.2f}, ATR_ratio={current_atr/avg_atr:.2f}"
            
        except Exception as e:
            logger.error(f"Ошибка в фильтрах режимов для {symbol}: {e}")
            volume_24h = data['volume'].iloc[-24:].sum() * data['close'].iloc[-1]
            logger.debug(f"Символ {symbol} отклонен: объем 24h = {volume_24h:.0f} USDT")
            return False, f"Низкий объем торгов: {volume_24h:.0f} USDT, Ошибка фильтров: {str(e)}"


    def _calculate_choppiness_index(self, data: pd.DataFrame, period: int) -> Optional[float]:
        """Расчет Choppiness Index согласно исследованию"""
        try:
            if len(data) < period:
                return None
            
            recent_data = data.tail(period)
            
            # Сумма True Range за период
            tr_sum = 0
            for i in range(1, len(recent_data)):
                high = recent_data['high'].iloc[i]
                low = recent_data['low'].iloc[i]
                prev_close = recent_data['close'].iloc[i-1]
                
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                tr_sum += tr
            
            # Максимальный и минимальный уровни за период
            highest_high = recent_data['high'].max()
            lowest_low = recent_data['low'].min()
            
            if highest_high == lowest_low or tr_sum == 0:
                return 50  # Нейтральное значение
            
            # Формула Choppiness Index
            chop = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
            
            return float(chop)
            
        except Exception as e:
            logger.error(f"Ошибка расчета Choppiness Index: {e}")
            return None

    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Главная функция генерации сигналов Stop-and-Reverse
        Реализует полную логику из исследования
        """
        try:
            components = SARSignalComponents()



            # Расчет индикаторов (предполагаем что они уже рассчитаны в data)
            psar_value = data['psar'].iloc[-1] if 'psar' in data.columns else None
            rsi_value = data['rsi'].iloc[-1] if 'rsi' in data.columns else 50.0
            adx_value = data['adx'].iloc[-1] if 'adx' in data.columns else 20.0

            # Проверяем минимальное количество данных
            min_required = max(200, self.ema_long, self.ichimoku_span_b + self.ichimoku_displacement)
            if len(data) < min_required:
                return None
            
            # Проверяем фильтры режимов
            can_trade, filter_reason = await self.should_trade_symbol(symbol, data)
            if not can_trade:
                logger.debug(f"SAR фильтры не пройдены для {symbol}: {filter_reason}")
                return None
            
            # Анализируем компоненты сигнала
            signal_components = await self._analyze_signal_components(symbol, data)
            
            # Определяем направление сигнала
            signal_type = await self._determine_signal_direction(symbol, data, signal_components)
            if signal_type is None:
                return None
            
            # Проверяем систему оценок
            if signal_components.total_score < self.min_signal_score:
                logger.debug(f"SAR сигнал для {symbol} не достиг минимального балла: {signal_components.total_score} < {self.min_signal_score}")
                return None
            
            # Проверяем защиту от быстрых разворотов
            if not await self._check_reversal_protection(symbol, data, signal_type):
                logger.debug(f"SAR сигнал для {symbol} заблокирован защитой от быстрых разворотов")
                return None
            
            # Получаем данные Shadow System
            shadow_score = await self._get_shadow_system_score(symbol, signal_type) if self.use_shadow_system else 0.5
            
            # Формируем итоговый сигнал
            confidence = self._calculate_signal_confidence(signal_components, shadow_score)

            config = getattr(self, 'config', {})
            strategy_settings = config.get('strategy_settings', {})
            ltf_str = strategy_settings.get('ltf_entry_timeframe', '5m')

            config = getattr(self, 'config', {})
            strategy_settings = config.get('strategy_settings', {})
            ltf_str = strategy_settings.get('ltf_entry_timeframe', '5m')
            # psar_value = data['psar'].iloc[-1]
            # Создаем сигнал с правильными параметрами
            signal = TradingSignal(
                signal_type=signal_type,
                symbol=symbol,
                price=float(data['close'].iloc[-1]),  # Используем price вместо entry_price
                confidence=confidence,
                strategy_name=self.strategy_name,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    'sar_components': signal_components.__dict__,
                    'components': components.__dict__,
                    'psar_value': float(psar_value) if psar_value else None,
                    'rsi': float(rsi_value),
                    'adx': float(adx_value),
                    'shadow_score': shadow_score,
                    'filter_reason': filter_reason,
                    'volume': float(data['volume'].iloc[-1]) if 'volume' in data.columns else 0,
                    'timeframe': ltf_str,  # Сохраняем в metadata как строку
                    'signal_score': signal_components.total_score,
                    'min_required_score': self.min_signal_score,
                    'protection_checks': 'passed',
                    'entry_timeframe': ltf_str
                }
            )

            logger.info(f"✅ SAR сигнал сгенерирован для {symbol}: {signal_type.value}, confidence={confidence:.3f}, score={signal_components.total_score}")
            return signal
            
        except Exception as e:
            logger.error(f"Ошибка генерации SAR сигнала для {symbol}: {e}")
            return None

    async def _analyze_signal_components(self, symbol: str, data: pd.DataFrame) -> SARSignalComponents:
        """Анализирует все компоненты сигнала согласно исследованию"""
        components = SARSignalComponents()
        
        try:
            # 1. Основной триггер - Parabolic SAR
            components.psar_trigger = self._check_psar_trigger(data)
            if components.psar_trigger:
                components.total_score += self.psar_score_weight
            
            # 2. Подтверждение 1: Дивергенция моментума (RSI/MACD)
            rsi_div = self._check_rsi_divergence(data)
            macd_div = self._check_macd_divergence(data)
            components.rsi_divergence = rsi_div
            components.macd_divergence = macd_div
            if rsi_div or macd_div:
                components.total_score += self.divergence_score_weight
            
            # 3. Подтверждение 2: Состояние осциллятора (RSI/MFI экстремальные зоны)
            rsi_extreme = self._check_rsi_extreme_zone(data)
            mfi_extreme = self._check_mfi_extreme_zone(data)
            components.rsi_extreme_zone = rsi_extreme
            components.mfi_extreme_zone = mfi_extreme
            if rsi_extreme or mfi_extreme:
                components.total_score += self.extreme_zone_score_weight
            
            # 4. Подтверждение 3: Структура тренда (Aroon)
            components.aroon_confirmation = self._check_aroon_structure(data)
            if components.aroon_confirmation:
                components.total_score += self.aroon_score_weight
            
            # 5. Макро-контекст (EMA 50/200 на 1H)
            macro_data = await self._get_macro_timeframe_data(symbol)
            if macro_data is not None:
                components.macro_trend_alignment = self._check_macro_trend_alignment(macro_data)
                if components.macro_trend_alignment:
                    components.total_score += self.macro_alignment_score_weight
            
            # 6. HMA RSI Trend сигнал (4H таймфрейм)
            hma_data = await self._get_hma_timeframe_data(symbol)
            if hma_data is not None:
                components.hma_rsi_trend_signal = self._check_hma_rsi_trend_signal(hma_data)
                if components.hma_rsi_trend_signal:
                    components.total_score += 1
            
            # 7. Aroon Oscillator подтверждение
            components.aroon_oscillator_signal = self._check_aroon_oscillator(data)
            if components.aroon_oscillator_signal:
                components.total_score += 1
            
            # 8. Ichimoku Cloud подтверждение
            try:
                components.ichimoku_confirmation = self._check_ichimoku_confirmation(data)
            except Exception as e:
                logger.warning(f"Ошибка Ichimoku, использую fallback: {e}")
                components.ichimoku_confirmation = self._check_ichimoku_confirmation_fallback(data)
            if components.ichimoku_confirmation:
                components.total_score += 1
            
            # Отмечаем пройденные фильтры
            components.chop_filter_passed = True  # Уже проверено в should_trade_symbol
            components.adx_filter_passed = True
            components.atr_filter_passed = True
            
            return components
            
        except Exception as e:
            logger.error(f"Ошибка анализа компонентов сигнала для {symbol}: {e}")
            return components

    def _check_psar_trigger(self, data: pd.DataFrame) -> bool:
        """Проверка пересечения PSAR с ценой"""
        try:
            if len(data) < 3 or 'psar' not in data.columns:
                return False

            # Безопасное извлечение PSAR значений
            psar_data = data['psar']
            if psar_data is None or len(psar_data) < 2:
                return False

            # Получаем последние значения
            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-2]

            # Безопасное получение PSAR значений
            if hasattr(psar_data, 'iloc'):
                current_psar = psar_data.iloc[-1]
                prev_psar = psar_data.iloc[-2]
            else:
                psar_values = np.array(psar_data)
                current_psar = psar_values[-1]
                prev_psar = psar_values[-2]

            # Проверяем на NaN
            if pd.isna(current_psar) or pd.isna(prev_psar):
                return False

            # Проверяем переворот PSAR
            buy_signal = (prev_price <= prev_psar) and (current_price > current_psar)
            sell_signal = (prev_price >= prev_psar) and (current_price < current_psar)

            return buy_signal or sell_signal

        except Exception as e:
            logger.error(f"Ошибка проверки PSAR триггера: {e}")
            return False

    def _check_rsi_divergence(self, data: pd.DataFrame) -> bool:
        """Проверка дивергенции RSI (15м/30м таймфрейм)"""
        try:
            if len(data) < 50:
                return False
            
            rsi = ta.rsi(data['close'], length=self.rsi_period)
            if rsi is None or rsi.empty:
                return False
            
            # Ищем локальные максимумы и минимумы
            prices = data['close'].values[-30:]  # Последние 30 баров
            # Безопасное извлечение значений RSI
            if hasattr(rsi, 'values'):
                rsi_values = rsi.values[-30:]
            elif hasattr(rsi, 'iloc'):
                rsi_values = rsi.iloc[-30:].values
            else:
                rsi_values = np.array(rsi)[-30:]
            
            # Находим пики и впадины
            price_peaks = self._find_peaks(prices)
            rsi_peaks = self._find_peaks(rsi_values)
            price_troughs = self._find_troughs(prices)
            rsi_troughs = self._find_troughs(rsi_values)
            
            # Медвежья дивергенция: цена растет, RSI падает
            bearish_div = (len(price_peaks) >= 2 and len(rsi_peaks) >= 2 and
                          price_peaks[-1] > price_peaks[-2] and rsi_peaks[-1] < rsi_peaks[-2])
            
            # Бычья дивергенция: цена падает, RSI растет
            bullish_div = (len(price_troughs) >= 2 and len(rsi_troughs) >= 2 and
                          price_troughs[-1] < price_troughs[-2] and rsi_troughs[-1] > rsi_troughs[-2])
            
            return bearish_div or bullish_div
            
        except Exception as e:
            logger.error(f"Ошибка проверки RSI дивергенции: {e}")
            return False

    def _check_macd_divergence(self, data: pd.DataFrame) -> bool:
        """Проверка дивергенции MACD (15м/30м таймфрейм)"""
        try:
            if len(data) < 50:
                return False
            
            macd_data = ta.macd(data['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            if macd_data is None or macd_data.empty:
                return False

            # Используем MACD гистограмму для дивергенции
            if isinstance(macd_data, pd.DataFrame) and len(macd_data.columns) > 2:
                macd_hist = macd_data.iloc[:, 2]
            elif isinstance(macd_data, tuple) and len(macd_data) > 2:
                macd_hist = macd_data[2]  # Третий элемент tuple - гистограмма
            else:
                logger.debug("MACD данные не содержат гистограмму")
                return False

            if macd_hist is None:
                return False

            # Безопасное извлечение значений
            prices = data['close'].values[-30:]
            if hasattr(macd_hist, 'values'):
                hist_values = macd_hist.values[-30:]
            elif hasattr(macd_hist, 'iloc'):
                hist_values = macd_hist.iloc[-30:].values
            else:
                hist_values = np.array(macd_hist)[-30:]
            
            price_peaks = self._find_peaks(prices)
            hist_peaks = self._find_peaks(hist_values)
            price_troughs = self._find_troughs(prices)
            hist_troughs = self._find_troughs(hist_values)
            
            # Медвежья дивергенция
            bearish_div = (len(price_peaks) >= 2 and len(hist_peaks) >= 2 and
                          price_peaks[-1] > price_peaks[-2] and hist_peaks[-1] < hist_peaks[-2])
            
            # Бычья дивергенция
            bullish_div = (len(price_troughs) >= 2 and len(hist_troughs) >= 2 and
                          price_troughs[-1] < price_troughs[-2] and hist_troughs[-1] > hist_troughs[-2])
            
            return bearish_div or bullish_div
            
        except Exception as e:
            logger.error(f"Ошибка проверки MACD дивергенции: {e}")
            return False

    def _find_peaks(self, data: np.ndarray, min_distance: int = 3) -> List[float]:
        """Находит локальные максимумы в данных, корректно извлекая значения."""
        try:
            # Преобразуем в numpy array для универсальности
            if hasattr(data, 'values'):
                values = data.values
            elif hasattr(data, 'iloc'):
                values = data.values
            else:
                values = np.array(data)

            peaks = []
            for i in range(min_distance, len(values) - min_distance):
                is_peak = True
                for j in range(i - min_distance, i + min_distance + 1):
                    if j != i and values[j] >= values[i]:
                        is_peak = False
                        break
                if is_peak:
                    peaks.append(values[i])

            return peaks

        except Exception as e:
            logger.error(f"Ошибка поиска пиков: {e}")
            return []

    # def _find_peaks(self, data: np.ndarray, min_distance: int = 3) -> List[float]:
    #     """Находит локальные максимумы в данных"""
    #     peaks = []
    #     for i in range(min_distance, len(data) - min_distance):
    #         if all(data[i] >= data[i-j] for j in range(1, min_distance + 1)) and \
    #            all(data[i] >= data[i+j] for j in range(1, min_distance + 1)):
    #             peaks.append(data[i])
    #     return peaks

    # def _find_troughs(self, data: np.ndarray, min_distance: int = 3) -> List[float]:
    #     """Находит локальные минимумы в данных"""
    #     troughs = []
    #     for i in range(min_distance, len(data) - min_distance):
    #         if all(data[i] <= data[i-j] for j in range(1, min_distance + 1)) and \
    #            all(data[i] <= data[i+j] for j in range(1, min_distance + 1)):
    #             troughs.append(data[i])
    #     return troughs

    def _find_troughs(self, data: np.ndarray, min_distance: int = 3) -> List[float]:
        """Находит локальные минимумы в данных."""
        try:
            # Преобразуем в numpy array для универсальности
            if hasattr(data, 'values'):
                values = data.values
            elif hasattr(data, 'iloc'):
                values = data.values
            else:
                values = np.array(data)

            troughs = []
            for i in range(min_distance, len(values) - min_distance):
                is_trough = True
                for j in range(i - min_distance, i + min_distance + 1):
                    if j != i and values[j] <= values[i]:
                        is_trough = False
                        break
                if is_trough:
                    troughs.append(values[i])

            return troughs

        except Exception as e:
            logger.error(f"Ошибка поиска впадин: {e}")
            return []

    def _check_rsi_extreme_zone(self, data: pd.DataFrame) -> bool:
        """Проверка RSI в экстремальной зоне (5м/15м)"""
        try:
            rsi = ta.rsi(data['close'], length=self.rsi_period)
            if rsi is None or rsi.empty:
                return False
            
            current_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else current_rsi
            
            # RSI выходит из зоны перекупленности
            rsi_falling_from_overbought = prev_rsi > self.rsi_overbought and current_rsi <= self.rsi_overbought
            
            # RSI выходит из зоны перепроданности
            rsi_rising_from_oversold = prev_rsi < self.rsi_oversold and current_rsi >= self.rsi_oversold
            
            return rsi_falling_from_overbought or rsi_rising_from_oversold
            
        except Exception as e:
            logger.error(f"Ошибка проверки RSI экстремальной зоны: {e}")
            return False

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

    def _check_mfi_extreme_zone(self, data: pd.DataFrame) -> bool:
        """Проверка Money Flow Index в экстремальной зоне"""
        try:
            if not all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
                return False
            
            mfi = self.calculate_mfi_manual(data['high'], data['low'], data['close'], data['volume'], length=self.mfi_period)
            if mfi is None or mfi.empty:
                return False
            
            current_mfi = mfi.iloc[-1]
            prev_mfi = mfi.iloc[-2] if len(mfi) > 1 else current_mfi
            
            # MFI выходит из зоны перекупленности
            mfi_falling_from_overbought = prev_mfi > self.mfi_overbought and current_mfi <= self.mfi_overbought
            
            # MFI выходит из зоны перепроданности
            mfi_rising_from_oversold = prev_mfi < self.mfi_oversold and current_mfi >= self.mfi_oversold
            
            return mfi_falling_from_overbought or mfi_rising_from_oversold
            
        except Exception as e:
            logger.error(f"Ошибка проверки MFI экстремальной зоны: {e}")
            return False

    def _check_aroon_structure(self, data: pd.DataFrame) -> bool:
        """Проверка структуры тренда через Aroon (15м/30м)"""
        try:
            aroon_data = ta.aroon(data['high'], data['low'], length=self.aroon_period)
            if aroon_data is None or aroon_data.empty or len(aroon_data.columns) < 2:
                return False
            
            aroon_up = aroon_data.iloc[:, 0]  # Aroon Up
            aroon_down = aroon_data.iloc[:, 1]  # Aroon Down
            
            if len(aroon_up) < 2:
                return False
            
            current_aroon_up = aroon_up.iloc[-1]
            current_aroon_down = aroon_down.iloc[-1]
            prev_aroon_up = aroon_up.iloc[-2]
            prev_aroon_down = aroon_down.iloc[-2]
            
            # Бычий сигнал: Aroon Up пересекает Aroon Down снизу вверх
            bullish_cross = (prev_aroon_up <= prev_aroon_down) and (current_aroon_up > current_aroon_down)
            
            # Медвежий сигнал: Aroon Down пересекает Aroon Up снизу вверх
            bearish_cross = (prev_aroon_down <= prev_aroon_up) and (current_aroon_down > current_aroon_up)
            
            return bullish_cross or bearish_cross
            
        except Exception as e:
            logger.error(f"Ошибка проверки Aroon структуры: {e}")
            return False

    def _check_aroon_oscillator(self, data: pd.DataFrame) -> bool:
        """Проверка Aroon Oscillator для дополнительного подтверждения"""
        try:
            aroon_data = ta.aroon(data['high'], data['low'], length=14)
            if aroon_data is None or aroon_data.empty or len(aroon_data.columns) < 2:
                return False
            
            aroon_up = aroon_data.iloc[:, 0]
            aroon_down = aroon_data.iloc[:, 1]
            
            # Aroon Oscillator = Aroon Up - Aroon Down
            aroon_osc = aroon_up - aroon_down
            
            if len(aroon_osc) < 3:
                return False
            
            current_osc = aroon_osc.iloc[-1]
            prev_osc = aroon_osc.iloc[-2]
            prev2_osc = aroon_osc.iloc[-3]
            
            # Сигнал на покупку: осциллятор пересекает 0 снизу вверх
            bullish_signal = prev_osc <= 0 and current_osc > 0
            
            # Сигнал на продажу: осциллятор пересекает 0 сверху вниз
            bearish_signal = prev_osc >= 0 and current_osc < 0
            
            # Дополнительно проверяем импульс
            momentum_confirmation = abs(current_osc - prev2_osc) > 20  # Значительное изменение
            
            return (bullish_signal or bearish_signal) and momentum_confirmation
            
        except Exception as e:
            logger.error(f"Ошибка проверки Aroon Oscillator: {e}")
            return False

    async def _get_macro_timeframe_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Получает данные 1H таймфрейма для макро-контекста"""
        try:
            if not self.data_fetcher:
                logger.warning("DataFetcher не установлен для SAR стратегии")
                return None

            # Проверяем кэш
            cache_key = f"{symbol}_1H"
            if (cache_key in self.data_cache and
                datetime.now() - self.data_cache[cache_key]['timestamp'] < self.cache_ttl):
                return self.data_cache[cache_key]['data']

            # Получаем данные 1H
            data_1h = await self.data_fetcher.get_historical_candles(
                symbol, Timeframe.ONE_HOUR, limit=250
            )

            if data_1h is not None and not data_1h.empty:
                # Сохраняем в кэш
                self.data_cache[cache_key] = {
                    'data': data_1h,
                    'timestamp': datetime.now()
                }
                return data_1h

            return None

        except Exception as e:
            logger.error(f"Ошибка получения макро данных для {symbol}: {e}")
            return None

    async def _get_hma_timeframe_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Получает данные 4H таймфрейма для HMA RSI Trend"""
        try:
            if not self.data_fetcher:
                return None

            # Проверяем кэш
            cache_key = f"{symbol}_4H"
            if (cache_key in self.data_cache and
                datetime.now() - self.data_cache[cache_key]['timestamp'] < self.cache_ttl):
                return self.data_cache[cache_key]['data']

            # Получаем данные 4H
            data_4h = await self.data_fetcher.get_historical_candles(
                symbol, Timeframe.FOUR_HOURS, limit=150
            )

            if data_4h is not None and not data_4h.empty:
                # Сохраняем в кэш
                self.data_cache[cache_key] = {
                    'data': data_4h,
                    'timestamp': datetime.now()
                }
                return data_4h

            return None

        except Exception as e:
            logger.error(f"Ошибка получения 4H данных для {symbol}: {e}")
            return None

    def _check_macro_trend_alignment(self, macro_data: pd.DataFrame) -> bool:
        """Проверка выравнивания с макро-трендом (EMA 50/200 на 1H)"""
        try:
            if len(macro_data) < self.ema_long:
                return False
            
            ema_50 = ta.ema(macro_data['close'], length=self.ema_short)
            ema_200 = ta.ema(macro_data['close'], length=self.ema_long)
            
            if ema_50 is None or ema_200 is None or ema_50.empty or ema_200.empty:
                return False
            
            current_price = macro_data['close'].iloc[-1]
            current_ema_50 = ema_50.iloc[-1]
            current_ema_200 = ema_200.iloc[-1]
            
            # Бычье выравнивание: цена > EMA(50) > EMA(200)
            bullish_alignment = current_price > current_ema_50 > current_ema_200
            
            # Медвежье выравнивание: цена < EMA(50) < EMA(200)
            bearish_alignment = current_price < current_ema_50 < current_ema_200
            
            return bullish_alignment or bearish_alignment
            
        except Exception as e:
            logger.error(f"Ошибка проверки макро-выравнивания: {e}")
            return False

    def _check_hma_rsi_trend_signal(self, hma_data: pd.DataFrame) -> bool:
        """
        Проверка HMA RSI Trend сигнала на 4H таймфрейме
        HMA быстрый пересекает медленный + RSI от HMA + ADX фильтр
        """
        try:
            if len(hma_data) < max(self.hma_slow_period, 50):
                return False
            
            # Рассчитываем Hull Moving Average
            hma_fast = self._calculate_hma(hma_data['close'], self.hma_fast_period)
            hma_slow = self._calculate_hma(hma_data['close'], self.hma_slow_period)
            
            if hma_fast is None or hma_slow is None or len(hma_fast) < 2 or len(hma_slow) < 2:
                return False
            
            # Проверяем пересечение HMA
            current_fast = hma_fast.iloc[-1]
            current_slow = hma_slow.iloc[-1]
            prev_fast = hma_fast.iloc[-2]
            prev_slow = hma_slow.iloc[-2]
            
            # Пересечение снизу вверх (бычий сигнал)
            bullish_cross = (prev_fast <= prev_slow) and (current_fast > current_slow)
            
            # Пересечение сверху вниз (медвежий сигнал)
            bearish_cross = (prev_fast >= prev_slow) and (current_fast < current_slow)
            
            if not (bullish_cross or bearish_cross):
                return False
            
            # RSI от HMA для подтверждения
            hma_rsi = ta.rsi(hma_fast, length=self.hma_rsi_period)
            if hma_rsi is None or hma_rsi.empty:
                return False
            
            current_hma_rsi = hma_rsi.iloc[-1]
            
            # ADX фильтр бокового движения
            adx_data = ta.adx(hma_data['high'], hma_data['low'], hma_data['close'], length=20)
            if adx_data is None or adx_data.empty:
                return False
            
            current_adx = adx_data.iloc[-1, 0] if len(adx_data.columns) > 0 else 0
            
            # Проверяем все условия
            if bullish_cross:
                return current_hma_rsi > 50 and current_adx >= self.hma_adx_threshold
            elif bearish_cross:
                return current_hma_rsi < 50 and current_adx >= self.hma_adx_threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка проверки HMA RSI Trend: {e}")
            return False

    def _calculate_hma(self, prices: pd.Series, period: int) -> Optional[pd.Series]:
        """Расчет Hull Moving Average"""
        try:
            if len(prices) < period:
                return None
            
            # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
            half_period = int(period / 2)
            sqrt_period = int(np.sqrt(period))
            
            wma_half = ta.wma(prices, length=half_period)
            wma_full = ta.wma(prices, length=period)
            
            if wma_half is None or wma_full is None:
                return None
            
            # 2*WMA(n/2) - WMA(n)
            raw_hma = 2 * wma_half - wma_full
            
            # WMA от полученных значений
            hma = ta.wma(raw_hma, length=sqrt_period)
            
            return hma
            
        except Exception as e:
            logger.error(f"Ошибка расчета HMA: {e}")
            return None

    def _check_ichimoku_confirmation(self, data: pd.DataFrame) -> bool:
        """
        Проверка подтверждения Ichimoku Cloud
        ИСПРАВЛЕННАЯ версия для работы с разными форматами возврата ta.ichimoku
        """
        try:
            min_length = max(self.ichimoku_span_b, self.ichimoku_displacement) + 10
            if len(data) < min_length:
                logger.debug("Недостаточно данных для Ichimoku")
                return False

            # Рассчитываем компоненты Ichimoku
            try:
                ichimoku_result = ta.ichimoku(
                    data['high'],
                    data['low'],
                    data['close'],
                    tenkan=self.ichimoku_conversion,
                    kijun=self.ichimoku_base,
                    senkou=self.ichimoku_span_b,
                    offset=self.ichimoku_displacement
                )
            except Exception as e:
                logger.warning(f"Ошибка расчета Ichimoku: {e}")
                return False

            # Обработка разных типов возврата
            if ichimoku_result is None:
                logger.debug("Ichimoku вернул None")
                return False

            # Проверяем тип возврата
            if isinstance(ichimoku_result, tuple):
                # Некоторые версии pandas_ta возвращают tuple
                if len(ichimoku_result) < 4:
                    logger.debug("Ichimoku tuple содержит недостаточно элементов")
                    return False

                tenkan_values, kijun_values, span_a_values, span_b_values = ichimoku_result[:4]

                # Проверяем, что это Series с данными
                if (not hasattr(tenkan_values, 'iloc') or len(tenkan_values) == 0 or
                    not hasattr(kijun_values, 'iloc') or len(kijun_values) == 0):
                    logger.debug("Ichimoku tuple содержит пустые Series")
                    return False

            elif isinstance(ichimoku_result, pd.DataFrame):
                # Стандартный возврат DataFrame
                if ichimoku_result.empty or len(ichimoku_result.columns) < 4:
                    logger.debug("Ichimoku DataFrame пуст или содержит недостаточно колонок")
                    return False

                # Извлекаем колонки по позиции (более надежно чем по имени)
                tenkan_values = ichimoku_result.iloc[:, 0]  # Tenkan-sen
                kijun_values = ichimoku_result.iloc[:, 1]  # Kijun-sen
                span_a_values = ichimoku_result.iloc[:, 2]  # Senkou Span A
                span_b_values = ichimoku_result.iloc[:, 3]  # Senkou Span B

            else:
                logger.warning(f"Неожиданный тип данных Ichimoku: {type(ichimoku_result)}")
                return False

            # Получаем текущие значения с защитой от ошибок
            try:
                current_price = float(data['close'].iloc[-1])
                current_tenkan = float(tenkan_values.iloc[-1])
                current_kijun = float(kijun_values.iloc[-1])

                # Облако (смещенное назад для текущего времени)
                if len(span_a_values) >= self.ichimoku_displacement:
                    current_span_a = float(span_a_values.iloc[-(self.ichimoku_displacement)])
                    current_span_b = float(span_b_values.iloc[-(self.ichimoku_displacement)])
                else:
                    current_span_a = float(span_a_values.iloc[-1])
                    current_span_b = float(span_b_values.iloc[-1])

            except (IndexError, ValueError, TypeError) as e:
                logger.debug(f"Ошибка извлечения значений Ichimoku: {e}")
                return False

            # Проверяем на NaN
            if any(
                pd.isna(val) for val in [current_price, current_tenkan, current_kijun, current_span_a, current_span_b]):
                logger.debug("Ichimoku содержит NaN значения")
                return False

            # Вычисляем границы облака
            cloud_top = max(current_span_a, current_span_b)
            cloud_bottom = min(current_span_a, current_span_b)

            # Условия для бычьего подтверждения
            bullish_conditions = [
                current_price > cloud_top,  # Цена над облаком
                current_tenkan > current_kijun,  # Tenkan выше Kijun
                current_price > current_tenkan,  # Цена выше Tenkan
                current_span_a > current_span_b  # Восходящее облако
            ]

            # Условия для медвежьего подтверждения
            bearish_conditions = [
                current_price < cloud_bottom,  # Цена под облаком
                current_tenkan < current_kijun,  # Tenkan ниже Kijun
                current_price < current_tenkan,  # Цена ниже Tenkan
                current_span_a < current_span_b  # Нисходящее облако
            ]

            # Подтверждение, если выполнено большинство условий (3 из 4)
            bullish_score = sum(bullish_conditions)
            bearish_score = sum(bearish_conditions)

            result = bullish_score >= 3 or bearish_score >= 3

            if result:
                logger.debug(f"Ichimoku подтверждение: bullish_score={bullish_score}, bearish_score={bearish_score}")

            return result

        except Exception as e:
            logger.error(f"Критическая ошибка в Ichimoku: {e}")
            return False

    # ДОПОЛНИТЕЛЬНО: Добавить fallback метод без Ichimoku

    def _check_ichimoku_confirmation_fallback(self, data: pd.DataFrame) -> bool:
        """
        Fallback метод без Ichimoku - использует простые скользящие средние
        """
        try:
            if len(data) < 50:
                return False

            # Используем простые EMA как замену
            ema_9 = ta.ema(data['close'], length=9)
            ema_26 = ta.ema(data['close'], length=26)
            ema_52 = ta.ema(data['close'], length=52)

            if any(x is None or x.empty for x in [ema_9, ema_26, ema_52]):
                return False

            current_price = data['close'].iloc[-1]
            current_ema_9 = ema_9.iloc[-1]
            current_ema_26 = ema_26.iloc[-1]
            current_ema_52 = ema_52.iloc[-1]

            # Бычье выравнивание
            bullish = (current_price > current_ema_9 > current_ema_26 > current_ema_52)

            # Медвежье выравнивание
            bearish = (current_price < current_ema_9 < current_ema_26 < current_ema_52)

            return bullish or bearish

        except Exception as e:
            logger.error(f"Ошибка в fallback Ichimoku: {e}")
            return False

    async def _determine_signal_direction(self, symbol: str, data: pd.DataFrame, components: SARSignalComponents) -> Optional[SignalType]:
        """Определяет направление сигнала на основе PSAR и дополнительных индикаторов"""
        try:
            if not components.psar_trigger:
                return None
            
            # Получаем PSAR для определения направления
            psar_data = ta.psar(data['high'], data['low'], data['close'],
                              af0=self.psar_start, af=self.psar_step, max_af=self.psar_max)
            
            if psar_data is None:
                return None
            
            # Получаем PSAR значения
            if isinstance(psar_data, pd.DataFrame):
                psar_col = next((col for col in psar_data.columns if 'PSAR' in col), None)
                if psar_col is None:
                    return None
                psar_values = psar_data[psar_col]
            else:
                psar_values = psar_data
                
            current_price = data['close'].iloc[-1]
            current_psar = psar_values.iloc[-1]
            
            # Определяем направление
            if current_price > current_psar:
                signal_direction = SignalType.BUY
            else:
                signal_direction = SignalType.SELL
            
            # Дополнительная проверка с RSI для подтверждения направления
            rsi = ta.rsi(data['close'], length=self.rsi_period)
            if rsi is not None and not rsi.empty:
                current_rsi = rsi.iloc[-1]
                
                # Для BUY сигналов RSI должен быть не в зоне перекупленности
                if signal_direction == SignalType.BUY and current_rsi > 80:
                    logger.debug(f"BUY сигнал отклонен для {symbol}: RSI слишком высокий ({current_rsi:.1f})")
                    return None
                
                # Для SELL сигналов RSI должен быть не в зоне перепроданности
                if signal_direction == SignalType.SELL and current_rsi < 20:
                    logger.debug(f"SELL сигнал отклонен для {symbol}: RSI слишком низкий ({current_rsi:.1f})")
                    return None
            
            return signal_direction
            
        except Exception as e:
            logger.error(f"Ошибка определения направления сигнала для {symbol}: {e}")
            return None

    async def _check_reversal_protection(self, symbol: str, data: pd.DataFrame, signal_type: SignalType) -> bool:
        """
        Защита от убыточных быстрых разворотов
        Проверяет, что потенциальная прибыль покроет минимум 3 комиссии
        """
        try:
            # Проверяем, есть ли текущая позиция по символу
            current_position = self.current_positions.get(symbol)
            if current_position is None:
                return True  # Нет позиции - можем торговать
            
            current_price = data['close'].iloc[-1]
            entry_price = current_position.get('entry_price', current_price)
            position_side = current_position.get('side')
            
            # Рассчитываем текущий P&L в процентах
            if position_side == 'BUY':
                current_pnl_pct = (current_price - entry_price) / entry_price
            else:  # SELL
                current_pnl_pct = (entry_price - current_price) / entry_price
            
            # Проверяем, покроет ли текущая прибыль минимальную защиту
            if current_pnl_pct < self.min_profit_protection:
                logger.debug(f"Разворот заблокирован для {symbol}: текущий P&L {current_pnl_pct:.4f} < минимум {self.min_profit_protection:.4f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка проверки защиты от разворотов для {symbol}: {e}")
            return True  # В случае ошибки разрешаем торговлю

    async def handle_removed_symbols(self, removed_symbols: List[str], position_manager) -> None:
        """
        Обрабатывает символы, исключенные из SAR мониторинга
        Передает их в основную обработку открытых позиций бота
        """
        try:
            for symbol in removed_symbols:
                # Проверяем, есть ли открытая позиция по символу
                if symbol in self.current_positions:
                    logger.info(f"🔄 Передача позиции {symbol} из SAR в основную обработку")

                    # Получаем данные позиции
                    position_data = self.current_positions[symbol]

                    # Добавляем метаданные о том, что позиция была от SAR стратегии
                    position_data['transferred_from'] = 'SAR_strategy'
                    position_data['transfer_reason'] = 'excluded_from_monitoring'
                    position_data['transfer_time'] = datetime.now()

                    # Передаем позицию в основную систему
                    if position_manager:
                        await position_manager.transfer_position_from_strategy(
                            symbol, position_data, 'Stop_and_Reverse'
                        )

                    # Удаляем из нашего списка
                    del self.current_positions[symbol]

                    logger.info(f"✅ Позиция {symbol} успешно передана в основную обработку")

        except Exception as e:
            logger.error(f"Ошибка обработки исключенных символов: {e}")

    async def handle_position_update(self, symbol: str, position_update: Dict) -> None:
        """
        Обрабатывает обновления позиций от основной системы
        Вызывается когда позиция SAR закрывается/обновляется
        """
        try:
            if position_update is None:
                # Позиция закрыта
                if symbol in self.current_positions:
                    closed_position = self.current_positions[symbol]

                    # Логируем результат
                    logger.info(f"📊 SAR позиция {symbol} закрыта:")
                    logger.info(
                        f"  - Время удержания: {datetime.now() - closed_position.get('updated_at', datetime.now())}")

                    # Удаляем из отслеживания
                    del self.current_positions[symbol]

                    # Обновляем статистику стратегии
                    await self._update_strategy_performance(symbol, position_update)
            else:
                # Позиция обновлена
                profit_loss = position_update.get('profit_loss', 0)
                close_price = position_update.get('close_price')
                close_timestamp = position_update.get('close_timestamp', datetime.now())
                close_reason = position_update.get('close_reason', 'unknown')  # Устанавливаем по умолчанию
                open_price = position_update.get('open_price')

                # Обновляем статистику производительности
                self._update_performance_metrics({
                    'profit_loss': profit_loss,
                    'close_price': close_price,
                    'close_timestamp': close_timestamp,
                    'close_reason': close_reason,
                    'open_price': open_price,
                    'symbol': symbol
                })

                await self.update_position_status(symbol, position_update)

        except Exception as e:
            logger.error(f"Ошибка обработки обновления позиции SAR {symbol}: {e}")

    async def _update_strategy_performance(self, symbol: str, closed_position: Dict) -> None:
        """Обновляет статистику производительности SAR стратегии с анализом компонентов"""
        try:
            # Сбор статистики винрейта SAR
            pnl_pct = closed_position.get('profit_loss', 0.0)

            component_analysis = {
                'psar_effectiveness': self._analyze_psar_component(symbol, closed_position),
                'trend_alignment': self._analyze_trend_component(symbol, closed_position),
                'momentum_strength': self._analyze_momentum_component(symbol, closed_position)
            }

            self.performance_stats['component_effectiveness'].update(component_analysis)

            # Корректировка параметров на основе результатов
            if self.performance_stats['total_trades'] >= 10:  # Минимум для анализа
                win_rate = self.performance_stats['winning_trades'] / self.performance_stats['total_trades']

                # Автоматическая корректировка параметров
                if win_rate < 0.4:  # Низкий винрейт
                    await self._adjust_strategy_parameters('conservative')
                    logger.info(f"SAR параметры скорректированы в консервативную сторону. Win rate: {win_rate:.2%}")
                elif win_rate > 0.7:  # Высокий винрейт
                    await self._adjust_strategy_parameters('aggressive')
                    logger.info(f"SAR параметры скорректированы в агрессивную сторону. Win rate: {win_rate:.2%}")

            logger.info(
                f"✅ SAR статистика обновлена для {symbol}. Total trades: {self.performance_stats['total_trades']}")

            if not hasattr(self, 'performance_stats'):
                self.performance_stats = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_pnl': 0.0,
                    'component_effectiveness': {
                        'psar_only': {'total': 0, 'wins': 0},
                        'with_divergence': {'total': 0, 'wins': 0},
                        'with_hma_rsi': {'total': 0, 'wins': 0},
                        'with_ichimoku': {'total': 0, 'wins': 0},
                        'high_score': {'total': 0, 'wins': 0}  # score >= 6
                    }
                }

            # Анализ эффективности компонентов сигнала
            components = closed_position.get('metadata', {}).get('sar_components', {})
            is_win = pnl_pct > 0

            # Анализируем какие компоненты были активны
            if components.get('rsi_divergence') or components.get('macd_divergence'):
                stats = self.performance_stats['component_effectiveness']['with_divergence']
                stats['total'] += 1
                if is_win: stats['wins'] += 1

            if components.get('hma_rsi_trend_signal'):
                stats = self.performance_stats['component_effectiveness']['with_hma_rsi']
                stats['total'] += 1
                if is_win: stats['wins'] += 1

            if components.get('ichimoku_confirmation'):
                stats = self.performance_stats['component_effectiveness']['with_ichimoku']
                stats['total'] += 1
                if is_win: stats['wins'] += 1

            if components.get('total_score', 0) >= 6:
                stats = self.performance_stats['component_effectiveness']['high_score']
                stats['total'] += 1
                if is_win: stats['wins'] += 1

            # Корректировка параметров на основе результатов
            await self._adaptive_parameter_adjustment()

        except Exception as e:
            logger.error(f"Ошибка обновления статистики SAR: {e}")

    def _analyze_psar_component(self, symbol: str, position: Dict) -> float:
        """Анализирует эффективность PSAR компонента на основе реальных данных"""
        try:
            profit_loss = position.get('profit_loss', 0)
            profit_pct = position.get('profit_pct', 0)
            metadata = position.get('metadata', {})

            # Базовая оценка на основе прибыльности
            base_score = 0.5  # Нейтральная оценка

            # Анализ компонентов сигнала SAR из метаданных
            sar_components = metadata.get('sar_components', {})
            psar_trigger = sar_components.get('psar_trigger', False)

            # Если PSAR был триггером сигнала
            if psar_trigger:
                if profit_loss > 0:
                    # Прибыльная сделка с PSAR триггером - высокая оценка
                    base_score = min(1.0, 0.7 + (profit_pct / 100) * 0.3)
                else:
                    # Убыточная сделка с PSAR триггером - низкая оценка
                    base_score = max(0.0, 0.3 + (profit_pct / 100) * 0.3)
            else:
                # PSAR не был основным триггером
                if profit_loss > 0:
                    base_score = 0.6  # Средняя оценка для прибыльных
                else:
                    base_score = 0.4  # Ниже средней для убыточных

            # Учитываем силу сигнала (если есть в метаданных)
            signal_strength = metadata.get('signal_strength', 0.5)
            volatility_regime = metadata.get('volatility_regime', 'normal')

            # Корректировка на основе режима волатильности
            if volatility_regime == 'high':
                base_score *= 0.9  # PSAR менее эффективен в высокой волатильности
            elif volatility_regime == 'low':
                base_score *= 1.1  # PSAR более эффективен в низкой волатильности

            # Обновляем статистику компонента
            if 'psar_trigger' not in self.performance_stats['component_effectiveness']:
                self.performance_stats['component_effectiveness']['psar_trigger'] = {
                    'total': 0, 'wins': 0, 'avg_profit': 0.0
                }

            comp_stats = self.performance_stats['component_effectiveness']['psar_trigger']
            comp_stats['total'] += 1
            if profit_loss > 0:
                comp_stats['wins'] += 1

            # Обновляем среднюю прибыль
            comp_stats['avg_profit'] = (
                (comp_stats['avg_profit'] * (comp_stats['total'] - 1) + profit_loss) /
                comp_stats['total']
            )

            return min(1.0, max(0.0, base_score))

        except Exception as e:
            logger.error(f"Ошибка анализа PSAR компонента для {symbol}: {e}")
            return 0.5  # Нейтральная оценка при ошибке

    def _analyze_trend_component(self, symbol: str, position: Dict) -> float:
        """Анализирует эффективность трендового компонента"""
        try:
            profit_loss = position.get('profit_loss', 0)
            profit_pct = position.get('profit_pct', 0)
            metadata = position.get('metadata', {})
            side = position.get('side', 'BUY')

            # Получаем информацию о тренде из метаданных
            sar_components = metadata.get('sar_components', {})
            macro_trend_alignment = sar_components.get('macro_trend_alignment', False)

            # Анализ рыночных данных для определения эффективности тренда
            trend_score = 0.5  # Базовая оценка

            # Если сделка была в направлении основного тренда
            if macro_trend_alignment:
                if profit_loss > 0:
                    # Прибыльная сделка по тренду - отличная оценка
                    trend_score = min(1.0, 0.8 + (abs(profit_pct) / 100) * 0.2)
                else:
                    # Убыточная сделка по тренду - возможно ложный пробой
                    trend_score = max(0.2, 0.4 + (profit_pct / 100) * 0.2)
            else:
                # Сделка против тренда
                if profit_loss > 0:
                    # Прибыльная контртрендовая сделка - хорошая оценка
                    trend_score = min(0.8, 0.6 + (abs(profit_pct) / 100) * 0.2)
                else:
                    # Убыточная контртрендовая сделка - ожидаемо
                    trend_score = max(0.1, 0.3 + (profit_pct / 100) * 0.2)

            # Дополнительные факторы из метаданных
            adx_strength = metadata.get('adx_value', 25)  # Сила тренда
            if adx_strength > 40:
                trend_score *= 1.1  # Сильный тренд увеличивает эффективность
            elif adx_strength < 20:
                trend_score *= 0.9  # Слабый тренд снижает эффективность

            # Учитываем время удержания позиции
            open_time = position.get('open_timestamp')
            close_time = position.get('close_timestamp')
            if open_time and close_time:
                try:
                    if isinstance(open_time, str):
                        open_time = datetime.fromisoformat(open_time.replace('Z', '+00:00'))
                    if isinstance(close_time, str):
                        close_time = datetime.fromisoformat(close_time.replace('Z', '+00:00'))

                    hold_duration = (close_time - open_time).total_seconds() / 3600  # часы

                    # Оптимальное время удержания для трендовых сделок 4-24 часа
                    if 4 <= hold_duration <= 24:
                        trend_score *= 1.05  # Бонус за оптимальное время
                    elif hold_duration < 1:
                        trend_score *= 0.9  # Штраф за слишком быстрое закрытие
                    elif hold_duration > 48:
                        trend_score *= 0.95  # Небольшой штраф за очень долгое удержание
                except Exception:
                    pass  # Игнорируем ошибки парсинга времени

            # Обновляем статистику трендового компонента
            if 'trend_alignment' not in self.performance_stats['component_effectiveness']:
                self.performance_stats['component_effectiveness']['trend_alignment'] = {
                    'total': 0, 'wins': 0, 'avg_profit': 0.0
                }

            comp_stats = self.performance_stats['component_effectiveness']['trend_alignment']
            comp_stats['total'] += 1
            if profit_loss > 0:
                comp_stats['wins'] += 1
            comp_stats['avg_profit'] = (
                (comp_stats['avg_profit'] * (comp_stats['total'] - 1) + profit_loss) /
                comp_stats['total']
            )

            return min(1.0, max(0.0, trend_score))

        except Exception as e:
            logger.error(f"Ошибка анализа трендового компонента для {symbol}: {e}")
            return 0.5  # Нейтральная оценка при ошибке

    def _analyze_momentum_component(self, symbol: str, position: Dict) -> float:
        """Анализирует эффективность моментум компонента"""
        try:
            profit_loss = position.get('profit_loss', 0)
            profit_pct = position.get('profit_pct', 0)
            metadata = position.get('metadata', {})

            # Получаем компоненты моментума из метаданных
            sar_components = metadata.get('sar_components', {})
            rsi_divergence = sar_components.get('rsi_divergence', False)
            macd_divergence = sar_components.get('macd_divergence', False)
            aroon_confirmation = sar_components.get('aroon_confirmation', False)

            # Базовая оценка моментума
            momentum_score = 0.5
            momentum_signals = 0

            # Анализ RSI дивергенции
            if rsi_divergence:
                momentum_signals += 1
                if profit_loss > 0:
                    momentum_score += 0.15  # RSI дивергенция и прибыль
                else:
                    momentum_score -= 0.05  # RSI дивергенция но убыток

            # Анализ MACD дивергенции
            if macd_divergence:
                momentum_signals += 1
                if profit_loss > 0:
                    momentum_score += 0.15  # MACD дивергенция и прибыль
                else:
                    momentum_score -= 0.05  # MACD дивергенция но убыток

            # Анализ подтверждения Aroon
            if aroon_confirmation:
                momentum_signals += 1
                if profit_loss > 0:
                    momentum_score += 0.1  # Aroon подтверждение и прибыль
                else:
                    momentum_score -= 0.03  # Aroon подтверждение но убыток

            # Бонус за множественные сигналы моментума
            if momentum_signals >= 2:
                momentum_score += 0.1  # Бонус за конвергенцию сигналов
            elif momentum_signals == 0:
                momentum_score -= 0.1  # Штраф за отсутствие моментума

            # Учитываем величину движения цены
            if abs(profit_pct) > 2.0:
                # Сильное движение цены
                if profit_loss > 0:
                    momentum_score += 0.1  # Бонус за поимку сильного движения
                else:
                    momentum_score -= 0.15  # Большой штраф за пропуск движения

            # Анализ скорости достижения цели
            roi_targets = metadata.get('roi_targets', {})
            if roi_targets:
                stop_loss_roi = roi_targets.get('stop_loss', {}).get('roi_pct', -2.0)
                take_profit_roi = roi_targets.get('take_profit', {}).get('roi_pct', 4.0)

                # Если достигли цели быстро - хороший моментум
                if profit_pct >= take_profit_roi * 0.8:
                    momentum_score += 0.1
                elif profit_pct <= stop_loss_roi * 1.2:
                    momentum_score -= 0.1

            # Временной анализ моментума
            signal_strength = metadata.get('signal_strength', 0.5)
            confidence = metadata.get('confidence', 0.5)

            # Корректировка на основе уверенности в сигнале
            momentum_score *= (0.8 + confidence * 0.4)  # Диапазон 0.8-1.2

            # Обновляем статистику моментум компонента
            if 'momentum_strength' not in self.performance_stats['component_effectiveness']:
                self.performance_stats['component_effectiveness']['momentum_strength'] = {
                    'total': 0, 'wins': 0, 'avg_profit': 0.0, 'signal_count': 0
                }

            comp_stats = self.performance_stats['component_effectiveness']['momentum_strength']
            comp_stats['total'] += 1
            comp_stats['signal_count'] += momentum_signals
            if profit_loss > 0:
                comp_stats['wins'] += 1
            comp_stats['avg_profit'] = (
                (comp_stats['avg_profit'] * (comp_stats['total'] - 1) + profit_loss) /
                comp_stats['total']
            )

            return min(1.0, max(0.0, momentum_score))

        except Exception as e:
            logger.error(f"Ошибка анализа моментум компонента для {symbol}: {e}")
            return 0.5  # Нейтральная оценка при ошибке

    async def _adjust_strategy_parameters(self, mode: str):
        """Корректирует параметры стратегии на основе производительности"""
        try:
            old_params = self.psar_params.copy()
            adjustment_reason = f"Performance-based adjustment: {mode}"

            if mode == 'conservative':
                # Делаем стратегию более консервативной
                self.psar_params['acceleration_factor'] = min(
                    0.015,
                    self.psar_params.get('acceleration_factor', 0.02) * 0.8
                )
                self.psar_params['sensitivity'] = max(
                    0.6,
                    self.psar_params.get('sensitivity', 0.8) * 0.9
                )
                self.psar_params['confidence_threshold'] = min(
                    0.9,
                    self.psar_params.get('confidence_threshold', 0.7) * 1.1
                )
                # Увеличиваем пороги для более строгой фильтрации
                self.chop_threshold = min(50, self.chop_threshold * 1.1)
                self.adx_threshold = min(35, self.adx_threshold * 1.1)

            elif mode == 'aggressive':
                # Делаем стратегию более агрессивной
                self.psar_params['acceleration_factor'] = min(
                    0.035,
                    self.psar_params.get('acceleration_factor', 0.02) * 1.3
                )
                self.psar_params['sensitivity'] = min(
                    1.0,
                    self.psar_params.get('sensitivity', 0.8) * 1.1
                )
                self.psar_params['confidence_threshold'] = max(
                    0.5,
                    self.psar_params.get('confidence_threshold', 0.7) * 0.9
                )
                # Снижаем пороги для более частых сигналов
                self.chop_threshold = max(30, self.chop_threshold * 0.9)
                self.adx_threshold = max(20, self.adx_threshold * 0.9)

            elif mode == 'adaptive':
                # Адаптивная корректировка на основе статистики компонентов
                component_stats = self.performance_stats['component_effectiveness']

                # Анализируем эффективность PSAR триггера
                psar_stats = component_stats.get('psar_trigger', {})
                psar_winrate = (psar_stats.get('wins', 0) / max(1, psar_stats.get('total', 1)))

                if psar_winrate > 0.7:
                    # PSAR очень эффективен - увеличиваем чувствительность
                    self.psar_params['acceleration_factor'] *= 1.1
                    self.psar_params['sensitivity'] *= 1.05
                elif psar_winrate < 0.4:
                    # PSAR неэффективен - уменьшаем чувствительность
                    self.psar_params['acceleration_factor'] *= 0.9
                    self.psar_params['sensitivity'] *= 0.95

                # Анализируем эффективность тренда
                trend_stats = component_stats.get('trend_alignment', {})
                trend_winrate = (trend_stats.get('wins', 0) / max(1, trend_stats.get('total', 1)))

                if trend_winrate > 0.7:
                    # Тренд эффективен - снижаем пороги фильтров
                    self.adx_threshold = max(20, self.adx_threshold * 0.95)
                elif trend_winrate < 0.4:
                    # Тренд неэффективен - повышаем пороги фильтров
                    self.adx_threshold = min(35, self.adx_threshold * 1.05)

                adjustment_reason = f"Adaptive adjustment based on component analysis"

            # Обновляем временные метки
            self.psar_params['last_modified'] = datetime.now()

            # Записываем изменения в историю
            parameter_change = {
                'timestamp': datetime.now().isoformat(),
                'mode': mode,
                'reason': adjustment_reason,
                'old_params': old_params,
                'new_params': self.psar_params.copy(),
                'performance_trigger': {
                    'total_trades': self.performance_stats.get('total_trades', 0),
                    'win_rate': self.performance_stats.get('win_rate', 0),
                    'profit_factor': self.performance_stats.get('profit_factor', 0)
                }
            }

            self.performance_stats['parameter_adjustments'].append(parameter_change)

            # Ограничиваем размер истории изменений (последние 50 записей)
            if len(self.performance_stats['parameter_adjustments']) > 50:
                self.performance_stats['parameter_adjustments'] = \
                    self.performance_stats['parameter_adjustments'][-50:]

            logger.info(f"SAR параметры скорректированы в режим {mode}")
            logger.info(
                f"Acceleration factor: {old_params.get('acceleration_factor', 0)} → {self.psar_params['acceleration_factor']}")
            logger.info(f"Sensitivity: {old_params.get('sensitivity', 0)} → {self.psar_params['sensitivity']}")
            logger.info(
                f"Confidence threshold: {old_params.get('confidence_threshold', 0)} → {self.psar_params['confidence_threshold']}")

        except Exception as e:
            logger.error(f"Ошибка корректировки параметров SAR: {e}")

    async def _adaptive_parameter_adjustment(self):
        """Корректирует параметры SAR на основе результатов"""
        try:
            if not hasattr(self, 'performance_stats'):
                return

            # Сохраняем текущие параметры в истории
            self.parameter_history.append({
                'timestamp': datetime.now().isoformat(),
                'acceleration': self.acceleration,
                'max_acceleration': self.max_acceleration,
                'sensitivity': self.sensitivity,
                'confidence_threshold': self.confidence_threshold,
                'stop_loss_atr_multiplier': self.stop_loss_atr_multiplier,
                'take_profit_atr_multiplier': self.take_profit_atr_multiplier,
                'total_trades': self.performance_metrics.get('total_trades', 0),
                'win_rate': self.performance_metrics.get('win_rate', 0)
            })

            # Ограничиваем размер истории
            if len(self.parameter_history) > 100:
                self.parameter_history = self.parameter_history[-50:]

            stats = self.performance_stats['component_effectiveness']

            # Если дивергенции показывают плохие результаты, корректируем их вес
            if stats['with_divergence']['total'] >= 10:
                win_rate = stats['with_divergence']['wins'] / stats['with_divergence']['total']
                if win_rate < 0.4:
                    self.divergence_score_weight = max(1, self.divergence_score_weight - 1)
                    logger.info(f"Снижен вес дивергенций до {self.divergence_score_weight}")
                elif win_rate > 0.7:
                    self.divergence_score_weight = min(3, self.divergence_score_weight + 1)
                    logger.info(f"Увеличен вес дивергенций до {self.divergence_score_weight}")

            # Адаптация параметров Stop Loss на основе волатильности
            if self.current_data is not None and 'atr' in self.current_data.columns:
                current_atr = self.current_data['atr'].iloc[-1]
                avg_atr = self.current_data['atr'].rolling(20).mean().iloc[-1]
                atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

                if atr_ratio > 1.5:  # Высокая волатильность
                    self.stop_loss_atr_multiplier = min(self.stop_loss_atr_multiplier * 1.1, 3.0)
                    logger.info(
                        f"Увеличен SL multiplier до {self.stop_loss_atr_multiplier:.2f} из-за высокой волатильности")
                elif atr_ratio < 0.7:  # Низкая волатильность
                    self.stop_loss_atr_multiplier = max(self.stop_loss_atr_multiplier * 0.95, 0.5)
                    logger.info(
                        f"Уменьшен SL multiplier до {self.stop_loss_atr_multiplier:.2f} из-за низкой волатильности")

            # Адаптация параметров Take Profit на основе трендовой силы
            if hasattr(self, 'trend_strength'):
                if self.trend_strength > 0.7:  # Сильный тренд
                    self.take_profit_atr_multiplier = min(self.take_profit_atr_multiplier * 1.05, 4.0)
                    logger.info(f"Увеличен TP multiplier до {self.take_profit_atr_multiplier:.2f} для сильного тренда")
                elif self.trend_strength < 0.3:  # Слабый тренд
                    self.take_profit_atr_multiplier = max(self.take_profit_atr_multiplier * 0.98, 1.5)
                    logger.info(f"Уменьшен TP multiplier до {self.take_profit_atr_multiplier:.2f} для слабого тренда")

            # Адаптация sensitivity на основе количества ложных сигналов
            recent_trades = self.performance_metrics.get('recent_trades', [])
            if len(recent_trades) >= 10:
                false_signals = sum(1 for trade in recent_trades[-10:]
                                    if trade['profit'] < 0)
                false_signal_rate = false_signals / 10

                if false_signal_rate > 0.6:  # Много ложных сигналов
                    self.sensitivity = max(self.sensitivity * 0.95, 0.5)
                    logger.info(f"Снижена чувствительность до {self.sensitivity:.3f} из-за ложных сигналов")
                elif false_signal_rate < 0.3:  # Мало ложных сигналов
                    self.sensitivity = min(self.sensitivity * 1.02, 0.99)
                    logger.info(f"Повышена чувствительность до {self.sensitivity:.3f}")

            # Адаптация confidence_threshold на основе рыночного режима
            if hasattr(self, 'market_regime'):
                if self.market_regime == 'trending':
                    self.confidence_threshold = max(self.confidence_threshold * 0.98, 0.6)
                elif self.market_regime == 'ranging':
                    self.confidence_threshold = min(self.confidence_threshold * 1.02, 0.85)
                elif self.market_regime == 'volatile':
                    self.confidence_threshold = min(self.confidence_threshold * 1.05, 0.9)

                logger.info(
                    f"Адаптирован порог уверенности до {self.confidence_threshold:.2f} для режима {self.market_regime}")

        except Exception as e:
            logger.error(f"Ошибка адаптивной корректировки: {e}")

    def _clear_old_cache(self):
        """Очищает устаревший кэш данных"""
        try:
            current_time = datetime.now()
            expired_keys = []

            for key, cache_entry in self.data_cache.items():
                if current_time - cache_entry['timestamp'] > self.cache_ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.data_cache[key]

            if expired_keys:
                logger.debug(f"Очищен кэш SAR для {len(expired_keys)} записей")

        except Exception as e:
            logger.error(f"Ошибка очистки кэша SAR: {e}")

    async def _get_shadow_system_score(self, symbol: str, signal_type: SignalType) -> float:
        """
        Получает оценку от Shadow Trading System
        РЕАЛЬНАЯ интеграция с существующей системой теневой торговли
        """
        try:
            if not self.use_shadow_system:
                return 0.5  # Нейтральная оценка

            # Интеграция с shadow_trading_manager через integrated_system
            # Получаем ссылку на shadow trading через data_fetcher
            if hasattr(self.data_fetcher, 'shadow_trading_manager'):
                shadow_manager = self.data_fetcher.shadow_trading_manager

                # Получаем исторические данные о производительности для символа
                recent_performance = await shadow_manager.get_symbol_performance(symbol, days=7)

                if recent_performance:
                    # Рассчитываем оценку на основе исторической производительности
                    win_rate = recent_performance.get('win_rate', 0.5)
                    avg_profit = recent_performance.get('avg_profit_pct', 0.0)
                    signal_count = recent_performance.get('signal_count', 0)

                    # Формула оценки Shadow System
                    base_score = win_rate
                    profit_bonus = min(avg_profit * 5, 0.2)  # Макс бонус 20%
                    confidence_penalty = max(0, (10 - signal_count) * 0.02)  # Штраф за мало сигналов

                    shadow_score = base_score + profit_bonus - confidence_penalty
                    return max(0.1, min(0.9, shadow_score))

            # Fallback: базовая оценка
            return 0.6

        except Exception as e:
            logger.error(f"Ошибка получения Shadow System оценки для {symbol}: {e}")
            return 0.5

    def _calculate_signal_confidence(self, components: SARSignalComponents, shadow_score: float) -> float:
        """Рассчитывает итоговую уверенность в сигнале"""
        try:
            # Базовая уверенность от количества баллов
            score_confidence = min(components.total_score / (self.min_signal_score * 1.5), 1.0)

            # Бонус за качественные подтверждения
            quality_bonus = 0.0
            if components.rsi_divergence or components.macd_divergence:
                quality_bonus += 0.1  # Дивергенция - сильный сигнал

            if components.hma_rsi_trend_signal:
                quality_bonus += 0.1  # HMA подтверждение

            if components.ichimoku_confirmation:
                quality_bonus += 0.05  # Ichimoku поддержка

            # Интеграция Shadow System
            shadow_weight = self.shadow_weight if self.use_shadow_system else 0.0

            # Итоговая формула
            confidence = (score_confidence * (1 - shadow_weight) +
                         shadow_score * shadow_weight +
                         quality_bonus)
            
            # ML модели (если включены)
            if self.use_ml_confirmation:
                # TODO: Интегрировать с ML моделями
                ml_boost = 0.0
                # await self._get_ml_confirmation_boost(symbol, signal_type)
                confidence += ml_boost * self.ml_weight
            
            return min(max(confidence, 0.1), 0.95)  # Ограничиваем в разумных пределах
            
        except Exception as e:
            logger.error(f"Ошибка расчета уверенности сигнала: {e}")
            return 0.5

    async def _get_ml_confirmation_boost(self, symbol: str, signal_type: SignalType) -> float:
        """
        Получает подтверждение от ML моделей (если включено)
        """
        try:
            if not self.use_ml_confirmation:
                return 0.0

            # Здесь будет интеграция с существующими ML моделями системы
            # Получаем доступ к ML стратегии через data_fetcher
            if (hasattr(self.data_fetcher, 'integrated_system') and
                hasattr(self.data_fetcher.integrated_system, 'strategy_manager')):

                strategy_manager = self.data_fetcher.integrated_system.strategy_manager
                ml_strategy = strategy_manager.strategies.get('Live_ML_Strategy')

                if ml_strategy and hasattr(ml_strategy, 'predict_signal_success'):
                    # Получаем предсказание успешности сигнала
                    ml_prediction = await ml_strategy.predict_signal_success(symbol, signal_type)

                    if ml_prediction:
                        confidence = ml_prediction.get('confidence', 0.5)
                        prediction = ml_prediction.get('prediction', 0.5)

                        # Конвертируем в бонус/штраф
                        if prediction > 0.6:
                            return min(0.15, (prediction - 0.5) * 0.3)  # Макс бонус 15%
                        elif prediction < 0.4:
                            return max(-0.1, (prediction - 0.5) * 0.2)  # Макс штраф 10%

            return 0.0

        except Exception as e:
            logger.error(f"Ошибка получения ML подтверждения для {symbol}: {e}")
            return 0.0

    async def update_monitored_symbols(self, data_fetcher) -> List[str]:
        """
        Обновляет список отслеживаемых символов каждый час
        Фильтрует по дневному объему торгов > 1 М USD
        """
        try:
            current_time = datetime.now()
            
            # Проверяем, нужно ли обновление
            if current_time - self.last_symbol_update < self.symbol_update_interval:
                return list(self.monitored_symbols.keys())
            
            logger.info("🔄 Обновление списка символов для SAR стратегии...")
            
            # Получаем символы с высоким объемом
            all_symbols = await data_fetcher.get_active_symbols_by_volume(

                limit=150  # Ограничиваем количество для производительности
            )
            
            new_monitored = {}
            removed_symbols = []
            logger.info(f"Получено {len(all_symbols)} символов для проверки")

            for symbol in all_symbols:
                # Получаем данные для проверки фильтров
                try:
                    data = await data_fetcher.get_historical_candles(symbol, Timeframe.FIFTEEN_MINUTES, limit=200)
                    if data is None or len(data) < 100:
                        logger.debug(f"Недостаточно данных для {symbol}: {len(data) if data is not None else 0}")
                        continue
                    
                    # Проверяем, подходит ли символ для SAR стратегии
                    can_trade, reason = await self.should_trade_symbol(symbol, data)
                    
                    if can_trade:
                        new_monitored[symbol] = {
                            'added_at': current_time,
                            'last_check': current_time,
                            'filter_reason': reason
                        }
                        
                        if symbol not in self.monitored_symbols:
                            logger.info(f"➕ Добавлен символ в SAR мониторинг: {symbol} ({reason})")
                    else:
                        logger.debug(f"Символ {symbol} не прошел фильтры: {reason}")

                except Exception as e:
                    logger.error(f"Ошибка проверки символа {symbol}: {e}")
                    continue
            
            # Определяем удаленные символы
            for symbol in self.monitored_symbols:
                if symbol not in new_monitored:
                    removed_symbols.append(symbol)
            
            # Обновляем список
            self.monitored_symbols = new_monitored
            self.last_symbol_update = current_time
            
            # Логируем изменения
            if removed_symbols:
                logger.info(f"➖ Удалены символы из SAR мониторинга: {removed_symbols}")
            
            logger.info(f"✅ SAR мониторинг обновлен: {len(self.monitored_symbols)} символов")
            
            return list(self.monitored_symbols.keys())
            
        except Exception as e:
            logger.error(f"Ошибка обновления списка символов для SAR: {e}")
            return list(self.monitored_symbols.keys())

    async def update_position_status(self, symbol: str, position_data: Dict) -> None:
        """Обновляет статус позиции для стратегии"""
        if position_data is None:
            # Позиция закрыта
            if symbol in self.current_positions:
                del self.current_positions[symbol]
                logger.debug(f"Позиция по {symbol} удалена из SAR стратегии")
        else:
            # Позиция открыта или обновлена
            self.current_positions[symbol] = {
                'entry_price': float(position_data.get('avgPrice', 0)),
                'side': position_data.get('side', 'BUY'),
                'size': float(position_data.get('size', 0)),
                'updated_at': datetime.now()
            }
            logger.debug(f"Позиция по {symbol} обновлена в SAR стратегии")

    def get_strategy_status(self) -> Dict[str, Any]:
        """Возвращает текущий статус стратегии для дашборда"""
        return {
            'strategy_name': self.strategy_name,
            'monitored_symbols_count': len(self.monitored_symbols),
            'monitored_symbols': list(self.monitored_symbols.keys()),
            'current_positions_count': len(self.current_positions),
            'current_positions': list(self.current_positions.keys()),
            'last_symbol_update': self.last_symbol_update.isoformat() if self.last_symbol_update != datetime.min else None,
            'config': {
                'min_signal_score': self.min_signal_score,
                'chop_threshold': self.chop_threshold,
                'adx_threshold': self.adx_threshold,
                'use_shadow_system': self.use_shadow_system,
                'use_ml_confirmation': self.use_ml_confirmation,
                'min_daily_volume_usd': self.min_daily_volume_usd
            }
        }

    def export_performance_report(self, filepath: str = None):
        """Экспортирует детальный отчет о производительности стратегии"""
        try:
            if not filepath:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"reports/sar_strategy_report_{timestamp}.json"

            # Создаем директорию если не существует
            import os
            dir_path = os.path.dirname(filepath)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            # Собираем полную статистику
            report_data = {
                'export_timestamp': datetime.now().isoformat(),
                'strategy_name': self.strategy_name,
                'performance_metrics': self.performance_metrics,
                'current_parameters': {
                    'acceleration': self.acceleration,
                    'max_acceleration': self.max_acceleration,
                    'sensitivity': self.sensitivity,
                    'confidence_threshold': self.confidence_threshold,
                    'stop_loss_atr_multiplier': self.stop_loss_atr_multiplier,
                    'take_profit_atr_multiplier': self.take_profit_atr_multiplier
                },
                'parameter_adaptation_history': self.parameter_history,
                'active_positions': list(self.current_positions.keys()),
                'total_symbols_monitored': len(getattr(self, 'monitored_symbols', [])),
                'last_signal_timestamp': self.last_signal_time.isoformat() if hasattr(self,
                                                                                      'last_signal_time') else None
            }

            # Сохраняем в JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Отчет SAR стратегии экспортирован в {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Ошибка экспорта отчета SAR стратегии: {e}")
            return None

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Возвращает метрики для отображения в дашборде"""
        try:
            # Рассчитываем актуальные метрики
            recent_trades_count = len(self.performance_metrics.get('recent_trades', []))

            return {
                'total_trades': self.performance_metrics.get('total_trades', 0),
                'winning_trades': self.performance_metrics.get('winning_trades', 0),
                'losing_trades': self.performance_metrics.get('losing_trades', 0),
                'win_rate': round(self.performance_metrics.get('win_rate', 0), 3),
                'recent_win_rate': round(self.performance_metrics.get('recent_win_rate', 0), 3),
                'profit_factor': round(self.performance_metrics.get('profit_factor', 0), 2),
                'total_profit': round(self.performance_metrics.get('total_profit', 0), 2),
                'total_loss': round(self.performance_metrics.get('total_loss', 0), 2),
                'avg_profit_per_trade': round(self.performance_metrics.get('avg_profit_per_trade', 0), 2),
                'recent_trades_count': recent_trades_count,
                'last_update': self.performance_metrics.get('last_update', datetime.min).isoformat(),

                # Параметры стратегии
                'current_parameters': {
                    'acceleration': round(self.acceleration, 4),
                    'max_acceleration': round(self.max_acceleration, 2),
                    'sensitivity': round(self.sensitivity, 3),
                    'confidence_threshold': round(self.confidence_threshold, 2),
                    'stop_loss_atr_multiplier': round(self.stop_loss_atr_multiplier, 2),
                    'take_profit_atr_multiplier': round(self.take_profit_atr_multiplier, 2)
                },

                # История изменений (последние 20 записей)
                'parameter_history': self.parameter_history[-20:] if len(
                    self.parameter_history) > 20 else self.parameter_history,

                # Состояние стратегии
                'monitored_symbols': len(self.monitored_symbols),
                'active_positions': len(self.current_positions),
                'market_regime': self.market_regime,
                'trend_strength': round(self.trend_strength, 2)
            }
        except Exception as e:
            logger.error(f"Ошибка получения метрик дашборда SAR: {e}")
            return {
                'total_trades': 0,
                'win_rate': 0,
                'recent_win_rate': 0,
                'profit_factor': 0,
                'total_profit': 0,
                'error': str(e)
            }

    def _update_performance_metrics(self, trade_result: Dict) -> None:
        """Обновляет метрики производительности стратегии"""
        try:
            profit_loss = trade_result.get('profit_loss', 0)

            # Обновляем базовые метрики
            self.performance_metrics['total_trades'] += 1

            if profit_loss > 0:
                self.performance_metrics['winning_trades'] += 1
                self.performance_metrics['total_profit'] += profit_loss
            else:
                self.performance_metrics['losing_trades'] += 1
                self.performance_metrics['total_loss'] += abs(profit_loss)

            # Добавляем в историю недавних сделок (максимум 20)
            recent_trades = self.performance_metrics.get('recent_trades', [])
            recent_trades.append({
                'profit': profit_loss,
                'timestamp': trade_result.get('close_timestamp', datetime.now()),
                'symbol': trade_result.get('symbol'),
                'close_reason': trade_result.get('close_reason', 'unknown')
            })

            # Ограничиваем размер истории
            if len(recent_trades) > 20:
                recent_trades = recent_trades[-20:]

            self.performance_metrics['recent_trades'] = recent_trades

            # Пересчитываем производные метрики
            total_trades = self.performance_metrics['total_trades']
            if total_trades > 0:
                self.performance_metrics['win_rate'] = self.performance_metrics['winning_trades'] / total_trades

                net_profit = self.performance_metrics['total_profit'] - self.performance_metrics['total_loss']
                self.performance_metrics['avg_profit_per_trade'] = net_profit / total_trades

                if self.performance_metrics['total_loss'] > 0:
                    self.performance_metrics['profit_factor'] = self.performance_metrics['total_profit'] / \
                                                                self.performance_metrics['total_loss']
                else:
                    self.performance_metrics['profit_factor'] = float('inf') if self.performance_metrics[
                                                                                    'total_profit'] > 0 else 0

            # Рассчитываем win rate для недавних сделок
            if len(recent_trades) > 0:
                recent_wins = sum(1 for trade in recent_trades if trade['profit'] > 0)
                self.performance_metrics['recent_win_rate'] = recent_wins / len(recent_trades)

            self.performance_metrics['last_update'] = datetime.now()

            logger.debug(
                f"Обновлены метрики SAR стратегии: WR={self.performance_metrics['win_rate']:.2f}, trades={total_trades}")

        except Exception as e:
            logger.error(f"Ошибка обновления метрик производительности SAR: {e}")

    async def check_exit_conditions(self, symbol: str, data: pd.DataFrame,
                                    position: Dict) -> Optional[TradingSignal]:
        """
        Проверяет условия выхода для открытой позиции.
        Возвращает сигнал разворота если условия соблюдены.
        """
        try:
            logger.debug(f"Проверка условий выхода SAR для {symbol}, позиция: {position.get('side')}")
            if len(data) < 20:
                logger.debug(f"Недостаточно данных для SAR анализа {symbol}: {len(data)} свечей")
                return None

            current_price = data['close'].iloc[-1]
            position_side = position.get('side')

            # Рассчитываем PSAR если не в данных
            if 'psar' not in data.columns:
                psar_df = ta.psar(data['high'], data['low'], data['close'])
                if psar_df is not None and not psar_df.empty:
                    psar_col = [col for col in psar_df.columns if
                                'PSAR' in col and 'PSARl' not in col and 'PSARs' not in col]
                    if psar_col:
                        data['psar'] = psar_df[psar_col[0]]

            if 'psar' not in data.columns:
                return None

            psar_value = data['psar'].iloc[-1]

            # Проверяем сигнал разворота
            is_reversal = False
            new_signal_type = None

            if position_side == 'BUY' and current_price < psar_value:
                # Разворот из лонга в шорт
                is_reversal = True
                new_signal_type = SignalType.SELL
            elif position_side == 'SELL' and current_price > psar_value:
                # Разворот из шорта в лонг
                is_reversal = True
                new_signal_type = SignalType.BUY

            if is_reversal:
                # Создаем сигнал разворота с ПРАВИЛЬНЫМИ параметрами
                reversal_signal = TradingSignal(
                    symbol=symbol,
                    signal_type=new_signal_type,
                    price=current_price,
                    confidence=0.7,  # Базовая уверенность для SAR разворота
                    strategy_name="SAR_Reversal",  # Используем strategy_name вместо strategy
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        'psar_value': float(psar_value),
                        'is_reversal': True,
                        'original_side': position_side,
                        'timeframe': data.attrs.get('timeframe', '1h'),
                        'volume': float(data['volume'].iloc[-1]) if 'volume' in data.columns else 0
                    }
                )
                # Добавляем флаг как атрибут после создания
                reversal_signal.is_reversal = True

                return reversal_signal

            return None

        except Exception as e:
            logger.error(f"Ошибка проверки условий выхода SAR для {symbol}: {e}")
            return None
