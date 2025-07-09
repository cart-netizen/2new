# core/signal_filter.py

import pandas as pd
import pandas_ta as ta
from typing import Tuple, Dict, Any
from core.correlation_manager import CorrelationManager
from core.data_fetcher import DataFetcher
from core.market_regime_detector import MarketRegimeDetector, MarketRegime
from core.schemas import TradingSignal
from core.enums import SignalType, Timeframe
from utils.logging_config import get_logger
import logging
signal_logger = logging.getLogger('SignalTrace')
logger = get_logger(__name__)


class SignalFilter:
  """
  Класс для фильтрации торговых сигналов на основе настроек из конфига.
  """

  def __init__(self, config: Dict[str, Any], data_fetcher: DataFetcher, market_regime_detector: MarketRegimeDetector, correlation_manager: CorrelationManager):
    self.config = config.get('filters', {})
    self.market_regime_detector = market_regime_detector
    self.correlation_manager = correlation_manager  # Добавляем менеджер корреляции

    # Настройки для фильтра по тренду BTC
    btc_filter_config = self.config.get('btc_trend_filter', {})
    self.btc_trend_filter_enabled = btc_filter_config.get('enabled', True)
    # Новый параметр: порог корреляции. По умолчанию 0.75
    self.correlation_threshold = btc_filter_config.get('correlation_threshold', 0.75)
    # >>> КОНЕЦ ПАТЧА (Часть 2)
    self.data_fetcher = data_fetcher
    logger.info(f"SignalFilter инициализирован с настройками: {self.config}")

  async def filter_signal(self, signal: TradingSignal, data: pd.DataFrame) -> Tuple[bool, str]:
    if not signal:
      return False, "Нет сигнала для фильтрации"

    try:
      # Добавить после try:
      current_price = signal.price

      # Проверяем, является ли символ приоритетным
      integrated_system = getattr(self, '_integrated_system', None)
      is_priority_symbol = False

      if integrated_system and hasattr(integrated_system, 'focus_list_symbols'):
        is_priority_symbol = signal.symbol in integrated_system.focus_list_symbols

      if is_priority_symbol:
        logger.info(f"🎯 {signal.symbol} в приоритетном списке - применяем адаптивные фильтры")



      # --- НОВЫЙ УМНЫЙ БЛОК: ФИЛЬТР ПО ТРЕНДУ BTC ---
      if self.config.get('use_btc_trend_filter', True) and 'BTC' not in signal.symbol:

        # Проверяем, нет ли на текущем активе аномального всплеска объема
        # Мы уже рассчитываем `volume_spike_ratio` в FeatureEngineer
        volume_spike_ratio = data.get('volume_spike_ratio', pd.Series([0])).iloc[-1]

        # Если объем вырос более чем в 2 раза по сравнению со средним, считаем это аномалией
        if volume_spike_ratio > 2.0:
          logger.warning(
            f"ФИЛЬТР для {signal.symbol}: Обнаружен аномальный всплеск объема (x{volume_spike_ratio:.1f}). Фильтр по тренду BTC временно игнорируется.")
        # else:
          # Если всплеска нет, проводим стандартную проверку по BTC
          #---------Закрыто в 08.07 для теста отсутствия фильтра по BTC-------------------------------------------
          # logger.debug(f"ФИЛЬТР для {signal.symbol}: Проверка тренда BTC...")
          # btc_data = await self.data_fetcher.get_historical_candles("BTCUSDT", Timeframe.ONE_HOUR, limit=50)
          # if not btc_data.empty:
          #   btc_ema = ta.ema(btc_data['close'], length=21)
          #   if btc_ema is not None and not btc_ema.empty:
          #     last_btc_price = btc_data['close'].iloc[-1]
          #     last_btc_ema = btc_ema.iloc[-1]
          #
          #     if signal.signal_type == SignalType.BUY and last_btc_price < last_btc_ema:
          #       return False, f"Отклонено: сигнал BUY, но BTC в нисходящем тренде"
          #     if signal.signal_type == SignalType.SELL and last_btc_price > last_btc_ema:
          #       return False, f"Отклонено: сигнал SELL, но BTC в восходящем тренде"
      # --- КОНЕЦ НОВОГО БЛОКА ---

      # --- 1. Фильтр по тренду (EMA) ---
      if self.config.get('use_trend_filter'):
        ema_period = self.config.get('ema_period', 200)
        ema_long = ta.ema(data['close'], length=ema_period)
        if ema_long is not None and not ema_long.empty:
          last_ema = ema_long.iloc[-1]
          logger.info(
            f"ФИЛЬТР для {signal.symbol}: Проверка EMA({ema_period}). Цена={current_price:.2f}, EMA={last_ema:.2f}")
          if signal.signal_type == SignalType.BUY and current_price < last_ema:
            return False, f"Цена ниже EMA({ema_period})"
          if signal.signal_type == SignalType.SELL and current_price > last_ema:
            return False, f"Цена выше EMA({ema_period})"

      # --- 2. Фильтр силы тренда (ADX) ---
      if self.config.get('use_adx_filter'):
        adx_threshold = self.config.get('adx_threshold', 20)
        adx_data = ta.adx(data['high'], data['low'], data['close'], length=14)
        if adx_data is not None and not adx_data.empty:
          last_adx = adx_data.iloc[-1, 0]
          logger.info(f"ФИЛЬТР для {signal.symbol}: Проверка ADX. ADX={last_adx:.2f}, Порог={adx_threshold}")
          if last_adx < adx_threshold:
            return False, f"Слабый тренд (ADX < {adx_threshold})"

      # --- НОВЫЙ БЛОК: Фильтр по силе и направлению тренда (Aroon) ---
      if self.config.get('use_aroon_filter', True):  # Добавим возможность отключать
        aroon_up_col = next((col for col in data.columns if 'AROONU' in col), None)
        aroon_down_col = next((col for col in data.columns if 'AROOND' in col), None)

        if aroon_up_col and aroon_down_col:
          last_aroon_up = data[aroon_up_col].iloc[-1]
          last_aroon_down = data[aroon_down_col].iloc[-1]
          logger.info(
            f"ФИЛЬТР для {signal.symbol}: Проверка Aroon. Up={last_aroon_up:.2f}, Down={last_aroon_down:.2f}")

          if signal.signal_type == SignalType.BUY and (last_aroon_up < 70 or last_aroon_down > 30):
            return False, f"Слабый бычий тренд по Aroon (Up < 70 или Down > 30)"
          if signal.signal_type == SignalType.SELL and (last_aroon_down < 70 or last_aroon_up > 30):
            return False, f"Слабый медвежий тренд по Aroon (Down < 70 или Up > 30)"

      # --- КОНЕЦ НОВОГО БЛОКА ---

      if self.btc_trend_filter_enabled and signal.symbol != 'BTCUSDT':
        # Получаем данные BTC для анализа режима
        btc_data = await self.data_fetcher.get_historical_candles('BTCUSDT', Timeframe.ONE_HOUR, limit=100)
        if not btc_data.empty:
          btc_regime = await self.market_regime_detector.detect_regime('BTCUSDT', btc_data)
          if btc_regime:
            # Используем signal.signal_type вместо signal.side
            is_sell_vs_up = signal.signal_type == SignalType.SELL and btc_regime.primary_regime in [
              MarketRegime.TREND_UP,
              MarketRegime.STRONG_TREND_UP]
            is_buy_vs_down = signal.signal_type == SignalType.BUY and btc_regime.primary_regime in [
              MarketRegime.TREND_DOWN,
              MarketRegime.STRONG_TREND_DOWN]

            # Проверяем, идет ли сигнал против тренда BTC
            if is_sell_vs_up or is_buy_vs_down:
              # Если да, то проверяем корреляцию (синхронный метод)
              correlation = self.correlation_manager.get_correlation_between(signal.symbol, 'BTCUSDT')

              # Отклоняем только если корреляция высокая
              # if correlation is not None and correlation >= self.correlation_threshold:
              #   trend_direction = "восходящем" if is_sell_vs_up else "нисходящем"
              #   reason = f"Отклонено: сигнал {signal.signal_type.value}, но BTC в {trend_direction} тренде и корреляция высока ({correlation:.2f})"
              #   logger.warning(f"ФИЛЬТР BTC: {reason}")
              #   return False, reason
              if correlation is not None and correlation >= 0.95:  # Повысили порог с 0.75
                # И даже тогда, проверяем силу сигнала
                if signal.confidence < 0.8:  # Только для слабых сигналов
                  return False, f"Слабый сигнал против тренда BTC (корреляция {correlation:.2f})"
              else:
                logger.info(
                  f"ФИЛЬТР BTC: Сигнал {signal.symbol} пропущен, т.к. идет против тренда BTC, но корреляция низкая ({correlation if correlation is not None else 'N/A'})")

      # --- 3. Фильтр по волатильности (ATR) ---
      if self.config.get('use_volatility_filter'):
        max_atr_percentage = self.config.get('max_atr_percentage', 5.0) / 100

        # Для приоритетных символов увеличиваем порог
        if is_priority_symbol:
          max_atr_percentage *= 3.0  # Утраиваем допустимую волатильность
          logger.debug(f"Увеличен порог волатильности для {signal.symbol}: {max_atr_percentage * 100:.1f}%")

        atr_data = ta.atr(data['high'], data['low'], data['close'], length=14)
        if atr_data is not None and not atr_data.empty:
          last_atr = atr_data.iloc[-1]
          atr_percent = (last_atr / current_price)
          if atr_percent > max_atr_percentage:
            if not is_priority_symbol:  # Только для обычных символов
              return False, f"Высокая волатильность (ATR={atr_percent:.2%})"
            else:
              logger.info(f"Приоритетный символ {signal.symbol}: игнорируем высокую волатильность")

      logger.info("СИГНАЛ-ФИЛЬТР: Сигнал прошел все проверки.")
      signal_logger.info(f"ФИЛЬТР: ПРОЙДЕН.")
      return True, "Все фильтры пройдены"

    except Exception as e:
      logger.error(f"Ошибка в SignalFilter: {e}", exc_info=True)
      return False, f"Исключение в фильтре: {e}"