# strategies/ichimoku_strategy.py

import pandas as pd
import pandas_ta as ta
from typing import Optional
from datetime import datetime

from core.schemas import TradingSignal
from core.enums import SignalType
from strategies.base_strategy import BaseStrategy
from utils.logging_config import get_logger

logger = get_logger(__name__)


class IchimokuStrategy(BaseStrategy):
  """
  Стратегия, основанная на индикаторе Ichimoku Kinko Hyo.
  Генерирует сигналы на основе расположения цены относительно облака (Kumo)
  и пересечения линий Tenkan-sen и Kijun-sen.
  """

  def __init__(self, strategy_name: str = "Ichimoku_Cloud"):
    super().__init__(strategy_name)

    # strategies/ichimoku_strategy.py

  # async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
  #     """
  #     Генерирует торговый сигнал на основе анализа Ichimoku с корректной
  #     обработкой и именованием колонок.
  #     """
  #     if data.empty or len(data) < 52:
  #       return None
  #
  #     try:
  #       ichimoku_tuple = ta.ichimoku(data['high'], data['low'], data['close'], tenkan=9, kijun=26, senkou=52)
  #
  #       if ichimoku_tuple is None or not isinstance(ichimoku_tuple, tuple) or len(ichimoku_tuple) < 2:
  #         return None
  #
  #       ichimoku_df = ichimoku_tuple[0]
  #       ichimoku_span = ichimoku_tuple[1]
  #
  #       if ichimoku_df is None or ichimoku_df.empty:
  #         return None
  #
  #       # Берем только первые 4 колонки, чтобы избежать ошибки несоответствия
  #       ichimoku_main_lines = ichimoku_df.iloc[:, :4]
  #       ichimoku_main_lines.columns = ['tenkan', 'kijun', 'senkou_a', 'senkou_b']
  #
  #       # Явно создаем новую колонку с именем 'chikou'
  #       data['chikou'] = ichimoku_span
  #
  #       # Объединяем все части
  #       df = pd.concat([data, ichimoku_main_lines], axis=1)
  #       last_candle = df.iloc[-1]
  #
  #       # --- ИСПРАВЛЕНИЕ ЗДЕСЬ: ИСПОЛЬЗУЕМ НОВЫЕ ИМЕНА ---
  #       price = last_candle['close']
  #       tenkan = last_candle['tenkan']
  #       kijun = last_candle['kijun']
  #       senkou_a = last_candle['senkou_a']
  #       senkou_b = last_candle['senkou_b']
  #       chikou = last_candle['chikou']
  #       # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
  #
  #       price_26_periods_ago = df['close'].get(len(df) - 27)
  #       if pd.isna(price_26_periods_ago):
  #         return None
  #
  #       # --- Логика для сигнала на ПОКУПКУ (BUY) ---
  #       is_bullish_kumo = price > senkou_a and price > senkou_b
  #       is_bullish_cross = tenkan > kijun
  #       is_bullish_chikou = chikou > price_26_periods_ago
  #
  #       if is_bullish_kumo and is_bullish_cross and is_bullish_chikou:
  #         logger.info(f"[{self.strategy_name}] Обнаружен сигнал BUY для {symbol} по цене {price}")
  #         stop_loss = kijun
  #         take_profit = price + 3 * (price - stop_loss)
  #         return TradingSignal(
  #           signal_type=SignalType.BUY, symbol=symbol, price=price, confidence=0.85,
  #           strategy_name=self.strategy_name, timestamp=datetime.now(),
  #           stop_loss=stop_loss, take_profit=take_profit
  #         )
  #
  #       # --- Логика для сигнала на ПРОДАЖУ (SELL) ---
  #       is_bearish_kumo = price < senkou_a and price < senkou_b
  #       is_bearish_cross = tenkan < kijun
  #       is_bearish_chikou = chikou < price_26_periods_ago
  #
  #       if is_bearish_kumo and is_bearish_cross and is_bearish_chikou:
  #         logger.info(f"[{self.strategy_name}] Обнаружен сигнал SELL для {symbol} по цене {price}")
  #         stop_loss = kijun
  #         take_profit = price - 3 * abs(price - stop_loss)
  #         return TradingSignal(
  #           signal_type=SignalType.SELL, symbol=symbol, price=price, confidence=0.85,
  #           strategy_name=self.strategy_name, timestamp=datetime.now(),
  #           stop_loss=stop_loss, take_profit=take_profit
  #         )
  #
  #       return None
  #
  #     except Exception as e:
  #       logger.error(f"[{self.strategy_name}] Ошибка при генерации сигнала для {symbol}: {e}", exc_info=True)
  #       return None

  async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
    """
    ФИНАЛЬНАЯ ВЕРСИЯ: Генерирует сигнал Ichimoku с надежной обработкой
    результатов от pandas-ta.
    """
    if data.empty or len(data) < 52:
      return None

    try:
      # 1. Рассчитываем Ichimoku
      ichimoku_tuple = ta.ichimoku(data['high'], data['low'], data['close'], tenkan=9, kijun=26, senkou=52)

      # 2. Проверяем, что результат корректен
      if ichimoku_tuple is None or not isinstance(ichimoku_tuple, tuple) or len(ichimoku_tuple) < 2:
        logger.warning(f"[{self.strategy_name}] ta.ichimoku не вернул ожидаемые данные для {symbol}.")
        return None

      ichimoku_main_lines = ichimoku_tuple[0]
      ichimoku_chikou = ichimoku_tuple[1]

      if ichimoku_main_lines is None or ichimoku_main_lines.empty:
        return None

      # --- ИСПРАВЛЕНИЕ: Безопасное переименование колонок ---
      # Определяем количество колонок и их названия
      num_columns = len(ichimoku_main_lines.columns)

      # Стандартные названия для Ichimoku
      standard_names = ['tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou_span']

      # Присваиваем названия в зависимости от количества колонок
      if num_columns >= 4:
        # Используем только первые 4 колонки для основных линий
        ichimoku_main_lines = ichimoku_main_lines.iloc[:, :4].copy()
        ichimoku_main_lines.columns = ['tenkan', 'kijun', 'senkou_a', 'senkou_b']
      else:
        # Если колонок меньше 4, используем доступные
        ichimoku_main_lines.columns = standard_names[:num_columns]
        logger.warning(f"[{self.strategy_name}] Получено только {num_columns} колонок Ichimoku для {symbol}")

      # 3. Обработка Chikou отдельно
      if ichimoku_chikou is not None and not ichimoku_chikou.empty:
        if isinstance(ichimoku_chikou, pd.DataFrame):
          ichimoku_chikou = ichimoku_chikou.iloc[:, 0]  # Берем первую колонку
        chikou_series = ichimoku_chikou.rename('chikou')
      else:
        # Создаем пустую серию если Chikou недоступна
        chikou_series = pd.Series(index=data.index, name='chikou', dtype=float)

      # 4. Объединяем данные
      df = pd.concat([data, ichimoku_main_lines, chikou_series], axis=1)

      # Проверяем наличие необходимых колонок
      required_columns = ['tenkan', 'kijun', 'senkou_a', 'senkou_b']
      missing_columns = [col for col in required_columns if col not in df.columns]

      if missing_columns:
        logger.error(f"[{self.strategy_name}] Отсутствуют колонки: {missing_columns} для {symbol}")
        return None

      last_candle = df.iloc[-1]

      # Получаем значения индикаторов
      price = last_candle['close']
      tenkan = last_candle['tenkan']
      kijun = last_candle['kijun']
      senkou_a = last_candle['senkou_a']
      senkou_b = last_candle['senkou_b']

      # Безопасное получение Chikou (может отсутствовать)
      chikou = last_candle.get('chikou', None)

      # Проверяем на NaN значения
      if pd.isna(price) or pd.isna(tenkan) or pd.isna(kijun) or pd.isna(senkou_a) or pd.isna(senkou_b):
        logger.debug(f"[{self.strategy_name}] NaN значения в индикаторах для {symbol}")
        return None

      # Получаем цену 26 периодов назад для сравнения с Chikou
      price_26_periods_ago = None
      if len(df) >= 27:
        price_26_periods_ago = df['close'].iloc[-27]

      # --- Логика для сигнала BUY ---
      is_bullish_kumo = price > senkou_a and price > senkou_b
      is_bullish_cross = tenkan > kijun

      # Проверяем Chikou только если он доступен
      is_bullish_chikou = True  # По умолчанию нейтрально
      if chikou is not None and price_26_periods_ago is not None and not pd.isna(chikou):
        is_bullish_chikou = chikou > price_26_periods_ago

      if is_bullish_kumo and is_bullish_cross and is_bullish_chikou:
        confidence = 0.85 if chikou is not None else 0.75  # Снижаем уверенность без Chikou
        return TradingSignal(
          signal_type=SignalType.BUY,
          symbol=symbol,
          price=price,
          confidence=confidence,
          strategy_name=self.strategy_name,
          timestamp=datetime.now()
        )

      # --- Логика для сигнала SELL ---
      is_bearish_kumo = price < senkou_a and price < senkou_b
      is_bearish_cross = tenkan < kijun

      is_bearish_chikou = True  # По умолчанию нейтрально
      if chikou is not None and price_26_periods_ago is not None and not pd.isna(chikou):
        is_bearish_chikou = chikou < price_26_periods_ago

      if is_bearish_kumo and is_bearish_cross and is_bearish_chikou:
        confidence = 0.85 if chikou is not None else 0.75  # Снижаем уверенность без Chikou
        return TradingSignal(
          signal_type=SignalType.SELL,
          symbol=symbol,
          price=price,
          confidence=confidence,
          strategy_name=self.strategy_name,
          timestamp=datetime.now()
        )

      return None

    except Exception as e:
      logger.error(f"[{self.strategy_name}] Ошибка при генерации сигнала для {symbol}: {e}", exc_info=True)
      return None

