# strategies/ensemble_ml_strategy.py

import pandas as pd
import numpy as np
import joblib
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from pandas.core.interchange.dataframe_protocol import DataFrame

from core.data_fetcher import DataFetcher
from core.enums import SignalType
from core.schemas import TradingSignal
from strategies.base_strategy import BaseStrategy
from ml.feature_engineering import feature_engineer
from utils.logging_config import get_logger

logger = get_logger(__name__)


class EnsembleMLStrategy(BaseStrategy):
  """
  ФИНАЛЬНАЯ ВЕРСИЯ: ML стратегия, которая только ИСПОЛЬЗУЕТ обученную модель.
  Работает с атрибутом self.model (в единственном числе).
  """

  def __init__(self, model_path: str, settings: Dict[str, Any], data_fetcher: DataFetcher):
    strategy_name = "Live_ML_Strategy"
    super().__init__(strategy_name=strategy_name)
    self.model_path = model_path
    self.settings = settings
    self.data_fetcher = data_fetcher
    self.model = self._load_model()  # <-- Загружает ОДНУ модель в self.model

  def _load_model(self):
    """Загружает обученную модель из файла."""
    try:
      model = joblib.load(self.model_path)
      logger.info(f"Модель успешно загружена из {self.model_path}")
      return model
    except FileNotFoundError:
      logger.warning(f"Файл модели не найден по пути: {self.model_path}. Стратегия будет использовать fallback.")
      return None
    except Exception as e:
      logger.error(f"Ошибка при загрузке модели из {self.model_path}: {e}")
      return None

  async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
    """
    ФИНАЛЬНАЯ ВЕРСИЯ: Генерирует торговый сигнал с использованием вероятностей
    и с гарантированно правильным расчетом Stop-Loss и Take-Profit.
    """
    # Простая проверка, есть ли у нас загруженная модель
    if self.model is None:
      return await self._fallback_strategy(symbol, data)

    try:
      logger.debug(f"Подготовка признаков для {symbol}...")
      # 1. Подготовка признаков
      # for_prediction=True может быть полезным, чтобы не считать метки, если они не нужны
      # features_df, _ = feature_engineer.create_features_and_labels(data, for_prediction=True)
      features_df, _ = await feature_engineer.create_multi_timeframe_features(symbol, self.data_fetcher)


      if features_df is None or features_df.empty:
        logger.warning(f"Не удалось создать признаки для {symbol}. Используем fallback.")
        return await self._fallback_strategy(symbol, data)

      latest_features = features_df.tail(1)

      logger.debug(f"Выполнение предсказания для {symbol}...")
      prediction_proba = self.model.predict_proba(latest_features)

      # Если модель вернула некорректный результат
      if prediction_proba is None or prediction_proba.shape[1] < 3:
        logger.error(f"Модель вернула некорректные вероятности для {symbol}. Используем fallback.")
        return await self._fallback_strategy(symbol, data)

      # # Логика принятия решения по вероятностям
      # # Наши метки: 0=SELL, 1=HOLD, 2=BUY
      # sell_prob, hold_prob, buy_prob = prediction_proba[0]
      # confidence = float(np.max(prediction_proba[0]))
      #
      # # Порог уверенности из конфига
      # confidence_threshold = self.settings.get('signal_confidence_threshold', 0.55)
      # if confidence < confidence_threshold:
      #   return None  # Сигнал недостаточно уверенный
      #
      # # Определяем тип сигнала
      # if buy_prob > sell_prob and buy_prob > hold_prob:
      #   signal_type = SignalType.BUY
      # elif sell_prob > buy_prob and sell_prob > hold_prob:
      #   signal_type = SignalType.SELL
      # else:
      #   return None  # Сигнал HOLD, пропускаем
      # Логика принятия решения по вероятностям (ИСПРАВЛЕННАЯ)
      # Наши метки: 0=SELL, 1=HOLD, 2=BUY
      sell_prob, hold_prob, buy_prob = prediction_proba[0]
      predicted_class = np.argmax(prediction_proba[0])
      confidence = float(np.max(prediction_proba[0]))

      logger.debug(f"Вероятности для {symbol}: SELL={sell_prob:.3f}, HOLD={hold_prob:.3f}, BUY={buy_prob:.3f}")
      logger.debug(f"Предсказанный класс: {predicted_class}, уверенность: {confidence:.3f}")

      # Порог уверенности из конфига
      base_confidence_threshold = self.settings.get('signal_confidence_threshold', 0.55)

      # Адаптивные пороги для разных классов (HOLD должен иметь более низкий порог)
      confidence_thresholds = {
        0: base_confidence_threshold,  # SELL
        1: base_confidence_threshold * 0.7,  # HOLD - более низкий порог
        2: base_confidence_threshold  # BUY
      }

      current_threshold = confidence_thresholds.get(predicted_class, base_confidence_threshold)

      if confidence < current_threshold:
        logger.debug(f"Уверенность {confidence:.3f} ниже порога {current_threshold:.3f} для класса {predicted_class}")
        return None  # Сигнал недостаточно уверенный

      # Определяем тип сигнала на основе предсказанного класса
      if predicted_class == 0:
        signal_type = SignalType.SELL
      elif predicted_class == 1:
        logger.debug(f"Модель предсказала HOLD для {symbol}, пропускаем")
        return None  # Сигнал HOLD, не торгуем
      elif predicted_class == 2:
        signal_type = SignalType.BUY
      else:
        logger.warning(f"Неожиданный predicted_class: {predicted_class} для {symbol}")
        return None

      # Дополнительная проверка: убеждаемся, что выбранный сигнал действительно доминирует
      if signal_type == SignalType.BUY and buy_prob <= hold_prob:
        logger.debug(f"BUY вероятность {buy_prob:.3f} не доминирует над HOLD {hold_prob:.3f}")
        return None
      elif signal_type == SignalType.SELL and sell_prob <= hold_prob:
        logger.debug(f"SELL вероятность {sell_prob:.3f} не доминирует над HOLD {hold_prob:.3f}")
        return None

      logger.info(f"Финальный сигнал для {symbol}: {signal_type.value} с уверенностью {confidence:.3f}")

      # --- КОРРЕКТНЫЙ РАСЧЕТ STOP-LOSS И TAKE-PROFIT ---
      # --- ФИНАЛЬНЫЙ БЛОК РАСЧЕТА STOP-LOSS И TAKE-PROFIT НА ОСНОВЕ ROI ---
      if len(data) >= 2:
        # Проверяем порядок сортировки данных
        if 'timestamp' in data.columns:
          first_ts = data['timestamp'].iloc[0]
          second_ts = data['timestamp'].iloc[1]
          is_desc_order = first_ts > second_ts
        elif hasattr(data.index, 'to_timestamp'):
          first_idx = data.index[0]
          second_idx = data.index[1]
          is_desc_order = first_idx > second_idx
        else:
          is_desc_order = False

        # Берем правильную цену
        current_price = float(data['close'].iloc[0] if is_desc_order else data['close'].iloc[-1])
        logger.debug(f"🔍 Ensemble ML: цена для сигнала = {current_price}, порядок desc = {is_desc_order}")
      else:
        current_price = float(data['close'].iloc[-1])
      stop_loss, take_profit = 0.0, 0.0


      return TradingSignal(
        signal_type=signal_type,
        symbol=symbol,
        price=current_price,
        confidence=confidence,
        strategy_name=self.strategy_name,
        timestamp=datetime.now(timezone.utc),
        # stop_loss=stop_loss,
        # take_profit=take_profit
      )

    except Exception as e:
      logger.error(f"❌ Критическая ошибка генерации сигнала для {symbol}: {e}", exc_info=True)
      return await self._fallback_strategy(symbol, data)

  async def _fallback_strategy(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
    """Резервная стратегия на основе классических индикаторов."""
    logger.info(f"FALLBACK СТРАТЕГИЯ для {symbol} активирована.")
    try:
      # Для pandas_ta импорт лучше делать внутри, чтобы не замедлять старт
      import pandas_ta as ta
      rsi = ta.rsi(data['close'], length=14)
      if rsi is None or rsi.empty or pd.isna(rsi.iloc[-1]):
        return None

      latest_rsi = rsi.iloc[-1]
      logger.info(f"FALLBACK СТРАТЕГИЯ для {symbol}: RSI = {latest_rsi:.2f}")

      if latest_rsi < 30:
        signal_type = SignalType.BUY
      elif latest_rsi > 70:
        signal_type = SignalType.SELL
      else:
        return None

      current_price = float(data['close'].iloc[-1])
      return TradingSignal(signal_type=signal_type, symbol=symbol, price=current_price, confidence=0.5,
                           strategy_name="Fallback", timestamp=datetime.now(timezone.utc))
    except Exception as e:
      logger.error(f"Ошибка в Fallback стратегии для {symbol}: {e}")
      return None