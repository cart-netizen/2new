# strategies/ensemble_ml_strategy.py

import pandas as pd
import numpy as np
import joblib
from typing import Optional, Dict, Any
from datetime import datetime

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

      # Логика принятия решения по вероятностям
      # Наши метки: 0=SELL, 1=HOLD, 2=BUY
      sell_prob, hold_prob, buy_prob = prediction_proba[0]
      confidence = float(np.max(prediction_proba[0]))

      # Порог уверенности из конфига
      confidence_threshold = self.settings.get('signal_confidence_threshold', 0.55)
      if confidence < confidence_threshold:
        return None  # Сигнал недостаточно уверенный

      # Определяем тип сигнала
      if buy_prob > sell_prob and buy_prob > hold_prob:
        signal_type = SignalType.BUY
      elif sell_prob > buy_prob and sell_prob > hold_prob:
        signal_type = SignalType.SELL
      else:
        return None  # Сигнал HOLD, пропускаем

      # --- КОРРЕКТНЫЙ РАСЧЕТ STOP-LOSS И TAKE-PROFIT ---
      # --- ФИНАЛЬНЫЙ БЛОК РАСЧЕТА STOP-LOSS И TAKE-PROFIT НА ОСНОВЕ ROI ---
      current_price = float(data['close'].iloc[-1])
      stop_loss, take_profit = 0.0, 0.0

      # # Получаем настройки из конфига, который мы передали в __init__
      # trade_settings = self.settings.get('trade_settings', {})
      # leverage = trade_settings.get('leverage', 10)
      # sl_roi_pct = trade_settings.get('roi_stop_loss_pct', 20.0)
      # tp_roi_pct = trade_settings.get('roi_take_profit_pct', 60.0)
      #
      # if leverage <= 0: leverage = 1  # Защита от деления на ноль

      # # Рассчитываем процентное изменение цены для достижения нужного ROI
      # sl_price_change_pct = (sl_roi_pct / 100) / leverage
      # tp_price_change_pct = (tp_roi_pct / 100) / leverage

      # Рассчитываем процентное изменение цены для достижения нужного ROI
      # sl_price_change_pct = (sl_roi_pct / 100.0) / leverage
      # tp_price_change_pct = (tp_roi_pct / 100.0) / leverage

      # Для ПОКУПКИ (BUY): Stop-Loss < Цена < Take-Profit
      # if signal_type == SignalType.BUY:
      #   stop_loss = current_price * (1 - sl_price_change_pct)
      #   take_profit = current_price * (1 + tp_price_change_pct)
      # # Для ПРОДАЖИ (SELL): Take-Profit < Цена < Stop-Loss
      # else:  # signal_type == SignalType.SELL
      #   stop_loss = current_price * (1 + sl_price_change_pct)
      #   take_profit = current_price * (1 - tp_price_change_pct)


      # --- КОНЕЦ ФИНАЛЬНОГО БЛОКА ---

      return TradingSignal(
        signal_type=signal_type,
        symbol=symbol,
        price=current_price,
        confidence=confidence,
        strategy_name=self.strategy_name,
        timestamp=datetime.now(),
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
                           strategy_name="Fallback", timestamp=datetime.now())
    except Exception as e:
      logger.error(f"Ошибка в Fallback стратегии для {symbol}: {e}")
      return None