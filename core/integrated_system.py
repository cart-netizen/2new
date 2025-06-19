import asyncio
import json
from contextlib import suppress
from datetime import datetime, timedelta, time
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import pandas_ta as ta
from core.indicators import crossover_series, crossunder_series
from ml.volatility_system import VolatilityPredictor, VolatilityPredictionSystem
import joblib
from config.config_manager import ConfigManager
from core.enums import Timeframe
from core.position_manager import PositionManager
from core.signal_filter import SignalFilter
from ml.lorentzian_classifier import LorentzianClassifier
from strategies.dual_thrust_strategy import DualThrustStrategy
from strategies.ensemble_ml_strategy import EnsembleMLStrategy
from strategies.ichimoku_strategy import IchimokuStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy
from utils.logging_config import get_logger
from config import trading_params, api_keys, settings
from core.data_fetcher import DataFetcher
from core.bybit_connector import BybitConnector
from core.strategy_manager import StrategyManager  # Будет использоваться позже
from core.risk_manager import AdvancedRiskManager # Будет использоваться позже
from core.trade_executor import TradeExecutor # Будет использоваться позже
from data.database_manager import AdvancedDatabaseManager # Будет использоваться позже
from core.enums import Timeframe, SignalType  # Для запроса свечей
from core.schemas import RiskMetrics, TradingSignal  # Для отображения баланса
from ml.model_retraining_task import ModelRetrainingManager
from data.state_manager import StateManager
import os
from ml.anomaly_detector import MarketAnomalyDetector, AnomalyType, AnomalyReport
from ml.enhanced_ml_system import EnhancedEnsembleModel, MLPrediction
import logging # <--- Добавьте импорт
from core.correlation_manager import CorrelationManager, PortfolioRiskMetrics
from core.signal_quality_analyzer import SignalQualityAnalyzer, QualityScore

signal_logger = logging.getLogger('SignalTrace') # <--- Получаем наш спец. логгер
logger = get_logger(__name__)


class IntegratedTradingSystem:
  def __init__(self):
    logger.info("Инициализация IntegratedTradingSystem...")

    # 1. Загружаем конфигурацию
    self.config_manager = ConfigManager()
    self.config = self.config_manager.load_config()

    # 2. Инициализируем основные компоненты
    self.connector = BybitConnector()
    self.db_manager = AdvancedDatabaseManager(settings.DATABASE_PATH)
    self.state_manager = StateManager()
    self.data_fetcher = DataFetcher(
      self.connector,
      settings=self.config.get('general_settings', {})  # <--- ИСПРАВЛЕНО
    )


    # 3. Передаем соответствующие части конфига в дочерние модули
    trade_settings = self.config.get('trade_settings', {})
    strategy_settings = self.config.get('strategy_settings', {})

    self.LIVE_MODEL_PATH = "ml_models/live_model.pkl"
    ml_strategy = EnsembleMLStrategy(model_path=self.LIVE_MODEL_PATH, settings=strategy_settings, data_fetcher=self.data_fetcher)

    # Инициализация новых ML компонентов
    self.anomaly_detector: Optional[MarketAnomalyDetector] = None
    self.enhanced_ml_model: Optional[EnhancedEnsembleModel] = None
    self._anomaly_check_interval = 300  # Проверка аномалий каждые 5 минут
    self._last_anomaly_check = {}

    # Загрузка детектора аномалий
    try:
      self.anomaly_detector = MarketAnomalyDetector.load("ml_models/anomaly_detector.pkl")
      logger.info("✅ Детектор аномалий успешно загружен")
    except FileNotFoundError:
      logger.warning("Файл детектора аномалий не найден. Будет использоваться эвристический режим")
      self.anomaly_detector = MarketAnomalyDetector()
    except Exception as e:
      logger.error(f"Ошибка при загрузке детектора аномалий: {e}")

    # Загрузка расширенной ML модели
    try:
      self.enhanced_ml_model = EnhancedEnsembleModel.load(
        "ml_models/enhanced_model.pkl",
        anomaly_detector=self.anomaly_detector
      )
      logger.info("✅ Расширенная ML модель успешно загружена")
    except FileNotFoundError:
      logger.warning("Файл расширенной ML модели не найден")
    except Exception as e:
      logger.error(f"Ошибка при загрузке расширенной ML модели: {e}")

    self.strategy_manager = StrategyManager()
    self.strategy_manager.add_strategy(ml_strategy)

    ichimoku_strategy = IchimokuStrategy()
    # "Регистрируем" ее в менеджере стратегий
    self.strategy_manager.add_strategy(ichimoku_strategy)

    # Создаем экземпляр Dual Thrust, передавая ему конфиг и data_fetcher
    dual_thrust_strategy = DualThrustStrategy(config=self.config, data_fetcher=self.data_fetcher)
    self.strategy_manager.add_strategy(dual_thrust_strategy)

    mean_reversion_strategy = MeanReversionStrategy()
    self.strategy_manager.add_strategy(mean_reversion_strategy)

    momentum_strategy = MomentumStrategy()
    self.strategy_manager.add_strategy(momentum_strategy)
    self.volatility_predictor: Optional[VolatilityPredictor] = None
    # --- НОВЫЙ БЛОК: ЗАГРУЗКА СИСТЕМЫ ВОЛАТИЛЬНОСТИ ---
    self.volatility_system: Optional[VolatilityPredictionSystem] = None
    try:
      self.volatility_system = joblib.load("ml_models/volatility_system.pkl")
      logger.info("✅ Система прогнозирования волатильности успешно загружена.")
    except FileNotFoundError:
      logger.warning("Файл volatility_system.pkl не найден. SL/TP будут рассчитываться по стандартной схеме.")
    # --- КОНЕЦ БЛОКА ---
    self.risk_manager = AdvancedRiskManager(
      db_manager=self.db_manager,
      settings=self.config,
      data_fetcher=self.data_fetcher,
      volatility_predictor=self.volatility_system,

    )
    self.trade_executor = TradeExecutor(
      connector=self.connector,
      db_manager=self.db_manager,
      data_fetcher=self.data_fetcher,
      settings=self.config
    )

    self.signal_filter = SignalFilter(
      settings=strategy_settings,
      data_fetcher=self.data_fetcher
    )
    self.position_manager = PositionManager(
      db_manager=self.db_manager,
      trade_executor=self.trade_executor,
      data_fetcher=self.data_fetcher,
      connector=self.connector,
      signal_filter = self.signal_filter,
      risk_manager=self.risk_manager
    )

    self.active_symbols: List[str] = []
    self.account_balance: Optional[RiskMetrics] = None
    self.is_running = False
    self._monitoring_task: Optional[asyncio.Task] = None

    # Инициализируем RetrainingManager без лишних зависимостей
    self.retraining_manager = ModelRetrainingManager(data_fetcher=self.data_fetcher)
    self._retraining_task: Optional[asyncio.Task] = None
    self._time_sync_task: Optional[asyncio.Task] = None

    # --- НОВЫЙ БЛОК: ЗАГРУЗКА ПРЕДИКТОРА ВОЛАТИЛЬНОСТИ ---

    try:
      self.volatility_predictor = joblib.load("ml_models/volatility_system.pkl")
      logger.info("Предиктор волатильности успешно загружен.")
    except FileNotFoundError:
      logger.warning("Файл предиктора волатильности не найден. Расчет SL/TP будет производиться по стандартной схеме.")
    except Exception as e:
      logger.error(f"Ошибка при загрузке предиктора волатильности: {e}")
    # --- КОНЕЦ НОВОГО БЛОКА ---

    # Инициализация корреляционного менеджера
    self.correlation_manager = CorrelationManager(self.data_fetcher)
    self._correlation_update_interval = 3600  # Обновление корреляций каждый час
    self._last_correlation_update = 0
    self._correlation_task: Optional[asyncio.Task] = None

    self.signal_quality_analyzer = SignalQualityAnalyzer(self.data_fetcher, self.db_manager)
    self.min_quality_score = 0.6  # Минимальный балл качества для исполнения

    logger.info("IntegratedTradingSystem полностью инициализирован.")

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

  async def _monitor_symbol_for_entry(self, symbol: str):
    """
    ФИНАЛЬНАЯ ВЕРСИЯ: Определяет режим рынка и использует соответствующий
    ансамбль стратегий для генерации и подтверждения сигнала.
    """
    logger.debug(f"Поиск сигнала на HTF для символа: {symbol}")
    logger.info(f"🔍 Начало стандартного мониторинга для {symbol}")

    try:
      htf_data = await self.data_fetcher.get_historical_candles(symbol, Timeframe.ONE_HOUR, limit=300)
      if htf_data.empty or len(htf_data) < 52:  # 52 нужно для Ichimoku
        return

      # --- 1. ОПРЕДЕЛЕНИЕ РЕЖИМА РЫНКА ПО ADX ---
      adx_data = ta.adx(htf_data['high'], htf_data['low'], htf_data['close'], length=14)
      last_adx = adx_data.iloc[-1, 0] if adx_data is not None and not adx_data.empty else 25

      final_signal = None

      # --- 2. ЛОГИКА ДЛЯ ТРЕНДОВОГО РЕЖИМА ---
      if last_adx > 25:
        logger.debug(f"Режим для {symbol}: ТРЕНД (ADX={last_adx:.2f}). Используем ансамбль трендовых стратегий.")
        # В сильном тренде используем ML и подтверждающие
        target_strategy_name = "Live_ML_Strategy"
        # Но можно сначала проверить на сверхсильный импульс
        impulse_signal = await self.strategy_manager.get_signal(symbol, htf_data, "Momentum_Spike")
        if impulse_signal:
          final_signal = impulse_signal
        else:

          # Получаем базовый сигнал от ML-модели
          ml_signal = await self.strategy_manager.get_signal(symbol, htf_data, "Live_ML_Strategy")
          if ml_signal and ml_signal.signal_type != SignalType.HOLD:
            # Используем Ichimoku и Dual Thrust как подтверждение
            ichimoku_signal = await self.strategy_manager.get_signal(symbol, htf_data, "Ichimoku_Cloud")
            dual_thrust_signal = await self.strategy_manager.get_signal(symbol, htf_data, "Dual_Thrust")

            if (ichimoku_signal and ichimoku_signal.signal_type == ml_signal.signal_type) and \
                (dual_thrust_signal and dual_thrust_signal.signal_type == ml_signal.signal_type):

              logger.info(f"✅✅✅ ТРОЙНОЕ ПОДТВЕРЖДЕНИЕ для {symbol}! Сигнал: {ml_signal.signal_type.value}")
              final_signal = ml_signal
              final_signal.confidence = 0.95  # Повышаем уверенность до почти максимальной
            else:
              logger.info(f"Сигнал от ML для {symbol} не был подтвержден другими стратегиями. Вход отменен.")

      # --- 3. ЛОГИКА ДЛЯ ФЛЭТОВОГО РЕЖИМА ---
      elif last_adx < 20:
        logger.debug(f"Режим для {symbol}: ФЛЭТ (ADX={last_adx:.2f}). Используем контртрендовую стратегию.")
        final_signal = await self.strategy_manager.get_signal(symbol, htf_data, "Mean_Reversion_BB")

      # --- 4. ЕСЛИ РЕЖИМ НЕОПРЕДЕЛЕННЫЙ, НИЧЕГО НЕ ДЕЛАЕМ ---
      else:
        logger.debug(f"Режим для {symbol}: НЕОПРЕДЕЛЕННЫЙ (ADX={last_adx:.2f}). Пропускаем.")
        return

      # --- 5. ЕСЛИ ПО ИТОГУ ЕСТЬ ОДОБРЕННЫЙ СИГНАЛ, ОБРАБАТЫВАЕМ ЕГО ---
      if final_signal and final_signal.signal_type != SignalType.HOLD:
        signal_logger.info(f"====== СИГНАЛ ДЛЯ {symbol} ПОЛУЧЕН ({final_signal.strategy_name}) ======")
        signal_logger.info(
          f"Тип: {final_signal.signal_type.value}, Уверенность: {final_signal.confidence:.2f}, Цена: {final_signal.price}")

        # # --- НОВЫЙ БЛОК: ЦЕНТРАЛИЗОВАННЫЙ РАСЧЕТ SL/TP НА ОСНОВЕ ROI ---
        # trade_settings = self.config.get('trade_settings', {})
        # leverage = trade_settings.get('leverage', 10)
        # sl_roi_pct = trade_settings.get('roi_stop_loss_pct', 5.0)
        # tp_roi_pct = trade_settings.get('roi_take_profit_pct', 60.0)
        # if leverage <= 0: leverage = 1
        #
        # sl_price_change_pct = (sl_roi_pct / 100.0) / leverage
        # tp_price_change_pct = (tp_roi_pct / 100.0) / leverage
        #
        # current_price = final_signal.price
        #
        # if final_signal.signal_type == SignalType.BUY:
        #   final_signal.stop_loss = current_price * (1 - sl_price_change_pct)
        #   final_signal.take_profit = current_price * (1 + tp_price_change_pct)
        # else:  # SELL
        #   final_signal.stop_loss = current_price * (1 + sl_price_change_pct)
        #   final_signal.take_profit = current_price * (1 - tp_price_change_pct)
        #
        # logger.info(
        #   f"Для сигнала {final_signal.signal_type.value} по {symbol} рассчитаны SL={final_signal.stop_loss:.4f}, TP={final_signal.take_profit:.4f}")
        # # --- КОНЕЦ НОВОГО БЛОКА ---



        risk_decision = await self.risk_manager.validate_signal(
          signal=final_signal, symbol=symbol, account_balance=self.account_balance.available_balance_usdt, market_data=htf_data
        )
        if not risk_decision.get('approved'):
          logger.info(f"СИГНАЛ для {symbol} ОТКЛОНЕН риск-менеджером. Причины: {risk_decision.get('reasons')}")
          signal_logger.warning(f"РИСК-МЕНЕДЖЕР: ОТКЛОНЕНО. Причины: {risk_decision.get('reasons')}")
          return

        # Ставим одобренный сигнал в очередь на поиск точки входа
        pending_signals = self.state_manager.get_pending_signals()
        signal_dict = final_signal.to_dict()
        signal_dict['metadata']['approved_size'] = risk_decision.get('recommended_size', 0)
        signal_dict['metadata']['signal_time'] = datetime.now().isoformat()
        pending_signals[symbol] = signal_dict
        self.state_manager.update_pending_signals(pending_signals)

        logger.info(f"СИГНАЛ HTF для {symbol} ОДОБРЕН и поставлен в очередь на поиск точки входа.")
        signal_logger.info(f"РИСК-МЕНЕДЖЕР: ОДОБРЕНО. Размер: {risk_decision.get('recommended_size'):.4f}")
        signal_logger.info(f"====== СИГНАЛ ДЛЯ {symbol} ПОСТАВЛЕН В ОЧЕРЕДЬ ======\n")

    except Exception as e:
      logger.error(f"Ошибка при поиске входа на HTF для {symbol}: {e}", exc_info=True)

  async def _monitor_symbol_for_entry_enhanced(self, symbol: str):
    """
    Расширенная версия мониторинга с использованием Enhanced ML
    """
    logger.debug(f"Расширенный поиск сигнала для символа: {symbol}")
    logger.info(f"🔍 Начало расширенного мониторинга для {symbol}")
    try:
      # 1. Получаем данные HTF
      htf_data = await self.data_fetcher.get_historical_candles(symbol, Timeframe.ONE_HOUR, limit=300)
      if htf_data.empty or len(htf_data) < 100:
        return

      # 2. Проверяем на аномалии
      anomalies = await self._check_market_anomalies(symbol, htf_data)

      # Блокируем торговлю при критических аномалиях
      critical_anomalies = [a for a in anomalies if a.severity > 0.8]
      if critical_anomalies:
        logger.warning(f"Торговля {symbol} временно заблокирована из-за критических аномалий")
        return

      # 3. Получаем внешние данные для межрыночного анализа
      external_data = {}
      if symbol != "BTCUSDT":
        btc_data = await self.data_fetcher.get_historical_candles("BTCUSDT", Timeframe.ONE_HOUR, limit=300)
        if not btc_data.empty:
          external_data['BTC'] = btc_data

      # Можно добавить индекс страха и жадности, данные о финансировании и т.д.

      # 4. Используем Enhanced ML модель если доступна
      if self.enhanced_ml_model and self.enhanced_ml_model.is_fitted:
        try:
          _, ml_prediction = self.enhanced_ml_model.predict_proba(htf_data, external_data)

          # Логируем детали предсказания
          logger.info(f"Enhanced ML предсказание для {symbol}:")
          logger.info(f"  Сигнал: {ml_prediction.signal_type.value}")
          logger.info(f"  Вероятность: {ml_prediction.probability:.3f}")
          logger.info(f"  Согласованность моделей: {ml_prediction.model_agreement:.3f}")

          if ml_prediction.risk_assessment['anomaly_detected']:
            logger.warning(f"  ⚠️ Обнаружена аномалия: {ml_prediction.risk_assessment['anomaly_type']}")

          # Корректируем уверенность на основе аномалий
          confidence_adjustment = 1.0
          if anomalies:
            max_severity = max(a.severity for a in anomalies)
            confidence_adjustment = 1.0 - (max_severity * 0.5)  # Снижаем уверенность до 50%

          adjusted_confidence = ml_prediction.confidence * confidence_adjustment

          # Создаем торговый сигнал
          if ml_prediction.signal_type != SignalType.HOLD and adjusted_confidence > 0.6:
            current_price = htf_data['close'].iloc[-1]

            trading_signal = TradingSignal(
              signal_type=ml_prediction.signal_type,
              symbol=symbol,
              price=current_price,
              confidence=adjusted_confidence,
              strategy_name="Enhanced_ML_Strategy",
              timestamp=datetime.now(),
              metadata={
                'ml_prediction': ml_prediction.__dict__,
                'anomalies': [a.to_dict() for a in anomalies],
                'feature_importance': ml_prediction.feature_importance
              }
            )

            # Продолжаем стандартную обработку сигнала
            # await self._process_trading_signal(trading_signal, symbol, htf_data)
            await self._process_trading_signal_with_correlation_and_quality(trading_signal, symbol, htf_data)

        except Exception as e:
          logger.error(f"Ошибка Enhanced ML для {symbol}: {e}")
          # Fallback на стандартные стратегии
          await self._monitor_symbol_for_entry(symbol)
      else:
        # Используем стандартный мониторинг
        await self._monitor_symbol_for_entry(symbol)

    except Exception as e:
      logger.error(f"Ошибка в расширенном мониторинге для {symbol}: {e}", exc_info=True)

  async def _process_trading_signal(self, signal: TradingSignal, symbol: str, market_data: pd.DataFrame):
      """
      Обработка торгового сигнала с учетом аномалий
      """
      logger.info(f"⚠️ Используется старый метод обработки сигналов для {symbol}")

      # Стандартная фильтрация
      is_approved, reason = await self.signal_filter.filter_signal(signal, market_data)
      if not is_approved:
        logger.info(f"Сигнал для {symbol} отклонен фильтром: {reason}")
        return

      # Проверка рисков с учетом аномалий
      await self.update_account_balance()
      if not self.account_balance or self.account_balance.available_balance_usdt <= 0:
        return

      # Корректируем размер позиции на основе обнаруженных аномалий
      position_size_multiplier = 1.0

      if 'anomalies' in signal.metadata:
        anomalies = signal.metadata['anomalies']
        if anomalies:
          # Уменьшаем размер позиции при аномалиях
          max_severity = max(a['severity'] for a in anomalies)
          position_size_multiplier = max(0.3, 1.0 - max_severity)
          logger.info(f"Размер позиции скорректирован на {position_size_multiplier:.2f} из-за аномалий")

      # Валидация сигнала риск-менеджером
      risk_decision = await self.risk_manager.validate_signal(
        signal=signal,
        symbol=symbol,
        account_balance=self.account_balance.available_balance_usdt,
        market_data=market_data
      )

      if not risk_decision.get('approved'):
        logger.info(f"Сигнал для {symbol} отклонен риск-менеджером: {risk_decision.get('reasons')}")
        return

      # Корректируем размер с учетом аномалий
      final_size = risk_decision.get('recommended_size', 0) * position_size_multiplier

      # Ставим в очередь на исполнение
      pending_signals = self.state_manager.get_pending_signals()
      signal_dict = signal.to_dict()
      signal_dict['metadata']['approved_size'] = final_size
      signal_dict['metadata']['signal_time'] = datetime.now().isoformat()
      signal_dict['metadata']['position_size_multiplier'] = position_size_multiplier

      pending_signals[symbol] = signal_dict
      self.state_manager.update_pending_signals(pending_signals)

      logger.info(f"Enhanced сигнал для {symbol} одобрен и поставлен в очередь")

  async def train_anomaly_detector(self, symbols: List[str], lookback_days: int = 45):
    """
    Обучает детектор аномалий на исторических данных
    """
    logger.info(f"Начало обучения детектора аномалий на {len(symbols)} символах...")

    if not self.anomaly_detector:
      self.anomaly_detector = MarketAnomalyDetector()

    all_data = []

    for symbol in symbols:
      try:
        # Получаем исторические данные
        data = await self.data_fetcher.get_historical_candles(
          symbol,
          Timeframe.ONE_HOUR,
          limit=24 * lookback_days
        )

        if not data.empty and len(data) > 100:
          all_data.append(data)
          logger.info(f"Загружены данные для {symbol}: {len(data)} свечей")

      except Exception as e:
        logger.error(f"Ошибка загрузки данных для {symbol}: {e}")

    if all_data:
      # Объединяем все данные
      combined_data = pd.concat(all_data, ignore_index=True)

      # Обучаем детектор
      self.anomaly_detector.fit(combined_data)

      # Сохраняем
      self.anomaly_detector.save("ml_models/anomaly_detector.pkl")

      # Выводим статистику
      stats = self.anomaly_detector.get_statistics()
      logger.info(f"Детектор аномалий обучен. Статистика: {stats}")
    else:
      logger.error("Недостаточно данных для обучения детектора аномалий")

  async def train_enhanced_ml_model(self, symbols: List[str], lookback_days: int = 60):
    """
    Обучает расширенную ML модель
    """
    logger.info(f"Начало обучения Enhanced ML модели на {len(symbols)} символах...")

    if not self.enhanced_ml_model:
      self.enhanced_ml_model = EnhancedEnsembleModel(self.anomaly_detector)

    all_features = []
    all_labels = []

    for symbol in symbols[:20]:  # Ограничиваем для демонстрации
      try:
        # Получаем данные
        data = await self.data_fetcher.get_historical_candles(
          symbol,
          Timeframe.ONE_HOUR,
          limit=24 * lookback_days
        )

        if data.empty or len(data) < 100:
          continue

        # Создаем метки (пример - можно использовать вашу логику)
        labels = self._create_ml_labels(data)

        if labels is not None and len(labels) > 100:
          all_features.append(data)
          all_labels.append(labels)
          logger.info(f"Подготовлены данные для {symbol}")

      except Exception as e:
        logger.error(f"Ошибка подготовки данных для {symbol}: {e}")

    if all_features:
      # Объединяем данные
      combined_features = pd.concat(all_features, ignore_index=True)
      combined_labels = pd.concat(all_labels, ignore_index=True)

      # Получаем внешние данные (BTC как пример)
      btc_data = await self.data_fetcher.get_historical_candles(
        "BTCUSDT",
        Timeframe.ONE_HOUR,
        limit=24 * lookback_days
      )

      external_data = {'BTC': btc_data} if not btc_data.empty else None

      # Обучаем модель
      self.enhanced_ml_model.fit(
        combined_features,
        combined_labels,
        external_data=external_data,
        optimize_features=True
      )

      # Сохраняем
      self.enhanced_ml_model.save("ml_models/enhanced_model.pkl")

      logger.info("Enhanced ML модель успешно обучена и сохранена")
    else:
      logger.error("Недостаточно данных для обучения Enhanced ML модели")

  def _create_ml_labels(self, data: pd.DataFrame) -> Optional[pd.Series]:
    """
    Создает метки для обучения ML
    """
    # Пример создания меток на основе будущих движений цены
    future_returns = data['close'].pct_change(periods=10).shift(-10)

    # Пороги для классификации
    buy_threshold = 0.02  # 2% рост
    sell_threshold = -0.02  # 2% падение

    labels = pd.Series(index=data.index, dtype=int)
    labels[future_returns > buy_threshold] = 2  # BUY
    labels[future_returns < sell_threshold] = 0  # SELL
    labels[(future_returns >= sell_threshold) & (future_returns <= buy_threshold)] = 1  # HOLD

    return labels.dropna()

  async def get_system_health_report(self) -> Dict[str, Any]:
    """
    Расширенный отчет о состоянии системы
    """
    report = {
      'timestamp': datetime.now().isoformat(),
      'components': {
        'anomaly_detector': {
          'loaded': self.anomaly_detector is not None,
          'fitted': self.anomaly_detector.is_fitted if self.anomaly_detector else False,
          'statistics': self.anomaly_detector.get_statistics() if self.anomaly_detector else None
        },
        'enhanced_ml': {
          'loaded': self.enhanced_ml_model is not None,
          'fitted': self.enhanced_ml_model.is_fitted if self.enhanced_ml_model else False
        },
        'performance': {
          'cache_stats': self.data_fetcher.get_cache_stats(),
          'api_requests': getattr(self.connector, 'request_stats', {})
        }
      }
    }

    # Добавляем статистику по аномалиям за последние 24 часа
    if self.anomaly_detector and hasattr(self.anomaly_detector, 'anomaly_history'):
      recent_anomalies = [
        a for a in self.anomaly_detector.anomaly_history
        if (datetime.now() - a.timestamp).total_seconds() < 86400
      ]

      report['anomalies_24h'] = {
        'total': len(recent_anomalies),
        'by_type': {},
        'critical': len([a for a in recent_anomalies if a.severity > 0.8])
      }

      for anomaly in recent_anomalies:
        anomaly_type = anomaly.anomaly_type.value
        report['anomalies_24h']['by_type'][anomaly_type] = \
          report['anomalies_24h']['by_type'].get(anomaly_type, 0) + 1

    return report

  # async def _monitor_symbol_for_entry(self, symbol: str):
  #   """
  #   НОВАЯ ВЕРСИЯ: Ищет сигнал на ВЫСОКОМ таймфрейме (HTF) и, если он одобрен,
  #   ставит его в очередь на ожидание точки входа на LTF.
  #   """
  #   logger.debug(f"Поиск сигнала на HTF для символа: {symbol}")
  #   try:
  #     # 1. Получаем данные HTF (1 час)
  #     htf_data = await self.data_fetcher.get_historical_candles(symbol, Timeframe.ONE_HOUR, limit=300)
  #     if htf_data.empty: return
  #
  #     # 2. Получаем и фильтруем сигнал
  #     trading_signal = await self.strategy_manager.get_signal(symbol, htf_data)
  #     if not trading_signal or trading_signal.signal_type == SignalType.HOLD:
  #       return
  #
  #     is_approved, reason = await self.signal_filter.filter_signal(trading_signal, htf_data)
  #     if not is_approved:
  #       return
  #
  #     # 3. Проверяем риски
  #     await self.update_account_balance()
  #     if not self.account_balance or self.account_balance.available_balance_usdt <= 0:
  #       return
  #
  #     risk_decision = await self.risk_manager.validate_signal(
  #       signal=trading_signal,
  #       symbol=symbol,
  #       account_balance=self.account_balance.available_balance_usdt
  #     )
  #     if not risk_decision.get('approved'):
  #       return
  #
  #     # 4. Если все проверки пройдены, НЕ ИСПОЛНЯЕМ, а ставим в ОЖИДАНИЕ
  #     pending_signals = self.state_manager.get_pending_signals()
  #
  #     # Добавляем в сигнал доп. информацию, которая понадобится для исполнения
  #     trading_signal.metadata['approved_size'] = risk_decision.get('recommended_size', 0)
  #     trading_signal.metadata['signal_time'] = datetime.now().isoformat()
  #
  #     # Явно преобразуем TradingSignal в словарь с простыми типами
  #     signal_dict = {
  #       "signal_type": trading_signal.signal_type.value,
  #       "symbol": trading_signal.symbol,
  #       "price": trading_signal.price,
  #       "confidence": trading_signal.confidence,
  #       "strategy_name": trading_signal.strategy_name,
  #       "timestamp": trading_signal.timestamp.isoformat(),
  #       "stop_loss": trading_signal.stop_loss,
  #       "take_profit": trading_signal.take_profit,
  #       "metadata": {
  #         'approved_size': risk_decision.get('recommended_size', 0),
  #         'signal_time': datetime.now().isoformat()
  #       }
  #     }
  #
  #     pending_signals[symbol] = signal_dict  # Сохраняем как словарь
  #     self.state_manager.update_pending_signals(pending_signals)
  #
  #     logger.info(f"СИГНАЛ HTF для {symbol} ОДОБРЕН и поставлен в очередь на поиск точки входа.")
  #     signal_logger.info(f"====== СИГНАЛ ДЛЯ {symbol} ОДОБРЕН И ПОСТАВЛЕН В ОЧЕРЕДЬ ======")
  #
  #   except Exception as e:
  #     logger.error(f"Ошибка при поиске входа на HTF для {symbol}: {e}", exc_info=True)

  async def initialize(self):
    """
    Инициализация системы с ГИБРИДНОЙ логикой выбора символов.
    """
    logger.info("Начало инициализации системы...")

    # Загружаем конфиг
    config_manager = ConfigManager()  # Убедитесь, что ConfigManager импортирован
    self.config = config_manager.load_config()

    mode = self.config.get('general_settings', {}).get('symbol_selection_mode', 'dynamic')
    blacklist = self.config.get('general_settings', {}).get('symbol_blacklist', [])

    if mode == 'static':
      logger.info("Выбран статический режим выбора символов.")
      self.active_symbols = self.config.get('general_settings', {}).get('static_symbol_list', [])
    else:  # Динамический режим
      logger.info("Выбран динамический режим выбора символов.")
      limit = self.config.get('general_settings', {}).get('dynamic_symbols_count', 20)
      all_symbols = await self.data_fetcher.get_active_symbols_by_volume(limit=limit)
      # Применяем черный список
      self.active_symbols = [s for s in all_symbols if s not in blacklist]

    if not self.active_symbols:
      logger.error("Не удалось выбрать ни одного активного символа для торговли. Проверьте config.json.")
      return False

    logger.info(f"Активные символы для торговли ({len(self.active_symbols)}): {self.active_symbols}")

    await self.update_account_balance()

    # Устанавливаем плечо для всех активных символов
    leverage = self.config.get('trade_settings', {}).get('leverage', 10)
    for symbol in self.active_symbols:
      # self.current_leverage.setdefault(symbol, leverage) # Эта строка не нужна
      await self.set_leverage_for_symbol(symbol, leverage)

    logger.info("Инициализация системы завершена.")
    return True

  async def _ensure_model_exists(self):
    """
    УЛУЧШЕННАЯ ВЕРСИЯ: Проверяет, загружена ли модель в саму стратегию,
    а не просто наличие файла на диске.
    """
    # Находим нашу ML-стратегию в менеджере
    ml_strategy = self.strategy_manager.strategies.get('Live_ML_Strategy')

    # Проверяем, есть ли у стратегии загруженная модель в памяти
    if ml_strategy and ml_strategy.model is not None:
      logger.info("Рабочая модель уже загружена в стратегию. Пропуск первичного обучения.")
      return True  # Модель на месте, все в порядке

    # Если модель не загружена, запускаем первичное обучение
    logger.warning("Рабочая модель не загружена в стратегию. Запуск первичного обучения...")

    limit = self.config.get('general_settings', {}).get('dynamic_symbols_count', 20)
    symbols_for_training = await self.data_fetcher.get_active_symbols_by_volume(limit=limit)

    if not symbols_for_training:
      logger.error("Не удалось получить символы для первичного обучения. Бот не может продолжить.")
      return False

    # Запускаем переобучение
    success, message = await self.retraining_manager.retrain_model(
      symbols_for_training, timeframe=Timeframe.ONE_HOUR
    )

    if not success:
      logger.error(f"Первичное обучение модели провалилось: {message}")
      return False

    # После успешного обучения ПЕРЕЗАГРУЖАЕМ модель в нашей стратегии
    logger.info("Первичное обучение модели успешно завершено. Перезагрузка модели в стратегию...")
    if ml_strategy:
      ml_strategy.model = ml_strategy._load_model()
      if not ml_strategy.model:
        logger.error("Не удалось загрузить только что обученную модель в стратегию!")
        return False

    return True

  async def update_account_balance(self):
    logger.info("Запрос баланса аккаунта...")
    balance_data = await self.connector.get_account_balance(account_type="UNIFIED", coin="USDT")

    # ИСПРАВЛЕННАЯ ЛОГИКА: Проверяем наличие ключа 'coin', и что это непустой список
    if (balance_data
        and 'coin' in balance_data
        and isinstance(balance_data.get('coin'), list)
        and len(balance_data['coin']) > 0):

      # Данные по конкретной монете (USDT) находятся внутри первого элемента списка 'coin'
      coin_data = balance_data['coin'][0]

      self.account_balance = RiskMetrics(
        # Общий баланс кошелька берем из данных по конкретной монете
        total_balance_usdt=float(coin_data.get('walletBalance', 0)),

        # Доступный баланс надежнее брать из общего поля 'totalAvailableBalance'
        available_balance_usdt=float(balance_data.get('totalAvailableBalance', 0)),

        # Нереализованный и реализованный PnL берем из данных по монете
        unrealized_pnl_total=float(coin_data.get('unrealisedPnl', 0)),
        realized_pnl_total=float(coin_data.get('cumRealisedPnl', 0))
      )
      logger.info(f"Баланс обновлен: Всего={self.account_balance.total_balance_usdt:.2f} USDT, "
                  f"Доступно={self.account_balance.available_balance_usdt:.2f} USDT, "
                  f"Нереализ. PNL={self.account_balance.unrealized_pnl_total:.2f} USDT, "
                  f"Реализ. PNL={self.account_balance.realized_pnl_total:.2f} USDT")
    else:
      logger.error(f"Не удалось получить или распарсить данные о балансе. Ответ: {balance_data}")
      self.account_balance = RiskMetrics()

  async def set_leverage_for_symbol(self, symbol: str, leverage: int) -> bool:
    """ИСПРАВЛЕНО: Обновлен для работы с новым методом connector.set_leverage"""
    logger.info(f"Попытка установить плечо {leverage}x для {symbol}")
    if not (1 <= leverage <= 100):  # Примерный диапазон, уточнить для Bybit
      logger.error(f"Некорректное значение плеча: {leverage}. Должно быть в диапазоне [1-100].")
      return False

    try:
      success = await self.connector.set_leverage(symbol, leverage, leverage)
      if success:
        logger.info(f"Кредитное плечо {leverage}x успешно установлено для {symbol}.")

        return True
      else:
        logger.error(f"Не удалось установить плечо для {symbol}.")
        return False
    except Exception as e:
      logger.error(f"Ошибка при установке плеча для {symbol}: {e}", exc_info=True)
      return False

  async def _monitoring_loop(self):
    """
    Главный цикл, управляющий всей логикой.
    """
    await self.position_manager.load_open_positions()
    while self.is_running:
      logger.info("--- Начало нового цикла мониторинга ---")
      await self.update_account_balance()
      if self.account_balance:
        self.state_manager.update_metrics(self.account_balance)

      # Управляем открытыми позициями
      await self.position_manager.manage_open_positions(self.account_balance)
      # Сверяем закрытые сделки
      await self.position_manager.reconcile_filled_orders()
      # Обновляем состояние для дашборда
      self.state_manager.update_open_positions(self.position_manager.open_positions)

      # Проверяем сигналы в ожидании
      pending_signals = self.state_manager.get_pending_signals()
      if pending_signals:
        tasks = [self._check_and_execute_pending_signal(s, d) for s, d in pending_signals.items()]
        await asyncio.gather(*tasks)

      # Ищем новые сигналы
      open_and_pending = set(self.position_manager.open_positions.keys()) | set(pending_signals.keys())
      symbols_for_new_search = [s for s in self.active_symbols if s not in open_and_pending]

      if symbols_for_new_search:
        use_enhanced = self.config.get('ml_settings', {}).get('use_enhanced_processing', True)
        if use_enhanced and self.enhanced_ml_model:
          tasks = [self._monitor_symbol_for_entry_enhanced(symbol) for symbol in symbols_for_new_search]
        else:
          tasks = [self._monitor_symbol_for_entry(symbol) for symbol in symbols_for_new_search]
        await asyncio.gather(*tasks)

      # --- НОВЫЙ БЛОК: ПРОВЕРКА КОМАНД ИЗ ДАШБОРДА ---
      command_data = self.state_manager.get_command()
      if command_data:
        command_name = command_data.get('name')
        logger.info(f"Получена новая команда из дашборда: {command_name}")

        if command_name == 'generate_report':
          if self.retraining_manager:
            self.retraining_manager.export_performance_report()

        # Очищаем команду, чтобы не выполнять ее повторно
        self.state_manager.clear_command()
      # --- КОНЕЦ НОВОГО БЛОКА ---

      interval = self.config.get('general_settings', {}).get('monitoring_interval_seconds', 30)
      await asyncio.sleep(interval)

  async def initialize_symbols_if_empty(self):
    if not self.active_symbols:
      logger.info("Список активных символов пуст, попытка повторной инициализации...")
      self.active_symbols = await self.data_fetcher.get_active_symbols_by_volume()
      if self.active_symbols:
        logger.info(f"Символы успешно реинициализированы: {self.active_symbols}")
      else:
        logger.warning("Не удалось реинициализировать символы.")

  async def start(self):
    if self.is_running:
      logger.warning("Система уже запущена.")
      return

    # ++ СИНХРОНИЗИРУЕМ ВРЕМЯ ПЕРЕД НАЧАЛОМ РАБОТЫ ++
    await self.connector.sync_time()

      # Инициализация БД
    await self.db_manager._create_tables_if_not_exist()
    # await self.state_manager.initialize_state()

    # Проверка и первичное обучение модели
    if not await self._ensure_model_exists():
      logger.critical("Не удалось создать первичную ML модель. Запуск отменен.")
      return

    if not await self.initialize():
      logger.error("Сбой инициализации системы. Запуск отменен.")
      return

    self.is_running = True
    # ++ СООБЩАЕМ, ЧТО БОТ ЗАПУЩЕН ++
    self.state_manager.set_status('running')
    logger.info("Торговая система запускается...")
    self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    # Запускаем фоновое переобучение
    self._retraining_task = self.retraining_manager.start_scheduled_retraining(
      self.active_symbols, timeframe=Timeframe.ONE_HOUR)
    self._time_sync_task = asyncio.create_task(self._time_sync_loop())
    logger.info("Торговая система и планировщик переобучения успешно запущены.")

  async def stop(self):
    if not self.is_running:
      logger.warning("Система не запущена.")
      return

    self.is_running = False
    # ++ СООБЩАЕМ, ЧТО БОТ ОСТАНОВЛЕН ++
    self.state_manager.set_status('stopped')
    logger.info("Остановка торговой системы...")

    if self._correlation_task and not self._correlation_task.done():
      self._correlation_task.cancel()
      with suppress(asyncio.CancelledError):
        await self._correlation_task

    # Отменяем все задачи мониторинга
    if self._monitoring_task:
      self._monitoring_task.cancel()
      try:
        await self._monitoring_task
      except asyncio.CancelledError:
        logger.info("Цикл мониторинга успешно отменен.")

    if self._retraining_task:
      # Вызываем новый метод stop_scheduled_retraining
      self.retraining_manager.stop_scheduled_retraining()  # <--- УБЕДИТЕСЬ, ЧТО ВЫЗОВ ВЫГЛЯДИТ ТАК
      self._retraining_task.cancel()
      with suppress(asyncio.CancelledError):
        await self._retraining_task

    if self._time_sync_task:
      self._time_sync_task.cancel()
      with suppress(asyncio.CancelledError):
        await self._time_sync_task

    # Закрываем соединения коннектора
    if self.connector:
      await self.connector.close()

    logger.info("Торговая система остановлена.")

  # Методы для взаимодействия с GUI (пока будут выводить в консоль)
  def display_balance(self):
    if self.account_balance:
      print(f"\n--- Текущий баланс ---")
      print(f"Общий баланс USDT: {self.account_balance.total_balance_usdt:.2f}")
      print(f"Доступный баланс USDT: {self.account_balance.available_balance_usdt:.2f}")
      print(f"Нереализованный PNL: {self.account_balance.unrealized_pnl_total:.2f}")
      print(f"Реализованный PNL: {self.account_balance.realized_pnl_total:.2f}")
      print(f"----------------------\n")
    else:
      print("Баланс еще не загружен.")

  def display_active_symbols(self):
    print(f"\n--- Активные торговые пары ---")
    if self.active_symbols:
      # Получаем актуальное плечо из нашего конфига
      leverage = self.config.get('trade_settings', {}).get('leverage', 'N/A')
      for i, symbol in enumerate(self.active_symbols):
        # Больше не используем self.current_leverage
        print(f"{i + 1}. {symbol} (Плечо: {leverage}x)")
    else:
      print("Нет активных торговых пар.")
    print(f"----------------------------\n")

  # Заглушки для управления символами и плечом (позже будут вызываться из GUI)
  async def add_symbol_manual(self, symbol: str):
    if symbol not in self.active_symbols:
      # TODO: Добавить проверку, существует ли такой символ на бирже
      self.active_symbols.append(symbol)
      self.current_leverage.setdefault(symbol, trading_params.DEFAULT_LEVERAGE)
      logger.info(f"Символ {symbol} добавлен вручную.")
      # await self.set_leverage_for_symbol(symbol, self.current_leverage[symbol])
    else:
      logger.info(f"Символ {symbol} уже в списке активных.")

  async def remove_symbol_manual(self, symbol: str):
    if symbol in self.active_symbols:
      self.active_symbols.remove(symbol)
      if symbol in self.current_leverage:
        del self.current_leverage[symbol]
      logger.info(f"Символ {symbol} удален из списка активных.")
    else:
      logger.warning(f"Символ {symbol} не найден в списке активных.")


  def get_risk_metrics(self, symbol: str = None):
    """Получить риск-метрики для символа"""
    try:
      metrics = RiskMetrics()

      # Получаем сделки
      if symbol:
        trades = self.get_trades_for_symbol(symbol)
      else:
        trades = self.get_all_trades(limit=1000)

      if not trades:
        return metrics

      # Основные метрики
      metrics.total_trades = len(trades)
      profitable_trades = [t for t in trades if t.get('pnl', 0) > 0]
      losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

      metrics.winning_trades = len(profitable_trades)
      metrics.losing_trades = len(losing_trades)

      if metrics.total_trades > 0:
        metrics.win_rate = metrics.winning_trades / metrics.total_trades

      # PnL метрики
      all_pnl = [t.get('pnl', 0) for t in trades]
      metrics.total_pnl = sum(all_pnl)

      if profitable_trades:
        metrics.avg_win = sum(t.get('pnl', 0) for t in profitable_trades) / len(profitable_trades)

      if losing_trades:
        metrics.avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades)

      # Profit Factor
      total_profit = sum(t.get('pnl', 0) for t in profitable_trades)
      total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))

      if total_loss > 0:
        metrics.profit_factor = total_profit / total_loss

      # Временные PnL
      metrics.daily_pnl = self._calculate_daily_pnl(trades)
      metrics.weekly_pnl = self._calculate_weekly_pnl(trades)
      metrics.monthly_pnl = self._calculate_monthly_pnl(trades)

      # Риск метрики
      metrics.max_drawdown = self._calculate_max_drawdown(all_pnl)
      metrics.sharpe_ratio = self._calculate_sharpe_ratio(all_pnl)
      metrics.volatility = self._calculate_volatility(all_pnl)

      # Дополнительные метрики
      metrics.max_consecutive_wins = self._calculate_max_consecutive_wins(trades)
      metrics.max_consecutive_losses = self._calculate_max_consecutive_losses(trades)

      if metrics.avg_loss != 0:
        metrics.risk_reward_ratio = abs(metrics.avg_win / metrics.avg_loss)

      return metrics

    except Exception as e:
      print(f"Ошибка при расчете риск-метрик: {e}")
      return RiskMetrics()

  def _calculate_daily_pnl(self, trades: list) -> float:
      """Рассчитать дневной PnL"""
      try:
        from datetime import datetime, timedelta

        today = datetime.now().date()
        daily_trades = []

        for trade in trades:
          # Попробуем извлечь дату из разных возможных полей
          trade_date = None

          if 'created_at' in trade and trade['created_at']:
            try:
              if isinstance(trade['created_at'], str):
                trade_date = datetime.strptime(trade['created_at'][:10], '%Y-%m-%d').date()
              else:
                trade_date = trade['created_at'].date()
            except:
              pass

          if trade_date and trade_date == today:
            daily_trades.append(trade)

        return sum(t.get('pnl', 0) for t in daily_trades)

      except Exception as e:
        print(f"Ошибка при расчете дневного PnL: {e}")
        return 0.0

  def _calculate_weekly_pnl(self, trades: list) -> float:
    """Рассчитать недельный PnL"""
    try:
      from datetime import datetime, timedelta

      today = datetime.now().date()
      week_ago = today - timedelta(days=7)
      weekly_trades = []

      for trade in trades:
        trade_date = None

        if 'created_at' in trade and trade['created_at']:
          try:
            if isinstance(trade['created_at'], str):
              trade_date = datetime.strptime(trade['created_at'][:10], '%Y-%m-%d').date()
            else:
              trade_date = trade['created_at'].date()
          except:
            pass

        if trade_date and week_ago <= trade_date <= today:
          weekly_trades.append(trade)

      return sum(t.get('pnl', 0) for t in weekly_trades)

    except Exception as e:
      print(f"Ошибка при расчете недельного PnL: {e}")
      return 0.0

  def _calculate_monthly_pnl(self, trades: list) -> float:
    """Рассчитать месячный PnL"""
    try:
      from datetime import datetime, timedelta

      today = datetime.now().date()
      month_ago = today - timedelta(days=30)
      monthly_trades = []

      for trade in trades:
        trade_date = None

        if 'created_at' in trade and trade['created_at']:
          try:
            if isinstance(trade['created_at'], str):
              trade_date = datetime.strptime(trade['created_at'][:10], '%Y-%m-%d').date()
            else:
              trade_date = trade['created_at'].date()
          except:
            pass

        if trade_date and month_ago <= trade_date <= today:
          monthly_trades.append(trade)

      return sum(t.get('pnl', 0) for t in monthly_trades)

    except Exception as e:
      print(f"Ошибка при расчете месячного PnL: {e}")
      return 0.0

  def _calculate_sharpe_ratio(self, pnl_series: list) -> float:
    """Рассчитать коэффициент Шарпа"""
    try:
      if len(pnl_series) < 2:
        return 0.0

      import statistics

      mean_return = statistics.mean(pnl_series)
      std_return = statistics.stdev(pnl_series)

      if std_return == 0:
        return 0.0

      return mean_return / std_return

    except Exception as e:
      print(f"Ошибка при расчете коэффициента Шарпа: {e}")
      return 0.0

  def _calculate_volatility(self, pnl_series: list) -> float:
    """Рассчитать волатильность"""
    try:
      if len(pnl_series) < 2:
        return 0.0

      import statistics
      return statistics.stdev(pnl_series)

    except Exception as e:
      print(f"Ошибка при расчете волатильности: {e}")
      return 0.0

  def _calculate_max_consecutive_wins(self, trades: list) -> int:
    """Рассчитать максимальное количество последовательных выигрышей"""
    try:
      max_wins = 0
      current_wins = 0

      for trade in trades:
        if trade.get('pnl', 0) > 0:
          current_wins += 1
          max_wins = max(max_wins, current_wins)
        else:
          current_wins = 0

      return max_wins

    except Exception as e:
      print(f"Ошибка при расчете максимальных последовательных выигрышей: {e}")
      return 0

  def _calculate_max_consecutive_losses(self, trades: list) -> int:
    """Рассчитать максимальное количество последовательных проигрышей"""
    try:
      max_losses = 0
      current_losses = 0

      for trade in trades:
        if trade.get('pnl', 0) < 0:
          current_losses += 1
          max_losses = max(max_losses, current_losses)
        else:
          current_losses = 0

      return max_losses

    except Exception as e:
      print(f"Ошибка при расчете максимальных последовательных проигрышей: {e}")
      return 0

  def _calculate_max_drawdown(self, pnl_series: list) -> float:
    """Вычислить максимальную просадку"""
    if not pnl_series:
      return 0.0

    try:
      cumulative_pnl = []
      running_total = 0

      for pnl in pnl_series:
        running_total += pnl
        cumulative_pnl.append(running_total)

      if not cumulative_pnl:
        return 0.0

      max_drawdown = 0.0
      peak = cumulative_pnl[0]

      for current_value in cumulative_pnl:
        if current_value > peak:
          peak = current_value

        if peak > 0:
          drawdown = (peak - current_value) / peak
          max_drawdown = max(max_drawdown, drawdown)

      return max_drawdown

    except Exception as e:
      print(f"Ошибка при расчете максимальной просадки: {e}")
      return 0.0

  def _calculate_drawdown(self, profits: List[float]) -> float:
    """Вычисляет текущую просадку"""
    if not profits:
      return 0

    cumulative = np.cumsum(profits)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / (running_max + 1e-8)
    return float(np.min(drawdown))

  def get_trades_for_symbol(self, symbol: str) -> List[Dict]:
    """Заглушка для получения сделок по символу"""
    # TODO: Реализовать когда будет подключена база данных
    logger.debug(f"Заглушка: запрос сделок для символа {symbol}")
    return []

  def get_all_trades(self, limit: int = 1000) -> List[Dict]:
    """Заглушка для получения всех сделок"""
    # TODO: Реализовать когда будет подключена база данных
    logger.debug(f"Заглушка: запрос всех сделок с лимитом {limit}")
    return []

  async def _time_sync_loop(self):
    """
    Фоновый цикл, который периодически ресинхронизирует время с сервером биржи.
    """
    while self.is_running:
      try:
        # Пауза в 5 мин (300 секунд) перед следующей синхронизацией
        await asyncio.sleep(300)

        logger.info("Выполнение плановой ресинхронизации времени...")
        await self.connector.sync_time()

      except asyncio.CancelledError:
        logger.info("Цикл синхронизации времени отменен.")
        break
      except Exception as e:
        logger.error(f"Ошибка в цикле синхронизации времени: {e}", exc_info=True)
        # В случае ошибки попробуем снова через 5 минут
        await asyncio.sleep(300)

  def _check_ltf_entry_trigger(self, data: pd.DataFrame, signal_type: SignalType) -> bool:
    """
    УЛУЧШЕННАЯ ВЕРСИЯ: Проверяет триггер для входа на малом таймфрейме (LTF),
    используя комплексную логику "MFI + RSI + EMA Dynamic Signals".
    """
    if data.empty or len(data) < 30:  # Нужно достаточно данных для всех индикаторов
      return False

    try:
      df = data.copy()
      # --- ШАГ 1: АГРЕССИВНАЯ ОЧИСТКА ДАННЫХ (как мы делали в FeatureEngineer) ---
      required_cols = ['open', 'high', 'low', 'close', 'volume']
      for col in required_cols:
        if col in df.columns:
          df[col] = pd.to_numeric(df[col], errors='coerce')
      df.dropna(subset=required_cols, inplace=True)
      if len(df) < 30: return False
      # --- КОНЕЦ ОЧИСТКИ ---


      # --- 1. Рассчитываем все необходимые индикаторы ---

      # Настройки, взятые из Pine Script индикатора
      mfi_length = 14
      mfi_overbought = 70
      mfi_oversold = 30
      rsi_length = 14
      rsi_buy_threshold = 45
      rsi_sell_threshold = 55
      fast_ema_length = 9
      slow_ema_length = 21
      ema_proximity_pct = 0.5

      df['mfi'] = self.calculate_mfi_manual(df['high'], df['low'], df['close'], df['volume'], length=mfi_length)
      df['rsi'] = ta.rsi(df['close'], length=rsi_length)
      df['ema_fast'] = ta.ema(df['close'], length=fast_ema_length)
      df['ema_slow'] = ta.ema(df['close'], length=slow_ema_length)

      if df.isnull().any().any():  # Если есть пропуски после расчетов
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        if df.isnull().any().any():  # Если пропуски остались
          logger.warning(f"Не удалось рассчитать все индикаторы для триггера LTF.")
          return False

      # --- 2. Определяем логические условия, как в индикаторе ---

      # Условия импульса
      bullish_momentum = df['rsi'].iloc[-1] > rsi_buy_threshold
      bearish_momentum = df['rsi'].iloc[-1] < rsi_sell_threshold

      # Условия близости EMA
      ema_diff = abs((df['ema_fast'].iloc[-1] - df['ema_slow'].iloc[-1]) / df['ema_slow'].iloc[-1]) * 100
      ema_near_crossover = ema_diff <= ema_proximity_pct

      # Условия пересечения (используем [-2], чтобы поймать самое свежее пересечение)
      ema_crossover = crossover_series(df['ema_fast'], df['ema_slow']).iloc[-2]
      ema_crossunder = crossunder_series(df['ema_fast'], df['ema_slow']).iloc[-2]
      mfi_oversold_crossover = crossover_series(df['mfi'], pd.Series(mfi_oversold, index=df.index)).iloc[-2]
      mfi_overbought_crossunder = crossunder_series(df['mfi'], pd.Series(mfi_overbought, index=df.index)).iloc[-2]

      # --- 3. Финальная логика триггера ---

      if signal_type == SignalType.BUY:
        # Вход в LONG, если (MFI вышел из перепроданности ИЛИ было пересечение EMA) И (есть бычий импульс ИЛИ EMA близки к развороту)
        if (mfi_oversold_crossover or ema_crossover) and (bullish_momentum or ema_near_crossover):
          logger.info(f"✅ ТРИГГЕР LTF для BUY сработал!")
          return True

      elif signal_type == SignalType.SELL:
        # Вход в SHORT, если (MFI вышел из перекупленности ИЛИ было пересечение EMA) И (есть медвежий импульс ИЛИ EMA близки к развороту)
        if (mfi_overbought_crossunder or ema_crossunder) and (bearish_momentum or ema_near_crossover):
          logger.info(f"✅ ТРИГГЕР LTF для SELL сработал!")
          return True


        logger.debug(
          f"Триггер LTF для {signal_type} не сработал. MFI_OB_Cross={mfi_overbought_crossunder}, MFI_OS_Cross={mfi_oversold_crossover}, EMA_Cross={ema_crossover or ema_crossunder}, MomentumOK={bullish_momentum if signal_type == 'BUY' else bearish_momentum}")
      return False

    except Exception as e:
      logger.error(f"Ошибка в триггере LTF: {e}", exc_info=True)
      return False

  async def _check_and_execute_pending_signal(self, symbol: str, signal_data: dict):
    """Проверяет триггер для сигнала в ожидании и исполняет его."""
    try:
      signal_time = datetime.fromisoformat(signal_data['metadata'].get('signal_time'))
      if datetime.now() - signal_time > timedelta(hours=2):
        logger.warning(f"Сигнал для {symbol} просрочен и будет удален из очереди.")
        pending_signals = self.state_manager.get_pending_signals()
        pending_signals.pop(symbol, None)
        self.state_manager.update_pending_signals(pending_signals)
        return

      strategy_settings = self.config.get('strategy_settings', {})
      ltf_str = strategy_settings.get('ltf_entry_timeframe', '5m')

      timeframe_map = {"1m": Timeframe.ONE_MINUTE, "5m": Timeframe.FIVE_MINUTES, "15m": Timeframe.FIFTEEN_MINUTES}
      ltf_timeframe = timeframe_map.get(ltf_str, Timeframe.FIFTEEN_MINUTES)

      logger.debug(f"Проверка триггера для {symbol} на таймфрейме {ltf_str}...")
      ltf_data = await self.data_fetcher.get_historical_candles(symbol, ltf_timeframe, limit=100)

      signal_data['signal_type'] = SignalType(signal_data['signal_type'])
      signal_data['timestamp'] = datetime.fromisoformat(signal_data['timestamp'])
      signal = TradingSignal(**signal_data)

      if self._check_ltf_entry_trigger(ltf_data, signal.signal_type):
        logger.info(f"✅ ТРИГГЕР НА LTF ДЛЯ {symbol} СРАБОТАЛ! Исполнение ордера...")
        quantity = signal.metadata.get('approved_size', 0)
        success, trade_details = await self.trade_executor.execute_trade(signal, symbol, quantity)

        if success and trade_details:
          self.position_manager.add_position_to_cache(trade_details)

        pending_signals = self.state_manager.get_pending_signals()
        pending_signals.pop(symbol, None)
        self.state_manager.update_pending_signals(pending_signals)
    except Exception as e:
      logger.error(f"Ошибка при обработке сигнала в ожидании для {symbol}: {e}", exc_info=True)

  async def initialize_with_optimization(self):
      """
      Оптимизированная инициализация системы с предзагрузкой кэшей
      """
      logger.info("Начало оптимизированной инициализации системы...")

      # 1. Инициализируем базовые компоненты
      await self.initialize()

      if not self.active_symbols:
        return

      # 2. Предзагружаем данные в кэш для активных символов
      logger.info("Предзагрузка данных в кэш...")

      # Определяем таймфреймы для предзагрузки
      preload_timeframes = [
        Timeframe.FIFTEEN_MINUTES,  # Для точек входа
        Timeframe.ONE_HOUR,  # Для основных сигналов
        Timeframe.FOUR_HOURS,  # Для подтверждения трендов
      ]

      # Предзагружаем данные параллельно
      await self.data_fetcher.preload_cache(
        symbols=self.active_symbols[:10],  # Топ-10 символов
        timeframes=preload_timeframes
      )

      # 3. Предзагружаем информацию об инструментах
      logger.info("Предзагрузка информации об инструментах...")
      instrument_tasks = [
        self.data_fetcher.get_instrument_info(symbol)
        for symbol in self.active_symbols
      ]
      await asyncio.gather(*instrument_tasks, return_exceptions=True)

      # 4. Оптимизируем параметры коннектора для активных символов
      if len(self.active_symbols) > 20:
        # Увеличиваем лимиты для большого количества символов
        self.connector.semaphore = asyncio.Semaphore(30)
        logger.info("Увеличены лимиты параллельных запросов для большого количества символов")

      # 5. Выводим статистику кэша
      cache_stats = self.data_fetcher.get_cache_stats()
      logger.info(f"Статистика кэша после предзагрузки: {cache_stats}")

      logger.info("Оптимизированная инициализация завершена")

  async def _monitoring_loop_optimized(self):
    """
    Оптимизированный мониторинг с батчингом запросов
    """
    logger.info("Запуск оптимизированного цикла мониторинга...")

    monitoring_interval = self.config.get('general_settings', {}).get('monitoring_interval_seconds', 60)
    batch_size = 5  # Обрабатываем символы батчами

    while self.is_running:
      try:
        # Обновляем баланс один раз за цикл
        await self.update_account_balance()

        # Разбиваем символы на батчи для параллельной обработки
        for i in range(0, len(self.active_symbols), batch_size):
          if not self.is_running:
            break

          batch = self.active_symbols[i:i + batch_size]

          # Параллельная обработка батча символов
          tasks = []

          # 1. Проверяем ожидающие сигналы
          for symbol in batch:
            if symbol in self.state_manager.get_pending_signals():
              tasks.append(self._check_pending_signal_for_entry(symbol))

          # 2. Мониторим открытые позиции
          for symbol in batch:
            if symbol in self.position_manager.open_positions:
              tasks.append(self.position_manager.monitor_single_position(symbol))

          # 3. Ищем новые сигналы для символов без позиций
          for symbol in batch:
            if (symbol not in self.position_manager.open_positions and
                symbol not in self.state_manager.get_pending_signals()):
              # Используем enhanced версию если модели загружены
              if self.enhanced_ml_model and self.anomaly_detector:
                tasks.append(self._monitor_symbol_for_entry_enhanced(symbol))
              else:
                tasks.append(self._monitor_symbol_for_entry(symbol))


          # Выполняем все задачи батча параллельно
          if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Логируем ошибки, если есть
            for result in results:
              if isinstance(result, Exception):
                logger.error(f"Ошибка в мониторинге: {result}")

        # Выводим статистику производительности каждые 10 циклов
        if hasattr(self, '_monitoring_cycles'):
          self._monitoring_cycles += 1
        else:
          self._monitoring_cycles = 1

        if self._monitoring_cycles % 10 == 0:
          await self._log_performance_stats()

        if self._monitoring_cycles % 20 == 0:
          await self.display_quality_statistics()

        # Ожидание перед следующим циклом
        await asyncio.sleep(monitoring_interval)

      except asyncio.CancelledError:
        logger.info("Мониторинг остановлен по запросу")
        break
      except Exception as e:
        logger.error(f"Ошибка в оптимизированном цикле мониторинга: {e}", exc_info=True)
        await asyncio.sleep(monitoring_interval)

  async def _check_pending_signal_for_entry(self, symbol: str):
    """Проверяет ожидающий сигнал на точку входа"""
    pending_signals = self.state_manager.get_pending_signals()

    if symbol not in pending_signals:
      return

    try:
      signal_data = pending_signals[symbol]

      # Проверяем таймаут сигнала (например, 30 минут)
      signal_time = datetime.fromisoformat(signal_data['metadata']['signal_time'])
      if (datetime.now() - signal_time).seconds > 1800:
        logger.info(f"Сигнал для {symbol} устарел, удаляем из очереди")
        del pending_signals[symbol]
        self.state_manager.update_pending_signals(pending_signals)
        return

      # Получаем данные LTF для поиска точки входа
      strategy_settings = self.config.get('strategy_settings', {})
      ltf_timeframe = strategy_settings.get('ltf_entry_timeframe', '5m')

      ltf_data = await self.data_fetcher.get_historical_candles(symbol, ltf_timeframe, limit=50)
      if ltf_data.empty:
        return

      # Проверяем условия входа на LTF
      entry_found = await self._check_ltf_entry_conditions(signal_data, ltf_data)

      if entry_found:
        # Исполняем сделку
        size = signal_data['metadata']['approved_size']

        # Восстанавливаем TradingSignal из словаря
        trading_signal = TradingSignal(
          signal_type=SignalType[signal_data['signal_type']],
          symbol=signal_data['symbol'],
          price=signal_data['price'],
          confidence=signal_data['confidence'],
          strategy_name=signal_data['strategy_name'],
          timestamp=datetime.fromisoformat(signal_data['timestamp']),
          stop_loss=signal_data.get('stop_loss'),
          take_profit=signal_data.get('take_profit'),
          metadata=signal_data.get('metadata', {})
        )

        success, order_details = await self.trade_executor.execute_trade(
          trading_signal, symbol, size
        )

        if success:
          logger.info(f"✅ Сделка по {symbol} успешно исполнена")
          # Удаляем из очереди
          del pending_signals[symbol]
          self.state_manager.update_pending_signals(pending_signals)

    except Exception as e:
      logger.error(f"Ошибка при проверке точки входа для {symbol}: {e}")

  async def _check_ltf_entry_conditions(self, signal_data: Dict, ltf_data: pd.DataFrame) -> bool:
    """Проверяет условия входа на младшем таймфрейме"""
    try:
      signal_type = SignalType[signal_data['signal_type']]

      # Пример простых условий входа
      last_close = ltf_data['close'].iloc[-1]
      sma_10 = ltf_data['close'].rolling(10).mean().iloc[-1]

      if signal_type == SignalType.BUY:
        # Для покупки: цена выше SMA10
        return last_close > sma_10
      else:  # SELL
        # Для продажи: цена ниже SMA10
        return last_close < sma_10

    except Exception as e:
      logger.error(f"Ошибка при проверке LTF условий: {e}")
      return False


  async def _log_performance_stats(self):
    """
    Выводит статистику производительности системы
    """
    # Статистика кэша DataFetcher
    cache_stats = self.data_fetcher.get_cache_stats()
    logger.info(f"📊 Статистика кэша DataFetcher: Hit rate: {cache_stats['hit_rate']:.2%}, "
                f"Hits: {cache_stats['cache_hits']}, Misses: {cache_stats['cache_misses']}")

    # Статистика запросов Bybit
    if hasattr(self.connector, 'request_stats'):
      total_requests = sum(self.connector.request_stats.values())
      logger.info(f"📊 Всего API запросов: {total_requests}")

      # Топ-5 endpoint'ов по количеству запросов
      top_endpoints = sorted(
        self.connector.request_stats.items(),
        key=lambda x: x[1],
        reverse=True
      )[:5]

      if top_endpoints:
        logger.info("📊 Топ-5 API endpoints:")
        for endpoint, count in top_endpoints:
          logger.info(f"  - {endpoint}: {count} запросов")

  async def cleanup_caches(self):
    """
    Периодическая очистка кэшей для освобождения памяти
    """
    while self.is_running:
      try:
        await asyncio.sleep(3600)  # Каждый час

        logger.info("Запуск очистки кэшей...")

        # Очищаем устаревшие данные в DataFetcher
        self.data_fetcher._clean_expired_cache()

        # Очищаем кэш базы данных для старых данных
        self.db_manager.clear_cache()

        # Собираем мусор
        import gc
        gc.collect()

        logger.info("Очистка кэшей завершена")

      except asyncio.CancelledError:
        break
      except Exception as e:
        logger.error(f"Ошибка при очистке кэшей: {e}")

  async def _periodic_retraining(self):
      """Периодическое переобучение моделей"""
      while self.is_running:
        try:
          # Ждем 24 часа перед переобучением
          await asyncio.sleep(86400)

          logger.info("Запуск периодического переобучения моделей...")

          # Запускаем переобучение в фоне
          if self.retraining_manager:
            asyncio.create_task(
              self.retraining_manager.check_and_retrain_if_needed(
                self.active_symbols[:10]  # Топ 10 символов
              )
            )

        except asyncio.CancelledError:
          break
        except Exception as e:
          logger.error(f"Ошибка при периодическом переобучении: {e}")

  async def _periodic_time_sync(self):
    """Периодическая синхронизация времени"""
    while self.is_running:
      try:
        await asyncio.sleep(3600)  # Каждый час
        await self.connector.sync_time()
        logger.debug("Выполнена периодическая синхронизация времени")
      except asyncio.CancelledError:
        break
      except Exception as e:
        logger.error(f"Ошибка при синхронизации времени: {e}")

  async def start_optimized(self):
    """
    Оптимизированный запуск системы
    """
    try:
      # Инициализация с оптимизацией
      await self.initialize_with_optimization()

      if not self.active_symbols:
        logger.error("Нет активных символов для торговли")
        return

      # Синхронизация времени
      await self.connector.sync_time()

      # Установка плеча для всех символов параллельно
      leverage = self.config.get('trade_settings', {}).get('leverage', 10)
      logger.info(f"Установка плеча {leverage} для {len(self.active_symbols)} символов...")

      successful_leverages = 0
      for i, symbol in enumerate(self.active_symbols):
        try:
          result = await self.connector.set_leverage(symbol, leverage, leverage)
          if result:
            successful_leverages += 1

          # Задержка между запросами для избежания rate limit
          if i < len(self.active_symbols) - 1:
            await asyncio.sleep(0.2)  # 200мс между запросами

        except Exception as e:
          logger.warning(f"Не удалось установить плечо для {symbol}: {e}")

      logger.info(f"Плечо установлено для {successful_leverages}/{len(self.active_symbols)} символов")

      # Загрузка открытых позиций
      await self.position_manager.load_open_positions()

      # Запуск фоновых задач
      self.is_running = True

      # Оптимизированный мониторинг
      self._monitoring_task = asyncio.create_task(self._monitoring_loop_optimized())

      # Периодическое переобучение моделей
      self._retraining_task = asyncio.create_task(self._periodic_retraining())

      # Периодическая синхронизация времени
      self._time_sync_task = asyncio.create_task(self._periodic_time_sync())

      # Периодическая очистка кэшей
      self._cache_cleanup_task = asyncio.create_task(self.cleanup_caches())

      self._correlation_task = asyncio.create_task(self._update_portfolio_correlations())

      # Обновление статуса
      self.state_manager.set_status('running')

      logger.info("✅ Торговая система успешно запущена в оптимизированном режиме")

    except Exception as e:
      logger.critical(f"Критическая ошибка при запуске системы: {e}", exc_info=True)
      self.is_running = False
      raise

  async def _check_market_anomalies(self, symbol: str, data: pd.DataFrame) -> List[AnomalyReport]:
    """
    Проверяет рыночные аномалии для символа
    """
    # Проверяем, нужно ли выполнять проверку
    current_time = time.time()
    last_check = self._last_anomaly_check.get(symbol, 0)

    if current_time - last_check < self._anomaly_check_interval:
      return []

    self._last_anomaly_check[symbol] = current_time

    if not self.anomaly_detector:
      return []

    try:
      anomalies = self.anomaly_detector.detect_anomalies(data, symbol)

      if anomalies:
        logger.warning(f"🚨 Обнаружены аномалии для {symbol}:")
        for anomaly in anomalies:
          logger.warning(f"  - {anomaly.anomaly_type.value}: {anomaly.description}")
          logger.warning(f"    Серьезность: {anomaly.severity:.2f}, Рекомендация: {anomaly.recommended_action}")

          # Отправляем критические аномалии в телеграм (если подключен)
          if anomaly.severity > 0.8:
            signal_logger.critical(
              f"🚨 КРИТИЧЕСКАЯ АНОМАЛИЯ {symbol}: {anomaly.anomaly_type.value}\n"
              f"{anomaly.description}\n"
              f"Действие: {anomaly.recommended_action}"
            )

      return anomalies

    except Exception as e:
      logger.error(f"Ошибка при проверке аномалий для {symbol}: {e}")
      return []

  async def display_ml_statistics(self):
    """Выводит статистику ML моделей"""
    if self.anomaly_detector:
      stats = self.anomaly_detector.get_statistics()
      logger.info(f"📊 Статистика детектора аномалий:")
      logger.info(f"  Проверок: {stats['total_checks']}")
      logger.info(f"  Обнаружено аномалий: {stats['anomalies_detected']}")
      logger.info(f"  По типам: {stats['by_type']}")

    if self.enhanced_ml_model and hasattr(self.enhanced_ml_model, 'performance_history'):
      logger.info(f"📊 Статистика Enhanced ML:")
      logger.info(f"  Обучена: {self.enhanced_ml_model.is_fitted}")
      logger.info(
        f"  Признаков: {len(self.enhanced_ml_model.selected_features) if self.enhanced_ml_model.selected_features else 0}")

  async def _update_portfolio_correlations(self):
      """Периодическое обновление корреляций портфеля"""
      while self.is_running:
        try:
          # Ждем перед первым обновлением
          await asyncio.sleep(300)  # 5 минут после старта

          while self.is_running:
            logger.info("Обновление корреляций портфеля...")

            # Получаем все активные символы (позиции + мониторимые)
            active_symbols = list(self.position_manager.open_positions.keys())
            monitored_symbols = self.active_symbols[:20]  # Топ 20

            all_symbols = list(set(active_symbols + monitored_symbols))

            if len(all_symbols) >= 2:
              # Анализируем корреляции
              correlation_report = await self.correlation_manager.analyze_portfolio_correlation(
                symbols=all_symbols,
                timeframe=Timeframe.ONE_HOUR,
                lookback_days=30
              )

              if correlation_report:
                # Логируем важную информацию
                risk_metrics = correlation_report.get('risk_metrics')
                if risk_metrics:
                  logger.info(f"📊 Метрики риска портфеля:")
                  logger.info(f"  Волатильность портфеля: {risk_metrics.portfolio_volatility:.4f}")
                  logger.info(f"  Коэффициент диверсификации: {risk_metrics.diversification_ratio:.2f}")
                  logger.info(f"  Эффективное кол-во активов: {risk_metrics.effective_assets:.1f}")
                  logger.info(f"  Макс. корреляция: {risk_metrics.max_correlation:.2f}")

                # Проверяем рекомендации
                recommendations = correlation_report.get('recommendations', {})
                warnings = recommendations.get('warnings', [])

                for warning in warnings:
                  logger.warning(f"⚠️ Корреляция: {warning}")
                  signal_logger.warning(f"КОРРЕЛЯЦИЯ: {warning}")

                # Проверяем высокие корреляции
                high_correlations = correlation_report.get('high_correlations', [])
                for corr_data in high_correlations[:3]:  # Топ 3
                  logger.warning(
                    f"🔗 Высокая корреляция: {corr_data['symbol1']}-{corr_data['symbol2']} "
                    f"= {corr_data['correlation']:.2f}"
                  )

            # Ждем до следующего обновления
            await asyncio.sleep(self._correlation_update_interval)

        except asyncio.CancelledError:
          break
        except Exception as e:
          logger.error(f"Ошибка при обновлении корреляций: {e}")
          await asyncio.sleep(300)  # Retry через 5 минут

  async def _process_trading_signal_with_correlation_and_quality(self, signal: TradingSignal, symbol: str,
                                                     market_data: pd.DataFrame):
    """
    Обработка торгового сигнала с учетом аномалий, корреляций и качества
    """
    # 1. Оценка качества сигнала
    logger.info(f"Оценка качества сигнала для {symbol}...")
    signal_logger.info(f"КАЧЕСТВО: Начата оценка сигнала {symbol}")

    # Загружаем дополнительные таймфреймы для анализа
    additional_timeframes = {}
    for tf in [Timeframe.FIFTEEN_MINUTES, Timeframe.FOUR_HOURS]:
      try:
        tf_data = await self.data_fetcher.get_historical_candles(symbol, tf, limit=100)
        if not tf_data.empty:
          additional_timeframes[tf] = tf_data
      except Exception as e:
        logger.debug(f"Не удалось загрузить {tf} для {symbol}: {e}")

    quality_metrics = await self.signal_quality_analyzer.rate_signal_quality(
      signal, market_data, additional_timeframes
    )

    # Логируем результаты оценки
    logger.info(
      f"Качество сигнала {symbol}: {quality_metrics.overall_score:.2f} ({quality_metrics.quality_category.value})")
    signal_logger.info(
      f"КАЧЕСТВО: Оценка {quality_metrics.overall_score:.2f} - {quality_metrics.quality_category.value}")

    if quality_metrics.strengths:
      logger.info(f"Сильные стороны: {', '.join(quality_metrics.strengths[:3])}")
    if quality_metrics.weaknesses:
      logger.warning(f"Слабые стороны: {', '.join(quality_metrics.weaknesses[:3])}")
    if quality_metrics.recommendations:
      for rec in quality_metrics.recommendations[:2]:
        signal_logger.info(f"РЕКОМЕНДАЦИЯ: {rec}")

    # Проверяем минимальное качество
    if quality_metrics.overall_score < self.min_quality_score:
      logger.warning(
        f"Сигнал {symbol} отклонен из-за низкого качества: "
        f"{quality_metrics.overall_score:.2f} < {self.min_quality_score}"
      )
      signal_logger.warning(f"КАЧЕСТВО: Сигнал отклонен - низкий балл {quality_metrics.overall_score:.2f}")
      return

    # 2. Стандартная фильтрация
    is_approved, reason = await self.signal_filter.filter_signal(signal, market_data)
    if not is_approved:
      logger.info(f"Сигнал для {symbol} отклонен фильтром: {reason}")
      return

    # 3. Проверка корреляций
    open_symbols = list(self.position_manager.open_positions.keys())
    if open_symbols:
      should_block, block_reason = self.correlation_manager.should_block_signal_due_to_correlation(
        symbol, open_symbols
      )
      if should_block:
        logger.warning(f"Сигнал для {symbol} заблокирован: {block_reason}")
        signal_logger.warning(f"КОРРЕЛЯЦИЯ: Сигнал {symbol} отклонен - {block_reason}")
        return

    # 4. Проверка рисков
    await self.update_account_balance()
    if not self.account_balance or self.account_balance.available_balance_usdt <= 0:
      return

    # 5. Валидация риск-менеджером
    risk_decision = await self.risk_manager.validate_signal(
      signal=signal,
      symbol=symbol,
      account_balance=self.account_balance.available_balance_usdt,
      market_data=market_data
    )

    if not risk_decision.get('approved'):
      logger.info(f"Сигнал для {symbol} отклонен риск-менеджером: {risk_decision.get('reasons')}")
      return

    # 6. Корректировка размера позиции на основе качества
    base_size = risk_decision.get('recommended_size', 0)

    # Масштабируем размер в зависимости от качества
    quality_multiplier = 1.0
    if quality_metrics.quality_category == QualityScore.EXCELLENT:
      quality_multiplier = 1.2  # Увеличиваем на 20% для отличных сигналов
    elif quality_metrics.quality_category == QualityScore.GOOD:
      quality_multiplier = 1.0  # Стандартный размер
    elif quality_metrics.quality_category == QualityScore.FAIR:
      quality_multiplier = 0.7  # Уменьшаем на 30%
    else:
      quality_multiplier = 0.5  # Минимальный размер для слабых сигналов

    quality_adjusted_size = base_size * quality_multiplier

    # 7. Корректировка с учетом корреляций
    signals_dict = {symbol: {'size': quality_adjusted_size}}
    current_positions = {
      sym: pos.get('quantity', 0)
      for sym, pos in self.position_manager.open_positions.items()
    }

    adjusted_sizes = await self.correlation_manager.adjust_position_sizes_by_correlation(
      signals_dict, current_positions
    )

    final_size = adjusted_sizes.get(symbol, quality_adjusted_size)

    logger.info(
      f"Размер позиции {symbol}: база={base_size:.4f}, "
      f"качество={quality_adjusted_size:.4f}, финал={final_size:.4f}"
    )

    # 8. Обогащаем сигнал информацией о качестве
    signal_dict = signal.to_dict()
    signal_dict['metadata'].update({
      'approved_size': final_size,
      'quality_score': quality_metrics.overall_score,
      'quality_category': quality_metrics.quality_category.value,
      'risk_reward_ratio': quality_metrics.risk_reward_ratio,
      'expected_win_rate': quality_metrics.expected_win_rate,
      'signal_percentile': quality_metrics.signal_strength_percentile,
      'quality_adjusted': True,
      'correlation_adjusted': final_size != quality_adjusted_size,
      'signal_time': datetime.now().isoformat()
    })

    # 9. Ставим в очередь
    pending_signals = self.state_manager.get_pending_signals()
    pending_signals[symbol] = signal_dict
    self.state_manager.update_pending_signals(pending_signals)

    logger.info(
      f"✅ Сигнал {symbol} одобрен: Качество={quality_metrics.overall_score:.2f}, "
      f"Категория={quality_metrics.quality_category.value}, Размер={final_size:.4f}"
    )
    signal_logger.info(
      f"ОДОБРЕНО: {symbol} - Качество {quality_metrics.overall_score:.2f}, "
      f"Размер {final_size:.4f}"
    )

  def _generate_quality_recommendation(self, results: Dict[str, Any]) -> str:
    """Генерирует рекомендации на основе анализа качества"""
    if not results:
      return "Недостаточно данных для рекомендаций"

    # Анализируем win rate по категориям
    excellent_wr = results.get('excellent', {}).get('win_rate', 0)
    good_wr = results.get('good', {}).get('win_rate', 0)
    fair_wr = results.get('fair', {}).get('win_rate', 0)

    recommendations = []

    if excellent_wr > 70:
      recommendations.append("Отличные сигналы показывают высокую эффективность - увеличьте размеры позиций для них")

    if fair_wr > good_wr:
      recommendations.append("⚠️ Сигналы среднего качества работают лучше хороших - проверьте настройки оценки")

    avg_wr = np.mean([r.get('win_rate', 0) for r in results.values() if r])
    if avg_wr < 50:
      recommendations.append("Общий win rate ниже 50% - рекомендуется повысить минимальный порог качества")

    return " | ".join(recommendations) if recommendations else "Система работает в нормальном режиме рекомендации на основе анализа качества"""

  if not results:
    return 'Недостаточно данных для рекомендаций'

    # Анализируем win rate по категориям
  excellent_wr = results.get('excellent', {}).get('win_rate', 0)
  good_wr = results.get('good', {}).get('win_rate', 0)
  fair_wr = results.get('fair', {}).get('win_rate', 0)

  recommendations = []

  if excellent_wr > 70:
    recommendations.append("Отличные сигналы показывают высокую эффективность - увеличьте размеры позиций для них")

  if fair_wr > good_wr:
    recommendations.append(f"⚠️ Сигналы среднего качества работают лучше хороших - проверьте настройки о {logger.info({category} : {count}) сигналов

  def set_quality_thresholds(self, min_score: float = 0.6,
                               quality_weights: Optional[Dict[str, float]] = None):
      """Настраивает пороги качества для системы"""
      self.min_quality_score = min_score

      if quality_weights:
        self.signal_quality_analyzer.quality_weights.update(quality_weights)

      logger.info(f"Обновлены пороги качества: минимальный балл = {min_score}")

  async def analyze_historical_signal_quality(self, days: int = 30) -> Dict[str, Any]:
    """Анализирует качество сигналов за период и их результаты"""
    logger.info(f"Анализ качества сигналов за последние {days} дней...")

    # Получаем закрытые сделки за период
    since_date = datetime.now() - timedelta(days=days)
    query = """
        SELECT symbol, strategy, side, open_price, close_price, 
               profit_loss, metadata, open_timestamp, close_timestamp
        FROM trades
        WHERE status = 'CLOSED' AND open_timestamp >= ?
        ORDER BY open_timestamp DESC
    """

    trades = await self.db_manager._execute(query, (since_date,), fetch='all')

    if not trades:
      return {"status": "no_trades"}

    quality_vs_performance = {
      QualityScore.EXCELLENT: {'total': 0, 'profitable': 0, 'avg_pnl': 0},
      QualityScore.GOOD: {'total': 0, 'profitable': 0, 'avg_pnl': 0},
      QualityScore.FAIR: {'total': 0, 'profitable': 0, 'avg_pnl': 0},
      QualityScore.POOR: {'total': 0, 'profitable': 0, 'avg_pnl': 0}
    }

    for trade in trades:
      try:
        metadata = json.loads(trade['metadata']) if trade['metadata'] else {}
        quality_score = metadata.get('quality_score', 0.5)
        quality_category = metadata.get('quality_category', 'fair')

        # Находим категорию
        category = None
        for cat in QualityScore:
          if cat.value == quality_category:
            category = cat
            break

        if category and category in quality_vs_performance:
          stats = quality_vs_performance[category]
          stats['total'] += 1
          if trade['profit_loss'] > 0:
            stats['profitable'] += 1
          stats['avg_pnl'] += trade['profit_loss']

      except Exception as e:
        logger.debug(f"Ошибка обработки сделки: {e}")
        continue

    # Вычисляем средние значения и win rate
    results = {}
    for category, stats in quality_vs_performance.items():
      if stats['total'] > 0:
        results[category.value] = {
          'total_trades': stats['total'],
          'win_rate': stats['profitable'] / stats['total'] * 100,
          'avg_pnl': stats['avg_pnl'] / stats['total']
        }

    return {
      'period_days': days,
      'total_trades_analyzed': len(trades),
      'quality_performance': results,
      'recommendation': self._generate_quality_recommendation(results)
    }

  def _generate_quality_recommendation(self, results: Dict[str, Any]) -> str:
    """
    Генерирует рекоменд# Обновления для integrated_system.py для интеграции системы оценки качества
    """

    # Добавьте импорт в начало файла:
    from core.signal_quality_analyzer import SignalQualityAnalyzer, QualityScore

    # В методе __init__ добавьте инициализацию после correlation_manager:
          # Инициализация анализатора качества сигналов
          self.signal_quality_analyzer = SignalQualityAnalyzer(self.data_fetcher, self.db_manager)
          self.min_quality_score = 0.6  # Минимальный балл качества для исполнения

    # Обновите метод _process_trading_signal_with_correlation, добавив оценку качества:

