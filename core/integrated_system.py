import asyncio
import json
from contextlib import suppress
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import pandas_ta as ta
import sys
from core.adaptive_strategy_selector import AdaptiveStrategySelector
from core.indicators import crossover_series, crossunder_series
from core.market_regime_detector import MarketRegimeDetector, RegimeCharacteristics, MarketRegime
from ml.feature_engineering import unified_feature_engineer
from ml.volatility_system import VolatilityPredictor, VolatilityPredictionSystem
import joblib
from config.config_manager import ConfigManager
from core.enums import Timeframe
from core.position_manager import PositionManager
from core.signal_filter import SignalFilter
from shadow_trading.signal_tracker import DatabaseMonitor
from strategies.GridStrategy import GridStrategy
from shadow_trading.shadow_trading_manager import ShadowTradingManager, FilterReason


from strategies.dual_thrust_strategy import DualThrustStrategy
from strategies.ensemble_ml_strategy import EnsembleMLStrategy
from strategies.ichimoku_strategy import IchimokuStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.sar_strategy import StopAndReverseStrategy
from utils.logging_config import get_logger
from config import trading_params, api_keys, settings
from core.data_fetcher import DataFetcher
from core.bybit_connector import BybitConnector
from core.strategy_manager import StrategyManager  # Будет использоваться позже
from core.risk_manager import AdvancedRiskManager # Будет использоваться позже
from core.trade_executor import TradeExecutor # Будет использоваться позже
from data.database_manager import AdvancedDatabaseManager # Будет использоваться позже
from core.enums import Timeframe, SignalType  # Для запроса свечей
from core.schemas import RiskMetrics, TradingSignal, GridSignal  # Для отображения баланса
from ml.model_retraining_task import ModelRetrainingManager
from data.state_manager import StateManager
import os
from ml.anomaly_detector import MarketAnomalyDetector, AnomalyType, AnomalyReport
from ml.enhanced_ml_system import EnhancedEnsembleModel, MLPrediction
import logging # <--- Добавьте импорт
from core.correlation_manager import CorrelationManager, PortfolioRiskMetrics
from core.signal_quality_analyzer import SignalQualityAnalyzer, QualityScore
# from shadow_trading import EnhancedShadowTradingManager
import time
signal_logger = logging.getLogger('SignalTrace') # <--- Получаем наш спец. логгер
logger = get_logger(__name__)


class IntegratedTradingSystem:
  def __init__(self, db_manager: AdvancedDatabaseManager = None, config: Dict[str, Any] = None):
    logger.info("Инициализация IntegratedTradingSystem...")

    # 1. Загружаем конфигурацию
    self.config_manager = ConfigManager()
    self.config = self.config_manager.load_config()

    # 2. Инициализируем основные компоненты
    self.connector = BybitConnector()
    self.db_manager = AdvancedDatabaseManager(settings.DATABASE_PATH)
    self.db_monitor = DatabaseMonitor(self.db_manager)
    self._monitoring_tasks = []
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

    # Загрузка детектора аномалий
    try:
      self.anomaly_detector = MarketAnomalyDetector.load("ml_models/anomaly_detector.pkl")
      logger.info("✅ Детектор аномалий успешно загружен")
    except FileNotFoundError:
      logger.warning("Файл детектора аномалий не найден. Будет использоваться эвристический режим")
      self.anomaly_detector = MarketAnomalyDetector()
    except Exception as e:
      logger.error(f"Ошибка при загрузке детектора аномалий: {e}")



    self.strategy_manager = StrategyManager()
    self.strategy_manager.add_strategy(ml_strategy)

    self.adaptive_selector = AdaptiveStrategySelector(
      db_manager=self.db_manager,
      min_trades_for_evaluation=10
    )
    self._evaluation_task: Optional[asyncio.Task] = None

    ichimoku_strategy = IchimokuStrategy()
    # "Регистрируем" ее в менеджере стратегий
    self.strategy_manager.add_strategy(ichimoku_strategy)

    # Создаем экземпляр Dual Thrust, передавая ему конфиг и data_fetcher
    dual_thrust_strategy = DualThrustStrategy(config=self.config, data_fetcher=self.data_fetcher)
    self.strategy_manager.add_strategy(dual_thrust_strategy)

    mean_reversion_strategy = MeanReversionStrategy()
    self.strategy_manager.add_strategy(mean_reversion_strategy)

    grid_strategy = GridStrategy(config=self.config)
    self.strategy_manager.add_strategy(grid_strategy)

    momentum_strategy = MomentumStrategy()
    self.strategy_manager.add_strategy(momentum_strategy)

    try:

      self.sar_strategy = StopAndReverseStrategy(
        config=self.config,
        data_fetcher=self.data_fetcher
      )
      self.strategy_manager.add_strategy(self.sar_strategy)
      logger.info("✅ Stop-and-Reverse стратегия зарегистрирована")
    except Exception as e:
      logger.error(f"Ошибка инициализации SAR стратегии: {e}")
      self.sar_strategy = None

    self.volatility_predictor: Optional[VolatilityPredictor] = None
    # --- НОВЫЙ БЛОК: ЗАГРУЗКА СИСТЕМЫ ВОЛАТИЛЬНОСТИ ---
    self.volatility_system: Optional[VolatilityPredictionSystem] = None
    try:
      self.volatility_system = joblib.load("ml_models/volatility_system.pkl")
      logger.info("✅ Система прогнозирования волатильности успешно загружена.")
    except FileNotFoundError:
      logger.warning("Файл volatility_system.pkl не найден. SL/TP будут рассчитываться по стандартной схеме.")
    # --- КОНЕЦ БЛОКА ---

    #ДОБАВИТЬ: Enhanced Shadow Trading
    # self.shadow_trading = None
    # self.shadow_trading_enabled = True  # Можно вынести в конфиг

    self.shadow_trading = None
    if self.config.get('enhanced_shadow_trading', {}).get('enabled', False):
      try:
        self.shadow_trading = ShadowTradingManager(self.db_manager, self.data_fetcher)
        logger.info("✅ Shadow Trading система инициализирована")
      except Exception as e:
        logger.error(f"Ошибка инициализации Shadow Trading: {e}")


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
      settings=self.config,
      risk_manager=self.risk_manager
    )
    self.trade_executor.integrated_system = self
    if self.shadow_trading:
      self.trade_executor.shadow_trading = self.shadow_trading

    if self.shadow_trading and self.data_fetcher:
      self.data_fetcher.shadow_trading_manager = self.shadow_trading

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

    self.market_regime_detector = MarketRegimeDetector(self.data_fetcher)
    # Флаги для включения/выключения ML моделей (оставляем как есть)
    self.use_enhanced_ml = True
    self.use_base_ml = True
    self._last_regime_check = {}
    self._regime_check_interval = 300

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
    ОБНОВЛЕННАЯ ВЕРСИЯ: Использует MarketRegimeDetector
    """
    logger.debug(f"Поиск сигнала для символа: {symbol}")

    try:
      # 1. Получаем рыночный режим
      regime_characteristics = await self.get_market_regime(symbol)
      if not regime_characteristics:
        logger.warning(f"Не удалось определить режим для {symbol}")
        return

      # Проверяем необходимость адаптации стратегий
      await self.check_strategy_adaptation(symbol)

      # 2. Получаем оптимальные параметры для режима
      regime_params = self.market_regime_detector.get_regime_parameters(symbol)

      # 3. Получаем данные для стратегий
      htf_data = await self.data_fetcher.get_historical_candles(
        symbol, Timeframe.ONE_HOUR, limit=300
      )

      unified_features = await unified_feature_engineer.get_unified_features(
        symbol, htf_data, self.data_fetcher, include_multiframe=True
      )

      if htf_data.empty or len(htf_data) < 52:
        return

      # 4. Проверяем, стоит ли вообще торговать в этом режиме
      if regime_characteristics.confidence < regime_params.min_signal_quality:
        logger.info(f"Пропускаем {symbol}: низкая уверенность режима "
                    f"({regime_characteristics.confidence:.2f} < {regime_params.min_signal_quality})")
        return

      # 5. Собираем сигналы от рекомендованных стратегий
      signals = []

      for strategy_name in regime_params.recommended_strategies:
        # Проверяем флаги для ML стратегий
        if strategy_name == "Live_ML_Strategy" and not self.use_base_ml:
          continue

        # Проверяем адаптивную активность стратегии
        if not self.adaptive_selector.should_activate_strategy(
            strategy_name, regime_characteristics.primary_regime.value
        ):
          logger.debug(f"Стратегия {strategy_name} отключена адаптивным селектором")
          continue

        # Пропускаем стратегии из avoided_strategies
        if strategy_name in regime_params.avoided_strategies:
          continue

        try:
          # Используем унифицированные признаки для ML стратегий
          if "ML" in strategy_name:
            signal = await self.strategy_manager.get_signal(symbol, unified_features, strategy_name)
          else:
            signal = await self.strategy_manager.get_signal(symbol, htf_data, strategy_name)

          if signal and signal.signal_type != SignalType.HOLD:
            # Применяем адаптивный вес
            weight = self.adaptive_selector.get_strategy_weight(
              strategy_name, regime_characteristics.primary_regime.value
            )
            signal.confidence *= weight

            signals.append((strategy_name, signal))
            logger.info(f"Сигнал от {strategy_name} для {symbol}: {signal.signal_type.value}, "
                        f"вес={weight:.2f}")
        except Exception as e:
          logger.error(f"Ошибка получения сигнала от {strategy_name}: {e}")

      # 6. Используем Enhanced ML для финального решения (если включена)
      final_signal = None

      if self.use_enhanced_ml and self.enhanced_ml_model and signals:
        try:
          # Подготавливаем данные для enhanced модели
          enhanced_prediction = self.enhanced_ml_model.predict_proba(htf_data)

          if enhanced_prediction:
            proba, ml_prediction = enhanced_prediction

            # Фильтруем сигналы на основе enhanced модели
            for strategy_name, signal in signals:
              if ml_prediction.signal_type == signal.signal_type:
                final_signal = signal
                # Корректируем уверенность
                final_signal.confidence = min(0.95,
                                              (signal.confidence + ml_prediction.probability) / 2)
                logger.info(f"✅ Enhanced ML подтвердила сигнал {strategy_name} для {symbol}")
                break
        except Exception as e:
          logger.error(f"Ошибка Enhanced ML для {symbol}: {e}")

      # 7. Если Enhanced ML не используется или не дала результат
      if not final_signal and signals:
        best_signal = max(signals, key=lambda x: x[1].confidence)
        final_signal = best_signal[1]
        logger.info(f"Выбран сигнал от {best_signal[0]} для {symbol}")

      # 8. Применяем параметры режима к сигналу
      if final_signal:
        # Корректируем на основе режима
        original_confidence = final_signal.confidence
        final_signal.confidence *= regime_params.position_size_multiplier

        # Добавляем метаданные о режиме
        final_signal.metadata = {
          'regime': regime_characteristics.primary_regime.value,
          'regime_confidence': regime_characteristics.confidence,
          'regime_strength': self.market_regime_detector.get_regime_strength_score(regime_characteristics),
          'use_limit_orders': regime_params.use_limit_orders,
          'sl_multiplier': regime_params.stop_loss_multiplier,
          'tp_multiplier': regime_params.take_profit_multiplier
        }

        logger.info(f"Финальный сигнал для {symbol}: {final_signal.signal_type.value}, "
                    f"уверенность: {original_confidence:.2f} -> {final_signal.confidence:.2f} "
                    f"(режим: {regime_characteristics.primary_regime.value})")

        # Проверяем минимальную уверенность после корректировок
        if final_signal.confidence >= regime_params.min_signal_quality:
          await self._process_trading_signal(symbol, final_signal)
        else:
          logger.info(f"Сигнал отклонен: уверенность {final_signal.confidence:.2f} "
                      f"< минимум {regime_params.min_signal_quality}")

    except Exception as e:
      logger.error(f"Ошибка мониторинга {symbol}: {e}", exc_info=True)

  def get_regime_statistics_for_dashboard(self) -> Dict[str, Any]:
    """Получает статистику режимов для отображения в дашборде"""
    stats = {}

    for symbol in self.active_symbols:
      if symbol in self.market_regime_detector.current_regimes:
        regime = self.market_regime_detector.current_regimes[symbol]
        stats[symbol] = {
          'regime': regime.primary_regime.value,
          'confidence': regime.confidence,
          'trend_strength': regime.trend_strength,
          'volatility': regime.volatility_level,
          'duration': str(regime.regime_duration)
        }

    return stats

  async def _monitor_symbol_for_entry_enhanced(self, symbol: str):
    """
    ОБНОВЛЕННАЯ ВЕРСИЯ с полной интеграцией Shadow Trading
    """
    logger.info(f"🔍 Поиск сигнала для {symbol}...")
    signal_logger.info(f"====== НАЧАЛО ЦИКЛА ДЛЯ {symbol} ======")

    try:
      # --- УРОВЕНЬ 1: ДЕТЕКЦИЯ РЕЖИМА РЫНКА ---
      htf_data = await self.data_fetcher.get_historical_candles(symbol, Timeframe.ONE_HOUR, limit=300)
      if htf_data.empty or len(htf_data) < 100:
        logger.debug(f"Недостаточно данных для анализа {symbol}")
        signal_logger.info(f"АНАЛИЗ: Пропущено - недостаточно данных.")
        return

      regime_characteristics = await self.get_market_regime(symbol, force_check=True)
      if not regime_characteristics:
        logger.warning(f"Не удалось определить режим для {symbol}")
        signal_logger.warning(f"АНАЛИЗ: Не удалось определить режим.")
        return

      signal_logger.info(
        f"РЕЖИМ: {regime_characteristics.primary_regime.value} (Уверенность: {regime_characteristics.confidence:.2f})")

      anomalies = await self._check_market_anomalies(symbol, htf_data)
      if any(a.severity > self.config.get('strategy_settings', {}).get('anomaly_severity_threshold', 0.8) for a in
             anomalies):
        logger.warning(f"Торговля по {symbol} заблокирована из-за критических аномалий.")
        signal_logger.critical(f"АНОМАЛИЯ: Торговля заблокирована.")
        return

      # --- УРОВЕНЬ 2: ВЫБОР И ГЕНЕРАЦИЯ СИГНАЛОВ ОТ СТРАТЕГИЙ ---
      await self.check_strategy_adaptation(symbol)

      regime_params = self.market_regime_detector.get_regime_parameters(symbol)
      if not regime_params.recommended_strategies or 'ALL' in regime_params.avoided_strategies:
        logger.info(f"Торговля в режиме '{regime_characteristics.primary_regime.value}' не рекомендуется для {symbol}.")
        signal_logger.info(f"РЕЖИМ: Торговля не рекомендуется.")
        return

      # --- ПРИОРИТЕТНАЯ ПРОВЕРКА ДЛЯ СЕТОЧНОЙ СТРАТЕГИИ и SAR ---
      active_strategies_from_dashboard = self.state_manager.get_custom_data('active_strategies') or {}
      candidate_signals: Dict[str, TradingSignal] = {}
      # Проверяем SAR стратегию отдельно
      if "Stop_and_Reverse" in regime_params.recommended_strategies and active_strategies_from_dashboard.get(
          "Stop_and_Reverse", True):
        if self.sar_strategy and symbol in self.sar_strategy.monitored_symbols:
          try:
            # Очищаем старый кэш перед генерацией сигнала
            self.sar_strategy._clear_old_cache()

            sar_signal = await self.sar_strategy.generate_signal(symbol, htf_data)
            if sar_signal and sar_signal.signal_type != SignalType.HOLD:
              # Обновляем статус позиции в SAR стратегии
              current_position = self.position_manager.open_positions.get(symbol)
              await self.sar_strategy.update_position_status(symbol, current_position)

              # Применяем адаптивный вес
              weight = self.adaptive_selector.get_strategy_weight(
                "Stop_and_Reverse", regime_characteristics.primary_regime.value
              )
              sar_signal.confidence *= weight

              # Интеграция с Shadow Trading
              if self.shadow_trading:
                signal_id = await self.shadow_trading.process_signal(
                  signal=sar_signal,
                  metadata={
                    'source': 'sar_strategy',
                    'strategy_name': 'Stop_and_Reverse',
                    'signal_score': sar_signal.metadata.get('signal_score', 0),
                    'sar_components': sar_signal.metadata.get('sar_components', {}),
                    'filter_reason': sar_signal.metadata.get('filter_reason', ''),
                    'market_regime': regime_characteristics.primary_regime.value,
                    'volatility_level': 'normal',  # TODO: динамически определять
                    'confidence_score': sar_signal.confidence
                  },
                  was_filtered=False
                )
                # Сохраняем ID для отслеживания
                sar_signal.metadata['shadow_tracking_id'] = signal_id

              candidate_signals["Stop_and_Reverse"] = sar_signal
              logger.info(f"SAR сигнал для {symbol}: {sar_signal.signal_type.value}, "
                          f"confidence={sar_signal.confidence:.3f}, вес={weight:.2f}")
          except Exception as e:
            logger.error(f"Ошибка получения SAR сигнала для {symbol}: {e}")

      if "Grid_Trading" in regime_params.recommended_strategies and active_strategies_from_dashboard.get("Grid_Trading",
                                                                                                         True):
        logger.info(
          f"Режим {regime_characteristics.primary_regime.value} подходит для сеточной торговли. Проверка GridStrategy...")
        grid_signal = await self.strategy_manager.get_signal(symbol, htf_data, "Grid_Trading")

        if isinstance(grid_signal, GridSignal):
          logger.info(f"Получен сеточный сигнал для {symbol}. Отправка на исполнение...")
          await self.trade_executor.execute_grid_trade(grid_signal)
          return
        else:
          logger.info("GridStrategy не сгенерировала сигнал. Переход к стандартной логике.")

      candidate_signals: Dict[str, TradingSignal] = {}
      for strategy_name in regime_params.recommended_strategies:
        if strategy_name == "Grid_Trading":
          continue

        if not active_strategies_from_dashboard.get(strategy_name, True):
          logger.debug(f"Стратегия {strategy_name} отключена в дашборде, пропускаем.")
          continue

        if not self.adaptive_selector.should_activate_strategy(strategy_name,
                                                               regime_characteristics.primary_regime.value):
          logger.debug(f"Стратегия {strategy_name} неактивна для {symbol}")
          continue

        signal = await self.strategy_manager.get_signal(symbol, htf_data, strategy_name)
        if signal and signal.signal_type != SignalType.HOLD:
          weight = self.adaptive_selector.get_strategy_weight(strategy_name,
                                                              regime_characteristics.primary_regime.value)
          signal.confidence *= weight
          candidate_signals[strategy_name] = signal
          signal_logger.info(
            f"СТРАТЕГИЯ ({strategy_name}): Сигнал {signal.signal_type.value}, Уверенность: {signal.confidence:.2f}")

      # --- УРОВЕНЬ 3: МЕТА-МОДЕЛЬ И ПРИНЯТИЕ РЕШЕНИЙ ---
      final_signal: Optional[TradingSignal] = None
      if self.enhanced_ml_model and self.use_enhanced_ml:
        logger.debug(f"Использование EnhancedEnsembleModel как финального арбитра для {symbol}...")
        _, ml_prediction = self.enhanced_ml_model.predict_proba(htf_data)

        if ml_prediction and ml_prediction.signal_type != SignalType.HOLD:
          signal_logger.info(
            f"МЕТА-МОДЕЛЬ: Предсказание {ml_prediction.signal_type.value}, Уверенность: {ml_prediction.confidence:.2f}")

          if any(s.signal_type == ml_prediction.signal_type for s in candidate_signals.values()):
            logger.info(f"Мета-модель подтверждена сигналом от другой стратегии для {symbol}.")
            final_signal = TradingSignal(
              signal_type=ml_prediction.signal_type,
              symbol=symbol,
              price=htf_data['close'].iloc[-1],
              confidence=ml_prediction.confidence,
              strategy_name="Ensemble_Confirmed",
              timestamp=datetime.now(),
              metadata={'ml_prediction': ml_prediction.metadata}
            )
            signal_logger.info(f"РЕШЕНИЕ: Сигнал мета-модели принят.")
          else:
            signal_logger.warning(f"РЕШЕНИЕ: Сигнал мета-модели отклонен - нет подтверждения.")
        else:
          signal_logger.info(f"РЕШЕНИЕ: Мета-модель предсказывает HOLD, сигнал не генерируется.")

      # Если мета-модель не дала сигнал, выбираем лучший из кандидатов
      if not final_signal and candidate_signals:
        best_signal = max(candidate_signals.values(), key=lambda s: s.confidence)
        if best_signal.confidence > regime_params.min_signal_quality:
          final_signal = best_signal
          signal_logger.info(f"РЕШЕНИЕ: Принят лучший сигнал от стратегии {best_signal.strategy_name}.")
        else:
          signal_logger.warning(f"РЕШЕНИЕ: Лучший сигнал ({best_signal.strategy_name}) отклонен - низкая уверенность.")

      # ============ ИНТЕГРАЦИЯ SHADOW TRADING ============
      if final_signal and final_signal.signal_type != SignalType.HOLD:
        signal_logger.info(f"🎯 НОВЫЙ СИГНАЛ {symbol}: {final_signal.signal_type.value} @ {final_signal.price}")

        # ЭТАП 1: Подготовка расширенных метаданных для Shadow Trading
        signal_metadata = await self._prepare_signal_metadata(symbol, final_signal, htf_data)

        # ЭТАП 2: Регистрация в Shadow Trading ДО фильтрации
        shadow_signal_id = ""
        if self.shadow_trading:
          try:
            shadow_signal_id = await self.shadow_trading.process_signal(
              signal=final_signal,
              metadata=signal_metadata,
              was_filtered=False
            )

            if shadow_signal_id:
              signal_logger.info(f"📊 Shadow ID: {shadow_signal_id}")
              # Добавляем связь с Shadow Trading в метаданные сигнала
              final_signal.metadata = final_signal.metadata or {}
              final_signal.metadata['shadow_tracking_id'] = shadow_signal_id

          except Exception as shadow_error:
            logger.warning(f"Ошибка регистрации в Shadow Trading: {shadow_error}")

        # ЭТАП 3: Проверка риск-менеджером
        risk_decision = await self.risk_manager.validate_signal(
          signal=final_signal,
          symbol=symbol,
          account_balance=self.account_balance.available_balance_usdt,
          market_data=htf_data
        )

        # ЭТАП 4: Обработка результата риск-менеджера
        if not risk_decision.get('approved'):
          rejection_reasons = risk_decision.get('reasons', [])
          signal_logger.warning(f"🚫 СИГНАЛ {symbol} ОТКЛОНЕН: {rejection_reasons}")

          # ВАЖНО: Отмечаем в Shadow Trading как отфильтрованный
          if self.shadow_trading and shadow_signal_id:
            try:
              filter_reasons = self._convert_rejection_reasons_to_filter_reasons(rejection_reasons)
              await self.shadow_trading.signal_tracker.mark_signal_filtered(
                shadow_signal_id, filter_reasons
              )
              logger.debug(f"🚫 Сигнал {shadow_signal_id} помечен как отфильтрованный")
            except Exception as filter_error:
              logger.warning(f"Ошибка отметки фильтрации в Shadow Trading: {filter_error}")
          return

        # ЭТАП 5: Сигнал одобрен - исполняем
        recommended_size = risk_decision.get('recommended_size')
        signal_logger.info(f"✅ СИГНАЛ {symbol} ОДОБРЕН, размер: {recommended_size}")

        success, order_data = await self.trade_executor.execute_trade(
          final_signal, symbol, recommended_size
        )

        # ЭТАП 6: Обновление Shadow Trading по результату исполнения
        if self.shadow_trading and shadow_signal_id:
          try:
            execution_result = {
              'executed': success,
              'execution_price': order_data.get('price') if order_data and success else final_signal.price,
              'quantity': recommended_size,
              'order_id': order_data.get('order_id') if order_data and success else None,
              'execution_time': datetime.now().isoformat(),
              'execution_success': success,
              'risk_manager_data': {
                'recommended_size': recommended_size,
                'risk_level': risk_decision.get('risk_level'),
                'position_impact': risk_decision.get('position_impact')
              }
            }

            # Обновляем метаданные в Shadow Trading
            tracked_signal = self.shadow_trading.signal_tracker.tracked_signals.get(shadow_signal_id)
            if tracked_signal:
              if not hasattr(tracked_signal, 'execution_data'):
                tracked_signal.execution_data = {}
              tracked_signal.execution_data.update(execution_result)
              tracked_signal.updated_at = datetime.now()

            if success:
              signal_logger.info(f"✅ Исполнение {symbol} успешно, отслеживается в Shadow Trading")
            else:
              signal_logger.error(f"❌ Ошибка исполнения {symbol}, продолжаем отслеживание в Shadow Trading")

          except Exception as execution_update_error:
            logger.warning(f"Ошибка обновления исполнения в Shadow Trading: {execution_update_error}")
      else:
        logger.info(f"Для {symbol} не найдено подходящего сигнала в текущем режиме.")
        signal_logger.info(f"ИТОГ: Сигнал не сформирован.")

    except Exception as e:
      logger.error(f"Критическая ошибка в _monitor_symbol_for_entry_enhanced для {symbol}: {e}", exc_info=True)
      signal_logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
    finally:
      signal_logger.info(f"====== КОНЕЦ ЦИКЛА ДЛЯ {symbol} ======\n")

  async def _process_trading_signal(self, signal: TradingSignal, symbol: str, market_data: pd.DataFrame):
    """
    Обработка торгового сигнала с учетом аномалий
    """
    logger.info(
      f"🔄 НАЧАЛО ОБРАБОТКИ СИГНАЛА для {symbol}: {signal.signal_type.value}, confidence={signal.confidence:.3f}")

    # Стандартная фильтрация
    logger.info(f"📋 Проверка сигнального фильтра...")
    is_approved, reason = await self.signal_filter.filter_signal(signal, market_data)
    if not is_approved:
      logger.info(f"❌ Сигнал для {symbol} отклонен фильтром: {reason}")
      return
    logger.info(f"✅ Сигнальный фильтр пройден")

    # Проверка рисков с учетом аномалий
    logger.info(f"💰 Обновление баланса аккаунта...")
    await self.update_account_balance()
    if not self.account_balance or self.account_balance.available_balance_usdt <= 0:
      logger.error(
        f"❌ Недостаточный баланс: {self.account_balance.available_balance_usdt if self.account_balance else 'None'}")
      return
    logger.info(f"✅ Баланс проверен: {self.account_balance.available_balance_usdt:.2f} USDT")

    # Корректируем размер позиции на основе обнаруженных аномалий
    position_size_multiplier = 1.0

    if 'anomalies' in signal.metadata:
      anomalies = signal.metadata['anomalies']
      if anomalies:
        # Уменьшаем размер позиции при аномалиях
        max_severity = max(a['severity'] for a in anomalies)
        position_size_multiplier = max(0.3, 1.0 - max_severity)
        logger.info(f"🔧 Размер позиции скорректирован на {position_size_multiplier:.2f} из-за аномалий")

    # Валидация сигнала риск-менеджером
    logger.info(f"⚠️ Валидация риск-менеджером...")
    risk_decision = await self.risk_manager.validate_signal(
      signal=signal,
      symbol=symbol,
      account_balance=self.account_balance.available_balance_usdt,
      market_data=market_data
    )

    if not risk_decision.get('approved'):
      logger.info(f"❌ Сигнал для {symbol} отклонен риск-менеджером: {risk_decision.get('reasons')}")
      return

    logger.info(f"✅ Риск-менеджер одобрил сигнал. Рекомендуемый размер: {risk_decision.get('recommended_size', 0):.6f}")

    # Корректируем размер с учетом аномалий
    final_size = risk_decision.get('recommended_size', 0) * position_size_multiplier
    logger.info(f"📊 Финальный размер позиции: {final_size:.6f}")

    # Ставим в очередь на исполнение
    logger.info(f"📥 Постановка сигнала в очередь ожидания...")
    pending_signals = self.state_manager.get_pending_signals()
    signal_dict = signal.to_dict()
    signal_dict['metadata']['approved_size'] = final_size
    signal_dict['metadata']['signal_time'] = datetime.now().isoformat()
    signal_dict['metadata']['position_size_multiplier'] = position_size_multiplier

    pending_signals[symbol] = signal_dict
    self.state_manager.update_pending_signals(pending_signals)

    logger.info(f"✅ Enhanced сигнал для {symbol} одобрен и поставлен в очередь")
    signal_logger.info(f"====== ENHANCED СИГНАЛ ДЛЯ {symbol} ПОСТАВЛЕН В ОЧЕРЕДЬ ======")

  async def update_sar_symbols_task(self):
    """Обновляет список символов для SAR стратегии каждый час"""
    while self.is_running:
      try:
        if self.sar_strategy:
          # Сохраняем текущий список для сравнения
          old_symbols = set(self.sar_strategy.monitored_symbols.keys())

          # Обновляем список
          updated_symbols = await self.sar_strategy.update_monitored_symbols(self.data_fetcher)
          new_symbols = set(updated_symbols)

          # Определяем изменения
          added_symbols = new_symbols - old_symbols
          removed_symbols = old_symbols - new_symbols

          # Обрабатываем исключенные символы
          if removed_symbols:
            await self.sar_strategy.handle_removed_symbols(
              list(removed_symbols), self.position_manager
            )

          # Логируем изменения
          if added_symbols or removed_symbols:
            logger.info(f"🔄 SAR символы обновлены: +{len(added_symbols)}, -{len(removed_symbols)}")

          # Обновляем статус в state_manager
          sar_status = self.sar_strategy.get_strategy_status()
          self.state_manager.set_custom_data('sar_strategy_status', sar_status)

      except Exception as e:
        logger.error(f"Ошибка обновления SAR символов: {e}")

      await asyncio.sleep(3600)  # 1 час

  # Задача очистки кэша SAR стратегии
  async def cleanup_sar_cache_task(self):
    """Периодически очищает кэш SAR стратегии"""
    while self.is_running:
      try:
        if self.sar_strategy:
          self.sar_strategy._clear_old_cache()
      except Exception as e:
        logger.error(f"Ошибка очистки кэша SAR: {e}")
      await asyncio.sleep(300)  # 5 минут

  async def transfer_position_from_strategy(self, symbol: str, position_data: Dict, strategy_name: str):
    """
    Принимает позицию от стратегии для дальнейшей обработки основной системой
    """
    try:
      logger.info(f"📥 Получена позиция {symbol} от стратегии {strategy_name}")

      # Добавляем позицию в основной список для мониторинга
      self.open_positions[symbol] = position_data

      # Логируем детали передачи
      transfer_reason = position_data.get('transfer_reason', 'unknown')
      logger.info(f"📋 Позиция {symbol} принята в основную обработку. Причина: {transfer_reason}")

      # Уведомляем систему о необходимости особого внимания к этой позиции
      if hasattr(self, 'special_monitoring_positions'):
        self.special_monitoring_positions.add(symbol)

      return True

    except Exception as e:
      logger.error(f"Ошибка передачи позиции {symbol} от {strategy_name}: {e}")
      return False

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
    Обучает расширенную ML модель с правильным выравниванием данных
    """
    logger.info(f"Начало обучения Enhanced ML модели на {len(symbols)} символах...")

    logger.info("=== ОТЛАДКА СОЗДАНИЯ ДАННЫХ ===")

    # Тестируем на одном символе
    test_symbol = symbols[0]
    test_data = await self.data_fetcher.get_historical_candles(
      test_symbol, Timeframe.ONE_HOUR, limit=100
    )

    logger.info(f"Тестовые данные {test_symbol}:")
    logger.info(f"  Размер: {test_data.shape}")
    logger.info(f"  Индекс: {type(test_data.index)}")
    logger.info(f"  Колонки: {test_data.columns.tolist()}")

    test_labels = self._create_ml_labels(test_data)
    if test_labels is not None:
      logger.info(f"Тестовые метки:")
      logger.info(f"  Размер: {len(test_labels)}")
      logger.info(f"  Индекс: {type(test_labels.index)}")
      logger.info(f"  Распределение: {test_labels.value_counts().to_dict()}")

    logger.info("=== КОНЕЦ ОТЛАДКИ ===")

    if not self.enhanced_ml_model:
      self.enhanced_ml_model = EnhancedEnsembleModel(self.anomaly_detector)

    all_features = []
    all_labels = []

    for symbol in symbols:  # Ограничиваем для демонстрации
      try:
        # Получаем данные
        data = await self.data_fetcher.get_historical_candles(
          symbol,
          Timeframe.ONE_HOUR,
          limit=24 * lookback_days
        )

        if data.empty or len(data) < 100:
          logger.warning(f"Недостаточно данных для {symbol}: {len(data)} свечей")
          continue

        # Убеждаемся, что данные имеют datetime индекс
        if not isinstance(data.index, pd.DatetimeIndex):
          if 'timestamp' in data.columns:
            data = data.set_index('timestamp')
          else:
            # Создаем datetime индекс
            start_time = datetime.now() - timedelta(hours=len(data))
            data.index = pd.date_range(start=start_time, periods=len(data), freq='1H')

        # Создаем метки
        labels = self._create_ml_labels(data)
        if labels is None or len(labels) < 50:
          logger.warning(f"Не удалось создать достаточно меток для {symbol}")
          continue

        # Убеждаемся, что данные и метки имеют пересекающиеся индексы
        common_index = data.index.intersection(labels.index)
        if len(common_index) < 50:
          logger.warning(f"Мало общих временных точек для {symbol}: {len(common_index)}")
          continue

        # Берем только общие данные
        data_aligned = data.loc[common_index]
        labels_aligned = labels.loc[common_index]

        # Добавляем символ к данным для идентификации
        data_aligned = data_aligned.copy()
        data_aligned['symbol'] = symbol

        all_features.append(data_aligned)
        all_labels.append(labels_aligned)

        logger.info(f"Подготовлены данные для {symbol}: {len(data_aligned)} образцов")

      except Exception as e:
        logger.error(f"Ошибка подготовки данных для {symbol}: {e}")
        continue

    if not all_features:
      logger.error("Нет данных для обучения Enhanced ML модели")
      return

    try:
      # Объединяем данные
      combined_features = pd.concat(all_features, ignore_index=False)
      combined_labels = pd.concat(all_labels, ignore_index=False)

      # Сбрасываем индекс для избежания конфликтов при объединении
      combined_features = combined_features.reset_index(drop=True)
      combined_labels = combined_labels.reset_index(drop=True)

      # Убеждаемся, что индексы соответствуют
      if len(combined_features) != len(combined_labels):
        min_len = min(len(combined_features), len(combined_labels))
        combined_features = combined_features.iloc[:min_len]
        combined_labels = combined_labels.iloc[:min_len]

      logger.info(f"Финальные размеры данных: features={combined_features.shape}, labels={combined_labels.shape}")

      # Получаем внешние данные (BTC как пример)
      btc_data = await self.data_fetcher.get_historical_candles(
        "BTCUSDT",
        Timeframe.ONE_HOUR,
        limit=24 * lookback_days
      )

      external_data = None
      if not btc_data.empty:
        # Приводим BTC данные к тому же формату
        if not isinstance(btc_data.index, pd.DatetimeIndex):
          if 'timestamp' in btc_data.columns:
            btc_data = btc_data.set_index('timestamp')
          else:
            start_time = datetime.now() - timedelta(hours=len(btc_data))
            btc_data.index = pd.date_range(start=start_time, periods=len(btc_data), freq='1H')

        external_data = {'BTC': btc_data}

      logger.info(f"Размеры данных перед обучением: features={combined_features.shape}, labels={combined_labels.shape}")
      logger.info(f"Индексы: features={combined_features.index.min()} - {combined_features.index.max()}")
      logger.info(f"Индексы: labels={combined_labels.index.min()} - {combined_labels.index.max()}")

      # Убеждаемся, что индексы корректны
      if not combined_features.index.equals(combined_labels.index):
        logger.warning("Индексы признаков и меток не совпадают, выполняем выравнивание...")
        common_idx = combined_features.index.intersection(combined_labels.index)
        combined_features = combined_features.loc[common_idx]
        combined_labels = combined_labels.loc[common_idx]
        logger.info(f"После выравнивания: {len(common_idx)} общих образцов")

      diagnosis = self.enhanced_ml_model.diagnose_training_issues(
        combined_features,
        combined_labels,
        )
      try:
        diagnosis_status = diagnosis.get('overall_status', 'НЕИЗВЕСТНО')
        logger.info(f"Диагностика: {diagnosis_status}")

        # Дополнительная информация
        if diagnosis.get('issues_found'):
          logger.warning(f"Обнаружено проблем: {len(diagnosis['issues_found'])}")
          for issue in diagnosis['issues_found'][:3]:  # Показываем первые 3
            logger.warning(f"  - {issue}")

        if diagnosis.get('warnings'):
          logger.info(f"Предупреждений: {len(diagnosis['warnings'])}")

      except Exception as log_error:
        logger.error(f"Ошибка логирования диагностики: {log_error}")

      # Обучаем модель
      # self.enhanced_ml_model.fit_with_diagnostics(
      #   combined_features,
      #   combined_labels,
      #   external_data=external_data,
      #   optimize_features=True,
      #   verbose=True
      # )
      # self.enhanced_ml_model.print_training_report(combined_features,
      #   combined_labels, diagnosis)

      # Обучаем модель с использованием подбора гиперпараметров
      logger.info("Запуск обучения с подбором гиперпараметров...")
      self.enhanced_ml_model.fit_with_hyperparameter_tuning(
        X_train_data=combined_features,  # Используем подготовленные данные
        y_train_data=combined_labels,  # Используем подготовленные метки
        external_data=external_data
      )

      # Отчет можно оставить, он покажет результаты уже после тюнинга
      self.enhanced_ml_model.print_training_report(combined_features,
                                                   combined_labels, diagnosis)


      health = self.enhanced_ml_model.get_model_health_status()
      logger.info(f"Здоровье модели: {health['overall_health']}")


      # Сохраняем
      self.enhanced_ml_model.save("ml_models/enhanced_model.pkl")

      logger.info("Enhanced ML модель успешно обучена и сохранена")

    except Exception as e:
      logger.error(f"Ошибка при финальном обучении модели: {e}")
      raise

  async def get_market_regime(self, symbol: str, force_check: bool = False) -> Optional[RegimeCharacteristics]:
    """
    Получает текущий рыночный режим для символа
    """
    try:
      # Получаем данные для анализа
      data = await self.data_fetcher.get_historical_candles(
        symbol, Timeframe.ONE_HOUR, limit=200
      )

      if data.empty or len(data) < 50:
        return None

      # Используем существующий детектор
      regime_characteristics = await self.market_regime_detector.detect_regime(symbol, data)

      # Логируем с учетом аномалий
      if self.anomaly_detector and regime_characteristics:
        anomalies = self.anomaly_detector.detect_anomalies(data, symbol)
        if anomalies:
          logger.warning(f"⚠️ Режим {symbol}: {regime_characteristics.primary_regime.value} "
                         f"+ обнаружены аномалии!")

      return regime_characteristics

    except Exception as e:
      logger.error(f"Ошибка определения режима для {symbol}: {e}")
      return None


  def _create_ml_labels(self, data: pd.DataFrame) -> Optional[pd.Series]:
    """
    Создает метки для обучения ML с правильными индексами
    """
    try:
      # Убеждаемся, что данные имеют правильный индекс
      if not isinstance(data.index, pd.DatetimeIndex):
        if 'timestamp' in data.columns:
          data = data.set_index('timestamp')
        else:
          # Создаем datetime индекс на основе позиции
          start_time = datetime.now() - timedelta(hours=len(data))
          data.index = pd.date_range(start=start_time, periods=len(data), freq='1H')

      if 'close' not in data.columns:
        logger.warning("Нет колонки 'close' для создания меток")
        return None

      # Вычисляем будущие доходности
      future_periods = 10  # Смотрим на 10 периодов вперед
      future_returns = data['close'].pct_change(periods=future_periods).shift(-future_periods)

      # Пороги для классификации
      buy_threshold = 0.02  # 2% рост
      sell_threshold = -0.02  # 2% падение

      # Создаем метки с тем же индексом, что и у данных
      labels = pd.Series(index=data.index, dtype=int, name='labels')

      # Заполняем метки
      labels[future_returns > buy_threshold] = 2  # BUY
      labels[future_returns < sell_threshold] = 0  # SELL
      labels[(future_returns >= sell_threshold) & (future_returns <= buy_threshold)] = 1  # HOLD

      # Удаляем NaN в конце (где нет будущих данных)
      labels = labels.dropna()

      logger.debug(f"Создано меток: {len(labels)}, распределение: {labels.value_counts().to_dict()}")

      return labels

    except Exception as e:
      logger.error(f"Ошибка при создании меток: {e}")
      return None

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

    try:
      # Инициализация Shadow Trading
      self.shadow_trading = ShadowTradingManager(self.db_manager, self.data_fetcher)
      await self.shadow_trading.start_enhanced_monitoring()
      logger.info("✅ Shadow Trading система инициализирована и запущена")
    except Exception as e:
      logger.error(f"Ошибка инициализации Shadow Trading: {e}")
      self.shadow_trading = None

    # # Устанавливаем плечо для всех активных символов
    # leverage = self.config.get('trade_settings', {}).get('leverage', 10)
    # for symbol in self.active_symbols:
    #   # self.current_leverage.setdefault(symbol, leverage) # Эта строка не нужна
    #   await self.set_leverage_for_symbol(symbol, leverage)

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
    # Запрашиваем баланс для UNIFIED аккаунта по USDT
    balance_data = await self.connector.get_account_balance(account_type="UNIFIED", coin="USDT")

    # >>> НАЧАЛО ПАТЧА <<<
    # Новая, более надежная логика обработки ответа
    if balance_data and balance_data.get('coin'):
      coin_data_list = balance_data.get('coin', [])
      if coin_data_list:
        coin_data = coin_data_list[0]
        try:
          self.account_balance = RiskMetrics(
            total_balance_usdt=float(coin_data.get('walletBalance', 0)),
            available_balance_usdt=float(balance_data.get('totalAvailableBalance', 0)),
            unrealized_pnl_total=float(coin_data.get('unrealisedPnl', 0)),
            realized_pnl_total=float(coin_data.get('cumRealisedPnl', 0))
          )
          # После успешного обновления баланса, обновляем state_manager
          self.state_manager.update_metrics(self.account_balance)

          logger.info(f"Баланс обновлен: Всего={self.account_balance.total_balance_usdt:.2f} USDT, "
                      f"Доступно={self.account_balance.available_balance_usdt:.2f} USDT")
          return  # Явный выход после успеха

        except (ValueError, TypeError) as e:
          logger.error(f"Ошибка преобразования данных баланса: {e}. Ответ: {coin_data}")
          self.account_balance = self.account_balance or RiskMetrics()  # Сохраняем старое значение, если есть
      else:
        logger.error(f"Список 'coin' в ответе о балансе пуст. Ответ: {balance_data}")
        self.account_balance = self.account_balance or RiskMetrics()
    else:
      logger.error(f"Не удалось получить или распарсить данные о балансе. Ответ: {balance_data}")
      # Если не удалось обновить, оставляем старое значение, чтобы не обнулять баланс
      self.account_balance = self.account_balance or RiskMetrics()
    # >>> КОНЕЦ ПАТЧА <<<
  # async def update_account_balance(self):
  #   logger.info("Запрос баланса аккаунта...")
  #   balance_data = await self.connector.get_account_balance(account_type="UNIFIED", coin="USDT")
  #
  #   # ИСПРАВЛЕННАЯ ЛОГИКА: Проверяем наличие ключа 'coin', и что это непустой список
  #   if (balance_data
  #       and 'coin' in balance_data
  #       and isinstance(balance_data.get('coin'), list)
  #       and len(balance_data['coin']) > 0):
  #
  #     # Данные по конкретной монете (USDT) находятся внутри первого элемента списка 'coin'
  #     coin_data = balance_data['coin'][0]
  #
  #     self.account_balance = RiskMetrics(
  #       # Общий баланс кошелька берем из данных по конкретной монете
  #       total_balance_usdt=float(coin_data.get('walletBalance', 0)),
  #
  #       # Доступный баланс надежнее брать из общего поля 'totalAvailableBalance'
  #       available_balance_usdt=float(balance_data.get('totalAvailableBalance', 0)),
  #
  #       # Нереализованный и реализованный PnL берем из данных по монете
  #       unrealized_pnl_total=float(coin_data.get('unrealisedPnl', 0)),
  #       realized_pnl_total=float(coin_data.get('cumRealisedPnl', 0))
  #     )
  #     logger.info(f"Баланс обновлен: Всего={self.account_balance.total_balance_usdt:.2f} USDT, "
  #                 f"Доступно={self.account_balance.available_balance_usdt:.2f} USDT, "
  #                 f"Нереализ. PNL={self.account_balance.unrealized_pnl_total:.2f} USDT, "
  #                 f"Реализ. PNL={self.account_balance.realized_pnl_total:.2f} USDT")
  #   else:
  #     logger.error(f"Не удалось получить или распарсить данные о балансе. Ответ: {balance_data}")
  #     self.account_balance = RiskMetrics()

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
        tasks = [self._check_pending_signal_for_entry(s) for s in pending_signals.keys()]
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

        # Очищаем команду, чтобы не выполнять ее повторно
        self.state_manager.clear_command()
      # --- КОНЕЦ НОВОГО БЛОКА ---

        if command_name == 'generate_report':
          if self.retraining_manager:
            self.retraining_manager.export_performance_report()


        elif command_name == 'update_ml_models':
          # Обновляем состояние ML моделей
          ml_state = self.state_manager.get_custom_data('ml_models_state')
          if ml_state:
            self.use_enhanced_ml = ml_state.get('use_enhanced_ml', True)
            self.use_base_ml = ml_state.get('use_base_ml', True)
            logger.info(f"ML модели обновлены: Enhanced={self.use_enhanced_ml}, Base={self.use_base_ml}")

        elif command_name == 'update_strategies':
          # Обновляем активные стратегии
          active_strategies = self.state_manager.get_custom_data('active_strategies')
          if active_strategies and hasattr(self, 'adaptive_selector'):
            for strategy_name, is_active in active_strategies.items():
              self.adaptive_selector.active_strategies[strategy_name] = is_active
            logger.info(f"Стратегии обновлены: {active_strategies}")

        elif command_name == 'retrain_model':
          # Запускаем переобучение
          if self.retraining_manager:
            asyncio.create_task(self.retraining_manager.retrain_model(
              self.active_symbols, timeframe=Timeframe.ONE_HOUR
            ))
            logger.info("Запущено переобучение модели")
        # Обработка команды экспорта отчета SAR
        elif command_name == 'export_sar_report':
          if hasattr(self, 'sar_strategy') and self.sar_strategy:
            report_path = self.sar_strategy.export_performance_report()
            if report_path:
              logger.info(f"Отчет SAR стратегии сохранен: {report_path}")

        # Обновляем метрики SAR стратегии для дашборда
        if hasattr(self, 'sar_strategy') and self.sar_strategy:
          try:
            sar_metrics = self.sar_strategy.get_dashboard_metrics()
            self.state_manager.set_custom_data('sar_strategy_performance', sar_metrics)
            logger.debug(f"SAR метрики обновлены для дашборда: {len(sar_metrics)} параметров")
          except Exception as e:
            logger.error(f"Ошибка обновления SAR метрик: {e}")

        if hasattr(self, 'adaptive_selector') and self.adaptive_selector:
          try:
            performance_summary = self.adaptive_selector.get_performance_summary()

            # Извлекаем только веса для удобства
            weights = {}
            for strategy_name, perf in performance_summary.items():
              weights[strategy_name] = perf.get('weight', 1.0)

            self.state_manager.set_custom_data('adaptive_weights', weights)
            self.state_manager.set_custom_data('strategy_performance_summary', performance_summary)
            logger.debug(f"Адаптивные веса обновлены: {len(weights)} стратегий")
          except Exception as e:
            logger.error(f"Ошибка обновления адаптивных весов: {e}")


      interval = self.config.get('general_settings', {}).get('monitoring_interval_seconds', 30)
      await asyncio.sleep(interval)

  async def _prepare_signal_metadata(self, symbol: str, signal: TradingSignal, data: pd.DataFrame) -> Dict[str, Any]:
      """Подготовка метаданных сигнала для Shadow Trading"""
      try:
        metadata = {
          'source': self._determine_signal_source(signal),
          'indicators_triggered': self._get_triggered_indicators(symbol, data),
          'market_regime': await self._determine_market_regime(data, symbol),
          'volatility_level': self._determine_volatility_level(data),
          'confidence_score': signal.confidence,
          'strategy_name': signal.strategy_name or 'unknown',
          'volume': float(data['volume'].iloc[-1]) if 'volume' in data.columns else 0,
          'price_action_score': self._calculate_price_action_score(data),
          'market_session': self._determine_market_session(),
          'correlation_data': await self._get_correlation_data(symbol) if hasattr(self,
                                                                                  '_get_correlation_data') else {},
          'liquidity_score': self._calculate_liquidity_score(data) if hasattr(self,
                                                                              '_calculate_liquidity_score') else 0,
          'signal_timestamp': signal.timestamp.isoformat(),
          'symbol': symbol
        }
        # Если доступен продвинутый детектор режимов, добавляем расширенные данные
        if hasattr(self, 'market_regime_detector') and self.market_regime_detector:
          try:
            regime_characteristics = await self.market_regime_detector.detect_regime(symbol, data)
            if regime_characteristics:
              metadata.update({
                'regime_confidence': regime_characteristics.confidence,
                'trend_strength': regime_characteristics.trend_strength,
                'volatility_level_detailed': regime_characteristics.volatility_level,
                'momentum_score': regime_characteristics.momentum_score,
                'regime_duration_hours': regime_characteristics.regime_duration.total_seconds() / 3600,
                'secondary_regime': regime_characteristics.secondary_regime.value if regime_characteristics.secondary_regime else None,
                'supporting_indicators': regime_characteristics.supporting_indicators
              })
          except Exception as e:
            logger.debug(f"Не удалось получить расширенные данные режима: {e}")


        # ML данные если доступны
        if hasattr(signal, 'metadata') and signal.metadata:
          metadata['ml_prediction_data'] = signal.metadata

        # Добавляем технические уровни
        metadata['technical_levels'] = self._get_technical_levels(data)

        return metadata

      except Exception as e:
        logger.warning(f"Ошибка подготовки метаданных для {symbol}: {e}")
        return {'source': 'unknown', 'error': str(e)}

  async def _update_dashboard_metrics(self):
    """Обновляет метрики для дашборда"""
    try:
      # 1. Обновляем SAR метрики
      if hasattr(self, 'sar_strategy') and self.sar_strategy:
        try:
          sar_metrics = self.sar_strategy.get_dashboard_metrics()
          self.state_manager.set_custom_data('sar_strategy_performance', sar_metrics)
          logger.debug(f"SAR метрики обновлены: {len(sar_metrics)} параметров")
        except Exception as e:
          logger.error(f"Ошибка обновления SAR метрик: {e}")

      # 2. Обновляем адаптивные веса
      if hasattr(self, 'adaptive_selector') and self.adaptive_selector:
        try:
          performance_summary = self.adaptive_selector.get_performance_summary()

          # Извлекаем веса
          weights = {}
          for strategy_name, perf in performance_summary.items():
            weights[strategy_name] = perf.get('weight', 1.0)

          self.state_manager.set_custom_data('adaptive_weights', weights)
          self.state_manager.set_custom_data('strategy_performance_summary', performance_summary)
          logger.debug(f"Адаптивные веса обновлены: {len(weights)} стратегий")
        except Exception as e:
          logger.error(f"Ошибка обновления адаптивных весов: {e}")

      # 3. Обновляем статус SAR стратегии
      if hasattr(self, 'sar_strategy') and self.sar_strategy:
        try:
          sar_status = self.sar_strategy.get_strategy_status()
          self.state_manager.set_custom_data('sar_strategy_status', sar_status)
        except Exception as e:
          logger.error(f"Ошибка обновления статуса SAR: {e}")

    except Exception as e:
      logger.error(f"Ошибка обновления метрик дашборда: {e}")

  def _determine_signal_source(self, signal: TradingSignal) -> str:
    """Определить источник сигнала с полным сопоставлением стратегий"""
    if not signal or not hasattr(signal, 'strategy_name'):
      return 'unknown'

    strategy_name = str(getattr(signal, 'strategy_name', '')).lower().strip()

    if not strategy_name or strategy_name == 'unknown':
      return 'unknown'

    # Точное сопоставление по названиям стратегий из системы
    strategy_mapping = {
      'live_ml_strategy': 'ml_model',
      'ensemble_confirmed': 'ml_ensemble',
      'enhanced_ml': 'ml_enhanced',
      'reversalsar': 'sar_strategy',
      'sar_strategy': 'sar_strategy',
      'stop_and_reverse': 'sar_strategy',
      'ichimoku_cloud': 'ichimoku_cloud',
      'dual_thrust': 'dual_thrust',
      'momentum_spike': 'momentum_spike',
      'mean_reversion_bb': 'mean_reversion',
      'bollinger_bands': 'mean_reversion',
      'grid_trading': 'grid_trading',
      'scalping_strategy': 'scalping',
      'swing_strategy': 'swing_trading',
      'arbitrage_strategy': 'arbitrage'
    }

    # Прямое сопоставление
    if strategy_name in strategy_mapping:
      return strategy_mapping[strategy_name]

    # Частичное сопоставление для составных названий
    for pattern, source in strategy_mapping.items():
      if pattern in strategy_name:
        return source

    # Если точное сопоставление не найдено, классифицируем по ключевым словам
    if any(word in strategy_name for word in ['ml', 'machine', 'neural', 'ensemble']):
      return 'ml_model'
    elif any(word in strategy_name for word in ['sar', 'parabolic', 'reversal']):
      return 'sar_strategy'
    elif any(word in strategy_name for word in ['bollinger', 'mean_reversion', 'reversion']):
      return 'mean_reversion'
    elif any(word in strategy_name for word in ['momentum', 'breakout', 'spike']):
      return 'breakout'
    elif any(word in strategy_name for word in ['ichimoku', 'cloud']):
      return 'ichimoku_cloud'
    elif any(word in strategy_name for word in ['dual', 'thrust']):
      return 'dual_thrust'
    elif any(word in strategy_name for word in ['grid', 'martingale']):
      return 'grid_trading'
    elif any(word in strategy_name for word in ['scalp']):
      return 'scalping'
    elif any(word in strategy_name for word in ['swing']):
      return 'swing_trading'
    else:
      # Возвращаем оригинальное название если не удалось классифицировать
      return strategy_name.replace(' ', '_')

  def _get_triggered_indicators(self, symbol: str, data: pd.DataFrame) -> List[str]:
    """Получить список сработавших индикаторов"""
    triggered = []

    try:
      latest = data.iloc[-1]

      # RSI анализ
      if 'rsi_14' in data.columns:
        rsi = latest['rsi_14']
        if rsi > 70:
          triggered.append('rsi_overbought')
        elif rsi < 30:
          triggered.append('rsi_oversold')
        elif 50 < rsi < 60:
          triggered.append('rsi_bullish_zone')
        elif 40 < rsi < 50:
          triggered.append('rsi_bearish_zone')

      # MACD анализ
      if all(col in data.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
        macd = latest['macd']
        macd_signal = latest['macd_signal']
        macd_hist = latest['macd_histogram']

        if macd > macd_signal:
          triggered.append('macd_bullish')
        else:
          triggered.append('macd_bearish')

        if macd_hist > 0 and data['macd_histogram'].iloc[-2] <= 0:
          triggered.append('macd_histogram_cross_up')
        elif macd_hist < 0 and data['macd_histogram'].iloc[-2] >= 0:
          triggered.append('macd_histogram_cross_down')

      # Bollinger Bands
      if all(col in data.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
        price = latest['close']
        bb_upper = latest['bb_upper']
        bb_lower = latest['bb_lower']
        bb_middle = latest['bb_middle']

        if price > bb_upper:
          triggered.append('bb_upper_breach')
        elif price < bb_lower:
          triggered.append('bb_lower_breach')
        elif price > bb_middle:
          triggered.append('bb_above_middle')
        else:
          triggered.append('bb_below_middle')

      # Moving Averages
      if all(col in data.columns for col in ['ema_20', 'ema_50']):
        ema_20 = latest['ema_20']
        ema_50 = latest['ema_50']
        price = latest['close']

        if ema_20 > ema_50:
          triggered.append('ema_bullish_alignment')
        else:
          triggered.append('ema_bearish_alignment')

        if price > ema_20:
          triggered.append('price_above_ema20')
        if price > ema_50:
          triggered.append('price_above_ema50')

      # Volume анализ
      if 'volume' in data.columns:
        volume = latest['volume']
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]

        if volume > avg_volume * 1.5:
          triggered.append('high_volume')
        elif volume < avg_volume * 0.5:
          triggered.append('low_volume')

      # ADX для определения тренда
      if 'adx' in data.columns:
        adx = latest['adx']
        if adx > 25:
          triggered.append('strong_trend')
        elif adx < 20:
          triggered.append('weak_trend')

      # Stochastic
      if all(col in data.columns for col in ['stoch_k', 'stoch_d']):
        stoch_k = latest['stoch_k']
        stoch_d = latest['stoch_d']

        if stoch_k > 80 and stoch_d > 80:
          triggered.append('stoch_overbought')
        elif stoch_k < 20 and stoch_d < 20:
          triggered.append('stoch_oversold')

        if stoch_k > stoch_d:
          triggered.append('stoch_bullish_cross')
        else:
          triggered.append('stoch_bearish_cross')

    except Exception as e:
      logger.warning(f"Ошибка анализа индикаторов для {symbol}: {e}")
      triggered.append('indicator_analysis_error')

    return triggered

  async def _determine_market_regime(self, data: pd.DataFrame, symbol: str = None) -> str:
    """Определить рыночный режим через продвинутый детектор"""
    try:
      # Используем продвинутый MarketRegimeDetector если доступен
      if hasattr(self, 'market_regime_detector') and self.market_regime_detector and symbol:
        # Получаем характеристики режима
        regime_characteristics = await self.market_regime_detector.detect_regime(symbol, data)

        if regime_characteristics:
          # Возвращаем строковое представление режима
          regime_name = regime_characteristics.primary_regime.value.lower()

          # Логируем детали для отладки
          logger.debug(f"Режим для {symbol}: {regime_name} "
                       f"(уверенность: {regime_characteristics.confidence:.2f}, "
                       f"сила тренда: {regime_characteristics.trend_strength:.2f})")

          return regime_name

      # Fallback к простому методу если продвинутый недоступен
      return self._simple_market_regime_fallback(data)

    except Exception as e:
      logger.warning(f"Ошибка определения режима: {e}")
      return self._simple_market_regime_fallback(data)

  def _simple_market_regime_fallback(self, data: pd.DataFrame) -> str:
    """Простое определение режима как fallback"""
    try:
      if len(data) < 50:
        return 'insufficient_data'

      close_prices = data['close'].tail(50)
      sma_20 = close_prices.rolling(20).mean()
      sma_50 = close_prices.rolling(50).mean()

      current_price = close_prices.iloc[-1]
      sma_20_current = sma_20.iloc[-1]
      sma_50_current = sma_50.iloc[-1]

      volatility = close_prices.pct_change().std() * 100

      # Классификация режима
      if current_price > sma_20_current > sma_50_current:
        return 'strong_trend_up' if volatility > 3.0 else 'trend_up'
      elif current_price < sma_20_current < sma_50_current:
        return 'strong_trend_down' if volatility > 3.0 else 'trend_down'
      elif volatility > 4.0:
        return 'volatile'
      else:
        return 'ranging'

    except Exception:
      return 'unknown'

  def _determine_volatility_level(self, data: pd.DataFrame) -> str:
    """Определить уровень волатильности"""
    try:
      latest = data.iloc[-1]

      if 'atr' in data.columns:
        atr = latest['atr']
        price = latest['close']
        atr_pct = (atr / price) * 100

        if atr_pct > 4:
          return 'very_high'
        elif atr_pct > 2.5:
          return 'high'
        elif atr_pct > 1.5:
          return 'normal'
        elif atr_pct > 0.8:
          return 'low'
        else:
          return 'very_low'

      # Fallback: анализ по Bollinger Bands
      if all(col in data.columns for col in ['bb_upper', 'bb_lower']):
        bb_width = (latest['bb_upper'] - latest['bb_lower']) / latest['close'] * 100

        if bb_width > 5:
          return 'high'
        elif bb_width > 2:
          return 'normal'
        else:
          return 'low'

    except Exception as e:
      logger.warning(f"Ошибка определения волатильности: {e}")

    return 'unknown'

  def _calculate_price_action_score(self, data: pd.DataFrame) -> float:
    """Рассчитать оценку price action"""
    try:
      # Анализ последних 10 свечей
      recent_data = data.tail(10)
      score = 0.0

      # Анализ паттернов свечей
      for i in range(len(recent_data)):
        candle = recent_data.iloc[i]

        # Размер тела свечи
        body_size = abs(candle['close'] - candle['open']) / candle['open']

        # Размер теней
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        total_range = candle['high'] - candle['low']

        if total_range > 0:
          upper_shadow_pct = upper_shadow / total_range
          lower_shadow_pct = lower_shadow / total_range

          # Оценка силы свечи
          if body_size > 0.02:  # Сильное тело
            score += 0.3

          # Доджи или spinning top
          if body_size < 0.005:
            score += 0.1

          # Hammer/Shooting star patterns
          if lower_shadow_pct > 0.6 and upper_shadow_pct < 0.2:
            score += 0.2  # Hammer
          elif upper_shadow_pct > 0.6 and lower_shadow_pct < 0.2:
            score += 0.2  # Shooting star

      # Нормализуем к диапазону 0-1
      return min(score / len(recent_data), 1.0)

    except Exception as e:
      logger.warning(f"Ошибка расчета price action score: {e}")
      return 0.0

  def _determine_market_session(self) -> str:
    """Определить текущую торговую сессию"""
    try:
      current_hour = datetime.now().hour

      # UTC время сессий
      if 22 <= current_hour or current_hour < 8:
        return 'asian'
      elif 8 <= current_hour < 16:
        return 'european'
      elif 16 <= current_hour < 22:
        return 'american'
      else:
        return 'overnight'

    except Exception as e:
      logger.warning(f"Ошибка определения сессии: {e}")
      return 'unknown'

  async def _get_correlation_data(self, symbol: str) -> Dict[str, float]:
    """Получить данные корреляции с основными активами"""
    try:
      correlation_data = {}

      # Получаем данные BTC для корреляции
      if symbol != "BTCUSDT":
        try:
          btc_data = await self.data_fetcher.get_historical_candles(
            "BTCUSDT", Timeframe.ONE_HOUR, limit=100
          )

          symbol_data = await self.data_fetcher.get_historical_candles(
            symbol, Timeframe.ONE_HOUR, limit=100
          )

          if not btc_data.empty and not symbol_data.empty:
            # Выравниваем данные по времени
            btc_returns = btc_data['close'].pct_change().dropna()
            symbol_returns = symbol_data['close'].pct_change().dropna()

            if len(btc_returns) > 10 and len(symbol_returns) > 10:
              # Обрезаем до одинакового размера
              min_length = min(len(btc_returns), len(symbol_returns))
              btc_returns = btc_returns.tail(min_length)
              symbol_returns = symbol_returns.tail(min_length)

              correlation = btc_returns.corr(symbol_returns)
              if not np.isnan(correlation):
                correlation_data['btc_correlation'] = float(correlation)

        except Exception as corr_error:
          logger.debug(f"Ошибка расчета корреляции с BTC: {corr_error}")

      return correlation_data

    except Exception as e:
      logger.warning(f"Ошибка получения данных корреляции: {e}")
      return {}

  def _calculate_liquidity_score(self, data: pd.DataFrame) -> float:
    """Рассчитать оценку ликвидности"""
    try:
      if 'volume' not in data.columns:
        return 0.0

      # Анализ объемов за последние 20 периодов
      recent_volumes = data['volume'].tail(20)

      # Средний объем
      avg_volume = recent_volumes.mean()

      # Стабильность объемов (низкая волатильность = хорошая ликвидность)
      volume_std = recent_volumes.std()
      volume_cv = volume_std / avg_volume if avg_volume > 0 else 1.0

      # Текущий объем относительно среднего
      current_volume = recent_volumes.iloc[-1]
      volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0.0

      # Оценка ликвидности (0-1)
      liquidity_score = min(volume_ratio * (1 - min(volume_cv, 1.0)), 2.0) / 2.0

      return float(liquidity_score)

    except Exception as e:
      logger.warning(f"Ошибка расчета ликвидности: {e}")
      return 0.0

  def _get_technical_levels(self, data: pd.DataFrame) -> Dict[str, float]:
    """Получить технические уровни поддержки/сопротивления"""
    try:
      levels = {}
      latest = data.iloc[-1]
      current_price = latest['close']

      # Pivot Points
      if len(data) >= 2:
        prev_candle = data.iloc[-2]
        high = prev_candle['high']
        low = prev_candle['low']
        close = prev_candle['close']

        pivot = (high + low + close) / 3
        levels['pivot_point'] = float(pivot)

        # Уровни поддержки и сопротивления
        levels['resistance_1'] = float(2 * pivot - low)
        levels['support_1'] = float(2 * pivot - high)
        levels['resistance_2'] = float(pivot + (high - low))
        levels['support_2'] = float(pivot - (high - low))

      # Простые уровни на основе максимумов/минимумов
      if len(data) >= 20:
        recent_data = data.tail(20)
        levels['recent_high'] = float(recent_data['high'].max())
        levels['recent_low'] = float(recent_data['low'].min())

      return levels

    except Exception as e:
      logger.warning(f"Ошибка получения технических уровней: {e}")
      return {}

  def _convert_rejection_reasons_to_filter_reasons(self, reasons: List[str]) -> List[FilterReason]:
    """Конвертировать причины отклонения в причины фильтрации"""
    filter_reasons = []

    for reason in reasons:
      reason_lower = reason.lower()

      if any(word in reason_lower for word in ['confidence', 'уверенность', 'certainty']):
        filter_reasons.append(FilterReason.LOW_CONFIDENCE)
      elif any(word in reason_lower for word in ['risk', 'риск', 'exposure']):
        filter_reasons.append(FilterReason.RISK_MANAGER)
      elif any(word in reason_lower for word in ['market', 'рынок', 'condition', 'условия']):
        filter_reasons.append(FilterReason.MARKET_CONDITIONS)
      elif any(word in reason_lower for word in ['position', 'позиция', 'limit', 'лимит']):
        filter_reasons.append(FilterReason.POSITION_LIMIT)
      elif any(word in reason_lower for word in ['correlation', 'корреляция']):
        filter_reasons.append(FilterReason.CORRELATION_FILTER)
      elif any(word in reason_lower for word in ['volatility', 'волатильность', 'волат']):
        filter_reasons.append(FilterReason.VOLATILITY_FILTER)
      else:
        filter_reasons.append(FilterReason.RISK_MANAGER)  # По умолчанию

    return filter_reasons if filter_reasons else [FilterReason.RISK_MANAGER]

  async def initialize_symbols_if_empty(self):
    if not self.active_symbols:
      logger.info("Список активных символов пуст, попытка повторной инициализации...")
      self.active_symbols = await self.data_fetcher.get_active_symbols_by_volume()
      if self.active_symbols:
        logger.info(f"Символы успешно реинициализированы: {self.active_symbols}")
      else:
        logger.warning("Не удалось реинициализировать символы.")

  async def periodic_regime_analysis(self):
    """Периодический анализ и экспорт статистики режимов"""
    while self.is_running:
      try:
        await asyncio.sleep(3600 * 4)  # Каждые 4 часа

        # Экспортируем статистику
        await self.export_regime_statistics()

        # Анализируем эффективность режимов
        for symbol in self.active_symbols:
          stats = self.market_regime_detector.get_regime_statistics(symbol)
          if stats and stats.get('total_observations', 0) > 100:
            logger.info(f"Статистика режимов для {symbol}:")
            logger.info(f"  Распределение: {stats.get('regime_distribution')}")
            logger.info(f"  Средние метрики: {stats.get('average_metrics')}")

      except Exception as e:
        logger.error(f"Ошибка периодического анализа режимов: {e}")


  # async def start(self):
  #   if self.is_running:
  #     logger.warning("Система уже запущена.")
  #     return
  #
  #   # ++ СИНХРОНИЗИРУЕМ ВРЕМЯ ПЕРЕД НАЧАЛОМ РАБОТЫ ++
  #   await self.connector.sync_time()
  #
  #     # Инициализация БД
  #   await self.db_manager._create_tables_if_not_exist()
  #   # await self.state_manager.initialize_state()
  #
  #   # Проверка и первичное обучение модели
  #   if not await self._ensure_model_exists():
  #     logger.critical("Не удалось создать первичную ML модель. Запуск отменен.")
  #     return
  #
  #   if not await self.initialize():
  #     logger.error("Сбой инициализации системы. Запуск отменен.")
  #     return
  #
  #   self.is_running = True
  #   # ++ СООБЩАЕМ, ЧТО БОТ ЗАПУЩЕН ++
  #   self.state_manager.set_status('running')
  #   logger.info("Торговая система запускается...")
  #   self._monitoring_task = asyncio.create_task(self._monitoring_loop())
  #   # Запускаем фоновое переобучение
  #   self._retraining_task = self.retraining_manager.start_scheduled_retraining(
  #     self.active_symbols, timeframe=Timeframe.ONE_HOUR)
  #   self._time_sync_task = asyncio.create_task(self._time_sync_loop())
  #   self.is_running = True
  #
  #   # Добавить задачу проверки ROI
  #   self._roi_check_task = asyncio.create_task(self.periodic_roi_check())
  #
  #   try:
  #     from analytics.roi_analytics import ROIAnalytics
  #     roi_analytics = ROIAnalytics(self.db_manager)
  #
  #     logger.info("=== АНАЛИТИКА ROI НАСТРОЕК ===")
  #
  #     # Анализ за последние 7 дней
  #     weekly_analysis = await roi_analytics.analyze_roi_performance(days=7)
  #     if 'error' not in weekly_analysis:
  #       logger.info(f"📊 Статистика за 7 дней:")
  #       logger.info(f"  Сделок: {weekly_analysis['total_trades']}")
  #       logger.info(f"  Винрейт: {weekly_analysis['win_rate']:.1f}%")
  #       logger.info(f"  Общий PnL: {weekly_analysis['total_pnl']:.2f}")
  #       logger.info(f"  SL срабатываний: {weekly_analysis['sl_hit_rate']:.1f}%")
  #       logger.info(f"  TP достижений: {weekly_analysis['tp_hit_rate']:.1f}%")
  #       logger.info(f"  💡 {weekly_analysis['recommendation']}")
  #
  #   except Exception as analytics_error:
  #     logger.warning(f"Не удалось загрузить ROI аналитику: {analytics_error}")
  #
  #   # Запускаем синхронизацию времени
  #   self._time_sync_task = asyncio.create_task(self._periodic_time_sync())
  #
  #   # Запускаем периодическую оценку стратегий
  #   self._evaluation_task = asyncio.create_task(self.periodic_strategy_evaluation())
  #
  #   await self.periodic_regime_analysis()
  #
  #   logger.info("Торговая система и планировщик переобучения успешно запущены.")
  #   logger.info("✅ Система успешно запущена")
  #   return True


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

    if self._evaluation_task:
      self._evaluation_task.cancel()
      with suppress(asyncio.CancelledError):
        await self._evaluation_task

    if self._time_sync_task:
      self._time_sync_task.cancel()
      with suppress(asyncio.CancelledError):
        await self._time_sync_task

    if self.shadow_trading:
      try:
        # Генерируем финальный отчет
        final_report = await self.shadow_trading.force_comprehensive_report()
        logger.info("📊 === ФИНАЛЬНЫЙ ОТЧЕТ SHADOW TRADING ===")
        logger.info(final_report)

        # Генерируем финальный отчет
        final_report = await self.shadow_trading.generate_daily_report()
        logger.info("📊 === ФИНАЛЬНЫЙ ОТЧЕТ SHADOW TRADING ===")

        overall = final_report.get('overall_performance', {})
        if overall and 'error' not in overall:
          logger.info(f"🎯 Всего сигналов: {overall.get('total_signals', 0)}")
          logger.info(f"✅ Win Rate: {overall.get('win_rate_pct', 0)}%")
          logger.info(f"💰 Общий P&L: {overall.get('total_pnl_pct', 0):+.2f}%")
          logger.info(f"⚖️ Profit Factor: {overall.get('profit_factor', 0)}")

        logger.info("=" * 50)
        await self.shadow_trading.stop_shadow_trading()
        logger.info("🌟 Shadow Trading система остановлена")

      except Exception as e:
        logger.error(f"Ошибка остановки Shadow Trading: {e}")

    # Закрываем соединения коннектора
    if self.connector:
      await self.connector.close()
    # Экспортируем финальную статистику
    if hasattr(self, 'adaptive_selector'):
      self.adaptive_selector.export_adaptation_history(
        f"logs/final_adaptation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
      )

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

      # Добавляем проверку волатильности для фильтрации шумных сигналов
      atr = ta.atr(df['high'], df['low'], df['close'], length=14)
      if atr is not None and len(atr) > 0:
        current_atr = atr.iloc[-1]
        avg_price = df['close'].mean()
        volatility_pct = (current_atr / avg_price) * 100

        # Если волатильность слишком низкая, не входим
        if volatility_pct < 0.1:  # менее 0.1%
          logger.debug(f"Волатильность слишком низкая ({volatility_pct:.3f}%), пропускаем вход")
          return False

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
# Метод ниже был в прошлой версии, его логика интегрирована в обновленный метод _check_pending_signal_for_entry
  # async def _check_and_execute_pending_signal(self, symbol: str, signal_data: dict):
  #   """Проверяет триггер для сигнала в ожидании и исполняет его."""
  #   try:
  #     signal_time = datetime.fromisoformat(signal_data['metadata'].get('signal_time'))
  #     if datetime.now() - signal_time > timedelta(hours=2):
  #       logger.warning(f"Сигнал для {symbol} просрочен и будет удален из очереди.")
  #       pending_signals = self.state_manager.get_pending_signals()
  #       pending_signals.pop(symbol, None)
  #       self.state_manager.update_pending_signals(pending_signals)
  #       return
  #
  #     strategy_settings = self.config.get('strategy_settings', {})
  #     ltf_str = strategy_settings.get('ltf_entry_timeframe', '5m')
  #
  #     timeframe_map = {"1m": Timeframe.ONE_MINUTE, "5m": Timeframe.FIVE_MINUTES, "15m": Timeframe.FIFTEEN_MINUTES}
  #     ltf_timeframe = timeframe_map.get(ltf_str, Timeframe.FIFTEEN_MINUTES)
  #
  #     logger.debug(f"Проверка триггера для {symbol} на таймфрейме {ltf_str}...")
  #     ltf_data = await self.data_fetcher.get_historical_candles(symbol, ltf_timeframe, limit=100)
  #
  #     signal_data['signal_type'] = SignalType(signal_data['signal_type'])
  #     signal_data['timestamp'] = datetime.fromisoformat(signal_data['timestamp'])
  #     signal = TradingSignal(**signal_data)
  #
  #     if self._check_ltf_entry_trigger(ltf_data, signal.signal_type):
  #       logger.info(f"✅ ТРИГГЕР НА LTF ДЛЯ {symbol} СРАБОТАЛ! Исполнение ордера...")
  #       quantity = signal.metadata.get('approved_size', 0)
  #       success, trade_details = await self.trade_executor.execute_trade(signal, symbol, quantity)
  #
  #       if success and trade_details:
  #         self.position_manager.add_position_to_cache(trade_details)
  #
  #       pending_signals = self.state_manager.get_pending_signals()
  #       pending_signals.pop(symbol, None)
  #       self.state_manager.update_pending_signals(pending_signals)
  #   except Exception as e:
  #     logger.error(f"Ошибка при обработке сигнала в ожидании для {symbol}: {e}", exc_info=True)

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

      if self.sar_strategy:
        try:
          # Запускаем первоначальное обновление символов
          initial_symbols = await self.sar_strategy.update_monitored_symbols(self.data_fetcher)
          logger.info(f"🎯 SAR стратегия готова к работе с {len(initial_symbols)} символами")

          # Сохраняем начальный статус
          sar_status = self.sar_strategy.get_strategy_status()
          self.state_manager.set_custom_data('sar_strategy_status', sar_status)

        except Exception as e:
          logger.error(f"Ошибка инициализации SAR стратегии: {e}")
      else:
        logger.warning("SAR стратегия не была инициализирована")

      logger.info("🚀 Все компоненты системы, включая SAR стратегию, готовы к работе")


      logger.info("Оптимизированная инициализация завершена")

  async def _monitoring_loop_optimized(self):
    """
    Оптимизированный мониторинг с батчингом запросов
    """
    logger.info("Запуск оптимизированного цикла мониторинга...")

    monitoring_interval = self.config.get('general_settings', {}).get('monitoring_interval_seconds', 60)
    batch_size = 5  # Обрабатываем символы батчами

    await self.position_manager.load_open_positions()

    while self.is_running:
      try:
        # Обновляем баланс один раз за цикл
        await self.update_account_balance()
        # Обновляем метрики баланса для дашборда
        if self.account_balance:
          self.state_manager.update_metrics(self.account_balance)

        # Управляем открытыми позициями
        await self.position_manager.manage_open_positions(self.account_balance)
        await self._update_dashboard_metrics()


        # Разбиваем символы на батчи для параллельной обработки
        for i in range(0, len(self.active_symbols), batch_size):
          if not self.is_running:
            break

          batch = self.active_symbols[i:i + batch_size]

          # Параллельная обработка батча символов
          tasks = []
# -------------
# # Проверяем сигналы в ожидании
#       pending_signals = self.state_manager.get_pending_signals()
#       if pending_signals:
#         tasks = [self._check_and_execute_pending_signal(s, d) for s, d in pending_signals.items()]
#         await asyncio.gather(*tasks)
#
#       # Ищем новые сигналы
#       open_and_pending = set(self.position_manager.open_positions.keys()) | set(pending_signals.keys())
#       symbols_for_new_search = [s for s in self.active_symbols if s not in open_and_pending]
#
#       if symbols_for_new_search:
#         use_enhanced = self.config.get('ml_settings', {}).get('use_enhanced_processing', True)
#         if use_enhanced and self.enhanced_ml_model:
#           tasks = [self._monitor_symbol_for_entry_enhanced(symbol) for symbol in symbols_for_new_search]
#         else:
#           tasks = [self._monitor_symbol_for_entry(symbol) for symbol in symbols_for_new_search]
#         await asyncio.gather(*tasks)
# --------------
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

        # # Управляем открытыми позициями
        # await self.position_manager.manage_open_positions(self.account_balance)
        # Сверяем закрытые сделки
        await self.position_manager.reconcile_filled_orders()
        # Обновляем состояние для дашборда
        self.state_manager.update_open_positions(self.position_manager.open_positions)

        if self.sar_strategy:
          asyncio.create_task(await self.cleanup_sar_cache_task())
          try:
            sar_status = self.sar_strategy.get_strategy_status()
            self.state_manager.set_custom_data('sar_strategy_status', sar_status)
          except Exception as e:
            logger.error(f"Ошибка обновления статуса SAR: {e}")

        if self.sar_strategy:
          asyncio.create_task(await self.update_sar_symbols_task())

        if hasattr(self, 'update_signal_outcomes'):
          await self.update_signal_outcomes()

        # Выводим статистику производительности каждые 10 циклов
        if hasattr(self, '_monitoring_cycles'):
          self._monitoring_cycles += 1
        else:
          self._monitoring_cycles = 1

        if self._monitoring_cycles % 10 == 0:
          await self._log_performance_stats()

        if self._monitoring_cycles % 20 == 0:
          await self.display_ml_statistics()

          # ======================= ИСПРАВЛЕНИЕ ЗДЕСЬ =======================
          # Этот блок был добавлен, чтобы правильно обрабатывать команды
        command_data = self.state_manager.get_command()
        if command_data:
          command_name = command_data.get('name')
          logger.info(f"Получена новая команда из дашборда: {command_name}")

          if command_name == 'generate_report':
            if self.retraining_manager:
              self.retraining_manager.export_performance_report()

          elif command_name == 'update_ml_models':
            # Обновляем состояние ML моделей
            ml_state = self.state_manager.get_custom_data('ml_models_state')
            if ml_state:
              self.use_enhanced_ml = ml_state.get('use_enhanced_ml', True)
              self.use_base_ml = ml_state.get('use_base_ml', True)
              logger.info(f"ML модели обновлены: enhanced={self.use_enhanced_ml}, base={self.use_base_ml}")

          elif command_name == 'export_regime_statistics':
            await self.export_regime_statistics()

          elif command_name == 'get_regime_statistics':
            symbol = command_data.get('data', {}).get('symbol')
            if symbol:
              stats = self.market_regime_detector.get_regime_statistics(symbol)
              self.state_manager.set_custom_data(f"regime_stats_{symbol}", stats)

          elif command_name == 'reload_sar_config':
            logger.info("🔄 Перезагрузка конфигурации SAR стратегии...")
            try:
              if self.sar_strategy:
                # Перезагружаем конфигурацию
                new_config = self.config_manager.load_config()
                new_sar_config = new_config.get('stop_and_reverse_strategy', {})

                # Обновляем параметры стратегии
                for key, value in new_sar_config.items():
                  if hasattr(self.sar_strategy, key):
                    setattr(self.sar_strategy, key, value)

                # Обновляем статус в state_manager
                sar_status = self.sar_strategy.get_strategy_status()
                self.state_manager.set_custom_data('sar_strategy_status', sar_status)

                logger.info("✅ Конфигурация SAR стратегии перезагружена")
              else:
                logger.warning("SAR стратегия не инициализирована")
            except Exception as e:
              logger.error(f"Ошибка перезагрузки SAR конфигурации: {e}")

          # Очищаем команду после выполнения
          self.state_manager.clear_command()

        # Ожидание перед следующим циклом
        await asyncio.sleep(monitoring_interval)

      except asyncio.CancelledError:
        logger.info("Мониторинг остановлен по запросу")
        break
      except Exception as e:
        logger.error(f"Ошибка в оптимизированном цикле мониторинга: {e}", exc_info=True)
        await asyncio.sleep(monitoring_interval)

  async def _check_pending_signal_for_entry(self, symbol: str):
    """Проверяет ожидающий сигнал на точку входа с использованием продвинутой логики"""
    pending_signals = self.state_manager.get_pending_signals()

    if symbol not in pending_signals:
      return

    try:
      signal_data = pending_signals[symbol]

      # Проверяем таймаут сигнала (2 часа вместо 30 минут для большей гибкости)
      signal_time = datetime.fromisoformat(signal_data['metadata']['signal_time'])
      if (datetime.now() - signal_time) > timedelta(hours=2):
        logger.info(f"Сигнал для {symbol} устарел, удаляем из очереди")
        del pending_signals[symbol]
        self.state_manager.update_pending_signals(pending_signals)
        return

      # Получаем настройки LTF
      strategy_settings = self.config.get('strategy_settings', {})
      ltf_str = strategy_settings.get('ltf_entry_timeframe', '5m')

      timeframe_map = {
        "1m": Timeframe.ONE_MINUTE,
        "5m": Timeframe.FIVE_MINUTES,
        "15m": Timeframe.FIFTEEN_MINUTES
      }
      ltf_timeframe = timeframe_map.get(ltf_str, Timeframe.FIVE_MINUTES)

      # Получаем данные LTF с достаточной историей для индикаторов
      logger.debug(f"Проверка триггера для {symbol} на таймфрейме {ltf_str}...")
      ltf_data = await self.data_fetcher.get_historical_candles(symbol, ltf_timeframe, limit=100)

      if ltf_data.empty or len(ltf_data) < 30:
        logger.debug(f"Недостаточно данных LTF для {symbol}")
        return

      # Восстанавливаем TradingSignal из данных
      signal_type = SignalType[signal_data['signal_type']]

      # Используем продвинутую проверку триггера
      if self._check_ltf_entry_trigger(ltf_data, signal_type):
        logger.info(f"✅ ТРИГГЕР LTF для {symbol} сработал! Исполняем сделку...")

        # Восстанавливаем полный TradingSignal
        trading_signal = TradingSignal(
          signal_type=signal_type,
          symbol=signal_data['symbol'],
          price=signal_data['price'],
          confidence=signal_data['confidence'],
          strategy_name=signal_data['strategy_name'],
          timestamp=datetime.fromisoformat(signal_data['timestamp']),
          stop_loss=signal_data.get('stop_loss'),
          take_profit=signal_data.get('take_profit'),
          metadata=signal_data.get('metadata', {})
        )

        # Исполняем сделку
        size = signal_data['metadata']['approved_size']
        success, order_details = await self.trade_executor.execute_trade(
          trading_signal, symbol, size
        )

        if success:
          logger.info(f"✅ Сделка по {symbol} успешно исполнена через LTF триггер")
          if order_details:
            self.position_manager.add_position_to_cache(order_details)

          # Удаляем из очереди
          del pending_signals[symbol]
          self.state_manager.update_pending_signals(pending_signals)

          # Синхронизируем с Shadow Trading
          if self.shadow_trading and order_details:
            asyncio.create_task(
              self.shadow_trading.signal_tracker.sync_with_real_trades(
                symbol,
                {
                  'open_price': order_details.get('open_price'),
                  'close_price': order_details.get('open_price'),  # При открытии
                  'profit_loss': 0,  # Пока 0
                  'profit_pct': 0
                }
              )
            )
        else:
          logger.error(f"❌ Не удалось исполнить сделку по {symbol}")
      else:
        # Логируем детали для отладки
        logger.debug(f"Триггер LTF для {symbol} пока не сработал, продолжаем мониторинг")

    except Exception as e:
      logger.error(f"Ошибка при проверке точки входа для {symbol}: {e}", exc_info=True)

  # async def _check_ltf_entry_conditions(self, signal_data: Dict, ltf_data: pd.DataFrame) -> bool:
  #   """Проверяет условия входа на младшем таймфрейме"""
  #   try:
  #     signal_type = SignalType[signal_data['signal_type']]
  #
  #     # Пример простых условий входа
  #     last_close = ltf_data['close'].iloc[-1]
  #     sma_10 = ltf_data['close'].rolling(10).mean().iloc[-1]
  #
  #     if signal_type == SignalType.BUY:
  #       # Для покупки: цена выше SMA10
  #       return last_close > sma_10
  #     else:  # SELL
  #       # Для продажи: цена ниже SMA10
  #       return last_close < sma_10
  #
  #   except Exception as e:
  #     logger.error(f"Ошибка при проверке LTF условий: {e}")
  #     return False


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

      # Инициализация БД
      await self.db_manager._create_tables_if_not_exist()

      # Проверяем БД перед запуском
      health = await self.db_monitor.check_database_health()
      if health['status'] != 'healthy':
        logger.warning(f"БД не в оптимальном состоянии при запуске: {health}")
      else:
        logger.info("✅ БД проверена, состояние нормальное")

      # Установка плеча для всех символов параллельно
      # leverage = self.config.get('trade_settings', {}).get('leverage', 10)
      # logger.info(f"Установка плеча {leverage} для {len(self.active_symbols)} символов...")
      #
      # successful_leverages = 0
      # for i, symbol in enumerate(self.active_symbols):
      #   try:
      #     result = await self.connector.set_leverage(symbol, leverage, leverage)
      #     if result:
      #       successful_leverages += 1
      #
      #     # Задержка между запросами для избежания rate limit
      #     if i < len(self.active_symbols) - 1:
      #       await asyncio.sleep(0.2)  # 200мс между запросами
      #
      #   except Exception as e:
      #     logger.warning(f"Не удалось установить плечо для {symbol}: {e}")
      #
      # logger.info(f"Плечо установлено для {successful_leverages}/{len(self.active_symbols)} символов")
        # Групповая установка плеча для всех символов
      leverage = self.config.get('trade_settings', {}).get('leverage', 10)
      logger.info(f"Установка плеча {leverage} для {len(self.active_symbols)} символов...")

      await self._set_leverage_for_all_symbols(leverage)

      if not await self._ensure_model_exists():
        logger.critical("Не удалось создать первичную ML модель. Запуск отменен.")
        return

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

      # Дополнительная задача синхронизации для совместимости
      self._time_sync_loop_task = asyncio.create_task(self._time_sync_loop())

      # Периодическая очистка кэшей
      self._cache_cleanup_task = asyncio.create_task(self.cleanup_caches())

      self._correlation_task = asyncio.create_task(self._update_portfolio_correlations())

      # Запускаем периодическую оценку стратегий
      self._evaluation_task = asyncio.create_task(self.periodic_strategy_evaluation())

      # await self.periodic_regime_analysis()

      # Обновление статуса
      self.state_manager.set_status('running')

      # Добавить задачу проверки ROI
      self._roi_check_task = asyncio.create_task(self.periodic_roi_check())

      monitoring_task = asyncio.create_task(self._database_monitoring_loop())
      self._monitoring_tasks.append(monitoring_task)

      logger.info("🚀 Система запущена с мониторингом БД")

      try:
        from analytics.roi_analytics import ROIAnalytics
        roi_analytics = ROIAnalytics(self.db_manager)

        logger.info("=== АНАЛИТИКА ROI НАСТРОЕК ===")

        # Анализ за последние 7 дней
        weekly_analysis = await roi_analytics.analyze_roi_performance(days=7)
        if 'error' not in weekly_analysis:
          logger.info(f"📊 Статистика за 7 дней:")
          logger.info(f"  Сделок: {weekly_analysis['total_trades']}")
          logger.info(f"  Винрейт: {weekly_analysis['win_rate']:.1f}%")
          logger.info(f"  Общий PnL: {weekly_analysis['total_pnl']:.2f}")
          logger.info(f"  SL срабатываний: {weekly_analysis['sl_hit_rate']:.1f}%")
          logger.info(f"  TP достижений: {weekly_analysis['tp_hit_rate']:.1f}%")
          logger.info(f"  💡 {weekly_analysis['recommendation']}")

      except Exception as analytics_error:
        logger.warning(f"Не удалось загрузить ROI аналитику: {analytics_error}")

      logger.info("✅ Торговая система успешно запущена в оптимизированном режиме")

    except Exception as e:
      logger.critical(f"Критическая ошибка при запуске системы: {e}", exc_info=True)
      self.is_running = False
      raise

  async def _set_leverage_for_all_symbols(self, leverage: int):
    """Устанавливает плечо для всех символов с оптимизацией и предотвращением дублирования"""
    if not self.active_symbols:
      logger.warning("Нет активных символов для установки плеча")
      return

    # Проверяем, нужно ли устанавливать плечо
    already_set = getattr(self, '_leverage_already_set', set())
    symbols_to_set = [s for s in self.active_symbols if s not in already_set]

    if not symbols_to_set:
      logger.info("Плечо уже установлено для всех активных символов")
      return

    successful_leverages = 0

    # Устанавливаем плечо батчами по 10 символов
    batch_size = 10
    for i in range(0, len(symbols_to_set), batch_size):
      batch = symbols_to_set[i:i + batch_size]

      # Параллельная установка плеча для батча
      tasks = []
      for symbol in batch:
        tasks.append(self._set_single_leverage(symbol, leverage))

      results = await asyncio.gather(*tasks, return_exceptions=True)

      # Подсчитываем успешные установки
      for j, result in enumerate(results):
        symbol = batch[j]
        if isinstance(result, Exception):
          logger.warning(f"Ошибка установки плеча для {symbol}: {result}")
        elif result:
          successful_leverages += 1
          already_set.add(symbol)

      # Задержка между батчами
      if i + batch_size < len(symbols_to_set):
        await asyncio.sleep(1.0)  # 1 секунда между батчами

    # Сохраняем состояние установленных плеч
    self._leverage_already_set = already_set

    logger.info(f"Плечо установлено для {successful_leverages}/{len(symbols_to_set)} новых символов")
    logger.info(f"Всего символов с установленным плечом: {len(already_set)}")

  async def _set_single_leverage(self, symbol: str, leverage: int) -> bool:
    """Устанавливает плечо для одного символа"""
    try:
      result = await self.connector.set_leverage(symbol, leverage, leverage)
      if result:
        logger.debug(f"Плечо {leverage}x установлено для {symbol}")
        return True
      else:
        logger.warning(f"Не удалось установить плечо для {symbol}")
        return False
    except Exception as e:
      logger.error(f"Ошибка установки плеча для {symbol}: {e}")
      return False

  async def _database_monitoring_loop(self):
      """Цикл мониторинга БД"""
      while self.is_running:
        try:
          await asyncio.sleep(300)  # Проверка каждые 5 минут

          health = await self.db_monitor.check_database_health()

          # Логируем статистику
          stats = health.get('stats', {})
          if stats.get('total_operations', 0) > 0:
            error_rate = (stats.get('failed_operations', 0) / stats['total_operations']) * 100
            lock_rate = (stats.get('lock_errors', 0) / stats['total_operations']) * 100

            logger.info(f"📊 Статистика БД: операций={stats['total_operations']}, "
                        f"ошибок={error_rate:.1f}%, блокировок={lock_rate:.1f}%")

          # Алерты при проблемах
          if health['status'] != 'healthy':
            logger.warning(f"⚠️ Проблемы с БД: {health['message']}")

          if stats.get('lock_errors', 0) > 50:
            logger.error(f"🚨 Критическое количество блокировок БД: {stats['lock_errors']}")

        except Exception as e:
          logger.error(f"Ошибка в цикле мониторинга БД: {e}")
          await asyncio.sleep(60)

  async def get_system_health(self) -> Dict[str, Any]:
    """Получить общее состояние системы включая БД"""
    try:
      db_health = await self.db_monitor.check_database_health()

      return {
        'system_status': 'running' if self.is_running else 'stopped',
        'database': db_health,
        'uptime_seconds': time.time() - self.start_time if hasattr(self, 'start_time') else 0,
        'active_components': {
          'data_fetcher': hasattr(self, 'data_fetcher'),
          'trade_executor': hasattr(self, 'trade_executor'),
          'risk_manager': hasattr(self, 'risk_manager'),
          'shadow_trading': hasattr(self, 'shadow_trading')
        }
      }

    except Exception as e:
      logger.error(f"Ошибка получения состояния системы: {e}")
      return {'error': str(e)}

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
    # --- НОВЫЙ БЛОК: ИНФОРМАЦИОННОЕ ЛОГИРОВАНИЕ ROI ЦЕЛЕЙ ---
    try:
      signal_logger.info(f"====== СИГНАЛ ДЛЯ {symbol} ПОЛУЧЕН ({signal.strategy_name}) ======")
      signal_logger.info(f"Тип: {signal.signal_type.value}, Уверенность: {signal.confidence:.2f}, Цена: {signal.price}")

      roi_targets = self.risk_manager.convert_roi_to_price_targets(
        entry_price=signal.price,
        signal_type=signal.signal_type
      )
      if roi_targets:
        signal_logger.info(f"ROI ЦЕЛИ для {symbol}:")
        signal_logger.info(
          f"  SL: {roi_targets['stop_loss']['price']:.6f} (ROI: {roi_targets['stop_loss']['roi_pct']:.1f}%)")
        signal_logger.info(
          f"  TP: {roi_targets['take_profit']['price']:.6f} (ROI: {roi_targets['take_profit']['roi_pct']:.1f}%)")
        signal_logger.info(f"  Risk/Reward: 1:{roi_targets['risk_reward_ratio']:.2f}")

    except Exception as roi_error:
      logger.warning(f"Ошибка получения ROI информации для {symbol}: {roi_error}")
    # --- КОНЕЦ НОВОГО БЛОКА ---


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

    recommendations: list[str] = []

    if excellent_wr > 70:
      recommendations.append("Отличные сигналы показывают высокую эффективность - увеличьте размеры позиций для них")

    if fair_wr > good_wr:
      recommendations.append("⚠️ Сигналы среднего качества работают лучше хороших - проверьте настройки оценки")

    avg_wr = np.mean([r.get('win_rate', 0) for r in results.values() if r])
    if avg_wr < 50:
      recommendations.append("Общий win rate ниже 50% - рекомендуется повысить минимальный порог качества")

    # Анализируем проблемы и даем рекомендации
    if any("отсутствующих значений" in issue for issue in results):
      recommendations.append(
        "• Рассмотрите использование более продвинутых методов заполнения пропусков "
        "(интерполяция, forward-fill с ограничением)"
      )

    if any("дубликатов" in issue for issue in results):
      recommendations.append(
        "• Проверьте источник данных на предмет дублирования при загрузке"
      )

    if any("выбросов" in issue for issue in results):
      recommendations.append(
        "• Примените робастное масштабирование или винсоризацию для уменьшения влияния выбросов"
      )

    if any("объем данных" in issue.lower() for issue in results):
      recommendations.append(
        "• Увеличьте период загрузки исторических данных или добавьте больше символов"
      )

    if any("несбалансированность" in issue for issue in results):
      recommendations.append(
        "• Используйте техники балансировки классов (SMOTE, undersampling) или "
        "настройте веса классов в модели"
      )

    # Общие рекомендации
    recommendations.extend([
      "• Проверьте корректность временных меток и отсутствие пропусков во временном ряде",
      "• Убедитесь, что все технические индикаторы рассчитаны корректно",
      "• Рассмотрите добавление дополнительных признаков для улучшения качества модели"
    ])

    if recommendations:
      return "Рекомендации по улучшению данных:\n" + "\n".join(recommendations)
    else:
      return "Система работает в нормальном режиме. Рекомендации на основе анализа качества."


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


  async def process_trade_feedback(self, symbol: str, trade_id: int, trade_result: Dict[str, Any]):
      """
      Обрабатывает обратную связь после закрытия сделки

      Args:
          symbol: Торговый символ
          trade_id: ID сделки
          trade_result: Результаты сделки (profit_loss, strategy_name, etc.)
      """
      try:
        strategy_name = trade_result.get('strategy_name')

        # 1. Обновляем производительность стратегии
        if hasattr(self, 'adaptive_selector'):
          self.adaptive_selector.update_strategy_performance(strategy_name, trade_result)

        # # 2. Сохраняем данные для переобучения ML
        if self.retraining_manager:
          await self.retraining_manager.record_trade_result(symbol, trade_result)

        # 3. Обновляем веса в Enhanced ML (если используется)
        if self.use_enhanced_ml and self.enhanced_ml_model:
          # Сохраняем результат для будущего переобучения
          feedback_data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'features': trade_result.get('entry_features', {}),
            'actual_outcome': 1 if trade_result['profit_loss'] > 0 else 0,
            'predicted_outcome': trade_result.get('predicted_signal'),
            'confidence': trade_result.get('confidence')
          }

          # Можно сохранить в отдельную таблицу для анализа
          self._save_ml_feedback(feedback_data)

        # 4. Адаптируем параметры риска на основе результатов
        await self._adapt_risk_parameters(symbol, trade_result)

        logger.info(f"Обработана обратная связь для сделки {trade_id}: "
                    f"стратегия={strategy_name}, результат={trade_result['profit_loss']:.2f}")

      except Exception as e:
        logger.error(f"Ошибка обработки обратной связи для сделки {trade_id}: {e}")

  def _save_ml_feedback(self, feedback_data: Dict[str, Any]):
    """Сохраняет обратную связь для ML моделей"""
    try:
      # Создаем таблицу если не существует
      self.db_manager.conn.execute("""
              CREATE TABLE IF NOT EXISTS ml_feedback (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  symbol TEXT NOT NULL,
                  timestamp DATETIME NOT NULL,
                  features TEXT,
                  actual_outcome INTEGER,
                  predicted_outcome INTEGER,
                  confidence REAL,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
              )
          """)

      self.db_manager.conn.execute("""
              INSERT INTO ml_feedback 
              (symbol, timestamp, features, actual_outcome, predicted_outcome, confidence)
              VALUES (?, ?, ?, ?, ?, ?)
          """, (
        feedback_data['symbol'],
        feedback_data['timestamp'],
        json.dumps(feedback_data['features']),
        feedback_data['actual_outcome'],
        feedback_data['predicted_outcome'],
        feedback_data['confidence']
      ))

      self.db_manager.conn.commit()

    except Exception as e:
      logger.error(f"Ошибка сохранения ML feedback: {e}")

  async def _adapt_risk_parameters(self, symbol: str, trade_result: Dict[str, Any]):
    """Адаптирует параметры риска на основе результатов"""
    # Анализируем результаты последних N сделок
    recent_trades = self.db_manager.get_recent_closed_trades(symbol, limit=20)

    if len(recent_trades) >= 10:
      wins = sum(1 for t in recent_trades if t['profit_loss'] > 0)
      win_rate = wins / len(recent_trades)

      # Адаптируем max_positions на основе win rate
      current_max_positions = self.config.get('risk_management', {}).get('max_positions_per_symbol', 3)

      if win_rate > 0.65:  # Хорошая производительность
        new_max_positions = min(current_max_positions + 1, 5)
      elif win_rate < 0.35:  # Плохая производительность
        new_max_positions = max(current_max_positions - 1, 1)
      else:
        new_max_positions = current_max_positions

      if new_max_positions != current_max_positions:
        self.config['risk_management']['max_positions_per_symbol'] = new_max_positions
        logger.info(f"Адаптирован max_positions для {symbol}: {current_max_positions} -> {new_max_positions}")

  async def periodic_strategy_evaluation(self):
    """Периодическая оценка и адаптация стратегий"""
    while self.is_running:
      try:
        await asyncio.sleep(3600)  # Каждый час

        if hasattr(self, 'adaptive_selector'):
          # Отключаем плохо работающие стратегии
          self.adaptive_selector.disable_poorly_performing_strategies()

          # Получаем сводку производительности
          performance = self.adaptive_selector.get_performance_summary()

          logger.info("Периодическая оценка стратегий:")
          for strategy, metrics in performance.items():
            logger.info(f"  {strategy}: активна={metrics['active']}, "
                        f"вес={metrics['weight']:.2f}, WR={metrics['win_rate']:.2f}")

            # Экспортируем историю адаптаций
            self.adaptive_selector.export_adaptation_history(
              f"logs/adaptation_history_{datetime.now().strftime('%Y%m%d')}.csv"
            )

      except Exception as e:
        logger.error(f"Ошибка периодической оценки стратегий: {e}")


  async def check_strategy_adaptation(self, symbol: str):
      """Проверяет необходимость адаптации стратегий для символа"""
      if not hasattr(self, 'market_regime_detector'):
        return

      # Получаем предыдущий режим из истории
      previous_regime = None
      if symbol in self.market_regime_detector.regime_history:
        history = list(self.market_regime_detector.regime_history[symbol])
        if len(history) >= 2:
          previous_regime = history[-2].primary_regime

      # Проверяем необходимость адаптации
      should_adapt, reason = self.market_regime_detector.should_adapt_strategy(
        symbol, previous_regime
      )

      if should_adapt:
        logger.info(f"Адаптация стратегий для {symbol}: {reason}")

        # Получаем статистику режимов
        stats = self.market_regime_detector.get_regime_statistics(symbol)

        # Корректируем веса стратегий на основе статистики
        if hasattr(self, 'adaptive_selector') and stats:
          regime_distribution = stats.get('regime_distribution', {})
          # Увеличиваем вес стратегий, которые хорошо работают в частых режимах
          for regime_name, count in regime_distribution.items():
            if count > 10:  # Если режим встречался часто
              recommended_strategies = self.market_regime_detector.regime_parameters.get(
                MarketRegime(regime_name),
                self.market_regime_detector.regime_parameters[MarketRegime.RANGING]
              ).recommended_strategies

              for strategy in recommended_strategies:
                if strategy in self.adaptive_selector.strategy_performance:
                  self.adaptive_selector._adapt_strategy_weight(strategy)

  async def export_regime_statistics(self):
    """Экспортирует статистику режимов для всех символов"""
    export_dir = "logs/regime_statistics"
    os.makedirs(export_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for symbol in self.active_symbols:
      if symbol in self.market_regime_detector.regime_history:
        filepath = f"{export_dir}/{symbol}_regimes_{timestamp}.csv"
        self.market_regime_detector.export_regime_data(symbol, filepath)

    logger.info(f"Статистика режимов экспортирована в {export_dir}")

  async def periodic_roi_check(self):
    """
    Периодическая проверка ROI настроек (каждые 24 часа)
    """
    while self.is_running:
      try:
        await asyncio.sleep(24 * 60 * 60)  # 24 часа

        logger.info("=== ПЕРИОДИЧЕСКАЯ ПРОВЕРКА ROI НАСТРОЕК ===")

        # Проверяем актуальность настроек
        validation = self.risk_manager.validate_roi_parameters()

        if validation['warnings']:
          logger.warning("Обнаружены предупреждения в ROI настройках:")
          for warning in validation['warnings']:
            logger.warning(f"  ⚠️  {warning}")

        # Выводим краткий отчет
        roi_report = self.risk_manager.get_roi_summary_report()
        logger.info("Текущие ROI настройки:")
        for line in roi_report.split('\n')[:10]:  # Первые 10 строк
          if line.strip():
            logger.info(line)

      except Exception as e:
        logger.error(f"Ошибка периодической проверки ROI: {e}")

