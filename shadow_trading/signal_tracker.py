# shadow_trading/signal_tracker.py
import time

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
import json

from core.schemas import TradingSignal
from core.enums import SignalType
from data.database_manager import AdvancedDatabaseManager
from utils.logging_config import get_logger

logger = get_logger(__name__)


class SignalOutcome(Enum):
  """Результаты сигналов"""
  PENDING = "pending"
  PROFITABLE = "profitable"
  LOSS = "loss"
  BREAKEVEN = "breakeven"
  TIMEOUT = "timeout"
  CANCELLED = "cancelled"


class FilterReason(Enum):
  """Причины фильтрации сигналов"""
  LOW_CONFIDENCE = "low_confidence"
  HIGH_RISK = "high_risk"
  MARKET_CONDITIONS = "market_conditions"
  POSITION_LIMIT = "position_limit"
  SYMBOL_BLACKLIST = "symbol_blacklist"
  TIME_RESTRICTION = "time_restriction"
  INSUFFICIENT_VOLUME = "insufficient_volume"
  TECHNICAL_FILTER = "technical_filter"
  ML_FILTER = "ml_filter"
  RISK_MANAGEMENT = "risk_management"


@dataclass
class SignalAnalysis:
  """Полный анализ сигнала"""
  signal_id: str
  symbol: str
  signal_type: SignalType
  entry_price: float
  entry_time: datetime
  confidence: float
  source: str  # "ml_model", "technical", "pattern", "combined"

  # Метаданные сигнала
  indicators_triggered: List[str] = field(default_factory=list)
  ml_prediction_data: Dict[str, Any] = field(default_factory=dict)
  market_regime: str = ""
  volatility_level: str = ""

  # Фильтрация
  was_filtered: bool = False
  filter_reasons: List[FilterReason] = field(default_factory=list)

  # Результаты отслеживания
  outcome: SignalOutcome = SignalOutcome.PENDING
  exit_price: Optional[float] = None
  exit_time: Optional[datetime] = None
  profit_loss_pct: Optional[float] = None
  profit_loss_usdt: Optional[float] = None

  # Статистика движения цены
  max_favorable_excursion_pct: float = 0.0  # Максимальное движение в правильную сторону
  max_adverse_excursion_pct: float = 0.0  # Максимальное движение против
  time_to_target: Optional[timedelta] = None
  time_to_max_profit: Optional[timedelta] = None

  # Дополнительная статистика
  volume_at_signal: float = 0.0
  price_action_score: float = 0.0
  created_at: datetime = field(default_factory=datetime.now)
  updated_at: datetime = field(default_factory=datetime.now)


class SignalTracker:
  """Трекер для сохранения и анализа всех сигналов"""

  def __init__(self, db_manager: AdvancedDatabaseManager, shadow_config: dict = None):
    self.db_manager = db_manager
    self.tracked_signals: Dict[str, SignalAnalysis] = {}

    self._pending_operations = asyncio.Queue()
    self._batch_size = 10
    self._batch_timeout = 5.0  # Секунд

    if shadow_config is None:
      shadow_config = self._load_shadow_config()

    self.config = shadow_config
    self._apply_config_settings()

    # asyncio.create_task(self._batch_processor())
    self._batch_processor_task = None
    self._initialize_batch_processor()

  def _initialize_batch_processor(self):
    """Безопасная инициализация batch processor"""
    try:
      # Проверяем, есть ли запущенный event loop
      try:
        loop = asyncio.get_running_loop()
        # Если loop запущен, создаем задачу
        self._batch_processor_task = loop.create_task(self._batch_processor())
      except RuntimeError:
        # Если нет запущенного loop, откладываем инициализацию
        logger.debug("Event loop не запущен, batch processor будет инициализирован позже")
        self._batch_processor_task = None
    except Exception as e:
      logger.warning(f"Не удалось инициализировать batch processor: {e}")
      self._batch_processor_task = None

  # ДОБАВИТЬ МЕТОД ДЛЯ ЛЕНИВОЙ ИНИЦИАЛИЗАЦИИ:
  async def _ensure_batch_processor(self):
    """Убеждается что batch processor запущен"""
    if self._batch_processor_task is None or self._batch_processor_task.done():
      try:
        self._batch_processor_task = asyncio.create_task(self._batch_processor())
        logger.debug("Batch processor инициализирован")
      except Exception as e:
        logger.error(f"Ошибка инициализации batch processor: {e}")

  async def _batch_processor(self):
    """Пакетная обработка операций БД для уменьшения блокировок"""
    batch = []
    last_process_time = time.time()

    while True:
      try:
        # Получаем операции с таймаутом
        try:
          operation = await asyncio.wait_for(
            self._pending_operations.get(),
            timeout=1.0
          )
          batch.append(operation)
        except asyncio.TimeoutError:
          pass

        current_time = time.time()

        # Обрабатываем пакет если он достаточно большой или прошло время
        if (len(batch) >= self._batch_size or
            (batch and current_time - last_process_time > self._batch_timeout)):
          await self._process_batch(batch)
          batch.clear()
          last_process_time = current_time

        await asyncio.sleep(0.1)

      except Exception as e:
        logger.error(f"Ошибка в пакетном процессоре: {e}")
        await asyncio.sleep(1)

  # async def _process_batch(self, batch: List[Dict]):
  #   """Обработка пакета операций"""
  #   if not batch:
  #     return
  #
  #   try:
  #     # Группируем операции по типу
  #     inserts = []
  #     updates = []
  #
  #     for operation in batch:
  #       if operation['type'] == 'insert':
  #         inserts.append(operation)
  #       elif operation['type'] == 'update':
  #         updates.append(operation)
  #
  #     # Пакетная вставка
  #     if inserts:
  #       await self._batch_insert_signals(inserts)
  #
  #     # Пакетное обновление
  #     if updates:
  #       await self._batch_update_signals(updates)
  #
  #     logger.debug(f"Обработан пакет: {len(inserts)} вставок, {len(updates)} обновлений")
  #
  #   except Exception as e:
  #     logger.error(f"Ошибка обработки пакета операций: {e}")
  async def _process_batch(self, batch: List):
    """Обработка пакета операций с правильной обработкой типов"""
    if not batch:
      return

    try:
      # Группируем операции по типу
      inserts = []
      updates = []

      for operation in batch:
        # Обрабатываем разные форматы операций
        if isinstance(operation, tuple) and len(operation) >= 2:
          # Формат: (operation_type, data)
          op_type = operation[0]
          op_data = operation[1]

          if op_type == 'save_signal':
            inserts.append({
              'type': 'insert',
              'analysis': op_data
            })
          elif op_type == 'update_signal':
            updates.append({
              'type': 'update',
              'analysis': op_data
            })
        elif isinstance(operation, dict):
          # Новый формат с явным типом
          if operation.get('type') == 'insert':
            inserts.append(operation)
          elif operation.get('type') == 'update':
            updates.append(operation)
        else:
          logger.warning(f"Неизвестный формат операции: {type(operation)}")

      # Пакетная вставка
      if inserts:
        await self._batch_insert_signals(inserts)

      # Пакетное обновление
      if updates:
        await self._batch_update_signals(updates)

      logger.debug(f"Обработан пакет: {len(inserts)} вставок, {len(updates)} обновлений")

    except Exception as e:
      logger.error(f"Ошибка обработки пакета операций: {e}")
      # При ошибке пытаемся обработать каждую операцию отдельно
      for operation in batch:
        try:
          if isinstance(operation, tuple) and operation[0] == 'save_signal':
            await self._direct_save_signal(operation[1])
          elif isinstance(operation, dict) and operation.get('type') == 'insert':
            await self._direct_save_signal(operation['analysis'])
        except Exception as individual_error:
          logger.error(f"Ошибка обработки отдельной операции: {individual_error}")



  async def _batch_insert_signals(self, insert_operations: List[Dict]):
    """Пакетная вставка сигналов"""
    try:
      if not insert_operations:
        return

      # Подготавливаем данные для пакетной вставки
      query = """
            INSERT INTO signal_analysis (
                signal_id, symbol, signal_type, entry_price, entry_time, confidence, source,
                indicators_triggered, ml_prediction_data, market_regime, volatility_level,
                was_filtered, filter_reasons, volume_at_signal, price_action_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

      params_list = []
      for operation in insert_operations:
        analysis = operation['analysis']
        params = (
          analysis.signal_id, analysis.symbol, analysis.signal_type.value,
          analysis.entry_price, analysis.entry_time, analysis.confidence, analysis.source,
          json.dumps(analysis.indicators_triggered), json.dumps(analysis.ml_prediction_data),
          analysis.market_regime, analysis.volatility_level,
          analysis.was_filtered, json.dumps([r.value for r in analysis.filter_reasons]),
          analysis.volume_at_signal, analysis.price_action_score
        )
        params_list.append(params)

      # Выполняем пакетную вставку
      await self.db_manager._execute_many(query, params_list)
      logger.debug(f"Пакетная вставка {len(params_list)} сигналов выполнена")

    except Exception as e:
      logger.error(f"Ошибка пакетной вставки сигналов: {e}")

  async def _batch_update_signals(self, update_operations: List[Dict]):
    """Пакетное обновление сигналов"""
    try:
      if not update_operations:
        return

      # Для обновлений выполняем по одному (они обычно редкие)
      for operation in update_operations:
        analysis = operation['analysis']

        time_to_target_seconds = int(analysis.time_to_target.total_seconds()) if analysis.time_to_target else None
        time_to_max_profit_seconds = int(
          analysis.time_to_max_profit.total_seconds()) if analysis.time_to_max_profit else None

        await self.db_manager._execute(
          """UPDATE signal_analysis SET
              was_filtered = ?, filter_reasons = ?, outcome = ?, exit_price = ?, exit_time = ?,
              profit_loss_pct = ?, max_favorable_excursion_pct = ?, max_adverse_excursion_pct = ?,
              time_to_target_seconds = ?, time_to_max_profit_seconds = ?, updated_at = ?
          WHERE signal_id = ?""",
          (
            analysis.was_filtered, json.dumps([r.value for r in analysis.filter_reasons]),
            analysis.outcome.value, analysis.exit_price, analysis.exit_time,
            analysis.profit_loss_pct, analysis.max_favorable_excursion_pct,
            analysis.max_adverse_excursion_pct, time_to_target_seconds,
            time_to_max_profit_seconds, analysis.updated_at, analysis.signal_id
          )
        )

      logger.debug(f"Пакетное обновление {len(update_operations)} сигналов выполнено")

    except Exception as e:
      logger.error(f"Ошибка пакетного обновления сигналов: {e}")

  async def _save_signal_to_db(self, analysis: SignalAnalysis):
    """Сохраняет сигнал в БД"""
    try:
      # Убеждаемся что batch processor запущен
      await self._ensure_batch_processor()
      operation = {
        'type': 'insert',  # Указываем правильный тип операции
        'analysis': analysis,
        'timestamp': time.time()
      }
      # Остальной код метода остается без изменений
      await self._pending_operations.put(('save_signal', analysis))

    except Exception as e:
      logger.error(f"Ошибка добавления операции в очередь: {e}")
      # Fallback - прямое сохранение в БД
      await self._direct_save_signal(analysis)

  async def _direct_save_signal(self, analysis: SignalAnalysis):
    """Прямое сохранение в БД без очереди (fallback)"""
    try:
      query = """
        INSERT OR REPLACE INTO signal_analysis 
        (signal_id, symbol, signal_type, entry_price, entry_time, confidence, source,
         indicators_triggered, ml_prediction_data, market_regime, volatility_level,
         was_filtered, filter_reasons, outcome, exit_price, exit_time, profit_loss_pct,
         max_favorable_excursion_pct, max_adverse_excursion_pct, volume_at_signal, price_action_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      """

      params = (
        analysis.signal_id, analysis.symbol, analysis.signal_type.value,
        analysis.entry_price, analysis.entry_time, analysis.confidence, analysis.source,
        json.dumps(analysis.indicators_triggered), json.dumps(analysis.ml_prediction_data),
        analysis.market_regime, analysis.volatility_level, analysis.was_filtered,
        json.dumps([r.value for r in analysis.filter_reasons]) if analysis.filter_reasons else None,
        analysis.outcome, analysis.exit_price, analysis.exit_time, analysis.profit_loss_pct,
        analysis.max_favorable_excursion_pct, analysis.max_adverse_excursion_pct,
        analysis.volume_at_signal, analysis.price_action_score
      )

      await self.db_manager._execute(query, params)
      logger.debug(f"Сигнал {analysis.signal_id} сохранен напрямую в БД")

    except Exception as e:
      logger.error(f"Ошибка прямого сохранения сигнала: {e}")

  async def _update_signal_in_db_async(self, analysis: SignalAnalysis):
    """Асинхронное обновление через очередь"""
    operation = {
      'type': 'update',
      'analysis': analysis,
      'timestamp': time.time()
    }
    await self._pending_operations.put(operation)


  def _load_shadow_config(self):
      """Загружает конфигурацию Shadow Trading из папки config"""
      try:
        import json
        import os

        config_path = "config/enhanced_shadow_trading_config.json"

        if not os.path.exists(config_path):
          logger.warning(f"Файл {config_path} не найден, используем настройки по умолчанию")
          return {}

        with open(config_path, 'r', encoding='utf-8') as f:
          config = json.load(f)
          return config.get('enhanced_shadow_trading', {})

      except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации SignalTracker: {e}")
        return {}

  def _apply_config_settings(self):
    """Применяет настройки из конфигурации"""
    # Настройки производительности
    performance = self.config.get('performance_thresholds', {})
    self.confidence_threshold = performance.get('confidence_threshold', 0.6)
    self.target_win_rate = performance.get('target_win_rate_pct', 60)
    self.min_profit_factor = performance.get('min_profit_factor', 1.5)

    # Настройки мониторинга
    monitoring = self.config.get('monitoring', {})
    self.signal_tracking_duration_hours = monitoring.get('signal_tracking_duration_hours', 24)
    self.max_concurrent_tracking = monitoring.get('max_concurrent_tracking', 1000)
    self.auto_finalize_expired = monitoring.get('auto_finalize_expired_signals', True)

    # Настройки аналитики
    analytics = self.config.get('analytics', {})
    self.min_signals_for_analysis = analytics.get('min_signals_for_pattern_analysis', 10)
    self.confidence_levels = analytics.get('confidence_levels', [0.5, 0.6, 0.7, 0.8, 0.9])

    logger.debug(f"SignalTracker настроен: confidence_threshold={self.confidence_threshold}")

  # def _load_shadow_config(self):
  #     """Загружает конфигурацию Shadow Trading"""
  #     try:
  #       import json
  #       with open("config/enhanced_shadow_trading_config.json", 'r', encoding='utf-8') as f:
  #         config = json.load(f)
  #         return config.get('enhanced_shadow_trading', {})
  #     except:
  #       return {}

  # async def track_signal(self, signal: TradingSignal, metadata: Dict[str, Any] = None) -> str:
  #   """Начать отслеживание сигнала с учетом конфигурации"""
  #
  #   # Проверяем пороги из конфигурации
  #   if signal.confidence < self.confidence_threshold:
  #     logger.debug(f"Сигнал {signal.symbol} пропущен: уверенность {signal.confidence} < {self.confidence_threshold}")
  #     return ""
  #
  #   # Проверяем лимит одновременного отслеживания
  #   if len(self.tracked_signals) >= self.max_concurrent_tracking:
  #     logger.warning(f"Достигнут лимит отслеживания ({self.max_concurrent_tracking})")
  #     return ""

  async def ensure_tables_exist(self):
    """Убеждается, что таблицы Shadow Trading существуют"""
    try:
      # Проверяем существование таблиц через асинхронный запрос
      check_query = "SELECT name FROM sqlite_master WHERE type='table' AND name='signal_analysis'"
      result = await self.db_manager._execute(check_query, fetch='one')

      if not result:
        logger.warning("Таблица signal_analysis не найдена, создаем через синхронный метод...")
        self.setup_database_sync()
      else:
        logger.info("✅ Таблицы Shadow Trading уже существуют")

    except Exception as e:
      logger.error(f"Ошибка проверки таблиц Shadow Trading: {e}")
      # Пытаемся создать таблицы синхронно в случае ошибки
      self.setup_database_sync()


  def setup_database_sync(self):
    """Создание таблиц для отслеживания сигналов"""
    try:
      # Таблица для анализа сигналов
      self.db_manager.execute_sync("""
                CREATE TABLE IF NOT EXISTS signal_analysis (
                    signal_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,

                    indicators_triggered TEXT,  -- JSON array
                    ml_prediction_data TEXT,    -- JSON object
                    market_regime TEXT,
                    volatility_level TEXT,

                    was_filtered BOOLEAN DEFAULT FALSE,
                    filter_reasons TEXT,        -- JSON array

                    outcome TEXT DEFAULT 'pending',
                    exit_price REAL,
                    exit_time TIMESTAMP,
                    profit_loss_pct REAL,
                    profit_loss_usdt REAL,

                    max_favorable_excursion_pct REAL DEFAULT 0.0,
                    max_adverse_excursion_pct REAL DEFAULT 0.0,
                    time_to_target_seconds INTEGER,
                    time_to_max_profit_seconds INTEGER,

                    volume_at_signal REAL DEFAULT 0.0,
                    price_action_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

      # Таблица для отслеживания цен
      self.db_manager.execute_sync("""
                CREATE TABLE IF NOT EXISTS price_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    minutes_elapsed INTEGER NOT NULL,
                    FOREIGN KEY (signal_id) REFERENCES signal_analysis (signal_id)
                )
            """)

      # Индексы для производительности
      self.db_manager.execute_sync("CREATE INDEX IF NOT EXISTS idx_signal_symbol ON signal_analysis(symbol)")
      self.db_manager.execute_sync("CREATE INDEX IF NOT EXISTS idx_signal_entry_time ON signal_analysis(entry_time)")
      self.db_manager.execute_sync("CREATE INDEX IF NOT EXISTS idx_signal_outcome ON signal_analysis(outcome)")
      self.db_manager.execute_sync("CREATE INDEX IF NOT EXISTS idx_price_tracking_signal ON price_tracking(signal_id)")

      logger.info("✅ База данных для Shadow Trading настроена")

    except Exception as e:
      logger.error(f"Ошибка настройки базы данных Shadow Trading: {e}")

  async def track_signal(self, signal: TradingSignal, metadata: Dict[str, Any] = None) -> str:
    """
    Начать отслеживание сигнала

    Args:
        signal: Торговый сигнал
        metadata: Дополнительные метаданные (индикаторы, ML данные и т.д.)

    Returns:
        signal_id для дальнейшего отслеживания
    """
    try:
      signal_id = f"{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}_{signal.signal_type.value}"

      # Создаем анализ сигнала
      analysis = SignalAnalysis(
        signal_id=signal_id,
        symbol=signal.symbol,
        signal_type=signal.signal_type,
        entry_price=signal.price,
        entry_time=signal.timestamp,
        confidence=signal.confidence,
        source=metadata.get('source', 'unknown') if metadata else 'unknown',
        indicators_triggered=metadata.get('indicators_triggered', []) if metadata else [],
        ml_prediction_data=metadata.get('ml_prediction_data', {}) if metadata else {},
        market_regime=metadata.get('market_regime', '') if metadata else '',
        volatility_level=metadata.get('volatility_level', '') if metadata else '',
        volume_at_signal=metadata.get('volume', 0.0) if metadata else 0.0,
        price_action_score=metadata.get('price_action_score', 0.0) if metadata else 0.0
      )

      # Сохраняем в память и БД
      self.tracked_signals[signal_id] = analysis
      await self._save_signal_to_db(analysis)

      logger.info(
        f"🎯 Начато отслеживание сигнала {signal_id}: {signal.signal_type.value} {signal.symbol} @ {signal.price}")

      return signal_id

    except Exception as e:
      logger.error(f"Ошибка отслеживания сигнала: {e}")
      return ""

  async def mark_signal_filtered(self, signal_id: str, filter_reasons: List[FilterReason]):
    """Отметить сигнал как отфильтрованный"""
    try:
      if signal_id in self.tracked_signals:
        analysis = self.tracked_signals[signal_id]
        analysis.was_filtered = True
        analysis.filter_reasons = filter_reasons
        analysis.updated_at = datetime.now()

        await self._update_signal_in_db(analysis)

        logger.info(f"🚫 Сигнал {signal_id} отфильтрован: {[r.value for r in filter_reasons]}")

    except Exception as e:
      logger.error(f"Ошибка отметки фильтрации сигнала {signal_id}: {e}")

  async def mark_signal_executed(self, signal_id: str, order_id: str, quantity: float, leverage: int):
    """Отметить сигнал как исполненный"""
    try:
      if signal_id in self.tracked_signals:
        analysis = self.tracked_signals[signal_id]
        analysis.was_executed = True
        analysis.order_id = order_id
        analysis.executed_quantity = quantity
        analysis.executed_leverage = leverage
        analysis.execution_time = datetime.now()
        analysis.updated_at = datetime.now()

        await self._update_signal_in_db(analysis)
        logger.info(f"✅ Сигнал {signal_id} исполнен с OrderID: {order_id}")

    except Exception as e:
      logger.error(f"Ошибка отметки исполнения сигнала {signal_id}: {e}")

  async def update_price_tracking(self, signal_id: str, current_price: float, timestamp: datetime):
    """Обновить отслеживание цены для сигнала"""
    try:
      if signal_id not in self.tracked_signals:
        return

      analysis = self.tracked_signals[signal_id]
      entry_time = analysis.entry_time
      minutes_elapsed = int((timestamp - entry_time).total_seconds() / 60)

      # Сохраняем данные о цене
      await self.db_manager._execute(
        """INSERT INTO price_tracking (signal_id, symbol, price, timestamp, minutes_elapsed)
           VALUES (?, ?, ?, ?, ?)""",
        (signal_id, analysis.symbol, current_price, timestamp, minutes_elapsed)
      )

      # Обновляем статистику движения цены
      price_change_pct = ((current_price - analysis.entry_price) / analysis.entry_price) * 100

      # Для BUY сигналов положительное изменение - это прибыль
      if analysis.signal_type == SignalType.BUY:
        if price_change_pct > analysis.max_favorable_excursion_pct:
          analysis.max_favorable_excursion_pct = price_change_pct
        elif price_change_pct < 0 and abs(price_change_pct) > analysis.max_adverse_excursion_pct:
          analysis.max_adverse_excursion_pct = abs(price_change_pct)
      # Для SELL сигналов отрицательное изменение - это прибыль
      elif analysis.signal_type == SignalType.SELL:
        if price_change_pct < 0 and abs(price_change_pct) > analysis.max_favorable_excursion_pct:
          analysis.max_favorable_excursion_pct = abs(price_change_pct)
        elif price_change_pct > analysis.max_adverse_excursion_pct:
          analysis.max_adverse_excursion_pct = price_change_pct

      analysis.updated_at = timestamp

    except Exception as e:
      logger.error(f"Ошибка обновления отслеживания цены для {signal_id}: {e}")

  async def finalize_signal(self, signal_id: str, exit_price: float, exit_time: datetime,
                            outcome: SignalOutcome):
    """Завершить отслеживание сигнала"""
    try:
      if signal_id not in self.tracked_signals:
        logger.warning(f"Сигнал {signal_id} не найден для финализации")
        return

      analysis = self.tracked_signals[signal_id]
      analysis.exit_price = exit_price
      analysis.exit_time = exit_time
      analysis.outcome = outcome
      analysis.updated_at = datetime.now()

      # Рассчитываем P&L
      price_change_pct = ((exit_price - analysis.entry_price) / analysis.entry_price) * 100

      if analysis.signal_type == SignalType.BUY:
        analysis.profit_loss_pct = price_change_pct
      elif analysis.signal_type == SignalType.SELL:
        analysis.profit_loss_pct = -price_change_pct

      # Рассчитываем время до цели
      if exit_time and analysis.entry_time:
        analysis.time_to_target = exit_time - analysis.entry_time

      await self._update_signal_in_db(analysis)

      logger.info(
        f"🏁 Сигнал {signal_id} завершен: {outcome.value}, P&L: {analysis.profit_loss_pct:.2f}%"
      )

    except Exception as e:
      logger.error(f"Ошибка финализации сигнала {signal_id}: {e}")

  async def get_signal_statistics(self, days: int = 7) -> Dict[str, Any]:
    """Получить статистику сигналов"""
    try:
      cutoff_date = datetime.now() - timedelta(days=days)

      query = """
            SELECT 
                COUNT(*) as total_signals,
                COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) as profitable_signals,
                COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as loss_signals,
                COUNT(CASE WHEN was_filtered = 1 THEN 1 END) as filtered_signals,
                AVG(CASE WHEN outcome = 'profitable' THEN profit_loss_pct END) as avg_win_pct,
                AVG(CASE WHEN outcome = 'loss' THEN profit_loss_pct END) as avg_loss_pct,
                AVG(confidence) as avg_confidence,
                MAX(profit_loss_pct) as max_win_pct,
                MIN(profit_loss_pct) as max_loss_pct
            FROM signal_analysis 
            WHERE entry_time >= ?
        """

      result = await self.db_manager._execute(query, (cutoff_date,), fetch='one')

      if result:
        # Считаем только завершенные сделки для винрейта
        completed_signals = (result['profitable_signals'] or 0) + (result['loss_signals'] or 0)
        win_rate = 0.0
        if completed_signals > 0:
          win_rate = (result['profitable_signals'] / completed_signals * 100)

        return {
          'total_signals': result['total_signals'] or 0,
          'profitable_signals': result['profitable_signals'] or 0,
          'loss_signals': result['loss_signals'] or 0,
          'filtered_signals': result['filtered_signals'] or 0,
          'avg_win_pct': result['avg_win_pct'] or 0.0,
          'avg_loss_pct': result['avg_loss_pct'] or 0.0,
          'avg_confidence': result['avg_confidence'] or 0.0,
          'max_win_pct': result['max_win_pct'] or 0.0,
          'max_loss_pct': result['max_loss_pct'] or 0.0,
          'win_rate': win_rate,
          'completed_signals': completed_signals
        }

      return {}

    except Exception as e:
      logger.error(f"Ошибка получения статистики сигналов: {e}")

  async def sync_with_real_trades(self, symbol: str, trade_data: Dict):
    """Синхронизирует результаты Shadow сигнала с реальной сделкой"""
    try:
      # Находим последний pending сигнал для символа
      query = """
        SELECT signal_id, entry_price, entry_time 
        FROM signal_analysis 
        WHERE symbol = ? AND outcome = 'pending'
        ORDER BY entry_time DESC
        LIMIT 1
      """

      result = await self.db_manager._execute(query, (symbol,), fetch='one')

      if result:
        signal_id = result['signal_id']

        # Обновляем сигнал результатами реальной сделки
        update_query = """
          UPDATE signal_analysis 
          SET outcome = ?,
              exit_price = ?,
              exit_time = ?,
              profit_loss_pct = ?,
              profit_loss_usdt = ?,
              updated_at = ?
          WHERE signal_id = ?
        """

        # Определяем outcome на основе profit_loss
        outcome = 'profitable' if trade_data.get('profit_loss', 0) > 0 else 'loss'

        await self.db_manager._execute(
          update_query,
          (
            outcome,
            trade_data.get('close_price'),
            trade_data.get('close_timestamp'),
            trade_data.get('profit_pct', 0),
            trade_data.get('profit_loss', 0),
            datetime.now(),
            signal_id
          )
        )

        logger.info(f"✅ Синхронизирован Shadow сигнал {signal_id} с реальной сделкой {symbol}")

    except Exception as e:
      logger.error(f"Ошибка синхронизации с реальной сделкой: {e}")

  # async def _save_signal_to_db(self, analysis: SignalAnalysis):
  #   """Сохранить анализ сигнала в БД"""
  #   try:
  #     await self.db_manager._execute(
  #       """INSERT INTO signal_analysis (
  #           signal_id, symbol, signal_type, entry_price, entry_time, confidence, source,
  #           indicators_triggered, ml_prediction_data, market_regime, volatility_level,
  #           was_filtered, filter_reasons, volume_at_signal, price_action_score
  #       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
  #       (
  #         analysis.signal_id, analysis.symbol, analysis.signal_type.value,
  #         analysis.entry_price, analysis.entry_time, analysis.confidence, analysis.source,
  #         json.dumps(analysis.indicators_triggered), json.dumps(analysis.ml_prediction_data),
  #         analysis.market_regime, analysis.volatility_level,
  #         analysis.was_filtered, json.dumps([r.value for r in analysis.filter_reasons]),
  #         analysis.volume_at_signal, analysis.price_action_score
  #       )
  #     )
  #   except Exception as e:
  #     logger.error(f"Ошибка сохранения сигнала в БД: {e}")

  async def get_signals_by_symbol(self, symbol: str, days: int = 30) -> List[Dict]:
    """Получить сигналы по символу"""
    try:
      cutoff_date = datetime.now() - timedelta(days=days)

      query = """
            SELECT * FROM signal_analysis 
            WHERE symbol = ? AND entry_time >= ?
            ORDER BY entry_time DESC
        """

      return await self.db_manager._execute(query, (symbol, cutoff_date), fetch='all') or []

    except Exception as e:
      logger.error(f"Ошибка получения сигналов по символу {symbol}: {e}")
      return []

  async def _update_signal_in_db(self, analysis: SignalAnalysis):
    """Обновить анализ сигнала в БД"""
    try:
      time_to_target_seconds = int(analysis.time_to_target.total_seconds()) if analysis.time_to_target else None
      time_to_max_profit_seconds = int(
        analysis.time_to_max_profit.total_seconds()) if analysis.time_to_max_profit else None

      await self.db_manager._execute(
        """UPDATE signal_analysis SET
            was_filtered = ?, filter_reasons = ?, outcome = ?, exit_price = ?, exit_time = ?,
            profit_loss_pct = ?, max_favorable_excursion_pct = ?, max_adverse_excursion_pct = ?,
            time_to_target_seconds = ?, time_to_max_profit_seconds = ?, updated_at = ?
        WHERE signal_id = ?""",
        (
          analysis.was_filtered, json.dumps([r.value for r in analysis.filter_reasons]),
          analysis.outcome.value, analysis.exit_price, analysis.exit_time,
          analysis.profit_loss_pct, analysis.max_favorable_excursion_pct,
          analysis.max_adverse_excursion_pct, time_to_target_seconds,
          time_to_max_profit_seconds, analysis.updated_at, analysis.signal_id
        )
      )
    except Exception as e:
      logger.error(f"Ошибка обновления сигнала в БД: {e}")


# shadow_trading/price_monitor.py

class PriceMonitor:
  """Мониторинг цен для отслеживания результатов сигналов"""

  def __init__(self, signal_tracker: SignalTracker, data_fetcher):
    self.signal_tracker = signal_tracker
    self.data_fetcher = data_fetcher
    self.monitoring_symbols: Dict[str, List[str]] = {}  # symbol -> list of signal_ids
    self.is_running = False
    self.price_update_interval = 30

  async def start_monitoring(self):
    """Запуск мониторинга цен"""
    if self.is_running:
      logger.warning("Мониторинг цен уже запущен")
      return

    self.is_running = True
    logger.info("🔄 Запущен мониторинг цен для Shadow Trading")

    # Запускаем фоновую задачу
    asyncio.create_task(self._monitoring_loop())

  async def stop_monitoring(self):
    """Остановка мониторинга"""
    self.is_running = False
    logger.info("⏹️ Остановлен мониторинг цен")

  async def add_signal_for_monitoring(self, signal_id: str, symbol: str):
    """Добавить сигнал для мониторинга"""
    if symbol not in self.monitoring_symbols:
      self.monitoring_symbols[symbol] = []

    if signal_id not in self.monitoring_symbols[symbol]:
      self.monitoring_symbols[symbol].append(signal_id)
      logger.debug(f"➕ Добавлен {signal_id} в мониторинг")

  async def remove_signal_from_monitoring(self, signal_id: str, symbol: str):
    """Убрать сигнал из мониторинга"""
    if symbol in self.monitoring_symbols and signal_id in self.monitoring_symbols[symbol]:
      self.monitoring_symbols[symbol].remove(signal_id)
      logger.debug(f"➖ Убран {signal_id} из мониторинга")

      # Убираем символ если нет активных сигналов
      if not self.monitoring_symbols[symbol]:
        del self.monitoring_symbols[symbol]

  # async def _monitoring_loop(self):
  #   """Основной цикл мониторинга"""
  #   while self.is_running:
  #     try:
  #       if not self.monitoring_symbols:
  #         await asyncio.sleep(10)
  #         continue
  #
  #       # Получаем текущие цены для всех отслеживаемых символов
  #       for symbol in list(self.monitoring_symbols.keys()):
  #         try:
  #           # ИСПРАВЛЕНИЕ: Используем существующий метод get_candles
  #           # Получаем последнюю свечу как текущую цену
  #           from core.enums import Timeframe
  #           df = await self.data_fetcher.get_historical_candles(
  #             symbol=symbol,
  #             timeframe=Timeframe.ONE_MINUTE,
  #             limit=1
  #           )
  #
  #           if df.empty:
  #             logger.warning(f"Не удалось получить данные для {symbol}")
  #             continue
  #
  #           current_price = float(df['close'].iloc[-1])
  #           timestamp = datetime.now()
  #
  #           # Обновляем отслеживание для всех сигналов этого символа
  #           for signal_id in self.monitoring_symbols[symbol]:
  #             await self.signal_tracker.update_price_tracking(
  #               signal_id, current_price, timestamp
  #             )
  #
  #         except Exception as e:
  #           logger.error(f"Ошибка мониторинга {symbol}: {e}")
  #
  #         # Пауза между циклами мониторинга
  #       await asyncio.sleep(30)  # Обновляем каждые 30 секунд
  #
  #     except asyncio.CancelledError:
  #       logger.info("Цикл мониторинга цен отменен")
  #       break
  #     except Exception as e:
  #       logger.error(f"Ошибка в цикле мониторинга цен: {e}")
  #       await asyncio.sleep(60)  # Больше пауза при ошибке
  async def _monitoring_loop(self):
    """Основной цикл мониторинга цен"""
    while self.is_running:
      try:
        if not self.monitoring_symbols:
          await asyncio.sleep(10)
          continue

        # Обновляем цены для всех отслеживаемых символов
        for symbol, signal_ids in list(self.monitoring_symbols.items()):
          try:
            # Получаем текущую цену
            current_price = await self.data_fetcher.get_current_price_safe(symbol)
            if not current_price:
              continue

            current_time = datetime.now()

            # Обновляем для каждого сигнала
            for signal_id in signal_ids[:]:  # Копия списка для безопасного удаления
              await self.signal_tracker.update_price_tracking(
                signal_id, current_price, current_time
              )

              # Проверяем, не пора ли прекратить отслеживание
              signal_info = self.signal_tracker.tracked_signals.get(signal_id)
              if signal_info:
                hours_elapsed = (current_time - signal_info.entry_time).total_seconds() / 3600
                if hours_elapsed > 24:  # 24 часа отслеживания
                  signal_ids.remove(signal_id)
                  await self.signal_tracker.finalize_signal(signal_id, current_price)

          except Exception as e:
            logger.error(f"Ошибка обновления цен для {symbol}: {e}")

        await asyncio.sleep(getattr(self, 'price_update_interval', 30))

      except Exception as e:
        logger.error(f"Ошибка в цикле мониторинга цен: {e}")
        await asyncio.sleep(60)

  async def finalize_signal(self, signal_id: str, final_price: float):
    """Финализировать отслеживание сигнала"""
    try:
      if signal_id not in self.tracked_signals:
        return

      analysis = self.tracked_signals[signal_id]
      analysis.exit_price = final_price
      analysis.exit_time = datetime.now()

      # Рассчитываем результат
      if analysis.signal_type == SignalType.BUY:
        analysis.profit_loss_pct = ((final_price - analysis.entry_price) / analysis.entry_price) * 100
      else:
        analysis.profit_loss_pct = ((analysis.entry_price - final_price) / analysis.entry_price) * 100

      # Определяем исход
      if analysis.profit_loss_pct > 0.5:
        analysis.outcome = SignalOutcome.PROFITABLE
      elif analysis.profit_loss_pct < -0.5:
        analysis.outcome = SignalOutcome.LOSS
      else:
        analysis.outcome = SignalOutcome.BREAKEVEN

      analysis.updated_at = datetime.now()
      await self._update_signal_in_db(analysis)

      # Удаляем из отслеживания
      del self.tracked_signals[signal_id]
      logger.info(f"📊 Сигнал {signal_id} финализирован с результатом: {analysis.profit_loss_pct:.2f}%")

    except Exception as e:
      logger.error(f"Ошибка финализации сигнала {signal_id}: {e}")

  async def _check_signal_completion(self, signal_id: str, current_price: float, timestamp: datetime):
    """Проверка условий завершения сигнала"""
    try:
      analysis = self.signal_tracker.tracked_signals.get(signal_id)
      if not analysis or analysis.outcome != SignalOutcome.PENDING:
        return

      # Проверяем время жизни сигнала
      time_elapsed = timestamp - analysis.entry_time

      # Автоматическое завершение через 24 часа
      if time_elapsed > timedelta(hours=24):
        await self.signal_tracker.finalize_signal(
          signal_id, current_price, timestamp, SignalOutcome.EXPIRED
        )
        await self.remove_signal_from_monitoring(signal_id, analysis.symbol)
        return

      # Проверяем целевые уровни (если есть)
      if hasattr(analysis, 'target_profit_pct'):
        target_profit = getattr(analysis, 'target_profit_pct', 3.0)  # 3% по умолчанию

        if analysis.signal_type == SignalType.BUY:
          profit_pct = (current_price - analysis.entry_price) / analysis.entry_price * 100
        else:
          profit_pct = (analysis.entry_price - current_price) / analysis.entry_price * 100

        if profit_pct >= target_profit:
          await self.signal_tracker.finalize_signal(
            signal_id, current_price, timestamp, SignalOutcome.PROFITABLE
          )
          await self.remove_signal_from_monitoring(signal_id, analysis.symbol)
          return

        # Проверяем stop loss (если есть)
        stop_loss = getattr(analysis, 'stop_loss_pct', -2.0)  # -2% по умолчанию
        if profit_pct <= stop_loss:
          await self.signal_tracker.finalize_signal(
            signal_id, current_price, timestamp, SignalOutcome.LOSS
          )
          await self.remove_signal_from_monitoring(signal_id, analysis.symbol)
          return

    except Exception as e:
      logger.error(f"Ошибка проверки завершения сигнала {signal_id}: {e}")


# shadow_trading/performance_analyzer.py

class PerformanceAnalyzer:
  """Анализатор производительности сигналов"""

  def __init__(self, db_manager: AdvancedDatabaseManager):
    self.db_manager = db_manager

  async def get_overall_performance(self, days: int = 30) -> Dict[str, Any]:
    """Общая производительность за период"""
    try:
      cutoff_date = datetime.now() - timedelta(days=days)

      query = """
                SELECT 
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) as profitable_signals,
                    COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as loss_signals,
                    COUNT(CASE WHEN was_filtered = 1 THEN 1 END) as filtered_signals,
                    AVG(CASE WHEN outcome = 'profitable' THEN profit_loss_pct END) as avg_win_pct,
                    AVG(CASE WHEN outcome = 'loss' THEN profit_loss_pct END) as avg_loss_pct,
                    AVG(max_favorable_excursion_pct) as avg_max_favorable,
                    AVG(max_adverse_excursion_pct) as avg_max_adverse,
                    AVG(confidence) as avg_confidence
                FROM signal_analysis 
                WHERE entry_time >= ?
            """

      result = await self.db_manager._execute(query, (cutoff_date,), fetch='one')

      if not result or result['total_signals'] == 0:
        return {'error': 'Нет данных за указанный период'}

      # Рассчитываем метрики
      total_signals = result['total_signals']
      profitable = result['profitable_signals'] or 0
      losses = result['loss_signals'] or 0
      completed = profitable + losses

      win_rate = (profitable / completed * 100) if completed > 0 else 0
      avg_win = result['avg_win_pct'] or 0
      avg_loss = result['avg_loss_pct'] or 0
      profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

      return {
        'period_days': days,
        'total_signals': total_signals,
        'completed_signals': completed,
        'filtered_signals': result['filtered_signals'] or 0,
        'win_rate_pct': round(win_rate, 2),
        'profitable_signals': profitable,
        'loss_signals': losses,
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2),
        'avg_max_favorable_pct': round(result['avg_max_favorable'] or 0, 2),
        'avg_max_adverse_pct': round(result['avg_max_adverse'] or 0, 2),
        'avg_confidence': round(result['avg_confidence'] or 0, 3)
      }

    except Exception as e:
      logger.error(f"Ошибка анализа общей производительности: {e}")
      return {'error': str(e)}

  async def get_performance_by_source(self, days: int = 30) -> List[Dict[str, Any]]:
    """Производительность по источникам сигналов"""
    try:
      cutoff_date = datetime.now() - timedelta(days=days)

      query = """
                SELECT 
                    source,
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) as profitable_signals,
                    COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as loss_signals,
                    AVG(CASE WHEN outcome = 'profitable' THEN profit_loss_pct END) as avg_win_pct,
                    AVG(CASE WHEN outcome = 'loss' THEN profit_loss_pct END) as avg_loss_pct,
                    AVG(confidence) as avg_confidence
                FROM signal_analysis 
                WHERE entry_time >= ?
                GROUP BY source
                ORDER BY COUNT(*) DESC
            """

      results = await self.db_manager._execute(query, (cutoff_date,), fetch='all')

      hourly_performance = {}
      for row in results:
        hour = row['hour']
        total = row['total_signals']
        profitable = row['profitable_signals'] or 0
        win_rate = (profitable / total * 100) if total > 0 else 0

        hourly_performance[hour] = {
          'total_signals': total,
          'win_rate_pct': round(win_rate, 2),
          'avg_return_pct': round(row['avg_return'] or 0, 2)
        }

      return hourly_performance

    except Exception as e:
      logger.error(f"Ошибка анализа по часам: {e}")
      return {}


  async def get_filter_analysis(self, days: int = 30) -> Dict[str, Any]:
    """Анализ эффективности фильтров"""
    try:
      cutoff_date = datetime.now() - timedelta(days=days)

      # Фильтрованные сигналы, которые могли бы быть прибыльными
      query_filtered = """
                  SELECT 
                      filter_reasons,
                      COUNT(*) as filtered_count
                  FROM signal_analysis 
                  WHERE entry_time >= ? AND was_filtered = 1
                  GROUP BY filter_reasons
              """

      filtered_results = await self.db_manager._execute(query_filtered, (cutoff_date,), fetch='all')

      # Анализ "упущенной прибыли" - это более сложная задача
      # Требует дополнительного анализа цен после фильтрации

      filter_stats = []
      for row in filtered_results:
        try:
          reasons = json.loads(row['filter_reasons']) if row['filter_reasons'] else []
          filter_stats.append({
            'filter_reasons': reasons,
            'count': row['filtered_count']
          })
        except json.JSONDecodeError:
          continue

      return {
        'total_filtered': sum(stat['count'] for stat in filter_stats),
        'filter_breakdown': filter_stats
      }

    except Exception as e:
      logger.error(f"Ошибка анализа фильтров: {e}")
      return {'error': str(e)}

  async def _analyze_missed_opportunity(self, filtered_signal: Dict, missed_opportunities: List):
    """Анализ упущенных возможностей"""
    try:
      # Получаем данные о движении цены после фильтрации
      symbol = filtered_signal['symbol']
      entry_time = datetime.fromisoformat(filtered_signal['entry_time'])
      entry_price = filtered_signal['entry_price']

      # Проверяем движение цены в течение 24 часов после фильтрации
      end_time = entry_time + timedelta(hours=24)

      # Здесь нужен доступ к историческим данным
      # Упрощенная версия - предполагаем, что у нас есть доступ к price_tracking
      query_price = """
            SELECT price, timestamp 
            FROM price_tracking 
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """

      price_data = await self.db_manager._execute(
        query_price, (symbol, entry_time, end_time), fetch='all'
      )

      if price_data:
        # Находим максимальное движение цены
        prices = [row['price'] for row in price_data]
        max_price = max(prices)
        min_price = min(prices)

        # Рассчитываем потенциальную прибыль
        upward_potential = (max_price - entry_price) / entry_price * 100
        downward_potential = (entry_price - min_price) / entry_price * 100

        # Предполагаем, что сигнал был BUY если upward_potential больше
        potential_profit = upward_potential if upward_potential > downward_potential else -downward_potential

        if abs(potential_profit) > 2.0:  # Только значительные движения
          missed_opportunities.append({
            'symbol': symbol,
            'entry_time': entry_time,
            'potential_profit': potential_profit,
            'confidence': filtered_signal['confidence']
          })

    except Exception as e:
      logger.warning(f"Ошибка анализа упущенной возможности: {e}")

  # async def get_hourly_performance(self, days: int = 7) -> Dict[int, Dict[str, Any]]:
  #   """Анализ производительности по часам дня"""
  #   try:
  #     cutoff_date = datetime.now() - timedelta(days=days)
  #
  #     query = """
  #       SELECT
  #         CAST(strftime('%H', entry_time) AS INTEGER) as hour,
  #         COUNT(*) as total,
  #         COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) as profitable,
  #         COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as losses,
  #         AVG(CASE WHEN outcome = 'profitable' THEN profit_loss_pct END) as avg_win_pct,
  #         AVG(CASE WHEN outcome = 'loss' THEN profit_loss_pct END) as avg_loss_pct
  #       FROM signal_analysis
  #       WHERE entry_time >= ? AND outcome IN ('profitable', 'loss')
  #       GROUP BY hour
  #       ORDER BY hour
  #     """
  #
  #     results = await self.db_manager._execute(query, (cutoff_date,), fetch='all')
  #
  #     hourly_stats = {}
  #     if results:
  #       for row in results:
  #         # Правильно извлекаем hour из словаря или кортежа
  #         if isinstance(row, dict):
  #           hour = row.get('hour', 0)
  #         else:
  #           hour = row[0] if row else 0
  #
  #         completed = (row.get('profitable', 0) if isinstance(row, dict) else row[2]) + \
  #                     (row.get('losses', 0) if isinstance(row, dict) else row[3])
  #
  #         win_rate = 0
  #         if completed > 0:
  #           profitable = row.get('profitable', 0) if isinstance(row, dict) else row[2]
  #           win_rate = (profitable / completed) * 100
  #
  #         hourly_stats[hour] = {
  #           'total_signals': row.get('total', 0) if isinstance(row, dict) else row[1],
  #           'win_rate_pct': round(win_rate, 1),
  #           'avg_win_pct': round(row.get('avg_win_pct', 0) if isinstance(row, dict) else (row[4] or 0), 2),
  #           'avg_loss_pct': round(row.get('avg_loss_pct', 0) if isinstance(row, dict) else (row[5] or 0), 2)
  #         }
  #
  #     return hourly_stats
  #
  #   except Exception as e:
  #     logger.error(f"Ошибка анализа по часам: {e}")
  #     return {}

  async def get_hourly_performance(self, days: int = 7) -> Dict[int, Dict[str, Any]]:
    """
    Анализ производительности по часам с улучшенной обработкой ошибок.
    """
    logger.debug(f"Запрос почасовой производительности за {days} дней.")
    try:
      cutoff_date = datetime.now() - timedelta(days=days)

      # ОБНОВЛЕННЫЙ ЗАПРОС:
      # 1. Используем COALESCE для всех агрегатных функций, чтобы избежать NULL/None.
      # 2. Переименовываем 'trade_hour' в 'hour' для консистентности.
      query = """
            SELECT 
                CAST(strftime('%H', entry_time) AS INTEGER) as hour,
                COALESCE(COUNT(*), 0) as total_signals,
                COALESCE(SUM(CASE WHEN outcome = 'profitable' THEN 1 ELSE 0 END), 0) as profitable_signals,
                COALESCE(AVG(profit_loss_pct), 0.0) as avg_return_pct
            FROM signal_analysis
            WHERE entry_time >= ? AND outcome IN ('profitable', 'loss')
            GROUP BY hour
        """

      results = await self.db_manager.execute_query(query, (cutoff_date.isoformat(),))

      hourly_stats = {}
      if results:
        for row in results:
          # Добавляем проверку, что row не None и является словарем
          if not row or not isinstance(row, dict):
            continue

          # Теперь ключ 'hour' будет существовать всегда
          hour = row.get('hour')
          if hour is None:
            continue

          total = row.get('total_signals', 0)
          profitable = row.get('profitable_signals', 0)

          # Благодаря COALESCE, avg_return_pct никогда не будет None
          avg_return = row.get('avg_return_pct', 0.0)

          win_rate = (profitable / total * 100) if total > 0 else 0.0

          hourly_stats[hour] = {
            'total_signals': total,
            'win_rate_pct': round(win_rate, 2),
            'avg_return_pct': round(avg_return, 4)
          }

      return hourly_stats

    except Exception as e:
      logger.error(f"Ошибка анализа по часам: {e}", exc_info=True)
      return {"error": str(e)}

  async def get_symbol_performance(self, days: int = 30) -> List[Dict[str, Any]]:
      """Производительность по символам"""
      try:
        cutoff_date = datetime.now() - timedelta(days=days)

        query = """
              SELECT 
                  symbol,
                  COUNT(*) as total_signals,
                  COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) as profitable_signals,
                  COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as loss_signals,
                  AVG(CASE WHEN outcome = 'profitable' THEN profit_loss_pct END) as avg_win_pct,
                  AVG(CASE WHEN outcome = 'loss' THEN profit_loss_pct END) as avg_loss_pct,
                  SUM(CASE WHEN outcome = 'profitable' THEN profit_loss_pct ELSE 0 END) +
                  SUM(CASE WHEN outcome = 'loss' THEN profit_loss_pct ELSE 0 END) as total_pnl_pct,
                  AVG(confidence) as avg_confidence
              FROM signal_analysis 
              WHERE entry_time >= ?
              GROUP BY symbol
              HAVING COUNT(*) >= 3  -- Минимум 3 сигнала для статистической значимости
              ORDER BY total_pnl_pct DESC
          """

        results = await self.db_manager._execute(query, (cutoff_date,), fetch='all')

        symbol_performance = []
        for row in results:
          profitable = row['profitable_signals'] or 0
          losses = row['loss_signals'] or 0
          completed = profitable + losses
          win_rate = (profitable / completed * 100) if completed > 0 else 0

          avg_win = row['avg_win_pct'] or 0
          avg_loss = row['avg_loss_pct'] or 0
          profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

          symbol_performance.append({
            'symbol': row['symbol'],
            'total_signals': row['total_signals'],
            'completed_signals': completed,
            'win_rate_pct': round(win_rate, 2),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'total_pnl_pct': round(row['total_pnl_pct'] or 0, 2),
            'avg_confidence': round(row['avg_confidence'] or 0, 3)
          })

        return symbol_performance

      except Exception as e:
        logger.error(f"Ошибка анализа по символам: {e}")
        return []

  async def get_confidence_analysis(self, days: int = 30) -> Dict[str, Any]:
    """Анализ производительности по уровням уверенности"""
    try:
      cutoff_date = datetime.now() - timedelta(days=days)

      query = """
              SELECT 
                  CASE 
                      WHEN confidence >= 0.9 THEN 'Very High (0.9+)'
                      WHEN confidence >= 0.8 THEN 'High (0.8-0.9)'
                      WHEN confidence >= 0.7 THEN 'Medium (0.7-0.8)'
                      WHEN confidence >= 0.6 THEN 'Low (0.6-0.7)'
                      ELSE 'Very Low (<0.6)'
                  END as confidence_level,
                  COUNT(*) as total_signals,
                  COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) as profitable_signals,
                  AVG(CASE WHEN outcome IN ('profitable', 'loss') THEN profit_loss_pct END) as avg_return,
                  AVG(confidence) as avg_confidence
              FROM signal_analysis 
              WHERE entry_time >= ? AND outcome IN ('profitable', 'loss')
              GROUP BY confidence_level
              ORDER BY avg_confidence DESC
          """

      results = await self.db_manager._execute(query, (cutoff_date,), fetch='all')

      confidence_analysis = []
      for row in results:
        total = row['total_signals']
        profitable = row['profitable_signals'] or 0
        win_rate = (profitable / total * 100) if total > 0 else 0

        confidence_analysis.append({
          'confidence_level': row['confidence_level'],
          'total_signals': total,
          'win_rate_pct': round(win_rate, 2),
          'avg_return_pct': round(row['avg_return'] or 0, 2),
          'avg_confidence': round(row['avg_confidence'] or 0, 3)
        })

      return {
        'confidence_breakdown': confidence_analysis,
        'optimal_threshold': self._calculate_optimal_confidence_threshold(confidence_analysis)
      }

    except Exception as e:
      logger.error(f"Ошибка анализа уверенности: {e}")
      return {'error': str(e)}

  def _calculate_optimal_confidence_threshold(self, confidence_data: List[Dict]) -> float:
    """Рассчитать оптимальный порог уверенности"""
    try:
      # Простая эвристика: найти уровень с лучшим соотношением win_rate и количества сигналов
      best_score = 0
      optimal_threshold = 0.6

      for level_data in confidence_data:
        win_rate = level_data['win_rate_pct']
        signal_count = level_data['total_signals']

        # Взвешенная оценка (win_rate важнее, но нужно минимальное количество сигналов)
        score = win_rate * (1 + min(signal_count / 50, 1))  # Бонус за количество до 50 сигналов

        if score > best_score:
          best_score = score
          # Извлекаем числовой порог из строки уровня
          if 'Very High' in level_data['confidence_level']:
            optimal_threshold = 0.9
          elif 'High' in level_data['confidence_level']:
            optimal_threshold = 0.8
          elif 'Medium' in level_data['confidence_level']:
            optimal_threshold = 0.7
          elif 'Low' in level_data['confidence_level']:
            optimal_threshold = 0.6
          else:
            optimal_threshold = 0.5

      return optimal_threshold

    except Exception as e:
      logger.warning(f"Ошибка расчета оптимального порога: {e}")
      return 0.6

  async def generate_optimization_recommendations(self, days: int = 30) -> Dict[str, Any]:
    """Генерация рекомендаций по оптимизации"""
    try:
      # Собираем все аналитические данные
      overall_perf = await self.get_overall_performance(days)
      source_perf = await self.get_performance_by_source(days)
      symbol_perf = await self.get_symbol_performance(days)
      confidence_analysis = await self.get_confidence_analysis(days)
      filter_analysis = await self.get_filter_analysis(days)

      recommendations = []

      # Анализ общей производительности
      if overall_perf.get('win_rate_pct', 0) < 60:
        recommendations.append({
          'type': 'confidence_threshold',
          'priority': 'high',
          'message': f"Win Rate {overall_perf.get('win_rate_pct')}% ниже целевого. Рекомендуется повысить порог уверенности.",
          'suggested_action': f"Установить минимальную уверенность {confidence_analysis.get('optimal_threshold', 0.7)}"
        })

      # Анализ источников сигналов
      if source_perf:
        best_source = max(source_perf, key=lambda x: x['win_rate_pct'])
        worst_source = min(source_perf, key=lambda x: x['win_rate_pct'])

        if worst_source['win_rate_pct'] < 50:
          recommendations.append({
            'type': 'disable_source',
            'priority': 'medium',
            'message': f"Источник '{worst_source['source']}' показывает низкую эффективность ({worst_source['win_rate_pct']}%)",
            'suggested_action': f"Рассмотреть отключение или снижение веса источника '{worst_source['source']}'"
          })

        recommendations.append({
          'type': 'boost_source',
          'priority': 'low',
          'message': f"Источник '{best_source['source']}' показывает лучшую эффективность ({best_source['win_rate_pct']}%)",
          'suggested_action': f"Увеличить вес источника '{best_source['source']}'"
        })

      # Анализ символов
      if symbol_perf:
        poor_symbols = [s for s in symbol_perf if s['win_rate_pct'] < 45]
        if poor_symbols:
          recommendations.append({
            'type': 'symbol_blacklist',
            'priority': 'medium',
            'message': f"Символы показывают стабильно низкую эффективность: {[s['symbol'] for s in poor_symbols[:3]]}",
            'suggested_action': "Добавить в черный список или снизить приоритет этих символов"
          })

      # Анализ фильтров
      if filter_analysis.get('avg_missed_profit_pct', 0) > 2:
        recommendations.append({
          'type': 'filter_adjustment',
          'priority': 'high',
          'message': f"Фильтры блокируют сигналы с потенциальной прибылью {filter_analysis.get('avg_missed_profit_pct')}%",
          'suggested_action': "Пересмотреть настройки фильтров рисков"
        })

      return {
        'total_recommendations': len(recommendations),
        'high_priority': len([r for r in recommendations if r['priority'] == 'high']),
        'recommendations': recommendations,
        'generated_at': datetime.now().isoformat()
      }

    except Exception as e:
      logger.error(f"Ошибка генерации рекомендаций: {e}")
      return {'error': str(e)}

  async def get_simple_time_analysis(self, days: int = 7) -> dict:
    """Простой анализ времени без сложной группировки"""
    try:
      cutoff_date = datetime.now() - timedelta(days=days)

      query = """
            SELECT entry_time, outcome 
            FROM signal_analysis 
            WHERE entry_time >= ? AND outcome != 'pending'
            ORDER BY entry_time DESC
        """

      results = await self.db_manager._execute(query, (cutoff_date,), fetch='all')

      if not results:
        return {'total_signals': 0, 'error': 'Нет данных'}

      # Простая статистика без pandas
      total_signals = len(results)
      profitable_signals = sum(1 for r in results if r.get('outcome') == 'profitable')

      return {
        'total_signals': total_signals,
        'profitable_signals': profitable_signals,
        'win_rate': (profitable_signals / total_signals * 100) if total_signals > 0 else 0,
        'period_days': days
      }

    except Exception as e:
      logger.error(f"Ошибка простого анализа времени: {e}")
      return {'error': str(e)}

class DatabaseMonitor:
    """Мониторинг состояния базы данных"""

    def __init__(self, db_manager: AdvancedDatabaseManager):
      self.db_manager = db_manager
      self.stats = {
        'total_operations': 0,
        'failed_operations': 0,
        'lock_errors': 0,
        'last_lock_time': None
      }

    async def check_database_health(self) -> Dict[str, Any]:
      """Проверка здоровья БД"""
      try:
        # Простой запрос для проверки
        result = await self.db_manager._execute("SELECT 1", fetch='one')

        if result:
          return {
            'status': 'healthy',
            'response_time_ms': 0,  # Можно замерить
            'stats': self.stats.copy()
          }
        else:
          return {
            'status': 'error',
            'message': 'Нет ответа от БД',
            'stats': self.stats.copy()
          }

      except Exception as e:
        self.stats['failed_operations'] += 1
        if "database is locked" in str(e).lower():
          self.stats['lock_errors'] += 1
          self.stats['last_lock_time'] = time.time()

        return {
          'status': 'error',
          'message': str(e),
          'stats': self.stats.copy()
        }