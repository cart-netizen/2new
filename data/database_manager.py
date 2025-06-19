# data/database_manager.py

import aiosqlite
import asyncio
import json
from functools import lru_cache
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import logging

from core.schemas import TradingSignal
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ConnectionPool:
  """Пул асинхронных соединений с БД для оптимизации производительности"""

  def __init__(self, db_path: str, pool_size: int = 10):
    self.db_path = db_path
    self.pool_size = pool_size
    self._pool = asyncio.Queue(maxsize=pool_size)
    self._all_connections = []
    self._initialized = False
    self._lock = asyncio.Lock()

  async def initialize(self):
    """Инициализирует пул соединений"""
    async with self._lock:
      if self._initialized:
        return

      logger.info(f"Инициализация пула соединений (размер: {self.pool_size})")

      for _ in range(self.pool_size):
        conn = await aiosqlite.connect(self.db_path)
        await conn.execute("PRAGMA journal_mode=WAL")  # Оптимизация для конкурентного доступа
        await conn.execute("PRAGMA synchronous=NORMAL")  # Баланс между производительностью и надежностью
        await conn.execute("PRAGMA cache_size=10000")  # Увеличиваем кэш
        await conn.execute("PRAGMA temp_store=MEMORY")  # Временные таблицы в памяти

        self._all_connections.append(conn)
        await self._pool.put(conn)

      self._initialized = True
      logger.info("Пул соединений успешно инициализирован")

  @asynccontextmanager
  async def acquire(self):
    """Получает соединение из пула"""
    if not self._initialized:
      await self.initialize()

    conn = await self._pool.get()
    try:
      yield conn
    finally:
      await self._pool.put(conn)

  async def close_all(self):
    """Закрывает все соединения в пуле"""
    async with self._lock:
      while not self._pool.empty():
        await self._pool.get()

      for conn in self._all_connections:
        await conn.close()

      self._all_connections.clear()
      self._initialized = False
      logger.info("Все соединения в пуле закрыты")


class AdvancedDatabaseManager:
  def __init__(self, db_path: str = "trading_data.db", pool_size: int = 10):
    self.db_path = db_path
    self.pool = ConnectionPool(db_path, pool_size)
    self._cache = {}  # Простой кэш для часто используемых данных
    self._cache_ttl = 300  # TTL кэша в секундах (5 минут)
    self._cache_timestamps = {}

  async def __aenter__(self):
    await self.pool.initialize()
    await self._create_tables_if_not_exist()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.pool.close_all()

  def _is_cache_valid(self, key: str) -> bool:
    """Проверяет, действителен ли кэш для данного ключа"""
    if key not in self._cache_timestamps:
      return False
    return (datetime.now() - self._cache_timestamps[key]).seconds < self._cache_ttl

  def _set_cache(self, key: str, value: Any):
    """Устанавливает значение в кэш"""
    self._cache[key] = value
    self._cache_timestamps[key] = datetime.now()

  def _get_cache(self, key: str) -> Optional[Any]:
    """Получает значение из кэша, если оно действительно"""
    if self._is_cache_valid(key):
      return self._cache.get(key)
    return None

  def clear_cache(self, pattern: Optional[str] = None):
    """Очищает кэш. Если указан pattern, очищает только соответствующие ключи"""
    if pattern:
      keys_to_remove = [k for k in self._cache.keys() if pattern in k]
      for key in keys_to_remove:
        self._cache.pop(key, None)
        self._cache_timestamps.pop(key, None)
    else:
      self._cache.clear()
      self._cache_timestamps.clear()

  async def _execute(self, query: str, params: tuple = (), fetch: str = None) -> Union[List[Dict], Dict, int, None]:
    """
    Оптимизированное выполнение SQL-запроса с использованием пула соединений
    """
    try:
      async with self.pool.acquire() as conn:
        async with conn.execute(query, params) as cursor:
          if fetch == 'all':
            rows = await cursor.fetchall()
            # Преобразуем в список словарей для удобства
            if rows and cursor.description:
              columns = [column[0] for column in cursor.description]
              return [dict(zip(columns, row)) for row in rows]
            return []
          elif fetch == 'one':
            row = await cursor.fetchone()
            if row and cursor.description:
              columns = [column[0] for column in cursor.description]
              return dict(zip(columns, row))
            return None
          else:
            await conn.commit()
            return cursor.lastrowid
    except Exception as e:
      logger.error(f"Ошибка выполнения запроса: {e}")
      logger.error(f"Запрос: {query}")
      logger.error(f"Параметры: {params}")
      return None

  async def _execute_many(self, query: str, params_list: List[tuple]):
    """Выполняет множественную вставку/обновление для оптимизации"""
    try:
      async with self.pool.acquire() as conn:
        await conn.executemany(query, params_list)
        await conn.commit()
        return True
    except Exception as e:
      logger.error(f"Ошибка выполнения executemany: {e}")
      return False

  async def _create_tables_if_not_exist(self):
    """Создает необходимые таблицы с оптимизированными индексами"""
    # Основная таблица сделок
    await self._execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                order_id TEXT UNIQUE,
                strategy TEXT,
                side TEXT,
                open_timestamp TIMESTAMP,
                close_timestamp TIMESTAMP,
                open_price REAL,
                close_price REAL,
                quantity REAL,
                leverage INTEGER,
                profit_loss REAL,
                commission REAL,
                status TEXT,
                confidence REAL,
                stop_loss REAL,
                take_profit REAL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

    # Создаем индексы для оптимизации запросов
    await self._execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
    await self._execute('CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)')
    await self._execute('CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at)')
    await self._execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol_status ON trades(symbol, status)')

    # Таблица логов сигналов
    # await self._execute('''
    #         CREATE TABLE IF NOT EXISTS signals_log (
    #             id INTEGER PRIMARY KEY AUTOINCREMENT,
    #             timestamp TIMESTAMP,
    #             symbol TEXT,
    #             strategy TEXT,
    #             signal TEXT,
    #             price REAL,
    #             confidence REAL,
    #             executed BOOLEAN,
    #             metadata TEXT,
    #             INDEX idx_signals_symbol (symbol),
    #             INDEX idx_signals_timestamp (timestamp)
    #         )
    #     ''')
    await self._execute('''
        CREATE TABLE IF NOT EXISTS signals_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TIMESTAMP, symbol TEXT, strategy TEXT,
            signal TEXT, price REAL, confidence REAL, executed BOOLEAN, metadata TEXT
        )
    ''')

    # Таблица производительности для быстрого доступа к метрикам
    await self._execute('''
            CREATE TABLE IF NOT EXISTS performance_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT UNIQUE,
                metric_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

    logger.info("Таблицы и индексы успешно созданы/проверены")

  # Кэшированные методы для часто используемых запросов

  async def get_open_positions_cached(self) -> List[Dict]:
    """Получает открытые позиции с кэшированием"""
    cache_key = "open_positions"
    cached_data = self._get_cache(cache_key)

    if cached_data is not None:
      return cached_data

    query = "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY created_at DESC"
    result = await self._execute(query, fetch='all') or []

    self._set_cache(cache_key, result)
    return result

  async def get_symbol_performance_cached(self, symbol: str, days: int = 30) -> Dict:
    """Получает производительность символа с кэшированием"""
    cache_key = f"symbol_performance_{symbol}_{days}"
    cached_data = self._get_cache(cache_key)

    if cached_data is not None:
      return cached_data

    since_date = datetime.now() - timedelta(days=days)
    query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(profit_loss) as total_pnl,
                AVG(profit_loss) as avg_pnl,
                MAX(profit_loss) as max_win,
                MIN(profit_loss) as max_loss
            FROM trades 
            WHERE symbol = ? AND status = 'CLOSED' AND close_timestamp >= ?
        """

    result = await self._execute(query, (symbol, since_date), fetch='one') or {}
    self._set_cache(cache_key, result)
    return result

  async def batch_insert_signals(self, signals: List[TradingSignal]):
    """Пакетная вставка сигналов для оптимизации"""
    query = '''
            INSERT INTO signals_log (timestamp, symbol, strategy, signal, price, confidence, executed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''

    params_list = []
    for signal in signals:
      metadata_json = json.dumps(signal.metadata) if signal.metadata else None
      params = (
        signal.timestamp, signal.symbol, signal.strategy_name,
        signal.signal_type.value, signal.price, signal.confidence,
        False, metadata_json
      )
      params_list.append(params)

    success = await self._execute_many(query, params_list)
    if success:
      logger.info(f"Успешно вставлено {len(signals)} сигналов")
      # Очищаем соответствующий кэш
      self.clear_cache("signals")

  # Метод для получения агрегированной статистики с кэшированием
  @lru_cache(maxsize=100)
  async def get_aggregated_stats(self, timeframe: str = 'day') -> Dict:
    """Получает агрегированную статистику с мемоизацией"""
    # Для демонстрации - этот метод будет кэшировать результаты в памяти
    # В реальности нужно использовать более сложное кэширование с учетом времени

    if timeframe == 'day':
      since = datetime.now() - timedelta(days=1)
    elif timeframe == 'week':
      since = datetime.now() - timedelta(days=7)
    else:
      since = datetime.now() - timedelta(days=30)

    query = """
            SELECT 
                COUNT(DISTINCT symbol) as active_symbols,
                COUNT(*) as total_trades,
                SUM(profit_loss) as total_pnl,
                AVG(profit_loss) as avg_pnl,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
            FROM trades 
            WHERE created_at >= ? AND status = 'CLOSED'
        """

    return await self._execute(query, (since,), fetch='one') or {}

  # Оригинальные методы с оптимизацией

  async def add_trade_with_signal(self, signal: TradingSignal, order_id: str, quantity: float, leverage: int = 1) -> \
  Optional[Dict]:
    """Добавляет сделку и возвращает полную информацию"""
    query = '''
            INSERT INTO trades (
                symbol, order_id, strategy, side, open_timestamp, open_price,
                quantity, leverage, confidence, stop_loss, take_profit, metadata, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
        '''

    metadata_json = json.dumps(signal.metadata) if signal.metadata else None
    params = (
      signal.symbol, order_id, signal.strategy_name, signal.signal_type.value,
      signal.timestamp, signal.price, quantity, leverage,
      signal.confidence, signal.stop_loss, signal.take_profit, metadata_json
    )

    trade_id = await self._execute(query, params)

    # Очищаем кэш открытых позиций
    self.clear_cache("open_positions")

    if trade_id and trade_id > 0:
      # Возвращаем полную информацию о сделке
      select_query = "SELECT * FROM trades WHERE id = ?"
      trade_details = await self._execute(select_query, (trade_id,), fetch='one')
      return trade_details

    return None

  async def update_closed_trade(self, trade_id: int, close_price: float, profit_loss: float, commission: float = 0.0):
    """Обновляет закрытую сделку"""
    query = """
            UPDATE trades 
            SET status = 'CLOSED',
                close_price = ?,
                close_timestamp = ?,
                profit_loss = ?,
                commission = ?
            WHERE id = ?
        """

    result = await self._execute(
      query,
      (close_price, datetime.now(), profit_loss, commission, trade_id)
    )

    # Очищаем соответствующие кэши
    self.clear_cache("open_positions")
    self.clear_cache("symbol_performance")

    return result is not None

  # Остальные методы остаются без изменений, но используют новый _execute с пулом
  async def get_open_trade_by_symbol(self, symbol: str) -> Optional[Dict]:
    query = "SELECT * FROM trades WHERE symbol = ? AND status = 'OPEN' LIMIT 1"
    return await self._execute(query, (symbol,), fetch='one')

  async def get_all_trades(self, limit: int = 50) -> List[Dict]:
    query = "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?"
    return await self._execute(query, (limit,), fetch='all')

  async def log_signal(self, signal: TradingSignal, symbol: str, executed: bool = False):
    query = '''
            INSERT INTO signals_log (timestamp, symbol, strategy, signal, price, confidence, executed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
    metadata_json = json.dumps(signal.metadata) if signal.metadata else None
    params = (
      signal.timestamp, symbol, signal.strategy_name,
      signal.signal_type.value, signal.price, signal.confidence,
      executed, metadata_json
    )
    await self._execute(query, params)

  async def get_all_closed_trades(self, limit: int = 50) -> List[Dict]:
    """
    Асинхронно получает из БД все сделки со статусом 'CLOSED',
    отсортированные по времени закрытия.
    """
    query = "SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY close_timestamp DESC LIMIT ?"
    # Используем наш универсальный метод _execute для выполнения запроса
    closed_trades = await self._execute(query, (limit,), fetch='all')
    return closed_trades

  async def get_all_open_trades(self) -> List[Dict]:
      """Получает все открытые сделки"""
      query = "SELECT * FROM trades WHERE status = 'OPEN'"
      return await self._execute(query, fetch='all') or []

  async def update_trade_as_closed(self, trade_id: int, close_price: float,
                                   pnl: float, commission: float = 0.0,
                                   close_timestamp: Optional[datetime] = None):
    """Обновляет сделку как закрытую"""
    if close_timestamp is None:
      close_timestamp = datetime.now()

    query = """
        UPDATE trades 
        SET status = 'CLOSED',
            close_price = ?,
            close_timestamp = ?,
            profit_loss = ?,
            commission = ?
        WHERE id = ?
    """

    result = await self._execute(
      query,
      (close_price, close_timestamp, pnl, commission, trade_id)
    )

    # Очищаем кэш
    self.clear_cache("open_positions")

    if result is not None:
      logger.info(f"Сделка {trade_id} закрыта. PnL: {pnl:.4f}, Комиссия: {commission:.4f}")

    return result is not None

  async def force_close_trade(self, trade_id: int, close_price: float, reason: str = "Forced closure") -> bool:
    """
    Принудительно закрывает сделку в БД с нулевым PnL.
    Используется для "зомби-позиций".
    """
    query = """
        UPDATE trades
        SET status = 'CLOSED',
            close_price = ?,
            close_timestamp = ?,
            profit_loss = 0,
            metadata = json_set(COALESCE(metadata, '{}'), '$.close_reason', ?)
        WHERE id = ?
    """

    result = await self._execute(
      query,
      (close_price, datetime.now(), reason, trade_id)
    )

    # Очищаем кэш
    self.clear_cache("open_positions")

    if result is not None:
      logger.warning(f"Сделка {trade_id} принудительно закрыта. Причина: {reason}")
      return True

    return False