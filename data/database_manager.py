# data/database_manager.py
import random
import sqlite3
import time

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

  def __init__(self, db_path: str, pool_size: int = 10, timeout: int = 30):
    self.db_path = db_path
    self.pool_size = pool_size
    self.timeout = timeout
    self._pool = asyncio.Queue(maxsize=pool_size)
    self._all_connections = []
    self._initialized = False
    self._lock = asyncio.Lock()

  async def initialize(self):
    """Инициализирует пул соединений с улучшенными настройками"""
    async with self._lock:
      if self._initialized:
        return

      logger.info(f"Инициализация пула соединений (размер: {self.pool_size})")

      for _ in range(self.pool_size):
        conn = await aiosqlite.connect(
          self.db_path,
          timeout=self.timeout  # Таймаут подключения
        )

        # УЛУЧШЕННЫЕ НАСТРОЙКИ SQLite для предотвращения блокировок
        await conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        await conn.execute("PRAGMA synchronous=NORMAL")  # Баланс производительности
        await conn.execute("PRAGMA cache_size=10000")  # Больше кэш
        await conn.execute("PRAGMA temp_store=MEMORY")  # Временные данные в памяти
        await conn.execute("PRAGMA busy_timeout=30000")  # 30 сек ожидания при блокировке
        await conn.execute("PRAGMA wal_autocheckpoint=1000")  # Автоматический checkpoint

        self._all_connections.append(conn)
        await self._pool.put(conn)

      self._initialized = True
      logger.info("Пул соединений успешно инициализирован")

  @asynccontextmanager
  async def acquire(self):
    """Получает соединение из пула с таймаутом"""
    if not self._initialized:
      await self.initialize()

    conn = None
    try:
      # Ожидаем соединение с таймаутом
      conn = await asyncio.wait_for(self._pool.get(), timeout=self.timeout)
      yield conn
    except asyncio.TimeoutError:
      logger.error("Таймаут получения соединения из пула")
      raise
    finally:
      if conn:
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
  def __init__(self, db_path: str = "trading_data.db", pool_size: int = 5):
    self.db_path = db_path
    self.pool = ConnectionPool(db_path, pool_size, timeout=30)
    self._cache = {}  # Простой кэш для часто используемых данных
    self._cache_ttl = 60  # TTL кэша в секундах (5 минут)
    self._cache_timestamps = {}

    # Настройки retry для операций
    self.max_retries = 3
    self.retry_delay_base = 0.1  # Базовая задержка между попытками

    self.stats = {
      'total_operations': 0,
      'failed_operations': 0,
      'lock_errors': 0,
      'last_lock_time': None,
      'last_successful_operation': None,
      'created_at': time.time()
    }

  async def _execute_with_retry(self, query: str, params: tuple = (), fetch: str = None,
                                  max_retries: int = None) -> any:
      """Выполнение запроса с повторными попытками при блокировке"""


      if max_retries is None:
        max_retries = self.max_retries

      last_error = None

      for attempt in range(max_retries + 1):
        try:
          return await self._execute_single(query, params, fetch)

        except Exception as e:
          last_error = e
          error_msg = str(e).lower()

          # Если БД заблокирована, пробуем еще раз
          if "database is locked" in error_msg or "database is busy" in error_msg:
            if attempt < max_retries:
              # Экспоненциальная задержка с джиттером
              delay = self.retry_delay_base * (2 ** attempt) + random.uniform(0, 0.1)
              logger.warning(f"БД заблокирована, попытка {attempt + 1}/{max_retries + 1}, ожидание {delay:.2f}с")
              await asyncio.sleep(delay)
              continue
            else:
              logger.error(f"БД заблокирована после {max_retries + 1} попыток")
              break
          else:
            # Другие ошибки не ретраим
            break

      # Если все попытки неудачны
      logger.error(f"Ошибка выполнения запроса после {max_retries + 1} попыток: {last_error}")
      logger.error(f"Запрос: {query[:100]}...")
      logger.error(f"Параметры: {params}")
      return None

  async def _execute_single(self, query: str, params: tuple = (), fetch: str = None) -> any:
    """Одиночное выполнение запроса"""
    try:
      async with self.pool.acquire() as conn:
        # Устанавливаем таймаут для операции
        async with asyncio.timeout(30):  # 30 секунд на операцию
          async with conn.execute(query, params) as cursor:
            if fetch == 'all':
              rows = await cursor.fetchall()
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
      raise e

  # # Синхронный метод для создания таблиц (нужен для SignalTracker)
  # def execute_sync(self, query: str, params: tuple = ()) -> Optional[Any]:
  #   """
  #   Синхронное выполнение SQL-запроса (для инициализации таблиц)
  #   """
  #   try:
  #     with sqlite3.connect(self.db_path) as conn:
  #       # Настраиваем оптимизацию для синхронного соединения
  #       conn.execute("PRAGMA journal_mode=WAL")
  #       conn.execute("PRAGMA synchronous=NORMAL")
  #       conn.execute("PRAGMA cache_size=10000")
  #       conn.execute("PRAGMA temp_store=MEMORY")
  #
  #       cursor = conn.execute(query, params)
  #       conn.commit()
  #       return cursor.lastrowid
  #   except Exception as e:
  #     logger.error(f"Ошибка выполнения синхронного запроса: {e}")
  #     logger.error(f"Запрос: {query}")
  #     logger.error(f"Параметры: {params}")
  #     return None

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

  async def _execute(self, query: str, params: tuple = (), fetch: str = None) -> any:
      """Основной метод выполнения запросов с retry логикой"""
      return await self._execute_with_retry(query, params, fetch)

  # Синхронный метод тоже улучшаем
  def execute_sync(self, query: str, params: tuple = ()) -> Optional[Any]:
    """
    Синхронное выполнение SQL-запроса с защитой от блокировок
    """
    import sqlite3
    import time

    max_retries = 3
    last_error = None

    for attempt in range(max_retries + 1):
      try:
        with sqlite3.connect(
            self.db_path,
            timeout=30.0  # 30 секунд таймаут
        ) as conn:
          # Настраиваем оптимизацию
          conn.execute("PRAGMA journal_mode=WAL")
          conn.execute("PRAGMA synchronous=NORMAL")
          conn.execute("PRAGMA cache_size=10000")
          conn.execute("PRAGMA temp_store=MEMORY")
          conn.execute("PRAGMA busy_timeout=30000")  # 30 сек ожидания

          cursor = conn.execute(query, params)
          conn.commit()
          return cursor.lastrowid

      except Exception as e:
        last_error = e
        error_msg = str(e).lower()

        if "database is locked" in error_msg or "database is busy" in error_msg:
          if attempt < max_retries:
            delay = 0.1 * (2 ** attempt) + random.uniform(0, 0.1)
            logger.warning(f"Синхронная БД заблокирована, попытка {attempt + 1}, ожидание {delay:.2f}с")
            time.sleep(delay)
            continue

        break

    logger.error(f"Ошибка синхронного выполнения запроса: {last_error}")
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
                strategy_name TEXT DEFAULT 'Unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

    # Создаем индексы для оптимизации запросов
    await self._execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
    await self._execute('CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)')
    await self._execute('CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at)')
    await self._execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol_status ON trades(symbol, status)')

    #Таблица логов сигналов
    await self._execute('''
            CREATE TABLE IF NOT EXISTS signals_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                symbol TEXT,
                strategy TEXT,
                signal TEXT,
                price REAL,
                confidence REAL,
                executed BOOLEAN,
                metadata TEXT
            )
        ''')
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
    # Проверяем и добавляем недостающие колонки для существующей таблицы
    async with self.pool.acquire() as conn:
      cursor = await conn.execute("PRAGMA table_info(trades)")
      columns = await cursor.fetchall()
      column_names = [col[1] for col in columns]

    # Добавляем strategy_name если её нет
    if 'strategy_name' not in column_names:
      await conn.execute("ALTER TABLE trades ADD COLUMN strategy_name TEXT DEFAULT 'Unknown'")
      await conn.commit()
      logger.info("Добавлена колонка strategy_name в таблицу trades")

    # Добавляем confidence если её нет
    if 'confidence' not in column_names:
      await conn.execute("ALTER TABLE trades ADD COLUMN confidence REAL DEFAULT 0.5")
      await conn.commit()
      logger.info("Добавлена колонка confidence в таблицу trades")

    await self.check_table_structure("trades")

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

  def execute_query_sync(self, query: str, params: tuple = None):
    """Синхронная версия execute_query для использования в не-асинхронном коде"""
    import sqlite3
    try:
      conn = sqlite3.connect(self.db_path)
      cursor = conn.cursor()
      cursor.execute(query, params or ())

      # Получаем имена колонок
      if cursor.description:
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
        result = [dict(zip(columns, row)) for row in rows]
      else:
        result = []

      conn.close()
      return result
    except Exception as e:
      logger.error(f"Ошибка выполнения синхронного запроса: {e}")
      return []

  async def check_table_structure(self, table_name: str):
    """Проверяет структуру таблицы"""
    try:
      async with self.pool.acquire() as conn:
        cursor = await conn.execute(f"PRAGMA table_info({table_name})")
        columns = await cursor.fetchall()

        logger.info(f"Структура таблицы {table_name}:")
        for col in columns:
          logger.info(f"  {col[1]} {col[2]} {'NOT NULL' if col[3] else 'NULL'} {f'DEFAULT {col[4]}' if col[4] else ''}")

        return columns
    except Exception as e:
      logger.error(f"Ошибка проверки структуры таблицы: {e}")
      return []

  async def get_trading_metrics_optimized(self, days: int = 30) -> Dict:
    """Оптимизированное получение торговых метрик с кэшированием"""
    cache_key = f"trading_metrics_{days}"
    cached_metrics = self._get_cache(cache_key)

    if cached_metrics is not None:
      return cached_metrics

    since = datetime.now() - timedelta(days=days)
    query = """
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as profitable_trades,
            SUM(profit_loss) as total_profit_loss,
            AVG(profit_loss) as avg_profit_loss,
            MAX(profit_loss) as max_profit,
            MIN(profit_loss) as max_loss,
            AVG(CASE WHEN profit_loss > 0 THEN profit_loss END) as avg_win,
            AVG(CASE WHEN profit_loss < 0 THEN profit_loss END) as avg_loss,
            SUM(commission) as total_commission
        FROM trades 
        WHERE close_timestamp >= ? 
        AND status = 'CLOSED'
    """

    result = await self._execute(query, (since,), fetch='one') or {}
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

    if not trade_id:
      logger.error(f"Не удалось добавить сделку в БД для {signal.symbol}")
      return None

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

  async def get_recent_closed_trades(self, limit: int = 100) -> List[Dict]:
    """Получает последние закрытые сделки"""
    try:
      query = """
            SELECT * FROM trades 
            WHERE status = 'CLOSED' 
            ORDER BY close_timestamp DESC 
            LIMIT ?
        """

      result = await self.execute_query(query, (limit,))
      return result if result else []

    except Exception as e:
      logger.error(f"Ошибка получения закрытых сделок: {e}")
      return []

  @property
  def conn(self):
    """Совместимость для старого кода"""
    # Возвращаем None, так как используем пул соединений
    return None

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

  async def execute_query(self, query: str, params: tuple = None):
    """Выполняет SQL запрос и возвращает результат"""
    try:
      async with aiosqlite.connect(self.db_path) as db:
        cursor = await db.execute(query, params or ())
        result = await cursor.fetchall()
        return result
    except Exception as e:
      logger.error(f"Ошибка выполнения запроса: {e}")
      return []

  def _update_stats(self, success: bool = True, is_lock_error: bool = False):
    """Обновление статистики операций"""
    self.stats['total_operations'] += 1

    if success:
      self.stats['last_successful_operation'] = time.time()
    else:
      self.stats['failed_operations'] += 1

    if is_lock_error:
      self.stats['lock_errors'] += 1
      self.stats['last_lock_time'] = time.time()

  def get_database_stats(self) -> Dict[str, Any]:
    """Получить статистику БД (синхронный метод)"""
    stats = self.stats.copy()

    # Дополнительные вычисления
    if stats['total_operations'] > 0:
      stats['error_rate_pct'] = (stats['failed_operations'] / stats['total_operations']) * 100
      stats['lock_rate_pct'] = (stats['lock_errors'] / stats['total_operations']) * 100
    else:
      stats['error_rate_pct'] = 0
      stats['lock_rate_pct'] = 0

    # Время с последней операции
    if stats['last_successful_operation']:
      stats['seconds_since_last_success'] = time.time() - stats['last_successful_operation']

    if stats['last_lock_time']:
      stats['seconds_since_last_lock'] = time.time() - stats['last_lock_time']

    return stats

  async def get_database_health(self) -> Dict[str, Any]:
    """Проверка состояния базы данных"""
    try:
      start_time = time.time()

      # Простой тестовый запрос
      result = await self._execute_single("SELECT 1 as test", (), fetch='one')

      response_time = (time.time() - start_time) * 1000  # в миллисекундах

      if result and result.get('test') == 1:
        status = 'healthy'
        message = 'БД работает нормально'
      else:
        status = 'warning'
        message = 'БД отвечает, но результат некорректный'

      return {
        'status': status,
        'message': message,
        'response_time_ms': round(response_time, 2),
        'stats': self.stats.copy(),
        'database_path': self.db_path,
        'pool_size': self.pool.pool_size if hasattr(self.pool, 'pool_size') else 'unknown',
        'timestamp': time.time()
      }

    except Exception as e:
      error_msg = str(e).lower()

      if "database is locked" in error_msg:
        status = 'locked'
        message = f'БД заблокирована: {e}'
      elif "no such file" in error_msg:
        status = 'missing'
        message = f'Файл БД не найден: {e}'
      else:
        status = 'error'
        message = f'Ошибка БД: {e}'

      return {
        'status': status,
        'message': message,
        'response_time_ms': -1,
        'stats': self.stats.copy(),
        'error': str(e),
        'timestamp': time.time()
      }

  # def execute_query(self, query: str, params: tuple = None):
  #   """Выполняет SQL запрос и возвращает результат"""
  #   try:
  #     conn = sqlite3.connect(self.db_path)
  #     cursor = conn.cursor()
  #     cursor.execute(query, params or ())
  #     result = cursor.fetchall()
  #     conn.close()
  #     return result
  #   except Exception as e:
  #     logger.error(f"Ошибка выполнения запроса: {e}")
  #     return []

