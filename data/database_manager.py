import sqlite3
import datetime

import logger
# import logger
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Optional, List, Tuple, Any, Dict
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import time
from abc import ABC, abstractmethod

# from core.trade_executor import TradeExecutor
from core.bybit_connector import BybitConnector
from core.schemas import TradingSignal
from ml.lorentzian_classifier import LorentzianClassifier
from strategies.base_strategy import BaseStrategy

class AdvancedDatabaseManager:
  """Продвинутый менеджер БД с кэшированием и оптимизацией"""

  def __init__(self, db_path: str = "advanced_trading.db"):
    self.db_path = db_path
    self.conn: Optional[sqlite3.Connection] = None
    self._connect()
    self._create_all_tables()
    self._cache = {}  # Простой кэш для часто используемых данных
    self.add_missing_columns()


  def _connect(self):
    """Устанавливает соединение с базой данных SQLite с оптимизацией"""
    try:
      self.conn = sqlite3.connect(
        self.db_path,
        # detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False
      )
      # Оптимизация SQLite для торговых данных
      self.conn.execute("PRAGMA journal_mode=WAL")
      self.conn.execute("PRAGMA synchronous=NORMAL")
      self.conn.execute("PRAGMA cache_size=10000")
      self.conn.execute("PRAGMA temp_store=memory")
      print(f"✅ Подключение к БД: {self.db_path}")
    except sqlite3.Error as e:
      print(f"❌ Ошибка подключения к SQLite: {e}")

  def _create_all_tables(self):
    """Создает все необходимые таблицы"""
    tables = {
      'trades': '''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    order_id TEXT UNIQUE,
                    strategy TEXT NOT NULL,
                    side TEXT NOT NULL,
                    open_timestamp TIMESTAMP NOT NULL,
                    close_timestamp TIMESTAMP,
                    open_price REAL NOT NULL,
                    close_price REAL,
                    quantity REAL NOT NULL,
                    leverage INTEGER DEFAULT 1,
                    profit_loss REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    status TEXT DEFAULT 'OPEN',
                    confidence REAL DEFAULT 0.5,
                    stop_loss REAL,
                    take_profit REAL,
                    metadata TEXT
                )
            ''',
      'model_performance': '''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall REAL,
                    f1_score REAL,
                    training_timestamp TIMESTAMP,
                    evaluation_data TEXT,
                    is_active BOOLEAN DEFAULT 1
                )
            ''',
      'risk_metrics': '''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    symbol TEXT,
                    current_drawdown REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    total_trades INTEGER,
                    daily_pnl REAL
                )
            ''',
      'signals_log': '''
                CREATE TABLE IF NOT EXISTS signals_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    price REAL NOT NULL,
                    confidence REAL,
                    executed BOOLEAN DEFAULT 0,
                    metadata TEXT
                )
            '''
    }

    for table_name, query in tables.items():
      try:
        self.conn.execute(query)
        print(f"✅ Таблица '{table_name}' готова")
      except sqlite3.Error as e:
        print(f"❌ Ошибка создания таблицы '{table_name}': {e}")

    self.conn.commit()

  def add_missing_columns(self):
    """Добавить отсутствующие столбцы в существующие таблицы"""
    try:
      cursor = self.conn.cursor()

      # Проверяем существование столбца created_at в таблице trades
      cursor.execute("PRAGMA table_info(trades)")
      columns = [column[1] for column in cursor.fetchall()]

      if 'created_at' not in columns:
        cursor.execute("""
                ALTER TABLE trades 
                ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            """)
        print("Добавлен столбец created_at в таблицу trades")

      self.conn.commit()

    except Exception as e:
      print(f"Ошибка при добавлении столбцов: {e}")

  def get_all_trades(self, limit: int = 50) -> List[Dict]:
    """Получить все сделки с лимитом"""
    try:
      cursor = self.conn.cursor()
      cursor.execute("""
              SELECT * FROM trades 
              ORDER BY created_at DESC 
              LIMIT ?
          """, (limit,))

      columns = [description[0] for description in cursor.description]
      trades = []

      for row in cursor.fetchall():
        trade_dict = dict(zip(columns, row))
        trades.append(trade_dict)

      return trades

    except Exception as e:
      print(f"Ошибка при получении сделок: {e}")
      return []

  def add_trade_with_signal(self, signal: TradingSignal, order_id: str, quantity: float, leverage: int = 1) -> Optional[int]:
    """Добавляет сделку на основе торгового сигнала"""
    query = '''
            INSERT INTO trades (
                symbol, order_id, strategy, side, open_timestamp, open_price, 
                quantity, leverage, confidence, stop_loss, take_profit, metadata, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
        '''

    try:
      metadata_json = json.dumps(signal.metadata) if signal.metadata else None
      cursor = self.conn.cursor()
      cursor.execute(query, (
        signal.metadata.get('symbol', 'UNKNOWN') if signal.metadata else 'UNKNOWN',
        order_id, signal.strategy_name, signal.signal.value,
        signal.timestamp, signal.price, quantity, leverage,
        signal.confidence, signal.stop_loss, signal.take_profit,
        metadata_json
      ))
      self.conn.commit()
      trade_id = cursor.lastrowid
      print(f"✅ Сделка добавлена (ID: {trade_id}): {signal.signal.value} {quantity} @ {signal.price}")
      return trade_id
    except sqlite3.Error as e:
      print(f"❌ Ошибка добавления сделки: {e}")
      return None

  def log_signal(self, signal: TradingSignal, symbol: str, executed: bool = False):
    """Логирует торговый сигнал"""
    query = '''
            INSERT INTO signals_log (timestamp, symbol, strategy, signal, price, confidence, executed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
    try:
      metadata_json = json.dumps(signal.metadata) if signal.metadata else None
      self.conn.execute(query, (
        signal.timestamp, symbol, signal.strategy_name,
        signal.signal.value, signal.price, signal.confidence,
        executed, metadata_json
      ))
      self.conn.commit()
    except sqlite3.Error as e:
      print(f"❌ Ошибка логирования сигнала: {e}")

  def update_model_performance(self, model_name: str, symbol: str, metrics: Dict[str, float]):
    """Обновляет метрики производительности модели"""
    query = '''
            INSERT INTO model_performance (
                model_name, symbol, accuracy, precision_score, recall, f1_score, 
                training_timestamp, evaluation_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
    try:
      self.conn.execute(query, (
        model_name, symbol, metrics.get('accuracy', 0),
        metrics.get('precision', 0), metrics.get('recall', 0),
        metrics.get('f1_score', 0), datetime.datetime.now(),
        json.dumps(metrics)
      ))
      self.conn.commit()
      print(f"✅ Метрики модели {model_name} обновлены для {symbol}")
    except sqlite3.Error as e:
      print(f"❌ Ошибка обновления метрик модели: {e}")

  def get_period_risk_metrics(self, days: int):
    """Получить метрики риска за определенный период"""
    try:
      end_date = datetime.datetime.now()
      start_date = end_date - datetime.timedelta(days=days)

      # Получаем сделки за период
      query = """
        SELECT profit_loss, entry_price, exit_price, quantity, 
               created_at, strategy_used
        FROM trades 
        WHERE created_at >= ? AND created_at <= ?
        ORDER BY created_at DESC
        """

      with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (start_date, end_date))
        trades = cursor.fetchall()

      if not trades:
        return {
          'total_trades': 0,
          'total_pnl': 0,
          'win_rate': 0,
          'avg_profit': 0,
          'max_drawdown': 0,
          'sharpe_ratio': 0
        }

      # Расчет метрик
      total_trades = len(trades)
      profits = [trade[0] for trade in trades if trade[0] is not None]

      total_pnl = sum(profits) if profits else 0
      winning_trades = len([p for p in profits if p > 0])
      win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

      avg_profit = total_pnl / total_trades if total_trades > 0 else 0

      # Расчет максимальной просадки
      cumulative_pnl = []
      running_total = 0
      for profit in profits:
        running_total += profit
        cumulative_pnl.append(running_total)

      max_drawdown = 0
      if cumulative_pnl:
        peak = cumulative_pnl[0]
        for value in cumulative_pnl:
          if value > peak:
            peak = value
          drawdown = peak - value
          max_drawdown = max(max_drawdown, drawdown)

      # Расчет коэффициента Шарпа (упрощенный)
      if profits and len(profits) > 1:
        import statistics
        avg_return = statistics.mean(profits)
        std_return = statistics.stdev(profits)
        sharpe_ratio = avg_return / std_return if std_return != 0 else 0
      else:
        sharpe_ratio = 0

      result = {
        'period_days': days,
        'total_trades': total_trades,
        'total_pnl': round(total_pnl, 2),
        'win_rate': round(win_rate, 2),
        'avg_profit': round(avg_profit, 2),
        'max_drawdown': round(max_drawdown, 2),
        'sharpe_ratio': round(sharpe_ratio, 3),
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d')
      }
      return result

    except Exception as e:
      print(f"Ошибка в get_period_risk_metrics: {e}")
      return {}

  def get_trades_for_symbol(self, symbol: str, limit: int = 100) -> List[Dict]:
    """Получить сделки для конкретного символа"""
    try:
      cursor = self.conn.cursor()

      # Проверяем структуру таблицы
      cursor.execute("PRAGMA table_info(trades)")
      columns_info = cursor.fetchall()
      column_names = [col[1] for col in columns_info]

      # Выбираем столбец для сортировки
      if 'created_at' in column_names:
        order_column = 'created_at'
      elif 'id' in column_names:
        order_column = 'id'
      else:
        order_column = 'rowid'

      cursor.execute(f"""
            SELECT * FROM trades 
            WHERE symbol = ?
            ORDER BY {order_column} DESC 
            LIMIT ?
        """, (symbol, limit))

      columns = [description[0] for description in cursor.description]
      trades = []

      for row in cursor.fetchall():
        trade_dict = dict(zip(columns, row))
        trades.append(trade_dict)

      return trades

    except Exception as e:
      print(f"Ошибка при получении сделок для символа {symbol}: {e}")
      return []
