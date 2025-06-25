#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для исправления и инициализации базы данных Shadow Trading

Использование:
    python shadow_db_init.py

Что делает:
1. Проверяет существование таблиц Shadow Trading
2. Создает недостающие таблицы
3. Проверяет целостность данных
4. Выводит статистику
"""

import sqlite3
import sys
import os
from datetime import datetime
from pathlib import Path


def check_database_file(db_path: str) -> bool:
  """Проверяет существование файла базы данных"""
  if not os.path.exists(db_path):
    print(f"❌ Файл базы данных не найден: {db_path}")
    return False

  print(f"✅ Файл базы данных найден: {db_path}")
  return True


def get_existing_tables(db_path: str) -> list:
  """Получает список существующих таблиц"""
  try:
    with sqlite3.connect(db_path) as conn:
      cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
      tables = [row[0] for row in cursor.fetchall()]
      return tables
  except Exception as e:
    print(f"❌ Ошибка получения списка таблиц: {e}")
    return []


def create_shadow_trading_tables(db_path: str) -> bool:
  """Создает таблицы Shadow Trading"""
  try:
    with sqlite3.connect(db_path) as conn:
      # Настраиваем оптимизацию
      conn.execute("PRAGMA journal_mode=WAL")
      conn.execute("PRAGMA synchronous=NORMAL")
      conn.execute("PRAGMA cache_size=10000")
      conn.execute("PRAGMA temp_store=MEMORY")

      print("🔨 Создание таблицы signal_analysis...")

      # Таблица для анализа сигналов
      conn.execute("""
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

      print("🔨 Создание таблицы price_tracking...")

      # Таблица для отслеживания цен
      conn.execute("""
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

      print("🔨 Создание индексов для производительности...")

      # Индексы для производительности
      conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_symbol ON signal_analysis(symbol)")
      conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_entry_time ON signal_analysis(entry_time)")
      conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_outcome ON signal_analysis(outcome)")
      conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_source ON signal_analysis(source)")
      conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_was_filtered ON signal_analysis(was_filtered)")
      conn.execute("CREATE INDEX IF NOT EXISTS idx_price_tracking_signal ON price_tracking(signal_id)")
      conn.execute("CREATE INDEX IF NOT EXISTS idx_price_tracking_timestamp ON price_tracking(timestamp)")

      conn.commit()
      print("✅ Таблицы Shadow Trading успешно созданы")
      return True

  except Exception as e:
    print(f"❌ Ошибка создания таблиц Shadow Trading: {e}")
    return False


def verify_tables_structure(db_path: str) -> bool:
  """Проверяет структуру таблиц"""
  try:
    with sqlite3.connect(db_path) as conn:
      # Проверяем signal_analysis
      cursor = conn.execute("PRAGMA table_info(signal_analysis)")
      signal_columns = [row[1] for row in cursor.fetchall()]

      required_signal_columns = [
        'signal_id', 'symbol', 'signal_type', 'entry_price', 'entry_time',
        'confidence', 'source', 'outcome', 'was_filtered'
      ]

      missing_columns = [col for col in required_signal_columns if col not in signal_columns]
      if missing_columns:
        print(f"❌ В таблице signal_analysis отсутствуют колонки: {missing_columns}")
        return False

      # Проверяем price_tracking
      cursor = conn.execute("PRAGMA table_info(price_tracking)")
      price_columns = [row[1] for row in cursor.fetchall()]

      required_price_columns = ['id', 'signal_id', 'symbol', 'price', 'timestamp']

      missing_price_columns = [col for col in required_price_columns if col not in price_columns]
      if missing_price_columns:
        print(f"❌ В таблице price_tracking отсутствуют колонки: {missing_price_columns}")
        return False

      print("✅ Структура таблиц корректна")
      return True

  except Exception as e:
    print(f"❌ Ошибка проверки структуры таблиц: {e}")
    return False


def get_database_statistics(db_path: str) -> dict:
  """Получает статистику базы данных"""
  try:
    with sqlite3.connect(db_path) as conn:
      stats = {}

      # Основные таблицы
      cursor = conn.execute("SELECT COUNT(*) FROM trades")
      stats['total_trades'] = cursor.fetchone()[0]

      cursor = conn.execute("SELECT COUNT(*) FROM signals_log")
      stats['total_signals_log'] = cursor.fetchone()[0]

      # Shadow Trading таблицы
      try:
        cursor = conn.execute("SELECT COUNT(*) FROM signal_analysis")
        stats['shadow_signals'] = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM signal_analysis WHERE outcome = 'profitable'")
        stats['shadow_profitable'] = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM signal_analysis WHERE was_filtered = 1")
        stats['shadow_filtered'] = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM price_tracking")
        stats['price_tracking_records'] = cursor.fetchone()[0]

      except sqlite3.OperationalError:
        stats['shadow_signals'] = 0
        stats['shadow_profitable'] = 0
        stats['shadow_filtered'] = 0
        stats['price_tracking_records'] = 0

      return stats

  except Exception as e:
    print(f"❌ Ошибка получения статистики: {e}")
    return {}


def add_sample_data(db_path: str) -> bool:
  """Добавляет тестовые данные для проверки"""
  try:
    with sqlite3.connect(db_path) as conn:
      # Добавляем тестовый сигнал
      test_signal_data = (
        'TEST_BTCUSDT_20250624_120000_BUY',
        'BTCUSDT',
        'BUY',
        50000.0,
        datetime.now().isoformat(),
        0.85,
        'test_init_script',
        '["rsi_oversold"]',
        '{"confidence": 0.85}',
        'trending',
        'normal',
        False,
        '[]',
        0.0,
        0.0
      )

      conn.execute("""
                INSERT OR IGNORE INTO signal_analysis (
                    signal_id, symbol, signal_type, entry_price, entry_time,
                    confidence, source, indicators_triggered, ml_prediction_data,
                    market_regime, volatility_level, was_filtered, filter_reasons,
                    volume_at_signal, price_action_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, test_signal_data)

      conn.commit()
      print("✅ Тестовые данные добавлены")
      return True

  except Exception as e:
    print(f"❌ Ошибка добавления тестовых данных: {e}")
    return False


def main():
  """Основная функция"""
  print("🚀 Инициализация Shadow Trading Database")
  print("=" * 50)

  # Определяем путь к базе данных
  db_path = "trading_data.db"

  # Ищем в текущей директории и родительских
  possible_paths = [
    db_path,
    f"../{db_path}",
    f"data/{db_path}",
    f"../data/{db_path}"
  ]

  actual_db_path = None
  for path in possible_paths:
    if os.path.exists(path):
      actual_db_path = path
      break

  if not actual_db_path:
    print(f"⚠️ База данных не найдена, создаем новую: {db_path}")
    actual_db_path = db_path

  print(f"📍 Используем базу данных: {actual_db_path}")

  # Проверяем существующие таблицы
  existing_tables = get_existing_tables(actual_db_path)
  print(f"📋 Существующие таблицы: {existing_tables}")

  shadow_tables = [t for t in existing_tables if t in ['signal_analysis', 'price_tracking']]

  if shadow_tables:
    print(f"✅ Найдены таблицы Shadow Trading: {shadow_tables}")
  else:
    print("❌ Таблицы Shadow Trading не найдены")

  # Создаем таблицы
  print("\n🔨 Создание/проверка таблиц...")
  if create_shadow_trading_tables(actual_db_path):
    print("✅ Таблицы Shadow Trading готовы к использованию")
  else:
    print("❌ Ошибка создания таблиц")
    return False

  # Проверяем структуру
  print("\n🔍 Проверка структуры таблиц...")
  if not verify_tables_structure(actual_db_path):
    print("❌ Структура таблиц некорректна")
    return False

  # Получаем статистику
  print("\n📊 Статистика базы данных:")
  stats = get_database_statistics(actual_db_path)

  for key, value in stats.items():
    print(f"   {key}: {value}")

  # Добавляем тестовые данные если нет данных
  if stats.get('shadow_signals', 0) == 0:
    print("\n🧪 Добавление тестовых данных...")
    add_sample_data(actual_db_path)

  print("\n" + "=" * 50)
  print("✅ Инициализация Shadow Trading завершена успешно!")
  print("\n📝 Следующие шаги:")
  print("   1. Запустите dashboard.py для проверки")
  print("   2. Запустите основной бот для начала сбора данных")
  print("   3. Мониторьте логи на предмет ошибок")

  return True


if __name__ == "__main__":
  try:
    success = main()
    sys.exit(0 if success else 1)
  except KeyboardInterrupt:
    print("\n⏹️ Прервано пользователем")
    sys.exit(1)
  except Exception as e:
    print(f"\n❌ Неожиданная ошибка: {e}")
    sys.exit(1)