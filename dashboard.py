# dashboard.py
import sys
from collections import deque
import json

import numpy as np
import psutil
import streamlit as st
import pandas as pd
import subprocess
import os
import signal
import time
import asyncio
from contextlib import suppress
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta


# from shadow_trading.dashboard_extensions import (
#     add_shadow_trading_section,
#     display_full_shadow_dashboard,
#     setup_shadow_dashboard_integration
# )
from data.database_manager import AdvancedDatabaseManager
from data.state_manager import StateManager
from config import settings
from config.config_manager import ConfigManager
from streamlit_autorefresh import st_autorefresh

from shadow_trading.dashboard_extensions import setup_shadow_dashboard_integration

# --- Настройка страницы ---
st.set_page_config(
  page_title="Панель управления торговым ботом",
  page_icon="🤖",
  layout="wide"
)
# st_autorefresh(interval=5000, key="data_refresher")

@st.cache_resource
def get_config_manager():
    """Создает и кэширует экземпляр ConfigManager."""
    return ConfigManager()

@st.cache_resource
def get_state_manager():
    """Создает и кэширует экземпляр StateManager."""
    return StateManager()

@st.cache_resource
def get_db_manager():
    """Создает и кэширует экземпляр AdvancedDatabaseManager."""
    # Примечание: асинхронная инициализация пула здесь не будет вызываться постоянно
    return AdvancedDatabaseManager(settings.DATABASE_PATH)
@st.cache_resource
def get_shadow_trading_initialized():
    """Кэширует инициализацию Shadow Trading."""
    return initialize_shadow_trading()

# --- Инициализация менеджеров ---
CONFIG_FILE_PATH = "config.json"
config_manager = ConfigManager()
state_manager = StateManager()
db_manager = AdvancedDatabaseManager(settings.DATABASE_PATH)

# Инициализируем таблицы
asyncio.run(db_manager._create_tables_if_not_exist())


def initialize_shadow_trading():
  """Инициализация Shadow Trading системы при запуске дашборда"""
  try:
    # Создаем таблицы Shadow Trading синхронно
    logger_query = "SELECT name FROM sqlite_master WHERE type='table' AND name='signal_analysis'"

    # Проверяем, существует ли таблица signal_analysis
    check_result = db_manager.execute_sync(logger_query)

    # Создаем таблицу для анализа сигналов
    create_signal_analysis_query = """
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
      """

    # Создаем таблицу для отслеживания цен
    create_price_tracking_query = """
          CREATE TABLE IF NOT EXISTS price_tracking (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              signal_id TEXT NOT NULL,
              symbol TEXT NOT NULL,
              price REAL NOT NULL,
              timestamp TIMESTAMP NOT NULL,
              minutes_elapsed INTEGER NOT NULL,
              FOREIGN KEY (signal_id) REFERENCES signal_analysis (signal_id)
          )
      """

    # Выполняем создание таблиц
    db_manager.execute_sync(create_signal_analysis_query)
    db_manager.execute_sync(create_price_tracking_query)

    # Создаем индексы для производительности
    db_manager.execute_sync("CREATE INDEX IF NOT EXISTS idx_signal_symbol ON signal_analysis(symbol)")
    db_manager.execute_sync("CREATE INDEX IF NOT EXISTS idx_signal_entry_time ON signal_analysis(entry_time)")
    db_manager.execute_sync("CREATE INDEX IF NOT EXISTS idx_signal_outcome ON signal_analysis(outcome)")
    db_manager.execute_sync("CREATE INDEX IF NOT EXISTS idx_price_tracking_signal ON price_tracking(signal_id)")

    print("✅ Shadow Trading база данных успешно инициализирована")
    return True

  except Exception as e:
    print(f"❌ Ошибка инициализации Shadow Trading: {e}")
    return False

# Инициализируем Shadow Trading при запуске дашборда
get_shadow_trading_initialized()

# --- Статус и основные метрики ---
if 'bot_process' not in st.session_state:
    st.session_state.bot_process = None

# is_bot_running = st.session_state.bot_process and st.session_state.bot_process.poll() is None

def get_shadow_trading_today_stats() -> dict:
  """Получает статистику Shadow Trading за сегодня"""
  try:
    query = """
            SELECT 
                COUNT(*) as today_signals,
                COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) as today_profitable
            FROM signal_analysis 
            WHERE entry_time >= datetime('now', '-1 day')
        """

    result = db_manager.execute_sync(query)
    if result:
      # Для синхронного SQLite получаем результат напрямую
      import sqlite3
      with sqlite3.connect(db_manager.db_path) as conn:
        cursor = conn.execute(query)
        row = cursor.fetchone()
        if row:
          return {
            'today_signals': row[0] or 0,
            'today_profitable': row[1] or 0
          }

    return {'today_signals': 0, 'today_profitable': 0}

  except Exception as e:
    print(f"Ошибка получения статистики за сегодня: {e}")
    return {'today_signals': 0, 'today_profitable': 0}


def get_shadow_trading_stats(days: int = 7) -> dict:
  """Получает детальную статистику Shadow Trading"""
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
                AVG(confidence) as avg_confidence
            FROM signal_analysis 
            WHERE entry_time >= ?
        """

    import sqlite3
    with sqlite3.connect(db_manager.db_path) as conn:
      cursor = conn.execute(query, (cutoff_date,))
      row = cursor.fetchone()

      if row:
        return {
          'total_signals': row[0] or 0,
          'profitable_signals': row[1] or 0,
          'loss_signals': row[2] or 0,
          'filtered_signals': row[3] or 0,
          'avg_win_pct': row[4] or 0.0,
          'avg_loss_pct': row[5] or 0.0,
          'avg_confidence': row[6] or 0.0
        }

    return {
      'total_signals': 0, 'profitable_signals': 0, 'loss_signals': 0,
      'filtered_signals': 0, 'avg_win_pct': 0.0, 'avg_loss_pct': 0.0, 'avg_confidence': 0.0
    }

  except Exception as e:
    print(f"Ошибка получения статистики Shadow Trading: {e}")
    return {
      'total_signals': 0, 'profitable_signals': 0, 'loss_signals': 0,
      'filtered_signals': 0, 'avg_win_pct': 0.0, 'avg_loss_pct': 0.0, 'avg_confidence': 0.0
    }

async def get_shadow_trading_stats_async(days: int = 7) -> dict:
    """Получает детальную статистику Shadow Trading через SignalTracker"""
    try:
      # Импортируем SignalTracker для использования его метода
      from shadow_trading.signal_tracker import SignalTracker

      # Создаем экземпляр SignalTracker
      signal_tracker = SignalTracker(db_manager)

      # Убеждаемся что таблицы существуют
      await signal_tracker.ensure_tables_exist()

      # Получаем статистику через метод SignalTracker
      stats = await signal_tracker.get_signal_statistics(days)

      if stats:
        return stats
      else:
        # Fallback к прямому запросу если SignalTracker не вернул данных
        return get_shadow_trading_stats(days)

    except Exception as e:
      print(f"Ошибка получения асинхронной статистики Shadow Trading: {e}")
      # Fallback к синхронному методу
      return get_shadow_trading_stats(days)


# ДОБАВИТЬ новую функцию для получения расширенной статистики:

def get_enhanced_shadow_stats(days: int = 7) -> dict:
  """Получает расширенную статистику Shadow Trading"""
  try:
    # Получаем базовую статистику
    base_stats = asyncio.run(get_shadow_trading_stats_async(days))

    # Добавляем дополнительные метрики из get_signal_statistics
    enhanced_stats = base_stats.copy()
    enhanced_stats.update({
      'max_win_pct': base_stats.get('max_win_pct', 0.0),
      'max_loss_pct': base_stats.get('max_loss_pct', 0.0),
      'win_rate': base_stats.get('win_rate', 0.0)
    })

    return enhanced_stats

  except Exception as e:
    print(f"Ошибка получения расширенной статистики: {e}")
    return get_shadow_trading_stats(days)

# --- ФУНКЦИИ SHADOW TRADING (БЕЗ ИМПОРТА КЛАССОВ) ---

def create_shadow_trading_summary(days: int = 7) -> str:
    """Создает краткую сводку Shadow Trading с расширенными метриками"""
    try:
      stats = get_enhanced_shadow_stats(days)

      if stats['total_signals'] == 0:
        return f"📊 Shadow Trading (за {days} дней): Нет данных о сигналах"

      win_rate = stats.get('win_rate', 0.0)
      filter_rate = (stats['filtered_signals'] / stats['total_signals']) * 100 if stats['total_signals'] > 0 else 0
      max_win = stats.get('max_win_pct', 0.0)
      max_loss = stats.get('max_loss_pct', 0.0)

      summary = f"""
📊 **Shadow Trading за {days} дней:**
• Всего сигналов: {stats['total_signals']}
• Прибыльных: {stats['profitable_signals']} ({win_rate:.1f}%)
• Убыточных: {stats['loss_signals']}
• Отфильтровано: {stats['filtered_signals']} ({filter_rate:.1f}%)
• Средняя прибыль: {stats['avg_win_pct']:.2f}%
• Средний убыток: {stats['avg_loss_pct']:.2f}%
• Лучший сигнал: +{max_win:.2f}%
• Худший сигнал: {max_loss:.2f}%
• Средняя уверенность: {stats['avg_confidence']:.2f}
        """

      return summary.strip()

    except Exception as e:
      return f"❌ Ошибка создания сводки Shadow Trading: {e}"


def add_shadow_trading_section():
  """Добавляет секцию Shadow Trading без импорта классов"""

  st.markdown("---")
  st.header("🌟 Shadow Trading Analytics")

  with st.expander("📊 Краткая сводка Shadow Trading", expanded=True):
    # Выбор периода
    col1, col2 = st.columns([2, 1])

    with col1:
      days = st.selectbox(
        "📅 Период анализа",
        options=[1, 3, 7, 14, 30],
        index=2,  # 7 дней по умолчанию
        key="shadow_period"
      )

    with col2:
      if st.button("🔄 Обновить Shadow", use_container_width=True):
        st.rerun()

    # Отображаем сводку
    summary = create_shadow_trading_summary(days)
    st.markdown(summary)

    # Базовая статистика по источникам (если есть данные)
    try:
      cutoff_date = datetime.now() - timedelta(days=days)
      source_query = """
                SELECT 
                    source,
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) as profitable_signals
                FROM signal_analysis 
                WHERE entry_time >= ?
                GROUP BY source
                ORDER BY COUNT(*) DESC
                LIMIT 5
            """

      source_results = asyncio.run(db_manager._execute(source_query, (cutoff_date,), fetch='all'))

      if source_results:
        st.markdown("**🏆 Топ источники сигналов:**")
        for row in source_results:
          total = row['total_signals']
          profitable = row['profitable_signals'] or 0
          win_rate = (profitable / total * 100) if total > 0 else 0
          st.markdown(f"• {row['source']}: {win_rate:.1f}% WR ({total} сигналов)")

    except Exception as source_error:
      st.info("Статистика по источникам временно недоступна")


def display_simple_shadow_metrics():
  """Отображает простые метрики Shadow Trading"""
  try:
    # Последние 24 часа
    today_query = """
            SELECT 
                COUNT(*) as today_signals,
                COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) as today_profitable
            FROM signal_analysis 
            WHERE entry_time >= datetime('now', '-1 day')
        """

    today_result = asyncio.run(db_manager._execute(today_query, (), fetch='one'))

    if today_result and today_result['today_signals'] > 0:
      today_total = today_result['today_signals']
      today_profitable = today_result['today_profitable'] or 0
      today_wr = (today_profitable / today_total * 100) if today_total > 0 else 0

      col1, col2, col3 = st.columns(3)

      with col1:
        st.metric("📊 Сигналов сегодня", today_total)

      with col2:
        st.metric("✅ Прибыльных", today_profitable)

      with col3:
        st.metric("🎯 Win Rate", f"{today_wr:.1f}%")
    else:
      st.info("🔄 Сегодня еще нет завершенных сигналов")

  except Exception as e:
    st.warning(f"Метрики Shadow Trading недоступны: {e}")

# def get_bot_pid():
#     """Получение PID процесса бота"""
#     try:
#       result = subprocess.run(['pgrep', '-f', 'main.py'], capture_output=True, text=True)
#       if result.returncode == 0 and result.stdout.strip():
#         return int(result.stdout.strip().split('\n')[0])
#     except:
#       pass
#     return None
def get_bot_pid():
  """Читает PID из файла состояния."""
  status = state_manager.get_status()
  if status and status.get('status') == 'running':
    return status.get('pid')
  return None

def is_bot_run():
  """Проверяет, запущен ли процесс бота по PID из файла состояния."""
  try:
    status = state_manager.get_status()
    if status and status.get('status') == 'running':
      pid = status.get('pid')
      if pid and psutil.pid_exists(pid):
        # Дополнительная проверка, что это действительно наш процесс
        try:
          process = psutil.Process(pid)
          # Проверяем, что процесс связан с Python и main.py
          cmdline = process.cmdline()
          if cmdline and any('main.py' in arg for arg in cmdline):
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
          pass
    return False
  except Exception as e:
    print(f"Ошибка проверки статуса бота: {e}")
    return False


def start_bot():
  """Запускает main.py как отдельный процесс и сохраняет его PID."""
  if is_bot_run():
    st.toast("⚠️ Бот уже запущен.")
    return

  try:
    # Используем Popen для неблокирующего запуска
    if sys.platform == 'win32':
      bot_process = subprocess.Popen(
        [sys.executable, "main.py"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
      )
    else:
      bot_process = subprocess.Popen([sys.executable, "main.py"])

    pid = bot_process.pid

    st.session_state.bot_process = bot_process

    # Сохраняем PID в файл состояния
    state_manager.set_status('running', pid)
    st.toast(f"🚀 Бот запущен с PID: {pid}")

  except Exception as e:
    st.error(f"Ошибка при запуске бота: {e}")


def stop_bot():
  """Находит и принудительно завершает процесс бота."""
  pid = get_bot_pid()
  if not pid or not is_bot_run():
    st.toast("⚠️ Бот не запущен.")
    state_manager.set_status('stopped', None)
    st.session_state.bot_process = None
    return

  try:
    if sys.platform == 'win32':
      # Windows
      subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        check=True, capture_output=True
      )
    else:
      # Linux/Mac
      parent = psutil.Process(pid)
      for child in parent.children(recursive=True):
        child.kill()
      parent.kill()

    st.toast(f"✅ Бот остановлен (PID: {pid})")

  except Exception as e:
    st.error(f"Ошибка при остановке бота: {e}")
  finally:
    state_manager.set_status('stopped', None)


def get_recent_trades():
  """Получение последних сделок"""
  try:
    return asyncio.run(db_manager.get_all_trades(10))
  except Exception as e:
    st.error(f"Ошибка получения сделок: {e}")
    return []


def get_trading_stats():
  """Получение торговой статистики"""
  try:
    return asyncio.run(db_manager.get_trading_metrics_optimized(30))
  except Exception as e:
    st.error(f"Ошибка получения статистики: {e}")
    return {}

# --- Вспомогательные функции ---
def get_bot_instance():
  """Получает экземпляр бота если он запущен"""
  # В реальной реализации здесь должен быть механизм получения ссылки на запущенный бот
  # Например, через shared memory, pickle файл или другой IPC механизм
  return None




def update_ml_models_state(use_enhanced: bool, use_base: bool):
  """Обновляет состояние ML моделей через StateManager"""
  ml_state = {
    'use_enhanced_ml': use_enhanced,
    'use_base_ml': use_base,
    'updated_at': datetime.now().isoformat()
  }
  state_manager.set_custom_data('ml_models_state', ml_state)
  state_manager.set_command('update_ml_models')


def get_ml_models_state():
  """Получает текущее состояние ML моделей"""
  ml_state = state_manager.get_custom_data('ml_models_state')
  if ml_state:
    return ml_state
  return {'use_enhanced_ml': True, 'use_base_ml': True}

def load_shadow_trading_config():
    """Загружает конфигурацию Shadow Trading из папки config"""
    try:
        import json
        import os

        config_path = "config/enhanced_shadow_trading_config.json"

        if not os.path.exists(config_path):
            return {}

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get('enhanced_shadow_trading', {})

    except Exception as e:
        print(f"Ошибка загрузки Shadow Trading конфигурации: {e}")
        return {}


def get_shadow_trading_config_display():
  """Получает настройки Shadow Trading для отображения в dashboard"""
  shadow_config = load_shadow_trading_config()

  if not shadow_config:
    return {
      "Shadow Trading": {
        "Статус": "❌ Конфигурация не найдена",
        "Файл": "enhanced_shadow_trading_config.json",
        "Расположение": "Папка config/"
      }
    }

  monitoring = shadow_config.get('monitoring', {})
  analytics = shadow_config.get('analytics', {})
  alerts = shadow_config.get('alerts', {})
  performance = shadow_config.get('performance_thresholds', {})
  reporting = shadow_config.get('reporting', {})
  optimization = shadow_config.get('optimization', {})

  return {
    "Shadow Trading": {
      "✅ Статус": "Включен" if shadow_config.get('enabled', False) else "Отключен",
      "📦 Версия": shadow_config.get('version', 'N/A'),
      "⏱️ Интервал обновления": f"{monitoring.get('price_update_interval_seconds', 30)} сек",
      "🕐 Время отслеживания": f"{monitoring.get('signal_tracking_duration_hours', 24)} ч",
      "📊 Макс. отслеживание": f"{monitoring.get('max_concurrent_tracking', 1000)} сигналов",
      "🧠 Продвинутая аналитика": "✅" if analytics.get('advanced_patterns_enabled', False) else "❌",
      "🔍 Детекция аномалий": "✅" if analytics.get('anomaly_detection_enabled', False) else "❌",
      "🚨 Алерты": "✅" if alerts.get('enabled', False) else "❌",
      "📱 Telegram": "✅" if alerts.get('telegram_integration', False) else "❌",
      "🎯 Целевой винрейт": f"{performance.get('target_win_rate_pct', 60)}%",
      "💰 Мин. Profit Factor": performance.get('min_profit_factor', 1.5),
      "🤖 Авто-оптимизация": "✅" if optimization.get('auto_optimization_enabled', False) else "❌",
      "📈 Авто-отчеты": "✅" if reporting.get('auto_reports_enabled', False) else "❌"
    }
  }


def get_database_health_minimal():
  """Минимальная проверка БД без зависимостей"""
  try:
    import sqlite3
    import os

    db_path = getattr(db_manager, 'db_path', 'trading_data.db')

    # Проверяем существование файла
    if not os.path.exists(db_path):
      return {
        'status': 'missing',
        'message': f'Файл БД не найден: {db_path}',
        'stats': {}
      }

    # Простая проверка соединения
    with sqlite3.connect(db_path, timeout=5.0) as conn:
      cursor = conn.execute("SELECT 1")
      result = cursor.fetchone()

      if result and result[0] == 1:
        return {
          'status': 'healthy',
          'message': 'БД работает нормально',
          'stats': {
            'database_path': db_path,
            'file_size_mb': round(os.path.getsize(db_path) / 1024 / 1024, 2)
          }
        }
      else:
        return {
          'status': 'error',
          'message': 'БД не отвечает корректно',
          'stats': {}
        }

  except Exception as e:
    return {
      'status': 'error',
      'message': f'Ошибка подключения к БД: {e}',
      'stats': {},
      'error': str(e)
    }

def get_strategy_performance():
  """Получает данные о производительности стратегий из БД"""
  try:
    # Получаем закрытые сделки за последние 30 дней
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Используем асинхронный запрос через asyncio.run
    query = """
          SELECT COALESCE(strategy_name, 'Unknown') as strategy_name, 
                 COUNT(*) as total_trades,
                 SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                 SUM(profit_loss) as total_profit,
                 AVG(profit_loss) as avg_profit,
                 MAX(profit_loss) as max_profit,
                 MIN(profit_loss) as max_loss
          FROM trades 
          WHERE status = 'CLOSED' 
          AND close_timestamp BETWEEN ? AND ?
          GROUP BY strategy_name
      """

    # Выполняем асинхронный запрос
    result = asyncio.run(db_manager.execute_query(query, (start_date.isoformat(), end_date.isoformat())))

    performance = {}
    for row in result:
      strategy_name = row[0]
      total_trades = row[1]
      wins = row[2]
      total_profit = row[3] or 0

      performance[strategy_name] = {
        'total_trades': total_trades,
        'wins': wins,
        'losses': total_trades - wins,
        'win_rate': wins / total_trades if total_trades > 0 else 0,
        'total_profit': total_profit,
        'avg_profit': row[4] or 0,
        'max_profit': row[5] or 0,
        'max_loss': row[6] or 0,
        'profit_factor': abs(row[5] / row[6]) if row[6] and row[6] != 0 else 0
      }

    return performance

  except Exception as e:
    st.error(f"Ошибка получения производительности стратегий: {e}")
    return {}

def get_market_regimes():
  """Получает текущие рыночные режимы из состояния"""
  regimes_data = state_manager.get_custom_data('market_regimes')
  return regimes_data or {}


# --- Боковая панель с управлением ---
# bot_pid = get_bot_pid()
# if bot_pid:
#   st.sidebar.success(f"✅ Бот запущен (PID: {bot_pid})")
#   if st.sidebar.button("🛑 Остановить бота"):
#     if stop_bot():
#       st.sidebar.success("Бот остановлен")
#       time.sleep(1)
#       st.rerun()
# else:
#   st.sidebar.error("❌ Бот не запущен")
#   if st.sidebar.button("▶️ Запустить бота"):
#     if start_bot():
#       st.sidebar.success("Бот запущен")
#       time.sleep(1)
#       st.rerun()

# Обновление страницы
if st.sidebar.button("🔄 Обновить данные"):
  st.rerun()

# Вкладки
tab1, tab2, tab3, tab4 = st.tabs(["📊 Общая статистика", "📈 Сделки", "🎯 Shadow Trading", "⚙️ Настройки"])

with tab1:
  st.header("📊 Общая статистика")

  # Метрики
  stats = get_trading_stats()

  col1, col2, col3, col4 = st.columns(4)

  with col1:
    st.metric(
      "Всего сделок",
      stats.get('total_trades', 0)
    )

  with col2:
    profitable = stats.get('profitable_trades', 0)
    total = stats.get('total_trades', 0)
    win_rate = (profitable / total * 100) if total > 0 else 0
    st.metric(
      "Прибыльных сделок",
      profitable,
      delta=f"{win_rate:.1f}% винрейт"
    )

  with col3:
    total_pnl = stats.get('total_profit_loss', 0)
    st.metric(
      "Общий P&L",
      f"{total_pnl:.2f} USDT",
      delta=total_pnl
    )

  with col4:
    avg_pnl = stats.get('avg_profit_loss', 0)
    st.metric(
      "Средний P&L",
      f"{avg_pnl:.2f} USDT"
    )

with tab2:
  st.header("📈 Последние сделки")

  trades = get_recent_trades()

  if trades:
    # Преобразуем в DataFrame для удобного отображения
    df = pd.DataFrame(trades)

    # Форматируем отображение
    display_columns = ['symbol', 'side', 'open_price', 'quantity', 'status', 'profit_loss', 'open_timestamp']
    if all(col in df.columns for col in display_columns):
      df_display = df[display_columns].copy()
      df_display['open_timestamp'] = pd.to_datetime(df_display['open_timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
      df_display = df_display.rename(columns={
        'symbol': 'Символ',
        'side': 'Сторона',
        'open_price': 'Цена входа',
        'quantity': 'Количество',
        'status': 'Статус',
        'profit_loss': 'P&L',
        'open_timestamp': 'Время открытия'
      })

      st.dataframe(df_display, use_container_width=True)
    else:
      st.dataframe(df, use_container_width=True)
  else:
    st.info("Нет данных о сделках")

with tab3:
  st.header("🎯 Shadow Trading System")

  # Сводка Shadow Trading
  col1, col2 = st.columns([2, 1])

  with col1:
    st.subheader("📊 Статистика за сегодня")
    today_stats = get_shadow_trading_today_stats()

    col_a, col_b = st.columns(2)
    with col_a:
      st.metric("Сигналов сегодня", today_stats['today_signals'])
    with col_b:
      st.metric("Прибыльных сегодня", today_stats['today_profitable'])

  with col2:
    st.subheader("🎛️ Настройки периода")
    days_period = st.selectbox("Период анализа", [1, 3, 7, 14, 30], index=2)

  # Детальная статистика
  st.subheader(f"📈 Подробная статистика за {days_period} дней")

  # Используем расширенную статистику
  detailed_stats = get_enhanced_shadow_stats(days_period)

  if detailed_stats['total_signals'] > 0:
    # Основные метрики (остается как есть)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
      st.metric("Всего сигналов", detailed_stats['total_signals'])

    with col2:
      win_rate = detailed_stats.get('win_rate', 0.0)
      st.metric(
        "Винрейт",
        f"{win_rate:.1f}%",
        delta=f"{detailed_stats['profitable_signals']}/{detailed_stats['total_signals']}"
      )

    with col3:
      filter_rate = (detailed_stats['filtered_signals'] / detailed_stats['total_signals']) * 100
      st.metric(
        "Отфильтровано",
        f"{filter_rate:.1f}%",
        delta=f"{detailed_stats['filtered_signals']} сигналов"
      )

    with col4:
      st.metric(
        "Средняя уверенность",
        f"{detailed_stats['avg_confidence']:.2f}"
      )

    # ДОБАВИТЬ новую секцию с расширенными метриками:
    st.subheader("📊 Расширенные метрики производительности")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
      max_win = detailed_stats.get('max_win_pct', 0.0)
      st.metric(
        "Лучший сигнал",
        f"+{max_win:.2f}%",
        delta="максимальная прибыль"
      )

    with col2:
      max_loss = detailed_stats.get('max_loss_pct', 0.0)
      st.metric(
        "Худший сигнал",
        f"{max_loss:.2f}%",
        delta="максимальный убыток"
      )

    with col3:
      avg_win = detailed_stats.get('avg_win_pct', 0.0)
      avg_loss = detailed_stats.get('avg_loss_pct', 0.0)
      if avg_loss != 0:
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
      else:
        profit_factor = float('inf') if avg_win > 0 else 0
      st.metric(
        "Profit Factor",
        f"{profit_factor:.2f}",
        delta="среднее соотношение"
      )

    with col4:
      pending_signals = (detailed_stats['total_signals'] -
                         detailed_stats['profitable_signals'] -
                         detailed_stats['loss_signals'] -
                         detailed_stats['filtered_signals'])
      st.metric(
        "В ожидании",
        pending_signals,
        delta="активных сигналов"
      )

  # ДОБАВИТЬ график сравнения с предыдущим периодом:

  st.subheader("📈 Динамика по периодам")

  # Получаем статистику за разные периоды для сравнения
  periods = [1, 3, 7, 14, 30]
  period_stats = {}

  for period in periods:
    try:
      stats = get_enhanced_shadow_stats(period)
      period_stats[f"{period}д"] = {
        'signals': stats.get('total_signals', 0),
        'win_rate': stats.get('win_rate', 0.0),
        'avg_confidence': stats.get('avg_confidence', 0.0)
      }
    except:
      period_stats[f"{period}д"] = {'signals': 0, 'win_rate': 0.0, 'avg_confidence': 0.0}

  if period_stats:
    # Создаем DataFrame для графика
    import pandas as pd

    df_periods = pd.DataFrame(period_stats).T
    df_periods.index.name = 'Период'

    # График винрейта по периодам
    col1, col2 = st.columns(2)

    with col1:
      fig_winrate = go.Figure()
      fig_winrate.add_trace(go.Bar(
        x=df_periods.index,
        y=df_periods['win_rate'],
        name='Винрейт %',
        marker_color='green'
      ))
      fig_winrate.update_layout(
        title="Винрейт по периодам",
        xaxis_title="Период",
        yaxis_title="Винрейт (%)",
        height=300
      )
      st.plotly_chart(fig_winrate, use_container_width=True)

    with col2:
      fig_signals = go.Figure()
      fig_signals.add_trace(go.Bar(
        x=df_periods.index,
        y=df_periods['signals'],
        name='Количество сигналов',
        marker_color='blue'
      ))
      fig_signals.update_layout(
        title="Количество сигналов по периодам",
        xaxis_title="Период",
        yaxis_title="Сигналов",
        height=300
      )
      st.plotly_chart(fig_signals, use_container_width=True)
    # График распределения результатов
    st.subheader("📊 Распределение результатов")

    # Данные для круговой диаграммы
    labels = ['Прибыльные', 'Убыточные', 'Отфильтрованные', 'В ожидании']
    values = [
      detailed_stats['profitable_signals'],
      detailed_stats['loss_signals'],
      detailed_stats['filtered_signals'],
      detailed_stats['total_signals'] - detailed_stats['profitable_signals'] -
      detailed_stats['loss_signals'] - detailed_stats['filtered_signals']
    ]

    fig = go.Figure(data=[go.Pie(
      labels=labels,
      values=values,
      hole=.3,
      textinfo='label+percent',
      textfont_size=12
    )])

    fig.update_layout(
      title="Распределение результатов сигналов",
      height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Детальная таблица с показателями
    st.subheader("📋 Детальные показатели")

    metrics_data = {
      'Показатель': [
        'Средняя прибыль на выигрышной сделке',
        'Средний убыток на убыточной сделке',
        'Общее количество сигналов',
        'Процент отфильтрованных сигналов',
        'Средний уровень уверенности'
      ],
      'Значение': [
        f"{detailed_stats['avg_win_pct']:.2f}%",
        f"{detailed_stats['avg_loss_pct']:.2f}%",
        f"{detailed_stats['total_signals']}",
        f"{filter_rate:.1f}%",
        f"{detailed_stats['avg_confidence']:.3f}"
      ]
    }

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

  else:
    st.info("Нет данных о сигналах Shadow Trading за выбранный период")

  # Краткая сводка
  st.subheader("📝 Краткая сводка")
  summary = create_shadow_trading_summary(days_period)
  st.markdown(summary)

with tab4:
  st.header("⚙️ Настройки системы")

  # Загрузка текущей конфигурации
  try:
    current_config = config_manager.load_config()
    st.subheader("📄 Текущая конфигурация")

    # Основные настройки (оставляем как есть)
    config_display = {
      "API настройки": {
        "Тестовая среда": current_config.get('testnet', False),
        "Таймаут запросов": f"{current_config.get('request_timeout', 30)} сек",
      },
      "Торговые настройки": {
        "Максимальный риск": f"{current_config.get('max_risk_per_trade', 2)}%",
        "Максимум открытых позиций": current_config.get('max_open_positions', 3),
      }
    }

    # Добавляем настройки Shadow Trading из отдельного файла
    shadow_display = get_shadow_trading_config_display()
    config_display.update(shadow_display)

    for section, settings in config_display.items():
      st.subheader(f"🔧 {section}")
      for setting, value in settings.items():
        st.write(f"**{setting}:** {value}")

    # Кнопка для перезагрузки конфигурации
    if st.button("🔄 Перезагрузить конфигурацию"):
      config_manager = ConfigManager(config_path=CONFIG_FILE_PATH)
      st.success("Конфигурация перезагружена")

  except Exception as e:
    st.error(f"Ошибка загрузки конфигурации: {e}")

  st.subheader("🗄️ Состояние базы данных")

  try:
    db_health = get_database_health_minimal()

    # Основные метрики
    col1, col2, col3, col4 = st.columns(4)

    with col1:
      status = db_health.get('status', 'unknown')
      status_icons = {
        'healthy': '🟢',
        'warning': '🟡',
        'error': '🔴',
        'locked': '🔒',
        'missing': '❌'
      }
      icon = status_icons.get(status, '❓')
      st.metric("Статус БД", f"{icon} {status.title()}")

    with col2:
      response_time = db_health.get('response_time_ms', -1)
      if response_time >= 0:
        st.metric("Время отклика", f"{response_time:.1f} мс")
      else:
        st.metric("Время отклика", "N/A")

    with col3:
      stats = db_health.get('stats', {})
      total_ops = stats.get('total_operations', 0)
      st.metric("Всего операций", total_ops)

    with col4:
      error_rate = stats.get('error_rate_pct', 0)
      st.metric("Процент ошибок", f"{error_rate:.1f}%")

    # Дополнительная информация
    if stats:
      col1, col2 = st.columns(2)

      with col1:
        lock_errors = stats.get('lock_errors', 0)
        st.metric("Блокировки БД", lock_errors)

        if lock_errors > 0:
          last_lock = stats.get('last_lock_time')
          if last_lock:
            time_ago = time.time() - last_lock
            if time_ago < 60:
              st.metric("Последняя блокировка", f"{time_ago:.0f} сек назад")
            else:
              st.metric("Последняя блокировка", f"{time_ago / 60:.0f} мин назад")

      with col2:
        db_path = db_health.get('database_path', 'unknown')
        st.metric("Путь к БД", os.path.basename(db_path) if db_path != 'unknown' else 'N/A')

        pool_size = db_health.get('pool_size', 'unknown')
        st.metric("Размер пула", pool_size)

    # Алерты и предупреждения
    if db_health['status'] == 'error':
      st.error(f"❌ Ошибка БД: {db_health.get('message', 'Неизвестная ошибка')}")
    elif db_health['status'] == 'locked':
      st.error(f"🔒 БД заблокирована: {db_health.get('message', '')}")
    elif db_health['status'] == 'warning':
      st.warning(f"⚠️ Предупреждение БД: {db_health.get('message', '')}")
    elif db_health['status'] == 'missing':
      st.error(f"❌ БД не найдена: {db_health.get('message', '')}")

    # Рекомендации
    if stats.get('lock_errors', 0) > 10:
      st.warning(f"🚨 Много блокировок БД ({stats['lock_errors']}). Рекомендации:")
      st.write("• Перезапустить систему")
      st.write("• Проверить нагрузку на БД")
      st.write("• Увеличить таймауты")

    if error_rate > 10:
      st.warning(f"⚠️ Высокий процент ошибок БД ({error_rate:.1f}%)")

  except Exception as e:
    st.error(f"❌ Ошибка получения состояния БД: {e}")
  # # Загрузка текущей конфигурации
  # try:
  #   current_config = config_manager.load_config()
  #   st.subheader("📄 Текущая конфигурация")
  #
  #   # Отображаем основные настройки в удобном виде
  #   config_display = {
  #     "API настройки": {
  #       "Тестовая среда": current_config.get('testnet', False),
  #       "Таймаут запросов": f"{current_config.get('request_timeout', 30)} сек",
  #     },
  #     "Торговые настройки": {
  #       "Максимальный риск": f"{current_config.get('max_risk_per_trade', 2)}%",
  #       "Максимум открытых позиций": current_config.get('max_open_positions', 3),
  #     },
  #     "Shadow Trading": {
  #       "Включен": current_config.get('shadow_trading', {}).get('enabled', False),
  #       "Отслеживание цен": current_config.get('shadow_trading', {}).get('price_monitoring', True),
  #     }
  #   }
  #
  #   for section, settings in config_display.items():
  #     st.subheader(f"🔧 {section}")
  #     for setting, value in settings.items():
  #       st.write(f"**{setting}:** {value}")
  #
  #   # Кнопка для перезагрузки конфигурации
  #   if st.button("🔄 Перезагрузить конфигурацию"):
  #     config_manager = ConfigManager(config_path=CONFIG_FILE_PATH)
  #     st.success("Конфигурация перезагружена")
  #
  # except Exception as e:
  #   st.error(f"Ошибка загрузки конфигурации: {e}")

  # Информация о базе данных
  st.subheader("🗄️ Информация о базе данных")
  st.write(f"**Путь к БД:** {db_manager.db_path}")

  # Проверка таблиц Shadow Trading
  try:
    import sqlite3

    with sqlite3.connect(db_manager.db_path) as conn:
      cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
      tables = [row[0] for row in cursor.fetchall()]

      shadow_tables = [t for t in tables if t in ['signal_analysis', 'price_tracking']]

      if shadow_tables:
        st.success(f"✅ Таблицы Shadow Trading найдены: {', '.join(shadow_tables)}")
      else:
        st.warning("⚠️ Таблицы Shadow Trading не найдены")

        if st.button("🔨 Создать таблицы Shadow Trading"):
          if initialize_shadow_trading():
            st.success("✅ Таблицы Shadow Trading созданы успешно")
            st.rerun()
          else:
            st.error("❌ Ошибка создания таблиц")

  except Exception as e:
    st.error(f"Ошибка проверки базы данных: {e}")

# --- АВТООБНОВЛЕНИЕ ---
# Автоматическое обновление каждые 30 секунд
time.sleep(0.1)  # Небольшая задержка для корректной работы
if st.checkbox("🔄 Автообновление (30 сек)", value=False):
  time.sleep(30)
  st.rerun()
with st.sidebar:
  st.title("🕹️ Управление ботом")



  col1, col2 = st.columns(2)

  with col1:
    if st.button("🚀 Запустить", type="primary", use_container_width=True):
      start_bot()
      time.sleep(1)
      st.rerun()

  with col2:
    if st.button("🛑 Остановить", use_container_width=True):
      stop_bot()
      time.sleep(1)
      st.rerun()

  # Показываем текущий статус
  if is_bot_run():
    pid = get_bot_pid()
    st.success(f"✅ Бот работает (PID: {pid})")
  else:
    st.warning("❌ Бот остановлен")

  st.divider()
  # --- УПРОЩЕННАЯ КНОПКА SHADOW TRADING ---
  st.subheader("🌟 Shadow Trading")

  # Показываем простые метрики
  display_simple_shadow_metrics()

  if st.button("📊 Подробная аналитика", use_container_width=True):
    st.info("🔄 Для подробной аналитики Shadow Trading запустите бота и дождитесь накопления данных")

  st.divider()

  # --- Управление ML моделями ---
  st.subheader("🤖 ML Модели")

  ml_state = get_ml_models_state()

  use_enhanced = st.checkbox(
    "Enhanced ML (Ансамбль)",
    value=ml_state.get('use_enhanced_ml', True),
    help="Расширенная модель с мета-обучением"
  )

  use_base = st.checkbox(
    "Base ML (Основная)",
    value=ml_state.get('use_base_ml', True),
    help="Базовая ML стратегия"
  )

  if st.button("Применить ML настройки", use_container_width=True):
    update_ml_models_state(use_enhanced, use_base)
    st.success("ML настройки обновлены!")
    if not use_enhanced and not use_base:
      st.warning("⚠️ Обе ML модели выключены!")

  st.divider()

  # Кнопка обновления
  if st.button("🔄 Обновить данные", use_container_width=True):
    st.rerun()

  st.divider()

  st.subheader("📊 Действия")
  if st.button("📈 Отчет о модели", use_container_width=True):
    state_manager.set_command("generate_report")
    st.toast("Команда отправлена!")

  if st.button("🔄 Переобучить модель", use_container_width=True):
    state_manager.set_command("retrain_model")
    st.toast("Запущено переобучение!")

# --- Основная часть дашборда ---
st.title("🤖 Панель управления торговым ботом")

# --- Загрузка данных ---
status = state_manager.get_status()
metrics = state_manager.get_metrics()
model_info = state_manager.get_model_info()

st.sidebar.write("🔍 **Отладка метрик:**")
if metrics:
  st.sidebar.success(f"✅ Метрики найдены")
  st.sidebar.write(f"Баланс: {metrics.total_balance_usdt:.2f}")
else:
  st.sidebar.error("❌ Метрики отсутствуют")

# Проверяем содержимое файла состояния
state_file_content = state_manager._read_state()
st.sidebar.write(f"📄 Ключи в файле: {list(state_file_content.keys())}")

# Получаем позиции
open_positions_list = state_manager.get_open_positions()
closed_trades_list = asyncio.run(db_manager.get_all_trades(limit=1000))

df_open = pd.DataFrame(open_positions_list) if open_positions_list else pd.DataFrame()
df_closed = pd.DataFrame(closed_trades_list) if closed_trades_list else pd.DataFrame()

# --- НОВЫЙ БЛОК ---
# Загружаем конфиг один раз для всего дашборда
current_config = config_manager.load_config()
trade_cfg = current_config.get('trade_settings', {})
strategy_cfg = current_config.get('strategy_settings', {})
# --- КОНЕЦ НОВОГО БЛОКА ---



col_status, col_ml = st.columns([3, 1])

with col_status:
  if is_bot_run():
    # Используем функцию get_bot_pid() для получения PID из файла состояния
    pid = get_bot_pid()
    st.success(f"🟢 **Статус: Бот работает** (PID: {pid})")
  else:
    st.warning("🟡 **Статус: Бот остановлен**")

with col_ml:
  ml_state = get_ml_models_state()
  enhanced_status = "✅" if ml_state.get('use_enhanced_ml', True) else "❌"
  base_status = "✅" if ml_state.get('use_base_ml', True) else "❌"
  st.metric("ML Модели", f"E:{enhanced_status} B:{base_status}")

# Финансовые метрики
st.subheader("💰 Финансовые показатели")
cols = st.columns(5)
if metrics:
  cols[0].metric("Общий баланс", f"${metrics.total_balance_usdt:.2f}")
  cols[1].metric("Доступный баланс", f"${metrics.available_balance_usdt:.2f}")
  cols[2].metric("Нереализованный PnL", f"${metrics.unrealized_pnl_total:.2f}")
  cols[3].metric("Реализованный PnL", f"${metrics.realized_pnl_total:.2f}")

  # Рассчитываем ROI
  if metrics.total_balance_usdt > 0:
    roi = (metrics.realized_pnl_total / metrics.total_balance_usdt) * 100
    cols[4].metric("ROI", f"{roi:.2f}%")
else:
  # Fallback: пробуем получить данные напрямую из БД
  try:
    recent_trades = asyncio.run(db_manager.get_all_trades(limit=100))
    if recent_trades:
      total_pnl = sum(trade.get('profit_loss', 0) for trade in recent_trades if trade.get('profit_loss'))
      cols[0].metric("Общий баланс", "Недоступно")
      cols[1].metric("Доступный баланс", "Недоступно")
      cols[2].metric("Нереализованный PnL", "Недоступно")
      cols[3].metric("Реализованный PnL", f"${total_pnl:.2f}")
      cols[4].metric("ROI", "Недоступно")
    else:
      for i, col in enumerate(cols):
        col.metric(["Общий баланс", "Доступный баланс", "Нереализованный PnL", "Реализованный PnL", "ROI"][i], "Нет данных")
  except Exception as e:
    st.error(f"Ошибка получения fallback метрик: {e}")
    for i, col in enumerate(cols):
      col.metric(["Общий баланс", "Доступный баланс", "Нереализованный PnL", "Реализованный PnL", "ROI"][i], "Ошибка")
st.divider()

# --- Вкладки ---
tabs = st.tabs([
  "📊 Мониторинг",
  "📈 Производительность",
  "🎯 Стратегии",
  "🌍 Режимы рынка",
  "📉 Анализ",
  "📊 ROI Калькулятор",
  "⚙️ Настройки"
])

#if st.button("🌟 Shadow Trading", use_container_width=True):
#  st.session_state.page = "shadow_trading"
#  st.rerun()

# --- Вкладка: Мониторинг ---
with tabs[0]:
  col1, col2 = st.columns([1, 1])

  with col1:
    st.subheader("🟢 Активные позиции")
    if not df_open.empty:
      # Добавляем текущий PnL для активных позиций
      if 'current_price' in df_open.columns and 'open_price' in df_open.columns:
        df_open['current_pnl'] = (df_open['current_price'] - df_open['open_price']) * df_open['quantity']
        df_open['current_pnl_pct'] = ((df_open['current_price'] - df_open['open_price']) / df_open['open_price']) * 100

      display_cols = ['open_timestamp', 'symbol', 'side', 'quantity', 'open_price']

      # Добавляем колонки PnL если они есть
      if 'current_price' in df_open.columns and 'open_price' in df_open.columns:
        # Рассчитываем текущий PnL
        df_open['current_pnl'] = (df_open['current_price'] - df_open['open_price']) * df_open.get('quantity', 0)
        df_open['current_pnl_pct'] = ((df_open['current_price'] - df_open['open_price']) / df_open['open_price']) * 100
        display_cols.extend(['current_price', 'current_pnl', 'current_pnl_pct'])

      # Проверяем какие колонки действительно существуют
      available_cols = [col for col in display_cols if col in df_open.columns]

      if available_cols:
        df_display = df_open[available_cols].copy()

        # Форматирование времени
        if 'open_timestamp' in df_display.columns:
          df_display['open_timestamp'] = pd.to_datetime(df_display['open_timestamp']).dt.strftime('%H:%M:%S')
      else:
        # Если нет нужных колонок, показываем все доступные
        df_display = df_open.copy()
        st.info("Показаны все доступные данные (стандартные колонки не найдены)")

      # Форматирование
      df_display = df_open[available_cols].copy()
      if 'open_timestamp' in df_display.columns:
        df_display['open_timestamp'] = pd.to_datetime(df_display['open_timestamp']).dt.strftime('%H:%M:%S')


      # Цветовое кодирование PnL
      def color_pnl(val):
        if isinstance(val, (int, float)):
          color = 'green' if val > 0 else 'red' if val < 0 else 'black'
          return f'color: {color}'
        return ''


      # styled_df = df_display.style.applymap(color_pnl, subset=['current_pnl',
      # 'current_pnl_pct'] if 'current_pnl' in df_display.columns else [])
      try:
        # Определяем какие колонки PnL есть в данных
        pnl_columns = []
        possible_pnl_cols = ['current_pnl', 'current_pnl_pct', 'profit_pct', 'profit_usd', 'profit_loss']

        for col in possible_pnl_cols:
          if col in df_display.columns:
            pnl_columns.append(col)

        if pnl_columns:
          # Применяем стили только к существующим колонкам PnL
          def color_pnl(val):
            if pd.isna(val):
              return ''
            try:
              num_val = float(val)
              return 'color: green' if num_val > 0 else 'color: red' if num_val < 0 else 'color: gray'
            except (ValueError, TypeError):
              return ''


          styled_df = df_display.style.map(color_pnl, subset=pnl_columns)
          st.dataframe(styled_df, use_container_width=True)
        else:
          # Если нет колонок PnL, показываем без стилей
          st.dataframe(df_display, use_container_width=True)

      except Exception as e:
        # Fallback: показываем данные без стилей
        st.warning(f"Ошибка применения стилей: {e}")
        st.dataframe(df_display, use_container_width=True)
    else:
      st.info("Нет активных позиций")

  with col2:
    st.subheader("📊 Кривая доходности")
    if not df_closed.empty and 'profit_loss' in df_closed.columns:
      df_closed['timestamp'] = pd.to_datetime(df_closed['close_timestamp'])
      df_closed_sorted = df_closed.sort_values('timestamp')
      df_closed_sorted['cumulative_pnl'] = df_closed_sorted['profit_loss'].cumsum()

      fig = go.Figure()
      fig.add_trace(go.Scatter(
        x=df_closed_sorted['timestamp'],
        y=df_closed_sorted['cumulative_pnl'],
        mode='lines',
        name='Накопленный PnL',
        line=dict(color='green' if df_closed_sorted['cumulative_pnl'].iloc[-1] > 0 else 'red')
      ))

      fig.update_layout(
        title="Накопленная прибыль",
        xaxis_title="Время",
        yaxis_title="PnL (USDT)",
        height=300
      )

      st.plotly_chart(fig, use_container_width=True)
    else:
      st.info("Нет данных для графика")

  # История сделок
  st.subheader("📋 История закрытых сделок")
  if not df_closed.empty:
    # Добавляем фильтры
    col_filter1, col_filter2, col_filter3 = st.columns(3)

    with col_filter1:
      symbol_filter = st.selectbox(
        "Символ",
        options=['Все'] + sorted(df_closed['symbol'].unique().tolist()),
        key='symbol_filter'
      )

    with col_filter2:
      strategy_filter = st.selectbox(
        "Стратегия",
        options=['Все'] + sorted(
          df_closed['strategy_name'].unique().tolist()) if 'strategy_name' in df_closed.columns else ['Все'],
        key='strategy_filter'
      )

    with col_filter3:
      profit_filter = st.selectbox(
        "Результат",
        options=['Все', 'Прибыльные', 'Убыточные'],
        key='profit_filter'
      )

    # Применяем фильтры
    df_filtered = df_closed.copy()

    if symbol_filter != 'Все':
      df_filtered = df_filtered[df_filtered['symbol'] == symbol_filter]

    if strategy_filter != 'Все' and 'strategy_name' in df_filtered.columns:
      df_filtered = df_filtered[df_filtered['strategy_name'] == strategy_filter]

    if profit_filter == 'Прибыльные':
      df_filtered = df_filtered[df_filtered['profit_loss'] > 0]
    elif profit_filter == 'Убыточные':
      df_filtered = df_filtered[df_filtered['profit_loss'] < 0]

    # Отображаем
    display_cols = ['close_timestamp', 'symbol', 'strategy_name', 'side', 'quantity',
                    'open_price', 'close_price', 'profit_loss', 'profit_pct']
    available_cols = [col for col in display_cols if col in df_filtered.columns]

    df_display = df_filtered[available_cols].copy()
    if 'close_timestamp' in df_display.columns:
      df_display['close_timestamp'] = pd.to_datetime(df_display['close_timestamp']).dt.strftime('%Y-%m-%d %H:%M')

    # Добавляем процент прибыли если нет
    if 'profit_pct' not in df_display.columns and 'profit_loss' in df_display.columns and 'open_price' in df_display.columns:
      df_display['profit_pct'] = (df_display['profit_loss'] / (df_display['open_price'] * df_display['quantity'])) * 100

    st.dataframe(df_display.sort_values('close_timestamp', ascending=False), use_container_width=True, height=400)

    # Статистика по фильтру
    if len(df_filtered) > 0:
      col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
      col_stat1.metric("Всего сделок", len(df_filtered))
      col_stat2.metric("Прибыльных", len(df_filtered[df_filtered['profit_loss'] > 0]))
      col_stat3.metric("Win Rate",
                       f"{(len(df_filtered[df_filtered['profit_loss'] > 0]) / len(df_filtered) * 100):.1f}%")
      col_stat4.metric("Общий PnL", f"${df_filtered['profit_loss'].sum():.2f}")

# --- Вкладка: Производительность ---
with tabs[1]:
  st.header("📈 Производительность стратегий")

  performance = get_strategy_performance()

  if 'shadow_manager' not in st.session_state and hasattr(st.session_state, 'trading_system'):
    setup_shadow_dashboard_integration(st.session_state.trading_system.shadow_trading)

  if performance:
    # Создаем DataFrame для удобства
    perf_data = []
    for strategy, metrics in performance.items():
      perf_data.append({
        'Стратегия': strategy,
        'Сделок': metrics['total_trades'],
        'Побед': metrics['wins'],
        'Поражений': metrics['losses'],
        'Win Rate': f"{metrics['win_rate'] * 100:.1f}%",
        'Общая прибыль': f"${metrics['total_profit']:.2f}",
        'Средняя прибыль': f"${metrics['avg_profit']:.2f}",
        'Макс прибыль': f"${metrics['max_profit']:.2f}",
        'Макс убыток': f"${metrics['max_loss']:.2f}",
        'Profit Factor': f"{metrics['profit_factor']:.2f}"
      })

    df_perf = pd.DataFrame(perf_data)

    # Сортируем по Win Rate
    df_perf['_wr'] = df_perf['Win Rate'].str.rstrip('%').astype(float)
    df_perf = df_perf.sort_values('_wr', ascending=False).drop('_wr', axis=1)

    st.dataframe(df_perf, use_container_width=True, hide_index=True)

    # Графики производительности
    col1, col2 = st.columns(2)

    with col1:
      # График Win Rate по стратегиям
      fig_wr = px.bar(
        x=list(performance.keys()),
        y=[m['win_rate'] * 100 for m in performance.values()],
        title="Win Rate по стратегиям",
        labels={'x': 'Стратегия', 'y': 'Win Rate (%)'}
      )
      fig_wr.add_hline(y=50, line_dash="dash", line_color="red")
      st.plotly_chart(fig_wr, use_container_width=True)

    with col2:
      # График прибыли по стратегиям
      fig_profit = px.bar(
        x=list(performance.keys()),
        y=[m['total_profit'] for m in performance.values()],
        title="Общая прибыль по стратегиям",
        labels={'x': 'Стратегия', 'y': 'Прибыль (USDT)'},
        color=[m['total_profit'] for m in performance.values()],
        color_continuous_scale=['red', 'yellow', 'green']
      )
      st.plotly_chart(fig_profit, use_container_width=True)

    # Адаптивные веса (если доступны)
    st.divider()
    adaptive_weights = state_manager.get_custom_data('adaptive_weights')

    if adaptive_weights:
      st.subheader("⚖️ Адаптивные веса стратегий")

      weights_data = []
      for strategy, weight in adaptive_weights.items():
        weights_data.append({
          'Стратегия': strategy,
          'Текущий вес': f"{weight:.2f}",
          'Статус': '✅ Активна' if weight > 0.5 else '⚠️ Снижен вес' if weight > 0 else '❌ Отключена'
        })

      df_weights = pd.DataFrame(weights_data)
      st.dataframe(df_weights, use_container_width=True, hide_index=True)
    else:
      st.info("ℹ️ Адаптивные веса стратегий пока недоступны")

      # Показываем почему нет данных
      with st.expander("🔍 Отладка адаптивных весов"):
        st.write("Возможные причины:")
        st.write("• Бот не запущен")
        st.write("• AdaptiveStrategySelector не инициализирован")
        st.write("• Недостаточно сделок для расчета весов")
        st.write("• Данные не обновляются в state_manager")
#---------------------------------------------------------------------------------------------------------------------
# st.divider()

sar_performance = state_manager.get_custom_data('sar_strategy_performance')
st.sidebar.write(f"SAR данные: {'✅ найдены' if sar_performance else '❌ отсутствуют'}")

sar_performance = state_manager.get_custom_data('sar_strategy_performance')
if sar_performance:
  st.subheader("🎯 Производительность SAR стратегии")

  col1, col2, col3, col4 = st.columns(4)

  with col1:
    st.metric(
      "Общие сделки",
      sar_performance.get('total_trades', 0),
      delta=f"+{sar_performance.get('recent_trades_count', 0)} за период"
    )

  with col2:
    win_rate = sar_performance.get('win_rate', 0) * 100
    st.metric(
      "Win Rate",
      f"{win_rate:.1f}%",
      delta=f"{sar_performance.get('recent_win_rate', 0) * 100:.1f}% недавних"
    )

  with col3:
    st.metric(
      "Profit Factor",
      f"{sar_performance.get('profit_factor', 0):.2f}",
      delta=f"Общая прибыль: {sar_performance.get('total_profit', 0):.2f} USDT"
    )

  with col4:
    avg_trade = sar_performance.get('avg_profit_per_trade', 0)
    st.metric(
      "Средн./сделка",
      f"{avg_trade:.2f} USDT",
      delta="положительная" if avg_trade > 0 else "отрицательная"
    )

  # График адаптивных параметров
  if 'parameter_history' in sar_performance:
    st.subheader("📊 История адаптации параметров")
    param_history = sar_performance['parameter_history']

    if param_history:
      df_params = pd.DataFrame(param_history)
      df_params['timestamp'] = pd.to_datetime(df_params['timestamp'])

      fig_params = px.line(
        df_params,
        x='timestamp',
        y=['acceleration', 'sensitivity', 'confidence_threshold'],
        title="Динамика адаптивных параметров SAR"
      )
      st.plotly_chart(fig_params, use_container_width=True)

  # Кнопка экспорта отчета SAR
  if st.button("📄 Экспортировать отчет SAR"):
    state_manager.set_command('export_sar_report')
    st.toast("Запрос на экспорт отчета SAR отправлен!")
#-------------------------------------------test_--------------------------
  with st.sidebar.expander("🔍 Отладка данных", expanded=False):
    st.write("**Проверка state_manager:**")

    # Проверяем все custom_data
    all_custom_data = state_manager._read_state().get('custom_data', {})
    st.write(f"Всего ключей в custom_data: {len(all_custom_data)}")

    for key in all_custom_data.keys():
      st.write(f"• {key}")

    st.write("**Специфичные проверки:**")

    # SAR данные
    sar_data = state_manager.get_custom_data('sar_strategy_performance')
    st.write(f"SAR данные: {'✅' if sar_data else '❌'}")
    if sar_data:
      st.write(f"Содержит ключей: {len(sar_data)}")
      st.json(sar_data)

    # Адаптивные веса
    adaptive_weights = state_manager.get_custom_data('adaptive_weights')
    st.write(f"Адаптивные веса: {'✅' if adaptive_weights else '❌'}")
    if adaptive_weights:
      st.json(adaptive_weights)

    # Активные стратегии
    active_strategies = state_manager.get_custom_data('active_strategies')
    st.write(f"Активные стратегии: {'✅' if active_strategies else '❌'}")
    if active_strategies:
      st.json(active_strategies)
    t.sidebar.divider()
    st.sidebar.subheader("🧪 Тестирование")

    if st.sidebar.button("Создать тестовые данные SAR"):
      from datetime import datetime, timedelta

      test_sar = {
        'total_trades': 25,
        'winning_trades': 17,
        'losing_trades': 8,
        'win_rate': 0.68,
        'recent_win_rate': 0.75,
        'profit_factor': 2.1,
        'total_profit': 245.30,
        'total_loss': 117.20,
        'avg_profit_per_trade': 5.12,
        'recent_trades_count': 12,
        'last_update': datetime.now().isoformat(),
        'current_parameters': {
          'acceleration': 0.025,
          'max_acceleration': 0.2,
          'sensitivity': 0.82,
          'confidence_threshold': 0.73,
          'stop_loss_atr_multiplier': 1.8,
          'take_profit_atr_multiplier': 3.2
        },
        'parameter_history': [
          {
            'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
            'acceleration': 0.02,
            'sensitivity': 0.8,
            'confidence_threshold': 0.7,
            'total_trades': 20,
            'win_rate': 0.65
          },
          {
            'timestamp': datetime.now().isoformat(),
            'acceleration': 0.025,
            'sensitivity': 0.82,
            'confidence_threshold': 0.73,
            'total_trades': 25,
            'win_rate': 0.68
          }
        ],
        'monitored_symbols': 15,
        'active_positions': 3,
        'market_regime': 'trending',
        'trend_strength': 0.78
      }

      state_manager.set_custom_data('sar_strategy_performance', test_sar)
      st.sidebar.success("✅ Тестовые данные SAR созданы!")

    if st.sidebar.button("Создать тестовые адаптивные веса"):
      test_weights = {
        'Live_ML_Strategy': 1.2,
        'Ichimoku_Cloud': 0.8,
        'Dual_Thrust': 1.1,
        'Mean_Reversion_BB': 0.9,
        'Momentum_Spike': 1.3,
        'Grid_Trading': 0.7,
        'Stop_and_Reverse': 1.15
      }

      state_manager.set_custom_data('adaptive_weights', test_weights)
      st.sidebar.success("✅ Тестовые адаптивные веса созданы!")

    if st.sidebar.button("Очистить тестовые данные"):
      state_manager.set_custom_data('sar_strategy_performance', None)
      state_manager.set_custom_data('adaptive_weights', None)
      st.sidebar.info("🗑️ Тестовые данные очищены")
#--------------------------------------------------------------------------------------------------
# --- Вкладка: Стратегии ---
with tabs[2]:
  st.header("🎯 Управление стратегиями")

  # Получаем список всех стратегий из конфига или состояния
  all_strategies = [
    "Live_ML_Strategy",
    "Ichimoku_Cloud",
    "Dual_Thrust",
    "Mean_Reversion_BB",
    "Momentum_Spike",
    "Grid_Trading",
    'Stop_and_Reverse'
  ]

  # Активные стратегии
  active_strategies = state_manager.get_custom_data('active_strategies') or {s: True for s in all_strategies}

  st.subheader("Активные стратегии")

  # Создаем колонки для чекбоксов
  cols = st.columns(3)

  updated_strategies = {}
  for i, strategy in enumerate(all_strategies):
    col_idx = i % 3
    with cols[col_idx]:
      is_active = st.checkbox(
        strategy,
        value=active_strategies.get(strategy, True),
        key=f"strat_{strategy}"
      )
      updated_strategies[strategy] = is_active

  if st.button("Применить изменения стратегий"):
    state_manager.set_custom_data('active_strategies', updated_strategies)
    state_manager.set_command('update_strategies')
    st.toast("Настройки стратегий обновлены!")

  # Параметры стратегий
  st.divider()
  st.subheader("Параметры адаптации")

  col1, col2 = st.columns(2)

  with col1:
    min_win_rate = st.slider(
      "Мин. Win Rate для активации",
      min_value=0.0,
      max_value=1.0,
      value=0.3,
      step=0.05,
      help="Стратегии с Win Rate ниже этого значения будут отключены"
    )

    weight_change_rate = st.slider(
      "Скорость изменения весов",
      min_value=0.01,
      max_value=0.5,
      value=0.1,
      step=0.01,
      help="Насколько быстро адаптируются веса стратегий"
    )

  with col2:
    min_trades_eval = st.number_input(
      "Мин. сделок для оценки",
      min_value=5,
      max_value=100,
      value=10,
      help="Минимальное количество сделок для оценки стратегии"
    )

    regime_weight_bonus = st.slider(
      "Бонус веса для режима",
      min_value=0.0,
      max_value=0.5,
      value=0.2,
      step=0.05,
      help="Дополнительный вес для стратегий в подходящем режиме"
    )

  if st.button("Сохранить параметры адаптации"):
    adaptation_params = {
      'min_win_rate': min_win_rate,
      'weight_change_rate': weight_change_rate,
      'min_trades_eval': min_trades_eval,
      'regime_weight_bonus': regime_weight_bonus
    }
    state_manager.set_custom_data('adaptation_params', adaptation_params)
    st.success("Параметры адаптации сохранены!")

# --- Вкладка: Режимы рынка ---
with tabs[3]:
  st.header("🌍 Режимы рынка")

  market_regimes = get_market_regimes()

  if market_regimes:
    # Создаем таблицу с текущими режимами
    regime_data = []
    for symbol, regime_info in market_regimes.items():
      regime_data.append({
        'Символ': symbol,
        'Режим': regime_info.get('regime', 'Неизвестно'),
        'Уверенность': f"{regime_info.get('confidence', 0) * 100:.1f}%",
        'Сила тренда': f"{regime_info.get('trend_strength', 0):.2f}",
        'Волатильность': f"{regime_info.get('volatility', 0):.3f}",
        'Длительность': regime_info.get('duration', 'N/A')
      })

    df_regimes = pd.DataFrame(regime_data)


    # Цветовое кодирование режимов
    def color_regime(val):
      regime_colors = {
        'strong_trend_up': 'background-color: #4CAF50',
        'trend_up': 'background-color: #8BC34A',
        'ranging': 'background-color: #FFC107',
        'trend_down': 'background-color: #FF9800',
        'strong_trend_down': 'background-color: #F44336',
        'volatile': 'background-color: #E91E63',
        'quiet': 'background-color: #9E9E9E'
      }

      for regime, color in regime_colors.items():
        if regime in str(val).lower():
          return color
      return ''


    styled_regimes = df_regimes.style.map(color_regime, subset=['Режим'])
    st.dataframe(styled_regimes, use_container_width=True, hide_index=True)

    # Статистика режимов
    st.subheader("📊 Распределение режимов")

    if df_regimes['Режим'].value_counts().any():
      fig_regimes = px.pie(
        values=df_regimes['Режим'].value_counts().values,
        names=df_regimes['Режим'].value_counts().index,
        title="Текущее распределение режимов"
      )
      st.plotly_chart(fig_regimes, use_container_width=True)
  else:
    st.info("Нет данных о режимах рынка")

  # Исторические данные о режимах
  st.divider()
  st.subheader("📈 История изменений режимов")

  regime_history = state_manager.get_custom_data('regime_history')
  if regime_history:
    # Здесь можно добавить график изменения режимов во времени
    st.info("История режимов доступна в логах")
  else:
    st.info("История режимов пока недоступна")
#--------------------------------------------------------------------------------новое
  # После отображения текущих режимов добавьте:
  if st.button("Экспортировать статистику режимов"):
    state_manager.set_command("export_regime_statistics")
    st.success("Команда на экспорт отправлена!")

  # Отображение статистики
  if st.checkbox("Показать детальную статистику"):
    selected_symbol = st.selectbox(
      "Выберите символ для анализа",
      options=list(market_regimes.keys()) if market_regimes else []
    )

    if selected_symbol:
      # Получаем статистику через command
      state_manager.set_command("get_regime_statistics", {"symbol": selected_symbol})
      time.sleep(1)  # Даем время на обработку

      stats = state_manager.get_custom_data(f"regime_stats_{selected_symbol}")
      if stats:
        st.json(stats)
#-------------------------------------------------------------------------------------------
# --- Вкладка: Анализ ---
with tabs[4]:
  st.header("📉 Аналитика")

  analysis_type = st.selectbox(
    "Тип анализа",
    ["Анализ по символам", "Анализ по времени", "Анализ рисков", "ML метрики"]
  )

  if analysis_type == "Анализ по символам":
    if not df_closed.empty:
      # Группируем по символам
      symbol_stats = df_closed.groupby('symbol').agg({
        'profit_loss': ['count', 'sum', 'mean'],
        'quantity': 'sum'
      }).round(2)

      symbol_stats.columns = ['Сделок', 'Общий PnL', 'Средний PnL', 'Общий объем']
      symbol_stats['Win Rate'] = df_closed[df_closed['profit_loss'] > 0].groupby('symbol').size() / symbol_stats[
        'Сделок'] * 100

      st.dataframe(symbol_stats.sort_values('Общий PnL', ascending=False), use_container_width=True)

      # График топ символов
      top_symbols = symbol_stats.nlargest(10, 'Общий PnL')
      fig_top = px.bar(
        x=top_symbols.index,
        y=top_symbols['Общий PnL'],
        title="Топ-10 символов по прибыли",
        labels={'x': 'Символ', 'y': 'PnL (USDT)'}
      )
      st.plotly_chart(fig_top, use_container_width=True)

  elif analysis_type == "Анализ по времени":
    if not df_closed.empty:
      df_time = df_closed.copy()
      df_time['timestamp'] = pd.to_datetime(df_time['close_timestamp'])
      df_time['hour'] = df_time['timestamp'].dt.hour
      df_time['weekday'] = df_time['timestamp'].dt.day_name()

      # Анализ по часам
      hourly_stats = df_time.groupby('hour')['profit_loss'].agg(['count', 'sum', 'mean'])

      fig_hourly = go.Figure()
      fig_hourly.add_trace(go.Bar(
        x=hourly_stats.index,
        y=hourly_stats['sum'],
        name='Общий PnL',
        yaxis='y'
      ))
      fig_hourly.add_trace(go.Scatter(
        x=hourly_stats.index,
        y=hourly_stats['count'],
        name='Количество сделок',
        yaxis='y2',
        line=dict(color='red')
      ))

      fig_hourly.update_xaxes(title_text="Час дня")
      fig_hourly.update_yaxes(title_text="PnL (USDT)", secondary_y=False)
      fig_hourly.update_yaxes(title_text="Количество сделок", secondary_y=True)
      fig_hourly.update_layout(
        title="Анализ прибыльности по часам",
        yaxis2=dict(overlaying='y', side='right')
      )

      st.plotly_chart(fig_hourly, use_container_width=True)

      # Анализ по дням недели
      weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
      weekday_stats = df_time.groupby('weekday')['profit_loss'].agg(['count', 'sum', 'mean']).reindex(weekday_order)

      fig_weekday = px.bar(
        x=weekday_stats.index,
        y=weekday_stats['sum'],
        title="Прибыльность по дням недели",
        labels={'x': 'День недели', 'y': 'PnL (USDT)'}
      )
      st.plotly_chart(fig_weekday, use_container_width=True)

  elif analysis_type == "Анализ рисков":
    st.subheader("⚠️ Анализ рисков")

    if not df_closed.empty:
      # Максимальная просадка
      df_risk = df_closed.copy()
      df_risk['timestamp'] = pd.to_datetime(df_risk['close_timestamp'])
      df_risk = df_risk.sort_values('timestamp')
      df_risk['cumulative_pnl'] = df_risk['profit_loss'].cumsum()
      df_risk['running_max'] = df_risk['cumulative_pnl'].cummax()
      df_risk['drawdown'] = df_risk['cumulative_pnl'] - df_risk['running_max']

      max_drawdown = df_risk['drawdown'].min()
      max_drawdown_pct = (max_drawdown / df_risk['running_max'].max() * 100) if df_risk['running_max'].max() > 0 else 0

      col1, col2, col3 = st.columns(3)
      col1.metric("Макс. просадка", f"${max_drawdown:.2f}")
      col2.metric("Макс. просадка %", f"{max_drawdown_pct:.2f}%")

      # Коэффициент Шарпа (упрощенный)
      if len(df_risk) > 1:
        daily_returns = df_risk.groupby(df_risk['timestamp'].dt.date)['profit_loss'].sum()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        col3.metric("Коэффициент Шарпа", f"{sharpe_ratio:.2f}")

      # График просадки
      fig_dd = go.Figure()
      fig_dd.add_trace(go.Scatter(
        x=df_risk['timestamp'],
        y=df_risk['cumulative_pnl'],
        name='Накопленный PnL',
        line=dict(color='green')
      ))
      fig_dd.add_trace(go.Scatter(
        x=df_risk['timestamp'],
        y=df_risk['running_max'],
        name='Максимум',
        line=dict(color='blue', dash='dash')
      ))
      fig_dd.add_trace(go.Scatter(
        x=df_risk['timestamp'],
        y=df_risk['drawdown'],
        name='Просадка',
        fill='tozeroy',
        line=dict(color='red')
      ))

      fig_dd.update_layout(
        title="Анализ просадки",
        xaxis_title="Время",
        yaxis_title="USDT",
        height=400
      )

      st.plotly_chart(fig_dd, use_container_width=True)

      # Распределение прибылей/убытков
      st.subheader("Распределение P&L")

      fig_dist = px.histogram(
        df_closed,
        x='profit_loss',
        nbins=50,
        title="Распределение прибылей и убытков",
        labels={'profit_loss': 'P&L (USDT)', 'count': 'Количество'}
      )
      fig_dist.add_vline(x=0, line_dash="dash", line_color="red")
      st.plotly_chart(fig_dist, use_container_width=True)

  elif analysis_type == "ML метрики":
    st.subheader("🤖 Метрики ML моделей")

    model_info = state_manager.get_model_info()

    if model_info:
      col1, col2 = st.columns(2)

      with col1:
        st.metric("Точность модели", f"{model_info.get('accuracy', 0) * 100:.2f}%")
        st.metric("F1 Score", f"{model_info.get('f1_score', 0):.3f}")
        st.metric("Последнее обучение", model_info.get('last_training', 'N/A'))

      with col2:
        st.metric("Количество признаков", model_info.get('features_count', 0))
        st.metric("Размер обучающей выборки", model_info.get('training_samples', 0))
        st.metric("Версия модели", model_info.get('version', 'N/A'))

      # График важности признаков (если доступен)
      feature_importance = model_info.get('feature_importance', {})
      if feature_importance:
        st.subheader("Важность признаков")

        # Берем топ-20 признаков
        top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20])

        fig_features = px.bar(
          x=list(top_features.values()),
          y=list(top_features.keys()),
          orientation='h',
          title="Топ-20 важных признаков",
          labels={'x': 'Важность', 'y': 'Признак'}
        )
        st.plotly_chart(fig_features, use_container_width=True)
    else:
      st.info("Метрики ML моделей пока недоступны")

# Добавить после существующих секций:
with tabs[5]:
  st.header("📊 ROI Калькулятор")

  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Текущие настройки")
    current_roi_sl = strategy_cfg.get('roi_stop_loss_pct', 20.0)
    current_roi_tp = strategy_cfg.get('roi_take_profit_pct', 60.0)
    current_leverage = trade_cfg.get('leverage', 10)

    st.metric("Stop-Loss ROI", f"{current_roi_sl}%", help="Процент от маржи")
    st.metric("Take-Profit ROI", f"{current_roi_tp}%", help="Процент от маржи")
    st.metric("Плечо", f"{current_leverage}x")
    st.metric("Risk/Reward", f"1:{current_roi_tp / current_roi_sl:.1f}")

  with col2:
    st.subheader("Калькулятор влияния на цену")

    test_price = st.number_input(
      "Тестовая цена для расчета",
      value=50000.0,
      step=100.0,
      help="Введите цену для примера расчета SL/TP"
    )

  # Рассчитываем изменения цены
  sl_price_change_pct = (current_roi_sl / 100.0) / current_leverage
  tp_price_change_pct = (current_roi_tp / 100.0) / current_leverage

  sl_price = test_price * (1 - sl_price_change_pct)
  tp_price = test_price * (1 + tp_price_change_pct)

  st.metric("SL Цена", f"{sl_price:.2f}", f"-{sl_price_change_pct * 100:.2f}%")
  st.metric("TP Цена", f"{tp_price:.2f}", f"+{tp_price_change_pct * 100:.2f}%")

  st.info(f"""
    📈 **Расчет для BUY сделки:**
    - Цена входа: {test_price:,.2f}
    - Stop-Loss: {sl_price:,.2f} (потеря {current_roi_sl}% ROI)
    - Take-Profit: {tp_price:,.2f} (прибыль {current_roi_tp}% ROI)
    """)



# --- Вкладка: Настройки ---
with tabs[6]:
  st.header("⚙️ Настройки бота")
  current_config = config_manager.load_config()

  with st.form("settings_form"):
    col1, col2 = st.columns(2)

    with col1:
      st.subheader("📊 Параметры торговли")
      general_cfg = current_config.get('general_settings', {})
      trade_cfg = current_config.get('trade_settings', {})

      leverage = st.slider("Кредитное плечо", 1, 100, value=int(trade_cfg.get('leverage', 10)))

      order_type = st.selectbox(
        "Тип размера ордера",
        options=["percentage", "fixed"],
        index=0 if trade_cfg.get('order_size_type') == 'percentage' else 1
      )

      order_value_label = f"Размер ордера ({'%' if order_type == 'percentage' else 'USDT'})"
      order_value = st.number_input(
        order_value_label,
        min_value=0.1,
        value=float(trade_cfg.get('order_size_value', 1.0))
      )

      min_order_value = st.number_input(
        "Мин. стоимость ордера (USDT)",
        min_value=1.0,
        value=float(trade_cfg.get('min_order_value_usdt', 5.5))
      )

      # --- НАЧАЛО НОВОГО КОДА ---
      grid_allocation = st.number_input(
        "Общая сумма для сетки (USDT)",
        min_value=10.0,
        value=float(trade_cfg.get('grid_total_usdt_allocation', 50.0)),
        step=10.0,
        help="Общая сумма в USDT, которая будет распределена по ордерам в одной сеточной стратегии."
      )
      # --- КОНЕЦ НОВОГО КОДА ---

      min_volume = st.number_input(
        "Мин. суточный объем (USDT)",
        min_value=100000,
        max_value=100000000,
        value=general_cfg.get('min_24h_volume_usdt', 1000000),
        step=100000
      )

      st.divider()

      mode = st.selectbox(
        "Режим выбора символов",
        options=["dynamic", "static"],
        index=0 if general_cfg.get('symbol_selection_mode') == 'dynamic' else 1
      )

      limit = st.number_input(
        "Количество символов (для dynamic)",
        min_value=1,
        max_value=500,
        value=general_cfg.get('dynamic_symbols_count', 20)
      )

      static_list_str = st.text_area(
        "Статический список (через запятую)",
        value=", ".join(general_cfg.get('static_symbol_list', []))
      )

      blacklist_str = st.text_area(
        "Черный список (через запятую)",
        value=", ".join(general_cfg.get('symbol_blacklist', []))
      )

      interval = st.slider(
        "Интервал мониторинга (сек)",
        10,
        300,
        value=general_cfg.get('monitoring_interval_seconds', 30)
      )

    with col2:
      st.subheader("🎯 Настройки стратегии")
      # strategy_cfg = current_config.get('strategy_settings', {})

      confidence = st.slider(
        "Мин. уверенность сигнала",
        0.50,
        1.0,
        value=float(strategy_cfg.get('signal_confidence_threshold', 0.55)),
        step=0.01
      )

      st.divider()

      use_trend = st.checkbox(
        "Фильтр по тренду (EMA)",
        value=strategy_cfg.get('use_trend_filter', True)
      )

      ema_period = st.number_input(
        "Период EMA",
        min_value=10,
        max_value=500,
        value=int(strategy_cfg.get('ema_period', 200))
      )

      st.divider()

      use_adx = st.checkbox(
        "Фильтр силы тренда (ADX)",
        value=strategy_cfg.get('use_adx_filter', True)
      )

      adx_thresh = st.slider(
        "Порог ADX",
        10,
        40,
        value=int(strategy_cfg.get('adx_threshold', 20))
      )

      st.divider()

      use_vol = st.checkbox(
        "Фильтр волатильности (ATR)",
        value=strategy_cfg.get('use_volatility_filter', True)
      )

      atr_pct = st.slider(
        "Макс. волатильность (ATR % от цены)",
        1.0,
        30.0,
        value=float(strategy_cfg.get('max_atr_percentage', 5.0)),
        step=0.1
      )

      st.divider()

      use_aroon = st.checkbox(
        "Фильтр направления тренда (Aroon)",
        value=strategy_cfg.get('use_aroon_filter', True)
      )

      use_psar = st.checkbox(
        "Динамический выход по Parabolic SAR",
        value=strategy_cfg.get('use_psar_exit', True)
      )

      st.divider()

      use_btc_filter = st.checkbox(
        "Фильтр по тренду BTC",
        value=strategy_cfg.get('use_btc_trend_filter', True)
      )

      use_atr_ts = st.checkbox(
        "Трейлинг-стоп по ATR",
        value=strategy_cfg.get('use_atr_trailing_stop', True)
      )

      atr_ts_mult = st.number_input(
        "Множитель ATR для трейлинга",
        min_value=0.5,
        max_value=10.0,
        value=float(strategy_cfg.get('atr_ts_multiplier', 1)),
        step=0.1
      )

      st.divider()

      st.subheader("📈 Risk Management")

      sl_mult = st.number_input(
        "Множитель Stop-Loss (ATR)",
        min_value=0.1,
        max_value=10.0,
        value=float(strategy_cfg.get('sl_multiplier', 0.1)),
        step=0.1
      )

      tp_mult = st.number_input(
        "Множитель Take-Profit (ATR)",
        min_value=0.1,
        max_value=10.0,
        value=float(strategy_cfg.get('tp_multiplier', 2.5)),
        step=0.1
      )

      st.divider()

      roi_sl = st.number_input(
        "Stop-Loss (% от маржи)",
        min_value=1.0,
        max_value=100.0,
        value=float(trade_cfg.get('roi_stop_loss_pct', 5.0)),
        step=1.0
      )

      roi_tp = st.number_input(
        "Take-Profit (% от маржи)",
        min_value=1.0,
        max_value=1000.0,
        value=float(trade_cfg.get('roi_take_profit_pct', 60.0)),
        step=5.0
      )

      ltf_timeframe = st.selectbox(
        "Таймфрейм для входа (LTF)",
        options=["1m", "5m", "15m"],
        index=["1m", "5m", "15m"].index(strategy_cfg.get('ltf_entry_timeframe', '15m'))
      )

    # Кнопка сохранения
    submitted = st.form_submit_button("💾 Сохранить настройки", use_container_width=True)

    if submitted:
      # Обновляем конфигурацию
      new_config = config_manager.load_config()

      try:
        temp_config = {
          'strategy_settings': {
            'roi_stop_loss_pct': roi_sl,
            'roi_take_profit_pct': roi_tp,
          },
          'trade_settings': {
            'leverage': leverage
          }
        }


        # Создаем временный объект для валидации
        class TempRiskManager:
          def __init__(self, config):
            self.config = config

          def validate_roi_parameters(self):
            # Копируем метод валидации
            strategy_settings = self.config.get('strategy_settings', {})
            trade_settings = self.config.get('trade_settings', {})

            roi_sl_pct = trade_settings.get('roi_stop_loss_pct', 5.0)
            roi_tp_pct = trade_settings.get('roi_take_profit_pct', 60.0)
            leverage = trade_settings.get('leverage', 10)

            validation_result = {'is_valid': True, 'warnings': [], 'errors': []}

            if roi_sl_pct < 1.0:
              validation_result['warnings'].append(f"Очень низкий SL ROI: {roi_sl_pct}%")
            elif roi_sl_pct > 50.0:
              validation_result['warnings'].append(f"Очень высокий SL ROI: {roi_sl_pct}%")

            if roi_tp_pct < 5.0:
              validation_result['warnings'].append(f"Очень низкий TP ROI: {roi_tp_pct}%")
            elif roi_tp_pct > 200.0:
              validation_result['warnings'].append(f"Очень высокий TP ROI: {roi_tp_pct}%")

            risk_reward_ratio = roi_tp_pct / roi_sl_pct
            if risk_reward_ratio < 1.5:
              validation_result['warnings'].append(f"Низкое соотношение риск/доходность: 1:{risk_reward_ratio:.1f}")

            if leverage < 1:
              validation_result['errors'].append("Плечо не может быть меньше 1")
              validation_result['is_valid'] = False

            return validation_result


        temp_risk_manager = TempRiskManager(temp_config)
        roi_validation = temp_risk_manager.validate_roi_parameters()

        # Показываем предупреждения пользователю
        if roi_validation['warnings']:
          st.warning("⚠️ Предупреждения по ROI настройкам:")
          for warning in roi_validation['warnings']:
            st.warning(f"• {warning}")

        if roi_validation['errors']:
          st.error("❌ Ошибки в ROI настройках:")
          for error in roi_validation['errors']:
            st.error(f"• {error}")
          st.error("Настройки не сохранены из-за критических ошибок!")
          st.stop()  # Останавливаем выполнение

        # Показываем расчетные изменения цены
        sl_price_change = (roi_sl / 100.0) / leverage * 100
        tp_price_change = (roi_tp / 100.0) / leverage * 100

        st.info(f"📊 Влияние на цену:")
        st.info(f"• SL потребует изменения цены на: {sl_price_change:.2f}%")
        st.info(f"• TP потребует изменения цены на: {tp_price_change:.2f}%")
        st.info(f"• Соотношение риск/доходность: 1:{roi_tp / roi_sl:.1f}")

      except Exception as validation_error:
        st.error(f"Ошибка валидации ROI: {validation_error}")
      # --- КОНЕЦ НОВОГО БЛОКА ---



      new_config['trade_settings'] = {
        "leverage": leverage,
        "order_size_type": order_type,
        "order_size_value": order_value,
        "min_order_value_usdt": min_order_value,
        "grid_total_usdt_allocation": grid_allocation,
        "roi_stop_loss_pct": roi_sl,
        "roi_take_profit_pct": roi_tp
      }

      new_config['strategy_settings'] = {
        "signal_confidence_threshold": confidence,
        "use_trend_filter": use_trend,
        "ema_period": ema_period,
        "use_adx_filter": use_adx,
        "adx_threshold": adx_thresh,
        "use_volatility_filter": use_vol,
        "max_atr_percentage": atr_pct,
        "use_aroon_filter": use_aroon,
        "use_psar_exit": use_psar,
        "use_btc_trend_filter": use_btc_filter,
        "use_atr_trailing_stop": use_atr_ts,
        "atr_ts_multiplier": atr_ts_mult,
        "sl_multiplier": sl_mult,
        "tp_multiplier": tp_mult,
        "ltf_entry_timeframe": ltf_timeframe,
      }

      new_config['general_settings'] = {
        "symbol_selection_mode": mode,
        "dynamic_symbols_count": limit,
        "static_symbol_list": [s.strip().upper() for s in static_list_str.split(',') if s.strip()],
        "symbol_blacklist": [s.strip().upper() for s in blacklist_str.split(',') if s.strip()],
        "min_24h_volume_usdt": min_volume,
        "monitoring_interval_seconds": interval
      }

      # Сохраняем feature_weights если они есть
      if 'feature_weights' not in new_config:
        new_config['feature_weights'] = current_config.get('feature_weights', {})
      if submitted:
        config_manager.save_config(new_config)
        st.toast("✅ Настройки сохранены! Применятся при следующем запуске бота.")
#add_shadow_trading_section()

# --- Футер с информацией ---
st.divider()

add_shadow_trading_section()

col1, col2, col3 = st.columns(3)

with col1:
  model_info = state_manager.get_model_info()
  if model_info:
    st.caption(f"📊 Модель: {model_info.get('version', 'N/A')}")

with col2:
  st.caption(f"🕐 Обновлено: {datetime.now().strftime('%H:%M:%S')}")

with col3:
  # is_bot_running = st.session_state.bot_process and st.session_state.bot_process.poll() is None
  if is_bot_run():
    st.caption("🟢 Бот активен")
  else:
    st.caption("🔴 Бот остановлен")

with st.expander("🎯 Stop-and-Reverse Strategy Settings", expanded=False):
    st.header("🎯 Настройки стратегии Stop-and-Reverse")

    # Загружаем текущую конфигурацию SAR
    try:
      current_config = config_manager.load_config()
      sar_config = current_config.get('stop_and_reverse_strategy', {})

      if not sar_config:
        st.warning("⚠️ Конфигурация SAR стратегии не найдена в config.json")
        st.stop()

      col1, col2 = st.columns(2)

      with col1:
        st.subheader("🚦 Фильтры режимов")

        # Фильтры режимов
        chop_threshold = st.slider(
          "Choppiness Index порог",
          min_value=20,
          max_value=60,
          value=sar_config.get('chop_threshold', 40),
          help="Рынки с CHOP > этого значения будут избегаться"
        )

        adx_threshold = st.slider(
          "ADX минимум для тренда",
          min_value=15,
          max_value=35,
          value=sar_config.get('adx_threshold', 25),
          help="Минимальная сила тренда для торговли"
        )

        atr_multiplier = st.slider(
          "ATR множитель волатильности",
          min_value=1.0,
          max_value=2.0,
          value=sar_config.get('atr_multiplier', 1.25),
          step=0.05,
          help="Текущая волатильность должна быть выше среднего в X раз"
        )

        st.subheader("📊 PSAR настройки")

        psar_start = st.slider(
          "PSAR начальный шаг",
          min_value=0.01,
          max_value=0.05,
          value=sar_config.get('psar_start', 0.02),
          step=0.001,
          format="%.3f"
        )

        psar_step = st.slider(
          "PSAR приращение",
          min_value=0.01,
          max_value=0.05,
          value=sar_config.get('psar_step', 0.02),
          step=0.001,
          format="%.3f"
        )

        psar_max = st.slider(
          "PSAR максимум",
          min_value=0.1,
          max_value=0.3,
          value=sar_config.get('psar_max', 0.2),
          step=0.01
        )

      with col2:
        st.subheader("🎯 Система оценок")

        min_signal_score = st.slider(
          "Минимальный балл сигнала",
          min_value=2,
          max_value=8,
          value=sar_config.get('min_signal_score', 4),
          help="Минимальная сумма баллов для генерации сигнала"
        )

        st.subheader("💰 Управление рисками")

        min_daily_volume = st.number_input(
          "Мин. дневной объем (USD)",
          min_value=100000,
          max_value=10000000,
          value=sar_config.get('min_daily_volume_usd', 1000000),
          step=100000,
          help="Минимальный дневной объем торгов для символа"
        )

        max_monitored_symbols = st.number_input(
          "Макс. отслеживаемых символов",
          min_value=10,
          max_value=100,
          value=sar_config.get('max_monitored_symbols', 50),
          step=5
        )

        st.subheader("🔧 Интеграции")

        use_shadow_system = st.checkbox(
          "Использовать Shadow System",
          value=sar_config.get('shadow_system_integration', {}).get('use_shadow_system', True),
          help="Интеграция с системой теневой торговли"
        )

        use_ml_confirmation = st.checkbox(
          "Использовать ML подтверждение",
          value=sar_config.get('ml_integration', {}).get('use_ml_confirmation', False),
          help="Дополнительное подтверждение от ML моделей"
        )

      # Кнопка сохранения настроек
      if st.button("💾 Сохранить настройки SAR", type="primary"):
        try:
          # Обновляем конфигурацию
          updated_sar_config = sar_config.copy()
          updated_sar_config.update({
            'chop_threshold': chop_threshold,
            'adx_threshold': adx_threshold,
            'atr_multiplier': atr_multiplier,
            'psar_start': psar_start,
            'psar_step': psar_step,
            'psar_max': psar_max,
            'min_signal_score': min_signal_score,
            'min_daily_volume_usd': min_daily_volume,
            'max_monitored_symbols': max_monitored_symbols,
          })

          # Обновляем интеграции
          updated_sar_config['shadow_system_integration']['use_shadow_system'] = use_shadow_system
          updated_sar_config['ml_integration']['use_ml_confirmation'] = use_ml_confirmation

          # Сохраняем в основную конфигурацию
          current_config['stop_and_reverse_strategy'] = updated_sar_config
          config_manager.save_config(current_config)

          # Уведомляем систему об изменениях
          state_manager.set_command('reload_sar_config')

          st.success("✅ Настройки SAR стратегии сохранены!")
          st.info("ℹ️ Изменения вступят в силу при следующем обновлении системы")

        except Exception as e:
          st.error(f"❌ Ошибка сохранения настроек: {e}")

      # Статус стратегии
      st.divider()
      st.subheader("📈 Статус стратегии")

      try:
        # Получаем статус SAR стратегии из состояния
        sar_status = state_manager.get_custom_data('sar_strategy_status')

        if sar_status:
          col1, col2, col3 = st.columns(3)

          with col1:
            st.metric(
              "Отслеживаемые символы",
              sar_status.get('monitored_symbols_count', 0)
            )

          with col2:
            st.metric(
              "Активные позиции",
              sar_status.get('current_positions_count', 0)
            )

          with col3:
            last_update = sar_status.get('last_symbol_update')
            if last_update:
              from datetime import datetime

              last_update_dt = datetime.fromisoformat(last_update)
              time_diff = datetime.now() - last_update_dt
              st.metric(
                "Последнее обновление",
                f"{time_diff.seconds // 60} мин назад"
              )

          # Список отслеживаемых символов
          monitored_symbols = sar_status.get('monitored_symbols', [])
          if monitored_symbols:
            st.subheader("📋 Отслеживаемые символы")

            # Разбиваем символы на колонки для компактного отображения
            cols = st.columns(4)
            for i, symbol in enumerate(monitored_symbols):
              col_idx = i % 4
              with cols[col_idx]:
                st.write(f"• {symbol}")

          # Текущие позиции
          current_positions = sar_status.get('current_positions', [])
          if current_positions:
            st.subheader("💼 Текущие позиции SAR")
            for position in current_positions:
              st.write(f"🔹 {position}")

        else:
          st.info("ℹ️ Статус SAR стратегии пока недоступен")

      except Exception as e:
        st.error(f"❌ Ошибка получения статуса SAR: {e}")

    except Exception as e:
      st.error(f"❌ Ошибка загрузки настроек SAR: {e}")


# --- Автообновление ---
auto_refresh = st.sidebar.checkbox("🔄 Автообновление (30 сек)", value=True)
if auto_refresh:
  # Используем st_autorefresh если доступен
  try:
    st_autorefresh(interval=30000, key="dashboard_refresh")  # 30 секунд
  except:
    # Fallback без автообновления
    st.sidebar.info("Автообновление недоступно. Используйте кнопку 'Обновить данные'")