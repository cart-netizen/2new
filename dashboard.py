# dashboard.py

import streamlit as st
import pandas as pd
import subprocess
import os
import signal
import time
import asyncio
from contextlib import suppress

from data.database_manager import AdvancedDatabaseManager
from data.state_manager import StateManager
from config import settings
from config.config_manager import ConfigManager

# --- Настройка страницы ---
st.set_page_config(
  page_title="Панель управления торговым ботом",
  page_icon="🤖",
  layout="wide"
)

# --- Инициализация менеджеров ---
# Используем один и тот же путь к конфигу для всех
CONFIG_FILE_PATH = "config.json"
state_manager = StateManager()
db_manager = AdvancedDatabaseManager() # Убедитесь, что db_path задан по умолчанию или передайте его
config_manager = ConfigManager(config_path=CONFIG_FILE_PATH)

# # Инициализируем таблицу состояний
# asyncio.run(state_manager.initialize_state())
# ДОБАВЛЯЕМ ПРОВЕРКУ И СОЗДАНИЕ ОСНОВНЫХ ТАБЛИЦ (`trades` и др.)
asyncio.run(db_manager._create_tables_if_not_exist())

# --- Боковая панель с управлением ---
with st.sidebar:
  st.title("🕹️ Управление ботом")

  if 'bot_process' not in st.session_state:
    st.session_state.bot_process = None

  if st.button("🚀 Запустить Бота", type="primary"):
    if st.session_state.bot_process and st.session_state.bot_process.poll() is None:
      st.warning("Бот уже запущен.")
    else:
      try:
        st.session_state.bot_process = subprocess.Popen(['python', 'main.py'])
        st.success("Команда на запуск отправлена!")
        time.sleep(2)
        st.rerun()
      except Exception as e:
        st.error(f"Ошибка при запуске бота: {e}")

  if st.button("🛑 Остановить Бота"):
    if st.session_state.bot_process and st.session_state.bot_process.poll() is None:
      try:
        st.session_state.bot_process.terminate()
        st.session_state.bot_process.wait(timeout=5)
        st.warning("Команда на остановку отправлена!")
        st.session_state.bot_process = None
        time.sleep(1)
        st.rerun()
      except Exception as e:
        st.error(f"Не удалось остановить процесс: {e}")
    else:
      st.error("Процесс бота не запущен через дашборд.")
      state_manager.set_status('stopped')

  st.divider()
  # Добавляем кнопку ручного обновления
  if st.button("🔄 Обновить данные"):
    st.rerun()

  st.divider()
  st.subheader("Действия")
  if st.button("📊 Сформировать отчет о модели"):
    # Создаем пустой файл-сигнал
    state_manager.set_command("generate_report")
    st.success(
      "Команда на создание отчета отправлена! Отчет появится в папке 'ml_models/performance_logs' в течение минуты.")


# --- Основная часть дашборда ---
st.title("🤖 Панель управления торговым ботом")

# --- Загрузка данных ---
# Загружаем все данные один раз при каждой отрисовке страницы
status = state_manager.get_status()
metrics = state_manager.get_metrics()
model_info = state_manager.get_model_info()
# Для асинхронных функций БД используем asyncio.run
# all_trades = asyncio.run(db_manager.get_all_trades(limit=1000))
# df_trades = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

# ++ НОВЫЙ ИСТОЧНИК ДАННЫХ ДЛЯ ТАБЛИЦЫ ++
# Получаем ОТКРЫТЫЕ позиции из файла состояния
open_positions_list = state_manager.get_open_positions()
# Получаем ИСТОРИЮ закрытых сделок из БД
closed_trades_list = asyncio.run(db_manager.get_all_trades(limit=1000))

# Объединяем их для отображения
all_trades_to_display = open_positions_list + closed_trades_list
df_trades = pd.DataFrame(all_trades_to_display) if all_trades_to_display else pd.DataFrame()
open_positions_from_state = state_manager.get_open_positions() # <-- Активные сделки
closed_trades_from_db = asyncio.run(db_manager.get_all_closed_trades(limit=1000)) # <-- Только закрытые

df_open = pd.DataFrame(open_positions_from_state) if open_positions_from_state else pd.DataFrame()
df_closed = pd.DataFrame(closed_trades_from_db) if closed_trades_from_db else pd.DataFrame()


# --- Отображение статуса и метрик ---
is_bot_running = st.session_state.bot_process and st.session_state.bot_process.poll() is None
if is_bot_running:
  st.success(f"🟢 **Статус: Бот работает** (PID: {st.session_state.bot_process.pid})")
else:
  st.error("🔴 **Статус: Бот остановлен**")

st.subheader("Финансовые показатели")
cols = st.columns(4)
if metrics:
  cols[0].metric("Общий баланс USDT", f"{metrics.total_balance_usdt:.2f}")
  cols[1].metric("Доступный баланс USDT", f"{metrics.available_balance_usdt:.2f}")
  cols[2].metric("Нереализованный PnL", f"{metrics.unrealized_pnl_total:.2f}")
  cols[3].metric("Реализованный PnL", f"{metrics.realized_pnl_total:.2f}")

st.divider()

# --- Вкладки с настройками и отображением данных ---
view_tab, analysis_tab, statistic_tab, settings_tab = st.tabs(["📊 Мониторинг", "📈 Анализ","📊 Статистика производительности", "⚙️ Конфигурация"])

with view_tab:
  st.subheader("Активные позиции")
  if not df_open.empty:
    display_cols_open = ['open_timestamp', 'symbol', 'side', 'quantity', 'open_price']
    df_open_display = df_open[[col for col in display_cols_open if col in df_open.columns]].copy()
    if 'open_timestamp' in df_open_display.columns:
      df_open_display['open_timestamp'] = pd.to_datetime(df_open_display['open_timestamp']).dt.strftime(
        '%Y-%m-%d %H:%M:%S')
    st.dataframe(df_open_display, use_container_width=True)
  else:
    st.info("Нет активных позиций.")

  st.subheader("История закрытых сделок")
  if not df_closed.empty:
    display_cols_closed = ['close_timestamp', 'symbol', 'side', 'quantity', 'open_price', 'close_price', 'profit_loss']
    df_closed_display = df_closed[[col for col in display_cols_closed if col in df_closed.columns]].copy()
    if 'close_timestamp' in df_closed_display.columns:
      df_closed_display['close_timestamp'] = pd.to_datetime(df_closed_display['close_timestamp']).dt.strftime(
        '%Y-%m-%d %H:%M:%S')
    st.dataframe(df_closed_display, use_container_width=True, height=350)
  else:
    st.info("Нет закрытых сделок.")

  st.subheader("Кривая доходности")
  if not df_trades.empty and 'profit_loss' in df_trades.columns and 'status' in df_trades.columns:
    closed_trades = df_trades[df_trades['status'] == 'CLOSED'].copy()
    if not closed_trades.empty and 'close_timestamp' in closed_trades.columns:
      closed_trades['timestamp'] = pd.to_datetime(closed_trades['close_timestamp'])
      closed_trades.sort_values('timestamp', inplace=True)
      closed_trades['cumulative_pnl'] = closed_trades['profit_loss'].cumsum()
      st.line_chart(closed_trades.set_index('timestamp')['cumulative_pnl'])
    else:
      st.info("Пока нет закрытых сделок для построения графика.")

with analysis_tab:
  st.header("Анализ производительности")
  st.info(
    "Эта вкладка зарезервирована для будущих аналитических инструментов, таких как отчеты о важности признаков, результаты бэктестов и т.д.")
  # Сюда можно будет добавить, например, кнопку для запуска offline_analysis.py
  # и отображения его результатов.
  if st.button("Запустить анализ важности признаков (оффлайн)"):
    st.warning("Эта функция пока в разработке.")
    st.divider()

with statistic_tab:
  with st.expander("📊 Статистика производительности"):
    col1, col2, col3 = st.columns(3)

    # Статистика кэша
    if hasattr(st.session_state, 'data_fetcher'):
      cache_stats = st.session_state.data_fetcher.get_cache_stats()
      with col1:
        st.metric("Cache Hit Rate", f"{cache_stats['hit_rate']:.1%}")
        st.metric("Total Requests", cache_stats['total_requests'])

    # Статистика БД
    with col2:
      st.metric("DB Pool Size", "10 connections")
      st.metric("Active Queries", "N/A")  # Можно добавить счетчик

    # Статистика API
    with col3:
      st.metric("API Requests/min", "50")
      st.metric("Avg Response Time", "N/A")


with settings_tab:
  st.header("Настройки Бота")
  current_config = config_manager.load_config()

  with st.form("settings_form"):
    col1, col2 = st.columns(2)

    with col1:
      # ... (настройки general_settings и trade_settings)
      st.subheader("Параметры торговли")
      general_cfg = current_config.get('general_settings', {})
      trade_cfg = current_config.get('trade_settings', {})
      leverage = st.slider("Кредитное плечо", 1, 100, value=int(trade_cfg.get('leverage', 10)), key='leverage')
      order_type = st.selectbox("Тип размера ордера", options=["percentage", "fixed"],
                                index=0 if trade_cfg.get('order_size_type') == 'percentage' else 1, key='order_type')
      order_value_label = f"Размер ордера ({'%' if order_type == 'percentage' else 'USDT'})"
      order_value = st.number_input(order_value_label, min_value=0.1,
                                    value=float(trade_cfg.get('order_size_value', 1.0)), key='order_value')
      min_order_value = st.number_input("Мин. стоимость ордера (USDT)", min_value=1.0,
                                        value=float(trade_cfg.get('min_order_value_usdt', 5.5)), key='min_order_value')
      # --- ДОБАВЛЯЕМ НОВЫЙ ВИДЖЕТ ---
      min_volume = st.number_input(
        "Мин. суточный объем (USDT)",
        min_value=100000,
        max_value=100000000,
        value=general_cfg.get('min_24h_volume_usdt', 1000000),
        step=100000,
        key='min_volume'
      )
      # --- КОНЕЦ НОВОГО БЛОКА ---
      st.write("---")
      mode = st.selectbox("Тип размера ордера", options=["dynamic", "static"],
                                index=0 if trade_cfg.get('symbol_selection_mode') == 'dynamic' else 1, key='mode')
      limit = st.number_input("Количество символов (для dynamic)", min_value=1, max_value=500, value=general_cfg.get('dynamic_symbols_count', 20))
      static_list_str = st.text_area("Статический список (через запятую)",
                                     value=", ".join(general_cfg.get('static_symbol_list', [])))
      blacklist_str = st.text_area("Черный список (через запятую)",
                                   value=", ".join(general_cfg.get('symbol_blacklist', [])))
      interval = st.slider("Интервал мониторинга (сек)", 10, 300, value=general_cfg.get('monitoring_interval_seconds', 30))

    with col2:
      st.subheader("Настройки стратегии")
      strategy_cfg = current_config.get('strategy_settings', {})
      confidence = st.slider("Мин. уверенность сигнала", 0.50, 1.0,
                             value=float(strategy_cfg.get('signal_confidence_threshold', 0.55)), step=0.01,
                             key='confidence')

      st.write("---")
      use_trend = st.checkbox("Фильтр по тренду (EMA)", value=strategy_cfg.get('use_trend_filter', True),
                              key='use_trend')
      ema_period = st.number_input("Период EMA", min_value=10, max_value=500,
                                   value=int(strategy_cfg.get('ema_period', 200)), key='ema_period')

      st.write("---")
      use_adx = st.checkbox("Фильтр силы тренда (ADX)", value=strategy_cfg.get('use_adx_filter', True), key='use_adx')
      adx_thresh = st.slider("Порог ADX", 10, 40, value=int(strategy_cfg.get('adx_threshold', 20)), key='adx_thresh')

      st.write("---")
      use_vol = st.checkbox("Фильтр волатильности (ATR)", value=strategy_cfg.get('use_volatility_filter', True),
                            key='use_vol')
      atr_pct = st.slider("Макс. волатильность (ATR % от цены)", 1.0, 30.0,
                          value=float(strategy_cfg.get('max_atr_percentage', 5.0)), step=0.1, key='atr_pct')

      # --- ДОБАВЛЯЕМ НОВЫЙ ВИДЖЕТ ЗДЕСЬ ---
      st.write("---")
      use_aroon = st.checkbox("Фильтр направления тренда (Aroon)", value=strategy_cfg.get('use_aroon_filter', True),
                              key='use_aroon')
      # --- КОНЕЦ НОВОГО БЛОКА ---

      use_psar = st.checkbox("Динамический выход по Parabolic SAR", value=strategy_cfg.get('use_psar_exit', True),
                             key='use_psar')

      st.write("---")

      use_btc_filter = st.checkbox("Фильтр по тренду BTC", value=strategy_cfg.get('use_btc_trend_filter', True),
                                   key='use_btc_filter')

      # --- ДОБАВЛЯЕМ НОВЫЙ ВИДЖЕТ ЗДЕСЬ ---
      use_atr_ts = st.checkbox("Трейлинг-стоп по ATR", value=strategy_cfg.get('use_atr_trailing_stop', True),
                               key='use_atr_ts')
      atr_ts_mult = st.number_input("Множитель ATR для трейлинга", min_value=0.5, max_value=10.0,
                                    value=float(strategy_cfg.get('atr_ts_multiplier', 1)), step=0.1,
                                    key='atr_ts_mult')
      # --- КОНЕЦ НОВОГО БЛОКА ---


      st.write("---")

      st.subheader("Множители SL/TP (на основе ATR)")
      # ЯВНО ПРИВОДИМ VALUE К ТИПУ FLOAT
      sl_mult = st.number_input("Множитель Stop-Loss", min_value=0.1, max_value=10.0,
                                value=float(strategy_cfg.get('sl_multiplier', 0.1)), step=0.1, key='sl_mult')
      tp_mult = st.number_input("Множитель Take-Profit", min_value=0.1, max_value=10.0,
                                value=float(strategy_cfg.get('tp_multiplier', 2.5)), step=0.1, key='tp_mult')

      # --- ДОБАВЛЯЕМ НОВЫЕ ВИДЖЕТЫ ЗДЕСЬ ---
      st.write("---")
      st.subheader("SL/TP на основе ROI (%)")
      roi_sl = st.number_input("Stop-Loss (% от маржи)", min_value=1.0, max_value=100.0,
                               value=float(trade_cfg.get('roi_stop_loss_pct', 5.0)), step=1.0, key='roi_sl')
      roi_tp = st.number_input("Take-Profit (% от маржи)", min_value=1.0, max_value=1000.0,
                               value=float(trade_cfg.get('roi_take_profit_pct', 60.0)), step=5.0, key='roi_tp')
      # --- КОНЕЦ НОВОГО БЛОКА ---

      ltf_timeframe = st.selectbox(
        "Таймфрейм для входа (LTF)",
        options=["1m", "5m", "15m"],
        # Устанавливаем индекс по умолчанию на основе значения из конфига
        index=["1m", "5m", "15m"].index(strategy_cfg.get('ltf_entry_timeframe', '15m')),
        key='ltf_timeframe'
      )
      st.divider()



      # Кнопка сохранения находится внутри формы
      submitted = st.form_submit_button("💾 Сохранить все настройки")
      if submitted:
        # 1. Загружаем текущую конфигурацию, чтобы не потерять данные,
        # которых нет в форме (например, feature_weights)
        new_config = config_manager.load_config()
        new_config['trade_settings']= {

            "leverage": leverage,
            "order_size_type": order_type,
            "order_size_value": order_value,
            "min_order_value_usdt": min_order_value


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
            "roi_stop_loss_pct": roi_sl,
            "roi_take_profit_pct": roi_tp,
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
        new_config["feature_weights"]= {
          "price_to_ema_200_4h": 1.5,
          "volume_spike_ratio_4h": 1.5,
          "rsi_14_4h": 1.4,
          "vpt": 1.3,
          "price_to_ema_200_1d": 1.2,
          "rsi_14_1d": 1.1,
          "volume_spike_ratio_1d": 1.1,
          "ema_50": 1.1,
          "volatility_20": 1.1,
          "AROONOSC_14_1H": 1.2,
          "rsi_14_15m": 0.5,
          "price_to_ema_200_30m": 0.6
        }
        config_manager.save_config(new_config)
        st.success("Настройки успешно сохранены! Они будут применены при следующем запуске бота.")
