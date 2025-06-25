# dashboard_extensions.py - РАСШИРЕНИЯ ДЛЯ DASHBOARD

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import json

from shadow_trading.shadow_trading_manager import ShadowTradingManager
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ShadowTradingDashboard:
  """Интеграция Shadow Trading с dashboard - ПОЛНАЯ ВЕРСИЯ"""

  def __init__(self, shadow_manager: ShadowTradingManager):
    self.shadow_manager = shadow_manager

  def display_main_dashboard(self):
    """Главная страница Shadow Trading dashboard"""

    st.header("🌟 Shadow Trading Analytics")

    # Выбор периода анализа
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
      days = st.selectbox(
        "📅 Период анализа",
        options=[1, 3, 7, 14, 30, 90],
        index=2,  # По умолчанию 7 дней
        format_func=lambda x: f"{x} {'день' if x == 1 else 'дней'}"
      )

    with col2:
      analysis_type = st.selectbox(
        "📊 Тип анализа",
        ["Общий обзор", "По источникам", "По символам", "По времени", "Фильтры"]
      )

    with col3:
      if st.button("🔄 Обновить", use_container_width=True):
        st.rerun()

    # Основные метрики
    self._display_main_metrics(days)

    # Детальный анализ в зависимости от выбранного типа
    if analysis_type == "Общий обзор":
      self._display_overall_analysis(days)
    elif analysis_type == "По источникам":
      self._display_source_analysis(days)
    elif analysis_type == "По символам":
      self._display_symbol_analysis(days)
    elif analysis_type == "По времени":
      self._display_time_analysis(days)
    elif analysis_type == "Фильтры":
      self._display_filter_analysis(days)

  def _display_main_metrics(self, days: int):
    """Отображение основных метрик"""
    try:
      # Получаем данные
      overall_perf = self._get_data_sync(
        self.shadow_manager.performance_analyzer.get_overall_performance(days)
      )

      if 'error' in overall_perf:
        st.warning(f"⚠️ {overall_perf['error']}")
        return

      # Основные метрики
      st.subheader("📈 Основные показатели")

      col1, col2, col3, col4, col5 = st.columns(5)

      with col1:
        win_rate = overall_perf.get('win_rate_pct', 0)
        color = "normal" if win_rate >= 60 else "inverse"
        st.metric(
          "🎯 Win Rate",
          f"{win_rate}%",
          delta=f"из {overall_perf.get('completed_signals', 0)} сигналов",
          delta_color=color
        )

      with col2:
        profit_factor = overall_perf.get('profit_factor', 0)
        st.metric(
          "⚖️ Profit Factor",
          f"{profit_factor}",
          delta="Соотношение прибыль/убыток"
        )

      with col3:
        total_pnl = overall_perf.get('total_pnl_pct', 0)
        st.metric(
          "💰 Общий P&L",
          f"{total_pnl:+.2f}%",
          delta=f"За {days} дней"
        )

      with col4:
        total_signals = overall_perf.get('total_signals', 0)
        filtered_signals = overall_perf.get('filtered_signals', 0)
        filter_rate = (filtered_signals / total_signals * 100) if total_signals > 0 else 0
        st.metric(
          "🚫 Фильтрация",
          f"{filter_rate:.1f}%",
          delta=f"{filtered_signals} из {total_signals}"
        )

      with col5:
        completion_rate = overall_perf.get('completion_rate_pct', 0)
        st.metric(
          "✅ Завершенность",
          f"{completion_rate:.1f}%",
          delta="Анализированных сигналов"
        )

    except Exception as e:
      st.error(f"Ошибка отображения метрик: {e}")
      logger.error(f"Dashboard метрики: {e}")

  def _display_overall_analysis(self, days: int):
    """Общий анализ производительности"""
    try:
      st.subheader("📊 Детальный анализ")

      # Получаем расширенные данные
      overall_perf = self._get_data_sync(
        self.shadow_manager.performance_analyzer.get_overall_performance(days)
      )
      confidence_analysis = self._get_data_sync(
        self.shadow_manager.performance_analyzer.get_confidence_analysis(days)
      )

      col1, col2 = st.columns(2)

      with col1:
        # График распределения результатов
        if 'error' not in overall_perf:
          profitable = overall_perf.get('profitable_signals', 0)
          losses = overall_perf.get('loss_signals', 0)
          pending = overall_perf.get('total_signals', 0) - profitable - losses

          fig_pie = go.Figure(data=[go.Pie(
            labels=['Прибыльные', 'Убыточные', 'В ожидании'],
            values=[profitable, losses, pending],
            hole=.3,
            marker_colors=['#2ecc71', '#e74c3c', '#f39c12']
          )])
          fig_pie.update_layout(
            title="Распределение сигналов по результатам",
            height=400
          )
          st.plotly_chart(fig_pie, use_container_width=True)

      with col2:
        # График производительности по уровням уверенности
        if 'error' not in confidence_analysis:
          conf_data = confidence_analysis.get('confidence_breakdown', [])
          if conf_data:
            df_conf = pd.DataFrame(conf_data)

            fig_conf = go.Figure()
            fig_conf.add_trace(go.Bar(
              x=df_conf['confidence_level'],
              y=df_conf['win_rate_pct'],
              name='Win Rate %',
              marker_color='#3498db'
            ))

            fig_conf.update_layout(
              title="Win Rate по уровням уверенности",
              xaxis_title="Уровень уверенности",
              yaxis_title="Win Rate (%)",
              height=400
            )
            st.plotly_chart(fig_conf, use_container_width=True)

      # Дополнительная статистика
      st.subheader("📋 Дополнительная статистика")

      col1, col2, col3 = st.columns(3)

      with col1:
        st.info(f"""
                **Экскурсии цены:**
                - Макс. в пользу: {overall_perf.get('avg_max_favorable_pct', 0):.2f}%
                - Макс. против: {overall_perf.get('avg_max_adverse_pct', 0):.2f}%
                """)

      with col2:
        avg_win = overall_perf.get('avg_win_pct', 0)
        avg_loss = overall_perf.get('avg_loss_pct', 0)
        st.info(f"""
                **Средние результаты:**
                - Прибыль: +{avg_win:.2f}%
                - Убыток: {avg_loss:.2f}%
                - Соотношение: {abs(avg_win / avg_loss):.2f}x
                """)

      with col3:
        st.info(f"""
                **Качество сигналов:**
                - Средняя уверенность: {overall_perf.get('avg_confidence', 0):.3f}
                - Оптимальный порог: {confidence_analysis.get('optimal_threshold', 0.6):.2f}
                """)

    except Exception as e:
      st.error(f"Ошибка общего анализа: {e}")

  def _display_source_analysis(self, days: int):
    """Анализ по источникам сигналов"""
    try:
      st.subheader("🎯 Анализ по источникам сигналов")

      source_perf = self._get_data_sync(
        self.shadow_manager.performance_analyzer.get_performance_by_source(days)
      )

      if not source_perf:
        st.warning("Нет данных по источникам сигналов")
        return

      # Таблица производительности источников
      df_sources = pd.DataFrame(source_perf)

      # Форматируем таблицу
      df_display = df_sources.copy()
      df_display['total_pnl_pct'] = df_display['total_pnl_pct'].apply(lambda x: f"{x:+.2f}%")
      df_display['win_rate_pct'] = df_display['win_rate_pct'].apply(lambda x: f"{x:.1f}%")
      df_display['avg_win_pct'] = df_display['avg_win_pct'].apply(lambda x: f"+{x:.2f}%")
      df_display['avg_loss_pct'] = df_display['avg_loss_pct'].apply(lambda x: f"{x:.2f}%")

      st.dataframe(
        df_display.rename(columns={
          'source': 'Источник',
          'total_signals': 'Всего сигналов',
          'win_rate_pct': 'Win Rate',
          'total_pnl_pct': 'Общий P&L',
          'profit_factor': 'Profit Factor',
          'avg_confidence': 'Ср. уверенность'
        }),
        use_container_width=True
      )

      # Графики сравнения источников
      col1, col2 = st.columns(2)

      with col1:
        # График Win Rate по источникам
        fig_wr = px.bar(
          df_sources,
          x='source',
          y='win_rate_pct',
          title='Win Rate по источникам',
          color='win_rate_pct',
          color_continuous_scale='RdYlGn'
        )
        fig_wr.update_layout(height=400)
        st.plotly_chart(fig_wr, use_container_width=True)

      with col2:
        # График общего P&L
        fig_pnl = px.bar(
          df_sources,
          x='source',
          y='total_pnl_pct',
          title='Общий P&L по источникам',
          color='total_pnl_pct',
          color_continuous_scale='RdYlGn'
        )
        fig_pnl.update_layout(height=400)
        st.plotly_chart(fig_pnl, use_container_width=True)

      # Рекомендации по источникам
      best_source = max(source_perf, key=lambda x: x['win_rate_pct'])
      worst_source = min(source_perf, key=lambda x: x['win_rate_pct'])

      st.info(f"""
            **💡 Рекомендации:**
            - Лучший источник: **{best_source['source']}** (Win Rate: {best_source['win_rate_pct']:.1f}%)
            - Требует внимания: **{worst_source['source']}** (Win Rate: {worst_source['win_rate_pct']:.1f}%)
            """)

    except Exception as e:
      st.error(f"Ошибка анализа источников: {e}")

  def _display_symbol_analysis(self, days: int):
    """Анализ по символам"""
    try:
      st.subheader("💎 Анализ по символам")

      symbol_perf = self._get_data_sync(
        self.shadow_manager.performance_analyzer.get_symbol_performance(days)
      )

      if not symbol_perf:
        st.warning("Недостаточно данных по символам (минимум 3 сигнала на символ)")
        return

      df_symbols = pd.DataFrame(symbol_perf)

      # Топ и худшие символы
      col1, col2 = st.columns(2)

      with col1:
        st.write("🏆 **Топ символы по P&L:**")
        top_symbols = df_symbols.head(10)
        for _, row in top_symbols.iterrows():
          delta_color = "normal" if row['total_pnl_pct'] > 0 else "inverse"
          st.metric(
            row['symbol'],
            f"{row['total_pnl_pct']:+.2f}%",
            delta=f"WR: {row['win_rate_pct']:.1f}% ({row['total_signals']} сигналов)"
          )

      with col2:
        st.write("⚠️ **Символы требующие внимания:**")
        poor_symbols = df_symbols[df_symbols['win_rate_pct'] < 50].head(5)
        if not poor_symbols.empty:
          for _, row in poor_symbols.iterrows():
            st.metric(
              row['symbol'],
              f"{row['win_rate_pct']:.1f}%",
              delta=f"P&L: {row['total_pnl_pct']:+.2f}% ({row['total_signals']} сигналов)",
              delta_color="inverse"
            )
        else:
          st.success("Все символы показывают приемлемую производительность!")

      # Детальная таблица
      st.subheader("📊 Детальная статистика по символам")

      # Сортировка
      sort_by = st.selectbox(
        "Сортировать по:",
        ["total_pnl_pct", "win_rate_pct", "total_signals", "profit_factor"],
        format_func=lambda x: {
          "total_pnl_pct": "Общий P&L",
          "win_rate_pct": "Win Rate",
          "total_signals": "Количество сигналов",
          "profit_factor": "Profit Factor"
        }[x]
      )

      df_sorted = df_symbols.sort_values(sort_by, ascending=False)

      # Форматируем таблицу для отображения
      df_display = df_sorted.copy()
      df_display['total_pnl_pct'] = df_display['total_pnl_pct'].apply(lambda x: f"{x:+.2f}%")
      df_display['win_rate_pct'] = df_display['win_rate_pct'].apply(lambda x: f"{x:.1f}%")

      st.dataframe(
        df_display.rename(columns={
          'symbol': 'Символ',
          'total_signals': 'Сигналов',
          'win_rate_pct': 'Win Rate',
          'total_pnl_pct': 'P&L',
          'profit_factor': 'PF'
        })[['Символ', 'Сигналов', 'Win Rate', 'P&L', 'PF']],
        use_container_width=True
      )

    except Exception as e:
      st.error(f"Ошибка анализа символов: {e}")

  def _display_time_analysis(self, days: int):
    """Анализ по времени"""
    try:
      st.subheader("⏰ Временной анализ")

      hourly_perf = self._get_data_sync(
        self.shadow_manager.performance_analyzer.get_hourly_performance(days)
      )

      if not hourly_perf:
        st.warning("Недостаточно данных для временного анализа")
        return

      # Подготавливаем данные для графиков
      hours = list(range(24))
      win_rates = []
      signal_counts = []
      avg_returns = []

      for hour in hours:
        if hour in hourly_perf:
          win_rates.append(hourly_perf[hour]['win_rate_pct'])
          signal_counts.append(hourly_perf[hour]['total_signals'])
          avg_returns.append(hourly_perf[hour]['avg_return_pct'])
        else:
          win_rates.append(0)
          signal_counts.append(0)
          avg_returns.append(0)

      # График Win Rate по часам
      fig_time = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Win Rate по часам', 'Количество сигналов по часам'),
        vertical_spacing=0.1
      )

      # Win Rate
      fig_time.add_trace(
        go.Scatter(
          x=hours,
          y=win_rates,
          mode='lines+markers',
          name='Win Rate %',
          line=dict(color='#2ecc71', width=3),
          marker=dict(size=8)
        ),
        row=1, col=1
      )

      # Количество сигналов
      fig_time.add_trace(
        go.Bar(
          x=hours,
          y=signal_counts,
          name='Количество сигналов',
          marker_color='#3498db',
          opacity=0.7
        ),
        row=2, col=1
      )

      fig_time.update_layout(
        height=600,
        title_text="Производительность по часам суток (UTC)",
        showlegend=False
      )

      fig_time.update_xaxes(title_text="Час (UTC)", row=2, col=1)
      fig_time.update_yaxes(title_text="Win Rate (%)", row=1, col=1)
      fig_time.update_yaxes(title_text="Сигналов", row=2, col=1)

      st.plotly_chart(fig_time, use_container_width=True)

      # Статистика по времени
      col1, col2, col3 = st.columns(3)

      # Лучший час
      best_hour_data = max(
        [(h, hourly_perf[h]) for h in hourly_perf.keys()],
        key=lambda x: x[1]['win_rate_pct']
      )
      best_hour, best_data = best_hour_data

      with col1:
        st.metric(
          "🕐 Лучший час",
          f"{best_hour:02d}:00 UTC",
          delta=f"WR: {best_data['win_rate_pct']:.1f}%"
        )

      # Самый активный час
      most_active_hour_data = max(
        [(h, hourly_perf[h]) for h in hourly_perf.keys()],
        key=lambda x: x[1]['total_signals']
      )
      active_hour, active_data = most_active_hour_data

      with col2:
        st.metric(
          "📈 Самый активный час",
          f"{active_hour:02d}:00 UTC",
          delta=f"{active_data['total_signals']} сигналов"
        )

      # Средний возврат
      avg_hourly_return = np.mean([data['avg_return_pct'] for data in hourly_perf.values()])

      with col3:
        st.metric(
          "📊 Средний возврат",
          f"{avg_hourly_return:+.2f}%",
          delta="За час"
        )

    except Exception as e:
      st.error(f"Ошибка временного анализа: {e}")

  def _display_filter_analysis(self, days: int):
    """Анализ эффективности фильтров"""
    try:
      st.subheader("🚫 Анализ фильтров")

      filter_analysis = self._get_data_sync(
        self.shadow_manager.performance_analyzer.get_filter_analysis(days)
      )

      if 'error' in filter_analysis:
        st.warning(f"Ошибка анализа фильтров: {filter_analysis['error']}")
        return

      # Основная статистика фильтрации
      col1, col2, col3 = st.columns(3)

      with col1:
        st.metric(
          "🚫 Всего отфильтровано",
          filter_analysis.get('total_filtered', 0),
          delta="сигналов"
        )

      with col2:
        st.metric(
          "💸 Упущенных возможностей",
          filter_analysis.get('missed_opportunities', 0),
          delta=f"Средняя прибыль: {filter_analysis.get('avg_missed_profit_pct', 0):+.2f}%"
        )

      with col3:
        st.metric(
          "📉 Общая упущенная прибыль",
          f"{filter_analysis.get('total_missed_profit_pct', 0):+.2f}%",
          delta="Потенциальная"
        )

      # График распределения причин фильтрации
      filter_breakdown = filter_analysis.get('filter_breakdown', [])
      if filter_breakdown:
        df_filters = pd.DataFrame(filter_breakdown)

        fig_filters = px.pie(
          df_filters,
          values='count',
          names='reason',
          title='Распределение причин фильтрации'
        )
        fig_filters.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_filters, use_container_width=True)

        # Детальная таблица фильтров
        st.subheader("📋 Детализация по фильтрам")

        df_display = df_filters.copy()
        total_filtered = df_display['count'].sum()
        df_display['percentage'] = (df_display['count'] / total_filtered * 100).round(1)

        st.dataframe(
          df_display.rename(columns={
            'reason': 'Причина фильтрации',
            'count': 'Количество',
            'percentage': 'Процент'
          }),
          use_container_width=True
        )

      # Рекомендации по фильтрам
      recommendations = self._get_data_sync(
        self.shadow_manager.performance_analyzer.generate_optimization_recommendations(days)
      )

      if 'error' not in recommendations:
        st.subheader("💡 Рекомендации по оптимизации")

        high_priority = [r for r in recommendations.get('recommendations', []) if r['priority'] == 'high']
        medium_priority = [r for r in recommendations.get('recommendations', []) if r['priority'] == 'medium']

        if high_priority:
          st.error("🔴 **Высокий приоритет:**")
          for rec in high_priority:
            st.write(f"- {rec['message']}")
            st.write(f"  💡 *{rec['suggested_action']}*")

        if medium_priority:
          st.warning("🟡 **Средний приоритет:**")
          for rec in medium_priority:
            st.write(f"- {rec['message']}")
            st.write(f"  💡 *{rec['suggested_action']}*")

        if not high_priority and not medium_priority:
          st.success("✅ Система фильтрации работает оптимально!")

    except Exception as e:
      st.error(f"Ошибка анализа фильтров: {e}")

  def _get_data_sync(self, coroutine):
    """Синхронное получение данных из асинхронного кода"""
    try:
      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      result = loop.run_until_complete(coroutine)
      loop.close()
      return result
    except Exception as e:
      logger.error(f"Ошибка получения данных: {e}")
      return {'error': str(e)}

  def create_performance_summary(self, days: int = 7) -> str:
    """Создать краткую сводку производительности для main dashboard"""
    try:
      overall_perf = self._get_data_sync(
        self.shadow_manager.performance_analyzer.get_overall_performance(days)
      )

      source_perf = self._get_data_sync(
        self.shadow_manager.performance_analyzer.get_performance_by_source(days)
      )

      if 'error' in overall_perf:
        return f"❌ Нет данных Shadow Trading за {days} дней"

      # Создаем краткую сводку
      summary_lines = [
        f"📊 **Shadow Trading за {days} дней:**",
        f"🎯 Сигналов: {overall_perf['total_signals']} (завершено: {overall_perf['completed_signals']})",
        f"✅ Win Rate: {overall_perf['win_rate_pct']}%",
        f"💰 Общий P&L: {overall_perf['total_pnl_pct']:+.2f}%",
        f"📈 Средняя прибыль: +{overall_perf['avg_win_pct']}%",
        f"📉 Средний убыток: {overall_perf['avg_loss_pct']}%",
        f"⚖️ Profit Factor: {overall_perf['profit_factor']}",
        ""
      ]

      # Топ источников
      if source_perf:
        summary_lines.append("🏆 **Лучшие источники:**")
        for source in source_perf[:3]:
          summary_lines.append(
            f"  • {source['source']}: {source['win_rate_pct']}% "
            f"({source['total_signals']} сигналов, P&L: {source['total_pnl_pct']:+.1f}%)"
          )

      return "\n".join(summary_lines)

    except Exception as e:
      logger.error(f"Ошибка создания сводки для dashboard: {e}")
      return f"❌ Ошибка получения данных Shadow Trading: {e}"


# ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ИНТЕГРАЦИИ В ОСНОВНОЙ DASHBOARD

def add_shadow_trading_section():
  """Добавить секцию Shadow Trading в основной dashboard"""

  st.markdown("---")
  st.header("🌟 Shadow Trading Analytics")

  with st.expander("📊 Краткая сводка Shadow Trading", expanded=True):
    # Здесь нужно получить доступ к shadow_trading_manager
    # Это можно сделать через session_state или глобальную переменную

    if 'shadow_manager' in st.session_state:
      dashboard = ShadowTradingDashboard(st.session_state.shadow_manager)
      summary = dashboard.create_performance_summary(7)
      st.markdown(summary)
    else:
      st.info("🔄 Shadow Trading система не инициализирована")

  # Кнопка для перехода к полному dashboard
  if st.button("🔍 Открыть полный анализ Shadow Trading", use_container_width=True):
    st.session_state.page = "shadow_trading"
    st.rerun()


def display_full_shadow_dashboard():
  """Отображение полного Shadow Trading dashboard"""

  if 'shadow_manager' not in st.session_state:
    st.error("Shadow Trading система не инициализирована")
    return

  dashboard = ShadowTradingDashboard(st.session_state.shadow_manager)
  dashboard.display_main_dashboard()


# ФУНКЦИИ ДЛЯ ИНТЕГРАЦИИ В MAIN.PY

def setup_shadow_dashboard_integration(shadow_manager: ShadowTradingManager):
  """Настройка интеграции dashboard с Shadow Trading"""

  # Сохраняем ссылку на shadow_manager в session_state для dashboard
  if 'shadow_manager' not in st.session_state:
    st.session_state.shadow_manager = shadow_manager
    logger.info("✅ Shadow Trading dashboard интеграция настроена")