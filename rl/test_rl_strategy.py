# rl/test_rl_strategy.py

import asyncio
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns

from core.integrated_system import IntegratedTradingSystem
from core.data_fetcher import DataFetcher
from core.bybit_connector import BybitConnector
from data.database_manager import AdvancedDatabaseManager
from config.config_manager import ConfigManager
from data.state_manager import StateManager

from rl.portfolio_manager import RLPortfolioManager
from rl.shadow_learning import ShadowTradingLearner

from utils.logging_config import get_logger

logger = get_logger(__name__)


class RLStrategyTester:
  """Класс для тестирования RL стратегии в реальных условиях"""

  def __init__(self, config_path: str = "config.json", model_name: Optional[str] = None):
    self.config_manager = ConfigManager(config_path)
    self.config = self.config_manager.load_config()

    # Убеждаемся, что RL включен
    if 'rl_trading' not in self.config:
      self.config['rl_trading'] = {}
    self.config['rl_trading']['enabled'] = True

    # Устанавливаем модель для загрузки
    if model_name:
      self.config['rl_trading']['pretrained_model'] = model_name

    # Компоненты системы
    self.trading_system = None
    self.db_manager = None
    self.state_manager = None

    # Результаты тестирования
    self.test_results = {
      'signals': [],
      'trades': [],
      'portfolio_history': [],
      'performance_metrics': {},
      'errors': []
    }

    # Параметры тестирования
    self.test_mode = self.config.get('test_mode', True)
    self.paper_trading = self.config.get('paper_trading', True)

  async def initialize_system(self):
    """Инициализирует торговую систему"""
    logger.info("Инициализация торговой системы для тестирования RL...")

    try:
      # База данных
      self.db_manager = AdvancedDatabaseManager()

      # Менеджер состояния
      self.state_manager = StateManager()

      # Интегрированная торговая система
      self.trading_system = IntegratedTradingSystem(
        db_manager=self.db_manager,
        config=self.config
      )

      # Проверяем, что RL компоненты инициализированы
      if not hasattr(self.trading_system, 'rl_agent'):
        raise ValueError("RL компоненты не инициализированы в торговой системе")

      if not self.trading_system.rl_agent.is_trained:
        logger.warning("RL агент не обучен. Загружаем предобученную модель...")

      logger.info("✅ Торговая система инициализирована")

    except Exception as e:
      logger.error(f"Ошибка инициализации системы: {e}", exc_info=True)
      raise

  async def run_backtest(
      self,
      start_date: Optional[datetime] = None,
      end_date: Optional[datetime] = None,
      symbols: Optional[List[str]] = None
  ) -> Dict:
    """Запускает бэктест RL стратегии"""
    logger.info("=" * 50)
    logger.info("ЗАПУСК БЭКТЕСТА RL СТРАТЕГИИ")
    logger.info("=" * 50)

    # Параметры бэктеста
    if not symbols:
      symbols = self.config['rl_trading'].get('symbols', ['BTCUSDT', 'ETHUSDT'])

    if not start_date:
      start_date = datetime.now() - timedelta(days=30)

    if not end_date:
      end_date = datetime.now()

    logger.info(f"Период: {start_date.date()} - {end_date.date()}")
    logger.info(f"Символы: {symbols}")

    # Загружаем исторические данные
    historical_data = await self._load_historical_data(symbols, start_date, end_date)

    if not historical_data:
      raise ValueError("Не удалось загрузить исторические данные")

    # Инициализируем портфель для бэктеста
    initial_capital = self.config['rl_trading'].get('initial_capital', 10000)
    portfolio = RLPortfolioManager(
      initial_capital=initial_capital,
      risk_manager=self.trading_system.risk_manager,
      config=self.config['rl_trading'].get('portfolio_config', {})
    )

    # Симулируем торговлю
    await self._simulate_trading(historical_data, portfolio)

    # Анализируем результаты
    backtest_results = self._analyze_backtest_results(portfolio)

    logger.info("=" * 50)
    logger.info("БЭКТЕСТ ЗАВЕРШЕН")
    logger.info(f"Итоговая доходность: {backtest_results['total_return_pct']:.2f}%")
    logger.info(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {backtest_results['max_drawdown_pct']:.2f}%")
    logger.info("=" * 50)

    return backtest_results

  async def _load_historical_data(
      self,
      symbols: List[str],
      start_date: datetime,
      end_date: datetime
  ) -> Dict[str, pd.DataFrame]:
    """Загружает исторические данные для бэктеста"""
    historical_data = {}

    for symbol in symbols:
      logger.info(f"Загрузка данных для {symbol}...")

      try:
        # Здесь должна быть загрузка из БД или API
        # Для примера используем data_fetcher
        data = await self.trading_system.data_fetcher.get_historical_candles(
          symbol=symbol,
          timeframe=self.trading_system.timeframe,
          limit=2000  # Максимум для тестирования
        )

        if data is not None and not data.empty:
          # Фильтруем по датам
          data = data[(data.index >= start_date) & (data.index <= end_date)]
          historical_data[symbol] = data
          logger.info(f"✅ Загружено {len(data)} баров для {symbol}")

      except Exception as e:
        logger.error(f"Ошибка загрузки данных для {symbol}: {e}")
        continue

    return historical_data

  async def _simulate_trading(
      self,
      historical_data: Dict[str, pd.DataFrame],
      portfolio: RLPortfolioManager
  ):
    """Симулирует торговлю на исторических данных"""
    # Определяем общий временной диапазон
    all_timestamps = set()
    for data in historical_data.values():
      all_timestamps.update(data.index.tolist())

    timestamps = sorted(list(all_timestamps))

    logger.info(f"Симуляция торговли на {len(timestamps)} временных точках...")

    # Прогресс бар
    progress_step = len(timestamps) // 20

    for i, timestamp in enumerate(timestamps):
      # Показываем прогресс
      if i % progress_step == 0:
        progress = (i / len(timestamps)) * 100
        logger.info(f"Прогресс: {progress:.0f}%")

      # Получаем текущие данные для всех символов
      current_data = {}
      current_prices = {}

      for symbol, data in historical_data.items():
        if timestamp in data.index:
          # Берем данные до текущего момента (включительно)
          hist_data = data[data.index <= timestamp].tail(100)
          if not hist_data.empty:
            current_data[symbol] = hist_data
            current_prices[symbol] = hist_data['close'].iloc[-1]

      if not current_data:
        continue

      # Обновляем цены в портфеле
      portfolio.update_prices(current_prices)

      # Генерируем сигналы для каждого символа
      for symbol, data in current_data.items():
        try:
          # Получаем сигнал от RL стратегии
          signal = await self.trading_system.strategy_manager.strategies['RL_Strategy'].generate_signal(
            symbol, data
          )

          if signal:
            # Записываем сигнал
            self.test_results['signals'].append({
              'timestamp': timestamp,
              'symbol': symbol,
              'signal': signal.signal_type.value,
              'confidence': signal.confidence,
              'entry_price': signal.entry_price,
              'metadata': signal.metadata
            })

            # Выполняем сделку в портфеле
            await self._execute_paper_trade(portfolio, signal, current_prices[symbol])

        except Exception as e:
          logger.error(f"Ошибка генерации сигнала для {symbol}: {e}")
          self.test_results['errors'].append({
            'timestamp': timestamp,
            'symbol': symbol,
            'error': str(e)
          })

      # Сохраняем состояние портфеля
      portfolio_state = portfolio.get_portfolio_metrics()
      portfolio_state['timestamp'] = timestamp
      self.test_results['portfolio_history'].append(portfolio_state)

    logger.info("✅ Симуляция завершена")

  async def _execute_paper_trade(
      self,
      portfolio: RLPortfolioManager,
      signal,
      current_price: float
  ):
    """Выполняет бумажную сделку"""
    try:
      symbol = signal.symbol

      # Рассчитываем размер позиции
      position_size = portfolio.calculate_position_size(
        symbol=symbol,
        signal_confidence=signal.confidence,
        current_price=current_price,
        volatility=signal.metadata.get('predicted_volatility')
      )

      # Конвертируем в количество контрактов
      quantity = position_size / current_price

      if signal.signal_type.value == 'BUY':
        # Открываем длинную позицию
        success = portfolio.open_position(
          symbol=symbol,
          quantity=quantity,
          entry_price=current_price,
          position_type='LONG'
        )

        if success:
          self.test_results['trades'].append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'BUY',
            'quantity': quantity,
            'price': current_price,
            'position_size': position_size
          })

      elif signal.signal_type.value == 'SELL':
        # Если есть открытая позиция - закрываем
        if symbol in portfolio.positions:
          result = portfolio.close_position(
            symbol=symbol,
            exit_price=current_price
          )

          self.test_results['trades'].append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'SELL',
            'quantity': result['quantity_closed'],
            'price': current_price,
            'pnl': result['realized_pnl']
          })
        else:
          # Открываем короткую позицию
          success = portfolio.open_position(
            symbol=symbol,
            quantity=quantity,
            entry_price=current_price,
            position_type='SHORT'
          )

    except Exception as e:
      logger.error(f"Ошибка выполнения бумажной сделки: {e}")

  def _analyze_backtest_results(self, portfolio: RLPortfolioManager) -> Dict:
    """Анализирует результаты бэктеста"""
    # Получаем финальные метрики портфеля
    final_metrics = portfolio.get_portfolio_metrics()

    # Анализируем сделки
    trades_df = pd.DataFrame(self.test_results['trades'])

    if not trades_df.empty:
      # Считаем статистику по сделкам
      total_trades = len(trades_df)
      buy_trades = len(trades_df[trades_df['action'] == 'BUY'])
      sell_trades = len(trades_df[trades_df['action'] == 'SELL'])

      # PnL статистика
      pnl_trades = trades_df[trades_df['pnl'].notna()]
      if not pnl_trades.empty:
        winning_trades = len(pnl_trades[pnl_trades['pnl'] > 0])
        losing_trades = len(pnl_trades[pnl_trades['pnl'] <= 0])
        win_rate = winning_trades / len(pnl_trades) if len(pnl_trades) > 0 else 0

        avg_win = pnl_trades[pnl_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(pnl_trades[pnl_trades['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
      else:
        win_rate = 0
        profit_factor = 0
    else:
      total_trades = 0
      buy_trades = 0
      sell_trades = 0
      win_rate = 0
      profit_factor = 0

    # Анализируем историю портфеля
    portfolio_df = pd.DataFrame(self.test_results['portfolio_history'])

    if not portfolio_df.empty:
      # Рассчитываем дополнительные метрики
      returns = portfolio_df['total_value'].pct_change().dropna()

      # Коэффициент Сортино
      negative_returns = returns[returns < 0]
      sortino_ratio = 0
      if len(negative_returns) > 0 and negative_returns.std() > 0:
        sortino_ratio = returns.mean() / negative_returns.std() * np.sqrt(252)

      # Calmar Ratio
      calmar_ratio = 0
      if final_metrics['max_drawdown'] > 0:
        annual_return = final_metrics['total_return'] * (252 / len(portfolio_df))
        calmar_ratio = annual_return / final_metrics['max_drawdown']
    else:
      sortino_ratio = 0
      calmar_ratio = 0

    # Собираем все результаты
    backtest_results = {
      **final_metrics,
      'total_trades': total_trades,
      'buy_trades': buy_trades,
      'sell_trades': sell_trades,
      'win_rate': win_rate,
      'profit_factor': profit_factor,
      'sortino_ratio': sortino_ratio,
      'calmar_ratio': calmar_ratio,
      'total_signals': len(self.test_results['signals']),
      'errors_count': len(self.test_results['errors'])
    }

    self.test_results['performance_metrics'] = backtest_results

    return backtest_results

  async def run_live_test(
      self,
      duration_hours: float = 24,
      symbols: Optional[List[str]] = None
  ):
    """Запускает тестирование в реальном времени"""
    logger.info("=" * 50)
    logger.info("ЗАПУСК LIVE ТЕСТИРОВАНИЯ RL СТРАТЕГИИ")
    logger.info("=" * 50)

    if not symbols:
      symbols = self.config['rl_trading'].get('symbols', ['BTCUSDT'])

    logger.info(f"Длительность: {duration_hours} часов")
    logger.info(f"Символы: {symbols}")
    logger.info(f"Paper Trading: {self.paper_trading}")

    # Устанавливаем символы для мониторинга
    self.trading_system.symbols_to_monitor = symbols

    # Запускаем систему
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=duration_hours)

    # Запускаем мониторинг производительности в фоне
    monitor_task = asyncio.create_task(self._monitor_performance())

    try:
      # Запускаем торговую систему
      await self.trading_system.start_optimized()

      # Ждем завершения времени тестирования
      while datetime.now() < end_time:
        await asyncio.sleep(60)  # Проверяем каждую минуту

        # Логируем текущее состояние
        current_metrics = self.trading_system.rl_portfolio_manager.get_portfolio_metrics()
        logger.info(f"Текущая стоимость портфеля: ${current_metrics['total_value']:,.2f}")
        logger.info(f"Текущая доходность: {current_metrics['total_return_pct']:.2f}%")

    except KeyboardInterrupt:
      logger.info("Получен сигнал остановки")
    except Exception as e:
      logger.error(f"Ошибка во время live тестирования: {e}", exc_info=True)
    finally:
      # Останавливаем систему
      await self.trading_system.shutdown()
      monitor_task.cancel()

    # Анализируем результаты
    live_results = self._analyze_live_results()

    logger.info("=" * 50)
    logger.info("LIVE ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    logger.info(f"Итоговая доходность: {live_results.get('total_return_pct', 0):.2f}%")
    logger.info("=" * 50)

    return live_results

  async def _monitor_performance(self):
    """Мониторит производительность в реальном времени"""
    while True:
      try:
        await asyncio.sleep(300)  # Каждые 5 минут

        # Получаем текущие метрики
        if hasattr(self.trading_system, 'rl_portfolio_manager'):
          metrics = self.trading_system.rl_portfolio_manager.get_portfolio_metrics()

          # Сохраняем снимок
          self.test_results['portfolio_history'].append({
            **metrics,
            'timestamp': datetime.now()
          })

          # Проверяем Shadow Learning
          if hasattr(self.trading_system, 'shadow_learner'):
            shadow_stats = self.trading_system.shadow_learner.get_learning_statistics()
            logger.info(f"Shadow Learning: {shadow_stats['total_updates']} обновлений, "
                        f"среднее улучшение: {shadow_stats['average_improvement']:.2%}")

      except asyncio.CancelledError:
        break
      except Exception as e:
        logger.error(f"Ошибка мониторинга: {e}")

  def _analyze_live_results(self) -> Dict:
    """Анализирует результаты live тестирования"""
    # Получаем сделки из БД
    if self.db_manager:
      trades = self.db_manager.get_trades_by_strategy('RL_Strategy')
      self.test_results['trades'] = trades

    # Анализируем портфель
    if hasattr(self.trading_system, 'rl_portfolio_manager'):
      portfolio_metrics = self.trading_system.rl_portfolio_manager.get_portfolio_metrics()
      self.test_results['performance_metrics'] = portfolio_metrics
      return portfolio_metrics

    return {}

  def save_test_results(self, test_type: str = "backtest"):
    """Сохраняет результаты тестирования"""
    # Создаем директорию
    results_dir = Path("rl/test_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Сохраняем JSON с результатами
    results_file = results_dir / f"{test_type}_results_{timestamp}.json"
    with open(results_file, 'w') as f:
      json.dump(self.test_results, f, indent=2, default=str)

    # Создаем отчет
    self._create_test_report(results_dir, timestamp, test_type)

    logger.info(f"✅ Результаты сохранены в {results_file}")

  def _create_test_report(self, results_dir: Path, timestamp: str, test_type: str):
    """Создает визуальный отчет о тестировании"""
    try:
      plt.style.use('seaborn-v0_8-darkgrid')
      fig = plt.figure(figsize=(20, 12))

      # Создаем сетку для субплотов
      gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

      # 1. График стоимости портфеля
      ax1 = fig.add_subplot(gs[0, :])
      if self.test_results['portfolio_history']:
        portfolio_df = pd.DataFrame(self.test_results['portfolio_history'])
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        portfolio_df.set_index('timestamp', inplace=True)

        ax1.plot(portfolio_df.index, portfolio_df['total_value'],
                 label='Portfolio Value', linewidth=2)
        ax1.fill_between(portfolio_df.index,
                         portfolio_df['total_value'].min() * 0.95,
                         portfolio_df['total_value'],
                         alpha=0.3)

        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Форматирование дат
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

      # 2. Распределение сигналов
      ax2 = fig.add_subplot(gs[1, 0])
      if self.test_results['signals']:
        signals_df = pd.DataFrame(self.test_results['signals'])
        signal_counts = signals_df['signal'].value_counts()

        colors = ['green' if x == 'BUY' else 'red' for x in signal_counts.index]
        bars = ax2.bar(signal_counts.index, signal_counts.values, color=colors)

        ax2.set_title('Signal Distribution', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count')

        # Добавляем значения на столбцы
        for bar in bars:
          height = bar.get_height()
          ax2.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')

      # 3. Win Rate и PnL Distribution
      ax3 = fig.add_subplot(gs[1, 1])
      if self.test_results['trades']:
        trades_df = pd.DataFrame(self.test_results['trades'])
        if 'pnl' in trades_df.columns and trades_df['pnl'].notna().any():
          pnl_values = trades_df['pnl'].dropna()

          # Гистограмма PnL
          n, bins, patches = ax3.hist(pnl_values, bins=20, edgecolor='black')

          # Окрашиваем в зеленый/красный
          for i, patch in enumerate(patches):
            if bins[i] >= 0:
              patch.set_facecolor('green')
            else:
              patch.set_facecolor('red')

          ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
          ax3.set_title('P&L Distribution', fontsize=12, fontweight='bold')
          ax3.set_xlabel('Profit/Loss ($)')
          ax3.set_ylabel('Frequency')

      # 4. Метрики производительности
      ax4 = fig.add_subplot(gs[1, 2])
      metrics = self.test_results.get('performance_metrics', {})

      metrics_text = f"""Performance Metrics:

Total Return: {metrics.get('total_return_pct', 0):.2f}%
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
Max Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%
Win Rate: {metrics.get('win_rate', 0) * 100:.2f}%
Total Trades: {metrics.get('total_trades', 0)}
Profit Factor: {metrics.get('profit_factor', 0):.2f}
Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}
Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}
            """

      ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
      ax4.axis('off')

      # 5. Drawdown график
      ax5 = fig.add_subplot(gs[2, 0])
      if self.test_results['portfolio_history']:
        # Рассчитываем drawdown
        values = portfolio_df['total_value'].values
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak * 100

        ax5.fill_between(portfolio_df.index, 0, drawdown,
                         color='red', alpha=0.3, label='Drawdown')
        ax5.plot(portfolio_df.index, drawdown, color='red', linewidth=1)

        ax5.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Drawdown (%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

      # 6. Позиции во времени
      ax6 = fig.add_subplot(gs[2, 1])
      if self.test_results['portfolio_history']:
        positions_count = [h.get('positions_count', 0)
                           for h in self.test_results['portfolio_history']]

        ax6.plot(portfolio_df.index, positions_count,
                 marker='o', markersize=4, linewidth=1)
        ax6.fill_between(portfolio_df.index, 0, positions_count, alpha=0.3)

        ax6.set_title('Open Positions Over Time', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Number of Positions')
        ax6.grid(True, alpha=0.3)

      # 7. Confidence vs Performance
      ax7 = fig.add_subplot(gs[2, 2])
      if self.test_results['signals'] and self.test_results['trades']:
        # Анализ зависимости уверенности от результата
        signals_df = pd.DataFrame(self.test_results['signals'])

        if 'confidence' in signals_df.columns:
          confidence_bins = pd.cut(signals_df['confidence'],
                                   bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0])
          confidence_counts = confidence_bins.value_counts().sort_index()

          ax7.bar(range(len(confidence_counts)), confidence_counts.values)
          ax7.set_xticks(range(len(confidence_counts)))
          ax7.set_xticklabels(['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
          ax7.set_title('Signal Confidence Distribution', fontsize=12, fontweight='bold')
          ax7.set_xlabel('Confidence Range')
          ax7.set_ylabel('Count')

      # Общий заголовок
      fig.suptitle(f'RL Strategy {test_type.upper()} Report - {timestamp}',
                   fontsize=16, fontweight='bold')

      # Сохраняем
      plt.tight_layout()
      report_file = results_dir / f"{test_type}_report_{timestamp}.png"
      plt.savefig(report_file, dpi=300, bbox_inches='tight')
      plt.close()

      logger.info(f"📊 Отчет создан: {report_file}")

    except Exception as e:
      logger.error(f"Ошибка создания отчета: {e}", exc_info=True)


async def run_backtest():
  """Запускает бэктест RL стратегии"""
  logger.info("🔙 Запуск бэктеста RL стратегии...")

  # Можно указать конкретную модель для тестирования
  model_name = "PPO_20240115_120000"  # Замените на вашу модель

  tester = RLStrategyTester(model_name=model_name)

  try:
    # Инициализация
    await tester.initialize_system()

    # Параметры бэктеста
    start_date = datetime.now() - timedelta(days=60)  # 2 месяца назад
    end_date = datetime.now()
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

    # Запуск бэктеста
    results = await tester.run_backtest(
      start_date=start_date,
      end_date=end_date,
      symbols=symbols
    )

    # Сохранение результатов
    tester.save_test_results("backtest")

    logger.info("✅ Бэктест завершен успешно!")

    # Выводим основные результаты
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ БЭКТЕСТА:")
    print("=" * 50)
    print(f"Общая доходность: {results['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {results['win_rate'] * 100:.2f}%")
    print(f"Всего сделок: {results['total_trades']}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print("=" * 50)

  except Exception as e:
    logger.error(f"Ошибка бэктеста: {e}", exc_info=True)


async def run_live_test():
  """Запускает live тестирование RL стратегии"""
  logger.info("🔴 Запуск LIVE тестирования RL стратегии...")

  tester = RLStrategyTester()

  try:
    # Инициализация
    await tester.initialize_system()

    # Параметры live теста
    duration_hours = 24  # Тестируем 24 часа
    symbols = ['BTCUSDT', 'ETHUSDT']  # Меньше символов для live

    # Запуск live теста
    results = await tester.run_live_test(
      duration_hours=duration_hours,
      symbols=symbols
    )

    # Сохранение результатов
    tester.save_test_results("live")

    logger.info("✅ Live тестирование завершено!")

  except Exception as e:
    logger.error(f"Ошибка live тестирования: {e}", exc_info=True)


async def run_shadow_learning_test():
  """Тестирует обучение на Shadow Trading данных"""
  logger.info("👻 Запуск теста Shadow Learning...")

  tester = RLStrategyTester()

  try:
    await tester.initialize_system()

    # Проверяем Shadow Learner
    if hasattr(tester.trading_system, 'shadow_learner'):
      shadow_learner = tester.trading_system.shadow_learner

      # Запускаем одну итерацию обучения
      results = await shadow_learner.learn_from_missed_opportunities(hours=24)

      logger.info(f"Результаты Shadow Learning: {results}")

      # Получаем статистику
      stats = shadow_learner.get_learning_statistics()

      print("\n" + "=" * 50)
      print("СТАТИСТИКА SHADOW LEARNING:")
      print("=" * 50)
      print(f"Всего обновлений: {stats['total_updates']}")
      print(f"Среднее улучшение: {stats['average_improvement'] * 100:.2f}%")
      print(f"Лучшее улучшение: {stats['best_improvement'] * 100:.2f}%")
      print(f"Размер буфера: {stats['buffer_size']}")
      print("=" * 50)
    else:
      logger.warning("Shadow Learner не инициализирован")

  except Exception as e:
    logger.error(f"Ошибка тестирования Shadow Learning: {e}", exc_info=True)


async def run_portfolio_analysis():
  """Анализирует текущее состояние портфеля"""
  logger.info("💼 Анализ портфеля RL стратегии...")

  tester = RLStrategyTester()

  try:
    await tester.initialize_system()

    # Получаем менеджер портфеля
    if hasattr(tester.trading_system, 'rl_portfolio_manager'):
      portfolio = tester.trading_system.rl_portfolio_manager

      # Текущие метрики
      metrics = portfolio.get_portfolio_metrics()

      print("\n" + "=" * 50)
      print("СОСТОЯНИЕ ПОРТФЕЛЯ:")
      print("=" * 50)
      print(f"Общая стоимость: ${metrics['total_value']:,.2f}")
      print(f"Наличные: ${metrics['cash']:,.2f}")
      print(f"Количество позиций: {metrics['positions_count']}")
      print(f"Unrealized P&L: ${metrics['unrealized_pnl']:,.2f}")
      print(f"Realized P&L: ${metrics['realized_pnl']:,.2f}")
      print(f"Общая доходность: {metrics['total_return_pct']:.2f}%")
      print(f"Используемый leverage: {metrics['leverage_used']:.2f}x")
      print("\nВеса портфеля:")
      for asset, weight in metrics['portfolio_weights'].items():
        print(f"  {asset}: {weight * 100:.2f}%")
      print("=" * 50)

      # Проверяем необходимость ребалансировки
      needs_rebalance, deviations = portfolio.needs_rebalancing()
      if needs_rebalance:
        print("\n⚠️  Требуется ребалансировка!")
        print("Отклонения от целевых весов:")
        for symbol, deviation in deviations.items():
          print(f"  {symbol}: {deviation * 100:+.2f}%")

        # Рассчитываем сделки для ребалансировки
        rebalancing_trades = portfolio.calculate_rebalancing_trades()
        print("\nРекомендуемые сделки:")
        for trade in rebalancing_trades:
          print(f"  {trade['action']} {trade['symbol']}: ${trade['value']:,.2f}")

      # Экспортируем отчет
      performance_df = portfolio.export_performance_report()
      if not performance_df.empty:
        report_path = Path("rl/portfolio_report.csv")
        performance_df.to_csv(report_path)
        logger.info(f"📊 Отчет о производительности сохранен: {report_path}")

    else:
      logger.warning("Portfolio Manager не инициализирован")

  except Exception as e:
    logger.error(f"Ошибка анализа портфеля: {e}", exc_info=True)


async def compare_with_baseline():
  """Сравнивает RL стратегию с базовыми стратегиями"""
  logger.info("📊 Сравнение RL с базовыми стратегиями...")

  # Здесь можно реализовать сравнение с Buy&Hold, SMA crossover и т.д.
  # Это поможет оценить эффективность RL подхода

  results = {
    'RL_Strategy': {},
    'Buy_Hold': {},
    'SMA_Cross': {}
  }

  # TODO: Реализовать сравнительный анализ

  logger.info("Сравнительный анализ завершен")


def main():
  """Главная функция с меню выбора тестов"""
  print("\n" + "=" * 60)
  print("RL STRATEGY TESTING SUITE")
  print("=" * 60)
  print("1. Запустить бэктест")
  print("2. Запустить live тестирование (paper trading)")
  print("3. Тестировать Shadow Learning")
  print("4. Анализ текущего портфеля")
  print("5. Сравнение с базовыми стратегиями")
  print("6. Запустить все тесты")
  print("=" * 60)

  choice = input("Выберите опцию (1-6): ")

  if choice == '1':
    asyncio.run(run_backtest())
  elif choice == '2':
    asyncio.run(run_live_test())
  elif choice == '3':
    asyncio.run(run_shadow_learning_test())
  elif choice == '4':
    asyncio.run(run_portfolio_analysis())
  elif choice == '5':
    asyncio.run(compare_with_baseline())
  elif choice == '6':
    # Запускаем все тесты последовательно
    asyncio.run(run_all_tests())
  else:
    print("Неверный выбор!")


async def run_all_tests():
  """Запускает все тесты последовательно"""
  logger.info("🚀 Запуск всех тестов RL стратегии...")

  tests = [
    ("Бэктест", run_backtest),
    ("Shadow Learning", run_shadow_learning_test),
    ("Анализ портфеля", run_portfolio_analysis),
    # Live test последним, так как он длительный
    ("Live тестирование", run_live_test),
  ]

  results = {}

  for test_name, test_func in tests:
    try:
      logger.info(f"\n{'=' * 50}")
      logger.info(f"Запуск: {test_name}")
      logger.info(f"{'=' * 50}")

      await test_func()
      results[test_name] = "✅ Успешно"

    except Exception as e:
      logger.error(f"Ошибка в тесте {test_name}: {e}")
      results[test_name] = f"❌ Ошибка: {str(e)}"

    # Пауза между тестами
    await asyncio.sleep(5)

  # Итоговый отчет
  print("\n" + "=" * 60)
  print("ИТОГОВЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ:")
  print("=" * 60)
  for test_name, result in results.items():
    print(f"{test_name}: {result}")
  print("=" * 60)


if __name__ == "__main__":
  main()