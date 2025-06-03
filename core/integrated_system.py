import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np

from utils.logging_config import get_logger
from config import trading_params, api_keys, settings
from core.data_fetcher import DataFetcher
from core.bybit_connector import BybitConnector
from core.strategy_manager import StrategyManager  # Будет использоваться позже
# from core.risk_manager import AdvancedRiskManager # Будет использоваться позже
# from core.trade_executor import TradeExecutor # Будет использоваться позже
# from data.database_manager import AdvancedDatabaseManager # Будет использоваться позже
from core.enums import Timeframe  # Для запроса свечей
from core.schemas import RiskMetrics  # Для отображения баланса

logger = get_logger(__name__)


class IntegratedTradingSystem:
  def __init__(self):
    logger.info("Инициализация IntegratedTradingSystem...")
    self.connector = BybitConnector()
    self.data_fetcher = DataFetcher(self.connector)
    self.strategy_manager = StrategyManager()  # Пока простой
    # self.db_manager = AdvancedDatabaseManager(settings.DATABASE_PATH) # Позже
    # self.risk_manager = AdvancedRiskManager(self.db_manager, self.connector) # Позже
    # self.trade_executor = TradeExecutor(self.db_manager, self.connector) # Позже

    self.active_symbols: List[str] = []
    self.account_balance: Optional[RiskMetrics] = None  # Будет хранить объект RiskMetrics
    self.current_leverage: Dict[str, int] = {}  # символ: плечо
    self.is_running = False
    self._monitoring_task: Optional[asyncio.Task] = None
    logger.info("IntegratedTradingSystem инициализирован.")

  async def initialize(self):
    """Инициализация системы: выбор символов, получение баланса."""
    logger.info("Начало инициализации системы...")

    await self.connector.initialize()

    self.active_symbols = await self.data_fetcher.get_active_symbols_by_volume()

    if not self.active_symbols:
      logger.warning("Не удалось выбрать активные символы. Система не может продолжить работу без символов.")
      # Можно либо завершить работу, либо перейти в режим ожидания/повторной попытки
      return False

    logger.info(f"Активные символы для торговли: {self.active_symbols}")
    await self.update_account_balance()
    # Установка плеча по умолчанию для всех активных символов (если нужно)
    for symbol in self.active_symbols:
      self.current_leverage.setdefault(symbol, trading_params.DEFAULT_LEVERAGE)
      await self.set_leverage_for_symbol(symbol, trading_params.DEFAULT_LEVERAGE) # Реальная установка плеча
    logger.info("Инициализация системы завершена.")
    return True

  async def update_account_balance(self):
    logger.info("Запрос баланса аккаунта...")
    balance_data = await self.connector.get_account_balance(account_type="CONTRACT", coin="USDT")
    if balance_data and 'walletBalance' in balance_data and 'availableToWithdraw' in balance_data:
      self.account_balance = RiskMetrics(
        total_balance_usdt=float(balance_data.get('total', 0)),
        available_balance_usdt=float(balance_data.get('free', 0)),

        unrealized_pnl_total=float(balance_data.get('used', 0))  # Пример
      )
      logger.info(f"Баланс обновлен: Всего={self.account_balance.total_balance_usdt:.2f} USDT, "
                  f"Доступно={self.account_balance.available_balance_usdt:.2f} USDT, "
                  f"Нереализ. PNL={self.account_balance.unrealized_pnl_total:.2f} USDT")
    else:
      logger.error(f"Не удалось получить или распарсить данные о балансе. Ответ: {balance_data}")
      self.account_balance = RiskMetrics()  # Устанавливаем значения по умолчанию

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
        self.current_leverage[symbol] = leverage
        return True
      else:
        logger.error(f"Не удалось установить плечо для {symbol}.")
        return False
    except Exception as e:
      logger.error(f"Ошибка при установке плеча для {symbol}: {e}", exc_info=True)
      return False

  async def _monitor_symbol(self, symbol: str):
    """Логика мониторинга для одного символа."""
    logger.info(f"Начат мониторинг символа: {symbol}")
    try:
      # 1. Получаем исторические данные (например, за последние 200 часов для анализа)
      # Таймфрейм для стратегии - пока 1 час, можно сделать настраиваемым
      historical_data = await self.data_fetcher.get_historical_candles(symbol, Timeframe.ONE_HOUR, limit=200)

      if historical_data.empty:
        logger.warning(f"Нет исторических данных для {symbol}, пропускаем цикл стратегии.")
        return

      # 2. Передаем данные в StrategyManager для получения сигнала (пока заглушка)
      # В будущем StrategyManager будет содержать вашу ML-стратегию
      # trading_signal = await self.strategy_manager.get_signal(symbol, historical_data)
      logger.debug(f"Заглушка: данные для {symbol} переданы в StrategyManager (размер: {len(historical_data)}).")
      # TODO: Когда StrategyManager будет реализован, раскомментировать и обработать trading_signal
      # if trading_signal:
      #     logger.info(f"Получен сигнал для {symbol}: {trading_signal}")
      #     # 3. Если есть сигнал, передаем в RiskManager (пока заглушка)
      #     # approved_decision = await self.risk_manager.validate_signal(trading_signal, ...)
      #     # if approved_decision and approved_decision['approved']:
      #     #     logger.info(f"Сигнал для {symbol} одобрен риск-менеджером: {approved_decision}")
      #     #     # 4. Если одобрено, передаем в TradeExecutor (пока заглушка)
      #     #     await self.trade_executor.execute_order(approved_decision)
      #     # else:
      #     #     logger.info(f"Сигнал для {symbol} отклонен риск-менеджером или нет одобрения.")
      # else:
      #     logger.debug(f"Нет сигнала от стратегии для {symbol}.")

    except Exception as e:
      logger.error(f"Ошибка при мониторинге символа {symbol}: {e}", exc_info=True)

  async def _monitoring_loop(self):
    logger.info("Запущен основной цикл мониторинга.")
    while self.is_running:
      if not self.active_symbols:
        logger.warning("Нет активных символов для мониторинга. Цикл ожидает.")
        await asyncio.sleep(trading_params.MONITORING_INTERVAL_SECONDS)
        # Попытка повторной инициализации символов, если они пропали
        await self.initialize_symbols_if_empty()
        continue

      logger.info(f"Начало нового цикла мониторинга для {len(self.active_symbols)} символов.")
      await self.update_account_balance()  # Обновляем баланс в начале каждого цикла

      # Создаем задачи для параллельного мониторинга каждого символа
      tasks = [self._monitor_symbol(symbol) for symbol in self.active_symbols]
      await asyncio.gather(*tasks,
                           return_exceptions=True)  # return_exceptions=True чтобы цикл не падал из-за одной ошибки

      logger.info(f"Цикл мониторинга завершен. Ожидание {trading_params.MONITORING_INTERVAL_SECONDS} секунд.")
      await asyncio.sleep(trading_params.MONITORING_INTERVAL_SECONDS)

  async def initialize_symbols_if_empty(self):
    if not self.active_symbols:
      logger.info("Список активных символов пуст, попытка повторной инициализации...")
      self.active_symbols = await self.data_fetcher.get_active_symbols_by_volume()
      if self.active_symbols:
        logger.info(f"Символы успешно реинициализированы: {self.active_symbols}")
      else:
        logger.warning("Не удалось реинициализировать символы.")

  async def start(self):
    if self.is_running:
      logger.warning("Система уже запущена.")
      return

    if not await self.initialize():
      logger.error("Сбой инициализации системы. Запуск отменен.")
      return

    self.is_running = True
    # ИСПРАВЛЕНО: Убрали дублирующий вызов initialize()
    logger.info("Торговая система запускается...")
    self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    logger.info("Торговая система успешно запущена.")

  async def stop(self):
    if not self.is_running:
      logger.warning("Система не запущена.")
      return

    self.is_running = False
    logger.info("Остановка торговой системы...")
    if self._monitoring_task:
      self._monitoring_task.cancel()
      try:
        await self._monitoring_task
      except asyncio.CancelledError:
        logger.info("Цикл мониторинга успешно отменен.")

    # ИСПРАВЛЕНО: Закрываем соединения коннектора
    await self.connector.close()
    logger.info("Торговая система остановлена.")

  # Методы для взаимодействия с GUI (пока будут выводить в консоль)
  def display_balance(self):
    if self.account_balance:
      print(f"\n--- Текущий баланс ---")
      print(f"Общий баланс USDT: {self.account_balance.total_balance_usdt:.2f}")
      print(f"Доступный баланс USDT: {self.account_balance.available_balance_usdt:.2f}")
      print(f"Нереализованный PNL: {self.account_balance.unrealized_pnl_total:.2f}")
      print(f"----------------------\n")
    else:
      print("Баланс еще не загружен.")

  def display_active_symbols(self):
    print(f"\n--- Активные торговые пары ---")
    if self.active_symbols:
      for i, symbol in enumerate(self.active_symbols):
        leverage = self.current_leverage.get(symbol, "N/A")
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
