import asyncio
from datetime import datetime, timezone

import pandas as pd
import pandas_ta as ta
from typing import Dict, Optional

from config.config_manager import ConfigManager
from core.bybit_connector import BybitConnector
from core.indicators import crossover_series, crossunder_series
# from core.integrated_system import IntegratedTradingSystem
from core.risk_manager import AdvancedRiskManager
from core.schemas import TradingSignal, RiskMetrics
from data.database_manager import AdvancedDatabaseManager
from core.trade_executor import TradeExecutor
from core.data_fetcher import DataFetcher
from core.enums import Timeframe, SignalType
from strategies.sar_strategy import StopAndReverseStrategy
from utils.logging_config import get_logger
from core.signal_filter import SignalFilter

logger = get_logger(__name__)


class PositionManager:
  """
  Класс для управления открытыми позициями и реализации логики выхода.
  """

  def __init__(self, db_manager: AdvancedDatabaseManager, trade_executor: TradeExecutor, data_fetcher: DataFetcher, connector: BybitConnector, signal_filter: SignalFilter, risk_manager: AdvancedRiskManager, sar_strategy:StopAndReverseStrategy):
    self.db_manager = db_manager
    self.trade_executor = trade_executor
    self.data_fetcher = data_fetcher
    self.connector = connector
    self.signal_filter = signal_filter
    self.open_positions: Dict[str, Dict] = {}  # Кэш открытых позиций для быстрой проверки
    self.risk_manager = risk_manager
    self.config_manager = ConfigManager()
    self.config = self.config_manager.load_config()
    self.trading_system = None
    # self.integrated_system: Optional[IntegratedTradingSystem] = None
    self.sar_strategy = sar_strategy
  async def load_open_positions(self):
    """
    Синхронизирует состояние открытых позиций.
    Получает ВСЕ позиции с биржи, а не только по топовым символам.
    """
    logger.info("Синхронизация открытых позиций с биржей...")
    self.open_positions = {}  # Очищаем старый кэш

    try:
      # НОВЫЙ ПОДХОД: Получаем ВСЕ позиции сразу одним запросом
      endpoint = "/v5/position/list"
      params = {
        'category': 'linear',
        'settleCoin': 'USDT'  # Только USDT позиции
      }

      # Запрашиваем напрямую все позиции
      result = await self.connector._make_request('GET', endpoint, params, use_cache=False)

      # После загрузки позиций с биржи добавить синхронизацию:
      # Удаляем из БД позиции, которых нет на бирже
      db_positions = await self.db_manager.get_all_open_trades()
      exchange_symbols = set(self.open_positions.keys())

      for db_pos in db_positions:
        if db_pos['symbol'] not in exchange_symbols:
          logger.warning(f"Найдена зомби-позиция {db_pos['symbol']} в БД, синхронизация...")
          # Проверяем еще раз напрямую
          positions = await self.connector.fetch_positions(db_pos['symbol'])
          if not any(float(p.get('size', 0)) > 0 for p in positions):
            await self.db_manager.force_close_trade(
              trade_id=db_pos['id'],
              close_price=0,
              reason="Position not found on exchange during sync"
            )


      if result and result.get('list'):
        all_positions = result.get('list', [])
        logger.info(f"Получено {len(all_positions)} позиций с биржи")

        # Обрабатываем только позиции с ненулевым размером
        for position in all_positions:
          size = float(position.get('size', 0))
          if size > 0:
            symbol = position.get('symbol')
            logger.info(f"На бирже найдена активная позиция по {symbol}. Размер: {size}")

            # Ищем соответствующую запись в локальной БД
            local_trade_data = await self.db_manager.get_open_trade_by_symbol(symbol)

            if local_trade_data:
              # Если нашли, используем данные из БД
              logger.info(f"Найдена соответствующая запись в локальной БД для {symbol}")
              self.open_positions[symbol] = local_trade_data
            else:
              # Если в БД нет, создаем заглушку из данных биржи
              logger.warning(f"Позиция по {symbol} существует на бирже, но отсутствует в локальной БД")
              self.open_positions[symbol] = {
                'symbol': symbol,
                'side': position.get('side', 'Buy').upper(),
                'open_price': float(position.get('avgPrice', 0)),
                'quantity': size,
                'stop_loss': float(position.get('stopLoss', 0)) if position.get('stopLoss') else None,
                'take_profit': float(position.get('takeProfit', 0)) if position.get('takeProfit') else None,
                'unrealizedPnl': float(position.get('unrealisedPnl', 0)),
                'leverage': int(position.get('leverage', 1)),
                'id': -1  # Указываем, что это "неизвестная" сделка
              }
      else:
        logger.info("Не получено позиций с биржи")

    except Exception as e:
      logger.error(f"Критическая ошибка при синхронизации позиций с биржей: {e}", exc_info=True)

    if self.open_positions:
      logger.info(f"Синхронизация завершена. Активные позиции: {list(self.open_positions.keys())}")
    else:
      logger.info("Синхронизация завершена. Активных позиций на бирже не найдено.")

  def _check_sl_tp(self, position: Dict, current_price: float) -> Optional[str]:
    """Проверяет, сработал ли Stop-Loss или Take-Profit."""
    side = position.get('side')
    stop_loss = position.get('stop_loss')
    take_profit = position.get('take_profit')

    if side == 'BUY':
      if stop_loss and current_price <= stop_loss:
        return f"Stop-Loss для BUY сработал: цена {current_price:.4f} <= SL {stop_loss:.4f}"
      if take_profit and current_price >= take_profit:
        return f"Take-Profit для BUY сработал: цена {current_price:.4f} >= TP {take_profit:.4f}"
    elif side == 'SELL':
      if stop_loss and current_price >= stop_loss:
        return f"Stop-Loss для SELL сработал: цена {current_price:.4f} >= SL {stop_loss:.4f}"
      if take_profit and current_price <= take_profit:
        return f"Take-Profit для SELL сработал: цена {current_price:.4f} <= TP {take_profit:.4f}"
    return None

  def _check_dynamic_exit(self, position: Dict, data: pd.DataFrame) -> Optional[str]:
    """Проверяет условие динамического выхода (пересечение EMA)."""
    try:
      ema_fast = ta.ema(data['close'], length=12)
      ema_slow = ta.ema(data['close'], length=26)
      if ema_fast is None or ema_slow is None or len(ema_fast) < 2: return None

      side = position.get('side')
      if side == 'BUY' and ema_fast.iloc[-2] > ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return "Обнаружен медвежий кроссовер EMA(12/26)"
      if side == 'SELL' and ema_fast.iloc[-2] < ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]:
        return "Обнаружен бычий кроссовер EMA(12/26)"
      return None
    except Exception as e:
      logger.error(f"Ошибка при проверке динамического выхода для {position['symbol']}: {e}")
      return None

  async def manage_open_positions(self, account_balance: Optional[RiskMetrics]):
    """
    ФИНАЛЬНАЯ ВЕРСИЯ: Управляет открытыми позициями, используя
    иерархию проверок: SL/TP -> PSAR -> Stop-and-Reverse.
    """
    if not hasattr(self, '_last_order_check'):
      self._last_order_check = datetime.now()

    if (datetime.now() - self._last_order_check).seconds > 30:  # Каждую минуту
      await self.track_pending_orders()
      self._last_order_check = datetime.now()

    if not self.open_positions:
      return


    logger.debug(f"Динамическое управление для {len(self.open_positions)} открытых позиций...")

    for symbol, position_data in list(self.open_positions.items()):
      try:
        # Загружаем данные разных таймфреймов для комплексного анализа
        timeframes_data = {}

        # Основные таймфреймы для анализа
        analysis_timeframes = {
          '1m': Timeframe.ONE_MINUTE,
          '5m': Timeframe.FIVE_MINUTES,
          '15m': Timeframe.FIFTEEN_MINUTES,
          '30m': Timeframe.THIRTY_MINUTES,
          '1h': Timeframe.ONE_HOUR
        }

        # Загружаем все необходимые данные
        for tf_name, tf_enum in analysis_timeframes.items():
          tf_data = await self.data_fetcher.get_historical_candles(
            symbol, tf_enum, limit=100
          )
          if not tf_data.empty:
            # Добавляем технические индикаторы
            tf_data['atr'] = ta.atr(tf_data['high'], tf_data['low'], tf_data['close'])

            # PSAR для основных таймфреймов
            if tf_name in ['1m', '5m', '15m', '1h']:
              psar_df = ta.psar(tf_data['high'], tf_data['low'], tf_data['close'])
              if psar_df is not None:
                psar_col = next(
                  (col for col in psar_df.columns if 'PSAR' in col and 'PSARl' not in col and 'PSARs' not in col),
                  None
                )
                if psar_col:
                  tf_data['psar'] = psar_df[psar_col]

            # Aroon для 5-минутного таймфрейма
            if tf_name == '5m':
              aroon = ta.aroon(tf_data['high'], tf_data['low'])
              if aroon is not None and not aroon.empty:
                tf_data['aroon_up'] = aroon.iloc[:, 0]
                tf_data['aroon_down'] = aroon.iloc[:, 1]
                tf_data['aroon_osc'] = tf_data['aroon_up'] - tf_data['aroon_down']

            timeframes_data[tf_name] = tf_data

        # Используем 1h как основной таймфрейм для обратной совместимости
        htf_data = timeframes_data.get('1h', pd.DataFrame())
        if htf_data.empty:
          continue

        current_price = htf_data['close'].iloc[-1]
        # # --- ИСПРАВЛЕНИЕ: ЗАГРУЖАЕМ ДАННЫЕ ЗАРАНЕЕ ---
        # # 1. Загружаем данные основного таймфрейма (1H) для анализа SL/TP и PSAR
        # htf_data = await self.data_fetcher.get_historical_candles(symbol, Timeframe.ONE_HOUR, limit=100)
        # if htf_data.empty:
        #   continue
        # # add_atr(htf_data)
        # current_price = htf_data['close'].iloc[-1]
        #
        # # 2. Загружаем данные малого таймфрейма (1m) для анализа разворота
        strategy_settings = self.config.get('strategy_settings', {})
        ltf_str = strategy_settings.get('ltf_entry_timeframe', '15m')
        timeframe_map = {"1m": Timeframe.ONE_MINUTE, "5m": Timeframe.FIVE_MINUTES, "15m": Timeframe.FIFTEEN_MINUTES}
        ltf_timeframe = timeframe_map.get(ltf_str, Timeframe.ONE_MINUTE)
        ltf_data = await self.data_fetcher.get_historical_candles(symbol, ltf_timeframe, limit=100)

        atr = htf_data['atr'].iloc[-1]
        price = htf_data['close'].iloc[-1]
        atr_percentage = (atr / price) * 100

        if atr_percentage > 5.0:  # Если ATR больше 5% от цены
          logger.info(f"Высокая волатильность для {symbol} (ATR: {atr_percentage:.2f}%), ужесточаем условия выхода")
          # Временно увеличиваем требования для выхода
          strategy_settings['atr_ts_multiplier'] = strategy_settings.get('atr_ts_multiplier', 2.5) * 1.5

        min_hold_time_minutes = 30  # Минимальное время удержания позиции

        if 'open_timestamp' in position_data:
          open_time = pd.to_datetime(position_data['open_timestamp'])
          current_time = datetime.now(timezone.utc)

          # Если времен��ая зона не установлена
          if open_time.tzinfo is None:
            open_time = open_time.replace(tzinfo=timezone.utc)

          time_held = (current_time - open_time).total_seconds() / 60

          if time_held < min_hold_time_minutes:
            logger.debug(f"Позиция {symbol} удерживается только {time_held:.1f} минут, пропускаем проверки выхода")
            continue  # Пропускаем все проверки выхода для новых позиций


        # # --- КОНЕЦ БЛОКА ЗАГРУЗКИ ---

        # --- ПРИОРИТЕТ 1: ЖЕСТКИЙ SL/TP ---
        exit_reason = self._check_sl_tp(position_data, current_price)
        if exit_reason:
          logger.info(f"ВЫХОД для {symbol}: Сработал стандартный SL/TP. Причина: {exit_reason}")
          await self.trade_executor.close_position(symbol=symbol)
          continue

        # --- ПРОВЕРКА STOP AND REVERSE ---
        if not exit_reason and self.sar_strategy and strategy_settings.get('use_sar_reversal', True):
          # Проверяем сигнал от SAR стратегии
          sar_signal = await self.sar_strategy.check_exit_conditions(
            symbol, htf_data, position_data
          )

          if sar_signal and sar_signal.is_reversal:
            # Проверяем направление сигнала
            current_side = position_data.get('side')
            new_direction = 'BUY' if sar_signal.signal_type == SignalType.BUY else 'SELL'

            # Если сигнал в том же направлении - пропускаем
            if current_side == new_direction:
              logger.debug(f"SAR сигнал для {symbol} в том же направлении, пропускаем")
            else:
              # Проверяем качество сигнала для разворота
              if sar_signal.confidence >= 0.7:
                logger.info(f"🔄 SAR разворот для {symbol}: {current_side} -> {new_direction}")

                # Выполняем разворот позиции
                reversal_success = await self.trade_executor.reverse_position(
                  symbol=symbol,
                  current_position=position_data,
                  new_signal=sar_signal
                )

                if reversal_success:
                  logger.info(f"✅ Разворот позиции {symbol} выполнен успешно")
                  continue
                else:
                  # Если разворот не удался, закрываем позицию
                  exit_reason = f"SAR сигнал на разворот (не удался автоматический разворот)"

        # --- ПРИОРИТЕТ 2: ТРЕЙЛИНГ-СТОП ПО ATR (на HTF)---
        if not exit_reason:
          exit_reason = self._check_atr_trailing_stop(
            position_data, htf_data, timeframes_data
          )
          if exit_reason:
            logger.info(f"ВЫХОД для {symbol}: {exit_reason}")
            await self.trade_executor.close_position(symbol=symbol)
            continue

        # --- ПРИОРИТЕТ 3: ТРЕЙЛИНГ-СТОП ПО PSAR ---
        if not exit_reason and strategy_settings.get('use_psar_exit', True):
          exit_reason = self._check_psar_exit(
            position_data, htf_data, timeframes_data
          )
          if exit_reason:
            logger.info(f"ВЫХОД для {symbol}: {exit_reason}")
            await self.trade_executor.close_position(symbol=symbol)
            continue

        # --- ПРИОРИТЕТ 4: STOP AND REVERSE ---
        if not exit_reason:
          reverse_signal = await self._check_reversal_exit(position_data, ltf_data)
          if reverse_signal:
            logger.info(f"🔄 Обнаружен сигнал разворота для {symbol}")

            # Проверяем, можем ли использовать функцию reverse
            use_reverse = self.config.get('strategy_settings', {}).get('use_reverse_function', True)

            if use_reverse and hasattr(self.trade_executor, 'reverse_position'):
              # Используем новую функцию reverse
              reverse_success = await self.trade_executor.reverse_position(
                symbol=symbol,
                current_position=position_data,
                new_signal=reverse_signal,
                force=False  # Требуем проверку прибыльности
              )

              if reverse_success:
                logger.info(f"✅ Позиция {symbol} успешно развернута")
                # Обновляем статистику

                if hasattr(self, 'trading_system') and self.trading_system:
                  await self.trading_system.update_strategy_performance(
                    symbol, 'reverse', True
                  )
              else:
                volume_check_window = 5  # Последние 5 свечей
                recent_volume = htf_data['volume'].tail(volume_check_window).mean()
                avg_volume = htf_data['volume'].mean()

                if recent_volume < avg_volume * 0.7:  # Если объем упал на 30%
                  logger.debug(f"Низкий объем для {symbol}, откладываем закрытие")
                  continue


                logger.warning(f"Не удалось развернуть позицию {symbol}, выполняем обычное закрытие")
                # Fallback к обычному закрытию + новый вход
                await self._execute_standard_exit_and_reentry(symbol, reverse_signal,
                                                              account_balance)
            else:
              volume_check_window = 5  # Последние 5 свечей
              recent_volume = htf_data['volume'].tail(volume_check_window).mean()
              avg_volume = htf_data['volume'].mean()

              if recent_volume < avg_volume * 0.7:  # Если объем упал на 30%
                logger.debug(f"Низкий объем для {symbol}, откладываем закрытие")
                continue

              # Стандартный путь: закрытие + новый вход
              await self._execute_standard_exit_and_reentry(symbol, reverse_signal,
                                                            account_balance)

        if exit_reason:
          logger.info(f"ВЫХОД для {symbol}: Обнаружен сигнал на закрытие. Причина: {exit_reason}")
          await self.trade_executor.close_position(symbol=symbol)
          continue

      except Exception as e:
        logger.error(f"Ошибка при динамическом управлении позицией {symbol}: {e}", exc_info=True)

  async def _execute_standard_exit_and_reentry(self, symbol: str,
                                               reverse_signal: TradingSignal,
                                               account_balance: Optional[RiskMetrics]):
    """
    Стандартная процедура: закрытие текущей позиции и открытие новой.
    """
    try:
      # Получаем данные текущей позиции для логирования
      position_data = self.open_positions.get(symbol, {})
      if position_data:
        current_side = position_data.get('side', 'Unknown')
        logger.info(f"Выполняется разворот позиции {symbol}: {current_side} -> {reverse_signal.signal_type.value}")

      # Закрываем текущую позицию
      close_success = await self.trade_executor.close_position(symbol=symbol)
      if not close_success:
        logger.error(f"Не удалось закрыть позицию {symbol} для разворота")
        return

      # Ждем подтверждения закрытия
      await asyncio.sleep(3)

      # Открываем новую позицию в противоположном направлении
      if account_balance:
        # ИСПРАВЛЕНИЕ: Получаем рыночные данные для validate_signal
        try:
          market_data = await self.data_fetcher.get_historical_candles(
            symbol=symbol,
            timeframe=Timeframe.ONE_HOUR,  # или другой подходящий таймфрейм
            limit=100
          )

          if market_data.empty:
            logger.error(f"Не удалось получить рыночные данные для {symbol}")
            return

        except Exception as e:
          logger.error(f"Ошибка получения рыночных данных для {symbol}: {e}")
          return

        # Теперь вызываем validate_signal с правильными параметрами
        risk_decision = await self.risk_manager.validate_signal(
          signal=reverse_signal,
          symbol=symbol,
          account_balance=account_balance.available_balance_usdt,
          market_data=market_data
        )

        if risk_decision.get('approved'):
          logger.info(f"Риск-менеджер одобрил новую позицию после разворота {symbol}")
          quantity = risk_decision.get('recommended_size')
          await self.trade_executor.execute_trade(reverse_signal, symbol, quantity)
        else:
          logger.warning(
            f"Риск-менеджер отклонил новую позицию после разворота {symbol}: {risk_decision.get('reasons')}")

      # Уведомляем SAR стратегию об успешном развороте
      if hasattr(self, 'trading_system') and self.trading_system:
        sar_strategy = getattr(self.trading_system, 'sar_strategy', None)
        if sar_strategy and hasattr(sar_strategy, 'handle_position_update'):
          reversal_data = {
            'symbol': symbol,
            'old_side': current_side,
            'new_side': reverse_signal.signal_type.value,
            'reversal_price': reverse_signal.price,
            'action': 'position_reversal'
          }
          await sar_strategy.handle_position_update(symbol, reversal_data)

    except Exception as e:
      logger.error(f"Ошибка при стандартном развороте {symbol}: {e}")

  async def reconcile_filled_orders(self):
      """
      ИСПРАВЛЕННАЯ ВЕРСИЯ: Сверяет исполненные ордера без дублирования логики
      """
      # Получаем из БД все сделки, которые у нас числятся как "OPEN"
      open_trades_in_db = await self.db_manager.get_all_open_trades()
      if not open_trades_in_db:
        return

      logger.debug(f"Сверка статуса для {len(open_trades_in_db)} открытых в БД сделок...")

      for trade in open_trades_in_db:
        symbol = trade.get('symbol')
        if not symbol:
          continue

        try:
          # 1. Проверяем, есть ли еще реальная позиция на бирже
          positions_on_exchange = await self.connector.fetch_positions(symbol)
          is_still_open = any(float(pos.get('size', 0)) > 0 for pos in positions_on_exchange)

          # 2. Если позиции на бирже уже нет, значит, она была закрыта
          if not is_still_open:
            logger.info(f"Позиция по {symbol} больше не активна на бирже. Поиск исполненной сделки...")

            # Ищем исполнение закрытия
            closing_exec = await self._find_closing_execution(symbol, trade)

            if closing_exec:
              # --- РАСЧЕТ PNL ---
              trade_data = await self._calculate_trade_pnl(trade, closing_exec)

              logger.info(f"ПОДТВЕРЖДЕНИЕ ЗАКРЫТИЯ {symbol}: PnL={trade_data['net_pnl']:.4f}")

              # --- ОБНОВЛЕНИЕ БД ---
              await self.db_manager.update_trade_as_closed(
                trade_id=trade['id'],
                close_price=trade_data['close_price'],
                pnl=trade_data['net_pnl'],
                commission=trade_data['commission'],
                close_timestamp=trade_data['close_timestamp']
              )

              # --- ЕДИНОЕ МЕСТО ДЛЯ ВСЕХ УВЕДОМЛЕНИЙ ---
              await self._notify_systems_about_closed_trade(symbol, trade, trade_data)

              # Удаляем из кэша
              if symbol in self.open_positions:
                del self.open_positions[symbol]

            else:
              # Принудительное закрытие если не найдено исполнение
              logger.warning(f"Не найдено исполнение для {symbol}. Принудительное закрытие.")
              await self.db_manager.force_close_trade(
                trade_id=trade['id'],
                close_price=trade['open_price']
              )

        except Exception as e:
          logger.error(f"Ошибка при сверке сделок для {symbol}: {e}", exc_info=True)

  async def _find_closing_execution(self, symbol: str, trade: dict) -> dict:
    """Находит исполнение закрытия позиции"""
    try:
      executions = await self.connector.get_execution_history(symbol=symbol, limit=20)

      for exec_trade in executions:
        if exec_trade.get('closedSize') and float(exec_trade.get('closedSize', 0)) > 0:
          return exec_trade

      return None
    except Exception as e:
      logger.error(f"Ошибка поиска исполнения для {symbol}: {e}")
      return None

  async def _calculate_trade_pnl(self, trade: dict, closing_exec: dict) -> dict:
    """Рассчитывает PnL и данные закрытия сделки"""
    open_price = float(trade['open_price'])
    close_price = float(closing_exec['execPrice'])
    quantity = float(trade['quantity'])
    commission = float(closing_exec.get('execFee', 0))
    side = trade.get('side')

    # Расчет PnL
    if side == 'BUY':
      gross_pnl = (close_price - open_price) * quantity
    else:
      gross_pnl = (open_price - close_price) * quantity

    net_pnl = gross_pnl - commission
    close_timestamp = datetime.fromtimestamp(int(closing_exec['execTime']) / 1000)

    # Расчет процентной прибыли
    profit_pct = ((close_price - open_price) / open_price * 100) if side == 'BUY' else \
      ((open_price - close_price) / open_price * 100)

    return {
      'open_price': open_price,
      'close_price': close_price,
      'quantity': quantity,
      'commission': commission,
      'side': side,
      'gross_pnl': gross_pnl,
      'net_pnl': net_pnl,
      'profit_pct': profit_pct,
      'close_timestamp': close_timestamp
    }

  async def _notify_systems_about_closed_trade(self, symbol: str, trade: dict, trade_data: dict):
    """
    ЕДИНОЕ МЕСТО для всех уведомлений о закрытой сделке
    Избегает дублирования и обеспечивает правильный порядок
    """
    strategy_name = trade.get('strategy_name', 'Unknown')
    net_pnl = trade_data['net_pnl']
    is_profitable = net_pnl > 0

    # Определяем ссылку на интегрированную систему
    integrated_system = None
    if hasattr(self, 'trading_system'):
      integrated_system = self.trading_system
    elif hasattr(self, 'integrated_system'):
      integrated_system = self.integrated_system
    elif hasattr(self.trade_executor, 'integrated_system'):
      integrated_system = self.trade_executor.integrated_system

    if not integrated_system:
      logger.warning("Не найдена ссылка на интегрированную систему")
      return

    try:
      # 1. ADAPTIVE SELECTOR - обновляем производительность стратегии
      if hasattr(integrated_system, 'adaptive_selector'):
        await integrated_system.adaptive_selector.update_strategy_performance(
          strategy_name=strategy_name,
          is_profitable=is_profitable,
          profit_amount=abs(net_pnl),
          symbol=symbol
        )
        logger.debug(f"✅ Adaptive Selector обновлен для {strategy_name}")

      # 2. SAR STRATEGY - уведомляем о закрытии (ТОЛЬКО ОДИН РАЗ)
      if (strategy_name == "Stop_and_Reverse" or "SAR" in strategy_name.upper()):
        sar_strategy = getattr(integrated_system, 'sar_strategy', None)
        if sar_strategy and hasattr(sar_strategy, 'handle_position_update'):
          if symbol in getattr(sar_strategy, 'current_positions', {}):
            closed_position_data = {
              'symbol': symbol,
              'side': trade_data['side'],
              'close_price': trade_data['close_price'],
              'profit_loss': net_pnl,
              'profit_pct': trade_data['profit_pct'],
              'close_reason': 'exchange_execution',
              'metadata': trade.get('metadata', {})
            }
            await sar_strategy.handle_position_update(symbol, closed_position_data)
            logger.debug(f"✅ SAR стратегия обновлена для {symbol}")

      # 3. SHADOW TRADING - синхронизация
      shadow_manager = getattr(integrated_system, 'shadow_trading', None)
      if shadow_manager and hasattr(shadow_manager, 'signal_tracker'):
        trade_result = {
          'symbol': symbol,
          'close_price': trade_data['close_price'],
          'close_timestamp': trade_data['close_timestamp'],
          'profit_loss': net_pnl,
          'profit_pct': trade_data['profit_pct'],
          'order_id': trade.get('order_id')
        }
        await shadow_manager.signal_tracker.sync_with_real_trades(symbol, trade_result)
        logger.debug(f"✅ Shadow Trading синхронизирован для {symbol}")

        # Финализация Shadow Trading сигнала
        await self._finalize_shadow_trading_signal(trade, trade_data, shadow_manager)

      # 4. PROCESS TRADE FEEDBACK - ML обратная связь
      if hasattr(integrated_system, 'process_trade_feedback'):
        trade_result = {
          'strategy_name': strategy_name,
          'profit_loss': net_pnl,
          'entry_price': trade_data['open_price'],
          'exit_price': trade_data['close_price'],
          'regime': self._extract_regime_from_metadata(trade),
          'confidence': trade.get('confidence', 0.5),
          'entry_features': self._extract_features_from_metadata(trade)
        }
        await integrated_system.process_trade_feedback(symbol, trade['id'], trade_result)
        logger.debug(f"✅ Trade feedback отправлен для {symbol}")

    except Exception as e:
      logger.error(f"Ошибка при уведомлении систем о закрытии {symbol}: {e}")

  async def _finalize_shadow_trading_signal(self, trade: dict, trade_data: dict, shadow_manager):
    """Финализирует сигнал в Shadow Trading"""
    try:
      metadata = trade.get('metadata')
      if not metadata:
        return

      # ИСПРАВЛЕНИЕ: Универсальный парсер метаданных
      metadata_dict = self._safe_parse_metadata(metadata)

      shadow_tracking_id = metadata_dict.get('shadow_tracking_id')
      if not shadow_tracking_id:
        return

      # Определяем исход
      from shadow_trading.signal_tracker import SignalOutcome
      outcome = SignalOutcome.PROFITABLE if trade_data['net_pnl'] > 0 else SignalOutcome.LOSS

      # Финализируем сигнал с правильными параметрами
      await shadow_manager.signal_tracker.finalize_signal(
        signal_id=shadow_tracking_id,
        final_price=trade_data['close_price'],
        exit_time=trade_data['close_timestamp'],
        outcome=outcome
      )
      logger.debug(f"✅ Shadow Trading сигнал {shadow_tracking_id} финализирован")

    except Exception as e:
      logger.warning(f"Не удалось финализировать Shadow Trading сигнал: {e}")

  def _safe_parse_metadata(self, metadata) -> dict:
    """Безопасно парсит метаданные независимо от типа"""
    try:
      if metadata is None:
        return {}

      if isinstance(metadata, dict):
        return metadata

      if isinstance(metadata, str):
        if metadata.strip() == "":
          return {}
        import json
        return json.loads(metadata)

      # Если это что-то другое, пытаемся конвертировать в строку и парсить
      import json
      return json.loads(str(metadata))

    except (json.JSONDecodeError, TypeError, ValueError) as e:
      logger.warning(f"Не удалось парсить метаданные: {e}")
      return {}

  # def _extract_regime_from_metadata(self, trade: dict) -> str:
  #   """Извлекает режим рынка из метаданных сделки"""
  #   try:
  #     metadata = trade.get('metadata', {})
  #     if isinstance(metadata, str):
  #       import json
  #       metadata = json.loads(metadata)
  #     return metadata.get('regime', 'unknown')
  #   except:
  #     return 'unknown'
  #
  # def _extract_features_from_metadata(self, trade: dict) -> dict:
  #   """Извлекает признаки из метаданных сделки"""
  #   try:
  #     metadata = trade.get('metadata', {})
  #     if isinstance(metadata, str):
  #       import json
  #       metadata = json.loads(metadata)
  #     return metadata.get('features', {})
  #   except:
  #     return {}
  def _extract_regime_from_metadata(self, trade: dict) -> str:
      """Извлекает режим рынка из метаданных сделки"""
      try:
        metadata_dict = self._safe_parse_metadata(trade.get('metadata'))
        return metadata_dict.get('regime', 'unknown')
      except:
        return 'unknown'

  def _extract_features_from_metadata(self, trade: dict) -> dict:
    """Извлекает признаки из метаданных сделки"""
    try:
      metadata_dict = self._safe_parse_metadata(trade.get('metadata'))
      return metadata_dict.get('features', {})
    except:
      return {}

        # 2. Полный метод reconcile_filled_orders с использованием fetch_positions_batch:

  # async def reconcile_filled_orders(self):
  #   """Оптимизированная версия с батчевой загрузкой позиций"""
  #
  #   # Получаем ВСЕ позиции одним запросом вместо N запросов
  #   try:
  #     all_positions_response = await self.connector._make_request(
  #       'GET',
  #       '/v5/position/list',
  #       {'category': 'linear', 'settleCoin': 'USDT'},
  #       use_cache=False
  #     )
  #
  #     all_positions = all_positions_response.get('list', []) if all_positions_response else []
  #
  #     # Создаем быстрый lookup по символам
  #     active_positions = {}
  #     for pos in all_positions:
  #       if float(pos.get('size', 0)) > 0:
  #         active_positions[pos['symbol']] = pos
  #
  #     logger.debug(f"Получено {len(active_positions)} активных позиций с биржи")
  #
  #   except Exception as e:
  #     logger.error(f"Ошибка получения всех позиций: {e}")
  #     # Fallback на старый метод
  #     active_positions = {}
  #
  #   # Получаем открытые сделки из БД
  #   open_trades_in_db = await self.db_manager.get_all_open_trades()
  #   if not open_trades_in_db:
  #     return
  #
  #   logger.debug(f"Сверка {len(open_trades_in_db)} сделок с биржей")
  #
  #   for trade in open_trades_in_db:
  #     symbol = trade.get('symbol')
  #     if not symbol:
  #       continue
  #
  #     try:
  #       # ОПТИМИЗАЦИЯ: используем уже загруженные данные вместо отдельного запроса
  #       is_still_open = symbol in active_positions
  #
  #       if not is_still_open:
  #         logger.info(f"Позиция по {symbol} больше не активна на бирже. Поиск исполненной сделки...")
  #
  #         # Ищем в истории исполнений
  #         executions = await self.connector.get_execution_history(symbol=symbol, limit=20)
  #         closing_exec = None
  #
  #         for exec_trade in executions:
  #           if exec_trade.get('closedSize') and float(exec_trade.get('closedSize', 0)) > 0:
  #             closing_exec = exec_trade
  #             break
  #
  #         if closing_exec:
  #           # Расчет PnL
  #           open_price = float(trade['open_price'])
  #           close_price = float(closing_exec['execPrice'])
  #           quantity = float(trade['quantity'])
  #           commission = float(closing_exec.get('execFee', 0))
  #           side = trade.get('side')
  #
  #           gross_pnl = (close_price - open_price) * quantity if side == 'BUY' else (
  #                                                                                         open_price - close_price) * quantity
  #           net_pnl = gross_pnl - commission
  #           close_timestamp = datetime.fromtimestamp(int(closing_exec['execTime']) / 1000)
  #
  #           logger.info(f"ПОДТВЕРЖДЕНИЕ ЗАКРЫТИЯ для {symbol}: Чистый PnL: {net_pnl:.4f}")
  #
  #           # Обновляем БД
  #           await self.db_manager.update_close_trade(
  #             trade['id'],
  #             close_timestamp=close_timestamp,
  #             close_price=close_price,
  #             profit_loss=net_pnl,
  #             commission=commission,
  #             close_reason='exchange_execution'
  #           )
  #
  #           # Обновляем статистику
  #           if hasattr(self, 'trading_system') and self.trading_system:
  #             if hasattr(self.trading_system, 'adaptive_selector'):
  #               await self.trading_system.adaptive_selector.update_strategy_performance(
  #                 trade.get('strategy_name', 'Unknown'),
  #                 net_pnl > 0,
  #                 abs(net_pnl)
  #               )
  #
  #           # Синхронизация с Shadow Trading
  #           if hasattr(self, 'trading_system') and self.trading_system:
  #             try:
  #               shadow_manager = getattr(self.trading_system, 'shadow_trading', None)
  #               if shadow_manager and hasattr(shadow_manager, 'signal_tracker'):
  #                 profit_pct = ((close_price - open_price) / open_price * 100) if side == 'BUY' \
  #                   else ((open_price - close_price) / open_price * 100)
  #
  #                 trade_result = {
  #                   'symbol': symbol,
  #                   'close_price': close_price,
  #                   'close_timestamp': close_timestamp,
  #                   'profit_loss': net_pnl,
  #                   'profit_pct': profit_pct,
  #                   'order_id': trade.get('order_id')
  #                 }
  #
  #                 await shadow_manager.signal_tracker.sync_with_real_trades(symbol, trade_result)
  #                 logger.info(f"✅ Shadow Trading синхронизирован для {symbol}")
  #
  #             except Exception as e:
  #               logger.error(f"Ошибка синхронизации Shadow Trading: {e}")
  #     except Exception as e:
  #      logger.error(f"Ошибка при сверке сделки {symbol}: {e}")

  # def add_position_to_cache(self, trade: Dict):
  #   """Добавляет информацию о новой сделке в кэш открытых позиций."""
  #   if 'symbol' in trade:
  #     symbol = trade['symbol']
  #     self.open_positions[symbol] = trade
  #     logger.info(f"Новая позиция по {symbol} добавлена в кэш PositionManager.")
  #   else:
  #     logger.error("Попытка добавить в кэш сделку без ключа 'symbol'.")
  async def _check_reversal_exit(self, position: Dict, data: pd.DataFrame) -> Optional[TradingSignal]:
    """
    Улучшенная версия: проверяет необходимость разворота позиции,
    используя стратегии и подтверждение от нескольких индикаторов.
    """
    try:
      symbol = position['symbol']
      current_side = position.get('side')

      # 1. Получаем сигналы от стратегий через integrated_system
      if hasattr(self, 'trading_system') and self.trading_system:
        # Проверяем SAR стратегию
        if hasattr(self.trading_system, 'sar_strategy') and self.trading_system.sar_strategy:
          sar_signal = await self.trading_system.sar_strategy.check_exit_conditions(
            symbol, data, position
          )
          if sar_signal and sar_signal.is_reversal:
            logger.info(f"SAR стратегия обнаружила сигнал разворота для {symbol}")

            # Подтверждаем другими индикаторами
            confirmations = await self._confirm_reversal_signal(
              symbol, data, current_side, sar_signal
            )

            if confirmations >= 2:  # Минимум 2 подтверждения
              logger.info(f"Разворот подтвержден {confirmations} индикаторами")
              return sar_signal

      # 2. Альтернативная проверка через базовые индикаторы
      # Если стратегии недоступны или не дали сигнала
      reversal_conditions = 0

      # EMA crossover
      if 'ema_12' in data.columns and 'ema_26' in data.columns:
        ema_fast = data['ema_12'].iloc[-1]
        ema_slow = data['ema_26'].iloc[-1]
        prev_ema_fast = data['ema_12'].iloc[-2]
        prev_ema_slow = data['ema_26'].iloc[-2]

        if current_side == 'BUY':
          # Медвежий кроссовер для выхода из лонга
          if prev_ema_fast > prev_ema_slow and ema_fast < ema_slow:
            reversal_conditions += 1
        else:
          # Бычий кроссовер для выхода из шорта
          if prev_ema_fast < prev_ema_slow and ema_fast > ema_slow:
            reversal_conditions += 1

      # RSI экстремумы
      if 'rsi' in data.columns:
        rsi = data['rsi'].iloc[-1]
        if current_side == 'BUY' and rsi > 75:  # Перекупленность
          reversal_conditions += 1
        elif current_side == 'SELL' and rsi < 25:  # Перепроданность
          reversal_conditions += 1

      # MACD дивергенция
      if all(col in data.columns for col in ['macd', 'macd_signal']):
        macd = data['macd'].iloc[-1]
        signal = data['macd_signal'].iloc[-1]
        prev_macd = data['macd'].iloc[-2]
        prev_signal = data['macd_signal'].iloc[-2]

        if current_side == 'BUY':
          if prev_macd > prev_signal and macd < signal:
            reversal_conditions += 1
        else:
          if prev_macd < prev_signal and macd > signal:
            reversal_conditions += 1

      # Если достаточно условий для разворота
      if reversal_conditions >= 2:
        # Создаем сигнал разворота
        new_signal_type = SignalType.SELL if current_side == 'BUY' else SignalType.BUY

        reversal_signal = TradingSignal(
          symbol=symbol,
          signal_type=new_signal_type,
          price=data['close'].iloc[-1],
          confidence=0.6 + (reversal_conditions * 0.1),  # Базовая + бонус за подтверждения
          strategy="reversal_exit",
          timeframe=position.get('timeframe', '1h'),
          volume=data['volume'].iloc[-1] if 'volume' in data.columns else 0,
          timestamp=datetime.now(),
          metadata={
            'reversal_conditions': reversal_conditions,
            'original_side': current_side,
            'position_pnl_pct': self._calculate_current_pnl(position, data['close'].iloc[-1])
          }
        )

        return reversal_signal

      return None

    except Exception as e:
      logger.error(f"Ошибка при проверке разворота для {position['symbol']}: {e}")
      return None

  async def _confirm_reversal_signal(self, symbol: str, data: pd.DataFrame,
                               current_side: str, signal: TradingSignal) -> int:
    """
    Подтверждает сигнал разворота дополнительными индикаторами.
    Возвращает количество подтверждений.
    """
    confirmations = 0

    try:
      # 1. Проверка объемов
      if 'volume' in data.columns and len(data) > 20:
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        if current_volume > avg_volume * 1.5:  # Повышенный объем
          confirmations += 1

      # 2. Проверка волатильности (ATR)
      if 'atr' in data.columns:
        atr = data['atr'].iloc[-1]
        price = data['close'].iloc[-1]
        atr_pct = (atr / price) * 100
        if atr_pct > 1.5:  # Высокая волатильность
          confirmations += 1

      # 3. Проверка моментума
      if len(data) > 10:
        momentum = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10] * 100
        if (current_side == 'BUY' and momentum < -2) or \
            (current_side == 'SELL' and momentum > 2):
          confirmations += 1

      # 4. Проверка структуры рынка (Higher High/Lower Low)
      if self._check_market_structure_break(data, current_side):
        confirmations += 1

      logger.debug(f"Разворот {symbol}: получено {confirmations} подтверждений")
      return confirmations

    except Exception as e:
      logger.error(f"Ошибка при подтверждении разворота: {e}")
      return 0

  def _check_market_structure_break(self, data: pd.DataFrame, current_side: str) -> bool:
    """
    Проверяет нарушение структуры рынка (пробой последнего high/low).
    """
    try:
      if len(data) < 20:
        return False

      # Находим последние локальные экстремумы
      highs = data['high'].rolling(5).max()
      lows = data['low'].rolling(5).min()

      current_price = data['close'].iloc[-1]

      if current_side == 'BUY':
        # Для лонга: проверяем пробой последнего значимого low
        recent_low = lows.iloc[-10:-1].min()
        return current_price < recent_low
      else:
        # Для шорта: проверяем пробой последнего значимого high
        recent_high = highs.iloc[-10:-1].max()
        return current_price > recent_high

    except Exception:
      return False

  def _calculate_current_pnl(self, position: Dict, current_price: float) -> float:
    """
    Рассчитывает текущий PnL позиции в процентах.
    """
    try:
      open_price = float(position.get('open_price', 0))
      if open_price == 0:
        return 0.0

      side = position.get('side')
      if side == 'BUY':
        return ((current_price - open_price) / open_price) * 100
      else:
        return ((open_price - current_price) / open_price) * 100
    except Exception:
      return 0.0

  def add_position_to_cache(self, trade: Dict):
    """Добавляет информацию о новой сделке в кэш открытых позиций."""
    if trade and 'symbol' in trade:
      symbol = trade['symbol']
      self.open_positions[symbol] = trade
      logger.info(f"Новая позиция по {symbol} добавлена в кэш PositionManager.")
    else:
      logger.error(f"Попытка добавить в кэш невалидную сделку: {trade}")

  def _check_psar_exit(self, position: Dict, data: pd.DataFrame,
                       timeframes_data: Dict[str, pd.DataFrame] = None) -> Optional[str]:
    """
    Проверка выхода по PSAR с мультитаймфреймовым анализом и Aroon подтверждением
    """
    if 'psar' not in data.columns or data['psar'].isnull().all():
      return None

    side = position.get('side')
    current_price = data['close'].iloc[-1]
    open_price = float(position.get('open_price', 0))

    if open_price == 0:
      return None

    # Базовая проверка прибыльности
    commission_rate = 0.00075
    total_commission_rate = commission_rate * 4
    min_profit_buffer = 0.001
    total_required_move = total_commission_rate + min_profit_buffer

    # Мультитаймфреймовый анализ PSAR
    if timeframes_data:
      psar_confirmations = 0
      checked_psar_timeframes = 0
      aroon_confirmation = False

      # Проверяем PSAR на разных таймфреймах
      for tf_name in ['1m', '5m', '15m', '1h']:
        if tf_name not in timeframes_data:
          continue

        tf_data = timeframes_data[tf_name]
        if 'psar' not in tf_data.columns or tf_data['psar'].isnull().all():
          continue

        checked_psar_timeframes += 1
        tf_price = tf_data['close'].iloc[-1]
        tf_psar = tf_data['psar'].iloc[-1]

        # Проверяем сигнал PSAR
        if side == 'BUY' and tf_price < tf_psar:
          psar_confirmations += 1
        elif side == 'SELL' and tf_price > tf_psar:
          psar_confirmations += 1

        # Проверяем Aroon на 5-минутном таймфрейме
        if tf_name == '5m' and 'aroon_osc' in tf_data.columns:
          aroon_osc = tf_data['aroon_osc'].iloc[-1]

          # Для выхода из лонга Aroon должен быть отрицательным
          if side == 'BUY' and aroon_osc < -20:
            aroon_confirmation = True
          # Для выхода из шорта Aroon должен быть положительным
          elif side == 'SELL' and aroon_osc > 20:
            aroon_confirmation = True

      # Проверяем условия выхода
      if checked_psar_timeframes >= 2 and psar_confirmations >= 2 and aroon_confirmation:
        # Проверяем прибыльность
        if side == 'BUY':
          actual_profit_pct = ((current_price - open_price) / open_price)
          if actual_profit_pct > total_required_move:
            net_profit_pct = (actual_profit_pct - total_commission_rate) * 100
            return (f"Мультитаймфреймовый PSAR выход с Aroon подтверждением "
                    f"({psar_confirmations}/{checked_psar_timeframes} PSAR, Aroon OK, прибыль: {net_profit_pct:.3f}%)")

        elif side == 'SELL':
          actual_profit_pct = ((open_price - current_price) / open_price)
          if actual_profit_pct > total_required_move:
            net_profit_pct = (actual_profit_pct - total_commission_rate) * 100
            return (f"Мультитаймфреймовый PSAR выход с Aroon подтверждением "
                    f"({psar_confirmations}/{checked_psar_timeframes} PSAR, Aroon OK, прибыль: {net_profit_pct:.3f}%)")

    # Fallback на стандартную проверку если мультитаймфреймовый анализ не сработал
    return self._check_psar_exit_single_tf(position, data)

  def _check_psar_exit_single_tf(self, position: Dict, data: pd.DataFrame) -> Optional[str]:
    """
    Проверяет, нужно ли выходить из сделки по сигналу Parabolic SAR,
    с ПРАВИЛЬНОЙ проверкой на безубыточность включая ВСЕ комиссии.
    """
    if 'psar' not in data.columns or data['psar'].isnull().all():
      return None

    side = position.get('side')
    current_price = data['close'].iloc[-1]
    psar_value = data['psar'].iloc[-1]
    open_price = float(position.get('open_price', 0))

    if open_price == 0:
      return None

    # Определяем, есть ли сигнал на выход по PSAR
    is_psar_exit_signal = False
    if side == 'BUY' and current_price < psar_value:
      is_psar_exit_signal = True
    elif side == 'SELL' and current_price > psar_value:
      is_psar_exit_signal = True

    if not is_psar_exit_signal:
      return None

    # --- ИСПРАВЛЕННАЯ ПРОВЕРКА НА БЕЗУБЫТОЧНОСТЬ ---
    # Комиссии: открытие + закрытие
    commission_rate = 0.0009  # Taker fee 0.075%
    total_commission_rate = commission_rate * 4  # За вход и выход

    # Добавляем небольшой буфер для гарантии прибыльности
    min_profit_buffer = 0.001  # 0.1% дополнительно
    total_required_move = total_commission_rate + min_profit_buffer  # ~0.25%

    # Расчет фактической прибыли с учетом направления
    if side == 'BUY':
      # Для лонга: (текущая - открытие) / открытие
      actual_profit_pct = ((current_price - open_price) / open_price)
      is_profitable = actual_profit_pct > total_required_move

      if is_profitable:
        net_profit_pct = (actual_profit_pct - total_commission_rate) * 100
        logger.info(
          f"✅ Выход по PSAR для BUY ({position['symbol']}) подтвержден. "
          f"Чистая прибыль: {net_profit_pct:.3f}%"
        )
        return f"Parabolic SAR для BUY: цена {current_price:.4f} < PSAR {psar_value:.4f}"

    elif side == 'SELL':
      # Для шорта: (открытие - текущая) / открытие
      actual_profit_pct = ((open_price - current_price) / open_price)
      is_profitable = actual_profit_pct > total_required_move

      if is_profitable:
        net_profit_pct = (actual_profit_pct - total_commission_rate) * 100
        logger.info(
          f"✅ Выход по PSAR для SELL ({position['symbol']}) подтвержден. "
          f"Чистая прибыль: {net_profit_pct:.3f}%"
        )
        return f"Parabolic SAR для SELL: цена {current_price:.4f} > PSAR {psar_value:.4f}"

    # Если сигнал есть, но выход приведет к убытку
    logger.debug(
      f"❌ PSAR сигнал для {position['symbol']} отклонен - недостаточная прибыль. "
      f"Требуется минимум {total_required_move * 100:.3f}% движения"
    )
    return None

  def _check_atr_trailing_stop(self, position: Dict, data: pd.DataFrame,
                               timeframes_data: Dict[str, pd.DataFrame] = None) -> Optional[str]:
    """
    Улучшенная версия трейлинг-стопа на основе ATR с мультитаймфреймовым анализом
    """
    strategy_settings = self.config.get('strategy_settings', {})
    if not strategy_settings.get('use_atr_trailing_stop', True):
      return None

    # Проверяем ATR на основном таймфрейме (1H)
    if 'atr' not in data.columns or data['atr'].isnull().all():
      logger.warning(f"ATR не найден в данных для {position['symbol']}")
      return None

    side = position.get('side')
    current_price = data['close'].iloc[-1]
    open_price = float(position.get('open_price', 0))

    if open_price == 0:
      return None

    # Параметры для расчета
    atr_multiplier = strategy_settings.get('atr_ts_multiplier', 2.5)
    commission_rate = 0.0009
    min_profit_buffer = 0.05

    # Мультитаймфреймовый анализ если доступен
    if timeframes_data:
      confirmations = 0
      checked_timeframes = 0

      # Проверяем ATR trailing на разных таймфреймах
      for tf_name in ['1m', '5m', '15m', '1h']:
        if tf_name not in timeframes_data:
          continue

        tf_data = timeframes_data[tf_name]
        if 'atr' not in tf_data.columns or tf_data['atr'].isnull().all():
          continue

        checked_timeframes += 1
        tf_atr = tf_data['atr'].iloc[-1]
        tf_price = tf_data['close'].iloc[-1]

        # Chandelier Exit для каждого таймфрейма
        lookback = min(20, len(tf_data))
        recent_data = tf_data.tail(lookback)

        if side == 'BUY':
          highest_high = recent_data['high'].max()
          chandelier_stop = highest_high - (tf_atr * atr_multiplier)
          minimum_stop = open_price * (1 + min_profit_buffer)
          effective_stop = max(chandelier_stop, minimum_stop)

          if tf_price < effective_stop:
            confirmations += 1

        elif side == 'SELL':
          lowest_low = recent_data['low'].min()
          chandelier_stop = lowest_low + (tf_atr * atr_multiplier)
          minimum_stop = open_price * (1 - min_profit_buffer)
          effective_stop = min(chandelier_stop, minimum_stop)

          if tf_price > effective_stop:
            confirmations += 1

      # Требуем подтверждение минимум на 2 из 4 таймфреймов
      if checked_timeframes >= 4 and confirmations >= 3:
        profit_pct = ((current_price - open_price) / open_price * 100) if side == 'BUY' else (
              (open_price - current_price) / open_price * 100)
        return (f"Мультитаймфреймовый ATR trailing stop сработал "
                f"({confirmations}/{checked_timeframes} подтверждений, прибыль: {profit_pct:.2f}%)")

        logger.debug(f"ATR проверка для {position['symbol']}:")
        logger.debug(f"  - Цена входа: {open_price:.6f}")
        logger.debug(f"  - Текущая цена: {current_price:.6f}")
        logger.debug(f"  - Изменение: {((current_price - open_price) / open_price * 100):.2f}%")
        logger.debug(f"  - ATR: {tf_atr:.6f}")
        logger.debug(f"  - Chandelier Stop: {effective_stop:.6f}")
        logger.debug(f"  - Минимальный буфер прибыли: {min_profit_buffer * 100:.2f}%")

    # Если мультитаймфреймовый анализ не дал результата, используем стандартную проверку
    return self._check_atr_trailing_stop_single_tf(position, data)

  def _calculate_profit_protection_stop(self, position: Dict, current_price: float,
                                        highest_since_entry: float) -> Optional[float]:
    """
    Рассчитывает уровень защиты прибыли.
    Активируется только после достижения определенной прибыли.
    """
    open_price = float(position.get('open_price', 0))
    if open_price == 0:
      return None

    side = position.get('side')

    if side == 'BUY':
      # Проверяем максимальную прибыль с момента входа
      max_profit_pct = ((highest_since_entry - open_price) / open_price) * 100
      current_profit_pct = ((current_price - open_price) / open_price) * 100

      # Активируем защиту только если была прибыль >= 2%
      if max_profit_pct >= 2.0:
        # Защищаем 50% от максимальной прибыли
        protection_level = open_price + (highest_since_entry - open_price) * 0.5

        if current_price < protection_level and current_profit_pct > 0.5:
          return protection_level

    elif side == 'SELL':
      # Аналогично для шорта
      max_profit_pct = ((open_price - lowest_since_entry) / open_price) * 100
      current_profit_pct = ((open_price - current_price) / open_price) * 100

      if max_profit_pct >= 2.0:
        protection_level = open_price - (open_price - lowest_since_entry) * 0.5

        if current_price > protection_level and current_profit_pct > 0.5:
          return protection_level

    return None

  def _check_atr_trailing_stop_single_tf(self, position: Dict, data: pd.DataFrame) -> Optional[str]:
    """
    Улучшенная версия трейлинг-стопа на основе ATR с поддержкой Chandelier Exit.
    """
    strategy_settings = self.config.get('strategy_settings', {})
    if not strategy_settings.get('use_atr_trailing_stop', True):
      return None

    if 'atr' not in data.columns or data['atr'].isnull().all():
      logger.warning(f"ATR не найден в данных для {position['symbol']}")
      return None

    if 'atr' not in data.columns and len(data) >= 14:
      atr = ta.atr(data['high'], data['low'], data['close'], length=14)
      if atr is not None:
        data['atr'] = atr

    side = position.get('side')
    current_price = data['close'].iloc[-1]
    atr_value = data['atr'].iloc[-1]
    atr_multiplier = strategy_settings.get('atr_ts_multiplier', 2.5)
    open_price = float(position.get('open_price', 0))

    if open_price == 0:
      return None

    # Расчет комиссий и минимальной прибыли
    commission_rate = 0.00075  # Taker fee
    min_profit_buffer = (commission_rate * 3) * 2.5  # 3 комиссии с запасом 70%

    # Получаем максимальную/минимальную цену с момента входа
    # Это ключевое отличие Chandelier Exit от обычного ATR stop
    entry_index = position.get('entry_bar_index', -20)  # По умолчанию последние 20 баров

    if entry_index < 0:
      # Если не знаем точный индекс входа, используем последние N баров
      lookback_bars = min(abs(entry_index), len(data))
      recent_data = data.tail(lookback_bars)
    else:
      recent_data = data.iloc[entry_index:]

    # Chandelier Exit логика
    if side == 'BUY':
      # Для лонга: стоп следует за максимумом минус ATR
      highest_high = recent_data['high'].max()
      chandelier_stop = highest_high - (atr_value * atr_multiplier)

      # Дополнительная защита: стоп не может быть ниже цены входа + минимальная прибыль
      minimum_stop = open_price * (1 + min_profit_buffer)
      effective_stop = max(chandelier_stop, minimum_stop)

      # Проверяем, пробита ли цена
      if current_price < effective_stop:
        profit_pct = ((current_price - open_price) / open_price) * 100
        return (f"Chandelier Exit сработал для BUY: цена {current_price:.4f} < "
                f"Stop {effective_stop:.4f} (прибыль: {profit_pct:.2f}%)")

    elif side == 'SELL':
      # Для шорта: стоп следует за минимумом плюс ATR
      lowest_low = recent_data['low'].min()
      chandelier_stop = lowest_low + (atr_value * atr_multiplier)

      # Защита для шорта
      minimum_stop = open_price * (1 - min_profit_buffer)
      effective_stop = min(chandelier_stop, minimum_stop)

      # Проверяем пробитие
      if current_price > effective_stop:
        profit_pct = ((open_price - current_price) / open_price) * 100
        return (f"Chandelier Exit сработал для SELL: цена {current_price:.4f} > "
                f"Stop {effective_stop:.4f} (прибыль: {profit_pct:.2f}%)")

    return None

  def update_position_extremes(self, symbol: str, current_price: float, high: float, low: float):
    """
    Обновляет экстремумы для позиции (используется для Chandelier Exit).
    """
    if symbol not in self.open_positions:
      return

    position = self.open_positions[symbol]

    # Инициализируем если не существует
    if 'highest_since_entry' not in position:
      position['highest_since_entry'] = high
      position['lowest_since_entry'] = low
    else:
      # Обновляем экстремумы
      position['highest_since_entry'] = max(position['highest_since_entry'], high)
      position['lowest_since_entry'] = min(position['lowest_since_entry'], low)

  async def monitor_single_position(self, symbol: str) -> bool:
    """Мониторит одну конкретную позицию"""
    if symbol not in self.open_positions:
      return False

    try:
      # Проверяем условия выхода для конкретной позиции
      await self.manage_open_positions(symbol)
      return True
    except Exception as e:
      logger.error(f"Ошибка при мониторинге позиции {symbol}: {e}")
      return False

  async def track_pending_orders(self):
    """Отслеживает статус pending ордеров"""
    # Получаем все открытые сделки
    open_trades = await self.db_manager.get_all_open_trades()

    for trade in open_trades:
      if trade.get('order_id'):
        # Используем метод из trade_executor для проверки статуса
        await self.trade_executor.update_trade_status_from_exchange(
          trade['order_id'],
          trade['symbol']
        )

    logger.debug("Проверка статусов ордеров завершена")