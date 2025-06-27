import asyncio
from datetime import datetime

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
from utils.logging_config import get_logger
from core.signal_filter import SignalFilter

logger = get_logger(__name__)


class PositionManager:
  """
  Класс для управления открытыми позициями и реализации логики выхода.
  """

  def __init__(self, db_manager: AdvancedDatabaseManager, trade_executor: TradeExecutor, data_fetcher: DataFetcher, connector: BybitConnector, signal_filter: SignalFilter, risk_manager: AdvancedRiskManager):
    self.db_manager = db_manager
    self.trade_executor = trade_executor
    self.data_fetcher = data_fetcher
    self.connector = connector
    self.signal_filter = signal_filter
    self.open_positions: Dict[str, Dict] = {}  # Кэш открытых позиций для быстрой проверки
    self.risk_manager = risk_manager
    self.config_manager = ConfigManager()
    self.config = self.config_manager.load_config()
    # self.integrated_system: Optional[IntegratedTradingSystem] = None

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

    if (datetime.now() - self._last_order_check).seconds > 60:  # Каждую минуту
      await self.track_pending_orders()
      self._last_order_check = datetime.now()

    if not self.open_positions:
      return

    logger.debug(f"Динамическое управление для {len(self.open_positions)} открытых позиций...")

    for symbol, position_data in list(self.open_positions.items()):
      try:
        # --- ИСПРАВЛЕНИЕ: ЗАГРУЖАЕМ ДАННЫЕ ЗАРАНЕЕ ---
        # 1. Загружаем данные основного таймфрейма (1H) для анализа SL/TP и PSAR
        htf_data = await self.data_fetcher.get_historical_candles(symbol, Timeframe.ONE_HOUR, limit=100)
        if htf_data.empty:
          continue
        current_price = htf_data['close'].iloc[-1]

        # 2. Загружаем данные малого таймфрейма (1m) для анализа разворота
        strategy_settings = self.config.get('strategy_settings', {})
        ltf_str = strategy_settings.get('ltf_entry_timeframe', '15m')
        timeframe_map = {"1m": Timeframe.ONE_MINUTE, "5m": Timeframe.FIVE_MINUTES, "15m": Timeframe.FIFTEEN_MINUTES}
        ltf_timeframe = timeframe_map.get(ltf_str, Timeframe.ONE_MINUTE)
        ltf_data = await self.data_fetcher.get_historical_candles(symbol, ltf_timeframe, limit=100)
        # --- КОНЕЦ БЛОКА ЗАГРУЗКИ ---

        # --- ПРИОРИТЕТ 1: ЖЕСТКИЙ SL/TP ---
        exit_reason = self._check_sl_tp(position_data, current_price)
        if exit_reason:
          logger.info(f"ВЫХОД для {symbol}: Сработал стандартный SL/TP. Причина: {exit_reason}")
          await self.trade_executor.close_position(symbol=symbol)
          continue

        # --- ПРИОРИТЕТ 2: ТРЕЙЛИНГ-СТОП ПО ATR (на HTF)---
        if not exit_reason:
          # Убедимся, что ATR рассчитан для htf_data
          if 'atr' not in htf_data.columns:
            htf_data.ta.atr(append=True)
          exit_reason = self._check_atr_trailing_stop(position_data, htf_data)

        # --- ПРИОРИТЕТ 3: ТРЕЙЛИНГ-СТОП ПО PSAR ---
        if not exit_reason and strategy_settings.get('use_psar_exit', True):
          # Важно: для PSAR используем данные ОСНОВНОГО таймфрейма (1H)
          # Рассчитываем PSAR для htf_data
          psar_df = ta.psar(htf_data['high'], htf_data['low'], htf_data['close'])
          if psar_df is not None:
            psar_col = next(
              (col for col in psar_df.columns if 'PSAR' in col and 'PSARl' not in col and 'PSARs' not in col), None)
            if psar_col:
              htf_data['psar'] = psar_df[psar_col]
              exit_reason = self._check_psar_exit(position_data, htf_data)
              if exit_reason:
                logger.info(f"ВЫХОД для {symbol}: Сработал трейлинг-стоп. Причина: {exit_reason}")
                await self.trade_executor.close_position(symbol=symbol)
                continue

        # --- ПРИОРИТЕТ 4: STOP AND REVERSE ---
        # Теперь вызываем _check_reversal_exit, передавая ему LTF данные
        if not exit_reason:
          reverse_signal = await self._check_reversal_exit(position_data, ltf_data)
          if reverse_signal:
            logger.info(f"Инициирован Stop and Reverse для {symbol}.")
            close_success = await self.trade_executor.close_position(symbol=symbol)
            if not close_success: continue

            await asyncio.sleep(3)

            if account_balance:
              # Важно: передаем htf_data в risk_manager, так как он ожидает полный набор признаков
              risk_decision = await self.risk_manager.validate_signal(reverse_signal, symbol,
                                                                      account_balance.available_balance_usdt)
              if risk_decision.get('approved'):
                logger.info(f"SAR для {symbol}: Риск-менеджер одобрил новую позицию.")
                quantity = risk_decision.get('recommended_size')
                await self.trade_executor.execute_trade(reverse_signal, symbol, quantity)

        if exit_reason:
          logger.info(f"ВЫХОД для {symbol}: Обнаружен сигнал на закрытие. Причина: {exit_reason}")
          await self.trade_executor.close_position(symbol=symbol)
          continue

      except Exception as e:
        logger.error(f"Ошибка при динамическом управлении позицией {symbol}: {e}", exc_info=True)


  async def reconcile_filled_orders(self):
        """
        ФИНАЛЬНАЯ ВЕРСИЯ: Сверяет исполненные ордера, корректно рассчитывая PnL
        с учетом комиссий и обрабатывая закрытия по TP/SL.
        """
        # all_positions = await self.connector.fetch_positions_batch()
        # active_symbols = {pos['symbol'] for pos in all_positions if float(pos.get('size', 0)) > 0}

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
              logger.info(
                f"Позиция по {symbol} больше не активна на бирже. Поиск исполненной сделки для расчета PnL...")

              # Ищем в истории исполнений сделку, которая закрыла нашу позицию
              executions = await self.connector.get_execution_history(symbol=symbol, limit=20)
              closing_exec = None
              for exec_trade in executions:
                # Ищем сделку, которая закрыла позицию (closedSize > 0)
                # и соответствует нашему ID ордера на открытие, если он есть
                if exec_trade.get('closedSize') and float(exec_trade.get('closedSize', 0)) > 0:
                  closing_exec = exec_trade
                  break

              if closing_exec:
                # --- КОРРЕКТНЫЙ РАСЧЕТ PNL ---
                open_price = float(trade['open_price'])
                close_price = float(closing_exec['execPrice'])
                quantity = float(trade['quantity'])
                # Важно: берем комиссию из данных об исполнении!
                commission = float(closing_exec.get('execFee', 0))
                side = trade.get('side')

                # Считаем "грязный" PnL
                gross_pnl = (close_price - open_price) * quantity if side == 'BUY' else (
                                                                                              open_price - close_price) * quantity

                # Вычитаем комиссию за ЗАКРЫТИЕ
                net_pnl = gross_pnl - commission
                # Примечание: комиссию за открытие мы здесь не учитываем, т.к. ее нет в данных о закрытии.
                # Для 100% точности ее нужно было бы хранить в БД. Но это уже 99% точности.

                close_timestamp = datetime.fromtimestamp(int(closing_exec['execTime']) / 1000)

                logger.info(
                  f"ПОДТВЕРЖДЕНИЕ ЗАКРЫТИЯ для {symbol}: Найдена исполненная сделка. Чистый PnL: {net_pnl:.4f}")

                await self.db_manager.update_trade_as_closed(
                  trade_id=trade['id'], close_price=close_price, pnl=net_pnl,
                  commission=commission, close_timestamp=close_timestamp
                )

                integrated_system = getattr(self.trade_executor, 'integrated_system', None)
                if integrated_system and hasattr(integrated_system, 'process_trade_feedback'):
                  try:
                    trade_result = {
                      'strategy_name': trade.get('strategy_name', 'Unknown'),
                      'profit_loss': net_pnl,
                      'entry_price': open_price,
                      'exit_price': close_price,
                      'regime': trade.get('metadata', {}).get('regime', 'unknown') if isinstance(trade.get('metadata'),
                                                                                                 dict) else 'unknown',
                      'confidence': trade.get('confidence', 0.5),
                      'entry_features': trade.get('metadata', {}).get('features', {}) if isinstance(
                        trade.get('metadata'), dict) else {}
                    }

                    await integrated_system.process_trade_feedback(symbol, trade['id'], trade_result)
                    logger.info(f"Обратная связь отправлена для {symbol}")
                  except Exception as e:
                    logger.error(f"Ошибка отправки обратной связи: {e}")

                # Уведомляем SAR стратегию об обновлении позиции для адаптивного обучения
                if (hasattr(self, 'integrated_system') and self.integrated_system and
                    hasattr(self.integrated_system, 'sar_strategy') and
                    self.integrated_system.sar_strategy):

                  try:
                    # Проверяем, что позиция действительно принадлежит SAR стратегии
                    if symbol in getattr(self.integrated_system.sar_strategy, 'current_positions', {}):
                      await self.integrated_system.sar_strategy.handle_position_update(symbol, {
                        'profit_loss': net_pnl,
                        'close_price': close_price,
                        'close_timestamp': close_timestamp,
                        'close_reason': locals().get('close_reason', 'position_manager'),
                        'open_price': locals().get('open_price', trade.get('open_price', 0)),
                        'net_pnl': net_pnl,
                        # 'close_reason': locals().get('close_reason', 'position_manager_close'),
                        # 'open_price': position_data.get('open_price', 0),
                      })
                      logger.debug(f"SAR стратегия уведомлена об обновлении позиции {symbol}")

                    # Также обновляем общую производительность стратегии
                    if hasattr(self.integrated_system, 'adaptive_selector'):
                      await self.integrated_system.adaptive_selector.update_strategy_performance(
                        'SAR_Strategy',
                        {
                          'profit_loss': net_pnl,
                          'symbol': symbol,
                          'close_timestamp': close_timestamp,
                          'regime': getattr(self.integrated_system.sar_strategy, 'current_market_regime', 'unknown')
                        }
                      )
                  except Exception as e:
                    logger.error(f"Ошибка при уведомлении SAR стратегии об обновлении позиции {symbol}: {e}")

                # Удаляем из кэша, если она там была
                if symbol in self.open_positions:
                  del self.open_positions[symbol]
              else:
                # Если не нашли исполненной сделки, используем старый метод "принудительного закрытия"
                logger.warning(f"Не удалось найти исполненную сделку для {symbol}. Принудительное закрытие с PnL=0.")
                await self.db_manager.force_close_trade(trade_id=trade['id'], close_price=trade['open_price'])

          except Exception as e:
            logger.error(f"Ошибка при сверке сделок для {symbol}: {e}", exc_info=True)




  # def add_position_to_cache(self, trade: Dict):
  #   """Добавляет информацию о новой сделке в кэш открытых позиций."""
  #   if 'symbol' in trade:
  #     symbol = trade['symbol']
  #     self.open_positions[symbol] = trade
  #     logger.info(f"Новая позиция по {symbol} добавлена в кэш PositionManager.")
  #   else:
  #     logger.error("Попытка добавить в кэш сделку без ключа 'symbol'.")
  async def _check_reversal_exit(self, position: Dict, ltf_data: pd.DataFrame) -> Optional[TradingSignal]:
    """
    Проверяет сигнал на разворот и возвращает ГОТОВЫЙ СИГНАЛ для новой сделки, если разворот оправдан.
    """
    current_price = ltf_data['close'].iloc[-1]
    open_price = float(position.get('open_price', 0))
    if open_price == 0: return None

    pnl_multiplier = 1 if position.get('side') == 'BUY' else -1
    unrealized_pnl_pct = ((current_price - open_price) / open_price) * pnl_multiplier * 100  # В процентах

    # ИЗМЕНИТЬ: Снизить порог для обнаружения разворота
    if unrealized_pnl_pct <= -1.0:  # Если убыток больше 1%
      logger.info(f"Позиция {position['symbol']} в убытке {unrealized_pnl_pct:.2f}%, проверяем разворот")
    else:
      return None

    opposite_signal_type = SignalType.SELL if position.get('side') == 'BUY' else SignalType.BUY

    # ДОБАВИТЬ: Дополнительные проверки для подтверждения разворота
    # Проверяем EMA кроссовер
    if 'ema_12' not in ltf_data.columns:
      ltf_data['ema_12'] = ta.ema(ltf_data['close'], length=12)
      ltf_data['ema_26'] = ta.ema(ltf_data['close'], length=26)

    ema_12 = ltf_data['ema_12'].iloc[-1]
    ema_26 = ltf_data['ema_26'].iloc[-1]

    # Подтверждаем разворот через EMA
    if position.get('side') == 'BUY' and ema_12 < ema_26:
      logger.info(f"EMA подтверждает разворот для SELL на {position['symbol']}")
    elif position.get('side') == 'SELL' and ema_12 > ema_26:
      logger.info(f"EMA подтверждает разворот для BUY на {position['symbol']}")
    else:
      return None

    reverse_signal_candidate = TradingSignal(
      signal_type=opposite_signal_type,
      symbol=position['symbol'],
      price=current_price,
      confidence=0.85,  # Снизить с 0.99
      strategy_name="ReversalSAR",
      timestamp=datetime.now()
    )

    is_strong_reverse, _ = await self.signal_filter.filter_signal(reverse_signal_candidate, ltf_data)
    if not is_strong_reverse:
      return None

    commission_rate = 0.00075
    safety_buffer_pct = commission_rate * 3  # Упростить расчет

    # ИЗМЕНИТЬ условие - разрешаем разворот даже при небольшом убытке
    if unrealized_pnl_pct > -10.0:  # Не разворачиваем при убытке больше 10%
      logger.info(f"Обнаружен разворот для {position['symbol']}. PnL: {unrealized_pnl_pct:.2f}%")
      return reverse_signal_candidate
    else:
      logger.info(f"Разворот для {position['symbol']} отклонен - слишком большой убыток: {unrealized_pnl_pct:.2f}%")
      return None
  # async def _check_reversal_exit(self, position: Dict, ltf_data: pd.DataFrame) -> Optional[TradingSignal]:
  #   """
  #   Проверяет сигнал на разворот и возвращает ГОТОВЫЙ СИГНАЛ для новой сделки, если разворот оправдан.
  #   """
  #   current_price = ltf_data['close'].iloc[-1]
  #   open_price = float(position.get('open_price', 0))
  #   if open_price == 0: return None
  #
  #   pnl_multiplier = 1 if position.get('side') == 'BUY' else -1
  #   unrealized_pnl_pct = ((current_price - open_price) / open_price) * pnl_multiplier
  #
  #   if unrealized_pnl_pct <= 0.005: return None
  #
  #   opposite_signal_type = SignalType.SELL if position.get('side') == 'BUY' else SignalType.BUY
  #   reverse_signal_candidate = TradingSignal(
  #     signal_type=opposite_signal_type, symbol=position['symbol'], price=current_price,
  #     confidence=0.99, strategy_name="ReversalSAR", timestamp=datetime.now()
  #   )
  #
  #   is_strong_reverse, _ = await self.signal_filter.filter_signal(reverse_signal_candidate, ltf_data)
  #   if not is_strong_reverse: return None
  #
  #   commission_rate = 0.00075
  #   safety_buffer_pct = (commission_rate * 3) * 1.5
  #
  #   if unrealized_pnl_pct > safety_buffer_pct:
  #     logger.info(f"Обнаружен экономически выгодный разворот для {position['symbol']}.")
  #     return reverse_signal_candidate
  #   else:
  #     logger.info(f"Разворот для {position['symbol']} обнаружен, но прибыль недостаточна.")
  #     return None

  def add_position_to_cache(self, trade: Dict):
    """Добавляет информацию о новой сделке в кэш открытых позиций."""
    if trade and 'symbol' in trade:
      symbol = trade['symbol']
      self.open_positions[symbol] = trade
      logger.info(f"Новая позиция по {symbol} добавлена в кэш PositionManager.")
    else:
      logger.error(f"Попытка добавить в кэш невалидную сделку: {trade}")

  def _check_psar_exit(self, position: Dict, data: pd.DataFrame) -> Optional[str]:
    """
    Проверяет, нужно ли выходить из сделки по сигналу Parabolic SAR,
    с обязательной проверкой на безубыточность.
    """
    if 'psar' not in data.columns or data['psar'].isnull().all():
      return None

    side = position.get('side')
    current_price = data['close'].iloc[-1]
    psar_value = data['psar'].iloc[-1]
    open_price = float(position.get('open_price', 0))

    if open_price == 0:
      return None  # Не можем проверить, если цена входа неизвестна

    # Определяем, есть ли сигнал на выход по PSAR
    is_psar_exit_signal = False
    if side == 'BUY' and current_price < psar_value:
      is_psar_exit_signal = True
    elif side == 'SELL' and current_price > psar_value:
      is_psar_exit_signal = True

    # Если сигнала на выход нет, прекращаем проверку
    if not is_psar_exit_signal:
      return None

    # --- НОВЫЙ БЛОК: ПРОВЕРКА НА БЕЗУБЫТОЧНОСТЬ ---
    # Если сигнал на выход есть, сначала проверяем, выгодно ли это

    # Минимальная прибыль для закрытия должна покрывать хотя бы комиссию за закрытие
    commission_rate = 0.00075  # Средняя комиссия тейкера ~0.075%

    # Для LONG позиции: выходим, только если цена закрытия выше цены входа + комиссия
    if side == 'BUY' and current_price > open_price * (1 + commission_rate):
      logger.info(f"Выход по PSAR для BUY ({position['symbol']}) подтвержден как безубыточный.")
      return f"Parabolic SAR для BUY сработал: цена {current_price:.4f} < PSAR {psar_value:.4f}"

    # Для SHORT позиции: выходим, только если цена закрытия ниже цены входа - комиссия
    elif side == 'SELL' and current_price < open_price * (1 - commission_rate):
      logger.info(f"Выход по PSAR для SELL ({position['symbol']}) подтвержден как безубыточный.")
      return f"Parabolic SAR для SELL сработал: цена {current_price:.4f} > PSAR {psar_value:.4f}"

    else:
      # Если сигнал PSAR есть, но закрытие приведет к убытку, мы его игнорируем
      logger.debug(
        f"Сигнал на выход по PSAR для {position['symbol']} проигнорирован, т.к. сделка не является безубыточной.")
      return None
    # --- КОНЕЦ НОВОГО БЛОКА ---

  def _check_atr_trailing_stop(self, position: Dict, data: pd.DataFrame) -> Optional[str]:
    """Проверяет, нужно ли выходить из сделки по трейлинг-стопу на основе ATR."""
    strategy_settings = self.config.get('strategy_settings', {})
    if not strategy_settings.get('use_atr_trailing_stop', True):
      return None

    if 'atr' not in data.columns or data['atr'].isnull().all():
      logger.warning(f"ATR не найден в данных для {position['symbol']}, проверка трейлинг-стопа пропущена.")
      return None

    side = position.get('side')
    current_price = data['close'].iloc[-1]
    atr_value = data['atr'].iloc[-1]
    atr_multiplier = strategy_settings.get('atr_ts_multiplier', 2.5)

    # --- ПРОВЕРКА НА БЕЗУБЫТОЧНОСТЬ (по вашему запросу) ---
    open_price = float(position.get('open_price', 0))
    if open_price == 0: return None

    # Минимальная прибыль для закрытия должна покрывать 3 комиссии с запасом 70%
    commission_rate = 0.00075
    breakeven_buffer = (commission_rate * 3) * 1.7

    # Для LONG позиции
    if side == 'BUY':
      trailing_stop_price = current_price - (atr_value * atr_multiplier)
      # Мы выходим, только если цена пробила наш трейлинг-стоп И мы все еще в плюсе с учетом буфера
      if trailing_stop_price > current_price > open_price * (1 + breakeven_buffer):
        return f"Трейлинг-стоп по ATR для BUY сработал: цена {current_price:.4f} < Stop {trailing_stop_price:.4f}"

    # Для SHORT позиции
    elif side == 'SELL':
      trailing_stop_price = current_price + (atr_value * atr_multiplier)
      # Аналогичная проверка для шорта
      if trailing_stop_price < current_price < open_price * (1 - breakeven_buffer):
        return f"Трейлинг-стоп по ATR для SELL сработал: цена {current_price:.4f} > Stop {trailing_stop_price:.4f}"

    return None

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