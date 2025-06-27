import asyncio
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, Tuple, Optional

import ccxt

import config
from core.bybit_connector import BybitConnector
from core.enums import Timeframe, SignalType
# from core.integrated_system import IntegratedTradingSystem
from core.schemas import TradingSignal, GridSignal
from data.database_manager import AdvancedDatabaseManager

from utils.logging_config import setup_logging, get_logger
# from config.trading_params import DEFAULT_LEVERAGE  # Глобальное плечо по умолчанию
from core.data_fetcher import DataFetcher
import logging
signal_logger = logging.getLogger('SignalTrace')
logger = get_logger(__name__)


class TradeExecutor:
  def __init__(self, connector: BybitConnector, db_manager: AdvancedDatabaseManager, data_fetcher: DataFetcher,settings: Dict[str, Any],risk_manager=None ):
    """

    """
    self.connector = connector
    self.db_manager = db_manager
    self.risk_manager = risk_manager
    # self.telegram_bot = telegram_bot
    self.data_fetcher = data_fetcher
    self.config = settings
    # self.trading_system = IntegratedTradingSystem(db_manager=db_manager)
    self.pending_orders = {}
    self.shadow_trading = None

    self.execution_stats = {
      'orders_placed': 0,
      'orders_filled': 0,
      'orders_failed': 0,
      'total_slippage': 0.0
    }

  async def _get_roi_details(self, symbol: str, signal: TradingSignal) -> Optional[Dict]:
    """Получает детали ROI для сигнала"""
    try:
      # ИСПРАВЛЕНИЕ: Проверяем наличие risk_manager
      if not self.risk_manager:
        logger.warning(f"Risk manager не настроен для получения ROI деталей {symbol}")
        return None

      # Получаем детали через risk_manager
      roi_details = await self.risk_manager.calculate_roi_details(symbol, signal)
      return roi_details

    except Exception as e:
      logger.error(f"Ошибка получения ROI деталей для {symbol}: {e}")
      return None

  async def execute_trade(self, signal: TradingSignal, symbol: str, quantity: float) -> Tuple[bool, Optional[Dict]]:
    """
    РЕАЛЬНАЯ ВЕРСИЯ: Исполняет торговый сигнал, отправляя ордер на биржу.
    """
    logger.info(
      f"ИСПОЛНИТЕЛЬ для {symbol}: Получена команда на реальное исполнение. Сигнал: {signal.signal_type.value}, Кол-во: {quantity:.5f}")
    logger.info(
        f"Стратегия: {signal.strategy_name}")

    try:
      # 1. Получаем настройки торговли из сохраненного конфига
      trade_settings = self.config.get('trade_settings', {})
      leverage = trade_settings.get('leverage', 10)

      balance_data = await self.connector.get_account_balance(account_type="UNIFIED", coin="USDT")
      if balance_data and 'coin' in balance_data and balance_data['coin']:
        available_balance = float(balance_data.get('totalAvailableBalance', 0))
        leverage = self.config.get('trade_settings', {}).get('leverage', 10)
        cost_of_trade = (signal.price * quantity) / leverage

        if cost_of_trade > available_balance:
          logger.error(
            f"ФИНАЛЬНАЯ ПРОВЕРКА: Недостаточно средств для {symbol}. Требуется: {cost_of_trade:.2f}, доступно: {available_balance:.2f}")
          signal_logger.error("ИСПОЛНИТЕЛЬ: ОТКЛОНЕНО. Недостаточно средств.")
          return False, None
      else:
        logger.warning("Не удалось выполнить финальную проверку баланса. Продолжаем с осторожностью.")


      # 1. Формируем параметры для ордера
      params = {
        'symbol': symbol,
        'side': 'Buy' if signal.signal_type == SignalType.BUY else 'Sell',
        'orderType': 'Market',
        'qty': str(quantity),
        'positionIdx': 0
      }

      # Bybit API требует, чтобы SL/TP были строками
      if signal.stop_loss and signal.stop_loss != 0:
        params['stopLoss'] = str(abs(signal.stop_loss))
      if signal.take_profit and signal.take_profit != 0:
        params['takeProfit'] = str(abs(signal.take_profit))

      # leverage = self.config.get('trade_settings', {}).get('leverage', 10)

      try:
        roi_info = self.risk_manager.convert_roi_to_price_targets(
          entry_price=signal.price,
          signal_type=signal.signal_type
        )

        logger.info(f"ROI ДЕТАЛИ СДЕЛКИ для {symbol}:")
        logger.info(f"  Цена входа: {signal.price:.6f}")
        logger.info(f"  SL: {signal.stop_loss:.6f} (ROI: {roi_info['stop_loss']['roi_pct']:.1f}%)")
        logger.info(f"  TP: {signal.take_profit:.6f} (ROI: {roi_info['take_profit']['roi_pct']:.1f}%)")
        logger.info(f"  Потенциальная потеря: ${roi_info['stop_loss']['distance_abs'] * quantity:.2f}")
        logger.info(f"  Потенциальная прибыль: ${roi_info['take_profit']['distance_abs'] * quantity:.2f}")

      except Exception as roi_error:
        logger.warning(f"Не удалось получить ROI детали для {symbol}: {roi_error}")

      # 2. Отправляем ордер на биржу
      logger.info(f"Отправка ордера на открытие: {params}")
      if not hasattr(signal, 'strategy_name') or not signal.strategy_name:
        signal.strategy_name = 'Unknown_new'

      order_response = await self.connector.place_order(**params)

      # 3. Обрабатываем ответ
      if order_response and order_response.get('orderId'):
        order_id = order_response.get('orderId')
        logger.info(f"✅ Ордер на открытие {symbol} успешно принят биржей. OrderID: {order_id}")

        # Убеждаемся, что strategy_name записан в метаданные
        if not signal.metadata:
          signal.metadata = {}
        signal.metadata['strategy_name'] = signal.strategy_name

        # Теперь этот метод будет возвращать созданную запись из БД
        trade_details = await self.db_manager.add_trade_with_signal(
          signal=signal,
          order_id=order_id,
          quantity=quantity,
          leverage=leverage
        )

        if trade_details:
          logger.info(f"✅ Сделка записана в БД: ID={trade_details.get('id')}, Symbol={symbol}")
        else:
          logger.error(f"❌ Не удалось записать сделку в БД для {symbol}")

        if hasattr(signal, 'metadata') and signal.metadata:
          shadow_id = signal.metadata.get('shadow_tracking_id')
          if shadow_id and hasattr(self, 'shadow_trading') and self.shadow_trading:
            try:
              await self.shadow_trading.signal_tracker.mark_signal_executed(
                shadow_id,
                order_id,
                quantity,
                leverage
              )
            except Exception as e:
              logger.warning(f"Ошибка обновления Shadow Trading: {e}")

        return True, trade_details
        # Возвращаем успех и детали сделки

      else:
        if order_response:
          # Если ответ есть, но в нем ошибка
          ret_msg = order_response.get('retMsg', 'Неизвестная ошибка API')
          logger.error(f"❌ Не удалось разместить ордер для {symbol}. Причина: {ret_msg}. Ответ биржи: {order_response}")
          signal_logger.error(f"ИСПОЛНИТЕЛЬ: ОШИБКА. Ордер не принят. Ответ: {ret_msg}")
        else:
          # Если ответа нет совсем (например, таймаут сети)
          logger.error(f"❌ Не удалось разместить ордер для {symbol}. Нет ответа от биржи.")
          signal_logger.error("ИСПОЛНИТЕЛЬ: ОШИБКА. Нет ответа от биржи.")

        signal_logger.info(f"====== ЦИКЛ СИГНАЛА ДЛЯ {symbol} ЗАВЕРШЕН ======\n")
        return False, None

    except Exception as e:
      logger.error(f"Критическая ошибка при исполнении сделки {symbol}: {e}", exc_info=True)
      return False, None

  async def close_position(self, symbol: str) -> bool:
    """
    Реализует полный алгоритм закрытия позиции по рынку.
    """
    logger.info(f"Попытка закрытия позиции по символу {symbol}...")

    try:
      # 1. Получаем реальную информацию о позициях с биржи
      positions = await self.connector.fetch_positions(symbol)
      active_position = next((pos for pos in positions if float(pos.get('size', 0)) > 0), None)

      # 2. Проверяем, есть ли что закрывать на бирже
      if not active_position:
        logger.warning(f"На бирже не найдено активной открытой позиции для {symbol}.")

        # --- ЛОГИКА ОБРАБОТКИ ЗОМБИ-ПОЗИЦИИ ---
        local_trade = await self.db_manager.get_open_trade_by_symbol(symbol)
        if local_trade:
          logger.warning(
            f"Найдена 'зомби-позиция' в локальной БД (ID: {local_trade['id']}). Принудительное закрытие...")
          # Получаем последнюю цену для записи в БД
          last_kline = await self.data_fetcher.get_historical_candles(symbol, Timeframe.ONE_MINUTE, limit=1)
          close_price = last_kline['close'].iloc[-1] if not last_kline.empty else 0

          await self.db_manager.force_close_trade(
            trade_id=local_trade['id'],
            close_price=close_price,
            reason=f"Forced closure due to no position on exchange"
          )
          logger.info(f"Зомби-позиция (ID: {local_trade['id']}) успешно закрыта в БД.")
        # --- КОНЕЦ ЛОГИКИ ---
        return True  # Считаем задачу выполненной в любом случае

      # Если позиция на бирже есть, продолжаем стандартное закрытие
      pos_size_str = active_position.get('size', '0')
      pos_side = active_position.get('side')
      logger.info(f"Найдена позиция на бирже: {pos_side} {pos_size_str} {symbol}")

      # 3. Формируем ордер на закрытие
      close_side = "Sell" if pos_side == "Buy" else "Buy"
      params = {
        'symbol': symbol,
        'side': close_side,
        'orderType': 'Market',
        'qty': str(float(pos_size_str)),
        'reduceOnly': True,
        'positionIdx': 0
      }
      # 4. Отправляем ордер
      order_response = await self.connector.place_order(**params)

      # 5. Проверяем результат
      if order_response and order_response.get('orderId'):
        logger.info(f"✅ Ордер на закрытие {symbol} успешно принят биржей. OrderID: {order_response.get('orderId')}")

        # ВАЖНО: На этом этапе мы только отправили ордер.
        # Расчет PnL и обновление статуса в БД на 'CLOSED' должно происходить
        # в отдельном процессе, который отслеживает исполнение ордеров.

        return True


      else:
        logger.error(f"❌ Не удалось разместить ордер на закрытие для {symbol}. Ответ биржи: {order_response}")
        return False

    except Exception as e:
      logger.error(f"Критическая ошибка при закрытии позиции {symbol}: {e}", exc_info=True)
      return False
#-----------------------------------------------------------
  # async def close_position(self, symbol: str, db_trade_id: [int] = None, open_order_id: [str] = None,
  #                          quantity_to_close: [float] = None) -> bool:
  #   """
  #   Закрывает существующую открытую позицию (или ее часть).
  #   Предполагается, что закрытие происходит рыночным ордером в противоположную сторону.
  #
  #   Args:
  #       symbol (str): Торговый символ.
  #       db_trade_id (Optional[int]): ID сделки в нашей БД (если закрываем конкретную).
  #       open_order_id (Optional[str]): ID ордера на открытие (если db_trade_id не известен).
  #       quantity_to_close (Optional[float]): Количество для закрытия. Если None, закрывается вся позиция из БД.
  #
  #   Returns:
  #       bool: True, если закрытие инициировано успешно, иначе False.
  #   """
  #   trade_info = None
  #   if db_trade_id:
  #     # Этот метод должен быть синхронным, если db_manager не async
  #     # trade_info = self.db_manager.get_trade_by_id(db_trade_id) # Предположим, есть такой метод
  #     # Если нет, то ищем по order_id или другим критериям
  #     pass  # Для примера, если бы мы искали по ID из нашей БД
  #
  #   if not trade_info and open_order_id:
  #     trade_info = self.db_manager.get_trade_by_order_id(open_order_id)
  #
  #   if not trade_info:
  #     # Если нет информации из БД, пытаемся получить текущую позицию с биржи
  #     logger.warning(
  #       f"Нет информации о сделке в БД для {symbol} (OrderID: {open_order_id}). Попытка получить позицию с биржи.")
  #     positions = await self.connector.fetch_positions(symbols=[symbol])
  #     if positions:
  #       # Ищем позицию по символу. Bybit fetch_positions может возвращать несколько (для hedge mode)
  #       # Для one-way mode должна быть одна или ни одной.
  #       current_pos = None
  #       for pos_item in positions:
  #         if pos_item['symbol'] == symbol and float(pos_item.get('contracts', 0)) != 0:
  #           current_pos = pos_item
  #           break
  #
  #       if current_pos:
  #         pos_size = float(current_pos.get('contracts', 0))
  #         pos_side = 'buy' if pos_size > 0 else 'sell'  # Направление открытой позиции
  #
  #         close_side = 'sell' if pos_side == 'buy' else 'buy'
  #         qty_to_close = abs(pos_size)
  #
  #         logger.info(f"Найдена активная позиция на бирже для {symbol}: {pos_side} {qty_to_close}. Попытка закрытия.")
  #         # Размещаем ордер на закрытие
  #         # Bybit API v5 для закрытия требует `reduceOnly=True`
  #         close_order_params = {
  #           'category': self.connector.exchange.options.get('defaultType', 'linear'),
  #           'reduceOnly': True
  #         }
  #         close_order_response = await self.connector.place_order(
  #           symbol=symbol,
  #           side=close_side,
  #           order_type='market',
  #           amount=qty_to_close,
  #           params=close_order_params
  #         )
  #
  #         if close_order_response and 'id' in close_order_response:
  #           close_order_id = close_order_response['id']
  #           # Цена закрытия и PnL будут известны после исполнения ордера.
  #           # Это требует дополнительной логики для отслеживания исполнения и обновления БД.
  #           logger.info(f"Ордер на закрытие позиции {symbol} (ID: {close_order_id}) успешно размещен.")
  #           # Здесь нужно будет дождаться исполнения и обновить БД, это упрощенный пример.
  #           # Для корректного обновления БД с PnL, нужна информация о цене закрытия.
  #           # Можно подписаться на WebSocket на обновления ордеров или периодически опрашивать.
  #           # Пока что просто логируем факт отправки ордера на закрытие.
  #           # db_manager.update_close_trade(...) будет вызван позже, когда будут данные.
  #           return True
  #         else:
  #           logger.error(f"Не удалось разместить ордер на закрытие для {symbol}. Ответ: {close_order_response}")
  #           return False
  #       else:
  #         logger.warning(f"Нет активной позиции на бирже для {symbol} для закрытия.")
  #         return False
  #     else:
  #       logger.error(f"Не удалось получить информацию о позициях с биржи для {symbol}.")
  #       return False
  #
  #   # Если же у нас есть trade_info из нашей БД (т.е. мы отслеживаем позицию)
  #   if trade_info and trade_info['status'] == 'OPEN':
  #     original_side = trade_info['side']
  #     original_quantity = trade_info['quantity']
  #     open_order_id_from_db = trade_info['order_id']
  #
  #     close_side = 'sell' if original_side == 'buy' else 'buy'
  #     qty_to_close = quantity_to_close if quantity_to_close else original_quantity
  #
  #     logger.info(
  #       f"Попытка закрыть позицию из БД (OrderID: {open_order_id_from_db}): {symbol} {close_side} {qty_to_close}")
  #
  #     close_order_params = {
  #       'category': self.connector.exchange.options.get('defaultType', 'linear'),
  #       'reduceOnly': True  # Важно для закрытия позиции
  #     }
  #
  #     # Если мы знаем ID ордера на открытие, его можно передать в `clientOrderId` для связи,
  #     # но это не стандартный параметр для закрытия в CCXT.
  #     # if open_order_id_from_db:
  #     #    close_order_params['clientOrderId'] = f"close_{open_order_id_from_db}"
  #
  #     close_order_response = await self.connector.place_order(
  #       symbol=symbol,
  #       side=close_side,
  #       order_type='market',  # Обычно закрывают рыночным
  #       amount=qty_to_close,
  #       params=close_order_params
  #     )
  #
  #     if close_order_response and 'id' in close_order_response:
  #       close_order_id = close_order_response['id']
  #       logger.info(
  #         f"Ордер на закрытие (ID: {close_order_id}) для позиции {open_order_id_from_db} ({symbol}) успешно размещен.")
  #
  #       # ВАЖНО: Обновление БД с P/L, комиссией и ценой закрытия должно происходить
  #       # ПОСЛЕ фактического исполнения ордера на закрытие и получения этих данных.
  #       # Это требует механизма отслеживания статуса ордеров (например, через WebSocket или polling).
  #       # Здесь мы только инициируем закрытие. Логика обновления БД будет в другом месте,
  #       # например, в основном цикле бота, который слушает обновления ордеров.
  #
  #       # Пока что, мы можем пометить в БД, что инициировано закрытие, или дождаться.
  #       # Для простоты, предположим, что мы получим коллбэк или проверим позже.
  #       return True
  #     else:
  #       logger.error(
  #         f"Не удалось разместить ордер на закрытие для позиции {open_order_id_from_db} ({symbol}). Ответ: {close_order_response}")
  #       return False
  #   else:
  #     logger.warning(
  #       f"Не найдена открытая сделка в БД с ID {db_trade_id} или OrderID {open_order_id} для {symbol}, или она уже не 'OPEN'.")
  #     return False

  async def update_trade_status_from_exchange(self, order_id: str, symbol: str):
    """
    Запрашивает статус ордера с биржи и обновляет БД, если ордер исполнен (полностью или частично).
    Этот метод будет вызываться периодически или по событию для ордеров, которые были отправлены на закрытие.
    """
    if not self.connector.exchange:
      logger.error("CCXT exchange не инициализирован для обновления статуса сделки.")
      return

    try:
      # Получаем информацию об ордере с биржи
      # Bybit требует 'category' в params
      order_info = await self.connector.exchange.fetch_order(order_id, symbol, params={'category': BYBIT_CATEGORY})
      logger.debug(f"Информация об ордере {order_id} ({symbol}) с биржи: {order_info}")

      if not order_info:
        logger.warning(f"Не удалось получить информацию об ордере {order_id} ({symbol}) с биржи.")
        return

      order_status = order_info.get('status')  # 'closed' (исполнен), 'open', 'canceled'

      # Нас интересует ордер, который был ордером на закрытие существующей позиции
      # и он был исполнен ('closed' в терминах CCXT означает filled)

      db_trade_record = self.db_manager.get_trade_by_order_id(
        order_id)  # Это если order_id - это ID ордера на ОТКРЫТИЕ.
      # Нам нужен механизм связи ордера на закрытие с ордером на открытие.

      # Допустим, у нас есть ID ордера на ОТКРЫТИЕ, и мы хотим обновить его статус
      # Это более сложная логика, т.к. `order_id` здесь - это ID ордера на ЗАКРЫТИЕ.
      # Нужно найти соответствующую ОТКРЫТУЮ сделку в нашей БД, которую этот ордер закрыл.
      # Это можно сделать, если при закрытии мы сохраняем связь, или по символу и противоположному сайду.

      # Упрощенный сценарий: мы получили коллбэк, что ордер на закрытие (close_order_id) исполнился.
      # Нам нужно найти исходную сделку (original_open_order_id) и обновить ее.
      # В данном методе `order_id` - это ID ордера, чей статус мы проверяем.
      # Если это ордер на закрытие, и он 'closed' (filled):
      if order_status == 'closed':  # 'filled' в терминах биржи
        filled_price = float(order_info.get('average', order_info.get('price', 0.0)))  # Средняя цена исполнения
        filled_qty = float(order_info.get('filled', 0.0))
        commission_cost = float(order_info.get('fee', {}).get('cost', 0.0)) if order_info.get('fee') else 0.0
        # commission_currency = order_info.get('fee', {}).get('currency')

        # Теперь нужно найти соответствующую ОТКРЫТУЮ сделку в нашей БД, которую этот ордер закрыл.
        # Это самая сложная часть, если мы не сохранили явную связь.
        # Предположим, что мы закрывали позицию по символу `symbol`.
        # Ищем в БД открытую позицию по этому символу.
        open_trades_in_db = self.db_manager.get_open_positions_from_db()
        target_trade_to_update = None
        for trade in open_trades_in_db:
          if trade['symbol'] == symbol:
            # Если мы закрывали часть позиции, логика усложняется.
            # Для простоты, если этот ордер закрыл количество, равное открытой позиции.
            if abs(filled_qty - trade['quantity']) < 1e-9:  # Сравнение float
              target_trade_to_update = trade
              break

        if target_trade_to_update:
          original_open_order_id = target_trade_to_update['order_id']
          open_price_db = target_trade_to_update['open_price']
          original_side_db = target_trade_to_update['side']
          original_qty_db = target_trade_to_update['quantity']

          # Расчет P/L
          pnl = 0
          if original_side_db == 'buy':  # Позиция была лонг, закрыли продажей
            pnl = (filled_price - open_price_db) * original_qty_db
          elif original_side_db == 'sell':  # Позиция была шорт, закрыли покупкой
            pnl = (open_price_db - filled_price) * original_qty_db

          # PnL с учетом кредитного плеча уже заложен в том, что quantity - это размер контракта.
          # Комиссия вычитается из PnL
          net_pnl = pnl - commission_cost

          logger.info(f"Ордер на закрытие {order_id} для {symbol} исполнен. "
                      f"Цена закрытия: {filled_price}, Кол-во: {filled_qty}, Комиссия: {commission_cost}. Расчетный P/L: {net_pnl}")

          await self.db_manager.update_close_trade(
            order_id=original_open_order_id,  # Обновляем запись об исходной сделке
            close_timestamp=datetime.fromtimestamp(order_info['timestamp'] / 1000, tz=timezone.utc) if order_info.get(
              'timestamp') else datetime.now(timezone.utc),
            close_price=filled_price,
            profit_loss=net_pnl,
            commission=commission_cost
          )
        else:
          logger.warning(
            f"Исполнен ордер на закрытие {order_id} ({symbol}), но не найдена соответствующая открытая сделка в БД для обновления.")
      elif order_status in ['open', 'partially_filled']:
        logger.info(f"Ордер {order_id} ({symbol}) все еще активен (статус: {order_status}).")
      elif order_status in ['canceled', 'rejected', 'expired']:
        logger.warning(f"Ордер {order_id} ({symbol}) не был исполнен (статус: {order_status}).")
        # Здесь может потребоваться логика для отмены/обновления в нашей БД, если это был ордер на открытие.
        # Если это был ордер на закрытие, то позиция все еще открыта.

    except ccxt.OrderNotFound:
      logger.warning(
        f"Ордер {order_id} для символа {symbol} не найден на бирже. Возможно, он был исполнен давно или ID неверен.")
      # Можно проверить, есть ли он в нашей БД как открытый и пометить его как "потерянный" или "ошибка".
    except Exception as e:
      logger.error(f"Ошибка при обновлении статуса ордера {order_id} ({symbol}): {e}", exc_info=True)

  async def execute_grid_trade(self, grid_signal: GridSignal) -> bool:
    """
    ИСПОЛНЯЕТ СЕТОЧНЫЙ СИГНАЛ С ПРОВЕРКОЙ БАЛАНСА И КОНТРОЛЕМ ЧАСТОТЫ ЗАПРОСОВ.
    """
    logger.info(f"ИСПОЛНИТЕЛЬ для {grid_signal.symbol}: Получена команда на развертывание СЕТКИ.")
    signal_logger.info(
      f"СЕТКА: Валидация для {grid_signal.symbol}. Buy: {len(grid_signal.buy_orders)}, Sell: {len(grid_signal.sell_orders)} уровней.")

    try:
      # --- ШАГ 1: Проверка на минимальные требования и баланс ПЕРЕД отправкой ---

      # Получаем настройки из конфига
      trade_settings = self.config.get('trade_settings', {})
      total_allocation_usdt = trade_settings.get('grid_total_usdt_allocation', 50.0)
      min_order_value_usdt = trade_settings.get('min_order_value_usdt', 5.5)

      # Рассчитываем размер ОДНОГО ордера в сетке
      num_buy_orders = len(grid_signal.buy_orders)
      num_sell_orders = len(grid_signal.sell_orders)

      if num_buy_orders == 0 or num_sell_orders == 0:
        logger.warning("Сетка не может быть развернута: нет ордеров на покупку или продажу.")
        signal_logger.warning("СЕТКА: Отклонено - нет уровней.")
        return False

      # Размер одного ордера на покупку и продажу
      buy_order_size_usdt = total_allocation_usdt / num_buy_orders
      sell_order_size_usdt = total_allocation_usdt / num_sell_orders

      # Проверяем, соответствует ли размер ОДНОГО ордера минимальным требованиям биржи
      if buy_order_size_usdt < min_order_value_usdt or sell_order_size_usdt < min_order_value_usdt:
        logger.error(
          f"Невозможно развернуть сетку: расчетный размер ордера ({buy_order_size_usdt:.2f} USDT) меньше минимального ({min_order_value_usdt} USDT).")
        signal_logger.error(f"СЕТКА: Отклонено - размер ордера слишком мал.")
        logger.info("РЕКОМЕНДАЦИЯ: Увеличьте 'grid_total_usdt_allocation' или уменьшите 'grid_levels' в настройках.")
        return False

      # --- ШАГ 2: Последовательное размещение ордеров с задержкой ---

      instrument_info = await self.data_fetcher.get_instrument_info(grid_signal.symbol)
      lot_size_filter = instrument_info.get('lotSizeFilter', {})
      qty_step_str = lot_size_filter.get('qtyStep', '1')

      all_orders_params = []

      # Подготовка параметров для ордеров на покупку
      for order in grid_signal.buy_orders:
        qty = (buy_order_size_usdt / order.price) if order.price > 0 else 0
        adjusted_qty = float(
          (Decimal(str(qty)) / Decimal(qty_step_str)).to_integral_value(rounding=ROUND_DOWN) * Decimal(qty_step_str))
        if adjusted_qty > 0:
          all_orders_params.append(
            {'symbol': grid_signal.symbol, 'side': 'Buy', 'orderType': 'Limit', 'qty': str(adjusted_qty),
             'price': str(order.price),'positionIdx': 0})

      # Подготовка параметров для ордеров на продажу
      for order in grid_signal.sell_orders:
        qty = (sell_order_size_usdt / order.price) if order.price > 0 else 0
        adjusted_qty = float(
          (Decimal(str(qty)) / Decimal(qty_step_str)).to_integral_value(rounding=ROUND_DOWN) * Decimal(qty_step_str))
        if adjusted_qty > 0:
          all_orders_params.append(
            {'symbol': grid_signal.symbol, 'side': 'Sell', 'orderType': 'Limit', 'qty': str(adjusted_qty),
             'price': str(order.price)})

      if not all_orders_params:
        logger.warning("Нет ордеров для размещения после корректировки количества.")
        return False

      logger.info(f"Размещение {len(all_orders_params)} лимитных ордеров для сетки {grid_signal.symbol} с задержкой...")

      success_count = 0
      # Размещаем ордера ПО ОДНОМУ с задержкой, чтобы не превысить rate limit
      for params in all_orders_params:
        try:
          result = await self.connector.place_order(**params)
          if result and result.get('orderId'):
            success_count += 1
            logger.debug(f"Успешно размещен ордер: {params}")
          else:
            logger.error(f"Ошибка размещения ордера в сетке: {result.get('retMsg', 'Нет ответа')}")

          # --- Контроль частоты запросов ---
          await asyncio.sleep(0.3)  # Задержка 120 мс (8 запросов/сек)
        except Exception as e:
          logger.error(f"Исключение при размещении ордера {params}: {e}")

      logger.info(f"Успешно размещено {success_count} из {len(all_orders_params)} ордеров сетки.")
      signal_logger.info(f"СЕТКА: Успешно размещено {success_count}/{len(all_orders_params)} ордеров.")
      return success_count > 0

    except Exception as e:
      logger.error(f"Критическая ошибка при исполнении сетки {grid_signal.symbol}: {e}", exc_info=True)
      return False