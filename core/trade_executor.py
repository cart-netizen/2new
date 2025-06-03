from datetime import datetime, timezone

import ccxt

import config
from core.bybit_connector import BybitConnector
from core.integrated_system import IntegratedTradingSystem
from core.schemas import TradingSignal
from data.database_manager import AdvancedDatabaseManager

from utils.logging_config import setup_logging, get_logger
from config import LEVERAGE  # Глобальное плечо по умолчанию


logger = get_logger(__name__)


class TradeExecutor:
  def __init__(self, connector: BybitConnector, db_manager: AdvancedDatabaseManager, telegram_bot=None):
    self.connector = connector
    self.db_manager = db_manager
    self.telegram_bot = telegram_bot
    self.trading_system = IntegratedTradingSystem(db_manager=db_manager)


  async def execute_trade(self, signal: TradingSignal, symbol: str, quantity: float):
    """Исполняет торговый сигнал через интегрированную систему"""
    try:
      # Создаем order_id
      order_id = f"{symbol}_{int(datetime.now().timestamp())}"

      # Добавляем сделку через AdvancedDatabaseManager
      trade_id = self.db_manager.add_trade_with_signal(
        signal=signal,
        order_id=order_id,
        quantity=quantity,
        leverage=config.LEVERAGE
      )

      if trade_id:
        # Логируем исполненный сигнал
        self.db_manager.log_signal(signal, symbol, executed=True)
        return True
      return False
    except Exception as e:
      logger.error(f"Ошибка исполнения сделки: {e}")
      return False

#--------прошлая реализация async def execute_trade---------
  # async def execute_trade(self, symbol: str, side: str, quantity: float, strategy_name: str,
  #                         order_type: str = "Market", price: [float] = None, leverage: int = LEVERAGE,
  #                         stop_loss: [float] = None, take_profit: [float] = None) -> [str]:
  #   """
  #   Исполняет торговый приказ (открытие позиции).
  #
  #   Args:
  #       symbol (str): Торговый символ.
  #       side (str): 'buy' или 'sell'.
  #       quantity (float): Количество для покупки/продажи.
  #       strategy_name (str): Название стратегии, сгенерировавшей сигнал.
  #       order_type (str): 'Market' или 'Limit'.
  #       price (Optional[float]): Цена для Limit ордера.
  #       leverage (int): Кредитное плечо.
  #       stop_loss (Optional[float]): Цена Stop Loss.
  #       take_profit (Optional[float]): Цена Take Profit.
  #
  #   Returns:
  #       Optional[str]: ID ордера на бирже в случае успеха, иначе None.
  #   """
  #   logger.info(f"Попытка исполнить сделку: {symbol} {side} {quantity} @ {price if price else 'Market'}, "
  #               f"Strategy: {strategy_name}, Leverage: {leverage}x")
  #   #self.log(f"Попытка разместить ордер: {symbol} | {side} | amount={quantity} | price={price or 'market'}")
  #
  #   # 1. Установка кредитного плеча (если оно не установлено глобально или отличается)
  #   # Bybit требует установку плеча перед размещением ордера для пары.
  #   # Этот вызов может быть избыточным, если плечо уже установлено и не меняется.
  #   # Можно добавить проверку текущего плеча или управлять этим более гранулярно.
  #   # ВАЖНО: set_leverage должно быть вызвано до create_order
  #   leverage_set = await self.connector.set_leverage(symbol, leverage)
  #   if not leverage_set:  # или если leverage_set вернул ошибку
  #     # Некоторые реализации set_leverage в ccxt могут не возвращать тело ответа, а просто не кидать исключение
  #     # Проверяем более тщательно
  #     logger.warning(
  #       f"Не удалось подтвердить установку плеча {leverage}x для {symbol}, но продолжаем с размещением ордера.")
  #     # В критических системах, если set_leverage не подтвердилось, лучше не продолжать.
  #
  #   # 2. Подготовка параметров ордера, включая SL/TP для Bybit API v5
  #   params = {'category': self.connector.exchange.options.get('defaultType', 'linear')}  # 'linear' или 'inverse'
  #   if order_type.lower() == 'market':
  #     price = None  # Для рыночного ордера цена не указывается
  #
  #   if stop_loss:
  #     params['stopLoss'] = str(stop_loss)
  #     # Для Bybit может потребоваться 'slTriggerBy': 'MarkPrice' или 'LastPrice'
  #     # params['slTriggerBy'] = 'MarkPrice'
  #   if take_profit:
  #     params['takeProfit'] = str(take_profit)
  #     # params['tpTriggerBy'] = 'MarkPrice'
  #
  #   # 3. Размещение ордера через BybitConnector
  #   try:
  #     order_response = await self.connector.place_order(
  #       symbol=symbol,
  #       side=side.lower(),
  #       order_type=order_type.lower(),
  #       amount=quantity,
  #       price=price,
  #       params=params
  #     )
  #
  #     if order_response and 'id' in order_response:
  #       order_id = order_response['id']
  #       open_price = float(
  #         order_response.get('price', 0.0))  # Для рыночных ордеров цена исполнения будет в 'average' или 'filledPrice'
  #
  #       # Если это рыночный ордер, цена исполнения может быть не сразу известна или отличаться.
  #       # Bybit часто возвращает 0 в поле 'price' для рыночных ордеров в ответе create_order.
  #       # Реальную цену исполнения нужно будет получить из fetch_order(order_id) или из WebSocket обновлений.
  #       # Для простоты здесь используем то, что вернулось, или цену запроса для лимитных.
  #       if order_type.lower() == 'market' and order_response.get('average'):
  #         actual_open_price = float(order_response['average'])
  #       elif order_type.lower() == 'limit' and order_response.get('price'):
  #         actual_open_price = float(order_response['price'])
  #       else:  # Запасной вариант или если цена не пришла сразу
  #         actual_open_price = price if price else 0.0  # Нужна лучшая логика для рыночных
  #         logger.warning(
  #           f"Цена открытия для ордера {order_id} не была четко определена в ответе: {order_response}. Используется {actual_open_price}")
  #
  #       logger.info(f"Ордер {order_id} ({symbol} {side} {quantity}) успешно размещен. "
  #                   f"Цена открытия (приблизительно): {actual_open_price}")
  #
  #       if "error" in order_response:
  #         msg = f"⚠️ Ордер {side} {symbol} НЕ размещен: {result['error']}"
  #         self.log(msg, level="warning")
  #         await self.notify(msg)
  #         return False
  #
  #       if "order" in order_response:
  #         order_data = order_response["order"]
  #         msg = (
  #           f"✅ Ордер размещен: {side.upper()} {symbol}\n"
  #           f"📦 Кол-во: {order_data['amount']}\n"
  #           f"💵 Цена: {order_data.get('price', 'market')}\n"
  #           f"🆔 ID: {order_data.get('id')}"
  #         )
  #         self.log(msg)
  #         await self.notify(msg)
  #
  #         return True
  #       self.log(f"⚠️ Неизвестный результат от биржи для {symbol}", level="warning")
  #
  #       # 4. Запись информации об открытой сделке в БД
  #       self.db_manager.add_open_trade(
  #         symbol=symbol,
  #         order_id=order_id,
  #         strategy=strategy_name,
  #         side=side.lower(),
  #         open_timestamp=datetime.now(timezone.utc),  # Используем UTC
  #         open_price=actual_open_price,  # Здесь должна быть фактическая цена исполнения
  #         quantity=quantity,
  #         leverage=leverage
  #       )
  #       return order_id
  #     else:
  #       logger.error(f"Не удалось разместить ордер для {symbol} {side}. Ответ API: {order_response}")
  #
  #   except RuntimeError as e:
  #     if "Недостаточно средств" in str(e):
  #       msg = f"❌ {symbol} | Ордер не выполнен: Недостаточно средств"
  #       self.log(msg, level="error")
  #       await self.notify(msg)
  #       return False
  #
  #
  #   except Exception as e:
  #     self.log(f"‼️ Неизвестная ошибка при торговле {symbol}: {e}", level="error")
  #     await self.notify(f"⚠️ Ошибка при размещении ордера {symbol}: {e}")
  #     return False
  #
  # def log(self, message: str, level="info"):
  #     logger_method = getattr(self.connector.logger, level, self.connector.logger.info)
  #     logger_method(message)
  #
  # async def notify(self, message: str):
  #   if self.telegram_bot:
  #     # try:
  #       await self.telegram_bot.send_message(message)
  #     # except Exception as e:
  #     #   self.connector.logger.warning(f"Ошибка при отправке сообщения в Telegram: {e}")
  #
  #
#-----------------------------------------------------------
  async def close_position(self, symbol: str, db_trade_id: [int] = None, open_order_id: [str] = None,
                           quantity_to_close: [float] = None) -> bool:
    """
    Закрывает существующую открытую позицию (или ее часть).
    Предполагается, что закрытие происходит рыночным ордером в противоположную сторону.

    Args:
        symbol (str): Торговый символ.
        db_trade_id (Optional[int]): ID сделки в нашей БД (если закрываем конкретную).
        open_order_id (Optional[str]): ID ордера на открытие (если db_trade_id не известен).
        quantity_to_close (Optional[float]): Количество для закрытия. Если None, закрывается вся позиция из БД.

    Returns:
        bool: True, если закрытие инициировано успешно, иначе False.
    """
    trade_info = None
    if db_trade_id:
      # Этот метод должен быть синхронным, если db_manager не async
      # trade_info = self.db_manager.get_trade_by_id(db_trade_id) # Предположим, есть такой метод
      # Если нет, то ищем по order_id или другим критериям
      pass  # Для примера, если бы мы искали по ID из нашей БД

    if not trade_info and open_order_id:
      trade_info = self.db_manager.get_trade_by_order_id(open_order_id)

    if not trade_info:
      # Если нет информации из БД, пытаемся получить текущую позицию с биржи
      logger.warning(
        f"Нет информации о сделке в БД для {symbol} (OrderID: {open_order_id}). Попытка получить позицию с биржи.")
      positions = await self.connector.fetch_positions(symbols=[symbol])
      if positions:
        # Ищем позицию по символу. Bybit fetch_positions может возвращать несколько (для hedge mode)
        # Для one-way mode должна быть одна или ни одной.
        current_pos = None
        for pos_item in positions:
          if pos_item['symbol'] == symbol and float(pos_item.get('contracts', 0)) != 0:
            current_pos = pos_item
            break

        if current_pos:
          pos_size = float(current_pos.get('contracts', 0))
          pos_side = 'buy' if pos_size > 0 else 'sell'  # Направление открытой позиции

          close_side = 'sell' if pos_side == 'buy' else 'buy'
          qty_to_close = abs(pos_size)

          logger.info(f"Найдена активная позиция на бирже для {symbol}: {pos_side} {qty_to_close}. Попытка закрытия.")
          # Размещаем ордер на закрытие
          # Bybit API v5 для закрытия требует `reduceOnly=True`
          close_order_params = {
            'category': self.connector.exchange.options.get('defaultType', 'linear'),
            'reduceOnly': True
          }
          close_order_response = await self.connector.place_order(
            symbol=symbol,
            side=close_side,
            order_type='market',
            amount=qty_to_close,
            params=close_order_params
          )

          if close_order_response and 'id' in close_order_response:
            close_order_id = close_order_response['id']
            # Цена закрытия и PnL будут известны после исполнения ордера.
            # Это требует дополнительной логики для отслеживания исполнения и обновления БД.
            logger.info(f"Ордер на закрытие позиции {symbol} (ID: {close_order_id}) успешно размещен.")
            # Здесь нужно будет дождаться исполнения и обновить БД, это упрощенный пример.
            # Для корректного обновления БД с PnL, нужна информация о цене закрытия.
            # Можно подписаться на WebSocket на обновления ордеров или периодически опрашивать.
            # Пока что просто логируем факт отправки ордера на закрытие.
            # db_manager.update_close_trade(...) будет вызван позже, когда будут данные.
            return True
          else:
            logger.error(f"Не удалось разместить ордер на закрытие для {symbol}. Ответ: {close_order_response}")
            return False
        else:
          logger.warning(f"Нет активной позиции на бирже для {symbol} для закрытия.")
          return False
      else:
        logger.error(f"Не удалось получить информацию о позициях с биржи для {symbol}.")
        return False

    # Если же у нас есть trade_info из нашей БД (т.е. мы отслеживаем позицию)
    if trade_info and trade_info['status'] == 'OPEN':
      original_side = trade_info['side']
      original_quantity = trade_info['quantity']
      open_order_id_from_db = trade_info['order_id']

      close_side = 'sell' if original_side == 'buy' else 'buy'
      qty_to_close = quantity_to_close if quantity_to_close else original_quantity

      logger.info(
        f"Попытка закрыть позицию из БД (OrderID: {open_order_id_from_db}): {symbol} {close_side} {qty_to_close}")

      close_order_params = {
        'category': self.connector.exchange.options.get('defaultType', 'linear'),
        'reduceOnly': True  # Важно для закрытия позиции
      }

      # Если мы знаем ID ордера на открытие, его можно передать в `clientOrderId` для связи,
      # но это не стандартный параметр для закрытия в CCXT.
      # if open_order_id_from_db:
      #    close_order_params['clientOrderId'] = f"close_{open_order_id_from_db}"

      close_order_response = await self.connector.place_order(
        symbol=symbol,
        side=close_side,
        order_type='market',  # Обычно закрывают рыночным
        amount=qty_to_close,
        params=close_order_params
      )

      if close_order_response and 'id' in close_order_response:
        close_order_id = close_order_response['id']
        logger.info(
          f"Ордер на закрытие (ID: {close_order_id}) для позиции {open_order_id_from_db} ({symbol}) успешно размещен.")

        # ВАЖНО: Обновление БД с P/L, комиссией и ценой закрытия должно происходить
        # ПОСЛЕ фактического исполнения ордера на закрытие и получения этих данных.
        # Это требует механизма отслеживания статуса ордеров (например, через WebSocket или polling).
        # Здесь мы только инициируем закрытие. Логика обновления БД будет в другом месте,
        # например, в основном цикле бота, который слушает обновления ордеров.

        # Пока что, мы можем пометить в БД, что инициировано закрытие, или дождаться.
        # Для простоты, предположим, что мы получим коллбэк или проверим позже.
        return True
      else:
        logger.error(
          f"Не удалось разместить ордер на закрытие для позиции {open_order_id_from_db} ({symbol}). Ответ: {close_order_response}")
        return False
    else:
      logger.warning(
        f"Не найдена открытая сделка в БД с ID {db_trade_id} или OrderID {open_order_id} для {symbol}, или она уже не 'OPEN'.")
      return False

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

          self.db_manager.update_close_trade(
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
