# connectors/bybit_websocket.py
"""
WebSocket клиент для Bybit API для получения данных в реальном времени
"""

import asyncio
import json
import time
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import hmac

from utils.logging_config import get_logger
from core.circuit_breaker import circuit_breaker, CircuitBreakerOpenError

logger = get_logger(__name__)


@dataclass
class SubscriptionInfo:
  """Информация о подписке"""
  topic: str
  symbols: Set[str] = field(default_factory=set)
  callback: Optional[Callable] = None
  last_update: Optional[datetime] = None
  message_count: int = 0


class BybitWebSocketClient:
  """
  WebSocket клиент для получения данных Bybit в реальном времени
  """

  def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
    self.api_key = api_key
    self.api_secret = api_secret
    self.testnet = testnet

    # WebSocket URL
    if testnet:
      self.ws_public_url = "wss://stream-testnet.bybit.com/v5/public/linear"
      self.ws_private_url = "wss://stream-testnet.bybit.com/v5/private"
    else:
      self.ws_public_url = "wss://stream.bybit.com/v5/public/linear"
      self.ws_private_url = "wss://stream.bybit.com/v5/private"

    # Соединения
    self.public_ws: Optional[websockets.WebSocketServerProtocol] = None
    self.private_ws: Optional[websockets.WebSocketServerProtocol] = None

    # Подписки
    self.subscriptions: Dict[str, SubscriptionInfo] = {}

    # Коллбэки для разных типов данных
    self.callbacks: Dict[str, List[Callable]] = defaultdict(list)

    # Статистика
    self.stats = {
      'messages_received': 0,
      'last_message_time': None,
      'connection_start_time': None,
      'reconnections': 0,
      'errors': 0
    }

    # Управление соединением
    self.is_running = False
    self.reconnect_interval = 5
    self.max_reconnect_attempts = 10

    # Heartbeat
    self.last_heartbeat = datetime.now()
    self.heartbeat_interval = 20

    logger.info(f"WebSocket клиент инициализирован (testnet: {testnet})")

  async def connect(self):
    """Устанавливает WebSocket соединения"""
    self.is_running = True
    self.stats['connection_start_time'] = datetime.now()

    try:
      # Подключаемся к публичному потоку
      await self._connect_public()

      # Подключаемся к приватному потоку если есть API ключи
      if self.api_key and self.api_secret:
        await self._connect_private()

      logger.info("WebSocket соединения установлены")

    except Exception as e:
      logger.error(f"Ошибка установки WebSocket соединений: {e}")
      raise

  async def _connect_public(self):
    """Подключается к публичному WebSocket"""
    try:
      self.public_ws = await websockets.connect(
        self.ws_public_url,
        ping_interval=20,
        ping_timeout=10,
        close_timeout=10
      )

      # Запускаем обработчик сообщений
      asyncio.create_task(self._handle_public_messages())

      logger.info("Подключен к публичному WebSocket")

    except Exception as e:
      logger.error(f"Ошибка подключения к публичному WebSocket: {e}")
      raise

  async def _connect_private(self):
    """Подключается к приватному WebSocket с аутентификацией"""
    try:
      self.private_ws = await websockets.connect(
        self.ws_private_url,
        ping_interval=20,
        ping_timeout=10,
        close_timeout=10
      )

      # Аутентификация
      await self._authenticate_private()

      # Запускаем обработчик приватных сообщений
      asyncio.create_task(self._handle_private_messages())

      logger.info("Подключен к приватному WebSocket")

    except Exception as e:
      logger.error(f"Ошибка подключения к приватному WebSocket: {e}")
      raise

  async def _authenticate_private(self):
    """Аутентификация для приватного WebSocket"""
    expires = int((time.time() + 10) * 1000)
    signature = hmac.new(
      self.api_secret.encode('utf-8'),
      f'GET/realtime{expires}'.encode('utf-8'),
      hashlib.sha256
    ).hexdigest()

    auth_message = {
      "op": "auth",
      "args": [self.api_key, expires, signature]
    }

    await self.private_ws.send(json.dumps(auth_message))
    logger.debug("Отправлен запрос аутентификации")

  async def _handle_public_messages(self):
    """Обрабатывает сообщения из публичного WebSocket"""
    try:
      async for message in self.public_ws:
        await self._process_message(message, 'public')
    except websockets.exceptions.ConnectionClosed:
      logger.warning("Публичное WebSocket соединение закрыто")
      if self.is_running:
        await self._reconnect_public()
    except Exception as e:
      logger.error(f"Ошибка обработки публичных сообщений: {e}")
      self.stats['errors'] += 1

  async def _handle_private_messages(self):
    """Обрабатывает сообщения из приватного WebSocket"""
    try:
      async for message in self.private_ws:
        await self._process_message(message, 'private')
    except websockets.exceptions.ConnectionClosed:
      logger.warning("Приватное WebSocket соединение закрыто")
      if self.is_running:
        await self._reconnect_private()
    except Exception as e:
      logger.error(f"Ошибка обработки приватных сообщений: {e}")
      self.stats['errors'] += 1

  async def _process_message(self, message: str, ws_type: str):
    """Обрабатывает входящее сообщение"""
    try:
      data = json.loads(message)
      self.stats['messages_received'] += 1
      self.stats['last_message_time'] = datetime.now()
      self.last_heartbeat = datetime.now()

      # Обрабатываем разные типы сообщений
      if 'topic' in data:
        topic = data['topic']

        # Обновляем статистику подписки
        if topic in self.subscriptions:
          self.subscriptions[topic].last_update = datetime.now()
          self.subscriptions[topic].message_count += 1

        # Вызываем коллбэки
        await self._trigger_callbacks(topic, data)

      elif data.get('op') == 'pong':
        # Ответ на ping
        logger.debug("Получен pong")

      elif data.get('op') == 'auth':
        # Результат аутентификации
        if data.get('success'):
          logger.info("Успешная аутентификация WebSocket")
        else:
          logger.error(f"Ошибка аутентификации WebSocket: {data}")

      elif 'success' in data and 'op' in data:
        # Ответ на операцию подписки/отписки
        op = data['op']
        success = data['success']
        if success:
          logger.debug(f"Операция {op} выполнена успешно")
        else:
          logger.error(f"Ошибка операции {op}: {data}")

    except json.JSONDecodeError:
      logger.error(f"Невалидный JSON: {message}")
    except Exception as e:
      logger.error(f"Ошибка обработки сообщения: {e}")

  async def _trigger_callbacks(self, topic: str, data: dict):
    """Вызывает коллбэки для определенного топика"""
    try:
      # Определяем тип данных по топику
      data_type = self._get_data_type_from_topic(topic)

      # Вызываем коллбэки для конкретного топика
      if topic in self.callbacks:
        for callback in self.callbacks[topic]:
          try:
            if asyncio.iscoroutinefunction(callback):
              await callback(data)
            else:
              callback(data)
          except Exception as e:
            logger.error(f"Ошибка в коллбэке для {topic}: {e}")

      # Вызываем коллбэки для типа данных
      if data_type in self.callbacks:
        for callback in self.callbacks[data_type]:
          try:
            if asyncio.iscoroutinefunction(callback):
              await callback(data)
            else:
              callback(data)
          except Exception as e:
            logger.error(f"Ошибка в коллбэке для {data_type}: {e}")

    except Exception as e:
      logger.error(f"Ошибка выполнения коллбэков: {e}")

  def _get_data_type_from_topic(self, topic: str) -> str:
    """Определяет тип данных по топику"""
    if 'tickers' in topic:
      return 'ticker'
    elif 'orderbook' in topic:
      return 'orderbook'
    elif 'publicTrade' in topic:
      return 'trade'
    elif 'kline' in topic:
      return 'kline'
    elif 'position' in topic:
      return 'position'
    elif 'execution' in topic:
      return 'execution'
    elif 'order' in topic:
      return 'order'
    else:
      return 'unknown'

  # Методы подписки
  async def subscribe_tickers(self, symbols: List[str], callback: Callable = None):
    """Подписывается на тикеры символов"""
    if not symbols:
      return

    # Bybit позволяет подписаться на все тикеры сразу
    topic = "tickers"

    subscription_message = {
      "op": "subscribe",
      "args": [f"tickers.{symbol}" for symbol in symbols]
    }

    await self._send_subscription(subscription_message, 'public')

    # Сохраняем информацию о подписке
    self.subscriptions[topic] = SubscriptionInfo(
      topic=topic,
      symbols=set(symbols),
      callback=callback
    )

    if callback:
      self.callbacks[topic].append(callback)

    logger.info(f"Подписка на тикеры {len(symbols)} символов")

  async def subscribe_orderbook(self, symbols: List[str], depth: int = 25, callback: Callable = None):
    """Подписывается на стакан символов"""
    for symbol in symbols:
      topic = f"orderbook.{depth}.{symbol}"

      subscription_message = {
        "op": "subscribe",
        "args": [topic]
      }

      await self._send_subscription(subscription_message, 'public')

      self.subscriptions[topic] = SubscriptionInfo(
        topic=topic,
        symbols={symbol},
        callback=callback
      )

      if callback:
        self.callbacks[topic].append(callback)

    logger.info(f"Подписка на стакан {len(symbols)} символов (глубина: {depth})")

  async def subscribe_trades(self, symbols: List[str], callback: Callable = None):
    """Подписывается на публичные сделки"""
    for symbol in symbols:
      topic = f"publicTrade.{symbol}"

      subscription_message = {
        "op": "subscribe",
        "args": [topic]
      }

      await self._send_subscription(subscription_message, 'public')

      self.subscriptions[topic] = SubscriptionInfo(
        topic=topic,
        symbols={symbol},
        callback=callback
      )

      if callback:
        self.callbacks[topic].append(callback)

    logger.info(f"Подписка на сделки {len(symbols)} символов")

  async def subscribe_klines(self, symbols: List[str], interval: str = "1", callback: Callable = None):
    """Подписывается на свечи (klines)"""
    for symbol in symbols:
      topic = f"kline.{interval}.{symbol}"

      subscription_message = {
        "op": "subscribe",
        "args": [topic]
      }

      await self._send_subscription(subscription_message, 'public')

      self.subscriptions[topic] = SubscriptionInfo(
        topic=topic,
        symbols={symbol},
        callback=callback
      )

      if callback:
        self.callbacks[topic].append(callback)

    logger.info(f"Подписка на свечи {len(symbols)} символов (интервал: {interval})")

  # Приватные подписки
  async def subscribe_positions(self, callback: Callable = None):
    """Подписывается на изменения позиций"""
    if not self.private_ws:
      logger.error("Приватное WebSocket соединение недоступно")
      return

    topic = "position"

    subscription_message = {
      "op": "subscribe",
      "args": [topic]
    }

    await self._send_subscription(subscription_message, 'private')

    self.subscriptions[topic] = SubscriptionInfo(
      topic=topic,
      callback=callback
    )

    if callback:
      self.callbacks[topic].append(callback)

    logger.info("Подписка на изменения позиций")

  async def subscribe_orders(self, callback: Callable = None):
    """Подписывается на изменения ордеров"""
    if not self.private_ws:
      logger.error("Приватное WebSocket соединение недоступно")
      return

    topic = "order"

    subscription_message = {
      "op": "subscribe",
      "args": [topic]
    }

    await self._send_subscription(subscription_message, 'private')

    self.subscriptions[topic] = SubscriptionInfo(
      topic=topic,
      callback=callback
    )

    if callback:
      self.callbacks[topic].append(callback)

    logger.info("Подписка на изменения ордеров")

  async def subscribe_executions(self, callback: Callable = None):
    """Подписывается на исполнения (fills)"""
    if not self.private_ws:
      logger.error("Приватное WebSocket соединение недоступно")
      return

    topic = "execution"

    subscription_message = {
      "op": "subscribe",
      "args": [topic]
    }

    await self._send_subscription(subscription_message, 'private')

    self.subscriptions[topic] = SubscriptionInfo(
      topic=topic,
      callback=callback
    )

    if callback:
      self.callbacks[topic].append(callback)

    logger.info("Подписка на исполнения сделок")

  async def _send_subscription(self, message: dict, ws_type: str):
    """Отправляет сообщение подписки"""
    try:
      if ws_type == 'public' and self.public_ws:
        await self.public_ws.send(json.dumps(message))
      elif ws_type == 'private' and self.private_ws:
        await self.private_ws.send(json.dumps(message))
      else:
        logger.error(f"WebSocket {ws_type} недоступен для отправки")

    except Exception as e:
      logger.error(f"Ошибка отправки подписки: {e}")

  # Управление коллбэками
  def add_callback(self, data_type: str, callback: Callable):
    """Добавляет коллбэк для типа данных"""
    self.callbacks[data_type].append(callback)
    logger.debug(f"Добавлен коллбэк для {data_type}")

  def remove_callback(self, data_type: str, callback: Callable):
    """Удаляет коллбэк для типа данных"""
    if callback in self.callbacks[data_type]:
      self.callbacks[data_type].remove(callback)
      logger.debug(f"Удален коллбэк для {data_type}")

  # Переподключение
  async def _reconnect_public(self):
    """Переподключается к публичному WebSocket"""
    if not self.is_running:
      return

    self.stats['reconnections'] += 1

    for attempt in range(self.max_reconnect_attempts):
      try:
        logger.info(f"Попытка переподключения публичного WS #{attempt + 1}")
        await asyncio.sleep(self.reconnect_interval)

        await self._connect_public()

        # Восстанавливаем подписки
        await self._restore_public_subscriptions()

        logger.info("Публичное WebSocket соединение восстановлено")
        return

      except Exception as e:
        logger.error(f"Ошибка переподключения публичного WS: {e}")

    logger.error("Не удалось восстановить публичное WebSocket соединение")

  async def _reconnect_private(self):
    """Переподключается к приватному WebSocket"""
    if not self.is_running or not (self.api_key and self.api_secret):
      return

    self.stats['reconnections'] += 1

    for attempt in range(self.max_reconnect_attempts):
      try:
        logger.info(f"Попытка переподключения приватного WS #{attempt + 1}")
        await asyncio.sleep(self.reconnect_interval)

        await self._connect_private()

        # Восстанавливаем подписки
        await self._restore_private_subscriptions()

        logger.info("Приватное WebSocket соединение восстановлено")
        return

      except Exception as e:
        logger.error(f"Ошибка переподключения приватного WS: {e}")

    logger.error("Не удалось восстановить приватное WebSocket соединение")

  async def _restore_public_subscriptions(self):
    """Восстанавливает публичные подписки"""
    for topic, sub_info in self.subscriptions.items():
      if any(keyword in topic for keyword in ['tickers', 'orderbook', 'publicTrade', 'kline']):
        try:
          # Здесь нужно восстановить подписку в зависимости от типа
          # Это упрощенная версия
          subscription_message = {
            "op": "subscribe",
            "args": [topic]
          }
          await self._send_subscription(subscription_message, 'public')

        except Exception as e:
          logger.error(f"Ошибка восстановления подписки {topic}: {e}")

  async def _restore_private_subscriptions(self):
    """Восстанавливает приватные подписки"""
    for topic, sub_info in self.subscriptions.items():
      if topic in ['position', 'order', 'execution']:
        try:
          subscription_message = {
            "op": "subscribe",
            "args": [topic]
          }
          await self._send_subscription(subscription_message, 'private')

        except Exception as e:
          logger.error(f"Ошибка восстановления приватной подписки {topic}: {e}")

  # Heartbeat и мониторинг
  async def start_heartbeat(self):
    """Запускает heartbeat для поддержания соединения"""
    while self.is_running:
      try:
        await asyncio.sleep(self.heartbeat_interval)

        # Проверяем время последнего сообщения
        if (datetime.now() - self.last_heartbeat).seconds > 60:
          logger.warning("Нет сообщений более 60 секунд, отправляем ping")

          # Отправляем ping
          if self.public_ws:
            await self.public_ws.ping()
          if self.private_ws:
            await self.private_ws.ping()

      except Exception as e:
        logger.error(f"Ошибка heartbeat: {e}")

  # Статистика и состояние
  def get_stats(self) -> Dict:
    """Возвращает статистику WebSocket клиента"""
    uptime = None
    if self.stats['connection_start_time']:
      uptime = (datetime.now() - self.stats['connection_start_time']).total_seconds()

    return {
      'messages_received': self.stats['messages_received'],
      'last_message_time': self.stats['last_message_time'].isoformat() if self.stats['last_message_time'] else None,
      'uptime_seconds': uptime,
      'reconnections': self.stats['reconnections'],
      'errors': self.stats['errors'],
      'active_subscriptions': len(self.subscriptions),
      'subscriptions': {
        topic: {
          'symbols_count': len(sub.symbols),
          'last_update': sub.last_update.isoformat() if sub.last_update else None,
          'message_count': sub.message_count
        }
        for topic, sub in self.subscriptions.items()
      },
      'connections': {
        'public': self.public_ws is not None and not self.public_ws.closed,
        'private': self.private_ws is not None and not self.private_ws.closed
      }
    }

  # Закрытие соединений
  async def close(self):
    """Закрывает WebSocket соединения"""
    self.is_running = False

    try:
      if self.public_ws:
        await self.public_ws.close()
        self.public_ws = None

      if self.private_ws:
        await self.private_ws.close()
        self.private_ws = None

      logger.info("WebSocket соединения закрыты")

    except Exception as e:
      logger.error(f"Ошибка закрытия WebSocket соединений: {e}")


class WebSocketDataManager:
  """
  Менеджер для обработки и распределения данных WebSocket
  """

  def __init__(self, ws_client: BybitWebSocketClient):
    self.ws_client = ws_client
    self.data_cache: Dict[str, Any] = {}
    self.subscribers: Dict[str, List[Callable]] = defaultdict(list)

    # Настраиваем обработчики данных
    self._setup_handlers()

  def _setup_handlers(self):
    """Настраивает обработчики для разных типов данных"""
    self.ws_client.add_callback('ticker', self._handle_ticker_update)
    self.ws_client.add_callback('orderbook', self._handle_orderbook_update)
    self.ws_client.add_callback('trade', self._handle_trade_update)
    self.ws_client.add_callback('position', self._handle_position_update)
    self.ws_client.add_callback('order', self._handle_order_update)
    self.ws_client.add_callback('execution', self._handle_execution_update)

  async def _handle_ticker_update(self, data: dict):
    """Обрабатывает обновления тикеров"""
    try:
      if 'data' not in data:
        logger.debug("Нет данных в ticker update")
        return

      ticker_data = data['data']

      # Обрабатываем как список, так и отдельные объекты
      if isinstance(ticker_data, list):
        ticker_list = ticker_data
      else:
        ticker_list = [ticker_data]

      for ticker in ticker_list:
        if not isinstance(ticker, dict):
          logger.warning(f"Неожиданный формат тикера: {type(ticker)}")
          continue

        symbol = ticker.get('symbol')
        if not symbol:
          continue

        processed_ticker = {
          'symbol': symbol,
          'price': float(ticker.get('lastPrice', 0)),
          'bid': float(ticker.get('bid1Price', 0)),
          'ask': float(ticker.get('ask1Price', 0)),
          'volume': float(ticker.get('volume24h', 0)),
          'change_24h': float(ticker.get('price24hPcnt', 0)) * 100,
          'timestamp': datetime.now()
        }

        self.data_cache[f"ticker_{symbol}"] = processed_ticker

        # Уведомляем подписчиков
        await self._notify_subscribers('ticker', symbol, processed_ticker)

    except Exception as e:
      logger.error(f"Ошибка обработки тикера: {e}")
      logger.debug(f"Данные тикера: {data}")

  async def _handle_orderbook_update(self, data: dict):
    """Обрабатывает обновления стакана с улучшенной проверкой"""
    try:
      topic = data.get('topic', '')
      if not topic or 'orderbook' not in topic:
        logger.debug("Некорректный topic для orderbook")
        return

      if 'data' not in data:
        logger.debug("Нет данных в orderbook update")
        return

      # Извлекаем символ из topic
      try:
        symbol = topic.split('.')[-1]
        if not symbol:
          logger.warning("Не удалось извлечь символ из topic")
          return
      except IndexError:
        logger.warning(f"Некорректный формат topic: {topic}")
        return

      orderbook_data = data['data']

      # Проверяем структуру данных
      if not isinstance(orderbook_data, dict):
        logger.warning(f"Неожиданный тип данных orderbook: {type(orderbook_data)}")
        return

      bids = orderbook_data.get('b', [])
      asks = orderbook_data.get('a', [])

      # Проверяем качество данных
      if not isinstance(bids, list) or not isinstance(asks, list):
        logger.warning(f"Некорректные типы bids/asks для {symbol}")
        return

      # Валидируем данные стакана
      valid_bids = []
      valid_asks = []

      for bid in bids:
        try:
          if len(bid) >= 2 and float(bid[0]) > 0 and float(bid[1]) > 0:
            valid_bids.append(bid)
        except (ValueError, TypeError, IndexError):
          continue

      for ask in asks:
        try:
          if len(ask) >= 2 and float(ask[0]) > 0 and float(ask[1]) > 0:
            valid_asks.append(ask)
        except (ValueError, TypeError, IndexError):
          continue

      if not valid_bids and not valid_asks:
        logger.debug(f"Нет валидных данных стакана для {symbol}")
        return

      # Сортируем данные правильно
      try:
        valid_bids.sort(key=lambda x: float(x[0]), reverse=True)  # По убыванию цены
        valid_asks.sort(key=lambda x: float(x[0]))  # По возрастанию цены
      except (ValueError, TypeError) as e:
        logger.warning(f"Ошибка сортировки стакана для {symbol}: {e}")
        return

      self.data_cache[f"orderbook_{symbol}"] = {
        'symbol': symbol,
        'bids': valid_bids,
        'asks': valid_asks,
        'timestamp': datetime.now(),
        'update_count': self.data_cache.get(f"orderbook_{symbol}", {}).get('update_count', 0) + 1
      }

      await self._notify_subscribers('orderbook', symbol, self.data_cache[f"orderbook_{symbol}"])

      logger.debug(f"✅ Обновлен стакан для {symbol}: {len(valid_bids)} bids, {len(valid_asks)} asks")

    except Exception as e:
      logger.error(f"❌ Критическая ошибка обработки стакана: {e}", exc_info=True)

  async def _handle_trade_update(self, data: dict):
    """Обрабатывает публичные сделки"""
    try:
      if 'data' in data:
        for trade in data['data']:
          symbol = trade.get('s')
          if symbol:
            trade_data = {
              'symbol': symbol,
              'price': float(trade.get('p', 0)),
              'quantity': float(trade.get('v', 0)),
              'side': trade.get('S'),
              'timestamp': datetime.fromtimestamp(int(trade.get('T', 0)) / 1000)
            }

            await self._notify_subscribers('trade', symbol, trade_data)

    except Exception as e:
      logger.error(f"Ошибка обработки сделок: {e}")

  async def _handle_position_update(self, data: dict):
    """Обрабатывает изменения позиций"""
    try:
      if 'data' in data:
        for position in data['data']:
          symbol = position.get('symbol')
          if symbol:
            position_data = {
              'symbol': symbol,
              'side': position.get('side'),
              'size': float(position.get('size', 0)),
              'entry_price': float(position.get('avgPrice', 0)),
              'mark_price': float(position.get('markPrice', 0)),
              'unrealized_pnl': float(position.get('unrealisedPnl', 0)),
              'timestamp': datetime.now()
            }

            self.data_cache[f"position_{symbol}"] = position_data
            await self._notify_subscribers('position', symbol, position_data)

    except Exception as e:
      logger.error(f"Ошибка обработки позиций: {e}")

  async def _handle_order_update(self, data: dict):
    """Обрабатывает изменения ордеров"""
    try:
      if 'data' in data:
        for order in data['data']:
          order_id = order.get('orderId')
          symbol = order.get('symbol')

          if order_id and symbol:
            order_data = {
              'order_id': order_id,
              'symbol': symbol,
              'side': order.get('side'),
              'order_type': order.get('orderType'),
              'quantity': float(order.get('qty', 0)),
              'price': float(order.get('price', 0)),
              'status': order.get('orderStatus'),
              'filled_qty': float(order.get('cumExecQty', 0)),
              'timestamp': datetime.now()
            }

            await self._notify_subscribers('order', symbol, order_data)

    except Exception as e:
      logger.error(f"Ошибка обработки ордеров: {e}")

  async def _handle_execution_update(self, data: dict):
    """Обрабатывает исполнения сделок"""
    try:
      if 'data' in data:
        for execution in data['data']:
          symbol = execution.get('symbol')
          if symbol:
            execution_data = {
              'symbol': symbol,
              'order_id': execution.get('orderId'),
              'execution_id': execution.get('execId'),
              'side': execution.get('side'),
              'quantity': float(execution.get('execQty', 0)),
              'price': float(execution.get('execPrice', 0)),
              'fee': float(execution.get('execFee', 0)),
              'timestamp': datetime.fromtimestamp(int(execution.get('execTime', 0)) / 1000)
            }

            await self._notify_subscribers('execution', symbol, execution_data)

    except Exception as e:
      logger.error(f"Ошибка обработки исполнений: {e}")

  async def _notify_subscribers(self, data_type: str, symbol: str, data: dict):
    """Уведомляет подписчиков о новых данных"""
    # Уведомляем подписчиков конкретного символа
    key = f"{data_type}_{symbol}"
    if key in self.subscribers:
      for callback in self.subscribers[key]:
        try:
          if asyncio.iscoroutinefunction(callback):
            await callback(data)
          else:
            callback(data)
        except Exception as e:
          logger.error(f"Ошибка в подписчике {key}: {e}")

    # Уведомляем подписчиков типа данных
    if data_type in self.subscribers:
      for callback in self.subscribers[data_type]:
        try:
          if asyncio.iscoroutinefunction(callback):
            await callback(data)
          else:
            callback(data)
        except Exception as e:
          logger.error(f"Ошибка в подписчике {data_type}: {e}")

  def subscribe(self, data_type: str, callback: Callable, symbol: str = None):
    """Подписывается на определенный тип данных"""
    if symbol:
      key = f"{data_type}_{symbol}"
    else:
      key = data_type

    self.subscribers[key].append(callback)
    logger.debug(f"Добавлен подписчик для {key}")

  def get_latest_data(self, data_type: str, symbol: str = None) -> Optional[Dict]:
    """Получает последние данные из кэша"""
    if symbol:
      key = f"{data_type}_{symbol}"
    else:
      key = data_type

    return self.data_cache.get(key)

  def get_cache_stats(self) -> Dict:
    """Возвращает статистику кэша данных"""
    return {
      'cached_items': len(self.data_cache),
      'subscribers_count': sum(len(subs) for subs in self.subscribers.values()),
      'data_types': list(set(key.split('_')[0] for key in self.data_cache.keys())),
      'symbols_count': len(set(
        key.split('_', 1)[1] for key in self.data_cache.keys()
        if '_' in key and len(key.split('_')) >= 2
      ))
    }