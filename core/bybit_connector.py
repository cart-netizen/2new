import hashlib
import hmac
import time
import requests  # Для синхронных запросов, можно заменить на aiohttp для async
import json
from typing import Optional, List, Dict, Any
import asyncio
import logging

import ccxt.async_support as ccxt
from config import api_keys, settings, trading_params
from utils.logging_config import get_logger

logger = get_logger(__name__)


class BybitConnector:
  def __init__(self, api_key: Optional[str] = api_keys.API_KEY,
                 api_secret: Optional[str] = api_keys.API_SECRET):
    self.api_key = api_keys.API_KEY
    self.api_secret = api_keys.API_SECRET
    self.base_url = settings.BYBIT_API_URL
    self.recv_window = 5000  # Рекомендовано Bybit
    self.exchange = None
    self.market_info_cache = None
    self.client_session = None

    if not self.api_key or "YOUR_" in self.api_key:
      logger.warning("API ключ Bybit не настроен или используется ключ-заглушка.")
    if not self.api_secret or "YOUR_" in self.api_secret:
      logger.warning("API секрет Bybit не настроен или используется секрет-заглушка.")

    exchange_params = {
      'apiKey': self.api_key,
      'secret': self.api_secret,
      'enableRateLimit': True,
      'timeout': 30000,
      'options': {
        'defaultType': 'linear',
        'adjustForTimeDifference': True,
        'recvWindow': 20000
      }
    }
    if settings.USE_TESTNET:
      exchange_params['urls'] = {'api': settings.BYBIT_API_URL}

    self.exchange = ccxt.bybit(exchange_params)


  def _generate_signature(self, params_str: str) -> str:
    return hmac.new(self.api_secret.encode('utf-8'), params_str.encode('utf-8'), hashlib.sha256).hexdigest()

  async def initialize(self):
    """Инициализирует сессию и синхронизирует время"""
    try:
      # Загрузка рынков
      await self.exchange.load_markets()
      self.market_info_cache = self.exchange.markets

      # Синхронизация времени
      server_ts = await self.get_server_time()
      local_ts = int(time.time() * 1000)
      time_diff = server_ts - local_ts
      self.exchange.options['timeDifference'] = time_diff
      logger.info(f"Синхронизация времени: timeDifference = {time_diff} ms")
    except Exception as e:
      logger.error(f"Ошибка инициализации: {e}", exc_info=True)

  async def close(self):
    """Закрывает соединения и освобождает ресурсы"""
    try:
      if self.exchange:
        await self.exchange.close()
      if self.client_session and not self.client_session.closed:
        await self.client_session.close()
      logger.info("Ресурсы BybitConnector закрыты")
    except Exception as e:
      logger.error(f"Ошибка при закрытии ресурсов: {e}")

  async def get_usdt_perpetual_contracts(self) -> List[Dict[str, Any]]:
    """Получает список всех бессрочных USDT контрактов"""
    try:
      tickers = await self.exchange.fetch_tickers(params={'category': 'linear'})
      return list(tickers.values())

    except Exception as e:
      logger.error(f"Ошибка получения контрактов: {e}")
      return []

  # def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
  #   if params is None:
  #     params = {}
  #
  #   timestamp = int(time.time() * 1000)
  #
  #   # Исправление: правильный порядок параметров для подписи
  #   sign_payload_str = str(timestamp) + self.api_key + str(self.recv_window)
  #
  #   if method.upper() == "GET":
  #     # Добавляем обязательные параметры в запрос
  #     params['api_key'] = self.api_key
  #     params['timestamp'] = timestamp
  #     params['recv_window'] = self.recv_window
  #
  #     # Сортируем все параметры (включая добавленные)
  #     sorted_params = sorted(params.items())
  #     query_string_list = []
  #     for key, value in sorted_params:
  #       # Исключаем пустые значения
  #       if value is not None:
  #         query_string_list.append(f"{key}={value}")
  #     param_str_for_signature = "&".join(query_string_list)
  #
  #     # Исправление: добавляем query string к payload
  #     sign_payload_str += param_str_for_signature
  #
  #   elif method.upper() == "POST":
  #     # Для POST используем JSON тело
  #     json_body = json.dumps(params) if params else ""
  #     sign_payload_str += json_body
  #
  #   # Генерируем подпись
  #   signature = self._generate_signature(sign_payload_str)
  #
  #   headers = {
  #     'X-BAPI-API-KEY': self.api_key,
  #     'X-BAPI-TIMESTAMP': str(timestamp),
  #     'X-BAPI-RECV-WINDOW': str(self.recv_window),
  #     'X-BAPI-SIGN': signature
  #   }
  #
  #   url = f"{self.base_url}{endpoint}"
  #
  #   try:
  #     if method.upper() == 'GET':
  #       response = requests.get(url, headers=headers, params=params)
  #     elif method.upper() == 'POST':
  #       headers['Content-Type'] = 'application/json'
  #       response = requests.post(url, headers=headers, data=json.dumps(params))
  #     else:
  #       logger.error(f"Неподдерживаемый HTTP метод: {method}")
  #       return None
  #
  #     response.raise_for_status()
  #     data = response.json()
  #
  #     if data.get('retCode') == 0:
  #       logger.debug(f"Успешный запрос {method} к {endpoint}. Результат: {data.get('retMsg')}")
  #       return data.get('result', {})
  #     else:
  #       logger.error(f"Ошибка API Bybit (Код: {data.get('retCode')}): {data.get('retMsg')}")
  #       logger.debug(f"Параметры запроса: {params}, Подписанная строка: {sign_payload_str}")
  #       return None
  #   except requests.exceptions.HTTPError as http_err:
  #     logger.error(f"HTTP ошибка при запросе к {url}: {http_err}")
  #     if 'response' in locals():
  #       logger.error(f"Тело ответа: {response.text}")
  #     logger.debug(f"Параметры запроса: {params}, Подписанная строка: {sign_payload_str}")
  #     return None
  #   except Exception as e:
  #     logger.error(f"Ошибка при выполнении запроса к {url}: {e}")
  #     logger.debug(f"Параметры запроса: {params}, Подписанная строка: {sign_payload_str}")
  #     return None
  #
  # def get_server_time(self) -> Optional[int]:
  #   """Получить текущее время сервера Bybit (в миллисекундах)"""
  #   result = self._make_request('GET', '/v5/market/time')
  #   return int(result['timeNano']) // 1_000_000 if result and 'timeNano' in result else None
  #
  # def get_usdt_perpetual_contracts(self) -> List[Dict[str, Any]]:
  #   """Получает список всех бессрочных USDT контрактов."""
  #   endpoint = "/v5/market/tickers"
  #   params = {'category': 'linear'}  # 'linear' для USDT-M контрактов
  #   result = self._make_request('GET', endpoint, params)
  #   return result.get('list', []) if result else []

  async def get_kline(self, symbol: str, interval: str, limit: int = 200,
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[List[Any]]:
        """
        Получает исторические данные K-line (свечи)
        interval: '1m', '5m', '15m', '1h', '4h', '1d'
        """
        params = {'category': 'linear'}
        if start_time:
          params['since'] = start_time
        if end_time:
          params['until'] = end_time

        try:
          return await self.exchange.fetch_ohlcv(
            symbol,
            timeframe=interval,
            limit=limit,
            params=params
          )
        except Exception as e:
          logger.error(f"Ошибка получения свечей для {symbol}: {e}")
          return []

  async def get_account_balance(self, account_type: str = "UNIFIED", coin: str = "USDT") -> Optional[Dict]:
      """Получает баланс аккаунта для указанной монеты"""
      try:
        balance = await self.exchange.fetch_balance(params={
          'accountType': account_type,
          'coin': coin
        })

        # Обработка баланса для Unified аккаунтов
        if 'info' in balance and 'result' in balance['info'] and 'list' in balance['info']['result']:
          for acc_info in balance['info']['result']['list']:
            if acc_info.get('accountType') == account_type:
              for coin_balance in acc_info.get('coin', []):
                if coin_balance.get('coin') == coin:
                  return {
                    'free': float(coin_balance.get('availableToWithdraw', 0)),
                    'used': float(coin_balance.get('usedMargin', 0)),
                    'total': float(coin_balance.get('walletBalance', 0))
                  }

        # Стандартная обработка для не-Unified аккаунтов
        if coin in balance:
          return balance[coin]

        logger.warning(f"Баланс для {coin} не найден в ответе")
        return None
      except Exception as e:
        logger.error(f"Ошибка получения баланса: {e}")
        return None

  async def set_leverage(self, symbol: str, buy_leverage: int, sell_leverage: int) -> bool:
        """Устанавливает кредитное плечо для символа"""
        try:
          # Проверяем текущее плечо
          positions = await self.exchange.fetch_positions([symbol], params={'category': 'linear'})
          current_leverage = None

          for position in positions:
            if position['symbol'] == symbol:
              current_leverage = position['leverage']
              break

          # Если текущее плечо уже соответствует - пропускаем
          if current_leverage and float(current_leverage) == float(buy_leverage):
            logger.info(f"Плечо для {symbol} уже установлено на {buy_leverage}")
            return True

          # Устанавливаем новое плечо
          await self.exchange.set_leverage(
            buy_leverage,
            symbol,
            params={
              'category': 'linear',
              'buyLeverage': str(buy_leverage),
              'sellLeverage': str(sell_leverage)
            }
          )
          logger.info(f"Установлено плечо для {symbol}: buy={buy_leverage}, sell={sell_leverage}")
          return True
        except ccxt.BadRequest as e:
          if "leverage not modified" in str(e):
            logger.info(f"Плечо для {symbol} уже установлено на {buy_leverage}")
            return True
          logger.error(f"Ошибка установки плеча: {e}")
          return False
        except Exception as e:
          logger.error(f"Ошибка установки плеча: {e}")
          return False

  async def place_order(self, symbol: str, order_type: str, side: str,
                        amount: float, price: Optional[float] = None, params: Optional[dict] = None) -> dict:
    """Размещает ордер на бирже"""
    try:
      # Загрузка информации о рынке при необходимости
      if not self.market_info_cache or symbol not in self.market_info_cache:
        await self.exchange.load_markets()
        self.market_info_cache = self.exchange.markets

      market = self.market_info_cache[symbol]

      # Получаем ограничения и точности
      precision_amount = market['precision']['amount']
      min_amount = market['limits']['amount']['min']

      # Корректируем количество
      adjusted_amount = max(amount, min_amount)
      adjusted_amount = self.exchange.amount_to_precision(symbol, adjusted_amount)

      # Корректируем цену если нужно
      if price:
        price = self.exchange.price_to_precision(symbol, price)

      # Создаем ордер
      order = await self.exchange.create_order(
        symbol=symbol,
        type=order_type,
        side=side,
        amount=float(adjusted_amount),
        price=float(price) if price else None,
        params=params or {}
      )

      logger.info(f"Ордер размещен: {order['id']} ({side} {adjusted_amount} {symbol} @ {price or 'market'})")
      return {
        'status': 'success',
        'order_id': order['id'],
        'symbol': symbol,
        'side': side,
        'amount': float(adjusted_amount),
        'price': float(price) if price else None
      }
    except ccxt.InsufficientFunds as e:
      logger.error(f"Недостаточно средств: {e}")
      return {'status': 'error', 'message': 'Недостаточно средств'}
    except ccxt.InvalidOrder as e:
      logger.error(f"Неверный ордер: {e}")
      return {'status': 'error', 'message': 'Неверные параметры ордера'}
    except Exception as e:
      logger.error(f"Ошибка размещения ордера: {e}")
      return {'status': 'error', 'message': str(e)}

  async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[dict]:
    """Получает список открытых ордеров"""
    try:
      return await self.exchange.fetch_open_orders(
        symbol=symbol,
        params={'category': 'linear'}
      )
    except Exception as e:
      logger.error(f"Ошибка получения открытых ордеров: {e}")
      return []

  async def fetch_positions(self, symbols: Optional[List[str]] = None) -> List[dict]:
    """Получает открытые позиции"""
    try:
      positions = await self.exchange.fetch_positions(
        symbols=symbols,
        params={'category': 'linear'}
      )
      # Фильтруем только активные позиции
      return [p for p in positions if p['contracts'] and float(p['contracts']) > 0]
    except Exception as e:
      logger.error(f"Ошибка получения позиций: {e}")
      return []

  async def get_order_book(self, symbol: str, depth: int = 25) -> dict:
    """Получает стакан ордеров"""
    try:
      return await self.exchange.fetch_order_book(symbol, limit=depth)
    except Exception as e:
      logger.error(f"Ошибка получения стакана для {symbol}: {e}")
      return {'bids': [], 'asks': []}
  # def get_kline(self, symbol: str, interval: str, limit: int = 200, start_time: Optional[int] = None,
  #               end_time: Optional[int] = None) -> List[List[Any]]:
  #   """
  #   Получает исторические данные K-line (свечи).
  #   interval: '1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'W', 'M'
  #   limit: Макс 1000 для V5
  #   start_time, end_time: в миллисекундах
  #   """
  #   endpoint = "/v5/market/kline"
  #   params = {
  #     'category': 'linear',
  #     'symbol': symbol,
  #     'interval': interval,
  #     'limit': limit
  #   }
  #   if start_time:
  #     params['start'] = start_time
  #   if end_time:
  #     params['end'] = end_time
  #
  #   result = self._make_request('GET', endpoint, params)
  #   # Данные приходят в формате [timestamp, open, high, low, close, volume, turnover]
  #   return result.get('list', []) if result else []
#---------------------------------------
#   def get_account_balance(self, account_type: str = "CONTRACT", coin: str = "USDT") -> Optional[Dict]:
#     """
#     Получает баланс аккаунта.
#     account_type: UNIFIED, CONTRACT, SPOT
#     """
#     endpoint = "/v5/account/wallet-balance"
#     params = {'accountType': account_type}
#     if coin:  # Можно указать конкретную монету, например USDT
#       params['coin'] = coin
#
#     result = self._make_request('GET', endpoint, params)
#     if result and result.get('list'):
#       # Обычно list содержит один элемент с информацией о балансе
#       # или несколько, если не указана монета 'coin'
#       for balance_info in result['list']:
#         if account_type == "CONTRACT" and coin:  # Для контрактного счета ищем по монете
#           if balance_info.get('coin') == [coin]:  # coin это список в ответе
#             return balance_info
#         elif account_type == "CONTRACT":  # Если монета не указана, но тип счета CONTRACT
#           # Вернем первый, обычно там основной баланс или сумма по всем
#           return balance_info
#           # Можно добавить логику для других типов счетов, если нужно
#       if result['list']:  # Если ничего не найдено по специфичным условиям, но список не пуст
#         logger.warning(f"Не найден баланс для {coin} в {account_type}, возвращаю первый элемент из списка.")
#         return result['list'][0]
#     logger.error(f"Не удалось получить баланс для {account_type} {coin}. Ответ: {result}")
#     return None
# #-----------------------------------------
#
#   # async def get_account_balance(self) -> Optional[Dict]:
#   #   """Получает баланс аккаунта (USDT)."""
#   #   if not self.exchange or not self.api_key:  # Баланс требует аутентификации
#   #     logger.error("CCXT exchange не инициализирован или отсутствуют API ключи для получения баланса.")
#   #     return None
#   #   try:
#   #     logger.info("Запрос баланса аккаунта.")
#   #     balance = await self.exchange.fetch_balance(
#   #       params={'accountType': 'UNIFIED', 'coin': 'USDT'})  # или 'CONTRACT' для старых аккаунтов
#   #     # Для Unified Trading Account (UTA) структура ответа может отличаться.
#   #     # Нужно найти USDT в общем списке или в секции контрактов.
#   #     # logger.debug(f"Полный ответ по балансу: {balance}")
#   #
#   #     # Попробуем найти USDT баланс
#   #     if 'USDT' in balance:
#   #       usdt_balance = balance['USDT']
#   #       logger.info(
#   #         f"Баланс USDT: Total={usdt_balance.get('total')}, Free={usdt_balance.get('free')}, Used={usdt_balance.get('used')}")
#   #       return usdt_balance
#   #     elif 'info' in balance and 'result' in balance['info'] and 'list' in balance['info']['result']:
#   #       # Для Unified аккаунтов
#   #       for acc_info in balance['info']['result']['list']:
#   #         if acc_info.get('accountType') == 'UNIFIED' or acc_info.get('accountType') == 'CONTRACT':
#   #           for coin_balance in acc_info.get('coin', []):
#   #             if coin_balance.get('coin') == 'USDT':
#   #               usdt_data = {
#   #                 'free': coin_balance.get('availableToWithdraw'),  # или 'availableToBorrow'
#   #                 'used': coin_balance.get('usedMargin'),  # Примерное поле
#   #                 'total': coin_balance.get('walletBalance')
#   #               }
#   #               logger.info(f"Баланс USDT (Unified): {usdt_data}")
#   #               return usdt_data
#   #     logger.warning(f"Не удалось извлечь баланс USDT из ответа: {balance}")
#   #     return None  # Или вернуть весь объект balance для дальнейшего анализа
#   #   except Exception as e:
#   #     logger.error(f"Ошибка CCXT при получении баланса: {e}", exc_info=True)
#   #     return None
#
#
#   def set_leverage(self, symbol: str, buy_leverage: str, sell_leverage: str) -> Optional[Dict]:
#     """Устанавливает кредитное плечо для символа."""
#     endpoint = "/v5/position/set-leverage"
#     params = {
#       'category': 'linear',
#       'symbol': symbol,
#       'buyLeverage': str(buy_leverage),
#       'sellLeverage': str(sell_leverage)
#     }
#     return self._make_request('POST', endpoint, params)

  # Другие методы для размещения/отмены ордеров, получения позиций и т.д. будут добавлены позже.

  async def get_server_time(self) -> Optional[int]:
    """Получает время сервера Bybit (v5 API)."""
    if not self.exchange: return None  # Используем CCXT если доступно
    try:
      server_time = await self.exchange.fetch_time()
      logger.info(f"Время сервера Bybit (CCXT): {server_time}")
      return server_time
    except Exception as e_ccxt:
      logger.warning(f"Не удалось получить время сервера через CCXT: {e_ccxt}. Попытка через прямой запрос.")
      # Запасной вариант через прямой запрос
      response = await self._request("GET", "/v5/market/time")
      if response and "timeNano" in response:  # или timeSecond
        server_time_ms = int(response["timeNano"]) // 1_000_000
        logger.info(f"Время сервера Bybit (прямой запрос): {server_time_ms}")
        return server_time_ms
      logger.error("Не удалось получить время сервера Bybit.")
      return None