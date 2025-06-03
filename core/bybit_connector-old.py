import asyncio
import logging

import ccxt.async_support as ccxt
import time
import hashlib
import hmac
import json
from typing import Optional, Dict, Any, List
import math

import aiohttp
from config import api_keys, settings, trading_params
from config.api_keys import BYBIT_API_KEY, BYBIT_API_SECRET, USE_TESTNET
from config.settings import BYBIT_API_URL, BYBIT_CATEGORY, BYBIT_WS_PUBLIC_URL
from utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


class BybitConnector:
  def __init__(self, api_key: Optional[str] = BYBIT_API_KEY, api_secret: Optional[str] = BYBIT_API_SECRET):
    self.api_key = api_key
    self.api_secret = api_secret
    self.exchange = None  # Для CCXT
    self.logger = logging.getLogger(__name__)

    if not self.api_key or not self.api_secret:
      logger.error("API ключ или секрет не предоставлены. Функционал, требующий аутентификации, будет недоступен.")
      # Возможно, стоит выбросить исключение или ограничить функционал
      # For now, CCXT will operate in public mode if keys are missing

    # Инициализация CCXT
    # CCXT сам разберется с testnet/mainnet на основе URL или опций
    exchange_params = {
      'apiKey': self.api_key,
      'secret': self.api_secret,
      'enableRateLimit': True,
      'timeout': 30000,
      'options': {
        'defaultType': 'linear',
        'adjustForTimeDifference': True,
        'recvWindow': 20000  # <- вот оно!
      },
      'defaultType': 'linear',  # продублируем, на всякий случай
    }
    if USE_TESTNET:
      exchange_params['urls'] = {'api': BYBIT_API_URL}  # CCXT может требовать это для testnet

    self.exchange = ccxt.bybit(exchange_params)

    # Для прямых запросов, если CCXT не покрывает что-то
    self.client_session: Optional[aiohttp.ClientSession] = None
    logger.info("BybitConnector инициализирован.")
    if USE_TESTNET:
      logger.warning("BybitConnector работает с TESTNET API Bybit.")

    masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}" if self.api_key else "NOT SET"
    masked_secret = f"{self.api_secret[:4]}...{self.api_secret[-4:]}" if self.api_secret else "NOT SET"
    logger.info(f"Настройка BybitConnector:")
    logger.info(f"  USE_TESTNET: {USE_TESTNET}")
    logger.info(f"  BYBIT_API_URL: {BYBIT_API_URL}")
    logger.info(f"  API_KEY: {masked_key}")
    logger.info(f"  API_SECRET: {masked_secret}")


  async def init_session(self):
      """Инициализирует aiohttp ClientSession и синхронизирует время."""
      if self.client_session is None or self.client_session.closed:
        self.client_session = aiohttp.ClientSession(
          headers={"Content-Type": "application/json"}
        )
        logger.info("aiohttp ClientSession инициализирована.")

      # Синхронизируем время CCXT относительно сервера Bybit
      try:
        server_ts = await self.get_server_time()  # в миллисекундах
        local_ts = int(time.time() * 1000)
        time_diff = server_ts - local_ts
        # Устанавливаем опцию CCXT
        if self.exchange and hasattr(self.exchange, 'options'):
          self.exchange.options['timeDifference'] = time_diff
          logger.info(f"Синхронизация времени: timeDifference = {time_diff} ms")
      except Exception as e:
        logger.warning(f"Не удалось синхронизировать время: {e}")

  async def close_session(self):
    """Закрывает aiohttp ClientSession и CCXT exchange."""
    try:
      # Закрываем ccxt exchange
      if self.exchange:
        await self.exchange.close()
        self.exchange = None
        print("✅ Ресурсы ccxt exchange закрыты")

      # Закрываем aiohttp сессию
      if self.session and not self.session.closed:
        await self.session.close()
        print("✅ aiohttp сессия закрыта")
    except Exception as e:
      print(f"❌ Ошибка при закрытии ресурсов: {e}")

  def _generate_signature(self, params_str: str, recv_window: int = 5000) -> str:
    """Генерирует подпись для Bybit API (v5)."""
    if not self.api_secret:
      logger.error("API secret не установлен. Невозможно сгенерировать подпись.")
      return ""
    timestamp = str(int(time.time() * 1000))
    # Для V5 API, строка для подписи: timestamp + api_key + recv_window + params_str
    # params_str - это query string для GET или JSON body для POST
    # В документации Bybit V5 указан конкретный порядок для конкатенации параметров
    # В данном примере, для простоты, предполагаем что params_str уже сформирован как query string или body
    # Для GET: timestamp + apiKey + recvWindow + queryString
    # Для POST: timestamp + apiKey + recvWindow + requestBody
    message = timestamp + self.api_key + str(recv_window) + params_str
    signature = hmac.new(
      bytes(self.api_secret, "utf-8"),
      bytes(message, "utf-8"),
      hashlib.sha256,
    ).hexdigest()
    return signature

  async def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, signed: bool = False) -> Optional[
    Dict]:
    """
    Осуществляет асинхронный HTTP запрос к Bybit API v5.
    CCXT обычно справляется с этим лучше и проще. Этот метод - для примера прямого запроса.
    """
    await self.init_session()
    if not self.client_session:
      logger.error("aiohttp ClientSession не инициализирована для запроса.")
      return None

    url = f"{BYBIT_API_URL}{endpoint}"
    headers = {
      "Content-Type": "application/json",
      "Accept": "application/json",
    }
    req_params = params if params else {}
    data_str = ""

    if signed:
      if not self.api_key or not self.api_secret:
        logger.error("API ключ/секрет не установлены для подписанного запроса.")
        return None

      timestamp = str(int(time.time() * 1000))
      recv_window = "20000"  # Bybit рекомендует 5000-20000ms

      if method.upper() == "GET":
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(req_params.items())])
        data_str = query_string
        if query_string:
          url += '?' + query_string
      elif method.upper() == "POST":
        data_str = json.dumps(req_params) if req_params else ""

      # Формируем строку для подписи согласно документации Bybit API v5
      # message = timestamp + self.api_key + recv_window + data_str (упрощенно, см. документацию)
      # Для v5: timestamp + apiKey + recvWindow + (queryString ИЛИ requestBody)

      # Для POST с json body:
      # payload_string = json.dumps(params, separators=(',', ':')) if params else ""
      # Для GET:
      # payload_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())]) if params else ""
      # string_to_sign = timestamp + self.api_key + recv_window + payload_string

      # Для GET запросов параметры должны быть в query string, для POST - в теле запроса
      if method.upper() == "GET":
        param_string_for_sign = '&'.join([f"{k}={v}" for k, v in sorted(req_params.items())]) if req_params else ""
      else:  # POST
        param_string_for_sign = json.dumps(req_params) if req_params else ""

      to_sign = f"{timestamp}{self.api_key}{recv_window}{param_string_for_sign}"
      signature = hmac.new(bytes(self.api_secret, 'utf-8'), bytes(to_sign, 'utf-8'), hashlib.sha256).hexdigest()

      headers.update({
        "X-BAPI-API-KEY": self.api_key,
        "X-BAPI-SIGN": signature,
        "X-BAPI-SIGN-TYPE": "2",
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
      })

    try:
      logger.debug(f"Запрос: {method} {url} | Params: {req_params} | Headers: {headers}")
      async with self.client_session.request(method, url, json=req_params if method.upper() == "POST" else None,
                                             params=req_params if method.upper() == "GET" else None,
                                             headers=headers) as response:
        response_text = await response.text()
        logger.debug(
          f"Ответ от {url}: {response.status} {response_text[:300]}")  # Логируем только начало длинных ответов
        response.raise_for_status()  # Вызовет исключение для 4xx/5xx
        data = await response.json()
        if data.get("retCode") == 0:
          return data.get("result", data)  # Bybit API v5
        else:
          logger.error(f"Ошибка API Bybit: {data.get('retMsg')} (Код: {data.get('retCode')}) | {data}")
          return None
    except aiohttp.ClientResponseError as e:
      logger.error(f"Ошибка HTTP запроса к {url}: {e.status} {e.message} {e.headers}")
      logger.error(f"Тело ответа при ошибке: {await response.text() if 'response' in locals() else 'Нет ответа'}")
    except aiohttp.ClientError as e:
      logger.error(f"Ошибка клиента aiohttp при запросе к {url}: {e}")
    except Exception as e:
      logger.error(f"Непредвиденная ошибка при запросе к {url}: {e}", exc_info=True)
    return None
  # async def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, signed: bool = False) -> Optional[Dict]:
  #   if not self.exchange:
  #     logger.error("CCXT exchange не инициализирован для _request.")
  #     return None
  #
  #   bucket = 'private' if signed else 'public'
  #   path = endpoint.lstrip('/')  # CCXT сам добавит ведущий '/'
  #   request_params = params or {}
  #
  #   try:
  #     response = await self.exchange.request(
  #       path,
  #       bucket,
  #       method.upper(),
  #       request_params
  #     )
  #   except Exception as e:
  #     logger.error(f"Ошибка CCXT при запросе {endpoint}: {e}", exc_info=True)
  #     return None
  #
  #   # Если это стандартный v5-ответ Bybit
  #   if isinstance(response, dict) and 'retCode' in response:
  #     if response['retCode'] != 0:
  #       # Тут ловим реальные ошибки API
  #       logger.error(f"Ошибка API Bybit [{endpoint}]: {response.get('retMsg')} (retCode={response.get('retCode')})")
  #       return None
  #     # Успешный запрос — отдаём секцию result целиком
  #     return response.get('result', {})
  #
  #   # Если структура нетипичная — просто вернём всё, как есть
  #   return response

  # --- Методы CCXT (предпочтительны) ---
  async def fetch_tickers(self, symbols: Optional[List[str]] = None) -> Optional[Dict]:
    """Получает тикеры для указанных символов или всех, если symbols is None."""
    if not self.exchange:
      logger.error("CCXT exchange не инициализирован.")
      return None
    try:
      logger.info(f"Запрос тикеров для символов: {symbols if symbols else 'все USDT Perpetual'}")
      # Bybit API v5: /v5/market/tickers, category=linear
      tickers = await self.exchange.fetch_tickers(symbols=symbols, params={'category': BYBIT_CATEGORY})
      logger.info(f"Получено {len(tickers)} тикеров.")
      return tickers
    except ccxt.NetworkError as e:
      logger.error(f"CCXT NetworkError при получении тикеров: {e}", exc_info=True)
    except ccxt.ExchangeError as e:
      logger.error(f"CCXT ExchangeError при получении тикеров: {e}", exc_info=True)
    except Exception as e:
      logger.error(f"Непредвиденная ошибка CCXT при получении тикеров: {e}", exc_info=True)
    return None

  async def fetch_ohlcv(self, symbol: str, timeframe: str = '5m', since: Optional[int] = None,
                        limit: Optional[int] = None) -> List:
    """Получает исторические данные OHLCV (свечи)."""
    if not self.exchange:
      logger.error("CCXT exchange не инициализирован.")
      return []
    try:
      logger.debug(f"Запрос OHLCV для {symbol}, timeframe {timeframe}, limit {limit}")
      # Для Bybit USDT Perpetual, ccxt сам выберет правильный 'type'
      ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit,
                                              params={'category': BYBIT_CATEGORY})
      logger.debug(f"Получено {len(ohlcv)} свечей для {symbol}.")
      return ohlcv
    except Exception as e:
      logger.error(f"Ошибка CCXT при получении OHLCV для {symbol}: {e}", exc_info=True)
      return []

  async def get_account_balance(self) -> Optional[Dict]:
    """Получает баланс аккаунта (USDT)."""
    if not self.exchange or not self.api_key:  # Баланс требует аутентификации
      logger.error("CCXT exchange не инициализирован или отсутствуют API ключи для получения баланса.")
      return None
    try:
      logger.info("Запрос баланса аккаунта.")
      balance = await self.exchange.fetch_balance(
        params={'accountType': 'UNIFIED', 'coin': 'USDT'})  # или 'CONTRACT' для старых аккаунтов
      # Для Unified Trading Account (UTA) структура ответа может отличаться.
      # Нужно найти USDT в общем списке или в секции контрактов.
      # logger.debug(f"Полный ответ по балансу: {balance}")

      # Попробуем найти USDT баланс
      if 'USDT' in balance:
        usdt_balance = balance['USDT']
        logger.info(
          f"Баланс USDT: Total={usdt_balance.get('total')}, Free={usdt_balance.get('free')}, Used={usdt_balance.get('used')}")
        return usdt_balance
      elif 'info' in balance and 'result' in balance['info'] and 'list' in balance['info']['result']:
        # Для Unified аккаунтов
        for acc_info in balance['info']['result']['list']:
          if acc_info.get('accountType') == 'UNIFIED' or acc_info.get('accountType') == 'CONTRACT':
            for coin_balance in acc_info.get('coin', []):
              if coin_balance.get('coin') == 'USDT':
                usdt_data = {
                  'free': coin_balance.get('availableToWithdraw'),  # или 'availableToBorrow'
                  'used': coin_balance.get('usedMargin'),  # Примерное поле
                  'total': coin_balance.get('walletBalance')
                }
                logger.info(f"Баланс USDT (Unified): {usdt_data}")
                return usdt_data
      logger.warning(f"Не удалось извлечь баланс USDT из ответа: {balance}")
      return None  # Или вернуть весь объект balance для дальнейшего анализа
    except Exception as e:
      logger.error(f"Ошибка CCXT при получении баланса: {e}", exc_info=True)
      return None

  async def place_order(self, symbol: str, order_type: str, side: str,
                      amount: float, price: float = None, params: dict = None):
    try:
        # Загрузка market info с кэшированием
        if not hasattr(self, 'market_info_cache') or not self.market_info_cache:
            await self.exchange.load_markets()
            self.market_info_cache = self.exchange.markets

        market = self.market_info_cache.get(symbol) or self.exchange.market(symbol)

        # Получаем ограничения и точности
        precision_amount = market['precision'].get('amount', 8)
        precision_price = market['precision'].get('price', 8)
        min_amount = market['limits']['amount'].get('min', 0)
        min_cost = market['limits']['cost'].get('min', 0)

        # Округляем вверх: сначала до min_amount, затем до precision
        raw_amount = max(amount, min_amount)
        factor = 10 ** precision_amount
        adjusted_amount = math.ceil(raw_amount * factor) / factor

        # Безопасность: если после округления всё ещё меньше min_amount — поднимаем явно
        if adjusted_amount < min_amount:
            adjusted_amount = min_amount

        # Проверка min_cost (стоимость ордера)
        if price is not None and (adjusted_amount * price) < min_cost:
            new_amount = math.ceil((min_cost / price) * factor) / factor
            logger.warning(f"[{symbol}] Стоимость {adjusted_amount * price:.4f} < min_cost {min_cost}. Повышаем объём до {new_amount}")
            adjusted_amount = max(adjusted_amount, new_amount)

        # Приведение к точности CCXT
        amount_precise = self.exchange.amount_to_precision(symbol, adjusted_amount)
        if price is not None:
            price = self.exchange.price_to_precision(symbol, price)

        final_params = params or {}

        # Отправка ордера
        order = await self.exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=float(amount_precise),
            price=float(price) if price else None,
            params=final_params
        )

        logger.info(f"[{symbol}] ✅ Ордер размещен: {side.upper()} {amount_precise} @ {price or 'market'}")
        return {"status": "success", "order": order}

    except ccxt.InsufficientFunds as e:
        msg = f"[{symbol}] ❌ Недостаточно средств: {e}"
        logger.error(msg)
        raise RuntimeError("Недостаточно средств на маржинальном счёте для размещения ордера.") from e

    except ccxt.InvalidOrder as e:
        msg = f"[{symbol}] ❗ Неверный ордер: {e}"
        logger.error(msg)
        return {"error": msg}

    except ccxt.BaseError as e:
        msg = f"[{symbol}] ⚠️ Ошибка биржи: {e}"
        logger.error(msg)
        return {"error": msg}

    except Exception as e:
        msg = f"[{symbol}] 🚨 Неизвестная ошибка при размещении ордера: {e}"
        logger.error(msg, exc_info=True)
        return {"error": msg}

  async def fetch_open_orders(self, symbol: Optional[str] = None) -> List:
    """Получает список открытых ордеров."""
    if not self.exchange or not self.api_key:
      logger.error("CCXT exchange не инициализирован или отсутствуют API ключи для получения открытых ордеров.")
      return []
    try:
      logger.info(f"Запрос открытых ордеров для {symbol if symbol else 'всех символов'}")
      # Bybit требует 'category' в params
      open_orders = await self.exchange.fetch_open_orders(symbol=symbol,
                                                          params={'category': BYBIT_CATEGORY, 'settleCoin': 'USDT'})
      logger.info(f"Получено {len(open_orders)} открытых ордеров.")
      return open_orders
    except Exception as e:
      logger.error(f"Ошибка CCXT при получении открытых ордеров: {e}", exc_info=True)
      return []

  async def fetch_positions(self, symbols: Optional[List[str]] = None) -> List:
    """Получает открытые позиции."""
    if not self.exchange or not self.api_key:
      logger.error("CCXT exchange не инициализирован или отсутствуют API ключи для получения позиций.")
      return []
    try:
      logger.info(f"Запрос открытых позиций для {symbols if symbols else 'всех символов'}")
      # Для Bybit, fetchPositions может потребовать указания категории
      positions = await self.exchange.fetch_positions(symbols=symbols,
                                                      params={'category': BYBIT_CATEGORY, 'settleCoin': 'USDT'})
      # Фильтруем позиции с ненулевым размером контрактов
      active_positions = [p for p in positions if p.get('contracts') is not None and float(p.get('contracts', 0)) != 0]
      logger.info(f"Получено {len(active_positions)} активных позиций.")
      return active_positions
    except Exception as e:
      logger.error(f"Ошибка CCXT при получении позиций: {e}", exc_info=True)
      return []

  async def set_leverage(self, symbol: str, leverage: int, position_mode: int = 0) -> Optional[Dict]:
    """
    Устанавливает кредитное плечо для символа.
    position_mode: 0 - One-Way Mode, 3 - Hedge Mode (для Hedge Mode нужно указывать buyLeverage и sellLeverage)
    Bybit API v5: POST /v5/position/set-leverage
    """
    if not self.api_key or not self.api_secret:
      logger.error("API ключи не установлены. Невозможно установить кредитное плечо.")
      return None

    endpoint = "/v5/position/set-leverage"
    params = {
      "category": BYBIT_CATEGORY,
      "symbol": symbol,
      "buyLeverage": str(leverage),
      "sellLeverage": str(leverage),
    }
    logger.info(f"Установка кредитного плеча {leverage}x для {symbol}...")
    # CCXT может иметь свой метод set_leverage, проверим
    try:
      if hasattr(self.exchange, 'set_leverage') and callable(getattr(self.exchange, 'set_leverage')):
        # Некоторые версии CCXT могут поддерживать это унифицированно
        response = await self.exchange.set_leverage(leverage, symbol,
                                                    params={"category": BYBIT_CATEGORY, "buyLeverage": str(leverage),
                                                            "sellLeverage": str(leverage)})
        logger.info(f"Кредитное плечо для {symbol} установлено на {leverage}x через CCXT. Ответ: {response}")
        return response
      else:  # Используем прямой запрос, если CCXT метод не универсален
        response = await self._request("POST", endpoint, params, signed=True)
        if response:
          logger.info(f"Кредитное плечо для {symbol} установлено на {leverage}x. Ответ: {response}")
        else:
          logger.error(f"Не удалось установить кредитное плечо для {symbol}.")
        return response
    except ccxt.BadRequest as e:
        error_message = str(e)
        if "leverage not modified" in error_message:
          logger.info(f"Кредитное плечо {symbol} уже равно {leverage}x. Пропускаем.")
          return None  # или return {"status": "unchanged"}
        else:
          logger.error(f"Ошибка при установке кредитного плеча для {symbol}: {e}")
          raise

  async def fetch_order_book(self, symbol: str, depth: int = 25) -> Dict:
    """Полная реализация получения стакана"""
    try:
      # Прямой API запрос к Bybit v5
      endpoint = "/v5/market/orderbook"
      params = {
        "symbol": symbol,
        "category": BYBIT_CATEGORY,
        "limit": depth
      }

      response = await self._request("GET", endpoint, params)
      if not response or 'bids' not in response:
        raise ValueError("Invalid orderbook response")

      return {
        'bids': [[float(b[0]), float(b[1])] for b in response['bids']],
        'asks': [[float(a[0]), float(a[1])] for a in response['asks']],
        'timestamp': response.get('ts', int(time.time() * 1000))
      }

    except Exception as e:
      logger.error(f"Orderbook fetch error: {str(e)}")
      # Fallback через CCXT
      if hasattr(self, 'exchange'):
        return await self.exchange.fetch_order_book(symbol, limit=depth)
      raise

  # async def fetch_order_book(self, symbol: str, depth: int = 25) -> Dict[str, List]:
  #   """Получает стакан ордеров с биржи"""
  #   try:
  #     if not self.exchange:
  #       logger.error("CCXT exchange не инициализирован")
  #       return {'bids': [], 'asks': []}
  #
  #     orderbook = await self.exchange.fetch_order_book(symbol, limit=depth)
  #     return {
  #       'bids': [[price, amount] for price, amount in orderbook['bids']],
  #       'asks': [[price, amount] for price, amount in orderbook['asks']]
  #     }
  #   except Exception as e:
  #     logger.error(f"Ошибка получения стакана для {symbol}: {e}")
  #     return {'bids': [], 'asks': []}


  async def get_kline(self, symbol: str, interval: str, limit: int = 200) -> List[Dict]:
    """
    Получение исторических свечей (k-lines).
    interval: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
    Bybit API v5: GET /v5/market/kline
    """
    endpoint = "/v5/market/kline"
    params = {
      "category": BYBIT_CATEGORY,
      "symbol": symbol,
      "interval": interval,
      "limit": limit
    }
    logger.debug(f"Запрос Klines для {symbol}, интервал {interval}, лимит {limit}")
    response_data = await self._request("GET", endpoint, params)
    if response_data and "list" in response_data:
      # Формат ответа Bybit: [timestamp, open, high, low, close, volume, turnover]
      # Преобразуем в более удобный формат, если нужно, или оставим как есть
      klines = []
      for k in response_data["list"]:
        klines.append({
          "timestamp": int(k[0]),
          "open": float(k[1]),
          "high": float(k[2]),
          "low": float(k[3]),
          "close": float(k[4]),
          "volume": float(k[5]),
          "turnover": float(k[6])
        })
      logger.debug(f"Получено {len(klines)} свечей для {symbol}.")
      return klines
    logger.warning(f"Не удалось получить klines для {symbol}. Ответ: {response_data}")
    return []

  # --- WebSocket ---
  # WebSocket часть сложнее и требует управления подписками, переподключениями.
  # Для WebSocket лучше использовать библиотеку `websocket-client` или встроенные возможности `aiohttp`.
  # CCXT также предлагает поддержку WebSocket, но она может быть менее гибкой для кастомных нужд.

  async def subscribe_to_orderbook(self, symbol: str, callback):
    """Подписывается на обновления стакана ордеров для символа."""
    # Примерная реализация. Полная реализация WebSocket требует больше кода.
    # ws_url = f"{BYBIT_WS_PUBLIC_URL}" # stream.bybit.com/v5/public/linear
    ws_url = BYBIT_WS_PUBLIC_URL
    logger.info(f"Подключение к WebSocket для стакана: {symbol} по {ws_url}")

    try:
      async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
          # Аутентификация для приватных каналов (если нужно)
          # expires = int((time.time() + 60) * 1000) # +1 минута
          # signature_payload = f"GET/realtime{expires}"
          # signature = hmac.new(self.api_secret.encode("utf-8"), signature_payload.encode("utf-8"), hashlib.sha256).hexdigest()
          # auth_msg = {
          #     "op": "auth",
          #     "args": [self.api_key, expires, signature]
          # }
          # await ws.send_json(auth_msg)
          # auth_resp = await ws.receive_json()
          # logger.info(f"WebSocket Auth response: {auth_resp}")
          # if not auth_resp.get("success"):
          #     logger.error(f"WebSocket auth failed: {auth_resp}")
          #     return

          # Подписка на стакан (orderbook.50.SYMBOL)
          subscribe_msg = {
            "op": "subscribe",
            "args": [f"orderbook.50.{symbol}"]  # Глубина 50
          }
          await ws.send_json(subscribe_msg)
          logger.info(f"Отправлен запрос на подписку на стакан {symbol}")

          async for msg_raw in ws:
            if msg_raw.type == aiohttp.WSMsgType.TEXT:
              message = json.loads(msg_raw.data)
              # logger.debug(f"WebSocket сообщение (orderbook {symbol}): {message}")
              if "topic" in message and message["topic"].startswith(f"orderbook.50.{symbol}"):
                if message.get("type") == "snapshot":  # Первый полный снимок
                  logger.info(f"Получен snapshot стакана для {symbol}")
                  await callback(symbol, message["data"])
                elif message.get("type") == "delta":  # Обновления
                  await callback(symbol, message["data"])
              elif message.get("op") == "subscribe":
                logger.info(f"Подписка на WebSocket подтверждена: {message}")
              elif message.get("success") is False:
                logger.error(f"Ошибка WebSocket: {message}")

            elif msg_raw.type == aiohttp.WSMsgType.ERROR:
              logger.error(f"Ошибка WebSocket соединения: {ws.exception()}")
              break
            elif msg_raw.type == aiohttp.WSMsgType.CLOSED:
              logger.warning("WebSocket соединение закрыто.")
              break
    except Exception as e:
      logger.error(f"Ошибка WebSocket для {symbol}: {e}", exc_info=True)
      # Здесь нужна логика переподключения
      await asyncio.sleep(5)  # Пауза перед попыткой переподключения

  # Другие методы: get_server_time, get_instrument_info и т.д.
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

  async def get_instruments_info(self, category: str = BYBIT_CATEGORY, symbol: Optional[str] = None) -> List[Dict]:
    """
    Получает информацию об инструментах (контрактах).
    Bybit API v5: GET /v5/market/instruments-info
    """
    if not self.exchange: return []
    try:
      params = {'category': category}
      if symbol:
        params['symbol'] = symbol

      # CCXT `load_markets` обычно загружает эту информацию при инициализации,
      # но можно и принудительно обновить или запросить для конкретной категории.
      # response = await self.exchange.fetch_markets(params=params) # CCXT
      # return response # CCXT уже возвращает в стандартизированном формате

      # Прямой запрос, если нужен именно оригинальный ответ API
      endpoint = "/v5/market/instruments-info"
      api_params = {"category": category}
      if symbol:
        api_params["symbol"] = symbol

      response_data = await self._request("GET", endpoint, api_params)
      if response_data:
        # Если list лежит в result
        if "result" in response_data and "list" in response_data["result"]:
          instruments = response_data["result"]["list"]
        # Если list сразу в корне
        elif "list" in response_data:
          instruments = response_data["list"]
        else:
          logger.error(f"Не удалось найти список инструментов в ответе: {response_data}")
          return []

        logger.info(f"Получено {len(instruments)} инструментов для категории {category}.")
        return instruments
      else:
        logger.error(f"Не удалось получить информацию об инструментах для {category}. Ответ: {response_data}")
        return []

    except Exception as e:
      logger.error(f"Ошибка при получении информации об инструментах: {e}", exc_info=True)
      return []


# Пример использования (для тестирования модуля)
async def main_test_connector():
  connector = BybitConnector()
  await connector.init_session()

  server_time = await connector.get_server_time()
  logger.info(f"Текущее время сервера: {server_time}")

  # Перед использованием приватных методов (баланс, ордера), убедитесь, что API ключи верны
  if BYBIT_API_KEY and BYBIT_API_SECRET:
    balance = await connector.get_account_balance()
    if balance:
      logger.info(f"Баланс USDT: {balance.get('total', 'N/A')}")
    else:
      logger.warning("Не удалось получить баланс.")

    # Установка плеча (пример) - ОСТОРОЖНО С РЕАЛЬНЫМИ СРЕДСТВАМИ
    # symbol_to_set_leverage = "BTCUSDT"
    # leverage_set_result = await connector.set_leverage(symbol_to_set_leverage, config.LEVERAGE)
    # if leverage_set_result:
    #     logger.info(f"Результат установки плеча для {symbol_to_set_leverage}: {leverage_set_result}")

    # # Пример размещения ордера (ТЕСТНЕТ!) - ОСТОРОЖНО
    # if USE_TESTNET:
    #     test_symbol = "BTCUSDT" # Убедитесь, что этот символ торгуется на тестнете
    #     # Перед размещением ордера, нужно получить актуальную цену
    #     tickers_info = await connector.fetch_tickers([test_symbol])
    #     if tickers_info and test_symbol in tickers_info:
    #         current_price = float(tickers_info[test_symbol]['last'])
    #         # Пример лимитного ордера на покупку чуть ниже текущей цены
    #         order_price = current_price * 0.998 # на 0.2% ниже
    #         order_amount = 0.001 # Минимальное количество BTC для тестнета
    #         logger.info(f"Попытка разместить тестовый ордер на {test_symbol}: BUY {order_amount} @ {order_price:.2f}")
    #         # order = await connector.place_order(test_symbol, 'buy', 'limit', order_amount, order_price)
    #         # if order:
    #         #     logger.info(f"Тестовый ордер размещен: {order}")
    #         #     # Отмена ордера (если нужно сразу отменить)
    #         #     # await asyncio.sleep(5) # Дать ордеру время появиться
    #         #     # cancel_result = await connector.cancel_order(order['id'], test_symbol)
    #         #     # logger.info(f"Результат отмены тестового ордера: {cancel_result}")
    #         # else:
    #         #     logger.error("Не удалось разместить тестовый ордер.")
    #     else:
    #         logger.warning(f"Не удалось получить информацию о тикере для {test_symbol} для тестового ордера.")
    # else:
    #     logger.warning("Тестовое размещение ордеров отключено, т.к. USE_TESTNET=False.")

    open_orders = await connector.fetch_open_orders()  # Для всех символов
    logger.info(f"Открытые ордера: {len(open_orders)}")
    # for o_order in open_orders:
    #     logger.info(f"  ID: {o_order['id']}, Sym: {o_order['symbol']}, Side: {o_order['side']}, Price: {o_order['price']}, Amount: {o_order['amount']}")

    positions = await connector.fetch_positions()
    logger.info(f"Открытые позиции: {len(positions)}")
    # for pos in positions:
    #     logger.info(f"  Sym: {pos['symbol']}, Side: {pos.get('side')}, Size: {pos.get('contracts')}, Entry: {pos.get('entryPrice')}, PnL: {pos.get('unrealisedPnl')}")

  else:
    logger.warning("API ключи не настроены. Приватные методы API недоступны.")

  # Получение списка инструментов для мониторинга
  # instruments = await connector.get_instruments_info(category=BYBIT_CATEGORY)
  # linear_usdt_symbols = [inst['symbol'] for inst in instruments if inst['quoteCoin'] == 'USDT' and inst['status'] == 'Trading']
  # logger.info(f"Найдено {len(linear_usdt_symbols)} активных USDT Perpetual контрактов для мониторинга.")
  # logger.debug(f"Первые 5 символов: {linear_usdt_symbols[:5]}")

  # Тестирование WebSocket (пример)
  # async def my_orderbook_handler(symbol, data):
  #     logger.info(f"Обработчик стакана для {symbol}: Bids[0]={data['b'][0] if data['b'] else 'N/A'}, Asks[0]={data['a'][0] if data['a'] else 'N/A'}")

  # test_ws_symbol = "BTCUSDT"
  # logger.info(f"Запуск теста WebSocket для {test_ws_symbol}")
  # # Запускаем в фоне, т.к. это бесконечный цикл
  # asyncio.create_task(connector.subscribe_to_orderbook(test_ws_symbol, my_orderbook_handler))
  # await asyncio.sleep(30) # Даем поработать WebSocket 30 секунд
  # logger.info(f"Завершение теста WebSocket для {test_ws_symbol}")

  await connector.close_session()


if __name__ == "__main__":
  # Настройка логирования должна быть вызвана до создания логгеров в модулях


  setup_logging("DEBUG")  # Устанавливаем уровень DEBUG для детального вывода
  asyncio.run(main_test_connector())