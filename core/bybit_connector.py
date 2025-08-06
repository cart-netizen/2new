# core/bybit_connector.py

import hashlib
import hmac
import time
import json
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from core.circuit_breaker import circuit_breaker, TradingCircuitBreakers, CircuitBreakerOpenError
import aiohttp
import pandas as pd
from aiolimiter import AsyncLimiter

from config import api_keys, settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


class RequestBatcher:
  """Класс для батчинга однотипных запросов"""

  def __init__(self, batch_window: float = 0.1):
    self.batch_window = batch_window  # Окно батчинга в секундах
    self.pending_requests = defaultdict(list)
    self.batch_locks = defaultdict(asyncio.Lock)


  async def add_request(self, request_type: str, params: Dict) -> Any:
    """Добавляет запрос в батч и возвращает результат"""
    future = asyncio.Future()

    async with self.batch_locks[request_type]:
      self.pending_requests[request_type].append((params, future))

      # Если это первый запрос в батче, запускаем таймер
      if len(self.pending_requests[request_type]) == 1:
        asyncio.create_task(self._process_batch(request_type))

    return await future

  async def _process_batch(self, request_type: str):
    """Обрабатывает батч запросов после истечения окна"""
    await asyncio.sleep(self.batch_window)

    async with self.batch_locks[request_type]:
      batch = self.pending_requests[request_type]
      self.pending_requests[request_type] = []

    # Здесь логика обработки батча в зависимости от типа
    # Для примера - просто возвращаем результаты
    for params, future in batch:
      if not future.done():
        future.set_result(None)


class OptimizedSession:
  """Оптимизированная сессия с переиспользованием соединений"""

  def __init__(self, connector_limit: int = 100, timeout: int = 60):
    self.connector = aiohttp.TCPConnector(
      limit=connector_limit,
      limit_per_host=30,
      ttl_dns_cache=300,
      enable_cleanup_closed=True
    )
    self.timeout = aiohttp.ClientTimeout(total=timeout)
    self.session: Optional[aiohttp.ClientSession] = None

  async def get_session(self) -> aiohttp.ClientSession:
    """Получает или создает сессию"""
    if self.session is None or self.session.closed:
      self.session = aiohttp.ClientSession(
        connector=self.connector,
        timeout=self.timeout,
        headers={
          'User-Agent': 'Bybit-Trading-Bot/1.0',
          'Connection': 'keep-alive'
        }
      )
    return self.session

  async def close(self):
    """Закрывает сессию и коннектор"""
    if self.session and not self.session.closed:
      await self.session.close()
    await self.connector.close()

class BybitConnector:
  def __init__(self):
    self.api_key = api_keys.API_KEY
    self.api_secret = api_keys.API_SECRET
    self.base_url = settings.BYBIT_API_URL
    self.recv_window = 20000

    # Оптимизированная сессия
    self.optimized_session = OptimizedSession()

    # Управление временем
    self.time_offset = 0
    self.last_sync_time = 0
    self.sync_interval = 3600  # Синхронизация времени каждый час

    # Rate limiting с оптимизацией
    self.rate_limiter = AsyncLimiter(10, 1)  # Увеличили лимит для батчинга
    self.endpoint_limiters = {
      '/v5/order/create': AsyncLimiter(10, 1),
      '/v5/market/kline': AsyncLimiter(15, 1),
      '/v5/position/list': AsyncLimiter(15, 1),
      '/v5/position/set-leverage': AsyncLimiter(5, 1)
    }

    # Семафор для параллельных запросов
    self.semaphore = asyncio.Semaphore(8)  # Увеличили для большей параллельности

    # Батчер для однотипных запросов
    self.batcher = RequestBatcher()

    # Кэш для повторяющихся запросов
    self.request_cache = {}
    self.cache_ttl = 0  # 5 секунд для краткосрочного кэша
    self.exchange = None

    # Статистика
    self.request_stats = defaultdict(int)
    self.error_stats = defaultdict(int)

    TradingCircuitBreakers.setup_trading_breakers()

    self.default_category = "linear"

    if not self.api_key or "YOUR_" in self.api_key:
      logger.warning("API ключ Bybit не настроен или используется ключ-заглушка.")
    if not self.api_secret or "YOUR_" in self.api_secret:
      logger.warning("API секрет Bybit не настроен или используется секрет-заглушка.")

  async def _get_session(self) -> aiohttp.ClientSession:
    """Получает или создает асинхронную сессию."""
    if self._session is None or self._session.closed:
      self._session = aiohttp.ClientSession()
    return self._session

  async def sync_time(self, force: bool = False):
    """Синхронизирует локальное время с временем сервера Bybit"""
    current_time = time.time()

    # Синхронизируем только если прошло достаточно времени или force=True
    if not force and (current_time - self.last_sync_time) < self.sync_interval:
      return

    logger.info("Синхронизация времени с сервером Bybit...")
    try:
      server_time_url = self.base_url + "/v5/market/time"
      session = await self.optimized_session.get_session()

      async with session.get(server_time_url) as response:
        if response.status == 200:
          data = await response.json()
          server_time = int(data['result']['timeNano']) // 1_000_000
          local_time = int(time.time() * 1000)
          self.time_offset = server_time - local_time
          self.last_sync_time = current_time
          logger.info(f"Смещение времени установлено: {self.time_offset} мс.")
        else:
          logger.error("Не удалось получить время сервера Bybit.")
    except Exception as e:
      logger.error(f"Ошибка синхронизации времени: {e}")

  async def close(self):
    """Закрывает асинхронную сессию и выводит статистику"""
    await self.optimized_session.close()

    # Выводим статистику запросов
    if self.request_stats:
      logger.info("Статистика запросов:")
      for endpoint, count in sorted(self.request_stats.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {endpoint}: {count} запросов")

    if self.error_stats:
      logger.warning("Статистика ошибок:")
      for error_type, count in sorted(self.error_stats.items(), key=lambda x: x[1], reverse=True):
        logger.warning(f"  {error_type}: {count} ошибок")

  def _generate_signature(self, params_str: str) -> str:
    """Генерирует подпись для запроса"""
    return hmac.new(self.api_secret.encode('utf-8'), params_str.encode('utf-8'), hashlib.sha256).hexdigest()

  def _check_cache(self, cache_key: str) -> Optional[Dict]:
      """Проверяет кэш на наличие актуальных данных"""
      if cache_key in self.request_cache:
        cached_data, timestamp = self.request_cache[cache_key]
        if (time.time() - timestamp) < self.cache_ttl:
          return cached_data
        else:
          del self.request_cache[cache_key]
      return None

  def _set_cache(self, cache_key: str, data: Dict):
    """Сохраняет данные в кэш"""
    self.request_cache[cache_key] = (data, time.time())

  # --- ПУБЛИЧНЫЕ МЕТОДЫ (теперь все асинхронные) ---
  # async def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, treat_as_success: List[int] = None, is_retry: bool = False) -> Optional[Dict]:
  #   """
  #   ФИНАЛЬНАЯ ВЕРСИЯ №2: Правильно обрабатывает подписи для GET (сортировка параметров)
  #   и POST (сортировка ключей в JSON).
  #   """
  #   async with self.rate_limiter:
  #     async with self.semaphore:
  #       if treat_as_success is None:
  #         treat_as_success = []
  #
  #       if params is None:
  #         params = {}
  #
  #       if time.time() - self.last_sync_time > 600:
  #         logger.warning("Последняя синхронизация времени была давно. Запуск принудительной ресинхронизации...")
  #         await self.sync_time()
  #
  #       timestamp = str(int(time.time() * 1000) + self.time_offset)
  #       recv_window = "10000"
  #
  #
  #
  #       if method.upper() == "GET":
  #
  #         param_str_for_signature = "&".join([f"{k}={v}" for k, v in params.items()])
  #       else:  # POST
  #         # Для POST запросов используется тело запроса в формате JSON
  #         param_str_for_signature = json.dumps(params) if params else ""
  #
  #       signature_raw = timestamp + self.api_key + recv_window + param_str_for_signature
  #       signature = self._generate_signature(signature_raw)
  #
  #       headers = {
  #         'X-BAPI-API-KEY': self.api_key,
  #         'X-BAPI-TIMESTAMP': timestamp,
  #         'X-BAPI-SIGN': signature,
  #         'X-BAPI-RECV-WINDOW': recv_window,
  #         'Content-Type': 'application/json'
  #       }
  #
  #       url = self.base_url + endpoint
  #       try:
  #         session = await self._get_session()
  #         # Для POST-запросов передаем несортированные параметры, так как aiohttp сам их кодирует
  #         async with session.request(
  #             method.upper(), url, headers=headers,
  #             json=params if method.upper() == 'POST' else None,
  #             params=params if method.upper() == 'GET' else None
  #         ) as response:
  #           data = await response.json()
  #           # ret_code = data.get('retCode', -1)
  #           ret_code = data.get('retCode')
  #
  #           # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ ЗДЕСЬ ---
  #           # Считаем успехом, если код 0 ИЛИ если он в списке исключений
  #           if response.status == 200 and (ret_code == 0 or ret_code in treat_as_success):
  #             return data.get('result', {})
  #           else:
  #             # --- НОВАЯ ЛОГИКА АДАПТИВНОЙ РЕСИНХРОНИЗАЦИИ ---
  #             if ret_code == 10002 and not is_retry:
  #               logger.warning("Обнаружена ошибка синхронизации времени (10002). Запуск принудительной ресинхронизации...")
  #               await self.sync_time()
  #               logger.info("Повторная отправка запроса после ресинхронизации...")
  #               return await self._make_request(method, endpoint, params, is_retry=True)
  #             # --- КОНЕЦ НОВОЙ ЛОГИКИ ---
  #
  #             logger.error(
  #               f"Ошибка API Bybit (HTTP: {response.status}, Код: {ret_code} для {endpoint}): {data.get('retMsg')}")
  #             # logger.error(f"Ошибка API Bybit (Код: {ret_code} для {endpoint}): {data.get('retMsg')}")
  #             return None
  #
  #       except asyncio.TimeoutError:
  #         logger.error(f"Таймаут запроса к {endpoint}")
  #         return None
  #
  #       except aiohttp.ClientConnectionError as e:
  #         logger.error(f"Ошибка соединения: {e}. Попытка пересоздать сессию...")
  #         # Закрываем "мертвую" сессию и пытаемся выполнить запрос еще раз
  #         await self.close()
  #         # Небольшая пауза перед повторной попыткой
  #         await asyncio.sleep(1)
  #         if not is_retry:
  #           return await self._make_request(method, endpoint, params, is_retry=True)
  #
  #       except Exception as e:
  #         logger.error(f"Непредвиденная ошибка в _make_request при запросе к {endpoint}: {e}", exc_info=True)
  #         return None
  #

  async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        treat_as_success: List[int] = None,
        is_retry: bool = False,
        use_cache: bool = True
    ) -> Optional[Dict]:
      """
      Оптимизированная версия с кэшированием и батчингом
      """
      import time
      start_time = time.time()

      if treat_as_success is None:
        treat_as_success = []

      # Проверяем кэш для GET запросов
      cache_key = None
      if method == 'GET' and use_cache:
        cache_key = f"{endpoint}:{json.dumps(params or {}, sort_keys=True)}"
        cached_result = self._check_cache(cache_key)
        if cached_result is not None:
          self.request_stats[f"{endpoint} (cached)"] += 1
          logger.debug(f"📦 Используется кэш для {endpoint}")
          return cached_result
      elif not use_cache:
        # НОВОЕ: Если use_cache=False, принудительно удаляем любой существующий кэш
        temp_cache_key = f"{endpoint}:{json.dumps(params or {}, sort_keys=True)}"
        if temp_cache_key in self.request_cache:
          del self.request_cache[temp_cache_key]
          logger.debug(f"🗑️ Принудительно удален кэш для {endpoint}")

      # Применяем rate limiting для конкретного endpoint
      endpoint_limiter = self.endpoint_limiters.get(endpoint, self.rate_limiter)

      max_retries = 3
      base_delay = 1  # Задержка в секундах

      for attempt in range(max_retries):
        async with endpoint_limiter:
          async with self.semaphore:
            url = self.base_url + endpoint

            # Обновляем статистику
            self.request_stats[endpoint] += 1

            # Автоматическая синхронизация времени при необходимости
            if not is_retry:
              await self.sync_time()

            timestamp = str(int((time.time() + self.time_offset / 1000) * 1000))

            headers = {
              'X-BAPI-API-KEY': self.api_key,
              'X-BAPI-TIMESTAMP': timestamp,
              'X-BAPI-RECV-WINDOW': str(self.recv_window),
              'Content-Type': 'application/json'
            }

            # Подготовка параметров
            if params is None:
              params = {}

            # Генерация подписи
            if method == 'GET':
              # Для GET запросов НЕ сортируем, сохраняем порядок как в оригинале
              query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
              param_str = f"{timestamp}{self.api_key}{self.recv_window}{query_string}"
            else:  # POST
              # Для POST запросов используем JSON без сортировки
              body_str = json.dumps(params) if params else ""
              param_str = f"{timestamp}{self.api_key}{self.recv_window}{body_str}"

            signature = self._generate_signature(param_str)
            headers['X-BAPI-SIGN'] = signature

            try:
              session = await self.optimized_session.get_session()

              # Выполнение запроса с оптимизированными таймаутами
              request_timeout = aiohttp.ClientTimeout(total=30, connect=10)

              if method == 'GET':
                async with session.get(url, params=params, headers=headers, timeout=request_timeout) as response:
                  data = await response.json()
              else:  # POST
                async with session.post(url, json=params, headers=headers, timeout=request_timeout) as response:
                  data = await response.json()

              ret_code = data.get('retCode', -1)

              if ret_code == 0 or ret_code in treat_as_success:
                result = data.get('result', {})

                # Кэшируем успешный результат
                if cache_key and use_cache:
                  self._set_cache(cache_key, result)

                # Собираем статистику
                response_time_ms = (time.time() - start_time) * 1000
                self._increment_request_stats(success=True, response_time_ms=response_time_ms)

                return result

              # Обработка ошибок
              self.error_stats[f"API Error {ret_code}"] += 1

              if ret_code == 10002 and not is_retry:
                logger.warning(f"Ошибка времени для {endpoint}. Запуск принудительной ресинхронизации...")
                await self.sync_time(force=True)
                logger.info("Повторная отправка запроса после ресинхронизации...")
                return await self._make_request(method, endpoint, params, treat_as_success, is_retry=True,
                                                use_cache=use_cache)

              logger.error(
                f"Ошибка API Bybit (HTTP: {response.status}, Код: {ret_code} для {endpoint}): {data.get('retMsg')}")
              return None

            except asyncio.TimeoutError:
              self.error_stats["Timeout"] += 1
              logger.error(f"Таймаут запроса к {endpoint}")
              return None

            except aiohttp.ClientConnectionError as e:
              self.error_stats["Connection Error"] += 1
              logger.error(f"Ошибка соединения: {e}")

              if not is_retry:
                await asyncio.sleep(1)
                return await self._make_request(method, endpoint, params, treat_as_success, is_retry=True,
                                                use_cache=use_cache)
              return None

            except Exception as e:
              self.error_stats["Unknown Error"] += 1
              logger.error(f"Непредвиденная ошибка в _make_request при запросе к {endpoint}: {e}", exc_info=True)

              response_time_ms = (time.time() - start_time) * 1000
              self._increment_request_stats(success=False, response_time_ms=response_time_ms)
              return None

      return None  # Возврат None, если все попытки провалились

  async def get_usdt_perpetual_contracts(self) -> List[Dict[str, Any]]:
    """Получает список всех бессрочных USDT контрактов с кэшированием"""
    endpoint = "/v5/market/tickers"
    params = {'category': 'linear'}
    return (await self._make_request('GET', endpoint, params, use_cache=True) or {}).get('list', [])

  async def get_kline(self, symbol: str, interval: str, limit: int = 200, force_fresh: bool = False, **kwargs) -> List[
    List[Any]]:
    """
    Получает исторические данные K-line (свечи).
    ИСПРАВЛЕНО: Корректно обрабатывает 'end' для запроса исторических данных.
    """
    endpoint = "/v5/market/kline"

    params = {
      'category': 'linear',
      'symbol': symbol,
      'interval': interval,
      'limit': limit,
      **kwargs  # ИЗМЕНЕНО: Сначала применяем переданные аргументы
    }

    # ИЗМЕНЕНО: Добавляем 'end' только если он не был передан явно в kwargs
    if 'end' not in params:
      current_time_ms = int(time.time() * 1000)
      params['end'] = current_time_ms

    use_cache_setting = not force_fresh

    if force_fresh:
      self.clear_symbol_cache(symbol)

    result = await self._make_request('GET', endpoint, params, use_cache=use_cache_setting)

    # ДЕТАЛЬНАЯ ДИАГНОСТИКА ОТВЕТА API
    if result:
      # logger.info(f"🔍 ДЕТАЛЬНЫЙ ответ API для {symbol}:")
      # logger.info(f"  - Получен result: {type(result)}")

      if result.get('list'):
        api_data = result['list']
        # logger.info(f"  - Количество свечей: {len(api_data)}")

        # Анализируем первые и последние свечи
        if api_data:
          first_candle = api_data[0]
          last_candle = api_data[-1]

          first_timestamp = int(first_candle[0])
          last_timestamp = int(last_candle[0])

          first_time = datetime.fromtimestamp(first_timestamp / 1000)
          last_time = datetime.fromtimestamp(last_timestamp / 1000)
          #Диагностика
          # logger.info(f"  - ПЕРВАЯ свеча: {first_time} (timestamp: {first_timestamp})")
          # logger.info(f"  - ПОСЛЕДНЯЯ свеча: {last_time} (timestamp: {last_timestamp})")

          # Проверяем логику сортировки - Bybit возвращает от новых к старым
          if first_timestamp > last_timestamp:
            # logger.info("  - ✅ Данные отсортированы правильно (новые -> старые)")
            fresh_age = (datetime.now() - first_time).total_seconds() / 3600
            # logger.info(f"  - 🕐 Возраст самых свежих данных: {fresh_age:.1f} часов")
          else:
            logger.warning("  - ❌ Данные отсортированы неправильно!")
      else:
        logger.warning(f"  - ❌ API вернул result без 'list'")
    else:
      logger.error(f"❌ API вернул None для {symbol}")

    return (result or {}).get('list', [])

  def clear_all_cache(self):
      """Полностью очищает весь кэш"""
      try:
        cache_cleared = len(self.request_cache)
        self.request_cache.clear()
        logger.info(f"🧹 Очищен request_cache коннектора: {cache_cleared} записей")
      except Exception as e:
        logger.error(f"Ошибка очистки кэша коннектора: {e}")

  def clear_symbol_cache(self, symbol: str):
    """Очищает кэш для конкретного символа"""
    try:
      keys_to_remove = []
      for key in self.request_cache.keys():
        if symbol in key:
          keys_to_remove.append(key)

      for key in keys_to_remove:
        del self.request_cache[key]

      logger.debug(f"🧹 Очищен кэш коннектора для {symbol}: {len(keys_to_remove)} записей")
    except Exception as e:
      logger.error(f"Ошибка очистки кэша для {symbol}: {e}")

  async def get_kline_batch(self, symbols: List[str], interval: str, limit: int = 200) -> dict[str, BaseException]:
    """Получает свечи для нескольких символов параллельно"""
    tasks = []
    for symbol in symbols:
      task = self.get_kline(symbol, interval, limit)
      tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
      symbol: result for symbol, result in zip(symbols, results)
      if not isinstance(result, Exception)
    }

  async def fetch_positions_batch(self, symbols: List[str]) -> Dict[str, List[Dict]]:
    """Получает позиции для нескольких символов параллельно"""
    tasks = []
    for symbol in symbols:
      task = self.fetch_positions(symbol)
      tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
      symbol: result for symbol, result in zip(symbols, results)
      if not isinstance(result, Exception) and result
    }

  async def get_account_balance(self, account_type: str = "CONTRACT", coin: str = "USDT") -> Optional[Dict]:
    """Получает баланс аккаунта (без кэширования для актуальности)"""
    endpoint = "/v5/account/wallet-balance"
    params = {'accountType': account_type}
    if coin:
      params['coin'] = coin

    result = await self._make_request('GET', endpoint, params, use_cache=False)
    if result and result.get('list'):
      return result['list'][0]
    return None

  async def set_leverage(self, symbol: str, buy_leverage: int, sell_leverage: int) -> bool:
    """Устанавливает кредитное плечо для символа"""
    endpoint = "/v5/position/set-leverage"
    params = {
      'category': 'linear',
      'symbol': symbol,
      'buyLeverage': str(buy_leverage),
      'sellLeverage': str(sell_leverage)
    }
    result = await self._make_request('POST', endpoint, params, treat_as_success=[110043], use_cache=False)
    return result is not None

  async def place_order(self, symbol: str, side: str, order_type: str, quantity: float,
                        price: float = None, time_in_force: str = "GTC",
                        category: str = "linear", positionIdx: int = 0, **kwargs) -> Optional[Dict]:
    """
    Размещает ордер с правильными параметрами для Bybit API v5
    """
    endpoint = "/v5/order/create"

    # Формируем правильные параметры
    params = {
      'category': category,
      'symbol': symbol,
      'side': side,  # Buy/Sell
      'orderType': order_type,  # Market/Limit
      'qty': str(quantity),  # Всегда строка
      'positionIdx': positionIdx  # 0 для one-way mode
    }

    # Добавляем цену только для лимитных ордеров
    if order_type.lower() == 'limit' and price is not None:
      params['price'] = str(price)
      params['timeInForce'] = time_in_force

    # Добавляем дополнительные параметры
    for key, value in kwargs.items():
      if key not in params and value is not None:
        # Специальная обработка для SL/TP параметров
        if key in ['stopLoss', 'takeProfit']:
          params[key] = str(float(value))  # Обеспечиваем правильный формат для SL/TP
          logger.info(f"🎯 Добавлен {key}: {params[key]} для ордера {symbol}")
        else:
          params[key] = str(value) if isinstance(value, (int, float)) else value

    logger.debug(f"Отправка ордера: {params}")

    return await self._make_request('POST', endpoint, params, use_cache=False)

  async def fetch_order_book(self, symbol: str, depth: int = 25) -> Optional[Dict]:
    """Получает стакан ордеров для символа"""
    try:
      params = {
        'category': self.default_category,
        'symbol': symbol,
        'limit': depth
      }

      response = await self._make_request('GET', '/v5/market/orderbook', params=params)

      if response and 'result' in response:
        data = response['result']

        # ИСПРАВЛЕННАЯ ВЕРСИЯ с улучшенной обработкой
        bids = data.get('b', [])
        asks = data.get('a', [])

        # Улучшенная проверка и исправление сортировки
        if bids and len(bids) > 1:
          try:
            # Проверяем правильность сортировки bids (должны быть по убыванию цены)
            first_bid_price = float(bids[0][0])
            second_bid_price = float(bids[1][0])

            if first_bid_price < second_bid_price:
              logger.debug(f"Исправляем сортировку bids: {first_bid_price} < {second_bid_price}")
              bids.sort(key=lambda x: float(x[0]), reverse=True)
              logger.debug("✅ Bids пересортированы по убыванию цены")
            else:
              logger.debug("✅ Bids уже правильно отсортированы")
          except (ValueError, IndexError) as e:
            logger.warning(f"Ошибка при проверке сортировки bids: {e}")

        if asks and len(asks) > 1:
          try:
            # Проверяем правильность сортировки asks (должны быть по возрастанию цены)
            first_ask_price = float(asks[0][0])
            second_ask_price = float(asks[1][0])

            if first_ask_price > second_ask_price:
              logger.debug(f"Исправляем сортировку asks: {first_ask_price} > {second_ask_price}")
              asks.sort(key=lambda x: float(x[0]))
              logger.debug("✅ Asks пересортированы по возрастанию цены")
            else:
              logger.debug("✅ Asks уже правильно отсортированы")
          except (ValueError, IndexError) as e:
            logger.warning(f"Ошибка при проверке сортировки asks: {e}")

        # Сортируем asks по возрастанию цены (лучшие предложения сверху)
        if asks:
          asks_sorted = sorted(asks, key=lambda x: float(x[0]))
          if asks != asks_sorted:
            logger.warning(f"⚠️ Исправлена сортировка asks для {symbol}")
            data['a'] = asks_sorted

        return {
          'bids': [[float(price), float(qty)] for price, qty in data.get('b', [])],
          'asks': [[float(price), float(qty)] for price, qty in data.get('a', [])],
          'timestamp': data.get('ts', int(datetime.now().timestamp() * 1000)),
          'symbol': symbol
        }

      return None

    except Exception as e:
      logger.error(f"❌ Ошибка получения стакана для {symbol}: {e}")
      return None

  async def fetch_positions(self, symbol: str) -> List[Dict]:
    """Получает открытые позиции для конкретного символа"""
    endpoint = "/v5/position/list"
    params = {
      'category': 'linear',
      'symbol': symbol
    }
    result = await self._make_request('GET', endpoint, params, use_cache=False)
    return result.get('list', []) if result else []

  async def get_execution_history(self, symbol: str, limit: int = 20) -> List[Dict]:
    """Асинхронно получает историю исполненных ордеров (сделок)."""
    endpoint = "/v5/execution/list"
    params = {
      'category': 'linear',
      'symbol': symbol,
      'limit': limit
    }
    result = await self._make_request('GET', endpoint, params)
    return result.get('list', []) if result else []

  async def get_instruments_info(self, category: str = 'linear') -> List[Dict]:
    """Получает информацию по всем инструментам с обработкой пагинации"""
    all_instruments = []
    cursor = ""

    while True:
      endpoint = "/v5/market/instruments-info"
      params = {'category': category, 'limit': 1000}

      if cursor:
        params['cursor'] = cursor

      result = await self._make_request('GET', endpoint, params, use_cache=True)

      if not result:
        break

      instruments = result.get('list', [])
      all_instruments.extend(instruments)

      cursor = result.get('nextPageCursor', '')
      if not cursor:
        break

    logger.info(f"Получено {len(all_instruments)} инструментов категории {category}")
    return all_instruments



  async def fetch_ticker(self, symbol: str) -> Optional[Dict]:
      """Получает текущие данные по тикеру"""
      try:
        endpoint = "/v5/market/tickers"
        params = {
          'category': 'linear',
          'symbol': symbol
        }

        result = await self._make_request('GET', endpoint, params, use_cache=True)

        if result and result.get('list') and len(result['list']) > 0:
          ticker = result['list'][0]
          return {
            'symbol': ticker.get('symbol'),
            'last': float(ticker.get('lastPrice', 0)),
            'bid': float(ticker.get('bid1Price', 0)),
            'ask': float(ticker.get('ask1Price', 0)),
            'high': float(ticker.get('highPrice24h', 0)),
            'low': float(ticker.get('lowPrice24h', 0)),
            'volume': float(ticker.get('volume24h', 0)),
            'timestamp': datetime.now()
          }

        return None

      except Exception as e:
        logger.error(f"Ошибка получения тикера {symbol}: {e}")
        return None

  async def get_multiple_tickers(self, symbols: List[str], batch_size: int = 50) -> Dict[str, dict]:
    """
    Получает тикеры для нескольких символов батчами
    Значительно быстрее чем отдельные запросы
    """
    results = {}

    # Разбиваем на батчи для обхода лимитов API
    for i in range(0, len(symbols), batch_size):
      batch = symbols[i:i + batch_size]

      try:
        # Bybit позволяет получать до 50 тикеров за раз
        params = {
          'category': 'linear',
          'symbol': ','.join(batch)  # Символы через запятую
        }

        response = await self._make_request('GET', '/v5/market/tickers', params)

        if response and 'result' in response and 'list' in response['result']:
          for ticker_data in response['result']['list']:
            symbol = ticker_data.get('symbol')
            if symbol:
              results[symbol] = ticker_data

        # Небольшая пауза между батчами
        if len(symbols) > batch_size:
          await asyncio.sleep(0.1)

      except Exception as e:
        logger.error(f"Ошибка получения батча тикеров: {e}")

        # Fallback: получаем по одному
        for symbol in batch:
          try:
            ticker = await self.fetch_ticker(symbol)
            if ticker:
              results[symbol] = ticker
            await asyncio.sleep(0.05)
          except Exception as ex:
            logger.error(f"Ошибка получения тикера {symbol}: {ex}")

    logger.debug(f"Получено {len(results)} тикеров из {len(symbols)} запрошенных")
    return results

  async def get_multiple_positions(self, symbols: List[str] = None) -> Dict[str, dict]:
    """
    Получает позиции для нескольких символов одним запросом
    """
    try:
      params = {
        'category': 'linear',
        'settleCoin': 'USDT'
      }

      # Если указаны конкретные символы, можем их добавить в фильтр
      if symbols and len(symbols) <= 200:  # Лимит Bybit
        params['symbol'] = ','.join(symbols)

      response = await self._make_request('GET', '/v5/position/list', params)

      results = {}
      if response and 'result' in response and 'list' in response['result']:
        for position_data in response['result']['list']:
          symbol = position_data.get('symbol')
          if symbol:
            results[symbol] = position_data

      return results

    except Exception as e:
      logger.error(f"Ошибка получения множественных позиций: {e}")
      return {}

  async def get_multiple_balances(self, coins: List[str] = None) -> Dict[str, dict]:
    """
    Получает балансы для нескольких монет одним запросом
    """
    try:
      params = {
        'accountType': 'UNIFIED'
      }

      if coins:
        params['coin'] = ','.join(coins[:200])  # Лимит API

      response = await self._make_request('GET', '/v5/account/wallet-balance', params)

      results = {}
      if response and 'result' in response and 'list' in response['result']:
        for account_data in response['result']['list']:
          if 'coin' in account_data:
            for coin_data in account_data['coin']:
              coin = coin_data.get('coin')
              if coin:
                results[coin] = coin_data

      return results

    except Exception as e:
      logger.error(f"Ошибка получения множественных балансов: {e}")
      return {}

  async def set_multiple_leverages(self, leverage_settings: Dict[str, float], batch_size: int = 10) -> Dict[str, bool]:
    """
    Устанавливает плечо для нескольких символов

    Args:
        leverage_settings: {symbol: leverage_value}
        batch_size: размер батча для параллельной обработки

    Returns:
        {symbol: success_status}
    """
    results = {}

    # Разбиваем на батчи для параллельной обработки
    symbols = list(leverage_settings.keys())

    for i in range(0, len(symbols), batch_size):
      batch_symbols = symbols[i:i + batch_size]
      batch_tasks = []

      for symbol in batch_symbols:
        leverage = leverage_settings[symbol]
        task = self._set_single_leverage(symbol, leverage)
        batch_tasks.append(task)

      # Выполняем батч параллельно
      batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

      # Обрабатываем результаты
      for idx, result in enumerate(batch_results):
        symbol = batch_symbols[idx]
        results[symbol] = not isinstance(result, Exception)

        if isinstance(result, Exception):
          logger.error(f"Ошибка установки плеча для {symbol}: {result}")

      # Пауза между батчами
      if i + batch_size < len(symbols):
        await asyncio.sleep(0.2)

    success_count = sum(results.values())
    logger.info(f"Плечо установлено для {success_count}/{len(symbols)} символов")

    return results

  async def _set_single_leverage(self, symbol: str, leverage: float) -> bool:
    """Устанавливает плечо для одного символа"""
    try:
      params = {
        'category': 'linear',
        'symbol': symbol,
        'buyLeverage': str(leverage),
        'sellLeverage': str(leverage)
      }

      response = await self._make_request('POST', '/v5/position/set-leverage', params)
      return response is not None

    except Exception as e:
      logger.error(f"Ошибка установки плеча {leverage}x для {symbol}: {e}")
      return False

  async def batch_place_orders(self, orders: List[Dict], batch_size: int = 5) -> Dict[str, dict]:
    """
    Размещает множественные ордера батчами

    Args:
        orders: список словарей с параметрами ордеров
        batch_size: размер батча

    Returns:
        {order_id: result}
    """
    results = {}

    for i in range(0, len(orders), batch_size):
      batch = orders[i:i + batch_size]
      batch_tasks = []

      for order_params in batch:
        task = self.place_order(**order_params)
        batch_tasks.append(task)

      # Выполняем батч параллельно
      batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

      # Обрабатываем результаты
      for idx, result in enumerate(batch_results):
        order_key = f"order_{i + idx}"

        if isinstance(result, Exception):
          results[order_key] = {'error': str(result)}
          logger.error(f"Ошибка размещения ордера {order_key}: {result}")
        else:
          results[order_key] = result

      # Пауза между батчами для соблюдения rate limits
      if i + batch_size < len(orders):
        await asyncio.sleep(0.3)

    return results

  async def get_symbols_info_batch(self, symbols: List[str], batch_size: int = 50) -> Dict[str, dict]:
    """
    Получает информацию об инструментах батчами
    """
    results = {}

    # Можем получить всю информацию сразу, а затем отфильтровать
    try:
      params = {
        'category': 'linear'
      }

      response = await self._make_request('GET', '/v5/market/instruments-info', params)

      if response and 'result' in response and 'list' in response['result']:
        # Создаем индекс для быстрого поиска
        all_instruments = {item['symbol']: item for item in response['result']['list']}

        # Возвращаем только запрошенные символы
        for symbol in symbols:
          if symbol in all_instruments:
            results[symbol] = all_instruments[symbol]

      logger.debug(f"Получена информация для {len(results)}/{len(symbols)} символов")

      return results

    except Exception as e:
      logger.error(f"Ошибка получения информации об инструментах: {e}")
      return {}

  async def get_batch_klines(self, symbol_timeframe_pairs: List[Tuple[str, str]], limit: int = 200) -> Dict[
    str, pd.DataFrame]:
    """
    Получает исторические данные для множественных пар символ-таймфрейм

    Args:
        symbol_timeframe_pairs: [(symbol, timeframe), ...]
        limit: количество свечей

    Returns:
        {f"{symbol}_{timeframe}": DataFrame}
    """
    results = {}
    batch_size = 3  # Консервативный размер для исторических данных

    for i in range(0, len(symbol_timeframe_pairs), batch_size):
      batch = symbol_timeframe_pairs[i:i + batch_size]
      batch_tasks = []

      for symbol, timeframe in batch:
        task = self._get_single_klines(symbol, timeframe, limit)
        batch_tasks.append(task)

      # Выполняем батч параллельно
      batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

      # Обрабатываем результаты
      for idx, result in enumerate(batch_results):
        symbol, timeframe = batch[idx]
        key = f"{symbol}_{timeframe}"

        if isinstance(result, Exception):
          logger.error(f"Ошибка получения данных для {key}: {result}")
          results[key] = None
        else:
          results[key] = result

      # Пауза между батчами
      if i + batch_size < len(symbol_timeframe_pairs):
        await asyncio.sleep(0.5)

    successful = sum(1 for v in results.values() if v is not None)
    logger.info(f"Получены исторические данные для {successful}/{len(symbol_timeframe_pairs)} пар")

    return results

  async def _get_single_klines(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Получает данные для одной пары символ-таймфрейм"""
    try:
      params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': timeframe,
        'limit': limit
      }

      response = await self._make_request('GET', '/v5/market/kline', params)

      if not response or 'result' not in response:
        return None

      data = response['result'].get('list', [])
      if not data:
        return None

      # Конвертируем в DataFrame
      df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
      ])

      # Обрабатываем типы данных
      df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
      for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

      df = df.sort_values('timestamp').reset_index(drop=True)
      return df

    except Exception as e:
      logger.error(f"Ошибка получения данных {symbol}_{timeframe}: {e}")
      return None

  # Метод для оптимизации rate limits
  async def optimize_request_rate(self):
    """
    Оптимизирует скорость запросов на основе текущих лимитов
    """
    try:
      # Получаем информацию о rate limits от API
      response = await self._make_request('GET', '/v5/market/time')

      if response:
        # Анализируем заголовки ответа для понимания текущих лимитов
        current_time = datetime.now()

        # Адаптируем семафор на основе доступных лимитов
        if hasattr(self, 'semaphore'):
          # Если нет ошибок - можем увеличить параллелизм
          if not hasattr(self, '_recent_errors'):
            self._recent_errors = 0

          if self._recent_errors == 0 and self.semaphore._value < 20:
            # Увеличиваем лимит
            self.semaphore = asyncio.Semaphore(min(20, self.semaphore._value + 2))
            logger.debug("Увеличен лимит параллельных запросов")

          elif self._recent_errors > 3 and self.semaphore._value > 5:
            # Уменьшаем лимит при ошибках
            self.semaphore = asyncio.Semaphore(max(5, self.semaphore._value - 2))
            logger.warning("Уменьшен лимит параллельных запросов из-за ошибок")

        # Сбрасываем счетчик ошибок каждые 5 минут
        if not hasattr(self, '_last_error_reset'):
          self._last_error_reset = current_time

        if (current_time - self._last_error_reset).seconds > 300:
          self._recent_errors = 0
          self._last_error_reset = current_time

    except Exception as e:
      logger.error(f"Ошибка оптимизации rate limits: {e}")

  # Обертка для добавления метрик к существующим методам
  async def _make_request_with_metrics(self, method: str, endpoint: str, params: dict = None, data: dict = None):
    """
    Обертка для _make_request с добавлением метрик производительности
    """
    start_time = time.time()

    try:
      result = await self._make_request(method, endpoint, params, data)

      # Записываем успешный запрос
      response_time = (time.time() - start_time) * 1000  # в миллисекундах

      if not hasattr(self, 'request_stats'):
        self.request_stats = {
          'total_requests': 0,
          'successful_requests': 0,
          'failed_requests': 0,
          'avg_response_time_ms': 0,
          'last_24h_requests': []
        }

      self.request_stats['total_requests'] += 1
      self.request_stats['successful_requests'] += 1

      # Обновляем среднее время ответа
      current_avg = self.request_stats['avg_response_time_ms']
      total = self.request_stats['total_requests']
      self.request_stats['avg_response_time_ms'] = (current_avg * (total - 1) + response_time) / total

      # Добавляем в статистику за 24 часа
      self.request_stats['last_24h_requests'].append({
        'timestamp': datetime.now(),
        'endpoint': endpoint,
        'response_time_ms': response_time,
        'success': True
      })

      # Очищаем старые записи (старше 24 часов)
      cutoff_time = datetime.now() - timedelta(hours=24)
      self.request_stats['last_24h_requests'] = [
        req for req in self.request_stats['last_24h_requests']
        if req['timestamp'] > cutoff_time
      ]

      return result

    except Exception as e:
      # Записываем неудачный запрос
      if hasattr(self, 'request_stats'):
        self.request_stats['failed_requests'] += 1

      if hasattr(self, '_recent_errors'):
        self._recent_errors += 1

      logger.error(f"Ошибка запроса {method} {endpoint}: {e}")
      raise

  def get_performance_stats(self) -> dict:
      """Возвращает статистику производительности коннектора"""
      if not hasattr(self, 'request_stats'):
        self.request_stats = {
          'total_requests': 0,
          'successful_requests': 0,
          'failed_requests': 0,
          'avg_response_time_ms': 0,
          'last_24h_requests': []
        }

      stats = self.request_stats.copy()

      # Добавляем базовую статистику
      if hasattr(self, '_total_requests'):
        stats['total_requests'] = self._total_requests
      if hasattr(self, '_successful_requests'):
        stats['successful_requests'] = self._successful_requests
      if hasattr(self, '_failed_requests'):
        stats['failed_requests'] = self._failed_requests

      return stats

  def _increment_request_stats(self, success: bool = True, response_time_ms: float = 0):
    """Увеличивает счетчики статистики"""
    if not hasattr(self, 'request_stats'):
      self.request_stats = {
        'total_requests': 0,
        'successful_requests': 0,
        'failed_requests': 0,
        'avg_response_time_ms': 0,
        'last_24h_requests': []
      }

    self.request_stats['total_requests'] += 1
    if success:
      self.request_stats['successful_requests'] += 1
    else:
      self.request_stats['failed_requests'] += 1

    # Обновляем среднее время ответа
    current_avg = self.request_stats['avg_response_time_ms']
    total = self.request_stats['total_requests']
    self.request_stats['avg_response_time_ms'] = (current_avg * (total - 1) + response_time_ms) / total