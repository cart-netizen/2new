# core/data_fetcher.py

import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from functools import lru_cache
import asyncio
from collections import defaultdict

from core.bybit_connector import BybitConnector
from utils.logging_config import get_logger
from config import trading_params
from core.enums import Timeframe

logger = get_logger(__name__)


class CachedData:
  """Класс для хранения кэшированных данных с временем жизни"""

  def __init__(self, data: Any, ttl: int = 300):
    self.data = data
    self.timestamp = datetime.now()
    self.ttl = ttl  # Time to live в секундах

  def is_valid(self) -> bool:
    """Проверяет, действительны ли еще данные"""
    return (datetime.now() - self.timestamp).seconds < self.ttl


class DataFetcher:
  def __init__(self, connector: BybitConnector, settings: Dict[str, Any]):
    self.connector = connector
    self.settings = settings

    # Кэши для различных типов данных
    self.instrument_info_cache: Dict[str, CachedData] = {}
    self.symbols_cache: Optional[CachedData] = None
    self.candles_cache: Dict[str, CachedData] = {}

    # Параметры кэширования
    self.instrument_cache_ttl = 3600  # 1 час для информации об инструментах
    self.symbols_cache_ttl = 600  # 10 минут для списка символов
    self.candles_cache_ttl = 60  # 1 минута для свечей

    # Блокировки для предотвращения дублирующих запросов
    self.fetch_locks = defaultdict(asyncio.Lock)

    # Счетчики для мониторинга эффективности кэша
    self.cache_hits = 0
    self.cache_misses = 0
    self.total_requests = 0

  def get_cache_stats(self) -> Dict[str, Any]:
    """Возвращает статистику использования кэша"""
    hit_rate = self.cache_hits / self.total_requests if self.total_requests > 0 else 0
    return {
      'cache_hits': self.cache_hits,
      'cache_misses': self.cache_misses,
      'total_requests': self.total_requests,
      'hit_rate': hit_rate,
      'cached_instruments': len(self.instrument_info_cache),
      'cached_candles': len(self.candles_cache)
    }

  def clear_cache(self, cache_type: Optional[str] = None):
    """Очищает кэш"""
    if cache_type == 'instruments':
      self.instrument_info_cache.clear()
    elif cache_type == 'symbols':
      self.symbols_cache = None
    elif cache_type == 'candles':
      self.candles_cache.clear()
    else:
      # Очищаем все кэши
      self.instrument_info_cache.clear()
      self.symbols_cache = None
      self.candles_cache.clear()
    logger.info(f"Кэш очищен: {cache_type or 'все'}")

  def _clean_expired_cache(self):
    """Удаляет устаревшие записи из кэшей"""
    # Очистка кэша инструментов
    expired_instruments = [
      symbol for symbol, cached in self.instrument_info_cache.items()
      if not cached.is_valid()
    ]
    for symbol in expired_instruments:
      del self.instrument_info_cache[symbol]

    # Очистка кэша свечей
    expired_candles = [
      key for key, cached in self.candles_cache.items()
      if not cached.is_valid()
    ]
    for key in expired_candles:
      del self.candles_cache[key]

    if expired_instruments or expired_candles:
      logger.debug(f"Очищено устаревших записей: инструменты={len(expired_instruments)}, свечи={len(expired_candles)}")

  async def get_instrument_info(self, symbol: str) -> Optional[Dict[str, Any]]:
    """Получает информацию об инструменте с кэшированием"""
    self.total_requests += 1

    # Проверяем кэш
    if symbol in self.instrument_info_cache:
      cached = self.instrument_info_cache[symbol]
      if cached.is_valid():
        self.cache_hits += 1
        logger.debug(f"Информация об инструменте {symbol} получена из кэша")
        return cached.data

    self.cache_misses += 1

    # Используем блокировку для предотвращения дублирующих запросов
    async with self.fetch_locks[f"instrument_{symbol}"]:
      # Проверяем еще раз после получения блокировки
      if symbol in self.instrument_info_cache and self.instrument_info_cache[symbol].is_valid():
        return self.instrument_info_cache[symbol].data

      logger.debug(f"Запрашиваем информацию об инструменте {symbol} с биржи")

      try:
        # Получаем все инструменты за один запрос для оптимизации
        if not self.instrument_info_cache:  # Если кэш пустой, загружаем все
          instruments = await self.connector.get_instruments_info()

          # Кэшируем все инструменты
          for inst in instruments:
            if inst.get('symbol'):
              self.instrument_info_cache[inst['symbol']] = CachedData(
                inst, self.instrument_cache_ttl
              )

          # Возвращаем нужный инструмент
          if symbol in self.instrument_info_cache:
            return self.instrument_info_cache[symbol].data
        else:
          # Если кэш не пустой, но нужного символа нет, запрашиваем отдельно
          instruments = await self.connector.get_instruments_info()
          for inst in instruments:
            if inst.get('symbol') == symbol:
              self.instrument_info_cache[symbol] = CachedData(
                inst, self.instrument_cache_ttl
              )
              return inst

        logger.warning(f"Информация об инструменте {symbol} не найдена")
        return None

      except Exception as e:
        logger.error(f"Ошибка при получении информации об инструменте {symbol}: {e}")
        return None

  async def get_active_symbols_by_volume(self, limit: int) -> List[str]:
    """Получает список активных символов с кэшированием"""
    self.total_requests += 1

    # Проверяем кэш
    if self.symbols_cache and self.symbols_cache.is_valid():
      self.cache_hits += 1
      cached_symbols = self.symbols_cache.data
      # Возвращаем нужное количество из кэша
      return cached_symbols[:limit]

    self.cache_misses += 1

    # Используем блокировку для предотвращения дублирующих запросов
    async with self.fetch_locks["symbols"]:
      # Проверяем еще раз после получения блокировки
      if self.symbols_cache and self.symbols_cache.is_valid():
        return self.symbols_cache.data[:limit]

      try:
        logger.info("Получение списка активных символов...")

        contracts = await self.connector.get_usdt_perpetual_contracts()
        if not contracts:
          logger.warning("Не удалось получить список контрактов от Bybit.")
          return []

        high_volume_symbols = []
        min_volume_usdt = self.settings.get('min_24h_volume_usdt', 1000000)

        # Обрабатываем контракты параллельно для ускорения
        tasks = []
        for contract in contracts:
          symbol = contract.get('symbol')
          if symbol and "USDT" in symbol and "USDC" not in symbol:
            volume_24h_usdt = float(contract.get('turnover24h', 0))
            if volume_24h_usdt >= min_volume_usdt:
              high_volume_symbols.append({
                'symbol': symbol,
                'volume24h_usdt': volume_24h_usdt
              })

        # Сортируем по объему
        high_volume_symbols.sort(key=lambda x: x['volume24h_usdt'], reverse=True)

        logger.info(f"Найдено {len(high_volume_symbols)} символов с объемом > {min_volume_usdt} USDT.")

        # Фильтруем по наличию исторических данных
        final_symbols = []

        # Параллельная проверка наличия данных
        check_tasks = []
        for item in high_volume_symbols[:limit * 2]:  # Проверяем больше, чем нужно
          symbol = item['symbol']
          check_tasks.append(self._check_symbol_data_availability(symbol))

        # Ждем завершения всех проверок
        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        for i, has_data in enumerate(results):
          if has_data and not isinstance(has_data, Exception):
            final_symbols.append(high_volume_symbols[i]['symbol'])
            if len(final_symbols) >= limit:
              break

        # Кэшируем результат
        self.symbols_cache = CachedData(final_symbols, self.symbols_cache_ttl)

        return final_symbols[:limit]

      except Exception as e:
        logger.error(f"Ошибка при получении активных символов: {e}")
        return []

  async def get_symbols_by_volatility(self, symbols: List[str], limit: int = 20) -> List[Dict[str, Any]]:
    """Получает символы, отсортированные по волатильности и активности"""
    try:
      volatility_data = []

      # Получаем данные о всех тикерах одним запросом
      endpoint = "/v5/market/tickers"
      params = {'category': 'linear'}

      tickers_result = await self.connector._make_request('GET', endpoint, params, use_cache=True)

      if not tickers_result or not tickers_result.get('list'):
        logger.error("Не удалось получить данные тикеров")
        return []

      all_tickers = tickers_result.get('list', [])

      # Создаем словарь для быстрого поиска
      ticker_dict = {ticker['symbol']: ticker for ticker in all_tickers}

      # Обрабатываем только интересующие нас символы
      for symbol in symbols:
        if symbol not in ticker_dict:
          continue

        ticker = ticker_dict[symbol]

        # Извлекаем данные о волатильности
        price_change_24h = float(ticker.get('price24hPcnt', 0)) * 100  # Конвертируем в проценты
        volume_24h = float(ticker.get('turnover24h', 0))  # Объем в USD

        # Получаем свечи для расчета ATR
        candles = await self.get_historical_candles(symbol, Timeframe.ONE_HOUR, limit=24)
        if not candles.empty and len(candles) >= 14:
          import pandas_ta as ta
          atr = ta.atr(candles['high'], candles['low'], candles['close'], length=14)
          if atr is not None and not atr.empty:
            current_atr = atr.iloc[-1]
            atr_percent = (current_atr / candles['close'].iloc[-1]) * 100
          else:
            atr_percent = abs(price_change_24h) / 10  # Примерная оценка
        else:
          atr_percent = abs(price_change_24h) / 10

        # Рассчитываем комплексный показатель волатильности
        volatility_score = abs(price_change_24h) * 0.5 + atr_percent * 0.5

        volatility_data.append({
          'symbol': symbol,
          'price_change_24h': price_change_24h,
          'volume_24h': volume_24h,
          'atr_percent': atr_percent,
          'volatility_score': volatility_score
        })

      # Сортируем по волатильности
      volatility_data.sort(key=lambda x: x['volatility_score'], reverse=True)

      logger.info(f"Найдено {len(volatility_data)} символов с данными о волатильности")

      return volatility_data[:limit]

    except Exception as e:
      logger.error(f"Ошибка при получении данных о волатильности: {e}")
      return []

  async def get_symbols_volatility_batch(self, symbols: List[str], limit: int = 20) -> List[Dict[str, Any]]:
    """Оптимизированная версия получения волатильности с батчевой загрузкой"""
    try:
      volatility_data = []

      # 1. Получаем все тикеры одним запросом
      endpoint = "/v5/market/tickers"
      params = {'category': 'linear'}

      tickers_result = await self.connector._make_request('GET', endpoint, params, use_cache=True)

      if not tickers_result or not tickers_result.get('list'):
        logger.error("Не удалось получить данные тикеров")
        return []

      all_tickers = tickers_result.get('list', [])
      ticker_dict = {ticker['symbol']: ticker for ticker in all_tickers}

      # 2. Фильтруем символы по базовым критериям
      candidates = []

      for symbol in symbols:
        if symbol not in ticker_dict:
          continue

        ticker = ticker_dict[symbol]
        price_change_24h = float(ticker.get('price24hPcnt', 0)) * 100
        volume_24h = float(ticker.get('turnover24h', 0))

        # Базовый фильтр по изменению цены
        if abs(price_change_24h) < 1.0:  # Менее 1% - пропускаем
          continue

        candidates.append({
          'symbol': symbol,
          'price_change_24h': price_change_24h,
          'volume_24h': volume_24h,
          'ticker': ticker
        })

      # 3. Сортируем кандидатов по изменению цены и берем топ для расчета ATR
      candidates.sort(key=lambda x: abs(x['price_change_24h']), reverse=True)
      top_candidates = candidates[:limit * 2]  # Берем с запасом

      # 4. Батчевая загрузка свечей для расчета ATR
      atr_tasks = []
      for candidate in top_candidates:
        task = self._calculate_atr_for_symbol(candidate['symbol'])
        atr_tasks.append(task)

      atr_results = await asyncio.gather(*atr_tasks, return_exceptions=True)

      # 5. Собираем финальные данные
      for i, candidate in enumerate(top_candidates):
        atr_percent = atr_results[i] if not isinstance(atr_results[i], Exception) else abs(
          candidate['price_change_24h']) / 10

        volatility_score = abs(candidate['price_change_24h']) * 0.5 + atr_percent * 0.5

        volatility_data.append({
          'symbol': candidate['symbol'],
          'price_change_24h': candidate['price_change_24h'],
          'volume_24h': candidate['volume_24h'],
          'atr_percent': atr_percent,
          'volatility_score': volatility_score
        })

      # 6. Финальная сортировка по волатильности
      volatility_data.sort(key=lambda x: x['volatility_score'], reverse=True)

      return volatility_data[:limit]

    except Exception as e:
      logger.error(f"Ошибка при батчевом получении волатильности: {e}")
      return []

  async def _calculate_atr_for_symbol(self, symbol: str) -> float:
    """Вспомогательный метод для расчета ATR"""
    try:
      candles = await self.get_historical_candles(symbol, Timeframe.ONE_HOUR, limit=24)
      if not candles.empty and len(candles) >= 14:
        import pandas_ta as ta
        atr = ta.atr(candles['high'], candles['low'], candles['close'], length=14)
        if atr is not None and not atr.empty:
          current_atr = atr.iloc[-1]
          return (current_atr / candles['close'].iloc[-1]) * 100
      return 0.0
    except Exception:
      return 0.0

  async def _check_symbol_data_availability(self, symbol: str) -> bool:
    """Проверяет наличие достаточных исторических данных для символа"""
    try:
      # Используем кэшированные данные, если есть
      cache_key = f"{symbol}_15m_check"
      if cache_key in self.candles_cache and self.candles_cache[cache_key].is_valid():
        return True

      test_data = await self.connector.get_kline(
        symbol, '15', limit=50
      )

      if test_data and len(test_data) >= 50:
        # Кэшируем успешную проверку
        self.candles_cache[cache_key] = CachedData(True, 300)  # 5 минут
        return True

      return False

    except Exception as e:
      logger.debug(f"Ошибка проверки данных для {symbol}: {e}")
      return False

  async def get_historical_candles(
      self,
      symbol: str,
      timeframe: Timeframe,
      limit: int = 250,
      use_cache: bool = True
  ) -> pd.DataFrame:
    """Получает исторические свечи с возможностью кэширования"""
    self.total_requests += 1

    cache_key = f"{symbol}_{timeframe.value}_{limit}"

    # Проверяем кэш, если разрешено
    if use_cache and cache_key in self.candles_cache:
      cached = self.candles_cache[cache_key]
      if cached.is_valid():
        self.cache_hits += 1
        logger.debug(f"Свечи {symbol} {timeframe.value} получены из кэша")
        return cached.data

    self.cache_misses += 1

    # Используем блокировку для предотвращения дублирующих запросов
    async with self.fetch_locks[cache_key]:
      # Проверяем еще раз после получения блокировки
      if use_cache and cache_key in self.candles_cache and self.candles_cache[cache_key].is_valid():
        return self.candles_cache[cache_key].data

      try:
        # Маппинг таймфреймов
        interval_map = {
          Timeframe.ONE_MINUTE: '1',
          Timeframe.FIVE_MINUTES: '5',
          Timeframe.FIFTEEN_MINUTES: '15',
          Timeframe.THIRTY_MINUTES: '30',
          Timeframe.ONE_HOUR: '60',
          Timeframe.FOUR_HOURS: '240',
          Timeframe.ONE_DAY: 'D'
        }

        interval = interval_map.get(timeframe)
        if not interval:
          logger.error(f"Неподдерживаемый таймфрейм: {timeframe}")
          return pd.DataFrame()

        logger.debug(f"Запрос свечей {symbol} {timeframe.value} с биржи")

        raw_candles = await self.connector.get_kline(symbol, interval, limit=limit)

        if not raw_candles:
          logger.warning(f"Не получены данные свечей для {symbol} {timeframe.value}")
          return pd.DataFrame()

        # Преобразуем в DataFrame
        df = pd.DataFrame(raw_candles)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']

        # Конвертируем типы данных
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
          df[col] = pd.to_numeric(df[col], errors='coerce')

        # Сортируем по времени
        df.sort_index(inplace=True)

        # Кэшируем результат
        if use_cache:
          ttl = self.candles_cache_ttl
          # Для больших таймфреймов увеличиваем TTL
          if timeframe in [Timeframe.FOUR_HOURS, Timeframe.ONE_DAY]:
            ttl = 300  # 5 минут

          self.candles_cache[cache_key] = CachedData(df.copy(), ttl)

        # Периодически чистим устаревший кэш
        if self.total_requests % 100 == 0:
          asyncio.create_task(asyncio.to_thread(self._clean_expired_cache))

        return df

      except Exception as e:
        logger.error(f"Ошибка при получении свечей для {symbol}: {e}")
        return pd.DataFrame()

  # Метод для предварительной загрузки данных в кэш
  async def preload_cache(self, symbols: List[str], timeframes: List[Timeframe]):
    """Предварительно загружает данные в кэш для оптимизации"""
    logger.info(f"Предварительная загрузка кэша для {len(symbols)} символов")

    tasks = []
    for symbol in symbols:
      # Загружаем информацию об инструментах
      tasks.append(self.get_instrument_info(symbol))

      # Загружаем свечи для разных таймфреймов
      for timeframe in timeframes:
        tasks.append(self.get_historical_candles(symbol, timeframe))

    # Выполняем все задачи параллельно
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for r in results if not isinstance(r, Exception))
    logger.info(f"Предзагрузка завершена: {successful}/{len(tasks)} успешно")

    return successful

  async def get_current_price_safe(self, symbol: str) -> Optional[float]:
    """Безопасное получение текущей цены"""
    try:
      # Используем существующий метод get_candles
      from core.enums import Timeframe
      df = await self.get_historical_candles(
        symbol=symbol,
        timeframe=Timeframe.ONE_MINUTE,
        limit=1
      )

      if not df.empty:
        return float(df['close'].iloc[-1])

      return None

    except Exception as e:
      logger.error(f"Ошибка получения цены для {symbol}: {e}")
      return None

  async def preload_market_data_batch(self, symbols: List[str], timeframe: Timeframe, limit: int = 100):
    """Предзагружает данные для множества символов параллельно."""
    try:
      batch_size = 10
      all_data = {}

      for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]

        # Параллельно загружаем данные для батча
        tasks = []
        for symbol in batch:
          task = self.get_historical_candles(symbol, timeframe, limit)
          tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, result in zip(batch, results):
          if isinstance(result, pd.DataFrame) and not result.empty:
            all_data[symbol] = result
          elif isinstance(result, Exception):
            logger.error(f"Ошибка загрузки данных для {symbol}: {result}")

      logger.info(f"✅ Предзагружены данные для {len(all_data)} из {len(symbols)} символов")
      return all_data

    except Exception as e:
      logger.error(f"Ошибка батчевой загрузки данных: {e}")
      return {}


