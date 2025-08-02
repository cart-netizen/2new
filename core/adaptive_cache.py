# core/adaptive_cache.py
"""
Адаптивная система кэширования для повышения производительности торгового бота
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import json

from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
  """Запись в кэше"""
  data: Any
  timestamp: datetime
  hit_count: int = 0
  priority: int = 1  # 1-низкий, 2-средний, 3-высокий
  ttl_override: Optional[int] = None

  @property
  def age_seconds(self) -> float:
    return (datetime.now() - self.timestamp).total_seconds()

  def is_expired(self, default_ttl: int) -> bool:
    ttl = self.ttl_override or default_ttl
    return self.age_seconds > ttl


@dataclass
class CacheStats:
  """Статистика кэша"""
  hits: int = 0
  misses: int = 0
  entries: int = 0
  memory_usage_mb: float = 0.0
  avg_response_time_ms: float = 0.0
  evictions: int = 0

  @property
  def hit_rate(self) -> float:
    total = self.hits + self.misses
    return (self.hits / total * 100) if total > 0 else 0.0


class AdaptiveCacheManager:
  """
  Адаптивный менеджер кэширования с приоритизацией для торгового бота
  """

  def __init__(self, max_entries: int = 10000, default_ttl: int = 300):
    self.max_entries = max_entries
    self.default_ttl = default_ttl

    # Основное хранилище кэша
    self.cache: Dict[str, CacheEntry] = {}

    # Статистика по символам
    self.symbol_stats: Dict[str, Dict] = defaultdict(lambda: {
      'requests': 0,
      'last_access': datetime.now(),
      'avg_response_time': 0.0,
      'priority_score': 1.0
    })

    # TTL конфигурация по типам данных
    self.ttl_config = {
      'candles': 30,  # Свечи - 30 сек
      'ticker': 15,  # Тикер - 15 сек
      'balance': 60,  # Баланс - 1 мин
      'positions': 45,  # Позиции - 45 сек
      'orders': 30,  # Ордера - 30 сек
      'instrument': 300,  # Информация об инструменте - 5 мин
      'volatility': 120,  # Волатильность - 2 мин
      'ml_features': 180,  # ML признаки - 3 мин
    }

    # Приоритетные символы получают больший TTL
    self.focus_symbols: set = set()

    # Статистика
    self.stats = CacheStats()

    # Блокировка для thread-safety
    self._lock = asyncio.Lock()

    # Время последней очистки
    self._last_cleanup = datetime.now()
    self._cleanup_interval = 300  # 5 минут

    logger.info("Адаптивный кэш-менеджер инициализирован")

  def _generate_key(self, symbol: str, data_type: str, **kwargs) -> str:
    """Генерирует уникальный ключ для кэша"""
    key_parts = [symbol, data_type]

    # Добавляем дополнительные параметры
    for key, value in sorted(kwargs.items()):
      key_parts.append(f"{key}:{value}")

    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()[:16]

  def update_focus_symbols(self, symbols: List[str]):
    """Обновляет список приоритетных символов"""
    self.focus_symbols = set(symbols)

    # Увеличиваем приоритет для focus символов
    for symbol in symbols:
      if symbol in self.symbol_stats:
        self.symbol_stats[symbol]['priority_score'] = min(
          self.symbol_stats[symbol]['priority_score'] * 1.5, 3.0
        )

    logger.debug(f"Обновлен список focus символов: {len(symbols)}")

  async def get(self, symbol: str, data_type: str, **kwargs) -> Optional[Any]:
    """Получает данные из кэша"""
    key = self._generate_key(symbol, data_type, **kwargs)

    async with self._lock:
      # Обновляем статистику доступа к символу
      self.symbol_stats[symbol]['requests'] += 1
      self.symbol_stats[symbol]['last_access'] = datetime.now()

      if key in self.cache:
        entry = self.cache[key]
        ttl = self._get_ttl(symbol, data_type)

        if not entry.is_expired(ttl):
          # Попадание в кэш
          entry.hit_count += 1
          self.stats.hits += 1

          logger.debug(f"Cache HIT: {symbol} - {data_type}")
          return entry.data
        else:
          # Данные устарели
          del self.cache[key]
          logger.debug(f"Cache EXPIRED: {symbol} - {data_type}")

      # Промах
      self.stats.misses += 1
      logger.debug(f"Cache MISS: {symbol} - {data_type}")
      return None

  async def set(self, symbol: str, data_type: str, data: Any, **kwargs):
    """Сохраняет данные в кэш"""
    key = self._generate_key(symbol, data_type, **kwargs)

    async with self._lock:
      # Определяем приоритет
      priority = self._calculate_priority(symbol, data_type)

      # Создаем запись
      entry = CacheEntry(
        data=data,
        timestamp=datetime.now(),
        priority=priority
      )

      # Проверяем лимиты кэша
      if len(self.cache) >= self.max_entries:
        await self._evict_entries()

      self.cache[key] = entry
      self.stats.entries = len(self.cache)

      logger.debug(f"Cache SET: {symbol} - {data_type} (priority: {priority})")

  def _get_ttl(self, symbol: str, data_type: str) -> int:
    """Вычисляет TTL для конкретного типа данных и символа"""
    base_ttl = self.ttl_config.get(data_type, self.default_ttl)

    # Увеличиваем TTL для приоритетных символов
    if symbol in self.focus_symbols:
      base_ttl = int(base_ttl * 1.5)

    # Адаптируем TTL на основе активности символа
    symbol_info = self.symbol_stats.get(symbol, {})
    if symbol_info.get('requests', 0) > 10:  # Активный символ
      base_ttl = int(base_ttl * 1.2)

    return base_ttl

  def _calculate_priority(self, symbol: str, data_type: str) -> int:
    """Вычисляет приоритет записи в кэше"""
    priority = 1

    # Высокий приоритет для focus символов
    if symbol in self.focus_symbols:
      priority = 3

    # Высокий приоритет для часто запрашиваемых данных
    symbol_info = self.symbol_stats.get(symbol, {})
    if symbol_info.get('requests', 0) > 5:
      priority = max(priority, 2)

    # Критически важные типы данных
    critical_types = ['positions', 'balance', 'orders']
    if data_type in critical_types:
      priority = 3

    return priority

  async def _evict_entries(self):
    """Удаляет старые записи из кэша"""
    # Сортируем по приоритету и времени доступа
    entries_to_remove = []

    for key, entry in self.cache.items():
      score = (
          entry.priority * 1000 +  # Приоритет важнее всего
          entry.hit_count * 10 +  # Частота использования
          max(0, 300 - entry.age_seconds)  # Свежесть данных
      )
      entries_to_remove.append((score, key))

    # Удаляем 20% записей с наименьшим счетом
    entries_to_remove.sort()
    remove_count = len(entries_to_remove) // 5

    for _, key in entries_to_remove[:remove_count]:
      del self.cache[key]
      self.stats.evictions += 1

    logger.info(f"Удалено {remove_count} записей из кэша")

  async def cleanup_expired(self):
    """Удаляет просроченные записи"""
    current_time = datetime.now()
    if (current_time - self._last_cleanup).seconds < self._cleanup_interval:
      return

    expired_keys = []
    for key, entry in self.cache.items():
      # Получаем TTL для этой записи (упрощенно используем default)
      if entry.is_expired(self.default_ttl):
        expired_keys.append(key)

    for key in expired_keys:
      del self.cache[key]

    if expired_keys:
      logger.info(f"Удалено {len(expired_keys)} просроченных записей")

    self._last_cleanup = current_time
    self.stats.entries = len(self.cache)

  def get_stats(self) -> Dict:
    """Возвращает статистику кэша"""
    return {
      'cache_stats': {
        'hits': self.stats.hits,
        'misses': self.stats.misses,
        'hit_rate': f"{self.stats.hit_rate:.1f}%",
        'entries': len(self.cache),
        'max_entries': self.max_entries,
        'evictions': self.stats.evictions
      },
      'top_symbols': sorted(
        [(symbol, info['requests']) for symbol, info in self.symbol_stats.items()],
        key=lambda x: x[1],
        reverse=True
      )[:10],
      'ttl_config': self.ttl_config,
      'focus_symbols_count': len(self.focus_symbols)
    }

  async def preload_focus_data(self, data_fetcher, timeframes: List[str]):
    """Предзагружает данные для focus символов"""
    if not self.focus_symbols:
      return

    logger.info(f"Предзагрузка данных для {len(self.focus_symbols)} focus символов")

    tasks = []
    for symbol in self.focus_symbols:
      for timeframe in timeframes:
        # Это будет интегрировано с data_fetcher
        task = self._preload_symbol_data(data_fetcher, symbol, timeframe)
        tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Предзагрузка focus данных завершена")

  async def _preload_symbol_data(self, data_fetcher, symbol: str, timeframe: str):
    """Предзагружает данные для одного символа"""
    try:
      # Получаем данные (они автоматически сохранятся в кэш)
      await data_fetcher.get_historical_candles(symbol, timeframe, limit=100)
    except Exception as e:
      logger.error(f"Ошибка предзагрузки {symbol} {timeframe}: {e}")


# Singleton для глобального использования
_global_cache_manager: Optional[AdaptiveCacheManager] = None


def get_cache_manager() -> AdaptiveCacheManager:
  """Возвращает глобальный экземпляр кэш-менеджера"""
  global _global_cache_manager
  if _global_cache_manager is None:
    _global_cache_manager = AdaptiveCacheManager()
  return _global_cache_manager