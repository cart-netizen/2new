# core/circuit_breaker.py
"""
Circuit Breaker для защиты торгового бота от каскадных сбоев
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import functools

from utils.logging_config import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
  """Состояния Circuit Breaker"""
  CLOSED = "closed"  # Нормальная работа
  OPEN = "open"  # Блокировка запросов
  HALF_OPEN = "half_open"  # Тестирование восстановления


@dataclass
class CircuitBreakerConfig:
  """Конфигурация Circuit Breaker"""
  failure_threshold: int = 5  # Количество ошибок для открытия
  timeout_seconds: int = 60  # Время до попытки восстановления
  success_threshold: int = 3  # Успешных запросов для закрытия
  monitoring_window: int = 300  # Окно мониторинга в секундах
  max_half_open_requests: int = 5  # Макс запросов в half-open состоянии


@dataclass
class RequestResult:
  """Результат запроса"""
  success: bool
  timestamp: datetime
  response_time_ms: float
  error_message: Optional[str] = None


class CircuitBreaker:
  """
  Circuit Breaker для защиты от каскадных сбоев
  """

  def __init__(self, name: str, config: CircuitBreakerConfig = None):
    self.name = name
    self.config = config or CircuitBreakerConfig()

    # Состояние
    self.state = CircuitState.CLOSED
    self.last_failure_time: Optional[datetime] = None
    self.consecutive_failures = 0
    self.consecutive_successes = 0
    self.half_open_requests = 0

    # История запросов
    self.request_history: List[RequestResult] = []

    # Блокировка для thread-safety
    self._lock = asyncio.Lock()

    logger.info(f"Circuit Breaker '{name}' инициализирован")

  async def call(self, func: Callable, *args, **kwargs) -> Any:
    """
    Выполняет функцию через Circuit Breaker
    """
    async with self._lock:
      # Проверяем текущее состояние
      await self._update_state()

      if self.state == CircuitState.OPEN:
        raise CircuitBreakerOpenError(
          f"Circuit breaker '{self.name}' is OPEN. "
          f"Next attempt in {self._time_until_retry():.1f} seconds"
        )

      if self.state == CircuitState.HALF_OPEN:
        if self.half_open_requests >= self.config.max_half_open_requests:
          raise CircuitBreakerOpenError(
            f"Circuit breaker '{self.name}' reached max half-open requests"
          )
        self.half_open_requests += 1

    # Выполняем функцию с измерением времени
    start_time = time.time()
    try:
      result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

      # Записываем успешный результат
      response_time = (time.time() - start_time) * 1000
      await self._record_success(response_time)

      return result

    except Exception as e:
      # Записываем ошибку
      response_time = (time.time() - start_time) * 1000
      await self._record_failure(str(e), response_time)
      raise

  async def _update_state(self):
    """Обновляет состояние Circuit Breaker"""
    now = datetime.now()

    if self.state == CircuitState.OPEN:
      # Проверяем, можно ли перейти в half-open
      if (self.last_failure_time and
          (now - self.last_failure_time).seconds >= self.config.timeout_seconds):
        self.state = CircuitState.HALF_OPEN
        self.half_open_requests = 0
        logger.info(f"Circuit breaker '{self.name}' переходит в HALF_OPEN")

    elif self.state == CircuitState.HALF_OPEN:
      # В half-open состоянии решение принимается после каждого запроса
      pass

    # Очищаем старую историю
    cutoff_time = now - timedelta(seconds=self.config.monitoring_window)
    self.request_history = [
      req for req in self.request_history
      if req.timestamp > cutoff_time
    ]

  async def _record_success(self, response_time_ms: float):
    """Записывает успешный запрос"""
    async with self._lock:
      result = RequestResult(
        success=True,
        timestamp=datetime.now(),
        response_time_ms=response_time_ms
      )
      self.request_history.append(result)

      self.consecutive_failures = 0
      self.consecutive_successes += 1

      # Переходы состояний при успехе
      if self.state == CircuitState.HALF_OPEN:
        if self.consecutive_successes >= self.config.success_threshold:
          self.state = CircuitState.CLOSED
          self.half_open_requests = 0
          logger.info(f"Circuit breaker '{self.name}' восстановлен (CLOSED)")

      logger.debug(f"Circuit breaker '{self.name}': успешный запрос ({response_time_ms:.1f}ms)")

  async def _record_failure(self, error_message: str, response_time_ms: float):
    """Записывает неудачный запрос"""
    async with self._lock:
      result = RequestResult(
        success=False,
        timestamp=datetime.now(),
        response_time_ms=response_time_ms,
        error_message=error_message
      )
      self.request_history.append(result)

      self.consecutive_successes = 0
      self.consecutive_failures += 1
      self.last_failure_time = datetime.now()

      # Переходы состояний при ошибке
      if (self.state == CircuitState.CLOSED and
          self.consecutive_failures >= self.config.failure_threshold):
        self.state = CircuitState.OPEN
        logger.warning(f"Circuit breaker '{self.name}' открыт (OPEN) после {self.consecutive_failures} ошибок")

      elif self.state == CircuitState.HALF_OPEN:
        self.state = CircuitState.OPEN
        self.half_open_requests = 0
        logger.warning(f"Circuit breaker '{self.name}' снова открыт (OPEN)")

      logger.warning(f"Circuit breaker '{self.name}': ошибка - {error_message}")

  def _time_until_retry(self) -> float:
    """Возвращает время до следующей попытки в секундах"""
    if not self.last_failure_time:
      return 0

    elapsed = (datetime.now() - self.last_failure_time).total_seconds()
    return max(0, self.config.timeout_seconds - elapsed)

  def get_stats(self) -> Dict:
    """Возвращает статистику Circuit Breaker"""
    recent_requests = self.request_history[-100:]  # Последние 100 запросов

    if not recent_requests:
      return {
        'name': self.name,
        'state': self.state.value,
        'total_requests': 0,
        'success_rate': 0,
        'avg_response_time_ms': 0
      }

    successful = [r for r in recent_requests if r.success]
    failed = [r for r in recent_requests if not r.success]

    stats = {
      'name': self.name,
      'state': self.state.value,
      'total_requests': len(recent_requests),
      'successful_requests': len(successful),
      'failed_requests': len(failed),
      'success_rate': (len(successful) / len(recent_requests)) * 100,
      'consecutive_failures': self.consecutive_failures,
      'consecutive_successes': self.consecutive_successes,
      'avg_response_time_ms': sum(r.response_time_ms for r in recent_requests) / len(recent_requests),
      'time_until_retry': self._time_until_retry() if self.state == CircuitState.OPEN else 0
    }

    if failed:
      # Топ ошибок
      error_counts = {}
      for req in failed:
        error = req.error_message or "Unknown error"
        error_counts[error] = error_counts.get(error, 0) + 1

      stats['top_errors'] = sorted(
        error_counts.items(),
        key=lambda x: x[1],
        reverse=True
      )[:5]

    return stats


class CircuitBreakerOpenError(Exception):
  """Исключение при открытом Circuit Breaker"""
  pass


class CircuitBreakerManager:
  """
  Менеджер для управления множественными Circuit Breaker'ами
  """

  def __init__(self):
    self.breakers: Dict[str, CircuitBreaker] = {}
    self._default_config = CircuitBreakerConfig()

  def get_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Получает или создает Circuit Breaker"""
    if name not in self.breakers:
      self.breakers[name] = CircuitBreaker(name, config or self._default_config)
    return self.breakers[name]

  def get_all_stats(self) -> Dict[str, Dict]:
    """Возвращает статистику всех Circuit Breaker'ов"""
    return {name: breaker.get_stats() for name, breaker in self.breakers.items()}

  async def health_check(self) -> Dict:
    """Проверка здоровья всех Circuit Breaker'ов"""
    stats = self.get_all_stats()

    healthy_count = sum(1 for s in stats.values() if s['state'] == 'closed')
    total_count = len(stats)

    return {
      'healthy_breakers': healthy_count,
      'total_breakers': total_count,
      'health_percentage': (healthy_count / total_count * 100) if total_count > 0 else 100,
      'unhealthy_breakers': [
        name for name, stat in stats.items()
        if stat['state'] != 'closed'
      ]
    }


# Глобальный менеджер
_global_breaker_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
  """Возвращает глобальный менеджер Circuit Breaker'ов"""
  global _global_breaker_manager
  if _global_breaker_manager is None:
    _global_breaker_manager = CircuitBreakerManager()
  return _global_breaker_manager


def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
  """
  Декоратор для применения Circuit Breaker к функции
  """

  def decorator(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
      manager = get_circuit_breaker_manager()
      breaker = manager.get_breaker(name, config)
      return await breaker.call(func, *args, **kwargs)

    return wrapper

  return decorator


# Предконфигурированные Circuit Breaker'ы для разных типов операций
class TradingCircuitBreakers:
  """Предконфигурированные Circuit Breaker'ы для торговых операций"""

  # Конфигурации для разных типов операций
  API_REQUESTS = CircuitBreakerConfig(
    failure_threshold=5,
    timeout_seconds=60,
    success_threshold=3,
    monitoring_window=300
  )

  ORDER_EXECUTION = CircuitBreakerConfig(
    failure_threshold=3,
    timeout_seconds=120,  # Больше времени для восстановления
    success_threshold=2,
    monitoring_window=600
  )

  DATA_FETCHING = CircuitBreakerConfig(
    failure_threshold=10,  # Больше терпимости к ошибкам данных
    timeout_seconds=30,  # Быстрое восстановление
    success_threshold=5,
    monitoring_window=180
  )

  ML_PROCESSING = CircuitBreakerConfig(
    failure_threshold=3,
    timeout_seconds=300,  # ML может требовать больше времени
    success_threshold=2,
    monitoring_window=900
  )

  @classmethod
  def setup_trading_breakers(cls):
    """Настраивает все торговые Circuit Breaker'ы"""
    manager = get_circuit_breaker_manager()

    # Создаем breaker'ы для разных операций
    manager.get_breaker('api_requests', cls.API_REQUESTS)
    manager.get_breaker('order_execution', cls.ORDER_EXECUTION)
    manager.get_breaker('data_fetching', cls.DATA_FETCHING)
    manager.get_breaker('ml_processing', cls.ML_PROCESSING)

    logger.info("Торговые Circuit Breaker'ы настроены")