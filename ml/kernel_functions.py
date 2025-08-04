import numpy as np
import pandas as pd
import numba
from typing import Union, Optional
from utils.logging_config import get_logger

logger = get_logger(__name__)


@numba.jit(nopython=True, cache=True, fastmath=True)
def gaussian_kernel_weights(size: int, lookback: float) -> np.ndarray:
  """
  Вычисляет веса Gaussian Kernel

  Args:
      size: Размер массива весов
      lookback: Параметр lookback (сигма)

  Returns:
      Массив весов
  """
  weights = np.zeros(size)
  for i in range(size):
    weights[i] = np.exp(-i ** 2 / (2 * lookback ** 2))
  return weights


@numba.jit(nopython=True, cache=True, fastmath=True)
def rational_quadratic_kernel_weights(size: int, lookback: float, relative_weight: float) -> np.ndarray:
  """
  Вычисляет веса Rational Quadratic Kernel

  Args:
      size: Размер массива весов
      lookback: Параметр lookback
      relative_weight: Относительный вес (альфа)

  Returns:
      Массив весов
  """
  weights = np.zeros(size)
  for i in range(size):
    weights[i] = (1 + (i ** 2 / (2 * relative_weight * lookback ** 2))) ** (-relative_weight)
  return weights


@numba.jit(nopython=True, cache=True, fastmath=True)
def periodic_kernel_weights(size: int, lookback: float, period: float) -> np.ndarray:
  """
  Вычисляет веса Periodic Kernel

  Args:
      size: Размер массива весов
      lookback: Параметр lookback
      period: Период повторения

  Returns:
      Массив весов
  """
  weights = np.zeros(size)
  for i in range(size):
    weights[i] = np.exp(-2 * np.sin(np.pi * i / period) ** 2 / lookback ** 2)
  return weights


@numba.jit(nopython=True, cache=True, fastmath=True)
def locally_periodic_kernel_weights(size: int, lookback: float, period: float) -> np.ndarray:
  """
  Вычисляет веса Locally Periodic Kernel (произведение Periodic и Gaussian)

  Args:
      size: Размер массива весов
      lookback: Параметр lookback
      period: Период повторения

  Returns:
      Массив весов
  """
  weights = np.zeros(size)
  for i in range(size):
    periodic_part = np.exp(-2 * np.sin(np.pi * i / period) ** 2 / lookback ** 2)
    gaussian_part = np.exp(-i ** 2 / (2 * lookback ** 2))
    weights[i] = periodic_part * gaussian_part
  return weights


@numba.jit(nopython=True, cache=True, fastmath=True)
def apply_kernel_regression(src: np.ndarray, weights: np.ndarray, start_at_bar: int) -> np.ndarray:
  """
  Применяет kernel регрессию к временному ряду

  Args:
      src: Исходный временной ряд
      weights: Веса kernel
      start_at_bar: Начальный бар для регрессии

  Returns:
      Сглаженный временной ряд
  """
  size = len(src)
  result = np.zeros(size)

  for i in range(size):
    if i < start_at_bar:
      result[i] = np.nan
      continue

    current_weight = 0.0
    cumulative_weight = 0.0

    # Ограничиваем окно весами
    window_size = min(i + 1, len(weights))

    for j in range(window_size):
      if i - j >= 0:
        y = src[i - j]
        w = weights[j]
        current_weight += y * w
        cumulative_weight += w

    if cumulative_weight > 0:
      result[i] = current_weight / cumulative_weight
    else:
      result[i] = src[i]

  return result


class KernelFunctions:
  """
  Реализация различных kernel функций для технического анализа
  Оптимизирована с использованием Numba для высокой производительности
  """

  @staticmethod
  def gaussian(src: Union[pd.Series, np.ndarray], lookback: int, start_at_bar: int) -> Union[pd.Series, np.ndarray]:
    """
    Gaussian Kernel - взвешенное среднее с весами по Гауссу

    Args:
        src: Исходный временной ряд
        lookback: Окно просмотра (стандартное отклонение)
        start_at_bar: Начальный бар для регрессии

    Returns:
        Сглаженный временной ряд
    """
    # Конвертируем в numpy array если нужно
    if isinstance(src, pd.Series):
      src_array = src.values
      index = src.index
      return_series = True
    else:
      src_array = src
      return_series = False

    # Вычисляем веса
    size = min(len(src_array), lookback * 4)  # Ограничиваем размер для производительности
    weights = gaussian_kernel_weights(size, float(lookback))

    # Применяем регрессию
    result = apply_kernel_regression(src_array, weights, start_at_bar)

    # Возвращаем в нужном формате
    if return_series:
      return pd.Series(result, index=index)
    return result

  @staticmethod
  def rational_quadratic(src: Union[pd.Series, np.ndarray], lookback: int,
                         relative_weight: float, start_at_bar: int) -> Union[pd.Series, np.ndarray]:
    """
    Rational Quadratic Kernel - бесконечная сумма Gaussian Kernels разных масштабов

    Args:
        src: Исходный временной ряд
        lookback: Окно просмотра
        relative_weight: Относительный вес временных рамок (альфа)
        start_at_bar: Начальный бар для регрессии

    Returns:
        Сглаженный временной ряд
    """
    # Конвертируем в numpy array если нужно
    if isinstance(src, pd.Series):
      src_array = src.values
      index = src.index
      return_series = True
    else:
      src_array = src
      return_series = False

    # Вычисляем веса
    size = min(len(src_array), lookback * 4)
    weights = rational_quadratic_kernel_weights(size, float(lookback), relative_weight)

    # Применяем регрессию
    result = apply_kernel_regression(src_array, weights, start_at_bar)

    # Возвращаем в нужном формате
    if return_series:
      return pd.Series(result, index=index)
    return result

  @staticmethod
  def periodic(src: Union[pd.Series, np.ndarray], lookback: int,
               period: int, start_at_bar: int) -> Union[pd.Series, np.ndarray]:
    """
    Periodic Kernel - для моделирования периодических функций

    Args:
        src: Исходный временной ряд
        lookback: Окно просмотра
        period: Расстояние между повторениями
        start_at_bar: Начальный бар для регрессии

    Returns:
        Сглаженный временной ряд
    """
    # Конвертируем в numpy array если нужно
    if isinstance(src, pd.Series):
      src_array = src.values
      index = src.index
      return_series = True
    else:
      src_array = src
      return_series = False

    # Вычисляем веса
    size = min(len(src_array), max(period * 2, lookback * 4))
    weights = periodic_kernel_weights(size, float(lookback), float(period))

    # Применяем регрессию
    result = apply_kernel_regression(src_array, weights, start_at_bar)

    # Возвращаем в нужном формате
    if return_series:
      return pd.Series(result, index=index)
    return result

  @staticmethod
  def locally_periodic(src: Union[pd.Series, np.ndarray], lookback: int,
                       period: int, start_at_bar: int) -> Union[pd.Series, np.ndarray]:
    """
    Locally Periodic Kernel - периодическая функция с медленным изменением во времени

    Args:
        src: Исходный временной ряд
        lookback: Окно просмотра
        period: Расстояние между повторениями
        start_at_bar: Начальный бар для регрессии

    Returns:
        Сглаженный временной ряд
    """
    # Конвертируем в numpy array если нужно
    if isinstance(src, pd.Series):
      src_array = src.values
      index = src.index
      return_series = True
    else:
      src_array = src
      return_series = False

    # Вычисляем веса
    size = min(len(src_array), max(period * 2, lookback * 4))
    weights = locally_periodic_kernel_weights(size, float(lookback), float(period))

    # Применяем регрессию
    result = apply_kernel_regression(src_array, weights, start_at_bar)

    # Возвращаем в нужном формате
    if return_series:
      return pd.Series(result, index=index)
    return result

  @staticmethod
  def adaptive_kernel(src: Union[pd.Series, np.ndarray], kernel_type: str = 'gaussian',
                      lookback: int = 8, relative_weight: float = 8.0,
                      period: int = 100, start_at_bar: int = 25) -> Union[pd.Series, np.ndarray]:
    """
    Адаптивный выбор kernel функции

    Args:
        src: Исходный временной ряд
        kernel_type: Тип kernel ('gaussian', 'rational_quadratic', 'periodic', 'locally_periodic')
        lookback: Окно просмотра
        relative_weight: Для rational_quadratic
        period: Для periodic kernels
        start_at_bar: Начальный бар

    Returns:
        Сглаженный временной ряд
    """
    if kernel_type == 'gaussian':
      return KernelFunctions.gaussian(src, lookback, start_at_bar)
    elif kernel_type == 'rational_quadratic':
      return KernelFunctions.rational_quadratic(src, lookback, relative_weight, start_at_bar)
    elif kernel_type == 'periodic':
      return KernelFunctions.periodic(src, lookback, period, start_at_bar)
    elif kernel_type == 'locally_periodic':
      return KernelFunctions.locally_periodic(src, lookback, period, start_at_bar)
    else:
      logger.warning(f"Неизвестный тип kernel: {kernel_type}, используется gaussian")
      return KernelFunctions.gaussian(src, lookback, start_at_bar)


# Дополнительные вспомогательные функции для WaveTrend 3D
@numba.jit(nopython=True, cache=True, fastmath=True)
def calculate_kernel_estimator_crossovers(series: np.ndarray, kernel_estimate: np.ndarray) -> tuple:
  """
  Определяет пересечения между серией и kernel оценкой

  Returns:
      (bullish_crosses, bearish_crosses) - массивы индексов пересечений
  """
  bullish = []
  bearish = []

  for i in range(1, len(series)):
    if series[i - 1] <= kernel_estimate[i - 1] and series[i] > kernel_estimate[i]:
      bullish.append(i)
    elif series[i - 1] >= kernel_estimate[i - 1] and series[i] < kernel_estimate[i]:
      bearish.append(i)

  return np.array(bullish), np.array(bearish)


@numba.jit(nopython=True, cache=True, fastmath=True)
def detect_kernel_trend(yhat0: np.ndarray, slow_series: np.ndarray) -> np.ndarray:
  """
  Определяет тренд на основе kernel оценки

  Returns:
      Массив: 1 для бычьего тренда, -1 для медвежьего, 0 для неопределенного
  """
  trend = np.zeros(len(yhat0))

  for i in range(len(yhat0)):
    if yhat0[i] > slow_series[i]:
      trend[i] = 1
    elif yhat0[i] < slow_series[i]:
      trend[i] = -1
    else:
      trend[i] = 0

  return trend