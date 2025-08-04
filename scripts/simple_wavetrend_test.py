#!/usr/bin/env python3
"""
Простой тест WaveTrend 3D без необходимости в API подключении
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.wavetrend_3d import WaveTrend3D
from ml.kernel_functions import KernelFunctions
from utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_test_data(periods=500, trend_type='sine'):
  """Генерирует тестовые OHLCV данные"""
  dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')

  if trend_type == 'sine':
    # Синусоидальный тренд
    trend = np.sin(np.linspace(0, 4 * np.pi, periods)) * 20 + 100
  elif trend_type == 'uptrend':
    # Восходящий тренд с коррекциями
    trend = np.linspace(80, 120, periods) + np.sin(np.linspace(0, 8 * np.pi, periods)) * 5
  elif trend_type == 'downtrend':
    # Нисходящий тренд
    trend = np.linspace(120, 80, periods) + np.sin(np.linspace(0, 8 * np.pi, periods)) * 5
  else:  # 'choppy'
    # Боковой рынок
    trend = 100 + np.random.normal(0, 5, periods).cumsum() * 0.1

  # Добавляем шум
  noise = np.random.normal(0, 0.5, periods)
  close = trend + noise

  # Генерируем OHLC
  high = close + np.abs(np.random.normal(0, 1, periods))
  low = close - np.abs(np.random.normal(0, 1, periods))
  open_price = close + np.random.normal(0, 0.5, periods)

  # Убеждаемся что high/low корректны
  high = np.maximum(high, np.maximum(close, open_price))
  low = np.minimum(low, np.minimum(close, open_price))

  volume = np.random.randint(10000, 100000, periods)

  return pd.DataFrame({
    'timestamp': dates,
    'open': open_price,
    'high': high,
    'low': low,
    'close': close,
    'volume': volume
  })


def test_kernel_functions():
  """Тестирует kernel функции"""
  print("\n" + "=" * 50)
  print("Тестирование Kernel функций")
  print("=" * 50)

  # Генерируем данные
  data = np.sin(np.linspace(0, 4 * np.pi, 200)) * 10 + 100
  series = pd.Series(data)

  kernel = KernelFunctions()

  # Тестируем разные kernel
  kernels_to_test = [
    ('Gaussian', lambda: kernel.gaussian(series, lookback=10, start_at_bar=5)),
    ('Rational Quadratic', lambda: kernel.rational_quadratic(series, lookback=10, relative_weight=8.0, start_at_bar=5)),
    ('Periodic', lambda: kernel.periodic(series, lookback=10, period=50, start_at_bar=5)),
    ('Locally Periodic', lambda: kernel.locally_periodic(series, lookback=10, period=50, start_at_bar=5))
  ]

  for name, func in kernels_to_test:
    try:
      result = func()
      print(f"✅ {name} kernel: OK (вычислено {len(result)} значений)")
    except Exception as e:
      print(f"❌ {name} kernel: ОШИБКА - {e}")


def test_wavetrend_basic():
  """Базовый тест WaveTrend 3D"""
  print("\n" + "=" * 50)
  print("Базовый тест WaveTrend 3D")
  print("=" * 50)

  # Генерируем разные типы рынка
  market_types = ['sine', 'uptrend', 'downtrend', 'choppy']

  wavetrend = WaveTrend3D({
    'oscillator_lookback': 20,
    'quadratic_mean_length': 50,
    'use_kernel_filter': True,
    'divergence_distance': 30
  })

  for market_type in market_types:
    print(f"\n📊 Тестирование на {market_type} рынке...")

    data = generate_test_data(periods=300, trend_type=market_type)
    result = wavetrend.calculate(data, f'TEST_{market_type.upper()}')

    # Анализ результатов
    oscillators = result['oscillators']
    signals = result['signals']
    divergences = result['divergences']

    print(f"  Осцилляторы:")
    print(f"    Fast (последнее):   {oscillators['fast'].iloc[-1]:.4f}")
    print(f"    Normal (последнее): {oscillators['normal'].iloc[-1]:.4f}")
    print(f"    Slow (последнее):   {oscillators['slow'].iloc[-1]:.4f}")

    print(f"  Сигналы: {len(signals)}")
    for signal in signals[-3:]:  # Последние 3 сигнала
      print(f"    - {signal.signal_type}: confidence={signal.confidence:.3f}")

    print(f"  Дивергенции:")
    print(f"    Бычьих: {len(divergences['bullish'])}")
    print(f"    Медвежьих: {len(divergences['bearish'])}")


def test_divergence_detection():
  """Тест детекции дивергенций на специально созданных данных"""
  print("\n" + "=" * 50)
  print("Тест детекции дивергенций")
  print("=" * 50)

  # Создаем данные с явной бычьей дивергенцией
  periods = 200

  # Цена: два минимума, второй ниже первого
  price_wave1 = np.linspace(100, 90, periods // 4)
  price_flat1 = np.full(periods // 4, 90)
  price_wave2 = np.linspace(90, 85, periods // 4)  # Новый минимум
  price_recovery = np.linspace(85, 95, periods // 4)

  price = np.concatenate([price_wave1, price_flat1, price_wave2, price_recovery])

  dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')

  data = pd.DataFrame({
    'timestamp': dates,
    'open': price + np.random.normal(0, 0.2, periods),
    'high': price + np.abs(np.random.normal(0, 0.5, periods)),
    'low': price - np.abs(np.random.normal(0, 0.5, periods)),
    'close': price,
    'volume': np.random.randint(10000, 50000, periods)
  })

  # Тестируем
  wavetrend = WaveTrend3D({
    'divergence_distance': 50,
    'divergence_size_percent': 40
  })

  result = wavetrend.calculate(data, 'DIVERGENCE_TEST')
  divergences = result['divergences']

  print(f"Найдено дивергенций:")
  print(f"  Бычьих: {len(divergences['bullish'])}")
  print(f"  Медвежьих: {len(divergences['bearish'])}")

  if divergences['bullish']:
    div = divergences['bullish'][0]
    print(f"\nПример бычьей дивергенции:")
    print(f"  Цена: {div['prev_price']:.2f} -> {div['price']:.2f} (падение)")
    print(f"  Осциллятор: {div['prev_oscillator']:.4f} -> {div['oscillator']:.4f} (рост)")


def test_performance():
  """Тест производительности"""
  print("\n" + "=" * 50)
  print("Тест производительности")
  print("=" * 50)

  import time

  wavetrend = WaveTrend3D()
  test_sizes = [100, 500, 1000, 2000]

  for size in test_sizes:
    data = generate_test_data(periods=size)

    start_time = time.time()
    result = wavetrend.calculate(data, 'PERF_TEST')
    end_time = time.time()

    execution_time = end_time - start_time
    per_bar_time = execution_time / size * 1000  # мс на бар

    print(f"  {size} баров: {execution_time:.3f} сек ({per_bar_time:.2f} мс/бар)")


def test_signal_types():
  """Тест разных типов сигналов"""
  print("\n" + "=" * 50)
  print("Тест типов сигналов")
  print("=" * 50)

  # Создаем данные с резкими движениями для генерации сигналов
  periods = 300

  # V-образное движение для пересечений
  v_bottom = np.concatenate([
    np.linspace(100, 80, periods // 2),
    np.linspace(80, 100, periods // 2)
  ])

  dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')

  data = pd.DataFrame({
    'timestamp': dates,
    'open': v_bottom + np.random.normal(0, 0.3, periods),
    'high': v_bottom + np.abs(np.random.normal(0, 0.5, periods)),
    'low': v_bottom - np.abs(np.random.normal(0, 0.5, periods)),
    'close': v_bottom,
    'volume': np.random.randint(10000, 50000, periods)
  })

  # Тестируем с разными настройками
  configs = [
    {'speed_to_emphasize': 'Fast', 'name': 'Fast emphasis'},
    {'speed_to_emphasize': 'Normal', 'name': 'Normal emphasis'},
    {'speed_to_emphasize': 'Slow', 'name': 'Slow emphasis'}
  ]

  for config in configs:
    wavetrend = WaveTrend3D(config)
    result = wavetrend.calculate(data, 'SIGNAL_TEST')

    signals = result['signals']
    print(f"\n{config['name']}:")
    print(f"  Всего сигналов: {len(signals)}")

    # Группируем по типам
    signal_types = {}
    for signal in signals:
      signal_types[signal.signal_type] = signal_types.get(signal.signal_type, 0) + 1

    for sig_type, count in signal_types.items():
      print(f"  {sig_type}: {count}")


def main():
  """Основная функция"""
  print("\n" + "=" * 60)
  print("WaveTrend 3D - Простое тестирование без API")
  print("=" * 60)

  tests = [
    ("Kernel функции", test_kernel_functions),
    ("Базовый WaveTrend", test_wavetrend_basic),
    ("Детекция дивергенций", test_divergence_detection),
    ("Типы сигналов", test_signal_types),
    ("Производительность", test_performance)
  ]

  print("\nВыберите тест:")
  for i, (name, _) in enumerate(tests, 1):
    print(f"{i}. {name}")
  print(f"{len(tests) + 1}. Все тесты")
  print("0. Выход")

  choice = input("\nВаш выбор: ").strip()

  if choice == '0':
    return
  elif choice == str(len(tests) + 1):
    # Запуск всех тестов
    for name, test_func in tests:
      try:
        test_func()
      except Exception as e:
        print(f"\n❌ Ошибка в тесте '{name}': {e}")
  else:
    try:
      idx = int(choice) - 1
      if 0 <= idx < len(tests):
        tests[idx][1]()
      else:
        print("Неверный выбор")
    except ValueError:
      print("Неверный ввод")

  print("\n✅ Тестирование завершено!")


if __name__ == "__main__":
  main()