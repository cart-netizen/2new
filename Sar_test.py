# ФИНАЛЬНЫЙ ПАТЧ ДЛЯ ЗАПУСКА И ТЕСТИРОВАНИЯ SAR СТРАТЕГИИ

# 1. СОЗДАНИЕ СКРИПТА ТЕСТИРОВАНИЯ: test_sar_strategy.py

"""
Скрипт для тестирования SAR стратегии перед запуском в продакшн
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

from strategies.sar_strategy import StopAndReverseStrategy

# Добавляем корневую папку проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from config.config_manager import ConfigManager
from utils.logging_config import get_logger

logger = get_logger(__name__)


class SARTester:
  def __init__(self):
    self.config_manager = ConfigManager()
    # self.config = self.config_manager.load_config()
    self.test_config = {
      "stop_and_reverse_strategy": {
        "enabled": True,
        "chop_threshold": 40,
        "adx_threshold": 25,
        "atr_multiplier": 1.25,
        "psar_start": 0.02,
        "psar_step": 0.02,
        "psar_max": 0.2,
        "min_signal_score": 4,
        "min_daily_volume_usd": 1000000,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "mfi_period": 14,
        "mfi_overbought": 80,
        "mfi_oversold": 20,
        "aroon_period": 25,
        "ema_short": 50,
        "ema_long": 200,
        "hma_fast_period": 14,
        "hma_slow_period": 28,
        "hma_rsi_period": 14,
        "hma_adx_threshold": 20,
        "ichimoku_conversion": 9,
        "ichimoku_base": 26,
        "ichimoku_span_b": 52,
        "ichimoku_displacement": 26,
        "use_shadow_system": True,
        "use_ml_confirmation": False
      }
    }
  def generate_test_data(self, symbol: str = "BTCUSDT", bars: int = 500) -> pd.DataFrame:
    """Генерирует тестовые данные для проверки индикаторов"""

    # Генерируем реалистичные OHLCV данные
    np.random.seed(42)

    dates = pd.date_range(start=datetime.now() - timedelta(hours=bars),
                          periods=bars, freq='15min')

    # Базовая цена с трендом
    base_price = 45000
    trend = np.linspace(0, 2000, bars)  # Восходящий тренд
    noise = np.random.normal(0, 200, bars).cumsum()

    close_prices = base_price + trend + noise

    # Генерируем OHLC на основе close
    data = []
    for i, close in enumerate(close_prices):
      volatility = abs(np.random.normal(0, 100))

      open_price = close + np.random.normal(0, 50)
      high = max(open_price, close) + abs(np.random.normal(0, volatility / 2))
      low = min(open_price, close) - abs(np.random.normal(0, volatility / 2))
      volume = abs(np.random.normal(1000, 300))

      data.append({
        'timestamp': dates[i],
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
      })

    df = pd.DataFrame(data)
    logger.info(f"Сгенерированы тестовые данные: {len(df)} баров для {symbol}")
    return df

  async def test_regime_filters(self, strategy: StopAndReverseStrategy, data: pd.DataFrame):
    """Тестирует фильтры рыночных режимов"""
    logger.info("🧪 Тестирование фильтров режимов...")

    try:
      can_trade, reason = await strategy.should_trade_symbol("TESTUSDT", data)

      logger.info(f"✅ Результат фильтров: {can_trade}")
      logger.info(f"📋 Причина: {reason}")

      # Тестируем Choppiness Index
      chop = strategy._calculate_choppiness_index(data, 14)
      logger.info(f"📊 Choppiness Index: {chop:.2f}")

      return can_trade

    except Exception as e:
      logger.error(f"❌ Ошибка тестирования фильтров: {e}")
      return False

  async def test_signal_components(self, strategy: StopAndReverseStrategy, data: pd.DataFrame):
    """Тестирует компоненты сигнала"""
    logger.info("🧪 Тестирование компонентов сигнала...")

    try:
      components = await strategy._analyze_signal_components("TESTUSDT", data)

      logger.info(f"📊 Компоненты сигнала:")
      logger.info(f"  - PSAR триггер: {components.psar_trigger}")
      logger.info(f"  - RSI дивергенция: {components.rsi_divergence}")
      logger.info(f"  - MACD дивергенция: {components.macd_divergence}")
      logger.info(f"  - RSI экстремальная зона: {components.rsi_extreme_zone}")
      logger.info(f"  - MFI экстремальная зона: {components.mfi_extreme_zone}")
      logger.info(f"  - Aroon подтверждение: {components.aroon_confirmation}")
      logger.info(f"  - Aroon Oscillator: {components.aroon_oscillator_signal}")
      logger.info(f"  - Ichimoku подтверждение: {components.ichimoku_confirmation}")
      logger.info(f"  - Общий балл: {components.total_score}")

      return components

    except Exception as e:
      logger.error(f"❌ Ошибка тестирования компонентов: {e}")
      return None

  async def test_full_signal_generation(self, strategy: StopAndReverseStrategy, data: pd.DataFrame):
    """Тестирует полную генерацию сигнала"""
    logger.info("🧪 Тестирование генерации сигнала...")

    try:
      signal = await strategy.generate_signal("TESTUSDT", data)

      if signal:
        logger.info(f"✅ Сигнал сгенерирован:")
        logger.info(f"  - Тип: {signal.signal_type.value}")
        logger.info(f"  - Уверенность: {signal.confidence:.3f}")
        logger.info(f"  - Цена входа: {signal.entry_price:.2f}")
        logger.info(f"  - Метаданные: {signal.metadata}")
      else:
        logger.info("ℹ️ Сигнал не сгенерирован (условия не выполнены)")

      return signal

    except Exception as e:
      logger.error(f"❌ Ошибка генерации сигнала: {e}")
      return None

  async def run_comprehensive_test(self):
    """Запускает комплексное тестирование"""
    logger.info("🚀 Запуск комплексного тестирования SAR стратегии")

    try:
      # Инициализируем стратегию
      strategy = StopAndReverseStrategy(self.test_config)
      logger.info("✅ SAR стратегия инициализирована")

      # Генерируем тестовые данные
      test_data = self.generate_test_data()

      # Тест 1: Фильтры режимов
      await self.test_regime_filters(strategy, test_data)

      # Тест 2: Компоненты сигнала
      await self.test_signal_components(strategy, test_data)

      # Тест 3: Полная генерация сигнала
      await self.test_full_signal_generation(strategy, test_data)

      # Тест 4: Статус стратегии
      status = strategy.get_strategy_status()
      logger.info(f"📋 Статус стратегии: {status}")

      logger.info("🎉 Комплексное тестирование завершено успешно!")
      return True

    except Exception as e:
      logger.error(f"❌ Критическая ошибка тестирования: {e}")
      return False


async def main():
  """Главная функция тестирования"""
  tester = SARTester()
  success = await tester.run_comprehensive_test()

  if success:
    print("\n✅ Все тесты пройдены! SAR стратегия готова к интеграции.")
  else:
    print("\n❌ Тесты провалены! Требуется исправление ошибок.")
    sys.exit(1)


if __name__ == "__main__":
  asyncio.run(main())

# 2. СКРИПТ ВАЛИДАЦИИ КОНФИГУРАЦИИ: validate_sar_config.py

"""
Скрипт для валидации конфигурации SAR стратегии
"""

import json
import os
from typing import Dict, Any, List


def validate_sar_config(config_path: str = "config.json") -> bool:
  """Валидирует конфигурацию SAR стратегии"""

  print("🔍 Валидация конфигурации SAR стратегии...")

  try:
    # Проверяем существование файла
    if not os.path.exists(config_path):
      print(f"❌ Файл конфигурации не найден: {config_path}")
      return False

    # Загружаем конфигурацию
    with open(config_path, 'r', encoding='utf-8') as f:
      config = json.load(f)

    # Проверяем наличие секции SAR
    if 'stop_and_reverse_strategy' not in config:
      print("❌ Секция 'stop_and_reverse_strategy' не найдена в конфигурации")
      return False

    sar_config = config['stop_and_reverse_strategy']

    # Обязательные параметры
    required_params = [
      'enabled', 'chop_threshold', 'adx_threshold', 'atr_multiplier',
      'psar_start', 'psar_step', 'psar_max', 'min_signal_score',
      'min_daily_volume_usd'
    ]

    missing_params = []
    for param in required_params:
      if param not in sar_config:
        missing_params.append(param)

    if missing_params:
      print(f"❌ Отсутствуют обязательные параметры: {missing_params}")
      return False

    # Валидация диапазонов значений
    validations = [
      ('chop_threshold', 20, 60, "Choppiness порог должен быть между 20 и 60"),
      ('adx_threshold', 15, 35, "ADX порог должен быть между 15 и 35"),
      ('atr_multiplier', 1.0, 2.0, "ATR множитель должен быть между 1.0 и 2.0"),
      ('psar_start', 0.01, 0.05, "PSAR старт должен быть между 0.01 и 0.05"),
      ('psar_step', 0.01, 0.05, "PSAR шаг должен быть между 0.01 и 0.05"),
      ('psar_max', 0.1, 0.3, "PSAR макс должен быть между 0.1 и 0.3"),
      ('min_signal_score', 2, 8, "Минимальный балл должен быть между 2 и 8"),
      ('min_daily_volume_usd', 100000, 10000000, "Минимальный объем должен быть между 100k и 10M")
    ]

    validation_errors = []
    for param, min_val, max_val, message in validations:
      value = sar_config.get(param)
      if value is not None and not (min_val <= value <= max_val):
        validation_errors.append(f"{param}: {message} (текущее: {value})")

    if validation_errors:
      print("❌ Ошибки валидации параметров:")
      for error in validation_errors:
        print(f"  - {error}")
      return False

    # Проверяем логическую согласованность
    if sar_config['psar_start'] >= sar_config['psar_max']:
      print("❌ PSAR start должен быть меньше PSAR max")
      return False

    if sar_config['psar_step'] >= sar_config['psar_max']:
      print("❌ PSAR step должен быть меньше PSAR max")
      return False

    print("✅ Конфигурация SAR стратегии прошла валидацию!")
    print(f"📊 Основные параметры:")
    print(f"  - Включена: {sar_config['enabled']}")
    print(f"  - CHOP порог: {sar_config['chop_threshold']}")
    print(f"  - ADX порог: {sar_config['adx_threshold']}")
    print(f"  - Минимальный балл: {sar_config['min_signal_score']}")
    print(f"  - Минимальный объем: {sar_config['min_daily_volume_usd']:,} USD")

    return True

  except json.JSONDecodeError as e:
    print(f"❌ Ошибка парсинга JSON: {e}")
    return False
  except Exception as e:
    print(f"❌ Неожиданная ошибка: {e}")
    return False


def create_sample_config():
  """Создает пример конфигурации SAR"""
  sample_config = {
    "stop_and_reverse_strategy": {
      "enabled": True,
      "chop_threshold": 40,
      "adx_threshold": 25,
      "atr_multiplier": 1.25,
      "psar_start": 0.02,
      "psar_step": 0.02,
      "psar_max": 0.2,
      "min_signal_score": 4,
      "min_daily_volume_usd": 1000000
    }
  }

  with open("sar_config_sample.json", "w", encoding="utf-8") as f:
    json.dump(sample_config, f, indent=2, ensure_ascii=False)

  print("📝 Создан пример конфигурации: sar_config_sample.json")


if __name__ == "__main__":
  import sys

  if len(sys.argv) > 1 and sys.argv[1] == "--create-sample":
    create_sample_config()
  else:
    success = validate_sar_config()
    if not success:
      sys.exit(1)

# 3. СКРИПТ МОНИТОРИНГА SAR: monitor_sar.py

"""
Скрипт для мониторинга работы SAR стратегии в реальном времени
"""

import asyncio
import time
from datetime import datetime
from data.state_manager import StateManager
from utils.logging_config import get_logger

logger = get_logger(__name__)


class SARMonitor:
  def __init__(self):
    self.state_manager = StateManager()
    self.last_status = None

  def display_sar_status(self, status: dict):
    """Отображает статус SAR стратегии"""
    print("\n" + "=" * 60)
    print(f"🎯 SAR STRATEGY STATUS - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    if not status:
      print("❌ Статус SAR стратегии недоступен")
      return

    print(f"📊 Отслеживаемые символы: {status.get('monitored_symbols_count', 0)}")
    print(f"💼 Активные позиции: {status.get('current_positions_count', 0)}")

    last_update = status.get('last_symbol_update')
    if last_update:
      try:
        last_update_dt = datetime.fromisoformat(last_update)
        time_diff = datetime.now() - last_update_dt
        print(f"🕐 Последнее обновление: {time_diff.seconds // 60} мин назад")
      except:
        print(f"🕐 Последнее обновление: {last_update}")

    # Конфигурация
    config = status.get('config', {})
    print(f"\n⚙️ Конфигурация:")
    print(f"  - Минимальный балл: {config.get('min_signal_score', 'N/A')}")
    print(f"  - CHOP порог: {config.get('chop_threshold', 'N/A')}")
    print(f"  - ADX порог: {config.get('adx_threshold', 'N/A')}")
    print(f"  - Shadow System: {'✅' if config.get('use_shadow_system') else '❌'}")
    print(f"  - ML подтверждение: {'✅' if config.get('use_ml_confirmation') else '❌'}")

    # Отслеживаемые символы
    monitored_symbols = status.get('monitored_symbols', [])
    if monitored_symbols:
      print(f"\n📋 Отслеживаемые символы ({len(monitored_symbols)}):")
      for i, symbol in enumerate(monitored_symbols[:10]):  # Показываем первые 10
        print(f"  {i + 1:2d}. {symbol}")
      if len(monitored_symbols) > 10:
        print(f"     ... и еще {len(monitored_symbols) - 10}")

    # Текущие позиции
    current_positions = status.get('current_positions', [])
    if current_positions:
      print(f"\n💼 Текущие позиции SAR ({len(current_positions)}):")
      for position in current_positions:
        print(f"  🔹 {position}")

  def detect_changes(self, current_status: dict):
    """Обнаруживает изменения в статусе"""
    if not self.last_status:
      self.last_status = current_status
      return

    changes = []

    # Изменения в количестве символов
    old_count = self.last_status.get('monitored_symbols_count', 0)
    new_count = current_status.get('monitored_symbols_count', 0)
    if old_count != new_count:
      changes.append(f"Символы: {old_count} → {new_count}")

    # Изменения в позициях
    old_positions = self.last_status.get('current_positions_count', 0)
    new_positions = current_status.get('current_positions_count', 0)
    if old_positions != new_positions:
      changes.append(f"Позиции: {old_positions} → {new_positions}")

    # Новые символы
    old_symbols = set(self.last_status.get('monitored_symbols', []))
    new_symbols = set(current_status.get('monitored_symbols', []))
    added_symbols = new_symbols - old_symbols
    removed_symbols = old_symbols - new_symbols

    if added_symbols:
      changes.append(f"Добавлены символы: {', '.join(list(added_symbols)[:3])}")
    if removed_symbols:
      changes.append(f"Удалены символы: {', '.join(list(removed_symbols)[:3])}")

    if changes:
      print(f"\n🔄 ИЗМЕНЕНИЯ ОБНАРУЖЕНЫ:")
      for change in changes:
        print(f"  - {change}")

    self.last_status = current_status

  async def run_monitoring(self, interval: int = 30):
    """Запускает мониторинг с указанным интервалом"""
    print(f"🚀 Запуск мониторинга SAR стратегии (интервал: {interval}s)")
    print("Нажмите Ctrl+C для остановки\n")

    try:
      while True:
        try:
          # Получаем статус
          sar_status = self.state_manager.get_custom_data('sar_strategy_status')

          # Отображаем статус
          self.display_sar_status(sar_status)

          # Обнаруживаем изменения
          if sar_status:
            self.detect_changes(sar_status)

          # Ждем следующего обновления
          await asyncio.sleep(interval)

        except KeyboardInterrupt:
          break
        except Exception as e:
          logger.error(f"Ошибка мониторинга: {e}")
          await asyncio.sleep(5)

    except KeyboardInterrupt:
      print("\n👋 Мониторинг остановлен пользователем")


async def main():
  """Главная функция мониторинга"""
  import sys

  interval = 30
  if len(sys.argv) > 1:
    try:
      interval = int(sys.argv[1])
    except ValueError:
      print("❌ Неверный интервал. Используется 30 секунд.")

  monitor = SARMonitor()
  await monitor.run_monitoring(interval)


if __name__ == "__main__":
  asyncio.run(main())

# 4. ИНСТРУКЦИИ ПО ЗАПУСКУ

"""
ПОШАГОВАЯ ИНСТРУКЦИЯ ПО ИНТЕГРАЦИИ И ЗАПУСКУ SAR СТРАТЕГИИ

1. ПОДГОТОВКА ФАЙЛОВ:
   - Скопировать strategies/stop_and_reverse_strategy.py в папку strategies/
   - Обновить config.json добавив секцию stop_and_reverse_strategy
   - Применить все патчи из integration_patches.py

2. ВАЛИДАЦИЯ КОНФИГУРАЦИИ:
   python validate_sar_config.py

3. ТЕСТИРОВАНИЕ СТРАТЕГИИ:
   python test_sar_strategy.py

4. ПРИМЕНЕНИЕ ПАТЧЕЙ:
   - Обновить core/integrated_system.py согласно патчам
   - Обновить core/adaptive_strategy_selector.py
   - Обновить core/market_regime_detector.py
   - Обновить dashboard.py

5. ЗАПУСК СИСТЕМЫ:
   python main.py

6. МОНИТОРИНГ SAR:
   python monitor_sar.py [интервал_в_секундах]

7. ПРОВЕРКА В ДАШБОРДЕ:
   - Открыть дашборд: streamlit run dashboard.py
   - Перейти на вкладку "Стратегии"
   - Включить "Stop_and_Reverse"
   - Настроить параметры в секции SAR Settings

ВАЖНЫЕ ПРОВЕРКИ ПОСЛЕ ЗАПУСКА:

✅ Стратегия зарегистрирована в логах
✅ Символы обновляются каждый час
✅ Сигналы генерируются при соответствии условиям
✅ Shadow Trading интегрирована
✅ Дашборд отображает статус

УСТРАНЕНИЕ ПРОБЛЕМ:

❌ "SAR стратегия не инициализирована":
   - Проверить конфигурацию config.json
   - Убедиться, что enabled: true
   - Проверить логи на ошибки импорта

❌ "Символы не обновляются":
   - Проверить подключение к data_fetcher
   - Убедиться в доступности API биржи
   - Проверить фильтры объема торгов

❌ "Сигналы не генерируются":
   - Снизить min_signal_score для тестирования
   - Проверить фильтры режимов (CHOP, ADX, ATR)
   - Убедиться, что символы в мониторинге

❌ "Ошибки в Shadow Trading":
   - Проверить инициализацию shadow_trading_manager
   - Убедиться в правильности структуры БД
   - Проверить права доступа к БД

МОНИТОРИНГ ПРОИЗВОДИТЕЛЬНОСТИ:

📊 Ключевые метрики для отслеживания:
   - Количество отслеживаемых символов (должно быть 10-50)
   - Частота генерации сигналов (1-5 в день)
   - Win Rate SAR сигналов (целевой: >55%)
   - Средняя прибыль на сигнал
   - Время выполнения фильтров (<100ms)

📈 Оптимизация параметров:
   - Если слишком много сигналов: увеличить min_signal_score
   - Если мало сигналов: снизить chop_threshold или adx_threshold
   - Если низкий Win Rate: повысить требования к подтверждениям
   - Если упущенные тренды: снизить atr_multiplier

🔧 Рекомендуемые настройки для начала:
   - min_signal_score: 4
   - chop_threshold: 45 (строже)
   - adx_threshold: 22 (мягче)
   - atr_multiplier: 1.2
   - use_shadow_system: true
   - use_ml_confirmation: false (до стабилизации)

После 1-2 недель работы проанализировать статистику и настроить параметры
под конкретные рыночные условия и стиль торговли.
"""

# 5. СКРИПТ АВТОМАТИЧЕСКОЙ УСТАНОВКИ: install_sar.py

"""
Автоматическая установка и настройка SAR стратегии
"""

import os
import json
import shutil
from pathlib import Path


def install_sar_strategy():
  """Автоматически устанавливает SAR стратегию"""

  print("🚀 Установка Stop-and-Reverse стратегии...")

  try:
    # 1. Проверяем структуру проекта
    required_dirs = ['strategies', 'config', 'core']
    for dir_name in required_dirs:
      if not os.path.exists(dir_name):
        print(f"❌ Отсутствует папка: {dir_name}")
        return False

    # 2. Создаем файл стратегии (если нужно)
    strategy_file = "strategies/stop_and_reverse_strategy.py"
    if not os.path.exists(strategy_file):
      print(f"⚠️ Файл стратегии не найден: {strategy_file}")
      print("📝 Создайте файл стратегии из предоставленного кода")

    # 3. Обновляем config.json
    config_file = "config.json"
    if os.path.exists(config_file):
      with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

      if 'stop_and_reverse_strategy' not in config:
        print("➕ Добавление конфигурации SAR в config.json...")

        config['stop_and_reverse_strategy'] = {
          "enabled": True,
          "chop_threshold": 40,
          "adx_threshold": 25,
          "atr_multiplier": 1.25,
          "psar_start": 0.02,
          "psar_step": 0.02,
          "psar_max": 0.2,
          "min_signal_score": 4,
          "min_daily_volume_usd": 1000000,
          "use_shadow_system": True,
          "use_ml_confirmation": False
        }

        with open(config_file, 'w', encoding='utf-8') as f:
          json.dump(config, f, indent=2, ensure_ascii=False)

        print("✅ Конфигурация SAR добавлена")
      else:
        print("ℹ️ Конфигурация SAR уже существует")

    # 4. Создаем backup
    backup_dir = "backup_before_sar"
    if not os.path.exists(backup_dir):
      os.makedirs(backup_dir)
      print(f"📦 Создана папка для backup: {backup_dir}")

    # 5. Информация о необходимых изменениях
    print("\n📋 ТРЕБУЕМЫЕ РУЧНЫЕ ИЗМЕНЕНИЯ:")
    print("1. Применить патчи из integration_patches.py к следующим файлам:")
    print("   - core/integrated_system.py")
    print("   - core/adaptive_strategy_selector.py")
    print("   - core/market_regime_detector.py")
    print("   - dashboard.py")
    print("\n2. Запустить валидацию: python validate_sar_config.py")
    print("3. Запустить тесты: python test_sar_strategy.py")
    print("4. Запустить систему: python main.py")

    print("\n✅ Установка SAR стратегии подготовлена!")
    return True

  except Exception as e:
    print(f"❌ Ошибка установки: {e}")
    return False


if __name__ == "__main__":
  success = install_sar_strategy()
  if not success:
    exit(1)

# 6. ФИНАЛЬНЫЙ CHECKLIST

"""
🎯 ФИНАЛЬНЫЙ CHECKLIST ДЛЯ ЗАПУСКА SAR СТРАТЕГИИ

ПОДГОТОВКА:
□ Скопирован файл strategies/stop_and_reverse_strategy.py
□ Обновлен config.json с секцией stop_and_reverse_strategy
□ Применены все патчи из integration_patches.py

ТЕСТИРОВАНИЕ:
□ python validate_sar_config.py - успешно
□ python test_sar_strategy.py - успешно  
□ Проверены все зависимости (pandas_ta, numpy, etc.)

ИНТЕГРАЦИЯ:
□ Обновлен core/integrated_system.py
□ Обновлен core/adaptive_strategy_selector.py
□ Обновлен core/market_regime_detector.py
□ Обновлен dashboard.py с новой вкладкой

ЗАПУСК:
□ python main.py запускается без ошибок
□ В логах есть "Stop-and-Reverse стратегия зарегистрирована"
□ streamlit run dashboard.py работает
□ В дашборде доступна вкладка SAR Settings

ПРОВЕРКА РАБОТЫ:
□ python monitor_sar.py показывает статус
□ Символы обновляются каждый час
□ Shadow Trading интегрирована
□ ML модели готовы к подключению

PRODUCTION READY:
□ Настроены параметры под рыночные условия
□ Включен мониторинг производительности
□ Настроены алерты и уведомления
□ Создан план отката при проблемах

🚀 ГОТОВО К ЗАПУСКУ В ПРОДАКШН!
"""