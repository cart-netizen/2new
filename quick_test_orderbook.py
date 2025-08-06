#!/usr/bin/env python3
"""
Быстрый тест для проверки orderbook в рамках существующего проекта
#запуст python quick_test_orderbook.py --mode logic
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_config import get_logger

logger = get_logger(__name__)


class QuickOrderBookTest:
  """Быстрый тест orderbook для интеграции с существующим проектом"""

  def __init__(self):
    self.test_results = []

  def log_result(self, test_name: str, success: bool, details: str = ""):
    """Логирует результат теста"""
    status = "✅ УСПЕХ" if success else "❌ ОШИБКА"
    message = f"{status}: {test_name}"
    if details:
      message += f" - {details}"

    logger.info(message)
    self.test_results.append({
      'test': test_name,
      'success': success,
      'details': details,
      'timestamp': datetime.now().isoformat()
    })

  def validate_and_fix_orderbook(self, orderbook_data: dict) -> tuple[bool, dict, str]:
    """
    Валидирует и исправляет данные orderbook

    Returns:
        (is_valid_after_fix, fixed_orderbook, message)
    """
    try:
      if not orderbook_data or not isinstance(orderbook_data, dict):
        return False, {}, "Некорректный формат данных"

      # Извлекаем данные
      if 'result' in orderbook_data:
        data = orderbook_data['result']
      else:
        data = orderbook_data

      bids = data.get('b', [])
      asks = data.get('a', [])

      if not bids or not asks:
        return False, orderbook_data, "Отсутствуют данные bids или asks"

      # Проверяем и исправляем сортировку bids (по убыванию цены)
      bids_fixed = False
      if len(bids) > 1:
        try:
          # Проверяем текущую сортировку
          first_bid_price = float(bids[0][0])
          second_bid_price = float(bids[1][0])

          if first_bid_price < second_bid_price:
            # Неправильная сортировка - исправляем
            bids.sort(key=lambda x: float(x[0]), reverse=True)
            bids_fixed = True
            logger.info(f"🔧 Исправлена сортировка bids: {first_bid_price} -> {float(bids[0][0])}")

        except (ValueError, IndexError) as e:
          return False, orderbook_data, f"Ошибка обработки bids: {e}"

      # Проверяем и исправляем сортировку asks (по возрастанию цены)
      asks_fixed = False
      if len(asks) > 1:
        try:
          # Проверяем текущую сортировку
          first_ask_price = float(asks[0][0])
          second_ask_price = float(asks[1][0])

          if first_ask_price > second_ask_price:
            # Неправильная сортировка - исправляем
            asks.sort(key=lambda x: float(x[0]))
            asks_fixed = True
            logger.info(f"🔧 Исправлена сортировка asks: {first_ask_price} -> {float(asks[0][0])}")

        except (ValueError, IndexError) as e:
          return False, orderbook_data, f"Ошибка обработки asks: {e}"

      # Обновляем данные
      if 'result' in orderbook_data:
        orderbook_data['result']['b'] = bids
        orderbook_data['result']['a'] = asks
      else:
        orderbook_data['b'] = bids
        orderbook_data['a'] = asks

      # Финальная валидация
      try:
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0

        if best_bid >= best_ask and best_bid > 0 and best_ask > 0:
          return False, orderbook_data, f"Некорректный spread: bid={best_bid} >= ask={best_ask}"

        spread_pct = (best_ask - best_bid) / best_bid * 100 if best_bid > 0 else 0

        fix_message = []
        if bids_fixed:
          fix_message.append("исправлены bids")
        if asks_fixed:
          fix_message.append("исправлены asks")

        if fix_message:
          message = f"Данные исправлены ({', '.join(fix_message)}). Spread: {spread_pct:.3f}%"
        else:
          message = f"Сортировка корректна. Spread: {spread_pct:.3f}%"

        return True, orderbook_data, message

      except Exception as e:
        return False, orderbook_data, f"Ошибка финальной валидации: {e}"

    except Exception as e:
      return False, orderbook_data, f"Критическая ошибка: {e}"

  async def test_with_bybit_connector(self, connector, test_symbols: list = None):
    """Тестирует с реальным BybitConnector"""
    if test_symbols is None:
      test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    logger.info("🚀 Запуск теста orderbook с реальным API")
    logger.info("=" * 60)

    successful_tests = 0
    total_tests = 0

    for symbol in test_symbols:
      total_tests += 1
      test_name = f"OrderBook Test - {symbol}"

      try:
        logger.info(f"📊 Тестирование получения orderbook для {symbol}...")

        # Получаем данные
        raw_response = await connector._make_request(
          'GET',
          '/v5/market/orderbook',
          params={
            'category': getattr(connector, 'default_category', 'linear'),
            'symbol': symbol,
            'limit': 10
          }
        )

        if not raw_response:
          self.log_result(test_name, False, "API вернул пустой ответ")
          continue

        # Валидируем и исправляем
        is_valid, fixed_data, message = self.validate_and_fix_orderbook(raw_response)

        if is_valid:
          # Дополнительные проверки
          result_data = fixed_data.get('result', fixed_data)
          bids = result_data.get('b', [])
          asks = result_data.get('a', [])

          if bids and asks:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = (best_ask - best_bid) / best_bid * 100

            details = f"Bids: {len(bids)}, Asks: {len(asks)}, Spread: {spread:.3f}%. {message}"
            self.log_result(test_name, True, details)
            successful_tests += 1
          else:
            self.log_result(test_name, False, "Пустые bids/asks после обработки")
        else:
          self.log_result(test_name, False, message)

        # Пауза между запросами
        await asyncio.sleep(0.5)

      except Exception as e:
        self.log_result(test_name, False, f"Исключение: {str(e)[:100]}")

    # Итоги
    logger.info("=" * 60)
    logger.info(f"📈 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    logger.info(f"✅ Успешно: {successful_tests}/{total_tests}")
    logger.info(f"📊 Процент успеха: {(successful_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%")

    if successful_tests == total_tests:
      logger.info("🎉 Все тесты прошли успешно!")
    elif successful_tests > 0:
      logger.warning("⚠️ Некоторые тесты не прошли. Проверьте детали выше.")
    else:
      logger.error("❌ Все тесты провалились. Требуется диагностика API подключения.")

    return {
      'total': total_tests,
      'successful': successful_tests,
      'failed': total_tests - successful_tests,
      'success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
      'details': self.test_results
    }

  def create_mock_orderbook_data(self, symbol: str, correct_sorting: bool = True) -> dict:
    """Создает мок-данные orderbook для тестирования"""
    if correct_sorting:
      # Правильно отсортированные данные
      bids = [
        ['50200.50', '1.5'],  # Highest bid
        ['50200.00', '2.0'],
        ['50199.50', '1.0'],
        ['50199.00', '0.5']  # Lowest bid
      ]
      asks = [
        ['50205.00', '1.0'],  # Lowest ask
        ['50205.50', '1.5'],
        ['50206.00', '2.0'],
        ['50206.50', '0.8']  # Highest ask
      ]
    else:
      # Неправильно отсортированные данные
      bids = [
        ['50199.00', '0.5'],  # Should be last
        ['50200.50', '1.5'],  # Should be first
        ['50199.50', '1.0'],  # Should be third
        ['50200.00', '2.0']  # Should be second
      ]
      asks = [
        ['50206.00', '2.0'],  # Should be third
        ['50205.00', '1.0'],  # Should be first
        ['50206.50', '0.8'],  # Should be fourth
        ['50205.50', '1.5']  # Should be second
      ]

    return {
      'result': {
        'symbol': symbol,
        's': symbol,
        'b': bids,
        'a': asks,
        'ts': int(datetime.now().timestamp() * 1000),
        'u': 123456
      }
    }

  def test_sorting_logic(self):
    """Тестирует логику сортировки на мок-данных"""
    logger.info("🧪 Тестирование логики сортировки...")

    # Тест 1: Корректные данные
    correct_data = self.create_mock_orderbook_data("TESTUSDT", correct_sorting=True)
    is_valid, _, message = self.validate_and_fix_orderbook(correct_data)
    self.log_result("Корректные данные", is_valid, message)

    # Тест 2: Некорректные данные
    incorrect_data = self.create_mock_orderbook_data("TESTUSDT", correct_sorting=False)
    is_valid, fixed_data, message = self.validate_and_fix_orderbook(incorrect_data)
    self.log_result("Исправление некорректных данных", is_valid, message)

    # Тест 3: Проверяем результат исправления
    if is_valid:
      result = fixed_data['result']
      bids = result['b']
      asks = result['a']

      # Проверяем правильность сортировки после исправления
      bids_sorted_correctly = all(
        float(bids[i][0]) >= float(bids[i + 1][0])
        for i in range(len(bids) - 1)
      )
      asks_sorted_correctly = all(
        float(asks[i][0]) <= float(asks[i + 1][0])
        for i in range(len(asks) - 1)
      )

      if bids_sorted_correctly and asks_sorted_correctly:
        self.log_result("Проверка результата сортировки", True,
                        f"Bids и Asks правильно отсортированы после исправления")
      else:
        self.log_result("Проверка результата сортировки", False,
                        f"Сортировка не сработала: bids_ok={bids_sorted_correctly}, asks_ok={asks_sorted_correctly}")

    # Тест 4: Граничные случаи
    edge_cases = [
      {
        'name': 'Пустые bids',
        'data': {'result': {'symbol': 'TEST', 'b': [], 'a': [['50000', '1.0']]}},
        'should_pass': False
      },
      {
        'name': 'Пустые asks',
        'data': {'result': {'symbol': 'TEST', 'b': [['50000', '1.0']], 'a': []}},
        'should_pass': False
      },
      {
        'name': 'Некорректные цены',
        'data': {'result': {'symbol': 'TEST', 'b': [['abc', '1.0']], 'a': [['50000', '1.0']]}},
        'should_pass': False
      },
      {
        'name': 'Один уровень каждого типа',
        'data': {'result': {'symbol': 'TEST', 'b': [['49999.99', '1.0']], 'a': [['50000.01', '1.0']]}},
        'should_pass': True
      }
    ]

    for case in edge_cases:
      is_valid, _, message = self.validate_and_fix_orderbook(case['data'])
      expected_result = case['should_pass']
      test_passed = (is_valid == expected_result)

      self.log_result(f"Граничный случай: {case['name']}", test_passed,
                      f"Ожидалось: {'успех' if expected_result else 'ошибка'}, получено: {'успех' if is_valid else 'ошибка'}")

  def run_full_test_suite(self):
    """Запускает полный набор тестов без API"""
    logger.info("🚀 Запуск полного набора тестов (без API)")
    logger.info("=" * 60)

    self.test_sorting_logic()

    # Подсчет результатов
    successful = sum(1 for result in self.test_results if result['success'])
    total = len(self.test_results)

    logger.info("=" * 60)
    logger.info(f"📊 ОБЩИЕ РЕЗУЛЬТАТЫ:")
    logger.info(f"✅ Успешно: {successful}/{total}")
    logger.info(f"📈 Процент успеха: {(successful / total * 100):.1f}%" if total > 0 else "0%")

    return {
      'total': total,
      'successful': successful,
      'failed': total - successful,
      'success_rate': (successful / total * 100) if total > 0 else 0,
      'details': self.test_results
    }


# Главная функция для интеграции с существующим проектом
async def run_orderbook_diagnostics(bybit_connector=None):
  """
  Главная функция для диагностики orderbook

  Использование в main.py или другом файле:

  from quick_test_orderbook import run_orderbook_diagnostics

  # Только тестирование логики
  await run_orderbook_diagnostics()

  # С реальным API
  await run_orderbook_diagnostics(your_bybit_connector)
  """
  tester = QuickOrderBookTest()

  if bybit_connector:
    logger.info("🔌 Подключен реальный API connector - запуск полного теста")
    return await tester.test_with_bybit_connector(bybit_connector)
  else:
    logger.info("🧪 API connector не подключен - запуск только тестов логики")
    return tester.run_full_test_suite()


# Утилита для исправления orderbook в существующем коде
def fix_orderbook_sorting(orderbook_data: dict) -> tuple[bool, dict, str]:
  """
  Утилита для исправления сортировки orderbook в существующем коде

  Использование:
  from quick_test_orderbook import fix_orderbook_sorting

  # В вашем bybit_connector.py
  async def fetch_order_book(self, symbol: str, depth: int = 25):
      # ... получение данных ...

      # Исправление сортировки
      is_valid, fixed_data, message = fix_orderbook_sorting(response)
      if not is_valid:
          logger.warning(f"Проблема с orderbook для {symbol}: {message}")
          return None

      # Используем исправленные данные
      return self._process_orderbook(fixed_data)
  """
  tester = QuickOrderBookTest()
  return tester.validate_and_fix_orderbook(orderbook_data)


# Точка входа для прямого запуска
if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description='OrderBook Diagnostic Tool')
  parser.add_argument('--mode', choices=['logic', 'api'], default='logic',
                      help='Режим тестирования: logic (только логика) или api (с реальным API)')
  parser.add_argument('--symbols', nargs='*', default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
                      help='Символы для тестирования (только для режима api)')

  args = parser.parse_args()


  async def main():
    if args.mode == 'logic':
      await run_orderbook_diagnostics()
    else:
      print("Для режима 'api' необходимо:")
      print("1. Импортировать ваш BybitConnector")
      print("2. Создать экземпляр с API ключами")
      print("3. Передать его в run_orderbook_diagnostics()")
      print("\nПример:")
      print("""
from core.bybit_connector import BybitConnector

async def test_with_api():
    connector = BybitConnector(api_key="...", api_secret="...", testnet=True)
    return await run_orderbook_diagnostics(connector)

# Запуск
results = await test_with_api()
            """)

      # Запуск только логики как fallback
      print("\nЗапуск тестирования логики...")
      await run_orderbook_diagnostics()


  # Запуск
  asyncio.run(main())