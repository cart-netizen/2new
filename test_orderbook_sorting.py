#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест для проверки получения и сортировки стакана ордеров от Bybit API
(Версия без Unicode эмодзи для совместимости с Windows cp1251)

Проверяет:
1. Подключение к API
2. Получение данных orderbook
3. Правильность сортировки bids (по убыванию) и asks (по возрастанию)
4. Валидацию данных
5. Обработку различных сценариев ошибок
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch, MagicMock

# Настройка логирования для совместимости с Windows
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  handlers=[
    logging.StreamHandler(sys.stdout),
    logging.FileHandler('orderbook_test.log', encoding='utf-8')
  ]
)

logger = logging.getLogger(__name__)


class OrderBookTester:
  """Класс для тестирования функциональности orderbook"""

  def __init__(self, bybit_connector=None):
    self.connector = bybit_connector
    self.test_results = {
      'passed': 0,
      'failed': 0,
      'tests': []
    }

  def log_test_result(self, test_name: str, passed: bool, message: str = ""):
    """Логирует результат теста"""
    status = "[PASSED]" if passed else "[FAILED]"
    logger.info(f"{status}: {test_name} - {message}")

    self.test_results['tests'].append({
      'name': test_name,
      'passed': passed,
      'message': message,
      'timestamp': datetime.now().isoformat()
    })

    if passed:
      self.test_results['passed'] += 1
    else:
      self.test_results['failed'] += 1

  def validate_orderbook_structure(self, orderbook: Dict) -> Tuple[bool, str]:
    """Проверяет структуру данных orderbook"""
    try:
      # Проверяем наличие основных ключей
      required_keys = ['symbol', 'bids', 'asks']
      missing_keys = [key for key in required_keys if key not in orderbook]
      if missing_keys:
        return False, f"Missing keys: {missing_keys}"

      # Проверяем типы данных
      if not isinstance(orderbook['bids'], list):
        return False, "Bids is not a list"

      if not isinstance(orderbook['asks'], list):
        return False, "Asks is not a list"

      # Проверяем минимальное количество уровней
      if len(orderbook['bids']) == 0:
        return False, "No bids data"

      if len(orderbook['asks']) == 0:
        return False, "No asks data"

      # Проверяем структуру каждого уровня
      for i, bid in enumerate(orderbook['bids'][:3]):  # Проверяем первые 3
        if not isinstance(bid, list) or len(bid) < 2:
          return False, f"Invalid bid[{i}] structure: {bid}"

        try:
          price = float(bid[0])
          volume = float(bid[1])
          if price <= 0 or volume <= 0:
            return False, f"Invalid bid[{i}] values: price={price}, volume={volume}"
        except (ValueError, TypeError) as e:
          return False, f"Bid[{i}] conversion error: {e}"

      for i, ask in enumerate(orderbook['asks'][:3]):  # Проверяем первые 3
        if not isinstance(ask, list) or len(ask) < 2:
          return False, f"Invalid ask[{i}] structure: {ask}"

        try:
          price = float(ask[0])
          volume = float(ask[1])
          if price <= 0 or volume <= 0:
            return False, f"Invalid ask[{i}] values: price={price}, volume={volume}"
        except (ValueError, TypeError) as e:
          return False, f"Ask[{i}] conversion error: {e}"

      return True, "Structure is valid"

    except Exception as e:
      return False, f"Critical validation error: {e}"

  def check_sorting(self, orderbook: Dict) -> Tuple[bool, str]:
    """Проверяет правильность сортировки bids и asks"""
    try:
      bids = orderbook.get('bids', [])
      asks = orderbook.get('asks', [])

      # Проверяем сортировку bids (должны быть по убыванию цены)
      for i in range(len(bids) - 1):
        try:
          current_price = float(bids[i][0])
          next_price = float(bids[i + 1][0])

          if current_price < next_price:
            return False, f"Bids incorrectly sorted: {current_price} < {next_price} at positions {i}, {i + 1}"
        except (ValueError, IndexError) as e:
          return False, f"Bids check error at position {i}: {e}"

      # Проверяем сортировку asks (должны быть по возрастанию цены)
      for i in range(len(asks) - 1):
        try:
          current_price = float(asks[i][0])
          next_price = float(asks[i + 1][0])

          if current_price > next_price:
            return False, f"Asks incorrectly sorted: {current_price} > {next_price} at positions {i}, {i + 1}"
        except (ValueError, IndexError) as e:
          return False, f"Asks check error at position {i}: {e}"

      # Проверяем, что лучший bid меньше лучшего ask
      if bids and asks:
        try:
          best_bid = float(bids[0][0])
          best_ask = float(asks[0][0])

          if best_bid >= best_ask:
            return False, f"Invalid spread: best_bid ({best_bid}) >= best_ask ({best_ask})"

          # Проверяем разумность спреда (не более 10%)
          spread_pct = (best_ask - best_bid) / best_bid * 100
          if spread_pct > 10:
            return False, f"Suspicious large spread: {spread_pct:.2f}%"

        except (ValueError, IndexError) as e:
          return False, f"Spread check error: {e}"

      return True, f"Sorting is correct. Bids: {len(bids)} levels, Asks: {len(asks)} levels"

    except Exception as e:
      return False, f"Critical sorting check error: {e}"

  async def test_orderbook_retrieval(self, symbol: str = "BTCUSDT") -> bool:
    """Тестирует получение orderbook от API"""
    test_name = f"OrderBook Retrieval ({symbol})"

    try:
      if not self.connector:
        self.log_test_result(test_name, False, "Connector not initialized")
        return False

      # Получаем orderbook
      orderbook = await self.connector.fetch_order_book(symbol, depth=10)

      if not orderbook:
        self.log_test_result(test_name, False, "API returned None")
        return False

      # Проверяем структуру
      structure_valid, structure_msg = self.validate_orderbook_structure(orderbook)
      if not structure_valid:
        self.log_test_result(test_name, False, f"Invalid structure: {structure_msg}")
        return False

      # Проверяем сортировку
      sorting_valid, sorting_msg = self.check_sorting(orderbook)
      if not sorting_valid:
        self.log_test_result(test_name, False, f"Invalid sorting: {sorting_msg}")
        return False

      # Логируем успешную информацию
      bids_count = len(orderbook.get('bids', []))
      asks_count = len(orderbook.get('asks', []))
      best_bid = float(orderbook['bids'][0][0]) if orderbook.get('bids') else 0
      best_ask = float(orderbook['asks'][0][0]) if orderbook.get('asks') else 0
      spread = (best_ask - best_bid) / best_bid * 100 if best_bid > 0 else 0

      success_msg = f"Bids: {bids_count}, Asks: {asks_count}, Spread: {spread:.3f}%"
      self.log_test_result(test_name, True, success_msg)
      return True

    except Exception as e:
      self.log_test_result(test_name, False, f"Exception: {e}")
      return False

  def test_sorting_function(self) -> bool:
    """Тестирует функцию сортировки на мок-данных"""
    test_name = "Sorting Function Logic"

    try:
      # Создаем тестовые данные с неправильной сортировкой
      test_orderbook = {
        'symbol': 'TESTUSDT',
        'bids': [
          ['50000.0', '1.0'],  # Должен быть последним
          ['50200.0', '2.0'],  # Должен быть первым
          ['50100.0', '1.5']  # Должен быть вторым
        ],
        'asks': [
          ['50400.0', '1.0'],  # Should be second
          ['50300.0', '2.0'],  # Should be first
          ['50500.0', '1.5']  # Should be third
        ]
      }

      # Исходная сортировка должна быть неправильной
      initial_sorting_valid, _ = self.check_sorting(test_orderbook)
      if initial_sorting_valid:
        self.log_test_result(test_name, False, "Test data is already correctly sorted")
        return False

      # Применяем сортировку
      test_orderbook['bids'].sort(key=lambda x: float(x[0]), reverse=True)
      test_orderbook['asks'].sort(key=lambda x: float(x[0]))

      # Проверяем результат
      sorting_valid, sorting_msg = self.check_sorting(test_orderbook)
      if not sorting_valid:
        self.log_test_result(test_name, False, f"Sorting failed: {sorting_msg}")
        return False

      # Проверяем конкретные значения
      expected_bids_order = ['50200.0', '50100.0', '50000.0']
      expected_asks_order = ['50300.0', '50400.0', '50500.0']

      actual_bids_order = [bid[0] for bid in test_orderbook['bids']]
      actual_asks_order = [ask[0] for ask in test_orderbook['asks']]

      if actual_bids_order != expected_bids_order:
        self.log_test_result(test_name, False, f"Bids: expected {expected_bids_order}, got {actual_bids_order}")
        return False

      if actual_asks_order != expected_asks_order:
        self.log_test_result(test_name, False, f"Asks: expected {expected_asks_order}, got {actual_asks_order}")
        return False

      self.log_test_result(test_name, True, "Sorting logic works correctly")
      return True

    except Exception as e:
      self.log_test_result(test_name, False, f"Exception: {e}")
      return False

  def test_edge_cases(self) -> bool:
    """Тестирует граничные случаи"""
    test_name = "Edge Cases Handling"

    edge_cases = [
      # Пустые данные
      {
        'name': 'Empty orderbook',
        'data': {'symbol': 'TEST', 'bids': [], 'asks': []},
        'should_pass': False
      },
      # Некорректные цены
      {
        'name': 'Invalid prices',
        'data': {
          'symbol': 'TEST',
          'bids': [['abc', '1.0'], ['50000', '2.0']],
          'asks': [['50100', '1.0']]
        },
        'should_pass': False
      },
      # Отрицательные значения
      {
        'name': 'Negative values',
        'data': {
          'symbol': 'TEST',
          'bids': [['-50000', '1.0']],
          'asks': [['50100', '1.0']]
        },
        'should_pass': False
      },
      # Нулевые объемы
      {
        'name': 'Zero volumes',
        'data': {
          'symbol': 'TEST',
          'bids': [['50000', '0']],
          'asks': [['50100', '1.0']]
        },
        'should_pass': False
      },
      # Корректные минимальные данные
      {
        'name': 'Valid minimal data',
        'data': {
          'symbol': 'TEST',
          'bids': [['50000.0', '1.0']],
          'asks': [['50100.0', '1.0']]
        },
        'should_pass': True
      }
    ]

    all_passed = True

    for case in edge_cases:
      try:
        structure_valid, structure_msg = self.validate_orderbook_structure(case['data'])

        if case['should_pass'] and not structure_valid:
          logger.error(f"  [FAILED] {case['name']}: expected success, got error: {structure_msg}")
          all_passed = False
        elif not case['should_pass'] and structure_valid:
          logger.error(f"  [FAILED] {case['name']}: expected error, but validation passed")
          all_passed = False
        else:
          logger.info(f"  [OK] {case['name']}: result matches expectations")

      except Exception as e:
        logger.error(f"  [FAILED] {case['name']}: exception: {e}")
        all_passed = False

    self.log_test_result(test_name, all_passed, f"Checked {len(edge_cases)} edge cases")
    return all_passed

  def validate_and_fix_orderbook(self, orderbook_response: dict, symbol: str) -> Tuple[bool, Optional[Dict], str]:
    """
    Валидирует и исправляет данные orderbook от Bybit API
    """
    try:
      if not orderbook_response or not isinstance(orderbook_response, dict):
        return False, None, "Invalid API response format"

      # Извлекаем данные из ответа
      if 'result' in orderbook_response:
        data = orderbook_response['result']
      else:
        data = orderbook_response

      # Проверяем наличие основных полей
      bids = data.get('b', [])
      asks = data.get('a', [])

      if not bids or not asks:
        return False, None, f"Missing bids ({len(bids)}) or asks ({len(asks)}) data"

      # Флаги для отслеживания исправлений
      bids_fixed = False
      asks_fixed = False

      # Проверяем и исправляем сортировку bids (должны быть по убыванию цены)
      if len(bids) > 1:
        try:
          first_bid_price = float(bids[0][0])
          second_bid_price = float(bids[1][0])

          if first_bid_price < second_bid_price:
            # Неправильная сортировка - исправляем
            bids.sort(key=lambda x: float(x[0]), reverse=True)
            bids_fixed = True
            logger.debug(f"Fixed bids sorting for {symbol}: {first_bid_price} -> {float(bids[0][0])}")

        except (ValueError, IndexError, TypeError) as e:
          return False, None, f"Bids processing error: {e}"

      # Проверяем и исправляем сортировку asks (должны быть по возрастанию цены)
      if len(asks) > 1:
        try:
          first_ask_price = float(asks[0][0])
          second_ask_price = float(asks[1][0])

          if first_ask_price > second_ask_price:
            # Неправильная сортировка - исправляем
            asks.sort(key=lambda x: float(x[0]))
            asks_fixed = True
            logger.debug(f"Fixed asks sorting for {symbol}: {first_ask_price} -> {float(asks[0][0])}")

        except (ValueError, IndexError, TypeError) as e:
          return False, None, f"Asks processing error: {e}"

      # Дополнительная валидация данных
      try:
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0

        # Проверяем что bid < ask
        if best_bid >= best_ask and best_bid > 0:
          return False, None, f"Invalid spread: bid={best_bid} >= ask={best_ask}"

        # Проверяем разумность спреда (не более 10%)
        if best_bid > 0:
          spread_pct = (best_ask - best_bid) / best_bid * 100
          if spread_pct > 10:
            logger.warning(f"Large spread for {symbol}: {spread_pct:.2f}%")

      except (ValueError, TypeError) as e:
        return False, None, f"Price validation error: {e}"

      # Обновляем данные в исходной структуре
      if 'result' in orderbook_response:
        orderbook_response['result']['b'] = bids
        orderbook_response['result']['a'] = asks
      else:
        orderbook_response['b'] = bids
        orderbook_response['a'] = asks

      # Формируем сообщение о статусе
      fixes = []
      if bids_fixed:
        fixes.append("bids")
      if asks_fixed:
        fixes.append("asks")

      if fixes:
        status_msg = f"Fixed sorting: {', '.join(fixes)}"
      else:
        status_msg = "Sorting is correct"

      return True, orderbook_response, status_msg

    except Exception as e:
      return False, None, f"Critical validation error: {e}"

  async def run_comprehensive_test(self, symbols: List[str] = None) -> Dict:
    """Запускает комплексный тест"""
    if symbols is None:
      symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]

    logger.info("STARTING COMPREHENSIVE ORDERBOOK TEST")
    logger.info("=" * 60)

    # Тест 1: Логика сортировки
    logger.info("Test 1: Checking sorting logic")
    self.test_sorting_function()

    # Тест 2: Граничные случаи
    logger.info("Test 2: Checking edge cases")
    self.test_edge_cases()

    # Тест 3: Реальные данные от API
    if self.connector:
      logger.info("Test 3: Getting real data from API")
      for symbol in symbols:
        await self.test_orderbook_retrieval(symbol)
        await asyncio.sleep(0.5)  # Пауза между запросами
    else:
      logger.warning("Test 3: Skipped (connector not initialized)")

    # Итоги
    logger.info("=" * 60)
    logger.info("TEST RESULTS:")
    logger.info(f"Successful: {self.test_results['passed']}")
    logger.info(f"Failed: {self.test_results['failed']}")

    success_rate = (self.test_results['passed'] /
                    (self.test_results['passed'] + self.test_results['failed']) * 100
                    if (self.test_results['passed'] + self.test_results['failed']) > 0 else 0)

    logger.info(f"Success rate: {success_rate:.1f}%")

    # Сохраняем детальный отчет
    report_file = f"orderbook_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
      json.dump(self.test_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Detailed report saved: {report_file}")

    return self.test_results


# Функции для интеграции с существующим кодом

async def test_bybit_orderbook(bybit_connector, symbols: List[str] = None):
  """
  Основная функция для тестирования orderbook с реальным connector

  Usage:
      from core.bybit_connector import BybitConnector
      connector = BybitConnector(api_key="your_key", api_secret="your_secret")
      await test_bybit_orderbook(connector, ["BTCUSDT", "ETHUSDT"])
  """
  tester = OrderBookTester(bybit_connector)
  results = await tester.run_comprehensive_test(symbols)
  return results


def test_sorting_only():
  """
  Функция для тестирования только логики сортировки (без API)

  Usage:
      test_sorting_only()
  """
  tester = OrderBookTester()

  logger.info("Testing orderbook sorting logic")
  logger.info("=" * 50)

  tester.test_sorting_function()
  tester.test_edge_cases()

  logger.info("=" * 50)
  logger.info(f"Successful: {tester.test_results['passed']}")
  logger.info(f"Failed: {tester.test_results['failed']}")

  return tester.test_results


def fix_orderbook_sorting_data(orderbook_data: dict, symbol: str = "UNKNOWN") -> Tuple[bool, dict, str]:
  """
  Утилита для исправления сортировки orderbook

  Usage:
      is_valid, fixed_data, message = fix_orderbook_sorting_data(api_response, "BTCUSDT")
      if is_valid:
          # используем fixed_data
      else:
          # обрабатываем ошибку
  """
  tester = OrderBookTester()
  return tester.validate_and_fix_orderbook(orderbook_data, symbol)


# Примеры использования
if __name__ == "__main__":
  print("OrderBook Tester (Windows Compatible Version)")
  print("Choose test mode:")
  print("1. Sorting logic only (no API)")
  print("2. Full test with API (requires connector setup)")

  try:
    choice = input("Enter number (1 or 2): ").strip()
  except KeyboardInterrupt:
    print("\nTest cancelled.")
    sys.exit(0)

  if choice == "1":
    # Тест только логики
    test_sorting_only()
  elif choice == "2":
    print("\nFor full test you need to:")
    print("1. Import your BybitConnector")
    print("2. Create instance with API keys")
    print("3. Call test_bybit_orderbook(connector)")
    print("\nExample code:")

from core.bybit_connector import BybitConnector


async def main():
  connector = BybitConnector(

  )

      # data_fetcher = DataFetcher(connector, settings_dict)
  results = await test_bybit_orderbook(connector)
  return results

  # Запуск
asyncio.run(main())
