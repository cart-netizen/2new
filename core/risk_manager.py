from datetime import datetime, time
from typing import Dict, List, Tuple, Any

import pandas as pd

from core.bybit_connector import BybitConnector, logger
from core.schemas import TradingSignal
from data.database_manager import AdvancedDatabaseManager


class AdvancedRiskManager:
  """Продвинутый риск-менеджер с динамическим управлением позициями"""

  def __init__(self, db_manager: AdvancedDatabaseManager, connector: BybitConnector = None):
    self.connector = connector
    self.db_manager = db_manager
    self.max_daily_loss_percent = 0.02  # 2% от депозита в день
    self.max_position_size_percent = 0.10  # Максимум 10% депозита на одну позицию
    self.max_correlation_positions = 3  # Максимум коррелированных позиций
    self.min_confidence_threshold = 0.65  # Минимальная уверенность для открытия
    self.correlation_threshold = 0.7  # Порог корреляции для уменьшения позиции
    self.volatility_threshold = 0.8  # Порог волатильности
    self.slippage_threshold = 0.001  # 0.1% допустимого скольжения

    # Кэш для хранения корреляционной матрицы
    self.correlation_cache = {
      'timestamp': None,
      'matrix': None,
      'valid_period': 3600  # Актуальность 1 час
    }

  async def calculate_correlation_matrix(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """Вычисляет корреляционную матрицу для активов на основе исторических данных"""
    try:
      # Проверяем актуальность кэша
      if (self.correlation_cache['timestamp'] and
          (datetime.datetime.now() - self.correlation_cache['timestamp']).seconds < self.correlation_cache[
            'valid_period']):
        return self.correlation_cache['matrix']

      # Получаем исторические данные для всех символов
      cursor = self.db_manager.conn.cursor()
      correlations = {}

      for sym1 in symbols:
        cursor.execute("""
                SELECT close_price, open_timestamp 
                FROM trades 
                WHERE symbol = ? AND status = 'CLOSED'
                ORDER BY open_timestamp DESC LIMIT 1000
            """, (sym1,))
        data1 = cursor.fetchall()

        if not data1:
          continue

        prices1 = pd.Series([x[0] for x in data1])
        correlations[sym1] = {}

        for sym2 in symbols:
          if sym1 == sym2:
            correlations[sym1][sym2] = 1.0
            continue

          cursor.execute("""
                    SELECT close_price 
                    FROM trades 
                    WHERE symbol = ? AND status = 'CLOSED'
                    ORDER BY open_timestamp DESC LIMIT 1000
                """, (sym2,))
          data2 = cursor.fetchall()

          if not data2:
            continue

          prices2 = pd.Series([x[0] for x in data2])

          # Выравниваем по длине
          min_len = min(len(prices1), len(prices2))
          corr = prices1.iloc[:min_len].corr(prices2.iloc[:min_len])
          correlations[sym1][sym2] = corr if not pd.isna(corr) else 0.0

      # Обновляем кэш
      self.correlation_cache = {
        'timestamp': datetime.datetime.now(),
        'matrix': correlations,
        'valid_period': 3600
      }

      return correlations

    except Exception as e:
      print(f"❌ Ошибка расчета корреляционной матрицы: {e}")
      return {}

  async def get_liquidity_metrics(self, symbol: str, quantity: float) -> Dict[str, float]:
    """Оценивает ликвидность и потенциальное скольжение"""
    try:
      # В реальной реализации здесь будет запрос к API биржи
      order_book = await self._fetch_order_book(symbol)

      if not order_book or 'bids' not in order_book or 'asks' not in order_book:
        return {
          'slippage': 0.0,
          'impact': 0.0,
          'spread': 0.0
        }

      best_bid = order_book['bids'][0][0]
      best_ask = order_book['asks'][0][0]
      spread = best_ask - best_bid

      # Расчет скольжения для заданного объема
      slippage_buy = self._calculate_slippage(order_book['asks'], quantity)
      slippage_sell = self._calculate_slippage(order_book['bids'], quantity, is_bid=True)

      # Расчет рыночного воздействия
      impact = quantity / sum([x[1] for x in order_book['asks'][:5]])

      return {
        'slippage': (slippage_buy + slippage_sell) / 2,
        'impact': impact,
        'spread': spread,
        'best_bid': best_bid,
        'best_ask': best_ask
      }

    except Exception as e:
      print(f"❌ Ошибка оценки ликвидности: {e}")
      return {
        'slippage': 0.0,
        'impact': 0.0,
        'spread': 0.0
      }

  def _calculate_slippage(self, orders: List[Tuple[float, float]], quantity: float, is_bid: bool = False) -> float:
    """Вычисляет ожидаемое скольжение для заданного объема"""
    remaining = quantity
    total_cost = 0.0
    slippage = 0.0

    for price, size in orders:
      if remaining <= 0:
        break

      fill_size = min(remaining, size)
      total_cost += fill_size * price
      remaining -= fill_size

    if quantity > 0:
      avg_price = total_cost / quantity
      if is_bid:
        slippage = (orders[0][0] - avg_price) / orders[0][0]
      else:
        slippage = (avg_price - orders[0][0]) / orders[0][0]

    return slippage

  def _get_active_symbols(self, current_symbol: str) -> List[str]:
    """Интеграция с существующей логикой"""
    cursor = self.db_manager.conn.cursor()
    cursor.execute("""
        SELECT DISTINCT symbol 
        FROM trades 
        WHERE status = 'OPEN' AND symbol != ?
    """, (current_symbol,))
    return [row[0] for row in cursor.fetchall()]


  async def validate_signal(self, signal: TradingSignal, symbol: str, account_balance: float) -> Dict[str, Any]:
    """Расширенная валидация сигнала с учетом корреляции и ликвидности"""
    validation_result = {
        'approved': False,
        'recommended_size': 0.0,
        'risk_score': 0.0,
        'liquidity_metrics': None,
        'correlation_risk': 0.0,
        'warnings': [],
        'reasons': []
    }

    # 1. Проверка уверенности
    if signal.confidence < self.min_confidence_threshold:
        validation_result['warnings'].append(f"Низкая уверенность: {signal.confidence:.2%}")
        validation_result['reasons'].append("Сигнал отклонен из-за низкой уверенности")
        return validation_result

    # 2. Проверка дневного лимита потерь (НОВОЕ)
    daily_loss = await self._check_daily_loss(account_balance)
    if daily_loss['exceeded']:
        validation_result['warnings'].append(f"Превышен дневной лимит потерь: {daily_loss['current']:.2%}/{daily_loss['limit']:.2%}")
        validation_result['reasons'].append("Дневной лимит потерь превышен")
        return validation_result

    # 3. Расчет размера позиции на основе риска
    risk_per_trade = abs(signal.price - signal.stop_loss) / signal.price
    max_loss_per_trade = account_balance * 0.01  # 1% от депозита на сделку

    if risk_per_trade > 0:
      base_size = min(
        max_loss_per_trade / (signal.price * risk_per_trade),
        account_balance * self.max_position_size_percent / signal.price
      )
    else:
      base_size = account_balance * 0.02 / signal.price

    # 4. Оценка ликвидности
    liquidity = await self.get_liquidity_metrics(symbol, base_size)
    validation_result['liquidity_metrics'] = liquidity

    if liquidity['slippage'] > self.slippage_threshold:
      # Автоматическое уменьшение размера позиции
      reduction_factor = min(0.7, self.slippage_threshold / liquidity['slippage'])
      base_size *= reduction_factor
      validation_result['warnings'].append(
        f"Высокое скольжение: {liquidity['slippage']:.2%}. Размер уменьшен в {reduction_factor:.1f} раз"
      )

    # 5. Корреляционный анализ
    active_symbols = self._get_active_symbols(symbol)
    corr_matrix = await self.calculate_correlation_matrix(active_symbols + [symbol])

    if corr_matrix:
      max_corr = max(
        [corr_matrix.get(symbol, {}).get(s, 0) for s in active_symbols],
        default=0
      )
      validation_result['correlation_risk'] = max_corr

      if max_corr > self.correlation_threshold:
        base_size *= 0.6  # Значительное уменьшение при высокой корреляции
        validation_result['warnings'].append(
          f"Высокая корреляция ({max_corr:.2f}) с открытыми позициями"
        )

    # 6. Проверка волатильности
    volatility = await self._check_volatility_risk(symbol, signal.price)
    if volatility > self.volatility_threshold:
      base_size *= 0.5
      validation_result['warnings'].append("Экстремальная волатильность рынка")

    # 7. Финальная проверка минимального размера
    if base_size < account_balance * 0.001:  # Минимум 0.1%
      validation_result['reasons'].append("Размер позиции слишком мал")
      return validation_result

    validation_result.update({
      'approved': True,
      'recommended_size': base_size,
      'risk_score': risk_per_trade,
      'reasons': ["Сигнал одобрен с учетом всех факторов риска"]
    })

    return validation_result

  async def _fetch_order_book(self, symbol: str, depth: int = 25) -> Dict[str, List]:
    """
    Получает стакан ордеров с биржи через connector
    с обработкой ошибок и fallback-логикой

    Args:
        symbol: Торговый символ (например 'BTCUSDT')
        depth: Глубина стакана (по умолчанию 25 уровней)

    Returns:
        Словарь с ключами 'bids' и 'asks', где каждый элемент - список [цена, объем]
        Пример: {'bids': [[50000, 1.5], [49900, 2.3]], 'asks': [[50100, 2.1], [50200, 1.8]]}
    """
    try:
      # Основной запрос через connector
      orderbook = await self.connector.fetch_order_book(symbol, depth)

      # Валидация структуры ответа
      if not isinstance(orderbook, dict) or 'bids' not in orderbook or 'asks' not in orderbook:
        raise ValueError("Некорректная структура стакана от биржи")

      # Нормализация данных
      normalized_bids = []
      for bid in orderbook['bids']:
        if len(bid) >= 2 and isinstance(bid[0], (int, float)) and isinstance(bid[1], (int, float)):
          normalized_bids.append([float(bid[0]), float(bid[1])])

      normalized_asks = []
      for ask in orderbook['asks']:
        if len(ask) >= 2 and isinstance(ask[0], (int, float)) and isinstance(ask[1], (int, float)):
          normalized_asks.append([float(ask[0]), float(ask[1])])

      logger.debug(f"Получен стакан для {symbol}: {len(normalized_bids)} bids, {len(normalized_asks)} asks")
      return {
        'bids': normalized_bids,
        'asks': normalized_asks,
        'timestamp': int(time.time() * 1000),  # Мс timestamp
        'symbol': symbol
      }

    except Exception as e:
      logger.error(f"Ошибка получения стакана для {symbol}: {str(e)}")

      # Fallback: пытаемся получить через CCXT напрямую
      try:
        if hasattr(self.connector, 'exchange'):
          ccxt_orderbook = await self.connector.exchange.fetch_order_book(symbol, limit=depth)
          return {
            'bids': ccxt_orderbook['bids'],
            'asks': ccxt_orderbook['asks'],
            'timestamp': ccxt_orderbook['timestamp'],
            'symbol': symbol
          }
      except Exception as ccxt_e:
        logger.error(f"CCXT fallback также failed для {symbol}: {str(ccxt_e)}")

      # Ultimate fallback - пустой стакан
      return {
        'bids': [],
        'asks': [],
        'timestamp': int(time.time() * 1000),
        'symbol': symbol
      }

  async def _check_daily_loss(self, account_balance: float) -> Dict[str, Any]:
    """Проверяет, превышен ли дневной лимит потерь"""
    cursor = self.db_manager.conn.cursor()
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0)

    cursor.execute("""
        SELECT SUM(profit_loss) 
        FROM trades 
        WHERE status = 'CLOSED' AND open_timestamp >= ? AND profit_loss < 0
    """, (today,))

    total_loss = cursor.fetchone()[0] or 0
    loss_percent = abs(total_loss) / account_balance
    limit_percent = self.max_daily_loss_percent

    return {
      'exceeded': loss_percent >= limit_percent,
      'current': loss_percent,
      'limit': limit_percent,
      'amount': total_loss
    }


  async def _check_correlation_risk(self, symbol: str) -> float:
      """Проверяет корреляцию с открытыми позициями"""
      # Простая имитация проверки корреляции
      # В реальности здесь был бы анализ корреляции между активами
      return 0.3  # Низкая корреляция

  async def _check_volatility_risk(self, symbol: str, price: float) -> float:
      """Проверяет риск волатильности"""
      # Простая имитация анализа волатильности
      return 0.5  # Средняя волатильность

  def calculate_position_size_kelly(self, win_rate: float, avg_win: float, avg_loss: float,
                                      account_balance: float) -> float:
      """Вычисляет оптимальный размер позиции по критерию Келли"""
      if avg_loss <= 0 or win_rate <= 0:
        return 0

      # Формула Келли: f = (bp - q) / b
      # где b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
      b = avg_win / abs(avg_loss)
      p = win_rate
      q = 1 - win_rate

      kelly_fraction = (b * p - q) / b

      # Ограничиваем максимальный размер для безопасности
      kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Максимум 25%

      return account_balance * kelly_fraction

  async def update_risk_metrics(self, symbol: str = None):
    """Обновить отображение риск-метрик"""
    try:
      if hasattr(self, 'trade_executor') and self.trade_executor:
        symbols = getattr(self.trade_executor.trading_system, 'active_symbols', []) + [None]
      else:
        symbols = [None]

      self.risk_metrics_table.setRowCount(len(symbols))

      for row, symbol in enumerate(symbols):
        # Получаем метрики
        if hasattr(self, 'trade_executor') and hasattr(self.trade_executor, 'risk_manager'):
          metrics = self.trade_executor.risk_manager.get_risk_metrics(symbol)
        else:
          metrics = None

        # Безопасное заполнение таблицы
        symbol_text = symbol if symbol else "Общие"
        self.risk_metrics_table.setItem(row, 0, QTableWidgetItem(symbol_text))

        if metrics:
          # Безопасное получение атрибутов с значениями по умолчанию
          win_rate = getattr(metrics, 'win_rate', 0.0)
          max_drawdown = getattr(metrics, 'max_drawdown', 0.0)
          profit_factor = getattr(metrics, 'profit_factor', 0.0)
          total_pnl = getattr(metrics, 'total_pnl', 0.0)
          sharpe_ratio = getattr(metrics, 'sharpe_ratio', 0.0)

          self.risk_metrics_table.setItem(row, 1, QTableWidgetItem(f"{win_rate:.1%}"))
          self.risk_metrics_table.setItem(row, 2, QTableWidgetItem(f"{max_drawdown:.2%}"))
          self.risk_metrics_table.setItem(row, 3, QTableWidgetItem(f"{profit_factor:.2f}"))
          self.risk_metrics_table.setItem(row, 4, QTableWidgetItem(f"{total_pnl:.2f}"))
          self.risk_metrics_table.setItem(row, 5, QTableWidgetItem(f"{sharpe_ratio:.2f}"))
        else:
          # Заполняем пустыми значениями
          for col in range(1, 6):
            self.risk_metrics_table.setItem(row, col, QTableWidgetItem("N/A"))

    except Exception as e:
      print(f"Ошибка при обновлении риск-метрик: {e}")

class SignalProcessor:
    """Отдельный процессор для верификации сигналов"""

    def __init__(self, risk_manager: AdvancedRiskManager):
      self.risk_manager = risk_manager

    async def verify_signal(self, signal: TradingSignal, symbol: str, balance: float) -> Dict[str, Any]:
      return await self.risk_manager.validate_signal(signal, symbol, balance)

