
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from core.enums import SignalType
from ml.volatility_system import VolatilityPredictor, VolatilityRegime, VolatilityPredictionSystem

import pandas as pd
from decimal import Decimal, ROUND_DOWN
from core.bybit_connector import BybitConnector
from core.data_fetcher import DataFetcher
from core.schemas import TradingSignal
from data.database_manager import AdvancedDatabaseManager
from utils.logging_config import get_logger
import logging
signal_logger = logging.getLogger('SignalTrace')
logger = get_logger(__name__)


class AdvancedRiskManager:
  """
  Продвинутый риск-менеджер, использующий настройки из конфига.
  """

  def __init__(self, db_manager: AdvancedDatabaseManager, settings: Dict[str, Any], data_fetcher: DataFetcher, volatility_predictor: Optional[VolatilityPredictionSystem] = None):
    self.db_manager = db_manager
    self.config = settings
    self.data_fetcher = data_fetcher
    self.trade_settings = self.config.get('trade_settings', {})
    self.strategy_settings = self.config.get('strategy_settings', {})
    self.volatility_system = volatility_predictor
    # Копируем коэффициенты из EnhancedRiskManager
    self.volatility_multipliers = {
      VolatilityRegime.VERY_LOW: {
        'stop_loss': 0.7,  # Уменьшаем SL при низкой волатильности
        'take_profit': 0.8,  # Уменьшаем TP
        'position_size': 1.2  # Увеличиваем размер позиции
      },
      VolatilityRegime.LOW: {
        'stop_loss': 0.85,
        'take_profit': 0.9,
        'position_size': 1.1
      },
      VolatilityRegime.NORMAL: {
        'stop_loss': 1.0,
        'take_profit': 1.0,
        'position_size': 1.0
      },
      VolatilityRegime.HIGH: {
        'stop_loss': 1.3,
        'take_profit': 1.4,
        'position_size': 0.8
      },
      VolatilityRegime.VERY_HIGH: {
        'stop_loss': 1.6,  # Увеличиваем SL при высокой волатильности
        'take_profit': 1.8,  # Увеличиваем TP
        'position_size': 0.6  # Уменьшаем размер позиции
      }
    }  # Вставьте сюда словарь multipliers
    logger.info("AdvancedRiskManager инициализирован с настройками из config.")

  async def _get_active_symbols(self, current_symbol: str) -> List[str]:
    """Асинхронно получает список активных символов из БД."""
    query = "SELECT DISTINCT symbol FROM trades WHERE status = 'OPEN' AND symbol != ?"
    rows = await self.db_manager._execute(query, (current_symbol,), fetch='all')
    return [row['symbol'] for row in rows] if rows else []

  async def _check_daily_loss(self, account_balance: float) -> Dict[str, Any]:
    """Асинхронно проверяет, превышен ли дневной лимит потерь."""
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    query = """
        SELECT SUM(profit_loss) as total_loss
        FROM trades 
        WHERE status = 'CLOSED' AND close_timestamp >= ? AND profit_loss < 0
    """
    result = await self.db_manager._execute(query, (today_start,), fetch='one')

    total_loss = result['total_loss'] if result and result['total_loss'] is not None else 0

    # Избегаем деления на ноль, если баланс нулевой
    loss_percent = abs(total_loss) / account_balance if account_balance > 0 else 0
    limit_percent = self.max_daily_loss_percent

    return {
      'exceeded': loss_percent >= limit_percent,
      'current': loss_percent,
      'limit': limit_percent,
      'amount': total_loss
    }



  async def validate_signal(self, signal: TradingSignal, symbol: str, account_balance: float, market_data: pd.DataFrame) -> Dict[str, Any]:
    """
    ФИНАЛЬНАЯ ВЕРСИЯ №2: Сначала определяет самое строгое минимальное требование (по количеству или по стоимости),
    а затем корректирует размер сделки.
    """
    validation_result = {'approved': False, 'recommended_size': 0.0, 'reasons': []}

    # Проверка уверенности сигнала
    confidence_threshold = self.strategy_settings.get('signal_confidence_threshold', 0.55)
    if signal.confidence < confidence_threshold:
      validation_result['reasons'].append(f"Низкая уверенность")
      return validation_result

    # 1. Рассчитываем наш ИДЕАЛЬНЫЙ размер позиции на основе нашего риска
    recommended_size = self._calculate_position_size(account_balance, signal.price)
    if recommended_size <= 0:
      validation_result['reasons'].append("Нулевой баланс или цена")
      return validation_result

    # 2. Получаем правила торговли для инструмента
    instrument_info = await self.data_fetcher.get_instrument_info(symbol)
    if not instrument_info:
      validation_result['reasons'].append(f"Нет правил торговли для {symbol}")
      return validation_result

    try:
      # 3. Извлекаем все лимиты из правил биржи и нашего конфига
      lot_size_filter = instrument_info.get('lotSizeFilter', {})
      min_order_qty = float(lot_size_filter.get('minOrderQty', '0'))
      max_order_qty = float(lot_size_filter.get('maxOrderQty', '1e12'))
      qty_step_str = lot_size_filter.get('qtyStep')
      min_order_value = self.trade_settings.get('min_order_value_usdt', 5.5)

      # --- НОВЫЙ, УМНЫЙ АЛГОРИТМ ---

      # 4. Рассчитываем, какое количество монет нужно, чтобы удовлетворить минимальной СТОИМОСТИ ордера
      qty_needed_for_min_value = (min_order_value / signal.price) if signal.price > 0 else float('inf')

      # 5. Определяем "ИСТИННЫЙ МИНИМУМ" - выбираем большее из двух минимальных требований
      true_min_qty = max(min_order_qty, qty_needed_for_min_value)
      logger.debug(
        f"РИСК-МЕНЕДЖЕР для {symbol}: Истинный минимальный размер сделки: {true_min_qty:.4f} (max из {min_order_qty} по кол-ву и {qty_needed_for_min_value:.4f} по стоимости)")

      final_size = recommended_size

      # 6. Сравниваем наш идеальный размер с истинным минимумом
      if final_size < true_min_qty:
        logger.warning(
          f"РИСК-МЕНЕДЖЕР для {symbol}: Расчетный размер {final_size:.4f} меньше истинного минимума {true_min_qty:.4f}. Размер будет увеличен.")
        final_size = true_min_qty

      # 7. Проверяем на максимальный размер и корректируем под шаг лота
      if final_size > max_order_qty:
        final_size = max_order_qty

      adjusted_size = float(
        (Decimal(str(final_size)) / Decimal(qty_step_str)).to_integral_value(rounding=ROUND_DOWN) * Decimal(
          qty_step_str))

      # Вместо старого блока расчета SL/TP вставляем новый
      # --- НОВЫЙ БЛОК: АДАПТИВНЫЙ РАСЧЕТ SL/TP ---
      signal.stop_loss, signal.take_profit, new_size = self._calculate_adaptive_risk_params(signal, market_data,
                                                                                            adjusted_size)
      adjusted_size = new_size  # Обновляем размер позиции
      # --- КОНЕЦ НОВОГО БЛОКА ---

      # 8. Финальная проверка баланса
      cost_of_trade = adjusted_size * signal.price / self.trade_settings.get('leverage', 1)
      if cost_of_trade > account_balance:
        validation_result['reasons'].append(f"Недостаточно баланса для итоговой сделки ({cost_of_trade:.2f} USDT)")
        return validation_result

      validation_result.update({
        'approved': True,
        'recommended_size': adjusted_size,
        'reasons': ["Сигнал одобрен с адаптивными SL/TP и размером"],
        'stop_loss': signal.stop_loss,  # <--- ИСПРАВЛЕНО
        'take_profit': signal.take_profit  # <--- ИСПРАВЛЕНО
      })

    except Exception as e:
      logger.error(f"Ошибка парсинга правил торговли для {symbol}: {e}")
      validation_result['reasons'].append("Ошибка обработки правил торговли")
      return validation_result

    logger.info(f"РИСК-МЕНЕДЖЕР для {symbol}: Итоговое решение -> {validation_result}")
    return validation_result


  def _calculate_position_size(self, balance: float, price: float) -> float:
    """Рассчитывает размер позиции на основе настроек из конфига."""
    size_type = self.trade_settings.get('order_size_type', 'percentage')
    size_value = self.trade_settings.get('order_size_value', 1.0)
    leverage = self.trade_settings.get('leverage', 10)

    if price <= 0: return 0.0

    if size_type == 'fixed':
      # Фиксированный размер в USDT
      total_usdt = size_value * leverage
      return total_usdt / price
    else:  # percentage
      # Процент от баланса
      usdt_at_risk = balance * (size_value / 100.0)
      total_usdt = usdt_at_risk * leverage
      return total_usdt / price

  def _calculate_adaptive_risk_params(self, signal: TradingSignal, market_data: pd.DataFrame, base_size: float) -> \
  Tuple[float, float, float]:
    trade_settings = self.config.get('trade_settings', {})
    leverage = trade_settings.get('leverage', 10)
    base_sl_pct = (trade_settings.get('roi_stop_loss_pct', 20.0) / 100.0) / leverage
    base_tp_pct = (trade_settings.get('roi_take_profit_pct', 60.0) / 100.0) / leverage

    if not self.volatility_system or not self.volatility_system.predictor.is_fitted:
      if signal.signal_type == SignalType.BUY:
        return signal.price * (1 - base_sl_pct), signal.price * (1 + base_tp_pct), base_size
      else:
        return signal.price * (1 + base_sl_pct), signal.price * (1 - base_tp_pct), base_size

    try:
      prediction = self.volatility_system.get_prediction(market_data)
      if not prediction: raise ValueError("Прогноз волатильности не получен")

      multipliers = self.volatility_multipliers[prediction.volatility_regime]

      adapted_sl_pct = base_sl_pct * multipliers['stop_loss']
      adapted_tp_pct = base_tp_pct * multipliers['take_profit']
      adapted_size = base_size * multipliers['position_size']

      # --- НОВЫЙ БЛОК ЛОГИРОВАНИЯ ---
      logger.info(f"РИСК-МЕНЕДЖЕР для {signal.symbol}:")
      logger.info(
        f"  - Прогноз волатильности: {prediction.predicted_volatility:.4f} (Режим: {prediction.volatility_regime.value})")
      logger.info(
        f"  - Применены множители: SL x{multipliers['stop_loss']:.2f}, TP x{multipliers['take_profit']:.2f}, Размер x{multipliers['position_size']:.2f}")
      logger.info(f"  - Базовый SL/TP (% от цены): SL={base_sl_pct:.2%}, TP={base_tp_pct:.2%}")
      logger.info(f"  - Адаптивный SL/TP (% от цены): SL={adapted_sl_pct:.2%}, TP={adapted_tp_pct:.2%}")
      # --- КОНЕЦ НОВОГО БЛОКА ---

      if signal.signal_type == SignalType.BUY:
        sl = signal.price * (1 - adapted_sl_pct)
        tp = signal.price * (1 + adapted_tp_pct)
      else:
        sl = signal.price * (1 + adapted_sl_pct)
        tp = signal.price * (1 - adapted_tp_pct)

      return sl, tp, adapted_size
    except Exception as e:
      logger.error(f"Ошибка при расчете адаптивных SL/TP: {e}. Используются базовые значения.")
      return self._calculate_adaptive_risk_params(signal, market_data, base_size)