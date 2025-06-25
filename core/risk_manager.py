
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

    # Выводим отчет о ROI настройках при инициализации
    logger.info("=== ИНИЦИАЛИЗАЦИЯ ROI СИСТЕМЫ ===")

    roi_report = self.get_roi_summary_report()
    logger.info("ROI КОНФИГУРАЦИЯ:")
    for line in roi_report.split('\n'):
      if line.strip():
        logger.info(line)

    # Валидируем ROI параметры
    roi_validation = self.validate_roi_parameters()
    if not roi_validation['is_valid']:
      logger.error("КРИТИЧЕСКИЕ ОШИБКИ В ROI НАСТРОЙКАХ:")
      for error in roi_validation['errors']:
        logger.error(f"  ❌ {error}")

    if roi_validation['warnings']:
      logger.warning("ПРЕДУПРЕЖДЕНИЯ ПО ROI НАСТРОЙКАМ:")
      for warning in roi_validation['warnings']:
        logger.warning(f"  ⚠️  {warning}")

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

      try:
        sl_tp_result = await self.calculate_unified_sl_tp(signal, market_data, method='auto')

        # Устанавливаем рассчитанные SL/TP в сигнал


        signal.stop_loss = sl_tp_result['stop_loss']
        signal.take_profit = sl_tp_result['take_profit']
        logger.info(
          f"Унифицированные SL/TP: метод={sl_tp_result['method_used']}, SL={signal.stop_loss:.6f}, TP={signal.take_profit:.6f}")
      except Exception as sl_tp_error:
        logger.error(f"Ошибка унифицированного расчета SL/TP: {sl_tp_error}")
        # Fallback к старому методу
        signal.stop_loss, signal.take_profit, adjusted_size = self._calculate_adaptive_risk_params(signal, market_data,
                                                                                                   adjusted_size)

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
        # 'stop_loss': signal.stop_loss,  # <--- ИСПРАВЛЕНО
        # 'take_profit': signal.take_profit  # <--- ИСПРАВЛЕНО
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
    """
    ИСПРАВЛЕННАЯ ВЕРСИЯ: Расчет адаптивных параметров риска на основе ROI без рекурсии
    """
    # Получаем настройки ROI из конфигурации
    trade_settings = self.config.get('trade_settings', {})
    strategy_settings = self.config.get('strategy_settings', {})

    leverage = trade_settings.get('leverage', 10)

    # Базовые параметры ROI из конфига
    roi_stop_loss_pct = trade_settings.get('roi_stop_loss_pct', 5.0)  # % от маржи
    roi_take_profit_pct = trade_settings.get('roi_take_profit_pct', 60.0)  # % от маржи

    # Преобразуем ROI в процент от цены с учетом плеча
    base_sl_pct = (roi_stop_loss_pct / 100.0) / leverage
    base_tp_pct = (roi_take_profit_pct / 100.0) / leverage

    # Проверяем доступность системы волатильности
    if not self.volatility_system or not self.volatility_system.predictor.is_fitted:
      logger.info(f"Система волатильности недоступна для {signal.symbol}, используем базовые ROI значения")
      logger.info(f"  - ROI SL: {roi_stop_loss_pct}% от маржи → {base_sl_pct:.2%} от цены")
      logger.info(f"  - ROI TP: {roi_take_profit_pct}% от маржи → {base_tp_pct:.2%} от цены")

      if signal.signal_type == SignalType.BUY:
        return signal.price * (1 - base_sl_pct), signal.price * (1 + base_tp_pct), base_size
      else:
        return signal.price * (1 + base_sl_pct), signal.price * (1 - base_tp_pct), base_size

    try:
      # Получаем прогноз волатильности
      prediction = self.volatility_system.get_prediction(market_data)
      if not prediction:
        logger.warning(f"Не удалось получить прогноз волатильности для {signal.symbol}")
      # Возвращаем базовые ROI значения БЕЗ рекурсии
      logger.warning(f"Используем базовые ROI значения: SL={roi_stop_loss_pct}%, TP={roi_take_profit_pct}%")
      if signal.signal_type == SignalType.BUY:
        return signal.price * (1 - base_sl_pct), signal.price * (1 + base_tp_pct), base_size
      else:
        return signal.price * (1 + base_sl_pct), signal.price * (1 - base_tp_pct), base_size

      # Применяем множители на основе режима волатильности
      multipliers = self.volatility_multipliers[prediction.volatility_regime]

      adapted_sl_pct = base_sl_pct * multipliers['stop_loss']
      adapted_tp_pct = base_tp_pct * multipliers['take_profit']
      adapted_size = base_size * multipliers['position_size']

      # Ограничиваем экстремальные значения
      adapted_sl_pct = max(0.001, min(0.1, adapted_sl_pct))  # От 0.1% до 10%
      adapted_tp_pct = max(0.001, min(0.2, adapted_tp_pct))  # От 0.1% до 20%

      # --- БЛОК ЛОГИРОВАНИЯ ---
      logger.info(f"РИСК-МЕНЕДЖЕР для {signal.symbol}:")
      logger.info(f"  - ROI параметры: SL={roi_stop_loss_pct}% от маржи, TP={roi_take_profit_pct}% от маржи")
      logger.info(f"  - Плечо: {leverage}x")
      logger.info(
        f"  - Прогноз волатильности: {prediction.predicted_volatility:.4f} (Режим: {prediction.volatility_regime.value})")
      logger.info(
        f"  - Применены множители: SL x{multipliers['stop_loss']:.2f}, TP x{multipliers['take_profit']:.2f}, Размер x{multipliers['position_size']:.2f}")
      logger.info(f"  - Базовый SL/TP (% от цены): SL={base_sl_pct:.2%}, TP={base_tp_pct:.2%}")
      logger.info(f"  - Адаптивный SL/TP (% от цены): SL={adapted_sl_pct:.2%}, TP={adapted_tp_pct:.2%}")
      # --- КОНЕЦ БЛОКА ЛОГИРОВАНИЯ ---

      # Рассчитываем финальные уровни
      if signal.signal_type == SignalType.BUY:
        sl = signal.price * (1 - adapted_sl_pct)
        tp = signal.price * (1 + adapted_tp_pct)
      else:
        sl = signal.price * (1 + adapted_sl_pct)
        tp = signal.price * (1 - adapted_tp_pct)

      logger.info(f"  - Итоговые значения: SL={sl:.6f}, TP={tp:.6f}, Размер={adapted_size:.4f}")

      return sl, tp, adapted_size

    except Exception as e:
      # Возвращаем базовые ROI значения БЕЗ рекурсии
      logger.error(f"Ошибка при расчете адаптивных SL/TP для {signal.symbol}: {e}")
      logger.warning(f"Используем базовые ROI значения: SL={roi_stop_loss_pct}%, TP={roi_take_profit_pct}%")
      if signal.signal_type == SignalType.BUY:
        return signal.price * (1 - base_sl_pct), signal.price * (1 + base_tp_pct), base_size
      else:
        return signal.price * (1 + base_sl_pct), signal.price * (1 - base_tp_pct), base_size

  async def calculate_adaptive_sl_tp(self, symbol: str, entry_price: float, signal_type: SignalType,
                                     market_data: pd.DataFrame) -> Dict[str, float]:
    """
    Рассчитывает адаптивные уровни SL/TP на основе ROI и волатильности

    Args:
        symbol: Торговый символ
        entry_price: Цена входа
        signal_type: Тип сигнала (BUY/SELL)
        market_data: Рыночные данные

    Returns:
        Dict с параметрами SL/TP
    """
    try:
      # Получаем ROI параметры из конфигурации
      strategy_settings = self.config.get('strategy_settings', {})
      trade_settings = self.config.get('trade_settings', {})

      # ROI параметры (% от маржи)
      roi_stop_loss_pct = trade_settings.get('roi_stop_loss_pct', 5.0)
      roi_take_profit_pct = trade_settings.get('roi_take_profit_pct', 60.0)
      leverage = trade_settings.get('leverage', 10)

      # Множители ATR (для резервного расчета)
      sl_multiplier = float(strategy_settings.get('sl_multiplier', 1.5))
      tp_multiplier = float(strategy_settings.get('tp_multiplier', 3.0))

      # Преобразуем ROI в процент изменения цены
      base_sl_pct = (roi_stop_loss_pct / 100.0) / leverage
      base_tp_pct = (roi_take_profit_pct / 100.0) / leverage

      result = {
        'stop_loss': 0,
        'take_profit': 0,
        'sl_distance_pct': roi_stop_loss_pct,  # ROI в процентах от маржи
        'tp_distance_pct': roi_take_profit_pct,  # ROI в процентах от маржи
        'method_used': 'roi_based'
      }

      logger.info(f"Расчет SL/TP для {symbol} на основе ROI:")
      logger.info(f"  - ROI SL: {roi_stop_loss_pct}% от маржи")
      logger.info(f"  - ROI TP: {roi_take_profit_pct}% от маржи")
      logger.info(f"  - Плечо: {leverage}x")
      logger.info(f"  - SL % от цены: {base_sl_pct:.2%}")
      logger.info(f"  - TP % от цены: {base_tp_pct:.2%}")

      # Проверяем наличие системы волатильности для адаптации
      adaptive_sl_pct = base_sl_pct
      adaptive_tp_pct = base_tp_pct

      if self.volatility_system and self.volatility_system.predictor.is_fitted:
        try:
          prediction = self.volatility_system.get_prediction(market_data)

          if prediction and prediction.volatility_regime:
            # Применяем адаптивные множители к ROI значениям
            regime_multipliers = self.volatility_multipliers.get(prediction.volatility_regime, {
              'stop_loss': 1.0,
              'take_profit': 1.0
            })

            adaptive_sl_pct = base_sl_pct * regime_multipliers.get('stop_loss', 1.0)
            adaptive_tp_pct = base_tp_pct * regime_multipliers.get('take_profit', 1.0)

            # Ограничиваем экстремальные значения
            adaptive_sl_pct = max(0.001, min(0.1, adaptive_sl_pct))  # От 0.1% до 10%
            adaptive_tp_pct = max(0.001, min(0.2, adaptive_tp_pct))  # От 0.1% до 20%

            logger.info(f"  - Режим волатильности: {prediction.volatility_regime.value}")
            logger.info(
              f"  - Адаптивные множители: SL x{regime_multipliers.get('stop_loss', 1.0):.2f}, TP x{regime_multipliers.get('take_profit', 1.0):.2f}")
            logger.info(f"  - Адаптивный SL: {adaptive_sl_pct:.2%}, TP: {adaptive_tp_pct:.2%}")

            result['method_used'] = 'roi_volatility_adaptive'

        except Exception as vol_error:
          logger.warning(f"Ошибка адаптации волатильности для {symbol}: {vol_error}")
          logger.info("Используем базовые ROI значения")
      else:
        logger.info(f"Система волатильности недоступна для {symbol}, используем базовые ROI значения")
      # Рассчитываем финальные уровни SL/TP
      if signal_type == SignalType.BUY:
        result['stop_loss'] = entry_price * (1 - adaptive_sl_pct)
        result['take_profit'] = entry_price * (1 + adaptive_tp_pct)
      else:
        result['stop_loss'] = entry_price * (1 + adaptive_sl_pct)
        result['take_profit'] = entry_price * (1 - adaptive_tp_pct)

      # Рассчитываем фактические ROI значения для логирования
      actual_sl_roi = (adaptive_sl_pct * leverage * 100)
      actual_tp_roi = (adaptive_tp_pct * leverage * 100)

      result['sl_distance_pct'] = actual_sl_roi
      result['tp_distance_pct'] = actual_tp_roi

      logger.info(f"Финальные значения для {symbol}:")
      logger.info(f"  - SL цена: {result['stop_loss']:.6f} (ROI: {actual_sl_roi:.1f}%)")
      logger.info(f"  - TP цена: {result['take_profit']:.6f} (ROI: {actual_tp_roi:.1f}%)")

      return result

    except Exception as e:
      logger.error(f"Критическая ошибка в calculate_adaptive_sl_tp для {symbol}: {e}")

      # Возвращаем безопасные ROI значения по умолчанию
      strategy_settings = self.config.get('strategy_settings', {})
      trade_settings = self.config.get('trade_settings', {})

      safe_roi_sl = trade_settings.get('roi_stop_loss_pct', 5.0)  # 10% ROI
      safe_roi_tp = trade_settings.get('roi_take_profit_pct', 20.0)  # 20% ROI
      leverage = trade_settings.get('leverage', 10)

      safe_sl_pct = (safe_roi_sl / 100.0) / leverage
      safe_tp_pct = (safe_roi_tp / 100.0) / leverage

      return {
        'stop_loss': entry_price * (0.95 if signal_type == SignalType.BUY else 1.05),
        'take_profit': entry_price * (1.05 if signal_type == SignalType.BUY else 0.95),
        'sl_distance_pct': safe_roi_sl,
        'tp_distance_pct': safe_roi_tp,
        'method_used': 'emergency_fallback'
      }

  def calculate_roi_based_sl_tp(self, entry_price: float, signal_type: SignalType,
                                custom_sl_roi: Optional[float] = None,
                                custom_tp_roi: Optional[float] = None) -> Dict[str, float]:
    """
    Прямой расчет SL/TP на основе ROI без адаптации по волатильности

    Args:
        entry_price: Цена входа
        signal_type: Тип сигнала (BUY/SELL)
        custom_sl_roi: Пользовательский ROI для SL (% от маржи)
        custom_tp_roi: Пользовательский ROI для TP (% от маржи)

    Returns:
        Dict с параметрами SL/TP
    """
    try:
      # Получаем настройки из конфигурации
      strategy_settings = self.config.get('strategy_settings', {})
      trade_settings = self.config.get('trade_settings', {})

      # Используем пользовательские значения или из конфига
      roi_sl_pct = custom_sl_roi if custom_sl_roi is not None else strategy_settings.get('roi_stop_loss_pct', 20.0)
      roi_tp_pct = custom_tp_roi if custom_tp_roi is not None else strategy_settings.get('roi_take_profit_pct', 60.0)
      leverage = trade_settings.get('leverage', 10)

      # Преобразуем ROI в процент изменения цены
      sl_price_change_pct = (roi_sl_pct / 100.0) / leverage
      tp_price_change_pct = (roi_tp_pct / 100.0) / leverage

      # Рассчитываем уровни
      if signal_type == SignalType.BUY:
        stop_loss = entry_price * (1 - sl_price_change_pct)
        take_profit = entry_price * (1 + tp_price_change_pct)
      else:  # SELL
        stop_loss = entry_price * (1 + sl_price_change_pct)
        take_profit = entry_price * (1 - tp_price_change_pct)

      logger.debug(f"ROI расчет SL/TP:")
      logger.debug(f"  - ROI SL: {roi_sl_pct}% → цена изменится на {sl_price_change_pct:.2%}")
      logger.debug(f"  - ROI TP: {roi_tp_pct}% → цена изменится на {tp_price_change_pct:.2%}")
      logger.debug(f"  - SL: {stop_loss:.6f}, TP: {take_profit:.6f}")

      return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'sl_distance_pct': roi_sl_pct,
        'tp_distance_pct': roi_tp_pct,
        'sl_price_change_pct': sl_price_change_pct * 100,
        'tp_price_change_pct': tp_price_change_pct * 100,
        'leverage_used': leverage,
        'method_used': 'pure_roi'
      }

    except Exception as e:
      logger.error(f"Ошибка в calculate_roi_based_sl_tp: {e}")
      return {
        'stop_loss': entry_price * 0.98,
        'take_profit': entry_price * 1.02,
        'sl_distance_pct': 10.0,
        'tp_distance_pct': 10.0,
        'method_used': 'error_fallback'
      }

  def validate_roi_parameters(self) -> Dict[str, Any]:
      """
      Валидация ROI параметров из конфигурации

      Returns:
          Dict с результатами валидации и рекомендациями
      """
      try:
        strategy_settings = self.config.get('strategy_settings', {})
        trade_settings = self.config.get('trade_settings', {})

        roi_sl_pct = trade_settings.get('roi_stop_loss_pct', 5.0)
        roi_tp_pct = trade_settings.get('roi_take_profit_pct', 60.0)
        leverage = trade_settings.get('leverage', 10)

        validation_result = {
          'is_valid': True,
          'warnings': [],
          'errors': [],
          'recommendations': [],
          'current_settings': {
            'roi_sl_pct': roi_sl_pct,
            'roi_tp_pct': roi_tp_pct,
            'leverage': leverage
          }
        }

        # Проверка разумности значений
        if roi_sl_pct < 1.0:
          validation_result['warnings'].append(f"Очень низкий SL ROI: {roi_sl_pct}% - может привести к частым потерям")
        elif roi_sl_pct > 50.0:
          validation_result['warnings'].append(
            f"Очень высокий SL ROI: {roi_sl_pct}% - может привести к большим потерям")

        if roi_tp_pct < 5.0:
          validation_result['warnings'].append(f"Очень низкий TP ROI: {roi_tp_pct}% - малая прибыль")
        elif roi_tp_pct > 200.0:
          validation_result['warnings'].append(f"Очень высокий TP ROI: {roi_tp_pct}% - может редко достигаться")

        # Проверка соотношения риск/доходность
        risk_reward_ratio = roi_tp_pct / roi_sl_pct
        if risk_reward_ratio < 1.5:
          validation_result['warnings'].append(f"Низкое соотношение риск/доходность: 1:{risk_reward_ratio:.1f}")
          validation_result['recommendations'].append("Рекомендуется соотношение минимум 1:2")
        elif risk_reward_ratio > 10:
          validation_result['warnings'].append(f"Очень высокое соотношение риск/доходность: 1:{risk_reward_ratio:.1f}")
          validation_result['recommendations'].append("Слишком оптимистичные ожидания прибыли")

        # Проверка плеча
        if leverage < 1:
          validation_result['errors'].append("Плечо не может быть меньше 1")
          validation_result['is_valid'] = False
        elif leverage > 100:
          validation_result['warnings'].append(f"Очень высокое плечо: {leverage}x - повышенный риск")

        # Расчет фактических изменений цены
        sl_price_change = (roi_sl_pct / 100.0) / leverage * 100
        tp_price_change = (roi_tp_pct / 100.0) / leverage * 100

        validation_result['price_impact'] = {
          'sl_price_change_pct': sl_price_change,
          'tp_price_change_pct': tp_price_change
        }

        # Предупреждения о малых изменениях цены
        if sl_price_change < 0.1:
          validation_result['warnings'].append(f"Очень маленькое изменение цены для SL: {sl_price_change:.2f}%")
        if tp_price_change < 0.2:
          validation_result['warnings'].append(f"Очень маленькое изменение цены для TP: {tp_price_change:.2f}%")

        # Рекомендации по оптимизации
        if len(validation_result['warnings']) == 0:
          validation_result['recommendations'].append("Настройки ROI выглядят сбалансированными")
        else:
          validation_result['recommendations'].append("Рассмотрите корректировку параметров на основе предупреждений")

        return validation_result

      except Exception as e:
        logger.error(f"Ошибка валидации ROI параметров: {e}")
        return {
          'is_valid': False,
          'errors': [f"Ошибка валидации: {str(e)}"],
          'warnings': [],
          'recommendations': ["Проверьте корректность конфигурации"]
        }

  def convert_roi_to_price_targets(self, entry_price: float, signal_type: SignalType) -> Dict[str, Any]:
    """
    Конвертирует ROI параметры в конкретные ценовые цели

    Args:
        entry_price: Цена входа
        signal_type: Тип сигнала

    Returns:
        Dict с детальной информацией о ценовых целях
    """
    try:
      strategy_settings = self.config.get('strategy_settings', {})
      trade_settings = self.config.get('trade_settings', {})

      roi_sl_pct = trade_settings.get('roi_stop_loss_pct', 5.0)
      roi_tp_pct = trade_settings.get('roi_take_profit_pct', 60.0)
      leverage = trade_settings.get('leverage', 10)

      # Расчет изменений цены
      sl_price_change_pct = (roi_sl_pct / 100.0) / leverage
      tp_price_change_pct = (roi_tp_pct / 100.0) / leverage

      if signal_type == SignalType.BUY:
        sl_price = entry_price * (1 - sl_price_change_pct)
        tp_price = entry_price * (1 + tp_price_change_pct)
        sl_distance = entry_price - sl_price
        tp_distance = tp_price - entry_price
      else:  # SELL
        sl_price = entry_price * (1 + sl_price_change_pct)
        tp_price = entry_price * (1 - tp_price_change_pct)
        sl_distance = sl_price - entry_price
        tp_distance = entry_price - tp_price

      return {
        'entry_price': entry_price,
        'signal_type': signal_type.value,
        'stop_loss': {
          'price': sl_price,
          'distance_abs': sl_distance,
          'distance_pct': sl_price_change_pct * 100,
          'roi_pct': roi_sl_pct
        },
        'take_profit': {
          'price': tp_price,
          'distance_abs': tp_distance,
          'distance_pct': tp_price_change_pct * 100,
          'roi_pct': roi_tp_pct
        },
        'risk_reward_ratio': roi_tp_pct / roi_sl_pct,
        'leverage': leverage,
        'calculation_method': 'roi_based'
      }

    except Exception as e:
      logger.error(f"Ошибка конвертации ROI в ценовые цели: {e}")
      return {
        'error': str(e),
        'entry_price': entry_price,
        'signal_type': signal_type.value
      }

  def get_roi_summary_report(self) -> str:
    """
    Создает текстовый отчет о текущих ROI настройках

    Returns:
        Форматированный текстовый отчет
    """
    try:
      validation = self.validate_roi_parameters()

      report = []
      report.append("=== ОТЧЕТ О НАСТРОЙКАХ ROI ===")
      report.append("")

      settings = validation.get('current_settings', {})
      report.append("Текущие настройки:")
      report.append(f"  • Stop-Loss ROI: {settings.get('roi_sl_pct', 'N/A')}% от маржи")
      report.append(f"  • Take-Profit ROI: {settings.get('roi_tp_pct', 'N/A')}% от маржи")
      report.append(f"  • Плечо: {settings.get('leverage', 'N/A')}x")

      if 'price_impact' in validation:
        impact = validation['price_impact']
        report.append("")
        report.append("Влияние на цену:")
        report.append(f"  • SL требует изменения цены на: {impact.get('sl_price_change_pct', 0):.2f}%")
        report.append(f"  • TP требует изменения цены на: {impact.get('tp_price_change_pct', 0):.2f}%")

      if validation.get('warnings'):
        report.append("")
        report.append("⚠️  ПРЕДУПРЕЖДЕНИЯ:")
        for warning in validation['warnings']:
          report.append(f"  • {warning}")

      if validation.get('errors'):
        report.append("")
        report.append("❌ ОШИБКИ:")
        for error in validation['errors']:
          report.append(f"  • {error}")

      if validation.get('recommendations'):
        report.append("")
        report.append("💡 РЕКОМЕНДАЦИИ:")
        for rec in validation['recommendations']:
          report.append(f"  • {rec}")

      report.append("")
      report.append("=" * 30)

      return "\n".join(report)

    except Exception as e:
      return f"Ошибка создания отчета ROI: {e}"

  async def calculate_unified_sl_tp(self, signal: TradingSignal, market_data: pd.DataFrame,
                                      method: str = 'auto') -> Dict[str, Any]:
      """
      Унифицированный метод расчета SL/TP с выбором стратегии

      Args:
          signal: Торговый сигнал
          market_data: Рыночные данные
          method: Метод расчета ('auto', 'roi', 'atr', 'volatility', 'hybrid')

      Returns:
          Dict с полной информацией о SL/TP
      """
      try:
        result = {
          'symbol': signal.symbol,
          'signal_type': signal.signal_type.value,
          'entry_price': signal.price,
          'timestamp': datetime.now().isoformat(),
          'method_used': method,
          'stop_loss': 0.0,
          'take_profit': 0.0,
          'metadata': {}
        }

        logger.info(f"Расчет SL/TP для {signal.symbol} методом '{method}'")

        # Автоматический выбор метода
        if method == 'auto':
          if self.volatility_system and self.volatility_system.predictor.is_fitted:
            method = 'hybrid'  # ROI + волатильность
            logger.info(f"Автовыбор: используем гибридный метод (ROI + волатильность)")
          else:
            method = 'roi'  # Только ROI
            logger.info(f"Автовыбор: система волатильности недоступна, используем ROI")

        # 1. ROI-based метод (базовый)
        if method in ['roi', 'hybrid']:
          roi_result = self.calculate_roi_based_sl_tp(signal.price, signal.signal_type)

          result.update({
            'stop_loss': roi_result['stop_loss'],
            'take_profit': roi_result['take_profit'],
            'sl_roi_pct': roi_result['sl_distance_pct'],
            'tp_roi_pct': roi_result['tp_distance_pct']
          })

          result['metadata']['roi_calculation'] = roi_result
          logger.info(f"ROI расчет: SL={roi_result['stop_loss']:.6f}, TP={roi_result['take_profit']:.6f}")

        # 2. Адаптация по волатильности (если доступна)
        if method in ['volatility', 'hybrid'] and self.volatility_system:
          try:
            adaptive_result = await self.calculate_adaptive_sl_tp(
              signal.symbol, signal.price, signal.signal_type, market_data
            )

            if method == 'hybrid':
              # Комбинируем ROI и волатильность
              logger.info("Применяем адаптацию по волатильности к ROI базе")
              result.update({
                'stop_loss': adaptive_result['stop_loss'],
                'take_profit': adaptive_result['take_profit'],
                'method_used': 'roi_volatility_hybrid'
              })
            else:
              # Только волатильность
              result.update({
                'stop_loss': adaptive_result['stop_loss'],
                'take_profit': adaptive_result['take_profit'],
                'method_used': 'volatility_adaptive'
              })

            result['metadata']['volatility_calculation'] = adaptive_result
            logger.info(
              f"Адаптивный расчет: SL={adaptive_result['stop_loss']:.6f}, TP={adaptive_result['take_profit']:.6f}")

          except Exception as vol_error:
            logger.warning(f"Ошибка адаптации по волатильности: {vol_error}, используем ROI")
            if 'roi_calculation' not in result['metadata']:
              # Fallback к ROI если волатильность не сработала
              roi_result = self.calculate_roi_based_sl_tp(signal.price, signal.signal_type)
              result.update({
                'stop_loss': roi_result['stop_loss'],
                'take_profit': roi_result['take_profit']
              })

        # 3. ATR-based метод (альтернативный)
        elif method == 'atr':
          try:
            atr_result = await self._calculate_atr_based_sl_tp(signal, market_data)
            result.update({
              'stop_loss': atr_result['stop_loss'],
              'take_profit': atr_result['take_profit'],
              'method_used': 'atr_based'
            })
            result['metadata']['atr_calculation'] = atr_result
            logger.info(f"ATR расчет: SL={atr_result['stop_loss']:.6f}, TP={atr_result['take_profit']:.6f}")

          except Exception as atr_error:
            logger.warning(f"Ошибка ATR расчета: {atr_error}, fallback к ROI")
            roi_result = self.calculate_roi_based_sl_tp(signal.price, signal.signal_type)
            result.update({
              'stop_loss': roi_result['stop_loss'],
              'take_profit': roi_result['take_profit'],
              'method_used': 'roi_fallback'
            })

        # Валидация результатов
        if result['stop_loss'] <= 0 or result['take_profit'] <= 0:
          logger.error(f"Некорректные SL/TP значения для {signal.symbol}")
          raise ValueError("Получены некорректные значения SL/TP")

        # Добавляем дополнительную информацию
        sl_distance_pct = abs(result['stop_loss'] - signal.price) / signal.price * 100
        tp_distance_pct = abs(result['take_profit'] - signal.price) / signal.price * 100

        result.update({
          'sl_distance_abs': abs(result['stop_loss'] - signal.price),
          'tp_distance_abs': abs(result['take_profit'] - signal.price),
          'sl_distance_pct': sl_distance_pct,
          'tp_distance_pct': tp_distance_pct,
          'risk_reward_ratio': tp_distance_pct / sl_distance_pct if sl_distance_pct > 0 else 0
        })

        # Финальное логирование
        logger.info(f"ИТОГОВЫЕ SL/TP для {signal.symbol}:")
        logger.info(f"  Метод: {result['method_used']}")
        logger.info(f"  SL: {result['stop_loss']:.6f} (-{sl_distance_pct:.2f}%)")
        logger.info(f"  TP: {result['take_profit']:.6f} (+{tp_distance_pct:.2f}%)")
        logger.info(f"  Risk/Reward: 1:{result['risk_reward_ratio']:.2f}")

        return result

      except Exception as e:
        logger.error(f"Критическая ошибка в calculate_unified_sl_tp для {signal.symbol}: {e}")

        # Emergency fallback
        emergency_sl_pct = 0.02  # 2%
        emergency_tp_pct = 0.04  # 4%

        if signal.signal_type == SignalType.BUY:
          emergency_sl = signal.price * (1 - emergency_sl_pct)
          emergency_tp = signal.price * (1 + emergency_tp_pct)
        else:
          emergency_sl = signal.price * (1 + emergency_sl_pct)
          emergency_tp = signal.price * (1 - emergency_tp_pct)

        return {
          'symbol': signal.symbol,
          'signal_type': signal.signal_type.value,
          'entry_price': signal.price,
          'stop_loss': emergency_sl,
          'take_profit': emergency_tp,
          'method_used': 'emergency_fallback',
          'error': str(e),
          'timestamp': datetime.now().isoformat()
        }

  async def _calculate_atr_based_sl_tp(self, signal: TradingSignal, market_data: pd.DataFrame) -> Dict[str, float]:
    """
    Расчет SL/TP на основе ATR (резервный метод)
    """
    try:
      strategy_settings = self.config.get('strategy_settings', {})
      sl_multiplier = strategy_settings.get('sl_multiplier', 1.5)
      tp_multiplier = strategy_settings.get('tp_multiplier', 3.0)

      # Рассчитываем ATR
      if len(market_data) >= 14:
        high_low = market_data['high'] - market_data['low']
        high_close = abs(market_data['high'] - market_data['close'].shift())
        low_close = abs(market_data['low'] - market_data['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_value = true_range.rolling(14).mean().iloc[-1]

        if pd.notna(atr_value) and atr_value > 0:
          if signal.signal_type == SignalType.BUY:
            stop_loss = signal.price - (atr_value * sl_multiplier)
            take_profit = signal.price + (atr_value * tp_multiplier)
          else:
            stop_loss = signal.price + (atr_value * sl_multiplier)
            take_profit = signal.price - (atr_value * tp_multiplier)

          return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr_value': atr_value,
            'sl_multiplier': sl_multiplier,
            'tp_multiplier': tp_multiplier,
            'method': 'atr_based'
          }

      raise ValueError("Недостаточно данных для ATR или некорректное значение ATR")

    except Exception as e:
      logger.error(f"Ошибка ATR расчета для {signal.symbol}: {e}")
      raise

  async def calculate_sl_tp_levels(self, signal: TradingSignal, method: str = "auto") -> Dict[str, float]:
    """Расчет уровней SL/TP с улучшенной обработкой ошибок"""
    try:
      logger.info(f"Расчет SL/TP для {signal.symbol} методом '{method}'")

      # Проверяем входные данные
      if not signal or not signal.symbol:
        raise ValueError("Некорректный сигнал")

      if method == "auto":
        method = self._select_optimal_method(signal)
        logger.info(f"Автовыбор: используем {method} метод")

      # Получаем прогноз волатильности с проверкой
      volatility_prediction = await self._get_volatility_prediction(signal.symbol)

      # Рассчитываем уровни
      if method == "roi":
        levels = await self._calculate_roi_levels(signal)
      elif method == "atr":
        levels = await self._calculate_atr_levels(signal)
      else:  # hybrid
        levels = await self._calculate_hybrid_levels(signal, volatility_prediction)

      # Валидируем результаты
      if not self._validate_levels(levels, signal):
        logger.warning(f"Некорректные уровни SL/TP для {signal.symbol}, используем запасные")
        levels = self._get_fallback_levels(signal)

      return levels

    except Exception as e:
      logger.error(f"Ошибка расчета SL/TP для {signal.symbol}: {e}")
      return self._get_fallback_levels(signal)

  def _get_fallback_levels(self, signal: TradingSignal) -> Dict[str, float]:
    """Запасные уровни SL/TP"""
    price = signal.price
    if signal.signal_type == SignalType.BUY:
      return {
        'stop_loss': price * 0.98,  # 2% ниже
        'take_profit': price * 1.04  # 4% выше
      }
    else:
      return {
        'stop_loss': price * 1.02,  # 2% выше
        'take_profit': price * 0.96  # 4% ниже
      }