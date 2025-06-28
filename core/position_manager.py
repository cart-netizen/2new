import asyncio
from datetime import datetime

import pandas as pd
import pandas_ta as ta
from typing import Dict, Optional

from config.config_manager import ConfigManager
from core.bybit_connector import BybitConnector
from core.indicators import crossover_series, crossunder_series
# from core.integrated_system import IntegratedTradingSystem
from core.risk_manager import AdvancedRiskManager
from core.schemas import TradingSignal, RiskMetrics
from data.database_manager import AdvancedDatabaseManager
from core.trade_executor import TradeExecutor
from core.data_fetcher import DataFetcher
from core.enums import Timeframe, SignalType
from utils.logging_config import get_logger
from core.signal_filter import SignalFilter

logger = get_logger(__name__)


class PositionManager:
  """
  Класс для управления открытыми позициями и реализации логики выхода.
  """

  def __init__(self, db_manager: AdvancedDatabaseManager, trade_executor: TradeExecutor, data_fetcher: DataFetcher, connector: BybitConnector, signal_filter: SignalFilter, risk_manager: AdvancedRiskManager):
    self.db_manager = db_manager
    self.trade_executor = trade_executor
    self.data_fetcher = data_fetcher
    self.connector = connector
    self.signal_filter = signal_filter
    self.open_positions: Dict[str, Dict] = {}  # Кэш открытых позиций для быстрой проверки
    self.risk_manager = risk_manager
    self.config_manager = ConfigManager()
    self.config = self.config_manager.load_config()
    self.trading_system = None
    # self.integrated_system: Optional[IntegratedTradingSystem] = None

  async def load_open_positions(self):
    """
    Синхронизирует состояние открытых позиций.
    Получает ВСЕ позиции с биржи, а не только по топовым символам.
    """
    logger.info("Синхронизация открытых позиций с биржей...")
    self.open_positions = {}  # Очищаем старый кэш

    try:
      # НОВЫЙ ПОДХОД: Получаем ВСЕ позиции сразу одним запросом
      endpoint = "/v5/position/list"
      params = {
        'category': 'linear',
        'settleCoin': 'USDT'  # Только USDT позиции
      }

      # Запрашиваем напрямую все позиции
      result = await self.connector._make_request('GET', endpoint, params, use_cache=False)

      if result and result.get('list'):
        all_positions = result.get('list', [])
        logger.info(f"Получено {len(all_positions)} позиций с биржи")

        # Обрабатываем только позиции с ненулевым размером
        for position in all_positions:
          size = float(position.get('size', 0))
          if size > 0:
            symbol = position.get('symbol')
            logger.info(f"На бирже найдена активная позиция по {symbol}. Размер: {size}")

            # Ищем соответствующую запись в локальной БД
            local_trade_data = await self.db_manager.get_open_trade_by_symbol(symbol)

            if local_trade_data:
              # Если нашли, используем данные из БД
              logger.info(f"Найдена соответствующая запись в локальной БД для {symbol}")
              self.open_positions[symbol] = local_trade_data
            else:
              # Если в БД нет, создаем заглушку из данных биржи
              logger.warning(f"Позиция по {symbol} существует на бирже, но отсутствует в локальной БД")
              self.open_positions[symbol] = {
                'symbol': symbol,
                'side': position.get('side', 'Buy').upper(),
                'open_price': float(position.get('avgPrice', 0)),
                'quantity': size,
                'stop_loss': float(position.get('stopLoss', 0)) if position.get('stopLoss') else None,
                'take_profit': float(position.get('takeProfit', 0)) if position.get('takeProfit') else None,
                'unrealizedPnl': float(position.get('unrealisedPnl', 0)),
                'leverage': int(position.get('leverage', 1)),
                'id': -1  # Указываем, что это "неизвестная" сделка
              }
      else:
        logger.info("Не получено позиций с биржи")

    except Exception as e:
      logger.error(f"Критическая ошибка при синхронизации позиций с биржей: {e}", exc_info=True)

    if self.open_positions:
      logger.info(f"Синхронизация завершена. Активные позиции: {list(self.open_positions.keys())}")
    else:
      logger.info("Синхронизация завершена. Активных позиций на бирже не найдено.")

  def _check_sl_tp(self, position: Dict, current_price: float) -> Optional[str]:
    """Проверяет, сработал ли Stop-Loss или Take-Profit."""
    side = position.get('side')
    stop_loss = position.get('stop_loss')
    take_profit = position.get('take_profit')

    if side == 'BUY':
      if stop_loss and current_price <= stop_loss:
        return f"Stop-Loss для BUY сработал: цена {current_price:.4f} <= SL {stop_loss:.4f}"
      if take_profit and current_price >= take_profit:
        return f"Take-Profit для BUY сработал: цена {current_price:.4f} >= TP {take_profit:.4f}"
    elif side == 'SELL':
      if stop_loss and current_price >= stop_loss:
        return f"Stop-Loss для SELL сработал: цена {current_price:.4f} >= SL {stop_loss:.4f}"
      if take_profit and current_price <= take_profit:
        return f"Take-Profit для SELL сработал: цена {current_price:.4f} <= TP {take_profit:.4f}"
    return None

  def _check_dynamic_exit(self, position: Dict, data: pd.DataFrame) -> Optional[str]:
    """Проверяет условие динамического выхода (пересечение EMA)."""
    try:
      ema_fast = ta.ema(data['close'], length=12)
      ema_slow = ta.ema(data['close'], length=26)
      if ema_fast is None or ema_slow is None or len(ema_fast) < 2: return None

      side = position.get('side')
      if side == 'BUY' and ema_fast.iloc[-2] > ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return "Обнаружен медвежий кроссовер EMA(12/26)"
      if side == 'SELL' and ema_fast.iloc[-2] < ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]:
        return "Обнаружен бычий кроссовер EMA(12/26)"
      return None
    except Exception as e:
      logger.error(f"Ошибка при проверке динамического выхода для {position['symbol']}: {e}")
      return None

  async def manage_open_positions(self, account_balance: Optional[RiskMetrics]):
    """
    ФИНАЛЬНАЯ ВЕРСИЯ: Управляет открытыми позициями, используя
    иерархию проверок: SL/TP -> PSAR -> Stop-and-Reverse.
    """
    if not hasattr(self, '_last_order_check'):
      self._last_order_check = datetime.now()

    if (datetime.now() - self._last_order_check).seconds > 30:  # Каждую минуту
      await self.track_pending_orders()
      self._last_order_check = datetime.now()

    if not self.open_positions:
      return

    logger.debug(f"Динамическое управление для {len(self.open_positions)} открытых позиций...")

    for symbol, position_data in list(self.open_positions.items()):
      try:
        # --- ИСПРАВЛЕНИЕ: ЗАГРУЖАЕМ ДАННЫЕ ЗАРАНЕЕ ---
        # 1. Загружаем данные основного таймфрейма (1H) для анализа SL/TP и PSAR
        htf_data = await self.data_fetcher.get_historical_candles(symbol, Timeframe.ONE_HOUR, limit=100)
        if htf_data.empty:
          continue
        current_price = htf_data['close'].iloc[-1]

        # 2. Загружаем данные малого таймфрейма (1m) для анализа разворота
        strategy_settings = self.config.get('strategy_settings', {})
        ltf_str = strategy_settings.get('ltf_entry_timeframe', '15m')
        timeframe_map = {"1m": Timeframe.ONE_MINUTE, "5m": Timeframe.FIVE_MINUTES, "15m": Timeframe.FIFTEEN_MINUTES}
        ltf_timeframe = timeframe_map.get(ltf_str, Timeframe.ONE_MINUTE)
        ltf_data = await self.data_fetcher.get_historical_candles(symbol, ltf_timeframe, limit=100)
        # --- КОНЕЦ БЛОКА ЗАГРУЗКИ ---

        # --- ПРИОРИТЕТ 1: ЖЕСТКИЙ SL/TP ---
        exit_reason = self._check_sl_tp(position_data, current_price)
        if exit_reason:
          logger.info(f"ВЫХОД для {symbol}: Сработал стандартный SL/TP. Причина: {exit_reason}")
          await self.trade_executor.close_position(symbol=symbol)
          continue

        # --- ПРИОРИТЕТ 2: ТРЕЙЛИНГ-СТОП ПО ATR (на HTF)---
        if not exit_reason:
          # Убедимся, что ATR рассчитан для htf_data
          if 'atr' not in htf_data.columns:
            htf_data.ta.atr(append=True)
          exit_reason = self._check_atr_trailing_stop(position_data, htf_data)

        # --- ПРИОРИТЕТ 3: ТРЕЙЛИНГ-СТОП ПО PSAR ---
        if not exit_reason and strategy_settings.get('use_psar_exit', True):
          # Важно: для PSAR используем данные ОСНОВНОГО таймфрейма (1H)
          # Рассчитываем PSAR для htf_data
          psar_df = ta.psar(htf_data['high'], htf_data['low'], htf_data['close'])
          if psar_df is not None:
            psar_col = next(
              (col for col in psar_df.columns if 'PSAR' in col and 'PSARl' not in col and 'PSARs' not in col), None)
            if psar_col:
              htf_data['psar'] = psar_df[psar_col]
              exit_reason = self._check_psar_exit(position_data, htf_data)
              if exit_reason:
                logger.info(f"ВЫХОД для {symbol}: Сработал трейлинг-стоп. Причина: {exit_reason}")
                await self.trade_executor.close_position(symbol=symbol)
                continue

        # --- ПРИОРИТЕТ 4: STOP AND REVERSE ---
        if not exit_reason:
          reverse_signal = await self._check_reversal_exit(position_data, ltf_data)
          if reverse_signal:
            logger.info(f"🔄 Обнаружен сигнал разворота для {symbol}")

            # Проверяем, можем ли использовать функцию reverse
            use_reverse = self.config.get('strategy_settings', {}).get('use_reverse_function', True)

            if use_reverse and hasattr(self.trade_executor, 'reverse_position'):
              # Используем новую функцию reverse
              reverse_success = await self.trade_executor.reverse_position(
                symbol=symbol,
                current_position=position_data,
                new_signal=reverse_signal,
                force=False  # Требуем проверку прибыльности
              )

              if reverse_success:
                logger.info(f"✅ Позиция {symbol} успешно развернута")
                # Обновляем статистику
                if hasattr(self, 'trading_system') and self.trading_system:
                  await self.trading_system.update_strategy_performance(
                    symbol, 'reverse', True
                  )
              else:
                logger.warning(f"Не удалось развернуть позицию {symbol}, выполняем обычное закрытие")
                # Fallback к обычному закрытию + новый вход
                await self._execute_standard_exit_and_reentry(symbol, position_data, reverse_signal,
                                                              account_balance)
            else:
              # Стандартный путь: закрытие + новый вход
              await self._execute_standard_exit_and_reentry(symbol, position_data, reverse_signal,
                                                            account_balance)

        if exit_reason:
          logger.info(f"ВЫХОД для {symbol}: Обнаружен сигнал на закрытие. Причина: {exit_reason}")
          await self.trade_executor.close_position(symbol=symbol)
          continue

      except Exception as e:
        logger.error(f"Ошибка при динамическом управлении позицией {symbol}: {e}", exc_info=True)

  async def _execute_standard_exit_and_reentry(self, symbol: str, position_data: Dict,
                                               reverse_signal: TradingSignal, account_balance: Optional[RiskMetrics]):
    """
    Стандартная процедура: закрытие текущей позиции и открытие новой.
    """
    try:
      # Закрываем текущую позицию
      close_success = await self.trade_executor.close_position(symbol=symbol)
      if not close_success:
        logger.error(f"Не удалось закрыть позицию {symbol} для разворота")
        return

      # Ждем подтверждения закрытия
      await asyncio.sleep(3)

      # Открываем новую позицию в противоположном направлении
      if account_balance:
        risk_decision = await self.risk_manager.validate_signal(
          reverse_signal, symbol, account_balance.available_balance_usdt
        )

        if risk_decision.get('approved'):
          logger.info(f"Риск-менеджер одобрил новую позицию после разворота {symbol}")
          quantity = risk_decision.get('recommended_size')
          await self.trade_executor.execute_trade(reverse_signal, symbol, quantity)
        else:
          logger.warning(f"Риск-менеджер отклонил новую позицию после разворота {symbol}")

    except Exception as e:
      logger.error(f"Ошибка при стандартном развороте {symbol}: {e}")

  async def reconcile_filled_orders(self):
        """
        ФИНАЛЬНАЯ ВЕРСИЯ: Сверяет исполненные ордера, корректно рассчитывая PnL
        с учетом комиссий и обрабатывая закрытия по TP/SL.
        """
        # all_positions = await self.connector.fetch_positions_batch()
        # active_symbols = {pos['symbol'] for pos in all_positions if float(pos.get('size', 0)) > 0}

        # Получаем из БД все сделки, которые у нас числятся как "OPEN"
        open_trades_in_db = await self.db_manager.get_all_open_trades()
        if not open_trades_in_db:
          return

        logger.debug(f"Сверка статуса для {len(open_trades_in_db)} открытых в БД сделок...")

        for trade in open_trades_in_db:
          symbol = trade.get('symbol')
          if not symbol:
            continue

          try:
            # 1. Проверяем, есть ли еще реальная позиция на бирже
            positions_on_exchange = await self.connector.fetch_positions(symbol)
            is_still_open = any(float(pos.get('size', 0)) > 0 for pos in positions_on_exchange)

            # 2. Если позиции на бирже уже нет, значит, она была закрыта
            if not is_still_open:
              logger.info(
                f"Позиция по {symbol} больше не активна на бирже. Поиск исполненной сделки для расчета PnL...")

              # Ищем в истории исполнений сделку, которая закрыла нашу позицию
              executions = await self.connector.get_execution_history(symbol=symbol, limit=20)
              closing_exec = None
              for exec_trade in executions:
                # Ищем сделку, которая закрыла позицию (closedSize > 0)
                # и соответствует нашему ID ордера на открытие, если он есть
                if exec_trade.get('closedSize') and float(exec_trade.get('closedSize', 0)) > 0:
                  closing_exec = exec_trade
                  break

              if closing_exec:
                # --- КОРРЕКТНЫЙ РАСЧЕТ PNL ---
                open_price = float(trade['open_price'])
                close_price = float(closing_exec['execPrice'])
                quantity = float(trade['quantity'])
                # Важно: берем комиссию из данных об исполнении!
                commission = float(closing_exec.get('execFee', 0))
                side = trade.get('side')

                # Считаем "грязный" PnL
                gross_pnl = (close_price - open_price) * quantity if side == 'BUY' else (
                                                                                              open_price - close_price) * quantity

                # Вычитаем комиссию за ЗАКРЫТИЕ
                net_pnl = gross_pnl - commission
                # Примечание: комиссию за открытие мы здесь не учитываем, т.к. ее нет в данных о закрытии.
                # Для 100% точности ее нужно было бы хранить в БД. Но это уже 99% точности.

                close_timestamp = datetime.fromtimestamp(int(closing_exec['execTime']) / 1000)

                logger.info(
                  f"ПОДТВЕРЖДЕНИЕ ЗАКРЫТИЯ для {symbol}: Найдена исполненная сделка. Чистый PnL: {net_pnl:.4f}")

                await self.db_manager.update_trade_as_closed(
                  trade_id=trade['id'], close_price=close_price, pnl=net_pnl,
                  commission=commission, close_timestamp=close_timestamp
                )

                integrated_system = getattr(self.trade_executor, 'integrated_system', None)
                if integrated_system and hasattr(integrated_system, 'process_trade_feedback'):
                  try:
                    trade_result = {
                      'strategy_name': trade.get('strategy_name', 'Unknown'),
                      'profit_loss': net_pnl,
                      'entry_price': open_price,
                      'exit_price': close_price,
                      'regime': trade.get('metadata', {}).get('regime', 'unknown') if isinstance(trade.get('metadata'),
                                                                                                 dict) else 'unknown',
                      'confidence': trade.get('confidence', 0.5),
                      'entry_features': trade.get('metadata', {}).get('features', {}) if isinstance(
                        trade.get('metadata'), dict) else {}
                    }

                    await integrated_system.process_trade_feedback(symbol, trade['id'], trade_result)
                    logger.info(f"Обратная связь отправлена для {symbol}")
                  except Exception as e:
                    logger.error(f"Ошибка отправки обратной связи: {e}")

                # Уведомляем SAR стратегию об обновлении позиции для адаптивного обучения
                if (hasattr(self, 'integrated_system') and self.integrated_system and
                    hasattr(self.integrated_system, 'sar_strategy') and
                    self.integrated_system.sar_strategy):

                  try:
                    # Проверяем, что позиция действительно принадлежит SAR стратегии
                    if symbol in getattr(self.integrated_system.sar_strategy, 'current_positions', {}):
                      await self.integrated_system.sar_strategy.handle_position_update(symbol, {
                        'profit_loss': net_pnl,
                        'close_price': close_price,
                        'close_timestamp': close_timestamp,
                        'close_reason': locals().get('close_reason', 'position_manager'),
                        'open_price': locals().get('open_price', trade.get('open_price', 0)),
                        'net_pnl': net_pnl,
                        # 'close_reason': locals().get('close_reason', 'position_manager_close'),
                        # 'open_price': position_data.get('open_price', 0),
                      })
                      logger.debug(f"SAR стратегия уведомлена об обновлении позиции {symbol}")

                    # Также обновляем общую производительность стратегии
                    if hasattr(self.integrated_system, 'adaptive_selector'):
                      await self.integrated_system.adaptive_selector.update_strategy_performance(
                        'SAR_Strategy',
                        {
                          'profit_loss': net_pnl,
                          'symbol': symbol,
                          'close_timestamp': close_timestamp,
                          'regime': getattr(self.integrated_system.sar_strategy, 'current_market_regime', 'unknown')
                        }
                      )
                  except Exception as e:
                    logger.error(f"Ошибка при уведомлении SAR стратегии об обновлении позиции {symbol}: {e}")

                # Удаляем из кэша, если она там была
                if symbol in self.open_positions:
                  del self.open_positions[symbol]

                # Синхронизируем с Shadow Trading
                if hasattr(self, 'trading_system') and self.trading_system and self.trading_system.shadow_trading:
                  # Получаем shadow_tracking_id из метаданных сделки
                  metadata = trade.get('metadata')
                  if metadata:
                    try:
                      import json
                      metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                      shadow_tracking_id = metadata_dict.get('shadow_tracking_id')

                      if shadow_tracking_id:
                        # Определяем исход на основе PnL
                        from shadow_trading.signal_tracker import SignalOutcome
                        outcome = SignalOutcome.PROFITABLE if net_pnl > 0 else SignalOutcome.LOSS

                        # Финализируем сигнал в Shadow Trading
                        await self.trading_system.shadow_trading.signal_tracker.finalize_signal(
                          shadow_tracking_id,
                          close_price,
                          datetime.now(),
                          outcome
                        )
                        logger.info(f"✅ Shadow Trading сигнал {shadow_tracking_id} синхронизирован")
                    except Exception as e:
                      logger.warning(f"Не удалось синхронизировать с Shadow Trading: {e}")
              else:
                # Если не нашли исполненной сделки, используем старый метод "принудительного закрытия"
                logger.warning(f"Не удалось найти исполненную сделку для {symbol}. Принудительное закрытие с PnL=0.")
                await self.db_manager.force_close_trade(trade_id=trade['id'], close_price=trade['open_price'])

          except Exception as e:
            logger.error(f"Ошибка при сверке сделок для {symbol}: {e}", exc_info=True)




  # def add_position_to_cache(self, trade: Dict):
  #   """Добавляет информацию о новой сделке в кэш открытых позиций."""
  #   if 'symbol' in trade:
  #     symbol = trade['symbol']
  #     self.open_positions[symbol] = trade
  #     logger.info(f"Новая позиция по {symbol} добавлена в кэш PositionManager.")
  #   else:
  #     logger.error("Попытка добавить в кэш сделку без ключа 'symbol'.")
  async def _check_reversal_exit(self, position: Dict, data: pd.DataFrame) -> Optional[TradingSignal]:
    """
    Улучшенная версия: проверяет необходимость разворота позиции,
    используя стратегии и подтверждение от нескольких индикаторов.
    """
    try:
      symbol = position['symbol']
      current_side = position.get('side')

      # 1. Получаем сигналы от стратегий через integrated_system
      if hasattr(self, 'trading_system') and self.trading_system:
        # Проверяем SAR стратегию
        if hasattr(self.trading_system, 'sar_strategy') and self.trading_system.sar_strategy:
          sar_signal = await self.trading_system.sar_strategy.check_exit_conditions(
            symbol, data, position
          )
          if sar_signal and sar_signal.is_reversal:
            logger.info(f"SAR стратегия обнаружила сигнал разворота для {symbol}")

            # Подтверждаем другими индикаторами
            confirmations = await self._confirm_reversal_signal(
              symbol, data, current_side, sar_signal
            )

            if confirmations >= 2:  # Минимум 2 подтверждения
              logger.info(f"Разворот подтвержден {confirmations} индикаторами")
              return sar_signal

      # 2. Альтернативная проверка через базовые индикаторы
      # Если стратегии недоступны или не дали сигнала
      reversal_conditions = 0

      # EMA crossover
      if 'ema_12' in data.columns and 'ema_26' in data.columns:
        ema_fast = data['ema_12'].iloc[-1]
        ema_slow = data['ema_26'].iloc[-1]
        prev_ema_fast = data['ema_12'].iloc[-2]
        prev_ema_slow = data['ema_26'].iloc[-2]

        if current_side == 'BUY':
          # Медвежий кроссовер для выхода из лонга
          if prev_ema_fast > prev_ema_slow and ema_fast < ema_slow:
            reversal_conditions += 1
        else:
          # Бычий кроссовер для выхода из шорта
          if prev_ema_fast < prev_ema_slow and ema_fast > ema_slow:
            reversal_conditions += 1

      # RSI экстремумы
      if 'rsi' in data.columns:
        rsi = data['rsi'].iloc[-1]
        if current_side == 'BUY' and rsi > 75:  # Перекупленность
          reversal_conditions += 1
        elif current_side == 'SELL' and rsi < 25:  # Перепроданность
          reversal_conditions += 1

      # MACD дивергенция
      if all(col in data.columns for col in ['macd', 'macd_signal']):
        macd = data['macd'].iloc[-1]
        signal = data['macd_signal'].iloc[-1]
        prev_macd = data['macd'].iloc[-2]
        prev_signal = data['macd_signal'].iloc[-2]

        if current_side == 'BUY':
          if prev_macd > prev_signal and macd < signal:
            reversal_conditions += 1
        else:
          if prev_macd < prev_signal and macd > signal:
            reversal_conditions += 1

      # Если достаточно условий для разворота
      if reversal_conditions >= 2:
        # Создаем сигнал разворота
        new_signal_type = SignalType.SELL if current_side == 'BUY' else SignalType.BUY

        reversal_signal = TradingSignal(
          symbol=symbol,
          signal_type=new_signal_type,
          price=data['close'].iloc[-1],
          confidence=0.6 + (reversal_conditions * 0.1),  # Базовая + бонус за подтверждения
          strategy="reversal_exit",
          timeframe=position.get('timeframe', '1h'),
          volume=data['volume'].iloc[-1] if 'volume' in data.columns else 0,
          timestamp=datetime.now(),
          metadata={
            'reversal_conditions': reversal_conditions,
            'original_side': current_side,
            'position_pnl_pct': self._calculate_current_pnl(position, data['close'].iloc[-1])
          }
        )

        return reversal_signal

      return None

    except Exception as e:
      logger.error(f"Ошибка при проверке разворота для {position['symbol']}: {e}")
      return None

  async def _confirm_reversal_signal(self, symbol: str, data: pd.DataFrame,
                               current_side: str, signal: TradingSignal) -> int:
    """
    Подтверждает сигнал разворота дополнительными индикаторами.
    Возвращает количество подтверждений.
    """
    confirmations = 0

    try:
      # 1. Проверка объемов
      if 'volume' in data.columns and len(data) > 20:
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        if current_volume > avg_volume * 1.5:  # Повышенный объем
          confirmations += 1

      # 2. Проверка волатильности (ATR)
      if 'atr' in data.columns:
        atr = data['atr'].iloc[-1]
        price = data['close'].iloc[-1]
        atr_pct = (atr / price) * 100
        if atr_pct > 1.5:  # Высокая волатильность
          confirmations += 1

      # 3. Проверка моментума
      if len(data) > 10:
        momentum = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10] * 100
        if (current_side == 'BUY' and momentum < -2) or \
            (current_side == 'SELL' and momentum > 2):
          confirmations += 1

      # 4. Проверка структуры рынка (Higher High/Lower Low)
      if self._check_market_structure_break(data, current_side):
        confirmations += 1

      logger.debug(f"Разворот {symbol}: получено {confirmations} подтверждений")
      return confirmations

    except Exception as e:
      logger.error(f"Ошибка при подтверждении разворота: {e}")
      return 0

  def _check_market_structure_break(self, data: pd.DataFrame, current_side: str) -> bool:
    """
    Проверяет нарушение структуры рынка (пробой последнего high/low).
    """
    try:
      if len(data) < 20:
        return False

      # Находим последние локальные экстремумы
      highs = data['high'].rolling(5).max()
      lows = data['low'].rolling(5).min()

      current_price = data['close'].iloc[-1]

      if current_side == 'BUY':
        # Для лонга: проверяем пробой последнего значимого low
        recent_low = lows.iloc[-10:-1].min()
        return current_price < recent_low
      else:
        # Для шорта: проверяем пробой последнего значимого high
        recent_high = highs.iloc[-10:-1].max()
        return current_price > recent_high

    except Exception:
      return False

  def _calculate_current_pnl(self, position: Dict, current_price: float) -> float:
    """
    Рассчитывает текущий PnL позиции в процентах.
    """
    try:
      open_price = float(position.get('open_price', 0))
      if open_price == 0:
        return 0.0

      side = position.get('side')
      if side == 'BUY':
        return ((current_price - open_price) / open_price) * 100
      else:
        return ((open_price - current_price) / open_price) * 100
    except Exception:
      return 0.0

  def add_position_to_cache(self, trade: Dict):
    """Добавляет информацию о новой сделке в кэш открытых позиций."""
    if trade and 'symbol' in trade:
      symbol = trade['symbol']
      self.open_positions[symbol] = trade
      logger.info(f"Новая позиция по {symbol} добавлена в кэш PositionManager.")
    else:
      logger.error(f"Попытка добавить в кэш невалидную сделку: {trade}")

  def _check_psar_exit(self, position: Dict, data: pd.DataFrame) -> Optional[str]:
    """
    Проверяет, нужно ли выходить из сделки по сигналу Parabolic SAR,
    с обязательной проверкой на безубыточность.
    """
    if 'psar' not in data.columns or data['psar'].isnull().all():
      return None

    side = position.get('side')
    current_price = data['close'].iloc[-1]
    psar_value = data['psar'].iloc[-1]
    open_price = float(position.get('open_price', 0))

    if open_price == 0:
      return None  # Не можем проверить, если цена входа неизвестна

    # Определяем, есть ли сигнал на выход по PSAR
    is_psar_exit_signal = False
    if side == 'BUY' and current_price < psar_value:
      is_psar_exit_signal = True
    elif side == 'SELL' and current_price > psar_value:
      is_psar_exit_signal = True

    # Если сигнала на выход нет, прекращаем проверку
    if not is_psar_exit_signal:
      return None

    # --- НОВЫЙ БЛОК: ПРОВЕРКА НА БЕЗУБЫТОЧНОСТЬ ---
    # Если сигнал на выход есть, сначала проверяем, выгодно ли это

    # Минимальная прибыль для закрытия должна покрывать хотя бы комиссию за закрытие
    commission_rate = 0.00075  # Средняя комиссия тейкера ~0.075%

    # Для LONG позиции: выходим, только если цена закрытия выше цены входа + комиссия
    if side == 'BUY' and current_price > open_price * (1 + commission_rate):
      logger.info(f"Выход по PSAR для BUY ({position['symbol']}) подтвержден как безубыточный.")
      return f"Parabolic SAR для BUY сработал: цена {current_price:.4f} < PSAR {psar_value:.4f}"

    # Для SHORT позиции: выходим, только если цена закрытия ниже цены входа - комиссия
    elif side == 'SELL' and current_price < open_price * (1 - commission_rate):
      logger.info(f"Выход по PSAR для SELL ({position['symbol']}) подтвержден как безубыточный.")
      return f"Parabolic SAR для SELL сработал: цена {current_price:.4f} > PSAR {psar_value:.4f}"

    else:
      # Если сигнал PSAR есть, но закрытие приведет к убытку, мы его игнорируем
      logger.debug(
        f"Сигнал на выход по PSAR для {position['symbol']} проигнорирован, т.к. сделка не является безубыточной.")
      return None
    # --- КОНЕЦ НОВОГО БЛОКА ---

  def _check_atr_trailing_stop(self, position: Dict, data: pd.DataFrame) -> Optional[str]:
    """
    Улучшенная версия трейлинг-стопа на основе ATR с поддержкой Chandelier Exit.
    """
    strategy_settings = self.config.get('strategy_settings', {})
    if not strategy_settings.get('use_atr_trailing_stop', True):
      return None

    if 'atr' not in data.columns or data['atr'].isnull().all():
      logger.warning(f"ATR не найден в данных для {position['symbol']}")
      return None

    side = position.get('side')
    current_price = data['close'].iloc[-1]
    atr_value = data['atr'].iloc[-1]
    atr_multiplier = strategy_settings.get('atr_ts_multiplier', 2.5)
    open_price = float(position.get('open_price', 0))

    if open_price == 0:
      return None

    # Расчет комиссий и минимальной прибыли
    commission_rate = 0.00075  # Taker fee
    min_profit_buffer = (commission_rate * 3) * 1.7  # 3 комиссии с запасом 70%

    # Получаем максимальную/минимальную цену с момента входа
    # Это ключевое отличие Chandelier Exit от обычного ATR stop
    entry_index = position.get('entry_bar_index', -20)  # По умолчанию последние 20 баров

    if entry_index < 0:
      # Если не знаем точный индекс входа, используем последние N баров
      lookback_bars = min(abs(entry_index), len(data))
      recent_data = data.tail(lookback_bars)
    else:
      recent_data = data.iloc[entry_index:]

    # Chandelier Exit логика
    if side == 'BUY':
      # Для лонга: стоп следует за максимумом минус ATR
      highest_high = recent_data['high'].max()
      chandelier_stop = highest_high - (atr_value * atr_multiplier)

      # Дополнительная защита: стоп не может быть ниже цены входа + минимальная прибыль
      minimum_stop = open_price * (1 + min_profit_buffer)
      effective_stop = max(chandelier_stop, minimum_stop)

      # Проверяем, пробита ли цена
      if current_price < effective_stop:
        profit_pct = ((current_price - open_price) / open_price) * 100
        return (f"Chandelier Exit сработал для BUY: цена {current_price:.4f} < "
                f"Stop {effective_stop:.4f} (прибыль: {profit_pct:.2f}%)")

    elif side == 'SELL':
      # Для шорта: стоп следует за минимумом плюс ATR
      lowest_low = recent_data['low'].min()
      chandelier_stop = lowest_low + (atr_value * atr_multiplier)

      # Защита для шорта
      minimum_stop = open_price * (1 - min_profit_buffer)
      effective_stop = min(chandelier_stop, minimum_stop)

      # Проверяем пробитие
      if current_price > effective_stop:
        profit_pct = ((open_price - current_price) / open_price) * 100
        return (f"Chandelier Exit сработал для SELL: цена {current_price:.4f} > "
                f"Stop {effective_stop:.4f} (прибыль: {profit_pct:.2f}%)")

    return None

  def update_position_extremes(self, symbol: str, current_price: float, high: float, low: float):
    """
    Обновляет экстремумы для позиции (используется для Chandelier Exit).
    """
    if symbol not in self.open_positions:
      return

    position = self.open_positions[symbol]

    # Инициализируем если не существует
    if 'highest_since_entry' not in position:
      position['highest_since_entry'] = high
      position['lowest_since_entry'] = low
    else:
      # Обновляем экстремумы
      position['highest_since_entry'] = max(position['highest_since_entry'], high)
      position['lowest_since_entry'] = min(position['lowest_since_entry'], low)

  async def monitor_single_position(self, symbol: str) -> bool:
    """Мониторит одну конкретную позицию"""
    if symbol not in self.open_positions:
      return False

    try:
      # Проверяем условия выхода для конкретной позиции
      await self.manage_open_positions(symbol)
      return True
    except Exception as e:
      logger.error(f"Ошибка при мониторинге позиции {symbol}: {e}")
      return False

  async def track_pending_orders(self):
    """Отслеживает статус pending ордеров"""
    # Получаем все открытые сделки
    open_trades = await self.db_manager.get_all_open_trades()

    for trade in open_trades:
      if trade.get('order_id'):
        # Используем метод из trade_executor для проверки статуса
        await self.trade_executor.update_trade_status_from_exchange(
          trade['order_id'],
          trade['symbol']
        )

    logger.debug("Проверка статусов ордеров завершена")