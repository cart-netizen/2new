import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from core.bybit_connector import BybitConnector
from utils.logging_config import get_logger
from config import trading_params
from core.enums import Timeframe

logger = get_logger(__name__)


class DataFetcher:
  def __init__(self, connector: BybitConnector):
    self.connector = connector

  async def get_active_symbols_by_volume(self) -> List[str]:
    """
    Получает список активных бессрочных USDT контрактов,
    отсортированных по объему торгов за 24 часа и отфильтрованных.
    """
    try:
      # ИСПРАВЛЕНО: Добавлен await для асинхронного метода
      contracts = await self.connector.get_usdt_perpetual_contracts()
      if not contracts:
        logger.warning("Не удалось получить список контрактов от Bybit.")
        return []

      valid_symbols = []
      for contract in contracts:
        try:
          # Для CCXT данные приходят в другом формате
          # Проверяем различные возможные поля для объема
          volume_24h_usdt = 0
          symbol = contract.get('symbol')

          # Пробуем разные поля для объема в USDT
          if 'quoteVolume' in contract:
            volume_24h_usdt = float(contract.get('quoteVolume', 0))
          elif 'turnover24h' in contract:
            volume_24h_usdt = float(contract.get('turnover24h', 0))
          elif 'info' in contract and 'turnover24h' in contract['info']:
            volume_24h_usdt = float(contract['info'].get('turnover24h', 0))
          elif 'info' in contract and 'quoteVolume' in contract['info']:
            volume_24h_usdt = float(contract['info'].get('quoteVolume', 0))

          if symbol and "USDT" in symbol and volume_24h_usdt >= trading_params.MIN_24H_VOLUME_USDT:
            valid_symbols.append({
              'symbol': symbol,
              'volume24h_usdt': volume_24h_usdt
            })
        except (ValueError, TypeError) as e:
          logger.warning(f"Некорректные данные объема для контракта: {contract.get('symbol')} - {e}")
          continue
        except Exception as e:
          logger.error(f"Ошибка обработки контракта {contract.get('symbol')}: {e}")
          continue

      # Сортируем по объему (от большего к меньшему)
      valid_symbols.sort(key=lambda x: x['volume24h_usdt'], reverse=True)

      logger.info(
        f"Найдено {len(valid_symbols)} валидных символов с объемом > {trading_params.MIN_24H_VOLUME_USDT} USDT.")

      # Ограничиваем количество символов
      selected_symbols = [s['symbol'] for s in valid_symbols[:trading_params.SYMBOLS_LIMIT]]
      logger.info(f"Выбрано топ {len(selected_symbols)} символов для работы: {selected_symbols}")

      return selected_symbols

    except Exception as e:
      logger.error(f"Ошибка при получении и фильтрации активных символов: {e}", exc_info=True)
      return []

  async def get_historical_candles(self, symbol: str, timeframe: Timeframe, limit: int = 200) -> pd.DataFrame:
    """
    Получает исторические свечи для символа.
    """
    logger.debug(f"Запрос исторических свечей для {symbol}, таймфрейм {timeframe.value}, лимит {limit}")
    try:
      # Преобразуем Timeframe enum в строку, которую понимает Bybit
      interval_str = timeframe.value

      raw_candles = await self.connector.get_kline(symbol, interval_str, limit)
      if not raw_candles:
        logger.warning(f"Нет данных свечей для {symbol} с таймфреймом {interval_str}")
        return pd.DataFrame()

      # CCXT возвращает данные в формате [timestamp, open, high, low, close, volume]
      df = pd.DataFrame(raw_candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume'
      ])

      # Преобразование типов
      df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
      for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])

      df.set_index('timestamp', inplace=True)
      df.sort_index(inplace=True)  # Сортируем по времени

      logger.info(f"Получено {len(df)} свечей для {symbol} ({interval_str})")
      return df
    except Exception as e:
      logger.error(f"Ошибка при получении исторических свечей для {symbol}: {e}", exc_info=True)
    return pd.DataFrame()

  # async def get_current_price(self, symbol: str) -> Optional[float]:
    # """Получает текущую цену для символа (цена последней сделки)"""
    # try:
    #   tickers = self.connector.get_usdt_perpetual_contracts()  # Можно оптимизировать, чтобы не запрашивать все тикеры
    #   if tickers:
    #     for ticker in tickers:
    #       if ticker.get('symbol') == symbol:
    #         return float(ticker.get('lastPrice', 0.0))
    #   logger.warning(f"Не удалось получить текущую цену для {symbol}")
    #   return None
    # except Exception as e:
    #   logger.error(f"Ошибка при получении текущей цены для {symbol}: {e}")
    #   return None

  async def get_current_price(self, symbol: str) -> Optional[float]:
      """Получает текущую цену для символа (цена последней сделки)"""
      try:
        # ИСПРАВЛЕНО: Используем fetch_ticker для получения текущей цены
        ticker = await self.connector.exchange.fetch_ticker(symbol)
        if ticker and 'last' in ticker:
          return float(ticker['last'])

        logger.warning(f"Не удалось получить текущую цену для {symbol}")
        return None
      except Exception as e:
        logger.error(f"Ошибка при получении текущей цены для {symbol}: {e}", exc_info=True)
        return None