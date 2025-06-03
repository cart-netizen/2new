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
      contracts = self.connector.get_usdt_perpetual_contracts()
      if not contracts:
        logger.warning("Не удалось получить список контрактов от Bybit.")
        return []

      valid_symbols = []
      for contract in contracts:
        try:
          # Bybit V5 API возвращает 'turnover24h' (оборот в котируемой валюте, USDT)
          # и 'volume24h' (объем в базовой валюте)
          # Нам нужен оборот в USDT
          volume_24h_usdt = float(contract.get('turnover24h', 0))
          symbol = contract.get('symbol')

          if symbol and "USDT" in symbol and volume_24h_usdt >= trading_params.MIN_24H_VOLUME_USDT:
            valid_symbols.append({
              'symbol': symbol,
              'volume24h_usdt': volume_24h_usdt
            })
        except ValueError:
          logger.warning(f"Некорректные данные объема для контракта: {contract.get('symbol')}")
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
      logger.error(f"Ошибка при получении и фильтрации активных символов: {e}")
      return []

  async def get_historical_candles(self, symbol: str, timeframe: Timeframe, limit: int = 200) -> pd.DataFrame:
    """
    Получает исторические свечи для символа.
    """
    logger.debug(f"Запрос исторических свечей для {symbol}, таймфрейм {timeframe.value}, лимит {limit}")
    try:
      # Преобразуем Timeframe enum в строку, которую понимает Bybit
      interval_str = timeframe.value

      raw_candles = self.connector.get_kline(symbol, interval_str, limit)
      if not raw_candles:
        logger.warning(f"Нет данных свечей для {symbol} с таймфреймом {interval_str}")
        return pd.DataFrame()

      # Bybit возвращает данные в обратном порядке (самые новые сначала), переворачиваем для pandas_ta
      df = pd.DataFrame(raw_candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
      ]).iloc[::-1]

      # Преобразование типов
      df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
      for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = pd.to_numeric(df[col])

      df.set_index('timestamp', inplace=True)
      logger.info(f"Получено {len(df)} свечей для {symbol} ({interval_str})")
      return df
    except Exception as e:
      logger.error(f"Ошибка при получении исторических свечей для {symbol}: {e}")
      return pd.DataFrame()

  async def get_current_price(self, symbol: str) -> Optional[float]:
    """Получает текущую цену для символа (цена последней сделки)"""
    try:
      tickers = self.connector.get_usdt_perpetual_contracts()  # Можно оптимизировать, чтобы не запрашивать все тикеры
      if tickers:
        for ticker in tickers:
          if ticker.get('symbol') == symbol:
            return float(ticker.get('lastPrice', 0.0))
      logger.warning(f"Не удалось получить текущую цену для {symbol}")
      return None
    except Exception as e:
      logger.error(f"Ошибка при получении текущей цены для {symbol}: {e}")
      return None