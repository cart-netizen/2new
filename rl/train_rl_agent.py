# rl/train_rl_agent.py

import asyncio
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt

from config.config_manager import ConfigManager
from core.data_fetcher import DataFetcher
from core.bybit_connector import BybitConnector
from core.enums import Timeframe
from data.database_manager import AdvancedDatabaseManager
from ml.feature_engineering import AdvancedFeatureEngineer
from ml.enhanced_ml_system import EnhancedEnsembleModel
from ml.anomaly_detector import MarketAnomalyDetector
from ml.volatility_system import VolatilityPredictionSystem, ModelType
from core.market_regime_detector import MarketRegimeDetector
from core.risk_manager import AdvancedRiskManager

from rl.environment import BybitTradingEnvironment
from rl.finrl_agent import EnhancedRLAgent
from rl.feature_processor import RLFeatureProcessor
from rl.reward_functions import RiskAdjustedRewardFunction
from rl.data_preprocessor import prepare_data_for_finrl
from stable_baselines3.common.callbacks import BaseCallback

from rl.safe_wrapper import SafeEnvironmentWrapper
from utils.logging_config import get_logger

logger = get_logger(__name__)


class RLTrainer:
  """Класс для обучения RL агента с полной интеграцией"""

  def __init__(self, config: Dict[str, Any]):
    self.config_manager = ConfigManager(config_path='../config.json')
    self.config = self.config_manager.load_config()

    # Инициализация компонентов
    self.connector = None
    self.data_fetcher = None
    self.db_manager = None
    self.feature_engineer = None
    self.ml_model = None
    self.anomaly_detector = None
    self.volatility_predictor = None
    self.market_regime_detector = None
    self.risk_manager = None

    # RL компоненты
    self.rl_agent = None
    self.feature_processor = None
    self.environment = None

    # Результаты обучения
    self.training_results = {}

  def _load_config(self, config_path: str) -> Dict:
    """Загружает конфигурацию"""
    try:
      with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    except Exception as e:
      logger.error(f"Ошибка загрузки конфигурации: {e}")
      return {}

  async def initialize_components(self):
    """Инициализирует все необходимые компоненты"""
    logger.info("Инициализация компонентов для обучения RL...")

    try:
      # Базовые компоненты
      self.connector = BybitConnector()
      self.data_fetcher = DataFetcher(self.connector, self.config)
      self.db_manager = AdvancedDatabaseManager()
      self.feature_processor = RLFeatureProcessor(self.config)

      # ML компоненты
      self.feature_engineer = AdvancedFeatureEngineer()

      # Инициализируем ML модель
      self.ml_model = EnhancedEnsembleModel(
        # feature_columns=[],  # Будут определены при обучении
        # use_market_regime=True,
        # use_correlation_filter=True
      )

      # Детектор аномалий
      self.anomaly_detector = MarketAnomalyDetector(
        contamination=0.1,
        # n_estimators=100
        lookback_periods=100
      )
      logger.info("Детектор аномалий инициализирован с эвристиками")
      # Создаем простой детектор с эвристиками, если обучить не удается
      if self.anomaly_detector:
        logger.info("Инициализация детектора аномалий с эвристиками...")
        # Детектор будет использовать правила по умолчанию

      # Предсказатель волатильности
      self.volatility_predictor = VolatilityPredictionSystem(
        model_type=ModelType.LIGHTGBM,
        # prediction_horizon=[1, 3, 5, 10],
        prediction_horizon=5,
        auto_retrain=True
      )

      # Детектор режимов рынка
      self.market_regime_detector = MarketRegimeDetector(self.data_fetcher)

      # Риск-менеджер
      self.risk_manager = AdvancedRiskManager(
        # initial_capital=self.rl_config.get('initial_capital', 10000),
        # db_manager=self.db_manager
        db_manager=self.db_manager,
        settings=self.config,  # self.config уже загружен ранее
        data_fetcher=self.data_fetcher,  # self.data_fetcher уже создан
        volatility_predictor=None
      )

      # RL Feature Processor
      self.feature_processor = RLFeatureProcessor(
        feature_engineer=self.feature_engineer,
        config=self.config.get('feature_config', {})
      )

      logger.info("✅ Все компоненты успешно инициализированы")

    except Exception as e:
      logger.error(f"Ошибка инициализации компонентов: {e}", exc_info=True)
      raise

  # async def load_training_data(self) -> Optional[pd.DataFrame]:
  #   """
  #   Загружает и подготавливает данные для обучения с корректной обработкой
  #   и обогащением признаками для каждой группы символов.
  #   """
  #   logger.info("Загрузка данных для обучения...")
  #
  #   # --- Шаг 1: Загрузка "сырых" данных ---
  #   symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
  #   timeframe = Timeframe.ONE_HOUR
  #   limit = self.config.get('training_config', {}).get('history_bars', 2000)
  #
  #   raw_data_dict = {}
  #   for symbol in symbols:
  #     data = await self.data_fetcher.get_historical_candles(
  #       symbol=symbol, timeframe=timeframe, limit=limit
  #     )
  #     if data is not None and not data.empty:
  #       raw_data_dict[symbol] = data
  #
  #   if not raw_data_dict:
  #     raise ValueError("Не удалось загрузить сырые данные ни для одного символа")
  #
  #   # --- Шаг 2: Выравнивание и добавление базовых индикаторов ---
  #   unaligned_df = prepare_data_for_finrl(raw_data_dict, list(raw_data_dict.keys()))
  #
  #   df_pivot = unaligned_df.pivot(index='date', columns='tic', values='close').dropna()
  #   aligned_df = unaligned_df[unaligned_df.date.isin(df_pivot.index)]
  #
  #   data_with_custom_features = await self._add_technical_indicators(aligned_df)
  #
  #   # --- Шаг 3: Асинхронное добавление ML признаков для каждой группы ---
  #   logger.info("Добавление ML-признаков для каждой группы символов...")
  #
  #   tasks = []
  #   # Группируем по 'tic' и итерируем, получая имя группы (symbol) и саму группу (group_df)
  #   for symbol, group_df in data_with_custom_features.groupby('tic'):
  #     tasks.append(self._add_ml_features(group_df, symbol))
  #
  #   # Запускаем задачи параллельно
  #   results = await asyncio.gather(*tasks, return_exceptions=True)
  #
  #   # Собираем обработанные группы обратно в один DataFrame
  #   processed_groups = [res for res in results if isinstance(res, pd.DataFrame)]
  #   if not processed_groups:
  #     raise ValueError("Не удалось добавить ML признаки ни для одной группы.")
  #
  #   data_with_ml_features = pd.concat(processed_groups, ignore_index=True)
  #   data_with_ml_features.sort_values(['date', 'tic'], inplace=True)
  #
  #   # --- Шаг 4: Добавление стандартных FinRL индикаторов ---
  #   finrl_ready_df = self._add_finrl_indicators(data_with_ml_features)
  #
  #   if finrl_ready_df is None or finrl_ready_df.empty:
  #     raise ValueError("Нет данных после добавления индикаторов FinRL")
  #
  #   logger.info(f"📊 Финально подготовлено {len(finrl_ready_df)} записей для обучения")
  #
  #   return finrl_ready_df

  async def load_training_data(self) -> pd.DataFrame:
    """Загружает данные для обучения"""
    try:
      symbols = self.config.get('symbols', ["BTCUSDT",
                "ETHUSDT",
                "SOLUSDT",
                "XRPUSDT"])
      timeframe = Timeframe.ONE_HOUR

      logger.info(f"Загрузка данных для символов: {symbols}")

      # Словарь для хранения данных по символам
      all_data = {}

      # Загружаем максимально доступное количество свечей
      max_limit = 1000

      for symbol in symbols:
        logger.info(f"Загрузка данных для {symbol}...")

        # Собираем данные через несколько последовательных запросов
        all_candles = []
        last_timestamp = None

        # Делаем до 10 запросов по 200 свечей = 2000 свечей максимум
        for i in range(10):
          try:
            if i > 0:
              await asyncio.sleep(0.2)  # Задержка между запросами

            # Подготавливаем параметры для запроса
            if hasattr(self.connector, '_make_request'):
              # Прямой запрос к API Bybit
              params = {
                "symbol": symbol,
                "interval": "60",  # 1 час
                "limit": 200
              }

              # Если есть последняя временная метка, запрашиваем данные ДО нее
              if last_timestamp:
                params["endTime"] = last_timestamp

              result = await self.connector._make_request("GET", "/v5/market/kline", params)

              if result and result.get("result") and result["result"].get("list"):
                klines = result["result"]["list"]

                # Преобразуем в DataFrame
                df = pd.DataFrame(klines)
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']

                # Конвертируем типы
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                  df[col] = pd.to_numeric(df[col], errors='coerce')

                all_candles.append(df)

                # Обновляем last_timestamp для следующего запроса
                last_timestamp = int(df['timestamp'].min().timestamp() * 1000) - 1

                logger.info(f"  Запрос {i + 1}: получено {len(df)} свечей")
              else:
                logger.warning(f"  Запрос {i + 1}: нет данных")
                break

            else:
              # Fallback на стандартный метод
              data = await self.data_fetcher.get_historical_candles(
                symbol=symbol,
                timeframe=timeframe,
                limit=200,
                use_cache=False
              )

              if data is not None and not data.empty:
                all_candles.append(data)
                logger.info(f"  Запрос {i + 1}: получено {len(data)} свечей")
              else:
                break

          except Exception as e:
            logger.error(f"Ошибка при запросе {i + 1} для {symbol}: {e}")
            break

        # Объединяем все полученные данные
        if all_candles:
          combined_data = pd.concat(all_candles, ignore_index=True)

          # Убираем дубликаты по timestamp
          if 'timestamp' in combined_data.columns:
            combined_data = combined_data.drop_duplicates(subset=['timestamp'], keep='first')
            combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)

          logger.info(f"✅ Всего загружено для {symbol}: {len(combined_data)} уникальных свечей")

          # Проверяем и исправляем структуру данных
          if 'timestamp' not in combined_data.columns and isinstance(combined_data.index, pd.DatetimeIndex):
            combined_data = combined_data.reset_index()
            if 'index' in combined_data.columns:
              combined_data.rename(columns={'index': 'timestamp'}, inplace=True)

          # Добавляем базовые индикаторы
          if 'volume_ratio' not in combined_data.columns:
            combined_data['volume_ratio'] = combined_data['volume'] / combined_data['volume'].rolling(window=20,
                                                                                                      min_periods=1).mean()
            combined_data['volume_ratio'] = combined_data['volume_ratio'].fillna(1.0)

          if 'rsi' not in combined_data.columns:
            delta = combined_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            combined_data['rsi'] = 100 - (100 / (1 + rs))
            combined_data['rsi'] = combined_data['rsi'].fillna(50)

          if 'sma_20' not in combined_data.columns:
            combined_data['sma_20'] = combined_data['close'].rolling(window=20, min_periods=1).mean()

          # Финальная очистка
          combined_data = combined_data.fillna(method='ffill').fillna(method='bfill').fillna(0)

          # Сохраняем
          all_data[symbol] = combined_data
        else:
          logger.error(f"❌ Не удалось загрузить данные для {symbol}")
        return pd.DataFrame()

      # ДОБАВИТЬ ДИАГНОСТИКУ:
      logger.info("📊 Диагностика загруженных данных:")
      for symbol, data in all_data.items():
        logger.info(f"  {symbol}: {len(data)} свечей")
        if 'timestamp' in data.columns:
          logger.info(f"    Период: {data['timestamp'].min()} - {data['timestamp'].max()}")

      # Теперь передаем словарь в prepare_data_for_finrl
      df = prepare_data_for_finrl(all_data, list(all_data.keys()))

      # Проверяем результат
      if df is not None and not df.empty:
        unique_dates = df['date'].nunique()
        total_rows = len(df)
        rows_per_symbol = total_rows // len(symbols)

        logger.info(f"📊 Итоговый размер данных:")
        logger.info(f"   - Всего строк: {total_rows}")
        logger.info(f"   - Уникальных дат: {unique_dates}")
        logger.info(f"   - Строк на символ: {rows_per_symbol}")

        # Минимальная проверка
        min_required_per_symbol = 800
        if rows_per_symbol < min_required_per_symbol:
          logger.warning(f"⚠️ Данных меньше рекомендуемого: {rows_per_symbol} < {min_required_per_symbol}")

      return df

    except Exception as e:
      logger.error(f"Ошибка загрузки данных: {e}", exc_info=True)
      return pd.DataFrame()

  async def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
    """Добавляет технические индикаторы"""
    import pandas_ta as ta

    # RSI
    data['rsi'] = ta.rsi(data['close'], length=14)

    # MACD
    macd = ta.macd(data['close'])
    if macd is not None and not macd.empty:
      data['macd'] = macd.iloc[:, 0]
      data['macd_signal'] = macd.iloc[:, 1]
      data['macd_diff'] = macd.iloc[:, 2]

    # Bollinger Bands
    bb = ta.bbands(data['close'], length=20, std=2)
    if bb is not None and not bb.empty:
      data['bb_lower'] = bb.iloc[:, 0]
      data['bb_middle'] = bb.iloc[:, 1]
      data['bb_upper'] = bb.iloc[:, 2]

    # ATR
    data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)

    # ADX
    adx = ta.adx(data['high'], data['low'], data['close'], length=14)
    if adx is not None and not adx.empty:
      data['adx'] = adx.iloc[:, 0]

    # CCI
    data['cci'] = ta.cci(data['high'], data['low'], data['close'], length=20)

    # Заполняем пропуски
    data = data.fillna(method='bfill').fillna(0)

    return data

  async def _add_ml_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Добавляет ML признаки"""
    try:
      # Детекция режима рынка
      regime = await self.market_regime_detector.detect_regime(symbol, data)

      # ИСПРАВЛЕНО: RegimeCharacteristics не имеет .value
      if hasattr(regime, 'name'):
        data['market_regime'] = regime.name
      else:
        data['market_regime'] = str(regime) if regime else 'UNKNOWN'

      # Числовое представление режима
      regime_mapping = {
        'STRONG_TREND_UP': 4,
        'TREND_UP': 3,
        'RANGE_BOUND': 2,
        'TREND_DOWN': 1,
        'STRONG_TREND_DOWN': 0,
        'UNKNOWN': -1
      }
      data['market_regime_numeric'] = regime_mapping.get(data['market_regime'].iloc[-1], -1)

      # Прогноз волатильности
      try:
        if hasattr(self.volatility_predictor, 'predict_volatility'):
          vol_pred = self.volatility_predictor.predict_volatility(data)
        elif hasattr(self.volatility_predictor, 'predict_future_volatility'):
          vol_pred = self.volatility_predictor.predict_future_volatility(data)
        else:
          vol_pred = data['close'].pct_change().rolling(20).std().iloc[-1]

        if isinstance(vol_pred, dict):
          data['predicted_volatility'] = vol_pred.get('volatility', 0) or vol_pred.get('predictions', {}).get(1, 0)
        else:
          data['predicted_volatility'] = float(vol_pred) if vol_pred else 0.0
      except Exception as e:
        logger.warning(f"Не удалось получить прогноз волатильности: {e}")
        data['predicted_volatility'] = 0.0

      # ИСПРАВЛЕНО: убираем await, так как detect_anomalies синхронный
      anomaly_reports = self.anomaly_detector.detect_anomalies(data, symbol)
      if anomaly_reports:

        anomaly_score = max(report.severity for report in anomaly_reports)
      else:
        anomaly_score = 0.0
      data['anomaly_score'] = anomaly_score

    except Exception as e:
      logger.error(f"Ошибка добавления ML признаков: {e}")
      # Добавляем значения по умолчанию
      data['market_regime'] = 'UNKNOWN'
      data['market_regime_numeric'] = -1
      data['predicted_volatility'] = 0.0
      data['anomaly_score'] = 0.0

    return data

  def _add_finrl_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет индикаторы в формате FinRL для всех символов"""
    if 'tic' not in df.columns:
      raise ValueError("DataFrame должен содержать колонку 'tic'")

    # Список индикаторов, которые мы будем добавлять
    indicators_config = {
      'rsi': {'default': 50.0, 'required': True},
      'macd': {'default': 0.0, 'required': True},
      'macd_signal': {'default': 0.0, 'required': True},
      'macd_diff': {'default': 0.0, 'required': True},
      'cci': {'default': 0.0, 'required': True},
      'adx': {'default': 25.0, 'required': True},
      'atr': {'default': 0.0, 'required': True}
    }

    result_dfs = []

    for tic in df['tic'].unique():
      tic_df = df[df['tic'] == tic].copy()

      # Проверяем наличие ценовых данных
      price_cols = ['open', 'high', 'low', 'close', 'volume']
      for col in price_cols:
        if col not in tic_df.columns:
          raise ValueError(f"Отсутствует колонка {col} для {tic}")

        # Убеждаемся, что данные числовые
        tic_df[col] = pd.to_numeric(tic_df[col], errors='coerce')

      # Добавляем индикаторы
      for indicator, config in indicators_config.items():
        if indicator not in tic_df.columns:
          tic_df[indicator] = config['default']
        else:
          # Проверяем и исправляем тип данных
          tic_df[indicator] = pd.to_numeric(tic_df[indicator], errors='coerce').fillna(config['default'])

      # Убеждаемся, что нет NaN
      tic_df = tic_df.fillna(method='ffill').fillna(method='bfill')

      # Если все еще есть NaN, заполняем дефолтными значениями
      for col in tic_df.columns:
        if tic_df[col].isna().any():
          if col in indicators_config:
            tic_df[col] = tic_df[col].fillna(indicators_config[col]['default'])
          else:
            tic_df[col] = tic_df[col].fillna(0)

      result_dfs.append(tic_df)

    if not result_dfs:
      raise ValueError("Нет данных после обработки")

    # Объединяем и сортируем
    final_df = pd.concat(result_dfs, ignore_index=True)
    final_df = final_df.sort_values(['date', 'tic']).reset_index(drop=True)

    logger.info(f"Добавлены индикаторы. Финальная форма: {final_df.shape}")
    logger.info(f"Колонки: {final_df.columns.tolist()}")

    return final_df

  async def create_environment(self, df: pd.DataFrame) -> SafeEnvironmentWrapper:
    """Создает торговую среду"""
    logger.info("Создание торговой среды...")

    unique_dates = df['date'].nunique()
    unique_symbols = df['tic'].nunique()

    logger.info(f"Создание среды: {unique_dates} дней, {unique_symbols} символов")

    # Минимальная проверка для FinRL
    min_days_required = 800  # Для эпизодов по 800 дней
    if unique_dates < min_days_required:
      logger.warning(f"Недостаточно дней: {unique_dates} < {min_days_required}")

    if len(df) < 800:  # Минимум для полноценного эпизода
      logger.warning(f"Недостаточно данных: {len(df)} строк. Дополняем...")
      # Здесь можно либо загрузить больше данных, либо использовать то что есть

    # Базовые колонки, которые не являются индикаторами
    base_columns = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume', 'timestamp', 'turnover']

    # Находим технические индикаторы
    tech_indicators = []
    for col in df.columns:
      if col not in base_columns and df[col].dtype in ['float64', 'int64', 'float32']:
        tech_indicators.append(col)

    logger.info(f"Найдено технических индикаторов: {len(tech_indicators)}")
    logger.info(f"Индикаторы: {tech_indicators[:10]}...")  # Показываем первые 10

    # Детальная диагностика
    logger.info(f"Входной DataFrame shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    if 'tic' in df.columns:
      logger.info(f"Уникальные tickers: {df['tic'].unique()}")

    if 'date' in df.columns:
      logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Проверка структуры
    logger.info(f"DataFrame dtypes:\n{df.dtypes}")
    logger.info(f"Первые 5 строк:\n{df.head()}")

    # Отладка структуры данных
    debug_dataframe_structure(df, "Before environment creation")

    # КРИТИЧНО: Убедимся, что данные в правильном формате для FinRL
    if 'tic' not in df.columns or 'date' not in df.columns:
      raise ValueError("DataFrame должен содержать колонки 'tic' и 'date'")

    # Убедимся, что индекс - это RangeIndex, а не даты
    if not isinstance(df.index, pd.RangeIndex):
      df = df.reset_index(drop=True)

    # Проверим и исправим типы данных
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
      if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Проверяем на NaN
        if df[col].isna().any():
          logger.warning(f"Found NaN in {col}, filling with forward fill")
          df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

    # ВАЖНО: FinRL требует, чтобы данные были отсортированы и выровнены
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)

    # Создаем функцию вознаграждения
    reward_function = RiskAdjustedRewardFunction(
      risk_manager=self.risk_manager,
      config=self.config.get('reward_config', {})
    )

    # Параметры среды
    env_config = {
      'hmax': 100,
      'initial_amount': self.config.get('initial_capital', 10000),
      'transaction_cost_pct': 0.001,
      'reward_scaling': 1e-4,
      'buy_cost_pct': 0.001,
      'sell_cost_pct': 0.001,
      'tech_indicator_list': tech_indicators
    }

    # Финальная отладка перед созданием среды
    debug_dataframe_structure(df, "Final check before environment")

    try:
      # Создаем среду
      environment = BybitTradingEnvironment(
        df=df,
        data_fetcher=self.data_fetcher,
        market_regime_detector=self.market_regime_detector,
        risk_manager=self.risk_manager,
        shadow_trading_manager=None,
        feature_engineer=self.feature_engineer,
        initial_balance=env_config['initial_amount'],
        commission_rate=env_config['transaction_cost_pct'],
        leverage=self.config.get('leverage', 10),
        max_positions=self.config.get('portfolio_config', {}).get('max_positions', 10),
        config=env_config
      )

      logger.info("✅ Торговая среда создана успешно")
      # Проверяем размерности
      logger.info(f"State space: {environment.state_space}")
      logger.info(f"Action space: {environment.action_space}")
      logger.info(f"Stock dim: {environment.stock_dim}")

    except Exception as e:
      logger.error(f"Ошибка создания среды: {e}")
      logger.error(f"DataFrame info:\n{df.info()}")
      raise

    from rl.safe_wrapper import SafeEnvironmentWrapper
    safe_environment = SafeEnvironmentWrapper(environment)

    # Устанавливаем функцию вознаграждения

    # environment.reward_function = reward_function
    safe_environment.reward_function = reward_function

    return safe_environment

  async def train_agent(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Обучает RL агента"""
    logger.info("=" * 50)
    logger.info("НАЧАЛО ОБУЧЕНИЯ RL АГЕНТА")
    logger.info("=" * 50)

    # Создаем среды для обучения и тестирования
    train_env = await self.create_environment(train_df)
    test_env = await self.create_environment(test_df)

    # Добавьте проверку:
    logger.info(f"Test environment info: obs_space={test_env.observation_space.shape}, "
                f"action_space={test_env.action_space.shape}")

    # Убедитесь, что eval_env правильно инициализирована
    if hasattr(test_env, 'df'):
      logger.info(f"Test env data shape: {test_env.df.shape}")

    # Создаем RL агента
    self.rl_agent = EnhancedRLAgent(
      environment=train_env,
      ml_model=self.ml_model,
      anomaly_detector=self.anomaly_detector,
      volatility_predictor=self.volatility_predictor,
      algorithm=self.config.get('algorithm', 'PPO'),
      config=self.config
    )



    # Параметры обучения
    training_config = self.config.get('training_config', {})
    total_timesteps = training_config.get('total_timesteps', 100000)
    eval_freq = training_config.get('eval_frequency', 10000)
    save_freq = training_config.get('save_frequency', 20000)



    # Создаем callback для мониторинга
    class TrainingCallback(BaseCallback):
      """
      Кастомный callback для мониторинга процесса обучения
      """

      def __init__(self, trainer, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.trainer = trainer
        self.episode_rewards = []
        self.episode_lengths = []
        self.n_episodes = 0

      def _on_step(self) -> bool:
        """
        Вызывается на каждом шаге среды
        """
        # Проверяем последнее вознаграждение
        if hasattr(self.locals, 'rewards') and len(self.locals['rewards']) > 0:
          last_reward = self.locals['rewards'][-1]
          if last_reward != 0:
            logger.info(f"🎯 Ненулевая награда: {last_reward}")

        # Получаем текущую информацию
        if self.num_timesteps % 100 == 0:
          # Получаем текущие наблюдения из среды
          if hasattr(self.model, '_last_obs') and self.model._last_obs is not None:
            obs = self.model._last_obs
            if isinstance(obs, np.ndarray) and len(obs) > 0:
              # Проверяем первый элемент (баланс)
              balance = obs[0][0] if obs.ndim > 1 else obs[0]
              logger.debug(f"Шаг {self.num_timesteps}: Баланс = {balance:.2f}")

              # Проверка на аномальные значения
              if balance <= 0 or balance > 1e6:
                logger.warning(f"Аномальный баланс на шаге {self.num_timesteps}: {balance}")

        # Проверяем завершение эпизода
        if self.locals.get('dones', [False])[0]:
          self.n_episodes += 1

          # Получаем информацию о последнем эпизоде
          info = self.locals.get('infos', [{}])[0]
          episode_reward = info.get('episode', {}).get('r', 0)
          episode_length = info.get('episode', {}).get('l', 0)

          self.episode_rewards.append(episode_reward)
          self.episode_lengths.append(episode_length)

          # Выводим статистику каждые 10 эпизодов
          if self.n_episodes % 10 == 0:
            avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
            avg_length = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0

            logger.info(f"\n{'=' * 50}")
            logger.info(f"Эпизод: {self.n_episodes}")
            logger.info(f"Средняя награда (последние 10): {avg_reward:.2f}")
            logger.info(f"Средняя длина эпизода: {avg_length:.0f}")
            logger.info(f"Всего шагов: {self.num_timesteps}")
            logger.info(f"{'=' * 50}\n")

        # Сохраняем метрики из info_buffer если есть
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
          for ep_info in self.model.ep_info_buffer:
            if 'r' in ep_info and 'l' in ep_info:
              self.episode_rewards.append(ep_info['r'])
              self.episode_lengths.append(ep_info['l'])
              self.n_episodes += 1

        # Логируем прогресс каждые 1000 шагов
        if self.num_timesteps % 1000 == 0:
          logger.info(
            f"Шаг {self.num_timesteps}/{self.trainer.config.get('training_config', {}).get('total_timesteps', 100000)}")
          if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards[-10:])
            logger.info(f"Средняя награда (последние 10 эпизодов): {avg_reward:.2f}")

        # # Логируем прогресс каждые 5000 шагов
        # if self.num_timesteps % 5000 == 0 and self.num_timesteps > 0:
        #   logger.info(f"Прогресс: {self.num_timesteps} шагов выполнено")

        if len(self.episode_rewards) > 20:
          recent_avg = np.mean(self.episode_rewards[-10:])
          older_avg = np.mean(self.episode_rewards[-20:-10])

          if recent_avg > older_avg * 1.1:  # Улучшение на 10%
            logger.info(f"📈 Улучшение! Старое: {older_avg:.2f}, Новое: {recent_avg:.2f}")


        return True  # Продолжаем обучение

      def _on_rollout_end(self) -> None:
        """
        Вызывается в конце rollout
        """
        pass

      def _on_training_end(self) -> None:
        """
        Вызывается в конце обучения
        """
        logger.info("Обучение завершено!")

      def __call__(self, locals_dict, globals_dict):
        # Сохраняем метрики
        if 'episode_rewards' in locals_dict:
          self.episode_rewards.extend(locals_dict['episode_rewards'])
        if 'episode_lengths' in locals_dict:
          self.episode_lengths.extend(locals_dict['episode_lengths'])

        # Логируем прогресс каждые 1000 шагов
        if locals_dict['self'].num_timesteps % 1000 == 0:
          logger.info(f"Шаг {locals_dict['self'].num_timesteps}/{total_timesteps}")
          if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards[-10:])
            logger.info(f"Средняя награда (последние 10 эпизодов): {avg_reward:.2f}")

        return True

    callback = TrainingCallback(self, verbose=1)

    # Обучаем агента
    logger.info(f"Обучение {self.config.get('algorithm', 'PPO')} на {total_timesteps} шагов...")

    self.rl_agent.train(
      total_timesteps=total_timesteps,
      callback=callback,
      log_interval=100,
      eval_env=test_env,
      eval_freq=eval_freq,
      n_eval_episodes=5,
      save_freq=save_freq
    )

    # Анализ результатов обучения
    logger.info("\n" + "=" * 60)
    logger.info("АНАЛИЗ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ")
    logger.info("=" * 60)

    if hasattr(callback, 'episode_rewards') and callback.episode_rewards:
      total_episodes = len(callback.episode_rewards)
      avg_reward = np.mean(callback.episode_rewards)
      std_reward = np.std(callback.episode_rewards)
      max_reward = np.max(callback.episode_rewards)
      min_reward = np.min(callback.episode_rewards)

      logger.info(f"Всего эпизодов: {total_episodes}")
      logger.info(f"Средняя награда: {avg_reward:.2f} ± {std_reward:.2f}")
      logger.info(f"Максимальная награда: {max_reward:.2f}")
      logger.info(f"Минимальная награда: {min_reward:.2f}")

      # Показываем тренд
      if total_episodes > 20:
        early_avg = np.mean(callback.episode_rewards[:10])
        late_avg = np.mean(callback.episode_rewards[-10:])
        improvement = ((late_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0

        logger.info(f"\nУлучшение производительности:")
        logger.info(f"Первые 10 эпизодов: {early_avg:.2f}")
        logger.info(f"Последние 10 эпизодов: {late_avg:.2f}")
        logger.info(f"Изменение: {improvement:+.1f}%")

    # Сохраняем результаты обучения
    self.training_results = {
      'algorithm': self.config.get('algorithm', 'PPO'),
      'total_timesteps': total_timesteps,
      'episode_rewards': callback.episode_rewards,
      'episode_lengths': callback.episode_lengths,
      'training_time': datetime.now().isoformat(),
      'symbols': self.config.get('symbols', []),
      'final_stats': self.rl_agent.get_training_stats()
    }

    logger.info("=" * 50)
    logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    logger.info(f"Финальная статистика: {self.training_results['final_stats']}")
    logger.info("=" * 50)

  async def evaluate_agent(self, test_df: pd.DataFrame):
    """Оценивает производительность агента"""
    logger.info("Оценка производительности агента...")

    # Создаем тестовую среду
    test_env = await self.create_environment(test_df)

    # Запускаем тестирование
    # ИСПРАВЛЕНО: правильно обрабатываем возврат reset()
    obs, info = test_env.reset()  # reset возвращает кортеж
    done = False
    truncated = False

    rewards = []
    actions = []
    portfolio_values = []

    # ИСПРАВЛЕНО: используем переменную для хранения текущего баланса
    initial_balance = test_env.initial_amount if hasattr(test_env, 'initial_amount') else 10000
    portfolio_values.append(initial_balance)

    while not done and not truncated:
      # Получаем действие от агента
      action, _ = self.rl_agent.predict(obs, deterministic=True)  # obs уже numpy array

      # Выполняем действие
      obs, reward, done, truncated, info = test_env.step(action)  # 5 возвращаемых значений

      # Сохраняем метрики
      rewards.append(reward)
      actions.append(action)

      # Получаем текущий баланс из состояния
      if isinstance(obs, np.ndarray) and len(obs) > 0:
        current_balance = obs[0]  # Первый элемент состояния - это баланс
        portfolio_values.append(current_balance)

    # Рассчитываем метрики производительности
    if len(portfolio_values) > 1:
      total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    else:
      total_return = 0

    sharpe_ratio = self._calculate_sharpe_ratio(rewards)
    max_drawdown = self._calculate_max_drawdown(portfolio_values)
    win_rate = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0

    evaluation_results = {
      'total_return': total_return,
      'total_return_pct': total_return * 100,
      'sharpe_ratio': sharpe_ratio,
      'max_drawdown': max_drawdown,
      'win_rate': win_rate,
      'total_trades': len([a for a in actions if np.any(a != 0)]),  # Не считаем нулевые действия
      'final_portfolio_value': portfolio_values[-1] if portfolio_values else initial_balance,
      'total_rewards': sum(rewards)
    }

    self.training_results['evaluation'] = evaluation_results

    logger.info("📊 Результаты оценки:")
    logger.info(f"  Общая доходность: {total_return * 100:.2f}%")
    logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    logger.info(f"  Максимальная просадка: {max_drawdown * 100:.2f}%")
    logger.info(f"  Win Rate: {win_rate * 100:.2f}%")
    logger.info(f"  Всего сделок: {evaluation_results['total_trades']}")

    return evaluation_results

  def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
    """Рассчитывает Sharpe Ratio"""
    if not returns or len(returns) < 2:
      return 0

    returns_array = np.array(returns)
    if np.std(returns_array) == 0:
      return 0

    # Annualized Sharpe
    sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
    return sharpe

  def _calculate_max_drawdown(self, values: List[float]) -> float:
    """Рассчитывает максимальную просадку"""
    if not values:
      return 0

    peak = values[0]
    max_dd = 0

    for value in values:
      if value > peak:
        peak = value
      drawdown = (peak - value) / peak
      max_dd = max(max_dd, drawdown)

    return max_dd

  def save_results(self):
    """Сохраняет результаты обучения"""
    # Создаем директорию для результатов
    results_dir = Path("rl/training_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем модель
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{self.config.get('algorithm', 'PPO')}_{timestamp}"
    self.rl_agent.save_model(model_name)

    # Сохраняем результаты
    results_file = results_dir / f"training_results_{timestamp}.json"
    with open(results_file, 'w') as f:
      json.dump(self.training_results, f, indent=2, default=str)

    # Создаем визуализацию
    self._create_visualizations(results_dir, timestamp)

    logger.info(f"✅ Результаты сохранены в {results_dir}")
    logger.info(f"✅ Модель сохранена как {model_name}")

  def _create_visualizations(self, results_dir: Path, timestamp: str):
    """Создает графики результатов"""
    try:
      plt.style.use('seaborn-v0_8-darkgrid')
      fig, axes = plt.subplots(2, 2, figsize=(15, 10))

      # График наград по эпизодам
      if self.training_results.get('episode_rewards'):
        ax = axes[0, 0]
        rewards = self.training_results['episode_rewards']
        ax.plot(rewards, alpha=0.3, label='Episode Rewards')

        # Скользящее среднее
        if len(rewards) > 10:
          ma = pd.Series(rewards).rolling(10).mean()
          ax.plot(ma, label='MA(10)', linewidth=2)

        ax.set_title('Episode Rewards During Training')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()

      # График длины эпизодов
      if self.training_results.get('episode_lengths'):
        ax = axes[0, 1]
        lengths = self.training_results['episode_lengths']
        ax.plot(lengths, alpha=0.5)
        ax.set_title('Episode Lengths')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Length')

      # График производительности на тестовых данных
      if 'evaluation' in self.training_results:
        ax = axes[1, 0]
        eval_data = self.training_results['evaluation']

        metrics = ['total_return_pct', 'sharpe_ratio', 'win_rate']
        values = [
          eval_data.get('total_return_pct', 0),
          eval_data.get('sharpe_ratio', 0) * 10,  # Масштабируем для визуализации
          eval_data.get('win_rate', 0) * 100
        ]
        labels = ['Return %', 'Sharpe x10', 'Win Rate %']

        bars = ax.bar(labels, values)
        ax.set_title('Performance Metrics')
        ax.set_ylabel('Value')

        # Добавляем значения на столбцы
        for bar, value in zip(bars, values):
          height = bar.get_height()
          ax.text(bar.get_x() + bar.get_width() / 2., height,
                  f'{value:.1f}',
                  ha='center', va='bottom')

      # Информация о модели
      ax = axes[1, 1]
      ax.axis('off')

      info_text = f"""
            Model Information:

            Algorithm: {self.config.get('algorithm', 'PPO')}
            Total Timesteps: {self.training_results.get('total_timesteps', 0):,}
            Training Time: {timestamp}

            Symbols: {', '.join(self.config.get('symbols', []))}

            Final Stats:
            {json.dumps(self.training_results.get('final_stats', {}), indent=2)}
            """

      ax.text(0.1, 0.9, info_text, transform=ax.transAxes,
              fontsize=10, verticalalignment='top', fontfamily='monospace')

      plt.tight_layout()
      plt.savefig(results_dir / f"training_results_{timestamp}.png", dpi=300, bbox_inches='tight')
      plt.close()

      logger.info("📊 Визуализация создана")

    except Exception as e:
      logger.error(f"Ошибка создания визуализации: {e}")


async def main_training():
  """Основная функция для запуска обучения"""
  logger.info("=" * 80)
  logger.info("ЗАПУСК ОБУЧЕНИЯ RL АГЕНТА")
  logger.info("=" * 80)

  # Создаем необходимые директории
  import os
  os.makedirs('results', exist_ok=True)
  os.makedirs('rl/models', exist_ok=True)
  os.makedirs('rl/models/logs', exist_ok=True)
  os.makedirs('rl/models/best_model', exist_ok=True)
  os.makedirs('rl/models/eval_logs', exist_ok=True)
  logger.info("🚀 Запуск обучения RL агента")
  trainer = None
  # 1. Загружаем конфигурацию ПЕРЕД созданием трейнера.
  # Путь '../config.json' указывает, что файл находится в папке config
  # на один уровень выше, чем папка rl.
  # Если config.json в корне, используйте path='../config.json'
  try:
    # Корректируем путь: ../ означает "на один уровень вверх" от папки rl
    config_manager = ConfigManager(config_path='../config.json')
    config = config_manager.load_config()
    logger.info("✅ Конфигурация успешно загружена.")
  except Exception as e:
    logger.error(f"❌ Критическая ошибка загрузки конфигурации: {e}")
    return

  # 2. Создаем экземпляр RLTrainer, ПЕРЕДАВАЯ ему загруженный конфиг.
  trainer = RLTrainer(config)

  # 3. Сохраняем оригинальную логику работы
  try:
    # Инициализация компонентов
    await trainer.initialize_components()

    # Загрузка данных
    df = await trainer.load_training_data()

    # Если данные не загрузились, прекращаем работу
    if df is None or df.empty:
      logger.error("❌ Не удалось загрузить данные для обучения. Процесс остановлен.")
      return

    # Дополнительная проверка размера
    unique_dates = df['date'].nunique()
    unique_symbols = df['tic'].nunique()
    total_rows = len(df)

    logger.info(f"📊 Загружено данных:")
    logger.info(f"   - Уникальных дат: {unique_dates}")
    logger.info(f"   - Символов: {unique_symbols}")
    logger.info(f"   - Всего строк: {total_rows}")

    # Минимальные требования FinRL
    min_required = 1000 * unique_symbols
    if total_rows < min_required:
      logger.error(f"❌ Недостаточно данных: {total_rows} < {min_required}")
      logger.error("   FinRL требует минимум 1000 точек на символ")
      return

    # Разделение на train/test
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]

    logger.info(f"📊 Данные разделены: train={len(train_df)}, test={len(test_df)}")

    # Обучение агента
    await trainer.train_agent(train_df, test_df)

    # Оценка производительности
    await trainer.evaluate_agent(test_df)

    # Сохранение результатов
    trainer.save_results()

    logger.info("✅ Обучение успешно завершено!")

  except Exception as e:
    logger.error(f"❌ Ошибка во время основного процесса обучения: {e}", exc_info=True)
    raise  # Повторно вызываем ошибку для полной диагностики
  finally:
    # Закрываем все соединения
    if trainer and hasattr(trainer, 'connector') and trainer.connector:
      await trainer.connector.close()
    if trainer and hasattr(trainer, 'data_fetcher') and hasattr(trainer.data_fetcher, 'connector'):
      await trainer.data_fetcher.connector.close()

def validate_finrl_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Валидирует и исправляет DataFrame для FinRL
    """
    # Проверяем необходимые колонки
    required_columns = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']

    for col in required_columns:
      if col not in df.columns:
        raise ValueError(f"Отсутствует обязательная колонка: {col}")

    # Проверяем, что у нас есть данные для всех символов на каждую дату
    # Это критично для FinRL!
    date_tic_combinations = df.groupby(['date', 'tic']).size()
    dates = df['date'].unique()
    tics = df['tic'].unique()

    # Создаем полный набор комбинаций дата-символ
    full_index = pd.MultiIndex.from_product([dates, tics], names=['date', 'tic'])

    # Проверяем пропуски
    missing_combinations = set(full_index) - set(date_tic_combinations.index)

    if missing_combinations:
      logger.warning(f"Обнаружено {len(missing_combinations)} пропущенных комбинаций дата-символ")

      # Заполняем пропуски
      for date, tic in missing_combinations:
        # Находим ближайшую доступную запись для этого символа
        tic_data = df[df['tic'] == tic]
        if len(tic_data) > 0:
          # Используем последнюю известную цену
          last_known = tic_data[tic_data['date'] < date].iloc[-1] if len(tic_data[tic_data['date'] < date]) > 0 else \
          tic_data.iloc[0]

          new_row = last_known.copy()
          new_row['date'] = date
          new_row['volume'] = 0  # Нулевой объем для пропущенных дней

          df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Сортируем по дате и символу
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)

    # Финальная проверка - убеждаемся, что все числовые колонки действительно числовые
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
      df[col] = pd.to_numeric(df[col], errors='coerce').fillna(method='ffill').fillna(0)

    return df


def debug_dataframe_structure(df: pd.DataFrame, stage: str = ""):
  """Отладка структуры DataFrame для FinRL"""
  logger.info(f"\n{'=' * 50}")
  logger.info(f"DEBUG DataFrame Structure - {stage}")
  logger.info(f"{'=' * 50}")
  logger.info(f"Shape: {df.shape}")
  logger.info(f"Columns: {df.columns.tolist()}")
  logger.info(f"Index: {df.index.name} - {type(df.index)}")
  logger.info(f"Dtypes:\n{df.dtypes}")

  if 'tic' in df.columns:
    logger.info(f"Unique tickers: {df['tic'].unique()}")
    logger.info(f"Ticker counts:\n{df['tic'].value_counts()}")

  if 'date' in df.columns:
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Unique dates: {df['date'].nunique()}")

  # Проверка на дубликаты
  if 'date' in df.columns and 'tic' in df.columns:
    duplicates = df.duplicated(subset=['date', 'tic'])
    if duplicates.any():
      logger.warning(f"Found {duplicates.sum()} duplicate date-tic combinations!")

  # Пример первых строк
  logger.info(f"First 5 rows:\n{df.head()}")
  logger.info(f"{'=' * 50}\n")


if __name__ == "__main__":
  asyncio.run(main_training())