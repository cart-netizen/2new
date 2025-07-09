# rl/train_rl_agent.py

import asyncio
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

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
    """Загружает и подготавливает данные для обучения"""
    logger.info("Загрузка данных для обучения...")

    symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
    timeframe = Timeframe.ONE_HOUR
    limit = self.config.get('training_config', {}).get('history_bars', 2000)

    all_data = {}

    # Загружаем данные для каждого символа
    for symbol in symbols:
      logger.info(f"Загрузка данных для {symbol}...")

      try:
        # Получаем исторические данные
        data = await self.data_fetcher.get_historical_candles(
          symbol=symbol,
          timeframe=timeframe,
          limit=limit
        )

        if data is not None and not data.empty:
          # Добавляем технические индикаторы
          data = await self._add_technical_indicators(data)

          # Добавляем ML признаки
          data = await self._add_ml_features(data, symbol)

          all_data[symbol] = data
          logger.info(f"✅ Загружено {len(data)} баров для {symbol}")
        else:
          logger.warning(f"⚠️ Нет данных для {symbol}")

      except Exception as e:
        logger.error(f"Ошибка загрузки данных для {symbol}: {e}")
        continue

    if not all_data:
      raise ValueError("Не удалось загрузить данные ни для одного символа")

    # Преобразуем в формат FinRL
    finrl_df = prepare_data_for_finrl(all_data, list(all_data.keys()))

    # ДОБАВЛЯЕМ ПРОВЕРКУ СТРУКТУРЫ ДАННЫХ
    logger.info(f"Структура данных после prepare_data_for_finrl:")
    logger.info(f"Колонки: {finrl_df.columns.tolist()}")
    logger.info(f"Типы данных:\n{finrl_df.dtypes}")

    # Проверяем критические колонки
    for col in ['open', 'high', 'low', 'close', 'volume']:
      if col not in finrl_df.columns:
        raise ValueError(f"Отсутствует обязательная колонка: {col}")

      # Проверяем, что данные числовые
      sample_value = finrl_df[col].iloc[0] if len(finrl_df) > 0 else None
      logger.info(f"Пример значения {col}: {sample_value}, тип: {type(sample_value)}")

    # Добавляем технические индикаторы в формате FinRL
    finrl_df = self._add_finrl_indicators(finrl_df)

    logger.info(f"📊 Подготовлено {len(finrl_df)} записей для обучения")

    return finrl_df

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
        # Преобразуем в строку или числовое значение
        data['market_regime'] = str(regime) if regime else 'UNKNOWN'

      # Если нужно числовое представление:
      regime_mapping = {
        'STRONG_TREND_UP': 4,
        'TREND_UP': 3,
        'RANGE_BOUND': 2,
        'TREND_DOWN': 1,
        'STRONG_TREND_DOWN': 0,
        'UNKNOWN': -1
      }
      data['market_regime_numeric'] = regime_mapping.get(data['market_regime'].iloc[-1], -1)

      # ИСПРАВЛЕНО: используем правильный метод для прогноза волатильности
      try:
        if hasattr(self.volatility_predictor, 'predict_volatility'):
          vol_pred = await self.volatility_predictor.predict_volatility(symbol, data)
        elif hasattr(self.volatility_predictor, 'predict_future_volatility'):
          vol_pred = self.volatility_predictor.predict_future_volatility(data)
        else:
          # Если нет подходящего метода, используем простой расчет
          vol_pred = data['close'].pct_change().rolling(20).std().iloc[-1]

        if isinstance(vol_pred, dict):
          data['predicted_volatility'] = vol_pred.get('volatility', 0) or vol_pred.get('predictions', {}).get(1, 0)
        else:
          data['predicted_volatility'] = float(vol_pred) if vol_pred else 0.0
      except Exception as e:
        logger.warning(f"Не удалось получить прогноз волатильности: {e}")
        data['predicted_volatility'] = 0.0

      # Детекция аномалий
      anomaly_reports = await self.anomaly_detector.detect_anomalies(data, symbol)
      if anomaly_reports:
        # В качестве оценки берем максимальную "серьезность" аномалии
        anomaly_score = max(report.severity for report in anomaly_reports)
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
    # Verify input structure
    if 'tic' not in df.columns:
      raise ValueError("Input DataFrame must contain 'tic' column")

    result_dfs = []

    for tic in df['tic'].unique():
      tic_df = df[df['tic'] == tic].copy()

      # Validate price data
      price_cols = ['open', 'high', 'low', 'close']
      if not all(col in tic_df.columns for col in price_cols):
        raise ValueError(f"Missing price columns for {tic}")

      # Ensure numeric values
      for col in price_cols:
        tic_df[col] = pd.to_numeric(tic_df[col], errors='coerce')
        if tic_df[col].isna().any():
          raise ValueError(f"Non-numeric values in {col} for {tic}")

      # Add indicators with proper defaults
      indicators = {
        'rsi': 50.0,
        'macd': 0.0,
        'macd_signal': 0.0,
        'macd_diff': 0.0,
        'cci': 0.0,
        'adx': 25.0,
        'atr': 0.0
      }

      for indicator, default in indicators.items():
        if indicator not in tic_df.columns:
          tic_df[indicator] = default
        tic_df[indicator] = pd.to_numeric(tic_df[indicator], errors='coerce').fillna(default)

      result_dfs.append(tic_df)

    if not result_dfs:
      raise ValueError("No valid data after processing")

    return pd.concat(result_dfs).sort_values(['date', 'tic']).reset_index(drop=True)

  async def create_environment(self, df: pd.DataFrame) -> BybitTradingEnvironment:
    """Создает торговую среду"""
    logger.info("Создание торговой среды...")

    # Создаем функцию вознаграждения
    reward_function = RiskAdjustedRewardFunction(
        risk_manager=self.risk_manager,
        config=self.config.get('reward_config', {})
    )

    # Параметры среды - добавляем reward_scaling
    env_config = {
        'hmax': 100,
        'initial_amount': self.config.get('initial_capital', 10000),
        'transaction_cost_pct': 0.001,
        'reward_scaling': 1e-4,  # Убедитесь, что это здесь
        'buy_cost_pct': 0.001,
        'sell_cost_pct': 0.001
    }

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
        config=env_config  # Передаем полный конфиг
    )

    # Устанавливаем функцию вознаграждения
    environment.reward_function = reward_function

    logger.info("✅ Торговая среда создана")

    return environment

  async def train_agent(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Обучает RL агента"""
    logger.info("=" * 50)
    logger.info("НАЧАЛО ОБУЧЕНИЯ RL АГЕНТА")
    logger.info("=" * 50)

    # Создаем среды для обучения и тестирования
    train_env = await self.create_environment(train_df)
    test_env = await self.create_environment(test_df)

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
    class TrainingCallback:
      def __init__(self, trainer):
        self.trainer = trainer
        self.episode_rewards = []
        self.episode_lengths = []

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

    callback = TrainingCallback(self)

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
    obs = test_env.reset()
    done = False

    rewards = []
    actions = []
    portfolio_values = []

    while not done:
      # Получаем действие от агента
      action, _ = self.rl_agent.predict(obs, deterministic=True)

      # Выполняем действие
      obs, reward, done, _, info = test_env.step(action)

      # Сохраняем метрики
      rewards.append(reward)
      actions.append(action)
      portfolio_values.append(test_env.amount)

    # Рассчитываем метрики производительности
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    sharpe_ratio = self._calculate_sharpe_ratio(rewards)
    max_drawdown = self._calculate_max_drawdown(portfolio_values)
    win_rate = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0

    evaluation_results = {
      'total_return': total_return,
      'total_return_pct': total_return * 100,
      'sharpe_ratio': sharpe_ratio,
      'max_drawdown': max_drawdown,
      'win_rate': win_rate,
      'total_trades': len([a for a in actions if a != 1]),  # Не считаем HOLD
      'final_portfolio_value': portfolio_values[-1],
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
  """Основная функция обучения, исправленная и с сохранением оригинальной логики."""
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


if __name__ == "__main__":
  asyncio.run(main_training())