# rl/finrl_agent.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import torch
import joblib
from pathlib import Path

from utils.logging_config import get_logger
from core.enums import SignalType

logger = get_logger(__name__)


class EnhancedRLAgent:
  """
  RL агент, интегрированный с ML системой проекта
  Поддерживает различные алгоритмы и режимы работы
  """

  def __init__(
      self,
      environment,
      ml_model=None,
      anomaly_detector=None,
      volatility_predictor=None,
      algorithm: str = 'PPO',
      config: Dict[str, Any] = None
  ):
    """
    Инициализация агента с интеграцией ML компонентов

    Args:
        environment: Торговая среда
        ml_model: EnhancedMLModel из вашего проекта
        anomaly_detector: MarketAnomalyDetector
        volatility_predictor: VolatilityPredictor
        algorithm: Алгоритм RL (PPO, A2C, SAC, TD3)
        config: Конфигурация агента
    """
    self.env = environment
    self.ml_model = ml_model
    self.anomaly_detector = anomaly_detector
    self.volatility_predictor = volatility_predictor
    self.algorithm = algorithm.upper()
    self.config = config or {}

    # Параметры модели
    self.model = None
    self.is_trained = False
    self.training_history = []

    # Путь для сохранения моделей
    self.model_path = Path(self.config.get('model_path', 'rl/models'))
    self.model_path.mkdir(parents=True, exist_ok=True)

    # Инициализация модели
    self._initialize_model()

    logger.info(f"Инициализирован RL агент с алгоритмом {self.algorithm}")

  def _initialize_model(self):
    """Инициализация модели RL в зависимости от выбранного алгоритма"""
    # Оборачиваем среду для стабильности
    env = DummyVecEnv([lambda: self.env])

    # Параметры по умолчанию для разных алгоритмов
    default_params = {
      'PPO': {
        'learning_rate': 0.00003,  # Уменьшено с 0.0003
        'n_steps': 1024,  # Уменьшено с 2048
        'batch_size': 32,  # Уменьшено с 64
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': True,
        'sde_sample_freq': 4,
        'policy': 'MlpPolicy',
        'verbose': 1
      },
      'A2C': {
        'learning_rate': 0.0007,
        'n_steps': 5,
        'gamma': 0.99,
        'gae_lambda': 1.0,
        'ent_coef': 0.01,
        'vf_coef': 0.25,
        'max_grad_norm': 0.5,
        'use_rms_prop': True,
        'policy': 'MlpPolicy',
        'verbose': 1
      },
      'SAC': {
        'learning_rate': 0.0003,
        'buffer_size': 1000000,
        'learning_starts': 100,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'ent_coef': 'auto',
        'target_update_interval': 1,
        'target_entropy': 'auto',
        'use_sde': True,
        'sde_sample_freq': 64,
        'policy': 'MlpPolicy',
        'verbose': 1
      },
      'TD3': {
        'learning_rate': 0.0003,
        'buffer_size': 1000000,
        'learning_starts': 100,
        'batch_size': 100,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'policy_delay': 2,
        'target_policy_noise': 0.2,
        'target_noise_clip': 0.5,
        'policy': 'MlpPolicy',
        'verbose': 1
      }
    }

    # Получаем параметры для выбранного алгоритма
    params = default_params.get(self.algorithm, default_params['PPO'])

    # Обновляем параметры из конфигурации
    if 'model_params' in self.config:
      params.update(self.config['model_params'])

    # Создаем модель
    if self.algorithm == 'PPO':
      self.model = PPO(**params, env=env, tensorboard_log=f"{self.model_path}/logs/")
    elif self.algorithm == 'A2C':
      self.model = A2C(**params, env=env, tensorboard_log=f"{self.model_path}/logs/")
    elif self.algorithm == 'SAC':
      self.model = SAC(**params, env=env, tensorboard_log=f"{self.model_path}/logs/")
    elif self.algorithm == 'TD3':
      self.model = TD3(**params, env=env, tensorboard_log=f"{self.model_path}/logs/")
    else:
      raise ValueError(f"Неподдерживаемый алгоритм: {self.algorithm}")

    logger.info(f"Модель {self.algorithm} инициализирована с параметрами: {params}")

  async def get_enhanced_state(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Создание расширенного состояния с использованием ML компонентов
    """
    enhanced_state = {
      'market_features': data,
      'ml_signals': None,
      'anomaly_score': 0,
      'volatility_forecast': None,
      'regime_info': None
    }

    try:
      # Получаем предсказания ML модели
      if self.ml_model and hasattr(self.ml_model, 'predict'):
        ml_prediction = await self.ml_model.predict(data)
        enhanced_state['ml_signals'] = {
          'signal_type': ml_prediction.signal_type,
          'probability': ml_prediction.probability,
          'confidence': ml_prediction.confidence
        }

      # Детекция аномалий
      if self.anomaly_detector and hasattr(self.anomaly_detector, 'detect_anomaly'):
        anomaly_result = await self.anomaly_detector.detect_anomaly(data)
        enhanced_state['anomaly_score'] = anomaly_result.get('anomaly_score', 0)

      # Прогноз волатильности
      if self.volatility_predictor and hasattr(self.volatility_predictor, 'predict'):
        volatility_result = await self.volatility_predictor.predict(symbol, data)
        enhanced_state['volatility_forecast'] = {
          'predicted_volatility': volatility_result.get('volatility', 0),
          'confidence_interval': volatility_result.get('confidence_interval', [0, 0])
        }

      # Информация о режиме рынка
      if hasattr(self.env, 'market_regime_detector'):
        regime = self.env.market_regime_detector.current_regime
        enhanced_state['regime_info'] = {
          'current_regime': regime,
          'regime_stability': self.env.market_regime_detector.get_regime_stability()
        }

    except Exception as e:
      logger.error(f"Ошибка создания расширенного состояния: {e}")

    return enhanced_state

  def train(
      self,
      total_timesteps: int = 100000,
      callback: Optional[Any] = None,
      log_interval: int = 100,
      eval_env: Optional[Any] = None,
      eval_freq: int = 10000,
      n_eval_episodes: int = 5,
      save_freq: int = 10000
  ) -> None:
    """
    Обучение агента
    """
    logger.info(f"Начало обучения {self.algorithm} на {total_timesteps} шагов")

    # Создаем callback для оценки
    callbacks = []

    if eval_env:
      eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{self.model_path}/best_model/",
        log_path=f"{self.model_path}/eval_logs/",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
      )
      callbacks.append(eval_callback)

    # Добавляем пользовательский callback
    if callback:
      callbacks.append(callback)

    # Обучаем модель
    try:
      self.model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=log_interval,
        tb_log_name=f"{self.algorithm}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
      )

      self.is_trained = True
      logger.info("Обучение завершено успешно")

      # Сохраняем модель
      self.save_model(f"{self.algorithm}_final")

    except Exception as e:
      logger.error(f"Ошибка при обучении: {e}")
      raise

  def predict(
      self,
      observation: np.ndarray,
      deterministic: bool = True
  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Получение предсказания от модели
    """
    if not self.is_trained and self.model is None:
      raise ValueError("Модель не обучена. Сначала обучите модель или загрузите сохраненную.")

    action, _states = self.model.predict(observation, deterministic=deterministic)
    return action, _states

  def predict_with_analysis(
      self,
      observation: np.ndarray,
      market_data: pd.DataFrame
  ) -> Dict[str, Any]:
    """
    Предсказание с дополнительным анализом
    """
    # Получаем действие
    action, _ = self.predict(observation)

    # Анализируем действие
    action_mapping = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    signal_type = action_mapping.get(int(action[0]), 'HOLD')

    # Получаем Q-values если возможно
    q_values = None
    if hasattr(self.model, 'q_net') and self.algorithm in ['SAC', 'TD3']:
      with torch.no_grad():
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        q_values = self.model.q_net(obs_tensor).numpy()[0]

    # Рассчитываем уверенность
    if q_values is not None:
      confidence = float(np.exp(q_values[action[0]]) / np.sum(np.exp(q_values)))
    else:
      # Простая эвристика для уверенности
      confidence = 0.7 if action[0] != 1 else 0.5

    return {
      'action': int(action[0]),
      'signal_type': signal_type,
      'confidence': confidence,
      'q_values': q_values,
      'market_conditions': {
        'current_price': float(market_data['close'].iloc[-1]),
        'volume': float(market_data['volume'].iloc[-1]),
        'volatility': float(market_data['close'].pct_change().std() * np.sqrt(252))
      }
    }

  def save_model(self, name: str) -> str:
    """Сохранение модели"""
    save_path = self.model_path / f"{name}.zip"
    self.model.save(save_path)

    # Сохраняем дополнительную информацию
    metadata = {
      'algorithm': self.algorithm,
      'is_trained': self.is_trained,
      'training_timesteps': self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else 0,
      'config': self.config
    }

    metadata_path = self.model_path / f"{name}_metadata.pkl"
    joblib.dump(metadata, metadata_path)

    logger.info(f"Модель сохранена: {save_path}")
    return str(save_path)

  def load_model(self, name: str) -> None:
    """Загрузка модели"""
    load_path = self.model_path / f"{name}.zip"

    if not load_path.exists():
      raise FileNotFoundError(f"Модель не найдена: {load_path}")

    # Загружаем модель
    if self.algorithm == 'PPO':
      self.model = PPO.load(load_path, env=self.env)
    elif self.algorithm == 'A2C':
      self.model = A2C.load(load_path, env=self.env)
    elif self.algorithm == 'SAC':
      self.model = SAC.load(load_path, env=self.env)
    elif self.algorithm == 'TD3':
      self.model = TD3.load(load_path, env=self.env)

    # Загружаем метаданные
    metadata_path = self.model_path / f"{name}_metadata.pkl"
    if metadata_path.exists():
      metadata = joblib.load(metadata_path)
      self.is_trained = metadata.get('is_trained', True)
      self.config.update(metadata.get('config', {}))

    logger.info(f"Модель загружена: {load_path}")

  def get_training_stats(self) -> Dict[str, Any]:
    """Получение статистики обучения"""
    stats = {
      'algorithm': self.algorithm,
      'is_trained': self.is_trained,
      'total_timesteps': 0,
      'n_episodes': 0,
      'average_reward': 0,
      'std_reward': 0
    }

    if hasattr(self.model, 'num_timesteps'):
      stats['total_timesteps'] = self.model.num_timesteps

    if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
      episode_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
      stats['n_episodes'] = len(episode_rewards)
      stats['average_reward'] = np.mean(episode_rewards)
      stats['std_reward'] = np.std(episode_rewards)

    return stats