import numpy as np
from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces
from utils.logging_config import get_logger

logger = get_logger(__name__)


class SafeEnvironmentWrapper(gym.Env):
  """Обертка для безопасности среды от NaN и других проблем"""

  def __init__(self, env):
    super().__init__()
    self.env = env
    self._last_valid_obs = None

    # Копируем пространства из оригинальной среды
    self.observation_space = env.observation_space
    self.action_space = env.action_space

    # Копируем важные атрибуты
    self.state_space = getattr(env, 'state_space', self.observation_space.shape[0])
    self.stock_dim = getattr(env, 'stock_dim', 1)
    self.spec = getattr(env, 'spec', None)
    self.metadata = getattr(env, 'metadata', {'render_modes': ['human']})

    # Прокидываем другие атрибуты
    for attr in ['reward_function', 'initial_amount', 'day', 'data', 'terminal']:
      if hasattr(env, attr):
        setattr(self, attr, getattr(env, attr))

  def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
    """Безопасный reset совместимый с Gymnasium"""
    # Устанавливаем seed если передан
    if seed is not None:
      super().reset(seed=seed)

    # Вызываем reset оригинальной среды
    result = self.env.reset(seed=seed, options=options)

    # Обрабатываем разные форматы возврата
    if isinstance(result, tuple):
      obs, info = result
    else:
      obs = result
      info = {}

    # Очищаем наблюдение
    obs = self._sanitize_obs(obs)
    self._last_valid_obs = obs.copy() if obs is not None else None

    return obs, info

  def step(self, action):
    """Безопасный step совместимый с Gymnasium"""
    # Проверяем action
    action = self._sanitize_action(action)

    # Выполняем шаг
    try:
      result = self.env.step(action)

      # Обрабатываем разные форматы возврата (4 или 5 значений)
      if len(result) == 4:
        obs, reward, done, info = result
        terminated = done
        truncated = False
      else:
        obs, reward, terminated, truncated, info = result

    except Exception as e:
      logger.error(f"Ошибка в step: {e}")
      # Возвращаем безопасное состояние
      obs = self._last_valid_obs if self._last_valid_obs is not None else np.zeros(self.observation_space.shape)
      reward = -10.0  # Штраф за ошибку
      terminated = True
      truncated = False
      info = {'error': str(e)}

    # Очищаем результаты
    obs = self._sanitize_obs(obs)
    reward = self._sanitize_reward(reward)

    # Сохраняем валидное состояние
    if obs is not None and not np.any(np.isnan(obs)):
      self._last_valid_obs = obs.copy()

    return obs, reward, terminated, truncated, info

  def _sanitize_obs(self, obs):
    """Очищает наблюдение от невалидных значений"""
    if obs is None:
      logger.warning("Observation is None")
      return np.zeros(self.observation_space.shape)

    if isinstance(obs, np.ndarray):
      # Проверяем на NaN
      if np.any(np.isnan(obs)):
        logger.warning(f"NaN в observation, заменяем")
        obs = np.nan_to_num(obs, nan=0.0)

      # Проверяем на Inf
      if np.any(np.isinf(obs)):
        logger.warning(f"Inf в observation, ограничиваем")
        obs = np.clip(obs, -1e10, 1e10)

      # Проверка размерности
      expected_shape = self.observation_space.shape
      if obs.shape != expected_shape:
        logger.error(f"Неверная размерность obs: {obs.shape} != {expected_shape}")
        # Пытаемся исправить
        if obs.size < expected_shape[0]:
          obs = np.pad(obs, (0, expected_shape[0] - obs.size), 'constant')
        else:
          obs = obs[:expected_shape[0]]
        obs = obs.reshape(expected_shape)

    return obs

  def _sanitize_action(self, action):
    """Очищает действия от невалидных значений"""
    if isinstance(action, np.ndarray):
      if np.any(np.isnan(action)) or np.any(np.isinf(action)):
        logger.warning("Invalid action, заменяем на 0")
        action = np.zeros(self.action_space.shape)
      else:
        # Ограничиваем диапазон для Box space
        if isinstance(self.action_space, spaces.Box):
          action = np.clip(action, self.action_space.low, self.action_space.high)
    return action

  def _sanitize_reward(self, reward):
    """Очищает награду от невалидных значений"""
    try:
      reward = float(reward)
      if np.isnan(reward) or np.isinf(reward):
        logger.warning(f"Invalid reward: {reward}, заменяем на 0")
        return 0.0
      # Ограничиваем разумными пределами
      return np.clip(reward, -1000, 1000)
    except:
      logger.error(f"Не могу преобразовать reward: {reward}")
      return 0.0

  def render(self):
    """Прокидываем render"""
    if hasattr(self.env, 'render'):
      return self.env.render()

  def close(self):
    """Прокидываем close"""
    if hasattr(self.env, 'close'):
      return self.env.close()

  def __getattr__(self, name):
    """Прокидываем атрибуты из оригинальной среды"""
    if name.startswith('_'):
      raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    return getattr(self.env, name)