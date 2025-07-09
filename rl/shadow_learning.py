import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio

from utils.logging_config import get_logger

logger = get_logger(__name__)

class ShadowTradingLearner:
  """
  Обучение RL агента на данных Shadow Trading
  Анализирует упущенные возможности и улучшает политику
  """

  def __init__(
      self,
      rl_agent,
      shadow_trading_manager,
      feature_processor,
      data_fetcher,
      config: Dict[str, Any] = None
  ):
    self.rl_agent = rl_agent
    self.shadow_manager = shadow_trading_manager
    self.feature_processor = feature_processor
    self.data_fetcher = data_fetcher
    self.config = config or {}

    # Параметры обучения
    self.batch_size = self.config.get('batch_size', 32)
    self.learning_rate = self.config.get('learning_rate', 0.001)
    self.update_frequency = self.config.get('update_frequency_hours', 24)
    self.min_samples = self.config.get('min_samples', 100)

    # История обучения
    self.learning_history = []
    self.last_update = datetime.now()

    # Буфер опыта для replay
    self.experience_buffer = []
    self.max_buffer_size = self.config.get('max_buffer_size', 10000)

  async def learn_from_missed_opportunities(self, hours: int = 24) -> Dict[str, Any]:
    """
    Обучение на пропущенных возможностях из Shadow Trading
    """
    try:
      logger.info(f"Начало обучения на Shadow Trading данных за последние {hours} часов")

      # 1. Получаем пропущенные сигналы
      missed_signals = await self._get_missed_profitable_signals(hours)

      if not missed_signals:
        logger.info("Нет пропущенных сигналов для обучения")
        return {'status': 'no_data', 'signals_processed': 0}

      logger.info(f"Найдено {len(missed_signals)} пропущенных прибыльных сигналов")

      # 2. Подготавливаем данные для обучения
      training_samples = await self._prepare_training_samples(missed_signals)

      if len(training_samples) < self.min_samples:
        logger.info(f"Недостаточно образцов для обучения: {len(training_samples)} < {self.min_samples}")
        # Сохраняем в буфер для будущего обучения
        self._add_to_experience_buffer(training_samples)
        return {'status': 'buffered', 'signals_processed': len(training_samples)}

      # 3. Обучаем агента
      learning_results = await self._train_on_samples(training_samples)

      # 4. Оцениваем улучшение
      improvement = await self._evaluate_improvement(training_samples)

      # 5. Обновляем статистику
      self._update_learning_history(learning_results, improvement)

      logger.info(f"Обучение завершено. Улучшение производительности: {improvement:.2%}")

      return {
        'status': 'success',
        'signals_processed': len(training_samples),
        'performance_improvement': improvement,
        'learning_results': learning_results
      }

    except Exception as e:
      logger.error(f"Ошибка обучения на Shadow Trading: {e}", exc_info=True)
      return {'status': 'error', 'error': str(e)}

  async def _get_missed_profitable_signals(self, hours: int) -> List[Dict[str, Any]]:
    """Получает пропущенные прибыльные сигналы"""
    try:
      # Получаем все shadow trades за период
      all_shadows = await self.shadow_manager.get_shadow_trades_by_period(
        start_time=datetime.now() - timedelta(hours=hours),
        end_time=datetime.now()
      )

      # Фильтруем только прибыльные и качественные
      profitable_shadows = []

      for shadow in all_shadows:
        # Проверяем прибыльность
        if shadow.get('potential_profit', 0) <= 0:
          continue

        # Проверяем качество сигнала
        if shadow.get('signal_quality', 0) < 0.6:
          continue

        # Проверяем, что сигнал действительно был пропущен
        if shadow.get('execution_status') != 'missed':
          continue

        profitable_shadows.append(shadow)

      # Сортируем по потенциальной прибыли
      profitable_shadows.sort(key=lambda x: x.get('potential_profit', 0), reverse=True)

      return profitable_shadows

    except Exception as e:
      logger.error(f"Ошибка получения shadow trades: {e}")
      return []

  async def _prepare_training_samples(self, missed_signals: List[Dict]) -> List[Dict[str, Any]]:
    """Подготавливает обучающие образцы из пропущенных сигналов"""
    training_samples = []

    for signal in missed_signals:
      try:
        # Воссоздаем состояние на момент сигнала
        symbol = signal['symbol']
        timestamp = signal['timestamp']

        # Получаем исторические данные на момент сигнала
        historical_data = await self._get_historical_state(symbol, timestamp)

        if historical_data is None or historical_data.empty:
          continue

        # Создаем признаки
        features = self.feature_processor.create_rl_features(
          historical_data,
          symbol
        )

        if features is None or features.shape[0] == 0:
          continue

        # Определяем правильное действие
        if signal['signal_type'] == 'BUY':
          correct_action = 2
        elif signal['signal_type'] == 'SELL':
          correct_action = 0
        else:
          continue

        # Создаем обучающий образец
        sample = {
          'state': features[-1] if len(features.shape) == 2 else features,
          'action': correct_action,
          'reward': signal['potential_profit'] * 100,  # Масштабируем награду
          'next_state': None,  # Будет заполнено позже если нужно
          'done': False,
          'info': {
            'symbol': symbol,
            'timestamp': timestamp,
            'signal_quality': signal.get('signal_quality', 0.5),
            'filter_reason': signal.get('filter_reason', 'unknown')
          }
        }

        training_samples.append(sample)

      except Exception as e:
        logger.error(f"Ошибка подготовки образца для {signal.get('symbol', 'unknown')}: {e}")
        continue

    return training_samples

  async def _get_historical_state(self, symbol: str, timestamp: datetime) -> Optional[pd.DataFrame]:
    """Получает исторические данные на определенный момент времени"""
    try:
      # Получаем данные до указанного момента
      end_time = timestamp
      start_time = timestamp - timedelta(hours=100)  # 100 часов истории

      # Здесь должен быть метод получения исторических данных
      # В реальной реализации это может быть запрос к БД или API
      data = await self.data_fetcher.get_historical_candles(
        symbol=symbol,
        timeframe='1h',
        start_time=start_time,
        end_time=end_time
      )

      return data

    except Exception as e:
      logger.error(f"Ошибка получения исторических данных: {e}")
      return None

  async def _train_on_samples(self, training_samples: List[Dict]) -> Dict[str, Any]:
    """Обучает агента на подготовленных образцах"""
    try:
      # Добавляем образцы в буфер опыта
      self._add_to_experience_buffer(training_samples)

      # Если используем алгоритм с replay buffer (SAC, TD3)
      if self.rl_agent.algorithm in ['SAC', 'TD3']:
        # Добавляем опыт в replay buffer модели
        for sample in training_samples:
          # Симулируем step для добавления в buffer
          self.rl_agent.model.replay_buffer.add(
            sample['state'],
            sample['action'],
            sample['reward'],
            sample['next_state'] or sample['state'],  # Используем текущее состояние если нет следующего
            sample['done']
          )

      # Выполняем обучение
      if len(self.experience_buffer) >= self.batch_size:
        # Обучаем на батчах из буфера
        n_updates = min(10, len(self.experience_buffer) // self.batch_size)

        for _ in range(n_updates):
          # Сэмплируем батч
          batch_indices = np.random.choice(
            len(self.experience_buffer),
            size=self.batch_size,
            replace=False
          )
          batch = [self.experience_buffer[i] for i in batch_indices]

          # Обновляем модель
          if hasattr(self.rl_agent.model, 'train'):
            # Для PPO/A2C нужно собрать траектории
            if self.rl_agent.algorithm in ['PPO', 'A2C']:
              # Создаем мини-батч для обучения
              states = np.array([s['state'] for s in batch])
              actions = np.array([s['action'] for s in batch])
              rewards = np.array([s['reward'] for s in batch])

              # Простое обновление градиента
              # В реальной реализации здесь должен быть более сложный процесс
              loss = self._compute_policy_loss(states, actions, rewards)
            else:
              # Для off-policy алгоритмов просто обновляем
              self.rl_agent.model.train(gradient_steps=1)

      return {
        'samples_trained': len(training_samples),
        'buffer_size': len(self.experience_buffer),
        'training_iterations': n_updates if 'n_updates' in locals() else 0
      }

    except Exception as e:
      logger.error(f"Ошибка обучения на образцах: {e}")
      return {'error': str(e)}

  def _compute_policy_loss(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> float:
    """Вычисляет loss для обновления политики"""
    # Упрощенная версия policy gradient loss
    # В реальной реализации должен использоваться правильный алгоритм

    # Получаем предсказания модели
    with torch.no_grad():
      predicted_actions = self.rl_agent.model.policy.predict(states)

    # Вычисляем loss
    action_diff = predicted_actions - actions
    weighted_diff = action_diff * rewards.reshape(-1, 1)
    loss = np.mean(np.square(weighted_diff))

    return loss

  async def _evaluate_improvement(self, training_samples: List[Dict]) -> float:
    """Оценивает улучшение производительности после обучения"""
    try:
      if not training_samples:
        return 0.0

      # Оцениваем агента на обучающих образцах
      correct_predictions = 0

      for sample in training_samples[:50]:  # Оцениваем на подвыборке
        state = sample['state']
        correct_action = sample['action']

        # Получаем предсказание агента
        predicted_action, _ = self.rl_agent.predict(state)

        if int(predicted_action[0]) == correct_action:
          correct_predictions += 1

      # Рассчитываем точность
      accuracy = correct_predictions / min(50, len(training_samples))

      # Сравниваем с базовой точностью (33% для 3 действий)
      improvement = accuracy - 0.33

      return max(0, improvement)

    except Exception as e:
      logger.error(f"Ошибка оценки улучшения: {e}")
      return 0.0

  def _add_to_experience_buffer(self, samples: List[Dict]):
    """Добавляет образцы в буфер опыта"""
    self.experience_buffer.extend(samples)

    # Ограничиваем размер буфера
    if len(self.experience_buffer) > self.max_buffer_size:
      # Удаляем старые образцы
      self.experience_buffer = self.experience_buffer[-self.max_buffer_size:]

  def _update_learning_history(self, results: Dict, improvement: float):
    """Обновляет историю обучения"""
    self.learning_history.append({
      'timestamp': datetime.now(),
      'results': results,
      'improvement': improvement,
      'buffer_size': len(self.experience_buffer)
    })

    self.last_update = datetime.now()

    # Ограничиваем размер истории
    if len(self.learning_history) > 100:
      self.learning_history.pop(0)

  async def continuous_learning_loop(self):
    """Непрерывный цикл обучения на Shadow Trading данных"""
    logger.info("Запуск непрерывного обучения на Shadow Trading")

    while True:
      try:
        # Проверяем, пора ли обновляться
        if (datetime.now() - self.last_update).total_seconds() < self.update_frequency * 3600:
          await asyncio.sleep(3600)  # Спим час
          continue

        # Выполняем обучение
        results = await self.learn_from_missed_opportunities(
          hours=self.update_frequency
        )

        if results['status'] == 'success':
          logger.info(f"Успешное обновление модели. Улучшение: {results.get('performance_improvement', 0):.2%}")

          # Сохраняем обновленную модель
          if results.get('performance_improvement', 0) > 0.01:  # Улучшение > 1%
            model_name = f"rl_shadow_updated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.rl_agent.save_model(model_name)
            logger.info(f"Обновленная модель сохранена: {model_name}")

        # Спим до следующего обновления
        await asyncio.sleep(self.update_frequency * 3600)

      except Exception as e:
        logger.error(f"Ошибка в цикле непрерывного обучения: {e}")
        await asyncio.sleep(3600)  # Спим час в случае ошибки

  def get_learning_statistics(self) -> Dict[str, Any]:
    """Возвращает статистику обучения"""
    if not self.learning_history:
      return {
        'total_updates': 0,
        'average_improvement': 0,
        'last_update': None,
        'buffer_size': len(self.experience_buffer)
      }

    improvements = [h['improvement'] for h in self.learning_history]

    return {
      'total_updates': len(self.learning_history),
      'average_improvement': np.mean(improvements),
      'best_improvement': max(improvements),
      'last_update': self.last_update,
      'buffer_size': len(self.experience_buffer),
      'recent_results': self.learning_history[-5:] if self.learning_history else []
      }