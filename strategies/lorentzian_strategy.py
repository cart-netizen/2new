import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from strategies.base_strategy import BaseStrategy
from core.schemas import TradingSignal
from core.enums import SignalType, Timeframe
from utils.logging_config import get_logger

# Импортируем наши новые компоненты
from ml.lorentzian_indicators import LorentzianIndicators, LorentzianFilters
from ml.enhanced_lorentzian_classifier import EnhancedLorentzianClassifier, create_lorentzian_labels

logger = get_logger(__name__)


class LorentzianStrategy(BaseStrategy):
  """
  Стратегия на основе Machine Learning: Lorentzian Classification
  Полностью соответствует оригинальному TradingView индикатору
  """

  def __init__(self, config: Dict[str, Any]):
    super().__init__(strategy_name="Lorentzian_Classification")

    # Параметры из конфигурации
    self.settings = config.get('strategy_settings', {})

    # Настройки модели (как в оригинале)
    self.neighbors_count = self.settings.get('neighbors_count', 8)
    self.max_bars_back = self.settings.get('max_bars_back', 2000)
    self.feature_count = self.settings.get('feature_count', 5)

    # Настройки признаков (по умолчанию как в оригинале)
    self.feature_configs = {
      'f1': {'type': 'RSI', 'paramA': 14, 'paramB': 1},
      'f2': {'type': 'WT', 'paramA': 10, 'paramB': 11},
      'f3': {'type': 'CCI', 'paramA': 20, 'paramB': 1},
      'f4': {'type': 'ADX', 'paramA': 20, 'paramB': 2},
      'f5': {'type': 'RSI', 'paramA': 9, 'paramB': 1}
    }

    # Переопределяем из конфига если есть
    for i in range(1, 6):
      feature_key = f'feature_{i}'
      if feature_key in self.settings:
        self.feature_configs[f'f{i}'] = self.settings[feature_key]

    # Настройки фильтров
    self.filter_settings = {
      'use_volatility_filter': self.settings.get('use_volatility_filter', True),
      'use_regime_filter': self.settings.get('use_regime_filter', True),
      'use_adx_filter': self.settings.get('use_adx_filter', False),
      'regime_threshold': self.settings.get('regime_threshold', -0.1),
      'adx_threshold': self.settings.get('adx_threshold', 20)
    }

    # Настройки для торговли
    self.use_dynamic_exits = self.settings.get('use_dynamic_exits', False)
    self.show_bar_predictions = self.settings.get('show_bar_predictions', True)

    # Инициализация компонентов
    self.indicators = LorentzianIndicators()
    self.filters = LorentzianFilters()
    self.classifier = None
    self.is_trained = False

    # # История для обучения
    # self.training_history = []
    # self.min_history_size = 500  # Минимум данных для обучения

    # История для обучения - ИЗМЕНЕНО
    self.training_history = []
    # Адаптивный минимум данных в зависимости от таймфрейма
    self.min_history_size = self.settings.get('min_history_size', 200)  # Уменьшили с 500
    self.force_training = self.settings.get('force_training', True)  # Принудительное обучение

    # Кеш для моделей по символам
    self.models_cache = {}  # {symbol: classifier}
    self.training_data_cache = {}  # {symbol: (features, labels)}
    self.data_fetcher = None  # Будет установлен позже

  def set_data_fetcher(self, data_fetcher):
    """Устанавливает data_fetcher после инициализации"""
    self.data_fetcher = data_fetcher

  async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
    """
    Генерация торгового сигнала на основе Lorentzian Classification
    """
    try:
      # # Минимальная проверка данных
      # if len(data) < 50:
      #   return None

      # Проверяем кеш моделей для символа
      if len(data) < self.min_history_size and self.data_fetcher:
        logger.info(f"Загружаем дополнительные данные для {symbol}: текущее количество {len(data)}")
        try:
          # Загружаем больше исторических данных
          extended_data = await self.data_fetcher.get_historical_candles(
            symbol,
            Timeframe.FIVE_MINUTES,
            limit=2000  # Загружаем 1000 свечей
          )

          if not extended_data.empty and len(extended_data) > len(data):
            data = extended_data
            logger.info(f"Загружено {len(data)} свечей для {symbol}")
        except Exception as e:
          logger.error(f"Ошибка загрузки дополнительных данных: {e}")

      # Проверяем минимум после попытки загрузки
      if len(data) < 100:  # Абсолютный минимум
        logger.warning(f"Недостаточно данных для {symbol}: {len(data)} < 100")
        return None

      # 1. Рассчитываем признаки
      features = self.indicators.calculate_features(
        data,
        self.feature_configs['f1'],
        self.feature_configs['f2'],
        self.feature_configs['f3'],
        self.feature_configs['f4'],
        self.feature_configs['f5']
      )

      if features.empty or len(features) < 10:
        logger.warning(f"Недостаточно данных для расчета признаков {symbol}")
        return None

      # 2. Обучаем модель если нужно
      if not self.is_trained:
        if not await self._train_model(data, features):
          return None

      # 3. Получаем последние признаки для предсказания
      last_features = features.tail(1)

      # 4. Делаем предсказание
      prediction = self.classifier.predict(last_features)[0]
      prediction_proba = self.classifier.predict_proba(last_features)[0]

      # 5. Применяем фильтры
      filter_results = self._apply_filters(data)

      # Если фильтры не пройдены - нет сигнала
      if not all(filter_results.values()):
        logger.debug(f"Сигнал отфильтрован для {symbol}: {filter_results}")
        return None

      # 6. Преобразуем предсказание в сигнал
      signal_type = None
      confidence = 0.0

      if prediction == 1:  # BUY
        signal_type = SignalType.BUY
        confidence = prediction_proba[1]
      elif prediction == 2:  # SELL
        signal_type = SignalType.SELL
        confidence = prediction_proba[2]
      else:  # HOLD
        return None

      # 7. Рассчитываем параметры позиции
      current_price = float(data['close'].iloc[-1])
      atr = float(data['atr'].iloc[-1]) if 'atr' in data.columns else current_price * 0.02

      # Stop Loss и Take Profit
      if signal_type == SignalType.BUY:
        stop_loss = current_price - 2 * atr
        take_profit = current_price + 3 * atr
      else:  # SELL
        stop_loss = current_price + 2 * atr
        take_profit = current_price - 3 * atr

      # 8. Создаем сигнал
      signal = TradingSignal(
        symbol=symbol,
        signal_type=signal_type,
        price=current_price,
        confidence=confidence,
        stop_loss=stop_loss,
        take_profit=take_profit,
        strategy_name=self.strategy_name,
        timestamp=datetime.now(timezone.utc),
        metadata={
          'prediction_raw': int(prediction),
          'probabilities': {
            'hold': float(prediction_proba[0]),
            'buy': float(prediction_proba[1]),
            'sell': float(prediction_proba[2])
          },
          'filters_passed': filter_results,
          'features': {
            'f1': float(last_features['f1'].iloc[0]),
            'f2': float(last_features['f2'].iloc[0]),
            'f3': float(last_features['f3'].iloc[0]),
            'f4': float(last_features['f4'].iloc[0]),
            'f5': float(last_features['f5'].iloc[0])
          }
        }
      )

      logger.info(f"🎯 Lorentzian сигнал для {symbol}: {signal_type.value} с уверенностью {confidence:.3f}")

      return signal

    except Exception as e:
      logger.error(f"Ошибка генерации Lorentzian сигнала для {symbol}: {e}", exc_info=True)
      return None

  async def _train_model(self, data: pd.DataFrame, features: pd.DataFrame) -> bool:
    """
    Обучение модели на исторических данных
    """
    try:
      logger.info("Начинаем обучение Lorentzian модели...")

      # Создаем метки для обучения (смотрим на 4 бара вперед как в оригинале)
      labels = create_lorentzian_labels(data, future_bars=4, threshold_percent=0.0)

      # Выравниваем индексы
      common_index = features.index.intersection(labels.index)
      features_train = features.loc[common_index]
      labels_train = labels.loc[common_index]

      if len(features_train) < self.min_history_size:
        if self.force_training and len(features_train) >= 100:
          logger.warning(f"Принудительное обучение с {len(features_train)} примерами")
        else:
          logger.warning(f"Недостаточно данных для обучения: {len(features_train)} < {self.min_history_size}")
          return False

      # Создаем и обучаем классификатор
      self.classifier = EnhancedLorentzianClassifier(
        k_neighbors=self.neighbors_count,
        max_bars_back=self.max_bars_back,
        feature_count=self.feature_count,
        use_dynamic_exits=self.use_dynamic_exits,
        filters=self.filter_settings
      )

      self.classifier.fit(features_train, labels_train)
      self.is_trained = True

      # Оценка на обучающей выборке
      train_predictions = self.classifier.predict(features_train.tail(100))
      train_labels = labels_train.tail(100).values
      accuracy = np.mean(train_predictions == train_labels)

      logger.info(f"✅ Модель обучена. Точность на последних 100 примерах: {accuracy:.3f}")

      return True

    except Exception as e:
      logger.error(f"Ошибка при обучении модели: {e}", exc_info=True)
      return False

  def _apply_filters(self, data: pd.DataFrame) -> Dict[str, bool]:
    """
    Применение фильтров из оригинального индикатора
    """
    results = {}

    # Volatility Filter
    if self.filter_settings['use_volatility_filter']:
      volatility_pass = self.filters.volatility_filter(data)
      results['volatility'] = bool(volatility_pass.iloc[-1]) if not volatility_pass.empty else True
    else:
      results['volatility'] = True

    # Regime Filter
    if self.filter_settings['use_regime_filter']:
      regime_pass = self.filters.regime_filter(data, self.filter_settings['regime_threshold'])
      results['regime'] = bool(regime_pass.iloc[-1]) if not regime_pass.empty else True
    else:
      results['regime'] = True

    # ADX Filter
    if self.filter_settings['use_adx_filter']:
      adx_pass = self.filters.adx_filter(data, self.filter_settings['adx_threshold'])
      results['adx'] = bool(adx_pass.iloc[-1]) if not adx_pass.empty else True
    else:
      results['adx'] = True

    return results

  def update_model(self, symbol: str, new_data: pd.DataFrame, actual_outcome: int):
    """
    Полноценное инкрементальное обучение модели

    Args:
        symbol: Торговый символ
        new_data: DataFrame с новым баром (OHLCV)
        actual_outcome: Фактический результат (-1=SELL, 0=HOLD, 1=BUY)
    """
    if symbol not in self.models_cache:
      logger.warning(f"Модель для {symbol} не найдена в кеше")
      return

    try:
      # Рассчитываем признаки для нового бара
      new_features = self.indicators.calculate_features(
        new_data,
        self.feature_configs['f1'],
        self.feature_configs['f2'],
        self.feature_configs['f3'],
        self.feature_configs['f4'],
        self.feature_configs['f5']
      )

      if new_features.empty:
        return

      # Получаем вектор признаков
      feature_vector = new_features.iloc[-1].values

      # Инкрементально обновляем модель
      classifier = self.models_cache[symbol]
      classifier.incremental_update(feature_vector, actual_outcome, symbol)

      # Обновляем кеш данных
      if symbol not in self.training_data_cache:
        self.training_data_cache[symbol] = {
          'features': [],
          'labels': [],
          'update_count': 0,
          'last_accuracy': 0.0
        }

      cache = self.training_data_cache[symbol]
      cache['features'].append(feature_vector)
      cache['labels'].append(actual_outcome)
      cache['update_count'] += 1

      # Ограничиваем размер кеша
      if len(cache['features']) > self.max_bars_back:
        cache['features'].pop(0)
        cache['labels'].pop(0)

      # Периодическая оценка качества
      if cache['update_count'] % 20 == 0:
        self._evaluate_model_performance(symbol)

      # Сохранение модели каждые 100 обновлений
      if cache['update_count'] % 100 == 0 and self.settings.get('save_models', True):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"ml_models/lorentzian_{symbol}_{timestamp}.pkl"
        classifier.save_model(model_path)
        logger.info(f"Модель {symbol} сохранена после {cache['update_count']} обновлений")

      logger.debug(f"Модель {symbol} обновлена. Всего обновлений: {cache['update_count']}")

    except Exception as e:
      logger.error(f"Ошибка инкрементального обновления для {symbol}: {e}", exc_info=True)

  def _evaluate_model_performance(self, symbol: str):
    """Оценивает текущую производительность модели"""
    try:
      cache = self.training_data_cache.get(symbol)
      if not cache or len(cache['features']) < 50:
        return

      classifier = self.models_cache.get(symbol)
      if not classifier:
        return

      # Берем последние 50 примеров для оценки
      test_features = np.array(cache['features'][-50:])
      test_labels = np.array(cache['labels'][-50:])

      # Создаем DataFrame для предсказания
      feature_names = [f'f{i + 1}' for i in range(self.feature_count)]
      test_df = pd.DataFrame(test_features, columns=feature_names)

      # Получаем предсказания
      predictions = classifier.predict(test_df)

      # Считаем точность
      accuracy = np.mean(predictions == test_labels)

      # Сравниваем с предыдущей точностью
      improvement = accuracy - cache['last_accuracy']
      cache['last_accuracy'] = accuracy

      logger.info(f"Производительность {symbol}: точность={accuracy:.3f}, "
                  f"изменение={improvement:+.3f}")

      # Если производительность сильно упала, можем переобучить модель
      if improvement < -0.1 and len(cache['features']) >= self.min_history_size:
        logger.warning(f"Значительное падение производительности {symbol}. "
                       f"Рекомендуется полное переобучение")

    except Exception as e:
      logger.error(f"Ошибка оценки производительности {symbol}: {e}")

  def _update_nearest_neighbors(self, symbol: str, new_feature_vector: np.ndarray, new_label: int):
    """
    Быстрое обновление ближайших соседей без полного переобучения
    Эмулирует поведение индикатора в TradingView
    """
    if symbol not in self.models_cache:
      return

    classifier = self.models_cache[symbol]

    # Добавляем новую точку в обучающие данные
    # Это упрощенная версия - в реальности нужно обновлять
    # внутренние структуры EnhancedLorentzianClassifier

    # Временное решение - помечаем модель для обновления
    if not hasattr(classifier, 'pending_updates'):
      classifier.pending_updates = []

    classifier.pending_updates.append((new_feature_vector, new_label))

    # Применяем обновления при следующем predict
    if len(classifier.pending_updates) > 10:
      logger.debug(f"Накоплено {len(classifier.pending_updates)} обновлений для {symbol}")