import asyncio
import shutil

import joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timezone
from ml.wavetrend_3d import WaveTrend3D

from strategies.base_strategy import BaseStrategy
from core.schemas import TradingSignal
from core.enums import SignalType, Timeframe
from utils.logging_config import get_logger

# Импортируем наши новые компоненты
from ml.lorentzian_indicators import LorentzianIndicators, LorentzianFilters
from ml.enhanced_lorentzian_classifier import EnhancedLorentzianClassifier, create_lorentzian_labels
import pickle
import os
from functools import lru_cache
from pathlib import Path  # Добавь этот импорт
import joblib
import pickle
import os
from functools import lru_cache

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
    self.neighbors_count = self.settings.get('neighbors_count', 12)
    self.max_bars_back = self.settings.get('max_bars_back', 5000)
    self.feature_count = self.settings.get('feature_count', 5)

    # Настройки признаков (по умолчанию как в оригинале)
    self.feature_configs = {
      'f1': {'type': 'RSI', 'paramA': 7, 'paramB': 1},
      'f2': {'type': 'WT', 'paramA': 5, 'paramB': 7},
      'f3': {'type': 'CCI', 'paramA': 10, 'paramB': 1},
      'f4': {'type': 'ADX', 'paramA': 10, 'paramB': 2},
      'f5': {'type': 'RSI', 'paramA': 5, 'paramB': 1}
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

    self.use_wavetrend_3d = self.settings.get('use_wavetrend_3d', True)
    self.wavetrend_3d_config = self.settings.get('wavetrend_3d', {})

    # Инициализируем WaveTrend 3D если включен
    if self.use_wavetrend_3d:
      self.wavetrend_3d = WaveTrend3D(self.wavetrend_3d_config)
      logger.info("✅ WaveTrend 3D интегрирован в Lorentzian стратегию")
    else:
      self.wavetrend_3d = None
    # Настройки для торговли
    self.use_dynamic_exits = self.settings.get('use_dynamic_exits', False)
    self.show_bar_predictions = self.settings.get('show_bar_predictions', True)

    # Инициализация компонентов
    self.indicators = LorentzianIndicators()
    self.filters = LorentzianFilters()
    self.classifier = None
    self.is_trained = False

    # # История для обучения

    # self.min_history_size = 500  # Минимум данных для обучения

    # История для обучения - ИЗМЕНЕНО
    self.training_history = {}
    # Адаптивный минимум данных в зависимости от таймфрейма
    self.min_history_size = self.settings.get('min_history_size', 5000)  # Уменьшили с 500
    self.min_initial_training_size = 5000
    self.force_training = self.settings.get('force_training', False)  # Принудительное обучение

    # Кеш для моделей по символам
    self.models_cache = {}  # {symbol: classifier}
    self.training_data_cache = {}  # {symbol: (features, labels)}
    self.data_fetcher = None  # Будет установлен позже
    self.performance_optimization = config.get('advanced_settings', {}).get('performance_optimization', {})
    self.use_numba = self.performance_optimization.get('use_numba', True)
    self.cache_predictions = self.performance_optimization.get('cache_predictions', True)
    self.cache_ttl_seconds = self.performance_optimization.get('cache_ttl_seconds', 60)
    self.min_trading_signals_ratio = self.settings.get('min_trading_signals_ratio', 0.15)
    self.quality_threshold = self.settings.get('quality_threshold', 0.55)
    self.trading_accuracy_threshold = self.settings.get('trading_accuracy_threshold', 0.5)
    self.auto_retrain_on_poor_quality = self.settings.get('auto_retrain_on_poor_quality', True)
    # Директория для сохранения моделей
    # self.models_dir = "ml_models/lorentzian"
    # os.makedirs(self.models_dir, exist_ok=True)

    self.models_dir = Path("ml_models/lorentzian")
    self.models_dir.mkdir(parents=True, exist_ok=True)

    self._load_existing_models()

  def _load_existing_models(self):
    """Загружает существующие модели при старте"""
    try:
      # Убеждаемся что директория существует
      if not self.models_dir.exists():
        logger.info("Директория моделей не существует, создаем...")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        return

      # Ищем файлы моделей
      model_files = list(self.models_dir.glob("lorentzian_*.pkl"))

      if not model_files:
        logger.info("Предобученные модели не найдены, будет выполнено обучение с нуля")
        return

      loaded_count = 0
      for model_file in model_files:
        try:
          # Извлекаем символ из имени файла
          # Формат: lorentzian_BTCUSDT_20240804_123456.pkl или lorentzian_BTCUSDT_latest.pkl
          filename = model_file.stem  # Убираем .pkl
          parts = filename.split("_")

          if len(parts) >= 2:
            symbol = parts[1]  # BTCUSDT

            # Загружаем модель
            logger.debug(f"Попытка загрузки модели для {symbol} из {model_file.name}")

            # Создаем пустой классификатор
            classifier = EnhancedLorentzianClassifier(
              k_neighbors=self.neighbors_count,
              max_bars_back=self.max_bars_back,
              feature_count=self.feature_count,
              use_dynamic_exits=self.use_dynamic_exits,
              filters=self.filter_settings
            )

            # Загружаем состояние модели
            if classifier.load_model(str(model_file)):
              self.models_cache[symbol] = classifier
              loaded_count += 1
              logger.info(f"✅ Загружена модель для {symbol} из {model_file.name}")
            else:
              logger.warning(f"⚠️ Не удалось загрузить модель из {model_file.name}")

        except Exception as e:
          logger.warning(f"Не удалось загрузить модель из {model_file}: {e}")
          continue

      if loaded_count > 0:
        logger.info(f"✅ Загружено {loaded_count} предобученных моделей")
      else:
        logger.info("Предобученные модели не найдены, будет выполнено обучение с нуля")

    except Exception as e:
      logger.error(f"Ошибка загрузки моделей: {e}")

  def _save_model(self, symbol: str, classifier):
    """Сохраняет модель для символа"""
    try:
      timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
      model_path = self.models_dir / f"lorentzian_{symbol}_{timestamp}.pkl"

      # Создаем резервную копию старой модели
      old_models = list(self.models_dir.glob(f"lorentzian_{symbol}_*.pkl"))
      if old_models:
        old_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        for old_model in old_models[2:]:
          old_model.unlink()
          logger.debug(f"Удалена старая модель: {old_model.name}")

      # # Сохраняем новую модель
      # with open(model_path, 'wb') as f:
      #   joblib.dump(classifier, f)

      # Сохраняем новую модель используя встроенный метод
      classifier.save_model(str(model_path))

      # --- ИСПРАВЛЕНИЕ: Заменяем символическую ссылку на простое копирование ---
      # Этот метод не требует прав администратора в Windows.
      latest_path = self.models_dir / f"lorentzian_{symbol}_latest.pkl"
      try:
          shutil.copy(model_path, latest_path)
          logger.info(f"✅ Модель {symbol} сохранена: {model_path.name} и обновлена копия 'latest'")
      except Exception as copy_error:
          logger.error(f"Не удалось создать копию 'latest' для модели {symbol}: {copy_error}")
      # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

      return True

    except Exception as e:
      logger.error(f"Ошибка сохранения модели {symbol}: {e}")
      return False

  def set_data_fetcher(self, data_fetcher):
    """Устанавливает data_fetcher после инициализации"""
    self.data_fetcher = data_fetcher

  async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
      """
      Генерация торгового сигнала на основе УЖЕ ОБУЧЕННОЙ Lorentzian Classification
      """

      # 1. Проверяем, готова ли модель. Если нет - сигнал не генерируем.
      if symbol not in self.models_cache or not self.models_cache[symbol].is_fitted:
          logger.warning(f"⏳ {symbol}: Модель еще не обучена. Пропуск генерации сигнала.")
          # Можно добавить логику для запуска preload_training_data, если это нужно
          # await self.preload_training_data(symbol)
          return None

      if len(data) < 100:
          logger.warning(f"Недостаточно актуальных данных для {symbol}: {len(data)} < 100")
          return None

      # Используем готовую модель
      self.classifier = self.models_cache[symbol]

      # 2. Рассчитываем признаки для ПОСЛЕДНИХ данных
      features = self.indicators.calculate_features(
          data,
          self.feature_configs['f1'], self.feature_configs['f2'], self.feature_configs['f3'],
          self.feature_configs['f4'], self.feature_configs['f5']
      )

      if features.empty:
          logger.warning(f"Не удалось рассчитать признаки для актуальных данных {symbol}")
          return None

      # 3. Получаем последние признаки для предсказания
      last_features = features.tail(1)

      # 4. Делаем предсказание
      prediction = self.classifier.predict(last_features)[0]
      prediction_proba = self.classifier.predict_proba(last_features)[0]

      # 5. Применяем фильтры
      filter_results = self._apply_filters(data, symbol)

      # Если фильтры не пройдены - нет сигнала
      if not all(filter_results.values()):
        logger.debug(f"Сигнал отфильтрован для {symbol}: {filter_results}")
        return None

      # 6. Преобразуем предсказание в сигнал
      signal_type = None
      confidence = 0.0

      # 6a. Определяем базовые параметры сигнала
      if prediction == 1:  # BUY
        signal_type = SignalType.BUY
        confidence = prediction_proba[1]
      elif prediction == 2:  # SELL
        signal_type = SignalType.SELL
        confidence = prediction_proba[2]
      else:  # HOLD
        return None

      # 6b. Применяем WaveTrend 3D для подтверждения и корректировки
      if self.use_wavetrend_3d and 'wavetrend_direction' in filter_results:
        wt_direction = filter_results['wavetrend_direction']
        wt_confidence = filter_results['wavetrend_confidence']

        # Логирование с учетом типа сигнала
        if (signal_type == SignalType.BUY and wt_direction > 0) or (
            signal_type == SignalType.SELL and wt_direction < 0):
          logger.info(f"✅ WaveTrend 3D подтверждает Lorentzian для {symbol}")
          confidence = min(0.95, confidence + wt_confidence * 0.15)
        elif (signal_type == SignalType.BUY and wt_direction < 0) or (
            signal_type == SignalType.SELL and wt_direction > 0):
          logger.warning(f"⚠️ WaveTrend 3D противоречит Lorentzian для {symbol}")
          if wt_confidence > 0.7:  # Сильное противоречие
            return None
          else:
            confidence = max(0.3, confidence - wt_confidence * 0.2)

        # Проверяем на дивергенции
        wavetrend_result = self.wavetrend_3d.calculate(data, symbol)
        if wavetrend_result and wavetrend_result.get('divergences'):
          divergences = wavetrend_result['divergences']
          if divergences.get('bullish') or divergences.get('bearish'):
            divergence_type = 'bullish_divergence' if divergences.get('bullish') else 'bearish_divergence'
            logger.info(f"🎯 WaveTrend дивергенция обнаружена: {divergence_type}")


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
          'wavetrend_3d': {
            'used': self.use_wavetrend_3d,
            'direction': filter_results.get('wavetrend_direction', 0),
            'confidence': filter_results.get('wavetrend_confidence', 0),
            'passed': filter_results.get('wavetrend_3d', True)
          },
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
      if len(self.training_history) > 0 and (len(self.training_history) % 10 == 0):
        self.log_accumulation_summary()

      return signal



  def analyze_signal_alignment(self, lorentzian_signal: int, wavetrend_data: Dict) -> Dict[str, Any]:
    """
    Анализирует согласованность сигналов Lorentzian и WaveTrend 3D

    Returns:
        Dict с анализом согласованности и рекомендациями
    """
    wt_direction = wavetrend_data.get('direction', 0)
    wt_confidence = wavetrend_data.get('confidence', 0.5)

    # Полное согласование
    if (lorentzian_signal == 1 and wt_direction > 0) or \
        (lorentzian_signal == 2 and wt_direction < 0):
      return {
        'aligned': True,
        'strength': 'strong',
        'confidence_multiplier': 1.0 + wt_confidence * 0.3,
        'recommendation': 'proceed'
      }

    # Противоречие
    elif (lorentzian_signal == 1 and wt_direction < 0) or \
        (lorentzian_signal == 2 and wt_direction > 0):
      return {
        'aligned': False,
        'strength': 'conflict',
        'confidence_multiplier': 1.0 - wt_confidence * 0.4,
        'recommendation': 'caution' if wt_confidence < 0.7 else 'skip'
      }

    # Нейтральный WaveTrend
    else:
      return {
        'aligned': None,
        'strength': 'neutral',
        'confidence_multiplier': 1.0,
        'recommendation': 'proceed_normal'
      }

  # async def _train_model(self, data: pd.DataFrame, features: pd.DataFrame) -> bool:
  #   """
  #   Обучение модели на исторических данных
  #   """
  #   try:
  #     logger.info("Начинаем обучение Lorentzian модели...")
  #
  #     # Создаем метки для обучения (смотрим на 4 бара вперед как в оригинале)
  #     labels = create_lorentzian_labels(data, future_bars=4, threshold_percent=0.0)
  #
  #     # Выравниваем индексы
  #     common_index = features.index.intersection(labels.index)
  #     features_train = features.loc[common_index]
  #     labels_train = labels.loc[common_index]
  #
  #     if len(features_train) < self.min_history_size:
  #       if self.force_training and len(features_train) >= 100:
  #         logger.warning(f"Принудительное обучение с {len(features_train)} примерами")
  #       else:
  #         logger.warning(f"Недостаточно данных для обучения: {len(features_train)} < {self.min_history_size}")
  #         return False
  #
  #     # Создаем и обучаем классификатор
  #     self.classifier = EnhancedLorentzianClassifier(
  #       k_neighbors=self.neighbors_count,
  #       max_bars_back=self.max_bars_back,
  #       feature_count=self.feature_count,
  #       use_dynamic_exits=self.use_dynamic_exits,
  #       filters=self.filter_settings
  #     )
  #
  #     self.classifier.fit(features_train, labels_train)
  #     self.is_trained = True
  #
  #     # Оценка на обучающей выборке
  #     train_predictions = self.classifier.predict(features_train.tail(300))
  #     train_labels = labels_train.tail(300).values
  #     accuracy = np.mean(train_predictions == train_labels)
  #
  #     logger.info(f"✅ Модель обучена. Точность на последних 300 примерах: {accuracy:.3f}")
  #
  #     return True
  #
  #   except Exception as e:
  #     logger.error(f"Ошибка при обучении модели: {e}", exc_info=True)
  #     return False

  async def _train_model(self, symbol: str, data: pd.DataFrame, features: pd.DataFrame) -> bool:
      """
      Обучение модели на исторических данных с накоплением
      """
      try:
        # Сначала проверяем, есть ли уже обученная модель для этого символа
        if symbol in self.models_cache and self.models_cache[symbol].is_fitted:
          logger.info(f"✅ {symbol}: модель уже обучена, используем существующую")
          return True

        logger.info(f"🎯 {symbol}: начинаем процесс обучения модели...")

        # УПРОЩЕННАЯ ЛОГИКА: если данных много - сразу обучаем
        if len(data) >= self.min_initial_training_size and len(features) >= self.min_initial_training_size:
          logger.info(f"🚀 {symbol}: достаточно данных ({len(data)}>={self.min_initial_training_size}), "
                      f"пропускаем накопление и сразу обучаем!")

          # Берем последние данные для обучения
          training_data = data.tail(self.min_initial_training_size).copy()
          training_features = features.tail(self.min_initial_training_size).copy()

          return await self._immediate_training(symbol, training_data, training_features)

        # Если данных мало - накапливаем
        logger.info(f"⏳ {symbol}: данных мало ({len(data)}<{self.min_initial_training_size}), накапливаем...")
        ready_for_training = self._accumulate_training_data(symbol, data, features)

        if not ready_for_training:
          logger.info(f"⏳ {symbol}: накопление не завершено, ждем больше данных...")
          return False

        logger.info(f"✅ {symbol}: накопление завершено, начинаем обучение!")

        # Получаем накопленные данные
        combined_data, combined_features = self._get_accumulated_data(symbol)

        if combined_data.empty or combined_features.empty:
          logger.error(f"❌ {symbol}: не удалось получить накопленные данные")
          return False

        return await self._immediate_training(symbol, combined_data, combined_features)

      except Exception as e:
        logger.error(f"❌ Ошибка при обучении модели {symbol}: {e}", exc_info=True)
        return False

  async def _immediate_training(self, symbol: str, data: pd.DataFrame, features: pd.DataFrame) -> bool:
    """Немедленное обучение модели на подготовленных данных с улучшенной генерацией меток"""
    try:
      logger.info(f"🧠 {symbol}: немедленное обучение на {len(data)} примерах...")

      # УЛУЧШЕННАЯ ГЕНЕРАЦИЯ МЕТОК
      # Анализируем волатильность символа для адаптации порогов
      atr_values = data['high'] - data['low']
      avg_volatility = atr_values.rolling(window=20).mean().iloc[-1] if len(atr_values) > 20 else atr_values.mean()
      current_price = data['close'].iloc[-1]
      volatility_ratio = avg_volatility / current_price if current_price > 0 else 0.02

      # Адаптивные пороги для 15M таймфрейма
      if volatility_ratio > 0.015:  # Высокая волатильность для 15M
        threshold = 0.4  # Снижено с 1.2% до 0.4%
        future_bars = 4  # 4 бара = 1 час
        logger.info(f"📊 {symbol}: высокая волатильность {volatility_ratio:.4f}, порог={threshold}%")
      elif volatility_ratio > 0.008:  # Средняя волатильность для 15M
        threshold = 0.25  # Снижено с 0.9% до 0.25%
        future_bars = 4
        logger.info(f"📊 {symbol}: средняя волатильность {volatility_ratio:.4f}, порог={threshold}%")
      else:  # Низкая волатильность для 15M
        threshold = 0.15  # Снижено с 0.6% до 0.15%
        future_bars = 4  # Оставляем 4 бара для стабильности
        logger.info(f"📊 {symbol}: низкая волатильность {volatility_ratio:.4f}, порог={threshold}%")

      # Создаем метки с новой улучшенной функцией
      labels = self._create_enhanced_labels(
        data,
        future_bars=future_bars,
        threshold_percent=threshold,
        symbol=symbol
      )

      # Выравниваем индексы
      common_index = features.index.intersection(labels.index)
      features_train = features.loc[common_index]
      labels_train = labels.loc[common_index]

      if len(features_train) < 1000:  # Минимальный порог
        logger.warning(f"❌ {symbol}: после обработки мало данных: {len(features_train)}")
        return False

      # Проверяем распределение классов
      class_counts = labels_train.value_counts()
      total_samples = len(labels_train)

      buy_ratio = class_counts.get(1, 0) / total_samples
      sell_ratio = class_counts.get(2, 0) / total_samples
      hold_ratio = class_counts.get(0, 0) / total_samples

      logger.info(
        f"📈 {symbol}: распределение классов - BUY: {buy_ratio:.3f}, SELL: {sell_ratio:.3f}, HOLD: {hold_ratio:.3f}")

      # Если слишком много HOLD сигналов, применяем балансировку
      if hold_ratio > 0.8:
        logger.warning(f"⚠️ {symbol}: слишком много HOLD ({hold_ratio:.3f}), применяем балансировку")
        labels_train = self._balance_labels(labels_train, max_hold_ratio=0.7)

        # Пересчитываем статистику
        class_counts = labels_train.value_counts()
        buy_ratio = class_counts.get(1, 0) / len(labels_train)
        sell_ratio = class_counts.get(2, 0) / len(labels_train)
        hold_ratio = class_counts.get(0, 0) / len(labels_train)
        logger.info(
          f"📊 {symbol}: после балансировки - BUY: {buy_ratio:.3f}, SELL: {sell_ratio:.3f}, HOLD: {hold_ratio:.3f}")

      # Создаем и обучаем классификатор
      classifier = EnhancedLorentzianClassifier(
        k_neighbors=self.neighbors_count,
        max_bars_back=self.max_bars_back,
        feature_count=self.feature_count,
        use_dynamic_exits=self.use_dynamic_exits,
        filters=self.filter_settings
      )

      classifier.fit(features_train, labels_train)

      # Сохраняем обученную модель в кеш
      self.models_cache[symbol] = classifier

      # УЛУЧШЕННАЯ ОЦЕНКА КАЧЕСТВА модели
      test_size = min(1000, len(features_train) // 2)

      # Разделяем на обучение и тест по времени (последние данные для теста)
      train_features = features_train.iloc[:-test_size]
      train_labels_subset = labels_train.iloc[:-test_size]
      test_features = features_train.iloc[-test_size:]
      test_labels_subset = labels_train.iloc[-test_size:]

      # Переобучаем на уменьшенной выборке
      classifier_test = EnhancedLorentzianClassifier(
        k_neighbors=self.neighbors_count,
        max_bars_back=self.max_bars_back,
        feature_count=self.feature_count,
        use_dynamic_exits=self.use_dynamic_exits,
        filters=self.filter_settings
      )

      classifier_test.fit(train_features, train_labels_subset)

      # Тестируем на out-of-sample данных
      test_predictions = classifier_test.predict(test_features)
      test_labels_array = test_labels_subset.values

      # Общая точность
      accuracy = np.mean(test_predictions == test_labels_array)

      # Точность по классам
      class_accuracies = {}
      for class_id in [0, 1, 2]:
        class_mask = test_labels_array == class_id
        if np.sum(class_mask) > 0:
          class_acc = np.mean(test_predictions[class_mask] == test_labels_array[class_mask])
          class_accuracies[class_id] = class_acc

      # Точность торговых сигналов (BUY/SELL)
      trading_mask = (test_labels_array == 1) | (test_labels_array == 2)
      trading_accuracy = 0.0
      if np.sum(trading_mask) > 0:
        trading_accuracy = np.mean(test_predictions[trading_mask] == test_labels_array[trading_mask])

      logger.info(f"🎯 {symbol}: общая точность={accuracy:.3f}, торговые сигналы={trading_accuracy:.3f}")
      logger.info(f"📊 {symbol}: точность по классам - HOLD={class_accuracies.get(0, 0):.3f}, "
                  f"BUY={class_accuracies.get(1, 0):.3f}, SELL={class_accuracies.get(2, 0):.3f}")

      # АДАПТИВНЫЕ КРИТЕРИИ СОХРАНЕНИЯ для 15M
      # Более мягкие требования для высокочастотной торговли
      # Сохраняем модель если:
      # 1. Общая точность > 0.45 ИЛИ
      # 2. Точность торговых сигналов > 0.35 И общая точность > 0.35 ИЛИ
      # 3. Есть хотя бы какая-то точность по BUY или SELL классам
      has_trading_accuracy = (class_accuracies.get(1, 0) > 0.2 or class_accuracies.get(2, 0) > 0.2)

      save_model = (
          accuracy > 0.45 or
          (trading_accuracy > 0.35 and accuracy > 0.35) or
          (has_trading_accuracy and accuracy > 0.30)
      )

      # Специальное условие для символов с хорошим распределением классов
      good_distribution = (buy_ratio > 0.15 and sell_ratio > 0.15 and hold_ratio < 0.7)
      if good_distribution and accuracy > 0.30:
        save_model = True

      if save_model:
        self._save_model(symbol, classifier)
        logger.info(f"💾 {symbol}: модель сохранена (общая точность: {accuracy:.3f}, торговая: {trading_accuracy:.3f})")
      else:
        logger.warning(
          f"⚠️ {symbol}: модель не сохранена - низкое качество (общая: {accuracy:.3f}, торговая: {trading_accuracy:.3f})")

      return True

    except Exception as e:
      logger.error(f"❌ Ошибка немедленного обучения {symbol}: {e}", exc_info=True)
      return False

  def _balance_labels(self, labels: pd.Series, max_hold_ratio: float = 0.7) -> pd.Series:
    """Балансировка меток для уменьшения доли HOLD"""

    class_counts = labels.value_counts()
    total_samples = len(labels)

    hold_count = class_counts.get(0, 0)
    buy_count = class_counts.get(1, 0)
    sell_count = class_counts.get(2, 0)

    current_hold_ratio = hold_count / total_samples

    # Для 15M разрешаем больше HOLD (до 85% вместо 70%)
    if current_hold_ratio <= 0.85:
      return labels  # Уже достаточно сбалансировано для 15M

    # Вычисляем сколько HOLD нужно перевести в торговые сигналы
    target_hold_count = int(total_samples * max_hold_ratio)
    excess_hold = hold_count - target_hold_count

    # Находим индексы HOLD меток
    hold_indices = labels[labels == 0].index.tolist()

    # Случайно выбираем HOLD метки для конвертации
    import random
    random.seed(42)  # Для воспроизводимости
    convert_indices = random.sample(hold_indices, min(excess_hold, len(hold_indices)))

    # Создаем новые метки
    balanced_labels = labels.copy()

    for idx in convert_indices:
      # Конвертируем в BUY или SELL на основе локального контекста
      if random.random() < 0.5:
        balanced_labels.loc[idx] = 1  # BUY
      else:
        balanced_labels.loc[idx] = 2  # SELL

    return balanced_labels

  def _create_enhanced_labels(self, data: pd.DataFrame, future_bars: int = 4,
                              threshold_percent: float = 0.85, symbol: str = "") -> pd.Series:
    """Создание улучшенных меток с адаптивными порогами"""

    df = data.copy()

    # Добавляем технические индикаторы
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

    # ATR для адаптивных порогов
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()

    labels = []
    buy_count = 0
    sell_count = 0

    for i in range(len(df)):
      if i + future_bars >= len(df):
        labels.append(0)  # HOLD для последних баров
        continue

      current_price = df.iloc[i]['close']
      current_atr = df.iloc[i]['atr'] if pd.notna(df.iloc[i]['atr']) else current_price * 0.02
      current_vol_ratio = df.iloc[i]['volume_ratio'] if pd.notna(df.iloc[i]['volume_ratio']) else 1.0


      # Адаптивный порог на основе ATR (оптимизировано для 15M)
      atr_factor = current_atr / current_price

      # Для 15M используем более агрессивные пороги
      # ATR обычно 0.001-0.003 для крипто на 15M, что дает 0.1-0.3%
      # Умножаем на 0.5-1.0 вместо 2.0 для большего количества сигналов
      dynamic_threshold = max(threshold_percent / 100, atr_factor * 0.7)

      # Ограничиваем порог для 15M
      dynamic_threshold = min(dynamic_threshold, 0.004)  # Максимум 0.4%
      dynamic_threshold = max(dynamic_threshold, 0.001)  # Минимум 0.1%

      # Анализ будущих цен
      future_slice = df.iloc[i + 1:i + future_bars + 1]
      future_highs = future_slice['high']
      future_lows = future_slice['low']
      future_closes = future_slice['close']

      max_high = future_highs.max()
      min_low = future_lows.min()
      final_close = future_closes.iloc[-1]

      # Потенциальные движения
      max_upside = (max_high - current_price) / current_price
      max_downside = (current_price - min_low) / current_price

      if i % 500 == 0:  # Логируем каждую 500-ю свечу
        logger.debug(f"{symbol} [{i}]: upside={max_upside:.4f}, downside={max_downside:.4f}, "
                     f"threshold={dynamic_threshold:.4f}, atr_factor={atr_factor:.4f}")
      final_return = (final_close - current_price) / current_price

      # Анализ качества движения
      price_momentum = final_return / dynamic_threshold if dynamic_threshold > 0 else 0
      volume_confirmation = current_vol_ratio > 1.1

      # Критерии для BUY
      strong_buy = (
          max_upside > dynamic_threshold * 1.5 and
          final_return > dynamic_threshold * 0.7 and
          max_upside > max_downside * 1.3 and
          abs(price_momentum) > 1.0
      )

      # Критерии для SELL
      strong_sell = (
          max_downside > dynamic_threshold * 1.5 and
          final_return < -dynamic_threshold * 0.7 and
          max_downside > max_upside * 1.3 and
          abs(price_momentum) > 1.0
      )

      # Принятие решения
      if strong_buy and (not volume_confirmation or buy_count < sell_count + 50):
        labels.append(1)  # BUY
        buy_count += 1
      elif strong_sell and (not volume_confirmation or sell_count < buy_count + 50):
        labels.append(2)  # SELL
        sell_count += 1
      else:
        labels.append(0)  # HOLD

    labels_series = pd.Series(labels, index=df.index)

    # Логируем статистику
    class_counts = labels_series.value_counts()
    total = len(labels_series)
    logger.info(
      f"🏷️ {symbol}: метки созданы - BUY={class_counts.get(1, 0)} ({class_counts.get(1, 0) / total * 100:.1f}%), "
      f"SELL={class_counts.get(2, 0)} ({class_counts.get(2, 0) / total * 100:.1f}%), "
      f"HOLD={class_counts.get(0, 0)} ({class_counts.get(0, 0) / total * 100:.1f}%)")

    # Итоговая статистика
    buy_final = labels_series.eq(1).sum()
    sell_final = labels_series.eq(2).sum()
    hold_final = labels_series.eq(0).sum()
    total = len(labels_series)

    logger.info(f"🏷️ {symbol}: метки созданы - BUY={buy_final} ({buy_final / total * 100:.1f}%), "
                f"SELL={sell_final} ({sell_final / total * 100:.1f}%), "
                f"HOLD={hold_final} ({hold_final / total * 100:.1f}%)")

    if (buy_final + sell_final) / total < 0.03:
      logger.warning(f"⚠️ {symbol}: Очень мало торговых сигналов! "
                     f"Рекомендуется еще снизить пороги")



    return labels_series

  def _apply_filters(self, data: pd.DataFrame, symbol: str = None) -> Dict[str, bool]:
    """
    Применение фильтров из оригинального индикатора + WaveTrend 3D
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

    # WaveTrend 3D Filter
    if self.use_wavetrend_3d and self.wavetrend_3d and symbol:
      wavetrend_signal = self.wavetrend_3d.get_signal_for_lorentzian(data, symbol)
      if wavetrend_signal:
        # Проверяем согласованность направления
        results['wavetrend_3d'] = True
        results['wavetrend_direction'] = wavetrend_signal['direction']
        results['wavetrend_confidence'] = wavetrend_signal['confidence']
      else:
        results['wavetrend_3d'] = True  # Не блокируем если нет сигнала
        results['wavetrend_direction'] = 0
        results['wavetrend_confidence'] = 0.5
    else:
      results['wavetrend_3d'] = True

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

      if cache['update_count'] % 50 == 0 and self.settings.get('save_models', True):
        self._save_model(symbol, classifier)
        logger.info(f"Модель {symbol} сохранена после {cache['update_count']} обновлений")

      # Сохранение модели каждые 100 обновлений
      if cache['update_count'] % 2 == 0 and self.settings.get('save_models', True):
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
    эмулирует поведение индикатора в TradingView
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

  async def process_trade_feedback(self, symbol: str, trade_result: Dict) -> None:
    """Обрабатывает результаты сделки для переобучения модели"""
    try:
      # Определяем фактический результат
      profit_loss = trade_result.get('profit_loss', 0)
      if profit_loss > 0:
        actual_outcome = 1  # Сигнал был правильным
      elif profit_loss < 0:
        actual_outcome = -1  # Сигнал был неправильным
      else:
        actual_outcome = 0  # Нейтральный результат

      # Получаем последние данные для обновления модели
      if hasattr(self, 'data_fetcher') and self.data_fetcher:
        recent_data = await self.data_fetcher.get_historical_candles(
          symbol,
          Timeframe.FIFTEEN_MINUTES,
          limit=50
        )

        if not recent_data.empty:
          self.update_model(symbol, recent_data, actual_outcome)
          logger.info(f"✅ Lorentzian модель обновлена для {symbol}, результат: {actual_outcome}")

    except Exception as e:
      logger.error(f"Ошибка обработки торгового отзыва в Lorentzian: {e}")

  def _accumulate_training_data(self, symbol: str, data: pd.DataFrame, features: pd.DataFrame) -> bool:
    """
    Накапливает данные для обучения до достижения минимального размера
    УПРОЩЕННАЯ ВЕРСИЯ - берем все доступные данные сразу

    Returns:
        True если накоплено достаточно данных для обучения
    """
    try:
      # Инициализируем историю для символа если её нет
      if symbol not in self.training_history:
        self.training_history[symbol] = {
          'combined_data': pd.DataFrame(),
          'combined_features': pd.DataFrame(),
          'accumulated_count': 0,
          'last_update': datetime.now(),
          'is_ready': False
        }
        logger.info(f"🆕 {symbol}: инициализировано накопление данных для обучения")

      history = self.training_history[symbol]

      # Проверяем валидность входных данных
      if data.empty or features.empty:
        logger.warning(f"❌ {symbol}: пустые данные для накопления")
        return False

      # БЕРЕМ ВСЕ ДОСТУПНЫЕ ДАННЫЕ СРАЗУ
      logger.info(f"📥 {symbol}: получено {len(data)} точек данных, {len(features)} признаков")

      # Проверяем размер данных
      min_len = min(len(data), len(features))
      if min_len == 0:
        logger.warning(f"❌ {symbol}: нет данных для обработки")
        return False

      # Выравниваем данные по размеру
      aligned_data = data.head(min_len).copy()
      aligned_features = features.head(min_len).copy()

      # Сохраняем данные (перезаписываем каждый раз - берем самые свежие)
      history['combined_data'] = aligned_data
      history['combined_features'] = aligned_features
      history['accumulated_count'] = len(aligned_data)
      history['last_update'] = datetime.now()

      logger.info(f"✅ {symbol}: сохранено {history['accumulated_count']} точек данных")

      # Проверяем готовность к обучению
      is_ready = history['accumulated_count'] >= self.min_initial_training_size
      history['is_ready'] = is_ready

      if is_ready:
        logger.info(
          f"🎯 {symbol}: ✅ ДОСТАТОЧНО ДАННЫХ ({history['accumulated_count']}>={self.min_initial_training_size})! "
          f"Готов к обучению!")
      else:
        needed = self.min_initial_training_size - history['accumulated_count']
        logger.info(f"⏳ {symbol}: накоплено {history['accumulated_count']}/{self.min_initial_training_size}, "
                    f"нужно еще {needed} точек")

      return is_ready

    except Exception as e:
      logger.error(f"❌ Ошибка накопления данных для {symbol}: {e}", exc_info=True)
      return False

  def log_accumulation_summary(self):
    """Логирует сводку по накоплению данных для всех символов"""
    try:
      if not self.training_history:
        logger.info("📊 СВОДКА НАКОПЛЕНИЯ: нет данных")
        return

      ready_count = 0
      total_symbols = len(self.training_history)

      logger.info("📊 СВОДКА НАКОПЛЕНИЯ ДАННЫХ:")
      logger.info("=" * 60)

      for symbol, history in self.training_history.items():
        progress_pct = (history['accumulated_count'] / self.min_initial_training_size) * 100
        status = "✅ ГОТОВ" if history['accumulated_count'] >= self.min_initial_training_size else "⏳ Накопление"

        if status == "✅ ГОТОВ":
          ready_count += 1

        logger.info(f"{symbol:12} | {history['accumulated_count']:5}/{self.min_initial_training_size} "
                    f"({progress_pct:5.1f}%) | {status} | Сессий: {history['session_count']:3}")

      logger.info("=" * 60)
      logger.info(f"ИТОГО: {ready_count}/{total_symbols} символов готовы к обучению")

    except Exception as e:
      logger.error(f"Ошибка в сводке накопления: {e}")

  def _get_accumulated_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает накопленные данные для символа
    УПРОЩЕННАЯ ВЕРСИЯ

    Returns:
        Tuple of (combined_data, combined_features)
    """
    try:
      if symbol not in self.training_history:
        logger.warning(f"❌ {symbol}: нет накопленных данных")
        return pd.DataFrame(), pd.DataFrame()

      history = self.training_history[symbol]

      combined_data = history.get('combined_data', pd.DataFrame())
      combined_features = history.get('combined_features', pd.DataFrame())

      if combined_data.empty or combined_features.empty:
        logger.warning(f"❌ {symbol}: пустые накопленные данные")
        return pd.DataFrame(), pd.DataFrame()

      logger.info(f"📤 {symbol}: возвращаем {len(combined_data)} точек данных, "
                  f"{len(combined_features)} точек признаков")

      return combined_data, combined_features

    except Exception as e:
      logger.error(f"❌ Ошибка получения данных для {symbol}: {e}", exc_info=True)
      return pd.DataFrame(), pd.DataFrame()

  def get_accumulation_status(self, symbol: str = None) -> Dict[str, Any]:
    """Возвращает статус накопления данных"""
    if symbol:
      # Статус для конкретного символа
      if symbol in self.training_history:
        history = self.training_history[symbol]
        return {
          'symbol': symbol,
          'accumulated_count': history['accumulated_count'],
          'required_count': self.min_initial_training_size,
          'progress_pct': (history['accumulated_count'] / self.min_initial_training_size) * 100,
          'ready_for_training': history['accumulated_count'] >= self.min_initial_training_size,
          'last_update': history['last_update']
        }
      else:
        return {
          'symbol': symbol,
          'accumulated_count': 0,
          'required_count': self.min_initial_training_size,
          'progress_pct': 0.0,
          'ready_for_training': False,
          'last_update': None
        }
    else:
      # Статус для всех символов
      status = {}
      for sym, history in self.training_history.items():
        status[sym] = {
          'accumulated_count': history['accumulated_count'],
          'required_count': self.min_initial_training_size,
          'progress_pct': (history['accumulated_count'] / self.min_initial_training_size) * 100,
          'ready_for_training': history['accumulated_count'] >= self.min_initial_training_size,
          'last_update': history['last_update']
        }
      return status

  async def preload_training_data(self, symbol: str, force_reload: bool = False):
    """
    Предварительная загрузка больших объемов данных для обучения с итеративными запросами.
    """
    target_count = self.min_initial_training_size
    timeframe = Timeframe.FIFTEEN_MINUTES  # Укажите нужный таймфрейм

    # Проверяем, нужна ли загрузка
    if not force_reload and symbol in self.training_history:
      current_count = self.training_history[symbol].get('accumulated_count', 0)
      if current_count >= target_count:
        logger.info(f"✅ {symbol}: Данные уже накоплены ({current_count} точек), пропуск.")
        # Убедимся, что модель обучена, если есть накопленные данные
        # if symbol not in self.models_cache or not self.models_cache[symbol].is_fitted:
        #   logger.info(f"Модель для {symbol} не обучена, запускаем обучение на существующих данных.")
        #   # Запускаем немедленное обучение на уже имеющихся данных
        #   try:
        #     data, features = self._get_accumulated_data(symbol)
        #     if not data.empty and not features.empty:
        #       await self._immediate_training(symbol, data, features)
        #   except Exception as e:
        #     logger.error(f"Ошибка обучения на существующих данных для {symbol}: {e}")
        # return True
        if symbol not in self.models_cache or not self.models_cache[symbol].is_fitted:
          logger.info(f"⏳ {symbol}: Модель не готова, запускаем предзагрузку...")
          # Запускаем предзагрузку в фоне если еще не запущена
          asyncio.create_task(self.preload_training_data(symbol))
          return None

    if not self.data_fetcher:
      logger.error(f"❌ DataFetcher не установлен для предзагрузки {symbol}")
      return False

    logger.info(f"🚀 Начинаем итеративную предзагрузку для {symbol}. Цель: {target_count} свечей.")

    all_candles = pd.DataFrame()
    api_limit_per_request = 1000  # Стандартный лимит API бирж
    end_time_ms = None  # Для пагинации запросов в прошлое

    max_requests = 10  # Защита от бесконечного цикла
    for i in range(max_requests):
      if len(all_candles) >= target_count:
        logger.info(f"🎯 {symbol}: Цель в {target_count} свечей достигнута. Собрано: {len(all_candles)}.")
        break

      logger.info(f"⏳ {symbol}: Запрос #{i + 1}. Собрано {len(all_candles)}/{target_count}...")

      # ВАЖНО: Предполагаем, что ваш data_fetcher может принимать 'params' для передачи 'endTime' в API
      # Это стандартная практика для получения старых данных (например, на Binance)
      # Словарь с аргументами для запроса
      request_kwargs = {
        'symbol': symbol,
        'timeframe': timeframe,
        'limit': api_limit_per_request
      }
      # Добавляем 'end' только если он есть, чтобы запрашивать историю
      if end_time_ms:
        request_kwargs['end'] = end_time_ms

      try:
        # ИСПРАВЛЕННЫЙ ВЫЗОВ: передаем аргументы через **
        chunk = await self.data_fetcher.get_historical_candles(**request_kwargs)

      except Exception as e:
        logger.error(f"❌ Ошибка API при загрузке части данных для {symbol}: {e}")
        await asyncio.sleep(5)
        continue

      if chunk is None or chunk.empty:
        logger.warning(f"⚠️ {symbol}: Больше нет доступных исторических данных. Собрано {len(all_candles)}.")
        break
      # --- УЛУЧШЕННАЯ И БОЛЕЕ НАДЕЖНАЯ ЛОГИКА ---
      # Явно берем самую старую метку времени из полученного чанка
      oldest_timestamp = chunk.index.min()
      end_time_ms = int(oldest_timestamp.timestamp() * 1000) - 1
      # -------------------------------------------

      # Добавляем новые данные в начало общего датафрейма
      all_candles = pd.concat([chunk, all_candles])

      # Устанавливаем `endTime` для следующего запроса на 1мс раньше самой старой свечи из полученных
      # Данные обычно приходят отсортированными от новых к старым
      end_time_ms = int(chunk.index[0].timestamp() * 1000) - 1

      # Добавляем новые данные в начало общего датафрейма
      all_candles = pd.concat([chunk, all_candles])

      # Удаляем дубликаты на случай пересечения данных
      all_candles = all_candles[~all_candles.index.duplicated(keep='first')]

      # Пауза между запросами, чтобы не перегружать API
      await asyncio.sleep(2)

    if all_candles.empty:
      logger.error(f"❌ {symbol}: Не удалось загрузить исторические данные.")
      return False

    # Сортируем индекс, чтобы данные шли в хронологическом порядке
    all_candles.sort_index(inplace=True)
    logger.info(f"🧮 {symbol}: Данные собраны ({len(all_candles)} свечей). Начинаем расчет признаков...")

    # Расчет признаков для всего датасета
    features = self.indicators.calculate_features(
      all_candles,
      self.feature_configs['f1'], self.feature_configs['f2'], self.feature_configs['f3'],
      self.feature_configs['f4'], self.feature_configs['f5']
    )

    if features.empty:
      logger.error(f"❌ {symbol}: Не удалось рассчитать признаки.")
      return False

    # Сохраняем все данные и сразу обучаем модель
    self._accumulate_training_data(symbol, all_candles, features)
    success = await self._immediate_training(symbol, all_candles, features)

    if success:
      logger.info(f"✅✅✅ {symbol}: Предзагрузка и обучение успешно завершены!")
    else:
      logger.warning(f"⚠️⚠️⚠️ {symbol}: Предзагрузка завершена, но обучение не удалось.")

    return success

  async def preload_multiple_symbols(self, symbols: List[str], max_concurrent: int = 3):
    """Предзагрузка данных для нескольких символов параллельно"""
    import asyncio

    async def preload_single(symbol):
      try:
        return await self.preload_training_data(symbol)
      except Exception as e:
        logger.error(f"Ошибка предзагрузки {symbol}: {e}")
        return False

    # Разбиваем на батчи чтобы не перегружать API
    results = {}
    for i in range(0, len(symbols), max_concurrent):
      batch = symbols[i:i + max_concurrent]
      logger.info(f"🔄 Предзагрузка батча {i // max_concurrent + 1}: {batch}")

      batch_tasks = [preload_single(symbol) for symbol in batch]
      batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

      for symbol, result in zip(batch, batch_results):
        results[symbol] = result

      # Небольшая пауза между батчами
      if i + max_concurrent < len(symbols):
        await asyncio.sleep(1)

    successful = sum(1 for r in results.values() if r is True)
    logger.info(f"✅ Предзагрузка завершена: {successful}/{len(symbols)} символов готовы")

    return results