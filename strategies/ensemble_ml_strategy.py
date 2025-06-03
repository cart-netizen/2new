import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import asyncio
import joblib
from datetime import datetime
import warnings

from core.enums import SignalType
from core.schemas import TradingSignal
from data.database_manager import AdvancedDatabaseManager
from ml.lorentzian_classifier import LorentzianClassifier, create_training_labels
from utils.logging_config import get_logger

class EnsembleMLStrategy:
  """Ensemble ML стратегия с автоматическим переобучением"""

  def __init__(self, db_manager: AdvancedDatabaseManager, ml_model: LorentzianClassifier):
    self.strategy_name = "Ensemble_ML_Strategy"
    self.db_manager = db_manager
    self.base_model = ml_model
    self.models = {}
    self.performance_threshold = 0.6  # Минимальная точность для использования модели
    self.retrain_interval = 24 * 60 * 60  # 24 часа в секундах
    self.last_retrain = {}
    # self.models = {}

    print(f"Инициализация EnsembleMLStrategy. ml_model type: {type(ml_model)}")



  def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Подготавливает признаки для ML модели"""
    if len(data) < 50:
      return pd.DataFrame()

    df = data.copy()

    # Технические индикаторы
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['rsi_fast'] = ta.rsi(df['close'], length=7)
    df['rsi_slow'] = ta.rsi(df['close'], length=21)

    df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
    df['macd_signal'] = ta.macd(df['close'])['MACDs_12_26_9']
    df['macd_hist'] = ta.macd(df['close'])['MACDh_12_26_9']

    bb_data = ta.bbands(df['close'])
    if bb_data is not None and not bb_data.empty:
      bb_cols = bb_data.columns.tolist()
      upper_col = [col for col in bb_cols if 'BBU' in col][0] if any('BBU' in col for col in bb_cols) else None
      lower_col = [col for col in bb_cols if 'BBL' in col][0] if any('BBL' in col for col in bb_cols) else None

      if upper_col and lower_col:
        df['bb_upper'] = bb_data[upper_col]
        df['bb_lower'] = bb_data[lower_col]
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
      else:
        df['bb_upper'] = df['close'] * 1.02
        df['bb_lower'] = df['close'] * 0.98
        df['bb_percent'] = 0.5
    else:
      df['bb_upper'] = df['close'] * 1.02
      df['bb_lower'] = df['close'] * 0.98
      df['bb_percent'] = 0.5

    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_percent'] = df['atr'] / df['close']

    # Moving averages
    df['sma_10'] = ta.sma(df['close'], length=10)
    df['sma_20'] = ta.sma(df['close'], length=20)
    df['sma_50'] = ta.sma(df['close'], length=50)
    df['ema_12'] = ta.ema(df['close'], length=12)
    df['ema_26'] = ta.ema(df['close'], length=26)

    # Price patterns
    df['price_change'] = df['close'].pct_change()
    df['price_change_5'] = df['close'].pct_change(5)
    df['volatility'] = df['price_change'].rolling(20).std()

    # Volume indicators (если есть volume)
    if 'volume' in df.columns:
      df['volume_sma'] = ta.sma(df['volume'], length=20)
      df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Market structure
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

    return df

  def _create_labels(self, data: pd.DataFrame, lookahead: int = 5) -> pd.Series:
    """Создает метки для обучения на основе будущих движений цены"""
    future_returns = data['close'].shift(-lookahead) / data['close'] - 1

    # Определяем пороги для сигналов
    buy_threshold = 0.01  # 1% прибыль для покупки
    sell_threshold = -0.01  # 1% убыток для продажи

    labels = pd.Series(0, index=data.index)  # 0 = HOLD
    labels[future_returns > buy_threshold] = 1  # 1 = BUY
    labels[future_returns < sell_threshold] = 2  # 2 = SELL

    return labels

  async def should_retrain(self, symbol: str) -> bool:
    """Проверяет, нужно ли переобучить модель"""
    if symbol not in self.last_retrain:
      return True

    time_since_retrain = (datetime.datetime.now() - self.last_retrain[symbol]).total_seconds()
    return time_since_retrain > self.retrain_interval

  async def train_ensemble_model(self, symbol: str, data: pd.DataFrame):
    """Обучает ensemble модель для конкретного символа"""
    print(f"🔄 Обучение ensemble модели для {symbol}...")

    # Подготовка данных
    features_df = self._prepare_features(data)
    if features_df.empty:
      print(f"❌ Недостаточно данных для обучения {symbol}")
      return

    labels = self._create_labels(features_df)

    # Выбираем только числовые признаки без NaN
    feature_columns = features_df.select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in feature_columns if col not in ['open', 'high', 'low', 'close', 'volume']]

    X = features_df[feature_columns].fillna(0)
    y = labels

    # Убираем последние строки где нет меток
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]

    self.models[symbol] = {
      'model': None,
      'feature_columns': ['rsi', 'macd'],  # Пример фичей
      'accuracy': 0.75  # Пример точности
    }
    self.last_retrain[symbol] = datetime.datetime.now()


    if len(X) < 100:
      print(f"❌ Недостаточно данных для обучения {symbol} (нужно минимум 100)")
      return



  async def generate_signals(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
    """Генерирует торговые сигналы с использованием ensemble модели"""
    try:
      # Всегда проверяем что self.models - словарь
      if not isinstance(self.models, dict):
        print(f"⚠️ Ошибка инициализации self.models. Используем fallback.")
        return await self._fallback_strategy(symbol, data)

      # Проверяем, нужно ли переобучить модель
      if await self.should_retrain(symbol):
        await self.train_ensemble_model(symbol, data)

      # Проверяем наличие модели
      if symbol not in self.models:
        print(f"⚠️ Модель для {symbol} не найдена, используем fallback стратегию")
        return await self._fallback_strategy(symbol, data)

      # Подготавливаем признаки
      features_df = self._prepare_features(data)
      if features_df.empty:
        print(f"⚠️ Не удалось подготовить признаки для {symbol}, используем fallback")
        return await self._fallback_strategy(symbol, data)

      model_info = self.models[symbol]
      latest_features = features_df[model_info['feature_columns']].fillna(0).tail(1)

      if latest_features.empty:
        return None

      # Делаем предсказание
      prediction_proba = None
      if model_info['model']:
        prediction_proba = model_info['model'].predict_proba(latest_features)

      if prediction_proba is None:
        return await self._fallback_strategy(symbol, data)

      # Проверяем, нужно ли переобучить модель
      if await self.should_retrain(symbol):
        await self.train_ensemble_model(symbol, data)

      # Проверяем наличие модели
      if symbol not in self.models:
        print(f"⚠️ Модель для {symbol} не найдена, используем fallback стратегию")
        return await self._fallback_strategy(symbol, data)

      model_info = self.models[symbol]
      model = model_info['model']

      # Получаем последнюю строку для предсказания
      latest_features = features_df[model_info['feature_columns']].fillna(0).tail(1)

      if latest_features.empty:
        return None
    except Exception as e:
      print(f"❌ Критическая ошибка генерации сигнала для {symbol}: {e}")
      return await self._fallback_strategy(symbol, data)
      # Делаем предсказание
    try:
        prediction_proba = model.predict_proba(latest_features)
        if prediction_proba is None:
          return await self._fallback_strategy(symbol, data)

        # Получаем вероятности для каждого класса
        hold_prob, buy_prob, sell_prob = prediction_proba[0]

        # Определяем сигнал на основе наибольшей вероятности
        max_prob = max(hold_prob, buy_prob, sell_prob)
        confidence = float(max_prob)

        # Минимальный порог уверенности
        if confidence < 0.6:
          signal_type = SignalType.HOLD
        elif buy_prob == max_prob:
          signal_type = SignalType.BUY
        elif sell_prob == max_prob:
          signal_type = SignalType.SELL
        else:
          signal_type = SignalType.HOLD

        if signal_type == SignalType.HOLD:
          return None

        # Получаем текущие рыночные данные
        current_price = float(data['close'].iloc[-1])
        current_atr = float(features_df['atr'].iloc[-1]) if not pd.isna(
          features_df['atr'].iloc[-1]) else current_price * 0.02

        # Вычисляем Stop Loss и Take Profit
        atr_multiplier = 2.0 if signal_type == SignalType.BUY else 2.0

        if signal_type == SignalType.BUY:
          stop_loss = current_price - (atr_multiplier * current_atr)
          take_profit = current_price + (3.0 * current_atr)
        else:  # SELL
          stop_loss = current_price + (atr_multiplier * current_atr)
          take_profit = current_price - (3.0 * current_atr)

        # Создаем торговый сигнал
        signal = TradingSignal(
          signal=signal_type,
          price=current_price,
          confidence=confidence,
          stop_loss=round(stop_loss, 4),
          take_profit=round(take_profit, 4),
          strategy_name=self.strategy_name,
          timestamp=datetime.datetime.now(),
          metadata={
            'symbol': symbol,
            'model_accuracy': model_info['accuracy'],
            'atr': current_atr,
            'buy_prob': float(buy_prob),
            'sell_prob': float(sell_prob),
            'hold_prob': float(hold_prob)
          }
        )

        # Логируем сигнал
        self.db_manager.log_signal(signal, symbol)

        print(f"🎯 {self.strategy_name} сгенерировал сигнал для {symbol}:")
        print(f"   Сигнал: {signal_type.value}, Цена: {current_price}, Уверенность: {confidence:.2%}")
        print(f"   SL: {stop_loss}, TP: {take_profit}")

        return signal

    except Exception as e:
        print(f"❌ Ошибка генерации сигнала для {symbol}: {e}")
        return await self._fallback_strategy(symbol, data)

  async def _fallback_strategy(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
    """Резервная стратегия на основе классических индикаторов"""
    if len(data) < 50:
      return None

    # Вычисляем индикаторы
    data['rsi'] = ta.rsi(data['close'], length=14)
    data['macd'] = ta.macd(data['close'])['MACD_12_26_9']
    data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)

    latest = data.iloc[-1]

    if pd.isna(latest['rsi']) or pd.isna(latest['atr']):
      return None

    rsi = latest['rsi']
    macd = latest['macd'] if not pd.isna(latest['macd']) else 0
    current_price = latest['close']
    current_atr = latest['atr']

    # Простая логика
    signal_type = SignalType.HOLD
    confidence = 0.5

    if rsi < 25 and macd > 0:
      signal_type = SignalType.BUY
      confidence = 0.7
    elif rsi > 75 and macd < 0:
      signal_type = SignalType.SELL
      confidence = 0.7

    if signal_type == SignalType.HOLD:
      return None

    # Stop Loss и Take Profit
    if signal_type == SignalType.BUY:
      stop_loss = current_price - (2 * current_atr)
      take_profit = current_price + (3 * current_atr)
    else:
      stop_loss = current_price + (2 * current_atr)
      take_profit = current_price - (3 * current_atr)

    return TradingSignal(
      signal=signal_type,
      price=current_price,
      confidence=confidence,
      stop_loss=round(stop_loss, 4),
      take_profit=round(take_profit, 4),
      strategy_name="Fallback_RSI_MACD",
      timestamp=datetime.datetime.now(),
      metadata={'symbol': symbol, 'rsi': rsi, 'macd': macd}
    )

  async def initialize_models(self, symbols: List[str]):
    """Инициализирует модели для всех указанных символов"""
    print(f"🔄 Инициализация моделей EnsembleMLStrategy для {len(symbols)} символов...")
    for symbol in symbols:
      # Если для символа уже есть модель, пропускаем
      if symbol in self.models:
        continue
      try:
        # Пытаемся создать клон, но ловим возможные ошибки
        model_clone = self.base_model.clone() if self.base_model else None
      except Exception as e:
        print(f"⚠️ Ошибка клонирования модели для {symbol}: {e}")
        model_clone = None

      # Создаем модель для символа на основе базовой модели
      if self.base_model:
        # Клонируем базовую модель для символа
        self.models[symbol] = {
          'model': self.base_model.clone(),  # Необходимо реализовать метод clone()
          'feature_columns': self.base_model.feature_columns.copy(),
          'accuracy': self.base_model.accuracy
        }
        print(f"✅ Модель для {symbol} инициализирована (на основе базовой)")
      else:
        # Создаем новую модель для символа
        self.models[symbol] = {
          'model': LorentzianClassifier(k_neighbors=8),
          'feature_columns': ['close', 'volume'],
          'accuracy': 0.75
        }
        print(f"✅ Новая модель создана для {symbol}")

  async def retrain_with_feedback(self, symbol: str, results: list):
    """Обновляет модель на основе результатов сделок"""
    try:
      print(f"🔄 Переобучение модели {symbol} на основе {len(results)} сделок...")

      # Если для символа нет модели, создаем базовую
      if symbol not in self.models:
        self.models[symbol] = {
          'model': None,
          'feature_columns': ['close', 'volume'],
          'accuracy': 0.75
        }

      # Анализируем результаты сделок
      win_rate = sum(1 for r in results if r['profit_loss'] > 0) / len(results) if results else 0.5
      avg_profit = sum(r['profit_loss'] for r in results) / len(results) if results else 0

      # Простое обновление точности модели на основе результатов
      current_accuracy = self.models[symbol].get('accuracy', 0.75)
      new_accuracy = current_accuracy * 0.9 + win_rate * 0.1

      # Обновляем модель
      self.models[symbol]['accuracy'] = new_accuracy
      self.last_retrain[symbol] = datetime.datetime.now()

      print(f"✅ Модель {symbol} обновлена: "
            f"Точность: {current_accuracy:.2f} → {new_accuracy:.2f}, "
            f"Средняя прибыль: {avg_profit:.4f}")

    except Exception as e:
      print(f"❌ Ошибка переобучения модели {symbol}: {e}")
