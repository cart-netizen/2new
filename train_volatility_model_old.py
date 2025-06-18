
import asyncio
import joblib
import pandas as pd
import numpy as np

# Импортируем компоненты нашего бота
from core.data_fetcher import DataFetcher
from core.bybit_connector import BybitConnector
from core.enums import Timeframe
from ml.volatility_system import VolatilityPredictionSystem, ModelType
from ml.feature_engineering import feature_engineer  # <-- Импортируем наш главный FeatureEngineer
from utils.logging_config import setup_logging


async def main():
  """
  ФИНАЛЬНАЯ ВЕРСИЯ: Скрипт для обучения модели волатильности
  на МНОЖЕСТВЕ символов с использованием МУЛЬТИТАЙМФРЕЙМ-признаков.
  """
  setup_logging("INFO")

  # 1. Инициализация
  connector = BybitConnector()
  await connector.sync_time()
  data_fetcher = DataFetcher(connector, settings={})

  # 2. Загрузка данных для множества символов
  symbols_to_train = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'FARTCOINUSDT', 'ZKJUSDT', 'HYPEUSDT', 'DOGEUSDT', '1000PEPEUSDT', 'SUIUSDT', 'WIFUSDT', 'AAVEUSDT', 'ALTUSDT', 'SPXUSDT',
                      'ADAUSDT', 'TRXUSDT', 'UNIUSDT', 'LINKUSDT', 'POPCATUSDT', 'MAGICUSDT', 'AI16ZUSDT', 'JTOUSDT', 'TRUMPUSDT', 'ENAUSDT', 'VIRTUALUSDT', 'BCHUSDT', 'OPUSDT', 'AVAXUSDT', 'ARBUSDT', 'ANIMEUSDT', 'TUSDT', 'MOODENGUSDT', 'LAUNCHCOINUSDT', 'ONDOUSDT', 'RVNUSDT', 'INJUSDT', 'WLDUSDT',
'SNTUSDT', 'TIAUSDT', 'NEARUSDT', 'LTCUSDT', 'GALAUSDT', 'TRBUSDT', 'CRVUSDT', '1000BONKUSDT', 'COOKIEUSDT', 'MKRUSDT', 'TONUSDT', 'KAIAUSDT', 'SYRUPUSDT', 'AEROUSDT', 'DOTUSDT', 'TAOUSDT', 'BMTUSDT', 'BNBUSDT', 'JUPUSDT', 'INITUSDT', 'BRETTUSDT', 'KASUSDT', 'LAUSDT', 'NXPCUSDT', 'ETHFIUSDT', 'ORDIUSDT', 'HAEDALUSDT', 'VELOUSDT', 'WCTUSDT', 'PNUTUSDT', 'LDOUSDT', 'AXLUSDT', 'ALCHUSDT', 'HBARUSDT', 'AIXBTUSDT', 'MASKUSDT', '1000NEIROCTOUSDT', 'FORMUSDT', 'XAUTUSDT', 'GRIFFAINUSDT', 'AGTUSDT', 'SOPHUSDT', 'EIGENUSDT', 'APTUSDT', 'KAITOUSDT', 'ENSUSDT', 'CHILLGUYUSDT', 'SHIB1000USDT', 'GOATUSDT', 'SEIUSDT', 'POLUSDT', 'ATOMUSDT', 'OMUSDT', 'PENGUUSDT', 'DYDXUSDT', 'ICPUSDT', 'STRKUSDT', 'FLOCKUSDT', '1000FLOKIUSDT', 'FILUSDT', 'ZBCNUSDT', 'APEUSDT', 'XLMUSDT', 'SUSDT', 'ZEREBROUSDT', 'SWEATUSDT', 'GRASSUSDT', 'COMPUSDT', '1000000MOGUSDT', 'ETCUSDT', 'PAXGUSDT', 'MOVEUSDT', 'RENDERUSDT', 'REXUSDT', 'ALGOUSDT', 'JELLYJELLYUSDT', 'LPTUSDT', 'BERAUSDT', 'HUMAUSDT', 'PENDLEUSDT', 'ATHUSDT', 'STXUSDT', 'B3USDT', 'PEOPLEUSDT', 'ZROUSDT', 'HMSTRUSDT', 'MEWUSDT', 'MNTUSDT', 'GIGAUSDT', 'NEIROETHUSDT', 'SOONUSDT','PYTHUSDT', 'NOTUSDT', 'KMNOUSDT', 'SANDUSDT', 'XAIUSDT', 'THEUSDT', 'ARCUSDT', 'BDXNUSDT', 'SERAPHUSDT', 'BOMEUSDT', 'ORBSUSDT', 'JASMYUSDT', 'DEEPUSDT', 'SUNDOGUSDT', '1000RATSUSDT', 'MUBARAKUSDT', 'GMTUSDT', 'FIDAUSDT', 'PUMPBTCUSDT', 'ZKUSDT', 'GRTUSDT', 'XMRUSDT', '1000TURBOUSDT', 'USUALUSDT', 'ARUSDT', '10000SATSUSDT', 'TAIKOUSDT', 'TAIUSDT', 'BIGTIMEUSDT',
                      'ZECUSDT', 'RUNEUSDT', 'ICXUSDT', 'RAYDIUMUSDT', 'BLURUSDT', 'DARKUSDT', 'SUSHIUSDT', 'BUSDT', 'CLOUDUSDT', 'RPLUSDT', 'DOGSUSDT', 'WALUSDT', 'FUSDT', '1000CATSUSDT', 'FWOGUSDT', 'USTCUSDT', 'ARKMUSDT', 'TUTUSDT', 'SOLAYERUSDT', 'MOCAUSDT', 'EPTUSDT', 'OLUSDT', 'WUSDT', 'ARPAUSDT', 'IMXUSDT', 'IPUSDT', 'VANAUSDT', 'VVVUSDT', 'CAKEUSDT', 'SIGNUSDT', 'SUNUSDT', 'SWARMSUSDT', 'AERGOUSDT', 'THETAUSDT', 'UXLINKUSDT', 'PARTIUSDT', 'MEUSDT', 'MELANIAUSDT', 'UMAUSDT', 'AXSUSDT', 'GORKUSDT', 'SAROSUSDT', 'AVAAIUSDT']
  # Можно расширить список
  print(f"Загрузка и обработка данных для {len(symbols_to_train)} символов...")

  all_features = []
  all_targets = []

  # Создаем задачи для параллельной обработки каждого символа
  tasks = [feature_engineer.create_multi_timeframe_features(symbol, data_fetcher) for symbol in symbols_to_train]
  results = await asyncio.gather(*tasks)

  for features, _ in results:  # Нам не нужны метки классификации, только признаки
    if features is not None and 'volatility_20' in features.columns:
      # Создаем регрессионную цель (будущую волатильность)
      target = features['volatility_20'].shift(-5)

      # Объединяем признаки и цель, удаляем строки с пропусками
      combined = pd.concat([features, target.rename('target')], axis=1)
      combined.dropna(inplace=True)

      if not combined.empty:
        all_features.append(combined.drop(columns=['target']))
        all_targets.append(combined['target'])

  if not all_features:
    print("❌ Не удалось создать обучающие данные ни для одного символа.")
    await connector.close()
    return

  # Объединяем данные со всех символов в единый датасет
  X_train = pd.concat(all_features)
  y_train = pd.concat(all_targets)

  split_point = int(len(X_train) * 0.8)
  X_train_final, X_test_final = X_train[:split_point], X_train[split_point:]
  y_train_final, y_test_final = y_train[:split_point], y_train[split_point:]

  print(f"✅ Итоговый размер обучающей выборки: {len(X_train)} наблюдений, {len(X_train.columns)} признаков.")

  # 3. Инициализация и обучение системы
  print("\nИнициализация и обучение системы прогнозирования волатильности...")
  # Ваш скрипт `volatility_prediction_system.py` содержит более сложную систему,
  # которая обучает сразу несколько моделей. Мы будем использовать ее.
  system = VolatilityPredictionSystem(
    model_type=ModelType.XGBOOST,  # Используем ансамбль моделей для надежности
    prediction_horizon=5
  )

  # Передаем уже готовые X и y в адаптированный метод initialize
  # (потребуется небольшое изменение в volatility_system.py)
  init_result = system.initialize(X_train_final, y_train_final, X_test_final, y_test_final)

  if init_result['status'] != 'success':
    print(f"❌ Ошибка обучения: {init_result.get('error')}")
    await connector.close()
    return

  print("\n--- Отчет об обучении моделей волатильности ---")
  if init_result.get('model_scores'):
    for model_name, scores in init_result['model_scores'].items():
      if 'error' not in scores:
        print(f"  Модель: {model_name:<18} | R²: {scores.get('r2', 0):.4f} | RMSE: {scores.get('rmse', 0):.6f}")

  # 4. Сохранение всего обученного объекта системы
  save_path = "ml_models/volatility_system.pkl"
  joblib.dump(system, save_path)
  print(f"\n✅ Система прогнозирования волатильности успешно обучена и сохранена в: {save_path}")

  await connector.close()


if __name__ == "__main__":
  asyncio.run(main())