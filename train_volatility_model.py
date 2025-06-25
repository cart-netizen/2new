# train_volatility_model.py
import asyncio
import os
import sys
from typing import List

import joblib
import pandas as pd
import numpy as np
import logging

from matplotlib import pyplot as plt

# Импортируем компоненты нашего бота
from core.data_fetcher import DataFetcher
from core.bybit_connector import BybitConnector, logger
from core.enums import Timeframe
from ml.volatility_system import VolatilityPredictionSystem, ModelType
from ml.feature_engineering import feature_engineer
# from utils.logging_config import setup_logging, get_logger
from sklearn.model_selection import train_test_split # Добавляем для разделения

# --- НАСТРОЙКА ЛОГИРОВАНИЯ (единая функция) ---
def setup_logging(log_level_str: str = "INFO"):
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    # Удаляем все предыдущие обработчики, чтобы избежать дублирования
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def _plot_predictions_vs_actual(self, predictions: List[float], actual: List[float],
                                    model_name: str, plots_dir: str):
        """График предсказанных vs фактических значений"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Scatter plot
        ax1.scatter(actual, predictions, alpha=0.6)
        ax1.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', lw=2)
        ax1.set_xlabel('Фактическая волатильность')
        ax1.set_ylabel('Предсказанная волатильность')
        ax1.set_title(f'Предсказания vs Факт: {model_name}')
        ax1.grid(True, alpha=0.3)

        # Временной ряд
        ax2.plot(actual, label='Фактическая', alpha=0.7)
        ax2.plot(predictions, label='Предсказанная', alpha=0.7)
        ax2.set_xlabel('Время')
        ax2.set_ylabel('Волатильность')
        ax2.set_title(f'Временной ряд: {model_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f'predictions_{model_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

async def main():
    """
    ФИНАЛЬНАЯ ВЕРСИЯ: Скрипт для обучения модели волатильности
    на МНОЖЕСТВЕ символов с использованием МУЛЬТИТАЙМФРЕЙМ-признаков.
    """
    logger = setup_logging("INFO")
    logger.info("=== ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ ВОЛАТИЛЬНОСТИ ===")

    # 1. Инициализация
    connector = BybitConnector()
    await connector.sync_time()
    data_fetcher = DataFetcher(connector, settings={})

    # 2. Загрузка и подготовка данных
    symbols_to_train = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'FARTCOINUSDT', 'ZKJUSDT', 'HYPEUSDT', 'DOGEUSDT', '1000PEPEUSDT', 'SUIUSDT', 'WIFUSDT', 'AAVEUSDT', 'ALTUSDT', 'SPXUSDT',
                      'ADAUSDT', 'TRXUSDT', 'UNIUSDT', 'LINKUSDT', 'POPCATUSDT', 'MAGICUSDT', 'AI16ZUSDT', 'JTOUSDT', 'TRUMPUSDT', 'ENAUSDT', 'VIRTUALUSDT', 'BCHUSDT', 'OPUSDT', 'AVAXUSDT', 'ARBUSDT', 'ANIMEUSDT', 'TUSDT', 'MOODENGUSDT', 'LAUNCHCOINUSDT', 'ONDOUSDT', 'RVNUSDT', 'INJUSDT', 'WLDUSDT',
'SNTUSDT', 'TIAUSDT', 'NEARUSDT', 'LTCUSDT', 'GALAUSDT', 'TRBUSDT', 'CRVUSDT', '1000BONKUSDT', 'COOKIEUSDT', 'MKRUSDT', 'TONUSDT', 'KAIAUSDT', 'SYRUPUSDT', 'AEROUSDT', 'DOTUSDT', 'TAOUSDT', 'BMTUSDT', 'BNBUSDT', 'JUPUSDT', 'INITUSDT', 'BRETTUSDT', 'KASUSDT', 'LAUSDT', 'NXPCUSDT', 'ETHFIUSDT', 'ORDIUSDT', 'HAEDALUSDT', 'VELOUSDT', 'WCTUSDT', 'PNUTUSDT', 'LDOUSDT', 'AXLUSDT', 'ALCHUSDT', 'HBARUSDT', 'AIXBTUSDT', 'MASKUSDT', '1000NEIROCTOUSDT', 'FORMUSDT', 'XAUTUSDT', 'GRIFFAINUSDT', 'AGTUSDT', 'SOPHUSDT', 'EIGENUSDT', 'APTUSDT', 'KAITOUSDT', 'ENSUSDT', 'CHILLGUYUSDT', 'SHIB1000USDT', 'GOATUSDT', 'SEIUSDT', 'POLUSDT', 'ATOMUSDT', 'OMUSDT', 'PENGUUSDT', 'DYDXUSDT', 'ICPUSDT', 'STRKUSDT', 'FLOCKUSDT', '1000FLOKIUSDT', 'FILUSDT', 'ZBCNUSDT', 'APEUSDT', 'XLMUSDT', 'SUSDT', 'ZEREBROUSDT', 'SWEATUSDT', 'GRASSUSDT', 'COMPUSDT', '1000000MOGUSDT', 'ETCUSDT', 'PAXGUSDT', 'MOVEUSDT', 'RENDERUSDT', 'REXUSDT', 'ALGOUSDT', 'JELLYJELLYUSDT', 'LPTUSDT', 'BERAUSDT', 'HUMAUSDT', 'PENDLEUSDT', 'ATHUSDT', 'STXUSDT', 'B3USDT', 'PEOPLEUSDT', 'ZROUSDT', 'HMSTRUSDT', 'MEWUSDT', 'MNTUSDT', 'GIGAUSDT', 'NEIROETHUSDT', 'SOONUSDT','PYTHUSDT', 'NOTUSDT', 'KMNOUSDT', 'SANDUSDT', 'XAIUSDT', 'THEUSDT', 'ARCUSDT', 'BDXNUSDT', 'SERAPHUSDT', 'BOMEUSDT', 'ORBSUSDT', 'JASMYUSDT', 'DEEPUSDT', 'SUNDOGUSDT', '1000RATSUSDT', 'MUBARAKUSDT', 'GMTUSDT', 'FIDAUSDT', 'PUMPBTCUSDT', 'ZKUSDT', 'GRTUSDT', 'XMRUSDT', '1000TURBOUSDT', 'USUALUSDT', 'ARUSDT', '10000SATSUSDT', 'TAIKOUSDT', 'TAIUSDT', 'BIGTIMEUSDT']


    logger.info(f"Подготовка обучающих данных для {len(symbols_to_train)} символов...")

    tasks = [feature_engineer.create_multi_timeframe_features(symbol, data_fetcher) for symbol in symbols_to_train]
    results = await asyncio.gather(*tasks)

    all_features = []
    all_targets = []
    for features, _ in results:
        if features is not None and 'volatility_20' in features.columns:
            target = features['volatility_20'].shift(-5).rename('target')
            combined = pd.concat([features, target], axis=1).dropna()
            if not combined.empty:
                all_features.append(combined.drop(columns=['target']))
                all_targets.append(combined['target'])

    if not all_features:
        logger.error("Не удалось создать обучающие данные ни для одного символа.")
        await connector.close()
        return

    X = pd.concat(all_features)
    y = pd.concat(all_targets)
    logger.info(f"✅ Итоговый размер данных: {len(X)} наблюдений, {len(X.columns)} признаков.")

    # 3. Разделение данных на обучающую и тестовую выборки
    # shuffle=False - важно для временных рядов, чтобы не перемешивать прошлое и будущее
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    logger.info(f"Данные разделены: {len(X_train)} для обучения, {len(X_test)} для теста.")

    # 4. Инициализация и обучение системы
    logger.info("\nИнициализация и обучение системы прогнозирования волатильности...")
    system = VolatilityPredictionSystem(
        model_type=ModelType.LIGHTGBM,
        prediction_horizon=5
    )

    # Передаем все четыре компонента в метод initialize
    init_result = system.initialize(X_train, y_train, X_test, y_test)

    if init_result['status'] != 'success':
        logger.error(f"❌ Ошибка обучения: {init_result.get('error')}")
        await connector.close()
        return

    logger.info("\n--- Отчет об обучении моделей волатильности (на тестовой выборке) ---")
    if init_result.get('model_scores'):
        for model_name, scores in init_result['model_scores'].items():
            if 'error' not in scores:
                print(f"  Модель: {model_name:<18} | R²: {scores.get('r2', 0):.4f} | RMSE: {scores.get('rmse', 0):.6f}")

    # 5. Сохранение всего обученного объекта системы
    save_path = "ml_models/volatility_system.pkl"
    joblib.dump(system, save_path)
    logger.info(f"\n✅ Система прогнозирования волатильности успешно обучена и сохранена в: {save_path}")



    await connector.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nОбучение прервано пользователем.")