# config/config_manager.py

import json
from pathlib import Path
from typing import Dict, Any

from utils.logging_config import get_logger

logger = get_logger(__name__)


class ConfigManager:
  """
  Класс для управления конфигурацией из JSON-файла.
  Обеспечивает безопасное чтение и запись настроек.
  """

  def __init__(self, config_path: str = "config.json"):
    self.config_path = Path(config_path)

  def load_config(self) -> Dict[str, Any]:
    """
    Загружает конфигурацию из файла.
    Если файл не найден, создает его с настройками по умолчанию.
    """
    if not self.config_path.exists():
      logger.warning(
        f"Файл конфигурации не найден. Создание нового файла '{self.config_path}' с настройками по умолчанию.")
      default_config = {
        "trade_settings": {"leverage": 10, "order_size_type": "percentage", "order_size_value": 1.0},
        "strategy_settings": {"signal_confidence_threshold": 0.55, "use_trend_filter": True, "ema_period": 200,
                              "use_adx_filter": True, "adx_threshold": 20, "use_volatility_filter": True,
                              "max_atr_percentage": 5.0},
        "general_settings": {"active_symbols": ["BTCUSDT", "ETHUSDT"], "monitoring_interval_seconds": 30}
      }
      self.save_config(default_config)
      return default_config

    try:
      with open(self.config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
      logger.error(f"Ошибка чтения файла конфигурации '{self.config_path}': {e}")
      # Возвращаем пустой словарь, чтобы избежать падения, но это плохая ситуация
      return {}

  def save_config(self, data: Dict[str, Any]):
    """
    Сохраняет переданный словарь с настройками в JSON-файл.
    """
    try:
      with open(self.config_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
      logger.info(f"Конфигурация успешно сохранена в '{self.config_path}'")
    except IOError as e:
      logger.error(f"Ошибка сохранения конфигурации в файл '{self.config_path}': {e}")