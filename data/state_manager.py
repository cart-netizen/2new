# data/state_manager.py

import json
import os
from datetime import datetime, date
from enum import Enum
from typing import Dict, Any, Optional, List

from core.schemas import RiskMetrics
from utils.logging_config import get_logger

logger = get_logger(__name__)

# --- НОВЫЙ КЛАСС-КОДИРОВЩИК ---
class CustomJSONEncoder(json.JSONEncoder):
    """
    Кастомный кодировщик JSON, который умеет обрабатывать
    объекты datetime и Enum.
    """
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value # Сохраняем только значение, например "BUY"
        return json.JSONEncoder.default(self, obj)
# --- КОНЕЦ НОВОГО КЛАССА ---


class StateManager:
    """
    Упрощенный менеджер состояний, использующий JSON-файл для избежания блокировок БД.
    """
    def __init__(self, state_file: str = "bot_status.json"):
        self.state_file = state_file

    def _read_state(self) -> Dict[str, Any]:
        """Читает состояние из JSON-файла."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Ошибка чтения файла состояния: {e}")
        # Возвращаем состояние по умолчанию, если файл не найден или поврежден
        return {'bot_status': {'status': 'stopped', 'pid': None}, 'latest_metrics': {}}

    def _write_state(self, state: Dict[str, Any]):
        """Записывает состояние в JSON-файл."""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
                json.dump(state, f, indent=4, cls=CustomJSONEncoder)
        except IOError as e:
            logger.error(f"Ошибка записи в файл состояния: {e}")

    # --- Публичные методы (теперь они СИНХРОННЫЕ) ---

    def set_status(self, status: str, pid: Optional[int] = None):
        """Устанавливает статус бота (running/stopped) и его PID."""
        state = self._read_state()
        state['bot_status'] = {'status': status, 'pid': pid}
        self._write_state(state)

    def get_status(self) -> Dict[str, Any]:
        """Получает текущий статус бота."""
        return self._read_state().get('bot_status', {'status': 'unknown', 'pid': None})

    def update_metrics(self, metrics: RiskMetrics):
        """Обновляет последние метрики баланса."""
        state = self._read_state()
        state['latest_metrics'] = metrics.__dict__
        self._write_state(state)

    def get_metrics(self) -> Optional[RiskMetrics]:
        """Получает последние метрики баланса."""
        metrics_dict = self._read_state().get('latest_metrics')
        return RiskMetrics(**metrics_dict) if metrics_dict else None

    def update_open_positions(self, positions: Dict[str, Any]):
        """Обновляет список открытых позиций в файле состояния."""
        state = self._read_state()
        # Преобразуем словарь объектов в список словарей для JSON-сериализации
        state['open_positions'] = list(positions.values())
        self._write_state(state)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Получает список открытых позиций из файла состояния."""
        return self._read_state().get('open_positions', [])

    def get_pending_signals(self) -> Dict[str, Any]:
        """Получает словарь сигналов, ожидающих входа."""
        return self._read_state().get('pending_entry_signals', {})

    def update_pending_signals(self, pending_signals: Dict[str, Any]):
        """Обновляет словарь сигналов, ожидающих входа."""
        state = self._read_state()
        state['pending_entry_signals'] = pending_signals
        self._write_state(state)

    def update_model_info(self, model_info: Dict[str, Any]):
        """Обновляет информацию о состоянии модели в файле состояния."""
        state = self._read_state()
        state['model_info'] = model_info
        self._write_state(state)

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Получает информацию о состоянии модели из файла состояния."""
        return self._read_state().get('model_info')

    def set_command(self, command: str, data: Any = None):
        """Устанавливает команду для выполнения ботом."""
        state = self._read_state()
        state['command'] = {'name': command, 'data': data, 'timestamp': datetime.now().isoformat()}
        self._write_state(state)

    def get_command(self) -> Optional[Dict[str, Any]]:
        """Читает команду."""
        return self._read_state().get('command')

    def clear_command(self):
        """Очищает команду после выполнения."""
        state = self._read_state()
        if 'command' in state:
            del state['command']
            self._write_state(state)

    def get_custom_data(self, key: str) -> Any:
        """Получает пользовательские данные по ключу"""
        try:
            state = self._read_state()
            custom_data = state.get('custom_data', {})
            return custom_data.get(key)
        except Exception as e:
            logger.error(f"Ошибка получения custom_data[{key}]: {e}")
            return None

    def set_custom_data(self, key: str, value: Any):
        """Устанавливает пользовательские данные"""
        try:
            state = self._read_state()
            if 'custom_data' not in state:
                state['custom_data'] = {}
            state['custom_data'][key] = value
            self._write_state(state)
        except Exception as e:
            logger.error(f"Ошибка установки custom_data[{key}]: {e}")
