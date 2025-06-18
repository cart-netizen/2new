# utils/logging_config.py

import logging
import sys


# --- НОВЫЙ БЛОК: Настройка выделенного логгера для трассировки сигналов ---
def setup_signal_logger():
  """Настраивает отдельный логгер для записи жизненного цикла сигнала в файл."""
  # Создаем логгер с уникальным именем
  signal_logger = logging.getLogger('SignalTrace')
  signal_logger.setLevel(logging.INFO)

  # Предотвращаем двойной вывод, если обработчики уже добавлены
  if signal_logger.hasHandlers():
    signal_logger.handlers.clear()

  # Создаем обработчик, который пишет в файл 'signal_trace.log'
  # mode='a' означает 'append' - дописывать в конец файла
  # encoding='utf-8' для корректной работы с кириллицей
  file_handler = logging.FileHandler('signal_trace.log', mode='a', encoding='utf-8')

  # Создаем форматтер, чтобы сообщения были красивыми и с временной меткой
  formatter = logging.Formatter('%(asctime)s - %(message)s')
  file_handler.setFormatter(formatter)

  # Добавляем обработчик к нашему логгеру
  signal_logger.addHandler(file_handler)

  # Важно: мы не отключаем распространение (propagate=False),
  # чтобы эти же сообщения попадали и в основной логгер (в консоль).


def setup_logging(log_level_str: str = "INFO"):
  """
    Настраивает базовое логирование в консоль.
    """
  log_level = getattr(logging, log_level_str.upper(), logging.INFO)

  # Настраиваем корневой логгер, который выводит все в консоль
  logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
      logging.StreamHandler(sys.stdout)
    ]
  )
  logger = logging.getLogger(__name__)
  logger.info(f"Основное логирование настроено на уровень: {log_level_str}")


def get_logger(name: str) -> logging.Logger:
  """
    Возвращает экземпляр логгера.
    """
  return logging.getLogger(name)