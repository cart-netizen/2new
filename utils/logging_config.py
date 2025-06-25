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
  Настраивает базовое логирование в консоль идемпотентно,
  чтобы избежать дублирования логов при повторных вызовах.
  """
  log_level = getattr(logging, log_level_str.upper(), logging.INFO)

  # Получаем корневой логгер, с которым будем работать
  root_logger = logging.getLogger()
  root_logger.setLevel(log_level)

  # ПРОВЕРКА И ОЧИСТКА: Удаляем все старые обработчики перед добавлением нового.
  # Это и есть ключ к решению проблемы дублирования.
  if root_logger.hasHandlers():
    root_logger.handlers.clear()

  # Создаем и настраиваем новый, единственный обработчик для консоли
  handler = logging.StreamHandler(sys.stdout)
  formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  )
  handler.setFormatter(formatter)

  # Добавляем наш единственный обработчик
  root_logger.addHandler(handler)

  # Этот флаг гарантирует, что сообщение ниже будет выведено в лог только один раз
  # при самом первом вызове функции.
  if not getattr(setup_logging, "has_run", False):
    logger = logging.getLogger(__name__)
    logger.info(f"Основное логирование настроено на уровень: {log_level_str}")
    setup_logging.has_run = True


def get_logger(name: str) -> logging.Logger:
  """
    Возвращает экземпляр логгера.
    """
  return logging.getLogger(name)