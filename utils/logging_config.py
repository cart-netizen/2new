import logging
import sys

def setup_logging(log_level_str: str = "INFO"):
    """
    Настраивает базовое логирование.
    """
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
            # Можно добавить FileHandler для записи в файл:
            # logging.FileHandler("trading_bot.log")
        ]
    )
    # Подавление слишком "шумных" логов от сторонних библиотек, если необходимо
    # logging.getLogger("some_library").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info(f"Логирование настроено на уровень: {log_level_str}")

def get_logger(name: str) -> logging.Logger:
    """
    Возвращает экземпляр логгера.
    """
    return logging.getLogger(name)