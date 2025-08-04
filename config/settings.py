import os

from config.api_keys import BYBIT_API_KEY, BYBIT_API_SECRET

# Общие настройки
LOG_LEVEL = "INFO"  # Уровни: DEBUG, INFO, WARNING, ERROR, CRITICAL
# BYBIT_API_URL = "https://api-testnet.bybit.com" if __import__('config.api_keys').USE_TESTNET else "https://api.bybit.com"
BYBIT_API_URL = "https://api.bybit.com"

# Настройки для подключения к Bybit
BYBIT_BASE_URL_MAINNET = "https://api.bybit.com"
# BYBIT_BASE_URL_TESTNET = "https://api-testnet.bybit.com" # Для тестирования
# Выбираем, какой URL использовать (например, на основе переменной окружения)
USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "False"
# BYBIT_API_URL = BYBIT_BASE_URL_TESTNET if USE_TESTNET else BYBIT_BASE_URL_MAINNET
BYBIT_API_URL = BYBIT_BASE_URL_MAINNET if USE_TESTNET else BYBIT_BASE_URL_MAINNET

# Путь к базе данных
DATABASE_PATH = "trading_data.db"

BYBIT_CATEGORY = "linear"

# WebSocket
BYBIT_WS_URL_PUBLIC_MAINNET = "wss://stream.bybit.com/v5/public/linear"
BYBIT_WS_URL_PUBLIC_TESTNET = "wss://stream-testnet.bybit.com/v5/public/linear"
BYBIT_WS_URL_PRIVATE_MAINNET = "wss://stream.bybit.com/v5/private"
BYBIT_WS_URL_PRIVATE_TESTNET = "wss://stream-testnet.bybit.com/v5/private"

BYBIT_WS_PUBLIC_URL = BYBIT_WS_URL_PUBLIC_MAINNET if USE_TESTNET else BYBIT_WS_URL_PUBLIC_TESTNET
BYBIT_WS_PRIVATE_URL = BYBIT_WS_URL_PRIVATE_MAINNET if USE_TESTNET else BYBIT_WS_URL_PRIVATE_TESTNET

api_keys = {
    'api_key': BYBIT_API_KEY,
    'api_secret': BYBIT_API_SECRET
}