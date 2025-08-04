# НЕ КОММИТИТЕ ЭТОТ ФАЙЛ С РЕАЛЬНЫМИ КЛЮЧАМИ В GIT!
# Лучше использовать переменные окружения для реального использования.

# Для реального рынка (mainnet)
BYBIT_API_KEY = "qzUXBFV2Eu9WRcLtJQ"
BYBIT_API_SECRET = "nPKPrENpQk3LtOjHjhXatfnpIuaqse6LEa0R"

# Для тестовой сети (testnet)
BYBIT_TESTNET_API_KEY = "YOUR_TESTNET_API_KEY"
BYBIT_TESTNET_API_SECRET = "YOUR_TESTNET_API_SECRET"

# Выберите, какие ключи использовать (True для testnet, False для mainnet)
USE_TESTNET = False # ВАЖНО: Установите False для реальной торговли

API_KEY = BYBIT_TESTNET_API_KEY if USE_TESTNET else BYBIT_API_KEY
API_SECRET = BYBIT_TESTNET_API_SECRET if USE_TESTNET else BYBIT_API_SECRET

api_keys = {
    'api_key': BYBIT_API_KEY,
    'api_secret': BYBIT_API_SECRET
}