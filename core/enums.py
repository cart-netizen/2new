from enum import Enum

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    PENDING = "PENDING" # Добавим статус ожидания

class OrderType(Enum):
    MARKET = "Market"
    LIMIT = "Limit"

class Timeframe(Enum):
    ONE_MINUTE = "1"
    THREE_MINUTES = "3"
    FIVE_MINUTES = "5"
    FIFTEEN_MINUTES = "15"
    THIRTY_MINUTES = "30"
    ONE_HOUR = "60" # Bybit использует минуты для часовых таймфреймов
    TWO_HOURS = "120"
    FOUR_HOURS = "240"
    SIX_HOURS = "360"
    TWELVE_HOURS = "720"
    ONE_DAY = "D"
    ONE_WEEK = "W"
    ONE_MONTH = "M"