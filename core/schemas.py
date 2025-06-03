from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
from core.enums import SignalType

@dataclass
class TradingSignal:
    signal_type: SignalType
    symbol: str
    price: Optional[float] # Цена генерации сигнала
    confidence: float
    strategy_name: str
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Добавим order_type и quantity для более полного сигнала
    order_type: Optional[str] = "Market" # e.g., OrderType.MARKET.value
    quantity_usdt: Optional[float] = None # Количество в USDT для расчета размера

@dataclass
class Order:
    order_id: str
    symbol: str
    order_type: str # Market, Limit
    side: str # Buy, Sell
    price: Optional[float] # Для Limit ордеров
    quantity: float
    status: str # New, PartiallyFilled, Filled, Canceled, Rejected
    timestamp: datetime
    avg_fill_price: Optional[float] = None
    commission: Optional[float] = None
    pnl: Optional[float] = None # Для закрытых позиций

@dataclass
class Position:
    symbol: str
    side: str # Buy (long), Sell (short)
    entry_price: float
    quantity: float
    leverage: int
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    liquidation_price: Optional[float] = None
    margin: Optional[float] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class RiskMetrics: # Сильно упрощено для начала
    total_balance_usdt: float = 0.0
    available_balance_usdt: float = 0.0
    unrealized_pnl_total: float = 0.0
    daily_loss_limit_pct: float = 0.02 # 2%
    current_daily_loss_usdt: float = 0.0
    max_position_size_pct: float = 0.10 # 10%