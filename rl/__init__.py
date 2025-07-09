"""
Модуль интеграции FinRL для обучения с подкреплением в торговле
"""

from .environment import BybitTradingEnvironment
from .finrl_agent import EnhancedRLAgent
from .reward_functions import RiskAdjustedRewardFunction
from .portfolio_manager import RLPortfolioManager
from .shadow_learning import ShadowTradingLearner

__all__ = [
    'BybitTradingEnvironment',
    'EnhancedRLAgent',
    'RiskAdjustedRewardFunction',
    'RLPortfolioManager',
    'ShadowTradingLearner'
]