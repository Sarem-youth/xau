
from datetime import datetime

CONFIG = {
    'start_date': datetime(2024, 1, 1),
    'end_date': datetime(2024, 1, 7),
    'symbol': 'XAUUSD',
    'timeframes': ['5m', '4h', '1d'],
    'ma_periods': [20, 50],
    'risk_reward_ratio': 2,
    'risk_per_trade': 0.02,  # 2% risk per trade
    'stoch_rsi_periods': 14,
    'initial_balance': 100000
}