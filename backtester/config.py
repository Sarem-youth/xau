import os
from datetime import datetime

CONFIG = {
    'start_date': datetime(2024, 1, 1, 0, 0, 0),  # Start at beginning of day
    'end_date': datetime(2024, 1, 4, 23, 59, 59),  # End at end of day
    'symbol': 'XAUUSD',
    'timeframes': ['5m', '4h', '1d'],
    'ma_periods': [20, 50],
    'risk_reward_ratio': 1.5,        # Reduced from 3.0 for more realistic targets
    'risk_per_trade': 0.01,          # Reduced risk per trade
    'profit_levels': [               # Better partial profit structure
        {'size': 0.5, 'target': 100},  # Take half at 10 pips
        {'size': 0.3, 'target': 200},  # Take 30% at 20 pips
        {'size': 0.2, 'target': 300},  # Let 20% run to 30 pips
    ],
    'breakeven_rules': {
        'min_profit_ticks': 50,      # Move to breakeven sooner
        'buffer_ticks': 20,          # Protect profits better
    },
    'trailing_stop': {
        'activation_ticks': 80,      # Start trailing earlier
        'trail_distance': 30,        # Tighter trail
        'step_size': 10,            # More frequent updates
    },
    'session_filters': {
        'london_open': '08:00',      # London session
        'london_close': '16:30',
        'ny_open': '13:30',          # New York session
        'ny_close': '20:00',
        'enable_session_filter': False,  # Disable session filter
    },
    'volatility_filter': {
        'min_atr_threshold': 0.0001,  # More lenient volatility requirements
        'max_atr_threshold': 0.01,
        'enable_volatility_filter': False,  # Disable initially
    },
    'position_scaling': {
        'enable_scaling': True,
        'scale_in_levels': [
            {'price': 0.5, 'size': 0.5},  # Add 50% at halfway to target
            {'price': 0.7, 'size': 0.3},  # Add 30% closer to target
        ],
    },
    'position_sizing': {
        'min_lot': 0.01,
        'max_lot': 0.1,
        'quality_tiers': {
            'excellent': {'min_score': 0.8, 'multiplier': 1.0},
            'good': {'min_score': 0.6, 'multiplier': 0.75},
            'moderate': {'min_score': 0.4, 'multiplier': 0.5},
            'poor': {'min_score': 0.0, 'multiplier': 0.25}
        },
        'adjustment_factors': {
            'high_volatility': 0.7,
            'counter_trend': 0.5,
            'weak_momentum': 0.6,
            'strong_momentum': 1.2
        }
    },
    'dynamic_targets': {
        'atr_multiplier': 2.0,       # Use ATR for dynamic targets
        'min_target_distance': 100,   # Minimum target distance
        'max_target_distance': 1000,  # Maximum target distance
    },
    'profit_target_ticks': 300,      # Smaller targets for quicker trades
    'partial_profit_ticks': 150,     # Take partial at 15 pips
    'partial_profit_size': 0.5,      # Close half position
    'breakeven_ticks': 100,          # Faster breakeven
    'stoch_rsi_periods': 14,
    'stoch_rsi_overbought': 0.85,    # More lenient
    'stoch_rsi_oversold': 0.15,
    'initial_balance': 100000,
    'min_volume_threshold': 1,       # Minimal volume requirement
    'consolidation_threshold': 0.0015,# Tighter consolidation definition
    'trend_strength_min': 3,         # More sensitive trend detection
    'min_stop_distance': 100,        # Tighter stops
    'max_stop_distance': 500,        # Allow wider stops
    'min_lot_size': 0.01,           # Smaller positions
    'max_spread_points': 50,         # Maximum allowed spread

    'trend_filters': {
        'min_ma_separation': 0.0001,  # More lenient trend requirements
        'momentum_threshold': 0.01,
        'enable_trend_filter': False,  # Disable initially
    },

    'stop_loss': {
        'atr_multiplier': 1.5,       # Tighter stops
        'min_distance': 100,         # Minimum 10 pip stop
        'max_distance': 200,         # Maximum 20 pip stop
    },

    'restrict_trading_hours': True,

    # Trading hours filter (all times in UTC)
    'trading_hours': {
        'start': '00:00',
        'end': '23:59',
        'enabled': True
    },

    # Market hours configuration (all times in UTC)
    'market_hours': {
        'forex': {
            'sunday_open': '22:00',    # Sunday open (Monday in Asia)
            'friday_close': '22:00',   # Friday close
            'daily_close_duration': 0,  # No break between sessions
        },
        'ignore_holidays': False,      # Set to True to skip known holidays
        'holidays': [
            # Add specific dates in 'YYYY-MM-DD' format if needed
        ],
        'include_start_date': True     # Always include start date regardless of market hours
    },

    # Trading hours restriction
    'restrict_trading_hours': True,
    'trading_hours': {
        'enabled': True,
        'sessions': [
            {'start': '00:00', 'end': '23:59', 'name': 'All Day Jan 1'},  # Special session for Jan 1
            {'start': '08:00', 'end': '16:30', 'name': 'London'},
            {'start': '13:30', 'end': '20:00', 'name': 'New York'}
        ]
    },

    # MT5 specific settings
    'mt5_account': int(os.getenv('MT5_ACCOUNT', '0')),
    'mt5_password': os.getenv('MT5_PASSWORD', ''),
    'mt5_server': os.getenv('MT5_SERVER', 'OctaFX-Demo'),
}