import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import pytz
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import pytz
from datetime import datetime, timedelta
from xau.backtester.config import CONFIG

class DataLoader:
    def __init__(self):
        # Initialize MT5 connection
        if not mt5.initialize():
            print("MT5 initialization failed. Please make sure MetaTrader 5 is installed and running.")
            mt5.shutdown()
            raise Exception("MT5 initialization failed")
        
        # Validate MT5 credentials
        if not CONFIG['mt5_account'] or not CONFIG['mt5_password']:
            raise ValueError("MT5 credentials not found. Please check your .env file.")
        
        # Login to MT5
        authorized = mt5.login(
            CONFIG['mt5_account'],
            password=CONFIG['mt5_password'],
            server=CONFIG['mt5_server']
        )
        
        if not authorized:
            error = mt5.last_error()
            print(f"MT5 login failed. Error code: {error[0]}, Error message: {error[1]}")
            mt5.shutdown()
            raise Exception(f"MT5 login failed: {error[1]}")
    
    def fetch_data(self, timeframe):
        # Convert timeframe string to MT5 timeframe constant
        mt5_timeframes = {
            '1m': mt5.TIMEFRAME_M1,
            '5m': mt5.TIMEFRAME_M5,
            '4h': mt5.TIMEFRAME_H4,
            '1d': mt5.TIMEFRAME_D1
        }
        
        tf = mt5_timeframes.get(timeframe)
        if tf is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        # Get historical data from MT5
        timezone = pytz.timezone("Etc/UTC")
        
        # Add buffer days to ensure we have enough data for indicators
        buffer_days = 5  # Add 5 days of buffer
        utc_from = (CONFIG['start_date'] - timedelta(days=buffer_days)).replace(tzinfo=timezone)
        utc_to = (CONFIG['end_date'] + timedelta(days=1)).replace(tzinfo=timezone)  # Add one day to include full end date
        
        print(f"Fetching {timeframe} data from {utc_from} to {utc_to}")
        rates = mt5.copy_rates_range(CONFIG['symbol'], tf, utc_from, utc_to)
        
        if rates is None:
            raise Exception("Failed to get data from MT5")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('timestamp', inplace=True)
        df.drop('time', axis=1, inplace=True)
        
        # First filter by date range to ensure we have the full range including Jan 1st
        df = df[(df.index >= CONFIG['start_date']) & (df.index <= CONFIG['end_date'])]
        
        # Then apply market hours filter except for Jan 1st
        if CONFIG['restrict_trading_hours']:
            df['market_open'] = df.index.map(self._is_market_open)
            df['trading_hour'] = df.index.map(self._is_valid_trading_hour)
            df = df[df['market_open'] & df['trading_hour']]
            df = df.drop(['market_open', 'trading_hour'], axis=1)
        
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Loaded {len(df)} bars for {timeframe} after filtering")
        return df
    
    def add_indicators(self, df):
        """Add basic indicators with proper initialization"""
        # Use ffill/bfill instead of fillna with method
        df = df.copy()
        
        # Basic moving averages
        for period in [10, 20, 50]:
            df[f'MA_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
        
        # RSI calculation
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)
        
        # StochRSI calculation
        rsi_min = df['RSI'].rolling(window=14, min_periods=1).min()
        rsi_max = df['RSI'].rolling(window=14, min_periods=1).max()
        rsi_diff = rsi_max - rsi_min
        df['StochRSI'] = (df['RSI'] - rsi_min) / rsi_diff.replace(0, 1)
        df['StochRSI'] = df['StochRSI'].fillna(0.5)
        
        # Volume analysis
        df['volume_ma'] = df['tick_volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
        
        # ATR calculation
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(window=14, min_periods=1).mean()
        
        # Price swings
        df['high_10'] = df['high'].rolling(window=10, min_periods=1).max()
        df['low_10'] = df['low'].rolling(window=10, min_periods=1).min()
        df['momentum'] = (df['close'] - df['MA_20']) / df['MA_20'] * 100
        
        # Forward fill remaining NaN values
        df = df.ffill().bfill()
        
        # Debug prints
        print(f"\nIndicator check for {len(df)} bars:")
        print(f"MA20 range: {df['MA_20'].min():.2f} - {df['MA_20'].max():.2f}")
        print(f"RSI range: {df['RSI'].min():.2f} - {df['RSI'].max():.2f}")
        print(f"StochRSI range: {df['StochRSI'].min():.2f} - {df['StochRSI'].max():.2f}")
        print(f"ATR range: {df['atr'].min():.5f} - {df['atr'].max():.5f}")
        
        return df

    def _calculate_atr(self, df, period=14):
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        return tr.rolling(window=period, min_periods=1).mean()

    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gains / avg_losses.replace(0, np.nan)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Default to neutral RSI

    def _calculate_stoch_rsi(self, rsi_values, period=14):
        min_rsi = rsi_values.rolling(window=period, min_periods=1).min()
        max_rsi = rsi_values.rolling(window=period, min_periods=1).max()
        
        # Avoid division by zero
        denominator = (max_rsi - min_rsi).replace(0, np.nan)
        stoch_rsi = (rsi_values - min_rsi) / denominator
        
        return stoch_rsi.fillna(0.5)  # Default to middle value
    
    def load_data(self):
        # ...existing code...
        df = df[(df.index >= CONFIG['start_date']) & (df.index <= CONFIG['end_date'])]

        if CONFIG.get('restrict_trading_hours'):
            # Exclude weekends or other closed-market periods
            df = df[df.index.dayofweek < 5]
            # ...any additional holiday or market-closed filtering...

        # ...existing code...

    def _is_market_open(self, timestamp):
        """Check if market is open at given timestamp"""
        weekday = timestamp.weekday()
        current_time = timestamp.time()
        
        # Check if date is Jan 1st and include it regardless of time
        if timestamp.date() == CONFIG['start_date'].date():
            return True
            
        # Normal market hour checks
        if weekday == 5:  # Saturday
            return False
        elif weekday == 6:  # Sunday
            sunday_open = datetime.strptime(CONFIG['market_hours']['forex']['sunday_open'], '%H:%M').time()
            return current_time >= sunday_open
        elif weekday == 4:  # Friday
            friday_close = datetime.strptime(CONFIG['market_hours']['forex']['friday_close'], '%H:%M').time()
            return current_time <= friday_close
        return True

    def _is_valid_trading_hour(self, timestamp):
        """Check if current time is within allowed trading sessions"""
        if not CONFIG['trading_hours']['enabled']:
            return True
            
        current_time = timestamp.time()
        
        # Check each trading session
        for session in CONFIG['trading_hours']['sessions']:
            session_start = datetime.strptime(session['start'], '%H:%M').time()
            session_end = datetime.strptime(session['end'], '%H:%M').time()
            
            if session_start <= current_time <= session_end:
                return True
                
        return False

    def __del__(self):
        mt5.shutdown()
