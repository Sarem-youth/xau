import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import pytz
from datetime import datetime, timedelta
from config import CONFIG

class DataLoader:
    def __init__(self):
        # Initialize MT5 connection
        if not mt5.initialize():
            print("MT5 initialization failed")
            mt5.shutdown()
            raise Exception("MT5 initialization failed")
        
        # Login to MT5
        authorized = mt5.login(
            CONFIG['mt5_account'],
            password=CONFIG['mt5_password'],
            server=CONFIG['mt5_server']
        )
        
        if not authorized:
            print("MT5 login failed")
            mt5.shutdown()
            raise Exception("MT5 login failed")
    
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
        utc_from = CONFIG['start_date'].replace(tzinfo=timezone)
        utc_to = CONFIG['end_date'].replace(tzinfo=timezone)
        
        rates = mt5.copy_rates_range(CONFIG['symbol'], tf, utc_from, utc_to)
        
        if rates is None:
            raise Exception("Failed to get data from MT5")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('timestamp', inplace=True)
        df.drop('time', axis=1, inplace=True)
        
        return df
    
    def add_indicators(self, df):
        # Add moving averages
        for period in CONFIG['ma_periods']:
            df[f'MA_{period}'] = df['close'].rolling(window=period).mean()
        
        # Add Stochastic RSI
        df['RSI'] = self._calculate_rsi(df['close'])
        df['StochRSI'] = self._calculate_stoch_rsi(df['RSI'])
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_stoch_rsi(self, rsi_values, period=14):
        min_rsi = rsi_values.rolling(window=period).min()
        max_rsi = rsi_values.rolling(window=period).max()
        return (rsi_values - min_rsi) / (max_rsi - min_rsi)
    
    def __del__(self):
        mt5.shutdown()
