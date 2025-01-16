import pandas as pd
import numpy as np
from binance.client import Client
from config import CONFIG

class DataLoader:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)
        
    def fetch_data(self, timeframe):
        klines = self.client.get_historical_klines(
            CONFIG['symbol'],
            timeframe,
            str(CONFIG['start_date']),
            str(CONFIG['end_date'])
        )
        
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 
                                         'volume', 'close_time', 'quote_volume', 'trades', 
                                         'taker_buy_base', 'taker_buy_quote', 'ignore'])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
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
