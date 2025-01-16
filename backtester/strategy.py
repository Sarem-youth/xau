import numpy as np
from config import CONFIG

class TradingStrategy:
    def __init__(self):
        self.positions = []
        self.trades = []
    
    def analyze_trend(self, df_daily, df_4h, df_5m):
        signals = []
        
        for idx in df_5m.index:
            if idx in df_daily.index:
                daily_trend = self._get_trend(df_daily.loc[idx])
                h4_trend = self._get_trend(df_4h.loc[idx])
                m5_trend = self._get_trend(df_5m.loc[idx])
                
                if self._check_entry_conditions(daily_trend, h4_trend, m5_trend, df_5m.loc[idx]):
                    entry_price = df_5m.loc[idx]['close']
                    stop_loss = self._calculate_stop_loss(df_5m, idx)
                    take_profit = self._calculate_take_profit(entry_price, stop_loss)
                    
                    signals.append({
                        'timestamp': idx,
                        'action': 'SELL' if daily_trend == 'DOWN' else 'BUY',
                        'price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'reason': self._generate_signal_reason(daily_trend, h4_trend, m5_trend)
                    })
        
        return signals
    
    def _get_trend(self, row):
        if row['MA_20'] > row['MA_50']:
            return 'UP'
        return 'DOWN'
    
    def _check_entry_conditions(self, daily_trend, h4_trend, m5_trend, row):
        if daily_trend == h4_trend and h4_trend == m5_trend:
            if row['StochRSI'] > 0.8 and daily_trend == 'DOWN':
                return True
            if row['StochRSI'] < 0.2 and daily_trend == 'UP':
                return True
        return False
    
    def _calculate_stop_loss(self, df, idx):
        lookback = 20
        if idx >= lookback:
            price_range = df.loc[:idx].tail(lookback)
            return price_range['low'].min() if df.loc[idx]['close'] > df.loc[idx]['MA_50'] else price_range['high'].max()
        return None
    
    def _calculate_take_profit(self, entry_price, stop_loss):
        risk = abs(entry_price - stop_loss)
        return entry_price + (risk * CONFIG['risk_reward_ratio']) if entry_price > stop_loss else entry_price - (risk * CONFIG['risk_reward_ratio'])
    
    def _generate_signal_reason(self, daily_trend, h4_trend, m5_trend):
        return f"Aligned trends: Daily={daily_trend}, 4H={h4_trend}, 5M={m5_trend}"
