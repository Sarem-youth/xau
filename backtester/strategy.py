import numpy as np
import pandas as pd
from backtester.config import CONFIG

class TradingStrategy:
    def __init__(self):
        self.positions = []
        self.trades = []
        self.trade_id = 0
        self.current_position = None
    
    def analyze_trend(self, df_daily, df_4h, df_5m):
        signals = []
        print("Starting trend analysis...")  # Debug print
        
        # Create copies and add indicators
        df_5m = df_5m.copy()
        df_4h = df_4h.copy()
        df_daily = df_daily.copy()
        
        # Pre-calculate values
        df_5m['volume_ma'] = df_5m['tick_volume'].rolling(window=20).mean()
        df_5m['volume_ma'] = df_5m['volume_ma'].fillna(df_5m['tick_volume'])
        df_5m['low_10'] = df_5m['low'].rolling(window=10).min()
        df_5m['high_10'] = df_5m['high'].rolling(window=10).max()
        df_5m['price_ma10'] = df_5m['close'].rolling(window=10).mean()
        df_5m['momentum'] = (df_5m['close'] - df_5m['price_ma10']) / df_5m['close'] * 100
        
        # Add pattern recognition data
        df_5m['prev_open'] = df_5m['open'].shift(1)
        df_5m['prev_close'] = df_5m['close'].shift(1)
        df_5m['prev_high'] = df_5m['high'].shift(1)
        df_5m['prev_low'] = df_5m['low'].shift(1)
        
        # Add volatility metrics
        df_5m['atr_ma'] = df_5m['atr'].rolling(window=14).mean()
        df_5m['volume_ratio'] = df_5m['tick_volume'] / df_5m['volume_ma']
        
        # Fill NaN values
        df_5m = df_5m.fillna(method='bfill').fillna(method='ffill')
        
        # Ensure data alignment
        df_4h = df_4h.reindex(df_5m.index, method='ffill')
        df_daily = df_daily.reindex(df_5m.index, method='ffill')
        
        for idx in df_5m.index:
            current_bar = df_5m.loc[idx]
            
            # Check for exit conditions if we have an open position
            if self.current_position:
                if self._check_exit_conditions(self.current_position, current_bar):
                    exit_signal = self._generate_exit_signal(self.current_position, current_bar)
                    signals.append(exit_signal)
                    self._record_trade(exit_signal)
                    print(f"Exit signal generated at {idx}, PnL: {exit_signal['pnl']}")  # Debug print
                    self.current_position = None
            
            # Look for new entry only if no position is open
            if not self.current_position:
                daily_trend = self._get_trend(df_daily.loc[idx])
                h4_trend = self._get_trend(df_4h.loc[idx])
                m5_trend = self._get_trend(current_bar)
                
                if self._check_entry_conditions(daily_trend, h4_trend, m5_trend, current_bar):
                    entry_signal = self._generate_entry_signal(
                        current_bar,
                        'SELL' if daily_trend == 'DOWN' else 'BUY',
                        daily_trend, h4_trend, m5_trend
                    )
                    signals.append(entry_signal)
                    self.current_position = entry_signal
                    print(f"Entry signal generated at {idx}, Direction: {entry_signal['action']}")  # Debug print
        
        print(f"Analysis complete. Found {len(self.trades)} trades")  # Debug print
        return signals

    def _generate_exit_signal(self, position, current_bar, reason="Target reached or stop loss hit"):
        """Enhanced exit signal with descriptive reasons"""
        # Calculate exit price and PnL first
        exit_price = current_bar['close']
        points_diff = (exit_price - position['entry_price']) if position['action'] == 'BUY' \
                     else (position['entry_price'] - exit_price)
        pnl = points_diff * 100 * position['lot_size']  # Multiply by 100 for gold's point value

        # Create professional exit description
        exit_details = self._create_professional_exit_description(position, current_bar)
        
        return {
            'id': position['id'],
            'timestamp': current_bar.name,
            'action': 'EXIT',
            'entry_time': position['timestamp'],
            'exit_time': current_bar.name,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'lot_size': position['lot_size'],
            'pnl': pnl,
            'exit_type': exit_details['type'],
            'trigger': exit_details['description'],
            'market_condition': exit_details['market_state'],
            'indicators_at_exit': exit_details['technical_data']
        }

    def _create_professional_exit_description(self, position, current_bar):
        """Create detailed, professional exit description"""
        if self._is_strong_reversal(current_bar, position['action']):
            exit_type = "Price Reversal Signal"
            description = self._get_reversal_description(current_bar, position['action'])
        elif self._conditions_deteriorating(current_bar, position):
            exit_type = "Market Condition Change"
            description = self._get_deterioration_description(current_bar, position)
        elif self._hit_stop_loss(position, current_bar):
            exit_type = "Risk Management - Stop Loss"
            description = (f"Price action triggered protective stop at {position['stop_loss']:.2f}. "
                         f"Market moved {abs(current_bar['close'] - position['entry_price']):.2f} points against position.")
        else:
            exit_type = "Technical Exit"
            description = "Multiple technical factors signaled optimal exit point"

        # Build detailed market state description
        market_state = (
            f"Market Structure: {self._get_market_structure(current_bar)}\n"
            f"Trend State: {self._get_trend_state(current_bar)}\n"
            f"Volatility: {self._get_volatility_state(current_bar)}\n"
            f"Volume: {self._get_volume_state(current_bar)}"
        )

        # Compile technical indicators
        technical_data = {
            'rsi': current_bar['RSI'],
            'price_action': self._get_price_action_description(current_bar),
            'volatility': current_bar['atr'],
            'momentum': current_bar['momentum'],
            'ma_alignment': self._get_ma_alignment_state(current_bar)
        }

        return {
            'type': exit_type,
            'description': description,
            'market_state': market_state,
            'technical_data': technical_data
        }

    def _get_market_structure(self, bar):
        """Analyze overall market structure"""
        if bar['close'] > bar['MA_20'] > bar['MA_50']:
            return "Bullish trend structure with strong momentum"
        elif bar['close'] < bar['MA_20'] < bar['MA_50']:
            return "Bearish trend structure with strong momentum"
        elif bar['close'] > bar['MA_20']:
            return "Short-term bullish structure"
        else:
            return "Short-term bearish structure"

    def _get_trend_state(self, bar):
        """Get detailed trend state description"""
        trend_strength = abs(bar['momentum'])
        if trend_strength > 2.0:
            strength = "Strong"
        elif trend_strength > 1.0:
            strength = "Moderate"
        else:
            strength = "Weak"

        direction = "Upward" if bar['close'] > bar['MA_20'] else "Downward"
        return f"{strength} {direction} trend"

    def _get_volatility_state(self, bar):
        """Get volatility state description"""
        if self._is_high_volatility(bar):
            return "Elevated volatility conditions"
        return "Normal volatility conditions"

    def _get_volume_state(self, bar):
        """Get volume analysis description"""
        if bar['volume_ratio'] > 2.0:
            return "Significantly above average volume"
        elif bar['volume_ratio'] > 1.5:
            return "Above average volume"
        elif bar['volume_ratio'] < 0.7:
            return "Below average volume"
        else:
            return "Normal volume conditions"

    def _get_ma_alignment_state(self, bar):
        """Get moving average alignment description"""
        if bar['MA_20'] > bar['MA_50']:
            separation = (bar['MA_20'] - bar['MA_50']) / bar['MA_50'] * 100
            if separation > 0.1:
                return "Strong bullish MA alignment"
            return "Slight bullish MA alignment"
        else:
            separation = (bar['MA_50'] - bar['MA_20']) / bar['MA_50'] * 100
            if separation > 0.1:
                return "Strong bearish MA alignment"
            return "Slight bearish MA alignment"

    def _record_trade(self, exit_signal):
        """Enhanced trade recording with detailed metadata"""
        trade_details = {
            'id': exit_signal['id'],
            'entry_time': exit_signal['entry_time'],
            'exit_time': exit_signal['exit_time'],
            'entry_price': exit_signal['entry_price'],
            'exit_price': exit_signal['exit_price'],
            'action': self.current_position['action'],
            'lot_size': self.current_position['lot_size'],
            'pnl': exit_signal['pnl'],
            'balance': CONFIG['initial_balance'] + sum(t['pnl'] for t in self.trades) + exit_signal['pnl'],
            
            # Add new detailed metadata
            'entry_reason': {
                'trend_alignment': self.current_position.get('trend_alignment', ''),
                'pattern_detected': self.current_position.get('pattern_detected', ''),
                'signal_quality': self.current_position.get('signal_quality', 0),
                'market_context': self.current_position.get('market_context', ''),
                'key_levels': self.current_position.get('key_levels', []),
                'indicators': {
                    'rsi': self.current_position.get('entry_rsi', 0),
                    'stoch_rsi': self.current_position.get('entry_stoch_rsi', 0),
                    'ma_alignment': self.current_position.get('ma_alignment', ''),
                    'momentum': self.current_position.get('momentum', 0)
                }
            },
            'exit_reason': {
                'type': exit_signal.get('exit_type', 'unknown'),
                'trigger': exit_signal.get('trigger', ''),
                'market_condition': exit_signal.get('market_condition', ''),
                'indicators_at_exit': {
                    'rsi': exit_signal.get('exit_rsi', 0),
                    'price_action': exit_signal.get('price_action', ''),
                    'volatility': exit_signal.get('volatility', 0)
                }
            },
            'trade_metrics': {
                'risk_reward': self.current_position.get('risk_reward', 0),
                'risk_amount': self.current_position.get('risk_amount', 0),
                'max_favorable_excursion': self.current_position.get('max_favorable', 0),
                'max_adverse_excursion': self.current_position.get('max_adverse', 0),
                'time_in_trade': (exit_signal['exit_time'] - exit_signal['entry_time']).total_seconds() / 3600,
                'profit_factor': exit_signal['pnl'] / self.current_position.get('risk_amount', 1) if exit_signal['pnl'] > 0 else 0
            }
        }
        
        self.trades.append(trade_details)
        print(f"Trade recorded: {exit_signal}")

    def _get_reversal_description(self, bar, position_type):
        """Get detailed reversal description"""
        reasons = []
        
        if self._is_engulfing_pattern(bar):
            reasons.append(f"{'Bearish' if position_type == 'BUY' else 'Bullish'} engulfing pattern")
        
        if self._is_pin_bar(bar):
            reasons.append(f"{'Bearish' if position_type == 'BUY' else 'Bullish'} pin bar")
        
        if bar['RSI'] > 70 and position_type == 'BUY':
            reasons.append("Overbought RSI condition")
        elif bar['RSI'] < 30 and position_type == 'SELL':
            reasons.append("Oversold RSI condition")
        
        if bar['volume_ratio'] > 2.0:
            reasons.append("High volume spike")
        
        return " with ".join(reasons) if reasons else "Price action reversal"

    def _get_deterioration_description(self, bar, position):
        """Get detailed market deterioration description"""
        reasons = []
        
        if position['action'] == 'BUY':
            if bar['close'] < bar['MA_20']:
                reasons.append("Price broke below 20MA")
            if bar['MA_20'] < bar['MA_50']:
                reasons.append("20MA crossed below 50MA")
        else:
            if bar['close'] > bar['MA_20']:
                reasons.append("Price broke above 20MA")
            if bar['MA_20'] > bar['MA_50']:
                reasons.append("20MA crossed above 50MA")
        
        if abs(bar['momentum']) < abs(bar['momentum']) * 0.5:
            reasons.append("Momentum weakening")
        
        if bar['volume_ratio'] < 0.7:
            reasons.append("Volume declining")
        
        if self._is_high_volatility(bar):
            reasons.append("Volatility increasing")
        
        return ", ".join(reasons) if reasons else "Deteriorating market conditions"

    def _get_market_description(self, bar):
        """Get current market condition description"""
        conditions = []
        
        # Trend description
        if bar['close'] > bar['MA_20'] > bar['MA_50']:
            conditions.append("Strong uptrend")
        elif bar['close'] < bar['MA_20'] < bar['MA_50']:
            conditions.append("Strong downtrend")
        elif bar['close'] > bar['MA_20']:
            conditions.append("Short-term bullish")
        else:
            conditions.append("Short-term bearish")
        
        # Volatility description
        if self._is_high_volatility(bar):
            conditions.append("high volatility")
        else:
            conditions.append("normal volatility")
        
        # Volume description
        if bar['volume_ratio'] > 1.5:
            conditions.append("strong volume")
        elif bar['volume_ratio'] < 0.7:
            conditions.append("weak volume")
        
        return " with ".join(conditions)

    def _get_price_action_description(self, bar):
        """Get descriptive price action summary"""
        body_size = abs(bar['close'] - bar['open'])
        upper_wick = bar['high'] - max(bar['open'], bar['close'])
        lower_wick = min(bar['open'], bar['close']) - bar['low']
        
        if bar['close'] > bar['open']:
            color = "bullish"
        else:
            color = "bearish"
        
        if body_size < (bar['high'] - bar['low']) * 0.3:
            size = "doji"
        elif body_size > bar['atr']:
            size = "large"
        else:
            size = "normal"
        
        if upper_wick > body_size * 2:
            wick = "long upper wick"
        elif lower_wick > body_size * 2:
            wick = "long lower wick"
        else:
            wick = "normal wicks"
        
        return f"{size} {color} candle with {wick}"

    def _get_trend(self, row):
        """Simplified trend determination"""
        if row['close'] > row['MA_20']:
            return 'UP'
        return 'DOWN'
    
    def _check_entry_conditions(self, daily_trend, h4_trend, m5_trend, row):
        """Simplified entry conditions for debugging"""
        try:
            print(f"\nChecking entry at {row.name}:")
            print(f"Price: {row['close']:.2f}, MA20: {row['MA_20']:.2f}")
            print(f"StochRSI: {row['StochRSI']:.2f}, RSI: {row['RSI']:.2f}")
            
            # Basic trend check
            trend_aligned = (daily_trend == h4_trend)  # Only require daily and H4 alignment
            
            # Entry logic
            if h4_trend == 'DOWN':
                entry_condition = (
                    row['StochRSI'] > CONFIG['stoch_rsi_overbought'] and 
                    row['close'] < row['MA_20'] and
                    trend_aligned
                )
            else:  # UP trend
                entry_condition = (
                    row['StochRSI'] < CONFIG['stoch_rsi_oversold'] and 
                    row['close'] > row['MA_20'] and
                    trend_aligned
                )
            
            if entry_condition:
                print(f"Entry signal found! Direction: {h4_trend}")
            return entry_condition

        except Exception as e:
            print(f"Entry check error: {e}")
            return False

    def _check_trend_alignment(self, daily_trend, h4_trend, m5_trend):
        """Ensure strong trend alignment"""
        if any(trend == 'NEUTRAL' for trend in [daily_trend, h4_trend, m5_trend]):
            return False
            
        return daily_trend == h4_trend == m5_trend

    def _check_exit_conditions(self, position, current_bar):
        """Enhanced exit conditions"""
        if not position:
            return False
            
        # Calculate current profit/loss
        profit_pips = self._calculate_profit_pips(position, current_bar)
        
        # Fast exit on strong reversal
        if self._is_strong_reversal(current_bar, position['action']):
            return True
        
        # Exit on deteriorating market conditions
        if self._conditions_deteriorating(current_bar, position):
            return True
        
        # Dynamic stop loss adjustment
        if self._should_adjust_stop(position, current_bar):
            self._adjust_stop_loss(position, current_bar)
        
        # Check if stop loss is hit
        if self._hit_stop_loss(position, current_bar):
            return True
        
        return False

    def _calculate_dynamic_trailing_stop(self, position, current_bar):
        """Calculate trailing stop based on volatility"""
        atr = current_bar['atr']
        trail_distance = max(
            CONFIG['trailing_stop']['trail_distance'] / 10000,
            atr * 1.5
        )
        
        if position['action'] == 'BUY':
            return current_bar['close'] - trail_distance
        else:
            return current_bar['close'] + trail_distance

    def _calculate_breakeven_level(self, position):
        """Calculate breakeven level with buffer"""
        rules = CONFIG['breakeven_rules']
        buffer_pips = rules['buffer_ticks'] / 10000
        
        if position['action'] == 'BUY':
            return position['entry_price'] + buffer_pips
        else:
            return position['entry_price'] - buffer_pips

    def _should_trail_stop(self, profit_pips):
        """Check if we should start trailing stop"""
        return profit_pips >= CONFIG['trailing_stop']['activation_ticks']

    def _calculate_trailing_stop(self, position, current_bar, profit_pips):
        """Calculate trailing stop with step-based movement"""
        trail_cfg = CONFIG['trailing_stop']
        
        # Calculate number of steps moved
        steps_moved = int((profit_pips - trail_cfg['activation_ticks']) / trail_cfg['step_size'])
        trail_distance = trail_cfg['trail_distance'] / 10000  # Convert to price points
        
        if position['action'] == 'BUY':
            return max(
                position['stop_loss'],
                current_bar['close'] - trail_distance - (steps_moved * trail_cfg['step_size'] / 10000)
            )
        else:
            return min(
                position['stop_loss'],
                current_bar['close'] + trail_distance + (steps_moved * trail_cfg['step_size'] / 10000)
            )

    def _check_scale_in_conditions(self, position, current_bar):
        """Check if we should add to position"""
        if not position.get('scaled_in'):
            for level in CONFIG['position_scaling']['scale_in_levels']:
                price_level = position['entry_price'] + (
                    (position['take_profit'] - position['entry_price']) * level['price']
                )
                
                if (position['action'] == 'BUY' and current_bar['close'] >= price_level) or \
                   (position['action'] == 'SELL' and current_bar['close'] <= price_level):
                    self._scale_in_position(position, level['size'], current_bar)
                    position['scaled_in'] = True

    def _scale_in_position(self, position, scale_size, current_bar):
        """Add to existing position"""
        additional_size = position['original_lot_size'] * scale_size
        position['lot_size'] += additional_size
        print(f"Scaling in: Adding {additional_size} lots at {current_bar['close']}")

    def _check_session_filter(self, timestamp):
        """Check if we're in valid trading sessions"""
        time = timestamp.strftime('%H:%M')
        sessions = CONFIG['session_filters']
        
        in_london = sessions['london_open'] <= time <= sessions['london_close']
        in_ny = sessions['ny_open'] <= time <= sessions['ny_close']
        
        return in_london or in_ny

    def _reached_breakeven_level(self, position, current_bar):
        ticks_moved = abs(current_bar['close'] - position['entry_price']) * 10000
        return ticks_moved >= CONFIG['breakeven_ticks']
    
    def _calculate_stop_loss(self, row, action):
        """Enhanced stop loss calculation"""
        atr = row['atr']
        entry_price = row['close']
        
        # Use ATR for dynamic stop distance
        stop_distance = max(2 * atr, CONFIG['min_stop_distance'] / 10000)
        stop_distance = min(stop_distance, CONFIG['max_stop_distance'] / 10000)
        
        if action == 'BUY':
            return min(
                entry_price - stop_distance,
                row['low_10']  # Use recent low as maximum stop distance
            )
        else:
            return max(
                entry_price + stop_distance,
                row['high_10']  # Use recent high as maximum stop distance
            )
    
    def _calculate_atr(self, row, period=14):
        """Calculate Average True Range for dynamic stops"""
        high = row['high']
        low = row['low']
        close = row['close']
        
        tr = max([
            high - low,  # Current high-low range
            abs(high - close),  # Distance from current high to previous close
            abs(low - close)   # Distance from current low to previous close
        ])
        
        return tr

    def _calculate_lot_size(self, row, risk_amount, risk_pips):
        """Advanced dynamic lot size calculation"""
        # Get trade quality score (0-1)
        trend_quality = self._calculate_trend_quality(row)
        volatility_score = self._calculate_volatility_score(row)
        momentum_score = self._calculate_momentum_quality(row)
        pattern_score = self._calculate_pattern_quality(row)
        
        # Combined quality score (weighted)
        quality_score = (
            trend_quality * 0.35 +
            volatility_score * 0.25 +
            momentum_score * 0.25 +
            pattern_score * 0.15
        )
        
        # Base lot size calculation
        base_lot = risk_amount / (risk_pips * 10)  # Standard formula
        
        # Apply quality multiplier
        if quality_score >= 0.8:  # High probability setup
            lot_multiplier = 1.0  # Full size
        elif quality_score >= 0.6:  # Good setup
            lot_multiplier = 0.75  # 75% size
        elif quality_score >= 0.4:  # Moderate setup
            lot_multiplier = 0.5   # Half size
        else:  # Lower quality setup
            lot_multiplier = 0.25  # Quarter size
        
        # Apply market condition adjustments
        if self._is_high_volatility(row):
            lot_multiplier *= 0.7  # Reduce size in high volatility
        
        if self._is_against_major_trend(row):
            lot_multiplier *= 0.5  # Reduce size for counter-trend trades
        
        # Calculate final lot size
        final_lot = base_lot * lot_multiplier
        
        # Apply limits and round to 0.01
        final_lot = max(CONFIG['min_lot_size'], 
                       min(CONFIG['position_sizing']['max_lot'], 
                           round(final_lot, 2)))
        
        return final_lot

    def _calculate_trend_quality(self, row):
        """Calculate trend strength and quality"""
        score = 0
        
        # MA alignment check
        if row['MA_20'] > row['MA_50']:
            ma_diff = (row['MA_20'] - row['MA_50']) / row['MA_50']
            score += min(ma_diff * 100, 0.4)  # Max 0.4 from MA alignment
        
        # Momentum strength
        momentum_str = abs(row['momentum'])
        score += min(momentum_str / 100, 0.3)  # Max 0.3 from momentum
        
        # Volume confirmation
        if row['volume_ratio'] > 1.5:
            score += 0.3  # Max 0.3 from volume
        
        return min(score, 1.0)

    def _calculate_volatility_score(self, row):
        """Score trade based on volatility conditions"""
        atr = row['atr']
        optimal_atr = 0.001  # Optimal ATR value
        
        # Calculate how close current ATR is to optimal
        volatility_score = 1 - min(abs(atr - optimal_atr) / optimal_atr, 1)
        
        # Adjust for extreme volatility
        if atr > optimal_atr * 2:
            volatility_score *= 0.5
        
        return volatility_score

    def _calculate_momentum_quality(self, row):
        """Evaluate momentum quality"""
        score = 0
        
        # RSI alignment
        if row['RSI'] > 70 or row['RSI'] < 30:
            score += 0.4
        
        # StochRSI confirmation
        if (row['RSI'] > 70 and row['StochRSI'] > 0.8) or \
           (row['RSI'] < 30 and row['StochRSI'] < 0.2):
            score += 0.3
        
        # Momentum strength
        momentum_change = abs(row['momentum'])
        score += min(momentum_change / 50, 0.3)
        
        return min(score, 1.0)

    def _calculate_pattern_quality(self, row):
        """Evaluate chart pattern quality"""
        score = 0
        
        # Price action patterns
        if self._is_engulfing_pattern(row):
            score += 0.4
        elif self._is_pin_bar(row):
            score += 0.3
        
        # Support/Resistance proximity
        if self._near_key_level(row):
            score += 0.3
        
        # Candlestick size
        body_size = abs(row['close'] - row['open'])
        avg_body = row['atr'] * 0.5
        if body_size > avg_body:
            score += 0.3
        
        return min(score, 1.0)

    def _generate_entry_signal(self, row, action, daily_trend, h4_trend, m5_trend):
        self.trade_id += 1
        entry_price = row['close']
        stop_loss = self._calculate_stop_loss(row, action)
        risk_pips = abs(entry_price - stop_loss) * 10000  # Convert to pips
        risk_amount = CONFIG['initial_balance'] * CONFIG['risk_per_trade']
        lot_size = self._calculate_lot_size(row, risk_amount, risk_pips)
        
        # Calculate take profit based on risk-reward ratio
        take_profit = entry_price + (risk_pips * CONFIG['risk_reward_ratio'] / 10000) if action == 'BUY' \
                     else entry_price - (risk_pips * CONFIG['risk_reward_ratio'] / 10000)
        
        signal = {
            'id': self.trade_id,
            'timestamp': row.name,
            'action': action,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'lot_size': lot_size,
            'position_size': lot_size,  # Use lot_size as position_size
            'reason': f"Aligned trends: Daily={daily_trend}, 4H={h4_trend}, 5M={m5_trend}"
        }
        
        # Add detailed entry analysis
        entry_analysis = self._create_entry_analysis(row, action, daily_trend, h4_trend, m5_trend)
        signal.update({
            'entry_analysis': entry_analysis,
            'modifications': [],  # Track all trade modifications
        })
        
        return signal

    def _create_entry_analysis(self, row, action, daily_trend, h4_trend, m5_trend):
        """Create detailed entry analysis"""
        return {
            'setup_type': self._determine_setup_type(row, action),
            'trend_analysis': {
                'daily': daily_trend,
                'h4': h4_trend,
                'm5': m5_trend,
                'alignment': self._analyze_trend_alignment(daily_trend, h4_trend, m5_trend),
                'strength': self._calculate_trend_strength(row)
            },
            'technical_signals': {
                'rsi': row['RSI'],
                'stochastic': row['StochRSI'],
                'momentum': row['momentum'],
                'volume_context': self._analyze_volume_context(row)
            },
            'price_action': {
                'pattern': self._identify_candlestick_pattern(row),
                'context': self._analyze_price_context(row)
            },
            'support_resistance': self._identify_key_levels(row),
            'risk_metrics': {
                'risk_reward': self._calculate_risk_reward(row),
                'stop_distance': self._calculate_stop_distance(row),
                'quality_score': self._calculate_setup_quality(row)
            }
        }

    def _calculate_profit_pips(self, position, current_bar):
        """Calculate current profit in pips"""
        profit_points = (current_bar['close'] - position['entry_price']) if position['action'] == 'BUY' \
                       else (position['entry_price'] - current_bar['close'])
        return profit_points * 10000  # Convert to pips

    def _hit_stop_loss(self, position, current_bar):
        """Check if stop loss is hit"""
        if position['action'] == 'BUY':
            return current_bar['low'] <= position['stop_loss']
        else:  # SELL position
            return current_bar['high'] >= position['stop_loss']

    def _should_move_to_breakeven(self, profit_pips):
        """Check if position should be moved to breakeven"""
        return profit_pips >= CONFIG['breakeven_rules']['min_profit_ticks']

    def _calculate_partial_exit_size(self, position, profit_pips):
        """Calculate position size for partial exit"""
        for level in CONFIG['profit_levels']:
            if profit_pips >= level['target'] and not position.get(f"target_{level['target']}_hit"):
                return position['lot_size'] * level['size']
        return 0

    def _calculate_dynamic_targets(self, row, action, entry_price):
        """Calculate dynamic profit targets based on volatility"""
        atr = row['atr']
        cfg = CONFIG['dynamic_targets']
        
        # Base target distance on ATR
        target_distance = min(
            max(atr * cfg['atr_multiplier'], cfg['min_target_distance'] / 10000),
            cfg['max_target_distance'] / 10000
        )
        
        targets = []
        for level in CONFIG['profit_levels']:
            target_price = (
                entry_price + (target_distance * level['target'] / 100) 
                if action == 'BUY' else
                entry_price - (target_distance * level['target'] / 100)
            )
            targets.append({
                'price': target_price,
                'size': level['size']
            })
        
        return targets

    def _update_trailing_stop(self, position, current_bar):
        """Dynamic trailing stop based on volatility and price action"""
        atr = current_bar['atr']
        profit_pips = self._calculate_profit_pips(position, current_bar)
        
        # Start with base trail distance
        trail_distance = CONFIG['trailing_stop']['trail_distance'] / 10000
        
        # Adjust based on volatility
        trail_distance = max(trail_distance, atr * 1.5)
        
        # Tighten trail as profit increases
        if profit_pips > 200:  # Significant profit
            trail_distance *= 0.7  # Tighter trail
        elif profit_pips > 100:
            trail_distance *= 0.85
        
        if position['action'] == 'BUY':
            return max(
                current_bar['close'] - trail_distance,
                position['stop_loss']  # Never widen the stop
            )
        else:
            return min(
                current_bar['close'] + trail_distance,
                position['stop_loss']  # Never widen the stop
            )

    def _is_target_hit(self, position, current_bar, target_price):
        """Check if profit target is hit"""
        if position['action'] == 'BUY':
            return current_bar['high'] >= target_price
        else:
            return current_bar['low'] <= target_price

    def _calculate_position_size(self, entry_price, stop_loss, risk_amount):
        """Better position sizing"""
        risk_pips = abs(entry_price - stop_loss) * 10000
        position_value = risk_amount / (risk_pips * 0.1)  # $0.10 per pip per 0.01 lot
        
        # Apply position size limits
        position_value = max(
            CONFIG['position_sizing']['min_position_size'],
            min(CONFIG['position_sizing']['max_position_size'], position_value)
        )
        
        # Round to nearest increment
        increment = CONFIG['position_sizing']['size_increment']
        return round(position_value / increment) * increment

    def _check_partial_exits(self, position, current_bar, profit_pips):
        """Handle partial profit taking"""
        if not hasattr(position, 'partial_exits'):
            position['partial_exits'] = []
        
        for level in CONFIG['profit_levels']:
            if profit_pips >= level['target'] and level not in position['partial_exits']:
                position['partial_exits'].append(level)
                position['lot_size'] *= (1 - level['size'])
                if position['lot_size'] < CONFIG['position_sizing']['min_position_size']:
                    return True  # Close remaining position
        
        return False

    def _calculate_trade_quality(self, row, daily_trend, h4_trend, m5_trend):
        """Calculate trade quality score (0-1) based on multiple factors"""
        quality_score = 0
        factors = CONFIG['position_sizing']['quality_factors']
        
        # Trend alignment score (0-0.3)
        trend_score = 0
        if daily_trend == h4_trend == m5_trend:
            trend_score = 0.3
        elif daily_trend == h4_trend or h4_trend == m5_trend:
            trend_score = 0.15
        quality_score += trend_score * factors['trend_alignment']
        
        # Volatility score (0-0.2)
        atr = row['atr']
        vol_score = 0
        if 0.0005 <= atr <= 0.002:  # Ideal volatility range
            vol_score = 0.2
        elif 0.002 < atr <= 0.003:
            vol_score = 0.1
        quality_score += vol_score * factors['volatility']
        
        # Momentum score (0-0.2)
        momentum = abs(row['momentum'])
        mom_score = min(momentum / 0.5, 1.0) * 0.2  # Normalize to 0-0.2
        quality_score += mom_score * factors['momentum']
        
        # RSI signal quality (0-0.15)
        rsi_score = 0
        if row['StochRSI'] > 0.8 or row['StochRSI'] < 0.2:
            rsi_score = 0.15
        elif row['StochRSI'] > 0.7 or row['StochRSI'] < 0.3:
            rsi_score = 0.1
        quality_score += rsi_score * factors['rsi_signal']
        
        # Volume confirmation (0-0.15)
        vol_confirm = row['volume_ratio']
        if vol_confirm > 1.5:
            vol_score = 0.15
        elif vol_confirm > 1.0:
            vol_score = 0.1
        else:
            vol_score = 0.05
        quality_score += vol_score * factors['volume_confirmation']
        
        return quality_score

    def _is_engulfing_pattern(self, row):
        """Check for bullish/bearish engulfing pattern"""
        try:
            # Get previous bar
            prev_open = row['prev_open']
            prev_close = row['prev_close']
            curr_open = row['open']
            curr_close = row['close']
            
            # Bullish engulfing
            if curr_close > curr_open and  \
               curr_close > prev_open and \
               curr_open < prev_close and \
               abs(curr_close - curr_open) > abs(prev_close - prev_open):
                return True
                
            # Bearish engulfing    
            if curr_close < curr_open and \
               curr_close < prev_open and \
               curr_open > prev_close and \
               abs(curr_close - curr_open) > abs(prev_close - prev_open):
                return True
                
            return False
            
        except Exception as e:
            print(f"Error checking engulfing pattern: {e}")
            return False

    def _is_pin_bar(self, row):
        """Check for pin bar pattern"""
        try:
            body = abs(row['close'] - row['open'])
            upper_wick = row['high'] - max(row['open'], row['close'])
            lower_wick = min(row['open'], row['close']) - row['low']
            total_range = row['high'] - row['low']
            
            # Body should be small relative to total range
            small_body = body <= total_range * 0.3
            
            # For bullish pin (hammer)
            if lower_wick >= total_range * 0.6 and upper_wick <= total_range * 0.1:
                return True
                
            # For bearish pin (shooting star)    
            if upper_wick >= total_range * 0.6 and lower_wick <= total_range * 0.1:
                return True
                
            return False
            
        except Exception as e:
            print(f"Error checking pin bar: {e}")
            return False

    def _near_key_level(self, row):
        """Check if price is near key support/resistance level"""
        try:
            # Use previous swing highs/lows
            nearby_threshold = row['atr'] * 1.5
            
            # Check if current price is near recent swing high/low
            near_high = abs(row['close'] - row['high_10']) <= nearby_threshold
            near_low = abs(row['close'] - row['low_10']) <= nearby_threshold
            
            # Check psychological levels (round numbers)
            price = row['close']
            round_level = round(price / 100) * 100  # Round to nearest 100
            near_round = abs(price - round_level) <= nearby_threshold
            
            return near_high or near_low or near_round
            
        except Exception as e:
            print(f"Error checking key levels: {e}")
            return False

    def _is_high_volatility(self, row):
        """Check if current volatility is high"""
        try:
            current_atr = row['atr']
            avg_atr = row.get('atr_ma', current_atr)  # Use MA of ATR if available
            
            return current_atr > avg_atr * 1.5
            
        except Exception as e:
            print(f"Error checking volatility: {e}")
            return False

    def _is_against_major_trend(self, row):
        """Check if trade is against major trend"""
        try:
            # Use MA alignment to determine major trend
            ma20 = row['MA_20']
            ma50 = row['MA_50']
            current_close = row['close']
            
            # Potential long trade against bearish trend
            if current_close > ma20 and ma20 < ma50:
                return True
                
            # Potential short trade against bullish trend    
            if current_close < ma20 and ma20 > ma50:
                return True
                
            return False
            
        except Exception as e:
            print(f"Error checking trend alignment: {e}")
            return False

    def _is_strong_reversal(self, current_bar, position_type):
        """Detect strong price reversal signals"""
        try:
            # Get momentum and volatility metrics
            momentum = current_bar['momentum']
            atr = current_bar['atr']
            
            # Check for reversal candle patterns
            is_reversal_pattern = (
                self._is_engulfing_pattern(current_bar) or 
                self._is_pin_bar(current_bar)
            )
            
            # Strong momentum against position
            if position_type == 'BUY':
                strong_counter_momentum = (
                    momentum < -2 * atr and 
                    current_bar['close'] < current_bar['MA_20']
                )
            else:  # SELL position
                strong_counter_momentum = (
                    momentum > 2 * atr and 
                    current_bar['close'] > current_bar['MA_20']
                )
            
            # RSI reversal signals
            rsi_reversal = (
                (position_type == 'BUY' and current_bar['RSI'] > 75) or
                (position_type == 'SELL' and current_bar['RSI'] < 25)
            )
            
            # Volume spike with reversal
            volume_spike = current_bar['volume_ratio'] > 2.0
            
            # Combine signals - need at least 2 confirmation factors
            reversal_signals = sum([
                is_reversal_pattern,
                strong_counter_momentum,
                rsi_reversal,
                volume_spike
            ])
            
            return reversal_signals >= 2
            
        except Exception as e:
            print(f"Error checking reversal: {e}")
            return False

    def _conditions_deteriorating(self, current_bar, position):
        """Check if market conditions are deteriorating"""
        try:
            # Track multiple deterioration factors
            deterioration_score = 0
            
            # 1. Trend weakness
            if position['action'] == 'BUY':
                if current_bar['close'] < current_bar['MA_20']:
                    deterioration_score += 1
                if current_bar['MA_20'] < current_bar['MA_50']:
                    deterioration_score += 1
            else:  # SELL position
                if current_bar['close'] > current_bar['MA_20']:
                    deterioration_score += 1
                if current_bar['MA_20'] > current_bar['MA_50']:
                    deterioration_score += 1
            
            # 2. Momentum loss
            momentum_weakening = (
                abs(current_bar['momentum']) < 
                abs(current_bar['momentum']) * 0.5  # 50% momentum loss
            )
            if momentum_weakening:
                deterioration_score += 1
            
            # 3. Volume decline
            if current_bar['volume_ratio'] < 0.7:  # Volume below average
                deterioration_score += 1
            
            # 4. Volatility increase
            if self._is_high_volatility(current_bar):
                deterioration_score += 1
            
            # 5. Price action warnings
            if self._is_exhaustion_pattern(current_bar, position['action']):
                deterioration_score += 2
            
            return deterioration_score >= 3  # Exit if multiple warning signs
            
        except Exception as e:
            print(f"Error checking market conditions: {e}")
            return False

    def _should_adjust_stop(self, position, current_bar):
        """Determine if stop loss should be adjusted"""
        try:
            profit_pips = self._calculate_profit_pips(position, current_bar)
            
            # Move to breakeven
            if profit_pips >= CONFIG['breakeven_rules']['min_profit_ticks']:
                return True
            
            # Trail profit
            if profit_pips >= CONFIG['trailing_stop']['activation_ticks']:
                return True
            
            # Adjust for increased volatility
            current_atr = current_bar['atr']
            entry_atr = position.get('entry_atr', current_atr)
            
            if current_atr > entry_atr * 1.5:  # 50% volatility increase
                return True
            
            return False
            
        except Exception as e:
            print(f"Error checking stop adjustment: {e}")
            return False

    def _is_exhaustion_pattern(self, row, position_type):
        """Check for trend exhaustion patterns"""
        try:
            # Long upper wick for buys, long lower wick for sells
            total_range = row['high'] - row['low']
            body = abs(row['close'] - row['open'])
            upper_wick = row['high'] - max(row['open'], row['close'])
            lower_wick = min(row['open'], row['close']) - row['low']
            
            if position_type == 'BUY':
                return (upper_wick > body * 2 and  # Long upper wick
                        body < total_range * 0.3)  # Small body
            else:
                return (lower_wick > body * 2 and  # Long lower wick
                        body < total_range * 0.3)  # Small body
                        
        except Exception as e:
            print(f"Error checking exhaustion pattern: {e}")
            return False

    def _adjust_stop_loss(self, position, current_bar):
        """Enhanced stop loss adjustment with tracking"""
        try:
            old_stop = position['stop_loss']
            
            # Existing stop loss adjustment logic
            profit_pips = self._calculate_profit_pips(position, current_bar)
            
            if profit_pips >= CONFIG['breakeven_rules']['min_profit_ticks']:
                # Move to breakeven with buffer
                new_stop = self._calculate_breakeven_level(position)
            elif profit_pips >= CONFIG['trailing_stop']['activation_ticks']:
                # Use trailing stop
                new_stop = self._calculate_trailing_stop(position, current_bar, profit_pips)
            else:
                # Adjust based on volatility
                atr = current_bar['atr']
                if position['action'] == 'BUY':
                    new_stop = current_bar['close'] - (atr * 2)
                else:
                    new_stop = current_bar['close'] + (atr * 2)
            
            # Never widen the stop
            if position['action'] == 'BUY':
                position['stop_loss'] = max(position['stop_loss'], new_stop)
            else:
                position['stop_loss'] = min(position['stop_loss'], new_stop)
            
            # Record the modification if stop changed
            if position['stop_loss'] != old_stop:
                modification = {
                    'time': current_bar.name,
                    'type': 'Stop Loss Adjustment',
                    'old_value': old_stop,
                    'new_value': position['stop_loss'],
                    'reason': self._get_stop_adjustment_reason(position, current_bar),
                    'market_context': self._get_market_description(current_bar)
                }
                position['modifications'].append(modification)
                
        except Exception as e:
            print(f"Error adjusting stop loss: {e}")

    def _get_stop_adjustment_reason(self, position, current_bar):
        """Get detailed reason for stop loss adjustment"""
        profit_pips = self._calculate_profit_pips(position, current_bar)
        
        if profit_pips >= CONFIG['breakeven_rules']['min_profit_ticks']:
            return f"Moving to breakeven after {profit_pips:.1f} pips in profit"
        elif profit_pips >= CONFIG['trailing_stop']['activation_ticks']:
            return f"Activating trailing stop at {profit_pips:.1f} pips profit"
        else:
            return "Volatility-based stop adjustment"

    def _determine_setup_type(self, row, action):
        """Determine the type of trading setup"""
        # Check for trend-following setup
        if self._is_trend_following_setup(row, action):
            return "Trend Following"
        
        # Check for reversal setup
        if self._is_reversal_setup(row, action):
            return "Counter-Trend Reversal"
        
        # Check for range/breakout setup
        if self._is_range_breakout(row, action):
            return "Range Breakout"
            
        return "Momentum/Scalp"

    def _is_trend_following_setup(self, row, action):
        """Check if setup is trend-following"""
        trend_aligned = (
            (action == 'BUY' and row['MA_20'] > row['MA_50']) or
            (action == 'SELL' and row['MA_20'] < row['MA_50'])
        )
        momentum_aligned = (
            (action == 'BUY' and row['momentum'] > 0) or
            (action == 'SELL' and row['momentum'] < 0)
        )
        return trend_aligned and momentum_aligned

    def _is_reversal_setup(self, row, action):
        """Check if setup is a reversal"""
        # Oversold conditions for buy, overbought for sell
        rsi_reversal = (
            (action == 'BUY' and row['RSI'] < 30) or
            (action == 'SELL' and row['RSI'] > 70)
        )
        # Price action confirmation
        pattern_reversal = (
            self._is_engulfing_pattern(row) or
            self._is_pin_bar(row)
        )
        return rsi_reversal and pattern_reversal

    def _is_range_breakout(self, row, action):
        """Check if setup is a range breakout"""
        # Calculate recent price range
        range_high = row['high_10']
        range_low = row['low_10']
        range_size = range_high - range_low
        
        # Check if price is breaking out of range
        breakout = (
            (action == 'BUY' and row['close'] > range_high) or
            (action == 'SELL' and row['close'] < range_low)
        )
        
        # Volume confirmation
        volume_surge = row['volume_ratio'] > 1.5
        
        return breakout and volume_surge

    def _analyze_trend_alignment(self, daily_trend, h4_trend, m5_trend):
        """Analyze trend alignment across timeframes"""
        alignments = []
        
        if daily_trend == h4_trend == m5_trend:
            alignments.append("Strong alignment across all timeframes")
        elif daily_trend == h4_trend:
            alignments.append("Higher timeframes aligned")
        elif h4_trend == m5_trend:
            alignments.append("Lower timeframes aligned")
        else:
            alignments.append("Mixed trend signals")
            
        return ", ".join(alignments)

    def _calculate_trend_strength(self, row):
        """Calculate trend strength based on multiple factors"""
        strength = 0
        
        # MA alignment strength
        ma_diff = (row['MA_20'] - row['MA_50']) / row['MA_50']
        strength += abs(ma_diff) * 5
        
        # Momentum contribution
        strength += abs(row['momentum']) / 100
        
        # Volume confirmation
        if row['volume_ratio'] > 1.5:
            strength += 0.2
        
        # Normalize to 0-1 range
        return min(max(strength, 0), 1)

    def _analyze_volume_context(self, row):
        """Analyze volume context"""
        contexts = []
        
        ratio = row['volume_ratio']
        if (ratio > 2.0):
            contexts.append("Very high volume")
        elif (ratio > 1.5):
            contexts.append("Above average volume")
        elif (ratio < 0.7):
            contexts.append("Below average volume")
        else:
            contexts.append("Normal volume")
            
        # Add volume trend
        if (ratio > row.get('prev_volume_ratio', ratio)):
            contexts.append("increasing")
        else:
            contexts.append("decreasing")
            
        return " - ".join(contexts)

    def _identify_candlestick_pattern(self, row):
        """Identify candlestick patterns"""
        patterns = []
        
        if self._is_engulfing_pattern(row):
            patterns.append("Engulfing")
        if self._is_pin_bar(row):
            patterns.append("Pin Bar")
        if self._is_doji(row):
            patterns.append("Doji")
            
        return patterns if patterns else ["No significant pattern"]

    def _analyze_price_context(self, row):
        """Analyze price action context"""
        context = []
        
        # Trend context
        if row['close'] > row['MA_20'] > row['MA_50']:
            context.append("Uptrend")
        elif row['close'] < row['MA_20'] < row['MA_50']:
            context.append("Downtrend")
        else:
            context.append("Sideways")
            
        # Volatility context
        if self._is_high_volatility(row):
            context.append("High volatility")
        else:
            context.append("Normal volatility")
            
        # Support/Resistance context
        if self._near_key_level(row):
            context.append("Near key level")
            
        return " - ".join(context)

    def _calculate_risk_reward(self, row):
        """Calculate initial risk/reward ratio"""
        atr = row['atr']
        potential_risk = 2 * atr  # 2x ATR for stop loss
        potential_reward = 3 * atr  # 3x ATR for target
        return potential_reward / potential_risk if potential_risk > 0 else 0

    def _calculate_stop_distance(self, row):
        """Calculate and validate stop distance"""
        atr = row['atr']
        return max(2 * atr, CONFIG['min_stop_distance'] / 10000)

    def _calculate_setup_quality(self, row):
        """Calculate overall setup quality score"""
        # Reuse existing quality calculation with additional factors
        base_quality = self._calculate_pattern_quality(row)
        
        # Add time-based factors
        if self._is_optimal_trading_hour(row.name):
            base_quality *= 1.2
        
        # Add volatility adjustment
        if self._is_optimal_volatility(row):
            base_quality *= 1.1
            
        return min(base_quality, 1.0)

    def _is_optimal_trading_hour(self, timestamp):
        """Check if current time is during optimal trading hours"""
        hour = timestamp.hour
        return (
            (8 <= hour <= 16) or  # London session
            (13 <= hour <= 20)    # NY session
        )

    def _is_optimal_volatility(self, row):
        """Check if volatility is in optimal range"""
        atr = row['atr']
        return 0.0005 <= atr <= 0.002

    def _is_doji(self, row):
        """Check for doji pattern"""
        body = abs(row['close'] - row['open'])
        total_range = row['high'] - row['low']
        return body <= total_range * 0.1

    def _identify_key_levels(self, row):
        """Identify key support and resistance levels"""
        levels = []
        
        # Use recent highs/lows
        if hasattr(row, 'high_10'):
            levels.append({
                'type': 'Resistance',
                'level': row['high_10'],
                'strength': 'Recent High'
            })
        
        if hasattr(row, 'low_10'):
            levels.append({
                'type': 'Support',
                'level': row['low_10'],
                'strength': 'Recent Low'
            })
        
        # Add moving averages as dynamic levels
        if hasattr(row, 'MA_20'):
            levels.append({
                'type': 'Dynamic',
                'level': row['MA_20'],
                'strength': 'MA20'
            })
        
        if hasattr(row, 'MA_50'):
            levels.append({
                'type': 'Dynamic',
                'level': row['MA_50'],
                'strength': 'MA50'
            })
        
        # Find psychological levels (round numbers)
        price = row['close']
        round_100 = round(price / 100) * 100
        round_50 = round(price / 50) * 50
        
        levels.append({
            'type': 'Psychological',
            'level': round_100,
            'strength': 'Round 100'
        })
        
        levels.append({
            'type': 'Psychological',
            'level': round_50,
            'strength': 'Round 50'
        })
        
        # Sort levels by price
        levels.sort(key=lambda x: x['level'])
        
        return levels

# ... existing code for other helper methods ...
