import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jinja2
import os
from xau.backtester.config import CONFIG

class ReportGenerator:
    def __init__(self, results, trades, market_data):
        # Initialize core attributes first
        self.initial_balance = CONFIG['initial_balance']
        self.current_balance = self.initial_balance
        self.results = results
        self.market_data = market_data
        
        # Get the directory where this script is located
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.template_dir = os.path.join(self.current_dir, 'templates')
        
        # Process trades last, after all attributes are set
        self.trades = self._preprocess_trades(trades)

    def _preprocess_trades(self, trades):
        """Convert numpy types to Python native types for Jinja2 compatibility."""
        processed_trades = []
        running_balance = self.initial_balance
        
        for trade in trades:
            try:
                # Convert PnL to float and ensure it's properly calculated
                pnl = float(trade['pnl'])
                running_balance += pnl
                
                processed_trade = {
                    'id': int(trade['id']),
                    'entry_time': pd.to_datetime(trade['entry_time']),
                    'exit_time': pd.to_datetime(trade['exit_time']),
                    'entry_price': float(trade['entry_price']),
                    'exit_price': float(trade['exit_price']),
                    'action': str(trade['action']),
                    'lot_size': float(trade.get('lot_size', 0.01)),  # Default to 0.01 if not present
                    'pnl': pnl,  # Already in USD
                    'balance': running_balance  # Running balance in USD
                }
                processed_trades.append(processed_trade)
                
            except (KeyError, ValueError) as e:
                print(f"Error processing trade: {e}")
                print(f"Trade data: {trade}")
                continue
        
        print(f"Successfully processed {len(processed_trades)} trades")
        if processed_trades:
            print(f"Sample trade: {processed_trades[0]}")
            print(f"Final balance: ${running_balance:.2f}")
        
        return processed_trades

    def generate_report(self):
        if not self.trades:
            print("Warning: No trades to report")
            
        metrics = self._calculate_metrics()
        charts = self._create_charts()
        
        # Calculate initial and final balance
        initial_balance = CONFIG['initial_balance']
        final_balance = initial_balance + sum(trade['pnl'] for trade in self.trades)
        
        # Debug prints
        print(f"Metrics calculated: {metrics}")
        print(f"Number of trades for report: {len(self.trades)}")
        
        # Create template loader with absolute path
        template_loader = jinja2.FileSystemLoader(searchpath=self.template_dir)
        template_env = jinja2.Environment(loader=template_loader)
        template = template_env.get_template("report_template.html")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(self.current_dir, 'reports')
        os.makedirs(output_dir, exist_ok=True)
        
        output_text = template.render(
            metrics=metrics,
            charts=charts,
            trades=self.trades,
            initial_balance=self.initial_balance,
            final_balance=final_balance
        )
        
        # Save report to reports directory
        output_path = os.path.join(output_dir, "backtest_report.html")
        # Use UTF-8 encoding when writing the file
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(output_text)
            
        print(f"Report generated at: {output_path}")

    def _calculate_metrics(self):
        return {
            'total_trades': len(self.trades),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor(),
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'average_trade': self._calculate_average_trade()
        }
    
    def _create_charts(self):
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price Chart', 'StochRSI')
        )
        
        # Price and MA chart
        fig.add_trace(go.Candlestick(
            x=self.market_data.index,
            open=self.market_data['open'],
            high=self.market_data['high'],
            low=self.market_data['low'],
            close=self.market_data['close'],
            name='XAUUSD'
        ), row=1, col=1)
        
        # Add MA lines
        for period in [20, 50]:
            fig.add_trace(go.Scatter(
                x=self.market_data.index,
                y=self.market_data[f'MA_{period}'],
                name=f'MA_{period}',
                line=dict(width=1)
            ), row=1, col=1)
        
        # Add trades
        for trade in self.trades:
            color = 'green' if trade['pnl'] > 0 else 'red'
            fig.add_trace(go.Scatter(
                x=[trade['entry_time'], trade['exit_time']],
                y=[trade['entry_price'], trade['exit_price']],
                mode='markers+lines',
                name=f"Trade {trade['id']}",
                line=dict(color=color),
                showlegend=False
            ), row=1, col=1)
        
        # Add StochRSI
        fig.add_trace(go.Scatter(
            x=self.market_data.index,
            y=self.market_data['StochRSI'],
            name='StochRSI',
            line=dict(color='purple', width=1)
        ), row=2, col=1)

        # Update layout
        fig.update_layout(
            height=800,
            xaxis_rangeslider_visible=False,
            title_text="XAU/USD Trading Strategy Backtest"
        )

        return fig.to_html(include_plotlyjs=True, full_html=True)

    def _calculate_win_rate(self):

        if not self.trades:
            return 0
        winning_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
        return (winning_trades / len(self.trades)) * 100

    def _calculate_profit_factor(self):
        gross_profit = sum(trade['pnl'] for trade in self.trades if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in self.trades if trade['pnl'] < 0))
        return gross_profit / gross_loss if gross_loss != 0 else 0

    def _calculate_max_drawdown(self):
        if not self.results:
            return 0
        peak = self.results[0]
        max_dd = 0
        
        for value in self.results:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            max_dd = max(max_dd, dd)
        
        return max_dd

    def _calculate_sharpe_ratio(self):
        if not self.trades:
            return 0
        returns = [trade['pnl'] for trade in self.trades]
        if len(returns) < 2:
            return 0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
            
        return (avg_return / std_return) * np.sqrt(252)  # Annualized

    def _calculate_average_trade(self):
        if not self.trades:
            return 0
        return sum(trade['pnl'] for trade in self.trades) / len(self.trades)
