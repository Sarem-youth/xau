import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jinja2
import os

class ReportGenerator:
    def __init__(self, results, trades, market_data):
        self.results = results
        self.trades = trades
        self.market_data = market_data
        
    def generate_report(self):
        metrics = self._calculate_metrics()
        charts = self._create_charts()
        
        template_loader = jinja2.FileSystemLoader(searchpath="./templates")
        template_env = jinja2.Environment(loader=template_loader)
        template = template_env.get_template("report_template.html")
        
        output_text = template.render(
            metrics=metrics,
            charts=charts,
            trades=self.trades
        )
        
        with open("backtest_report.html", "w") as f:
            f.write(output_text)
    
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
        fig = make_subplots(rows=3, cols=1, shared_xaxis=True)
        
        # Price and MA chart
        fig.add_trace(go.Candlestick(
            x=self.market_data.index,
            open=self.market_data['open'],
            high=self.market_data['high'],
            low=self.market_data['low'],
            close=self.market_data['close']
        ), row=1, col=1)
        
        # Add MA lines
        for period in [20, 50]:
            fig.add_trace(go.Scatter(
                x=self.market_data.index,
                y=self.market_data[f'MA_{period}'],
                name=f'MA_{period}'
            ), row=1, col=1)
        
        # Add trades
        for trade in self.trades:
            fig.add_trace(go.Scatter(
                x=[trade['entry_time'], trade['exit_time']],
                y=[trade['entry_price'], trade['exit_price']],
                mode='markers',
                name=f"Trade {trade['id']}"
            ), row=1, col=1)
        
        return fig.to_html()
