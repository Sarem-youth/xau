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
        """Enhanced trade preprocessing with modifications"""
        processed_trades = []
        running_balance = self.initial_balance
        
        for trade in trades:
            try:
                # Basic trade data processing
                pnl = float(trade['pnl'])
                running_balance += pnl
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time'])
                time_in_trade = (exit_time - entry_time).total_seconds() / 3600
                
                # Calculate signal quality score
                signal_quality = self._calculate_signal_quality(trade)
                
                # Enhanced trade metrics
                trade_metrics = {
                    'risk_amount': trade.get('risk_amount', abs(pnl) if pnl < 0 else pnl * 0.5),
                    'time_in_trade': time_in_trade,
                    'risk_reward': abs(pnl / trade.get('risk_amount', 1)) if pnl > 0 else 0,
                    'max_favorable_excursion': trade.get('max_favorable', abs(pnl) if pnl > 0 else 0),
                    'max_adverse_excursion': trade.get('max_adverse', abs(pnl) if pnl < 0 else 0)
                }
                
                # Enhanced entry reason with signal quality
                entry_reason = {
                    'market_context': trade.get('reason', 'N/A'),
                    'pattern_detected': trade.get('pattern_detected', 'No specific pattern'),
                    'signal_quality': signal_quality,
                    'indicators': {
                        'rsi': float(trade.get('entry_rsi', 50)),
                        'stoch_rsi': float(trade.get('entry_stoch_rsi', 0.5)),
                        'ma_alignment': trade.get('ma_alignment', 'Unknown'),
                        'momentum': float(trade.get('momentum', 0))
                    }
                }
                
                # Enhanced exit reason formatting
                exit_reason = {
                    'type': trade.get('exit_type', 'unknown').title(),  # Capitalize first letter
                    'description': self._format_exit_reason(trade),
                    'market_analysis': {
                        'trend': trade.get('exit_condition', 'Unknown market condition'),
                        'technical_signals': self._format_technical_signals(trade),
                        'key_levels': self._identify_key_levels(trade)
                    },
                    'indicators': {
                        'rsi': float(trade.get('exit_rsi', 0)),
                        'price_structure': trade.get('exit_price_action', 'N/A'),
                        'volatility': float(trade.get('exit_volatility', 0)),
                        'volume': trade.get('exit_volume', 'Normal')
                    }
                }
                
                processed_trade = {
                    'id': int(trade['id']),
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': float(trade['entry_price']),
                    'exit_price': float(trade['exit_price']),
                    'action': str(trade['action']),
                    'lot_size': float(trade.get('lot_size', 0.01)),
                    'pnl': pnl,
                    'balance': running_balance,
                    'trade_metrics': trade_metrics,
                    'entry_reason': entry_reason,
                    'exit_reason': exit_reason
                }
                
                processed_trade.update({
                    'entry_analysis': self._format_entry_analysis(trade),
                    'modifications': self._format_trade_modifications(trade),
                    'summary': self._create_trade_summary(trade)
                })
                
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

    def _calculate_signal_quality(self, trade):
        """Calculate overall signal quality score"""
        try:
            # Basic score starting point
            score = 0.5  # Start with neutral score
            
            # Adjust based on available indicators
            if 'entry_rsi' in trade:
                rsi = float(trade['entry_rsi'])
                if 30 <= rsi <= 70:
                    score += 0.1
                elif 20 <= rsi <= 80:
                    score += 0.05
                
            if 'entry_stoch_rsi' in trade:
                stoch = float(trade['entry_stoch_rsi'])
                if 0.2 <= stoch <= 0.8:
                    score += 0.1
                
            if 'ma_alignment' in trade:
                if trade['ma_alignment'] == 'strong':
                    score += 0.2
                elif trade['ma_alignment'] == 'moderate':
                    score += 0.1
                    
            # Normalize score between 0 and 1
            return min(max(score, 0), 1)
            
        except Exception as e:
            print(f"Error calculating signal quality: {e}")
            return 0.5  # Return neutral score on error

    def generate_report(self):
        """Enhanced report generation with complete data"""
        if not self.trades:
            print("Warning: No trades to report")
            return
        
        # Calculate all metrics
        metrics = self._calculate_metrics()
        
        # Create charts
        equity_chart = self._create_equity_chart()
        trade_distribution_chart = self._create_trade_distribution_chart()
        time_analysis_chart = self._create_time_analysis_chart()
        risk_analysis_chart = self._create_risk_management_chart()
        
        # Calculate detailed performance metrics
        initial_balance = self.initial_balance
        final_balance = initial_balance + sum(trade['pnl'] for trade in self.trades)
        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        
        # Get time period info - use configured dates instead of trade dates
        start_date = CONFIG['start_date']
        end_date = CONFIG['end_date']
        total_days = (end_date - start_date).days + 1  # Add 1 to include both start and end dates
        
        # Prepare comprehensive template data
        template_data = {
            'metrics': metrics,
            'equity_chart': equity_chart,
            'trade_distribution_chart': trade_distribution_chart,
            'time_analysis_chart': time_analysis_chart,
            'risk_analysis_chart': risk_analysis_chart,
            'trades': self.trades,
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'total_days': total_days,
            'strategy_name': 'XAU/USD Trading Strategy',
            'timeframe': '5-Minute Chart Analysis',
            'symbol': CONFIG['symbol'],
            'trade_summary': {
                'total_trades': len(self.trades),
                'winning_trades': len([t for t in self.trades if t['pnl'] > 0]),
                'losing_trades': len([t for t in self.trades if t['pnl'] < 0]),
                'breakeven_trades': len([t for t in self.trades if t['pnl'] == 0]),
                'avg_win': self._calculate_average_win(),
                'avg_loss': self._calculate_average_loss(),
                'largest_win': self._calculate_largest_win(),
                'largest_loss': self._calculate_largest_loss(),
                'avg_trade_duration': self._calculate_average_duration()
            }
        }
        
        # Generate and save report
        template_loader = jinja2.FileSystemLoader(searchpath=self.template_dir)
        template_env = jinja2.Environment(loader=template_loader)
        template = template_env.get_template("report_template.html")
        
        output_dir = os.path.join(self.current_dir, 'reports')
        os.makedirs(output_dir, exist_ok=True)
        
        output_text = template.render(template_data)
        
        output_path = os.path.join(output_dir, "backtest_report.html")
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(output_text)
            
        print(f"Complete report generated at: {output_path}")

    def _calculate_metrics(self):
        """Enhanced metrics calculation with descriptive names and trends"""
        metrics = []
        
        # Total trades
        metrics.append({
            'name': 'Total Trades',
            'value': len(self.trades),
            'trend': 'Based on market conditions',
            'class': 'neutral'
        })
        
        # Win rate
        win_rate = self._calculate_win_rate()
        metrics.append({
            'name': 'Win Rate',
            'value': f"{win_rate:.1f}%",
            'trend': 'Above target' if win_rate > 50 else 'Below target',
            'class': 'success' if win_rate > 50 else 'warning'
        })
        
        # Profit factor
        profit_factor = self._calculate_profit_factor()
        metrics.append({
            'name': 'Profit Factor',
            'value': f"{profit_factor:.2f}",
            'trend': 'Profitable' if profit_factor > 1 else 'Unprofitable',
            'class': 'success' if profit_factor > 1 else 'error'
        })
        
        # Max drawdown
        max_dd = self._calculate_max_drawdown()
        metrics.append({
            'name': 'Maximum Drawdown',
            'value': f"{max_dd:.1f}%",
            'trend': 'Within limits' if max_dd < 20 else 'Exceeded target',
            'class': 'success' if max_dd < 20 else 'error'
        })
        
        # Sharpe ratio
        sharpe = self._calculate_sharpe_ratio()
        metrics.append({
            'name': 'Sharpe Ratio',
            'value': f"{sharpe:.2f}",
            'trend': 'Good risk-adjusted returns' if sharpe > 1 else 'Poor risk-adjusted returns',
            'class': 'success' if sharpe > 1 else 'warning'
        })
        
        # Average trade
        avg_trade = self._calculate_average_trade()
        metrics.append({
            'name': 'Average Trade',
            'value': f"${avg_trade:.2f}",
            'trend': 'Profitable' if avg_trade > 0 else 'Unprofitable',
            'class': 'success' if avg_trade > 0 else 'error'
        })
        
        # Total profit/loss
        total_pnl = sum(trade['pnl'] for trade in self.trades)
        metrics.append({
            'name': 'Total P&L',
            'value': f"${total_pnl:.2f}",
            'trend': 'Net profitable' if total_pnl > 0 else 'Net loss',
            'class': 'success' if total_pnl > 0 else 'error'
        })
        
        return metrics

    def _create_equity_chart(self):
        """Create enhanced equity curve chart"""
        fig = make_subplots(rows=1, cols=1, subplot_titles=['Account Equity & Drawdown'], specs=[[{"secondary_y": True}]])
        fig = self._add_equity_curve(fig, row=1, col=1)
        
        # Enhanced layout
        fig.update_layout(
            height=500,
            template='plotly_white',
            title_font=dict(family="Poppins, sans-serif", size=20),
            font=dict(family="Inter, sans-serif"),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode='x unified',
            plot_bgcolor='rgba(248,250,252,0.5)'
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(226,232,240,0.5)',
            zeroline=False
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(226,232,240,0.5)',
            zeroline=False
        )

        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _create_trade_distribution_chart(self):
        """Create enhanced trade distribution chart"""
        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "domain"}]])
        
        # Color palette that matches the UI
        colors = ['#22c55e', '#ef4444', '#94a3b8']
        
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = len([t for t in self.trades if t['pnl'] < 0])
        breakeven_trades = len([t for t in self.trades if t['pnl'] == 0])
        
        fig.add_trace(go.Pie(
            labels=['Winning', 'Losing', 'Breakeven'],
            values=[winning_trades, losing_trades, breakeven_trades],
            hole=0.7,
            marker=dict(colors=colors),
            textinfo='percent+label',
            textfont=dict(family="Inter, sans-serif", size=12),
            hovertemplate="<b>%{label}</b><br>" +
                         "Count: %{value}<br>" +
                         "Percentage: %{percent}<br>" +
                         "<extra></extra>",
            domain=dict(x=[0.1, 0.9], y=[0.1, 0.9])  # Set domain in trace instead of layout
        ))
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            title=dict(
                text='Trade Distribution',
                font=dict(family="Poppins, sans-serif", size=20),
                x=0.5,
                y=0.95
            ),
            font=dict(family="Inter, sans-serif"),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=20, r=20, t=60, b=60)
        )

        return fig.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True})

    def _create_time_analysis_chart(self):
        """Create enhanced time analysis chart"""
        fig = make_subplots(rows=1, cols=1, subplot_titles=['Trading Activity by Hour'], specs=[[{"secondary_y": True}]])
        fig = self._add_time_analysis(fig, row=1, col=1)
        
        fig.update_layout(
            height=400,  # Fixed height
            template='plotly_white',
            title_font=dict(family="Poppins, sans-serif", size=20),
            font=dict(family="Inter, sans-serif"),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor='rgba(248,250,252,0.5)',
            bargap=0.3
        )

        return fig.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True})

    def _create_risk_management_chart(self):
        """Create enhanced risk management chart"""
        fig = make_subplots(rows=1, cols=1, subplot_titles=['Risk/Reward Analysis'])
        
        # Prepare data
        data = []
        for trade in self.trades:
            data.append({
                'risk': trade['trade_metrics']['risk_amount'],
                'reward': abs(trade['pnl']) if trade['pnl'] > 0 else 0,
                'pnl': trade['pnl'],
                'id': trade['id']
            })
        
        df = pd.DataFrame(data)
        
        # Add scatter plot with enhanced styling
        fig.add_trace(go.Scatter(
            x=df['risk'],
            y=df['reward'],
            mode='markers',
            marker=dict(
                size=10,
                color=df['pnl'],
                colorscale=[[0, '#ef4444'], [0.5, '#94a3b8'], [1, '#22c55e']],
                line=dict(width=1, color='rgba(255,255,255,0.8)'),
                symbol='circle',
                showscale=True,
                colorbar=dict(
                    title='P&L ($)',
                    titleside='right',
                    titlefont=dict(family="Inter, sans-serif", size=12),
                    tickfont=dict(family="Inter, sans-serif", size=10)
                )
            ),
            text=[f"Trade #{id}<br>Risk: ${risk:.2f}<br>Reward: ${reward:.2f}<br>P&L: ${pnl:.2f}" 
                  for id, risk, reward, pnl in zip(df['id'], df['risk'], df['reward'], df['pnl'])],
            hoverinfo='text',
            name='Trades'
        ))
        
        # Update layout with enhanced styling
        fig.update_layout(
            height=400,  # Fixed height
            template='plotly_white',
            title_font=dict(family="Poppins, sans-serif", size=20),
            font=dict(family="Inter, sans-serif"),
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor='rgba(248,250,252,0.5)'
        )

        return fig.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True})

    def _create_charts(self):
        """Enhanced chart creation with complete data display"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Price Action & Trade Entries/Exits',
                'Account Equity & Drawdown',
                'Trade Distribution',
                'Trading Activity by Hour',
                'Risk/Reward Analysis',
                'Trade Duration Distribution'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.12,  # Increased spacing
            specs=[
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"type": "domain"}, {"secondary_y": True}],  # Changed pie chart spec to domain
                [{"secondary_y": True}, {"type": "histogram"}]
            ]
        )
        
        # Add all charts with complete data
        fig = self._add_price_action_chart(fig, row=1, col=1)
        fig = self._add_equity_curve(fig, row=1, col=2)
        fig = self._add_trade_distribution(fig, row=2, col=1)
        fig = self._add_time_analysis(fig, row=2, col=2)
        fig = self._add_risk_management_analysis(fig, row=3, col=1)
        fig = self._add_duration_analysis(fig, row=3, col=2)
        
        # Add overall chart titles and axis labels
        fig.update_layout(
            height=1500,
            showlegend=True,
            template='plotly_white',
            title=dict(
                text="Trading Strategy Performance Analysis",
                x=0.5,
                font=dict(size=24)
            ),
            font=dict(family="Inter, sans-serif"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        # Update all subplot axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2, secondary_y=True)
        
        fig.update_xaxes(title_text="Hour of Day", row=2, col=2)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=2)
        
        fig.update_xaxes(title_text="Risk ($)", row=3, col=1)
        fig.update_yaxes(title_text="Reward ($)", row=3, col=1)
        
        fig.update_xaxes(title_text="Trade Duration (Hours)", row=3, col=2)
        fig.update_yaxes(title_text="Number of Trades", row=3, col=2)
        
        # Add grid lines and improve visibility
        for i in fig['layout']:
            if i.startswith('xaxis') or i.startswith('yaxis'):
                fig['layout'][i].update(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='#f0f0f0',
                    linecolor='#d0d0d0',
                    zeroline=True,
                    zerolinecolor='#e0e0e0'
                )
        
        # Return the complete figure
        return fig.to_html(
            include_plotlyjs=True,
            full_html=False,
            config={
                'displayModeBar': True,
                'responsive': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
            }
        )

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

    def _add_price_action_chart(self, fig, row, col):
        """Add price action chart with trade entries/exits"""
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.market_data.index,
                open=self.market_data['open'],
                high=self.market_data['high'],
                low=self.market_data['low'],
                close=self.market_data['close'],
                name='XAUUSD'
            ),
            row=row, col=col
        )
        
        # Add moving averages
        for period in [20, 50]:
            fig.add_trace(
                go.Scatter(
                    x=self.market_data.index,
                    y=self.market_data[f'MA_{period}'],
                    name=f'MA{period}',
                    line=dict(width=1)
                ),
                row=row, col=col
            )
        
        # Add trade entries and exits
        for trade in self.trades:
            color = 'green' if trade['pnl'] > 0 else 'red'
            
            # Entry point
            fig.add_trace(
                go.Scatter(
                    x=[trade['entry_time']],
                    y=[trade['entry_price']],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if trade['action'] == 'BUY' else 'triangle-down',
                        size=10,
                        color=color
                    ),
                    name=f"Trade {trade['id']} Entry",
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Exit point
            fig.add_trace(
                go.Scatter(
                    x=[trade['exit_time']],
                    y=[trade['exit_price']],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=10,
                        color=color
                    ),
                    name=f"Trade {trade['id']} Exit",
                    showlegend=False
                ),
                row=row, col=col
            )
        
        return fig  # Return the figure after adding traces

    def _add_equity_curve(self, fig, row, col):
        """Add equity curve with drawdown overlay"""
        equity_line = go.Scatter(
            x=[trade['exit_time'] for trade in self.trades],
            y=self.results,
            name='Equity',
            line=dict(color='blue', width=2)
        )
        
        # Calculate drawdown
        running_max = pd.Series(self.results).expanding().max()
        drawdown = (pd.Series(self.results) - running_max) / running_max * 100
        
        drawdown_line = go.Scatter(
            x=[trade['exit_time'] for trade in self.trades],
            y=drawdown,
            name='Drawdown %',
            line=dict(color='red', width=1),
            yaxis='y2'
        )
        
        fig.add_trace(equity_line, row=row, col=col)
        fig.add_trace(drawdown_line, row=row, col=col, secondary_y=True)
        
        return fig  # Return the figure after adding traces

    def _add_trade_distribution(self, fig, row, col):
        """Enhanced trade distribution pie chart with better formatting"""
        # Calculate trade statistics
        trades_by_type = {
            'Winning': len([t for t in self.trades if t['pnl'] > 0]),
            'Losing': len([t for t in self.trades if t['pnl'] < 0]),
            'Breakeven': len([t for t in self.trades if t['pnl'] == 0])
        }
        
        total_trades = sum(trades_by_type.values())
        percentages = {k: (v/total_trades*100) for k, v in trades_by_type.items()}
        
        # Create donut chart
        fig.add_trace(
            go.Pie(
                labels=list(trades_by_type.keys()),
                values=list(trades_by_type.values()),
                name='Trade Distribution',
                marker_colors=['#22c55e', '#ef4444', '#94a3b8'],
                hovertemplate="<b>%{label}</b><br>" +
                            "Count: %{value}<br>" +
                            "Percentage: %{percent:.1%}<br>" +
                            "<extra></extra>",
                hole=0.6,
                domain={'row': row-1, 'column': col-1},  # Specify domain for pie chart
                textinfo='percent+label',
                textposition='outside',
                textfont=dict(size=12)
            )
        )
        
        # Add center text annotation
        fig.add_annotation(
            text=f"Total Trades<br>{total_trades}",
            x=0.16, y=0.5,  # Adjusted position
            font=dict(size=14),
            showarrow=False,
            xanchor='center',
            yanchor='middle'
        )
        
        return fig

    def _add_time_analysis(self, fig, row, col):
        """Enhanced time analysis with better visualization"""
        # Prepare hourly data
        trade_hours = pd.Series([t['entry_time'].hour for t in self.trades])
        winning_hours = pd.Series([t['entry_time'].hour for t in self.trades if t['pnl'] > 0])
        
        # Create hour ranges (0-23)
        all_hours = pd.Series(range(24))
        hourly_trades = trade_hours.value_counts().reindex(all_hours).fillna(0)
        hourly_wins = winning_hours.value_counts().reindex(all_hours).fillna(0)
        
        # Calculate win rates with handling for zero division
        win_rates = (hourly_wins / hourly_trades * 100).fillna(0)
        
        # Add total trades bars
        fig.add_trace(
            go.Bar(
                x=hourly_trades.index,
                y=hourly_trades.values,
                name='Total Trades',
                marker_color='rgba(55, 125, 255, 0.7)',
                hovertemplate="Hour: %{x}<br>Trades: %{y}<extra></extra>"
            ),
            row=row, col=col
        )
        
        # Add win rate line
        fig.add_trace(
            go.Scatter(
                x=win_rates.index,
                y=win_rates.values,
                name='Win Rate %',
                line=dict(color='#22c55e', width=2),
                yaxis='y2',
                hovertemplate="Hour: %{x}<br>Win Rate: %{y:.1f}%<extra></extra>"
            ),
            row=row, col=col,
            secondary_y=True
        )
        
        # Update axes
        fig.update_xaxes(
            title_text='Hour of Day',
            tickmode='array',
            ticktext=[f'{i:02d}:00' for i in range(24)],
            tickvals=list(range(24)),
            row=row, col=col
        )
        fig.update_yaxes(title_text='Number of Trades', row=row, col=col)
        fig.update_yaxes(title_text='Win Rate %', secondary_y=True, row=row, col=col)
        
        return fig

    def _add_risk_management_analysis(self, fig, row, col):
        """Enhanced risk management analysis with better visualization"""
        # Prepare risk/reward data
        data = []
        for trade in self.trades:
            data.append({
                'risk': trade['trade_metrics']['risk_amount'],
                'reward': abs(trade['pnl']) if trade['pnl'] > 0 else 0,
                'pnl': trade['pnl'],
                'id': trade['id']
            })
        
        df = pd.DataFrame(data)
        df['rr_ratio'] = df['reward'] / df['risk'].where(df['risk'] != 0, 1)
        
        # Create scatter plot
        fig.add_trace(
            go.Scatter(
                x=df['risk'],
                y=df['reward'],
                mode='markers',
                name='Trades',
                marker=dict(
                    color=df['pnl'],
                    colorscale='RdYlGn',
                    size=10,
                    line=dict(width=1, color='black'),
                    showscale=True,
                    colorbar=dict(title='P&L ($)')
                ),
                text=[f"Trade #{id}<br>R/R: {rr:.2f}<br>PnL: ${pnl:.2f}" 
                      for id, rr, pnl in zip(df['id'], df['rr_ratio'], df['pnl'])],
                hoverinfo='text'
            ),
            row=row, col=col
        )
        
        # Add reference lines
        max_val = max(df['risk'].max(), df['reward'].max())
        for rr_line in [0.5, 1.0, 2.0]:
            fig.add_trace(
                go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val * rr_line],
                    mode='lines',
                    name=f'{rr_line}:1 R/R',
                    line=dict(dash='dash', color='rgba(128, 128, 128, 0.6)', width=1),
                    hoverinfo='none'
                ),
                row=row, col=col
            )
        
        # Add statistics annotation
        positive_rr = df[df['rr_ratio'] > 0]['rr_ratio']
        fig.add_annotation(
            text=(f"Avg R/R: {positive_rr.mean():.2f}<br>"
                  f"Max R/R: {positive_rr.max():.2f}<br>"
                  f"Trades > 1R: {(df['rr_ratio'] > 1).sum()}"),
            xref="x", yref="y",
            x=df['risk'].min(),
            y=df['reward'].max(),
            showarrow=False,
            font=dict(size=10),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
            borderpad=4,
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text='Risk ($)', row=row, col=col)
        fig.update_yaxes(title_text='Reward ($)', row=row, col=col)
        
        return fig

    def _add_duration_analysis(self, fig, row, col):
        """Add trade duration analysis"""
        durations = [t['trade_metrics']['time_in_trade'] for t in self.trades]
        
        fig.add_trace(
            go.Histogram(
                x=durations,
                name='Trade Duration',
                nbinsx=20,
                marker_color='lightblue'
            ),
            row=row, col=col
        )
        
        return fig  # Return the figure after adding trace

    def _format_exit_reason(self, trade):
        """Format exit reason in a professional manner"""
        if trade.get('exit_type') == 'Stop Loss':
            return f"Position stopped out at {trade.get('stop_loss', 0):.2f} to protect capital"
        elif trade.get('exit_type') == 'Take Profit':
            return f"Target reached at {trade.get('take_profit', 0):.2f}, securing profits"
        elif trade.get('exit_type') == 'Market Conditions':
            return "Market conditions deteriorated, defensive exit executed"
        else:
            return "Regular exit based on technical factors"

    def _format_technical_signals(self, trade):
        """Format technical analysis signals at exit"""
        signals = []
        
        if 'ma_cross' in trade:
            signals.append(f"MA Cross: {trade['ma_cross']}")
        if 'momentum' in trade:
            signals.append(f"Momentum: {trade['momentum']}")
        if 'volume_spike' in trade:
            signals.append(f"Volume: {trade['volume_spike']}")
            
        return signals if signals else ['No significant technical signals']

    def _identify_key_levels(self, trade):
        """Identify important price levels at exit"""
        levels = []
        
        if 'support' in trade:
            levels.append(f"Support: {trade['support']:.2f}")
        if 'resistance' in trade:
            levels.append(f"Resistance: {trade['resistance']:.2f}")
        if 'pivot' in trade:
            levels.append(f"Pivot: {trade['pivot']:.2f}")
            
        return levels if levels else ['No key levels identified']

    def _format_entry_analysis(self, trade):
        """Format entry analysis for display"""
        setup = trade.get('entry_analysis', {})
        return {
            'setup_type': setup.get('setup_type', 'Unknown Setup'),
            'trend_context': self._format_trend_context(setup.get('trend_analysis', {})),
            'signals': self._format_technical_signals(setup.get('technical_signals', {})),
            'price_action': setup.get('price_action', {}),
            'key_levels': setup.get('support_resistance', []),
            'quality_metrics': self._format_quality_metrics(setup.get('risk_metrics', {}))
        }

    def _format_trade_modifications(self, trade):
        """Format trade modifications for display"""
        mods = trade.get('modifications', [])
        return [{
            'time': mod['time'],
            'type': mod['type'],
            'description': f"Changed from {mod['old_value']:.2f} to {mod['new_value']:.2f}",
            'reason': mod['reason'],
            'market_context': mod['market_context']
        } for mod in mods]

    def _create_trade_summary(self, trade):
        """Create comprehensive trade summary"""
        return {
            'entry_quality': self._calculate_entry_quality(trade),
            'management_quality': self._calculate_management_quality(trade),
            'execution_notes': self._generate_execution_notes(trade),
            'learning_points': self._identify_learning_points(trade)
        }

    def _format_trend_context(self, trend_analysis):
        """Format trend analysis data into readable context"""
        try:
            # Skip if no trend analysis data
            if not trend_analysis:
                return "No trend analysis available"
                
            formatted_text = []
            
            # Format timeframe alignment
            if 'alignment' in trend_analysis:
                formatted_text.append(trend_analysis['alignment'])
            
            # Add trend details for each timeframe
            timeframes = {
                'daily': 'Daily trend',
                'h4': '4H trend',
                'm5': '5M trend'
            }
            
            for tf, label in timeframes.items():
                if tf in trend_analysis:
                    formatted_text.append(f"{label}: {trend_analysis[tf]}")
            
            # Add trend strength if available
            if 'strength' in trend_analysis:
                strength = float(trend_analysis['strength'])
                if strength > 0.8:
                    strength_desc = "Very strong"
                elif strength > 0.6:
                    strength_desc = "Strong"
                elif strength > 0.4:
                    strength_desc = "Moderate"
                else:
                    strength_desc = "Weak"
                formatted_text.append(f"Trend strength: {strength_desc} ({strength:.2f})")
            
            return "<br>".join(formatted_text)
            
        except Exception as e:
            print(f"Error formatting trend context: {e}")
            return "Error formatting trend analysis"

    def _format_quality_metrics(self, risk_metrics):
        """Format risk and quality metrics into readable format"""
        try:
            if not risk_metrics:
                return "Quality metrics not available"
                
            metrics = {
                'Risk/Reward Ratio': f"{risk_metrics.get('risk_reward', 0):.2f}",
                'Stop Distance': f"{risk_metrics.get('stop_distance', 0):.2f} pips",
                'Setup Quality': self._get_quality_description(risk_metrics.get('quality_score', 0)),
                'Risk Rating': self._get_risk_rating(risk_metrics)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error formatting quality metrics: {e}")
            return "Error formatting quality metrics"
            
    def _get_quality_description(self, quality_score):
        """Convert quality score to descriptive text"""
        try:
            score = float(quality_score)
            if score >= 0.8:
                return "Excellent (A)"
            elif score >= 0.6:
                return "Good (B)"
            elif score >= 0.4:
                return "Moderate (C)"
            else:
                return "Poor (D)"
        except (ValueError, TypeError):
            return "Unrated"
            
    def _get_risk_rating(self, metrics):
        """Calculate risk rating based on multiple factors"""
        try:
            risk_score = 0
            
            # Risk/Reward contribution
            rr_ratio = float(metrics.get('risk_reward', 0))
            if rr_ratio >= 2.0:
                risk_score += 3
            elif rr_ratio >= 1.5:
                risk_score += 2
            elif rr_ratio >= 1.0:
                risk_score += 1
                
            # Stop distance contribution
            stop_distance = float(metrics.get('stop_distance', 0))
            if stop_distance <= CONFIG['min_stop_distance']:
                risk_score += 1
            elif stop_distance <= CONFIG['max_stop_distance']:
                risk_score += 2
                
            # Quality score contribution
            quality = float(metrics.get('quality_score', 0))
            if quality >= 0.7:
                risk_score += 2
            elif quality >= 0.5:
                risk_score += 1
                
            # Convert score to rating
            if risk_score >= 6:
                return "Low Risk"
            elif risk_score >= 4:
                return "Moderate Risk"
            else:
                return "High Risk"
                
        except Exception:
            return "Risk Unrated"

    def _calculate_entry_quality(self, trade):
        """Calculate entry quality score based on multiple factors"""
        try:
            # Start with base score from signal quality
            score = float(trade.get('entry_reason', {}).get('signal_quality', 0.5))
            
            # Adjust for trend alignment
            if 'trend_alignment' in trade.get('entry_reason', {}):
                if trade['entry_reason']['trend_alignment'] == 'Strong alignment':
                    score += 0.2
                elif trade['entry_reason']['trend_alignment'] == 'Moderate alignment':
                    score += 0.1
            
            # Adjust for pattern quality
            if trade.get('entry_reason', {}).get('pattern_detected') not in ['No specific pattern', 'None']:
                score += 0.15
            
            # Key level proximity bonus
            if trade.get('entry_reason', {}).get('key_levels', []):
                score += 0.15
                
            return min(1.0, score)  # Cap at 1.0
            
        except Exception as e:
            print(f"Error calculating entry quality: {e}")
            return 0.5

    def _calculate_management_quality(self, trade):
        """Calculate trade management quality score"""
        try:
            score = 0.5  # Start with neutral score
            
            # Evaluate stop loss management
            if trade.get('modifications', []):
                score += 0.2  # Active management bonus
                
                # Check if stops were adjusted protectively
                protective_adjustments = sum(1 for mod in trade['modifications'] 
                                          if 'stop' in mod['type'].lower() and 
                                          float(mod['new_value']) > float(mod['old_value']))
                if protective_adjustments > 0:
                    score += 0.1
            
            # Evaluate profit taking
            pnl = float(trade['pnl'])
            risk_amount = float(trade['trade_metrics'].get('risk_amount', 0))
            
            # Avoid division by zero
            if risk_amount > 0:
                if pnl > 0:  # Winning trade
                    rr_achieved = pnl / risk_amount
                    if rr_achieved >= 2.0:
                        score += 0.2  # Good profit target
                elif pnl < 0:  # Losing trade
                    if abs(pnl) < risk_amount:
                        score += 0.1  # Good loss management
            
            return min(1.0, score)
            
        except Exception as e:
            print(f"Error calculating management quality: {e}")
            return 0.5  # Return neutral score on error

    def _generate_execution_notes(self, trade):
        """Generate notes about trade execution quality"""
        try:
            notes = []
            
            # Entry execution
            entry_slippage = abs(float(trade['entry_price']) - 
                               float(trade.get('intended_entry', trade['entry_price'])))
            if entry_slippage > 0:
                notes.append(f"Entry slippage: {entry_slippage:.2f} points")
            
            # Exit execution
            if trade.get('exit_type') == 'Stop Loss':
                notes.append("Stopped out at planned level")
            elif trade.get('exit_type') == 'Take Profit':
                notes.append("Target reached as planned")
            
            # Trade management
            if trade.get('modifications', []):
                notes.append(f"Active management with {len(trade['modifications'])} adjustments")
            else:
                notes.append("No position adjustments made")
            
            return notes if notes else ["Standard execution"]
            
        except Exception as e:
            print(f"Error generating execution notes: {e}")
            return ["Error analyzing execution"]

    def _identify_learning_points(self, trade):
        """Identify key learning points from the trade"""
        try:
            lessons = []
            
            # Entry analysis
            entry_quality = self._calculate_entry_quality(trade)
            if entry_quality < 0.6:
                lessons.append("Entry criteria could be more selective")
            
            # Management analysis
            management_quality = self._calculate_management_quality(trade)
            if management_quality < 0.6:
                lessons.append("Trade management could be more active")
            
            # Outcome analysis
            pnl = float(trade['pnl'])
            if pnl > 0:
                lessons.append("Successful trade - review positive factors")
            else:
                lessons.append("Review stop placement and market analysis")
            
            # Pattern recognition
            if trade.get('entry_reason', {}).get('pattern_detected') != 'None':
                lessons.append(f"Pattern ({trade['entry_reason']['pattern_detected']}) validation")
            
            return lessons if lessons else ["Standard trade execution"]
            
        except Exception as e:
            print(f"Error identifying learning points: {e}")
            return ["Unable to analyze learning points"]

    def _calculate_average_win(self):
        """Calculate average winning trade"""
        winning_trades = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        return np.mean(winning_trades) if winning_trades else 0

    def _calculate_average_loss(self):
        """Calculate average losing trade"""
        losing_trades = [t['pnl'] for t in self.trades if t['pnl'] < 0]
        return np.mean(losing_trades) if losing_trades else 0

    def _calculate_largest_win(self):
        """Find largest winning trade"""
        winning_trades = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        return max(winning_trades) if winning_trades else 0

    def _calculate_largest_loss(self):
        """Find largest losing trade"""
        losing_trades = [t['pnl'] for t in self.trades if t['pnl'] < 0]
        return min(losing_trades) if losing_trades else 0

    def _calculate_average_duration(self):
        """Find largest losing trade"""
        """Calculate average trade duration in hours"""
        durations = [t['trade_metrics']['time_in_trade'] for t in self.trades]
        return np.mean(durations) if durations else 0
        losing_trades = [t['pnl'] for t in self.trades if t['pnl'] < 0]
        return min(losing_trades) if losing_trades else 0

    def _calculate_average_duration(self):
        """Calculate average trade duration in hours"""
        durations = [t['trade_metrics']['time_in_trade'] for t in self.trades]
        return np.mean(durations) if durations else 0