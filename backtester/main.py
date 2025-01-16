from xau.backtester.data_loader import DataLoader
from xau.backtester.strategy import TradingStrategy
from xau.backtester.report_generator import ReportGenerator
from xau.backtester.config import CONFIG

def main():
    print("Starting backtest...")
    
    # Initialize data loader with MT5
    loader = DataLoader()
    
    # Load and prepare data
    print("Loading data...")
    daily_data = loader.fetch_data('1d')
    h4_data = loader.fetch_data('4h')
    m5_data = loader.fetch_data('5m')
    
    print(f"Data loaded: {len(m5_data)} 5m bars, {len(h4_data)} 4h bars, {len(daily_data)} daily bars")
    
    # Add technical indicators
    print("Adding indicators...")
    daily_data = loader.add_indicators(daily_data)
    h4_data = loader.add_indicators(h4_data)
    m5_data = loader.add_indicators(m5_data)
    
    # Run strategy
    print("Running strategy...")
    strategy = TradingStrategy()
    signals = strategy.analyze_trend(daily_data, h4_data, m5_data)
    print(f"Strategy complete. Generated {len(signals)} signals and {len(strategy.trades)} trades")
    
    # Calculate equity curve
    initial_balance = CONFIG['initial_balance']
    equity_curve = [initial_balance]
    
    for trade in strategy.trades:
        equity_curve.append(equity_curve[-1] + trade['pnl'])
    
    # Generate report
    print("Generating report...")
    report = ReportGenerator(equity_curve, strategy.trades, m5_data)
    report.generate_report()
    print("Backtest complete!")

if __name__ == "__main__":
    main()
