from backtester.data_loader import DataLoader
from backtester.strategy import TradingStrategy
from backtester.report_generator import ReportGenerator
from backtester.config import CONFIG

def main():
    # Initialize data loader with MT5
    loader = DataLoader()
    
    # Load and prepare data
    daily_data = loader.fetch_data('1d')
    h4_data = loader.fetch_data('4h')
    m5_data = loader.fetch_data('5m')
    
    # Add technical indicators
    daily_data = loader.add_indicators(daily_data)
    h4_data = loader.add_indicators(h4_data)
    m5_data = loader.add_indicators(m5_data)
    
    # Run strategy
    strategy = TradingStrategy()
    signals = strategy.analyze_trend(daily_data, h4_data, m5_data)
    
    # Generate report
    report = ReportGenerator(signals, strategy.trades, m5_data)
    report.generate_report()

if __name__ == "__main__":
    main()
