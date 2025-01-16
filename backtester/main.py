from data_loader import DataLoader
from strategy import TradingStrategy
from report_generator import ReportGenerator
from config import CONFIG
import os

def main():
    # Initialize with your API credentials
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    # Load and prepare data
    loader = DataLoader(api_key, api_secret)
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
