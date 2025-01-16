
import os
from dotenv import load_dotenv
from backtester.main import main

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the backtest
    main()