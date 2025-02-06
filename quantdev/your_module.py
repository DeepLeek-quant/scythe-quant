from .config import config
from pathlib import Path
import pandas as pd

def save_market_data(data: pd.DataFrame, symbol: str):
    """Save market data to specified location from config"""
    try:
        # Get save path from config
        save_dir = Path(config.data_config['save_directory'])
        save_path = save_dir / f"{symbol}_data.csv"
        
        # Create directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data.to_csv(save_path)
        logger.info(f"Successfully saved data for {symbol} to {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to save data for {symbol}: {str(e)}")
        raise

def get_api_credentials():
    """Get API credentials when needed"""
    return config.fugle_config['api_key'] 