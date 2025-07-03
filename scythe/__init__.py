from pathlib import Path
import logging

from .utils import *

__version__ = '2.0.1'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='scythe.log',
    filemode='a',
)
logger = logging.getLogger(__name__)

# Global config directory setting
_custom_config_dir = None

def set_config_dir(path: str | Path) -> None:
    """
    設定自訂的設定檔目錄路徑。
    
    Args:
        path: 設定檔目錄的路徑字串或 Path 物件
        
    Returns:
        None
        
    Examples:
        ```python
        # 設定自訂目錄
        import scythe
        scythe.set_config_dir('/path/to/config')
        ```
        
    Note:
        - 設定檔目錄必須包含所需的 JSON 設定檔
        - 如果未設定，則會使用預設的 config 目錄
    """
    global _custom_config_dir
    _custom_config_dir = Path(path)

def get_config_dir() -> Path:
    """
    Get the current configuration directory path.
    
    Returns:
        Path: The current config directory path
    """
    return _custom_config_dir or Path(__file__).parent.parent / 'config'
    

