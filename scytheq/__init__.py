from pathlib import Path
import logging
import re

from .utils import *

__version__ = '2.0.1'

def update_version(v):
    global __version__
    __version__ = v
    
    # Update this file
    with open(__file__, 'r') as f:
        content = f.read()
    content = re.sub(r"__version__ = '[\d.]+'", f"__version__ = '{v}'", content)
    with open(__file__, 'w') as f:
        f.write(content)
    
    # Update pyproject.toml
    with open('pyproject.toml', 'r') as f:
        content = f.read()
    content = re.sub(r'version = "[\d.]+"', f'version = "{v}"', content)
    with open('pyproject.toml', 'w') as f:
        f.write(content)
    
    print(f"Version updated to {v}")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='scytheq.log',
    filemode='a',
)
logger = logging.getLogger(__name__)

# Global config directory setting
_config_dir = None

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
        import scytheq
        scytheq.set_config_dir('/path/to/config')
        ```
        
    Note:
        - 設定檔目錄必須包含所需的 JSON 設定檔
        - 如果未設定，則會使用預設的 config 目錄
    """
    global _custom_config_dir
    _custom_config_dir = Path(path)

def get_config_dir() -> Path:
    config_path = _config_dir or Path(__file__).parent.parent / 'config'
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration directory not found: {config_path}\n"
            "Please ensure the config directory exists or set a custom config directory using set_config_dir()."
        )
    
    return config_path
