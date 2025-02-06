import logging

__version__ = '1.3.1'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='quantdev.log',
    filemode='a',
)
logger = logging.getLogger(__name__)