import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    BASE_DIR = Path(__file__).parent.parent
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Server settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    
    # Game settings
    MAX_CONCURRENT_GAMES = int(os.getenv('MAX_CONCURRENT_GAMES', 100))
    GAME_TIMEOUT = int(os.getenv('GAME_TIMEOUT', 3600))  # 1 hour
    
    # Data storage
    DATA_DIR = BASE_DIR / 'data'
    LOGS_DIR = BASE_DIR / 'logs'
    
    # Redis settings (for scaling)
    REDIS_URL = os.getenv('REDIS_URL', None)
    USE_REDIS = os.getenv('USE_REDIS', 'False').lower() == 'true'
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', str(BASE_DIR / 'models'))
    USE_GPU = os.getenv('USE_GPU', 'False').lower() == 'true'
    
    # MCTS settings
    MCTS_SIMULATIONS = int(os.getenv('MCTS_SIMULATIONS', 800))
    MCTS_CPUCT = float(os.getenv('MCTS_CPUCT', 1.0))
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        Path(cls.MODEL_PATH).mkdir(exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    HOST = 'localhost'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.getenv('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY must be set in production")

class CloudConfig(ProductionConfig):
    """Cloud deployment configuration"""
    USE_REDIS = True
    REDIS_URL = os.getenv('REDIS_URL')

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'cloud': CloudConfig,
    'default': DevelopmentConfig
}

def get_config(env=None):
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])
