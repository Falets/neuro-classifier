import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATA_FOLDER = os.getenv('DATA_FOLDER', './data')
    WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', 500))
    STEP = int(os.getenv('STEP', 100))
    EPOCHS = int(os.getenv('EPOCHS', 50))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
    CLASS_NAMES = ['EPSP', 'PS']