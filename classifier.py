import os
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pickle
import warnings
import sys
import os
import locale
# Основные зависимости
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
# Для трансформеров (опционально)
try:
    from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Внимание: библиотека transformers не установлена. Режим train_transformer будет недоступен.")
# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Конфигурация по умолчанию
DEFAULT_CONFIG = {
    "max_tokens": 20000,
    "max_len": 200,
    "embedding_dim": 128,
    "lstm_units": 128,
    "epochs": 8,
    "batch_size": 32,
    "val_size": 0.2,
    "n_classes": 2
}
