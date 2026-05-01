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


class AttentionLayer(layers.Layer):
    """
    Кастомный слой внимания для BiLSTM модели.
    Взвешивает важность каждого временного шага.
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Вычисление весов внимания
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

