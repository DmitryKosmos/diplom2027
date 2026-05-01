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


class TextClassifier:
    """
    Основной класс для классификации текстов.
    Поддерживает BiLSTM + Attention и Transformer модели.
    """

    def __init__(self, model_dir: str = "./models"):
        """
        Инициализация классификатора.

        Args:
            model_dir: Директория для сохранения моделей и артефактов
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.config = DEFAULT_CONFIG.copy()
        self.model_type = None  # 'bilstm' или 'transformer'

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Валидация входных данных.

        Args:
            df: DataFrame с данными

        Returns:
            True если данные валидны
        """
        required_columns = ['text', 'label']

        # Проверка наличия обязательных колонок
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует обязательная колонка: {col}")

        # Проверка типов данных
        if not pd.api.types.is_string_dtype(df['text']):
            logger.warning("Колонка 'text' должна быть строкового типа. Выполняется преобразование...")
            df['text'] = df['text'].astype(str)

        if not pd.api.types.is_integer_dtype(df['label']):
            try:
                df['label'] = df['label'].astype(int)
            except:
                raise ValueError("Колонка 'label' должна содержать целочисленные значения")

        # Проверка на пустые значения
        if df['text'].isnull().any() or df['label'].isnull().any():
            raise ValueError("Данные содержат пропущенные значения")

        return True

    def _prepare_data_bilstm(self, texts: List[str], labels: List[int], fit_tokenizer: bool = False):
        """
        Подготовка данных для BiLSTM модели.

        Args:
            texts: Список текстов
            labels: Список меток
            fit_tokenizer: Обучать ли токенизатор заново

        Returns:
            Подготовленные данные
        """
        if fit_tokenizer:
            self.tokenizer = Tokenizer(
                num_words=self.config['max_tokens'],
                oov_token='<OOV>'
            )
            self.tokenizer.fit_on_texts(texts)

            # Сохранение словаря
            word_index = self.tokenizer.word_index
            logger.info(f"Размер словаря: {len(word_index)} токенов")

        # Преобразование текстов в последовательности
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=self.config['max_len'],
            padding='post',
            truncating='post'
        )

        return padded, np.array(labels)


    def _build_bilstm_model(self, n_classes: int) -> Model:
        """
        Построение архитектуры BiLSTM + Attention.

        Args:
            n_classes: Количество классов

        Returns:
            Модель Keras
        """
        # Входной слой
        inputs = Input(shape=(self.config['max_len'],))

        # Embedding слой
        embedding = layers.Embedding(
            input_dim=self.config['max_tokens'],
            output_dim=self.config['embedding_dim'],
            input_length=self.config['max_len']
        )(inputs)

        # Bidirectional LSTM
        lstm = layers.Bidirectional(
            layers.LSTM(
                self.config['lstm_units'],
                return_sequences=True,
                dropout=0.3,
                recurrent_dropout=0.3
            )
        )(embedding)

        # Dropout для регуляризации
        dropout = layers.Dropout(0.3)(lstm)

        # Attention слой
        attention = AttentionLayer()(dropout)

        # Полносвязные слои
        dense1 = layers.Dense(64, activation='relu')(attention)
        dropout2 = layers.Dropout(0.3)(dense1)

        # Выходной слой
        if n_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(dropout2)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            outputs = layers.Dense(n_classes, activation='softmax')(dropout2)
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']

        # Создание модели
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=metrics
        )

        model.summary(print_fn=logger.info)
        return model

    def train_bilstm(self, data_path: str, config_path: Optional[str] = None):
        """
        Обучение BiLSTM модели.

        Args:
            data_path: Путь к CSV файлу с данными
            config_path: Путь к JSON файлу с конфигурацией (опционально)
        """
        logger.info("=" * 50)
        logger.info("Запуск обучения BiLSTM модели")
        logger.info("=" * 50)

        self.model_type = 'bilstm'

