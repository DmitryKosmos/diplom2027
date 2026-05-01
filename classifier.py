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


        # Загрузка конфигурации
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                self.config.update(user_config)
                logger.info(f"Загружена конфигурация из {config_path}")

        # Сохранение конфигурации
        config_save_path = self.model_dir / 'config.json'
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        # Загрузка данных
        logger.info(f"Загрузка данных из {data_path}")
        df = pd.read_csv(data_path, encoding='utf-8')
        self._validate_data(df)

        texts = df['text'].tolist()
        labels = df['label'].tolist()

        # Определение количества классов
        unique_labels = sorted(set(labels))
        n_classes = len(unique_labels)
        self.config['n_classes'] = n_classes
        logger.info(f"Количество классов: {n_classes}")
        logger.info(f"Уникальные метки: {unique_labels}")

        # Кодирование меток (для многоклассовой классификации)
        self.label_encoder = LabelEncoder()
        labels_encoded = self.label_encoder.fit_transform(labels)

        # Подготовка данных для BiLSTM
        X, y = self._prepare_data_bilstm(texts, labels_encoded, fit_tokenizer=True)

        # Сохранение токенизатора
        tokenizer_path = self.model_dir / 'tokenizer.pickle'
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        # Сохранение LabelEncoder
        encoder_path = self.model_dir / 'label_encoder.pickle'
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        # Разделение на train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config['val_size'],
            random_state=42,
            stratify=y
        )

        logger.info(f"Размер обучающей выборки: {len(X_train)}")
        logger.info(f"Размер валидационной выборки: {len(X_val)}")

        # Построение модели
        self.model = self._build_bilstm_model(n_classes)


        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=str(self.model_dir / 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            )
        ]

        # Обучение
        logger.info("Начало обучения...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Обучение завершено!")

        # Оценка на валидационной выборке
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

        return history

    def train_transformer(self, data_path: str, model_name: str = "distilbert-base-multilingual-cased"):
        """
        Обучение Transformer модели (DistilBERT).

        Args:
            data_path: Путь к CSV файлу с данными
            model_name: Название предобученной модели
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Библиотека transformers не установлена. Установите: pip install transformers")

        logger.info("=" * 50)
        logger.info("Запуск обучения Transformer модели")
        logger.info("=" * 50)

        self.model_type = 'transformer'

        # Загрузка данных
        logger.info(f"Загрузка данных из {data_path}")
        df = pd.read_csv(data_path, encoding='utf-8')
        self._validate_data(df)

        texts = df['text'].tolist()
        labels = df['label'].tolist()

        # Определение количества классов
        unique_labels = sorted(set(labels))
        n_classes = len(unique_labels)
        logger.info(f"Количество классов: {n_classes}")

        # Кодирование меток
        self.label_encoder = LabelEncoder()
        labels_encoded = self.label_encoder.fit_transform(labels)

        # Сохранение LabelEncoder
        encoder_path = self.model_dir / 'label_encoder.pickle'
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        # Разделение на train/validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels_encoded,
            test_size=0.2,
            random_state=42,
            stratify=labels_encoded
        )

        # Загрузка токенизатора
        logger.info(f"Загрузка токенизатора {model_name}")
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        # Токенизация
        def tokenize_function(texts):
            return tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='tf'
            )

        train_encodings = tokenize_function(train_texts)
        val_encodings = tokenize_function(val_texts)

        # Создание TensorFlow датасетов
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            train_labels
        )).batch(16)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            val_labels
        )).batch(16)

        # Загрузка модели
        logger.info(f"Загрузка модели {model_name}")
        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=n_classes
        )

        # Компиляция
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=str(self.model_dir / 'best_transformer'),
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True,
                verbose=1
            )
        ]

        # Обучение
        logger.info("Начало обучения...")
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=3,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Обучение завершено!")

        # Сохранение конфигурации
        config_save_path = self.model_dir / 'config.json'
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_type': 'transformer',
                'model_name': model_name,
                'n_classes': n_classes
            }, f, indent=2, ensure_ascii=False)

        # Сохранение токенизатора
        tokenizer.save_pretrained(str(self.model_dir / 'tokenizer'))

        return history

    def load_model(self, model_dir: str):
        """
        Загрузка сохраненной модели и артефактов.

        Args:
            model_dir: Директория с сохраненной моделью
        """
        model_dir = Path(model_dir)

        # Загрузка конфигурации
        config_path = model_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config.update(json.load(f))

        # Определение типа модели
        if (model_dir / 'best_model.h5').exists():
            self.model_type = 'bilstm'
            logger.info("Загрузка BiLSTM модели...")

            # Загрузка модели с пользовательским слоем
            self.model = tf.keras.models.load_model(
                str(model_dir / 'best_model.h5'),
                custom_objects={'AttentionLayer': AttentionLayer}
            )

            # Загрузка токенизатора
            tokenizer_path = model_dir / 'tokenizer.pickle'
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)

        elif (model_dir / 'best_transformer').exists():
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Библиотека transformers не установлена")

            self.model_type = 'transformer'
            logger.info("Загрузка Transformer модели...")

            # Загрузка модели
            self.model = TFDistilBertForSequenceClassification.from_pretrained(
                str(model_dir / 'best_transformer')
            )

            # Загрузка токенизатора
            tokenizer_path = model_dir / 'tokenizer'
            if tokenizer_path.exists():
                self.tokenizer = DistilBertTokenizer.from_pretrained(str(tokenizer_path))
        else:
            raise FileNotFoundError(f"Не найдена модель в директории {model_dir}")

        # Загрузка LabelEncoder
        encoder_path = model_dir / 'label_encoder.pickle'
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)

        logger.info(f"Модель загружена из {model_dir}")
