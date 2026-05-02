import subprocess
import sys
import os

# Исправление кодировки
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')


def create_better_data():
    """Создание данных с явными различиями между классами"""
    import pandas as pd
    import json
    import random

    print("Создание улучшенных данных...")

    # Явно положительные слова
    positive_words = ['отличный', 'прекрасный', 'хороший', 'супер', 'классный',
                      'доволен', 'рекомендую', 'качественный', 'понравился', 'спасибо']

    # Явно отрицательные слова
    negative_words = ['ужасный', 'плохой', 'отвратительный', 'брак', 'сломался',
                      'разочарование', 'кошмар', 'некачественный', 'обман', 'деньги на ветер']

    train_data = []

    # Создаем 150 положительных примеров
    for i in range(150):
        words = random.sample(positive_words, random.randint(2, 4))
        text = ' '.join(words) + '!'
        train_data.append({'text': text, 'label': 1})

    # Создаем 150 отрицательных примеров
    for i in range(150):
        words = random.sample(negative_words, random.randint(2, 4))
        text = ' '.join(words) + '!'
        train_data.append({'text': text, 'label': 0})

    # Тестовые данные
    test_data = [
        {'text': 'Отличный продукт, всем советую!', 'label': 1},
        {'text': 'Ужасное качество, полный брак', 'label': 0},
        {'text': 'Прекрасная вещь, спасибо', 'label': 1},
        {'text': 'Кошмар, не покупайте это', 'label': 0},
        {'text': 'Хороший товар, доволен', 'label': 1},
        {'text': 'Деньги на ветер, обман', 'label': 0},
        {'text': 'Качественно, рекомендую', 'label': 1},
        {'text': 'Сломалось в первый день', 'label': 0},
        {'text': 'Супер покупка!', 'label': 1},
        {'text': 'Полное разочарование', 'label': 0},
    ]

    # Сохраняем
    pd.DataFrame(train_data).sample(frac=1).to_csv('train.csv', index=False, encoding='utf-8')
    pd.DataFrame(test_data).to_csv('test.csv', index=False, encoding='utf-8')

    print(f"Создано: {len(train_data)} обучающих и {len(test_data)} тестовых примеров")

    # Конфигурация
    config = {
        "max_tokens": 5000,
        "max_len": 50,
        "epochs": 20,
        "batch_size": 32,
        "val_size": 0.2
    }

    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def run_training():
    print("\n" + "=" * 50)
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 50)
    subprocess.run([sys.executable, "classifier.py", "--mode", "train",
                    "--data", "train.csv", "--model_dir", "./my_model",
                    "--config", "config.json"], check=True)


def run_evaluation():
    print("\n" + "=" * 50)
    print("ОЦЕНКА МОДЕЛИ")
    print("=" * 50)
    subprocess.run([sys.executable, "classifier.py", "--mode", "evaluate",
                    "--data", "test.csv", "--model_dir", "./my_model"], check=True)


def run_predictions():
    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ")
    print("=" * 50)

    test_texts = [
        ("Отличный продукт!", 1),
        ("Ужасное качество", 0),
        ("Хорошая вещь", 1),
        ("Полный брак", 0),
    ]

    correct = 0
    for text, expected in test_texts:
        result = subprocess.run([sys.executable, "classifier.py", "--mode", "predict",
                                 "--model_dir", "./my_model", "--text", text],
                                capture_output=True, text=True, encoding='utf-8')

        # Извлекаем предсказание
        import re
        match = re.search(r'Предсказанный класс:\s*(\d+)', result.stdout)
        if match:
            pred = int(match.group(1))
            is_correct = pred == expected
            correct += is_correct
            print(f"Текст: {text}")
            print(f"Предсказано: {pred} | Ожидается: {expected} {'✓' if is_correct else '✗'}")

    print(f"\nТочность: {correct / len(test_texts):.2%}")


if __name__ == "__main__":
    create_better_data()
    run_training()
    run_evaluation()
    run_predictions()
