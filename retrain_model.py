# retrain_model.py
import subprocess
import sys
import os
import json
import pandas as pd  # Добавлен недостающий импорт
import random

# Исправление кодировки для Windows
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')


def create_optimized_config():
    """Создание оптимизированной конфигурации"""

    config = {
        "max_tokens": 20000,
        "max_len": 200,
        "embedding_dim": 512,
        "lstm_units": 512,
        "epochs": 30,
        "batch_size": 128,
        "val_size": 0.15,
        "n_classes": 2,
        "dropout_rate": 0.5,
        "learning_rate": 0.0005
    }

    with open('config_optimized.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("✅ Создан оптимизированный конфиг: config_optimized.json")
    return config


def add_more_training_data():
    """Добавление сложных случаев в обучающие данные"""

    # Загружаем существующие данные
    try:
        df = pd.read_csv('train_big.csv', encoding='utf-8')
        print(f"📂 Загружено {len(df)} существующих записей")
    except FileNotFoundError:
        print("⚠️ Файл train_big.csv не найден, создаю новый датасет")
        df = pd.DataFrame(columns=['text', 'label'])
    except Exception as e:
        print(f"⚠️ Ошибка при загрузке: {e}")
        df = pd.DataFrame(columns=['text', 'label'])

    # Сложные случаи, на которых модель может ошибаться
    hard_cases = [
        # Нейтральные/смешанные отзывы
        ("Нормальный товар, но дороговато", 1),
        ("Качество хорошее, но цена кусается", 1),
        ("Доставили быстро, но товар немного поцарапан", 0),
        ("В целом неплохо, но есть мелкие недостатки", 1),
        ("Ожидал лучшего за такие деньги", 0),
        ("Цена качество соответствуют", 1),
        ("Средненько, но пойдет", 1),
        ("Не фонтан, но жить можно", 1),
        ("На троечку", 1),
        ("Могло быть и хуже", 1),

        # Саркастичные отзывы
        ("Отличный товар, сломался через час", 0),
        ("Супер качество, развалилось в руках", 0),
        ("Рекомендую, если хотите выбросить деньги", 0),
        ("Лучший брак в моей жизни", 0),

        # Отзывы с конструктивной критикой
        ("Хороший товар, но можно улучшить упаковку", 1),
        ("Качественно, но долго ждать", 1),
        ("Работает, но шумноват", 1),
        ("Нормально, но инструкция на китайском", 1),

        # Эмоциональные
        ("Вау! Супер! 👌🔥", 1),
        ("😡 Зря потратил деньги", 0),
        ("💔 разочарован", 0),
        ("❤️ влюблена в этот товар", 1),

        # Сравнительные
        ("Лучше чем я ожидал", 1),
        ("Хуже чем в прошлый раз", 0),
        ("Дороже чем везде", 0),
        ("Дешевле чем в магазине", 1),
    ]

    # Добавляем сложные случаи с вариациями
    new_data = []
    for text, label in hard_cases:
        # Добавляем базовый вариант
        new_data.append({'text': text, 'label': label})
        # Добавляем вариант с восклицательным знаком
        new_data.append({'text': text + "!", 'label': label})
        # Добавляем вариант с точкой
        new_data.append({'text': text + ".", 'label': label})

    # Добавляем новые комбинации положительных слов
    positive_words = ['отличный', 'прекрасный', 'супер', 'классный', 'доволен',
                      'рекомендую', 'качественный', 'понравился', 'спасибо', 'хороший']
    for i in range(100):
        num_words = random.randint(2, 5)
        words = random.sample(positive_words, min(num_words, len(positive_words)))
        text = ' '.join(words) + random.choice(['', '!', '!!', '.'])
        new_data.append({'text': text, 'label': 1})

    # Добавляем новые комбинации отрицательных слов
    negative_words = ['ужасный', 'плохой', 'брак', 'кошмар', 'разочарование',
                      'обман', 'сломался', 'некачественный', 'отвратительный', 'деньги на ветер']
    for i in range(100):
        num_words = random.randint(2, 5)
        words = random.sample(negative_words, min(num_words, len(negative_words)))
        text = ' '.join(words) + random.choice(['', '!', '!!', '.'])
        new_data.append({'text': text, 'label': 0})

    # Объединяем с существующими данными
    new_df = pd.DataFrame(new_data)
    combined_df = pd.concat([df, new_df], ignore_index=True)

    # Удаляем дубликаты
    combined_df = combined_df.drop_duplicates(subset=['text'])

    # Перемешиваем
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)

    # Сохраняем
    output_file = 'train_enhanced.csv'
    combined_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"✅ Добавлено {len(new_df)} новых записей")
    print(f"📊 Всего уникальных записей: {len(combined_df)}")
    print(f"   Положительных (1): {sum(combined_df['label'] == 1)}")
    print(f"   Отрицательных (0): {sum(combined_df['label'] == 0)}")
    print(f"\n📁 Данные сохранены в: {output_file}")

    return combined_df


def create_test_set():
    """Создание расширенного тестового набора"""

    test_cases = [
        # Основные тесты
        ("Это отличный продукт! Очень доволен!", 1),
        ("Ужасное качество, полный брак", 0),
        ("Супер! Рекомендую всем!", 1),
        ("Кошмар, деньги на ветер", 0),

        # Смешанные
        ("Хороший товар, но цена высоковата", 1),
        ("Неплохо, но могло быть лучше", 1),
        ("Качество отличное, но упаковка плохая", 1),
        ("Доставили быстро, но товар не тот", 0),

        # Нейтральные
        ("Обычный товар, ничего особенного", 1),
        ("Средненько, на троечку", 1),
        ("Цена качество соответствуют", 1),
        ("Нормально за свои деньги", 1),

        # Короткие
        ("👍👍👍", 1),
        ("👎", 0),
        ("Отстой", 0),
        ("Класс!", 1),
        ("🔥", 1),
        ("💩", 0),

        # Эмоциональные
        ("Очень очень очень доволен!!!", 1),
        ("Ужас ужас ужасный!!!", 0),
        ("😍😍😍", 1),
        ("😡😡😡", 0),

        # Реалистичные
        ("Товар пришел быстро, упакован хорошо, качество отличное", 1),
        ("Продавец не отвечает, товар не соответствует описанию", 0),
        ("Доставка долгая, но товар хороший", 1),
        ("Цена завышена, качество среднее", 0),

        # Сложные случаи
        ("Нормально", 1),
        ("Так себе", 0),
        ("Пойдет", 1),
        ("Развод", 0),
        ("Ок", 1),
        ("Не ок", 0),
    ]

    # Создаем DataFrame и сохраняем
    test_df = pd.DataFrame(test_cases, columns=['text', 'label'])
    test_df.to_csv('test_enhanced.csv', index=False, encoding='utf-8')

    print(f"✅ Создан тестовый набор: test_enhanced.csv")
    print(f"   Всего примеров: {len(test_df)}")
    print(f"   Положительных: {sum(test_df['label'] == 1)}")
    print(f"   Отрицательных: {sum(test_df['label'] == 0)}")

    # Покажем несколько примеров
    print("\n📝 Примеры тестовых данных:")
    for i in range(min(5, len(test_df))):
        text = test_df.iloc[i]['text']
        label = test_df.iloc[i]['label']
        sentiment = "👍" if label == 1 else "👎"
        print(f"  {i + 1}. {text[:50]}... {sentiment}")


def run_training():
    """Запуск обучения с улучшенными параметрами"""

    print("\n" + "=" * 60)
    print("🎓 ЗАПУСК УЛУЧШЕННОГО ОБУЧЕНИЯ")
    print("=" * 60)

    # Проверяем наличие файлов
    if not os.path.exists('train_enhanced.csv'):
        print("❌ Файл train_enhanced.csv не найден!")
        return False

    cmd = [
        sys.executable, "classifier.py",
        "--mode", "train",
        "--data", "train_enhanced.csv",
        "--model_dir", "./my_enhanced_model",
        "--config", "config_optimized.json"
    ]

    print(f"🚀 Команда: {' '.join(cmd)}")
    print("\n⏳ Начинаю обучение (это может занять несколько минут)...\n")

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Обучение успешно завершено!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ Файл classifier.py не найден!")
        return False


def test_new_model():
    """Тестирование улучшенной модели"""

    print("\n" + "=" * 60)
    print("📊 ТЕСТИРОВАНИЕ УЛУЧШЕННОЙ МОДЕЛИ")
    print("=" * 60)

    # Проверяем существование модели
    if not os.path.exists('./my_enhanced_model'):
        print("❌ Модель не найдена в ./my_enhanced_model")
        return 0

    # Загружаем тестовые данные
    try:
        test_df = pd.read_csv('test_enhanced.csv', encoding='utf-8')
    except:
        print("❌ Файл test_enhanced.csv не найден")
        return 0

    import re
    correct = 0
    total = len(test_df)

    print(f"\n📋 Тестирование на {total} примерах:\n")
    print("-" * 80)
    print(f"{'ТЕКСТ':<50} {'ПРЕДСКАЗАНО':<12} {'ОЖИДАЕТСЯ':<12}")
    print("-" * 80)

    for idx, row in test_df.iterrows():
        text = row['text']
        expected = row['label']

        result = subprocess.run([
            sys.executable, "classifier.py",
            "--mode", "predict",
            "--model_dir", "./my_enhanced_model",
            "--text", text
        ], capture_output=True, text=True, encoding='utf-8')

        match = re.search(r'Предсказанный класс:\s*(\d+)', result.stdout)
        if match:
            pred = int(match.group(1))
            is_correct = pred == expected
            if is_correct:
                correct += 1

            # Обрезаем длинные тексты
            short_text = text[:47] + "..." if len(text) > 50 else text
            mark = "✓" if is_correct else "✗"
            print(f"{short_text:<50} {pred:<12} {expected:<12} {mark}")

    accuracy = (correct / total) * 100
    print("-" * 80)
    print(f"\n🎯 ИТОГОВАЯ ТОЧНОСТЬ: {accuracy:.1f}% ({correct}/{total})")

    return accuracy


def main():
    """Основная функция"""
    print("=" * 60)
    print("🚀 УЛУЧШЕНИЕ МОДЕЛИ КЛАССИФИКАЦИИ")
    print("=" * 60)

    # Шаг 1: Создаем улучшенную конфигурацию
    print("\n📝 Шаг 1 из 5: Создание конфигурации...")
    create_optimized_config()

    # Шаг 2: Добавляем больше данных
    print("\n📊 Шаг 2 из 5: Расширение обучающих данных...")
    add_more_training_data()

    # Шаг 3: Создаем тестовый набор
    print("\n🧪 Шаг 3 из 5: Создание тестового набора...")
    create_test_set()

    # Шаг 4: Обучаем улучшенную модель
    print("\n🎓 Шаг 4 из 5: Обучение улучшенной модели...")
    if not run_training():
        print("\n❌ Обучение прервано")
        return

    # Шаг 5: Тестируем
    print("\n📈 Шаг 5 из 5: Тестирование...")
    accuracy = test_new_model()

    # Финальный результат
    print("\n" + "=" * 60)
    if accuracy > 85:
        print("🎉 ОТЛИЧНО! Модель значительно улучшена!")
    elif accuracy > 80:
        print("👍 ХОРОШО! Модель стала лучше")
    elif accuracy > 75:
        print("👌 ПРИЕМЛЕМО, но можно ещё улучшить")
    else:
        print("🔄 Требуется дополнительная настройка")
    print("=" * 60)

    print("\n📁 Улучшенная модель сохранена в: ./my_enhanced_model")
    print("\n💡 Для использования модели:")
    print('   python classifier.py --mode predict --model_dir ./my_enhanced_model --text "Ваш текст"')
    print("\n📊 Для оценки на тестовых данных:")
    print('   python classifier.py --mode evaluate --data test_enhanced.csv --model_dir ./my_enhanced_model')


if __name__ == "__main__":
    main()