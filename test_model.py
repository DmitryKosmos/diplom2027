# test_model.py
import subprocess
import sys
import os
import re  # Добавлен недостающий импорт

# Исправление кодировки для Windows
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')


def test_predictions():
    """Тестирование модели на разных текстах"""

    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ОБУЧЕННОЙ МОДЕЛИ")
    print("=" * 60)

    test_cases = [
        # Явно положительные
        ("Это отличный продукт!", 1),
        ("Супер качество, всем советую!", 1),
        ("Очень доволен покупкой, спасибо!", 1),
        ("Прекрасная вещь, рекомендую!", 1),
        ("Лучшее, что я покупал!", 1),

        # Явно отрицательные
        ("Ужасное качество, полный брак", 0),
        ("Кошмар, деньги на ветер", 0),
        ("Не покупайте это, обман", 0),
        ("Сломалось в первый день", 0),
        ("Отвратительный товар, разочарование", 0),

        # Смешанные/нейтральные (интересно посмотреть)
        ("Нормальный товар за свои деньги", 1),
        ("Цена качество соответствуют", 1),
        ("Есть недостатки, но в целом ок", 1),
        ("Бракованный экземпляр попался", 0),
        ("Ожидал лучшего качества", 0),

        # Сложные случаи с эмодзи
        ("Отличный товар :)", 1),
        ("Ужасный товар :(", 0),
        ("сууупер!!!", 1),
        ("кошмааар!!!", 0),

        # Короткие
        ("Супер!", 1),
        ("Отстой!", 0),
        ("👍", 1),
        ("👎", 0),
    ]

    correct = 0
    total = len(test_cases)

    print("\nРезультаты тестирования:\n")
    print("-" * 80)
    print(f"{'ТЕКСТ':<50} {'ПРЕДСКАЗАНО':<15} {'ОЖИДАЕТСЯ':<15}")
    print("-" * 80)

    for text, expected in test_cases:
        try:
            # Запускаем предсказание
            result = subprocess.run([
                sys.executable, "classifier.py",
                "--mode", "predict",
                "--model_dir", "./my_big_model",
                "--text", text
            ], capture_output=True, text=True, encoding='utf-8')

            # Извлекаем предсказание
            match = re.search(r'Предсказанный класс:\s*(\d+)', result.stdout)

            if match:
                pred = int(match.group(1))
                is_correct = pred == expected
                if is_correct:
                    correct += 1

                # Обрезаем текст для красивого вывода
                short_text = text[:47] + "..." if len(text) > 50 else text
                mark = "✓" if is_correct else "✗"
                print(f"{short_text:<50} {pred:<15} {expected:<15} {mark}")
            else:
                print(f"Не удалось распарсить результат для: {text}")
                print(f"Вывод: {result.stdout}")

        except Exception as e:
            print(f"Ошибка при тестировании '{text}': {e}")

    print("-" * 80)
    accuracy = (correct / total) * 100
    print(f"\nИТОГОВАЯ ТОЧНОСТЬ: {accuracy:.1f}% ({correct}/{total})")

    if accuracy > 90:
        print("\n🎉 ОТЛИЧНО! Модель работает великолепно!")
    elif accuracy > 80:
        print("\n👍 Хорошо! Модель работает неплохо")
    elif accuracy > 70:
        print("\n👌 Приемлемо, но можно улучшить")
    else:
        print("\n⚠️ Модель нуждается в доработке")


def test_batch_prediction():
    """Тестирование пакетного предсказания"""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ НА РЕАЛЬНЫХ ПРИМЕРАХ")
    print("=" * 60)

    # Примеры из жизни
    real_examples = [
        "Товар пришел быстро, качество отличное, продавцу спасибо!",
        "Ужасный продавец, товар не соответствует описанию, деньги не вернули",
        "Нормально, но могло быть и лучше",
        "Купил, порадовало качество, буду заказывать еще",
        "Полное разочарование, бракованный товар, не советую",
        "Доставка быстрая, упаковка хорошая, товар целый",
        "Кошмарный сервис, неделю ждал заказ",
        "Отличный магазин, цены приемлемые, всем советую",
        "Товар средненький, цена завышена",
        "Спасибо, все пришло в сохранности, качеством доволен",
        "Бракованный товар, хочу вернуть деньги",
        "Очень качественный продукт, спасибо большое!",
        "Не советую этот магазин, обманщики",
        "Цена качество супер, буду заказывать ещё",
        "Доставка долгая, но товар хороший",
    ]

    print("\nАнализ реальных отзывов:\n")
    for i, text in enumerate(real_examples, 1):
        result = subprocess.run([
            sys.executable, "classifier.py",
            "--mode", "predict",
            "--model_dir", "./my_big_model",
            "--text", text
        ], capture_output=True, text=True, encoding='utf-8')

        match = re.search(r'Предсказанный класс:\s*(\d+)', result.stdout)
        if match:
            pred = int(match.group(1))
            sentiment = "👍 ПОЛОЖИТЕЛЬНЫЙ" if pred == 1 else "👎 ОТРИЦАТЕЛЬНЫЙ"

            # Определяем уверенность (если есть в выводе)
            confidence_match = re.search(r'Уверенность:\s*([\d.]+%)', result.stdout)
            confidence = f" ({confidence_match.group(1)})" if confidence_match else ""

            print(f"{i}. {text[:70]}...")
            print(f"   Оценка: {sentiment}{confidence}")
            print()


def quick_test():
    """Быстрый тест для одного текста"""
    print("\n" + "=" * 60)
    print("БЫСТРЫЙ ТЕСТ")
    print("=" * 60)

    test_text = input("\nВведите текст для анализа (или 'q' для выхода): ")
    if test_text.lower() != 'q':
        result = subprocess.run([
            sys.executable, "classifier.py",
            "--mode", "predict",
            "--model_dir", "./my_big_model",
            "--text", test_text
        ], capture_output=True, text=True, encoding='utf-8')

        print("\nРезультат:")
        print(result.stdout)


if __name__ == "__main__":
    # Проверяем наличие модели
    if not os.path.exists("./my_big_model"):
        print("❌ Модель не найдена! Сначала обучите модель:")
        print(
            'python classifier.py --mode train --data train_big.csv --model_dir ./my_big_model --config config_big.json')
        sys.exit(1)

    # Запускаем тесты
    test_predictions()
    test_batch_prediction()

    # Спрашиваем, хочет ли пользователь ввести свой текст
    response = input("\nХотите протестировать свой текст? (д/н): ")
    if response.lower() in ['д', 'да', 'yes', 'y']:
        quick_test()

    print("\n" + "=" * 60)
    print("ГОТОВО! Модель протестирована")
    print("=" * 60)
    print("\nКоманды для использования модели:")
    print('  - Для одного текста:')
    print('    python classifier.py --mode predict --model_dir ./my_big_model --text "Ваш текст"')
    print('\n  - Для оценки на тестовых данных:')
    print('    python classifier.py --mode evaluate --data test_big.csv --model_dir ./my_big_model')