import requests
from bs4 import BeautifulSoup
import csv
import time
import re
from datetime import datetime

def get_page_content(url):
      # Подбирал user-agent несколько часов, нашел рабочий вариант
    # Без правильного user-agent сервер возвращал 403 ошибку
   headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    try:
        # Использую сессию для сохранения cookies между запросами
        session = requests.Session()
        session.headers.update(headers)

        print(f"[ДЕНЬ 1] Отправка запроса к {url}")
        response = session.get(url, timeout=10)
        response.raise_for_status()  # Проверка на HTTP ошибки

        # Проверка на капчу - важный момент, потратил 2 часа на отладку
        if any(text in response.text for text in ['captcha', 'CAPTCHA', 'Доступ ограничен']):
            print("[ОШИБКА] Обнаружена защита от роботов, нужен более сложный обход")
            return None

        print(f"[УСПЕХ] Страница загружена, размер: {len(response.text)} символов")
        return response.text
        
    except requests.RequestException as e:
        print(f"[ОШИБКА] Не удалось загрузить страницу: {e}")
        return None
      
def parse_review_alternative(review_block):
    """
    Извлечение данных из HTML-блока отзыва
    
    Аргументы:
        review_block (BeautifulSoup): HTML элемент, содержащий отзыв
    
    Возвращает:
        dict: Словарь с полями 'title', 'content', 'rating' или None при ошибке
    
    История изменений:
        v1 (11:00) - только заголовок
        v2 (13:30) - добавил текст отзыва
        v3 (16:45) - добавил рейтинг через регулярные выражения
    """
    try:
        review_data = {}

        # --- Извлечение заголовка ---
        # Проблема: сайт использует разные классы для заголовков на разных страницах
        # Решение: перебираю несколько популярных вариантов
        title_selectors = [
            'a[class*="title"]',   # Наиболее частый вариант
            'h3',                   # Запасной вариант
            '.review-title',        # Альтернативный класс
            'a.title',              # Еще один вариант
            'div.title'             # Последняя надежда
        ]

        for selector in title_selectors:
            title_elem = review_block.select_one(selector)
            if title_elem and title_elem.text.strip():
                review_data['title'] = title_elem.text.strip()
                break
        else:
            review_data['title'] = ""  # Если ничего не нашли

        # --- Извлечение текста отзыва ---
        # Самая важная часть для анализа тональности
        content_selectors = [
            'div[class*="content"]',   # Основной контейнер
            'div[class*="text"]',      # Альтернативный вариант
            'div[class*="body"]',      # Встречается редко
            '.review-body',            # Специфичный класс
            '.review-content',         # Еще один вариант
            'p'                        # Самый общий случай
        ]

        for selector in content_selectors:
            content_elem = review_block.select_one(selector)
            if content_elem and content_elem.text.strip():
                review_data['content'] = content_elem.text.strip()
                break
        else:
            review_data['content'] = ""

        # --- Извлечение рейтинга ---
        # Здесь пришлось использовать regex, так как рейтинг часто встроен в CSS классы
        rating_selectors = [
            'div[class*="rating"]',
            'span[class*="rating"]',
            '.rating',
            'div[class*="star"]',
            'meta[itemprop="ratingValue"]'
        ]

        review_data['rating'] = None
        for selector in rating_selectors:
            rating_elem = review_block.select_one(selector)
            if rating_elem:
                # Пробуем извлечь число из текста
                rating_text = rating_elem.text.strip()
                rating_match = re.search(r'(\d+[,.]?\d*)', rating_text)
                if rating_match:
                    try:
                        review_data['rating'] = float(rating_match.group(1).replace(',', '.'))
                        print(f"[DEBUG] Найден рейтинг: {review_data['rating']}")
                        break
                    except ValueError:
                        pass

                # Если не нашли в тексте, ищем в CSS классах
                class_attr = rating_elem.get('class', [])
                for cls in class_attr:
                    if re.search(r'rating[-_]?(\d+)', cls):
                        match = re.search(r'rating[-_]?(\d+)', cls)
                        try:
                            review_data['rating'] = float(match.group(1))
                            print(f"[DEBUG] Найден рейтинг в классе: {review_data['rating']}")
                            break
                        except ValueError:
                            pass

        # Логируем успешный парсинг
        if review_data['title'] or review_data['content']:
            print(f"[ПАРСИНГ] Заголовок: {review_data['title'][:50]}...")

        return review_data

    except Exception as e:
        print(f"[ОШИБКА] Не удалось распарсить отзыв: {e}")
        return None

with open("debug_page.html", "r", encoding="utf-8") as f:
     test_html = f.read()
     test_soup = BeautifulSoup(test_html, 'html.parser')
     test_blocks = test_soup.select('div[class*="review"]')
     print(f"Найдено тестовых блоков: {len(test_blocks)}")
     for block in test_blocks[:3]:
         result = parse_review_alternative(block)
         if result:
             print(f"Результат: {result}")
