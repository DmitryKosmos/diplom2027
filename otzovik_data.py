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
      
 if __name__ == "__main__":
     test_url = "https://otzovik.com/reviews/moskovskiy_universitet_imeni_s_yu_vitte_muiv/"
     content = get_page_content(test_url)
     if content:
         print("Страница успешно загружена!")
         # Сохраняем для анализа структуры
         with open("debug_page.html", "w", encoding="utf-8") as f:
             f.write(content)
