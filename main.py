import os
from datetime import date

import instructor
from instructor import Mode
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel


# ==========================================
# ЗАГРУЖАЕМ API-КЛЮЧ
# ==========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY не найден в файле .env")


# ==========================================
# PYDANTIC-МОДЕЛЬ
# ==========================================

class Invoice(BaseModel):
    number: str
    amount: float
    currency: str
    issue_date: date


# ==========================================
# ИСХОДНЫЙ ТЕКСТ НАКЛАДНОЙ
# ==========================================

raw_text = """
Накладная № KG-2026/157 от 3 октября 2026 года.
Покупатель: ОсОО "Альфа".
Общая сумма составляет сорок восемь тысяч пятьсот сомов.
Оплата производится в кыргызских сомах.
"""


# ==========================================
# ПРОМПТ
# ==========================================

prompt = f"""
Извлеки данные из текста накладной.

Нужно определить:
- number — номер накладной
- amount — сумму числом
- currency — код валюты
- issue_date — дату в формате YYYY-MM-DD

Пример:

Текст:
"Накладная № НК-2024/088 от 14.02.2025.
Сумма: 32 500 KGS."

Результат:
{{
    "number": "НК-2024/088",
    "amount": 32500,
    "currency": "KGS",
    "issue_date": "2025-02-14"
}}

Теперь обработай следующий текст:

{raw_text}
"""


# ==========================================
# 1. ОБЫЧНЫЙ СЫРОЙ ОТВЕТ
# ==========================================

groq_client = Groq(api_key=api_key)

raw_response = groq_client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

raw_answer = raw_response.choices[0].message.content


print("\n======================================")
print("СЫРОЙ ОТВЕТ МОДЕЛИ")
print("======================================\n")

print(raw_answer)


# ==========================================
# 2. INSTRUCTOR + PYDANTIC
# ==========================================

instructor_client = instructor.from_groq(
    Groq(api_key=api_key),
    mode=Mode.JSON
)

invoice = instructor_client.chat.completions.create(
    model="openai/gpt-oss-120b",
    response_model=Invoice,
    max_retries=2,
    messages=[
        {
            "role": "user",
            "content": f"""
Извлеки данные из следующей накладной.

Верни только данные накладной.

{raw_text}
"""
        }
    ]
)


# ==========================================
# ВЫВОД ТИПИЗИРОВАННОГО ОТВЕТА
# ==========================================

print("\n======================================")
print("ТИПИЗИРОВАННЫЙ ОТВЕТ")
print("======================================\n")

print(invoice)


# ==========================================
# ПРОВЕРКА ТИПОВ
# ==========================================

print("\n======================================")
print("ТИПЫ ДАННЫХ PYDANTIC")
print("======================================\n")

print("number:", invoice.number, type(invoice.number))
print("amount:", invoice.amount, type(invoice.amount))
print("currency:", invoice.currency, type(invoice.currency))
print("issue_date:", invoice.issue_date, type(invoice.issue_date))


# ==========================================
# JSON ИЗ PYDANTIC
# ==========================================

print("\n======================================")
print("JSON ИЗ PYDANTIC")
print("======================================\n")

print(invoice.model_dump_json(indent=2))