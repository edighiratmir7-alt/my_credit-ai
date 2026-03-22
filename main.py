import telebot
import pickle
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. НАСТРОЙКИ (Берем токен из секретов сервера) ---
TOKEN = os.environ.get("TELEGRAM_TOKEN")
MODEL_PATH = 'credit_model.pkl'
bot = telebot.TeleBot(TOKEN)

# --- 2. МОЗГИ БОТА (Логистическая регрессия) ---
def get_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    
    # Обучающие данные (Доход, Возраст, Просрочки) -> (0 - Одобрить, 1 - Отказать)
    X = np.array([
        [10000, 35, 0], [5000, 25, 1], [1000, 19, 5], 
        [50000, 40, 0], [150000, 20, 10], [3000, 17, 0], 
        [8000, 30, 50], [1500000, 16, 99], [200000, 50, 0]
    ])
    y = np.array([0, 0, 1, 0, 1, 1, 1, 1, 0])
    model = LogisticRegression()
    model.fit(X, y)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    return model

model = get_model()

# --- 3. ОБРАБОТКА КОМАНД ---
@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "🤖 **ИИ Кредитный Инспектор онлайн**\n\nВведите через пробел:\nДоход | Возраст | Просрочки")

@bot.message_handler(func=lambda message: True)
def analyze(message):
    try:
        # Убираем лишние пробелы и запятые
        text = message.text.replace(',', '.').strip()
        parts = [float(p) for p in text.split() if p]
        
        if len(parts) != 3:
            bot.reply_to(message, "⚠️ Нужно 3 числа! Пример: 5000 25 0")
            return

        # Делаем предсказание
        df = pd.DataFrame([parts], columns=['income', 'age', 'delays'])
        prob = model.predict_proba(df)[0][1]
        score = int((1 - prob) * 1000)

        # Логика решения
        if parts[1] < 18:
            status, score = "❌ ОТКАЗ (Мало лет)", min(score, 200)
        elif parts[2] > 10:
            status, score = "❌ ОТКАЗ (Плохая история)", min(score, 300)
        else:
            status = "✅ ОДОБРЕНО" if prob < 0.5 else "❌ ОТКАЗ"

        res = (f"📊 **Результат анализа:**\n"
               f"🏆 Скоринг-балл: `{score}`\n"
               f"📉 Вероятность риска: `{prob:.1%}`\n"
               f"📢 Решение: **{status}**")
        
        bot.reply_to(message, res, parse_mode="Markdown")

    except Exception:
        bot.reply_to(message, "⚠️ Ошибка! Вводите только цифры через пробел.")

# --- 4. ЗАПУСК ---
if __name__ == "__main__":
    bot.infinity_polling()
