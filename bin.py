# Импортируем необходимые библиотеки
import requests
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Насраиваем параметры API Binance, чтобы получить данные по торговой паре BTCUSDT за последние 1000 часов (1 час
# - interval). Затем выполняем GET-запрос к API, и результаты сохраняем в переменной data

# Настройки API Binance
url = "https://api.binance.com/api/v3/klines"
symbol = "BTCUSDT"
interval = "1h"
limit = "1000"

# Получение данных
params = {"symbol": symbol, "interval": interval, "limit": limit}
response = requests.get(url, params=params)
data = response.json()

# Создается DataFrame из полученных данных с указанием названий столбцов
df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
df = df.drop(["timestamp", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"], axis=1)
df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric)

# Сохраняем данные в файл CSV
df.to_csv("binance_data.csv", index=False)

# Предварительная обработка данных
df = pd.read_csv("binance_data.csv") # Считывается CSV файл в DataFrame
df = df.dropna() # Удаляются строки, содержащие пропущенные значения
df["close"] = df["close"].pct_change() # Вычисляется процентное изменение значения столбца "close"
df = df.dropna()

# Разделение исходного набора данных на обучающую и тестовую выборки с помощью разделения индексов данных в
# соотношении 80/20
split = int(0.8 * len(df))
train_df = df[:split]
test_df = df[split:]

# Масштабирование данных для нормализации значений на основе среднего значения и стандартного отклонения обучающего
# набора данных
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# Преобразование в 3D-массив, где каждый элемент массива представляет собой последовательность 60 предыдущих значений
# цены закрытия актива для обучения модели. X_train и X_test - это матрицы функций, используемые для обучения и
# проверки модели, соответственно, а y_train и y_test - это векторы целевых переменных, которые модель пытается
# предсказать на основе предыдущих 60 значений.
train_data = train_df.values
test_data = test_df.values
X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = [], []
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i])
    y_test.append(test_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

# Создание LSTM-модели
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss="mae", optimizer="adam")

# Обучение модели
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1, shuffle=False)

# Визуализация обучения (графики функции потерь на обучающих и тестовых данных)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="test")
plt.legend()
plt.show()

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

y_pred = y_pred * train_std[0] + train_mean[0]
y_test = y_test * train_std[0] + train_mean[0]
plt.figure(figsize=(16,8))
plt.plot(y_test, label="actual")
plt.plot(y_pred, label="predicted")
plt.legend()
plt.show() # график фактических значений и предсказаний на тестовой выборке

# Оценка качества модели на тестовой выборке
mse = mean_squared_error(y_test, y_pred) # Функция mean_squared_error используется для вычисления среднеквадратичной
# ошибки между y_test (фактическими значениями) и y_pred (предсказанными значениями)
mae = mean_absolute_error(y_test, y_pred) # Функция mean_absolute_error используется для вычисления средней
# абсолютной ошибки между y_test и y_pred
print("MSE на тестовой выборке (обратное масштабирование):", mse)
print("MAE на тестовой выборке (обратное масштабирование):", mae)

# Берем последние 60 значений из тестовой выборки, затем преобразуем их в 3D-массив и подадим на вход
# обученной модели LSTM, которая предсказывает следующее значение. Далее предсказанное значение размасштабируется
# обратно и выводится на график вместе с остальными значениями тестовой выборки. На графике также указывается
# предсказанное значение на час вперед в виде маркера "o"

# Получение последних 60 значений из тестовой выборки
last_60 = test_data[-60:]

# Преобразование последних 60 значений в 3D-массив
last_60 = np.array(last_60)
last_60 = np.reshape(last_60, (1, last_60.shape[0], last_60.shape[1]))

# Предсказание следующего значения
next_value = model.predict(last_60)

# Размасштабирование предсказанного значения
next_value = next_value * train_std[0] + train_mean[0]

# Вывод графика с предсказанием на час вперед
plt.figure(figsize=(16,8))
plt.plot(y_test, label="actual")
plt.plot(y_pred, label="predicted")
plt.plot(len(y_test) + 1, next_value, marker="o", markersize=10, label="next value")
plt.legend()
plt.show()

# Сохранение модели
model.save("model.h5")