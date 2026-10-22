# PRACTICAL 7: LSTM forecasting using airline passenger data
# Install: python -m pip install tensorflow pandas matplotlib scikit-learn
# Internet is required to download the CSV dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
print("Loading dataset...")
df = pd.read_csv(url)
values = df["Passengers"].values.astype("float32").reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

def create_sequences(series, time_step=12):
    X, y = [], []
    for i in range(len(series) - time_step):
        X.append(series[i:i + time_step])
        y.append(series[i + time_step])
    return np.array(X), np.array(y)

time_step = 12
X, y = create_sequences(scaled, time_step)

model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(16),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=10, batch_size=16, verbose=1)

predicted_scaled = model.predict(X, verbose=0)
predictions = scaler.inverse_transform(predicted_scaled).flatten()
actual = values[time_step:].flatten()

plt.figure(figsize=(10, 5))
plt.plot(actual, label="Actual passengers")
plt.plot(predictions, label="Predicted passengers")
plt.title("Airline Passenger Forecast")
plt.xlabel("Month index")
plt.ylabel("Passengers")
plt.legend()
plt.tight_layout()
plt.show()
