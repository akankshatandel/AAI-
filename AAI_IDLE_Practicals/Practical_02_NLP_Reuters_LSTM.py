# PRACTICAL 2: NLP text classification using an LSTM and Reuters dataset
# Install: python -m pip install tensorflow matplotlib

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

vocab_size = 10000
max_length = 200

print("Loading Reuters dataset...")
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

num_classes = max(max(y_train), max(y_test)) + 1
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = models.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=64),
    layers.LSTM(32),
    layers.Dense(32, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    x_train, y_train, epochs=3, batch_size=64,
    validation_split=0.2, verbose=1
)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", round(accuracy, 4))

plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Reuters LSTM Classification")
plt.legend()
plt.show()
