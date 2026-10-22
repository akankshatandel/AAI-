# PRACTICAL 4: Neural-network recommendation system
# Install: python -m pip install tensorflow numpy scikit-learn

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Small sample: user ID, item ID, rating (1 to 5)
data = np.array([
    [1, 101, 5], [1, 102, 3], [2, 101, 4],
    [2, 103, 2], [3, 102, 4], [3, 104, 1],
    [4, 103, 5], [4, 104, 4], [2, 102, 3],
    [3, 101, 5]
], dtype=float)

users_raw = data[:, 0].astype(int)
items_raw = data[:, 1].astype(int)
ratings = data[:, 2].astype("float32") / 5.0

# Convert IDs to consecutive numbers for embedding layers.
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
users = user_encoder.fit_transform(users_raw)
items = item_encoder.fit_transform(items_raw)
num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)

train_u, test_u, train_i, test_i, train_r, test_r = train_test_split(
    users, items, ratings, test_size=0.2, random_state=42
)

user_input = Input(shape=(1,), name="user")
item_input = Input(shape=(1,), name="item")
user_vec = Flatten()(Embedding(num_users, 8)(user_input))
item_vec = Flatten()(Embedding(num_items, 8)(item_input))
joined = Concatenate()([user_vec, item_vec])
hidden = Dense(16, activation="relu")(joined)
output = Dense(1, activation="sigmoid")(hidden)

model = Model(inputs=[user_input, item_input], outputs=output)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit([train_u, train_i], train_r, epochs=30, batch_size=4, verbose=0)

predictions = model.predict([test_u, test_i], verbose=0).flatten() * 5
actual = test_r * 5
for i in range(len(predictions)):
    print("Actual rating:", round(float(actual[i]), 2),
          "| Predicted rating:", round(float(predictions[i]), 2))
