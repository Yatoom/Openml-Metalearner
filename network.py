import json
import numpy as np
import requests
import pandas as pd
from keras import Sequential
from keras.callbacks import History
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from better_converter import Converter

# Open files
params_json = open("flow_6794_100_params.json")
scores_json = open("flow_6794_100_scores.json")
quantities_json = open("openml_100_quantities.json")

# Load json
params_data = json.load(params_json)
scores_data = json.load(scores_json)
quantities_data = json.load(quantities_json)

# Handle parameters
description_data = requests.get("http://openml.org/api/v1/json/flow/6794").json()['flow']['parameter']
converter = Converter(params_data, description_data)
vectors = np.array(converter.get_vectors())

# Handle scores
scores = np.array(list(scores_data.values())).flatten()

# Handle quantities
quantities_data = dict([(i, quantities_data[i]) for i in params_data.keys()])
quantities_frame = pd.DataFrame(list(quantities_data.values())).dropna(axis=1)
quantities_frame = quantities_frame.apply(
    lambda x: StandardScaler().fit_transform(np.array(x).reshape(-1, 1)).flatten())
quantities = np.array(quantities_frame)

# Data
X = np.concatenate((vectors, quantities.repeat(100, axis=0)), axis=1)
y = scores

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# model

# --> 0.0017802061972600456
model = Sequential()
model.add(Dense(32, input_shape=(81,), activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['mse'])
model.summary()

history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.3, callbacks=[History()])

print("Best:", np.min(history.history['val_mean_squared_error']))
# summarize history for accuracy
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model accuracy')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()