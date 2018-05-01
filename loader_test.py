import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.callbacks import History
from keras.layers import Dense, BatchNormalization
from sklearn.model_selection import train_test_split

from loader import Loader
from merger import Merger

params, scores = Loader.get_flow_data(6767, 100)
qualities = Loader.get_task_qualities()
description = Loader.get_description(6767)

merger = Merger(params, description, scores, qualities)
X, y = merger.merge(100)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# model

# --> 0.0017802061972600456
model = Sequential()
model.add(Dense(32, input_shape=(X.shape[1],), activation='relu'))
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

history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1, callbacks=[History()])

print("Best:", np.min(history.history['val_mean_squared_error']))
# summarize history for accuracy
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model accuracy')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
