import tensorflow as tf
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

data = pd.read_csv("housing.csv")

data.pop('ocean_proximity')
data = data.dropna()

features = data
labels = data['median_house_value']

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

scaler = sklearn.preprocessing.MinMaxScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, 'linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(train_features, train_labels, epochs=100, validation_data=(test_features, test_labels))

model.evaluate(test_features, test_labels)

predictions = model.predict(test_features)
print(predictions[0])