import tensorflow as tf
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import numpy as np

data = pd.read_csv("housing.csv")

data = data.dropna()

data['ocean_proximity'], _ = pd.factorize(data['ocean_proximity'])

numeric_cols = data.select_dtypes(include=[np.number]).columns
data = data[(np.abs(zscore(data[numeric_cols])) < 3).all(axis=1)]

features = data.drop(['housing_median_age', 'total_bedrooms', 'population'], axis=1)
labels = data['median_house_value']

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

scaler = sklearn.preprocessing.MinMaxScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(train_features, train_labels, epochs=100, validation_data=(test_features, test_labels))

model.evaluate(test_features, test_labels)

predictions = model.predict(test_features)
print(predictions[0])