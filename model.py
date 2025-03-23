import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('housing.csv')

data.fillna(data.median(numeric_only=True), inplace=True)

data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

target = 'median_house_value'
features = data.drop(columns=target)
labels = data[target]
labels = np.log1p(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=64,
    callbacks=[early_stopping]
)

mse, mae = model.evaluate(X_test, y_test)
print(f"MSE: {mse}")
print(f"MAE: {mae}")

predictions = model.predict(X_test)

y_test = np.expm1(y_test)
predictions = np.expm1(predictions.flatten())

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"R2 score: {r2}")
print(f"y_MAE: {mae}")

print(f"Predicted Value: {predictions[0]}, Actual Value: {y_test.iloc[0]}")
