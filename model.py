import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScale

data = example_data #as np.array; shape (n,) where n is the length of the profile

# Prepare the data
x = np.arange(len(data)).reshape(-1, 1)
y = data.reshape(-1, 1)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalize the data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_train_normalized = scaler_x.fit_transform(x_train)
x_val_normalized = scaler_x.transform(x_val)
y_train_normalized = scaler_y.fit_transform(y_train)
y_val_normalized = scaler_y.transform(y_val)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(x_train_normalized, y_train_normalized, epochs=500, validation_data=(x_val_normalized, y_val_normalized), callbacks=[early_stopping], verbose=0)

# Predict the profile using the trained model
y_pred_train_normalized = model.predict(x_train_normalized).flatten()
y_pred_val_normalized = model.predict(x_val_normalized).flatten()

# Inverse transform the predictions
y_pred_train = scaler_y.inverse_transform(y_pred_train_normalized.reshape(-1, 1)).flatten()
y_pred_val = scaler_y.inverse_transform(y_pred_val_normalized.reshape(-1, 1)).flatten()

# Combine the training and validation predictions for full profile plotting
y_pred = np.zeros_like(y.flatten())
y_pred[x_train.flatten()] = y_pred_train
y_pred[x_val.flatten()] = y_pred_val

# Plot the original, smoothed signal, and model predictions
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Original Profile')
plt.plot(x, y_pred, label='Model Prediction', linestyle='--')

plt.xlabel('Position')
plt.ylabel('Intensity')
plt.title('Model Fit to Signal Profile')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(6, 3))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
