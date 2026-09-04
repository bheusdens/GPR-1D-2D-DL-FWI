import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Example dummy data for demonstration
import numpy as np
# train_X_array = np.random.randn(32, input_length)
# train_X_scalar = np.random.randn(32, 1)
# train_y = np.random.randn(32, *output_shape)

# load & split inputs
inputs = np.load('signal_list_multi_2D_pep_All_V2.npy') 
print(inputs.shape)

input_heights = np.load('rx_heights_2D_Data_Gen_Multi_All_V2.npy') 
print(input_heights.shape)

outputs = np.load('perm_list_multi_2D_pep_All_V2_trim.npy') 
print(outputs.shape)

# Ensure all arrays have the same number of samples
n_samples = min(len(inputs), len(input_heights), len(outputs))
inputs = inputs[:n_samples]
input_heights = input_heights[:n_samples]
outputs = outputs[:n_samples]

# Generate a random permutation of indices
np.random.seed(42)
indices = np.random.permutation(n_samples)

# Calculate split index for 90% train, 10% test
split = int(n_samples * 0.9)
train_idx = indices[:split]
test_idx = indices[split:]

# Use the same indices to split all arrays
train_X = inputs[train_idx]
test_X = inputs[test_idx]
train_X_height = input_heights[train_idx]
test_X_height = input_heights[test_idx]
train_y = outputs[train_idx]
test_y = outputs[test_idx]

# Make sure heights are 2D (N, 1)
train_X_height = train_X_height.reshape(-1, 1)
test_X_height = test_X_height.reshape(-1, 1)

#set shapes
input_length = train_X[0].shape[0]
print(input_length)
output_shape = train_y[0].shape
print(output_shape)

# Two inputs: 1D array and scalar
array_input = keras.Input(shape=(input_length,), name='array_input')
scalar_input = keras.Input(shape=(1,), name='scalar_input')

# Conv1D on array input
x = layers.Reshape((input_length, 1))(array_input)
x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
x = layers.Flatten()(x)

# Concatenate scalar input BEFORE Dense layers
x = layers.Concatenate()([x, scalar_input])

# Dense layers
x = layers.Dense(2048, activation='relu')(x)
x = layers.Dense(output_shape[0] * output_shape[1], activation='relu')(x)
x = layers.Reshape(output_shape + (1,))(x)  # Add channel for Conv2D

# Conv2D layers
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
x = layers.Conv2D(1, (3, 3), padding='same', activation='linear')(x)
outputs = layers.Reshape(output_shape)(x)

### LOSSES

import tensorflow as tf
import tensorflow.image as tfim
from keras.saving import register_keras_serializable

def autocorr2d(x):
    # x: (batch, height, width, 1)
    x = tf.squeeze(x, -1)  # (batch, height, width)
    x_mean = tf.reduce_mean(x, axis=[1,2], keepdims=True)
    x = x - x_mean
    fft = tf.signal.fft2d(tf.cast(x, tf.complex64))
    ps = fft * tf.math.conj(fft)
    ac = tf.signal.ifft2d(ps)
    ac = tf.math.real(ac)
    # Normalize autocorrelation
    ac = ac / (tf.reduce_max(tf.abs(ac), axis=[1,2], keepdims=True) + 1e-8)
    return ac

def spatial_correlation_loss(y_true, y_pred):
    # Ensure channel dimension
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
        y_pred = tf.expand_dims(y_pred, -1)
    ac_true = autocorr2d(y_true)
    ac_pred = autocorr2d(y_pred)
    return tf.reduce_mean(tf.square(ac_true - ac_pred))

def gradient_difference_loss(y_true, y_pred):
    # Ensure channel dimension
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
        y_pred = tf.expand_dims(y_pred, -1)
    # Compute gradients
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    # L1 difference of gradients
    return tf.reduce_mean(tf.abs(dy_true - dy_pred) + tf.abs(dx_true - dx_pred))

def Sobel_Loss(y_actual, y_predicted):
    # Ensure float32 and add channel dimension
    y_actual = tf.cast(y_actual, tf.float32)
    y_predicted = tf.cast(y_predicted, tf.float32)
    y_actual = tf.expand_dims(y_actual, -1)
    y_predicted = tf.expand_dims(y_predicted, -1)
    # Compute Sobel edges
    sobel_actual = tf.image.sobel_edges(y_actual)
    sobel_predicted = tf.image.sobel_edges(y_predicted)
    # Calculate L1 difference between edge maps
    edge_loss = tf.reduce_mean(tf.abs(sobel_actual - sobel_predicted))
    return edge_loss

def DSSIM_tf(y_actual, y_predicted, min_val, max_val):
    y_actual = tf.cast(y_actual, tf.float32)
    y_predicted = tf.cast(y_predicted, tf.float32)
    y_actual = (y_actual - min_val) / (max_val - min_val)
    y_predicted = (y_predicted - min_val) / (max_val - min_val)
    y_actual = tf.expand_dims(y_actual, -1)
    y_predicted = tf.expand_dims(y_predicted, -1)
    s = tfim.ssim(y_actual, y_predicted, max_val=1.0, filter_size=11)
    ssim_loss = (1 - s) / 2
    return ssim_loss

def directional_autocorr2d(x, direction='horizontal', lag=1):
    # x: (batch, height, width, 1)
    x = tf.squeeze(x, -1)  # (batch, height, width)
    x_mean = tf.reduce_mean(x, axis=[1,2], keepdims=True)
    x = x - x_mean
    if direction == 'horizontal':
        # Correlation between pixels separated by lag along width
        valid = x[:, :, :-lag] * x[:, :, lag:]
    elif direction == 'vertical':
        # Correlation between pixels separated by lag along height
        valid = x[:, :-lag, :] * x[:, lag:, :]
    else:
        raise ValueError('direction must be "horizontal" or "vertical"')
    # Mean over valid positions and batch
    return tf.reduce_mean(valid)

def anisotropy_loss(y_true, y_pred, lag=1):
    # Ensure channel dimension
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
        y_pred = tf.expand_dims(y_pred, -1)
    # Horizontal autocorrelation
    ac_h_true = directional_autocorr2d(y_true, 'horizontal', lag)
    ac_h_pred = directional_autocorr2d(y_pred, 'horizontal', lag)
    # Vertical autocorrelation
    ac_v_true = directional_autocorr2d(y_true, 'vertical', lag)
    ac_v_pred = directional_autocorr2d(y_pred, 'vertical', lag)
    # Penalize difference in each direction
    loss_h = tf.square(ac_h_true - ac_h_pred)
    loss_v = tf.square(ac_v_true - ac_v_pred)
    # Optionally, penalize the difference in anisotropy ratio
    ratio_true = ac_h_true / (ac_v_true + 1e-8)
    ratio_pred = ac_h_pred / (ac_v_pred + 1e-8)
    loss_ratio = tf.square(ratio_true - ratio_pred)
    # Combine losses (you can weight them as needed)
    return loss_h + loss_v + loss_ratio

# Optionally, you can define separate losses for horizontal and vertical directions:
def horizontal_autocorr_loss(y_true, y_pred, lag=1):
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
        y_pred = tf.expand_dims(y_pred, -1)
    ac_h_true = directional_autocorr2d(y_true, 'horizontal', lag)
    ac_h_pred = directional_autocorr2d(y_pred, 'horizontal', lag)
    return tf.square(ac_h_true - ac_h_pred)

def vertical_autocorr_loss(y_true, y_pred, lag=1):
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
        y_pred = tf.expand_dims(y_pred, -1)
    ac_v_true = directional_autocorr2d(y_true, 'vertical', lag)
    ac_v_pred = directional_autocorr2d(y_pred, 'vertical', lag)
    return tf.square(ac_v_true - ac_v_pred)

def variogram_loss(y_true, y_pred, lags=5):
    """
    Differentiable variogram-based loss for 2D tensors using TensorFlow ops.
    Computes the mean squared difference between empirical variograms of y_true and y_pred.
    Args:
        y_true: (batch, height, width) or (batch, height, width, 1)
        y_pred: same shape as y_true
        lags: number of pixel lags to compute variogram for
    Returns:
        Scalar loss value
    """
    # Ensure channel dimension
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
        y_pred = tf.expand_dims(y_pred, -1)
    # Remove channel for calculation
    y_true = tf.squeeze(y_true, -1)
    y_pred = tf.squeeze(y_pred, -1)
    # Compute variogram for each lag
    def empirical_variogram(img, lags):
        vgs = []
        weights = 1.0 / np.arange(1, lags+1)  # Example: inverse lag weighting
        for idx, h in enumerate(range(1, lags+1)):
            vg_h = tf.reduce_mean(tf.square(img[:, :-h] - img[:, h:]))
            vg_v = tf.reduce_mean(tf.square(img[:-h, :] - img[h:, :]))
            vgs.append(weights[idx] * (vg_h + vg_v) / 2.0)
        return tf.stack(vgs)
    vg_true = tf.map_fn(lambda x: empirical_variogram(x, lags), y_true)
    vg_pred = tf.map_fn(lambda x: empirical_variogram(x, lags), y_pred)
    return tf.reduce_mean(tf.square(vg_true - vg_pred))

@register_keras_serializable()
def combined_weighted_loss(
    y_true, y_pred,
    w_spatial=0.0, w_gdl=0.0, w_sobel=0.0, w_dssim=0.0,
    w_mse=0.25, w_logmse=0.0, w_mae=0.0,
    w_vario=0.75, w_aniso=0.0, w_horiz=0.0, w_vert=0.0,
    vario_lags=5, aniso_lag=1,
    min_val=None, max_val=None
):
    loss = 0.0
    if w_spatial != 0:
        loss += w_spatial * spatial_correlation_loss(y_true, y_pred)
    if w_gdl != 0:
        loss += w_gdl * gradient_difference_loss(y_true, y_pred)
    if w_sobel != 0:
        loss += w_sobel * Sobel_Loss(y_true, y_pred)
    if w_dssim != 0 and min_val is not None and max_val is not None:
        loss += w_dssim * tf.reduce_mean(DSSIM_tf(y_true, y_pred, min_val, max_val))
    if w_mse != 0:
        loss += w_mse * tf.reduce_mean(tf.square(y_true - y_pred))
    if w_logmse != 0:
        loss += w_logmse * tf.reduce_mean(tf.square(tf.math.log1p(tf.abs(y_true - y_pred))))
    if w_mae != 0:
        loss += w_mae * tf.reduce_mean(tf.abs(y_true - y_pred))
    if w_vario != 0:
        loss += w_vario * variogram_loss(y_true, y_pred, lags=vario_lags)
    if w_aniso != 0:
        loss += w_aniso * anisotropy_loss(y_true, y_pred, lag=aniso_lag)
    if w_horiz != 0:
        loss += w_horiz * horizontal_autocorr_loss(y_true, y_pred, lag=aniso_lag)
    if w_vert != 0:
        loss += w_vert * vertical_autocorr_loss(y_true, y_pred, lag=aniso_lag)
    return loss

min_val = np.min(train_y)
max_val = np.max(train_y)

def my_loss(y_true, y_pred):
    return combined_weighted_loss(y_true, y_pred, w_dssim=0.0, min_val=min_val, max_val=max_val)


# Load the previously saved model
saved_model = keras.models.load_model(
    '2D_pepML_2in_MSE25Var5.keras',
    custom_objects={'combined_weighted_loss': combined_weighted_loss,
        'my_loss': my_loss},
    safe_mode=False
)

# Build your new model architecture as usual
model = keras.Model([array_input, scalar_input], outputs)

# Set the weights from the saved model
model.set_weights(saved_model.get_weights())

model.summary()

model.compile(optimizer='adam', loss=combined_weighted_loss)

# early stopping callback
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

import csv

# Save batch losses by epoch
epoch_batch_losses = []

class BatchLossLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch_losses = []
    def on_train_batch_end(self, batch, logs=None):
        self.current_epoch_losses.append(logs['loss'])
    def on_epoch_end(self, epoch, logs=None):
        epoch_batch_losses.append(self.current_epoch_losses)

# train model
history = model.fit(
    [train_X, train_X_height], train_y,
    epochs=50,
    callbacks=[callback, BatchLossLogger()],
    verbose=2
)

# Save to CSV (each row is an epoch, columns are batch losses)
with open('batch_losses_pepML_MSE25Var5.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for epoch_losses in epoch_batch_losses:
        writer.writerow(epoch_losses)

# Loss curve
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.legend()
plt.savefig('loss_curve_2D_pepML_2in_MSE25Var5.png')

model.save('2D_pepML_2in_MSE25Var5.keras')

# Evaluate on test set
test_loss = model.evaluate([test_X, test_X_height], test_y, verbose=0)
print("Test loss:", test_loss)

# Predict on test set
predictions = model.predict([test_X, test_X_height], verbose=0)
print("Predictions shape:", predictions.shape)

np.save('2D_pepML_Pred_2in_MSE25Var5.npy', predictions)

# Evaluate on train set
train_loss = model.evaluate([train_X, train_X_height], train_y, verbose=0)
print("Train loss:", train_loss)

# Predict on train set
predictions = model.predict([train_X, train_X_height], verbose=0)
print("Predictions shape:", predictions.shape)

np.save('2D_pepML_Pred_2in_MSE25Var5_train.npy', predictions)
