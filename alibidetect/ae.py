import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, \
    Dense, Layer, Reshape, InputLayer, Flatten
from tqdm import tqdm



print(f"GPU status: {1 if tf.test.is_gpu_available() else 0}")

train_data_dir = "/home/vishesh/Desktop/datasets/ships-data/X_true_train_240_320"
test_data_path = "/home/vishesh/Desktop/datasets/ships-data/X_true_val_240_320/1.npy"


X_train = None



for i in range(1,10):
    input_data_path = os.path.join(train_data_dir, f"{i}.npy")
    
    if i == 1:
        X_train = np.load(input_data_path)
        continue
        
        
    new_input_data = np.load(input_data_path)
    X_train = np.vstack((X_train, new_input_data))


X_test = np.load(test_data_path)

print("Data has been loaded")

X_train_noise = X_train + np.random.normal(0,0.1,size=X_train.shape)


encoding_dim = 128
input_img = tf.keras.layers.Input(shape=(240, 320, 3))
x = tf.keras.layers.Conv2D(32, 3, strides=1, padding="same")(input_img)
x1 = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU())(x)
x = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same")(x1)
x2 = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU())(x)
x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU())(x2)
x = tf.keras.layers.Conv2D(256, 5, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU())(x)
x = tf.keras.layers.Conv2D(512, 5, strides=(3, 2), padding="same", activation=tf.keras.layers.LeakyReLU())(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(encoding_dim, activation=tf.keras.layers.LeakyReLU())(x)
x = tf.keras.layers.Dense(5*10*512)(x)
x = tf.keras.layers.Reshape (target_shape=(5, 10, 512))(x)
x = tf.keras.layers.Conv2DTranspose(512, 5, padding="same", activation=tf.keras.layers.LeakyReLU())(x)
x = tf.keras.layers.Conv2DTranspose (256, 5, strides=(3, 2), padding="same", activation=tf.keras.layers.LeakyReLU())(x)
x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU())(x)
x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU())(x)
x = tf.keras.layers.Add()([x, x2])
x = tf.keras.layers.Conv2DTranspose(64, 3, strides=1, padding="same")(x)
x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU())(x)
x = tf.keras.layers.Add()([x, x1])
x = tf.keras.layers.Conv2DTranspose(32, 3, strides=1, padding="same")(x)
x = tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding="same", activation="sigmoid")(x)
ae = tf.keras.models.Model(inputs=input_img, outputs=x)

opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
ae.compile(optimizer=opt, loss='mse')
ae.fit(X_train_noise, X_train, validation_split=0.1, epochs=20, batch_size=4)
ae.save('my_model.h5')