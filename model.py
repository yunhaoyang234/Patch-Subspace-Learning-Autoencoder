from skimage.util import random_noise
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
import cv2
import os
import glob
from keras.layers import Input, Dense, Lambda

regularizer = keras.regularizers.l1_l2(0.01)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class gumbel_softmax(keras.layers.Layer):
    def __init__(self, input_dim, tau):
        super(gumbel_softmax, self).__init__()
        w_init = tf.random_uniform_initializer(0.1,1)
        self.w = tf.Variable(
            initial_value=w_init(shape=[input_dim], dtype="float32"),
            trainable=False,
        )
        self.tau = tau

    def call(self, inputs):
        return (inputs - tf.math.log(-tf.math.log(self.w)))/self.tau

def build_encoder(latent_dim, shape, num_cluster):
    encoder_inputs = keras.Input(shape=shape)

    # x convolutional block
    x = layers.Conv2D(16, 3, activation="relu", strides=1, padding="same", 
                      kernel_regularizer=regularizer)(encoder_inputs)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same",
                      kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(48, 3, activation="relu", strides=1, padding="same", 
                      kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(72, 3, activation="relu", strides=2, padding="same",
                      kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(128, 3, activation="relu", strides=1, padding="same", 
                      kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)

    # y probability block
    y_hid = layers.Dense(128, activation="relu")(x)
    y_logits = layers.Dense(num_cluster, activation="linear")(y_hid)
    #y = gumbel_softmax(num_cluster, 1)(y_logits)
    y = layers.Dense(num_cluster, activation="linear")(y_hid)
    y = layers.Softmax()(y)
    y_logits = layers.Softmax()(y_logits)

    # z prior block
    z_prior_mean = layers.Dense(latent_dim)(y)
    z_prior_sig = layers.Dense(latent_dim, activation='softplus')(y)

    # Sampling
    h = layers.Dense(128, activation="relu")(layers.Dropout(rate=0.2)(x))
    z_mean = layers.Dense(latent_dim, name="z_mean")(h)
    z_sig = layers.Dense(latent_dim, activation='softplus', name="z_sig")(h)
    z = Sampling()([z_mean, z_sig])

    encoder = keras.Model(encoder_inputs, [z, z_mean, z_sig, y, y_logits, z_prior_mean, z_prior_sig], name="encoder")
    return encoder

def build_raw2rgb_encoder(latent_dim, shape, num_cluster):
    encoder_inputs = keras.Input(shape=shape)

    x = layers.UpSampling2D()(encoder_inputs)

    # x convolutional block
    x = layers.Conv2D(16, 3, activation="relu", strides=1, padding="same", 
                      kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(32, 3, activation="relu", strides=1, padding="same",
                      kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(48, 3, activation="relu", strides=2, padding="same", 
                      kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same", 
                      kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(72, 3, activation="relu", strides=1, padding="same",
                      kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(96, 3, activation="relu", strides=2, padding="same",
                      kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(128, 3, activation="relu", strides=1, padding="same", 
                      kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)

    # y probability block
    y_hid = layers.Dense(128, activation="relu")(x)
    y_logits = layers.Dense(num_cluster, activation="linear")(y_hid)
    #y = gumbel_softmax(num_cluster, 1)(y_logits)
    y = layers.Dense(num_cluster, activation="linear")(y_hid)
    y = layers.Softmax()(y)
    y_logits = layers.Softmax()(y_logits)

    # z prior block
    z_prior_mean = layers.Dense(latent_dim)(y)
    z_prior_sig = layers.Dense(latent_dim, activation='softplus')(y)

    # Sampling
    h = layers.Dense(128, activation="relu")(layers.Dropout(rate=0.2)(x))
    z_mean = layers.Dense(latent_dim, name="z_mean")(h)
    z_sig = layers.Dense(latent_dim, activation='softplus', name="z_sig")(h)
    z = Sampling()([z_mean, z_sig])

    encoder = keras.Model(encoder_inputs, [z, z_mean, z_sig, y, y_logits, z_prior_mean, z_prior_sig], name="encoder")
    return encoder

def build_decoder(latent_dim, shape, name):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(shape[0] * shape[1] * 16, activation="relu",
                    kernel_regularizer=regularizer)(latent_inputs)
    x = layers.Reshape((shape[0]//4, shape[1]//4, 256))(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", strides=1, 
                              kernel_regularizer=regularizer, padding="same")(x)
    x = layers.Conv2DTranspose(72, 3, activation="relu", strides=2,
                              kernel_regularizer=regularizer, padding="same")(x)
    x = layers.Conv2DTranspose(48, 3, activation="relu", strides=1,
                              kernel_regularizer=regularizer, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2,
                              kernel_regularizer=regularizer, padding="same")(x)
    output = layers.Conv2DTranspose(shape[2], 3, 
                                    activation="sigmoid", 
                                    kernel_regularizer=regularizer, 
                                    padding="same")(x)
    decoder = keras.Model(latent_inputs, output, name=name)
    return decoder

def build_raw2rgb_decoder(latent_dim, shape, name):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(shape[0] * shape[1] * 16, activation="relu",
                    kernel_regularizer=regularizer)(latent_inputs)
    x = layers.Reshape((shape[0]//4, shape[1]//4, 256))(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", strides=1, 
                              kernel_regularizer=regularizer, padding="same")(x)
    x = layers.Conv2DTranspose(96, 3, activation="relu", strides=2, 
                              kernel_regularizer=regularizer, padding="same")(x)
    x = layers.Conv2DTranspose(72, 3, activation="relu", strides=1,
                              kernel_regularizer=regularizer, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=1,
                              kernel_regularizer=regularizer, padding="same")(x)
    x = layers.Conv2DTranspose(48, 3, activation="relu", strides=2,
                              kernel_regularizer=regularizer, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=1,
                              kernel_regularizer=regularizer, padding="same")(x)
    output = layers.Conv2DTranspose(shape[2], 3, 
                                    activation="sigmoid", 
                                    kernel_regularizer=regularizer, 
                                    padding="same")(x)
    decoder = keras.Model(latent_inputs, output, name=name)
    return decoder


def kl_divergence_two_gauss(mean1,sig1,mean2,sig2):
    return tf.reduce_mean(tf.reduce_mean(tf.math.log(sig2) - tf.math.log(sig1) + ((tf.math.square(sig1) + tf.math.square(mean1-mean2)) / (2*tf.math.square(sig2))) - 0.5, axis=1))

class PSVAE_Encoder(keras.Model):
    def __init__(self, encoder, decoder, num_cluster, shape, **kwargs):
        super(PSVAE_Encoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.num_cluster = num_cluster
        self.shape = shape

    def train_step(self, data):
        test = data[0][1]
        data = data[0][0]

        with tf.GradientTape(persistent=True) as tape:
            z_x, z_mean_x, z_sig_x, y, y_logits, z_prior_mean, z_prior_sig = self.encoder(data)
            # reconstruct images
            reconstruction = self.decoder(z_x)
            # reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.MSE(test, reconstruction))
            reconstruction_loss *= self.shape[0] * self.shape[1]
            # KL loss
            x_kl_loss = kl_divergence_two_gauss(z_mean_x, z_sig_x, z_prior_mean, z_prior_sig)
            y_kl_loss = tf.reduce_mean(tf.reduce_sum(y_logits * (tf.math.log(y_logits + 1e-8) - tf.math.log(1.0/self.num_cluster)), axis=1))
            
            total_loss = reconstruction_loss + y_kl_loss + 0.01*x_kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "reconstruction_loss": reconstruction_loss,
            "x_kl_loss": x_kl_loss,
            "y_kl_loss": y_kl_loss,
        }

class PSVAE_Decoder(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(PSVAE_Decoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.trained_decoder = decoder

    def train_step(self, data):
        test = data[0][1]
        data = data[0][0]
        shape = (len(data[0]), len(data[0][0]))
        
        with tf.GradientTape() as tape:
            z, z_mean, z_sig, y, y_logits, z_prior_mean, z_prior_sig = self.encoder(data)
            # reconstruct images
            reconstruction = self.trained_decoder(z)
            # calculate loss
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.MSE(test, reconstruction))
            reconstruction_loss *= shape[0] * shape[1]
            
            total_loss = reconstruction_loss
        
        grads = tape.gradient(total_loss, self.trained_decoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trained_decoder.trainable_weights))
        return {
            "loss": reconstruction_loss,
        }
