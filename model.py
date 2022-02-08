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

def build_encoder(shape):
    encoder_inputs = keras.Input(shape=shape)
    regularizer = keras.regularizers.l1_l2(0.01)
    x = layers.Conv2D(16, 3, activation="relu", strides=1, padding="same", 
                      kernel_regularizer=regularizer)(encoder_inputs)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same",
                      kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(48, 3, activation="relu", strides=1, padding="same", 
                      kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same", 
                      kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(72, 3, activation="relu", strides=2, padding="same",
                      kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    z = layers.Dense(128, activation="relu")(x)
    encoder = keras.Model(encoder_inputs, z, name="encoder")
    return encoder

def build_classifier(num_cluster):
    x = keras.Input(shape=(128,))
    y_hid = layers.Dense(64, activation="relu")(x)
    y_logits = layers.Dense(num_cluster, activation="linear")(y_hid)
    y = layers.Softmax()(y_logits)
    y_logits = layers.Softmax()(y_logits)
    return keras.Model(x, [y, y_logits], name="classifier")

def build_decoder(shape):
    latent_inputs = keras.Input(shape=(128,))
    regularizer = keras.regularizers.l1_l2(0.01)
    x = layers.Dense(shape[0] * shape[1] * 16, activation="relu",
                    kernel_regularizer=regularizer)(latent_inputs)
    x = layers.Reshape((shape[0]//4, shape[1]//4, 256))(x)
    x = layers.Conv2DTranspose(72, 3, activation="relu", strides=2,
                              kernel_regularizer=regularizer, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=1,
                              kernel_regularizer=regularizer, padding="same")(x)
    x = layers.Conv2DTranspose(48, 3, activation="relu", strides=1,
                              kernel_regularizer=regularizer, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=1,
                               kernel_regularizer=regularizer, padding="same")(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, 
                              kernel_regularizer=regularizer, padding="same")(x)
    output = layers.Conv2DTranspose(3, 3, 
                                    activation="sigmoid", 
                                    kernel_regularizer=regularizer, 
                                    padding="same")(x)
    decoder = keras.Model(latent_inputs, output)
    return decoder

class My_Model(keras.Model):
    def __init__(self, encoder=None, num_cluster=1, **kwargs):
        super(My_Model, self).__init__(**kwargs)
        self.encoder = encoder
        if encoder == None:
            self.encoder = build_encoder(SHAPE)
        self.decoder = build_decoder(SHAPE)
        self.classifier = build_classifier(num_cluster)
        self.num_cluster = num_cluster
        self.shape = SHAPE

    def call(self, x):
        return self.decoder(self.encoder(x))

    def elbo_loss(self, x, target):
        mse = tf.reduce_mean(keras.losses.MSE(target, x))
        mse *= self.shape[0] * self.shape[1]
        kld = tf.keras.losses.KLDivergence()(target, x)
        return mse + kld

    def train_step(self, inputs):
        x = inputs[0][0]
        target = inputs[0][1]
        # x: [batch * height * width * channel], noise image
        # target: [batch * height * width * channel], clean image
        with tf.GradientTape() as tape:
            x = self.decoder(self.encoder(x))
            elbo_loss = self.elbo_loss(x, target)
        grads = tape.gradient(elbo_loss, self.encoder.trainable_weights + self.decoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights + self.decoder.trainable_weights))

        with tf.GradientTape() as tape_y:
            z = self.encoder(x)
            y, y_logits = self.classifier(z)
            y_kl_loss = tf.reduce_mean(tf.reduce_sum(y_logits * (tf.math.log(y_logits + 1e-8) - tf.math.log(1.0/self.num_cluster)), axis=1))
        grads = tape_y.gradient(y_kl_loss, self.classifier.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.classifier.trainable_weights))
        return {
            "elbo_loss": elbo_loss,
            "y_kl_loss": y_kl_loss,
        }

class Model_P(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(Model_P, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        return self.decoder(self.encoder(x))

    def train_step(self, inputs):
        x, target = inputs[0][0], inputs[0][1]
        with tf.GradientTape() as tape:
            x = self.decoder(self.encoder(x))
            elbo_loss = tf.reduce_mean(keras.losses.MSE(target, x)) * SHAPE[0] * SHAPE[1] + tf.keras.losses.KLDivergence()(target, x)
        grads = tape.gradient(elbo_loss, self.decoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.decoder.trainable_weights))
        return {
            "elbo_loss": elbo_loss,
        }

class ContrastiveModel(keras.Model):
    def __init__(self):
        super().__init__()

        self.encoder = build_encoder(SHAPE)
        self.temperature = 0.1
        # Non-linear MLP as projection head
        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=(128,)),
                layers.Dense(128, activation="relu"),
                layers.Dense(128),
            ],
            name="projection_head",
        )

    def compile(self, contrastive_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer

        # self.contrastive_loss will be defined as a method
        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        (augmented_images_1, augmented_images_2) = data[0]

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        return {m.name: m.result() for m in self.metrics}
        