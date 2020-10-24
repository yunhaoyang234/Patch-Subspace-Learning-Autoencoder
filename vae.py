import argparse
import pprint
import json

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=bool, default=True)
parser.add_argument('--block_size', type=int, default=18)
parser.add_argument('--num_block', type=int, default=18)
parser.add_argument('--overlap', type=int, default=4)
parser.add_argument('--batch', type=int, default=10000)
parser.add_argument('--latent_dim', type=int, default=60)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--num_cluster', type=int, default=4)

from utils import *
from evaluation import *

def build_encoder(latent_dim, shape, num_cluster):
  encoder_inputs = keras.Input(shape=shape)
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
  x = layers.Dense(96, activation="relu", 
                   kernel_regularizer=regularizer)(x)
  x = layers.Dense(latent_dim, activation="relu", 
                   kernel_regularizer=regularizer)(x)

  initializer = tf.keras.initializers.RandomNormal(1, 0.1)
  y = layers.Dense(num_cluster, activation="relu",
                   kernel_initializer=initializer, 
                   trainable = False)(x)
  y = layers.Softmax()(y)

  # Sampling
  z_mean = layers.Dense(latent_dim, name="z_mean")(x)
  z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
  z = Sampling()([z_mean, z_log_var])
  encoder = keras.Model(encoder_inputs, [x, y, z_mean, z_log_var, z], name="encoder")
  return encoder

def build_decoder(latent_dim, shape, name):
  latent_inputs = keras.Input(shape=(latent_dim,))
  x = layers.Dense(shape[0] * shape[1] * 4, activation="relu",
                   kernel_regularizer=regularizer)(latent_inputs)
  x = layers.Reshape((shape[0]//6, shape[1]//6, 144))(x)
  x = layers.Conv2DTranspose(128, 3, activation="relu", strides=1, 
                             kernel_regularizer=regularizer, padding="same")(x)
  x = layers.Conv2DTranspose(72, 3, activation="relu", strides=2,
                             kernel_regularizer=regularizer, padding="same")(x)
  x = layers.Conv2DTranspose(48, 3, activation="relu", strides=1, 
                             kernel_regularizer=regularizer, padding="same")(x)
  x = layers.Conv2DTranspose(32, 3, activation="relu", strides=3, 
                             kernel_regularizer=regularizer, padding="same")(x)
  x = layers.Conv2DTranspose(16, 3, activation="relu", strides=1, 
                             kernel_regularizer=regularizer, padding="same")(x)
  decoder_outputs = layers.Conv2DTranspose(shape[2], 3, 
                                           activation="sigmoid", 
                                           kernel_regularizer=regularizer, 
                                           padding="same")(x)
  decoder = keras.Model(latent_inputs, decoder_outputs, name=name)
  return decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, num_cluster, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.num_cluster = num_cluster

    def train_step(self, data):
        test = data[0][1]
        data = data[0][0]
        shape = (len(data[0]), len(data[0][0]))

        with tf.GradientTape() as tape:
            x, y, z_mean, z_log_var, z = self.encoder(data)
            
            # reconstruct images
            reconstruction = self.decoder(z)
            # calculate loss
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5

            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.MSE(test, reconstruction))
            reconstruction_loss *= shape[0] * shape[1]

            #soft_cut_loss = soft_n_cut_loss(z_mean, z_log_var, y, self.num_cluster)
            
            total_loss = reconstruction_loss + kl_loss# + soft_cut_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            #"soft_n_cut_loss": soft_cut_loss,
        }

def main(args):
    # Enable GPU
    if args.gpu:
        %tensorflow_version 2.x
        import tensorflow as tf
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
          raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    BLOCK_SIZE = args.block_size #28
    NUM_BLOCK = args.num_block #22
    BLOCK_PER_IMAGE = NUM_BLOCK * NUM_BLOCK
    OVERLAP = args.num_block #8
    WIDTH = len(images[0][0])
    HEIGHT = len(images[0])
    NUM_CLUSTER = args.num_cluster
    SHAPE = (BLOCK_SIZE, BLOCK_SIZE, 3)
    BATCH_SIZE = args.batch
    LATENT_DIM = args.latent_dim
    EPOCH = args.epoch

    '''
    Load Images
    '''
    validation_images = load_images(cwd + 'validation-r08-s-0000-of-0040.tfrecords')
    images = load_images(cwd + 'train-r08-s-0000-of-0120.tfrecords')
    blur_imgs = gen_noise(images)
    val_blur_imgs = gen_noise(validation_images)
    clear_images, blur_images = gen_large_train_set(images, blur_imgs, BLOCK_SIZE, BATCH_SIZE)


    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )

    encoder = build_encoder(LATENT_DIM, SHAPE, NUM_CLUSTER)
    decoder = build_decoder(LATENT_DIM, SHAPE,"decoder")
    model = VAE(encoder, decoder, NUM_CLUSTER)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
    model.fit((blur_images,clear_images), epochs=EPOCH, batch_size=128)

    test_images_clear, test_images_blur = gen_large_train_set(validation_images, val_blur_imgs, BLOCK_SIZE, BATCH_SIZE)

    batch = 10000
    x, y, z_mean, z_log_var, z = encoder.predict(test_images_blur[:batch])
    for i in range(batch, len(test_images_blur), batch):
      y = np.concatenate([y, encoder.predict(test_images_blur[i: i+batch])[1]], axis=0)
      z = np.concatenate([z, encoder.predict(test_images_blur[i: i+batch])[4]], axis=0)

    decoded_imgs = decoder.predict(z[:batch])
    for i in range(batch, len(z), batch):
      decoded_imgs = np.concatenate([decoded_imgs, decoder.predict(z[i:i+batch])], axis=0)

    """## Metrics"""
    recons_images = reconstruct_image(z, y_logits, [decoder]*NUM_CLUSTER, 
                                      BATCH_SIZE, BLOCK_PER_IMAGE, WIDTH, 
                                      HEIGHT, BLOCK_SIZE, OVERLAP)
    recons_images = tf.cast((recons_images*255), dtype=tf.uint8)
    comp_images = recons_images
    test_images = validation_images

    quality_evaluation(recons_images, test_images, comp_images, metric='PSNR')
    quality_evaluation(recons_images, test_images, comp_images, metric='SSIM')
    quality_evaluation(recons_images, test_images, comp_images, metric='UQI')
