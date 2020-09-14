# -*- coding: utf-8 -*-

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
import cv2
from keras.layers import Input, Dense, Lambda
from typing import List
import numpy as np
import scipy as sp
from sklearn import cluster
from sklearn import decomposition
from skimage.metrics import structural_similarity as ssim
import sewar
import random
import os
import glob

img_shape = (256, 256) #(448, 448)
block_size = 18
num_block = 18 #22
block_per_image = num_block * num_block #484
overlap = 4 # 8
num_cluster = 2
shape = (block_size, block_size, 3)
batch=10000

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)

def load_images(file_path):
    raw_image_dataset = tf.data.TFRecordDataset(file_path)

    # Create a dictionary describing the features.
    image_feature_description = {
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([1], tf.int64)
    }

    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, image_feature_description)
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
  
    images = []
    for image_features in parsed_image_dataset:
        image_raw = image_features['data'].numpy()
        shape = image_features['shape'].numpy()
        img = tf.io.decode_raw(image_raw, tf.uint8)
        img = tf.reshape(img, shape).numpy()
        images.append(img)
    return np.array(images)

def read_images(img_dir):
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path) 
    data = []
    for f1 in files: 
        img = cv2.imread(f1) 
        data.append(img)
    return np.array(data)

def imshow(img):
    cv2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def gen_blur(images):
    topleft = 64
    size = 64
    blur_imgs = []
    images_copy = np.copy(images)
    for img in images_copy:
        b_img = img[topleft:topleft+size, topleft:topleft+size]
        b_img = cv2.blur(b_img,(100,100))
        for i in range(size):
            for j in range(size):
                img[i+topleft][j+topleft] = b_img[i][j]
        blur_imgs.append(img)
    return blur_imgs

def noisy(image, prob=0.05):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def gen_noise(images, prob=0.05):
  noise = []
  for img in images:
    noise.append(noisy(img, prob))
  return noise


def divide_img(img, block_size=18, num_block=18, overlap=4):
    height = len(img)
    width = len(img[0])
    if not (block_size*num_block - (num_block - 1)*overlap == height):
        print('Block size mismatch', 
            block_size*num_block - (num_block - 1)*overlap, height)
        return None
    size = block_size - overlap
    blocks = np.array([img[i:i+block_size, j:j+block_size] 
                     for j in range(0,width - overlap,size) 
                     for i in range(0,height - overlap,size)])
    return blocks

def merge_img(blocks, width=256, height=256, block_size=18, overlap=4):
    num_block_per_row = (width - overlap)//(block_size - overlap)
    num_block_per_col = (height - overlap)//(block_size - overlap)
    def get_row_block(row):
        row_block = blocks[row]
        for j in range(1, num_block_per_row):
            cur_row_block = row_block[:, :len(row_block[0]) - overlap]
            block1 = blocks[row+j*num_block_per_col]
            cur_block = block1[:, overlap:]
            lapping = row_block[:, len(row_block[0]) - overlap:]
            lapping1 = block1[:, :overlap]
            for k in range(0, overlap):
                lapping[:, k] *= 1 - (k+1)/(overlap+1)
                lapping1[:, k] *= (k+1)/(overlap+1)
            lap = lapping + lapping1
            row_block = np.concatenate([cur_row_block, lap, cur_block], axis=1)
        return row_block

    img = get_row_block(0)
    for i in range(1, num_block_per_col):
        cur_block = img[:len(img)-overlap]
        cur_row = get_row_block(i)
        lapping = img[len(img)-overlap:]
        lapping1 = cur_row[:overlap]
        cur_row = cur_row[overlap:]
        for k in range(0, overlap):
            lapping[k,:] *= 1 - (k+1)/(overlap+1)
            lapping1[k,:] *= (k+1)/(overlap+1)
    
        lap = lapping + lapping1
        img = np.concatenate([cur_block, lap, cur_row], axis=0)
    return img

def gen_train_set(clear_imgs, blur_imgs, block_size):
    blur_images = []
    clear_images = []

    for i in range(len(clear_imgs)):
        blocks = divide_img(clear_imgs[i], block_size, num_block, overlap=overlap)
        for b in blocks:
            clear_images.append(b)
        blur_blocks = divide_img(blur_imgs[i], block_size, num_block, overlap=overlap)
        for bb in blur_blocks:
            blur_images.append(bb)
    return np.array(clear_images)/255, np.array(blur_images)/255

def gen_large_train_set(clear_imgs, blur_imgs):
    batch_size = 1000
    c, b = gen_train_set(clear_imgs[:batch_size], blur_imgs[:batch_size], block_size)
    blur_images = tf.convert_to_tensor(b, np.float32)
    clear_images = tf.convert_to_tensor(c, np.float32)
    
    for i in range(batch_size, len(blur_imgs), batch_size):
        c, b = gen_train_set(clear_imgs[i:i+batch_size], blur_imgs[i:i+batch_size], block_size)
        blur_images = tf.concat([blur_images, tf.convert_to_tensor(b, np.float32)], axis=0)
        clear_images = tf.concat([clear_images, tf.convert_to_tensor(c, np.float32)], axis=0)
    return clear_images, blur_images

latent_dim = 60
regularizer = keras.regularizers.l1_l2()

def build_encoder(latent_dim, shape, num_cluster=2):
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
    encoder = keras.Model(encoder_inputs, x, name="encoder")
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

def cluster_latent(y):
    y = np.array(y)
    labels = []
    for i in y:
        labels.append(np.argmax(i))
    return labels

def gen_clusters(imgs, labels, num_cluster):
    clusters = {}
    for label in range(num_cluster):
        clusters[label] = []
    for i in range(len(labels)):
        clusters[labels[i]].append(imgs[i])
    return clusters

def update_centers(x, r, K):
    centers = []
    for k in range(K):
        centers.append(tf.tensordot(r[:, k],x,axes=1) / tf.reduce_sum(r[:, k]))
    return centers

# https://github.com/Ashish77IITM/W-Net/blob/master/soft_n_cut_loss.py

def edge_weights(flatten_image, rows , cols, std_intensity=3, std_position=1, radius=5):
	A = outer_product(flatten_image, tf.ones_like(flatten_image))
	A_T = tf.transpose(A)
	intensity_weight = tf.exp(-1*tf.square((tf.divide((A - A_T), std_intensity))))

	xx, yy = tf.meshgrid(tf.range(rows), tf.range(cols))
	xx = tf.reshape(xx, (rows*cols,))
	yy = tf.reshape(yy, (rows*cols,))
	A_x = outer_product(xx, tf.ones_like(xx))
	A_y = outer_product(yy, tf.ones_like(yy))

	xi_xj = A_x - tf.transpose(A_x)
	yi_yj = A_y - tf.transpose(A_y)

	sq_distance_matrix = tf.square(xi_xj) + tf.square(yi_yj)

	dist_weight = tf.exp(-tf.divide(sq_distance_matrix,tf.square(std_position)))
	dist_weight = tf.cast(dist_weight, tf.float32)
	weight = tf.multiply(intensity_weight, dist_weight)
	return weight

def outer_product(v1,v2):
	v1 = tf.reshape(v1, (-1,))
	v2 = tf.reshape(v2, (-1,))
	v1 = tf.expand_dims((v1), axis=0)
	v2 = tf.expand_dims((v2), axis=0)
	return tf.matmul(tf.transpose(v1),(v2))

def numerator(k_class_prob,weights):
    k_class_prob = tf.reshape(k_class_prob, (-1,))
    return tf.reduce_sum(tf.multiply(weights,outer_product(k_class_prob,k_class_prob)))

def denominator(k_class_prob,weights):	
    k_class_prob = tf.cast(k_class_prob, tf.float32)
    k_class_prob = tf.reshape(k_class_prob, (-1,))	
    deno = tf.reduce_sum(tf.multiply(weights,outer_product(k_class_prob,tf.ones(tf.shape(k_class_prob)))))
    if deno == 0.0:
        return 0.1
    return deno

def soft_n_cut_loss(flatten_image, prob, k, rows, cols):
	soft_n_cut_loss = k
	weights = edge_weights(flatten_image, rows ,cols)
	for t in range(k): 
		soft_n_cut_loss = soft_n_cut_loss - (numerator(prob[:,t],weights)/denominator(prob[:,t],weights))
	return soft_n_cut_loss


class AutoEncoder(keras.Model):
    def __init__(self, encoder, decoder, alpha=1, num_cluster=2, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = alpha
        self.num_cluster = num_cluster

    def train_step(self, data):
        test = data[0][1]
        data = data[0][0]
        shape = (len(data[0]), len(data[0][0]))

        with tf.GradientTape() as tape:
            z, y, latent = self.encoder(data)
            
            # reconstruct images
            reconstruction = self.decoder(z)
            # calculate loss
            kl = tf.keras.losses.KLDivergence(
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
            kl_loss = kl(test, reconstruction)

            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.MSE(test, reconstruction))
            reconstruction_loss *= shape[0] * shape[1]

            # leng = 0
            # for k in range(len(y)):
            #     leng += 1
            # soft_cut_loss = 0
            # for i in range(8):
            #     soft_cut_loss += soft_n_cut_loss(latent[:,i], y, self.num_cluster, leng, 1)
            
            total_loss = reconstruction_loss + kl_loss# + soft_cut_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "reconstruction_loss": reconstruction_loss,
            #"soft_n_cut_loss": soft_cut_loss,
        }

class AutoEncoder_P(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(AutoEncoder_P, self).__init__(**kwargs)
        self.encoder = encoder
        self.trained_decoder = decoder

    def train_step(self, data):
        test = data[0][1]
        data = data[0][0]
        shape = (len(data[0]), len(data[0][0]))
        
        with tf.GradientTape() as tape:
            z, y, latent = self.encoder(data)
            # reconstruct images
            reconstruction = self.trained_decoder(z)
            # calculate loss
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.MSE(test, reconstruction))
            reconstruction_loss *= shape[0] * shape[1]
            kl = tf.keras.losses.KLDivergence(
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
            kl_loss = kl(test, reconstruction)
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trained_decoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trained_decoder.trainable_weights))
        return {
            "loss": total_loss,
        }

def clustering(blur_images, clear_images, encoder, num_clusters=2):
    x, y, latent = encoder(blur_images[:batch])
    # pca =  decomposition.PCA(n_components=2)
    for i in range(batch, len(blur_images), batch):
        y = np.concatenate([y, encoder(blur_images[i: i+batch])[1]], axis=0)
        x = np.concatenate([x, encoder(blur_images[i: i+batch])[0]], axis=0)
    labels = np.array(cluster_latent(y))

    clusters = gen_clusters(blur_images, labels, num_clusters)
    label_clusters = gen_clusters(clear_images, labels, num_clusters)
    clus = []
    for c in range(num_clusters):
        clus.append(np.array(clusters[c]))
    label_clus = []
    for c in range(num_clusters):
        label_clus.append(np.array(label_clusters[c]))
    return np.array(clus), np.array(label_clus)


def train_decoders(clus, label_clus, encoder, epochs=100, batch_size=128, lr=lr_schedule):
    decoders = []
    for i in range(len(clus)):
        decoder_i = build_decoder(latent_dim, shape,"decoder"+str(i))
        if len(clus[i]) > 0:
            model_i = VAE_P(encoder, decoder_i)
            model_i.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
            model_i.fit((clus[i],label_clus[i]), epochs=epochs, batch_size=batch_size)
        decoders.append(decoder_i)
    return decoders

def decode_images(z, labels, decoders):
    decoded_images = []
    decode = []
    for decoder in decoders:
        decode.append(decoder.predict(z))
    for num in range(len(z)):
        decode_img = decode[labels[num]][num]
        decoded_images.append(decode_img)
    return np.array(decoded_images)

def reconstruct_image(z, y, decoders, 
                      blocks_per_image=256, img_shape=(256,256), block_size=16):
    recons_images = []
    labels = cluster_latent(y)
    decoded_images = decode_images(z[:batch], labels[:batch], decoders)
    blocks = decoded_images[: blocks_per_image]
    image = merge_img(blocks, img_shape[0], img_shape[1], block_size, overlap=overlap)
    recons_images = tf.convert_to_tensor([image], np.float32)

    for i in range(batch, len(z), batch):
      decoded_images = np.concatenate([decoded_images, decode_images(z[i:i+batch], labels[i:i+batch], decoders)], axis=0)
    for i in range(blocks_per_image, len(decoded_images), blocks_per_image):
        blocks = decoded_images[i: i+blocks_per_image]
        image = merge_img(blocks, img_shape[0], img_shape[1], block_size, overlap=overlap)
        recons_images = tf.concat([recons_images, tf.convert_to_tensor([image], np.float32)], axis=0)
    return recons_images

def tune_parameters(param, blur_images, clear_images, validation_images, val_blur_imgs):
    results = []
    for i in param['init_lr']:
        for d in param['decay_steps']:
            for epoch in param['epochs']:
                for batch_size in param['batch_size']:
                    print('init_lr', i, 'decay_steps', d, 'epochs', epoch, 'batch_size', batch_size)
                    encoder = build_encoder(latent_dim, shape, num_cluster)
                    decoder = build_decoder(latent_dim, shape,"decoder")
                    model = AutoEncoder(encoder, decoder, 1, num_cluster)
                    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
                    model.fit((blur_images,clear_images), epochs=epoch, batch_size=batch_size)

                    clus, label_clus = clustering(blur_images, clear_images, encoder, num_cluster)
                    if len(clus[0]) < batch_size or len(clus[1]) < batch_size:
                        batch_size = min(len(clus[0]), len(clus[1])) // 2 + 1
                    decoders = train_decoders(clus, label_clus, encoder, epoch, batch_size=batch_size, lr=lr_schedule)

                    test_images_clear, test_images_blur = gen_train_set(
                        validation_images, val_blur_imgs, block_size=block_size)
                    z, y, latent= encoder.predict(test_images_blur)
                    decoded_imgs = decoder.predict(z)
                    recons_images = reconstruct_image(z, y,
                                                      decoders=decoders,
                                                      blocks_per_image=block_per_image,
                                                      block_size=block_size)
                    recons_images = (recons_images*255).astype('uint8')
                    test_images = []
                    for i in range(0, len(test_images_clear), block_per_image):
                        test_images.append(merge_img(test_images_clear[i:i+block_per_image],img_shape[0],img_shape[1], block_size, overlap=overlap))
                    test_images = (np.array(test_images)*255).astype('uint8')

                    recons_psnr = []
                    for i in range(len(recons_images)):
                        recons_psnr.append(cv2.PSNR(recons_images[i], test_images[i]))
                    results.append(np.array(recons_psnr).mean())
                    print(np.array(recons_psnr).mean())
    return results


    