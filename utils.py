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

cwd = ''

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

def gen_noise(images, x=100, y=50, z=150):
    noise = []
    for img in images:
        noise_image = np.copy(img)
        def get_random():
            p = random.randint(0, x)
            q = random.randint(0, z)
            h = random.randint(y, z)
            w = random.randint(y, x)
            return p, q, h, w

        p, q, h, w = get_random()
        noise_img = random_noise(noise_image[p:p+w, q:q+h], mode='s&p',amount=0.2)
        noise_image[p:p+w, q:q+h] = np.array(255*noise_img, dtype = 'uint8')

        p, q, h, w = get_random()
        noise_img = random_noise(noise_image[p:p+w, q:q+h], mode='gaussian', mean=0.2)
        noise_image[p:p+w, q:q+h] = np.array(255*noise_img, dtype = 'uint8')

        p, q, h, w = get_random()
        noise_img = random_noise(noise_image[p:p+w, q:q+h], mode='speckle', mean=0.2)
        noise_image[p:p+w, q:q+h] = np.array(255*noise_img, dtype = 'uint8')
        noise.append(noise_image)
    return noise

def divide_img(img, block_size=18, num_block=18, overlap=4):
    height = len(img)
    width = len(img[0])
    if not (block_size*num_block - (num_block - 1)*overlap == height):
        raise ValueError('Block size mismatch', block_size*num_block - (num_block - 1)*overlap, height)
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
    noise_images = []
    clear_images = []

    for i in range(len(clear_imgs)):
        blocks = divide_img(clear_imgs[i], block_size, NUM_BLOCK, OVERLAP)
        for b in blocks:
            clear_images.append(b)
        blur_blocks = divide_img(blur_imgs[i], block_size, NUM_BLOCK, OVERLAP)
        for bb in blur_blocks:
            noise_images.append(bb)
    return np.array(clear_images)/255, np.array(noise_images)/255

def gen_large_train_set(clear_imgs, blur_imgs, block_size, batch_size):
    c, b = gen_train_set(clear_imgs[:batch_size], blur_imgs[:batch_size], block_size)
    noise_images = tf.convert_to_tensor(b, np.float32)
    clear_images = tf.convert_to_tensor(c, np.float32)
    
    for i in range(batch_size, len(blur_imgs), batch_size):
        c, b = gen_train_set(clear_imgs[i:i+batch_size], blur_imgs[i:i+batch_size], block_size)
        noise_images = tf.concat([noise_images, tf.convert_to_tensor(b, np.float32)], axis=0)
        clear_images = tf.concat([clear_images, tf.convert_to_tensor(c, np.float32)], axis=0)
    return clear_images, noise_images

def decode_images(z, labels, decoders):
    decoded_images = []
    decode = []
    for decoder in decoders:
        decode.append(decoder.predict(z))
    for num in range(len(z)):
        decode_img = decode[labels[num]][num]
        decoded_images.append(decode_img)
    return decoded_images

def reconstruct_image(z, y, decoders, batch_size, block_per_image, width, height, block_size, overlap):
    recons_images = []
    labels = cluster_latent(y, batch_size)
    decoded_images = decode_images(z[:batch_size], labels[:batch_size], decoders)
    blocks = decoded_images[: block_per_image]
    image = merge_img(blocks, width, height, block_size, overlap)
    recons_images = tf.convert_to_tensor([image], np.float32)

    for i in range(batch_size, len(z), batch_size):
      decoded_images = np.concatenate([decoded_images, 
                                       decode_images(z[i:i+batch_size], 
                                                     labels[i:i+batch_size], 
                                                     decoders)], 
                                      axis=0)
    for i in range(block_per_image, len(decoded_images), block_per_image):
        blocks = decoded_images[i: i+block_per_image]
        image = merge_img(blocks, width, height, block_size, overlap)
        recons_images = tf.concat([recons_images, tf.convert_to_tensor([image], np.float32)], axis=0)
    return recons_images
