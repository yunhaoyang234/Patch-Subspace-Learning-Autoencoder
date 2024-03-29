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
import torch
import torchvision
import imageio
from keras.layers import Input, Dense, Lambda

cwd = ''

'''
PREPROCESSING: LOAD DATASET
'''
def load_celeb_images(file_path):
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
    return images

def read_images(files):
    data = []
    for f1 in files:
        img = []
        img = cv2.imread(f1)
        data.append(img)
    return data

def extract_bayer_channels(raw):
    # Reshape the input bayer image
    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm

def read_raw(files):
    data = []
    for f in files:
        I = np.asarray(imageio.imread((f)))
        I = extract_bayer_channels(I)
        data.append(I)
    return data

def crop_square(imgs, length = 3000):
    data = []
    for img in imgs:
        img1, img2 = img[:length, -length:], img[:length, :length]
        data.append(cv2.resize(img1, (length,length)))
        data.append(cv2.resize(img2, (length,length)))
    return data

def recover_square(img1, img2, shape, length=3000):
    h, w = shape[0], shape[1]
    if h > w:
        ol = np.array(np.mean([img1[:w-h+length, :], img2[h-w-length:, :]], axis=0), dtype='uint8')
        img = np.concatenate([img2[:h-length, :], ol, img1[length-h:,:]], axis=0)
    else:
        ol = np.array(np.mean([img1[:, :h-w+length], img2[:, w-h-length:]], axis=0), dtype='uint8')
        print(ol.shape, img2.shape, img1.shape)
        img = np.concatenate([img2[:, :w-length], ol, img1[:, length-w:]], axis=1)
    return img

def sidd_test_data(path, key, batch):
    import scipy.io
    mat = scipy.io.loadmat(path)
    tmp = mat.get(key)
    images = []
    for i in range(batch*10,batch*10+10):
        for j in range(32):
            images.append(cv2.resize(tmp[i][j],(248,248)))
    return images

'''
PREPROCESSING: GENERATE TRAINGING AND VALIDATION DATA
'''
def gen_global_noise(images):
    noise = []
    for img in images:
        noise_image = np.copy(img)
        noise_img = random_noise(noise_image, mode='gaussian',var=0.02)
        noise_image = np.array(255*noise_img, dtype = 'uint8')
        noise.append(noise_image)
    return noise

def gen_noise(images):
    noise = []
    ht, wd = len(images[0]), len(images[0][0])
    p, q, r, s = ht//10, ht//5, ht//3, ht//2
    for img in images:
        noise_image = np.copy(img)

        noise_img = random_noise(noise_image[p:p+s, q:wd-p], mode='s&p',amount=0.05)
        noise_image[p:p+s, q:wd-p] = np.array(255*noise_img, dtype = 'uint8')

        noise_img = random_noise(noise_image[q:, :wd-p], mode='poisson')
        noise_image[q:, :wd-p] = np.array(255*noise_img, dtype = 'uint8')

        noise_img = random_noise(noise_image[r:r+s, q:q+s], mode='speckle', mean=-0.1)
        noise_image[r:r+s, q:q+s] = np.array(255*noise_img, dtype = 'uint8')
        noise.append(noise_image)
    return noise

def gen_blur(images, x=50, y=150, z=200):
    noise = []
    for img in images:
        noise_image = np.copy(img)
        def get_random():
            p = random.randint(0, x)
            q = random.randint(0, x)
            h = random.randint(y, z)
            w = random.randint(y, z)
            return p, q, h, w

        p, q, h, w = get_random()
        noise_img = cv2.GaussianBlur(noise_image[p:p+w, q:q+h],(11,11),0)
        noise_image[p:p+w, q:q+h] = np.array(noise_img, dtype = 'uint8')

        p, q, h, w = get_random()
        noise_img = cv2.medianBlur(noise_image[p:p+w, q:q+h],5)
        noise_image[p:p+w, q:q+h] = np.array(noise_img, dtype = 'uint8')

        p, q, h, w = get_random()
        noise_img = cv2.bilateralFilter(noise_image[p:p+w, q:q+h],15,75,75)
        noise_image[p:p+w, q:q+h] = np.array(noise_img, dtype = 'uint8')
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

def gen_train_set(clear_imgs, blur_imgs, shape, block_size, num_block, overlap):
    noise_images = np.expand_dims(np.zeros(shape), 0)
    clear_images = np.expand_dims(np.zeros(shape), 0)

    for i in range(len(clear_imgs)):
        blocks = divide_img(clear_imgs[i], block_size, num_block, overlap)
        clear_images = np.concatenate([clear_images, blocks])
        blur_blocks = divide_img(blur_imgs[i], block_size, num_block, overlap)
        noise_images = np.concatenate([noise_images, blur_blocks])
    return clear_images[1:]/255, noise_images[1:]/255

'''
DECODE AND RECONSTRUCT IMAGES
'''
def cluster_latent(y, batch=10000):
    labels = []
    for i in range(0, len(y), batch):
        y_ = np.array(y[i:i+batch])
        for j in y_:
            labels.append(np.argmax(j))
    return labels

def gen_clusters(imgs, labels, num_cluster):
    clusters = []
    for label in range(num_cluster):
        clusters.append([])
    for i in range(len(labels)):
        clusters[labels[i]].append(imgs[i])
    for i in range(num_cluster):
        clusters[i] = np.array(clusters[i])
    return clusters

def clustering(noisy_patches, clean_patches, encoder, classifier, num_cluster, plot=False):
    z = encoder(noisy_patches[:10000])
    y, y_logits = classifier(z)
    for i in range(10000, len(noisy_patches), 10000):
        new_z = encoder(noisy_patches[i: i+10000])
        new_y, _ = classifier(new_z)
        z = np.concatenate([z, new_z], axis=0)
        y = np.concatenate([y, new_y], axis=0)
    labels = np.array(cluster_latent(y, 10000))

    if plot:
        pca =  decomposition.PCA(n_components=2)
        x = pca.fit_transform(z)
        colors = ['orange', 'blue', 'green', 'red', 'black', 'purple']
        for n in range(num_cluster):
          plt.scatter(x[labels==n, 0], x[labels==n, 1], s=5, c=colors[n], label ='Cluster'+str(n))
        plt.show()
    clus = gen_clusters(noisy_patches, labels, num_cluster)
    label_clus = gen_clusters(clean_patches, labels, num_cluster)
    return clus, label_clus

def decode_images(x, encoder, classifier, decoders):
    z = encoder.predict(x)
    y,_ = classifier.predict(z)
    decoded_images = []
    decode = []
    for i in range(len(decoders)):
        decode.append(decoders[i].predict(z))
    for i in range(len(y)):
        decode_img = decode[np.argmax(y[i])][i]
        decoded_images.append(decode_img)
    return decoded_images

def reconstruct_image(noise_blocks, encoder, classifier, decoders, 
                      block_per_image, width, height, block_size, overlap):
    recons_images = []
    decoded_images = decode_images(noise_blocks[:10000], encoder, classifier, decoders)
    for i in range(10000, len(noise_blocks), 10000):
        decoded_images = np.concatenate([decoded_images, 
                                         decode_images(noise_blocks[i:i+10000], encoder, classifier, decoders)], 
                                        axis=0)

    blocks = decoded_images[: block_per_image]
    image = merge_img(blocks, width, height, block_size, overlap)
    recons_images = np.array([image])
    for i in range(block_per_image, len(decoded_images), block_per_image):
        blocks = decoded_images[i: i+block_per_image]
        image = merge_img(blocks, width, height, block_size, overlap)
        recons_images = np.concatenate([recons_images, np.array([image])], axis=0)
    return (recons_images*255).astype(np.uint8)

'''
PLOT IMAGES
'''
def get_image_grid(images_np, nrow=8):
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=20, interpolation='lanczos'):
    images_np = np.swapaxes(np.swapaxes(images_np, 1, 3), 2,3)
    n_channels = max(x.shape[0] for x in images_np)
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]
    grid = get_image_grid(images_np, nrow)
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    plt.show()
