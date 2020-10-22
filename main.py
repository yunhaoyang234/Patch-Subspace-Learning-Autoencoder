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

def main(args):
    # Enable GPU
    if args.gpu:
        %tensorflow_version 2.x
        import tensorflow as tf
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
          raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    from train import *
    from evaluation import *

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

    '''
    Generate Noise
    '''
    blur_imgs = gen_noise(images, 200, 100, 350)
    val_blur_imgs = gen_noise(validation_images, 200, 100, 350)
    clear_images, noise_images = gen_large_train_set(images, blur_imgs, BLOCK_SIZE, BATCH_SIZE)

    '''
    Train Network
    '''
    encoder, decoder = train_encoder(noise_images, clear_images, LATENT_DIM, SHAPE, NUM_CLUSTER, EPOCH)
    clus, label_clus = clustering(noise_images.numpy(), clear_images.numpy(), encoder, NUM_CLUSTER, BATCH_SIZE)
    decoders = train_decoders(clus, label_clus, encoder, LATENT_DIM, SHAPE, EPOCH)

    '''
    Reconstruct
    '''
    test_images_clear, test_images_blur = gen_large_train_set(validation_images, val_blur_imgs)
    z, z_mean, z_sig, y, y_logits, z_prior_mean, z_prior_sig = encoder.predict(test_images_blur[:BATCH_SIZE])
    for i in range(BATCH_SIZE, len(test_images_blur), BATCH_SIZE):
        new_z, m, s, y, log, pm, ps = encoder.predict(test_images_blur[i: i+BATCH_SIZE])
        y_logits = np.concatenate([y_logits, log], axis=0)
        z = np.concatenate([z, new_z], axis=0)

    decoded_imgs = decoder.predict(z[:BATCH_SIZE])
    for i in range(BATCH_SIZE, len(z), BATCH_SIZE):
        decoded_imgs = np.concatenate([decoded_imgs, decoder.predict(z[i:i+BATCH_SIZE])], axis=0)

    recons_images = reconstruct_image(z, y_logits, [decoder]*NUM_CLUSTER, 
                                      BATCH_SIZE, BLOCK_PER_IMAGE, WIDTH, 
                                      HEIGHT, BLOCK_SIZE, OVERLAP)
    comp_images = reconstruct_image(z, y_logits, decoders, BATCH_SIZE, BLOCK_PER_IMAGE, 
                                    WIDTH, HEIGHT, BLOCK_SIZE, OVERLAP)
    recons_images = tf.cast((recons_images*255), dtype=tf.uint8)
    comp_images = tf.cast((comp_images*255), dtype=tf.uint8)
    test_images = validation_images

    '''
    Evaluate
    '''
    quality_evaluation(recons_images, test_images, comp_images, metric='PSNR')
    quality_evaluation(recons_images, test_images, comp_images, metric='SSIM')
    quality_evaluation(recons_images, test_images, comp_images, metric='UQI')

if __name__ == '__main__':
    main(parser.parse_args())
