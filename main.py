import argparse
import pprint
import json

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=bool, default=True)
parser.add_argument('--block_size', type=int, default=18)
parser.add_argument('--num_block', type=int, default=18)
parser.add_argument('--overlap', type=int, default=4)
parser.add_argument('--batch', type=int, default=15000)
parser.add_argument('--latent_dim', type=int, default=60)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--num_cluster', type=int, default=4)
parser.add_argument('--dataset', type=str, default='celeba', help='celeba, sidd, zurich')
parser.add_argument('--train_files_path', type=str, default='')
parser.add_argument('--test_files_path', type=str, default='')
parser.add_argument('--validation_files_path', type=str, default='')
parser.add_argument('--test_validation_files_path', type=str, default='')

from train import *
from evaluation import *

def main(args):
    # Enable GPU
    if args.gpu:
        #%tensorflow_version 2.x
        import tensorflow as tf
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
          raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    '''
    CelebA: bs=16, nb=21, ol=4
    Zurich:    28.    22.    8
    SIDD:      32.    107.   4
    '''

    BLOCK_SIZE = args.block_size
    NUM_BLOCK = args.num_block
    BLOCK_PER_IMAGE = NUM_BLOCK * NUM_BLOCK
    OVERLAP = args.overlap
    NUM_CLUSTER = args.num_cluster
    SHAPE = (BLOCK_SIZE, BLOCK_SIZE, 3)
    BATCH_SIZE = max(args.batch, BLOCK_PER_IMAGE)
    LATENT_DIM = args.latent_dim
    EPOCH = args.epoch
    DATASET = args.dataset
    FILE_BATCH = 2

    '''
    Load Images
    '''
    train_files = sorted(glob.glob(cwd + args.train_files_path + '*'))
    valid_files = sorted(glob.glob(cwd + args.validation_files_path + '*'))

    print(len(train_files))
    print()
    print(len(valid_files))
    print()
    if len(train_files) != len(valid_files):
        raise ValueError('Train and Validation file must have same length', len(train_files), len(valid_files))


    encoder = build_encoder(LATENT_DIM, SHAPE, NUM_CLUSTER)
    decoder = build_decoder(LATENT_DIM, SHAPE, 'decoder')
    decoders = [build_decoder(LATENT_DIM, SHAPE,"decoder"+str(i)) for i in range(NUM_CLUSTER)]

    '''
    Train Network
    '''
    for fb in range(0, len(train_files), FILE_BATCH):
        train = train_files[fb:fb+FILE_BATCH]
        valid = valid_files[fb:fb+FILE_BATCH]
        if DATASET == 'celeba':
            clear_images = [load_celeb_images(train[i]) for i in range(FILE_BATCH)]
            clear_images = np.concatenate(clear_images, axis=0)
            noise_images = gen_noise(clear_images)
        elif DATASET == 'sidd':
            clear_images = crop_square(read_images(valid))
            noise_images = crop_square(read_images(train))
        else:
            clear_images = read_images(valid)
            noise_images = read_images(train)
        WIDTH = len(clear_images[0][0])
        HEIGHT = len(clear_images[0])
        clear_images, noise_images = gen_large_train_set(clear_images, noise_images, SHAPE,
                                                         BLOCK_SIZE, BATCH_SIZE, NUM_BLOCK, OVERLAP)
        encoder, decoder = train_encoder(noise_images, clear_images, encoder, decoder, NUM_CLUSTER, SHAPE, EPOCH)
        clus, label_clus = clustering(noise_images.numpy(), clear_images.numpy(), encoder, 
                                      NUM_CLUSTER, BATCH_SIZE)
        decoders = train_decoders(clus, label_clus, encoder, decoders, EPOCH)

    # save_models(encoder, decoder, decoders, DATASET + '/')

    '''
    Evaluation
    '''
    # encoder, decoder, decoders = load_models(DATASET + '/')

    test_files = sorted(glob.glob(cwd + args.test_files_path + '*'))
    test_valid_files = sorted(glob.glob(cwd + args.test_validation_files_path + '*'))
    if len(test_files) != len(test_valid_files):
        raise ValueError('Test and Validation file must have same length', len(test_files), len(test_valid_files))

    psnr, ssim, uqi = 0, 0, 0

    for fb in range(0, len(test_files), FILE_BATCH):
        train = test_files[fb:fb+FILE_BATCH]
        valid = test_valid_files[fb:fb+FILE_BATCH]
        if DATASET == 'celeba':
            clear_images = [load_celeb_images(train[i]) for i in range(FILE_BATCH)]
            clear_images = np.concatenate(clear_images, axis=0)
            noise_images = gen_noise(clear_images)
        elif DATASET == 'sidd':
            clear_images = crop_square(read_images(valid))
            noise_images = crop_square(read_images(train))
        else:
            clear_images = read_images(valid)
            noise_images = read_images(train)
        test_images = np.array(clear_images)
        clear_images, noise_images = gen_large_train_set(clear_images, noise_images, SHAPE,
                                                         BLOCK_SIZE, BATCH_SIZE, NUM_BLOCK, OVERLAP)
        z, z_mean, z_sig, y, y_logits, z_prior_mean, z_prior_sig = encoder.predict(noise_images[:BATCH_SIZE])
        for i in range(BATCH_SIZE, len(noise_images), BATCH_SIZE):
            new_z, m, s, y, log, pm, ps = encoder.predict(noise_images[i: i+BATCH_SIZE])
            y_logits = np.concatenate([y_logits, log], axis=0)
            z = np.concatenate([z, new_z], axis=0)
        
        recons_images = reconstruct_image(z, y_logits, [decoder]*NUM_CLUSTER, 
                                          BATCH_SIZE, BLOCK_PER_IMAGE, WIDTH, 
                                          HEIGHT, BLOCK_SIZE, OVERLAP)
        comp_images = reconstruct_image(z, y_logits, decoders, BATCH_SIZE, BLOCK_PER_IMAGE, 
                                        WIDTH, HEIGHT, BLOCK_SIZE, OVERLAP)
        recons_images = tf.cast((recons_images*255), dtype=tf.uint8)
        comp_images = tf.cast((comp_images*255), dtype=tf.uint8)

        psnr += quality_evaluation(recons_images, test_images, comp_images, metric='PSNR')
        ssim += quality_evaluation(recons_images, test_images, comp_images, metric='SSIM')
        uqi += quality_evaluation(recons_images, test_images, comp_images, metric='UQI')
    print('***********************')
    print('Overall Results')
    print('PSNR: ', psnr/len(test_files))
    print('SSIM: ', ssim/len(test_files))
    print('UQI: ', uqi/len(test_files))
    print('***********************')

if __name__ == '__main__':
    main(parser.parse_args())
