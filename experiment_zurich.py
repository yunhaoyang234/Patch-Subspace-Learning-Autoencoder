import argparse
import pprint
import json

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=bool, default=True)
parser.add_argument('--block_size', type=int, default=28)
parser.add_argument('--num_block_per_row', type=int, default=22)
parser.add_argument('--overlap', type=int, default=8)
parser.add_argument('--file_batch', type=int, default=100)
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--num_filter', type=int, default=2)
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


    BLOCK_SIZE = args.block_size
    NUM_BLOCK = args.num_block_per_row
    BLOCK_PER_IMAGE = NUM_BLOCK * NUM_BLOCK
    OVERLAP = args.overlap
    NUM_CLUSTER = args.num_filter
    SHAPE = (BLOCK_SIZE, BLOCK_SIZE, 3)
    LATENT_DIM = args.latent_dim
    EPOCH = args.epoch
    FILE_BATCH = args.file_batch
    DATASET = 'zurich'

    '''
    Load Images
    '''
    files = glob.glob(cwd + args.train_files_path + '*')
    train_files = [cwd + args.train_files_path + str(i) + '.png' for i in range(len(files))]
    valid_files = [cwd + args.validation_files_path + str(i) + '.jpg' for i in range(len(files))]

    '''
    Build Networks
    '''
    encoder = build_encoder(LATENT_DIM, (SHAPE[0]//2, SHAPE[1]//2, 4), NUM_CLUSTER)
    decoder = build_decoder(LATENT_DIM, SHAPE, 'decoder')
    decoders = [build_decoder(LATENT_DIM, SHAPE,"decoder"+str(i)) for i in range(NUM_CLUSTER)]

    '''
    Train Network
    '''
    for fb in range(0, len(train_files), FILE_BATCH):
        train = train_files[fb:fb+FILE_BATCH]
        valid = valid_files[fb:fb+FILE_BATCH]
        rgb_images = read_images(valid)
        raw_images = read_raw(train)
        WIDTH = len(rgb_images[0][0])
        HEIGHT = len(rgb_images[0])
        rgb_images, raw_images = gen_zurich_set(rgb_images, raw_images, SHAPE,
                                                    BLOCK_SIZE, NUM_BLOCK, OVERLAP)
       
        encoder, decoder = train_encoder(raw_images, rgb_images, encoder, decoder, NUM_CLUSTER, SHAPE, EPOCH)
        clus, label_clus = clustering(raw_images, rgb_images, encoder, NUM_CLUSTER)
        decoders = train_decoders(clus, label_clus, encoder, decoders, EPOCH, decoder.get_weights())

    save_models(encoder, decoders, DATASET + 'model/')

    '''
    Evaluation
    '''
    # encoder, decoders = load_models(DATASET + 'model/')

    files = glob.glob(cwd + args.test_files_path + '*')
    test_files = [cwd + args.test_files_path + str(i) + '.png' for i in range(len(files))]
    test_valid_files = [cwd + args.test_validation_files_path + str(i) + '.jpg' for i in range(len(files))]
    
    avg_psnr, avg_ssim, avg_uqi = 0, 0, 0

    for fb in range(0, len(test_files), FILE_BATCH):
        train = test_files[fb:fb+FILE_BATCH]
        valid = test_valid_files[fb:fb+FILE_BATCH]
        rgb_images = read_images(valid)
        raw_images = read_raw(train)
        WIDTH = len(rgb_images[0][0])
        HEIGHT = len(rgb_images[0])
        test_images = np.array(rgb_images)
        rgb_images, raw_images = gen_zurich_set(rgb_images, raw_images, SHAPE,
                                                    BLOCK_SIZE, NUM_BLOCK, OVERLAP)

        z, z_mean, z_sig, y, y_logits, z_prior_mean, z_prior_sig = encoder.predict(raw_images)
        recons_images = reconstruct_image(z, y, decoders, BLOCK_PER_IMAGE, WIDTH, 
                                          HEIGHT, BLOCK_SIZE, OVERLAP)
        recons_images = tf.cast((recons_images*255), dtype=tf.uint8)

        avg_psnr += quality_evaluation(recons_images, test_images, metric='PSNR')
        avg_ssim += ms_ssim(recons_images, test_images)
    print('***********************')
    print('Overall Results')
    print('PSNR: ', avg_psnr/len(test_files)*FILE_BATCH)
    print('SSIM: ', avg_ssim/len(test_files)*FILE_BATCH)
    print('***********************')

if __name__ == '__main__':
    main(parser.parse_args())
