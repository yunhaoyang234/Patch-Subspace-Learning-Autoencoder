import argparse
import pprint
import json

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=bool, default=True)
parser.add_argument('--block_size', type=int, default=24)
parser.add_argument('--num_block_per_row', type=int, default=187)
parser.add_argument('--overlap', type=int, default=8)
parser.add_argument('--file_batch', type=int, default=5)
parser.add_argument('--latent_dim', type=int, default=96)
parser.add_argument('--epoch', type=int, default=25)
parser.add_argument('--num_filter', type=int, default=4)
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
    DATASET = 'sidd'
    FILE_BATCH = args.file_batch

    '''
    Load Images
    '''
    train_files = []
    valid_files = []
    for root_path, sub, files in os.walk(args.train_files_path):
        contents = files
        contents.sort()
        for f in contents:
            file_path = os.path.join(root_path,f)
            if os.path.isfile(file_path) and "GT" in f:
                train_files.append(file_path)
            if os.path.isfile(file_path) and "NOISY" in f:
                valid_files.append(file_path)
    if len(train_files) != len(valid_files):
        raise ValueError('Train and Validation file must have same length', len(train_files), len(valid_files))
    
    '''
    Build Networks
    '''
    encoder = build_encoder(LATENT_DIM, SHAPE, NUM_CLUSTER)
    decoder = build_decoder(LATENT_DIM, SHAPE, 'decoder')
    decoders = [build_decoder(LATENT_DIM, SHAPE,"decoder"+str(i)) for i in range(NUM_CLUSTER)]

    '''
    Train Network
    '''
    for fb in range(0, len(train_files), FILE_BATCH):
        train = train_files[fb:fb+FILE_BATCH]
        valid = valid_files[fb:fb+FILE_BATCH]
        clear_images = crop_square(read_images(valid))
        noise_images = crop_square(read_images(train))
        WIDTH = len(clear_images[0][0])
        HEIGHT = len(clear_images[0])
        
        clear_images, noise_images = gen_train_set(clear_images, noise_images, SHAPE,
                                                   BLOCK_SIZE, NUM_BLOCK, OVERLAP)

        encoder, decoder = train_encoder(noise_images, clear_images, encoder, decoder, NUM_CLUSTER, SHAPE, EPOCH)
        clus, label_clus = clustering(noise_images, clear_images, encoder, NUM_CLUSTER)
        decoders = train_decoders(clus, label_clus, encoder, decoders, EPOCH, decoder.get_weights())

    save_models(encoder, decoders, DATASET + 'denoise_model/')

    '''
    Evaluation
    '''
    # encoder, decoders = load_models(DATASET + 'denoise_model/')
    avg_psnr, avg_ssim, avg_uqi = 0, 0, 0
    for i in range(4):
        clear_images = sidd_test_data(args.test_files_path, 'ValidationGtBlocksSrgb', i)
        test_images = np.array(clear_images)
        noise_images = sidd_test_data(args.test_validation_files_path, 'ValidationNoisyBlocksSrgb', i)
        WIDTH = len(clear_images[0][0])
        HEIGHT = len(clear_images[0])

        clear_images, noise_images = gen_train_set(clear_images, noise_images, SHAPE,
                                                   BLOCK_SIZE, NUM_BLOCK, OVERLAP)

        z, z_mean, z_sig, y, y_logits, z_prior_mean, z_prior_sig = encoder.predict(noise_images)
            
        recons_images = reconstruct_image(z, y, decoders, BLOCK_PER_IMAGE, WIDTH, 
                                          HEIGHT, BLOCK_SIZE, OVERLAP)
        recons_images = tf.cast((recons_images*255), dtype=tf.uint8)

        avg_psnr += quality_evaluation(recons_images, test_images, metric='PSNR')
        avg_ssim += quality_evaluation(recons_images, test_images, metric='SSIM')
        avg_uqi += quality_evaluation(recons_images, test_images, metric='UQI')
    print('***********************')
    print('Overall Results')
    print('PSNR: ', avg_psnr/len(test_files)*FILE_BATCH)
    print('SSIM: ', avg_ssim/len(test_files)*FILE_BATCH)
    print('UQI: ', avg_uqi/len(test_files)*FILE_BATCH)
    print('***********************')

if __name__ == '__main__':
    main(parser.parse_args())
