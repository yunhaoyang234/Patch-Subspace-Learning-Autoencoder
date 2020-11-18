import argparse
import pprint
import json

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=bool, default=True)
parser.add_argument('--block_size', type=int, default=28)
parser.add_argument('--num_block_per_row', type=int, default=135)
parser.add_argument('--overlap', type=int, default=6)
parser.add_argument('--input_files_path', type=str, default='')
parser.add_argument('--output_files_path', type=str, default='huawei_rgb_full_resolution/')

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
    SHAPE = (BLOCK_SIZE, BLOCK_SIZE, 3)
    DATASET = 'zurich'
    
    encoder, decoder, decoders = load_models(DATASET + 'model/')

    files = glob.glob(cwd + args.input_files_path + '*')
    for i in range(len(files)):
        huawei_raw = read_raw([files[i]])
        raw_images = crop_square(huawei_raw, 1488)
        rgb_images = np.zeros((2, 2976, 2976, 3))
        rgb_images, raw_images = gen_zurich_set(rgb_images, raw_images, SHAPE,
                                                    BLOCK_SIZE, NUM_BLOCK, OVERLAP)

        z, z_mean, z_sig, y, y_logits, z_prior_mean, z_prior_sig = encoder.predict(raw_images)

        recons_images = reconstruct_image(z, y_logits, decoders, BLOCK_PER_IMAGE, 2976, 
                                      2976, BLOCK_SIZE, OVERLAP)

        recons_images = tf.cast((recons_images*255), dtype=tf.uint8)
        recons_images = recons_images.numpy()
        image = recover_square(recons_images[0], recons_images[1], (2976, 3968), 2976)
        print(cv2.imwrite(cwd + args.output_file_path + str(i+1) +'.jpg', image))

if __name__ == '__main__':
    main(parser.parse_args())
