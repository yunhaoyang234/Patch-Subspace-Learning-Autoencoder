# -*- coding: utf-8 -*-

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from autoencoder import *

print("load data")
cwd = ""
validation_images = load_images(cwd + 'validation-r08-s-0000-of-0040.tfrecords')
images = load_images(cwd + 'train-r08-s-0000-of-0120.tfrecords')
print(validation_images.shape)
print(images.shape)

"""## Generate Blur Images"""

print("generate noise")
blur_imgs = gen_noise(images)
val_blur_imgs = gen_noise(validation_images)
print(blur_imgs.shape)
print(cv2.PSNR(blur_imgs[0], images[0]))

"""## Divide and Merge Images"""

print("generate training set")

clear_images, blur_images = gen_train_set(images, blur_imgs, block_size=block_size)

print(clear_images.shape, blur_images.shape)

# print("tune hyperparameter")
# param = {'init_lr':[0.01, 0.001, 0.0001], 'decay_steps':[100,1000], 'epochs':[50,100],
#         'batch_size': [64, 128]}
# results = tune_parameters(param, blur_images, clear_images, validation_images, val_blur_imgs)
# print(results)
# exit()

print("train autoencoder")
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)

epoch = 100

encoder = build_encoder(latent_dim, shape, num_cluster)
decoder = build_decoder(latent_dim, shape,"decoder")
model = AutoEncoder(encoder, decoder, 1, num_cluster)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
model.fit((blur_images,clear_images), epochs=epoch, batch_size=128)

print("clustering")
clus, label_clus = clustering(blur_images, clear_images, encoder, num_cluster)
print(clus[0].shape, clus[1].shape)

print("train decoders")
decoders = train_decoders(clus, label_clus, encoder, epoch)

print("reconstruct images")
test_images_clear, test_images_blur = gen_train_set(
    validation_images, val_blur_imgs, block_size=block_size)

batch = 10000
z, y, latent = encoder.predict(test_images_blur[:batch])
for i in range(batch, len(test_images_blur), batch):
    y = np.concatenate([y, encoder.predict(test_images_blur[i: i+batch])[1]], axis=0)
    z = np.concatenate([z, encoder.predict(test_images_blur[i: i+batch])[0]], axis=0)

decoded_imgs = decoder.predict(z[:batch])
for i in range(batch, len(z), batch):
    decoded_imgs = np.concatenate([decoded_imgs, decoder.predict(z[i:i+batch])], axis=0)

recons_images = reconstruct_image(z, y,
                                  decoders=decoders,
                                  blocks_per_image=block_per_image,
                                  img_shape=img_shape,
                                  block_size=block_size)
recons_images = (recons_images*255).astype('uint8')
print(recons_images.shape)

comp_images = reconstruct_image(z, y, [decoder]*num_cluster,
                                blocks_per_image=block_per_image,
                                img_shape=img_shape,
                                block_size=block_size)
comp_images = (comp_images*255).astype('uint8')

test_images = []
for i in range(0, len(test_images_clear), block_per_image):
    test_images.append(merge_img(test_images_clear[i:i+block_per_image], img_shape[0], img_shape[1], block_size, overlap=overlap))
test_images = (np.array(test_images)*255).astype('uint8')
print(comp_images.shape)

print("Quality Metrics")
cnt = 0
recons_psnr = []
comp_psnr = []
for i in range(len(recons_images)):
    recons_psnr.append(cv2.PSNR(recons_images[i], test_images[i]))
    comp_psnr.append(cv2.PSNR(comp_images[i], test_images[i]))
    if cv2.PSNR(recons_images[i], test_images[i]) > cv2.PSNR(comp_images[i], test_images[i]):
        cnt += 1
print(np.array(recons_psnr).mean(), np.array(comp_psnr).mean())
print(cnt/len(test_images))

cnt = 0
recons_ssim = []
comp_ssim = []
for i in range(len(recons_images)):
    recons_ssim.append(ssim(recons_images[i], test_images[i], multichannel=True))
    comp_ssim.append(ssim(comp_images[i], test_images[i], multichannel=True))
    if ssim(recons_images[i], test_images[i], multichannel=True) > ssim(comp_images[i], test_images[i], multichannel=True):
        cnt += 1
print('SSIM')
print(np.array(recons_ssim).mean(), np.array(comp_ssim).mean())
print(cnt/len(test_images))

cnt = 0
recons_se = []
comp_se = []
for i in range(len(recons_images)):
    recons_se.append(sewar.full_ref.uqi(recons_images[i], test_images[i], ws=8))
    comp_se.append(sewar.full_ref.uqi(comp_images[i], test_images[i], ws=8))
    if sewar.full_ref.uqi(recons_images[i], test_images[i], 
                        ws=8) > sewar.full_ref.uqi(comp_images[i], test_images[i], ws=8):
        cnt += 1
print('UQI')
print(np.array(recons_se).mean(), np.array(comp_se).mean())
print(cnt/len(test_images))

