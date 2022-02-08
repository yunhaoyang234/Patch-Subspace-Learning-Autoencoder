from utils import *
from skimage.metrics import structural_similarity as ssim
import sewar

def original_psnr(blur_imgs):
    psnr = 0
    for b in range(len(blur_imgs)):
        psnr += cv2.PSNR(blur_imgs[b], images[b])
    print('Initial PSNR', psnr/len(blur_imgs))

def ms_ssim(recons_images, test_images):
    ms = tf.reduce_mean(tf.image.ssim_multiscale(test_images, recons_images, 255, k1=0.01, k2=0.07))
    return ms

def quality_evaluation(recons_images, test_images, metric='PSNR', display=True):
    recons = []
    for i in range(len(recons_images)):
        metric_recons = 0
        if metric == 'PSNR':
            metric_recons = cv2.PSNR(recons_images[i], test_images[i])
        elif metric == 'SSIM':
            metric_recons = ssim(recons_images[i], test_images[i], multichannel=True)
        elif metric == 'UQI':
            metric_recons = sewar.full_ref.uqi(recons_images[i], test_images[i], ws=8)
        recons.append(metric_recons)
    if display:
        print(metric)
        print(np.array(recons).mean())
    return np.array(recons).mean()