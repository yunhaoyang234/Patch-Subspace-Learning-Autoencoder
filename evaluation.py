from utils import *
from skimage.metrics import structural_similarity as ssim
import sewar

def original_psnr(blur_imgs):
    psnr = 0
    for b in range(len(blur_imgs)):
        psnr += cv2.PSNR(blur_imgs[b], images[b])
    print('Initial PSNR', psnr/len(blur_imgs))

def quality_evaluation(recons_images, test_images, comp_images, metric='PSNR'):
    cnt = 0
    recons = []
    comp = []
    for i in range(len(recons_images)):
        metric_recons, metric_comp = 0, 0
        if metric == 'PSNR':
            metric_recons = cv2.PSNR(recons_images[i].numpy(), test_images[i])
            metric_comp = cv2.PSNR(comp_images[i].numpy(), test_images[i])
        elif metric == 'SSIM':
            metric_recons = ssim(recons_images[i].numpy(), test_images[i], multichannel=True)
            metric_comp = ssim(comp_images[i].numpy(), test_images[i], multichannel=True)
        elif metric == 'UQI':
            metric_recons = sewar.full_ref.uqi(recons_images[i].numpy(), test_images[i], ws=8)
            metric_comp = sewar.full_ref.uqi(comp_images[i].numpy(), test_images[i], ws=8)
        recons.append(metric_recons)
        comp.append(metric_comp)
        if metric_recons > metric_comp:
            cnt += 1
    print(metric)
    print(np.array(recons).mean(), np.array(comp).mean())
    print(cnt/len(test_images))