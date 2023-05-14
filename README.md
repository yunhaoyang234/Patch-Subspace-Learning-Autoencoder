# Patch Subspace Learning Autoencoder (PS-VAE)
[[paper]](https://arxiv.org/abs/2104.00253) 
Code for reproducing results in **Learning Patch Subspaces on Autoencoders**.

We present a specific patch-based, local subspace deep neural network that improves Camera ISP to be robust to heterogeneous artifacts (especially image denoising). We call our three-fold deep-trained model the Patch Subspace Learning Autoencoder (PSL-AE).
The PSL-AE model does not make assumptions regarding uniform levels of image distortion. Instead, it first encodes patches extracted from noisy and clean image pairs, with different artifact types or distortion levels, by contrastive learning. Then, the patches of each image are encoded into corresponding soft clusters within their suitable latent sub-space, utilizing a prior mixture model. Furthermore, the decoders undergo training in an unsupervised manner, specifically trained for the image patches present in each cluster. The experiments highlight the adaptability and efficacy through enhanced heterogeneous filtering, both from synthesized artifacts but also realistic SIDD image pairs.

Patch-based self-supervised pretraining using contrastive learning:
![contrastive](https://github.com/yunhaoyang234/Patch-Subspace-Learning-Autoencoder/blob/master/figures/architecture_contrastive.png)

Training the encoder and a dummy decoder:
![encoder](https://github.com/yunhaoyang234/Patch-Subspace-Learning-Autoencoder/blob/master/figures/architecture_encoder.png)

Training multiple decoders:
![decoder](https://github.com/yunhaoyang234/Patch-Subspace-Learning-Autoencoder/blob/master/figures/architecture_decoder.png)

## Requirements:
See requirement.txt\
Run
`pip install -r requirement.txt` \
GPU is required

## Datasets:
- `CelebA` - 4GB. CelebA-HQ 256x256 dataset. Downloaded from [here](https://openaipublic.azureedge.net/glow-demo/data/celeba-tfr.tar). The synthesized noise is added by calling gen_noise() and gen_global_noise() functions in [utils.py](https://github.com/yunhaoyang234/Patch-Subspace-Learning-Autoencoder/blob/master/utils.py)
- `SIDD-Medium Dataset` - 12GB. Smartphone Image Denoising Dataset consists of 320 image pairs (noisy and ground-truth). Download from [here](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)

Please put the decompressed datasets in the same directory with the code during experiments, otherwise please set **cwd** in utils.py to the file directory where the datasets locate at.

## Experiments:
#### CelebA Denoising Experiment
```bash
$ python3 experiment_celeba.py \
          --file_batch 5\
          --epoch 50\
      	  --num_filter 1\
      	  --train_files_path "celeba_train(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
      	  --test_files_path "celeba_test(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"
```
The reconstruction quality of test data evaluated by PSNR, SSIM, and UQI will be printed out, and the trained model will be saved to the current working directory.

Denoising visualization:
![celeba](https://github.com/yunhaoyang234/Patch-Subspace-Learning-Autoencoder/blob/master/figures/denoise_zoom_celeb.png)

#### SIDD Denoising Experiment
```bash
$ python3 experiment_sidd_denoise.py \
          --file_batch 5\
          --epoch 25\
          --latent_dim 96\
      	  --num_filter 1\
      	  --train_files_path "sidd_noise(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
      	  --validation_file_path "sidd_ground_truth(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
      	  --test_files_path "sidd_test_noise(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
      	  --test_validation_files_path "sidd_test_GT(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"
    	  
```
The denoisng quality of the SIDD Benchmark images evaluated by PSNR, SSIM, and UQI will be printed out, and the trained model will be saved to the current working directory.

Denoising visualization:
![sidd1](https://github.com/yunhaoyang234/Patch-Subspace-Learning-Autoencoder/blob/master/figures/sidd1.png)
![sidd2](https://github.com/yunhaoyang234/Patch-Subspace-Learning-Autoencoder/blob/master/figures/sidd3.png)
