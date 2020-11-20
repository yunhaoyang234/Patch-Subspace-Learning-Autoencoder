# Patch Subspace Variational Autoencoder (PS-VAE)
Code for reproducing results in **Learning Latent Subspaces in Camera ISP Variational Autoencoders**.

## Requirements:
See requirement.txt\
Run
`pip install -r requirement.txt` \
GPU is required

## Datasets:
- `CelebA` - 4GB. CelebA-HQ 256x256 dataset. Downloaded from [here](https://openaipublic.azureedge.net/glow-demo/data/celeba-tfr.tar)
- `SIDD-Medium Dataset` - 12GB. Smartphone Image Denoising Dataset consists of 320 image pairs (noisy and ground-truth). Download from [here](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)
- `Zurich` - 22GB. Zurich RAW to RGB dataset. [here](https://docs.google.com/forms/d/e/1FAIpQLSdH6Pqdlu0pk2vGZlazqoRYwWsxN3nsLFwYY6Zc5-RUjw3SdQ/viewform)

Please put the decompressed datasets in the same directory with the code during experiments, otherwise please set **cwd** in utils.py to the file directory where the datasets locate at.

## Experiments:
#### CelebA Denoising Experiment
```bash
$ python3 experiment_celeba.py \
          --file_batch 5\
          --epoch 50\
          --latent_dim 96\
    	  --num_filter 4\
    	  --train_files_path "celeba_train(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
    	  --test_files_path "celeba_test(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"
```
The reconstruction quality of test data evaluated by PSNR, SSIM, and UQI will be printed out, and the trained model will be saved to the current working directory.

#### SIDD Denoising Experiment
```bash
$ python3 experiment_sidd_denoise.py \
          --file_batch 5\
          --epoch 25\
          --latent_dim 96\
    	  --num_filter 3\
    	  --train_files_path "sidd_noise(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
    	  --validation_file_path "sidd_ground_truth(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
    	  --test_files_path "sidd_test_noise(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
    	  --test_validation_files_path "sidd_test_GT(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"
    	  
```
The denoisng quality of the SIDD Benchmark images evaluated by PSNR, SSIM, and UQI will be printed out, and the trained model will be saved to the current working directory.

#### Zurich Raw to sRGB Experiment
```bash
$ python3 experiment_zurich.py \
          --file_batch 100\
          --epoch 50\
          --latent_dim 128\
    	  --num_filter 2\
    	  --train_files_path "Zurich_RAW(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
    	  --validation_file_path "Zurich_sRGB(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
    	  --test_files_path "Zurich_RAW_test(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
    	  --test_validation_files_path "Zurich_sRBG_test(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"
    	  
```
The reconstruction quality of test data evaluated by PSNR, and Multi-Scale SSIM will be printed out, and the trained model will be saved to the current working directory.

#### Reconstruct Full-Resolution sRGB Image
Please run *Zurich Raw to sRGB Experiment* first to obtain a saved model.
```bash
$ python3 zurich_raw_to_rgb.py \
    	  --input_files_path "Zurich_RAW_full_resolution(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
    	  --output_files_path "Zurich_sRGB_full_resolution(REPLACE THIS WITH YOUR DESIRED FILE DIRECTORY)/"
    	  
```
The reconstructed full-resolution RGB images will be stored to the directory defined by output_file_path.

