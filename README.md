# autoencoder-clustering

## Requirements:
See requirement.txt\
Run
`pip install -r requirement.txt` \
GPU is required

## Dataset:
Run\
`wget https://storage.googleapis.com/glow-demo/data/celeba-tfr.tar` \
`tar -xvf celeb-tfr.tar` \
Please put the dataset in the same directory with the code

## Python Files:
main.py: Patch Pased Gaussian Mixture Variational Autoencoder
```bash
$ python3 main.py \
    	--batch 10000\
    	--epoch 100\
    	--num_cluster 4\
```

autoencoder.py + ae.py: Convolutional Autoencoder with soft clustering\
Run `python ae.py` \

vae.py: Variational Autoencoder with soft clustering\
Run `python vae.py` \

## Report
See ae_clustering_report.pdf
