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
autoencoder.py + main.py: Convolutional Autoencoder with soft clustering\
Run `python main.py` \

vae.py: Variational Autoencoder with soft clustering\
Run `python vae.py` \

autoencoder&hard-clustering.py: Variational Autoencoder with hard clustering\
Run `python autoencoder&hard-clustering.py` \

## Report
See ae_clustering_report.pdf
