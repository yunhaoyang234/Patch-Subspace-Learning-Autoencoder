from model import *
from utils import *

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)

def train_encoder(noise_images, clear_images, latent_dim, shape, num_cluster, epoch):
    encoder = build_encoder(latent_dim, shape, num_cluster)
    decoder = build_decoder(latent_dim, shape, 'decoder')
    model = VAE(encoder, decoder)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
    model.fit((noise_images,clear_images), epochs=epoch, batch_size=128)
    return encoder, decoder

def cluster_latent(y, batch):
    labels = []
    for i in range(0, len(y), batch):
        y_ = np.array(y[i:i+batch])
        for j in y_:
            labels.append(np.argmax(j))
    return labels

def gen_clusters(imgs, labels, num_cluster):
    clusters = []
    for label in range(num_cluster):
        clusters.append([])
    for i in range(len(labels)):
        clusters[labels[i]].append(imgs[i])
    for i in range(num_cluster):
        clusters[i] = np.array(clusters[i])
    return clusters

def clustering(blur_images, clear_images, encoder, num_cluster, batch):
    z, z_mean, z_sig, y, y_logits, z_prior_mean, z_prior_sig = encoder(blur_images[:batch])
    for i in range(batch, len(blur_images), batch):
        new_z, m, sig, y, log, pm, ps = encoder(blur_images[i: i+batch])
        z = np.concatenate([z, new_z], axis=0)
        y_logits = np.concatenate([y_logits, log], axis=0)
    labels = np.array(cluster_latent(y_logits, batch))
    clus = gen_clusters(blur_images, labels, num_cluster)
    label_clus = gen_clusters(clear_images, labels, num_cluster)
    return clus, label_clus

def train_decoders(clus, label_clus, encoder, latent_dim, shape, epochs):
    decoders = []
    for i in range(len(clus)):
        decoder_i = build_decoder(latent_dim, shape,"decoder"+str(i))
        if len(clus[i]) > 0:
            model_i = VAE_P(encoder, decoder_i)
            model_i.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
            model_i.fit((clus[i],label_clus[i]), epochs=epochs, batch_size=128, verbose=0)
        decoders.append(decoder_i)
    return decoders

