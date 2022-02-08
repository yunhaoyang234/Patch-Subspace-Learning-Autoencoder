from model import *
from utils import *

def train_encoder(noise_images, clear_images, num_cluster, epoch):
    pretraining_model = ContrastiveModel()
    pretraining_model.compile(
        contrastive_optimizer=keras.optimizers.Adam(),
    )

    pretraining_history = pretraining_model.fit(
        (clear_images, noise_images), epochs=epoch, batch_size=128
    )
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )
    model = My_Model(pretraining_model.encoder, num_cluster)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
    model.fit((noise_images,clear_images), epochs=epoch, batch_size=128)
    return model

def train_decoders(clus, label_clus, encoder, epochs, default_weights):
    decoders = [build_decoder(SHAPE) for i in range(len(clus))]
    for i in range(len(clus)):
        decoders[i].set_weights(default_weights)
        if len(clus[i]) > 0:
            lr = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.0005,
                decay_steps=1000,
                decay_rate=0.9
            )
            model_i = Model_P(encoder, decoders[i])
            model_i.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
            model_i.fit((clus[i],label_clus[i]), epochs=epochs, batch_size=128, verbose=1)
    return decoders

