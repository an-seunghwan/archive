#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
# tf.debugging.set_log_device_placement(False)
#%%
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
# from pprint import pprint
from datetime import datetime
# import sys
# from PIL import Image
import time
# import os
#%%
'''dummy data'''
n = 10000
data_dim = 100
activation_true = 40
true = np.ones((1, data_dim))
true[0, :activation_true] = 0.

np.random.seed(10)
activation_index = [round(x) for x in np.random.normal(loc=activation_true, scale=7, size=(n, ))]
plt.hist(activation_index)
#%%
data = np.ones((n, data_dim))
for i in range(n):
    data[i, :activation_index[i]] = 0.

plt.plot(true[0])
plt.plot(np.mean(data, axis=0)) # MLE
#%%
'''MNIST'''
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
x_train = x_train / 255.
x_test = x_test / 255.
x_train = x_train[..., None]
x_test = x_test[..., None]

'''binarize'''
f = lambda x: 1 if x > 0.5 else 0
x_train = np.vectorize(f)(x_train)
assert len(np.unique(x_train)) == 2

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
#%%
def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]
    return x_var, y_var

def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(tf.reduce_sum(tf.abs(x_deltas), axis=[1, 2, 3]) + tf.reduce_sum(tf.abs(y_deltas), axis=[1, 2, 3]))
#%%
PARAMS = {
    "data": 'mnist',
    "batch_size": 256,
    "data_dim": 28,
    "channel": 1,
    "class_num": 10,
    "latent_dim": 128,
    "sigma": 1., 
    "activation": 'sigmoid',
    "iterations": 1000, 
    "learning_rate": 0.01,
    "hard": True,
    "observation": 'mse',
    "epsilon": 0.1,
}
#%%
total_variation = []
#%%
for obs in ['mse', 'taylor', 'bce']:
    
    PARAMS['observation'] = obs
    
    '''VAE'''
    class Sampling(layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    #%%
    encoder_inputs = K.layers.Input(shape=(PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(PARAMS['latent_dim'], name="z_mean")(x)
    z_log_var = layers.Dense(PARAMS['latent_dim'], name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = K.models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    #%%
    latent_inputs = K.layers.Input(shape=(PARAMS['latent_dim'],))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation=PARAMS['activation'], padding="same")(x)
    decoder = K.models.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    # latent_inputs = K.Input(shape=(PARAMS['latent_dim'], ))
    # x = layers.Dense(7 * 7 * 256, use_bias=False)(latent_inputs)
    # x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU()(x)
    # x = layers.Reshape((7, 7, 256))(x)

    # x = layers.Conv2DTranspose(128, 5, activation="relu", strides=1, padding="same", use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU()(x)

    # x = layers.Conv2DTranspose(64, 5, activation="relu", strides=2, padding="same", use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU()(x)

    # decoder_outputs = layers.Conv2DTranspose(1, 5, activation=PARAMS['activation'], strides=2, padding="same", use_bias=False)(x)
    # decoder = K.models.Model(latent_inputs, decoder_outputs, name="decoder")
    # decoder.summary()
    #%%
    class VAE(K.models.Model):
        def __init__(self, params, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.params = params
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = K.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = K.metrics.Mean(
                name="reconstruction_loss"
            )
            self.kl_loss_tracker = K.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                if self.params['observation'] == 'bce':
                    reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            K.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                        )
                    )
                elif self.params['observation'] == 'mse':
                    reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            0.5 * tf.square(data - reconstruction), axis=(1, 2, 3)
                        )
                    )
                elif self.params['observation'] == 'taylor':
                    reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            (1 / (data * (1 - data) + self.params['epsilon'])) * tf.square(data - reconstruction), axis=(1, 2, 3)
                        )
                    )
                else:
                    assert 0, 'Unsupported observation model: {}!'.format(self.params['observation'])
                    
                kl_loss = - 0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                total_loss = reconstruction_loss + kl_loss
                
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }
    #%%
    vae = VAE(PARAMS, encoder, decoder)
    vae.compile(optimizer=K.optimizers.Adam(learning_rate=0.001))
    vae.fit(x_train, epochs=30, batch_size=PARAMS['batch_size'])
    #%%
    np.random.seed(1)
    recon = vae.decoder(np.random.normal(size=(100, PARAMS['latent_dim'])))
    plt.imshow(recon.numpy()[0, ..., 0])
    plt.savefig('./recon_vae_{}.png'.format(obs))
    plt.show()
    plt.close()
    #%%
    plt.hist(recon.numpy().reshape(-1), density=True)
    plt.savefig('./density_vae_{}.png'.format(obs))
    plt.show()
    plt.close()
    #%%
    total_variation.append(total_variation_loss(recon).numpy())
    print(total_variation_loss(recon).numpy())
#%%
'''GAN'''
def make_generator_model():
    model = K.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(PARAMS['latent_dim'], )))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) 

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation=PARAMS['activation']))
    assert model.output_shape == (None, 28, 28, 1)

    return model
#%%
def make_discriminator_model():
    model = K.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
#%%
cross_entropy = K.losses.BinaryCrossentropy(from_logits=True)
#%%
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
#%%
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
#%%
generator = make_generator_model()
discriminator = make_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
#%%
@tf.function
def train_step(images):
    noise = tf.random.normal([PARAMS['batch_size'], PARAMS['latent_dim']])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
#%%
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
#%%
train_dataset = tf.data.Dataset.from_tensor_slices((x_train)).shuffle(len(x_train), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
train(train_dataset, epochs=30)
#%%
np.random.seed(1)
recon = generator(np.random.normal(size=(100, PARAMS['latent_dim'])))
plt.imshow(recon.numpy()[0, ..., 0])
plt.savefig('./recon_gan.png')
plt.show()
plt.close()
#%%
plt.hist(recon.numpy().reshape(-1), density=True)
plt.savefig('./density_gan.png')
plt.show()
plt.close()
#%%
total_variation.append(total_variation_loss(recon).numpy())
print(total_variation_loss(recon).numpy())
#%%
print(total_variation)
# [37.417057, 56.11975, 68.309326, 96.69542]
#%%
# (50000 - 5000) / 128
#%%