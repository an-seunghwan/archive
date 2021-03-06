"""
Title: Variational AutoEncoder
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/05/03
Last modified: 2020/05/03
Description: Convolutional Variational AutoEncoder (VAE) trained on MNIST digits.
"""
#%%
"""
## Setup
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image

#%%
"""
## Create a sampling layer
"""

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
#%%
"""
## Build the encoder
"""

latent_dim = 2

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()
#%%
"""
## Build the decoder
"""

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()
#%%
"""
## Define the VAE as a `Model` with a custom `train_step`
"""

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

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
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
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
"""
## Train the VAE
"""

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

#%%
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=30, batch_size=128)
#%%
"""
## Display a grid of sampled digits
"""

def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


plot_latent_space(vae)
#%%
"""
## Display how the latent space clusters different digit classes
"""

def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(10, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap=plt.cm.Reds)
    # plt.colorbar()
    plt.xlabel("$z_0$", fontsize=25)
    plt.ylabel("$z_1$", fontsize=25)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.show()
#%%
(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255
#%%
plot_label_clusters(vae, x_train, y_train)
#%%
vae.save_weights(r'D:\archive\assets\rebuttal_vae_model')
#%%
vae1 = VAE(encoder, decoder)
vae1.load_weights(r'D:\archive\assets\rebuttal_vae_model')
#%%
z_mean, _, _ = vae1.encoder.predict(x_train)
#%%
i = 500
j = 260
z_inter = (z_mean[np.where(y_train == 0)[0][[i]][0], :], z_mean[np.where(y_train == 1)[0][[j]][0], :])

plt.figure(figsize=(10, 10))
plt.tick_params(labelsize=30)    
plt.locator_params(axis='y', nbins=8)
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_train, s=10, cmap=plt.cm.Reds, alpha=1)
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.scatter(z_inter[0][0], z_inter[0][1], color='blue', s=100)
plt.annotate('A', (z_inter[0][0], z_inter[0][1]), fontsize=30)
plt.scatter(z_inter[1][0], z_inter[1][1], color='blue', s=100)
plt.annotate('B', (z_inter[1][0], z_inter[1][1]), fontsize=30)
plt.plot((z_inter[0][0], z_inter[1][0]), (z_inter[0][1], z_inter[1][1]), color='black', linewidth=2, linestyle='--')
plt.xlabel("$z_0$", fontsize=30)
plt.ylabel("$z_1$", fontsize=30)
# plt.xticks(fontsize = 25)
# plt.yticks(fontsize = 25)
plt.savefig(r'D:\archive\assets\naive_vae_interpolation_path.png', 
            dpi=100, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
inter = np.linspace(z_inter[0], z_inter[1], 10)
inter_recon = vae1.decoder.predict(inter)

figure = plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10+1, i+1)
    plt.imshow(inter_recon[i], cmap='gray_r')
    plt.axis('off')
plt.savefig(r'D:\archive\assets\naive_vae_interpolation_path_recon.png', 
            dpi=100, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
img = [Image.open(r'D:\archive\assets\naive_vae_interpolation_path.png'),
        Image.open(r'D:\archive\assets\naive_vae_interpolation_path_recon.png')]

f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 0.25]})
a0.imshow(img[0])    
a0.axis('off')
a1.imshow(img[1])    
a1.axis('off')
plt.tight_layout() 
plt.savefig(r'D:\archive\assets\naive_vae_interpolation_path_and_recon.png',
            dpi=100, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%