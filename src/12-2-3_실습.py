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
tf.debugging.set_log_device_placement(False)
#%%
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
#%%
'''beta-VAE'''
PARAMS = {
    "data": 'mnist',
    "batch_size": 256,
    "data_dim": 784,
    "latent_dim": 2,
    "activation": 'tanh',
    "iterations": 1000, 
    "learning_rate": 0.001,
    "beta": 1.,
}
#%%
class Encoder(K.models.Model):
    def __init__(self, params, name="Encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.enc_dense1 = layers.Dense(512, activation='relu')
        self.enc_dense2 = layers.Dense(256, activation='relu')
        
        self.mean_layer = layers.Dense(self.params['latent_dim'], activation='linear')
        self.logvar_layer = layers.Dense(self.params['latent_dim'], activation='linear')
    
    @tf.function
    def call(self, x):
        h = self.enc_dense1(x)
        h = self.enc_dense2(h)
        
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar
#%%
class Decoder(K.models.Model):
    def __init__(self, params, name="Decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.dec_dense1 = layers.Dense(256, activation='relu')
        self.dec_dense2 = layers.Dense(512, activation='relu')
        self.dec_dense3 = layers.Dense(self.params["data_dim"], activation=self.params['activation'])
    
    @tf.function
    def call(self, x):
        h = self.dec_dense1(x)
        h = self.dec_dense2(h)
        h = self.dec_dense3(h)
        return h
#%%
class VAE(K.models.Model):
    def __init__(self, params):
        super(VAE, self).__init__()
        self.params = params
        self.Encoder = Encoder(self.params)
        self.Decoder = Decoder(self.params)
    
    @tf.function
    def call(self, x):
        # Encoding (잠재변수의 평균과 분산 벡터)
        mean, logvar = self.Encoder(x)
        # 잠재변수 sampling
        epsilon = tf.random.normal((tf.shape(x)[0], self.params["latent_dim"]))
        z = mean + tf.math.exp(logvar / 2.) * epsilon 
        # Decoding (복원된 이미지의 평균 벡터)
        xhat = self.Decoder(z) 
        return mean, logvar, z, xhat
#%%
'''MNIST'''
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train)).shuffle(len(x_train), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
#%%
model = VAE(PARAMS)
# optimizer = K.optimizers.Adam(PARAMS["learning_rate"])
#%%
@tf.function
def loss_function(xhat, x, mean, logvar, PARAMS):
    # 데이터의 복원 정도를 판단하는 오차항 (reconstruction)
    error = tf.reduce_mean(tf.reduce_sum(tf.math.square(x - xhat) / 2., axis=-1))
    # encoder와 prior 사이의 KL divergence
    kl = - 0.5 * (1. + logvar - tf.square(mean) - tf.math.exp(logvar))
    kl = tf.reduce_mean(tf.reduce_sum(kl, axis=1))
    return error / PARAMS['beta'] + kl, error, kl
#%%
optimizer = K.optimizers.Adam(PARAMS["learning_rate"])
@tf.function
def train_step(x_batch, PARAMS):
    with tf.GradientTape() as tape:
        mean, logvar, z, xhat = model(x_batch)
        loss, error, kl = loss_function(xhat, x_batch, mean, logvar, PARAMS) 
    # 1차 미분 계산
    grad = tape.gradient(loss, model.trainable_weights)
    # 1차 미분을 이용해 모형의 가중치 update
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    return loss, error, kl, xhat
#%%
'''training'''
step = 0
progress_bar = tqdm(range(PARAMS['iterations']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, PARAMS['iterations']))

for _ in progress_bar:
    x_batch = next(iter(train_dataset))
    
    loss, error, kl, xhat = train_step(x_batch, PARAMS)
    
    progress_bar.set_description('iteration {}/{} | loss {:.3f}, recon {:.3f}, kl {:.3f}'.format(
        step, PARAMS['iterations'], 
        loss.numpy(), error.numpy(), kl.numpy())) 
    
    step += 1
    
    if step == PARAMS['iterations']: break
#%%
n = 20
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

from scipy.stats import norm
grid_x = 1.5 * norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = 1.5 * norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = model.Decoder(tf.cast(np.array(z_sample), tf.float32))
        digit = x_decoded[0].numpy().reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.axis('off')
plt.show()
#%%