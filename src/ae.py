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
from pprint import pprint
#%%
PARAMS = {
    "data": 'mnist',
    "batch_size": 512,
    "data_dim": 784,
    "class_num": 10,
    "latent_dim": 2,
    "sigma": 1., 
    "activation": 'tanh',
    "iterations": 2000, 
    "learning_rate": 0.001,
}
#%%
class Encoder(K.models.Model):
    def __init__(self, params, name="Encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.enc_dense1 = layers.Dense(512, activation='relu')
        self.enc_dense2 = layers.Dense(256, activation='relu')
        self.enc_dense3 = layers.Dense(self.params['latent_dim'], activation='linear')
        
    def call(self, x):
        h = self.enc_dense1(x)
        h = self.enc_dense2(h)
        h = self.enc_dense3(h)
        return h
#%%
class Decoder(K.models.Model):
    def __init__(self, params, name="Decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.dec_dense1 = layers.Dense(256, activation='relu')
        self.dec_dense2 = layers.Dense(512, activation='relu')
        self.dec_dense3 = layers.Dense(self.params["data_dim"], activation=self.params['activation'])
        
    def call(self, x):
        h = self.dec_dense1(x)
        h = self.dec_dense2(h)
        h = self.dec_dense3(h)
        return h
#%%
class AutoEncoder(K.models.Model):
    def __init__(self, params):
        super(AutoEncoder, self).__init__()
        self.params = params
        
        self.Encoder = Encoder(self.params)
        self.Decoder = Decoder(self.params)
        
    def call(self, x):
        z = self.Encoder(x)
        xhat = self.Decoder(z) 
        return z, xhat
#%%
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train)).shuffle(len(x_train), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(PARAMS['batch_size'])
#%%
model = AutoEncoder(PARAMS)
optimizer = K.optimizers.Adam(PARAMS["learning_rate"])
#%%
@tf.function
def train_step(x_batch):
    with tf.GradientTape() as tape:
        z, xhat = model(x_batch)
        loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(x_batch - xhat), axis=-1) / 2)
    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    return loss
#%%
'''training'''
step = 0
progress_bar = tqdm(range(PARAMS['iterations']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, PARAMS['iterations']))

for _ in progress_bar:
    x_batch = next(iter(train_dataset))
    
    loss = train_step(x_batch)
    
    progress_bar.set_description('iteration {}/{} | loss {:.3f}'.format(
        step, PARAMS['iterations'], loss.numpy()))
    
    step += 1
    
    if step == PARAMS['iterations']: break
#%%
z = model.Encoder(x_test)
#%%
plt.figure(figsize=(10, 10))
plt.scatter(z.numpy()[:, 0], z.numpy()[:, 1], c=y_test, cmap=plt.cm.Reds)
plt.show()
#%%
# idx1 = np.argmax(z.numpy()[:, 0])
# idx2 = (z.numpy()[:, 0] > 40) & (z.numpy()[:, 1] > 20) & (z.numpy()[:, 1] < 23)
# idx2 = np.where(idx2)[0][0]
# plt.scatter(z.numpy()[[idx1, idx2], 0], z.numpy()[[idx1, idx2], 1])
# plt.show()
# #%%
# z_interpolation = np.linspace(z.numpy()[idx1, :], z.numpy()[idx2, :], 10)
# plt.figure(figsize=(10, 3))
# for i in range(10):
#     plt.subplot(1, 10, i+1)
#     xhat = model.Decoder(z_interpolation[[i], :]).numpy()
#     xhat = np.reshape(xhat, (28, 28))
#     plt.imshow(xhat, 'gray_r')
#     plt.axis('off')
#%%
idx1 = 0
idx2 = 1
plt.scatter(z.numpy()[[idx1, idx2], 0], z.numpy()[[idx1, idx2], 1])
plt.show()
#%%
z_interpolation = np.linspace(z.numpy()[idx1, :], z.numpy()[idx2, :], 10)
plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    xhat = model.Decoder(z_interpolation[[i], :]).numpy()
    xhat = np.reshape(xhat, (28, 28))
    plt.imshow(xhat, 'gray_r')
    plt.axis('off')
#%%