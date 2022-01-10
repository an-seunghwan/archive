#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
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
from datetime import datetime
import sys
from PIL import Image
import os
os.chdir(r'D:\archive')
#%%
PARAMS = {
    "data": 'mnist',
    "batch_size": 256,
    "data_dim": 784,
    "class_num": 10,
    "latent_dim": 2,
    "activation": 'sigmoid',
    "epochs": 30, 
    "learning_rate": 0.001,
}
#%%
class Encoder(K.models.Model):
    def __init__(self, params, name="Encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.enc_dense1 = layers.Dense(512, activation='relu')
        self.enc_dense2 = layers.Dense(256, activation='relu')
        
        self.mean_layer = layers.Dense(self.params['latent_dim'], activation='linear')
        self.logsigma_layer = layers.Dense(self.params['latent_dim'], activation='linear')
        self.prob_layer = layers.Dense(self.params['class_num'], activation='linear')
    
    @tf.function
    def call(self, x):
        h = self.enc_dense1(x)
        h = self.enc_dense2(h)
        
        mean = self.mean_layer(h)
        logvar = self.logsigma_layer(h)
        prob = self.prob_layer(h)
        return mean, logvar, prob
#%%
class Decoder(K.models.Model):
    def __init__(self, params, name="Decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.dec_dense1 = layers.Dense(256, activation='relu')
        self.dec_dense2 = layers.Dense(512, activation='relu')
        self.dec_dense3 = layers.Dense(self.params["data_dim"], activation=self.params['activation'])
    
    @tf.function
    def call(self, z, prob):
        h = tf.concat([z, prob], axis=-1)
        h = self.dec_dense1(h)
        h = self.dec_dense2(h)
        h = self.dec_dense3(h)
        return h
#%%
class VAE(K.models.Model):
    def __init__(self, params):
        super(VAE, self).__init__()
        self.params = params
        self.encoder = Encoder(self.params)
        self.decoder = Decoder(self.params)
        self.temperature = 0.67
    
    def sample_gumbel(self, shape): 
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + 1e-8) + 1e-8)

    def gumbel_max_sample(self, log_prob): 
        y = log_prob + self.sample_gumbel(tf.shape(log_prob))
        y = tf.nn.softmax(y / self.temperature)
        return y
    
    @tf.function
    def call(self, x):
        mean, log_sigma, prob = self.encoder(x)
        
        epsilon = tf.random.normal((tf.shape(x)[0], self.params["latent_dim"]))
        z = mean + tf.math.exp(log_sigma) * epsilon 
        
        log_prob = tf.nn.log_softmax(prob)
        y = self.gumbel_max_sample(log_prob)
        
        xhat = self.decoder(z, y) 
        return mean, log_sigma, z, log_prob, xhat
#%%
# data
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
y_train = to_categorical(y_train, num_classes = PARAMS['class_num'])
y_test = to_categorical(y_test, num_classes = PARAMS['class_num'])

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#%%
def loss_function(xhat, x, mean, log_sigma, log_prob, PARAMS):
    # reconstruction error
    # error = tf.reduce_mean(tf.reduce_sum(tf.math.square(x - xhat), axis=-1) / 2)
    error = tf.reduce_mean(tf.reduce_sum(x * tf.math.log(tf.clip_by_value(xhat, 1e-10, 1.)) + 
                                        (1. - x) * tf.math.log(tf.clip_by_value(1. - xhat, 1e-10, 1.)), axis=-1))
    
    # KL divergence by closed form
    # kl = - 0.5 * (1 + logvar - tf.square(mean) - tf.math.exp(logvar))
    kl_z = tf.reduce_mean(0.5 * tf.reduce_sum(tf.math.square(mean) + tf.math.exp(2. * log_sigma) - (2. * log_sigma) - 1, axis=-1))
    kl_y = tf.reduce_mean(tf.reduce_sum(tf.math.exp(log_prob) * (log_prob - tf.math.log(1. / PARAMS['class_num'])), axis=1))
    return error, kl_z, kl_y
#%%
def train_step(x_batch, y_batch, PARAMS, dmi):
    with tf.GradientTape() as tape:
        mean, log_sigma, z, log_prob, xhat = model(x_batch)
        error, kl_z, kl_y = loss_function(xhat, x_batch, mean, log_sigma, log_prob, PARAMS) 
        classification = - tf.reduce_mean(tf.reduce_sum(y_batch * log_prob, axis=-1))
        loss = error * 0.01 + kl_z + tf.math.abs(kl_y - dmi) + classification
    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    
    kl_z_avg(kl_z)
    kl_y_avg(kl_y)
    recon_avg(error)
    loss_avg(error + kl_z + kl_y)
    accuracy(tf.argmax(log_prob, axis=1, output_type=tf.int32), 
            tf.argmax(y_batch, axis=1, output_type=tf.int32))
    
    return loss, error, kl_z, kl_y, classification, log_prob, xhat
#%%
'''training'''
dmi = 0

train_writer = tf.summary.create_file_writer(r'D:\archive\logs\dmi{}\train'.format(dmi))
test_writer = tf.summary.create_file_writer(r'D:\archive\logs\dmi{}\test'.format(dmi))

loss_avg = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
recon_avg = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
kl_z_avg = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
kl_y_avg = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
accuracy = tf.keras.metrics.Accuracy('train_accuracy')

test_loss_avg = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_recon_avg = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_kl_z_avg = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_kl_y_avg = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.Accuracy('test_accuracy')

model = VAE(PARAMS)
optimizer = K.optimizers.Adam(PARAMS["learning_rate"])

shuffle_and_batch = lambda dataset: dataset.shuffle(len(x_train), reshuffle_each_iteration=True).batch(PARAMS['batch_size'], drop_remainder=True)
iterator_ = iter(shuffle_and_batch(train_dataset))

dmi = tf.convert_to_tensor(dmi, dtype=tf.float32)

step = 0
total_length = sum(1 for _ in train_dataset)

for epoch in range(PARAMS['epochs']):
    progress_bar = tqdm(range(total_length // PARAMS['batch_size']))
    progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, total_length))
    
    for _ in progress_bar:
        try:
            x_batch, y_batch = next(iterator_)
        except:
            iterator_ = iter(shuffle_and_batch(train_dataset))
            imageL, labelL = next(iterator_)
        
        loss, error, kl_z, kl_y, classification, log_prob, xhat = train_step(x_batch, y_batch, PARAMS, dmi)
        
        progress_bar.set_description('loss {:.3f}, recon {:.3f}, kl_z {:.3f}, kl_y {:.3f}, cls {:.3f}'.format(
            loss.numpy(), error.numpy(), kl_z.numpy(), kl_y.numpy(), classification.numpy())) 
        
    template = 'TEST: epochs {}/{} | loss {:.3f}, recon {:.3f}, kl_z {:.3f}, kl_y {:.3f}, cls {:.3f}'
    print(template.format(
        epoch, PARAMS['epochs'], 
        loss_avg.result(), recon_avg.result(), kl_z_avg.result(), kl_y_avg.result(), accuracy.result() * 100))
    
    mean, log_sigma, z, log_prob, xhat = model(x_test)
    error, kl_z, kl_y = loss_function(xhat, x_test, mean, log_sigma, log_prob, PARAMS) 
    test_kl_z_avg(kl_z)
    test_kl_y_avg(kl_y)
    test_recon_avg(error)
    test_loss_avg(error * 0.01 + kl_z + kl_y)
    test_accuracy(tf.argmax(log_prob, axis=1, output_type=tf.int32), 
                    tf.argmax(y_test, axis=1, output_type=tf.int32))
    
    with train_writer.as_default():
        tf.summary.scalar('loss', loss_avg.result(), step=epoch)
        tf.summary.scalar('recon', recon_avg.result(), step=epoch)
        tf.summary.scalar('KL(z)', kl_z_avg.result(), step=epoch)
        tf.summary.scalar('KL(y)', kl_y_avg.result(), step=epoch)
        tf.summary.scalar('accuracy', accuracy.result(), step=epoch)
    
    with test_writer.as_default():
        tf.summary.scalar('loss', test_loss_avg.result(), step=epoch)
        tf.summary.scalar('recon', test_recon_avg.result(), step=epoch)
        tf.summary.scalar('KL(z)', test_kl_z_avg.result(), step=epoch)
        tf.summary.scalar('KL(y)', test_kl_y_avg.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
        
    # Reset metrics every epoch
    loss_avg.reset_states()
    recon_avg.reset_states()
    kl_z_avg.reset_states()
    kl_y_avg.reset_states()
    accuracy.reset_states()
    
    test_loss_avg.reset_states()
    test_recon_avg.reset_states()
    test_kl_z_avg.reset_states()
    test_kl_y_avg.reset_states()
    test_accuracy.reset_states()
#%%