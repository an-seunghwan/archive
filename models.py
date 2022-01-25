#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
import math
import time
import re
#%%
class KingmaM2(K.models.Model):
    def __init__(self, params):
        super(KingmaM2, self).__init__()
        self.params = params
        
        self.enc_dense1 = layers.Dense(500, activation='softplus')
        
        self.mean_layer = layers.Dense(self.params['latent_dim'], activation='linear')
        self.logvar_layer = layers.Dense(self.params['latent_dim'], activation='linear') 
        
        self.logits_feature = layers.Dense(500, activation='softplus')
        self.logits = layers.Dense(self.params["class_num"], activation='softmax') 

        self.dec_dense1 = layers.Dense(500, activation='softplus')
        self.dec_dense2 = layers.Dense(self.params["data_dim"], activation='sigmoid')
        
    def decoder(self, z, y):
        zy = tf.concat((z, y), axis=-1)
        h = self.dec_dense1(zy)
        h = tf.nn.leaky_relu(h, alpha=0.1)
        h = self.dec_dense2(h)
        return h

    def call(self, x, y):
        latent_dim = self.params["latent_dim"]   
        data_dim = self.params['data_dim']

        xy = tf.concat((x, y), axis=-1)
        h = self.enc_dense1(xy)
        
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        epsilon = tf.random.normal((tf.shape(x)[0], latent_dim))
        z = mean + tf.math.exp(logvar / 2) * epsilon 
        
        logits_ = self.logits_feature(x)
        logits = self.logits(logits_)
        
        xhat = self.decoder(z, y) 
        
        return mean, logvar, logits, z, xhat
#%%