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
from PIL import Image
import os
os.chdir('/home/jeon/Desktop/an/transformer')
#%%
# image_size = 32
# embedding_dim = 128
# patch_size = 8
# channels = 3
# patch_dim = channels * patch_size ** 2
# num_patches = (image_size // patch_size) ** 2
# #%%
# images = tf.random.normal(shape=(256, 32, 32, 3))

# def extract_patches(images):
#     batch_size = tf.shape(images)[0]
#     patches = tf.image.extract_patches(
#         images=images,
#         sizes=[1, patch_size, patch_size, 1],
#         strides=[1, patch_size, patch_size, 1],
#         rates=[1, 1, 1, 1],
#         padding="VALID",
#     )
#     patches = tf.reshape(patches, [batch_size, -1, patch_dim])
#     return patches

# patches = extract_patches(images)

# patch_proj = layers.Dense(embedding_dim)
# x = patch_proj(patches)

# batch_size = tf.shape(images)[0]
# class_emb = tf.Variable(tf.random.normal(shape=(1, 1, embedding_dim)))
# class_emb = tf.broadcast_to(class_emb, [batch_size, 1, embedding_dim])

# x = tf.concat([class_emb, x], axis=1)

# pos_emb = tf.Variable(tf.random.normal(shape=(1, num_patches + 1, embedding_dim)))
# x = x + pos_emb
# #%%
# '''multi-head attention'''
# query_dense = layers.Dense(embedding_dim)
# key_dense = layers.Dense(embedding_dim)
# value_dense = layers.Dense(embedding_dim)
# combine_heads = layers.Dense(embedding_dim)
        
# query = query_dense(x)
# key = key_dense(x)
# value = value_dense(x)

# num_heads = 8
# projection_dim = embedding_dim // num_heads

# def separate_heads(x, batch_size):
#     x = tf.reshape(x, (batch_size, -1, num_heads, projection_dim))
#     return tf.transpose(x, perm=[0, 2, 1, 3])

# query = separate_heads(query, batch_size)
# key = separate_heads(key, batch_size)
# value = separate_heads(value, batch_size)

# score = tf.matmul(query, key, transpose_b=True)
# scaled_score = score / tf.math.sqrt(tf.cast(projection_dim, tf.float32))
# weights = tf.nn.softmax(scaled_score, axis=-1)
# attention = tf.matmul(weights, value)

# attention = tf.transpose(attention, perm=[0, 2, 1, 3])
# concat_attention = tf.reshape(attention, (batch_size, -1, embedding_dim))
# output = combine_heads(concat_attention)
#%%
class MultiHeadSelfAttention(K.layers.Layer):
    def __init__(self, embedding_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embedding_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embedding_dim // num_heads
        self.query_dense = layers.Dense(embedding_dim)
        self.key_dense = layers.Dense(embedding_dim)
        self.value_dense = layers.Dense(embedding_dim)
        self.combine_heads = layers.Dense(embedding_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        scaled_score = score / tf.math.sqrt(tf.cast(self.projection_dim, tf.float32))
        weights = tf.nn.softmax(scaled_score, axis=-1)
        attention = tf.matmul(weights, value)
        return attention, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embedding_dim))
        output = self.combine_heads(concat_attention)
        return output
#%%
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.mlp = tf.keras.Sequential(
            [
                layers.Dense(mlp_dim, activation=K.activations.gelu),
                layers.Dropout(dropout),
                layers.Dense(embedding_dim),
                layers.Dropout(dropout),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1
#%%
class VisionTransformer(K.Model):
    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_classes,
        embedding_dim,
        num_heads,
        mlp_dim,
        channels=3,
        dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.pos_emb = self.add_weight("pos_emb", shape=(1, num_patches + 1, self.embedding_dim))
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, self.embedding_dim))
        
        self.patch_proj = layers.Dense(self.embedding_dim)
        self.enc_layers = [
            TransformerBlock(self.embedding_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        self.mlp_head = K.Sequential(
            [
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(mlp_dim, activation=K.activations.gelu),
                layers.Dropout(dropout),
                layers.Dense(num_classes, activation='softmax'),
            ]
        )

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        patches = self.extract_patches(x)
        x = self.patch_proj(patches)

        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.embedding_dim])
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x, training)

        # First (class token) is used for classification
        x = self.mlp_head(x[:, 0])
        return x
#%%
PARAMS = {
    "data": 'cifar10',
    "batch_size": 512,
    "learning_rate": 0.01,
    "epochs": 10000,
    "image_size": 32,
    "patch_size": 8,
    "num_layers": 8,
    "num_classes": 10,
    "embedding_dim": 256,
    "num_heads": 8,
    "mlp_dim": 512,
    "channels": 3,
    "dropout": 0.1,
    "ema": True,
}
#%%
# data
(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5

from tensorflow.keras.utils import to_categorical
y_train_onehot = to_categorical(y_train, num_classes=PARAMS['num_classes'])
y_test_onehot = to_categorical(y_test, num_classes=PARAMS['num_classes'])

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_onehot)).shuffle(len(x_train), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(PARAMS['batch_size'])
#%%
model = VisionTransformer(PARAMS['image_size'],
                        PARAMS['patch_size'],
                        PARAMS['num_layers'],
                        PARAMS['num_classes'],
                        PARAMS['embedding_dim'],
                        PARAMS['num_heads'],
                        PARAMS['mlp_dim'],
                        PARAMS['channels'],
                        PARAMS['dropout'])
optimizer = K.optimizers.Adam(PARAMS["learning_rate"])
if PARAMS['ema']:
    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
#%%
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        pred = model(x_batch, training=True)
        loss = - tf.reduce_mean(tf.reduce_sum(tf.multiply(y_batch, tf.math.log(pred + 1e-8)), axis=-1))
        
    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    ema.apply(model.trainable_weights)
    return loss
#%%
'''training'''
step = 0
progress_bar = tqdm(range(PARAMS['epochs']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, PARAMS['epochs']))

error = 1.
for _ in progress_bar:
    x_batch, y_batch = next(iter(train_dataset))
    
    loss = train_step(x_batch, y_batch)
    
    progress_bar.set_description('iteration {}/{} | loss {:.3f} error {:.3f}'.format(
        step, PARAMS['epochs'], 
        loss.numpy(), error)) 
    
    step += 1
    
    if step % 500 == 0:
        '''test classification error''' 
        error_count = 0
        for x_batch, y_batch in test_dataset:
            prob = model(x_batch, training=False)
            error_count += len(np.where(np.squeeze(y_batch) - np.argmax(prob.numpy(), axis=-1) != 0)[0])
            error = error_count / len(x_test)
        
    if step == PARAMS['epochs']: break
#%%
model.summary()
asset_path = 'weights'
model.save_weights('./assets/{}/{}/weights'.format(PARAMS['data'], asset_path))
#%%
'''test classification error''' 
error_count = 0
for x_batch, y_batch in test_dataset:
    prob = model(x_batch, training=False)
    error_count += len(np.where(np.squeeze(y_batch) - np.argmax(prob.numpy(), axis=-1) != 0)[0])
print(error_count / len(y_test))
#%%