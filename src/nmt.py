#%%
# !pip install tensorflow_text
#%%
import numpy as np

import typing
from typing import Any, Tuple

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import tensorflow_text as tf_text

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#%%
use_builtins = True
#%%
# Download the file
import pathlib

path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng/spa.txt'
#%%
def load_data(path):
  text = path.read_text(encoding='utf-8')

  lines = text.splitlines()
  pairs = [line.split('\t') for line in lines]

  inp = [inp for targ, inp in pairs]
  targ = [targ for targ, inp in pairs]

  return targ, inp
#%%
targ, inp = load_data(path_to_file)
print(inp[-1])
print(targ[-1])
#%%
BUFFER_SIZE = len(inp)
BATCH_SIZE = 64

dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
#%%
for example_input_batch, example_target_batch in dataset.take(1):
  print(example_input_batch[:5])
  print()
  print(example_target_batch[:5])
  break
#%%
'''
The first step is Unicode normalization 
to split accented characters and replace compatibility characters with their ASCII equivalents.
'''
example_text = tf.constant('¿Todavía está en casa?')

print(example_text.numpy())
print(tf_text.normalize_utf8(example_text, 'NFKD').numpy())
#%%
b'\xc2\xbf'.decode('UTF-8')
b'\xc3\xad'.decode('UTF-8')
b'\xc3\xa1'.decode('UTF-8')
#%%
b'\xc2\xbf'.decode('UTF-8')
b'\xcc\x81'.decode('UTF-8')
b'a\xcc\x81'.decode('UTF-8')
#%%
'''한글 예제'''
example_text = tf.constant('안녕하세요?')

print(example_text.numpy())
print(tf_text.normalize_utf8(example_text, 'NFKD').numpy())
#%%
b'\xec\x95\x88'.decode('UTF-8')

b'\xe1\x84\x8b'.decode('UTF-8')
b'\xe1\x85\xa1'.decode('UTF-8')
b'\xe1\x86\xab'.decode('UTF-8')
b'\xe1\x84\x8b\xe1\x85\xa1\xe1\x86\xab'.decode('UTF-8')
#%%
example_text = tf.constant('#우리나라.')
example_text = tf.strings.regex_replace(example_text, '[^ 가-힣a-z.?!,¿]', '')
example_text
a = tf_text.normalize_utf8(example_text, 'NFKD')
a = example_text.numpy()
print(a.decode())
#%%