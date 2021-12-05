#%%
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import re
from tqdm import tqdm
import numpy as np
#%%
data = pd.read_csv('/Users/anseunghwan/Documents/uos/tohim100.csv')
datalist=[i for i in data['content']]#일단 리스트로 가져와보기
datalist=datalist[:20]#일단 100개만 돌려보za
#%%
def clean_korean(sent):
    if type(sent) == str:
        h = re.compile('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]+')
        result = h.sub(' ', sent)
        result=result.replace('\n','').strip()
    else:
        result = ''
    return result
# i=0

datalist_clean=[clean_korean(i) for i in datalist]
#%%
'''
OKT 가 안깔려 있어서 띄어쓰기로만 정제하고 해볼게
'''
preprocessed_text = datalist_clean
#%%
from tensorflow.keras import preprocessing
tokenizer=preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(preprocessed_text)
sequences=tokenizer.texts_to_sequences(preprocessed_text)
vocab=tokenizer.word_index

preprocessed_text[0]
vocab_reverse = {i:x for x,i in vocab.items()}
vocab_reverse[364]

vocab['pad'] = 0
vocab_size = len(vocab)
#%%
# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for vocab_size tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in dataset.
  for sequence in tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels
#%%
window_size=3
num_ns=3

targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=window_size,
    num_ns=num_ns,
    vocab_size=vocab_size,
    seed=62)
#%%
# #padding
# MAX_PAD_LENGTH=max([len(x) for x in sequences])
# inputs=preprocessing.sequence.pad_sequences(sequences,
#                                             maxlen=MAX_PAD_LENGTH,
#                                             padding='post')
#%%
targets = np.array(targets)
contexts = np.array(contexts)[:,:,0]
labels = np.array(labels)

print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")
#%%
BATCH_SIZE = 4
BUFFER_SIZE = len(targets)

dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
#%%
class Word2Vec(tf.keras.Model):
    def __init__(self,vocab_size,embedding_dim):
        super(Word2Vec,self).__init__()
        self.target_embedding=layers.Embedding(vocab_size,
                                               embedding_dim,
                                               input_length=1,
                                               name="w2v_embedding")
        self.context_embedding=layers.Embedding(vocab_size,
                                                embedding_dim,
                                                input_length=num_ns+1)

    def call(self,pair):
        target,context=pair

        if len(target.shape)==2:
            target=tf.squeeze(target,axis=1)
        word_emb=self.target_embedding(target)
        context_emb=self.context_embedding(context)
        dots=tf.einsum('be,bce->bc',word_emb,context_emb)
        return dots
#%%
embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

word2vec.fit(dataset, epochs=20)
#%%
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
norms = np.linalg.norm(weights, axis=1)
#%%
'''maximum simliarity word 찾기'''
topk = 10
idx = vocab['열정'] # 찾고자 하는 단어
sim = weights[idx, :] @ weights.T
cosine_sim = sim / (np.linalg.norm(weights[idx, :]) * norms)
topk_words = np.argsort(cosine_sim)[-topk:] # 뒤에서부터 가장 비슷한 단어임!
topk_words_score = [(vocab_reverse.get(i), c) for i, c in zip(topk_words, cosine_sim[topk_words])][::-1]
print(topk_words_score)
#%%