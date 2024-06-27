import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import re
import unicodedata
import matplotlib.ticker as ticker
from transformers import BertTokenizer, AutoTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from parsivar import Normalizer
import tensorflow_text as tf_text
from keras.layers import TextVectorization
import time

from keras.layers import Layer, GRU, Dense, Embedding, Activation, Bidirectional
from keras.models import Model


class Encoder(Model):
    def __init__(self, rnn_units, vocab_size, embd_dim, batch_size):
        super(Encoder, self).__init__()
        self.rnn_units = rnn_units
        self.batch_size = batch_size
        self.embedding = Embedding(vocab_size, embd_dim)
        self.gru = Bidirectional(GRU(
            units=rnn_units,
            return_sequences=True,
            return_state=True
        ))
        self.initialize_state()

    def build(self):
        inputs = keras.layers.Input(shape=(None,))
        self.call(inputs)
        self.built = True

    def call(self, inputs, initial_state=None):
        if initial_state is not None:
            self._hidden = initial_state[0]
            self._backward_hidden = initial_state[1]

        x = self.embedding(inputs)
        sequences, h, bh = self.gru(x, initial_state=(self._hidden, self._backward_hidden))

        return sequences, (h, bh)

    def get_hidden_states(self):
        return self._hidden, self._backward_hidden

    def initialize_state(self):
        self._hidden = tf.zeros((self.batch_size, self.rnn_units))
        self._backward_hidden = tf.zeros((self.batch_size, self.rnn_units))
        return self.get_hidden_states()


class BahdanauAttention(Model):
  def __init__(self, units):
      super(BahdanauAttention, self).__init__()
      self.w1 = Dense(units)
      self.w2 = Dense(units)
      self.v = Dense(1)

  def call(self, sequences, s_prev):
      s_prev = tf.expand_dims(s_prev, 1)
      x = self.v(tf.nn.tanh(self.w1(sequences) + self.w2(s_prev)))
      attention_weights = tf.nn.softmax(x, axis=1)
      context = sequences * attention_weights
      context = tf.reduce_sum(context, axis=1)

      return context, attention_weights



class Decoder(Model):
  def __init__(self, rnn_units, attention_units, vocab_size, embd_dim):
      super(Decoder, self).__init__()
      self.embedding = Embedding(vocab_size, embd_dim)
      self.gru = GRU(rnn_units, return_sequences=True, return_state=True)
      self.attention = BahdanauAttention(attention_units)
      self.fc = Dense(vocab_size)

  def call(self, inputs, sequences, s_prev):
      x = self.embedding(inputs)

      context, attention_weights = self.attention(sequences, s_prev)
      context = tf.expand_dims(context, 1)
      x = tf.concat([context, x], axis=-1)
      output, state = self.gru(x, s_prev)
      output = tf.reshape(output, (-1, output.shape[2]))
      output = self.fc(output)

      return output, state, attention_weights