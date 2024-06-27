# Copyright 2024 Alireza Kamyab
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Dropout, Dense, MultiHeadAttention, Add, LayerNormalization, Embedding, Input
from keras.models import Model


def positional_encoding(length, depth):
    depth = depth / 2

    positions = tf.range(0, length, dtype='float32')[..., None]
    depths = tf.range(depth)[None, ...] / depth

    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates

    pos_encoding = tf.concat([tf.math.sin(angle_rads), tf.math.cos(angle_rads)], axis=-1)
    
    return pos_encoding


class PositionalEmbedding(Layer):
    def __init__(self, vocab_size, d_model, max_position=2048):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True, name='embedding')
        self.pos_encoding = positional_encoding(max_position, d_model)


    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, inputs):
        length = tf.shape(inputs)[1]
        x = self.embedding(inputs)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(Layer):
    def __init__(self, **kwargs):
        super(BaseAttention, self).__init__()

        self.mha = MultiHeadAttention(**kwargs)
        self.add = Add(name='add')
        self.layernorm = LayerNormalization(name='layernorm')


class CrossAttention(BaseAttention):
    def __init__(self, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)


    def call(self, query, context, training=False):
        # query has the shape [B, dec_seq_len, d_model]
        # context has the shape [B, enc_seq_len, d_model]
        outputs, attention_scores = self.mha(query=query, 
                           key=context, 
                           value=context, 
                           training=training,
                           return_attention_scores=True)

        # cache attention scores
        self.attention_scores = attention_scores

        outputs = self.add([query, outputs])
        outputs = self.layernorm(outputs)

        return outputs


class SelfAttention(BaseAttention):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)


    def call(self, inputs, training=False):
        # inputs has the shape [B, enc_seq_len, d_model]
        outputs = self.mha(query=inputs,
                           key=inputs,
                           value=inputs, 
                           training=training)

        # adds residual connection
        outputs = self.add([inputs, outputs])

        outputs = self.layernorm(outputs)
        return outputs


class MaskedMultiHeadAttention(BaseAttention):
    def __init__(self, **kwargs):
        super(MaskedMultiHeadAttention, self).__init__(**kwargs)


    def call(self, inputs, training=False):
        # inputs has the shape [B, dec_seq_len, d_model]
        outputs = self.mha(query=inputs, 
                           key=inputs, 
                           value=inputs, 
                           training=training, 
                           use_causal_mask=True)

        # adds residual connection
        outputs = self.add([inputs, outputs])

        outputs = self.layernorm(outputs)
        return outputs


class FeedForward(Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super(FeedForward, self).__init__()

        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.W1 = Dense(dff, activation='relu')
        self.W2 = Dense(d_model, activation='linear')

        if dropout_rate > 0.0:
            self.dropout = Dropout(dropout_rate, name='dropout')

        self.add = Add(name='add')
        self.layernorm = LayerNormalization(name='layernorm')


    def call(self, inputs, training=False):
        # inputs has the shape [B, Tq, d_model]
        outputs = self.W1(inputs)
        outputs = self.W2(outputs)

        if self.dropout_rate > 0.0:
            outputs = self.dropout(outputs, training=training)

        # adds residual connection
        outputs = self.add([inputs, outputs])
        outputs = self.layernorm(outputs)
        return outputs


class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.self_attention = SelfAttention(num_heads=num_heads, 
                                            key_dim=d_model, 
                                            dropout=dropout_rate)

        self.ffn = FeedForward(d_model=d_model, 
                               dff=dff, 
                               dropout_rate=dropout_rate)


    def call(self, inputs, training=False):
        # inputs has the shape [B, enc_seq_len, d_model]
        outputs = self.self_attention(inputs, training=training)
        outputs = self.ffn(outputs, training=training)
        return outputs


class Encoder(Layer):
    def __init__(self, *, d_model, num_heads, dff, N, vocab_size, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.N = N
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate

        self.positional_embedding = PositionalEmbedding(vocab_size=vocab_size, 
                                                        d_model=d_model)

        self.stacks = [EncoderLayer(d_model=d_model, 
                                    num_heads=num_heads, 
                                    dff=dff, 
                                    dropout_rate=dropout_rate) for _ in range(N)]

        if dropout_rate > 0.0:
            self.dropout = Dropout(dropout_rate, name='dropout')


    def call(self, inputs, training=False):
        # inputs has the shape [B, seq_len]
        x = self.positional_embedding(inputs)
        if self.dropout_rate > 0.0:
            x = self.dropout(x, training=training)

        for n in range(self.N):
            x = self.stacks[n](x, training=training)

        return x


class DecoderLayer(Layer):
    def __init__(self, d_model, dff, num_heads, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.num_heads = num_heads
        self.dff = dff
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        self.masked_attention = MaskedMultiHeadAttention(num_heads=num_heads, 
                                                         key_dim=d_model, 
                                                         dropout=dropout_rate)

        self.cross_attention = CrossAttention(num_heads=num_heads, 
                                              key_dim=d_model, 
                                              dropout=dropout_rate)

        self.ffn = FeedForward(d_model=d_model, 
                               dff=dff, 
                               dropout_rate=dropout_rate)


    def call(self, inputs, context, training=False):
        # inputs has the shape [B, dec_seq_len, d_model]
        # context has the shape [B, enc_seq_len, d_model]
        outputs = self.masked_attention(inputs, training=training)
        outputs = self.cross_attention(query=outputs, 
                                       context=context, 
                                       training=training)

        # cache the attention scores
        self.last_attention_scores = self.cross_attention.attention_scores

        outputs = self.ffn(outputs, training=training)
        return outputs


class Decoder(Layer):
    def __init__(self, *, d_model, num_heads, dff, N, vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.N = N
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate

        if dropout_rate > 0.0:
            self.dropout = Dropout(dropout_rate, name='dropout')

        self.positional_embedding = PositionalEmbedding(vocab_size=vocab_size, 
                                                        d_model=d_model)

        self.stacks = [DecoderLayer(d_model=d_model, 
                                    num_heads=num_heads, 
                                    dff=dff, 
                                    dropout_rate=dropout_rate) for _ in range(N)]

    
    def call(self, inputs, context, training=False):
        # inputs has the shape [B, dec_seq_len]
        # context has the shape [B, enc_seq_len, d_model]
        x = self.positional_embedding(inputs)
        if self.dropout_rate > 0.0:
            x = self.dropout(x, training=training)

        for n in range(self.N):
            x = self.stacks[n](x, context=context, training=training)


        # cache attention scores for plotting
        self.last_attention_scores = self.stacks[-1].last_attention_scores

        return x


class Transformer(Model):
    def __init__(self, *, d_model, 
                 dff, 
                 num_heads, 
                 N, 
                 source_vocab_size, 
                 target_vocab_size, 
                 dropout_rate=0.1):
        
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads
        self.N = N
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.dropout_rate = dropout_rate

        self.encoder = Encoder(d_model=d_model, 
                               num_heads=num_heads, 
                               dff=dff, 
                               N=N, 
                               vocab_size=source_vocab_size, 
                               dropout_rate=dropout_rate)


        self.decoder = Decoder(d_model=d_model, 
                               num_heads=num_heads, 
                               dff=dff, 
                               N=N, 
                               vocab_size=target_vocab_size, 
                               dropout_rate=dropout_rate)

        self.classifier = Dense(units=target_vocab_size, name='classifier')


    def build(self, input_shapes):
        source_shape, target_shape = input_shapes
        source = Input(shape=source_shape)
        target = Input(shape=target_shape)
        self.call(source, target)
        self.built = True


    def call(self, source, target, training=False):
        context = self.encoder(source, training=training)
        outputs = self.decoder(target, context=context, training=training)
        logits = self.classifier(outputs, training=training)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            del logits._keras_mask
        except AttributeError:
            pass


        return logits