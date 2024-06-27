#!/usr/bin/env python

# # Neural Machine Translation with Transformers


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '20'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import re
from transformers import BertTokenizer, AutoTokenizer
from sklearn.model_selection import train_test_split
import time
from Utils.Transformers import Transformer
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.metrics import Metric

device = tf.config.experimental.list_physical_devices('GPU')[0]
print(device)


en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
fa_tokenizer = AutoTokenizer.from_pretrained('bolbolzaban/gpt2-persian')


def create_dataset(file_path, num_examples=None):
    en_fa = []
    cnt = 0
    with open(file_path, 'r') as file:
        for line in file.readlines():
            if num_examples is not None and cnt >= num_examples: break
            line = line.split('\t')[:2]
            en = line[0]
            fa = line[1]
            fa = re.sub('\u200c', ' ', fa)
            en_fa.append([en, fa])
            cnt += 1
            if cnt % 100 == 0:
                print(f'\rRead {cnt:>5}', end='')

    print('\nRead', cnt)
    return zip(*en_fa)


def sort_dataset(source, target):
    xy = sorted(zip(source, target), key=lambda x: (len(x[0].split(' ')), len(x[1].split(' ')))) 
    x = [x[0] for x in xy]
    y = [y[1] for y in xy]
    return x, y


### Loading the dataset


NUM_SAMPLES = 2_000_000
FILE_PATH = './Data/combined.txt'
src_lang, targ_lang = create_dataset(FILE_PATH, NUM_SAMPLES)


src_train, src_val, targ_train, targ_val = train_test_split(src_lang, targ_lang, test_size=0.2, shuffle=True)
src_train, targ_train = sort_dataset(src_train, targ_train)
src_val, targ_val = sort_dataset(src_val, targ_val)
print('Source Train examples:', len(src_train))
print('Source Validation examples:', len(src_val))


### Create the pipeline


def en_vectorization(text):
    return en_tokenizer(text, max_length=50, truncation=True, padding='max_length')['input_ids']

def fa_vectorization(text):
    return fa_tokenizer(text, max_length=50, truncation=True, padding='max_length')['input_ids']

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128
STEPS_PER_EPOCH = len(src_train) // BATCH_SIZE
BUFFER_SIZE = 1000

def preprocess(src, targ):
    src = tf.py_function(func=en_vectorization, inp=[src], Tout=tf.TensorSpec(shape=(None, None), dtype=tf.int32))
    src = tf.cast(src, tf.int32)
    targ = tf.py_function(func=fa_vectorization, inp=[targ], Tout=tf.TensorSpec(shape=(None, None), dtype=tf.int32))
    targ = tf.cast(targ, tf.int32)
    return src, targ

train_ds = tf.data.Dataset.from_tensor_slices((src_train, targ_train))
train_ds = train_ds.batch(BATCH_SIZE, num_parallel_calls=AUTOTUNE, drop_remainder=True)
train_ds = train_ds.map(preprocess,
                       num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(STEPS_PER_EPOCH)
train_ds = train_ds.prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((src_val, targ_val))
test_ds = test_ds.batch(BATCH_SIZE, num_parallel_calls=AUTOTUNE, drop_remainder=True)
test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache()
test_ds = test_ds.prefetch(AUTOTUNE)

print('Loading cache...')
for i, (x, y) in enumerate(train_ds): 
    print(f'\r{i:>5}', end='')
print()
for i, (x, y) in enumerate(test_ds): 
    print(f'\r{i:>5}', end='')
print()

### Define the model
N = 6
d_model = 256
dff = 1024
num_heads = 8
dropout_rate = 0.1

VOCAB_SRC_SIZE = en_tokenizer.vocab_size
VOCAB_TARG_SIZE = fa_tokenizer.vocab_size

transformer = Transformer(d_model=d_model, 
                          num_heads=num_heads, 
                          dff=dff,
                          N=N,  
                          source_vocab_size=VOCAB_SRC_SIZE, 
                          target_vocab_size=VOCAB_TARG_SIZE,                           
                          dropout_rate=dropout_rate)

transformer.build(input_shapes=[(None, ), (None, )])
transformer.summary()


### Training


loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def masked_loss_function(y_true, pred):
    mask = tf.logical_not(tf.equal(y_true, fa_tokenizer.pad_token_id))
    loss = loss_object(y_true, pred)
    mask = tf.cast(mask, loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


class MaskedAccuracy(Metric):
    def __init__(self, name='masked_accuracy', **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)

        self.total = self.add_weight(name='total', 
                                     shape=(), 
                                     dtype='float32', 
                                     initializer='zeros')
        
        self.count = self.add_weight(name='count', 
                                     shape=(), 
                                     dtype='float32', 
                                     initializer='zeros')


    def update_state(self, y_true, pred, sample_weights=None):
        # pred = [B, dec_seq_len, target_vocab_size]
        # y_true = [B, dec_seq_len]
        y_pred = tf.argmax(pred, axis=-1)
        y_pred = tf.cast(y_pred, y_true.dtype)
        mask = tf.logical_not(tf.equal(y_true, fa_tokenizer.pad_token_id))

        match = tf.equal(y_pred, y_true)
        match = tf.logical_and(match, mask)

        match = tf.cast(match, tf.float32)
        mask = tf.cast(mask, tf.float32)

        self.total.assign_add(tf.reduce_sum(match))
        self.count.assign_add(tf.reduce_sum(mask))


    def result(self):
        return self.total / self.count if self.count != 0 else 0

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


def get_scheduler(d_model, warmup_steps=4000):
    def get_learning_rate(step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * tf.math.pow(warmup_steps * 1.0, -1.5)

        return tf.math.rsqrt(d_model * 1.0) * tf.minimum(arg1, arg2)

    return get_learning_rate


EPOCHS = 30
scheduler = get_scheduler(d_model, warmup_steps=int(0.15 * STEPS_PER_EPOCH * EPOCHS))
optimizer = Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9)

train_acc = MaskedAccuracy(name='train_acc')
valid_acc = MaskedAccuracy(name='valid_acc')



@tf.function(reduce_retracing=True, input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32), 
                                                     tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
def training_step(source, target):
    loss = 0
    y_true = target[:, 1:]
    target = target[:, :-1]

    with tf.GradientTape() as tape:
        pred = transformer(source, target, training=True)
        loss = masked_loss_function(y_true, pred)
        train_acc.update_state(y_true, pred)

    variables = transformer.trainable_variables
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))

    return loss


@tf.function(reduce_retracing=True, input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32), 
                                                     tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
def validation_step(source, target):
    loss = 0
    y_true = target[:, 1:]
    target = target[:, :-1]
    pred = transformer(source, target, training=False)
    loss = masked_loss_function(y_true, pred)
    valid_acc.update_state(y_true, pred)
    return loss


train_writer = tf.summary.create_file_writer(logdir='./logs/train')
test_writer = tf.summary.create_file_writer(logdir='./logs/test')


train_mean_losses = []
train_accs = []
valid_mean_losses = []
valid_accs = []
total_steps = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1:>3}')
    train_acc.reset_states()
    valid_acc.reset_states()
    train_mean_loss = 0
    valid_mean_loss = 0
    start = time.time()

    for step, (source, target) in enumerate(train_ds):
        learning_rate = scheduler(total_steps)
        optimizer.learning_rate = learning_rate
        loss = training_step(source, target)
        acc = train_acc.result()
        
        train_mean_loss = train_mean_loss + (1 / (step + 1)) * (loss - train_mean_loss)
        end = time.time()
        total_steps += 1
        print(f'\r{int(end - start):>4} sec | step {step:>4}\tloss {train_mean_loss:>3.4f}\taccuracy {acc:3.2f}', end='')

        with train_writer.as_default():
            tf.summary.scalar('Loss_Per_Step', train_mean_loss, step=total_steps)
            tf.summary.scalar('Accuracy_Per_Step', acc, step=total_steps)
            
    print()
    train_mean_losses.append(train_mean_loss)
    train_accs.append(acc)

    with train_writer.as_default():
        tf.summary.scalar('Loss_Per_Epoch', train_mean_loss, step=epoch)
        tf.summary.scalar('Accuracy_Per_Epoch', acc, step=epoch)
        

    for step, (source, target) in enumerate(test_ds):
        loss = validation_step(source, target)
        acc = valid_acc.result()

        valid_mean_loss = valid_mean_loss + (1 / (step + 1)) * (loss - valid_mean_loss)
        valid_accs.append(acc)
        end = time.time()
        print(f'\r{int(end - start):>4} sec | step {step:>4}\tloss {valid_mean_loss:>3.4f}\taccuracy {acc:3.2f}', end='')
    
    valid_mean_losses.append(train_mean_loss)
    valid_accs.append(acc)

    with test_writer.as_default():
        tf.summary.scalar('Loss_Per_Epoch', valid_mean_loss, step=epoch)
        tf.summary.scalar('Accuracy_Per_Epoch', acc, step=epoch)
    
    transformer.save_weights(f'./{epoch + 1}.h5')
    print('\n')


### Save the translator
transformer.save('./models/transformers_60M.tf')





