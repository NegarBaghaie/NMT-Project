import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '20'

import tensorflow as tf
import matplotlib.pyplot as plt
import re
from transformers import BertTokenizer, AutoTokenizer
from parsivar import Normalizer
import time
import sacrebleu
from Utils.BahdanauAttention import Encoder, Decoder
from Utils.Transformers import Transformer
from bidi.algorithm import get_display
import arabic_reshaper

device = tf.config.experimental.list_physical_devices('GPU')[0]
print(device)

NUM_SAMPLES = 1_000_000
FILE_PATH = './Data/combined.txt'
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128
BUFFER_SIZE = 1000

en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
fa_tokenizer = AutoTokenizer.from_pretrained('bolbolzaban/gpt2-persian')
VOCAB_SRC_SIZE = en_tokenizer.vocab_size
VOCAB_TARG_SIZE = fa_tokenizer.vocab_size


def main():
    global en_tokenizer
    global fa_tokenizer
    print('Loading data . . .')

    with open('./Evaluation/source.txt', 'r') as file:
        source = file.read()
        source = source.split('\n')

    with open('./Evaluation/reference1.txt', 'r') as file:
        reference1 = file.read()
        reference1 = reference1.replace('\u200c', ' ')
        reference1 = [Normalizer(half_space_char=' ', statistical_space_correction=True).normalize(x) for x in
                      reference1.split('\n')]

    print('Loading models . . .')

    encoder, decoder = getBahadanauAttention(BATCH_SIZE=1)
    encoder.load_weights('./Checkpoints/RNN/Encoder/29.h5')
    decoder.load_weights('./Checkpoints/RNN/Decoder/29.h5')

    transformer60M = Transformer(d_model=256,
                                 dff=1024,
                                 num_heads=8,
                                 N=6,
                                 source_vocab_size=VOCAB_SRC_SIZE,
                                 target_vocab_size=VOCAB_TARG_SIZE)
    transformer60M.build(input_shapes=[(None,), (None,)])
    transformer60M.load_weights('./Checkpoints/Transformers60M/30.h5')

    transformer16M = Transformer(d_model=128,
                                 dff=512,
                                 num_heads=8,
                                 N=4,
                                 source_vocab_size=VOCAB_SRC_SIZE,
                                 target_vocab_size=VOCAB_TARG_SIZE)
    transformer16M.build(input_shapes=[(None,), (None,)])
    transformer16M.load_weights('./Checkpoints/Transformers16M/26.h5')


    print('Evaluating RNN with Attention')
    greedyTranslator = GreedyTranslator(encoder, decoder, fa_tokenizer, max_length=100)
    predictions = []
    mean_time = 0

    for i, sentence in enumerate(source):
        tokenized = en_tokenizer.encode(sentence, add_special_tokens=True)
        tokenized = tf.expand_dims(tokenized, axis=0)

        start = time.time()
        translated = greedyTranslator(tokenized)[0]
        end = time.time()

        translation_decoded = fa_tokenizer.decode(translated[1:-1])
        predictions.append(translation_decoded)

        mean_time = mean_time + ((end - start) - mean_time) / (i + 1)
        bleu_score = sacrebleu.corpus_bleu(predictions[:i+1], [reference1[:i+1]]).score
        print(
            f'\r%{100 * (i + 1) / len(source):>4.2f} Completed | BLEU Score is {bleu_score:>3.2f} | Mean inference time {mean_time:>3.3f} seconds',
            end='')

    print(f'\n\nBLEU Score is {bleu_score:>3.2f}')

    with open('./Evaluation/RNN/predictions.txt', 'w') as file:
        file.write(f'RNN Evaluation: BLEU SCORE is {bleu_score}\n##################\n')

        for i in range(len(predictions)):
            file.write(f'{source[i]}\n{predictions[i]}\n\n')


    print('Evaluating Transformers60M')
    predictions = []
    greedy = TransformerGreedyTranslator(model=transformer60M,
                                         fa_tokenizer=fa_tokenizer,
                                         return_attention_weights=False)
    mean_time = 0
    for i, sentence in enumerate(source):
        tokenized = en_tokenizer.encode(sentence, add_special_tokens=True)
        tokenized = tf.expand_dims(tokenized, axis=0)
        start = time.time()
        translation = greedy(tokenized)[0]
        end = time.time()

        translation_decoded = fa_tokenizer.decode(translation[1:-1])
        predictions.append(translation_decoded)

        mean_time = mean_time + ((end - start) - mean_time) / (i + 1)
        bleu_score = sacrebleu.corpus_bleu(predictions[:i+1], [reference1[:i+1]]).score
        print(f'\r%{100 * (i + 1) / len(source):>4.2f} Completed | BLEU Score is {bleu_score:>3.2f} | Mean inference time {mean_time:>3.3f} seconds',
              end='')
    print(f'\n\nBLEU Score is {bleu_score:>3.2f}')

    with open('./Evaluation/Transformers60M/predictions.txt', 'w') as file:
        file.write(f'Transformer60M Evaluation: BLEU SCORE is {bleu_score}\n##################\n')

        for i in range(len(predictions)):
            file.write(f'{source[i]}\n{predictions[i]}\n\n')




    print('Evaluating Transformer16M')
    predictions = []
    greedy = TransformerGreedyTranslator(model=transformer16M,
                                         fa_tokenizer=fa_tokenizer,
                                         return_attention_weights=False)
    mean_time = 0
    for i, sentence in enumerate(source):
        tokenized = en_tokenizer.encode(sentence, add_special_tokens=True)
        tokenized = tf.expand_dims(tokenized, axis=0)
        start = time.time()
        translation = greedy(tokenized)[0]
        end = time.time()

        translation_decoded = fa_tokenizer.decode(translation[1:-1])
        predictions.append(translation_decoded)

        mean_time = mean_time + ((end - start) - mean_time) / (i + 1)
        bleu_score = sacrebleu.corpus_bleu(predictions[:i+1], [reference1[:i+1]]).score
        print(f'\r%{100 * (i + 1) / len(source):>4.2f} Completed | BLEU Score is {bleu_score:>3.2f} | Mean inference time {mean_time:>3.3f} seconds',
              end='')
    print(f'\n\nBLEU Score is {bleu_score:>3.2f}')

    with open('./Evaluation/Transformers16M/predictions.txt', 'w') as file:
        file.write(f'Transformer16M Evaluation: BLEU SCORE is {bleu_score}\n##################\n')

        for i in range(len(predictions)):
            file.write(f'{source[i]}\n{predictions[i]}\n\n')


def create_dataset(file_path, num_examples=None):
    en_fa = []
    cnt = 0
    fa_normalizer = Normalizer()
    with open(file_path, 'r') as file:
        for line in file.readlines():
            if num_examples is not None and cnt >= num_examples: break
            line = line.split('\t')[:2]
            en = line[0]
            fa = fa_normalizer.normalize(line[1])
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


def en_vectorization(text):
    text = [x.decode('utf-8') for x in text.numpy()]
    return en_tokenizer(text, padding=True, return_tensors='tf')['input_ids']


def fa_vectorization(text):
    text = [x.decode('utf-8') for x in text.numpy()]
    return fa_tokenizer(text, padding=True, return_tensors='tf')['input_ids']


def preprocess(src, targ):
    src = tf.py_function(func=en_vectorization, inp=[src], Tout=tf.TensorSpec(shape=(None, None), dtype=tf.int32))
    src = tf.cast(src, tf.int32)
    targ = tf.py_function(func=fa_vectorization, inp=[targ], Tout=tf.TensorSpec(shape=(None, None), dtype=tf.int32))
    targ = tf.cast(targ, tf.int32)
    return src, targ


def getBahadanauAttention(BATCH_SIZE):
    SOURCE_VOCAB_SIZE = len(en_tokenizer.get_vocab())
    TARGET_VOCAB_SIZE = len(fa_tokenizer.get_vocab())
    ENCODER_UNITS = 64
    DECODER_UNITS = 128
    EMBD_DIM = 128
    ATTENTION_UNITS = 10

    encoder = Encoder(rnn_units=ENCODER_UNITS,
                      vocab_size=SOURCE_VOCAB_SIZE,
                      embd_dim=EMBD_DIM,
                      batch_size=BATCH_SIZE)

    decoder = Decoder(rnn_units=DECODER_UNITS,
                      attention_units=ATTENTION_UNITS,
                      embd_dim=EMBD_DIM,
                      vocab_size=TARGET_VOCAB_SIZE)

    encoder.build()
    decoder = Decoder(DECODER_UNITS, ATTENTION_UNITS, TARGET_VOCAB_SIZE, EMBD_DIM)
    hidden = tf.zeros((BATCH_SIZE, ENCODER_UNITS * 2), dtype=tf.float32)
    sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                          tf.zeros((BATCH_SIZE, 10, ENCODER_UNITS * 2)), hidden)
    return encoder, decoder


class GreedyTranslator(tf.Module):
    def __init__(self, encoder, decoder, fa_tokenizer, *, max_length=50, return_attention_weights=False):
        super(GreedyTranslator, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.fa_tokenizer = fa_tokenizer
        self.max_length = max_length
        self.start_token = fa_tokenizer.cls_token_id
        self.end_token = fa_tokenizer.sep_token_id
        self.pad_token = fa_tokenizer.pad_token_id
        self.return_attention_weights = return_attention_weights

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def __call__(self, tokenized_sentence):
        batch_size = tf.shape(tokenized_sentence)[0]
        seqs, (h, bh) = self.encoder(tokenized_sentence)
        s_prev = tf.concat([h, bh], axis=-1)

        dec_inp_arr = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        dec_inp_arr = dec_inp_arr.write(0, tf.fill([batch_size], self.start_token))
        not_finished = tf.ones((batch_size,), tf.float32)
        attention_weights_arr = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        for i in tf.range(self.max_length):
            dec_inp = tf.transpose(dec_inp_arr.stack())
            dec_inp = tf.expand_dims(dec_inp[:, -1], axis=-1)
            output, s_prev, attention_weights = self.decoder(dec_inp, seqs, s_prev, training=False)
            attention_weights_arr = attention_weights_arr.write(i, attention_weights[0, :, 0])
            output = tf.argmax(output, axis=-1)
            output = tf.cast(output, tf.float32)
            not_seen_end = tf.cast(tf.logical_not(tf.equal(output, self.end_token)), tf.float32)
            not_finished = not_seen_end * not_finished
            output = output * not_finished + (1 - not_finished) * self.pad_token

            output = tf.cast(output, tf.int32)
            dec_inp_arr = dec_inp_arr.write(i + 1, output[:])

            if tf.reduce_sum(not_finished) == 0:
                break

        outputs = tf.transpose(dec_inp_arr.stack())
        decoder_input_array = dec_inp_arr.close()

        if self.return_attention_weights:
            return outputs, attention_weights_arr.stack()

        return outputs


class TransformerGreedyTranslator(tf.Module):
    def __init__(self, model, fa_tokenizer, *, max_length=50, return_attention_weights=False):
        super(TransformerGreedyTranslator, self).__init__()

        self.model = model
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.classifier = model.classifier
        self.fa_tokenizer = fa_tokenizer
        self.max_length = max_length
        self.start_token = fa_tokenizer.cls_token_id
        self.end_token = fa_tokenizer.sep_token_id
        self.pad_token = fa_tokenizer.pad_token_id
        self.return_attention_weights = return_attention_weights

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def __call__(self, tokenized_sentence):
        batch_size = tf.shape(tokenized_sentence)[0]
        context = self.encoder(tokenized_sentence)

        decoder_input_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        decoder_input_array = decoder_input_array.write(0, tf.fill([batch_size, ], self.start_token))

        not_finished = tf.ones(batch_size, dtype=tf.float32)

        for i in tf.range(self.max_length):
            decoder_inps = tf.transpose(decoder_input_array.stack())

            outputs = self.decoder(decoder_inps, context, training=False)
            outputs = self.classifier(outputs, training=False)
            outputs = tf.argmax(outputs, axis=-1)
            outputs = tf.cast(outputs[:, -1], tf.float32)

            outputs = outputs * not_finished + (1 - not_finished) * self.pad_token
            not_finished *= tf.cast(tf.logical_not(tf.equal(outputs, self.end_token)), tf.float32)
            outputs = tf.cast(outputs, tf.int32)

            decoder_input_array = decoder_input_array.write(i + 1, outputs)

            if tf.reduce_sum(not_finished) == 0:
                break

        outputs = tf.transpose(decoder_input_array.stack())
        decoder_input_array = decoder_input_array.close()

        if self.return_attention_weights:
            self.model(tokenized_sentence, outputs, training=False)
            attention_weights = self.model.decoder.last_attention_scores
            return outputs, attention_weights

        return outputs


def draw_transformer_attention(result, tokenized, attention_scores, filename='attention.png'):
    result = result.numpy()[0, 1:]

    x_tick_labels = [en_tokenizer.decode([i]) for i in tokenized[0]]
    y_tick_labels = [get_display(arabic_reshaper.reshape(fa_tokenizer.decode([i]))) for i in result]

    fig = plt.figure(figsize=(20, 10))
    for h in range(8):
        ax = plt.subplot(2, 4, h + 1)
        attentions = attention_scores[0][h]
        ax.matshow(attentions[:-1, :])
        ax.set_xticks(range(len(x_tick_labels)))
        ax.set_yticks(range(len(y_tick_labels)));
        ax.set_xticklabels(x_tick_labels, rotation=90);
        ax.set_yticklabels(y_tick_labels);

    fig.savefig(filename, dpi=320)


if __name__ == '__main__':
    main()