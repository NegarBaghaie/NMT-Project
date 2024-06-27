import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '20'

import tensorflow as tf
from transformers import BertTokenizer, AutoTokenizer
from Utils.Transformers import Transformer
from Utils.BeamSearch import BeamSearch
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
import arabic_reshaper


MODEL = './Checkpoints/Transformers60M/30.h5'
en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
fa_tokenizer = AutoTokenizer.from_pretrained('bolbolzaban/gpt2-persian')
VOCAB_SRC_SIZE = en_tokenizer.vocab_size
VOCAB_TARG_SIZE = fa_tokenizer.vocab_size

def main():
    print('Loading the model . . .')
    transformer = Transformer(d_model=256,
                              num_heads=8,
                              dff=1024,
                              N=6,
                              source_vocab_size=VOCAB_SRC_SIZE,
                              target_vocab_size=VOCAB_TARG_SIZE,
                              dropout_rate=0.0)

    transformer.build(input_shapes=[(None,), (None,)])
    transformer.load_weights(MODEL)

    greedy = input('Use greedy? (Y/N)')
    if greedy.lower() == 'y': greedy = True
    else: greedy = False

    if greedy:
        attention = input('Show attention scores? (Y/N)')
        if attention.lower() == 'y': attention = True
        else: attention = False
    else: attention = False

    beam_search = BeamSearch(transformer,
                             en_tokenizer,
                             fa_tokenizer,
                             beam_width=20,
                             max_length=100)

    greedy = GreedyTranslator(transformer,
                              fa_tokenizer,
                              max_length=100,
                              return_attention_weights=attention)

    cnt = 0
    while True:
        source = input('>> ')
        if not greedy:
            print(beam_search(source))
        else:
            source = en_tokenizer.encode(source, add_special_tokens=True)
            source = tf.expand_dims(source, axis=0)
            outputs = greedy(source)
            if attention:
                outputs, attention_scores = outputs

            translation = fa_tokenizer.decode(outputs[0][1:-1])
            print(translation)

            if attention:
                draw_transformer_attention(outputs, source, attention_scores, filename=f'./figures/{cnt}.png')


class GreedyTranslator(tf.Module):
    def __init__(self, model, fa_tokenizer, *, max_length=50, return_attention_weights=False):
        super(GreedyTranslator, self).__init__()

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

    plt.show()


if __name__ == '__main__':
    main()