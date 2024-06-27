import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '20'

import re
import sys
import tensorflow as tf
from transformers import BertTokenizer, AutoTokenizer


MODEL_PATH = '../models/transformers_60M.tf'

def main():
    print('Loading tokenizers . . .')
    en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    fa_tokenizer = AutoTokenizer.from_pretrained('bolbolzaban/gpt2-persian')
    
    print('Initializing . . .')

    # initialize tflite model
    model = tf.keras.models.load_model(MODEL_PATH)

    path = sys.argv[1]
    subtitles = get_subtitles(path)
    total_subtitles = len(subtitles)

    translated = []
    greedy = GreedyTranslator(model, fa_tokenizer, max_length=100)
    for i, sub in enumerate(subtitles):
        index = sub['index']
        start = sub['start']
        end = sub['end']
        text = sub['text']

        text = en_tokenizer.encode(text, add_special_tokens=True)
        text = tf.cast(text, dtype=tf.int32)
        text = tf.expand_dims(text, axis=0)
        translation = greedy(text)[0][1:-1]
        translation = fa_tokenizer.decode(translation)
        new_sub = {'index':index, 'start':start, 'end':end, 'text':translation}
        translated.append(new_sub)
        print(f'\r%{100 * (i + 1) / total_subtitles:>4.2f} Completed', end='')

    with open('result.srt', 'w') as file:
        texts = ''
        for t in translated:
            index = t['index']
            start = t['start']
            end = t['end']
            text = t['text']
            texts += f'{index}\n{start} --> {end}\n{text}\n\n'
        file.write(texts)



def get_subtitles(filename: str) -> list:
    with open(filename, 'r', encoding='utf8') as file:
        file = file.read()

    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})((\n.+)*)'
    matches = re.findall(pattern, file)
    subtitles = []
    for match in matches:
        index = int(match[0])
        start = match[1]
        end = match[2]
        lines = match[3].split('\n')[1:]
        text = ' '.join(lines).strip()
        subtitles.append({'index': index, 'start': start, 'end': end, 'text': text})
    return subtitles


class GreedyTranslator(tf.Module):
    def __init__(self, model, fa_tokenizer, *, max_length=50):
        super(GreedyTranslator, self).__init__()

        self.encoder = model.encoder
        self.decoder = model.decoder
        self.classifier = model.classifier
        self.fa_tokenizer = fa_tokenizer
        self.max_length = max_length
        self.start_token = fa_tokenizer.cls_token_id
        self.end_token = fa_tokenizer.sep_token_id
        self.pad_token = fa_tokenizer.pad_token_id

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def __call__(self, tokenized_sentence):
        batch_size = tf.shape(tokenized_sentence)[0]
        context = self.encoder(tokenized_sentence)
        
        decoder_input_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        decoder_input_array = decoder_input_array.write(0, tf.fill([batch_size,], self.start_token))

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
        return outputs


if __name__ == "__main__":
    main()
