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


class BeamSearch:
    def __init__(self, model, en_tokenizer, fa_tokenizer, *, alpha=0.7, beam_width=5, max_length=50):
        self.model = model
        self.en_tokenizer = en_tokenizer
        self.fa_tokenizer = fa_tokenizer
        self.alpha = alpha
        self.beam_width = beam_width
        self.max_length = max_length
        self.start_token = fa_tokenizer.cls_token_id
        self.end_token = fa_tokenizer.sep_token_id
        self.vocab_size = fa_tokenizer.vocab_size
        

    def __call__(self, sentence, *, only_return_best=True):
        beam_width = self.beam_width
        
        sentence = self.en_tokenizer.encode(sentence, add_special_tokens=True)
        sentence = tf.cast(sentence, tf.int32)
        sentence = tf.expand_dims(sentence, 0)
        
        # Beam search parameters
        lst = tf.expand_dims([self.start_token], 0)
        lst = tf.cast(lst, tf.int32)
        lst_score = tf.expand_dims([0.0], 0)
        
        # Finished flags
        finished = tf.zeros((beam_width,), dtype=tf.int32)
        fix_finished = tf.ones((beam_width,), dtype=tf.int32)
        answers = []
        
        for length in range(self.max_length):
            idx = []
            scores = []
            logits = []
            index = tf.where(fix_finished == 0)
            if len(index) != 0:
                index = tf.squeeze(index, axis=-1)
              
                answer_value = tf.gather(lst, index)
                answer_score = tf.gather(lst_score, index)
                for j in range(len(answer_score)):
                    answers.append([answer_value[j], answer_score[j]])
                    
                index_not_finished = tf.where(fix_finished == 1)
                index_not_finished = tf.squeeze(index_not_finished, axis=-1)
                
                lst = tf.gather(lst, index_not_finished)
                lst = tf.reshape(lst, (len(index_not_finished), -1))
                lst_score = tf.gather(lst_score, index_not_finished)
                lst_score = tf.reshape(lst_score, (len(index_not_finished), -1))
                
                fix_finished = tf.gather(fix_finished, index_not_finished)
                fix_finished = tf.reshape(fix_finished, (-1,))
        
                finished = tf.gather(finished, index_not_finished)
                finished = tf.reshape(finished, (-1,))
        
                beam_width -= len(index)
                
            if beam_width <= 0: break
            outputs = self.model(tf.repeat(sentence, len(lst), axis=0), lst, training=False)
            outputs = tf.nn.softmax(outputs)
            outputs = outputs[:, -1]
            # flatten outputs
            y = tf.reshape(outputs, (-1,))
            
            idx = tf.gather(tf.range(len(lst)), tf.zeros(self.vocab_size, dtype=tf.int32)) \
                        if len(lst) == 1 else tf.transpose(tf.broadcast_to(tf.range(len(lst)), (self.vocab_size, len(lst))))
            
            idx = tf.reshape(idx, (-1,))
        
            lst_score = tf.reshape(lst_score, (-1,))
            lst_score = tf.gather(lst_score, idx)
            y = lst_score + tf.math.log(y)
            top_values, top_indices = tf.nn.top_k(y, k=beam_width)
            lst_score = tf.gather(y, top_indices)
            base = tf.gather(idx, top_indices)
            top_indices = top_indices % self.vocab_size
        
            lst = tf.gather(lst, base)
            lst = tf.cast(lst, tf.int32)
        
            finished = top_indices != self.end_token
            finished = tf.cast(finished, tf.int32)
            # Replace the loop with tensor operations
            mask = tf.logical_and(tf.equal(fix_finished, 1), tf.equal(finished, 0))
            fix_finished = tf.where(mask, tf.zeros_like(fix_finished), fix_finished)
            
            lst = tf.concat([lst, top_indices[..., tf.newaxis]], axis=-1)
        
        prob = [p / tf.math.pow(tf.cast(i.shape[0], tf.float32), self.alpha) for i, p in answers]
        
        best = tf.argmax(prob)
        result = self.fa_tokenizer.decode(answers[best][0][1:-1])

        if only_return_best:
            return result
        return [[self.fa_tokenizer.decode(x[0][1:-1]), x[1]] for x in answers]