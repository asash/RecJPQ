from __future__ import annotations
import numpy as np

import numpy.typing as npt
import tensorflow as tf

class Tokenizer(tf.keras.layers.Layer):
    def __init__(self, tokens_per_item, values_per_dimension, num_items) -> None:
        super().__init__()
        self.tokens_per_item = tf.Variable(tokens_per_item, trainable=False, name="Tokenizer/TokensPerItem")
        self.num_items = tf.Variable(num_items, trainable=False, name="Tokenizer/NumItems")
        self.values_per_dimension = tf.Variable(values_per_dimension, trainable=False, name="Tokenizer/ValuesPerDimension")
        self.vocab_size = tf.Variable(tokens_per_item * values_per_dimension, trainable=False, name="Tokenizer/VocabSize")
        vocabulary_initaliser = tf.zeros_initializer()
        self.vocabulary = tf.Variable(vocabulary_initaliser((num_items + 1, tokens_per_item), dtype='int32'), 
                                      trainable=False, name="Tokenizer/Vocabulary")
        self.index = {}
        self.index_is_built = False #not a weight, so not saved in checkpoints 
    
    def assign(self, train_users):
        raise NotImplementedError()
    
    def build_index(self):
        for i in range(self.vocabulary.shape[0]):
            item_tokens = tuple(self.vocabulary[i].numpy())
            self.index[item_tokens] = i
    
    def decode(self, batch):
        if not self.index_is_built:
            self.build_index()
            self.index_is_built = True
            
        result = [] 
        for i in range(len(batch)):
           row = tuple(batch[i].numpy())
           item = self.index.get(row, self.num_items) # if incorrect sequence was generated return pad token
           result.append(item)
        return tf.constant(np.array(result))


    def call(self, batch: npt.ArrayLike, bs, seq_len):
        return tf.reshape(tf.gather(self.vocabulary, batch), (bs, seq_len*self.tokens_per_item))
    
