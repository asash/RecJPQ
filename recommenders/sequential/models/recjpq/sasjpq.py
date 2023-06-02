from __future__ import annotations
from typing import Type
import numpy as np
import tensorflow as tf
from aprec.recommenders.sequential.models.positional_encodings import  get_pos_embedding
from aprec.recommenders.sequential.models.recjpq.rec_jpq_layer import ItemCodeLayer

from aprec.recommenders.sequential.samplers.sampler import get_negatives_sampler
from scipy.sparse import csr_matrix
layers = tf.keras.layers

from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialDataParameters, SequentialModelConfig, SequentialRecsysModel
from ..sasrec.sasrec_multihead_attention import multihead_attention
import tensorflow as tf

#https://ieeexplore.ieee.org/abstract/document/8594844
#the code is ported from original code
#https://github.com/kang205/SASRec

class SASJPQModel(SequentialRecsysModel):
    @classmethod
    def get_model_config_class(cls) -> Type[SASJPQConfig]:
        return SASJPQConfig

    def fit_biases(self, train_users):
        self.item_codes_layer.assign_codes(train_users)

    def __init__(self, model_parameters, data_parameters, *args, **kwargs):
        super().__init__(model_parameters, data_parameters, *args, **kwargs)
        self.model_parameters: SASJPQConfig #just a hint for the static analyser
        self.positions = tf.constant(tf.expand_dims(tf.range(self.data_parameters.sequence_length), 0))
        if self.model_parameters.pos_emb_comb != 'ignore':
            self.postion_embedding_layer = get_pos_embedding(self.data_parameters.sequence_length, self.model_parameters.embedding_size, self.model_parameters.pos_embedding)
        self.embedding_dropout = layers.Dropout(self.model_parameters.dropout_rate, name='embedding_dropout')
        self.attention_blocks = []
        for i in range(self.model_parameters.num_blocks):
            block_layers = {
                "first_norm": layers.LayerNormalization(),
                "attention_layers": {
                    "query_proj": layers.Dense(self.model_parameters.embedding_size, activation='linear'),
                    "key_proj": layers.Dense(self.model_parameters.embedding_size, activation='linear'),
                    "val_proj": layers.Dense(self.model_parameters.embedding_size, activation='linear'),
                    "dropout": layers.Dropout(self.model_parameters.dropout_rate),
                },
                "second_norm": layers.LayerNormalization(),
                "dense1": layers.Dense(self.model_parameters.embedding_size, activation='relu'),
                "dense2": layers.Dense(self.model_parameters.embedding_size),
                "dropout": layers.Dropout(self.model_parameters.dropout_rate)
            }
            self.attention_blocks.append(block_layers)
        self.output_activation = tf.keras.activations.get(self.model_parameters.output_layer_activation)
        self.seq_norm = layers.LayerNormalization()
        self.all_items = tf.range(0, self.data_parameters.num_items)
        if self.model_parameters.encode_output_embeddings:
            self.output_item_embeddings_encode = layers.Dense(self.model_parameters.embedding_size, activation='gelu')

        self.sampler = get_negatives_sampler(self.model_parameters.vanilla_target_sampler, 
                                                 self.data_parameters, self.model_parameters.vanilla_num_negatives)

        self.item_codes_layer = ItemCodeLayer(self.model_parameters.embedding_size, 
                                              self.model_parameters.pq_m, 
                                              self.data_parameters.num_items, 
                                              self.data_parameters.sequence_length,
                                              self.model_parameters.centroid_strategy)


    def block(self, seq, mask, i):
        x = self.attention_blocks[i]["first_norm"](seq)
        queries = x
        keys = seq
        x, attentions = multihead_attention(queries, keys, self.model_parameters.num_heads, self.attention_blocks[i]["attention_layers"],
                                     causality=self.model_parameters.causal_attention)
        x =x + queries
        x = self.attention_blocks[i]["second_norm"](x)
        residual = x
        x = self.attention_blocks[i]["dense1"](x)
        x = self.attention_blocks[i]["dropout"](x)
        x = self.attention_blocks[i]["dense2"](x)
        x = self.attention_blocks[i]["dropout"](x)
        x += residual
        x *= mask
        return x, attentions

    def get_dummy_inputs(self):
        pad = tf.cast(tf.fill((self.data_parameters.batch_size, 1), self.data_parameters.num_items), 'int64')
        seq = tf.zeros((self.data_parameters.batch_size, self.data_parameters.sequence_length-1), 'int64')
        inputs = [tf.concat([pad, seq], -1)]
        positives = tf.zeros((self.data_parameters.batch_size, self.data_parameters.sequence_length), 'int64')
        inputs.append(positives)
        return inputs

    def call(self, inputs,  **kwargs):
        input_ids = inputs[0]
        training = kwargs['training']
        seq_emb, attentions = self.get_seq_embedding(input_ids, bs=self.data_parameters.batch_size, training=training)
        target_positives = tf.expand_dims(inputs[1], -1)
        target_negatives = self.sampler(input_ids, target_positives)
        target_ids = tf.concat([target_positives, target_negatives], -1)
        logits = self.item_codes_layer.score_sequence_items(seq_emb, target_ids, self.data_parameters.batch_size)
        positive_logits = logits[:, :, 0]
        negative_logits = logits[:,:,1:]
        minus_positive_logprobs = tf.math.softplus(-positive_logits)
        minus_negative_logprobs = tf.reduce_sum(tf.math.softplus(negative_logits) , axis=-1)
        minus_average_logprobs = (minus_positive_logprobs + minus_negative_logprobs) / (self.model_parameters.vanilla_num_negatives +1)
        mask = 1 - tf.cast(input_ids == self.data_parameters.num_items, 'float32')
        result = tf.reduce_sum(mask*minus_average_logprobs)/tf.reduce_sum(mask)
        return result
    
    def score_all_items(self, inputs):
        input_ids = inputs[0]
        seq_emb, attentions = self.get_seq_embedding(input_ids)
        last_item_emb = seq_emb[:, -1, :]
        return self.item_codes_layer.score_all_items(last_item_emb)


    def get_seq_embedding(self, input_ids, bs=None, training=None):
        if bs is None:
            bs = input_ids.shape[0]
        seq = self.item_codes_layer(input_ids, bs)
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_ids, self.data_parameters.num_items), dtype=tf.float32), -1)
        positions  = tf.tile(self.positions, [bs, 1])
        if training and self.model_parameters.pos_smoothing:
            smoothing = tf.random.normal(shape=positions.shape, mean=0, stddev=self.model_parameters.pos_smoothing)
            positions =  tf.maximum(0, smoothing + tf.cast(positions, 'float32'))
        if self.model_parameters.pos_emb_comb != 'ignore':
            pos_embeddings = self.postion_embedding_layer(positions)[:input_ids.shape[0]]
        if self.model_parameters.pos_emb_comb == 'add':
             seq += pos_embeddings
        elif self.model_parameters.pos_emb_comb == 'mult':
             seq *= pos_embeddings

        elif self.model_parameters.pos_emb_comb == 'ignore':
             seq = seq
        seq = self.embedding_dropout(seq)
        seq *= mask
        attentions = []
        for i in range(self.model_parameters.num_blocks):
            seq, attention = self.block(seq, mask, i)
            attentions.append(attention)
        seq_emb = self.seq_norm(seq)
        return seq_emb, attentions 

class SASJPQConfig(SequentialModelConfig):
    def __init__(self, output_layer_activation='linear', embedding_size=64,
                dropout_rate=0.5, num_blocks=2, num_heads=1,
                reuse_item_embeddings=False,
                encode_output_embeddings=False,
                pos_embedding = 'learnable', 
                pos_emb_comb = 'add',
                causal_attention = True,
                pos_smoothing = 0,
                max_targets_per_user=10,
                vanilla_num_negatives = 1,
                vanilla_bce_t = 0.0,
                vanilla_target_sampler = 'random',
                full_target = False,
                centroid_strategy = 'svd',
                pq_m = 4,
                ): 
        self.output_layer_activation=output_layer_activation
        self.embedding_size=embedding_size
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.reuse_item_embeddings = reuse_item_embeddings
        self.encode_output_embeddings = encode_output_embeddings
        self.pos_embedding = pos_embedding
        self.pos_emb_comb = pos_emb_comb
        self.causal_attention = causal_attention
        self.pos_smoothing = pos_smoothing
        self.max_targets_per_user = max_targets_per_user #only used with sparse positives
        self.full_target = full_target,
        self.vanilla_num_negatives = vanilla_num_negatives 
        self.vanilla_target_sampler = vanilla_target_sampler
        self.vanilla_bce_t = vanilla_bce_t
        self.pq_m = pq_m
        self.centroid_strategy = centroid_strategy

    def as_dict(self):
        result = self.__dict__
        return result
    
    def get_model_architecture(self) -> Type[SequentialRecsysModel]:
        return SASJPQModel


    
