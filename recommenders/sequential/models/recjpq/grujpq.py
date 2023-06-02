from __future__ import annotations
from typing import List, Type

import tensorflow as tf
from aprec.recommenders.sequential.models.recjpq.rec_jpq_layer import ItemCodeLayer
from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialDataParameters, SequentialModelConfig, SequentialRecsysModel

#https://arxiv.org/abs/1511.06939
class GRU4RecJPQConfig(SequentialModelConfig):
    def __init__(self,
                 output_layer_activation='linear',
                 embedding_size=64,
                 num_gru_layers=3,
                 num_dense_layers=0, #we also add 1 extra dense layer with linear projection before computing final output, so -1 compared to GRU4Rec
                 pq_m = 4,
                 centroid_strategy='svd',
                 activation='relu'):
        self.output_layer_activation = output_layer_activation
        self.embedding_size = embedding_size
        self.num_gru_layers = num_gru_layers
        self.num_dense_layers = num_dense_layers
        self.pq_m = pq_m
        self.activation = activation
        self.centroid_strategy = centroid_strategy

    def get_model_architecture(self) -> Type[GRU4RecJPQ]:
        return GRU4RecJPQ 
    
    def as_dict(self) -> dict:
        return self.__dict__


class GRU4RecJPQ(SequentialRecsysModel):
    def __init__(self, model_parameters: GRU4RecJPQConfig, data_parameters: SequentialDataParameters, *args, **kwargs):
        super().__init__(model_parameters, data_parameters, *args, **kwargs)    
        self.model_parameters: GRU4RecJPQConfig
        self.item_codes_layer = ItemCodeLayer(self.model_parameters.embedding_size, 
                                              self.model_parameters.pq_m, 
                                              self.data_parameters.num_items, 
                                              self.data_parameters.sequence_length,
                                              self.model_parameters.centroid_strategy)

        layers = tf.keras.layers
        self.gru_layers = []
        for _ in range(self.model_parameters.num_gru_layers - 1):
            self.gru_layers.append(layers.GRU(self.model_parameters.embedding_size, activation='tanh', return_sequences=True))
        self.gru_layers.append(layers.GRU(self.model_parameters.embedding_size, activation='tanh', return_sequences=False))

        self.dense_layers = []
        for i in range (self.model_parameters.num_dense_layers):
            self.dense_layers.append(layers.Dense(self.model_parameters.embedding_size, activation=self.model_parameters.activation))
        self.output_layer = layers.Dense(self.model_parameters.embedding_size, name="output", activation=self.model_parameters.output_layer_activation)

    def fit_biases(self, train_users):
        self.item_codes_layer.assign_codes(train_users)

    @classmethod
    def get_model_config_class(cls) -> Type[GRU4RecJPQConfig]:
        return GRU4RecJPQConfig

    def apply_network(self, x):
        for gru_layer in self.gru_layers:
            x = gru_layer(x)
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        x = self.output_layer(x)
        result = self.item_codes_layer.score_all_items(x)
        return result

    def call(self, inputs, **kwargs):
        x = inputs[0]
        x = self.item_codes_layer(x, self.data_parameters.batch_size) 
        return self.apply_network(x)

    def score_all_items(self, inputs, **kwargs):
        x = inputs[0]
        x = self.item_codes_layer(x, x.shape[0]) #embeddings
        return self.apply_network(x)
         
    def get_dummy_inputs(self) -> List[tf.Tensor]:
        input = tf.zeros((self.data_parameters.batch_size, self.data_parameters.sequence_length), 'int32')
        return [input]