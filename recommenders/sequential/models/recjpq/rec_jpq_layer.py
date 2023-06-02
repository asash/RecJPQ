import math
import tensorflow as tf

from aprec.recommenders.sequential.models.recjpq.centroid_assignment_strategies.bpr_strategy import BPRAssignmentStrategy
from aprec.recommenders.sequential.models.recjpq.centroid_assignment_strategies.qr_strategy import QuotientRemainder
from aprec.recommenders.sequential.models.recjpq.centroid_assignment_strategies.random_strategy import RandomAssignmentStrategy
from .centroid_assignment_strategies.centroid_strategy import CentroidAssignmentStragety
from .centroid_assignment_strategies.svd_strategy import SVDAssignmentStrategy

def get_codes_strategy(codes_strategy, item_code_bytes, num_items) -> CentroidAssignmentStragety:
    if codes_strategy == "svd":
        return SVDAssignmentStrategy(item_code_bytes, num_items)
    if codes_strategy == "bpr":
        return BPRAssignmentStrategy(item_code_bytes, num_items)
    if codes_strategy == "random":
        return RandomAssignmentStrategy(item_code_bytes, num_items) 
    if codes_strategy == "qr":
        return QuotientRemainder(item_code_bytes, num_items)
    
        
class ItemCodeLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, pq_m, num_items, sequence_length, codes_strategy):
        super().__init__()

        
        self.sub_embedding_size = embedding_size // pq_m
        self.item_code_bytes = embedding_size // self.sub_embedding_size
        item_initializer = tf.zeros_initializer()
        if codes_strategy != "qr":
            self.base_type = 'uint8' 
            self.vals_per_dim = 256
        else:
            self.base_type = 'int32'
            self.vals_per_dim = math.ceil(math.sqrt(num_items))

        self.item_codes = tf.Variable(item_initializer((num_items + 2, self.item_code_bytes), dtype=self.base_type), 
                                      trainable=False, name="ItemCodes/codes") # +2 for padding and mask

        centroid_initializer = tf.random_uniform_initializer()
        self.centroids = tf.Variable(centroid_initializer(shape=(self.item_code_bytes, self.vals_per_dim,
                                                                 self.sub_embedding_size)),
                                     name="ItemCodes/centroids")
        self.item_codes_strategy = get_codes_strategy(codes_strategy, self.item_code_bytes, num_items)
        self.sequence_length = sequence_length
        self.num_items = num_items

    def assign_codes(self, train_users):
        codes = self.item_codes_strategy.assign(train_users)
        self.item_codes.assign(codes)

    def call(self, input_ids, batch_size): #use instead of item embeddings
        input_codes = tf.stop_gradient(tf.cast(tf.gather(self.item_codes, input_ids), 'int32'))
        code_byte_indices = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(0, self.item_code_bytes), 0), 0), [batch_size, self.sequence_length,1])
        n_sub_embeddings = batch_size * self.sequence_length * self.item_code_bytes
        code_byte_indices_reshaped = tf.reshape(code_byte_indices, (n_sub_embeddings, ))
        input_codes_reshaped = tf.reshape(input_codes, (n_sub_embeddings,))
        indices = tf.stack([code_byte_indices_reshaped, input_codes_reshaped], axis=-1)
        input_sub_embeddings_reshaped = tf.gather_nd(self.centroids, indices)
        result = tf.reshape(input_sub_embeddings_reshaped,[batch_size, self.sequence_length, self.item_code_bytes * self.sub_embedding_size] )
        return result

    def score_sequence_items(self, seq_emb, target_ids, batch_size):
        """
            seq_emb - encoded embeddings of items in sequence batch_size x sequence_len x embedding_size 
            target_id - items to score batch_size x sequence_length x num_items_to_score  (ints) 
        """ 
        seq_sub_emb = tf.reshape(seq_emb, [batch_size, self.sequence_length, self.item_code_bytes, self.sub_embedding_size])
        centroid_scores = tf.einsum("bsie,ine->bsin", seq_sub_emb, self.centroids) 
        target_codes =tf.transpose(tf.cast(tf.gather(self.item_codes, tf.nn.relu(target_ids)), 'int32'), [0, 1, 3, 2])
        target_sub_scores = tf.gather(centroid_scores, target_codes, batch_dims=3)
        logits = tf.reduce_sum(target_sub_scores, -2)
        return logits

    def score_sequence_all_items(self, seq_emb, batch_size):
        seq_sub_emb = tf.reshape(seq_emb, [batch_size, self.sequence_length, self.item_code_bytes, self.sub_embedding_size])
        centroid_scores = tf.einsum("bsie,ine->bsin", seq_sub_emb, self.centroids) 
        centroid_scores = tf.transpose(tf.reshape(centroid_scores, [batch_size, self.sequence_length, self.item_code_bytes * self.vals_per_dim]), 
                                       [2, 0, 1])
        target_codes = tf.cast(tf.transpose(self.item_codes[:-2]), 'int32')
        offsets = tf.expand_dims(tf.range(self.item_code_bytes) * self.vals_per_dim, -1)
        target_codes += offsets
        item_sub_scores = tf.gather(centroid_scores, target_codes)
        logits = tf.transpose(tf.reduce_sum(item_sub_scores, axis=0), [1,2,0])
        return logits

 

    def score_all_items(self, seq_emb): #seq_emb: batch_size x embedding_size
        seq_sub_emb = tf.reshape(seq_emb, [seq_emb.shape[0], self.item_code_bytes, self.sub_embedding_size])
        centroid_scores = tf.einsum("bie,ine->bin", seq_sub_emb, self.centroids)
        centroid_scores = tf.transpose(tf.reshape(centroid_scores, 
                                                  [centroid_scores.shape[0], centroid_scores.shape[1] * centroid_scores.shape[2]]))
        target_codes = tf.cast(tf.transpose(self.item_codes[:-2]), 'int32')
        offsets = tf.expand_dims(tf.range(self.item_code_bytes) * self.vals_per_dim, -1)
        target_codes += offsets
        result = tf.zeros((self.num_items, centroid_scores.shape[1]))
        for i in range (self.item_code_bytes):
            result += tf.gather(centroid_scores, target_codes[i])
        return tf.transpose(result)

         


