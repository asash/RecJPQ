from __future__ import annotations
from aprec.recommenders.sequential.models.generative.tokenizer import Tokenizer
from aprec.recommenders.sequential.models.recjpq.centroid_assignment_strategies.svd_strategy import SVDAssignmentStrategy
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import KBinsDiscretizer
import tensorflow as tf
import numpy as np



class IDTokenizer(Tokenizer):
    def __init__(self, tokens_per_item, values_per_dimension, num_items) -> None:
        super().__init__(tokens_per_item, values_per_dimension, num_items)

        #ensure that tokens per item equals to 1:
        if (tokens_per_item != 1):
            raise ValueError("ID tokenizer only supports tokens_per_item=1")

        #ensure that values_per_dimension is larger than num_items:
        if (values_per_dimension <= num_items):
            raise ValueError("ID tokenizer requires values_per_dimension > num_items")

    def assign(self, train_users):
        assignment = np.arange(self.num_items+1)
        assignment[-1] = -100 #padding
        assignment = np.expand_dims(assignment, 1)
        self.vocabulary.assign(assignment)
        
        
        
        