from __future__ import annotations
from aprec.recommenders.sequential.models.generative.tokenizer import Tokenizer
from aprec.recommenders.sequential.models.recjpq.centroid_assignment_strategies.svd_strategy import SVDAssignmentStrategy
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import KBinsDiscretizer
import tensorflow as tf
import numpy as np



class SVDTokenizer(Tokenizer):
    def __init__(self, tokens_per_item, values_per_dimension, num_items) -> None:
        super().__init__(tokens_per_item, values_per_dimension, num_items)

    def assign(self, train_users):
        rows = []
        cols = []
        vals = []
        for i in range(len(train_users)):
            for j in range(len(train_users[i])):
                rows.append(i)
                cols.append(train_users[i][j][1])
                vals.append(1)
        matr = csr_matrix((vals, [rows, cols]), shape=(len(train_users), self.num_items+1))
        print("fitting svd tokenizer")
        svd = TruncatedSVD(n_components=int(self.tokens_per_item))
        svd.fit(matr)
        item_embeddings = svd.components_
        assignments = []
        print("done")
        for i in range(int(self.tokens_per_item)):
            discretizer = KBinsDiscretizer(n_bins=int(self.values_per_dimension), encode='ordinal', strategy='quantile')
            ith_component = item_embeddings[i:i+1][0]
            ith_component = (ith_component - np.min(ith_component))/(np.max(ith_component) - np.min(ith_component) + 1e-10)
            noise = np.random.normal(0, 1e-5, self.num_items + 1)
            ith_component += noise # make sure that every item has unique value
            ith_component = np.expand_dims(ith_component, 1)
            component_assignments = discretizer.fit_transform(ith_component).astype('int32')[:,0] + self.values_per_dimension*i
            assignments.append(component_assignments)
        result = np.array(assignments).T
        result[-1] = -100 #padding
        self.vocabulary.assign(result)
        