from lightfm import LightFM
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import KBinsDiscretizer
from .centroid_strategy import CentroidAssignmentStragety
import numpy as np

class BPRAssignmentStrategy(CentroidAssignmentStragety):
    def assign(self, train_users):
        rows = []
        cols = []
        vals = []
        for i in range(len(train_users)):
            for j in range(len(train_users[i])):
                rows.append(i)
                cols.append(train_users[i][j][1])
                vals.append(1)
        matr = csr_matrix((vals, [rows, cols]), shape=(len(train_users), self.num_items+2)) # +2 for padding and mask
        print("fitting MF-BPR for initial centroids assignments")

        model = LightFM(no_components=self.item_code_bytes, loss='bpr')
        model.fit(matr, epochs=20, verbose=True, num_threads=20)
        item_embeddings = model.get_item_representations()[1].T

        assignments = []
        print("done")
        for i in range(self.item_code_bytes):
            discretizer = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile')
            ith_component = item_embeddings[i:i+1][0]
            ith_component = (ith_component - np.min(ith_component))/(np.max(ith_component) - np.min(ith_component) + 1e-10)
            noise = np.random.normal(0, 1e-5, self.num_items + 2)
            ith_component += noise # make sure that every item has unique value
            ith_component = np.expand_dims(ith_component, 1)
            component_assignments = discretizer.fit_transform(ith_component).astype('uint8')[:,0]
            assignments.append(component_assignments)
        return np.transpose(np.array(assignments))

