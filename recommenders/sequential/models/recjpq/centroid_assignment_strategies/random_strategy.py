from .centroid_strategy import CentroidAssignmentStragety
import numpy as np

class RandomAssignmentStrategy(CentroidAssignmentStragety):
    def assign(self, train_users):
        assignments = []
        print("done")
        for i in range(self.item_code_bytes):
            component_assignments = np.random.randint(0, 256, self.num_items + 2) # +2 for padding and mask
            assignments.append(component_assignments)
        return np.transpose(np.array(assignments))


