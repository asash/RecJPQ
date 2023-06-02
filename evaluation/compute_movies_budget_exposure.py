import gzip
import sys
from aprec.datasets.movies_dataset import get_movies_budget_bands

class ExposureDistributionCalculator:
    def __init__(self, item_attributes, k=10):
        self.item_attributes = item_attributes
        k = k
        pass
    

if __name__ == "__main__":
    item_attributes = get_movies_budget_bands()
    exposure_distribution_calculator = ExposureDistributionCalculator(item_attributes, k=10)
    predictions_file = gzip.open(sys.argv[1])
    pass
    