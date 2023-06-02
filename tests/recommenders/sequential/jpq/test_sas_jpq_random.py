import os
import unittest

from aprec.recommenders.sequential.target_builders.positives_sequence_target_builder import PositivesSequenceTargetBuilder

class TestSASRecJPQRandom(unittest.TestCase):
    def setUp(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    def test_sasjpq_random(self):
        from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions, get_movies_catalog
        from aprec.recommenders.sequential.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.recommenders.sequential.models.recjpq.sasjpq import SASJPQConfig
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.utils.generator_limit import generator_limit

        USER_ID = '120'
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        model_config = SASJPQConfig(embedding_size=32, vanilla_num_negatives=5, vanilla_bce_t=1, centroid_strategy='random')

        recommender_config = SequentialRecommenderConfig(model_config, train_epochs=10000, early_stop_epochs=50000,
                                               batch_size=5,
                                               training_time_limit=10,  
                                               sequence_splitter=ShiftedSequenceSplitter, 
                                               targets_builder=PositivesSequenceTargetBuilder,
                                               use_keras_training=True
                                               )
        
        recommender = SequentialRecommender(recommender_config)

        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        catalog = get_movies_catalog()
        for rec in recs:
            print(catalog.get_item(rec[0]), "\t", rec[1])

if __name__ == "__main__":
    unittest.main()