import unittest
from aprec.losses.softmax_crossentropy import SoftmaxCrossEntropy

from aprec.recommenders.sequential.models.recjpq.grujpq import GRU4RecJPQConfig

class TestBertJPQ(unittest.TestCase):
    def test_bert4rec(self):
        from aprec.api.items_ranking_request import ItemsRankingRequest
        from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.utils.generator_limit import generator_limit
        from aprec.datasets.movielens20m import get_movielens20m_actions, get_movies_catalog

        USER_ID = '120'

        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        embedding_size=32
        model_config =  GRU4RecJPQConfig(embedding_size=embedding_size)
        recommender_config = SequentialRecommenderConfig(model_config, train_epochs=10, early_stop_epochs=5,
                                               batch_size=5, training_time_limit=10, loss=SoftmaxCrossEntropy(), 
                                               sequence_length=10)
        recommender = SequentialRecommender(recommender_config)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        ranking_request = ItemsRankingRequest('120', ['608', '294', '648'])
        recommender.add_test_items_ranking_request(ranking_request)
        batch1 = [('120', None), ('10', None)]
        recs = recommender.recommender.recommend_multiple(batch1, 10)        
        catalog = get_movies_catalog()
        for rec in recs[0]:
            print(catalog.get_item(rec[0]), "\t", rec[1])

if __name__ == "__main__":
    unittest.main()