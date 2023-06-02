from aprec.evaluation.metrics.entropy import Entropy
from aprec.evaluation.metrics.highest_score import HighestScore
from aprec.evaluation.metrics.model_confidence import Confidence
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.sequential.target_builders.positives_sequence_target_builder import PositivesSequenceTargetBuilder
import random

USERS_FRACTIONS = [1.0]

EXTRA_VAL_METRICS = [NDCG(10), HighestScore(), NDCG(40), HIT(1), MRR(), 
                     Confidence('Softmax'),
                     Confidence('Sigmoid'),
                     Entropy('Softmax', 10), 
                     HIT(10)]

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]
#TARGET_ITEMS_SAMPLER = PopTargetItemsWithReplacementSampler(101)

SEQUENCE_LENGTH=200


#SASREC MODELS 
def sasjpq(embedding_size, m, num_samples=1, strategy='svd'):
    from aprec.recommenders.sequential.models.recjpq.sasjpq import SASJPQConfig
    from aprec.recommenders.sequential.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
    model_config = SASJPQConfig(embedding_size=embedding_size, vanilla_num_negatives=num_samples, pq_m=m, centroid_strategy=strategy)
    return sasrec_style_model(model_config, 
            ShiftedSequenceSplitter,
            target_builder=lambda: PositivesSequenceTargetBuilder(SEQUENCE_LENGTH),
            batch_size=128)

def vanilla_sasrec(embedding_size, loss='bce', num_samples=1, batch_size=128):
    from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
    from aprec.recommenders.sequential.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
    model_config = SASRecConfig(vanilla=True, embedding_size=embedding_size, loss=loss, vanilla_num_negatives=num_samples)
    return sasrec_style_model(model_config, 
            ShiftedSequenceSplitter,
            target_builder=lambda: PositivesSequenceTargetBuilder(SEQUENCE_LENGTH),
            batch_size=batch_size)

def sasrec_style_model(model_config, sequence_splitter, 
                target_builder,
                max_epochs=10000, 
                batch_size=128,
                ):
    from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
    from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig

    config = SequentialRecommenderConfig(model_config,                       
                                train_epochs=max_epochs,
                                early_stop_epochs=200,
                                batch_size=batch_size,
                                eval_batch_size=256, #no need for gradients, should work ok
                                validation_batch_size=256,
                                max_batches_per_epoch=256,
                                sequence_splitter=sequence_splitter, 
                                targets_builder=target_builder, 
                                use_keras_training=True,
                                extra_val_metrics=EXTRA_VAL_METRICS, 
                                sequence_length=SEQUENCE_LENGTH
                                )
    
    return SequentialRecommender(config)

#BERT4Rec MODELS
def bertjpq(embedding_size, m, strategy='svd'):
    from aprec.recommenders.sequential.models.recjpq.bertjpq import BertJPQConfig
    model_config = BertJPQConfig(embedding_size=embedding_size, pq_m=m, centroid_strategy=strategy)
    return bert_style_model(model_config, 0) 

def full_bert(embedding_size):
        from aprec.recommenders.sequential.models.bert4rec.full_bert import FullBERTConfig
        model_config =  FullBERTConfig(embedding_size=embedding_size)
        return bert_style_model(model_config, tuning_samples_portion=0.0)
 
def bert_style_model(model_config, tuning_samples_portion, batch_size=8):
        import tensorflow as tf
        from aprec.recommenders.sequential.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.recommenders.sequential.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
        from aprec.recommenders.sequential.targetsplitters.items_masking import ItemsMasking
        recommender_config = SequentialRecommenderConfig(model_config, 
                                               train_epochs=10000, early_stop_epochs=200,
                                               batch_size=batch_size,
                                               eval_batch_size=256, #no need for gradients, should work ok
                                               validation_batch_size=256,
                                               sequence_splitter=lambda: ItemsMasking(tuning_samples_prob=tuning_samples_portion), 
                                               max_batches_per_epoch=1024,
                                               targets_builder=ItemsMaskingTargetsBuilder,
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               use_keras_training=True,
                                               sequence_length=SEQUENCE_LENGTH,
                                               optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                                               extra_val_metrics = EXTRA_VAL_METRICS)
        
        return SequentialRecommender(recommender_config)

        
#GRU4Rec MODELS
def grujpq(embedding_size, m, strategy='svd'):
    from aprec.recommenders.sequential.models.recjpq.grujpq import GRU4RecJPQConfig
    model_config = GRU4RecJPQConfig(embedding_size=embedding_size, pq_m=m, centroid_strategy=strategy)
    return gru_style_model(model_config) 

    
def full_gru(embedding_size):
    from aprec.recommenders.sequential.models.gru4rec import GRU4RecConfig
    model_config = GRU4RecConfig(embedding_size=embedding_size)
    return gru_style_model(model_config)


def gru_style_model(model_config, batch_size=128, max_epochs=10000):
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
        from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
        recommender_config = SequentialRecommenderConfig(model_config, train_epochs=max_epochs,
                                early_stop_epochs=200,
                                batch_size=batch_size,
                                eval_batch_size=256, #no need for gradients, should work ok
                                validation_batch_size=256,
                                max_batches_per_epoch=256,
                                use_keras_training=True,
                                extra_val_metrics=EXTRA_VAL_METRICS, 
                                sequence_length=SEQUENCE_LENGTH,
                                loss = LambdaGammaRankLoss(pred_truncate_at=3000))
        return SequentialRecommender(recommender_config)
recommenders = {}

for strategy in ('random', 'svd', 'bpr'):
    recommenders[f'grujpq_{strategy}'] = lambda embedding_size=512, m=8, strategy=strategy: grujpq(embedding_size, m, strategy=strategy)    
    recommenders[f'sasrec_{strategy}'] = lambda embedding_size=512, m=8, strategy=strategy: sasjpq(embedding_size, m, strategy=strategy)    
    recommenders[f'bert4rec_{strategy}'] = lambda embedding_size=512, m=8, strategy=strategy: bertjpq(embedding_size, m, strategy=strategy)    

recommenders['sasrec'] = lambda embedding_size=512: vanilla_sasrec(embedding_size)
recommenders['bert4rec'] = lambda embedding_size=512: full_bert(embedding_size)
recommenders['gru4rec'] = lambda embedding_size=512: full_gru(embedding_size)

recommenders["sasrec_qr"] = lambda embedding_size=512: sasjpq(embedding_size, 2 , strategy="qr")
recommenders["bert4rec_qr"] = lambda embedding_size=512: bertjpq(embedding_size, 2 , "qr")
recommenders["gru4rec_qr"] = lambda embedding_size=512: grujpq(embedding_size, 2 , "qr")

def get_recommenders(filter_seen: bool):
    result = {}
    all_recommenders = list(recommenders.keys())
    for recommender_name in all_recommenders:
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result

