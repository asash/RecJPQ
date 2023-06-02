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
 
def sasjpq(embedding_size, m, num_samples=1):
    from aprec.recommenders.sequential.models.recjpq.sasjpq import SASJPQConfig
    from aprec.recommenders.sequential.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
    model_config = SASJPQConfig(embedding_size=embedding_size, vanilla_num_negatives=num_samples, pq_m=m)
    return sasrec_style_model(model_config, 
            ShiftedSequenceSplitter,
            target_builder=lambda: PositivesSequenceTargetBuilder(SEQUENCE_LENGTH),
            batch_size=128)

def bertjpq(embedding_size, m):
    from aprec.recommenders.sequential.models.recjpq.bertjpq import BertJPQConfig
    model_config = BertJPQConfig(embedding_size=embedding_size, pq_m=m)
    return bert_style_model(model_config, 0) 

def full_bert(embedding_size):
        from aprec.recommenders.sequential.models.bert4rec.full_bert import FullBERTConfig
        model_config =  FullBERTConfig(embedding_size=embedding_sizes)
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
                                eval_batch_size=512, #no need for gradients, should work ok
                                validation_batch_size=512,
                                max_batches_per_epoch=256,
                                sequence_splitter=sequence_splitter, 
                                targets_builder=target_builder, 
                                use_keras_training=True,
                                extra_val_metrics=EXTRA_VAL_METRICS, 
                                sequence_length=SEQUENCE_LENGTH
                                )
    
    return SequentialRecommender(config)

recommenders_list = []

embedding_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
pq_ms=  [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


for emb in embedding_sizes:
    recommenders_list.append((f"bert4rec-emb:{emb}", lambda e=emb: full_bert(embedding_size=e)))
    for m in pq_ms:
        if m <= emb:
            recommenders_list.append((f"bert4recjpq-emb:{emb}-pqm:{m}", lambda e=emb, pqm=m: bertjpq(embedding_size=e, m=pqm)))
            
random.seed(31337)           
random.shuffle(recommenders_list)
recommenders = dict(recommenders_list)
 
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

