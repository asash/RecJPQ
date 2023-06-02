from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.jpq.common_benchmark_config_all_models_sas import *

DATASET = "gowalla_warm5"
N_VAL_USERS=1024
MAX_TEST_USERS=86168
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=False)

if __name__ == "__main__":

    from aprec.tests.misc.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)
