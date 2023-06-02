from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.jpq.common_benchmark_config_jpq import *

DATASET = "Amazon.Books_warm5"
N_VAL_USERS=512
MAX_TEST_USERS=100000
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)

if __name__ == "__main__":

    from aprec.tests.misc.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)