import sys
import gzip
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from collections import defaultdict
from analyze_experiment_in_progress import get_data_from_logs

baselines = ["top", "mf-bpr", "SASRec-vanilla",  "bert4rec-1h", "bert4rec-16h"]
controls = ["Sasrec-rss-bce", "Sasrec-rss-lambdarank"]
control_symbols = {"Sasrec-rss-lambdarank": '*', "Sasrec-rss-bce": "†"}

datasets = {
    "Yelp": ["yelp_benchmark_2022_04_08T08_58_25", "yelp_benchmark_bert4rec_16h_2022_04_10T17_27_18"], 
    "Gowalla": ["gowalla_benchmark_2022_04_17T07_37_46"],
    "Booking": ["booking_benchmark_2022_04_14T08_34_13", "booking_benchmark_bert4rec16h_2022_04_16T14_19_52"],
    "Movielens": ["ml_benchmark20m_bert4rec16h_2022_04_13T07_57_13", "ml_benchmark20m_2022_04_11T14_28_35"]
}

metrics = ["HIT@10", "ndcg@10"]
limit_per_file = None


def get_models(names, experiments):
    result = {}
    for experiment in experiments:
        dir = os.path.join("results", experiment, "predictions")
        for name in names:
            full_predictions_name = os.path.join(dir, name) + ".json.gz"
            if os.path.isfile(full_predictions_name):
               result[name] = full_predictions_name
    return result


def read_predictions_file(pred):
    lines_read = 0
    result = defaultdict(list)
    for line in gzip.open(pred):
        if limit_per_file is not None and lines_read >= limit_per_file:
            break
        data = json.loads(line)
        doc = json.loads(line)['metrics']
        for metric in metrics:
            result[metric].append(doc[metric])
        lines_read += 1
    return dict(result)


for dataset in datasets:
    print(f"analyzing dataset {dataset}")
    control_models = get_models(controls, datasets[dataset])
    baseline_models = get_models(baselines, datasets[dataset])
    model_metrics = {}
    for control in control_models:
        model_metrics[control] = read_predictions_file(control_models[control])

    for baseline in baseline_models:
        model_metrics[baseline] = read_predictions_file(baseline_models[baseline])

    result = []

    for model in baselines+controls:
        doc = {"model_name": model}
        for metric in metrics:
            if model in model_metrics:
                doc[metric] = "{:.4f}".format(float(np.mean(model_metrics[model][metric])))
                for control in controls:
                    t, pval = ttest_ind(model_metrics[model][metric], model_metrics[control][metric]) 
                    pval *= len(metrics) #Bonferoni correction
                    if pval < 0.05:
                        doc[metric] += control_symbols[control]

            else:
                doc[metric] = "N/A"
        result.append(doc)
    df = pd.DataFrame(result)
    print(pd.DataFrame(result))
    df.to_csv(dataset + "_rss.csv")

