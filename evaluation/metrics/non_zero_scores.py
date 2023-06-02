from aprec.evaluation.metrics.metric import Metric


class NonZeroScores(Metric):
    def __init__(self):
        self.name = "NonZeroScores"
        
    def __call__(self, recommendations, actual_actions):
        count = 0
        for (item, score) in recommendations:
            if score > 0:
                count += 1
        return count
    