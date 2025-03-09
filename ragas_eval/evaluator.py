from datasets import Dataset
from ragas import evaluate

class Evaluator:
    def __init__(self, data_samples):
        self.data_samples = data_samples
        self.dataset = Dataset.from_dict(data_samples)

    def run_evaluation(self, metrics):
        score = evaluate(self.dataset, metrics=metrics)
        return score.to_pandas()
