import json
import os
from time import perf_counter

import pandas as pd

from subwabbit import VowpalWabbitBaseFormatter


PREDICTION_TIMEOUT = 0.01


class BenchmarkFormatter(VowpalWabbitBaseFormatter):

    def format_common_features(self, common_features, debug_info=None):
        return ' '.join([
            '|d {}'.format(' '.join('d{}:{:.2f}'.format(k, v) for k, v in common_features['d'].items())),
            '|e {}'.format(' '.join('e{}:{:.2f}'.format(k, v) for k, v in common_features['e'].items())),
            '|f {}'.format(' '.join('f{}'.format(f) for f in common_features['f'])),
            '|g {}'.format(' '.join('g{}:{:.2f}'.format(k, v) for k, v in common_features['g'].items())),
            '|h h{}'.format(common_features['h']),
            '|i i{}'.format(common_features['i'])
        ])

    def format_item_features(self, common_features, item_features, debug_info=None):
        return ' '.join([
            '|a a{}'.format(item_features['a']),
            '|b b{}'.format(item_features['b']),
            '|c {}'.format(' '.join('c{}'.format(c) for c in item_features['c']))
        ])


def load_dataset():
    with open(os.path.join(os.path.dirname(__name__), 'requests.json')) as f:
        requests = json.load(f)
    with open(os.path.join(os.path.dirname(__name__), 'items.json')) as f:
        items = json.load(f)

    return requests, items


def run_benchmark_single_process(model_class):
    requests, items = load_dataset()

    model = model_class(
        formatter=BenchmarkFormatter(),
        vw_args=[
            '--initial_regressor', 'model.vw',
            '--quiet',
            '-t'
        ])

    results = []
    for r in requests:
        t0 = perf_counter()
        predictions = list(model.predict(common_features=r, items_features=items, timeout=PREDICTION_TIMEOUT))
        prediction_time = perf_counter()
        results.append({
            'Prediction time': prediction_time - t0,
            'Predicted lines': len(predictions)
        })

    results_df = pd.DataFrame(results)

    print(results_df.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]).to_string())