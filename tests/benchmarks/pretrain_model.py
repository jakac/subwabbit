import random

from subwabbit import VowpalWabbitProcess

from benchmark_utils import BenchmarkFormatter, load_dataset


if __name__ == '__main__':
    requests, items = load_dataset()

    model = VowpalWabbitProcess(
        formatter=BenchmarkFormatter(),
        vw_args=[
            '--bit_precision', '25',
            '--loss_function', 'logistic',
            '--link', 'logistic',
            '--l2', '1e-06',
            '-l', '0.1',
            '-q', 'ad', '-q', 'ae', '-q', 'af', '-q', 'ag', '-q', 'ah', '-q', 'ai',
            '-q', 'bd', '-q', 'be', '-q', 'bf', '-q', 'bg', '-q', 'bh', '-q', 'bi',
            '-q', 'cd', '-q', 'ce', '-q', 'cf', '-q', 'cg', '-q', 'ch', '-q', 'ci',
            '--progress', str(len(items)),
            '--passes', '1',
            '--preserve_performance_counters',
            '--holdout_off',
            '--kill_cache',
            '--final_regressor', 'model.vw'
        ],
        write_only=True
    )

    random.seed(42)
    for r in requests:
        model.train(
            common_features=r,
            items_features=items,
            labels=[-1.0 if random.random() < 0.5 else 1.0 for _ in range(len(items))],
            weights=[1.0 for _ in range(len(items))]
        )
