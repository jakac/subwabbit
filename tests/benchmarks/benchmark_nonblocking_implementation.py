from subwabbit import VowpalWabbitNonBlockingProcess

from benchmark_utils import BenchmarkFormatter, load_dataset, PREDICTION_TIMEOUT, run_benchmark_single_process


if __name__ == '__main__':
    run_benchmark_single_process(VowpalWabbitNonBlockingProcess)
