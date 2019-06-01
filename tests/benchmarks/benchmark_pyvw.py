"""
Please run:
```bash
    pip install vowpalwabbit
```
"""
from time import perf_counter

from vowpalwabbit import pyvw

from subwabbit import VowpalWabbitBaseModel

from benchmark_utils import BenchmarkFormatter, load_dataset, PREDICTION_TIMEOUT, run_benchmark_single_process


class PyVWModel(VowpalWabbitBaseModel):

    def __init__(self, formatter, vw_args):
        """
        :param vw_args: list of command line arguments for vw command, eg. ['-q', '::']
        """
        super().__init__(formatter)
        self.vw_model = pyvw.vw(' '.join(vw_args))

    def predict(self, common_features, items_features, timeout=None):
        please_respond_to = None
        if timeout:
            please_respond_to = perf_counter() + timeout
        common_line_part = self.formatter.get_common_line_part(common_features)
        for item_features in items_features:
            if please_respond_to is not None and perf_counter() > please_respond_to:
                break
            item_line_part = self.formatter.get_item_line_part(common_features, item_features)
            vw_line = self.formatter.get_vw_line(common_line_part, item_line_part)
            example = self.vw_model.example(vw_line)
            yield self.vw_model.predict(example)

    def train(self, common_features, items_features, labels, weights, debug_info=None):
        raise NotImplementedError()

    def explain_vw_line(self, vw_line, link_function=False):
        raise NotImplementedError()


if __name__ == '__main__':
    run_benchmark_single_process(PyVWModel)
