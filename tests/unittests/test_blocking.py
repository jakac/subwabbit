import copy
import pytest
import random
from unittest.mock import Mock, MagicMock, call, patch

from subwabbit.base import VowpalWabbitDummyFormatter
from subwabbit.blocking import VowpalWabbitProcess


@pytest.mark.parametrize(
    'return_predictions_batch',
    [
        [[1]],
        [[1, 2]],
        [[1, 2], [3]],
        [[1, 2], [3, 4]],
        [[1, 2], [3, 4], [5]],
        [[1, 2], [3, 4], [5, 6]]
    ],
    ids=[
        'Batch has less values than batch size',
        'Batch has same length as batch',
        'One and a half batches',
        'Two batches',
        'Two and a half batches',
        'Three batches'
    ]
)
def test_predict_without_timeout(return_predictions_batch):
    batch_size = 2
    num_items = sum(len(batch) for batch in return_predictions_batch)
    return_predictions_batch_copy = return_predictions_batch.copy()

    formatter = VowpalWabbitDummyFormatter()
    common_features = '|a user1'
    items_features = ['|b item{}'.format(i) for i in range(num_items)]

    self = Mock(
        formatter=formatter,
        batch_size=batch_size,
        write_only=False,
        _send_lines_to_vowpal=Mock(),
        _get_predictions_from_vowpal=Mock(side_effect=lambda detailed_metrics, debug_info: return_predictions_batch_copy.pop(0))
    )

    detailed_metrics = MagicMock()
    predictions = list(VowpalWabbitProcess.predict(self, common_features, iter(items_features),
                                                   detailed_metrics=detailed_metrics))
    for i, performed_call in enumerate(self._send_lines_to_vowpal.mock_calls):
        items_from = i * batch_size
        items_to = i * batch_size + batch_size
        assert performed_call == call(
            [formatter.get_formatted_example(common_features, item_features) for item_features in items_features[items_from:items_to]],
            detailed_metrics, debug_info=None
        )

    assert predictions == [prediction for batch in return_predictions_batch for prediction in batch]


@pytest.mark.parametrize(
    'return_predictions_batch, expected_predictions, timeout_after_item',
    [
        ([[1, 2], [3, 4], [5, 6]], [], 0),
        ([[1, 2], [3, 4], [5, 6]], [], 1),  # no prediction is provided because there was no batch in progress in the moment of timeout
        ([[1, 2], [3, 4], [5, 6]], [1, 2], 2),  # 2 predictions are returned
        ([[1, 2], [3, 4], [5, 6]], [1, 2, 3, 4, 5, 6], 8)  # all predictions are returned
    ],
    ids=[
        'Timeout immediately',
        'Timeout after first item',
        'Timeout after two items - ',
        'All items in time'
    ]
)
def test_predict_with_timeout(return_predictions_batch, expected_predictions, timeout_after_item):
    batch_size = 2
    num_items = sum(len(batch) for batch in return_predictions_batch)
    return_predictions_batch_copy = return_predictions_batch.copy()

    formatter = VowpalWabbitDummyFormatter()
    common_features = '|a user1'
    items_features = ['|b item{}'.format(i) for i in range(num_items)]

    processed_items = -1

    def perf_counter_side_effect():
        if processed_items >= timeout_after_item:
            return 1
        else:
            return 0

    perf_counter_mock = Mock(
        side_effect=perf_counter_side_effect
    )

    def get_items_iterator(items):
        nonlocal processed_items
        for item in items:
            processed_items += 1
            yield item

    self = Mock(
        formatter=formatter,
        batch_size=batch_size,
        write_only=False,
        _send_lines_to_vowpal=Mock(),
        _get_predictions_from_vowpal=Mock(side_effect=lambda detailed_metrics, debug_info: return_predictions_batch_copy.pop(0))
    )

    detailed_metrics = MagicMock()
    with patch('subwabbit.blocking.time.perf_counter', new=perf_counter_mock):
        predictions = list(VowpalWabbitProcess.predict(self, common_features, get_items_iterator(items_features),
                                                       timeout=0.5,
                                                       detailed_metrics=detailed_metrics))
    for i, performed_call in enumerate(self._send_lines_to_vowpal.mock_calls):
        items_from = i * batch_size
        items_to = i * batch_size + batch_size
        assert performed_call == call(
            [formatter.get_formatted_example(common_features, item_features) for item_features in items_features[items_from:items_to]],
            detailed_metrics, debug_info=None
        )

    assert predictions == expected_predictions


@pytest.mark.parametrize(
    'return_predictions_batch',
    [
        [[1]],
        [[1, 2]],
        [[1, 2], [3]],
        [[1, 2], [3, 4]],
        [[1, 2], [3, 4], [5]],
        [[1, 2], [3, 4], [5, 6]]
    ],
    ids=[
        'Batch has less values than batch size',
        'Batch has same length as batch',
        'One and a half batches',
        'Two batches',
        'Two and a half batches',
        'Three batches'
    ]
)
def test_predict_io_calls(return_predictions_batch):
    batch_size = 2
    num_items = sum(len(batch) for batch in return_predictions_batch)
    return_predictions_batch_copy = copy.deepcopy(return_predictions_batch)

    def get_next_prediction():
        if return_predictions_batch_copy[0]:
            return str(return_predictions_batch_copy[0].pop(0))
        else:
            return_predictions_batch_copy.pop(0)
            return get_next_prediction()

    formatter = VowpalWabbitDummyFormatter()

    common_features = '|a user1'
    items_features = ['|b item{}'.format(i) for i in range(num_items)]

    vw_process = Mock(
        stdin=Mock(),
        stdout=Mock(
            readline=Mock(side_effect=lambda: bytes(get_next_prediction() + '\n', encoding='utf-8'))
        )
    )
    popen = Mock(
        return_value=vw_process
    )
    with patch('subwabbit.blocking.subprocess.Popen', new=popen):
        model = VowpalWabbitProcess(
            formatter=formatter,
            batch_size=batch_size,
            vw_args=[]
        )

        predictions = list(model.predict(common_features, iter(items_features)))
        expected_calls = []
        for i, item_features in enumerate(return_predictions_batch):
            items_from = i * batch_size
            items_to = i * batch_size + batch_size
            expected_calls.append(
                call.write(
                    bytes(
                        '\n'.join([formatter.get_formatted_example(common_features, item_features) for item_features in
                                   items_features[items_from:items_to]]) + '\n',
                        encoding='utf-8'
                    )
                )
            )
            expected_calls.append(call.flush())
        vw_process.stdin.assert_has_calls(expected_calls)

        assert predictions == [prediction for batch in return_predictions_batch for prediction in batch]
        assert model.unprocessed_batch_sizes == []


@pytest.mark.parametrize(
    'return_predictions_batch',
    [
        [[1, 2], [3, 4], [5]],
        [[1, 2], [3, 4], [5, 6]]
    ],
    ids=[
        'Last batch is not full',
        'Last batch is full'
    ]
)
def test_train(return_predictions_batch):
    batch_size = 2
    num_items = sum(len(batch) for batch in return_predictions_batch)
    return_predictions_batch_copy = copy.deepcopy(return_predictions_batch)

    formatter = VowpalWabbitDummyFormatter()

    common_features = '|a user1'
    items_features = ['|b item{}'.format(i) for i in range(num_items)]
    weights = [random.random() for _ in range(num_items)]
    labels = [random.random() for _ in range(num_items)]

    def get_next_prediction():
        if return_predictions_batch_copy[0]:
            return str(return_predictions_batch_copy[0].pop(0))
        else:
            return_predictions_batch_copy.pop(0)
            return get_next_prediction()

    vw_process = Mock(
        stdin=Mock(),
        stdout=Mock(
            readline=Mock(side_effect=lambda: bytes(get_next_prediction() + '\n', encoding='utf-8'))
        )
    )
    popen = Mock(
        return_value=vw_process
    )
    with patch('subwabbit.blocking.subprocess.Popen', new=popen):
        model = VowpalWabbitProcess(
            formatter=formatter,
            batch_size=batch_size,
            vw_args=[]
        )
        assert model.vw_process == vw_process
        model.train(common_features, iter(items_features), iter(labels), iter(weights))

        expected_calls = []
        for i, item_features in enumerate(return_predictions_batch):
            items_from = i * batch_size
            items_to = i * batch_size + batch_size
            expected_calls.append(
                call.write(
                    bytes(
                        '\n'.join([
                            formatter.get_formatted_example(common_features, item_features, label, weight)
                            for item_features, label, weight in zip(
                                items_features[items_from:items_to],
                                labels[items_from:items_to],
                                weights[items_from:items_to])
                        ]) + '\n',
                        encoding='utf-8'
                    )
                )
            )
            expected_calls.append(call.flush())
        vw_process.stdin.assert_has_calls(expected_calls)
        assert model.unprocessed_batch_sizes == []


@pytest.mark.parametrize(
    'return_predictions_batch',
    [
        [[1, 2], [3, 4], [5]],
        [[1, 2], [3, 4], [5, 6]]
    ],
    ids=[
        'Last batch is not full',
        'Last batch is full'
    ]
)
def test_train_write_only(return_predictions_batch):
    batch_size = 2
    num_items = sum(len(batch) for batch in return_predictions_batch)

    formatter = VowpalWabbitDummyFormatter()

    common_features = '|a user1'
    items_features = ['|b item{}'.format(i) for i in range(num_items)]
    weights = [random.random() for _ in range(num_items)]
    labels = [random.random() for _ in range(num_items)]

    vw_process = Mock(
        stdin=Mock(),
        stdout=Mock()
    )
    popen = Mock(
        return_value=vw_process
    )
    with patch('subwabbit.blocking.subprocess.Popen', new=popen):
        model = VowpalWabbitProcess(
            formatter=formatter,
            batch_size=batch_size,
            write_only=True,
            vw_args=[]
        )
        assert model.vw_process == vw_process
        model.train(common_features, iter(items_features), iter(labels), iter(weights))

        expected_calls = []
        for i, item_features in enumerate(return_predictions_batch):
            items_from = i * batch_size
            items_to = i * batch_size + batch_size
            expected_calls.append(
                call.write(
                    bytes(
                        '\n'.join([
                            formatter.get_formatted_example(common_features, item_features, label, weight)
                            for item_features, label, weight in zip(
                                items_features[items_from:items_to],
                                labels[items_from:items_to],
                                weights[items_from:items_to])
                        ]) + '\n',
                        encoding='utf-8'
                    )
                )
            )
            expected_calls.append(call.flush())
        vw_process.stdin.assert_has_calls(expected_calls)
        vw_process.stdout.assert_not_called()
