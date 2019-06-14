import os
import pytest

from subwabbit import VowpalWabbitProcess, VowpalWabbitNonBlockingProcess, VowpalWabbitBaseFormatter


class TestFormatter(VowpalWabbitBaseFormatter):

    def format_common_features(self, user_features, debug_info=None):
        return '|u u{}'.format(user_features['user_id'])

    def format_item_features(self, user_features, item_features, debug_info=None):
        return '|i i{}'.format(item_features['item_id'])


class VWProcessTest(VowpalWabbitProcess):

    batch_size = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class VWNonBlockingProcessTest(VowpalWabbitNonBlockingProcess):

    batch_size = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


user_features1 = {
    'user_id': 1
}

user_features2 = {
    'user_id': 2
}

items_features = [
    {'item_id': 1},
    {'item_id': 2},
    {'item_id': 3}
]


@pytest.mark.parametrize(
    'model_class',
    [
        VWProcessTest,
        VWNonBlockingProcessTest
    ]
)
def test_vwprocess(model_class):
    """
    Test whether results are same as using vw command directly:
    run 'vw -q :: -p /dev/stdout --quiet' command:
0 1.0 |u u1 |i i1
1 1.0 |u u1 |i i2
1 1.0 |u u1 |i i3


1 1.0 |u u2 |i i1
1 1.0 |u u2 |i i2
0 1.0 |u u2 |i i3


|u u1 |i i1
|u u1 |i i2
|u u1 |i i3

|u u2 |i i1
|u u2 |i i2
|u u2 |i i3

0 1.0 |u u1 |i i1
1 1.0 |u u1 |i i2
1 1.0 |u u1 |i i3


1 1.0 |u u2 |i i1
1 1.0 |u u2 |i i2
0 1.0 |u u2 |i i3


|u u1 |i i1
|u u1 |i i2
|u u1 |i i3

|u u2 |i i1
|u u2 |i i2
|u u2 |i i3
    """
    try:
        if os.path.exists('/tmp/testmodel.vw'):
            os.remove('/tmp/testmodel.vw')

        if os.path.exists('/tmp/testmodel2.vw'):
            os.remove('/tmp/testmodel2.vw')

        if os.path.exists('/tmp/testmodel3.vw'):
            os.remove('/tmp/testmodel3.vw')

        vw_process = VWProcessTest(TestFormatter(), ['-q', '::', '--save_resume', '-f', '/tmp/testmodel.vw', '--quiet'],
                                   write_only=True)
        vw_process.train(common_features=user_features1, items_features=items_features, labels=[0, 1, 1], weights=[1.0, 1.0, 1.0])
        vw_process.train(common_features=user_features2, items_features=items_features, labels=[1, 1, 0], weights=[1.0, 1.0, 1.0])
        vw_process.close()

        assert os.path.exists('/tmp/testmodel.vw')

        vw_process = VWProcessTest(TestFormatter(),
                                   ['--save_resume', '-i', '/tmp/testmodel.vw', '-f', '/tmp/testmodel2.vw', '--quiet'])
        predictions = vw_process.predict(common_features=user_features1, items_features=items_features)
        assert list(predictions) == [0.777104, 0.96657, 0.69223]
        predictions = vw_process.predict(common_features=user_features2, items_features=items_features)
        assert list(predictions) == [0.666838, 0.727325, 0.242521]

        vw_process.train(common_features=user_features1, items_features=items_features, labels=[0, 1, 1],
                         weights=[1.0, 1.0, 1.0])
        vw_process.train(common_features=user_features2, items_features=items_features, labels=[1, 1, 0],
                         weights=[1.0, 1.0, 1.0])
        vw_process.close()

        assert os.path.exists('/tmp/testmodel2.vw')

        # 2nd time we test only len(batch) = 1 to test if batches size != batch_size will pass
        vw_process = model_class(TestFormatter(),
                                 ['--save_resume', '-i', '/tmp/testmodel2.vw', '-f', '/tmp/testmodel3.vw', '--quiet'])
        predictions = vw_process.predict(common_features=user_features1, items_features=items_features)
        assert list(predictions) == [0.558472, 1.0, 0.736521]
        vw_process.close()

        assert os.path.exists('/tmp/testmodel3.vw')
    finally:
        if os.path.exists('/tmp/testmodel.vw'):
            os.remove('/tmp/testmodel.vw')

        if os.path.exists('/tmp/testmodel2.vw'):
            os.remove('/tmp/testmodel2.vw')

        if os.path.exists('/tmp/testmodel3.vw'):
            os.remove('/tmp/testmodel3.vw')


def isclose(a, b, abs_tol=0.00001):
    assert len(a) == len(b)
    return [abs(x-y) <= abs_tol for x, y in zip(a, b)]
