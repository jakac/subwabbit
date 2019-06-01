import pytest
from unittest.mock import Mock, MagicMock, call, patch

from subwabbit.base import VowpalWabbitDummyFormatter
from subwabbit.nonblocking import VowpalWabbitNonBlockingProcess


@pytest.mark.parametrize(
    'unwritten_buffer, lines, num_bytes_written, expected_written, expected_unwritten_buffer',
    [
        (
            b'',
            [
                '|a u2 |b u2',
                '|a u3 |b u3',
            ],
            24,
            b'|a u2 |b u2\n|a u3 |b u3\n',
            b''
        ),
        (
            b'|a u1 |b i1\n',
            [
                '|a u2 |b u2',
                '|a u3 |b u3',
            ],
            36,
            b'|a u1 |b i1\n|a u2 |b u2\n|a u3 |b u3\n',
            b''
        ),
        (
            b'',
            [
                '|a u2 |b u2',
                '|a u3 |b u3',
            ],
            12,
            b'|a u2 |b u2\n|a u3 |b u3\n',
            b'|a u3 |b u3\n'
        ),
        (
            b'|a u1 |b i1\n',
            [
                '|a u2 |b u2',
                '|a u3 |b u3',
            ],
            12,
            b'|a u1 |b i1\n|a u2 |b u2\n|a u3 |b u3\n',
            b'|a u2 |b u2\n|a u3 |b u3\n'
        ),
        (
            b'|a u1 |b i1\n',
            [
                '|a u2 |b u2',
                '|a u3 |b u3',
            ],
            None,
            b'|a u1 |b i1\n|a u2 |b u2\n|a u3 |b u3\n',
            b'|a u1 |b i1\n|a u2 |b u2\n|a u3 |b u3\n',
        ),
        (
            b'',
            [],
            0,
            b'',
            b'',
        )
    ],
    ids=[
        'Empty buffer, all written',
        'Nonempty buffer, all written',
        'Empty buffer, partially written',
        'Nonempty buffer, partially written',
        'System buffer is full',
        'Nothing to write'
    ]
)
def test_send_lines_to_vowpal(unwritten_buffer, lines, num_bytes_written, expected_written, expected_unwritten_buffer):
    model = Mock(
        _unwritten_buffer=unwritten_buffer,
        vw_process=Mock(stdin=Mock(write=Mock(return_value=num_bytes_written))),
        _pending_lines=0
    )
    detailed_metrics = {
        'sending_bytes': []
    }
    VowpalWabbitNonBlockingProcess._send_lines_to_vowpal(model, lines, detailed_metrics)
    if lines:
        model.vw_process.stdin.write.assert_called_once_with(expected_written)
        assert detailed_metrics['sending_bytes'][0][1] == (num_bytes_written if num_bytes_written is not None else 0)
    assert model._unwritten_buffer == expected_unwritten_buffer
    assert model._pending_lines == len(lines)


@pytest.mark.parametrize(
    'read, unprocessed_buffer, expected_unprocessed_buffer, expected_predictions',
    [
        (
            None,
            b'',
            b'',
            []
        ),
        (
            b'0.12',
            b'',
            b'0.12',
            []
        ),
        (
            b'0.12\n',
            b'',
            b'',
            [0.12]
        ),
        (
            b'\n0.12',
            b'0.23',
            b'0.12',
            [0.23]
        ),
    ],
    ids=[
        'Nothing to read, empty unprocessed buffer',
        'Uncomplete prediction',
        'Complete prediction',
        'Nonempty unprocessed buffer'
    ]
)
def test_get_predictions_from_vowpal(read, unprocessed_buffer, expected_unprocessed_buffer, expected_predictions):
    model = Mock(
        vw_process=Mock(stdout=Mock(read=Mock(return_value=read))),
        _unprocessed_buffer=unprocessed_buffer,
        _pending_lines=0
    )
    detailed_metrics = {
        'receiving_bytes': []
    }
    predictions = list(VowpalWabbitNonBlockingProcess._get_predictions_from_vowpal(model, detailed_metrics))

    assert model._unprocessed_buffer == expected_unprocessed_buffer
    assert predictions == expected_predictions
    assert detailed_metrics['receiving_bytes'][0][1] == (len(read) if read is not None else 0)


@pytest.mark.parametrize(
    'written_bytes, vowpal_output_buffer_reads, num_items',
    [
        (
                [
                    # up to 10 items, size of one line is 18 bytes
                    18
                ],
                [
                    b'0.123\n'
                ],
                1
        ),
        (
                [
                    36
                ],
                [
                    b'0.123\n0.234\n'
                ],
                2
        ),
        (
                [
                    15,
                    21
                ],
                [
                    b'',
                    b'0.123\n0.234\n'
                ],
                2
        ),
        (
                [
                    21,
                    15
                ],
                [
                    b'0.12',
                    b'3\n0.234\n'
                ],
                2
        ),
(
                [
                    21,
                    None,
                    None,
                    15
                ],
                [
                    b'0.12',
                    None,
                    None,
                    b'3\n0.234\n'
                ],
                2
        ),
    ],
    ids=[
        'Simple case of one VW line',
        'Two lines in one batch processed immediately',
        'Two lines, first line is not completely written',
        'Two lines, second line is not completely written',
        'Two lines, second line is not completely written and VW blocks'
    ]
)
def test_predict_without_timeout(written_bytes, vowpal_output_buffer_reads, num_items):
    batch_size = 2

    formatter = VowpalWabbitDummyFormatter()
    common_features = '|a user1'
    items_features = ['|b item{}'.format(i) for i in range(num_items)]

    vw_process = Mock(
        stdin=Mock(
            write=Mock(side_effect=written_bytes)
        ),
        stdout=Mock(
            read=Mock(side_effect=vowpal_output_buffer_reads)
        )
    )
    popen = Mock(
        return_value=vw_process
    )
    with patch('subwabbit.nonblocking.fcntl') as fcntl:
        with patch('subwabbit.nonblocking.subprocess.Popen', new=popen):
            model = VowpalWabbitNonBlockingProcess(
                formatter=formatter,
                batch_size=batch_size,
                vw_args=[]
            )

            predictions = list(model.predict(common_features, iter(items_features)))

    written_stream = b''
    for c, num_bytes in zip(vw_process.stdin.write.mock_calls, written_bytes):
        written_stream += c[1][0][:num_bytes if num_bytes is not None else 0]
    expected_written_stream = bytes('\n'.join([formatter.get_vw_line(common_features, item_features) for item_features in items_features]) + '\n', 'utf-8')
    assert written_stream == expected_written_stream

    assert predictions == [float(p.strip()) for p in b''.join(b for b in vowpal_output_buffer_reads if b is not None).decode('utf-8').split()]
