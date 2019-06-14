import fcntl
import logging
import time
import os
import platform
import subprocess
from typing import Dict, Iterable, Any, List, Optional

from .base import VowpalWabbitError, VowpalWabbitBaseModel, VowpalWabbitBaseFormatter

logger = logging.getLogger(__name__)


if platform.system() != 'Linux':
    raise NotImplementedError('Currently only linux is supported')


class VowpalWabbitNonBlockingProcess(VowpalWabbitBaseModel):
    """
    Class representing Vowpal Wabbit model. It runs vw bash command
    through subprocess library and communicates through non-blocking pipes.

    .. warning::

        Available on Linux only.
    """

    # pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals,super-init-not-called
    def __init__(self, formatter: VowpalWabbitBaseFormatter, vw_args: List, batch_size: int = 20,
                 audit_mode: bool = False, max_pending_lines: int = 20,
                 write_timeout_ms: float = 0.001, pipe_buffer_size_bytes: Optional[int] = None):
        """
        :param formatter: Instance of :class:`subwabbit.base.VowpalWabbitBaseFormatter`
        :param vw_args: List of command line arguments for vw command, eg. ['-q', '::']
                        This list MUST NOT specify `-p` argument for `vw` command
        :param batch_size: Maximum number of lines communicated to Vowpal in one system call.
                           Smaller batches means less system calls overhead, but also higher risk of keeping
                           mess for other calls.
        :param audit_mode: When turned on, VW is launched in audit mode with `-a` argument (overwrites `-t` argument).
                            This allows to run `explain_vw_line` and `get_human_readable_explanation` methods.
        :param max_pending_lines: How many lines can wait for prediction in buffers. Recommended to set it to
                                    same value as `batch_size`, but it can be higher.
        :param write_timeout_ms: When `predict` is called with timeout, then `write_timeout_ms` before timeout
                                    sending lines to vowpal stops. It provides time to finish work without
                                    keeping mess that next call have to clean.
        :param pipe_buffer_size_bytes: Optionally set size of system buffer for sending lines to Vowpal.
                                        None means use default buffer size, for more details see
                                        http://man7.org/linux/man-pages/man7/pipe.7.html and ``detailed_metrics``
                                        argument of :func:`~VowpalWabbitNonBlockingProcess.predict` method

        .. warning::

                            WARNING: When audit_mode is turned on, it is not possible to call other methods then
                            `explain_vw_line`.
        """
        self.formatter = formatter

        self.vw_args = vw_args
        self.batch_size = batch_size
        self.audit_mode = audit_mode
        self.max_pending_lines = max_pending_lines
        self.write_timeout_ms = write_timeout_ms
        output = ['-p', '/dev/stdout']
        stdout = subprocess.PIPE
        if self.audit_mode:
            self.vw_args = self.vw_args.copy()
            if '-t' in vw_args:
                self.vw_args.remove('-t')
            if '-a' not in self.vw_args:
                self.vw_args.append('-a')
        logger.info(
            'Instantiating VW process with arguments %s, batch_size=%i, audit_mode=%s',
            str(self.vw_args), self.batch_size, self.audit_mode
        )
        self.vw_process = subprocess.Popen(['vw'] + output + self.vw_args,
                                           stdin=subprocess.PIPE, stdout=stdout, bufsize=0)

        if not self.audit_mode:
            if pipe_buffer_size_bytes is not None:
                # change buffer size for STDIN
                F_SETPIPE_SZ = 1031  # pylint: disable=invalid-name
                F_GETPIPE_SZ = 1032  # pylint: disable=invalid-name,unused-variable
                fcntl.fcntl(self.vw_process.stdin, F_SETPIPE_SZ, pipe_buffer_size_bytes)

            # set pipes as nonblocking
            old_in_flags = fcntl.fcntl(self.vw_process.stdin.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(self.vw_process.stdin.fileno(), fcntl.F_SETFL, old_in_flags | os.O_NONBLOCK)

            old_out_flags = fcntl.fcntl(self.vw_process.stdout.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(self.vw_process.stdout.fileno(), fcntl.F_SETFL, old_out_flags | os.O_NONBLOCK)

            if not self.audit_mode:
                self._pending_lines = 0

        self._unprocessed_buffer = b''
        self._unwritten_buffer = b''

    def __del__(self):
        try:
            self.close()
        except:  # pylint: disable=bare-except
            pass

    def close(self):
        self.vw_process.stdin.close()
        # We have to exhaust stdout of subprocess
        # or it can cause deadlock in calling wait()
        # see Note at: https://docs.python.org/3/library/subprocess.html#subprocess.Popen.wait
        self.vw_process.stdout.readall()
        self.vw_process.stdout.close()
        # VW should exit itself after closing its stdin, so lets wait for it
        if self.vw_process.wait(timeout=120) != 0:
            raise VowpalWabbitError('Vowpal Wabbit process returned non-zero return code')

    def cleanup(self, deadline: Optional[float] = None, debug_info: Any = None):  # pylint: disable=unused-argument
        """
        Cleans buffers after previous calls

        :param deadline: Optional unix timestamp to end
        """
        while self._pending_lines > 0 and (deadline is None or time.perf_counter() < deadline):
            self._send_lines_to_vowpal([])
            if deadline is None or time.perf_counter() < deadline:
                list(self._get_predictions_from_vowpal())

    # pylint: disable=too-many-statements,arguments-differ,too-many-branches
    def predict(
            self,
            common_features: Any,
            items_features: Iterable[Any],
            timeout: Optional[float] = None,
            debug_info: Any = None,
            metrics: Optional[Dict] = None,
            detailed_metrics: Optional[Dict] = None) -> Iterable[float]:
        """
        Transforms iterable with item features to iterator of predictions.

        :param common_features: Features common for all items
        :param items_features: Iterable with features for each item
        :param timeout: Optionally specify how much time in seconds is desired for computing predictions.
                        In case timeout is passed, returned iterator can has less items that items features iterable.
        :param debug_info: Some object that can be filled by information useful for debugging.
        :param metrics: Optional dict populated with metrics that are good to monitor:

                        - ``cleanup_time`` - Time spent on cleaning buffers after last calls
                        - ``before_cleanup_pending_lines`` - Count of lines pending in buffers before cleaning
                        - ``after_cleanup_pending_lines`` - Count of lines pending in buffers after cleaning
                        - ``prepare_time`` - Time from call start to start of prediction loop, including
                          ``format_common_features`` call
                        - ``total_time`` - Total time spend in predict call
                        - ``num_lines`` - Count of predictions performed

        :param detailed_metrics: Optional dict with more detailed (and more time consuming) metrics that are good
                                    for debugging and profiling:

                                 - ``sending_bytes`` - number of bytes (VW lines) sent to OS pipe buffer
                                 - ``receiving_bytes`` - number of bytes (predictions) received from OS pipe buffer
                                 - ``pending_lines`` - number of pending lines sent to VW at the time
                                 - ``generating_lines_time`` - time spent by generating VW lines batch
                                 - ``sending_lines_time`` - time spent by sending lines to OS pipe buffer
                                 - ``receiving_lines_time`` - time spent by receiving predictions from OS pipe buffer

                                 For each key, there will be list of tuples (time, metric value).

        :return: Iterable with predictions for each item from ``items_features``
        """
        total_t0 = time.perf_counter()
        deadline = None  # type: Optional[float]
        deadline_write = None  # type: Optional[float]
        if timeout:
            deadline = total_t0 + timeout
            deadline_write = deadline - self.write_timeout_ms

        if metrics is not None:
            metrics['num_lines'] = 0

        if detailed_metrics is not None:
            detailed_metrics['sending_bytes'] = []
            detailed_metrics['receiving_bytes'] = []
            detailed_metrics['pending_lines'] = []
            detailed_metrics['generating_lines_time'] = []
            detailed_metrics['sending_lines_time'] = []
            detailed_metrics['receiving_lines_time'] = []

        if metrics:
            metrics['before_cleanup_pending_lines'] = self._pending_lines
        t0 = time.perf_counter()
        self.cleanup(deadline, debug_info=debug_info)
        if metrics:
            metrics['cleanup_time'] = time.perf_counter() - t0
            metrics['after_cleanup_pending_lines'] = self._pending_lines

        common_line_part = self.formatter.format_common_features(common_features, debug_info=debug_info)
        batch = []
        all_lines_generated = False

        # we need to transform it to iterator, because we will consume it more than once
        # and we assume that each iteration will start where previous ended
        items_features = iter(items_features)

        _get_item_line_part = self.formatter.format_item_features  # for faster access in for-loop
        _get_vw_line = self.formatter.get_formatted_example  # for faster access in for-loop

        if metrics:
            metrics['prepare_time'] = time.perf_counter() - total_t0

        while deadline is None or time.perf_counter() < deadline:
            # generate batch of lines
            t0 = time.perf_counter()
            if deadline_write is None or time.perf_counter() < deadline_write:
                batch_size = min(self.batch_size, self.max_pending_lines - self._pending_lines)
                if batch_size >= 0:
                    for item_features in items_features:
                        item_line_part = _get_item_line_part(common_features, item_features, debug_info=debug_info)
                        vw_line = _get_vw_line(common_line_part, item_line_part, debug_info=debug_info)
                        batch.append(vw_line)
                        if len(batch) >= batch_size or (deadline_write is not None
                                                        and time.perf_counter() > deadline_write):
                            break
                    else:
                        all_lines_generated = True
            if detailed_metrics:
                detailed_metrics['generating_lines_time'].append((time.perf_counter(), time.perf_counter() - t0))
            # send lines to vowpal if there are some
            t0 = time.perf_counter()
            if deadline is None or time.perf_counter() < deadline:
                if batch or self._unwritten_buffer:
                    self._send_lines_to_vowpal(batch, detailed_metrics, debug_info=debug_info)
                    batch = []
            if detailed_metrics:
                detailed_metrics['sending_lines_time'].append((time.perf_counter(), time.perf_counter() - t0))
            # receiving predictions from vowpal
            t0 = time.perf_counter()
            if (deadline is None or time.perf_counter() < deadline) and self._pending_lines > 0:
                for prediction in self._get_predictions_from_vowpal(detailed_metrics, debug_info=debug_info):
                    yield prediction
                    if metrics:
                        metrics['num_lines'] += 1
            if detailed_metrics:
                detailed_metrics['receiving_lines_time'].append((time.perf_counter(), time.perf_counter() - t0))
                detailed_metrics['pending_lines'].append((time.perf_counter(), self._pending_lines))
            # Other stopping crierions:
            if self._pending_lines == 0:
                # - all available lines are generated (and sended) and there are no pending lines, all work is done
                if all_lines_generated:
                    break
                # - if timeout is passed, there are no pending lines and it is after deadline_write,
                #   no more lines will be written, all work is done
                if deadline_write is not None and time.perf_counter() > deadline_write:
                    break
        if metrics:
            metrics['total_time'] = time.perf_counter() - total_t0

    def train(
            self,
            common_features: Any,
            items_features: Iterable[Any],
            labels: Iterable[float],
            weights: Iterable[Optional[float]],
            debug_info: Any = None) -> None:
        raise NotImplementedError('Please use blocking implementation')

    def explain_vw_line(self, vw_line: str, link_function=False):
        if not self.audit_mode:
            raise Exception('Explaining is available only in audit mode')
        self.vw_process.stdin.write(bytes(vw_line.replace('\n', '').strip() + '\n', 'utf-8'))
        self.vw_process.stdin.flush()
        prediction, explain = (
            self.vw_process.stdout.readline().decode('utf-8').strip(),
            self.vw_process.stdout.readline().decode('utf-8').strip(),
        )
        if link_function:
            # When using link function, VW returns linked score as third result
            _ = self.vw_process.stdout.readline().decode('utf-8').strip()
        return float(prediction), explain

    def _send_lines_to_vowpal(self, lines, detailed_metrics=None, debug_info=None):  # pylint: disable=unused-argument
        if lines:
            write_bytes = self._unwritten_buffer + bytes('\n'.join(lines) + '\n', 'utf-8')
        else:
            write_bytes = self._unwritten_buffer
        if not write_bytes:
            return
        num_bytes_written = self.vw_process.stdin.write(write_bytes)
        if num_bytes_written is None:
            num_bytes_written = 0

        if num_bytes_written == 0:
            self._unwritten_buffer = write_bytes
        elif len(write_bytes) > num_bytes_written:
            self._unwritten_buffer = write_bytes[num_bytes_written:]
        else:
            self._unwritten_buffer = b''

        self._pending_lines += len(lines)

        if detailed_metrics:
            detailed_metrics['sending_bytes'].append((time.perf_counter(), num_bytes_written))

    def _get_predictions_from_vowpal(self, detailed_metrics=None, debug_info=None):  # pylint: disable=unused-argument
        read_buffer = self.vw_process.stdout.read(4096)
        if read_buffer is None:
            if detailed_metrics is not None:
                detailed_metrics['receiving_bytes'].append((time.perf_counter(), 0))
            return
        if detailed_metrics is not None:
            detailed_metrics['receiving_bytes'].append((time.perf_counter(), len(read_buffer)))
        last_newline_pos = read_buffer.rfind(b'\n')
        if last_newline_pos < 0:
            # we read some data, but we do not have complete line to process yet
            self._unprocessed_buffer = self._unprocessed_buffer + read_buffer
            return
        processed_lines = 0
        for prediction in (self._unprocessed_buffer + read_buffer[0:last_newline_pos+1]).splitlines():
            if prediction:
                yield float(prediction)
                processed_lines += 1
        self._unprocessed_buffer = read_buffer[last_newline_pos+1:]
        self._pending_lines -= processed_lines
