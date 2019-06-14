import logging
import time
import subprocess
from typing import Dict, Iterable, Any, List, Optional

from .base import VowpalWabbitError, VowpalWabbitBaseModel, VowpalWabbitBaseFormatter

logger = logging.getLogger(__name__)


class VowpalWabbitProcess(VowpalWabbitBaseModel):
    """
    Class representing Vowpal Wabbit model. It runs ``vw`` command
    through subprocess library and communicates through pipes.
    """

    # pylint: disable=super-init-not-called,too-many-arguments
    def __init__(self, formatter: VowpalWabbitBaseFormatter, vw_args: List, batch_size: int = 20,
                 write_only: bool = False, audit_mode: bool = False):
        """
        :param formatter: Instance of :class:`subwabbit.base.VowpalWabbitBaseFormatter`
        :param vw_args: List of command line arguments for vw command, eg. ['-q', '::']
                        This list MUST NOT specify `-p` argument for `vw` command
        :param batch_size: Number of lines communicated to Vowpal in one system call, has influence on performance.
                           Smaller batches slightly reduces latencies and throughput.
        :param write_only: whether we expect to get predictions or we will just train
                            This can greatly improve training performance but disables predicting.
        :param audit_mode: When set to True, VW is launched in audit mode with `-a` argument (overwrites `-t` argument).
                            This allows to run `explain_vw_line` and `get_human_readable_explanation` methods.

        .. warning::

                            WARNING: When audit_mode is turned on, it is not possible to call other methods then
                            `explain_vw_line`.
        """
        self.formatter = formatter
        self.vw_args = vw_args
        self.batch_size = batch_size
        self.write_only = write_only
        self.audit_mode = audit_mode
        if self.write_only:
            output = ['-p', '/dev/null']
            stdout = None
        else:
            output = ['-p', '/dev/stdout']
            stdout = subprocess.PIPE
        if self.audit_mode:
            self.vw_args = self.vw_args.copy()
            if '-t' in vw_args:
                self.vw_args.remove('-t')
            if '-a' not in self.vw_args:
                self.vw_args.append('-a')
        logger.info(
            'Instantiating VW process with arguments %s, batch_size=%i, write_only=%s, audit_mode=%s',
            str(self.vw_args), self.batch_size, self.write_only, self.audit_mode
        )
        self.vw_process = subprocess.Popen(['vw'] + output + self.vw_args,
                                           stdin=subprocess.PIPE, stdout=stdout)
        if not (self.write_only or self.audit_mode):
            self.unprocessed_batch_sizes = []  # type: List

    def __del__(self):
        try:
            self.close()
        except:  # pylint: disable=bare-except
            pass

    def close(self):
        """
        Gracefully stop Vowpal Wabbit process
        """
        self.vw_process.stdin.close()
        if not self.write_only:
            # We have to exhaust stdout of subprocess
            # or it can cause deadlock in calling wait()
            # see Note at: https://docs.python.org/3/library/subprocess.html#subprocess.Popen.wait
            stdout_not_empty = self.vw_process.stdout.readlines()
            if stdout_not_empty:
                logger.warning('You left some data in Vowpal stdout buffer: %s', str(stdout_not_empty))
            self.vw_process.stdout.close()
        # VW should exit after closing its stdin, so lets wait for it
        if self.vw_process.wait(timeout=120) != 0:
            raise VowpalWabbitError('Vowpal Wabbit process returned non-zero return code')

    def _check_empty_buffer(self):
        if hasattr(self, 'unprocessed_batch_sizes') and self.unprocessed_batch_sizes:
            raise Exception(
                'Unprocessed batches sended to vowpal, so you can not'
                ' communicate with vowpal synchronously'
            )

    # pylint: disable=arguments-differ,too-many-arguments,too-many-locals,too-many-branches
    def predict(
            self,
            common_features: Any,
            items_features: Iterable[Any],
            timeout: Optional[float] = None,
            debug_info: Any = None,
            metrics: Optional[Dict] = None,  # pylint: disable=unused-argument
            detailed_metrics: Optional[Dict] = None) -> Iterable[float]:  # pylint: disable=unused-argument
        """
        Transforms iterable with item features to iterator of predictions.

        :param common_features: Features common for all items
        :param items_features: Iterable with features for each item
        :param timeout: Optionally specify how much time in seconds is desired for computing predictions.
                        In case timeout is passed, returned iterator can has less items that items features iterable.
        :param debug_info: Some object that can be filled by information useful for debugging.
        :param metrics: Optional dict populated with metrics that are good to monitor:

                        - ``prepare_time`` - Time from call start to start of prediction loop, including
                          ``format_common_features`` call
                        - ``total_time`` - Total time spend in predict call
                        - ``num_lines`` - Count of predictions performed

        :param detailed_metrics: Optional dict with more detailed (and more time consuming) metrics that are good
                                    for debugging and profiling:

                                 - ``generating_lines_time`` - time spent by generating VW line
                                 - ``sending_lines_time`` - time spent by sending VW lines to OS pipe buffer
                                 - ``receiving_lines_time`` - time spent by reading predictions from OS pipe buffer

                                 For each key, there will be list of tuples (time, metric value).

        :return: Iterable with predictions for each item from ``items_features``
        """
        if self.write_only:
            raise VowpalWabbitError('Can not predict in write only mode')

        total_t0 = time.perf_counter()
        please_respond_to = None  # type: Optional[float]
        if timeout:
            please_respond_to = total_t0 + timeout

        if metrics is not None:
            metrics['num_lines'] = 0

        if detailed_metrics is not None:
            detailed_metrics['generating_lines_time'] = []
            detailed_metrics['sending_lines_time'] = []
            detailed_metrics['receiving_lines_time'] = []

        common_line_part = self.formatter.format_common_features(common_features, debug_info=debug_info)

        batch = []
        first_pass = True
        _get_item_line_part = self.formatter.format_item_features  # for faster access in for-loop
        _get_vw_line = self.formatter.get_formatted_example  # for faster access in for-loop

        if metrics:
            metrics['prepare_time'] = time.perf_counter() - total_t0

        for item_features in items_features:
            if please_respond_to is not None and time.perf_counter() > please_respond_to:
                break
            t0 = time.perf_counter()
            item_line_part = _get_item_line_part(common_features, item_features, debug_info=debug_info)
            vw_line = _get_vw_line(common_line_part, item_line_part, debug_info=debug_info)
            if detailed_metrics:
                detailed_metrics['generating_lines_time'].append((time.perf_counter(), time.perf_counter() - t0))
            batch.append(vw_line)
            if len(batch) >= self.batch_size:
                # Send data to vowpal process
                self._send_lines_to_vowpal(batch, detailed_metrics, debug_info=debug_info)
                # First pass we want to let vowpal do its work,
                # while we prepare next batch at parallel (big speedup ;)
                if not first_pass:
                    # Get predictions from previous batch
                    for prediction in self._get_predictions_from_vowpal(detailed_metrics, debug_info=debug_info):
                        yield prediction
                        if metrics:
                            metrics['num_lines'] += 1
                else:
                    first_pass = False
                batch = []
        if batch and (please_respond_to is None or time.perf_counter() < please_respond_to):
            # We have some items in batch yet
            self._send_lines_to_vowpal(batch, detailed_metrics, debug_info=debug_info)
            # Get predictions from last batch processed in for-loop
            for prediction in self._get_predictions_from_vowpal(detailed_metrics, debug_info=debug_info):
                yield prediction
                if metrics:
                    metrics['num_lines'] += 1
        # Get predictions from last batch processed in for-loop
        # Or from batch processed after for-loop,
        #   if we had some items in batch after exiting the loop
        if not first_pass:
            for prediction in self._get_predictions_from_vowpal(detailed_metrics, debug_info=debug_info):
                yield prediction
                if metrics:
                    metrics['num_lines'] += 1
        if metrics:
            metrics['total_time'] = time.perf_counter() - total_t0

    def train(
            self,
            common_features: Any,
            items_features: Iterable[Any],
            labels: Iterable[float],
            weights: Iterable[Optional[float]],
            debug_info: Any = None) -> None:
        common_line_part = self.formatter.format_common_features(common_features, debug_info=debug_info)
        batch = []
        first_pass = True
        _get_item_line_part = self.formatter.format_item_features  # for faster access in for-loop
        _get_vw_line = self.formatter.get_formatted_example  # for faster access in for-loop
        for item_features, label, weight in zip(items_features, labels, weights):
            item_line_part = _get_item_line_part(common_features, item_features, debug_info=debug_info)
            vw_line = _get_vw_line(common_line_part, item_line_part, label=label, weight=weight,
                                   debug_info=debug_info)
            batch.append(vw_line)
            if len(batch) >= self.batch_size:
                self._send_lines_to_vowpal(batch, debug_info=debug_info)
                batch = []
                # First pass we want to let vowpal do its work,
                # while we prepare next batch at parallel (big speedup ;)
                if not first_pass:
                    if not self.write_only:
                        # If we do not use write_only=True option for training
                        # we have to take care about reading what vowpal tells us
                        # as well. Other way potential next calls to
                        # predict() will fail and deadlocks can occur.
                        self._get_predictions_from_vowpal(debug_info=debug_info)
                else:
                    first_pass = False
        if batch:
            self._send_lines_to_vowpal(batch, debug_info=debug_info)
            if not self.write_only:
                # Get predictions from last batch processed in for-loop:
                # If we do not use write_only=True option for training
                # we have to take care about reading what vowpal tells us
                # as well. Other way potential next calls to
                # predict() will fail and deadlocks can occur.
                self._get_predictions_from_vowpal(debug_info=debug_info)
        # Get predictions from last batch processed in for-loop
        # Or from batch processed after for-loop,
        #   if we had some items in batch after exiting the loop.
        # If we do not use write_only=True option for training
        # we have to take care about reading what vowpal tells us
        # as well. Other way potential next calls to
        # predict() will fail and deadlocks can occur.
        if not self.write_only and not first_pass:
            self._get_predictions_from_vowpal(debug_info=debug_info)

    def explain_vw_line(self, vw_line: str, link_function=False):
        if not self.audit_mode:
            raise Exception('Explaining is available only in audit mode')
        self._check_empty_buffer()
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
        t0 = time.perf_counter()
        self.vw_process.stdin.write(bytes('\n'.join(lines) + '\n', 'utf-8'))
        self.vw_process.stdin.flush()
        if detailed_metrics:
            detailed_metrics['sending_lines_time'].append((time.perf_counter(), time.perf_counter() - t0))
        if not self.write_only:
            # We save number of lines sended to vowpal
            # so we can get predictions only for this batch in
            # next call of _get_predictions_from_vowpal method
            self.unprocessed_batch_sizes.append(len(lines))

    def _get_predictions_from_vowpal(self, detailed_metrics=None, debug_info=None):  # pylint:  disable=unused-argument
        # There should be check whether instance is not in write-only mode,
        # but for predictions we have to be damn fast so eventually
        # let this method fail on calling pop(0) on empty list
        num_predictions = self.unprocessed_batch_sizes.pop(0)
        t0 = time.perf_counter()
        predictions = []
        received_line = None
        try:
            for _ in range(num_predictions):
                received_line = self.vw_process.stdout.readline().strip()
                predictions.append(float(received_line.split()[0]))
        except (ValueError, IndexError):
            raise ValueError('Wrong format of prediction: "%s"', received_line)
        if detailed_metrics:
            detailed_metrics['receiving_lines_time'].append((time.perf_counter(), time.perf_counter() - t0))
        return predictions
