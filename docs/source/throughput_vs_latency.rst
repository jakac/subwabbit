Throughput vs. latency
======================

There are two implementations of :class:`subwabbit.base.VowpalWabbitBaseModel`. Both implementation
runs ``vw`` subprocess and communicates with subprocess through pipes, but implementation differs in whether
pipe is blocking or nonblocking.

Blocking
--------

:class:`subwabbit.blocking.VowpalWabbitProcess`

Blocking implementation use buffered binary IO. When `predict()` method is called,
there is loop that:

- creates batch of VW lines
- sends this batch to Vowpal and flush Python-side buffer into
  system pipe buffer
- waits for predictions from last but one batch (writing is one batch ahead,
  so Vowpal should always be busy with processing lines)

There is also :func:`~subwabbit.blocking.VowpalWabbitProcess.train()` method that looks very similar,
but usually you run training on instance with `write_only=True` so there is no need to wait for predictions.

Nonblocking
-----------

- :class:`subwabbit.nonblocking.VowpalWabbitNonBlockingProcess`

.. warning::

    Nonblocking implementation is available only for Linux based systems.


.. warning::

    Training is not implemented for nonblocking variant.


Blocking implementation have great throughput, depends on features you have and arguments of vw process, it can be
optimal in a sense that Vowpal itself is a bottleneck. However, due to blocking system calls, it can sometimes miss
`timeout`. It is problem if there is SLO with low-latency requirements.

Nonblocking implementation works similar to blocking, but it do not block for system calls when there are no predictions
to read or system level buffer for VW lines is full, which helps to keep latencies very stable.

There is comparison of running time of :func:`~subwabbit.base.VowpalWabbitBaseModel.predict`
method with `timeout` set to 10ms:

+------+----------+----------+-------------+
|      |   pyvw   | blocking | nonblocking |
+======+==========+==========+=============+
| mean | 0.010039 | 0.010929 |   0.009473  |
+------+----------+----------+-------------+
|  min | 0.010012 | 0.010054 |   0.009049  |
+------+----------+----------+-------------+
|  25% | 0.010025 | 0.010130 |   0.009142  |
+------+----------+----------+-------------+
|  50% | 0.010036 | 0.010312 |   0.009355  |
+------+----------+----------+-------------+
|  75% | 0.010048 | 0.010630 |   0.009804  |
+------+----------+----------+-------------+
|  90% | 0.010063 | 0.010950 |   0.010024  |
+------+----------+----------+-------------+
|  99% | 0.010091 | 0.013289 |   0.010140  |
+------+----------+----------+-------------+
|  max | 0.010138 | 0.468903 |   0.010999  |
+------+----------+----------+-------------+


Nonblocking implementation reduced latency peaks significantly, from almost 460ms to just 1ms.

Nonblocking implementation makes more system calls with smaller batches then blocking implementation and it comes
with price of slightly lower throughput.

Predicted lines per request:

+------+------------+------------+-------------+
|      |    pyvw    |  blocking  | nonblocking |
+======+============+============+=============+
| mean | 239.461000 | 1033.70000 |  911.890000 |
+------+------------+------------+-------------+
|  min |  83.000000 |  100.00000 |   0.000000  |
+------+------------+------------+-------------+
|  25% | 192.750000 |  650.00000 |  552.000000 |
+------+------------+------------+-------------+
|  50% | 240.000000 | 1000.00000 |  841.500000 |
+------+------------+------------+-------------+
|  75% | 288.000000 | 1350.00000 | 1271.750000 |
+------+------------+------------+-------------+
|  90% | 316.000000 | 1600.00000 | 1574.000000 |
+------+------------+------------+-------------+
|  99% | 349.000000 | 1900.00000 | 1900.130000 |
+------+------------+------------+-------------+
|  max | 362.000000 | 2050.00000 | 2022.000000 |
+------+------------+------------+-------------+


.. note::

    Nonblocking implementation can have even zero predictions per call. It can happen because if
    previous call had not enough time to clean buffers before timeout, next call have to do that and it can take all the time.
    See :func:`~subwabbit.nonblocking.VowpalWabbitNonBlockingProcess.predict` `metrics` argument for details how
    to monitor this behavior.