Monitoring and debugging
========================

This section gives overview of *subwabbit* monitoring and debugging capabilities.

Monitoring
----------

It is good practice to monitor your system's behavior and fire an alert when system behavior changes.

Both blocking and nonblocking implementations of :func:`~subwabbit.base.VowpalWabbitBaseModel.predict`
can collect some metrics that can be helpful. There are two kinds of
metrics:

- ``metrics`` - one numeric measurment per one call of `predict()` method. They are relatively cheap to collect and
  should be monitored in production.
- ``detailed_metrics`` - more measurements per one call of `predict()`. Each metric value is a list containing
  tuple ``(time, numeric value)``. Their collection brings some overhead,
  for example for reallocation of memory of growing lists as number of measurements grows.
  They are useful for profiling and can answer questions like "What is the bottleneck, formatting Vowpal lines
  or Vowpal itself?" or "Can change in some parameter bring some additional performance?".

See API documentation for more details about collected metrics for specific implementation.

See example of visualizing ``detailed_metrics``:

.. code-block:: bash

    pip install jupyter pandas matplotlib
    jupyter notebook examples/Detailed-metrics.ipynb


Debugging
---------

Sometimes it is useful to save some internal state like final formatted VW line. For these cases you can use
``debug_info`` parameter, which can be passed both to :func:`~subwabbit.base.VowpalWabbitBaseModel.predict`
and :func:`~subwabbit.base.VowpalWabbitBaseModel.train` methods and which is passed to all following
:class:`subwabbit.base.VowpalWabbitBaseFormatter` calls and to private method calls. You can pass dict
for example and fill it by some useful information.