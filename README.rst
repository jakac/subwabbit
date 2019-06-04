Fast Python Vowpal Wabbit wrapper
=================================

*For Kaggle playing use official vowpalwabbit package, for production use subwabbit.*

**subwabbit** is Python wrapper around great `Vowpal Wabbit <https://github.com/VowpalWabbit/vowpal_wabbit/>`_ tool
that aims to be as fast as Vowpal itself. It is ideal for real time use, when many lines need to be scored
in just few milliseconds or when high throughput is required.

**Advantages**:

- more then 4x faster then official Python wrapper
- good latency guarantees - give 10ms for prediction and it will end in 10ms
- explainability - API for explaining prediction value
- use just ``vw`` CLI - no compiling
- proven by reliably running in production at Seznam.cz where it makes hundreds of thousands
  of predictions per second per machine

Documentation
-------------
Full documentation can be found on `Read the docs <https://subwabbit.readthedocs.io>`_.

Requirements
------------

- Python 3.4+
- Vowpal Wabbit

You can install Vowpal Wabbit by running:

.. code-block:: bash

    sudo apt-get install vowpal-wabbit

on Debian-based systems or by using Homebrew:

.. code-block:: bash

    brew install vowpal-wabbit

You can also build Vowpal Wabbit from source, see `instructions <https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Dependencies>`_.

**subwabbit** will probably work on other Pythons than 3.4+ but it is not tested
(contribution welcomed).


Installation
------------

.. code-block:: bash

    pip install subwabbit


Example use
-----------


.. code-block:: python

    from subwabbit import VowpalWabbitProcess, VowpalWabbitDummyFormatter

    vw = VowpalWabbitProcess(VowpalWabbitDummyFormatter(), ['-q', 'ab'])

    common_features = '|a common_feature1:1.5 common_feature2:-0.3'
    items_features = [
        '|b item123',
        '|b item456',
        '|b item789'
    ]

    for prediction in vw.predict(common_features, items_features, timeout=0.001):
        print(prediction)
    0.4
    0.5
    0.6

This is the simplest use of *subwabbit* library. You have some common features that describe context
- it can be location of user or daytime for example. Then there is collection of items to score, each item has
its specific features. Use of `timeout` argument means "compute as many predictions as you can in 1ms", then stop.

More advanced use
`````````````````

With simple implementation above you will not use key feature of `subwabbit`:
**you can format your vw lines while Vowpal is busy with computing predictions**.
By using this trick, you can get great speedup and VW lines formatting abstraction as a bonus.


Suppose we have features as dicts:

.. code-block::  python

    common_features = {
        'common_feature1': 1.5,
        'common_feature2': -0.3
    }

    items_features = [
        {'id': 'item123'},
        {'id': 'item456'},
        {'id': 'item789'}
    ]


Then implementation with use of formatter can look like this:


.. code-block:: python

    from subwabbit import VowpalWabbitBaseFormatter, VowpalWabbitProcess

    class MyVowpalWabbitFormatter(VowpalWabbitBaseFormatter):

        def get_common_line_part(self, common_features, debug_info=None):
            return '|a ccommon_feature1:{:.2f} common_feature2:{:.2f}'.format(
                common_features['common_feature1'],
                common_features['common_feature2']
            )

        def get_item_line_part(self, common_features, item_features, debug_info=None):
            return '|b {}'.format(item_features['id'])

    vw = VowpalWabbitProcess(MyVowpalWabbitFormatter(), ['-q', 'ab'])

    for prediction in vw.predict(common_features, items_features, timeout=0.001):
        print(prediction)
    0.4
    0.5
    0.6



Benchmarks
----------

Benchmarks were made on logistic regression model with L2 regularization and with many quadratic combinations
to mimic real-world use case.
Real dataset containing 1000 contexts and 3000 items was used.
Model was pretrained on this dataset with random labels generated. You can see used features at:

- `tests/benchmarks/requests.json`
- `tests/benchmarks/items.json`

.. code-block:: bash

    # Prepare environment
    pip install pandas vowpalwabbit
    cd tests/benchmarks
    # benchmarks depends a lot whether Vowpal is trained or just initialized
    python pretrain_model.py

    # Benchmark official Python client
    python benchmark_pyvw.py

    # Benchmark blocking implementation
    python benchmark_blocking_implementation.py

    # Benchmark nonblocking implementation
    python benchmark_blocking_implementation.py


Benchmark results
`````````````````
Results on Dell Latitude E7470 with Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz.

Table shows how many lines implementation can predict in 10ms:

+------+------------+------------------+
|      |    pyvw    |     subwabbit    |
+======+============+==================+
| mean | 239.461000 |    1033.70000    |
+------+------------+------------------+
|  min |  83.000000 |     100.00000    |
+------+------------+------------------+
|  25% | 192.750000 |     650.00000    |
+------+------------+------------------+
|  50% | 240.000000 |    1000.00000    |
+------+------------+------------------+
|  75% | 288.000000 |    1350.00000    |
+------+------------+------------------+
|  90% | 316.000000 |    1600.00000    |
+------+------------+------------------+
|  99% | 349.000000 |    1900.00000    |
+------+------------+------------------+
|  max | 362.000000 |    2050.00000    |
+------+------------+------------------+

**subwabbit** is in average more then **4x** faster than official Python wrapper.


License
-------

Copyright (c) 2016 - 2018, Seznam.cz, a.s.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

