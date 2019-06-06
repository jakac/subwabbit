Explaining predictions
======================

It is practical to understand your model. There are various ways how to gain some insights about your model behavior,
see for example excellent Dan Becker's tutorial on Kaggle: https://www.kaggle.com/learn/machine-learning-explainability
.

Vowpal Wabbit offers various options how to inspect learned weights, **subwabbit** helps with use of
*audit mode*. It allows to easily compute which features contributes the most for particular line's prediction.


How to explain prediction
-------------------------

At first, you need to turn on `audit_mode` by passing ``audit_mode=True`` argument to
:class:`subwabbit.base.VowpalWabbitBaseModel` constructor.

.. warning::

    When audit mode is turned on, it is not possible to call
    :func:`~subwabbit.base.VowpalWabbitBaseModel.predict`
    and :func:`~subwabbit.base.VowpalWabbitBaseModel.train` methods.


Then use :func:`~subwabbit.base.VowpalWabbitBaseModel.explain_vw_line` to retrieve explanation string. It will
look like this:


.. code-block::

    c^c8*f^f10237121819548268936:23365229:1:0.0220863@0	a^a3426538138935958091*e^e115:1296634:0.2:0.0987504@0


Features used for prediction are separated by `tab` and for each feature, there is string in format:


.. code-block::

    namespace^feature:hashindex:value:weight[@ssgrad]


Then we can use :func:`~subwabbit.base.VowpalWabbitBaseFormatter.get_human_readable_explanation` function
to transform explanation string into more interpretable structure:

.. autofunction:: subwabbit.base.VowpalWabbitBaseFormatter.get_human_readable_explanation
    :noindex:


You may also want to overwrite :func:`~subwabbit.base.VowpalWabbitBaseFormatter.parse_element` method on your
formatter to translate Vowpal feature names into human readable form, for example translate IDs to their names,
potentialy using some mapping in database.


Example
-------

Feature importances can also be visualized in *Jupyter notebook*, see complete example of how to use *subwabbit*
for explaining predictions:

.. code-block:: bash

    pip install jupyter
    jupyter notebook examples/Explaining-prediction.ipynb


Notes
-----

.. note::

    This explanation is valid if you use sparse features, since expected value of every feature is close to zero.
    When you use dense features, you should normalize your features. If you do not normalize to zero mean,
    explaining features by their absolute contribution is not informative
    because you also need to consider how feature value differs from some expected value of that feature.
    In this case, you should use SHAP values for better interpretability,
    see https://www.kaggle.com/learn/machine-learning-explainability for more details. You still may find
    *subwabbit* explaining functionality useful, but interpreting results results won't be straightforward.

.. note::

    In case you have correlated features, it is better to sum their potentials and consider them as single feature,
    otherwise you may underestimate influence of these features.