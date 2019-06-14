from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Iterable, Any, Optional


class VowpalWabbitError(Exception):
    pass


class VowpalWabbitBaseFormatter(ABC):
    """
    Formatter translates structured information about context and items to
    Vowpal Wabbit's input format: https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format

    It also can implement reverse translation, from Vowpal Wabbits feature names into human readable feature names.
    """

    @abstractmethod
    def format_common_features(self, common_features: Any,
                               debug_info: Any = None) -> str:
        """
        Return part of VW line with features that are common for one call of predict/train.
        This method will run just once per one call of
        :class:`subwabbit.base.VowpalWabbitBaseModel`'s `predict()` or `train()` method.

        :param common_features: Features common for all items
        :param debug_info: Optional dict that can be filled by information useful for debugging
        :return: Part of line that is common for each item in one call. Returned string has to start with '|' symbol.
        """
        raise NotImplementedError()

    @abstractmethod
    def format_item_features(self, common_features: Any, item_features: Any,
                             debug_info: Any = None) -> str:
        """
        Return part of VW line with features specific to each item.
        This method will run for each item per one call of
        :class:`subwabbit.base.VowpalWabbitBaseModel`'s `predict()` or `train()` method.

        .. note::

            It is a good idea to cache results of this method.


        :param common_features: Features common for all items
        :param item_features: Features for item
        :param debug_info: Optional dict that can be filled by information useful for debugging
        :return: Part of line that is specific for item. Depends on whether namespaces are used or not in
                 ``format_common_features`` method:

                 - namespaces are used: returned string has to start with ``'|NAMESPACE_NAME'`` where `NAMESPACE_NAME`
                   is the name of some namespace
                 - namespaces are not used: returned string should not contain '|' symbol
        """
        raise NotImplementedError()

    # pylint: disable=too-many-arguments,no-self-use
    def get_formatted_example(self, common_line_part: str, item_line_part: str,
                              label: Optional[float] = None, weight: Optional[float] = None,
                              debug_info: Optional[Dict[Any, Any]] = None):  # pylint: disable=unused-argument
        """
        Compose valid VW line from its common and item-dependent parts.

        :param common_line_part: Part of line that is common for each item in one call.
        :param item_line_part: Part of line specific for each item
        :param label: Label of this row
        :param weight: Optional weight of row
        :param debug_info: Optional dict that can be filled by information useful for debugging
        :return: One VW line in input format: https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format
        """
        if label is not None:
            return ' '.join((
                str(label),
                str(weight) if weight is not None else '',
                common_line_part,
                item_line_part
            ))
        return ' '.join((common_line_part, item_line_part))

    def get_human_readable_explanation(self, explanation_string: str,
                                       feature_translator: Any = None) -> List[Dict]:
        """
        Transform explanation string into more readable form.
        Every feature used for prediction is translated into this structure:

        .. code-block:: python

            {
                # For each feature used in higher interaction there is a 2-tuple
                'names': [('Human readable namespace name 1', 'Human readable feature name 1'), ...],
                'original_feature_name': 'c^c8*f^f102'  # feature name how vowpal sees it,
                'hashindex': 123,  # Vowpal's internal hash of feature name
                'value': 0.123, # value for feature in input line
                'weight': -0.534, # weight learned by VW for this feature
                'potential': value * weight,
                'relative_potential': abs(potential) / sum_of_abs_potentials_for_all_features
            }

        :param explanation_string: Explanation string from :func:`~VowpalWabbitBaseModel.explain_vw_line`
        :param feature_translator: Any object that can help you with translation of feature names into human readable
                                   form, for example some database connection.
                                   See :func:`~VowpalWabbitBaseFormatter.parse_element`
        :return: List of dicts, sorted by contribution to final score
        """
        parsed_features = []
        potential_sum = 0.0
        for feature in [f.split(':') for f in explanation_string.split('\t')]:
            feature_name = feature[0]
            hash_index = feature[1]
            value = float(feature[2])
            weight = float(feature[3].split('@')[0])

            # quadratic and higher level interactions have multiple features for one weight
            feature_name_parts = feature_name.split('*')
            parsed_feature_name_parts = [self.parse_element(el, feature_translator) for el in feature_name_parts]
            parsed_features.append({
                'names': parsed_feature_name_parts,
                'original_feature_name': feature_name,
                'hashindex': int(hash_index),
                'value': value,
                'weight': weight,
                'potential': value * weight
            })
            potential_sum += abs(value * weight)
        if potential_sum == 0:
            # can happen in case all features are unknown
            potential_sum = 1
        for parsed_feature in parsed_features:
            parsed_feature['relative_potential'] = abs(parsed_feature['potential'] / potential_sum)  # type: ignore
        return list(sorted(parsed_features, key=lambda f: f['relative_potential'], reverse=True))

    # pylint: disable=invalid-name
    def get_human_readable_explanation_html(self, explanation_string: str, feature_translator: Any = None,
                                            max_rows: Optional[int] = None):
        """
        Visualize importance of features in Jupyter notebook.

        :param explanation_string: Explanation string from :func:`~VowpalWabbitBaseModel.explain_vw_line`
        :param feature_translator: Any object that can help you with translation, e.g. some database connection.
        :param max_rows: Maximum number of most important features. None return all used features.
        :return: `IPython.core.display.HTML`
        """
        try:
            from IPython.core.display import HTML
        except ImportError:
            raise ImportError('Please install IPython to use this method')

        explanation = self.get_human_readable_explanation(explanation_string, feature_translator)
        rows = []
        for row_number, feature in enumerate(explanation):
            if max_rows is not None and (row_number + 1) > max_rows:
                break
            feature_name = ''
            for name in feature['names']:
                if feature_name:
                    feature_name += '''
                        <span style="color: grey; margin-left: 10px; margin-right: 10px;">IN COMBINATION WITH</span>
                        '''
                feature_name += name[0]
                feature_name += ': <i>{}</i>'.format(name[1])

            rows.append(
                '''
                <tr>
                    <td>
                        <div style="display: block; width: 100px; border: solid 1px;
                                    -webkit-border-radius: 5px; -moz-border-radius: 5px; border-radius: 5px;">
                            <div style="display: block; width: {width}%; height: 20px; background-color: {color};
                                       overflow: hidden;"></div>
                        </div>
                    </td>
                    <td>{potential:.4f}</td>
                    <td>
                        {feature_value:.4f}
                    </td>
                    <td>
                        {feature_weight:.4f}
                    </td>
                    <td>
                        {feature_name}
                    </td>
                </tr>
                '''.format(
                    width=feature['relative_potential'] * 100,
                    color='green' if feature['potential'] > 0 else 'red',
                    potential=feature['potential'],
                    feature_value=feature['value'],
                    feature_weight=feature['weight'],
                    feature_name=feature_name
                )
            )

        return HTML('''
            <table>
                <thead>
                    <tr>
                        <th>Relative potential</th>
                        <th>Potential</th>
                        <th>Value</th>
                        <th>Weight</th>
                        <th>Feature name</th>
                    </tr>
                </thead>
                <tbody>
                ''' + ''.join(rows) + '''
                </tbody>
            </table>''')

    # pylint: disable=unused-argument,no-self-use
    def parse_element(self, element: str, feature_translator: Any = None) -> Tuple[str, str]:
        """
        This method is supposed to translate namespace name and feature name to human readable form.

        For example, element can be "a_item_id^i123" and result can be ('Item ID', 'News of the day: ID of item is 123')

        :param element: namespace name and feature name, e.g. a_item_id^i123
        :param feature_translator: Any object that can help you with translation, e.g. some database connection
        :return: tuple(human understandable namespace name, human understandable feature name)
        """
        splitted = element.split('^')
        if len(splitted) == 1:
            return '', splitted[0]
        return splitted[0], splitted[1]


class VowpalWabbitDummyFormatter(VowpalWabbitBaseFormatter):
    """
    Formatter that assumes that either common features and item features are already formatted VW input format strings.
    """

    def format_common_features(self, common_features: str,
                               debug_info: Optional[Dict[Any, Any]] = None) -> str:
        return common_features

    def format_item_features(self, common_features: Any, item_features: str,
                             debug_info: Optional[Dict[Any, Any]] = None) -> str:
        return item_features


class VowpalWabbitBaseModel(ABC):
    """
    Declaration of Vowpal Wabbit model interface.
    """

    def __init__(self, formatter: VowpalWabbitBaseFormatter):
        self.formatter = formatter
        super().__init__()

    # pylint: disable=too-many-arguments
    @abstractmethod
    def predict(
            self,
            common_features: Any,
            items_features: Iterable[Any],
            timeout: Optional[float] = None,
            debug_info: Any = None,
            metrics: Optional[Dict] = None,
            detailed_metrics: Optional[Dict] = None
        ) -> Iterable[float]:
        """
        Transforms iterable with item features to iterator of predictions.

        :param common_features: Features common for all items
        :param items_features: Iterable with features for each item
        :param timeout: Optionally specify how much time in seconds is desired for computing predictions.
                        In case timeout is passed, returned iterator can has less items that items features iterable.
        :param debug_info: Some object that can be filled by information useful for debugging.
        :param metrics: Optional dict that is populated with some metrics that are good to monitor.
        :param detailed_metrics: Optional dict with more detailed (and more time consuming) metrics that are good
                                    for debugging and profiling.
        :return: Iterable with predictions for each item from ``items_features``
        """
        raise NotImplementedError()

    # pylint: disable=too-many-arguments
    @abstractmethod
    def train(
            self,
            common_features: Any,
            items_features: Iterable[Any],
            labels: Iterable[float],
            weights: Iterable[Optional[float]],
            debug_info: Any = None
        ) -> None:
        """
        Transform features, label and weight into VW line format and send it to Vowpal.

        :param common_features: Features common for all items
        :param items_features: Iterable with features for each item
        :param labels: Iterable with same length as items features with label for each item
        :param weights: Iterable with same length as items features with optional weight for each item
        :param debug_info: Some object that can be filled by information useful for debugging
        """
        raise NotImplementedError()

    @abstractmethod
    def explain_vw_line(self, vw_line: str, link_function: bool = False):
        """
        Uses VW audit mode to inspect weights used for prediction. Audit mode has to be turned on
        by passing ``audit_mode=True`` to constructor.

        :param vw_line: String in VW line format
        :param link_function: If your model use link function, pass True
        :return: (raw prediction without use of link function, explanation string)
        """
        raise NotImplementedError()
