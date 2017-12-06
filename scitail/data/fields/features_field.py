import logging
import math
from typing import Dict, List

import numpy
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.fields.field import Field
from overrides import overrides

logger = logging.getLogger(__name__)


class FeaturesField(Field[numpy.ndarray]):
    """An AllenNLP field for fixed-length feature vectors"""
    def __init__(self, features: List[float]) -> None:
        self.features = features

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_features': len(self.features)}

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> numpy.array:
        padded_features = pad_sequence_to_length(self.features,
                                                 padding_lengths['num_features'],
                                                 (lambda: math.nan))
        return numpy.asarray(padded_features, dtype=numpy.float32)

    @overrides
    def empty_field(self):
        return FeaturesField([math.nan] * len(self.features))
