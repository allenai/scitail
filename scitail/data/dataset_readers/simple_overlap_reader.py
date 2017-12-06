import json
import logging
from typing import Dict

import tqdm
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from nltk.util import skipgrams, ngrams
from overrides import overrides

from scitail.data.fields.features_field import FeaturesField

logger = logging.getLogger(__name__)


@DatasetReader.register("simple_overlap")
class SimpleOverlapReader(DatasetReader):
    """
    A reader that converts an entailment dataset into simple overlap statistics that can be used
    as input features by a neural network. It reads the JSONL or TSV format to create three
    features: trigram overlap, bigram overlap, unigram overlap. The trigrams and bigrams allow
    for one skip word. The overlap is normalized by the number of corresponding n-grams in the
    hypothesis.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []
        with open(file_path, 'r') as snli_file:
            logger.info("Reading instances from tsv/jsonl dataset at: %s", file_path)
            for line in tqdm.tqdm(snli_file):
                if file_path.endswith(".jsonl"):
                    # SNLI format
                    example = json.loads(line)
                    label = example["gold_label"]
                    premise = example["sentence1"]
                    hypothesis = example["sentence2"]
                else:
                    # DGEM/TSV format
                    fields = line.split("\t")
                    premise = fields[0]
                    hypothesis = fields[1]
                    label = fields[2]
                if label == '-':
                    # ignore unknown examples
                    continue
                instances.append(self.text_to_instance(premise, hypothesis, label))
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @overrides
    def text_to_instance(self,
                         premise: str,
                         hypothesis: str,
                         label: str = None) -> Instance:
        fields: Dict[str, Field] = {}
        premise_tokens = [x.text for x in self._tokenizer.tokenize(premise)]
        hypothesis_tokens = [x.text for x in self._tokenizer.tokenize(hypothesis)]
        prem_trigrams = set(skipgrams(premise_tokens, 3, 1))
        prem_bigrams = set(skipgrams(premise_tokens, 2, 1))
        prem_unigrams = set(ngrams(premise_tokens, 1))

        hyp_trigrams = set(skipgrams(hypothesis_tokens, 3, 1))
        hyp_bigrams = set(skipgrams(hypothesis_tokens, 2, 1))
        hyp_unigrams = set(ngrams(hypothesis_tokens, 1))

        tri_overlap = float(len(prem_trigrams.intersection(hyp_trigrams))) / len(hyp_trigrams) \
            if len(hyp_trigrams) > 0 else 0.0
        bi_overlap = float(len(prem_bigrams.intersection(hyp_bigrams))) / len(hyp_bigrams) \
            if len(hyp_bigrams) > 0 else 0.0
        uni_overlap = float(len(prem_unigrams.intersection(hyp_unigrams))) / len(hyp_unigrams) \
            if len(hyp_unigrams) > 0 else 0.0

        fields['features'] = FeaturesField([tri_overlap, bi_overlap, uni_overlap])
        metadata = {
            'premise': premise,
            'hypothesis': hypothesis,
            'premise_tokens': [token.text for token in premise_tokens],
            'hypothesis_tokens': [token.text for token in hypothesis_tokens]
        }
        fields['metadata'] = MetadataField(metadata)
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)


    @classmethod
    def from_params(cls, params: Params) -> 'SimpleOverlapReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        params.assert_empty(cls.__name__)
        return SimpleOverlapReader(tokenizer=tokenizer)
