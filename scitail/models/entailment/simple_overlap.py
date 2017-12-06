from typing import Dict, List, Any

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("simple_overlap")
class SimpleOverlap(Model):
    """
    Simple model that applies a feedforward network on overlap-based feature vectors to
    compute entailment probability
    """

    def __init__(self, vocab: Vocabulary,
                 classifier: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(SimpleOverlap, self).__init__(vocab)
        self.linear_mlp = classifier
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,  # type: ignore
                features: torch.Tensor,
                metadata: List[Dict[str, Any]] = None,
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        features: torch.Tensor,
            From a ``FloatField`` over the overlap features computed by the SimpleOverlapReader
        metadata: List[Dict[str, Any]]
            Metadata information
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        label_logits = self.linear_mlp(features)
        label_probs = torch.nn.functional.softmax(label_logits)
        output_dict = {"label_logits": label_logits, "label_probs": label_probs}
        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label.squeeze(-1))
            output_dict["loss"] = loss
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SimpleOverlap':
        classifier = FeedForward.from_params(params.pop('classifier'))

        init_params = params.pop('initializer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())

        return cls(vocab=vocab,
                   classifier=classifier,
                   initializer=initializer)
