from torch import Tensor
from torch.nn import LogSoftmax, NLLLoss

class ce_count_loss:
    """
    Cross-entropy loss for count rate coding.

    Args:
        spk_out (Tensor): Spiking output of shape [Steps, Batch, Classes].
        targets (Tensor): Ground truth labels of shape [Batch].
        weight (Tensor, optional): Class weights for imbalanced datasets.
    """
    def __init__(self, weight=None):
        self.__name__ = "ce_count_loss"
        self.weight = weight

    def __call__(self, spk_out: Tensor, targets: Tensor):
        log_softmax_fn = LogSoftmax(dim=-1)
        loss_fn = NLLLoss(weight=self.weight)
        rates = self._encode(spk_out)
        probabilities = log_softmax_fn(rates)  
        loss = loss_fn(probabilities, targets)
        return loss
    
    def _encode(self, spk_out: Tensor):
        return spk_out.sum(dim=0)
    
    def spike_code(self, spk_out: Tensor):
        return self._encode(spk_out).argmax(dim=1)