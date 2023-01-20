import torch
from torch.nn import CrossEntropyLoss


def cal_loss(labels, logits, attention_mask, uni_w, ske_w):
    """Calculate the token classification loss.

    Args:
        uni_w: float. The weight for causal words.
        ske_w: float. The weight for background words.
    """
    w = torch.tensor([uni_w, ske_w]).cuda()
    loss_fct = CrossEntropyLoss(weight=w)
    if attention_mask is not None:
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, 2)
        active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
        loss = loss_fct(active_logits, active_labels)
    else:
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
    return loss