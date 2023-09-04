import torch
import torch.nn as nn
import torch.nn.functional as F


# ==== Regular Loss (for minimization, or grandient ascent) ====
class MarginLossOrig(nn.Module):
    """
        AKA CW loss
    """
    def __init__(self, reduction="mean", use_clip_loss=False):
        super(MarginLossOrig, self).__init__()
        self.reduction = reduction
        self.use_clip_loss = use_clip_loss

    def forward(self, logits, labels):
        correct_logits = torch.gather(logits, 1, labels.view(-1, 1)) # [n, 1]
        max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1) # [n, 2]
        top_max, second_max = max_2_logits.chunk(2, dim=1)  # [n, 2]
        top_argmax, _ = argmax_2_logits.chunk(2, dim=1)

        labels_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
        labels_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
        max_incorrect_logits = labels_eq_max * second_max + labels_ne_max * top_max # [n, 2]
        # loss = (max_incorrect_logits - correct_logits).clamp(max=1).squeeze().sum()
        if self.use_clip_loss:
            clamp_value = 0.01
        else:
            clamp_value = float("inf")

        if self.reduction == "none":
            loss = (max_incorrect_logits - correct_logits).clamp(max=clamp_value).squeeze(1)
        elif self.reduction == "sum":
            loss = (max_incorrect_logits - correct_logits).clamp(max=clamp_value).sum(dim=0)
        elif self.reduction == "mean":
            loss = (max_incorrect_logits - correct_logits).clamp(max=clamp_value).mean(dim=0)
        else:
            raise RuntimeError("Unsupported reduction method.")
        return loss


class MarginLossTrain(nn.Module):
    """
        AKA CW loss
    """
    def __init__(self, reduction="mean", use_clip_loss=False):
        super(MarginLossTrain, self).__init__()
        self.reduction = reduction
        self.use_flip_loss = use_clip_loss

    def forward(self, logits, labels):
        correct_logits = torch.gather(logits, 1, labels.view(-1, 1)) # [n, 1]
        max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1) # [n, 2]
        top_max, second_max = max_2_logits.chunk(2, dim=1)  # [n, 2]
        top_argmax, _ = argmax_2_logits.chunk(2, dim=1)

        labels_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
        labels_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
        max_incorrect_logits = labels_eq_max * second_max + labels_ne_max * top_max # [n, 2]
        # loss = (max_incorrect_logits - correct_logits).clamp(max=1).squeeze().sum()
        if self.use_flip_loss:
            clamp_value = -1
        else:
            raise RuntimeError("Not Good For Training Purpose.")

        if self.reduction == "none":
            loss = (max_incorrect_logits - correct_logits).clamp(min=clamp_value) - clamp_value
            loss = loss.squeeze(1)
        elif self.reduction == "sum":
            loss = (max_incorrect_logits - correct_logits).clamp(min=clamp_value) - clamp_value
            loss = loss.sum(dim=0)
        elif self.reduction == "mean":
            loss = (max_incorrect_logits - correct_logits).clamp(min=clamp_value) - clamp_value
            loss = loss.mean(dim=0)
        else:
            raise RuntimeError("Unsupported reduction method.")
        return loss


class CELoss(nn.Module):
    def __init__(self, reduction="mean", use_clip_loss=False) -> None:
        super().__init__()
        self.use_clip_loss = use_clip_loss
        self.reduction = reduction
        self.loss_func = nn.CrossEntropyLoss(reduction="none")
    
    def forward(self, logits, labels):
        if self.use_clip_loss:
            clamp_value = 10
        else:
            clamp_value = float("inf") 
        
        if self.reduction == "none":
            loss = self.loss_func(logits, labels).clamp(max=clamp_value)
        elif self.reduction == "sum":
            loss = self.loss_func(logits, labels).clamp(max=clamp_value).sum(dim=0).unsqueeze(0)
        elif self.reduction == "mean":
            loss = self.loss_func(logits, labels).clamp(max=clamp_value).mean(dim=0).unsqueeze(0)
        else:
            raise RuntimeError("Unsupported reduction method.")
        return loss


# ==== PyGranso Loss (negative regular loss) ====#
# ==== Create a different name just to minimize the coding err ====
class NegMarginLoss(nn.Module):
    """
        AKA CW loss
    """
    def __init__(self, reduction="mean", use_clip_loss=False):
        super(NegMarginLoss, self).__init__()
        self.reduction = reduction
        self.use_flip_loss = use_clip_loss

    def forward(self, logits, labels):
        correct_logits = torch.gather(logits, 1, labels.view(-1, 1)) # [n, 1]
        max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1) # [n, 2]
        top_max, second_max = max_2_logits.chunk(2, dim=1)  # [n, 2]
        top_argmax, _ = argmax_2_logits.chunk(2, dim=1)

        labels_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
        labels_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
        max_incorrect_logits = labels_eq_max * second_max + labels_ne_max * top_max # [n, 2]
        # loss = (max_incorrect_logits - correct_logits).clamp(max=1).squeeze().sum()
        
        if self.use_flip_loss:
            clamp_value = 0.01
        else:
            clamp_value = float("inf")

        if self.reduction == "none":
            loss = (max_incorrect_logits - correct_logits).clamp(max=clamp_value).squeeze(1)
        elif self.reduction == "sum":
            loss = (max_incorrect_logits - correct_logits).clamp(max=clamp_value).sum(axis=0)
        elif self.reduction == "mean":
            loss = (max_incorrect_logits - correct_logits).clamp(max=clamp_value).mean(axis=0)
        else:
            raise RuntimeError("Unsupported reduction method.")
        return -loss


class NegCELoss(nn.Module):
    def __init__(self, reduction="mean", use_clip_loss=False, clamp_value=10) -> None:
        super().__init__()
        self.use_clip_loss = use_clip_loss
        self.reduction = reduction
        self.loss_func = nn.CrossEntropyLoss(reduction="none")
        self.clamp_value = clamp_value
    
    def forward(self, logits, labels):
        if self.use_clip_loss:
            clamp_value = self.clamp_value
        else:
            clamp_value = float("inf") 
        
        if self.reduction == "none":
            loss = self.loss_func(logits, labels).clamp(max=clamp_value)
        elif self.reduction == "sum":
            loss = self.loss_func(logits, labels).clamp(max=clamp_value).sum(dim=0).unsqueeze(0)
        elif self.reduction == "mean":
            loss = self.loss_func(logits, labels).clamp(max=clamp_value).mean(dim=0).unsqueeze(0)
        else:
            raise RuntimeError("Unsupported reduction method.")
        return -loss

