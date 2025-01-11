import torch


class Loss:
    def __init__(
        self,
        name: str,
        backwards: bool = False,
        reg: float = 0.0,
    ):
        self.name = name
        self.backwards = backwards
        self.reg = reg

        # defaults
        self._flatten_target = True
        self._flatten_out = False

        if self.name == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss()

        elif self.name == "logistic":
            self.criterion = torch.nn.SoftMarginLoss()
            self._flatten_out = True

        elif self.name == "squared":
            self.criterion = torch.nn.MSELoss()
            self._flatten_out = True

        elif self.name == "cross_entropy_accuracy":
            assert (
                not self.backwards
            ), "For accuracy metrics, we never want to do backprop."
            self.criterion = cross_entropy_accuracy

        elif self.name == "logistic_accuracy":
            assert (
                not self.backwards
            ), "For accuracy metrics, we never want to do backprop."
            self.criterion = logistic_accuracy

        return

    def compute(self, out, targets, model=None, recompute=False, data=None):
        if recompute:
            out = model(data)

        if self._flatten_out:
            out = out.view(-1)

        if self._flatten_target:
            targets = targets.view(-1)

        loss = self.criterion(out, targets)

        # optional weight-decay penalty
        if self.reg > 0.0:
            if model is None:
                raise ValueError(
                    "Model cannot be None when regularization is non-zero."
                )
            penalty = 0
            for p in model.parameters():
                penalty += torch.sum(p**2)

            loss += self.reg * penalty

        if self.backwards and loss.requires_grad:
            loss.backward()

        return loss


##
# Accuracy functions
# ==========================


def logistic_accuracy(out, targets):
    logits = torch.sigmoid(out).view(-1)
    pred_labels = (logits >= 0.5) * 2 - 1  # map to{-1,1}
    acc = (pred_labels == targets).float().mean()
    return acc


def cross_entropy_accuracy(out, targets):
    pred_labels = out.argmax(dim=1)
    acc = (pred_labels == targets).float().mean()
    return acc
