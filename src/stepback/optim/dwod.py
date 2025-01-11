"""
Author: Aaron Mishkin 
"""

import torch
import math
import warnings

from ..types import Params, LossClosure, OptFloat


class DwoD(torch.optim.Optimizer):
    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        weight_decay: float = 0,
        smooth: bool = False,
    ) -> None:
        """

        Parameters
        ----------
        params :
            PyTorch model parameters.
        lr : float, optional
            The initial learning rate. The default is 1e-3.
        weight_decay : float, optional
            Weigt decay parameter. The default is 0.
            If specified, the term weight_decay/2 * ||w||^2 is added to
            objective, where w are all model weights.
        smooth: bool, optional
            Whether or not to smooth the directional smoothness parameters.
        """

        params = list(params)
        defaults = dict(lr=lr, weight_decay=weight_decay)

        super(DwoD, self).__init__(params, defaults)
        self.params = params

        self.lr = lr
        self.prev_lr = lr
        self.theta = torch.inf

        self.smooth = smooth

        self.state["step_size_list"] = list()

        self.prev_iter = None
        self.prev_grad = None

        if len(self.param_groups) > 1:
            warnings.warn("More than one parameter group for SPS.")

        return

    def step(self, closure: LossClosure = None) -> OptFloat:
        """
        Descent without descent update.
        See https://arxiv.org/pdf/1910.09529.pdf.
        """

        with torch.enable_grad():
            loss = closure()

        # handle "implict" weight-decay regularization.
        r = 0
        for group in self.param_groups:
            lmbda = group["weight_decay"]
            for p in group["params"]:
                p.grad.add_(lmbda * p.data)  # gradients
                r += (lmbda / 2) * (p.data**2).sum()  # loss

        loss.add_(r)

        # compute directional smoothness
        with torch.no_grad():
            if self.prev_grad is None:
                self.prev_grad = []
                self.prev_iter = []
                for i, p in enumerate(self.params):
                    # save gradient and iterate
                    self.prev_iter.append(p.data.clone())
                    self.prev_grad.append(p.grad.clone())

            else:
                iter_diff_norm = 0
                grad_diff_norm = 0
                for i, p in enumerate(self.params):
                    iter_diff = p.data - self.prev_iter[i]
                    grad_diff = p.grad - self.prev_grad[i]

                    iter_diff_norm += torch.sum(iter_diff**2)
                    grad_diff_norm += torch.sum(grad_diff**2)

                    self.prev_iter[i] = p.data.clone()
                    self.prev_grad[i] = p.grad.clone()

                iter_diff_norm = torch.sqrt(iter_diff_norm).item()
                grad_diff_norm = torch.sqrt(grad_diff_norm).item()

                self.prev_lr = self.lr

                # guard against dividing by 0.
                if grad_diff_norm > 0:
                    self.lr = iter_diff_norm / (2 * grad_diff_norm)
                else:
                    self.lr = math.inf

                if self.smooth:
                    self.lr = min(
                        self.lr, math.sqrt(1 + self.theta) * self.prev_lr
                    )
                    self.theta = self.lr / self.prev_lr

                if math.isnan(self.lr):
                    raise ValueError()

        ############################################################
        # update
        for group in self.param_groups:
            for p in group["params"]:
                p.data.add_(other=p.grad.data, alpha=-self.lr)

        ############################################################
        # update state with metrics
        self.state["step_size_list"].append(
            self.lr
        )  # works only if one param_group!

        return loss
