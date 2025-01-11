"""
Author: Fabian Schaipp

Adapted from https://github.com/fabian-sp/ProxSPS/blob/main/sps/sps.py.

Main changes:
    * use .data in all computations
    * rename 'fstar' to 'lb'
"""

import torch
import math
import warnings

from ..types import Params, LossClosure, OptFloat


class NormalizedSGD(torch.optim.Optimizer):
    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        weight_decay: float = 0,
        lr_decay=False,
    ) -> None:
        """
        Parameters
        ----------
        params :
            PyTorch model parameters.
        lr : float, optional
            Learning rate. The default is 1e-3.
        weight_decay : float, optional
            Weigt decay parameter. The default is 0.
            If specified, the term weight_decay/2 * ||w||^2 is added to objective, where w are all model weights.
        """

        params = list(params)
        defaults = dict(lr=lr, weight_decay=weight_decay)

        super(NormalizedSGD, self).__init__(params, defaults)
        self.params = params

        self.lr = lr
        self.lr_decay = lr_decay
        self.num_steps = 1

        self.state["step_size_list"] = list()

        if len(self.param_groups) > 1:
            warnings.warn("More than one parameter group for SPS.")

        return

    def step(self, closure: LossClosure = None) -> OptFloat:
        """
        Normalized SGD update.
        """
        lr = self.lr
        if self.lr_decay:
            lr = lr / math.sqrt(self.num_steps)

        with torch.enable_grad():
            loss = closure()

        grad_norm = self.compute_grad_terms()

        ############################################################
        # update
        for group in self.param_groups:
            lr = group["lr"] / grad_norm
            lmbda = group["weight_decay"]

            for p in group["params"]:
                g = p.grad
                if lmbda != 0:
                    g = g.add(p, alpha=lmbda)

                p.data.sub_(other=g, alpha=lr)

        ############################################################

        self.num_steps += 1
        return loss

    @torch.no_grad()
    def compute_grad_terms(self):
        """
        computes the norm of stochastic gradient ||grad||.
        """
        grad_norm = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    raise KeyError("None gradient")

                g = p.grad.data
                grad_norm += torch.sum(torch.mul(g, g))

        grad_norm = torch.sqrt(grad_norm)
        return grad_norm
