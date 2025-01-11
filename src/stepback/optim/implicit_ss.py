"""
Author: Aaron Mishkin 
"""

import torch
import math
import warnings

from scipy.optimize import root_scalar
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ..types import Params, LossClosure, OptFloat

MIN_LR = 1e-3


class ImplicitSS(torch.optim.Optimizer):
    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        weight_decay: float = 0,
        use_two: bool = False,
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
        """

        params = list(params)
        defaults = dict(lr=lr, weight_decay=weight_decay)

        super(ImplicitSS, self).__init__(params, defaults)
        self.params = params

        if weight_decay != 0:
            raise ValueError(
                "Implicit step-sizes do not support weight decay."
            )

        self.lr = self.lr_prev = 1
        self.use_two = use_two
        self.current_iter = []
        self.current_grad = []

        for i, p in enumerate(self.params):
            # save gradient and iterate
            self.current_iter.append(p.data.clone())
            self.current_grad.append(torch.zeros_like(p.data))

        self.state["step_size_list"] = list()

        if len(self.param_groups) > 1:
            warnings.warn("More than one parameter group for SPS.")

        self.hvp_model = None
        self.closure_fn = None

        return

    def set_current_iter(self):
        for i, p in enumerate(self.params):
            self.current_iter[i] = p.data.clone()
            self.current_grad[i] = p.grad.clone()

    def try_update(self, eta):
        # print(eta, self.theta, self.grad)
        return self.theta - eta * self.grad

    def get_root_finding_obj(self):
        def rf_obj_fn(eta):
            theta_prime = self.try_update(eta)

            with torch.enable_grad():
                _, grad_prime = self.obj_fn(theta_prime)

            rf_obj = torch.dot(grad_prime, grad_prime) / 2 - torch.dot(
                grad_prime, self.grad
            )

            if self.use_two:
                rf_obj = rf_obj - (torch.dot(self.grad, self.grad) / 2)

            rf_obj = rf_obj.item() / len(grad_prime)

            with torch.enable_grad():
                hvp = self.hvp_fn(theta_prime, self.grad)

            rf_grad = torch.dot(self.grad - grad_prime, hvp).item() / len(
                grad_prime
            )
            # print(rf_obj, rf_grad)

            return rf_obj, rf_grad

        return rf_obj_fn

    def step(self, closure: LossClosure = None) -> OptFloat:
        """
        Compute step-size by solving non-linear root-finding
        equation.
        """

        with torch.enable_grad():
            loss = closure()

        grad = []
        for p in self.params:
            grad.append(p.grad.clone())

        assert self.hvp_model is not None
        assert self.closure_fn is not None

        self.obj_fn, self.hvp_fn = self.closure_fn()
        self.theta = parameters_to_vector(self.params).detach().clone()

        with torch.enable_grad():
            loss, self.grad = self.obj_fn(self.theta)

        rf_obj_fn = self.get_root_finding_obj()

        # find step-size by solving implicit equation
        root_results = root_scalar(
            rf_obj_fn,
            fprime=True,
            method="newton",
            x0=self.lr,
            xtol=1e-6,
            rtol=1e-6,
            maxiter=20,
        )
        # print("\nLearning Rate", root_results)

        self.lr_prev = self.lr
        self.lr = root_results.root

        # safe-guard against negative steps
        if self.lr <= 0:
            self.lr = self.lr_prev

        # final update
        theta_prime = self.try_update(self.lr)
        vector_to_parameters(theta_prime, self.params)

        _, grad_prime = self.obj_fn(theta_prime)

        inv_smoothness = torch.sqrt(
            torch.sum((theta_prime - self.theta) ** 2)
            / torch.sum((grad_prime - self.grad) ** 2)
        )

        # update state with metrics
        self.state["step_size_list"].append(
            self.lr
        )  # works only if one param_group!

        return loss

    def set_hvp_model(self, hvp_model, closure_fn):
        self.hvp_model = hvp_model
        self.closure_fn = closure_fn
