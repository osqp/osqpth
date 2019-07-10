# -*- coding: utf-8 -*-

"""Main module."""

import torch
from torch.autograd import Function
import osqp


# TODO: Finish from this
# https://github.com/sbarratt/diff_osqp/blob/master/diff_osqp.py


class OSQP(Function):
    def __init__(self,
                 P_idx, P_size,
                 A_idx, A_size,
                 eps_rel=1e-04,
                 eps_abs=1e-04,
                 verbose=0,
                 max_iter=10000):
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.verbose = verbose
        self.max_iter = max_iter
        self.P_idx, self.P_size = P_idx, P_size
        self.A_idx, self.A_size = A_idx, A_size

        # TODO: Perform OSQP Setup first to allocate memory?

    def forward(self, P_val_, q_, A_val_, l_, u_):
        """Solve a batch of QPs using OSQP.

        This function solves a batch of QPs, each optimizing over
        `n` variables and having `m` constraints.

        The optimization problem for each instance in the batch
        (dropping indexing from the notation) is of the form

            \hat z =   argmin_z 1/2 z' P z + q' z
                       subject to l <= Ax <= u
                                
        where P \in S^{n,n},
              S^{n,n} is the set of all positive semi-definite matrices,
              q \in R^{n}
              A \in R^{m,n}
              l \in R^{m}
              u \in R^{m}
              
        These parameters should all be passed to this function as
        Variable- or Parameter-wrapped Tensors.
        (See torch.autograd.Variable and torch.nn.parameter.Parameter)

        If you want to solve a batch of QPs where `n` and `m`
        are the same, but some of the contents differ across the
        minibatch, you can pass in tensors in the standard way
        where the first dimension indicates the batch example.
        This can be done with some or all of the coefficients.

        You do not need to add an extra dimension to coefficients
        that will not change across all of the minibatch examples.
        This function is able to infer such cases.

        If you don't want to use any constraints, you can set the 
        appropriate values to:

            e = Variable(torch.Tensor())

        """

        # Convert batches to sparse matrices/vectors
        self.n_batch = P_val_.size(0)  # Size of the batch
        sparseTensor = torch.cuda.sparse.DoubleTensor if P_val_.is_cuda \
            else torch.sparse.DoubleTensor
        P = [sparseTensor(self.P_idx, P_val_[i], self.P_size)
             for i in range(self.n_batch)]
        A = [sparseTensor(self.A_idx, A_val_[i], self.A_size)
             for i in range(self.n_batch)]

        # Perform forward step solving the QPs
        for i in range(self.n_batch):
            # Solve QP

            # Append solutions

        # Return solutions

        return zhats

    def backward(self, dl_dzhat):

        # Construct linear system


        # Solve it


        # Get derivatives
        grads = (dPs, dqs, dAs, dls, dus)

        return grads
