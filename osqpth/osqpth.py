import torch
from torch.nn import Module
from torch.autograd import Function
import osqp
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as sla

from enum import IntEnum

from .util import to_numpy

DiffModes = IntEnum('DiffModes', 'ACTIVE FULL')

class OSQP(Module):
    def __init__(self, P_idx, P_shape, A_idx, A_shape,
                 eps_rel=1e-5, eps_abs=1e-5, verbose=False,
                 max_iter=10000, diff_mode=DiffModes.ACTIVE):
        super().__init__()
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.verbose = verbose
        self.max_iter = max_iter
        self.P_idx, self.P_shape = P_idx, P_shape
        self.A_idx, self.A_shape = A_idx, A_shape
        self.diff_mode = diff_mode

    def forward(self, P_val, q_val, A_val, l_val, u_val):
        return _OSQP.apply(
            P_val, q_val, A_val, l_val, u_val,
            self.P_idx, self.P_shape,
            self.A_idx, self.A_shape,
            self.eps_rel, self.eps_abs,
            self.verbose, self.max_iter,
            self.diff_mode,
        )


class _OSQP(Function):
    @staticmethod
    def forward(ctx, P_val, q_val, A_val, l_val, u_val,
                A_idx, A_shape, P_idx, P_shape,
                eps_rel, eps_abs, verbose, max_iter, diff_mode):
        """Solve a batch of QPs using OSQP.

        This function solves a batch of QPs, each optimizing over
        `n` variables and having `m` constraints.

        The optimization problem for each instance in the batch
        (dropping indexing from the notation) is of the form

            \\hat x =   argmin_x 1/2 x' P x + q' x
                       subject to l <= Ax <= u

        where P \\in S^{n,n},
              S^{n,n} is the set of all positive semi-definite matrices,
              q \\in R^{n}
              A \\in R^{m,n}
              l \\in R^{m}
              u \\in R^{m}

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

        ctx.eps_abs = eps_abs
        ctx.eps_rel = eps_rel
        ctx.verbose = verbose
        ctx.max_iter = max_iter
        ctx.P_idx, ctx.P_shape = P_idx, P_shape
        ctx.A_idx, ctx.A_shape = A_idx, A_shape
        ctx.diff_mode = diff_mode

        params = [P_val, q_val, A_val, l_val, u_val]

        for p in params:
            assert p.ndimension() <= 2, 'Unexpected number of dimensions'

        # Convert batches to sparse matrices/vectors
        batch_mode = np.all([t.ndimension() == 1 for t in params])
        if batch_mode:
            ctx.n_batch = 1
        else:
            batch_sizes = [t.size(0) if t.ndimension() == 2 else 1 for t in params]
            ctx.n_batch = max(batch_sizes)
        ctx.m, ctx.n = ctx.A_shape   # Problem size

        dtype = P_val.dtype
        device = P_val.device

        # Convert P and A to sparse matrices
        # TODO (Bart): create CSC matrix during initialization. Then
        # just reassign the mat.data vector with A_val and P_val

        for i, p in enumerate(params):
            if p.ndimension() == 1:
                params[i] = p.unsqueeze(0).expand(ctx.n_batch, p.size(0))

        [P_val, q_val, A_val, l_val, u_val] = params

        P = [spa.csc_matrix((to_numpy(P_val[i]), ctx.P_idx), shape=ctx.P_shape)
             for i in range(ctx.n_batch)]
        q = [to_numpy(q_val[i]) for i in range(ctx.n_batch)]
        A = [spa.csc_matrix((to_numpy(A_val[i]), ctx.A_idx), shape=ctx.A_shape)
             for i in range(ctx.n_batch)]
        l = [to_numpy(l_val[i]) for i in range(ctx.n_batch)]
        u = [to_numpy(u_val[i]) for i in range(ctx.n_batch)]

        # Perform forward step solving the QPs
        x_torch = torch.zeros((ctx.n_batch, ctx.n), dtype=dtype, device=device)

        x, y, z = [], [], []
        for i in range(ctx.n_batch):
            # Solve QP
            # TODO: Cache solver object in between
            m = osqp.OSQP()
            m.setup(P[i], q[i], A[i], l[i], u[i], verbose=ctx.verbose)
            result = m.solve()
            status = result.info.status
            if status != 'solved':
                # TODO: We can replace this with something calmer and
                # add some more options around potentially ignoring this.
                raise RuntimeError(f"Unable to solve QP, status: {status}")
            x.append(result.x)
            y.append(result.y)
            z.append(A[i].dot(result.x))

            # This is silently converting result.x to the same
            # dtype and device as x_torch.
            x_torch[i] = torch.from_numpy(result.x)

        # Save stuff for backpropagation
        ctx.backward_vars = (P, q, A, l, u, x, y, z)

        # Return solutions
        if not batch_mode:
            x_torch = x_torch.squeeze(0)
        return x_torch

    @staticmethod
    def backward(ctx, dl_dx_val):
        dtype = dl_dx_val.dtype
        device = dl_dx_val.device

        batch_mode = dl_dx_val.ndimension() == 2
        if not batch_mode:
            dl_dx_val = dl_dx_val.unsqueeze(0)

        # Convert dl_dx to numpy
        dl_dx = to_numpy(dl_dx_val)

        # Extract data from forward pass
        P, q, A, l, u, x, y, z = ctx.backward_vars

        # Convert to torch tensors
        nnz_P = len(ctx.P_idx[0])
        nnz_A = len(ctx.A_idx[0])
        dP = torch.zeros((ctx.n_batch, nnz_P), dtype=dtype, device=device)
        dq = torch.zeros((ctx.n_batch, ctx.n), dtype=dtype, device=device)
        dA = torch.zeros((ctx.n_batch, nnz_A), dtype=dtype, device=device)
        dl = torch.zeros((ctx.n_batch, ctx.m), dtype=dtype, device=device)
        du = torch.zeros((ctx.n_batch, ctx.m), dtype=dtype, device=device)

        # TODO: Improve this, reuse OSQP, port it in C

        for i in range(ctx.n_batch):
            # Construct linear system

            if ctx.diff_mode == DiffModes.ACTIVE:
                # Taken from https://github.com/oxfordcontrol/osqp-python/blob/0363d028b2321017049360d2eb3c0cf206028c43/modulepurepy/_osqp.py#L1717
                # Guess which linear constraints are lower-active, upper-active, free
                ind_low = np.where(z[i] - l[i] < - y[i])[0]
                ind_upp = np.where(u[i] - z[i] < y[i])[0]
                n_low = len(ind_low)
                n_upp = len(ind_upp)

                # Form A_red from the assumed active constraints
                A_red = spa.vstack([A[i][ind_low], A[i][ind_upp]])

                # Form KKT linear system
                KKT = spa.vstack([spa.hstack([P[i], A_red.T]),
                                spa.hstack([A_red, spa.csc_matrix((n_low + n_upp, n_low + n_upp))])])
                rhs = np.hstack([dl_dx[i], np.zeros(n_low + n_upp)])

                # Get solution
                r_sol = sla.spsolve(KKT, rhs)
            elif ctx.diff_mode == DiffModes.FULL:
                raise NotImplementedError
            else:
                raise RuntimeError(f"Unrecognized differentiation mode")

            r_x =  r_sol[:ctx.n]
            r_yl = r_sol[ctx.n:ctx.n + n_low]
            r_yu = r_sol[ctx.n + n_low:]
            r_y = np.zeros(ctx.m)
            r_y[ind_low] = r_yl
            r_y[ind_upp] = r_yu

             # Extract derivatives
            rows, cols = ctx.P_idx
            values = -.5 * (r_x[rows] * x[i][cols] + r_x[cols] * x[i][rows])
            dP[i] = torch.from_numpy(values)

            rows, cols = ctx.A_idx
            values = -(y[i][rows] * r_x[cols] + r_y[rows] * x[i][cols])
            dA[i] = torch.from_numpy(values)
            dq[i] = torch.from_numpy(-r_x)
            t = np.hstack([r_yl[np.where(ind_low == j)[0]] if j in ind_low else 0
                 for j in range(ctx.m)])
            dl[i] = torch.tensor(t)
            t = np.hstack([r_yu[np.where(ind_upp == j)[0]] if j in ind_upp else 0
                 for j in range(ctx.m)])
            du[i] = torch.tensor(t)

        grads = [dP, dq, dA, dl, du]

        if not batch_mode:
            for i, g in enumerate(grads):
                grads[i] = g.squeeze()

        grads += [None]*9

        return tuple(grads)
