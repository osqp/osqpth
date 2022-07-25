from functools import partial
import scipy.sparse as spa
from osqpth.osqpth import _OSQP_Fn
from osqpth.util import to_numpy
import osqp


import torch
from torch.autograd import Function

from enum import Enum



def QPFunction(**kwargs):

    verbose = kwargs.get('verbose')
    eps_abs = kwargs.get('eps')
    not_improved_lim = kwargs.get('notImprovedLim')
    max_iter = kwargs.get('maxIter')

    solvers = []

    class QPFunctionFn(Function):
        @staticmethod
        def forward(ctx, P, q, A, u, l, _ignored):

            assert l.nelement() == 0, 'Equality constraints not supported yet'
            if P.ndim > 2:
                batch_mode = True
                n_batch = P.shape[0]

            dtype = P.dtype
            device = P.device

            ctx.m = A.shape[1] if batch_mode else A.shape[0]
            ctx.n = P.shape[1] if batch_mode else P.shape[0]

            P = [spa.csc_matrix(to_numpy(P[i])) for i in range(n_batch)]
            q = [to_numpy(q[i]) for i in range(n_batch)]
            A = [spa.csc_matrix(to_numpy(A[i])) for i in range(n_batch)]
            u = [to_numpy(u[i]) for i in range(n_batch)]

            # Perform forward step solving the QPs
            x_torch = torch.zeros((n_batch, ctx.n), dtype=dtype, device=device)

            x = []
            for i in range(n_batch):
                # Solve QP
                # TODO: Cache solver object in between
                solver = osqp.OSQP()
                solver.setup(P[i], q[i], A[i], None, u[i], verbose=verbose,
                             eps_abs=eps_abs)
                result = solver.solve()
                solvers.append(solver)
                status = result.info.status
                if status != 'solved':
                    # TODO: We can replace this with something calmer and
                    # add some more options around potentially ignoring this.
                    raise RuntimeError(f"Unable to solve QP, status: {status}")
                x.append(result.x)

                # This is silently converting result.x to the same
                # dtype and device as x_torch.
                x_torch[i] = torch.from_numpy(result.x)

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

            n_batch = dl_dx_val.size(0)
            dtype = dl_dx_val.dtype
            device = dl_dx_val.device

            # Convert dl_dx to numpy
            dl_dx = to_numpy(dl_dx_val)

            # Convert to torch tensors
            dP = torch.zeros((n_batch, ctx.n, ctx.n), dtype=dtype, device=device)
            dq = torch.zeros((n_batch, ctx.n), dtype=dtype, device=device)
            dA = torch.zeros((n_batch, ctx.m, ctx.n), dtype=dtype, device=device)
            dl = torch.zeros((n_batch, ctx.m), dtype=dtype, device=device)
            du = torch.zeros((n_batch, ctx.m), dtype=dtype, device=device)

            for i in range(n_batch):
                derivatives_np = solvers[i].adjoint_derivative(
                    dx=dl_dx[i], dy_u=None, dy_l=None,
                    A_idx=None, P_idx=None,
                    as_dense=True, dP_as_triu=True

                )
                dPi_np, dqi_np, dAi_np, dli_np, dui_np = derivatives_np
                dq[i], dl[i], du[i] = [torch.from_numpy(d) for d in [dqi_np, dli_np, dui_np]]
                dP[i], dA[i] = [torch.from_numpy(d) for d in [dPi_np, dAi_np]]

            grads = [dP, dq, dA, dl, du, None]

            if not batch_mode:
                for i, g in enumerate(grads):
                    grads[i] = g.squeeze()

            return tuple(grads)

    return QPFunctionFn.apply
